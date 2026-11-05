"""Check one image with frozen torchvision ResNet50 and ConvNeXt-Tiny controls."""

from __future__ import annotations

# Directly invoking a Conda environment's python.exe on Windows can omit its
# Library/bin DLL directory. Register it before importing Pillow/torchvision.
import ctypes
import os
from pathlib import Path
import sys

_DLL_HANDLE = None
if os.name == "nt":
    dll_dir = Path(sys.prefix) / "Library" / "bin"
    if dll_dir.is_dir():
        os.environ["PATH"] = str(dll_dir) + os.pathsep + str(Path(sys.prefix)) + os.pathsep + os.environ["PATH"]
        _DLL_HANDLE = os.add_dll_directory(str(dll_dir))
        imaging_modules = list((Path(sys.prefix) / "Lib" / "site-packages" / "PIL").glob("_imaging.cp*-win_amd64.pyd"))
        if len(imaging_modules) == 1:
            ctypes.WinDLL(str(imaging_modules[0]))

import argparse
from datetime import datetime, timezone
import hashlib
import json
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch
from torch import nn
import torchvision
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    convnext_tiny,
    resnet50,
)


CONTROL_SPECS = {
    "resnet50": {
        "config": "control_resnet50.json",
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "builder": resnet50,
        "feature_dim": 2048,
    },
    "convnext_tiny": {
        "config": "control_convnext_tiny.json",
        "weights": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "builder": convnext_tiny,
        "feature_dim": 768,
    },
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_audited_image(root: Path, data_root: Path):
    manifest_path = root / "data" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = manifest["split_indices"]["train"][0]
    record = manifest["records"][index]
    image_path = data_root / record["relative_image"]
    if file_hash(image_path) != record["sha256"]:
        raise ValueError("Image differs from the audited manifest")
    return manifest_path, index, record, image_path


def load_control(root: Path, name: str, device: torch.device):
    spec = CONTROL_SPECS[name]
    config = json.loads((root / "configs" / spec["config"]).read_text(encoding="utf-8"))
    weights = spec["weights"]
    if config["allow_parameter_updates"] is not False or not config["backbone_frozen"]:
        raise ValueError("Control model must remain frozen")
    if str(weights) != config["weight_enum"] or weights.url != config["weights_url"]:
        raise ValueError("Torchvision weight identity differs from pinned configuration")
    if torchvision.__version__ != config["torchvision_version"]:
        raise ValueError("Torchvision version differs from pinned configuration")
    torch.hub.set_dir(str(root / "hf_cache" / "torch"))
    weight_path = root / "hf_cache" / "torch" / "checkpoints" / config["weights_filename"]
    if weight_path.is_file() and file_hash(weight_path) != config["weights_sha256"]:
        raise ValueError("Existing official checkpoint has the wrong SHA256")
    model = spec["builder"](weights=weights)
    if file_hash(weight_path) != config["weights_sha256"]:
        raise ValueError("Downloaded official checkpoint has the wrong SHA256")
    # Builders load the full checkpoint strictly. Remove only the original
    # ImageNet-1000 linear prediction layer after successful loading.
    if name == "resnet50":
        if not isinstance(model.fc, nn.Linear) or model.fc.in_features != 2048 or model.fc.out_features != 1000:
            raise RuntimeError("Unexpected ResNet50 classification head")
        model.fc = nn.Identity()
    elif name == "convnext_tiny":
        if (len(model.classifier) != 3 or not isinstance(model.classifier[2], nn.Linear)
                or model.classifier[2].in_features != 768 or model.classifier[2].out_features != 1000):
            raise RuntimeError("Unexpected ConvNeXt-Tiny classification head")
        model.classifier[2] = nn.Identity()
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    return config, weights, model, weight_path


def image_features(model: nn.Module, inputs: torch.Tensor, feature_dim: int) -> torch.Tensor:
    with torch.inference_mode():
        features = model(inputs)
    if tuple(features.shape) != (len(inputs), feature_dim):
        raise RuntimeError("Unexpected whole-image feature shape")
    if not torch.isfinite(features).all() or torch.any(features.norm(dim=1) == 0):
        raise RuntimeError("Invalid or zero-length whole-image features")
    return features.detach().float().cpu()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()
    manifest_path, index, record, image_path = find_audited_image(root, args.data_root)
    with Image.open(image_path) as original:
        image = original.convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("Image:", image_path)
    print("Device:", device)
    print("Official checkpoints are cached locally; no training or classification.")
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    for name, spec in CONTROL_SPECS.items():
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        config, weights, model, weight_path = load_control(root, name, device)
        transform = weights.transforms()
        inputs = transform(image).unsqueeze(0).to(device)
        features = image_features(model, inputs, spec["feature_dim"])
        frozen = all(not p.requires_grad for p in model.parameters())
        if not frozen or model.training:
            raise RuntimeError("Control backbone is not fully frozen in eval mode")
        peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        print(f"\nControl: {name}")
        print("Weight enum:", config["weight_enum"])
        print("Input shape:", tuple(inputs.shape))
        print("Feature shape:", tuple(features.shape))
        print("Feature dtype:", str(features.numpy().dtype))
        print("All finite:", bool(np.isfinite(features.numpy()).all()))
        print("All parameters frozen:", frozen)
        print("Optimizer updates:", 0)
        print("Test evaluated:", False)
        if peak is not None:
            print("Peak allocated GPU GiB:", round(peak, 3))
        report = {
            "timestamp": now.isoformat(),
            "status": "single_image_smoke_passed_no_classification",
            "control_name": name,
            "config": config,
            "code_commit": git.stdout.strip() if git.returncode == 0 else None,
            "script_sha256": file_hash(Path(__file__)),
            "manifest_sha256": file_hash(manifest_path),
            "weights_sha256": file_hash(weight_path),
            "sample_index": index,
            "image_relative": record["relative_image"],
            "image_sha256": record["sha256"],
            "original_size": list(image.size),
            "input_shape": list(inputs.shape),
            "feature_shape": list(features.shape),
            "feature_dtype": "float32",
            "all_finite": True,
            "all_official_weights_loaded_before_head_removal": True,
            "original_imagenet_head_used": False,
            "all_parameters_frozen": frozen,
            "trainable_parameter_count": 0,
            "backbone_parameter_count_after_head_removal": sum(p.numel() for p in model.parameters()),
            "optimizer_updates": 0,
            "classification_performed": False,
            "test_evaluated": False,
            "transform": str(transform),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_gpu_gib": peak,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "allow_tf32": False,
        }
        destination = root / "logs" / f"control_{name}_smoke_{run_id}.json"
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("Saved report:", destination)


if __name__ == "__main__":
    main()
