"""Smoke-test frozen OpenVision 2 ViT-L/14-224 (CVPR 2026) on one audited training image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import hf_hub_download
import numpy as np
import PIL
from PIL import Image
import torch


CONFIG_NAME = "baseline_openvision2_vitl14.json"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_image(image: Image.Image, res: int = 224) -> torch.Tensor:
    im = image.convert("RGB")
    w, h = im.size
    scale = res / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = im.size
    left, top = (w - res) // 2, (h - res) // 2
    im = im.crop((left, top, left + res, top + res))
    arr = (np.asarray(im, dtype=np.float32) - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def load_model(root: Path, config: dict, device: torch.device, local_files_only: bool = True):
    if config["allow_parameter_updates"] is not False or not config["backbone_frozen"]:
        raise ValueError("This step requires frozen inference")
    repo_dir = root / config["local_repo_dir"] / "src" / "convert_upload"
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from open_clip.factory import load_state_dict
    from open_clip.model import _build_vision_tower

    cfg_path = Path(
        hf_hub_download(
            repo_id=config["model_repo"],
            filename="open_clip_config.json",
            revision=config["requested_revision"],
            cache_dir=str(root / "hf_cache"),
            token=False,
            local_files_only=local_files_only,
        )
    )
    weights_path = Path(
        hf_hub_download(
            repo_id=config["model_repo"],
            filename=config["weights_filename"],
            revision=config["requested_revision"],
            cache_dir=str(root / "hf_cache"),
            token=False,
            local_files_only=local_files_only,
        )
    )
    if weights_path.stat().st_size != config["weights_size_bytes"]:
        raise RuntimeError("OpenVision 2 weights size mismatch")
    weights_sha256 = file_hash(weights_path)
    if "weights_sha256" in config and weights_sha256 != config["weights_sha256"]:
        raise RuntimeError("OpenVision 2 weights SHA256 mismatch")

    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["model_cfg"]
    vision_encoder = _build_vision_tower(
        embed_dim=model_cfg["embed_dim"],
        vision_cfg=model_cfg["vision_cfg"],
        cast_dtype=None,
    )
    state_dict = load_state_dict(str(weights_path))
    if any(k.startswith("visual.") for k in state_dict):
        state_dict = {k.replace("visual.", "", 1): v for k, v in state_dict.items() if k.startswith("visual.")}
    vision_encoder.load_state_dict(state_dict, strict=True)
    vision_encoder.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    return vision_encoder, weights_sha256


def extract_features(model, inputs: torch.Tensor) -> np.ndarray:
    with torch.inference_mode():
        out = model(inputs)
        pooled = out[0] if isinstance(out, (tuple, list)) else out
    return pooled.detach().float().cpu().numpy()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = root / "data" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = manifest["split_indices"]["train"][0]
    record = manifest["records"][index]
    image_path = args.data_root / record["relative_image"]
    if file_hash(image_path) != record["sha256"]:
        raise RuntimeError("Image hash mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model, weights_sha256 = load_model(root, config, device, local_files_only=True)
    load_seconds = time.perf_counter() - t0

    with Image.open(image_path) as source:
        inputs = preprocess_image(source, config["expected_image_size"]).unsqueeze(0)

    t1 = time.perf_counter()
    features = extract_features(model, inputs.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_seconds = time.perf_counter() - t1

    if features.shape != (1, config["expected_feature_dim"]) or not np.isfinite(features).all():
        raise RuntimeError(f"Feature shape/finite check failed: {features.shape}")

    frozen = all(not p.requires_grad for p in model.parameters())
    param_count = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated(device) / (2**30) if device.type == "cuda" else None
    print("Model:", config["paper_model"], "| Params:", param_count, "| Feature:", tuple(features.shape), "| Peak GPU GiB:", round(peak, 3))

    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "openvision2_vitl14_single_image_smoke_passed_no_training",
        "paper_model": config["paper_model"],
        "model_repo": config["model_repo"],
        "requested_revision": config["requested_revision"],
        "resolved_revision": config["requested_revision"],
        "weights_sha256": weights_sha256,
        "config_relative": f"configs/{CONFIG_NAME}",
        "config_sha256": file_hash(config_path),
        "script_sha256": file_hash(Path(__file__)),
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "manifest_sha256": file_hash(manifest_path),
        "sample_index": record["index"],
        "image_relative": record["relative_image"],
        "image_sha256": record["sha256"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_shape": list(inputs.shape),
        "feature_source": "official_openvision2_pooled_output",
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "all_finite": True,
        "all_parameters_frozen": frozen,
        "parameter_count": param_count,
        "load_seconds": round(load_seconds, 4),
        "single_image_infer_seconds": round(infer_seconds, 4),
        "peak_gpu_gib": round(peak, 4) if peak is not None else None,
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "allow_tf32": False,
        "torch_version": torch.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"openvision2_vitl14_smoke_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
