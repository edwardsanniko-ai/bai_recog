"""One training-image smoke test of frozen official EUPE ViT-B/16 features."""

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
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2


CONFIG_NAME = "baseline_eupe_vitb16.json"


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_transform(size: int = 256):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_model(root: Path, config: dict, device: torch.device, local_files_only: bool = True):
    if config["allow_parameter_updates"] is not False or not config["backbone_frozen"]:
        raise ValueError("This step requires frozen inference")
    repo_dir = root / config["local_repo_dir"]
    repo_rev = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if repo_rev != config["source_revision"]:
        raise ValueError(f"EUPE repo revision mismatch: {repo_rev} != {config['source_revision']}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from eupe.hub.backbones import eupe_vitb16

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = Path(
        hf_hub_download(
            repo_id=config["backbone"],
            filename=config["weights_filename"],
            revision=config["revision"],
            cache_dir=str(root / "hf_cache"),
            token=False,
            local_files_only=local_files_only,
        )
    )
    weights_sha256 = file_hash(path)
    if "weights_sha256" in config and weights_sha256 != config["weights_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from locked official hash")

    state = torch.load(path, map_location="cpu", weights_only=True)
    model = eupe_vitb16(pretrained=False)
    expected = set(model.state_dict())
    missing = sorted(expected - set(state))
    extra = sorted(set(state) - expected)
    allowed_extra = sorted(
        f"projectors.heads.dinov3radio_vitg16.{kind}.layers.{suffix}"
        for kind in ("cls", "patch")
        for suffix in ("0.weight", "1.weight", "1.bias", "3.weight")
    )
    if missing or extra != allowed_extra:
        raise ValueError(f"Unexpected checkpoint keys: missing={missing}, extra={extra}")
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    model.eval().requires_grad_(False)
    model.to(device=device, dtype=torch.float32)
    return model, path, weights_sha256, extra


def image_features(model, inputs: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        features = model.forward_features(inputs)["x_norm_clstoken"]
    if features.ndim != 2 or features.shape != (len(inputs), 768):
        raise RuntimeError(f"Expected EUPE-ViT-B whole-image features [batch,768], got {tuple(features.shape)}")
    if not torch.isfinite(features).all() or torch.any(features.norm(dim=1) == 0):
        raise RuntimeError("Invalid or zero-length features")
    return features.detach().float().cpu()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = root / "data/splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = manifest["split_indices"]["train"][0]
    record = manifest["records"][index]
    image_path = args.data_root / record["relative_image"]
    if file_hash(image_path) != record["sha256"]:
        raise ValueError("Image differs from audited manifest")
    with Image.open(image_path) as original:
        image = original.convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    model, weights_path, weights_sha256, excluded = load_model(root, config, device, local_files_only=True)
    load_seconds = time.perf_counter() - t0
    inputs = make_transform(config["image_size"])(image).unsqueeze(0).to(device)
    t1 = time.perf_counter()
    features = image_features(model, inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_seconds = time.perf_counter() - t1

    frozen = all(not p.requires_grad for p in model.parameters())
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    print("Model:", config["backbone"], "| Params:", sum(p.numel() for p in model.parameters()))
    print("Feature shape:", tuple(features.shape), "| Peak GPU GiB:", round(peak, 3))

    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = dict(
        timestamp=now.isoformat(),
        status="eupe_vitb16_single_image_smoke_passed_no_classification",
        paper_model=config["paper_model"],
        resolved_revision=config["revision"],
        config_relative=f"configs/{CONFIG_NAME}",
        config_sha256=file_hash(config_path),
        script_sha256=file_hash(Path(__file__)),
        code_commit=git.stdout.strip() if git.returncode == 0 else None,
        weights_sha256=weights_sha256,
        manifest_sha256=file_hash(manifest_path),
        sample_index=index,
        image_relative=record["relative_image"],
        image_sha256=record["sha256"],
        input_shape=list(inputs.shape),
        feature_source="official_eupe_vitb16_x_norm_clstoken",
        feature_shape=list(features.shape),
        feature_dtype="float32",
        all_finite=True,
        all_backbone_weights_loaded=True,
        excluded_distillation_projector_keys=excluded,
        all_parameters_frozen=frozen,
        parameter_count=sum(p.numel() for p in model.parameters()),
        load_seconds=round(load_seconds, 4),
        single_image_infer_seconds=round(infer_seconds, 4),
        optimizer_updates=0,
        test_evaluated=False,
        classification_performed=False,
        device=str(device),
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        allow_tf32=False,
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        peak_allocated_gpu_gib=round(peak, 4) if peak is not None else None,
    )
    destination = root / "logs" / ("eupe_vitb16_smoke_" + now.strftime("%Y%m%d_%H%M%S_%f") + ".json")
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Saved report:", destination)


if __name__ == "__main__":
    main()
