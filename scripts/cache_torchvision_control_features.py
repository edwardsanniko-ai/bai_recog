"""Cache both frozen torchvision control features; no training or evaluation."""

from __future__ import annotations

# Importing the smoke helper first also registers the Conda DLL directory on Windows.
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch
import torchvision

from smoke_test_torchvision_controls import CONTROL_SPECS, file_hash, image_features, load_control


def verify_manifest_and_smoke(root: Path, config: dict, name: str) -> dict:
    manifest_path = root / "data" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, split = manifest["records"], manifest["split_indices"]
    if len(records) != 620 or [r["index"] for r in records] != list(range(620)):
        raise ValueError("Expected 620 audited records in original order")
    if len({r["relative_image"] for r in records}) != 620:
        raise ValueError("Duplicate image paths")
    if sorted(i for key in ("train", "validation", "test") for i in split[key]) != list(range(620)):
        raise ValueError("Overlapping or incomplete splits")
    if [len(split[k]) for k in ("train", "validation", "test")] != [434, 93, 93]:
        raise ValueError("Unexpected split sizes")
    categories = manifest["categories"]
    if len(categories) != 6 or any(not 0 <= r["class_id"] < 6 for r in records):
        raise ValueError("Invalid classes")
    if any(r["class_name"] != categories[r["class_id"]] for r in records):
        raise ValueError("Class names and IDs disagree")
    smoke_path = root / config["smoke_report"]
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke["control_name"] != name or smoke["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Smoke report identity or manifest mismatch")
    for key in (
        "backbone", "weight_enum", "weights_url", "weights_filename", "weights_sha256",
        "torchvision_version", "feature_dim", "preprocessing", "pooling", "compute_dtype",
        "allow_tf32", "evaluation_protocol",
    ):
        if smoke["config"][key] != config[key]:
            raise ValueError(f"Configuration differs from successful smoke test: {key}")
    if (
        smoke["script_sha256"] != file_hash(root / "scripts/smoke_test_torchvision_controls.py")
        or smoke["weights_sha256"] != config["weights_sha256"]
        or not smoke["all_official_weights_loaded_before_head_removal"]
        or not smoke["all_parameters_frozen"]
        or not smoke["all_finite"]
        or smoke["original_imagenet_head_used"]
        or smoke["optimizer_updates"] != 0
        or smoke["classification_performed"]
        or smoke["test_evaluated"]
    ):
        raise ValueError("Successful frozen smoke-test requirements not met")
    return manifest


def extract_batch(records, data_root: Path, transform, model, device, feature_dim: int) -> np.ndarray:
    tensors = []
    for record in records:
        path = data_root / record["relative_image"]
        if file_hash(path) != record["sha256"]:
            raise ValueError(f"Image changed: sample {record['index']}")
        with Image.open(path) as image:
            tensors.append(transform(image.convert("RGB")))
    inputs = torch.stack(tensors).to(device)
    if tuple(inputs.shape) != (len(records), 3, 224, 224):
        raise ValueError("Unexpected official preprocessing output shape")
    return image_features(model, inputs, feature_dim).numpy()


def save_cache(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError("Cache readback differs from extracted arrays")


def cache_one(root: Path, data_root: Path, name: str, device: torch.device, run_id: str, now):
    spec = CONTROL_SPECS[name]
    config = json.loads((root / "configs" / spec["config"]).read_text(encoding="utf-8"))
    manifest = verify_manifest_and_smoke(root, config, name)
    manifest_path = root / "data" / "splits.json"
    manifest_hash = file_hash(manifest_path)
    records, split = manifest["records"], manifest["split_indices"]
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    config_loaded, weights, model, weight_path = load_control(root, name, device)
    if config_loaded != config:
        raise ValueError("Loaded configuration differs from selected configuration")
    transform = weights.transforms()
    batches = []
    print(f"\nControl: {name}", flush=True)
    # Fixed batch=1: on the current CUDA/cuDNN stack, ResNet50 uses slightly
    # different kernels for batch=1 and batch=16. Per-image extraction keeps
    # each feature independent of the other images grouped with it.
    batch_size = 1
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        batches.append(extract_batch(batch, data_root, transform, model, device, spec["feature_dim"]))
        completed = start + len(batch)
        if completed == 1 or completed % 25 == 0 or completed == len(records):
            print(f"Processed: {completed}/{len(records)}", flush=True)
    features = np.concatenate(batches, axis=0)
    expected_shape = (620, spec["feature_dim"])
    if features.shape != expected_shape or features.dtype != np.float32 or not np.isfinite(features).all():
        raise RuntimeError("Invalid complete feature array")
    if np.any(np.linalg.norm(features.astype(np.float64), axis=1) == 0):
        raise RuntimeError("Zero-length feature in complete cache")
    frozen = all(not p.requires_grad for p in model.parameters())
    if not frozen or model.training or file_hash(manifest_path) != manifest_hash:
        raise RuntimeError("Frozen model or manifest consistency check failed")
    labels = np.array([r["class_id"] for r in records], dtype=np.int64)
    output = root / "cache" / f"control_{name}_features_{run_id}.npz"
    arrays = {
        "features": features,
        "labels": labels,
        "sample_indices": np.arange(620, dtype=np.int64),
        "categories": np.array(manifest["categories"]),
        "model_id": np.array(config["backbone"]),
        "model_revision": np.array(config["revision"]),
        "manifest_sha256": np.array(manifest_hash),
        "train_idx": np.array(split["train"], dtype=np.int64),
        "val_idx": np.array(split["validation"], dtype=np.int64),
        "test_idx": np.array(split["test"], dtype=np.int64),
    }
    save_cache(output, arrays)
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "features_cached_no_training_or_metrics",
        "control_name": name,
        "config": config,
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "script_sha256": file_hash(Path(__file__)),
        "helper_sha256": file_hash(root / "scripts/smoke_test_torchvision_controls.py"),
        "smoke_report_sha256": file_hash(root / config["smoke_report"]),
        "manifest_sha256": manifest_hash,
        "weights_sha256": file_hash(weight_path),
        "cache_relative": output.relative_to(root).as_posix(),
        "cache_sha256": file_hash(output),
        "feature_shape": list(features.shape),
        "label_shape": list(labels.shape),
        "dtype": str(features.dtype),
        "all_finite": True,
        "all_nonzero": True,
        "readback_identical": True,
        "split_counts": {k: len(split[k]) for k in ("train", "validation", "test")},
        "extraction_batch_size": batch_size,
        "all_official_weights_loaded_before_head_removal": True,
        "original_imagenet_head_used": False,
        "all_parameters_frozen": frozen,
        "trainable_parameter_count": 0,
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "feature_normalization": "none in cache; shared evaluator applies per-image L2",
        "transform": str(transform),
        "allow_tf32": False,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    report_path = root / "logs" / f"control_{name}_cache_{run_id}.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)
    print("All finite:", True)
    print("Readback identical:", True)
    print("All parameters frozen:", frozen)
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)
    print("Saved features:", output)
    print("Saved report:", report_path)
    del model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    print("Device:", device)
    print("Extraction batch size: 1 (fixed for per-image numerical consistency)")
    print("Loading local official checkpoints; no download, training, classification, or evaluation.")
    for name in ("resnet50", "convnext_tiny"):
        cache_one(root, args.data_root, name, device, run_id, now)


if __name__ == "__main__":
    main()
