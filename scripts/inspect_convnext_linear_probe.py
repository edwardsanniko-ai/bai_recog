"""Preflight the ConvNeXt-Tiny head-only CNN baseline without updating parameters."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader

from cnn_baseline_common import (
    BaiPatternDataset,
    CONFIG_NAME,
    build_transforms,
    file_hash,
    load_and_verify_manifest,
    load_config,
    load_head_only_model,
    set_seed,
    trainable_parameter_names,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()
    config = load_config(root)
    seed = int(config["development_seed"])
    set_seed(seed)
    manifest_path = root / config["split"]["manifest"]
    print("Verifying all 620 audited image hashes ...", flush=True)
    manifest = load_and_verify_manifest(root, args.data_root, config, verify_hashes=True)
    records, split = manifest["records"], manifest["split_indices"]
    train_transform, eval_transform = build_transforms(config)
    train_dataset = BaiPatternDataset(records, split["train"], args.data_root, train_transform)
    validation_dataset = BaiPatternDataset(records, split["validation"], args.data_root, eval_transform)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
    )
    images, labels, sample_indices = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weight_path = load_head_only_model(root, config, device)
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device))
    names = trainable_parameter_names(model)
    allowed_prefixes = ("classifier.0.", "classifier.3.")
    if not names or any(not name.startswith(allowed_prefixes) for name in names):
        raise RuntimeError(f"Unexpected trainable parameters: {names}")
    if tuple(images.shape) != (config["batch_size"], 3, config["image_size"], config["image_size"]):
        raise RuntimeError("Unexpected first-batch image shape")
    if tuple(logits.shape) != (config["batch_size"], config["num_classes"]):
        raise RuntimeError("Unexpected class-score shape")
    if not torch.isfinite(images).all() or not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite tensor found")
    train_ids = set(split["train"])
    if not set(sample_indices.tolist()) <= train_ids:
        raise RuntimeError("Non-training image entered the training batch")
    if train_ids & set(split["validation"]) or train_ids & set(split["test"]):
        raise RuntimeError("Fixed splits overlap")
    train_counts = Counter(records[index]["class_name"] for index in split["train"])
    validation_counts = Counter(records[index]["class_name"] for index in split["validation"])
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_count = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "head_only_preflight_passed_no_updates",
        "config_relative": f"configs/{CONFIG_NAME}",
        "config": config,
        "code_commit_before_run": git.stdout.strip() if git.returncode == 0 else None,
        "script_sha256": file_hash(Path(__file__)),
        "helper_sha256": file_hash(root / "scripts" / "cnn_baseline_common.py"),
        "manifest_sha256": file_hash(manifest_path),
        "weights_sha256": file_hash(weight_path),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "split_counts": {name: len(split[name]) for name in ("train", "validation", "test")},
        "train_class_counts": dict(train_counts),
        "validation_class_counts": dict(validation_counts),
        "first_batch_shape": list(images.shape),
        "first_batch_label_shape": list(labels.shape),
        "first_batch_indices": sample_indices.tolist(),
        "class_score_shape": list(logits.shape),
        "all_finite": True,
        "trainable_parameter_names": names,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": frozen_count,
        "backbone_frozen": True,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "validation_metrics_computed": False,
        "test_dataset_instantiated": False,
        "test_evaluated": False,
        "train_transform": str(train_transform),
        "validation_transform": str(eval_transform),
    }
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    destination = root / "logs" / f"cnn_convnext_linear_probe_preflight_{run_id}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Split sizes:", report["split_counts"])
    print("Train class counts:", report["train_class_counts"])
    print("Validation class counts:", report["validation_class_counts"])
    print("First batch images:", tuple(images.shape))
    print("First batch labels:", tuple(labels.shape))
    print("Class score shape:", tuple(logits.shape))
    print("Trainable parameters:", trainable_count)
    print("Frozen parameters:", frozen_count)
    print("Trainable names:", names)
    print("Backbone frozen:", True)
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)
    print("Saved report:", destination)


if __name__ == "__main__":
    main()
