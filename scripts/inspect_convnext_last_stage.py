"""Preflight phase-B ConvNeXt last-stage fine-tuning without parameter updates."""

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
    build_transforms,
    file_hash,
    load_and_verify_manifest,
    set_seed,
)
from cnn_last_stage_common import (
    CONFIG_NAME,
    load_last_stage_config,
    load_last_stage_model,
    phase_b_parameter_groups,
    phase_b_parameter_names,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_last_stage_config(root)
    seed = int(config["development_seed"] if args.seed is None else args.seed)
    allowed_seeds = {int(config["development_seed"]), *map(int, config["confirmatory_seeds_after_protocol_lock"])}
    if seed not in allowed_seeds:
        raise ValueError(f"Seed {seed} is outside the locked seed set: {sorted(allowed_seeds)}")
    run_role = "development" if seed == int(config["development_seed"]) else "confirmatory"
    set_seed(seed)
    print("Verifying all 620 audited image hashes ...", flush=True)
    manifest = load_and_verify_manifest(root, args.data_root, config, verify_hashes=True)
    records, split = manifest["records"], manifest["split_indices"]
    train_transform, eval_transform = build_transforms(config)
    train_dataset = BaiPatternDataset(records, split["train"], args.data_root, train_transform)
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
    model, weight_path, checkpoint_path, checkpoint = load_last_stage_model(
        root, config, device, source_seed=seed
    )
    last_stage, head = phase_b_parameter_groups(model)
    names = phase_b_parameter_names(model)
    if any(not name.startswith(("features.7.", "classifier.0.", "classifier.3.")) for name in names):
        raise RuntimeError(f"Unexpected trainable parameter: {names}")
    if any(value.requires_grad for value in model.features[:7].parameters()):
        raise RuntimeError("A module before features.7 is trainable")
    model.eval()
    with torch.inference_mode():
        logits = model(images.to(device))
    if tuple(images.shape) != (config["batch_size"], 3, config["image_size"], config["image_size"]):
        raise RuntimeError("Unexpected first-batch image shape")
    if tuple(logits.shape) != (config["batch_size"], config["num_classes"]):
        raise RuntimeError("Unexpected class-score shape")
    if not torch.isfinite(images).all() or not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite tensor found")
    if not set(sample_indices.tolist()) <= set(split["train"]):
        raise RuntimeError("A non-training sample entered the training batch")
    trainable_count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    frozen_count = sum(value.numel() for value in model.parameters() if not value.requires_grad)
    last_stage_count = sum(value.numel() for value in last_stage)
    head_count = sum(value.numel() for value in head)
    if trainable_count != last_stage_count + head_count:
        raise RuntimeError("Parameter-group counts do not sum to all trainable parameters")
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "phase_b_last_stage_preflight_passed_no_updates",
        "source_seed": seed,
        "run_role": run_role,
        "config_relative": f"configs/{CONFIG_NAME}",
        "config": config,
        "code_commit_before_run": git.stdout.strip() if git.returncode == 0 else None,
        "script_sha256": file_hash(Path(__file__)),
        "helper_sha256": file_hash(root / "scripts" / "cnn_last_stage_common.py"),
        "shared_helper_sha256": file_hash(root / "scripts" / "cnn_baseline_common.py"),
        "manifest_sha256": file_hash(root / config["split"]["manifest"]),
        "weights_sha256": file_hash(weight_path),
        "phase_a_checkpoint_sha256": file_hash(checkpoint_path),
        "phase_a_checkpoint_epoch": checkpoint["completed_epoch"],
        "phase_a_checkpoint_validation_macro_f1": checkpoint["best_validation_macro_f1"],
        "split_counts": {name: len(split[name]) for name in ("train", "validation", "test")},
        "train_class_counts": dict(Counter(records[index]["class_name"] for index in split["train"])),
        "first_batch_shape": list(images.shape),
        "first_batch_label_shape": list(labels.shape),
        "first_batch_indices": sample_indices.tolist(),
        "class_score_shape": list(logits.shape),
        "all_finite": True,
        "trainable_parameter_names": names,
        "trainable_parameter_count": trainable_count,
        "last_stage_parameter_count": last_stage_count,
        "head_parameter_count": head_count,
        "frozen_parameter_count": frozen_count,
        "frozen_module_range": "features.0 through features.6",
        "trainable_module_range": "features.7 plus classifier.0 and classifier.3",
        "intended_optimizer_groups": [
            {"name": "last_stage", "parameter_count": last_stage_count, "learning_rate": config["last_stage_learning_rate"]},
            {"name": "head", "parameter_count": head_count, "learning_rate": config["head_learning_rate"]}
        ],
        "optimizer_created": False,
        "backward_performed": False,
        "optimizer_updates": 0,
        "validation_metrics_computed": False,
        "test_dataset_instantiated": False,
        "test_evaluated": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "train_transform": str(train_transform),
        "validation_transform": str(eval_transform),
    }
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    if run_role == "development":
        destination = root / "logs" / f"cnn_convnext_last_stage_preflight_{run_id}.json"
    else:
        destination = root / "logs" / f"cnn_convnext_last_stage_seed{seed}_preflight_{run_id}.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Source phase-A best epoch:", checkpoint["completed_epoch"])
    print("Source phase-A validation Macro-F1:", checkpoint["best_validation_macro_f1"])
    print("Split sizes:", report["split_counts"])
    print("First batch images:", tuple(images.shape))
    print("Class score shape:", tuple(logits.shape))
    print("Last-stage trainable parameters:", last_stage_count)
    print("Head trainable parameters:", head_count)
    print("Total trainable parameters:", trainable_count)
    print("Frozen parameters:", frozen_count)
    print("Trainable name prefixes valid:", True)
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)
    print("Saved report:", destination)


if __name__ == "__main__":
    main()
