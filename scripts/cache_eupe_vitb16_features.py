"""Cache frozen EUPE ViT-B/16 features in audited sample order; no training or classification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
import torchvision

from smoke_test_eupe_vitb16 import CONFIG_NAME, file_hash, image_features, load_model, make_transform


def latest_smoke_report(root: Path, config: dict) -> tuple[Path, dict]:
    candidates = sorted((root / "logs").glob("eupe_vitb16_smoke_*.json"))
    for path in reversed(candidates):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "eupe_vitb16_single_image_smoke_passed_no_classification"
            and report.get("paper_model") == config["paper_model"]
            and report.get("resolved_revision") == config["revision"]
            and report.get("all_parameters_frozen") is True
            and report.get("optimizer_updates") == 0
            and report.get("test_evaluated") is False
        ):
            return path, report
    raise RuntimeError("No verified EUPE ViT-B/16 smoke report found")


def save_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError("Cache readback differs from extracted arrays")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke_path, smoke = latest_smoke_report(root, config)

    manifest_path = root / "data/splits.json"
    manifest_hash = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, split = manifest["records"], manifest["split_indices"]
    if len(records) != 620 or [r["index"] for r in records] != list(range(620)):
        raise ValueError("Expected 620 records in audited sample order")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, _, weights_sha256, excluded = load_model(root, config, device, local_files_only=True)
    if weights_sha256 != smoke["weights_sha256"]:
        raise RuntimeError("Weights hash mismatch with smoke report")
    transform = make_transform(config["image_size"])

    batches = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        tensors = []
        for record in batch:
            path = args.data_root / record["relative_image"]
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"Image changed: sample {record['index']}")
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        inputs = torch.stack(tensors).to(device)
        batches.append(image_features(model, inputs).numpy())
        done = start + len(batch)
        if done % 160 == 0 or done == len(records):
            print(f"Processed: {done}/{len(records)} ({time.perf_counter() - started:.1f}s)", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    features = np.concatenate(batches, axis=0)
    labels = np.array([r["class_id"] for r in records], dtype=np.int64)
    now = datetime.now(timezone(timedelta(hours=8)))
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    output = root / "cache" / f"eupe_vitb16_features_{run_id}.npz"
    arrays = dict(
        features=features,
        labels=labels,
        sample_indices=np.arange(620, dtype=np.int64),
        categories=np.array(manifest["categories"]),
        model_id=np.array(config["backbone"]),
        model_revision=np.array(config["revision"]),
        manifest_sha256=np.array(manifest_hash),
        train_idx=np.array(split["train"], dtype=np.int64),
        val_idx=np.array(split["validation"], dtype=np.int64),
        test_idx=np.array(split["test"], dtype=np.int64),
    )
    save_cache(output, arrays)
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = dict(
        timestamp=now.isoformat(),
        status="eupe_vitb16_features_cached_no_training_or_metrics",
        paper_model=config["paper_model"],
        resolved_revision=config["revision"],
        config_relative=f"configs/{CONFIG_NAME}",
        config_sha256=file_hash(config_path),
        script_sha256=file_hash(Path(__file__)),
        smoke_report_relative=smoke_path.relative_to(root).as_posix(),
        smoke_report_sha256=file_hash(smoke_path),
        manifest_sha256=manifest_hash,
        weights_sha256=weights_sha256,
        code_commit=git.stdout.strip() if git.returncode == 0 else None,
        cache_relative=output.relative_to(root).as_posix(),
        cache_sha256=file_hash(output),
        feature_source="official_eupe_vitb16_x_norm_clstoken",
        feature_shape=list(features.shape),
        label_shape=list(labels.shape),
        dtype=str(features.dtype),
        all_finite=True,
        readback_identical=True,
        split_counts={k: len(split[k]) for k in ("train", "validation", "test")},
        extraction_batch_size=args.batch_size,
        all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
        excluded_distillation_projector_keys=excluded,
        optimizer_updates=0,
        test_evaluated=False,
        classification_performed=False,
        allow_tf32=False,
        torch_version=torch.__version__,
        torchvision_version=torchvision.__version__,
        device=str(device),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        peak_allocated_gpu_gib=round(peak, 4) if peak is not None else None,
        elapsed_seconds=round(elapsed, 3),
    )
    report_path = root / "logs" / f"eupe_vitb16_cache_{run_id}.json"
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("Feature shape:", features.shape, "| Elapsed:", f"{elapsed:.2f}s", "| Peak GPU GiB:", round(peak, 3))
    print("Saved features:", output)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
