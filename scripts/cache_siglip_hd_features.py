"""Cache frozen LiheYoung/SigLIP-HD embeddings in audited sample order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import PIL
from PIL import Image
import torch
import transformers

from smoke_test_siglip_hd import (
    CONFIG_NAME,
    extract_whole_image_embedding,
    file_hash,
    load_model_and_processor,
    validate_config,
)


def latest_smoke_report(root: Path, config: dict) -> tuple[Path, dict]:
    candidates = sorted((root / "logs").glob("siglip_hd_smoke_*.json"))
    for path in reversed(candidates):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "siglip_hd_single_image_smoke_passed_no_training"
            and report.get("paper_model") == config["paper_model"]
            and report.get("resolved_revision") == config["requested_revision"]
            and report.get("all_parameters_frozen") is True
            and report.get("optimizer_updates") == 0
            and report.get("test_evaluated") is False
        ):
            return path, report
    raise RuntimeError("没有找到与锁定版本一致的SigLIP-HD单图预检记录。")


def save_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError("缓存字段读回不一致。")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    smoke_path, smoke = latest_smoke_report(root, config)

    manifest_path = root / "data" / "splits.json"
    manifest_hash = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records, split = manifest["records"], manifest["split_indices"]
    if len(records) != 620 or [r["index"] for r in records] != list(range(620)):
        raise RuntimeError("审计清单不是固定的620张样本顺序。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, processor, weights_sha256 = load_model_and_processor(root, config, device, local_files_only=True)
    if weights_sha256 != smoke["weights_sha256"]:
        raise RuntimeError("本地权重哈希与预检记录不一致。")

    t0 = time.perf_counter()
    batches: list[np.ndarray] = []
    feat_source = None
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        pil_images = []
        for record in batch:
            path = args.data_root / record["relative_image"]
            if not path.is_file() or file_hash(path) != record["sha256"]:
                raise RuntimeError(f"图片与审计清单不一致: {record['relative_image']}")
            with Image.open(path) as source:
                pil_images.append(source.convert("RGB"))
        inputs = processor(images=pil_images, return_tensors="pt").pixel_values
        feats, feat_source = extract_whole_image_embedding(model, inputs.to(device))
        if feats.shape != (len(batch), config["expected_feature_dim"]) or not np.isfinite(feats).all():
            raise RuntimeError("批量特征形状或有限值检查失败。")
        batches.append(feats)
        done = start + len(batch)
        if done % 80 == 0 or done == len(records):
            print(f"Processed: {done}/620 ({time.perf_counter() - t0:.1f}s)", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device) / (2**30) if device.type == "cuda" else None

    features = np.concatenate(batches, axis=0)
    labels = np.asarray([r["class_id"] for r in records], dtype=np.int64)
    now = datetime.now(timezone(timedelta(hours=8)))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    output = root / "cache" / f"siglip_hd_features_{stamp}.npz"
    arrays = {
        "features": features,
        "labels": labels,
        "sample_indices": np.arange(620, dtype=np.int64),
        "categories": np.asarray(manifest["categories"]),
        "model_id": np.asarray(config["paper_model"]),
        "model_revision": np.asarray(smoke["resolved_revision"]),
        "manifest_sha256": np.asarray(manifest_hash),
        "train_idx": np.asarray(split["train"], dtype=np.int64),
        "val_idx": np.asarray(split["validation"], dtype=np.int64),
        "test_idx": np.asarray(split["test"], dtype=np.int64),
    }
    save_cache(output, arrays)
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "siglip_hd_features_cached_no_training_or_metrics",
        "paper_model": config["paper_model"],
        "model_repo": config["model_repo"],
        "resolved_revision": smoke["resolved_revision"],
        "weights_sha256": weights_sha256,
        "config_relative": f"configs/{CONFIG_NAME}",
        "config_sha256": file_hash(config_path),
        "script_sha256": file_hash(Path(__file__)),
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "smoke_report_relative": smoke_path.relative_to(root).as_posix(),
        "smoke_report_sha256": file_hash(smoke_path),
        "manifest_sha256": manifest_hash,
        "cache_relative": output.relative_to(root).as_posix(),
        "cache_sha256": file_hash(output),
        "feature_source": feat_source,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "label_shape": list(labels.shape),
        "all_finite": True,
        "readback_identical": True,
        "split_counts": {k: len(v) for k, v in split.items()},
        "extraction_batch_size": args.batch_size,
        "extraction_seconds": round(elapsed, 2),
        "peak_gpu_gib": round(peak, 4) if peak is not None else None,
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "allow_tf32": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"siglip_hd_cache_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Feature shape:", features.shape, "| Elapsed:", f"{elapsed:.2f}s", "| Peak GPU GiB:", round(peak, 3))
    print("Saved features:", output)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
