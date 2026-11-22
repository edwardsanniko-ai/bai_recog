"""Cache frozen nvidia/C-RADIOv4-SO400M summary embeddings in audited sample order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import einops
import numpy as np
import PIL
from PIL import Image
import torch
import transformers

from smoke_test_c_radiov4_so400m import (
    CONFIG_NAME,
    file_hash,
    load_model_and_processor,
    preprocess_images,
    validate_config,
    verified_snapshot,
)


def latest_smoke_report(root: Path, config: dict) -> tuple[Path, dict]:
    candidates = sorted((root / "logs").glob("c_radiov4_so400m_smoke_*.json"))
    for path in reversed(candidates):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "c_radiov4_so400m_single_image_smoke_passed_no_training"
            and report.get("paper_model") == config["paper_model"]
            and report.get("resolved_revision") == config["requested_revision"]
            and report.get("all_parameters_frozen") is True
            and report.get("optimizer_updates") == 0
            and report.get("test_evaluated") is False
        ):
            return path, report
    raise RuntimeError("没有找到与锁定版本一致的C-RADIOv4-SO400M单图预检记录。")


def save_cache(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise RuntimeError("缓存字段读回不一致。")
        for key, value in arrays.items():
            if not np.array_equal(saved[key], value):
                raise RuntimeError(f"缓存字段读回不一致: {key}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size必须为正整数。")

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    smoke_path, smoke = latest_smoke_report(root, config)
    snapshot_dir, weights = verified_snapshot(root, config)
    weights_sha256 = file_hash(weights)
    if weights_sha256 != smoke["weights_sha256"]:
        raise RuntimeError("本地权重哈希与预检记录不一致。")

    manifest_path = root / "data" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["records"]
    split = manifest["split_indices"]
    if len(records) != 620 or [record["index"] for record in records] != list(range(620)):
        raise RuntimeError("审计清单不是固定的620张样本顺序。")
    flattened = [index for name in ("train", "validation", "test") for index in split[name]]
    if sorted(flattened) != list(range(620)):
        raise RuntimeError("固定划分存在遗漏或重复。")
    if [len(split[name]) for name in ("train", "validation", "test")] != [434, 93, 93]:
        raise RuntimeError("固定划分数量不符合434/93/93。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Paper model:", config["paper_model"], flush=True)
    print("Resolved revision:", smoke["resolved_revision"])
    print("Device:", device, "| batch_size:", args.batch_size)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, processor = load_model_and_processor(snapshot_dir, device, config)

    t0 = time.perf_counter()
    batches: list[np.ndarray] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        pil_images = []
        for record in batch:
            path = args.data_root / record["relative_image"]
            if not path.is_file() or file_hash(path) != record["sha256"]:
                raise RuntimeError(f"图片与审计清单不一致: {record['relative_image']}")
            with Image.open(path) as source:
                pil_images.append(source.convert("RGB"))
        inputs = preprocess_images(pil_images, processor, config["expected_image_size"])
        expected = (len(batch), 3, config["expected_image_size"], config["expected_image_size"])
        if tuple(inputs.shape) != expected:
            raise RuntimeError("批量预处理输出形状不符合锁定配置。")
        with torch.inference_mode():
            summary, _ = model(inputs.to(device))
            features = summary.detach().float().cpu().numpy()
        if features.shape != (len(batch), config["expected_feature_dim"]):
            raise RuntimeError("批量C-RADIOv4 summary维度错误。")
        if not np.isfinite(features).all():
            raise RuntimeError("批量特征包含非有限数值。")
        batches.append(features)
        done = start + len(batch)
        if done % 40 == 0 or done == len(records):
            print(f"Processed: {done}/620 ({time.perf_counter() - t0:.1f}s)", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    extract_seconds = time.perf_counter() - t0
    peak_vram_gib = (
        torch.cuda.max_memory_allocated(device) / (2**30) if device.type == "cuda" else None
    )

    features = np.concatenate(batches, axis=0)
    labels = np.asarray([record["class_id"] for record in records], dtype=np.int64)
    if features.shape != (620, config["expected_feature_dim"]) or labels.shape != (620,):
        raise RuntimeError("完整缓存形状错误。")

    now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai"))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    output = root / "cache" / f"c_radiov4_so400m_features_{stamp}.npz"
    arrays = {
        "features": features,
        "labels": labels,
        "sample_indices": np.arange(620, dtype=np.int64),
        "categories": np.asarray(manifest["categories"]),
        "model_id": np.asarray(config["paper_model"]),
        "model_revision": np.asarray(smoke["resolved_revision"]),
        "manifest_sha256": np.asarray(file_hash(manifest_path)),
        "train_idx": np.asarray(split["train"], dtype=np.int64),
        "val_idx": np.asarray(split["validation"], dtype=np.int64),
        "test_idx": np.asarray(split["test"], dtype=np.int64),
    }
    save_cache(output, arrays)
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report = {
        "timestamp": now.isoformat(),
        "status": "c_radiov4_so400m_features_cached_no_training_or_metrics",
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
        "manifest_sha256": file_hash(manifest_path),
        "cache_relative": output.relative_to(root).as_posix(),
        "cache_sha256": file_hash(output),
        "feature_source": "official_c_radiov4_summary_output",
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "label_shape": list(labels.shape),
        "all_finite": True,
        "readback_identical": True,
        "split_counts": {name: len(indices) for name, indices in split.items()},
        "extraction_batch_size": args.batch_size,
        "extraction_seconds": round(extract_seconds, 2),
        "peak_gpu_gib": round(peak_vram_gib, 4) if peak_vram_gib is not None else None,
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "feature_normalization": "none during caching; locked comparison applies per-sample L2",
        "allow_tf32": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "einops_version": einops.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"c_radiov4_so400m_cache_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)
    print(f"Extraction time: {extract_seconds:.2f}s | Peak GPU GiB: {peak_vram_gib:.3f}")
    print("All finite:", True)
    print("Readback identical:", True)
    print("All parameters frozen:", report["all_parameters_frozen"])
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)
    print("Saved features:", output)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
