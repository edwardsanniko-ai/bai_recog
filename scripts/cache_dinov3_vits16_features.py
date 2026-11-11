"""Cache frozen DINOv3 ViT-S/16 CLS features in audited sample order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoImageProcessor, AutoModel

from smoke_test_dinov3_vits16 import CONFIG_NAME, file_hash, validate_config, validate_loaded_model


def latest_smoke_report(root: Path, config: dict) -> tuple[Path, dict]:
    candidates = sorted((root / "logs").glob("dinov3_vits16_smoke_*.json"))
    for path in reversed(candidates):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "dinov3_vits16_single_image_smoke_passed_no_training"
            and report.get("model") == config["backbone"]
            and report.get("all_parameters_frozen") is True
            and report.get("optimizer_updates") == 0
            and report.get("test_evaluated") is False
        ):
            return path, report
    raise RuntimeError("没有找到通过的 DINOv3 ViT-S/16 单图预检记录。请先运行 smoke test。")


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
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size 必须为正整数。")

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    smoke_path, smoke = latest_smoke_report(root, config)
    revision = smoke["resolved_revision"]
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
    kwargs = {
        "revision": revision,
        "cache_dir": str(root / "hf_cache"),
        "token": False,
        "trust_remote_code": False,
        "local_files_only": True,
    }
    print("Model:", config["backbone"], flush=True)
    print("Resolved revision:", revision)
    print("Device:", device)
    print("Loading the smoke-tested local checkpoint; no download or training.", flush=True)
    processor = AutoImageProcessor.from_pretrained(config["backbone"], use_fast=True, **kwargs)
    model = AutoModel.from_pretrained(config["backbone"], **kwargs).eval().to(device)
    model.requires_grad_(False)
    validate_loaded_model(model, config)

    batches: list[np.ndarray] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        images = []
        for record in batch:
            path = args.data_root / record["relative_image"]
            if file_hash(path) != record["sha256"]:
                raise RuntimeError(f"图片与审计清单不一致: {record['relative_image']}")
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        expected = (len(batch), 3, config["expected_image_size"], config["expected_image_size"])
        if tuple(inputs["pixel_values"].shape) != expected:
            raise RuntimeError("批量预处理输出形状不符合锁定配置。")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            features = model(**inputs).pooler_output.detach().float().cpu().numpy()
        if features.shape != (len(batch), config["expected_feature_dim"]):
            raise RuntimeError("批量 CLS 特征维度错误。")
        if not np.isfinite(features).all():
            raise RuntimeError("批量特征包含非有限数值。")
        batches.append(features)
        print(f"Processed: {start + len(batch)}/620", flush=True)

    features = np.concatenate(batches, axis=0)
    labels = np.asarray([record["class_id"] for record in records], dtype=np.int64)
    if features.shape != (620, config["expected_feature_dim"]) or labels.shape != (620,):
        raise RuntimeError("完整缓存形状错误。")
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    output = root / "cache" / f"dinov3_vits16_features_{stamp}.npz"
    arrays = {
        "features": features,
        "labels": labels,
        "sample_indices": np.arange(620, dtype=np.int64),
        "categories": np.asarray(manifest["categories"]),
        "model_id": np.asarray(config["backbone"]),
        "model_revision": np.asarray(revision),
        "manifest_sha256": np.asarray(file_hash(manifest_path)),
        "train_idx": np.asarray(split["train"], dtype=np.int64),
        "val_idx": np.asarray(split["validation"], dtype=np.int64),
        "test_idx": np.asarray(split["test"], dtype=np.int64),
    }
    save_cache(output, arrays)
    snapshot = root / "hf_cache" / ("models--" + config["backbone"].replace("/", "--")) / "snapshots" / revision
    weights_path = snapshot / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"找不到已加载权重文件: {weights_path}")
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report = {
        "timestamp": now.isoformat(),
        "status": "dinov3_vits16_features_cached_no_training_or_metrics",
        "model": config["backbone"],
        "resolved_revision": revision,
        "config_relative": f"configs/{CONFIG_NAME}",
        "config_sha256": file_hash(config_path),
        "script_sha256": file_hash(Path(__file__)),
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "smoke_report_relative": smoke_path.relative_to(root).as_posix(),
        "smoke_report_sha256": file_hash(smoke_path),
        "manifest_sha256": file_hash(manifest_path),
        "weights_sha256": file_hash(weights_path),
        "cache_relative": output.relative_to(root).as_posix(),
        "cache_sha256": file_hash(output),
        "feature_source": "pooler_output_final_normalized_cls_token",
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "label_shape": list(labels.shape),
        "all_finite": True,
        "readback_identical": True,
        "split_counts": {name: len(indices) for name, indices in split.items()},
        "extraction_batch_size": args.batch_size,
        "all_parameters_frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "feature_normalization": "none; raw final normalized CLS token",
        "allow_tf32": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    report_path = root / "logs" / f"dinov3_vits16_cache_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Feature shape:", features.shape)
    print("Label shape:", labels.shape)
    print("All finite:", True)
    print("Readback identical:", True)
    print("All parameters frozen:", report["all_parameters_frozen"])
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)
    print("Saved features:", output)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
