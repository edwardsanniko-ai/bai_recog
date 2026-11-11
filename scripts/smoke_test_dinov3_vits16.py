"""Download and smoke-test frozen DINOv3 ViT-S/16 on one audited training image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch
import transformers
from transformers import AutoImageProcessor, AutoModel


CONFIG_NAME = "baseline_dinov3_vits16.json"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> tuple[Path, dict]:
    path = root / "data" / "splits.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def audited_training_image(manifest: dict, data_root: Path) -> tuple[Path, dict]:
    index = manifest["split_indices"]["train"][0]
    record = manifest["records"][index]
    path = data_root / record["relative_image"]
    if not path.is_file():
        raise FileNotFoundError("找不到预检图片；请用 --data-root 指向 bai_data。")
    if file_hash(path) != record["sha256"]:
        raise RuntimeError("预检图片与审计清单不一致。")
    return path, record


def validate_config(config: dict) -> None:
    if config["backbone"] != MODEL_ID:
        raise ValueError("DINOv3 ViT-S/16 模型 ID 不符合已锁定配置。")
    if config["allow_parameter_updates"] or not config["backbone_frozen"]:
        raise ValueError("本轮必须冻结模型且禁止参数更新。")
    if config["test_split_allowed"]:
        raise ValueError("候选筛选阶段禁止访问测试集。")


def resolved_revision(model: torch.nn.Module) -> str:
    revision = getattr(model.config, "_commit_hash", None)
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise RuntimeError("无法从已下载模型读取精确 Hugging Face 提交号。")
    return revision


def validate_loaded_model(model: torch.nn.Module, config: dict) -> None:
    actual = model.config
    expected = {
        "model_type": config["expected_model_type"],
        "image_size": config["expected_image_size"],
        "patch_size": config["expected_patch_size"],
        "hidden_size": config["expected_feature_dim"],
    }
    differences = {
        name: {"expected": value, "actual": getattr(actual, name, None)}
        for name, value in expected.items()
        if getattr(actual, name, None) != value
    }
    if differences:
        raise RuntimeError(f"模型结构不符合 ViT-S/16 配置: {differences}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    manifest_path, manifest = load_manifest(root)
    image_path, record = audited_training_image(manifest, args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Model:", config["backbone"], flush=True)
    print("Architecture: DINOv3 ViT-S/16 (Transformer, patch size 16)")
    print("Image:", image_path)
    print("Device:", device)
    print("First run downloads the official gated checkpoint to local hf_cache.", flush=True)
    kwargs = {
        "revision": config["requested_revision"],
        "cache_dir": str(root / "hf_cache"),
        "token": True,
        "trust_remote_code": False,
    }
    processor = AutoImageProcessor.from_pretrained(config["backbone"], use_fast=True, **kwargs)
    model = AutoModel.from_pretrained(config["backbone"], **kwargs).eval().to(device)
    model.requires_grad_(False)
    validate_loaded_model(model, config)
    revision = resolved_revision(model)

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    expected_input = (1, 3, config["expected_image_size"], config["expected_image_size"])
    if tuple(inputs["pixel_values"].shape) != expected_input:
        raise RuntimeError(f"预处理输出形状错误: {tuple(inputs['pixel_values'].shape)}")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        features = outputs.pooler_output
    array = features.detach().float().cpu().numpy()
    expected_feature = (1, config["expected_feature_dim"])
    if array.shape != expected_feature or not np.isfinite(array).all():
        raise RuntimeError("CLS 特征形状或有限值检查失败。")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    frozen = all(not parameter.requires_grad for parameter in model.parameters())
    print("Resolved revision:", revision)
    print("Input shape:", tuple(inputs["pixel_values"].shape))
    print("Feature source: final normalized CLS token (pooler_output)")
    print("Feature shape:", tuple(array.shape))
    print("Feature dtype:", array.dtype)
    print("All finite:", bool(np.isfinite(array).all()))
    print("All parameters frozen:", frozen)
    print("Parameter count:", parameter_count)
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)
    if device.type == "cuda":
        print(f"Allocated GPU GiB: {torch.cuda.memory_allocated(device) / 2**30:.3f}")

    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report = {
        "timestamp": now.isoformat(),
        "status": "dinov3_vits16_single_image_smoke_passed_no_training",
        "model": config["backbone"],
        "requested_revision": config["requested_revision"],
        "resolved_revision": revision,
        "config_relative": f"configs/{CONFIG_NAME}",
        "config_sha256": file_hash(config_path),
        "script_sha256": file_hash(Path(__file__)),
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "manifest_sha256": file_hash(manifest_path),
        "sample_index": record["index"],
        "image_relative": record["relative_image"],
        "image_sha256": record["sha256"],
        "device": str(device),
        "input_shape": list(inputs["pixel_values"].shape),
        "feature_source": "pooler_output_final_normalized_cls_token",
        "feature_shape": list(array.shape),
        "feature_dtype": str(array.dtype),
        "all_finite": True,
        "all_parameters_frozen": frozen,
        "parameter_count": parameter_count,
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "allow_tf32": False,
        "use_fast": True,
        "processor": processor.to_dict(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    report_path = root / "logs" / f"dinov3_vits16_smoke_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
