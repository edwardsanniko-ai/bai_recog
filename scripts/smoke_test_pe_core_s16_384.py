"""Download and smoke-test frozen PE-Core-S16-384 on one audited training image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import PIL
from PIL import Image
import timm
from timm.data import create_transform, resolve_model_data_config
import torch


CONFIG_NAME = "baseline_pe_core_s16_384.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["allow_parameter_updates"] or not config["backbone_frozen"]:
        raise ValueError("本轮必须冻结PE模型且禁止参数更新。")
    if config["test_split_allowed"]:
        raise ValueError("候选筛选阶段禁止访问测试集。")
    if timm.__version__ != config["implementation_version"]:
        raise RuntimeError(f"需要timm {config['implementation_version']}，当前为{timm.__version__}。")


def audited_training_image(root: Path, data_root: Path) -> tuple[Path, Path, dict, dict]:
    manifest_path = root / "data" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = manifest["split_indices"]["train"][0]
    record = manifest["records"][index]
    image_path = data_root / record["relative_image"]
    if not image_path.is_file():
        raise FileNotFoundError("找不到预检图片；请用--data-root指向bai_data。")
    if file_hash(image_path) != record["sha256"]:
        raise RuntimeError("预检图片与审计清单不一致。")
    return manifest_path, image_path, record, manifest


def verified_weights(root: Path, config: dict) -> Path:
    path = root / config["local_weights"]
    if not path.is_file():
        raise FileNotFoundError(
            "找不到已校验的PE权重。请先运行scripts/download_pe_core_s16_384.ps1。"
        )
    if path.stat().st_size != config["weights_size_bytes"]:
        raise RuntimeError("PE权重文件大小不符合锁定值，请重新运行下载脚本断点续传。")
    actual_hash = file_hash(path)
    if actual_hash != config["weights_sha256"]:
        raise RuntimeError(f"PE权重SHA256不符合官方值: {actual_hash}")
    return path


def validate_model(model: torch.nn.Module, config: dict) -> dict:
    data_config = resolve_model_data_config(model)
    expected_data = {
        "input_size": (3, config["expected_image_size"], config["expected_image_size"]),
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "crop_pct": 1.0,
    }
    differences = {
        key: {"expected": value, "actual": data_config.get(key)}
        for key, value in expected_data.items() if data_config.get(key) != value
    }
    if differences:
        raise RuntimeError(f"官方预处理配置不符合锁定值: {differences}")
    if getattr(model.patch_embed, "patch_size", None) != (config["expected_patch_size"],) * 2:
        raise RuntimeError("模型patch size不符合S/16。")
    if getattr(model, "embed_dim", None) != config["expected_backbone_width"]:
        raise RuntimeError("视觉主干宽度不符合PE-Core-S。")
    if model.num_classes != config["expected_feature_dim"]:
        raise RuntimeError("最终投影维度不符合官方512维CLIP嵌入。")
    if model.pretrained_cfg.get("hf_hub_id") != config["weights_repo"]:
        raise RuntimeError("timm权重仓库与锁定配置不一致。")
    return data_config


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()
    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    manifest_path, image_path, record, _ = audited_training_image(root, args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Paper model:", config["paper_model"], flush=True)
    print("timm model:", config["model_name"])
    print("Weights repo:", config["weights_repo"])
    print("Image:", image_path)
    print("Device:", device)
    weights = verified_weights(root, config)
    print("Loading the pinned, SHA256-verified local checkpoint; no network access.", flush=True)
    model = timm.create_model(
        config["model_name"], pretrained=False, checkpoint_path=str(weights)
    ).eval().to(device)
    model.requires_grad_(False)
    data_config = validate_model(model, config)
    transform = create_transform(**data_config, is_training=False)
    revision = config["requested_revision"]
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = transform(image).unsqueeze(0)
    expected_input = (1, 3, config["expected_image_size"], config["expected_image_size"])
    if tuple(inputs.shape) != expected_input:
        raise RuntimeError(f"预处理输出形状错误: {tuple(inputs.shape)}")
    with torch.inference_mode():
        features = model(inputs.to(device)).detach().float().cpu().numpy()
    expected_feature = (1, config["expected_feature_dim"])
    if features.shape != expected_feature or not np.isfinite(features).all():
        raise RuntimeError("最终PE图像嵌入的形状或有限值检查失败。")
    frozen = all(not parameter.requires_grad for parameter in model.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Resolved revision:", revision)
    print("Input shape:", tuple(inputs.shape))
    print("Feature source: official final attention-pooled and projected image embedding")
    print("Feature shape:", tuple(features.shape))
    print("Feature dtype:", features.dtype)
    print("All finite:", bool(np.isfinite(features).all()))
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
        "status": "pe_core_s16_384_single_image_smoke_passed_no_training",
        "paper_model": config["paper_model"],
        "timm_model": config["model_name"],
        "weights_repo": config["weights_repo"],
        "requested_revision": config["requested_revision"],
        "resolved_revision": revision,
        "weights_sha256": file_hash(weights),
        "config_relative": f"configs/{CONFIG_NAME}",
        "config_sha256": file_hash(config_path),
        "script_sha256": file_hash(Path(__file__)),
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "manifest_sha256": file_hash(manifest_path),
        "sample_index": record["index"],
        "image_relative": record["relative_image"],
        "image_sha256": record["sha256"],
        "device": str(device),
        "input_shape": list(inputs.shape),
        "feature_source": "official_final_attention_pooled_projected_image_embedding",
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "all_finite": True,
        "all_parameters_frozen": frozen,
        "parameter_count": parameter_count,
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "allow_tf32": False,
        "data_config": data_config,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"pe_core_s16_384_smoke_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
