"""Smoke-test frozen LowFormer-E2 (timm/lowformer_e2.in1k) on one audited training image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import hf_hub_download
import numpy as np
import PIL
from PIL import Image
from safetensors.torch import load_file
import timm
from timm.data import create_transform, resolve_model_data_config
import torch


CONFIG_NAME = "baseline_lowformer_e2.json"


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["allow_parameter_updates"] or not config["backbone_frozen"]:
        raise ValueError("本轮必须冻结LowFormer-E2模型且禁止参数更新。")
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


def load_model_and_transform(root: Path, config: dict, device: torch.device, local_files_only: bool = True):
    weights_path = Path(
        hf_hub_download(
            repo_id=config["weights_repo"],
            filename=config["weights_filename"],
            revision=config["requested_revision"],
            cache_dir=str(root / "hf_cache"),
            token=False,
            local_files_only=local_files_only,
        )
    )
    if weights_path.stat().st_size != config["weights_size_bytes"]:
        raise RuntimeError("LowFormer-E2权重文件大小不符合官方值。")
    weights_sha256 = file_hash(weights_path)
    if "weights_sha256" in config and weights_sha256 != config["weights_sha256"]:
        raise RuntimeError(f"LowFormer-E2权重SHA256不符合锁定值: {weights_sha256}")

    model = timm.create_model(config["model_name"], pretrained=False, num_classes=0)
    state = load_file(str(weights_path), device="cpu")
    expected = set(model.state_dict())
    missing = sorted(expected - set(state))
    extra = sorted(set(state) - expected)
    allowed_extra = ["head.classifier.bias", "head.classifier.weight", "classifier.bias", "classifier.weight"]
    unexpected_extra = [k for k in extra if k not in allowed_extra]
    if missing or unexpected_extra:
        raise RuntimeError(f"权重键不匹配: missing={missing}, unexpected_extra={unexpected_extra}")
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    return model, transform, data_config, weights_path, weights_sha256, extra


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
    print("Device:", device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model, transform, data_config, weights_path, weights_sha256, extra = load_model_and_transform(
        root, config, device, local_files_only=True
    )
    load_seconds = time.perf_counter() - t0

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = transform(image).unsqueeze(0)
    expected_input = (1, 3, config["expected_image_size"], config["expected_image_size"])
    if tuple(inputs.shape) != expected_input:
        raise RuntimeError(f"预处理输出形状错误: {tuple(inputs.shape)}")

    t1 = time.perf_counter()
    with torch.inference_mode():
        features = model(inputs.to(device)).detach().float().cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    infer_seconds = time.perf_counter() - t1

    expected_feature = (1, config["expected_feature_dim"])
    if features.shape != expected_feature or not np.isfinite(features).all():
        raise RuntimeError(f"特征形状或有限值检查失败: {features.shape}")

    frozen = all(not p.requires_grad for p in model.parameters())
    param_count = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated(device) / (2**30) if device.type == "cuda" else None

    print("Resolved revision:", config["requested_revision"])
    print("Excluded classifier head keys:", extra)
    print("Input shape:", tuple(inputs.shape))
    print("Feature shape:", tuple(features.shape), "dtype:", features.dtype)
    print("All finite:", True, "| All frozen:", frozen, "| Params:", param_count)
    print(f"Load: {load_seconds:.2f}s | Infer: {infer_seconds:.3f}s | Peak GPU GiB: {peak:.3f}")
    print("Optimizer updates:", 0, "| Test evaluated:", False)

    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "lowformer_e2_single_image_smoke_passed_no_training",
        "paper_model": config["paper_model"],
        "timm_model": config["model_name"],
        "weights_repo": config["weights_repo"],
        "requested_revision": config["requested_revision"],
        "resolved_revision": config["requested_revision"],
        "weights_sha256": weights_sha256,
        "excluded_classifier_keys": extra,
        "config_relative": f"configs/{CONFIG_NAME}",
        "config_sha256": file_hash(config_path),
        "script_sha256": file_hash(Path(__file__)),
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "manifest_sha256": file_hash(manifest_path),
        "sample_index": record["index"],
        "image_relative": record["relative_image"],
        "image_sha256": record["sha256"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "input_shape": list(inputs.shape),
        "feature_source": "official_timm_num_classes_0_pooled_pre_logits",
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "all_finite": True,
        "all_parameters_frozen": frozen,
        "parameter_count": param_count,
        "load_seconds": round(load_seconds, 4),
        "single_image_infer_seconds": round(infer_seconds, 4),
        "peak_gpu_gib": round(peak, 4) if peak is not None else None,
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "allow_tf32": False,
        "data_config": data_config,
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"lowformer_e2_smoke_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
