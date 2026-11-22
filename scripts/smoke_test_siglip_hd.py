"""Smoke-test frozen LiheYoung/SigLIP-HD (ICLR 2026) on one audited training image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import numpy as np
import PIL
from PIL import Image
import torch
import transformers
from transformers import AutoImageProcessor, SiglipVisionModel


CONFIG_NAME = "baseline_siglip_hd.json"


def file_hash(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["allow_parameter_updates"] or not config["backbone_frozen"]:
        raise ValueError("本轮必须冻结SigLIP-HD模型且禁止参数更新。")
    if config["test_split_allowed"]:
        raise ValueError("候选筛选阶段禁止访问测试集。")
    if transformers.__version__ != config["implementation_version"]:
        raise RuntimeError(f"需要transformers {config['implementation_version']}，当前为{transformers.__version__}。")


def load_model_and_processor(root: Path, config: dict, device: torch.device, local_files_only: bool = True):
    snapshot_dir = Path(
        snapshot_download(
            repo_id=config["model_repo"],
            revision=config["requested_revision"],
            cache_dir=str(root / "hf_cache"),
            token=False,
            local_files_only=local_files_only,
        )
    )
    weights = snapshot_dir / config["weights_filename"]
    if weights.stat().st_size != config["weights_size_bytes"]:
        raise RuntimeError("SigLIP-HD权重文件大小不符合官方值。")
    weights_sha256 = file_hash(weights)
    if "weights_sha256" in config and weights_sha256 != config["weights_sha256"]:
        raise RuntimeError("SigLIP-HD权重SHA256不符合锁定值。")

    processor = AutoImageProcessor.from_pretrained(str(snapshot_dir), local_files_only=True)
    model = SiglipVisionModel.from_pretrained(str(snapshot_dir), local_files_only=True).eval().to(device)
    model.requires_grad_(False)
    return model, processor, weights_sha256


def extract_whole_image_embedding(model, pixel_values: torch.Tensor) -> tuple[np.ndarray, str]:
    with torch.inference_mode():
        outputs = model(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            feats = outputs.pooler_output
            source = "official_siglip_vision_model_pooler_output"
        else:
            feats = outputs.last_hidden_state.mean(dim=1)
            source = "official_siglip_vision_model_last_hidden_state_mean"
    return feats.detach().float().cpu().numpy(), source


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    manifest_path = root / "data" / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = manifest["split_indices"]["train"][0]
    record = manifest["records"][index]
    image_path = args.data_root / record["relative_image"]
    if file_hash(image_path) != record["sha256"]:
        raise RuntimeError("预检图片与审计清单不一致。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model, processor, weights_sha256 = load_model_and_processor(root, config, device, local_files_only=True)
    load_seconds = time.perf_counter() - t0

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").pixel_values
    if tuple(inputs.shape) != (1, 3, config["expected_image_size"], config["expected_image_size"]):
        raise RuntimeError(f"预处理输出形状错误: {tuple(inputs.shape)}")

    t1 = time.perf_counter()
    features, feat_source = extract_whole_image_embedding(model, inputs.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_seconds = time.perf_counter() - t1

    if features.shape != (1, config["expected_feature_dim"]) or not np.isfinite(features).all():
        raise RuntimeError(f"特征形状或有限值检查失败: {features.shape}")

    frozen = all(not p.requires_grad for p in model.parameters())
    param_count = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated(device) / (2**30) if device.type == "cuda" else None
    print("Model:", config["paper_model"], "| Source:", feat_source)
    print("Feature shape:", tuple(features.shape), "| Params:", param_count, "| Peak GPU GiB:", round(peak, 3))

    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "siglip_hd_single_image_smoke_passed_no_training",
        "paper_model": config["paper_model"],
        "model_repo": config["model_repo"],
        "requested_revision": config["requested_revision"],
        "resolved_revision": config["requested_revision"],
        "weights_sha256": weights_sha256,
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
        "feature_source": feat_source,
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
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"siglip_hd_smoke_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
