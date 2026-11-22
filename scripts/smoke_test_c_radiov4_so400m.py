"""Smoke-test frozen nvidia/C-RADIOv4-SO400M on one audited training image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import einops
import numpy as np
import PIL
from PIL import Image
import torch
import transformers
from transformers import AutoModel, CLIPImageProcessor
from transformers.dynamic_module_utils import HF_MODULES_CACHE


CONFIG_NAME = "baseline_c_radiov4_so400m.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["allow_parameter_updates"] or not config["backbone_frozen"]:
        raise ValueError("本轮必须冻结C-RADIOv4模型且禁止参数更新。")
    if config["test_split_allowed"]:
        raise ValueError("候选筛选阶段禁止访问测试集。")
    if transformers.__version__ != config["implementation_version"]:
        raise RuntimeError(
            f"需要transformers {config['implementation_version']}，当前为{transformers.__version__}。"
        )


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


def verified_snapshot(root: Path, config: dict) -> tuple[Path, Path]:
    snapshot_dir = root / config["local_snapshot_dir"]
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"找不到本地快照目录: {snapshot_dir}")
    weights = snapshot_dir / config["weights_filename"]
    if not weights.is_file():
        raise FileNotFoundError(f"找不到权重文件: {weights}")
    if weights.stat().st_size != config["weights_size_bytes"]:
        raise RuntimeError(
            f"权重文件大小不符合官方值: {weights.stat().st_size} != {config['weights_size_bytes']}"
        )
    return snapshot_dir, weights


def sync_local_dynamic_modules(snapshot_dir: Path) -> None:
    """Copy all local .py files into transformers' module cache for offline nested imports."""
    mod_name = f"_{snapshot_dir.name}" if snapshot_dir.name[0].isdigit() else snapshot_dir.name
    module_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / mod_name
    module_dir.mkdir(parents=True, exist_ok=True)
    init_file = module_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
    for py_file in snapshot_dir.glob("*.py"):
        target = module_dir / py_file.name
        if not target.exists() or target.stat().st_size != py_file.stat().st_size:
            shutil.copy2(py_file, target)


def load_model_and_processor(snapshot_dir: Path, device: torch.device, config: dict):
    sync_local_dynamic_modules(snapshot_dir)
    processor = CLIPImageProcessor.from_pretrained(str(snapshot_dir), local_files_only=True)
    model = AutoModel.from_pretrained(
        str(snapshot_dir),
        trust_remote_code=True,
        local_files_only=True,
    ).eval().to(device)
    model.requires_grad_(False)

    preferred = tuple(int(v) for v in model.preferred_resolution)
    expected_res = (config["expected_image_size"], config["expected_image_size"])
    if preferred != expected_res:
        raise RuntimeError(f"preferred_resolution不符合锁定值: {preferred} != {expected_res}")
    if model.patch_size != config["expected_patch_size"]:
        raise RuntimeError(f"patch_size不符合锁定值: {model.patch_size}")
    if getattr(model.radio_model, "embed_dim", None) != config["expected_embed_dim"]:
        raise RuntimeError(f"embed_dim不符合锁定值: {getattr(model.radio_model, 'embed_dim', None)}")
    if getattr(model.radio_model, "summary_dim", None) != config["expected_feature_dim"]:
        raise RuntimeError(f"summary_dim不符合锁定值: {getattr(model.radio_model, 'summary_dim', None)}")
    return model, processor


def preprocess_images(
    images: list[Image.Image], processor: CLIPImageProcessor, image_size: int
) -> torch.Tensor:
    encoded = processor(
        images=images,
        return_tensors="pt",
        do_resize=True,
        size={"height": image_size, "width": image_size},
        do_center_crop=False,
        do_rescale=True,
        do_normalize=False,
    )
    pixel_values = encoded.pixel_values.to(dtype=torch.float32)
    return pixel_values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    manifest_path, image_path, record, _ = audited_training_image(root, args.data_root)
    snapshot_dir, weights = verified_snapshot(root, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("Paper model:", config["paper_model"], flush=True)
    print("Snapshot dir:", snapshot_dir)
    print("Image:", image_path)
    print("Device:", device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model, processor = load_model_and_processor(snapshot_dir, device, config)
    load_seconds = time.perf_counter() - t0

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = preprocess_images([image], processor, config["expected_image_size"])
    expected_input = (1, 3, config["expected_image_size"], config["expected_image_size"])
    if tuple(inputs.shape) != expected_input:
        raise RuntimeError(f"预处理输出形状错误: {tuple(inputs.shape)}")
    min_val, max_val = float(inputs.min()), float(inputs.max())
    if min_val < -1e-6 or max_val > 1.0 + 1e-6:
        raise RuntimeError(f"输入像素值超出[0,1]范围: [{min_val}, {max_val}]")

    t1 = time.perf_counter()
    with torch.inference_mode():
        summary, spatial_features = model(inputs.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    infer_seconds = time.perf_counter() - t1

    features = summary.detach().float().cpu().numpy()
    spatial_shape = list(spatial_features.shape)
    expected_feature = (1, config["expected_feature_dim"])
    if features.shape != expected_feature or not np.isfinite(features).all():
        raise RuntimeError(f"summary特征形状或有限值检查失败: {features.shape}")

    frozen = all(not parameter.requires_grad for parameter in model.parameters())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    peak_vram_gib = (
        torch.cuda.max_memory_allocated(device) / (2**30) if device.type == "cuda" else None
    )
    alloc_vram_gib = (
        torch.cuda.memory_allocated(device) / (2**30) if device.type == "cuda" else None
    )

    print("Resolved revision:", config["requested_revision"])
    print("Input shape:", tuple(inputs.shape), f"range=[{min_val:.4f}, {max_val:.4f}]")
    print("Feature source: official C-RADIOv4 summary output")
    print("Summary shape:", tuple(features.shape), "dtype:", features.dtype)
    print("Spatial shape:", tuple(spatial_shape))
    print("All finite:", bool(np.isfinite(features).all()))
    print("All parameters frozen:", frozen)
    print("Parameter count:", parameter_count)
    print(f"Load time: {load_seconds:.2f}s | Single-image infer time: {infer_seconds:.3f}s")
    if peak_vram_gib is not None:
        print(f"Allocated GPU GiB: {alloc_vram_gib:.3f} | Peak GPU GiB: {peak_vram_gib:.3f}")
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)

    weights_sha256 = file_hash(weights)
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai"))
    git = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report = {
        "timestamp": now.isoformat(),
        "status": "c_radiov4_so400m_single_image_smoke_passed_no_training",
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
        "input_min": min_val,
        "input_max": max_val,
        "feature_source": "official_c_radiov4_summary_output",
        "feature_shape": list(features.shape),
        "spatial_feature_shape": spatial_shape,
        "feature_dtype": str(features.dtype),
        "all_finite": True,
        "all_parameters_frozen": frozen,
        "parameter_count": parameter_count,
        "load_seconds": round(load_seconds, 4),
        "single_image_infer_seconds": round(infer_seconds, 4),
        "allocated_gpu_gib": round(alloc_vram_gib, 4) if alloc_vram_gib is not None else None,
        "peak_gpu_gib": round(peak_vram_gib, 4) if peak_vram_gib is not None else None,
        "optimizer_updates": 0,
        "classification_performed": False,
        "test_evaluated": False,
        "allow_tf32": False,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "einops_version": einops.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"c_radiov4_so400m_smoke_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
