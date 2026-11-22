"""Smoke-test frozen nvidia/C-RADIOv4-H on one audited training image."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import subprocess
import time

import einops
import numpy as np
import PIL
from PIL import Image
import torch
import transformers

from smoke_test_c_radiov4_so400m import (
    audited_training_image,
    file_hash,
    load_model_and_processor,
    preprocess_images,
    validate_config,
    verified_snapshot,
)


CONFIG_NAME = "baseline_c_radiov4_h.json"


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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    model, processor = load_model_and_processor(snapshot_dir, device, config)
    load_seconds = time.perf_counter() - t0

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = preprocess_images([image], processor, config["expected_image_size"])
    t1 = time.perf_counter()
    with torch.inference_mode():
        summary, spatial_features = model(inputs.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    infer_seconds = time.perf_counter() - t1

    features = summary.detach().float().cpu().numpy()
    if features.shape != (1, config["expected_feature_dim"]) or not np.isfinite(features).all():
        raise RuntimeError(f"summary特征形状或有限值检查失败: {features.shape}")

    frozen = all(not p.requires_grad for p in model.parameters())
    param_count = sum(p.numel() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated(device) / (2**30) if device.type == "cuda" else None
    print("Model:", config["paper_model"], "| Params:", param_count, "| Summary:", tuple(features.shape), "| Peak GPU GiB:", round(peak, 3))

    weights_sha256 = file_hash(weights)
    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "c_radiov4_h_single_image_smoke_passed_no_training",
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
        "feature_source": "official_c_radiov4_h_summary_output",
        "feature_shape": list(features.shape),
        "spatial_feature_shape": list(spatial_features.shape),
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
        "einops_version": einops.__version__,
        "pillow_version": PIL.__version__,
    }
    report_path = root / "logs" / f"c_radiov4_h_smoke_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
