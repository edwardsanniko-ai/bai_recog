"""Run a one-image DINOv3 ConvNeXt-Tiny loading and feature smoke test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def find_image(manifest_path: Path, data_root: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = manifest["split_indices"]["train"][0]
    relative = manifest["records"][index]["relative_image"]
    candidate = data_root / relative
    if candidate.is_file():
        expected = manifest["records"][index]["sha256"]
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Image differs from the audited dataset manifest")
        return candidate
    raise FileNotFoundError(
        "找不到第一张本地图片。请用 --data-root 指向包含类别文件夹的数据目录。"
    )


def pooled_feature(outputs: object) -> torch.Tensor:
    value = getattr(outputs, "pooler_output", None)
    if value is None or value.ndim != 2:
        raise RuntimeError("Expected DINOv3 ConvNeXt pooler_output with shape [batch, features]")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    args = parser.parse_args()

    config = json.loads((root / "configs" / "baseline_convnext_tiny.json").read_text(encoding="utf-8"))
    image_path = find_image(root / "data" / "splits.json", args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"模型: {config['backbone']}", flush=True)
    print(f"图片: {image_path}")
    print(f"设备: {device}")

    with Image.open(image_path) as original:
        image = original.convert("RGB")
    print("首次运行会下载 DINOv3 权重到本地 hf_cache。", flush=True)
    kwargs = {"revision": config["revision"], "cache_dir": str(root / "hf_cache"),
              "token": True, "trust_remote_code": False}
    processor = AutoImageProcessor.from_pretrained(config["backbone"], use_fast=True, **kwargs)
    model = AutoModel.from_pretrained(config["backbone"], **kwargs).eval().to(device)
    model.requires_grad_(False)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        features = pooled_feature(outputs)
    array = features.detach().float().cpu().numpy()
    if array.shape != (1, 768) or not np.isfinite(array).all():
        raise RuntimeError("Feature shape or finite-value check failed")
    print(f"输入张量: {tuple(inputs['pixel_values'].shape)}")
    print(f"特征形状: {tuple(array.shape)}")
    print(f"特征 dtype: {array.dtype}")
    print(f"全部有限: {bool(np.isfinite(array).all())}")
    if device.type == "cuda":
        print(f"显存 GiB: {torch.cuda.memory_allocated(device) / 2**30:.2f}")
    now = datetime.now(timezone(timedelta(hours=8)))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=False)
    report = {"model": config["backbone"], "revision": config["revision"],
              "timestamp": now.isoformat(), "status": "single_image_forward_passed_not_trained",
              "config": config, "code_commit": git.stdout.strip() if git.returncode == 0 else None,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256((root / "data" / "splits.json").read_bytes()).hexdigest(),
              "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
              "device": str(device), "image_relative": image_path.relative_to(args.data_root).as_posix(),
              "input_shape": list(inputs["pixel_values"].shape), "feature_shape": list(array.shape),
              "finite": True, "torch_version": torch.__version__,
              "transformers_version": transformers.__version__,
              "allow_tf32": False, "use_fast": True,
              "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
              "processor": processor.to_dict()}
    report_path = root / "logs" / ("dinov3_smoke_" + now.strftime("%Y%m%d_%H%M%S_%f") + ".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
