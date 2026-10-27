"""Run a one-image DINOv3 ConvNeXt-Tiny loading and feature smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def find_image(manifest_path: Path, data_root: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = manifest["split_indices"]["train"][0]
    relative = manifest["records"][index]["relative_image"]
    candidate = data_root / relative
    if candidate.is_file():
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
    print(f"图片: {image_path}")
    print(f"设备: {device}")

    with Image.open(image_path) as original:
        image = original.convert("RGB")
    kwargs = {"revision": config["revision"], "cache_dir": str(root / "hf_cache")}
    processor = AutoImageProcessor.from_pretrained(config["backbone"], **kwargs)
    model = AutoModel.from_pretrained(config["backbone"], **kwargs).eval().to(device)
    model.requires_grad_(False)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        features = pooled_feature(outputs)
    array = features.detach().float().cpu().numpy()
    if array.shape[0] != 1 or not np.isfinite(array).all():
        raise RuntimeError("Feature shape or finite-value check failed")
    print(f"输入张量: {tuple(inputs['pixel_values'].shape)}")
    print(f"特征形状: {tuple(array.shape)}")
    print(f"特征 dtype: {array.dtype}")
    print(f"全部有限: {bool(np.isfinite(array).all())}")
    if device.type == "cuda":
        print(f"显存 GiB: {torch.cuda.memory_allocated(device) / 2**30:.2f}")
    report = {"model": config["backbone"], "revision": config["revision"],
              "device": str(device), "image_relative": image_path.relative_to(args.data_root).as_posix(),
              "input_shape": list(inputs["pixel_values"].shape), "feature_shape": list(array.shape),
              "finite": True, "torch_version": torch.__version__,
              "processor": processor.to_dict()}
    from datetime import datetime
    report_path = root / "logs" / ("smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
