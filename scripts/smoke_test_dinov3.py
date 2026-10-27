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
    relative = manifest["records"][0]["relative_image"]
    candidates = [
        data_root / relative,
        data_root / relative.split("\\", 1)[-1],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "找不到第一张本地图片。请用 --data-root 指向包含类别文件夹的数据目录。"
    )


def pooled_feature(outputs: object) -> torch.Tensor:
    for name in ("pooler_output", "image_embeds", "last_hidden_state"):
        value = getattr(outputs, name, None)
        if value is None:
            continue
        if value.ndim == 4:
            return value.mean(dim=(-2, -1))
        if value.ndim == 3:
            return value.mean(dim=1)
        if value.ndim == 2:
            return value
    raise RuntimeError(f"无法从模型输出中找到可池化特征: {type(outputs)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("..") / "bai_data")
    parser.add_argument(
        "--model", default="facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    image_path = find_image(root / "data" / "splits.json", args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"图片: {image_path}")
    print(f"设备: {device}")

    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        features = pooled_feature(outputs)
    array = features.detach().float().cpu().numpy()
    print(f"输入张量: {tuple(inputs['pixel_values'].shape)}")
    print(f"特征形状: {tuple(array.shape)}")
    print(f"特征 dtype: {array.dtype}")
    print(f"全部有限: {bool(np.isfinite(array).all())}")
    if device.type == "cuda":
        print(f"显存 GiB: {torch.cuda.memory_allocated(device) / 2**30:.2f}")


if __name__ == "__main__":
    main()
