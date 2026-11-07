"""Shared, audited components for the trainable CNN baseline experiments."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import sys

_DLL_HANDLE = None
if os.name == "nt":
    dll_dir = Path(sys.prefix) / "Library" / "bin"
    if dll_dir.is_dir():
        os.environ["PATH"] = str(dll_dir) + os.pathsep + str(Path(sys.prefix)) + os.pathsep + os.environ["PATH"]
        _DLL_HANDLE = os.add_dll_directory(str(dll_dir))
        modules = list((Path(sys.prefix) / "Lib" / "site-packages" / "PIL").glob("_imaging.cp*-win_amd64.pyd"))
        if len(modules) == 1:
            ctypes.WinDLL(str(modules[0]))

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


CONFIG_NAME = "cnn_convnext_tiny_linear_probe.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class ResizePadWhite:
    """Resize without distortion, then center on an RGB white square."""

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("Image has an invalid size")
        scale = self.size / max(width, height)
        resized_size = (
            max(1, min(self.size, round(width * scale))),
            max(1, min(self.size, round(height * scale))),
        )
        resized = image.resize(resized_size, resample=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        offset = ((self.size - resized.width) // 2, (self.size - resized.height) // 2)
        canvas.paste(resized, offset)
        return canvas

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, fill=(255,255,255))"


class BaiPatternDataset(Dataset):
    def __init__(self, records: list[dict], indices: list[int], data_root: Path, transform):
        self.records = records
        self.indices = list(indices)
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        sample_index = self.indices[position]
        record = self.records[sample_index]
        with Image.open(self.data_root / record["relative_image"]) as source:
            image = source.convert("RGB")
        return self.transform(image), int(record["class_id"]), sample_index


def load_config(root: Path) -> dict:
    config = json.loads((root / "configs" / CONFIG_NAME).read_text(encoding="utf-8"))
    if config["allow_test_during_training"] or config["allow_parameter_updates_in_smoke_test"]:
        raise ValueError("Safety boundary differs from the head-only protocol")
    if config["phase"] != "A_head_only" or not config["backbone_frozen"]:
        raise ValueError("This helper is restricted to phase A")
    return config


def load_and_verify_manifest(root: Path, data_root: Path, config: dict, verify_hashes: bool = True) -> dict:
    path = root / config["split"]["manifest"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest["records"]
    split = manifest["split_indices"]
    if len(records) != 620 or [record["index"] for record in records] != list(range(620)):
        raise ValueError("Manifest must contain 620 records in global-index order")
    if manifest["categories"] != config["class_order"]:
        raise ValueError("Class order differs from the locked protocol")
    expected = {name: config["split"][name] for name in ("train", "validation", "test")}
    actual = {name: len(split[name]) for name in ("train", "validation", "test")}
    if actual != expected:
        raise ValueError(f"Split counts differ: {actual} != {expected}")
    combined = [index for name in ("train", "validation", "test") for index in split[name]]
    if sorted(combined) != list(range(620)) or len(set(combined)) != 620:
        raise ValueError("Splits overlap or do not cover the dataset")
    if any(record["class_name"] != manifest["categories"][record["class_id"]] for record in records):
        raise ValueError("Class IDs and names disagree")
    if verify_hashes:
        for record in records:
            image_path = data_root / record["relative_image"]
            if not image_path.is_file() or file_hash(image_path) != record["sha256"]:
                raise ValueError(f"Image missing or changed: sample {record['index']}")
    return manifest


def build_transforms(config: dict):
    size = int(config["image_size"])
    aug = config["train_augmentation"]
    mean = config["normalization"]["mean"]
    std = config["normalization"]["std"]
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=aug["brightness_jitter"], contrast=aug["contrast_jitter"]),
        ResizePadWhite(size),
        transforms.RandomAffine(
            degrees=aug["random_affine_degrees"],
            translate=tuple(aug["random_affine_translate"]),
            scale=tuple(aug["random_affine_scale"]),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=tuple(aug["affine_fill_rgb"]),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        ResizePadWhite(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


def load_head_only_model(root: Path, config: dict, device: torch.device):
    if torchvision.__version__ != config["torchvision_version"]:
        raise ValueError("Torchvision version differs from the locked configuration")
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    if str(weights) != config["weight_enum"]:
        raise ValueError("Weight enum differs from the locked configuration")
    torch.hub.set_dir(str(root / "hf_cache" / "torch"))
    weight_path = root / "hf_cache" / "torch" / "checkpoints" / config["weights_filename"]
    if weight_path.is_file() and file_hash(weight_path) != config["weights_sha256"]:
        raise ValueError("Existing checkpoint has the wrong SHA256")
    model = convnext_tiny(weights=weights)
    if file_hash(weight_path) != config["weights_sha256"]:
        raise ValueError("Loaded checkpoint has the wrong SHA256")
    if len(model.classifier) != 3 or not isinstance(model.classifier[2], nn.Linear):
        raise RuntimeError("Unexpected torchvision ConvNeXt classifier")
    if model.classifier[2].in_features != 768 or model.classifier[2].out_features != 1000:
        raise RuntimeError("Unexpected ImageNet classifier dimensions")
    normalization = model.classifier[0]
    flatten = model.classifier[1]
    model.classifier = nn.Sequential(
        normalization,
        flatten,
        nn.Dropout(p=0.2),
        nn.Linear(768, config["num_classes"]),
    )
    model.requires_grad_(False)
    model.classifier[0].requires_grad_(True)
    model.classifier[3].requires_grad_(True)
    model.to(device=device, dtype=torch.float32)
    return model, weight_path


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]

