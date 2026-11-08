"""Shared model and protocol helpers for the locked 2x2 feature screen."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from cnn_baseline_common import file_hash
from cnn_last_stage_common import load_last_stage_config, load_last_stage_model


CONFIG_NAME = "cnn_feature_screen_2x2.json"


class SpatialPyramidClassifier(nn.Module):
    """Concatenate normalized 1x1 and 2x2 pooled ConvNeXt features."""

    def __init__(self, normalization: nn.Module, dropout: nn.Module, source_linear: nn.Linear):
        super().__init__()
        if source_linear.in_features != 768:
            raise ValueError("Expected a 768-D ConvNeXt classifier")
        self.normalization = normalization
        self.dropout = dropout
        self.linear = nn.Linear(768 * 5, source_linear.out_features)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.weight[:, :768].copy_(source_linear.weight)
            self.linear.bias.copy_(source_linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        global_features = self.normalization(F.adaptive_avg_pool2d(features, (1, 1))).flatten(1)
        local_features = self.normalization(F.adaptive_avg_pool2d(features, (2, 2))).flatten(1)
        return self.linear(self.dropout(torch.cat([global_features, local_features], dim=1)))


def load_screen_config(root: Path) -> dict:
    config = json.loads((root / "configs" / CONFIG_NAME).read_text(encoding="utf-8"))
    if len(config["source_checkpoint_sha256"]) != 64:
        raise ValueError("Source checkpoint SHA256 has the wrong length")
    if config["status"] != "locked_before_feature_screen_training":
        raise ValueError("Feature-screen protocol is not locked")
    if config["data"]["test_split_allowed"]:
        raise ValueError("Test access must remain disabled")
    if [row["id"] for row in config["variants"]] != ["B0", "B1", "B2", "B3"]:
        raise ValueError("Unexpected feature-screen variants")
    return config


def variant_config(config: dict, variant_id: str) -> dict:
    matches = [row for row in config["variants"] if row["id"] == variant_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown variant: {variant_id}")
    return matches[0]


def build_screen_model(
    root: Path, config: dict, variant_id: str, device: torch.device, source_seed: int | None = None
):
    variant = variant_config(config, variant_id)
    if variant["action"] != "train":
        raise ValueError(f"Variant {variant_id} is a historical control and must not be retrained")
    stage_b_config = load_last_stage_config(root)
    if source_seed is None:
        source_seed = int(config["source_seed"])
    model, weight_path, initialization_path, initialization_checkpoint = load_last_stage_model(
        root, stage_b_config, device, source_seed=int(source_seed)
    )
    if int(source_seed) == int(config["source_seed"]):
        if initialization_path.relative_to(root).as_posix() != config["source_checkpoint"]:
            raise ValueError("Phase-A initialization path differs")
        if file_hash(initialization_path) != config["source_checkpoint_sha256"]:
            raise ValueError("Phase-A initialization SHA256 differs")
    if variant["pooling"] == "spatial_pyramid_1x1_plus_2x2":
        source_classifier = model.classifier
        if len(source_classifier) != 4 or not isinstance(source_classifier[3], nn.Linear):
            raise RuntimeError("Unexpected source classifier")
        model.avgpool = nn.Identity()
        model.classifier = SpatialPyramidClassifier(
            source_classifier[0], source_classifier[2], source_classifier[3]
        ).to(device=device, dtype=torch.float32)
    elif variant["pooling"] != "global_average":
        raise ValueError("Unsupported pooling method")
    model.requires_grad_(False)
    model.features[7].requires_grad_(True)
    model.classifier.requires_grad_(True)
    return model, weight_path, initialization_path, initialization_checkpoint


def parameter_groups(model: nn.Module):
    stage = list(model.features[7].parameters())
    head = list(model.classifier.parameters())
    intended = {id(value) for value in stage + head}
    actual = {id(value) for value in model.parameters() if value.requires_grad}
    if not stage or not head or intended != actual:
        raise RuntimeError("Trainable parameter scope differs from the locked screen")
    return stage, head


def screen_transforms(stage_b_config: dict, image_size: int):
    from cnn_baseline_common import build_transforms

    copied = json.loads(json.dumps(stage_b_config))
    copied["image_size"] = int(image_size)
    return build_transforms(copied)
