"""Shared model setup for ConvNeXt-Tiny phase-B last-stage fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from cnn_baseline_common import file_hash, load_head_only_model


CONFIG_NAME = "cnn_convnext_tiny_last_stage.json"


def load_last_stage_config(root: Path) -> dict:
    config = json.loads((root / "configs" / CONFIG_NAME).read_text(encoding="utf-8"))
    if config["phase"] != "B_last_stage_finetune":
        raise ValueError("This helper is restricted to phase B")
    if config["allow_test_during_training"] or config["allow_parameter_updates_in_preflight"]:
        raise ValueError("Phase-B safety boundary differs from the locked protocol")
    return config


def phase_a_compatible_config(config: dict) -> dict:
    """Project phase-B shared fields into the phase-A model builder schema."""
    return {
        "torchvision_version": config["torchvision_version"],
        "weight_enum": config["weight_enum"],
        "weights_filename": config["weights_filename"],
        "weights_sha256": config["weights_sha256"],
        "num_classes": config["num_classes"],
    }


def load_last_stage_model(root: Path, config: dict, device: torch.device):
    # Construct the identical six-class architecture from official weights.
    model, weight_path = load_head_only_model(root, phase_a_compatible_config(config), device)
    initialization = config["initialization"]
    checkpoint_path = root / initialization["head_checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Phase-A best checkpoint is missing: {checkpoint_path}")
    if file_hash(checkpoint_path) != initialization["head_checkpoint_sha256"]:
        raise ValueError("Phase-A best checkpoint SHA256 differs from the phase-B protocol")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint["seed"] != initialization["source_seed"]
        or checkpoint["completed_epoch"] != initialization["source_best_epoch"]
        or checkpoint["best_epoch"] != initialization["source_best_epoch"]
        or checkpoint["best_validation_macro_f1"] != initialization["source_validation_macro_f1"]
    ):
        raise ValueError("Phase-A checkpoint identity or best metric differs")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    if len(model.features) != 8 or not isinstance(model.features[7], nn.Sequential):
        raise RuntimeError("Unexpected torchvision ConvNeXt feature layout")
    model.features[7].requires_grad_(True)
    model.classifier[0].requires_grad_(True)
    model.classifier[3].requires_grad_(True)
    return model, weight_path, checkpoint_path, checkpoint


def phase_b_parameter_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    last_stage = list(model.features[7].parameters())
    head = list(model.classifier[0].parameters()) + list(model.classifier[3].parameters())
    if not last_stage or not head:
        raise RuntimeError("Expected non-empty phase-B parameter groups")
    if {id(value) for value in last_stage} & {id(value) for value in head}:
        raise RuntimeError("Phase-B parameter groups overlap")
    intended = {id(value) for value in last_stage + head}
    actual = {id(value) for value in model.parameters() if value.requires_grad}
    if intended != actual:
        raise RuntimeError("Trainable parameters differ from the two phase-B groups")
    return last_stage, head


def phase_b_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, value in model.named_parameters() if value.requires_grad]

