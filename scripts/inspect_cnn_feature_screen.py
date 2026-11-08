"""No-update preflight for the locked ConvNeXt resolution/pooling screen."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from cnn_baseline_common import BaiPatternDataset, file_hash, load_and_verify_manifest, set_seed
from cnn_feature_screen_common import (
    CONFIG_NAME,
    build_screen_model,
    load_screen_config,
    parameter_groups,
    screen_transforms,
    variant_config,
)
from cnn_last_stage_common import load_last_stage_config


def tensor_state_hash(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_unchanged(before: dict[str, torch.Tensor], model: torch.nn.Module) -> bool:
    after = model.state_dict()
    return before.keys() == after.keys() and all(torch.equal(value, after[name].detach().cpu()) for name, value in before.items())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_root = root.parent / "bai_data"
    config = load_screen_config(root)
    stage_b_config = load_last_stage_config(root)
    set_seed(int(config["source_seed"]))
    manifest = load_and_verify_manifest(root, data_root, stage_b_config, verify_hashes=True)
    if len(manifest["split_indices"]["train"]) != 434 or len(manifest["split_indices"]["validation"]) != 93:
        raise ValueError("Development split counts differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_variants = []
    logits: dict[str, torch.Tensor] = {}
    for variant_id in ["B1", "B2", "B3"]:
        variant = variant_config(config, variant_id)
        _, eval_transform = screen_transforms(stage_b_config, int(variant["image_size"]))
        dataset = BaiPatternDataset(
            manifest["records"], [manifest["split_indices"]["validation"][0]], data_root, eval_transform
        )
        image, _, index = dataset[0]
        model, weight_path, initialization_path, initialization_checkpoint = build_screen_model(
            root, config, variant_id, device
        )
        before = tensor_state_hash(model)
        model.eval()
        with torch.inference_mode():
            output = model(image.unsqueeze(0).to(device))
        if output.shape != (1, 6) or not torch.isfinite(output).all():
            raise RuntimeError(f"Invalid output for {variant_id}")
        if not state_unchanged(before, model):
            raise RuntimeError(f"A parameter or buffer changed during {variant_id} preflight")
        stage, head = parameter_groups(model)
        logits[variant_id] = output.detach().cpu()
        report_variants.append({
            "variant": variant_id,
            "image_size": variant["image_size"],
            "pooling": variant["pooling"],
            "sample_index": index,
            "input_shape": list(image.unsqueeze(0).shape),
            "output_shape": list(output.shape),
            "all_finite": bool(torch.isfinite(output).all()),
            "trainable_last_stage_parameters": sum(value.numel() for value in stage),
            "trainable_head_parameters": sum(value.numel() for value in head),
            "all_nonselected_parameters_frozen": all(
                not value.requires_grad
                for name, value in model.named_parameters()
                if not name.startswith("features.7.") and not name.startswith("classifier.")
            ),
            "state_unchanged": True,
            "weights_sha256": file_hash(weight_path),
            "initialization_checkpoint_sha256": file_hash(initialization_path),
            "initialization_best_epoch": initialization_checkpoint["best_epoch"],
        })
        del model
    # At each resolution, zero-initialized local SPP blocks must reproduce GAP logits exactly.
    equivalence = {}
    for size, gap_id, spp_id in [(224, None, "B2"), (320, "B1", "B3")]:
        if gap_id is None:
            gap_variant = {"id": "B1", "image_size": 224, "pooling": "global_average", "action": "train"}
            config_copy = json.loads(json.dumps(config))
            config_copy["variants"] = [gap_variant if row["id"] == "B1" else row for row in config_copy["variants"]]
            _, transform = screen_transforms(stage_b_config, size)
            dataset = BaiPatternDataset(
                manifest["records"], [manifest["split_indices"]["validation"][0]], data_root, transform
            )
            image, _, _ = dataset[0]
            gap_model, _, _, _ = build_screen_model(root, config_copy, "B1", device)
            gap_model.eval()
            with torch.inference_mode():
                gap_logits = gap_model(image.unsqueeze(0).to(device)).cpu()
            del gap_model
        else:
            gap_logits = logits[gap_id]
        max_difference = float((gap_logits - logits[spp_id]).abs().max())
        equivalence[str(size)] = {"max_absolute_logit_difference": max_difference, "close": max_difference < 1e-6}
        if max_difference >= 1e-6:
            raise RuntimeError(f"SPP initialization does not reproduce GAP at {size}px")
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    report = {
        "timestamp": now.isoformat(),
        "status": "cnn_feature_screen_preflight_passed_no_updates",
        "config": config,
        "config_sha256": file_hash(root / "configs" / CONFIG_NAME),
        "manifest_sha256": file_hash(root / config["data"]["manifest"]),
        "variants": report_variants,
        "spp_initialization_equivalence": equivalence,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "test_dataset_instantiated": False,
        "test_evaluated": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    path = root / "logs" / f"cnn_feature_screen_preflight_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in report_variants:
        print(
            f"{row['variant']}: input={tuple(row['input_shape'])}, output={tuple(row['output_shape'])}, "
            f"trainable stage={row['trainable_last_stage_parameters']}, head={row['trainable_head_parameters']}"
        )
    print("SPP initialized equal to GAP at 224:", equivalence["224"]["close"])
    print("SPP initialized equal to GAP at 320:", equivalence["320"]["close"])
    print("Optimizer updates:", 0)
    print("Test evaluated:", False)
    print("Saved report:", path)


if __name__ == "__main__":
    main()
