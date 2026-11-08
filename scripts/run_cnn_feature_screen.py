"""Run/resume B1-B3 and summarize the locked validation-only 2x2 feature screen."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import torch

from cnn_baseline_common import file_hash
from cnn_feature_screen_common import CONFIG_NAME, load_screen_config, variant_config


def run_variant(root: Path, variant_id: str, seed: int) -> dict:
    checkpoint_dir = root / "checkpoints" / f"cnn_feature_screen_{variant_id}_seed{seed}"
    last_path = checkpoint_dir / "last.pt"
    command = [sys.executable, str(root / "scripts" / "train_cnn_feature_screen.py"), "--variant", variant_id]
    if last_path.is_file():
        command.extend(["--resume", str(last_path)])
    print("\n=== RUNNING", variant_id, "===", flush=True)
    subprocess.run(command, cwd=root, check=True)
    if not last_path.is_file():
        raise FileNotFoundError(f"Missing last checkpoint for {variant_id}")
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    completed = (
        checkpoint["patience_counter"] >= load_screen_config(root)["early_stopping_patience"]
        or checkpoint["completed_epoch"] >= checkpoint["variant"].get("max_epochs", 50)
        or checkpoint["completed_epoch"] >= load_screen_config(root)["max_epochs"]
    )
    if not completed:
        raise RuntimeError(f"{variant_id} returned before early stopping or max epochs")
    best_path = checkpoint_dir / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best["completed_epoch"] != best["best_epoch"]:
        raise ValueError(f"Best checkpoint identity differs for {variant_id}")
    return {
        "variant": variant_id,
        "image_size": checkpoint["variant"]["image_size"],
        "pooling": checkpoint["variant"]["pooling"],
        "stopped_epoch": checkpoint["completed_epoch"],
        "best_epoch": best["best_epoch"],
        "best_validation_macro_f1": best["best_validation_macro_f1"],
        "best_checkpoint_relative": best_path.relative_to(root).as_posix(),
        "best_checkpoint_sha256": file_hash(best_path),
        "test_evaluated": False,
    }


def choose_variant(config: dict, results: list[dict]) -> tuple[str, str]:
    values = {row["variant"]: float(row["best_validation_macro_f1"]) for row in results}
    baseline = values["B0"]
    best_value = max(values.values())
    gain_required = float(config["selection"]["minimum_absolute_gain_over_B0_for_confirmation"])
    if best_value < baseline + gain_required:
        return "B0", "No candidate met the locked minimum gain over B0"
    tolerance = float(config["selection"]["near_tie_tolerance"])
    near_best = {key for key, value in values.items() if best_value - value <= tolerance and value >= baseline + gain_required}
    for variant_id in config["selection"]["near_tie_preference_order"]:
        if variant_id in near_best:
            return variant_id, "Selected by validation Macro-F1 with locked near-tie simplicity rule"
    raise RuntimeError("Selection rule did not produce a variant")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_screen_config(root)
    seed = int(config["source_seed"])
    results = [{
        "variant": "B0",
        "image_size": 224,
        "pooling": "global_average",
        "stopped_epoch": 24,
        "best_epoch": config["historical_control"]["best_epoch"],
        "best_validation_macro_f1": config["historical_control"]["validation_macro_f1"],
        "best_checkpoint_relative": config["historical_control"]["checkpoint"],
        "best_checkpoint_sha256": config["historical_control"]["checkpoint_sha256"],
        "test_evaluated": False,
        "historical_control_reused_without_retraining": True,
    }]
    for variant_id in ["B1", "B2", "B3"]:
        results.append(run_variant(root, variant_id, seed))
    values = {row["variant"]: float(row["best_validation_macro_f1"]) for row in results}
    effects = {
        "resolution_effect_with_global_average_B1_minus_B0": values["B1"] - values["B0"],
        "resolution_effect_with_spatial_pyramid_B3_minus_B2": values["B3"] - values["B2"],
        "pooling_effect_at_224_B2_minus_B0": values["B2"] - values["B0"],
        "pooling_effect_at_320_B3_minus_B1": values["B3"] - values["B1"],
        "factor_interaction": (values["B3"] - values["B2"]) - (values["B1"] - values["B0"]),
    }
    selected, reason = choose_variant(config, results)
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    summary = {
        "timestamp": now.isoformat(),
        "status": "cnn_feature_screen_complete_validation_only_pending_confirmation",
        "config_relative": f"configs/{CONFIG_NAME}",
        "config_sha256": file_hash(root / "configs" / CONFIG_NAME),
        "development_seed": seed,
        "results": results,
        "factor_effects_on_validation_macro_f1": effects,
        "selected_for_multiseed_confirmation": selected,
        "selection_reason": reason,
        "selection_is_exploratory_single_seed": True,
        "confirmation_required": True,
        "test_dataset_instantiated": False,
        "test_evaluated": False,
        "post_test_disclosure": config["disclosure"],
    }
    summary_path = root / "logs" / f"cnn_feature_screen_summary_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== VALIDATION-ONLY 2x2 FEATURE SCREEN ===")
    for row in results:
        print(
            f"{row['variant']}: size={row['image_size']}, pooling={row['pooling']}, "
            f"best epoch={row['best_epoch']}, Macro-F1={row['best_validation_macro_f1']:.6f}"
        )
    print("Selected for multiseed confirmation:", selected)
    print("Selection reason:", reason)
    print("Test evaluated:", False)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
