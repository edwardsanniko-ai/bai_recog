"""Summarize validation-only L1 versus same-seed B0 confirmation results."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "cnn_label_smoothing_confirmation.json").read_text(encoding="utf-8"))
    rows = []
    for seed in map(int, config["confirmatory_seeds"]):
        b0 = load(root / "checkpoints" / f"cnn_convnext_last_stage_seed{seed}" / "best.pt")
        l1 = load(root / "checkpoints" / f"cnn_label_smoothing_L1_seed{seed}" / "best.pt")
        if b0["seed"] != seed or l1["seed"] != seed:
            raise ValueError(f"Seed identity mismatch for seed {seed}")
        b0_f1 = float(b0["best_validation_macro_f1"])
        l1_f1 = float(l1["best_validation_macro_f1"])
        rows.append({
            "seed": seed,
            "B0_best_epoch": int(b0["best_epoch"]),
            "B0_validation_macro_f1": b0_f1,
            "L1_best_epoch": int(l1["best_epoch"]),
            "L1_validation_macro_f1": l1_f1,
            "paired_gain_L1_minus_B0": l1_f1 - b0_f1,
        })
    gains = np.asarray([row["paired_gain_L1_minus_B0"] for row in rows], dtype=np.float64)
    l1_values = np.asarray([row["L1_validation_macro_f1"] for row in rows], dtype=np.float64)
    summary = {
        "status": "L1_validation_only_confirmation_complete",
        "candidate": "L1",
        "control": "B0",
        "seeds": [row["seed"] for row in rows],
        "results": rows,
        "L1_validation_macro_f1_mean": float(l1_values.mean()),
        "L1_validation_macro_f1_sample_std": float(l1_values.std(ddof=1)),
        "paired_gain_mean": float(gains.mean()),
        "paired_gain_sample_std": float(gains.std(ddof=1)),
        "positive_seed_count": int((gains > 0).sum()),
        "selection_rule": config["selection"],
        "locked_success_rule_passed": bool(
            gains.mean() >= float(config["selection"]["minimum_mean_gain_over_same_seed_B0"])
            and int((gains > 0).sum()) >= int(config["selection"]["minimum_positive_seed_count"])
        ),
        "test_evaluated": False,
    }
    destination = root / "logs" / (
        "cnn_label_smoothing_L1_confirmation_summary_"
        + datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("========== L1 VALIDATION-ONLY CONFIRMATION SUMMARY ==========")
    for row in rows:
        print(
            f"seed {row['seed']}: B0={row['B0_validation_macro_f1']:.6f}, "
            f"L1={row['L1_validation_macro_f1']:.6f}, "
            f"gain={row['paired_gain_L1_minus_B0']:+.6f}"
        )
    print(f"L1 Macro-F1 mean +/- sample std: {summary['L1_validation_macro_f1_mean']:.6f} +/- {summary['L1_validation_macro_f1_sample_std']:.6f}")
    print(f"Paired gain mean +/- sample std: {summary['paired_gain_mean']:+.6f} +/- {summary['paired_gain_sample_std']:.6f}")
    print("Positive seeds:", summary["positive_seed_count"], "/", len(rows))
    print("Locked success rule passed:", summary["locked_success_rule_passed"])
    print("Test evaluated:", False)
    print("Saved summary:", destination)


if __name__ == "__main__":
    main()
