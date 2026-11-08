"""Run/resume B1 seeds 43-46 and compare them with paired B0 validation results."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch

from cnn_baseline_common import file_hash


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "configs" / "cnn_b1_multiseed_confirmation.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "locked_before_confirmation_training":
        raise ValueError("B1 confirmation protocol is not locked")
    baseline_summaries = sorted((root / "logs").glob("cnn_convnext_confirmation_training_summary_*.json"))
    if len(baseline_summaries) != 1:
        raise ValueError("Expected exactly one B0 confirmation summary")
    baseline = json.loads(baseline_summaries[0].read_text(encoding="utf-8"))
    baseline_by_seed = {int(row["seed"]): row for row in baseline["results"]}
    rows = []
    for seed in protocol["confirmation_seeds"]:
        checkpoint_dir = root / "checkpoints" / f"cnn_feature_screen_B1_seed{seed}"
        last_path = checkpoint_dir / "last.pt"
        command = [
            sys.executable,
            str(root / "scripts" / "train_cnn_feature_screen.py"),
            "--variant", "B1",
            "--seed", str(seed),
        ]
        if last_path.is_file():
            command.extend(["--resume", str(last_path)])
        print(f"\n=== B1 CONFIRMATION SEED {seed} ===", flush=True)
        subprocess.run(command, cwd=root, check=True)
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        if (
            last["patience_counter"] < protocol["early_stopping_patience"]
            and last["completed_epoch"] < protocol["max_epochs"]
        ):
            raise RuntimeError(f"Seed {seed} returned before completion")
        best_path = checkpoint_dir / "best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        if best["seed"] != seed or best["completed_epoch"] != best["best_epoch"]:
            raise ValueError(f"Seed {seed} best checkpoint identity differs")
        b0 = float(baseline_by_seed[seed]["phase_b_validation_macro_f1"])
        b1 = float(best["best_validation_macro_f1"])
        rows.append({
            "seed": seed,
            "B0_validation_macro_f1": b0,
            "B1_validation_macro_f1": b1,
            "paired_gain_B1_minus_B0": b1 - b0,
            "B1_stopped_epoch": int(last["completed_epoch"]),
            "B1_best_epoch": int(best["best_epoch"]),
            "B1_best_checkpoint_relative": best_path.relative_to(root).as_posix(),
            "B1_best_checkpoint_sha256": file_hash(best_path),
        })
    gains = np.asarray([row["paired_gain_B1_minus_B0"] for row in rows], dtype=np.float64)
    b1_values = np.asarray([row["B1_validation_macro_f1"] for row in rows], dtype=np.float64)
    positive = int((gains > 0).sum())
    rule = protocol["success_rule"]
    success = bool(
        gains.mean() >= rule["minimum_mean_paired_macro_f1_gain"]
        and positive >= rule["minimum_seeds_with_positive_gain"]
    )
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    summary = {
        "timestamp": now.isoformat(),
        "status": "cnn_b1_multiseed_confirmation_complete_validation_only",
        "protocol_relative": protocol_path.relative_to(root).as_posix(),
        "protocol_sha256": file_hash(protocol_path),
        "paired_baseline_summary_relative": baseline_summaries[0].relative_to(root).as_posix(),
        "paired_baseline_summary_sha256": file_hash(baseline_summaries[0]),
        "results": rows,
        "B1_validation_macro_f1_mean": float(b1_values.mean()),
        "B1_validation_macro_f1_sample_std": float(b1_values.std(ddof=1)),
        "paired_gain_mean": float(gains.mean()),
        "paired_gain_sample_std": float(gains.std(ddof=1)),
        "seeds_with_positive_gain": positive,
        "locked_success_rule_passed": success,
        "same_fixed_split_across_seeds": True,
        "seeds_measure_training_randomness_not_independent_dataset_replication": True,
        "test_dataset_instantiated": False,
        "test_evaluated": False,
        "post_test_disclosure": protocol["post_test_disclosure"],
    }
    summary_path = root / "logs" / f"cnn_b1_confirmation_summary_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== B1 VALIDATION-ONLY MULTISEED CONFIRMATION ===")
    for row in rows:
        print(
            f"seed {row['seed']}: B0={row['B0_validation_macro_f1']:.6f}, "
            f"B1={row['B1_validation_macro_f1']:.6f}, gain={row['paired_gain_B1_minus_B0']:+.6f}"
        )
    print(f"B1 Macro-F1 mean +/- sample std: {b1_values.mean():.6f} +/- {b1_values.std(ddof=1):.6f}")
    print(f"Paired gain mean +/- sample std: {gains.mean():+.6f} +/- {gains.std(ddof=1):.6f}")
    print("Seeds with positive gain:", positive, "/ 4")
    print("Locked success rule passed:", success)
    print("Test evaluated:", False)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
