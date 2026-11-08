"""Verify B1 multiseed confirmation from the saved best validation CSV files."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from cnn_baseline_common import file_hash


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summaries = sorted((root / "logs").glob("cnn_b1_confirmation_summary_*.json"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one B1 confirmation summary, found {len(summaries)}")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/splits.json").read_text(encoding="utf-8"))
    categories = manifest["categories"]
    b0_best_epochs = {43: 20, 44: 20, 45: 14, 46: 22}
    per_class = {"B0": {name: [] for name in categories}, "B1": {name: [] for name in categories}}
    fixed_counts, regressed_counts = Counter(), Counter()
    for result in summary["results"]:
        seed = int(result["seed"])
        checkpoint = root / result["B1_best_checkpoint_relative"]
        if file_hash(checkpoint) != result["B1_best_checkpoint_sha256"]:
            raise ValueError(f"B1 checkpoint SHA256 differs for seed {seed}")
        paths = {
            "B0": root / "results" / f"cnn_convnext_last_stage_seed{seed}_validation_epoch{b0_best_epochs[seed]:03d}.csv",
            "B1": root / "results" / f"cnn_feature_screen_B1_seed{seed}_validation_epoch{result['B1_best_epoch']:03d}.csv",
        }
        loaded = {}
        for model_name, path in paths.items():
            with path.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 93:
                raise ValueError(f"Validation row count differs for {model_name} seed {seed}")
            loaded[model_name] = {int(row["sample_index"]): row for row in rows}
            truth = np.asarray([int(row["true_class_id"]) for row in rows], dtype=np.int64)
            predicted = np.asarray([int(row["predicted_class_id"]) for row in rows], dtype=np.int64)
            details = classification_report(
                truth, predicted, labels=range(6), target_names=categories, output_dict=True, zero_division=0
            )
            macro_f1 = float(details["macro avg"]["f1-score"])
            if not np.isclose(macro_f1, result[f"{model_name}_validation_macro_f1"], rtol=0.0, atol=1e-12):
                raise ValueError(f"Macro-F1 differs for {model_name} seed {seed}")
            for category in categories:
                per_class[model_name][category].append(float(details[category]["f1-score"]))
        fixed, regressed = [], []
        for index in loaded["B0"]:
            b0_correct = loaded["B0"][index]["correct"] == "True"
            b1_correct = loaded["B1"][index]["correct"] == "True"
            if not b0_correct and b1_correct:
                fixed.append(index)
                fixed_counts[index] += 1
            elif b0_correct and not b1_correct:
                regressed.append(index)
                regressed_counts[index] += 1
        print(f"seed {seed}: fixed={fixed}, regressed={regressed}, metrics_match=True")
    print("Mean classwise F1 change (B1 - B0):")
    for category in categories:
        change = float(np.mean(per_class["B1"][category]) - np.mean(per_class["B0"][category]))
        print(f"  {category}: {change:+.6f}")
    if fixed_counts[532] != 4:
        raise ValueError("Expected sample 532 to be fixed in all four seeds")
    gains = np.asarray([row["paired_gain_B1_minus_B0"] for row in summary["results"]], dtype=np.float64)
    if not np.isclose(gains.mean(), summary["paired_gain_mean"], rtol=0.0, atol=1e-12):
        raise ValueError("Paired mean gain differs")
    if not np.isclose(gains.std(ddof=1), summary["paired_gain_sample_std"], rtol=0.0, atol=1e-12):
        raise ValueError("Paired gain standard deviation differs")
    print("Sample 532 fixed in all seeds: True")
    print("Locked success rule passed:", summary["locked_success_rule_passed"])
    print("Test evaluated:", summary["test_evaluated"])


if __name__ == "__main__":
    main()
