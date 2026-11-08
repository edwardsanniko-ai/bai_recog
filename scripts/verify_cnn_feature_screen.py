"""Verify the validation-only 2x2 feature screen from saved prediction CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from cnn_baseline_common import file_hash


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summaries = sorted((root / "logs").glob("cnn_feature_screen_summary_*.json"))
    if len(summaries) != 1:
        raise ValueError(f"Expected one feature-screen summary, found {len(summaries)}")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/splits.json").read_text(encoding="utf-8"))
    categories = manifest["categories"]
    csv_paths = {
        "B0": root / "results/cnn_convnext_last_stage_seed42_validation_epoch014.csv",
        "B1": root / "results/cnn_feature_screen_B1_seed42_validation_epoch023.csv",
        "B2": root / "results/cnn_feature_screen_B2_seed42_validation_epoch005.csv",
        "B3": root / "results/cnn_feature_screen_B3_seed42_validation_epoch013.csv",
    }
    rows_by_variant = {}
    for result in summary["results"]:
        variant = result["variant"]
        checkpoint = root / result["best_checkpoint_relative"]
        if file_hash(checkpoint) != result["best_checkpoint_sha256"]:
            raise ValueError(f"Best checkpoint SHA256 differs for {variant}")
        with csv_paths[variant].open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 93:
            raise ValueError(f"Validation prediction count differs for {variant}")
        rows_by_variant[variant] = rows
        truth = np.asarray([int(row["true_class_id"]) for row in rows], dtype=np.int64)
        predicted = np.asarray([int(row["predicted_class_id"]) for row in rows], dtype=np.int64)
        details = classification_report(
            truth, predicted, labels=range(6), target_names=categories, output_dict=True, zero_division=0
        )
        macro_f1 = float(details["macro avg"]["f1-score"])
        if not np.isclose(macro_f1, result["best_validation_macro_f1"], rtol=0.0, atol=1e-12):
            raise ValueError(f"Recomputed Macro-F1 differs for {variant}")
        print(
            f"{variant}: rows=93, accuracy={accuracy_score(truth, predicted):.6f}, "
            f"Macro-F1={macro_f1:.6f}, matches summary=True"
        )
    old = {int(row["sample_index"]): row for row in rows_by_variant["B0"]}
    new = {int(row["sample_index"]): row for row in rows_by_variant["B1"]}
    fixed = [index for index in old if old[index]["correct"] == "False" and new[index]["correct"] == "True"]
    regressed = [index for index in old if old[index]["correct"] == "True" and new[index]["correct"] == "False"]
    if fixed != [532] or regressed:
        raise ValueError("B0/B1 paired prediction difference changed")
    print("B1 fixed validation sample indices:", fixed)
    print("B1 regressed validation sample indices:", regressed)
    print("Selected variant:", summary["selected_for_multiseed_confirmation"])
    print("Test evaluated:", summary["test_evaluated"])


if __name__ == "__main__":
    main()
