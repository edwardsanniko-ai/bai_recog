"""Independently verify the locked ConvNeXt final-test artifacts without inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cnn_baseline_common import file_hash


METRIC_NAMES = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]


def require_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{label} differs: recomputed={actual}, recorded={expected}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports = sorted((root / "logs").glob("cnn_convnext_final_test_*.json"))
    if len(reports) != 1:
        raise ValueError(f"Expected exactly one final-test report, found {len(reports)}")

    report_path = reports[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "locked_cnn_final_test_complete_no_post_test_tuning":
        raise ValueError("Unexpected final-test status")
    if report["seeds"] != [43, 44, 45, 46] or report["test_count"] != 93:
        raise ValueError("Locked seeds or test count differ")
    if report["ensemble_used"] or report["test_time_augmentation_used"]:
        raise ValueError("Ensemble or test-time augmentation was unexpectedly enabled")
    if report["optimizer_updates_during_test"] != 0 or report["post_test_tuning_allowed"]:
        raise ValueError("Final-test update/tuning boundary differs")

    for key, relative_key in [
        ("lock_sha256", "lock_relative"),
        ("validation_summary_sha256", "validation_summary_relative"),
    ]:
        artifact = root / report[relative_key]
        if file_hash(artifact) != report[key]:
            raise ValueError(f"SHA256 differs for {artifact}")

    all_metrics: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    expected_indices = list(map(int, report["test_indices"]))
    categories: list[str] | None = None

    for result in report["results"]:
        seed = int(result["seed"])
        checkpoint = root / result["locked_checkpoint_relative"]
        prediction_path = root / result["test_predictions_relative"]
        if file_hash(checkpoint) != result["locked_checkpoint_sha256"]:
            raise ValueError(f"Checkpoint SHA256 differs for seed {seed}")
        if file_hash(prediction_path) != result["test_predictions_sha256"]:
            raise ValueError(f"Prediction CSV SHA256 differs for seed {seed}")

        with prediction_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != report["test_count"]:
            raise ValueError(f"Prediction row count differs for seed {seed}")
        if [int(row["sample_index"]) for row in rows] != expected_indices:
            raise ValueError(f"Test sample order differs for seed {seed}")

        probability_columns = [name for name in rows[0] if name.startswith("probability_")]
        current_categories = [name.removeprefix("probability_") for name in probability_columns]
        if categories is None:
            categories = current_categories
        elif current_categories != categories:
            raise ValueError(f"Category order differs for seed {seed}")

        truth = np.asarray([int(row["true_class_id"]) for row in rows], dtype=np.int64)
        predicted = np.asarray([int(row["predicted_class_id"]) for row in rows], dtype=np.int64)
        probabilities = np.asarray(
            [[float(row[column]) for column in probability_columns] for row in rows], dtype=np.float64
        )
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"Invalid probability values for seed {seed}")
        if not np.array_equal(probabilities.argmax(axis=1), predicted):
            raise ValueError(f"Predicted class is not probability argmax for seed {seed}")
        for row, actual, prediction in zip(rows, truth, predicted):
            if row["true_class"] != categories[int(actual)]:
                raise ValueError(f"True class name differs for seed {seed}")
            if row["predicted_class"] != categories[int(prediction)]:
                raise ValueError(f"Predicted class name differs for seed {seed}")
            if (row["correct"] == "True") != bool(actual == prediction):
                raise ValueError(f"Correct flag differs for seed {seed}")

        class_ids = list(range(len(categories)))
        details = classification_report(
            truth,
            predicted,
            labels=class_ids,
            target_names=categories,
            output_dict=True,
            zero_division=0,
        )
        metrics = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_precision": float(details["macro avg"]["precision"]),
            "macro_recall": float(details["macro avg"]["recall"]),
            "macro_f1": float(details["macro avg"]["f1-score"]),
            "weighted_f1": float(details["weighted avg"]["f1-score"]),
        }
        for name, value in metrics.items():
            require_close(value, result["test_metrics"][name], f"seed {seed} {name}")
            all_metrics[name].append(value)
        matrix = confusion_matrix(truth, predicted, labels=class_ids).tolist()
        if matrix != result["test_confusion_matrix"]:
            raise ValueError(f"Confusion matrix differs for seed {seed}")
        print(
            f"seed {seed}: rows={len(rows)}, hashes=True, accuracy={metrics['accuracy']:.6f}, "
            f"Macro-F1={metrics['macro_f1']:.6f}, matches report=True"
        )

    for name, values in all_metrics.items():
        array = np.asarray(values, dtype=np.float64)
        require_close(float(array.mean()), report["aggregate"][name]["mean"], f"aggregate {name} mean")
        require_close(
            float(array.std(ddof=1)), report["aggregate"][name]["sample_std"], f"aggregate {name} std"
        )
    print("Test images per seed:", report["test_count"])
    print("Final-test report count:", len(reports))
    print("All artifact hashes and recomputed metrics match: True")
    print("Optimizer updates during verification:", 0)
    print("Inference rerun during verification:", False)
    print("Post-test tuning allowed:", False)


if __name__ == "__main__":
    main()
