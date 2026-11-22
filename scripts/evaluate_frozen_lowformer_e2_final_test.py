"""Evaluate locked frozen LowFormer-E2 with its validation-selected LinearSVC on the fixed 93-image test split."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize

from cnn_baseline_common import file_hash
from compare_frozen_linear_probes import load_cache, make_classifier, save_predictions
from evaluate_frozen_top4_final_test import calculate_metrics, checked_path, read_json


LOCK_NAME = "frozen_lowformer_e2_final_test.json"
MODEL_IDS = ["lowformer_e2"]
EXPECTED_C = [10.0]


def prepare_models(root: Path, lock: dict):
    if lock["status"] != "locked_before_test_evaluation":
        raise ValueError("Final-test configuration is not locked")
    if [model["id"] for model in lock["models"]] != MODEL_IDS:
        raise ValueError("Final-test candidates or order differ")
    if [model["C"] for model in lock["models"]] != EXPECTED_C:
        raise ValueError("Final-test regularization values differ")
    prohibited = (
        "validation_used_for_refit",
        "hyperparameter_search_during_final_test",
        "ensemble",
        "test_time_augmentation",
        "post_test_tuning_allowed",
    )
    if any(lock[name] is not False for name in prohibited) or lock["backbone_parameter_updates"] != 0:
        raise ValueError("Final-test protocol differs from the agreed frozen evaluation")
    if sklearn.__version__ != lock["sklearn_version"]:
        raise ValueError(f"Expected sklearn {lock['sklearn_version']}, got {sklearn.__version__}")

    report = read_json(checked_path(root, lock["validation_report"], lock["validation_report_sha256"]))
    config = read_json(checked_path(root, lock["validation_config"], lock["validation_config_sha256"]))
    manifest = read_json(checked_path(root, lock["manifest"], lock["manifest_sha256"]))
    if report["test_evaluated"] or report["backbone_parameter_updates"] != 0:
        raise ValueError("Source report is not a frozen validation-only report")

    split = manifest["split_indices"]
    if [len(split[name]) for name in ("train", "validation", "test")] != [434, 93, 93]:
        raise ValueError("Fixed split counts differ")
    if sorted(index for indices in split.values() for index in indices) != list(range(620)):
        raise ValueError("Fixed splits contain duplicates or omissions")

    train_idx, val_idx = np.asarray(split["train"]), np.asarray(split["validation"])
    labels = np.asarray([record["class_id"] for record in manifest["records"]], dtype=np.int64)
    categories = manifest["categories"]
    prepared = []
    warnings.simplefilter("error", ConvergenceWarning)

    for entry in lock["models"]:
        source = next(row for row in report["results"] if row["model_id"] == entry["id"])
        if (
            entry["C"] != source["selected_C"]
            or entry["cache"] != source["cache_relative"]
            or entry["cache_sha256"] != source["cache_sha256"]
        ):
            raise ValueError(f"Locked model differs from validation source: {entry['id']}")
        checked_path(root, entry["cache"], entry["cache_sha256"])
        _, data = load_cache(root, entry["cache"], lock["manifest_sha256"])
        if not np.array_equal(data["labels"], labels) or not np.array_equal(data["categories"], categories):
            raise ValueError(f"Cache labels or category order differ: {entry['id']}")
        for key, split_name in (("train_idx", "train"), ("val_idx", "validation"), ("test_idx", "test")):
            if not np.array_equal(data[key], split[split_name]):
                raise ValueError(f"Cache {split_name} indices differ: {entry['id']}")
        if data["features"].shape[1] != source["feature_dimension"]:
            raise ValueError("Cached feature dimension differs from validation source")

        x_train = normalize(data["features"][train_idx].astype(np.float64), norm="l2")
        x_val = normalize(data["features"][val_idx].astype(np.float64), norm="l2")
        classifier = make_classifier(config, entry["C"])
        classifier.fit(x_train, labels[train_idx])
        predictions = classifier.predict(x_val)
        scores = classifier.decision_function(x_val)

        original_csv = checked_path(root, source["prediction_relative"], source["prediction_sha256"])
        with original_csv.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [int(row["sample_index"]) for row in rows] != split["validation"]:
            raise ValueError("Original validation prediction order differs")
        if not np.array_equal(predictions, [int(row["predicted_class_id"]) for row in rows]):
            raise ValueError(f"Refitted validation predictions differ: {entry['id']}")
        original_scores = np.asarray(
            [[float(row[f"decision_score_{name}"]) for name in categories] for row in rows]
        )
        if not np.allclose(scores, original_scores, rtol=1e-10, atol=1e-10):
            raise ValueError(f"Refitted validation decision scores differ: {entry['id']}")
        metrics, _, _ = calculate_metrics(labels[val_idx], predictions, categories)
        if any(
            not np.isclose(value, source["validation_metrics"][name], rtol=0, atol=1e-12)
            for name, value in metrics.items()
        ):
            raise ValueError("Refitted validation metrics differ")
        prepared.append((entry, source, data, classifier))
        print(f"Verified {entry['id']}: fixed C={entry['C']:g}, validation predictions reproduced", flush=True)
    return manifest, prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-only", action="store_true", help="Verify and refit on train; no test predictions"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "configs" / LOCK_NAME
    lock = read_json(lock_path)

    existing = list((root / "logs").glob("frozen_lowformer_e2_final_test_*.json"))
    existing += list((root / "results").glob("frozen_lowformer_e2_*_test_*.csv"))
    if existing and not args.preflight_only:
        raise FileExistsError("Final-test outputs already exist; refusing to repeat or overwrite evaluation")

    manifest, prepared = prepare_models(root, lock)
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Shanghai"))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    common = {
        "timestamp": now.isoformat(),
        "lock_relative": f"configs/{LOCK_NAME}",
        "lock_sha256": file_hash(lock_path),
        "script_sha256": file_hash(Path(__file__)),
        "classifier_helper_sha256": file_hash(root / "scripts/compare_frozen_linear_probes.py"),
        "code_commit": git.stdout.strip() if git.returncode == 0 else None,
        "manifest_sha256": lock["manifest_sha256"],
        "sklearn_version": sklearn.__version__,
        "fit_count": 434,
        "fit_split": "train",
        "validation_used_for_refit": False,
        "metric_averaging": "Accuracy overall; Precision, Recall and F1 macro over all six classes",
        "backbone_parameter_updates": 0,
        "post_test_tuning_allowed": False,
        "historical_test_exposure": lock["historical_test_exposure"],
        "historical_test_exposure_note": lock["historical_test_exposure_note"],
    }
    if args.preflight_only:
        path = root / "logs" / f"frozen_lowformer_e2_preflight_{stamp}.json"
        result = {
            **common,
            "status": "preflight_passed_validation_reproduced_no_test_predictions",
            "models": MODEL_IDS,
            "fixed_C": EXPECTED_C,
            "validation_predictions_reproduced": True,
            "test_evaluated": False,
        }
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print("Preflight passed; test evaluated: False")
        print("Saved report:", path)
        return

    test_idx = np.asarray(manifest["split_indices"]["test"], dtype=np.int64)
    categories = manifest["categories"]
    results = []
    for entry, source, data, classifier in prepared:
        x_test = normalize(data["features"][test_idx].astype(np.float64), norm="l2")
        truth = data["labels"][test_idx]
        predictions = classifier.predict(x_test)
        scores = classifier.decision_function(x_test)
        metrics, per_class, matrix = calculate_metrics(truth, predictions, categories)
        prediction_path = root / "results" / f"frozen_lowformer_e2_test_{stamp}.csv"
        save_predictions(prediction_path, categories, test_idx, truth, predictions, scores)
        results.append(
            {
                "model_id": entry["id"],
                "C": entry["C"],
                "cache_relative": entry["cache"],
                "cache_sha256": entry["cache_sha256"],
                "model_metadata": source["model_metadata"],
                "validation_metrics": source["validation_metrics"],
                "test_metrics": metrics,
                "test_correct_count": int(np.sum(truth == predictions)),
                "test_count": len(test_idx),
                "test_per_class": per_class,
                "test_confusion_matrix": matrix,
                "prediction_relative": prediction_path.relative_to(root).as_posix(),
                "prediction_sha256": file_hash(prediction_path),
                "classifier_fit_count": 434,
                "fit_iterations": np.asarray(classifier.n_iter_).tolist(),
            }
        )
    summary_path = root / "results" / f"frozen_lowformer_e2_final_test_summary_{stamp}.csv"
    with summary_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "Accuracy", "Precision_macro", "Recall_macro", "F1_macro", "correct", "total"])
        for row in results:
            writer.writerow(
                [
                    row["model_id"],
                    *[row["test_metrics"][name] for name in lock["metrics"]],
                    row["test_correct_count"],
                    row["test_count"],
                ]
            )
    report = {
        **common,
        "status": "locked_frozen_lowformer_e2_final_test_complete",
        "results": results,
        "test_evaluated": True,
        "test_count": len(test_idx),
        "test_indices": test_idx.tolist(),
        "categories": categories,
        "confusion_matrix_rows": "true",
        "confusion_matrix_columns": "predicted",
        "hyperparameter_search_during_final_test": False,
        "ensemble_used": False,
        "test_time_augmentation_used": False,
        "summary_relative": summary_path.relative_to(root).as_posix(),
        "summary_sha256": file_hash(summary_path),
    }
    path = root / "logs" / f"frozen_lowformer_e2_final_test_{stamp}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("\n========== LOCKED LOWFORMER-E2 FINAL TEST ==========")
    print("Precision, Recall and F1 use macro averaging.")
    for row in results:
        values = row["test_metrics"]
        print(
            f"{row['model_id']}: Accuracy={values['accuracy']:.6f} | Precision={values['macro_precision']:.6f} | "
            f"Recall={values['macro_recall']:.6f} | F1={values['macro_f1']:.6f} | "
            f"correct={row['test_correct_count']}/93"
        )
    print("Backbone parameter updates:", 0)
    print("Post-test tuning allowed:", False)
    print("Saved summary:", summary_path)
    print("Saved report:", path)


if __name__ == "__main__":
    main()
