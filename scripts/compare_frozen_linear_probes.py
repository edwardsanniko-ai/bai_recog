"""Compare cached frozen representations using one locked linear-SVM protocol."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
import sklearn

from cnn_baseline_common import file_hash


CONFIG_NAME = "frozen_linear_probe_comparison.json"


def load_cache(root: Path, relative: str, manifest_sha256: str):
    path = root / relative
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    required = {"features", "labels", "sample_indices", "categories", "manifest_sha256", "train_idx", "val_idx", "test_idx"}
    if not required.issubset(arrays):
        raise ValueError(f"Missing cache arrays in {path}: {required - set(arrays)}")
    if str(arrays["manifest_sha256"].item()) != manifest_sha256:
        raise ValueError(f"Manifest hash mismatch in {path}")
    if arrays["features"].shape[0] != 620 or arrays["labels"].shape != (620,):
        raise ValueError(f"Unexpected cache shape in {path}")
    if not np.array_equal(arrays["sample_indices"], np.arange(620)):
        raise ValueError(f"Sample order differs in {path}")
    if not np.isfinite(arrays["features"]).all():
        raise ValueError(f"Non-finite features in {path}")
    return path, arrays


def make_classifier(config: dict, c_value: float) -> LinearSVC:
    settings = config["classifier_settings"]
    return LinearSVC(
        penalty=settings["penalty"], loss=settings["loss"], dual=settings["dual"],
        C=float(c_value), class_weight=settings["class_weight"],
        max_iter=int(settings["max_iter"]), tol=float(settings["tol"]),
        random_state=int(config["C_selection"]["seed"]),
    )


def save_predictions(path: Path, categories, indices, truth, predictions, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_index", "true_class_id", "true_class", "predicted_class_id", "predicted_class",
            *[f"decision_score_{name}" for name in categories], "correct",
        ])
        for index, actual, predicted, values in zip(indices, truth, predictions, scores):
            writer.writerow([
                int(index), int(actual), categories[int(actual)], int(predicted), categories[int(predicted)],
                *[float(value) for value in values], bool(actual == predicted),
            ])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "locked_before_evaluation" or config["test_split_allowed"]:
        raise ValueError("Linear-probe protocol is not locked or test access is enabled")
    manifest_path = root / config["fixed_split"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = file_hash(manifest_path)
    categories = manifest["categories"]
    split = manifest["split_indices"]
    train_idx = np.asarray(split["train"], dtype=np.int64)
    val_idx = np.asarray(split["validation"], dtype=np.int64)
    if len(train_idx) != 434 or len(val_idx) != 93:
        raise ValueError("Fixed split counts differ")
    cv = StratifiedKFold(
        n_splits=int(config["C_selection"]["folds"]), shuffle=True,
        random_state=int(config["C_selection"]["seed"]),
    )
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    results = []
    reference_identity = None
    warnings.simplefilter("error", ConvergenceWarning)
    for model in config["models"]:
        cache_path, data = load_cache(root, model["cache"], manifest_sha256)
        identity = (
            data["labels"], data["sample_indices"], data["categories"],
            data["train_idx"], data["val_idx"], data["test_idx"],
        )
        if reference_identity is None:
            reference_identity = tuple(value.copy() for value in identity)
        elif any(not np.array_equal(value, reference) for value, reference in zip(identity, reference_identity)):
            raise ValueError(f"Cached labels, categories, or splits differ for {model['id']}")
        if not np.array_equal(data["train_idx"], train_idx) or not np.array_equal(data["val_idx"], val_idx):
            raise ValueError(f"Cached split order differs from manifest for {model['id']}")
        x = normalize(data["features"].astype(np.float64, copy=False), norm="l2", axis=1)
        y = data["labels"].astype(np.int64, copy=False)
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        c_rows = []
        for c_value in map(float, config["C_grid"]):
            fold_scores = []
            for fold_train, fold_holdout in cv.split(x_train, y_train):
                classifier = make_classifier(config, c_value)
                classifier.fit(x_train[fold_train], y_train[fold_train])
                predicted = classifier.predict(x_train[fold_holdout])
                fold_scores.append(float(f1_score(y_train[fold_holdout], predicted, average="macro", zero_division=0)))
            c_rows.append({
                "C": c_value, "fold_macro_f1": fold_scores,
                "mean_macro_f1": float(np.mean(fold_scores)),
                "sample_std_macro_f1": float(np.std(fold_scores, ddof=1)),
            })
        best_cv = sorted(c_rows, key=lambda row: (-row["mean_macro_f1"], row["C"]))[0]
        classifier = make_classifier(config, best_cv["C"])
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_val)
        scores = classifier.decision_function(x_val)
        details = classification_report(
            y_val, prediction, labels=list(range(len(categories))), target_names=categories,
            output_dict=True, zero_division=0,
        )
        metrics = {
            "accuracy": float(accuracy_score(y_val, prediction)),
            "macro_precision": float(details["macro avg"]["precision"]),
            "macro_recall": float(details["macro avg"]["recall"]),
            "macro_f1": float(details["macro avg"]["f1-score"]),
            "weighted_f1": float(details["weighted avg"]["f1-score"]),
        }
        prediction_path = root / "results" / f"frozen_linear_probe_{model['id']}_validation_{stamp}.csv"
        save_predictions(prediction_path, categories, val_idx, y_val, prediction, scores)
        result = {
            "model_id": model["id"], "cache_relative": model["cache"],
            "cache_sha256": file_hash(cache_path), "feature_dimension": int(x.shape[1]),
            "model_metadata": {
                name: str(data[name].item()) for name in ("model_id", "model_revision") if name in data
            },
            "C_cross_validation": c_rows, "selected_C": best_cv["C"],
            "selected_C_train_cv_macro_f1": best_cv["mean_macro_f1"],
            "validation_metrics": metrics, "validation_per_class": details,
            "validation_confusion_matrix": confusion_matrix(
                y_val, prediction, labels=list(range(len(categories)))
            ).tolist(),
            "prediction_relative": prediction_path.relative_to(root).as_posix(),
            "prediction_sha256": file_hash(prediction_path),
            "backbone_parameter_updates": 0, "test_evaluated": False,
        }
        results.append(result)
        print(
            f"{model['id']}: selected C={best_cv['C']:g} | "
            f"train-CV F1={best_cv['mean_macro_f1']:.6f} | "
            f"validation F1={metrics['macro_f1']:.6f}", flush=True,
        )
    ranking = sorted(results, key=lambda row: (-row["validation_metrics"]["macro_f1"], row["model_id"]))
    report = {
        "timestamp": now.isoformat(), "status": "frozen_linear_probe_validation_comparison_complete",
        "config_relative": f"configs/{CONFIG_NAME}", "config_sha256": file_hash(config_path),
        "manifest_sha256": manifest_sha256, "train_count": 434, "validation_count": 93,
        "results": results, "ranking_by_validation_macro_f1": [row["model_id"] for row in ranking],
        "selected_by_validation_macro_f1": ranking[0]["model_id"],
        "interpretation_scope": "frozen_representation_linear_separability_only",
        "backbone_parameter_updates": 0, "test_dataset_instantiated": False, "test_evaluated": False,
        "sklearn_version": sklearn.__version__,
    }
    report_path = root / "logs" / f"frozen_linear_probe_comparison_{stamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n========== FROZEN LINEAR-PROBE VALIDATION RANKING ==========")
    for rank, row in enumerate(ranking, start=1):
        print(f"{rank}. {row['model_id']}: Macro-F1={row['validation_metrics']['macro_f1']:.6f}")
    print("Selected representation:", ranking[0]["model_id"])
    print("Backbone parameter updates:", 0)
    print("Test evaluated:", False)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
