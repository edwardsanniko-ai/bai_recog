"""Evaluate the four locked ConvNeXt checkpoints once on the final test split."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_baseline_common import BaiPatternDataset, build_transforms, file_hash, load_and_verify_manifest
from cnn_last_stage_common import load_last_stage_config, load_last_stage_model
from train_convnext_linear_probe import evaluate_validation


def save_test_predictions(path, categories, indices, truth, predictions, probabilities):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "sample_index", "true_class_id", "true_class", "predicted_class_id",
            "predicted_class", *[f"probability_{name}" for name in categories], "correct",
        ])
        for index, actual, predicted, probs in zip(indices, truth, predictions, probabilities):
            writer.writerow([
                index, actual, categories[actual], predicted, categories[predicted],
                *[float(value) for value in probs], bool(actual == predicted),
            ])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "configs" / "cnn_convnext_final_test.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_test_evaluation":
        raise ValueError("Final-test protocol was not locked before evaluation")
    if lock["ensemble"] or lock["test_time_augmentation"] or lock["post_test_model_or_hyperparameter_selection"]:
        raise ValueError("Unsupported final-test behavior in locked protocol")
    if list(map(int, lock["seeds"])) != [43, 44, 45, 46]:
        raise ValueError("Locked final-test seeds differ")
    existing_reports = list((root / "logs").glob("cnn_convnext_final_test_*.json"))
    existing_predictions = list((root / "results").glob("cnn_convnext_seed*_test_*.csv"))
    if existing_reports or existing_predictions:
        raise FileExistsError(
            "Final CNN test artifacts already exist; refusing to re-evaluate or overwrite them"
        )
    validation_summary = root / lock["validation_confirmation"]["summary"]
    summary = json.loads(validation_summary.read_text(encoding="utf-8"))
    if summary["test_evaluated"] or summary["seeds"] != lock["seeds"]:
        raise ValueError("Validation-only confirmation summary differs from final-test lock")
    config = load_last_stage_config(root)
    manifest = load_and_verify_manifest(root, root.parent / "bai_data", config, verify_hashes=True)
    if len(manifest["split_indices"]["test"]) != lock["test_count"]:
        raise ValueError("Test count differs from locked protocol")
    _, eval_transform = build_transforms(config)
    test_dataset = BaiPatternDataset(
        manifest["records"], manifest["split_indices"]["test"], root.parent / "bai_data", eval_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    results = []
    for entry in lock["locked_checkpoints"]:
        seed = int(entry["seed"])
        checkpoint_path = root / entry["relative_path"]
        if file_hash(checkpoint_path) != entry["sha256"].lower():
            raise ValueError(f"Locked checkpoint SHA256 differs for seed {seed}")
        model, _, _, _ = load_last_stage_model(root, config, device, source_seed=seed)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["seed"] != seed or checkpoint["best_epoch"] != entry["best_phase_b_epoch"]:
            raise ValueError(f"Locked checkpoint identity differs for seed {seed}")
        if checkpoint["best_validation_macro_f1"] != entry["validation_macro_f1"]:
            raise ValueError(f"Locked validation metric differs for seed {seed}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics, per_class, matrix, indices, truth, predictions, probabilities = evaluate_validation(
            model, test_loader, device, manifest["categories"]
        )
        prediction_path = root / "results" / f"cnn_convnext_seed{seed}_test_{run_id}.csv"
        save_test_predictions(
            prediction_path, manifest["categories"], indices, truth, predictions, probabilities
        )
        results.append({
            "seed": seed,
            "best_phase_b_epoch": entry["best_phase_b_epoch"],
            "locked_checkpoint_relative": entry["relative_path"],
            "locked_checkpoint_sha256": entry["sha256"].lower(),
            "validation_accuracy": entry["validation_accuracy"],
            "validation_macro_f1": entry["validation_macro_f1"],
            "test_metrics": metrics,
            "test_per_class": per_class,
            "test_confusion_matrix": matrix,
            "confusion_matrix_rows": "true",
            "confusion_matrix_columns": "predicted",
            "test_predictions_relative": prediction_path.relative_to(root).as_posix(),
            "test_predictions_sha256": file_hash(prediction_path),
        })
        print(
            f"seed {seed}: test accuracy {metrics['accuracy']:.6f}, "
            f"test Macro-F1 {metrics['macro_f1']:.6f}"
        )
        del model
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    aggregate = {}
    for name in metric_names:
        values = np.array([row["test_metrics"][name] for row in results], dtype=np.float64)
        aggregate[name] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
        }
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    report = {
        "timestamp": now.isoformat(),
        "status": "locked_cnn_final_test_complete_no_post_test_tuning",
        "model": lock["model"],
        "code_commit_before_test": git.stdout.strip() if git.returncode == 0 else None,
        "lock_relative": lock_path.relative_to(root).as_posix(),
        "lock_sha256": file_hash(lock_path),
        "script_sha256": file_hash(Path(__file__)),
        "manifest_sha256": file_hash(root / config["split"]["manifest"]),
        "validation_summary_relative": validation_summary.relative_to(root).as_posix(),
        "validation_summary_sha256": file_hash(validation_summary),
        "test_split": "test",
        "test_count": len(test_dataset),
        "test_indices": manifest["split_indices"]["test"],
        "seeds": lock["seeds"],
        "ensemble_used": False,
        "test_time_augmentation_used": False,
        "optimizer_updates_during_test": 0,
        "results": results,
        "aggregate": aggregate,
        "post_test_tuning_allowed": False,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    report_path = root / "logs" / f"cnn_convnext_final_test_{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nLOCKED FINAL TEST SUMMARY")
    for name in metric_names:
        print(f"{name}: {aggregate[name]['mean']:.6f} +/- {aggregate[name]['sample_std']:.6f}")
    print("Optimizer updates during test:", 0)
    print("Post-test tuning allowed:", False)
    print("Saved report:", report_path)


if __name__ == "__main__":
    main()
