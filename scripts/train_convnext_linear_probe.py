"""Train or resume the audited ConvNeXt-Tiny head-only CNN baseline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_baseline_common import (
    BaiPatternDataset,
    CONFIG_NAME,
    build_transforms,
    file_hash,
    load_and_verify_manifest,
    load_config,
    load_head_only_model,
    set_seed,
    trainable_parameter_names,
)


def parameter_hash(model: nn.Module, trainable: bool) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def restore_rng_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # The checkpoint is loaded with map_location=device so model/optimizer
    # tensors are ready for CUDA. RNG states are an exception: PyTorch's CPU
    # generator and DataLoader generator require CPU uint8 tensors, while
    # torch.cuda.set_rng_state_all also expects the states in their canonical
    # CPU ByteTensor representation.
    torch_cpu_state = state["torch_cpu"].detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(torch_cpu_state)
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        cuda_states = [value.detach().to(device="cpu", dtype=torch.uint8) for value in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)
    loader_state = state["loader_generator"].detach().to(device="cpu", dtype=torch.uint8)
    generator.set_state(loader_state)


def train_one_epoch(model, loader, criterion, optimizer, device, frozen_hash: str):
    # Keep the frozen ConvNeXt backbone in inference mode so stochastic depth is
    # disabled, while enabling Dropout only inside the trainable classifier.
    model.eval()
    model.classifier.train()
    total_loss = 0.0
    total_samples = 0
    updates = 0
    before_trainable = parameter_hash(model, trainable=True)
    for batch_number, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        batch_size = len(labels)
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
        updates += 1
        if batch_number == 1 or batch_number % 5 == 0 or batch_number == len(loader):
            print(f"  train batch {batch_number}/{len(loader)} | loss {float(loss.detach()):.4f}", flush=True)
    after_trainable = parameter_hash(model, trainable=True)
    after_frozen = parameter_hash(model, trainable=False)
    if before_trainable == after_trainable:
        raise RuntimeError("Trainable head parameters did not change")
    if frozen_hash != after_frozen:
        raise RuntimeError("A frozen backbone parameter changed")
    if total_samples != len(loader.dataset):
        raise RuntimeError("Training epoch did not see each dataset position once")
    return total_loss / total_samples, updates, before_trainable, after_trainable


def evaluate_validation(model, loader, device, categories: list[str]):
    model.eval()
    truth, predictions, sample_indices, probabilities = [], [], [], []
    with torch.inference_mode():
        for images, labels, indices in loader:
            logits = model(images.to(device, non_blocking=True))
            probs = logits.softmax(dim=1).cpu()
            if not torch.isfinite(probs).all():
                raise RuntimeError("Non-finite validation probability")
            truth.extend(labels.tolist())
            predictions.extend(probs.argmax(dim=1).tolist())
            sample_indices.extend(indices.tolist())
            probabilities.extend(probs.tolist())
    truth_array = np.asarray(truth, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)
    class_ids = list(range(len(categories)))
    details = classification_report(
        truth_array,
        prediction_array,
        labels=class_ids,
        target_names=categories,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(truth_array, prediction_array)),
        "macro_precision": float(details["macro avg"]["precision"]),
        "macro_recall": float(details["macro avg"]["recall"]),
        "macro_f1": float(details["macro avg"]["f1-score"]),
        "weighted_f1": float(details["weighted avg"]["f1-score"]),
    }
    matrix = confusion_matrix(truth_array, prediction_array, labels=class_ids)
    if len(truth) != len(loader.dataset) or matrix.sum() != len(loader.dataset):
        raise RuntimeError("Validation accounting failed")
    return metrics, details, matrix.tolist(), sample_indices, truth, predictions, probabilities


def save_predictions(path: Path, categories, indices, truth, predictions, probabilities) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    args = parser.parse_args()
    config = load_config(root)
    seed = int(config["development_seed"])
    max_epochs = int(config["max_epochs"])
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= max_epochs:
        raise ValueError("--stop-after-epoch must be within the configured epoch range")
    set_seed(seed)
    config_path = root / "configs" / CONFIG_NAME
    manifest_path = root / config["split"]["manifest"]
    config_sha256 = file_hash(config_path)
    print("Verifying all 620 audited image hashes ...", flush=True)
    manifest = load_and_verify_manifest(root, args.data_root, config, verify_hashes=True)
    manifest_sha256 = file_hash(manifest_path)
    records, split = manifest["records"], manifest["split_indices"]
    train_transform, eval_transform = build_transforms(config)
    train_dataset = BaiPatternDataset(records, split["train"], args.data_root, train_transform)
    validation_dataset = BaiPatternDataset(records, split["validation"], args.data_root, eval_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=generator,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weight_path = load_head_only_model(root, config, device)
    names = trainable_parameter_names(model)
    if any(not name.startswith(("classifier.0.", "classifier.3.")) for name in names):
        raise RuntimeError(f"Unexpected trainable parameters: {names}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["head_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    checkpoint_dir = root / "checkpoints" / f"cnn_convnext_linear_probe_seed{seed}"
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    start_epoch, best_f1, best_epoch, patience_counter = 1, float("-inf"), 0, 0
    total_updates = 0
    history = []
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if resume_path != last_checkpoint.resolve():
            raise ValueError(f"Resume path must be the audited last checkpoint: {last_checkpoint}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_sha256 or checkpoint["manifest_sha256"] != manifest_sha256:
            raise ValueError("Resume checkpoint protocol identity differs")
        if checkpoint["seed"] != seed or checkpoint["max_epochs"] != max_epochs:
            raise ValueError("Resume checkpoint seed or schedule differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["completed_epoch"] + 1
        best_f1 = checkpoint["best_validation_macro_f1"]
        best_epoch = checkpoint["best_epoch"]
        patience_counter = checkpoint["patience_counter"]
        total_updates = checkpoint["total_optimizer_updates"]
        history = checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"], generator)
        if patience_counter >= config["early_stopping_patience"]:
            print(f"Checkpoint already reached early stopping after epoch {start_epoch - 1}.")
            print("No training or parameter update was performed.")
            print("Best epoch:", best_epoch)
            print("Best validation Macro-F1:", f"{best_f1:.6f}")
            print("Test evaluated:", False)
            return
        if start_epoch > max_epochs:
            print(f"Checkpoint already completed the configured {max_epochs} epochs.")
            print("No training or parameter update was performed.")
            print("Best epoch:", best_epoch)
            print("Best validation Macro-F1:", f"{best_f1:.6f}")
            print("Test evaluated:", False)
            return
        print(f"Resumed after epoch {start_epoch - 1}; next epoch is {start_epoch}.")
    elif checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint directory already exists; use --resume with {last_checkpoint}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frozen_hash = parameter_hash(model, trainable=False)
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    code_commit = git.stdout.strip() if git.returncode == 0 else None
    final_epoch = start_epoch - 1
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, max_epochs + 1):
        print(f"Epoch {epoch}/{max_epochs} | lr {optimizer.param_groups[0]['lr']:.8f}", flush=True)
        train_loss, epoch_updates, head_before, head_after = train_one_epoch(
            model, train_loader, criterion, optimizer, device, frozen_hash
        )
        total_updates += epoch_updates
        metrics, per_class, matrix, indices, truth, predictions, probabilities = evaluate_validation(
            model, validation_loader, device, manifest["categories"]
        )
        improved = metrics["macro_f1"] > best_f1
        if improved:
            best_f1 = metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "optimizer_updates": epoch_updates,
            "validation_metrics": metrics,
            "improved": improved,
            "patience_counter": patience_counter,
        })
        prediction_path = root / "results" / f"cnn_convnext_linear_probe_seed{seed}_validation_epoch{epoch:03d}.csv"
        save_predictions(prediction_path, manifest["categories"], indices, truth, predictions, probabilities)
        checkpoint = {
            "format_version": 1,
            "experiment_name": config["experiment_name"],
            "seed": seed,
            "max_epochs": max_epochs,
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "total_optimizer_updates": total_updates,
            "history": history,
            "rng_state": rng_state(generator),
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "frozen_parameter_sha256": frozen_hash,
            "code_commit_before_run": code_commit,
        }
        torch.save(checkpoint, last_checkpoint)
        if improved:
            torch.save(checkpoint, best_checkpoint)
        now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
        report = {
            "timestamp": now.isoformat(),
            "status": "epoch_completed_resume_available",
            "config_relative": f"configs/{CONFIG_NAME}",
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "weights_sha256": file_hash(weight_path),
            "script_sha256": file_hash(Path(__file__)),
            "helper_sha256": file_hash(root / "scripts" / "cnn_baseline_common.py"),
            "code_commit_before_run": code_commit,
            "seed": seed,
            "epoch": epoch,
            "max_epochs": max_epochs,
            "train_images": len(train_dataset),
            "validation_images": len(validation_dataset),
            "train_batches": len(train_loader),
            "validation_batches": len(validation_loader),
            "train_loss": train_loss,
            "validation_metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": matrix,
            "confusion_matrix_rows": "true",
            "confusion_matrix_columns": "predicted",
            "learning_rate": current_lr,
            "epoch_optimizer_updates": epoch_updates,
            "total_optimizer_updates": total_updates,
            "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
            "trainable_parameter_names": names,
            "head_parameter_hash_before_epoch": head_before,
            "head_parameter_hash_after_epoch": head_after,
            "head_parameters_changed": head_before != head_after,
            "frozen_parameter_sha256": frozen_hash,
            "frozen_parameters_unchanged": parameter_hash(model, trainable=False) == frozen_hash,
            "best_epoch_so_far": best_epoch,
            "best_validation_macro_f1_so_far": best_f1,
            "patience_counter": patience_counter,
            "prediction_relative": prediction_path.relative_to(root).as_posix(),
            "prediction_sha256": file_hash(prediction_path),
            "last_checkpoint_relative": last_checkpoint.relative_to(root).as_posix(),
            "best_checkpoint_relative": best_checkpoint.relative_to(root).as_posix(),
            "resume_supported": True,
            "test_dataset_instantiated": False,
            "test_evaluated": False,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "sklearn_version": sklearn.__version__,
        }
        report_path = root / "logs" / f"cnn_convnext_linear_probe_seed{seed}_epoch{epoch:03d}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch} complete | train loss {train_loss:.4f} | "
            f"val accuracy {metrics['accuracy']:.4f} | val Macro-F1 {metrics['macro_f1']:.4f}",
            flush=True,
        )
        print("  head changed:", True, "| frozen backbone unchanged:", True, "| updates:", epoch_updates)
        print("  report:", report_path)
        final_epoch = epoch
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            stop_reason = "requested_epoch_checkpoint"
            break
        if patience_counter >= config["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break
    print("Stopped after epoch:", final_epoch)
    print("Stop reason:", stop_reason)
    print("Best epoch so far:", best_epoch)
    print("Best validation Macro-F1 so far:", f"{best_f1:.6f}")
    print("Total optimizer updates:", total_updates)
    print("Test evaluated:", False)
    print("Resume command uses:", last_checkpoint)


if __name__ == "__main__":
    main()
