"""Train or resume phase-B ConvNeXt-Tiny last-stage fine-tuning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_baseline_common import (
    BaiPatternDataset,
    build_transforms,
    file_hash,
    load_and_verify_manifest,
    set_seed,
)
from cnn_last_stage_common import (
    CONFIG_NAME,
    load_last_stage_config,
    load_last_stage_model,
    phase_b_parameter_groups,
    phase_b_parameter_names,
)
from train_convnext_linear_probe import (
    evaluate_validation,
    restore_rng_state,
    rng_state,
    save_predictions,
)


def parameters_hash(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def selected_named_parameters(model: nn.Module, prefixes: tuple[str, ...]):
    return [(name, value) for name, value in model.named_parameters() if name.startswith(prefixes)]


def frozen_named_parameters(model: nn.Module):
    return [(name, value) for name, value in model.named_parameters() if not value.requires_grad]


def train_one_epoch(model, loader, criterion, optimizer, device, frozen_hash: str):
    # Only the trainable last stage and classifier enter training mode. Earlier
    # frozen modules remain deterministic and cannot update any state.
    model.eval()
    model.features[7].train()
    model.classifier.train()
    stage_named = selected_named_parameters(model, ("features.7.",))
    head_named = selected_named_parameters(model, ("classifier.0.", "classifier.3."))
    stage_before = parameters_hash(stage_named)
    head_before = parameters_hash(head_named)
    total_loss, total_samples, updates = 0.0, 0, 0
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
    stage_after = parameters_hash(stage_named)
    head_after = parameters_hash(head_named)
    if stage_before == stage_after:
        raise RuntimeError("Trainable last-stage parameters did not change")
    if head_before == head_after:
        raise RuntimeError("Trainable head parameters did not change")
    if parameters_hash(frozen_named_parameters(model)) != frozen_hash:
        raise RuntimeError("A frozen parameter changed")
    if total_samples != len(loader.dataset):
        raise RuntimeError("Training epoch did not see each dataset position once")
    return {
        "loss": total_loss / total_samples,
        "updates": updates,
        "stage_before": stage_before,
        "stage_after": stage_after,
        "head_before": head_before,
        "head_after": head_after,
    }


def checkpoint_payload(
    config, seed, completed_epoch, model, optimizer, scheduler, best_f1, best_epoch,
    patience_counter, total_updates, history, generator, config_sha256,
    manifest_sha256, frozen_hash, code_commit, initialization_sha256,
):
    return {
        "format_version": 1,
        "experiment_name": config["experiment_name"],
        "phase": config["phase"],
        "seed": seed,
        "max_epochs": config["max_epochs"],
        "completed_epoch": completed_epoch,
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
        "initialization_checkpoint_sha256": initialization_sha256,
        "code_commit_before_run": code_commit,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    args = parser.parse_args()
    config = load_last_stage_config(root)
    seed = int(config["development_seed"])
    max_epochs = int(config["max_epochs"])
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= max_epochs:
        raise ValueError("--stop-after-epoch must be within the configured epoch range")
    preflight_paths = sorted((root / "logs").glob("cnn_convnext_last_stage_preflight_*.json"))
    if len(preflight_paths) != 1:
        raise RuntimeError("Expected exactly one locked phase-B preflight report")
    preflight = json.loads(preflight_paths[0].read_text(encoding="utf-8"))
    if (
        preflight["status"] != "phase_b_last_stage_preflight_passed_no_updates"
        or preflight["config"] != config
        or preflight["optimizer_updates"] != 0
        or preflight["test_evaluated"]
    ):
        raise ValueError("Phase-B preflight report differs from the locked protocol")
    set_seed(seed)
    config_path = root / "configs" / CONFIG_NAME
    config_sha256 = file_hash(config_path)
    manifest_path = root / config["split"]["manifest"]
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
    model, weight_path, initialization_path, _ = load_last_stage_model(root, config, device)
    initialization_sha256 = file_hash(initialization_path)
    last_stage, head = phase_b_parameter_groups(model)
    names = phase_b_parameter_names(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": last_stage, "lr": config["last_stage_learning_rate"], "group_name": "last_stage"},
            {"params": head, "lr": config["head_learning_rate"], "group_name": "head"},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    checkpoint_dir = root / "checkpoints" / f"cnn_convnext_last_stage_seed{seed}"
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    start_epoch = 1
    best_f1 = float(config["initialization"]["source_validation_macro_f1"])
    best_epoch = 0
    patience_counter = 0
    total_updates = 0
    history = []
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    code_commit = git.stdout.strip() if git.returncode == 0 else None
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if resume_path != last_checkpoint.resolve():
            raise ValueError(f"Resume path must be the audited last checkpoint: {last_checkpoint}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if (
            checkpoint["config_sha256"] != config_sha256
            or checkpoint["manifest_sha256"] != manifest_sha256
            or checkpoint["initialization_checkpoint_sha256"] != initialization_sha256
        ):
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
            print(f"Checkpoint already reached early stopping after phase-B epoch {start_epoch - 1}.")
            print("No training or parameter update was performed.")
            print("Best phase-B epoch (0 means phase-A initialization):", best_epoch)
            print("Best validation Macro-F1:", f"{best_f1:.6f}")
            print("Test evaluated:", False)
            return
        if start_epoch > max_epochs:
            print(f"Checkpoint already completed the configured {max_epochs} phase-B epochs.")
            print("No training or parameter update was performed.")
            print("Best phase-B epoch (0 means phase-A initialization):", best_epoch)
            print("Best validation Macro-F1:", f"{best_f1:.6f}")
            print("Test evaluated:", False)
            return
        print(f"Resumed after phase-B epoch {start_epoch - 1}; next epoch is {start_epoch}.")
    elif checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint directory already exists; use --resume with {last_checkpoint}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frozen_hash = parameters_hash(frozen_named_parameters(model))
    if args.resume is None:
        initial_payload = checkpoint_payload(
            config, seed, 0, model, optimizer, scheduler, best_f1, 0, 0, 0, [], generator,
            config_sha256, manifest_sha256, frozen_hash, code_commit, initialization_sha256,
        )
        torch.save(initial_payload, best_checkpoint)
    final_epoch = start_epoch - 1
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, max_epochs + 1):
        stage_lr = optimizer.param_groups[0]["lr"]
        head_lr = optimizer.param_groups[1]["lr"]
        print(f"Phase-B epoch {epoch}/{max_epochs} | stage lr {stage_lr:.8f} | head lr {head_lr:.8f}", flush=True)
        train_result = train_one_epoch(model, train_loader, criterion, optimizer, device, frozen_hash)
        total_updates += train_result["updates"]
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
        scheduler.step()
        history.append({
            "epoch": epoch,
            "stage_learning_rate": stage_lr,
            "head_learning_rate": head_lr,
            "train_loss": train_result["loss"],
            "optimizer_updates": train_result["updates"],
            "validation_metrics": metrics,
            "improved_over_all_previous_including_phase_a_initialization": improved,
            "patience_counter": patience_counter,
        })
        prediction_path = root / "results" / f"cnn_convnext_last_stage_seed{seed}_validation_epoch{epoch:03d}.csv"
        save_predictions(prediction_path, manifest["categories"], indices, truth, predictions, probabilities)
        payload = checkpoint_payload(
            config, seed, epoch, model, optimizer, scheduler, best_f1, best_epoch,
            patience_counter, total_updates, history, generator, config_sha256,
            manifest_sha256, frozen_hash, code_commit, initialization_sha256,
        )
        torch.save(payload, last_checkpoint)
        if improved:
            torch.save(payload, best_checkpoint)
        now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
        report = {
            "timestamp": now.isoformat(),
            "status": "phase_b_epoch_completed_resume_available",
            "config_relative": f"configs/{CONFIG_NAME}",
            "config_sha256": config_sha256,
            "preflight_relative": preflight_paths[0].relative_to(root).as_posix(),
            "preflight_sha256": file_hash(preflight_paths[0]),
            "manifest_sha256": manifest_sha256,
            "weights_sha256": file_hash(weight_path),
            "initialization_checkpoint_sha256": initialization_sha256,
            "script_sha256": file_hash(Path(__file__)),
            "last_stage_helper_sha256": file_hash(root / "scripts" / "cnn_last_stage_common.py"),
            "shared_model_helper_sha256": file_hash(root / "scripts" / "cnn_baseline_common.py"),
            "shared_training_helper_sha256": file_hash(root / "scripts" / "train_convnext_linear_probe.py"),
            "code_commit_before_run": code_commit,
            "seed": seed,
            "phase_b_epoch": epoch,
            "max_epochs": max_epochs,
            "phase_a_initialization_validation_macro_f1": config["initialization"]["source_validation_macro_f1"],
            "train_images": len(train_dataset),
            "validation_images": len(validation_dataset),
            "train_batches": len(train_loader),
            "validation_batches": len(validation_loader),
            "train_loss": train_result["loss"],
            "validation_metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": matrix,
            "confusion_matrix_rows": "true",
            "confusion_matrix_columns": "predicted",
            "last_stage_learning_rate": stage_lr,
            "head_learning_rate": head_lr,
            "epoch_optimizer_updates": train_result["updates"],
            "total_optimizer_updates": total_updates,
            "trainable_parameter_count": sum(value.numel() for value in last_stage + head),
            "trainable_parameter_names": names,
            "last_stage_parameters_changed": train_result["stage_before"] != train_result["stage_after"],
            "head_parameters_changed": train_result["head_before"] != train_result["head_after"],
            "frozen_parameter_sha256": frozen_hash,
            "frozen_parameters_unchanged": parameters_hash(frozen_named_parameters(model)) == frozen_hash,
            "best_phase_b_epoch_so_far": best_epoch,
            "best_epoch_zero_means_phase_a_initialization": True,
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
        }
        report_path = root / "logs" / f"cnn_convnext_last_stage_seed{seed}_epoch{epoch:03d}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"Phase-B epoch {epoch} complete | train loss {train_result['loss']:.4f} | "
            f"val accuracy {metrics['accuracy']:.4f} | val Macro-F1 {metrics['macro_f1']:.4f}",
            flush=True,
        )
        print("  last stage changed:", True, "| head changed:", True, "| frozen parameters unchanged:", True)
        print("  improved over phase-A/best-so-far:", improved, "| patience:", patience_counter)
        print("  report:", report_path)
        final_epoch = epoch
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            stop_reason = "requested_epoch_checkpoint"
            break
        if patience_counter >= config["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break
    print("Stopped after phase-B epoch:", final_epoch)
    print("Stop reason:", stop_reason)
    print("Best phase-B epoch so far (0 means phase-A initialization):", best_epoch)
    print("Best validation Macro-F1 so far:", f"{best_f1:.6f}")
    print("Total phase-B optimizer updates:", total_updates)
    print("Test evaluated:", False)
    print("Resume command uses:", last_checkpoint)


if __name__ == "__main__":
    main()
