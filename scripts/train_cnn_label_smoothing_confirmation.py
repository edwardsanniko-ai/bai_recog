"""Train or resume one locked L1 label-smoothing confirmation seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_baseline_common import BaiPatternDataset, build_transforms, file_hash, load_and_verify_manifest, set_seed
from cnn_last_stage_common import load_last_stage_config, load_last_stage_model, phase_b_parameter_groups
from train_cnn_feature_screen import frozen_named_parameters, parameters_hash, train_one_epoch
from train_convnext_linear_probe import evaluate_validation, restore_rng_state, rng_state, save_predictions


CONFIG_NAME = "cnn_label_smoothing_confirmation.json"
SCREEN_CONFIG_NAME = "cnn_label_smoothing_screen.json"


def make_checkpoint(config, seed, epoch, model, optimizer, scheduler, best_f1, best_epoch,
                    patience_counter, total_updates, history, generator, config_sha256,
                    screen_config_sha256, manifest_sha256, frozen_hash, code_commit):
    return {
        "format_version": 1,
        "experiment_name": config["experiment_name"],
        "candidate_id": config["candidate_id"],
        "seed": seed,
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
        "screen_config_sha256": screen_config_sha256,
        "manifest_sha256": manifest_sha256,
        "frozen_parameter_sha256": frozen_hash,
        "code_commit_before_run": code_commit,
        "test_evaluated": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    screen_config_path = root / "configs" / SCREEN_CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    screen_config = json.loads(screen_config_path.read_text(encoding="utf-8"))
    seed = int(args.seed)
    seeds = {int(value) for value in config["confirmatory_seeds"]}
    if seed not in seeds:
        raise ValueError(f"Seed {seed} is outside the locked L1 confirmation set: {sorted(seeds)}")
    if config["status"] != "locked_before_confirmation" or config["data"]["test_split_allowed"]:
        raise ValueError("L1 confirmation protocol is not locked or test access is enabled")
    if float(config["label_smoothing"]) != float(screen_config["label_smoothing"]):
        raise ValueError("Label smoothing differs between confirmation and screen configs")

    stage_b_config = load_last_stage_config(root)
    max_epochs = int(stage_b_config["max_epochs"])
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= max_epochs:
        raise ValueError("--stop-after-epoch is outside the configured range")
    set_seed(seed)
    config_sha256 = file_hash(config_path)
    screen_config_sha256 = file_hash(screen_config_path)
    manifest = load_and_verify_manifest(root, args.data_root, stage_b_config, verify_hashes=True)
    manifest_path = root / stage_b_config["split"]["manifest"]
    manifest_sha256 = file_hash(manifest_path)
    records, split = manifest["records"], manifest["split_indices"]
    train_transform, eval_transform = build_transforms(stage_b_config)
    train_dataset = BaiPatternDataset(records, split["train"], args.data_root, train_transform)
    validation_dataset = BaiPatternDataset(records, split["validation"], args.data_root, eval_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=stage_b_config["batch_size"], shuffle=True, num_workers=0,
        generator=generator, drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=stage_b_config["batch_size"], shuffle=False, num_workers=0,
        drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weight_path, initialization_path, initialization = load_last_stage_model(
        root, stage_b_config, device, source_seed=seed
    )
    stage, head = phase_b_parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": stage, "lr": stage_b_config["last_stage_learning_rate"], "group_name": "last_stage"},
            {"params": head, "lr": stage_b_config["head_learning_rate"], "group_name": "head"},
        ], weight_decay=stage_b_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    checkpoint_dir = root / "checkpoints" / f"cnn_label_smoothing_L1_seed{seed}"
    last_path, best_path = checkpoint_dir / "last.pt", checkpoint_dir / "best.pt"
    start_epoch, patience_counter, total_updates = 1, 0, 0
    best_f1 = float(initialization["best_validation_macro_f1"])
    best_epoch = 0
    history = []
    frozen_hash = parameters_hash(frozen_named_parameters(model))
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    code_commit = git.stdout.strip() if git.returncode == 0 else None

    if args.resume is not None:
        if args.resume.resolve() != last_path.resolve():
            raise ValueError(f"Resume must use {last_path}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_sha256 or checkpoint["screen_config_sha256"] != screen_config_sha256:
            raise ValueError("Resume checkpoint protocol identity differs")
        if checkpoint["seed"] != seed:
            raise ValueError("Resume checkpoint seed differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_f1, best_epoch = float(checkpoint["best_validation_macro_f1"]), int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        total_updates, history = int(checkpoint["total_optimizer_updates"]), checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"], generator)
        if patience_counter >= stage_b_config["early_stopping_patience"] or start_epoch > max_epochs:
            print("L1 confirmation checkpoint already completed; no training performed.")
            print("Best epoch:", best_epoch)
            print("Best validation Macro-F1:", f"{best_f1:.6f}")
            print("Test evaluated:", False)
            return
        print(f"Resumed seed {seed} after epoch {start_epoch - 1}; next epoch is {start_epoch}.")
    elif checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint directory exists; resume with {last_path}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        initial = make_checkpoint(
            config, seed, 0, model, optimizer, scheduler, best_f1, 0, 0, 0, [], generator,
            config_sha256, screen_config_sha256, manifest_sha256, frozen_hash, code_commit,
        )
        torch.save(initial, last_path)
        torch.save(initial, best_path)

    final_epoch, stop_reason = start_epoch - 1, "max_epochs"
    for epoch in range(start_epoch, max_epochs + 1):
        print(f"L1 seed {seed} epoch {epoch}/{max_epochs}", flush=True)
        train_loss, updates = train_one_epoch(model, train_loader, criterion, optimizer, device, frozen_hash)
        total_updates += updates
        metrics, per_class, matrix, indices, truth, predictions, probabilities = evaluate_validation(
            model, validation_loader, device, manifest["categories"]
        )
        improved = metrics["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, patience_counter = float(metrics["macro_f1"]), epoch, 0
        else:
            patience_counter += 1
        stage_lr, head_lr = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
        scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": train_loss, "validation_metrics": metrics,
            "stage_learning_rate": stage_lr, "head_learning_rate": head_lr,
            "improved": improved, "patience_counter": patience_counter,
        })
        prediction_path = root / "results" / f"cnn_label_smoothing_L1_seed{seed}_validation_epoch{epoch:03d}.csv"
        save_predictions(prediction_path, manifest["categories"], indices, truth, predictions, probabilities)
        checkpoint = make_checkpoint(
            config, seed, epoch, model, optimizer, scheduler, best_f1, best_epoch,
            patience_counter, total_updates, history, generator, config_sha256,
            screen_config_sha256, manifest_sha256, frozen_hash, code_commit,
        )
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, best_path)
        report = {
            "timestamp": datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).isoformat(),
            "status": "label_smoothing_confirmation_epoch_complete_validation_only",
            "candidate": "L1", "seed": seed, "epoch": epoch,
            "label_smoothing": config["label_smoothing"], "train_loss": train_loss,
            "validation_metrics": metrics, "per_class": per_class, "confusion_matrix": matrix,
            "best_epoch_so_far": best_epoch, "best_validation_macro_f1_so_far": best_f1,
            "patience_counter": patience_counter, "total_optimizer_updates": total_updates,
            "train_images": len(train_dataset), "validation_images": len(validation_dataset),
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "source_phase_a_checkpoint": initialization_path.relative_to(root).as_posix(),
            "source_phase_a_best_validation_macro_f1": initialization["best_validation_macro_f1"],
            "prediction_relative": prediction_path.relative_to(root).as_posix(),
            "prediction_sha256": file_hash(prediction_path), "config_sha256": config_sha256,
            "screen_config_sha256": screen_config_sha256, "weights_sha256": file_hash(weight_path),
            "frozen_parameters_unchanged": parameters_hash(frozen_named_parameters(model)) == frozen_hash,
            "test_dataset_instantiated": False, "test_evaluated": False,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__, "sklearn_version": sklearn.__version__,
        }
        report_path = root / "logs" / f"cnn_label_smoothing_L1_seed{seed}_epoch{epoch:03d}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"L1 seed {seed} epoch {epoch} complete | train loss {train_loss:.4f} | "
            f"val Macro-F1 {metrics['macro_f1']:.4f} | best {best_f1:.4f} | patience {patience_counter}",
            flush=True,
        )
        final_epoch = epoch
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            stop_reason = "requested_epoch_checkpoint"
            break
        if patience_counter >= stage_b_config["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break
    print("Stopped after epoch:", final_epoch)
    print("Stop reason:", stop_reason)
    print("Best epoch:", best_epoch)
    print("Best validation Macro-F1:", f"{best_f1:.6f}")
    print("Test evaluated:", False)
    print("Resume command uses:", last_path)


if __name__ == "__main__":
    main()
