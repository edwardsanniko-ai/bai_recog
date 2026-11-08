"""Train or resume one locked ConvNeXt resolution/pooling screen variant."""

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

from cnn_baseline_common import BaiPatternDataset, file_hash, load_and_verify_manifest, set_seed
from cnn_feature_screen_common import (
    CONFIG_NAME,
    build_screen_model,
    load_screen_config,
    parameter_groups,
    screen_transforms,
    variant_config,
)
from cnn_last_stage_common import load_last_stage_config
from train_convnext_linear_probe import evaluate_validation, restore_rng_state, rng_state, save_predictions


def parameters_hash(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def named_group(model: nn.Module, prefix: str):
    return [(name, value) for name, value in model.named_parameters() if name.startswith(prefix)]


def frozen_named_parameters(model: nn.Module):
    return [(name, value) for name, value in model.named_parameters() if not value.requires_grad]


def train_one_epoch(model, loader, criterion, optimizer, device, frozen_hash: str):
    model.eval()
    model.features[7].train()
    model.classifier.train()
    stage_named = named_group(model, "features.7.")
    head_named = named_group(model, "classifier.")
    stage_before = parameters_hash(stage_named)
    head_before = parameters_hash(head_named)
    total_loss, total_samples, updates = 0.0, 0, 0
    for batch_number, (images, labels, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
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
    if parameters_hash(stage_named) == stage_before or parameters_hash(head_named) == head_before:
        raise RuntimeError("A trainable parameter group did not change")
    if parameters_hash(frozen_named_parameters(model)) != frozen_hash:
        raise RuntimeError("A frozen parameter changed")
    if total_samples != len(loader.dataset):
        raise RuntimeError("Training sample accounting failed")
    return total_loss / total_samples, updates


def checkpoint_payload(
    config, variant, completed_epoch, model, optimizer, scheduler, best_f1, best_epoch,
    patience_counter, total_updates, history, generator, config_sha256, manifest_sha256,
    frozen_hash, code_commit,
):
    return {
        "format_version": 1,
        "experiment_name": config["experiment_name"],
        "variant": variant,
        "seed": config["source_seed"],
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
        "code_commit_before_run": code_commit,
        "test_evaluated": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["B1", "B2", "B3"])
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    args = parser.parse_args()
    config = load_screen_config(root)
    variant = variant_config(config, args.variant)
    seed = int(config["source_seed"])
    max_epochs = int(config["max_epochs"])
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= max_epochs:
        raise ValueError("--stop-after-epoch is outside the configured range")
    preflights = sorted((root / "logs").glob("cnn_feature_screen_preflight_*.json"))
    if len(preflights) != 1:
        raise RuntimeError("Expected exactly one feature-screen preflight")
    preflight = json.loads(preflights[0].read_text(encoding="utf-8"))
    if (
        preflight["status"] != "cnn_feature_screen_preflight_passed_no_updates"
        or preflight["config"] != config
        or preflight["optimizer_updates"] != 0
        or preflight["test_evaluated"]
    ):
        raise ValueError("Feature-screen preflight differs from the locked protocol")
    set_seed(seed)
    config_path = root / "configs" / CONFIG_NAME
    config_sha256 = file_hash(config_path)
    stage_b_config = load_last_stage_config(root)
    print("Verifying all 620 audited image hashes ...", flush=True)
    manifest = load_and_verify_manifest(root, args.data_root, stage_b_config, verify_hashes=True)
    records, split = manifest["records"], manifest["split_indices"]
    if set(split) != {"train", "validation", "test"}:
        raise ValueError("Unexpected split names")
    train_transform, eval_transform = screen_transforms(stage_b_config, int(variant["image_size"]))
    train_dataset = BaiPatternDataset(records, split["train"], args.data_root, train_transform)
    validation_dataset = BaiPatternDataset(records, split["validation"], args.data_root, eval_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
        generator=generator, drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0,
        drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, weight_path, initialization_path, _ = build_screen_model(root, config, args.variant, device)
    stage, head = parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": stage, "lr": config["last_stage_learning_rate"], "group_name": "last_stage"},
            {"params": head, "lr": config["head_learning_rate"], "group_name": "head"},
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    checkpoint_dir = root / "checkpoints" / f"cnn_feature_screen_{args.variant}_seed{seed}"
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    start_epoch, patience_counter, total_updates = 1, 0, 0
    history = []
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    code_commit = git.stdout.strip() if git.returncode == 0 else None
    frozen_hash = parameters_hash(frozen_named_parameters(model))
    if args.resume is not None:
        if args.resume.resolve() != last_checkpoint.resolve():
            raise ValueError(f"Resume must use {last_checkpoint}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["config_sha256"] != config_sha256 or checkpoint["variant"]["id"] != args.variant:
            raise ValueError("Resume checkpoint protocol differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_f1 = float(checkpoint["best_validation_macro_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        total_updates = int(checkpoint["total_optimizer_updates"])
        history = checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"], generator)
        if patience_counter >= config["early_stopping_patience"] or start_epoch > max_epochs:
            print(f"{args.variant} already completed; no training performed.")
            print("Best epoch:", best_epoch)
            print("Best validation Macro-F1:", f"{best_f1:.6f}")
            print("Test evaluated:", False)
            return
        print(f"Resumed {args.variant} after epoch {start_epoch - 1}; next epoch is {start_epoch}.")
    else:
        if checkpoint_dir.exists():
            raise FileExistsError(f"Checkpoint directory exists; resume with {last_checkpoint}")
        checkpoint_dir.mkdir(parents=True)
        metrics, _, _, indices, truth, predictions, probabilities = evaluate_validation(
            model, validation_loader, device, manifest["categories"]
        )
        best_f1 = float(metrics["macro_f1"])
        best_epoch = 0
        initial_predictions = root / "results" / f"cnn_feature_screen_{args.variant}_seed{seed}_validation_epoch000.csv"
        save_predictions(initial_predictions, manifest["categories"], indices, truth, predictions, probabilities)
        initial = checkpoint_payload(
            config, variant, 0, model, optimizer, scheduler, best_f1, 0, 0, 0, [], generator,
            config_sha256, file_hash(root / config["data"]["manifest"]), frozen_hash, code_commit,
        )
        torch.save(initial, best_checkpoint)
        torch.save(initial, last_checkpoint)
        print(f"{args.variant} epoch 0 | val accuracy {metrics['accuracy']:.4f} | val Macro-F1 {metrics['macro_f1']:.4f}")
    final_epoch = start_epoch - 1
    stop_reason = "max_epochs"
    for epoch in range(start_epoch, max_epochs + 1):
        stage_lr, head_lr = optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]
        print(
            f"{args.variant} epoch {epoch}/{max_epochs} | stage lr {stage_lr:.8f} | head lr {head_lr:.8f}",
            flush=True,
        )
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
        scheduler.step()
        history.append({
            "epoch": epoch,
            "stage_learning_rate": stage_lr,
            "head_learning_rate": head_lr,
            "train_loss": train_loss,
            "optimizer_updates": updates,
            "validation_metrics": metrics,
            "improved": improved,
            "patience_counter": patience_counter,
        })
        prediction_path = root / "results" / f"cnn_feature_screen_{args.variant}_seed{seed}_validation_epoch{epoch:03d}.csv"
        save_predictions(prediction_path, manifest["categories"], indices, truth, predictions, probabilities)
        payload = checkpoint_payload(
            config, variant, epoch, model, optimizer, scheduler, best_f1, best_epoch,
            patience_counter, total_updates, history, generator, config_sha256,
            file_hash(root / config["data"]["manifest"]), frozen_hash, code_commit,
        )
        torch.save(payload, last_checkpoint)
        if improved:
            torch.save(payload, best_checkpoint)
        now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
        report = {
            "timestamp": now.isoformat(),
            "status": "cnn_feature_screen_epoch_complete_validation_only",
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": matrix,
            "best_epoch_so_far": best_epoch,
            "best_validation_macro_f1_so_far": best_f1,
            "patience_counter": patience_counter,
            "epoch_optimizer_updates": updates,
            "total_optimizer_updates": total_updates,
            "frozen_parameters_unchanged": parameters_hash(frozen_named_parameters(model)) == frozen_hash,
            "config_sha256": config_sha256,
            "manifest_sha256": file_hash(root / config["data"]["manifest"]),
            "weights_sha256": file_hash(weight_path),
            "initialization_checkpoint_sha256": file_hash(initialization_path),
            "prediction_relative": prediction_path.relative_to(root).as_posix(),
            "prediction_sha256": file_hash(prediction_path),
            "test_dataset_instantiated": False,
            "test_evaluated": False,
        }
        report_path = root / "logs" / f"cnn_feature_screen_{args.variant}_seed{seed}_epoch{epoch:03d}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"{args.variant} epoch {epoch} complete | train loss {train_loss:.4f} | "
            f"val accuracy {metrics['accuracy']:.4f} | val Macro-F1 {metrics['macro_f1']:.4f} | "
            f"patience {patience_counter}", flush=True,
        )
        final_epoch = epoch
        if args.stop_after_epoch is not None and epoch >= args.stop_after_epoch:
            stop_reason = "requested_epoch_checkpoint"
            break
        if patience_counter >= config["early_stopping_patience"]:
            stop_reason = "early_stopping"
            break
    print("Variant:", args.variant)
    print("Stopped after epoch:", final_epoch)
    print("Stop reason:", stop_reason)
    print("Best epoch:", best_epoch)
    print("Best validation Macro-F1:", f"{best_f1:.6f}")
    print("Total optimizer updates:", total_updates)
    print("Test evaluated:", False)
    print("Resume command uses:", last_checkpoint)


if __name__ == "__main__":
    main()
