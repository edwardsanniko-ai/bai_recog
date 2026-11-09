"""Train or resume the audited from-scratch TinyCNN validation baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import numpy as np
import sklearn
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
from train_convnext_linear_probe import (
    evaluate_validation,
    restore_rng_state,
    rng_state,
    save_predictions,
)


CONFIG_NAME = "cnn_scratch_tiny.json"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ScratchTinyCNN(nn.Module):
    """Small CNN initialized from scratch; no pretrained checkpoint is used."""

    def __init__(self, num_classes: int, channels: tuple[int, ...] = (16, 32, 64, 128), dropout: float = 0.3):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("This locked protocol expects exactly four convolutional stages")
        blocks = []
        in_channels = 3
        for out_channels in channels:
            blocks.append(ConvBlock(in_channels, out_channels))
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))


def parameter_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    updates = 0
    before = parameter_hash(model)
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
    after = parameter_hash(model)
    if before == after:
        raise RuntimeError("Scratch CNN parameters did not change")
    if total_samples != len(loader.dataset):
        raise RuntimeError("Training epoch did not see each dataset position once")
    return total_loss / total_samples, updates, before, after


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=root.parent / "bai_data")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    args = parser.parse_args()

    config_path = root / "configs" / CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["status"] != "locked_before_training" or config["allow_test_during_training"]:
        raise ValueError("Scratch-CNN validation protocol is not locked or test access is enabled")
    seed = int(config["development_seed"] if args.seed is None else args.seed)
    allowed_seeds = {int(value) for value in config["confirmatory_seeds"]}
    if seed not in allowed_seeds:
        raise ValueError(f"Seed {seed} is outside the locked seed set: {sorted(allowed_seeds)}")
    max_epochs = int(config["max_epochs"])
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= max_epochs:
        raise ValueError("--stop-after-epoch must be within the configured epoch range")

    set_seed(seed)
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
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
        generator=generator, drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0,
        drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = tuple(int(value) for value in config["model"]["channels"])
    model = ScratchTinyCNN(config["num_classes"], channels, config["model"]["dropout"])
    model.to(device=device, dtype=torch.float32)
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Scratch CNN unexpectedly contains frozen parameters")
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    checkpoint_dir = root / "checkpoints" / f"cnn_scratch_tiny_seed{seed}"
    last_checkpoint = checkpoint_dir / "last.pt"
    best_checkpoint = checkpoint_dir / "best.pt"
    start_epoch, best_f1, best_epoch, patience_counter = 1, float("-inf"), 0, 0
    total_updates = 0
    history = []
    git = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    code_commit = git.stdout.strip() if git.returncode == 0 else None
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
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_f1 = float(checkpoint["best_validation_macro_f1"])
        best_epoch = int(checkpoint["best_epoch"])
        patience_counter = int(checkpoint["patience_counter"])
        total_updates = int(checkpoint["total_optimizer_updates"])
        history = checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"], generator)
        if patience_counter >= config["early_stopping_patience"] or start_epoch > max_epochs:
            print("Checkpoint already completed; no training performed.")
            print("Best epoch:", best_epoch)
            print("Best validation Macro-F1:", f"{best_f1:.6f}")
            print("Test evaluated:", False)
            return
        print(f"Resumed after epoch {start_epoch - 1}; next epoch is {start_epoch}.")
    elif checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint directory already exists; use --resume with {last_checkpoint}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_epoch, stop_reason = start_epoch - 1, "max_epochs"
    if args.resume is None:
        initial = {
            "format_version": 1, "experiment_name": config["experiment_name"], "seed": seed,
            "max_epochs": max_epochs, "completed_epoch": 0,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch, "patience_counter": patience_counter,
            "total_optimizer_updates": total_updates, "history": [], "rng_state": rng_state(generator),
            "config_sha256": config_sha256, "manifest_sha256": manifest_sha256,
            "code_commit_before_run": code_commit, "test_evaluated": False,
        }
        torch.save(initial, last_checkpoint)
        torch.save(initial, best_checkpoint)

    for epoch in range(start_epoch, max_epochs + 1):
        print(f"Epoch {epoch}/{max_epochs} | lr {optimizer.param_groups[0]['lr']:.8f}", flush=True)
        train_loss, epoch_updates, before_hash, after_hash = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        total_updates += epoch_updates
        metrics, per_class, matrix, indices, truth, predictions, probabilities = evaluate_validation(
            model, validation_loader, device, manifest["categories"]
        )
        improved = metrics["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, patience_counter = float(metrics["macro_f1"]), epoch, 0
        else:
            patience_counter += 1
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({
            "epoch": epoch, "learning_rate": current_lr, "train_loss": train_loss,
            "optimizer_updates": epoch_updates, "validation_metrics": metrics,
            "improved": improved, "patience_counter": patience_counter,
        })
        prediction_path = root / "results" / f"cnn_scratch_tiny_seed{seed}_validation_epoch{epoch:03d}.csv"
        save_predictions(prediction_path, manifest["categories"], indices, truth, predictions, probabilities)
        checkpoint = {
            "format_version": 1, "experiment_name": config["experiment_name"], "seed": seed,
            "max_epochs": max_epochs, "completed_epoch": epoch,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(), "best_validation_macro_f1": best_f1,
            "best_epoch": best_epoch, "patience_counter": patience_counter,
            "total_optimizer_updates": total_updates, "history": history, "rng_state": rng_state(generator),
            "config_sha256": config_sha256, "manifest_sha256": manifest_sha256,
            "code_commit_before_run": code_commit, "test_evaluated": False,
        }
        torch.save(checkpoint, last_checkpoint)
        if improved:
            torch.save(checkpoint, best_checkpoint)
        now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
        report = {
            "timestamp": now.isoformat(), "status": "scratch_cnn_epoch_complete_validation_only",
            "config_relative": f"configs/{CONFIG_NAME}", "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256, "script_sha256": file_hash(Path(__file__)),
            "code_commit_before_run": code_commit, "seed": seed, "epoch": epoch,
            "max_epochs": max_epochs, "train_images": len(train_dataset),
            "validation_images": len(validation_dataset), "train_batches": len(train_loader),
            "validation_batches": len(validation_loader), "train_loss": train_loss,
            "validation_metrics": metrics, "per_class": per_class, "confusion_matrix": matrix,
            "confusion_matrix_rows": "true", "confusion_matrix_columns": "predicted",
            "learning_rate": current_lr, "epoch_optimizer_updates": epoch_updates,
            "total_optimizer_updates": total_updates, "trainable_parameter_count": trainable_count,
            "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
            "parameters_changed_this_epoch": before_hash != after_hash,
            "best_epoch_so_far": best_epoch, "best_validation_macro_f1_so_far": best_f1,
            "patience_counter": patience_counter, "prediction_relative": prediction_path.relative_to(root).as_posix(),
            "prediction_sha256": file_hash(prediction_path),
            "last_checkpoint_relative": last_checkpoint.relative_to(root).as_posix(),
            "best_checkpoint_relative": best_checkpoint.relative_to(root).as_posix(),
            "resume_supported": True, "test_dataset_instantiated": False, "test_evaluated": False,
            "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__, "sklearn_version": sklearn.__version__,
        }
        report_path = root / "logs" / f"cnn_scratch_tiny_seed{seed}_epoch{epoch:03d}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"Epoch {epoch} complete | train loss {train_loss:.4f} | val accuracy {metrics['accuracy']:.4f} | "
            f"val Macro-F1 {metrics['macro_f1']:.4f}", flush=True,
        )
        print("  all parameters trainable:", True, "| parameters changed:", True, "| updates:", epoch_updates)
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
    print("Best validation Macro-F1:", f"{best_f1:.6f}")
    print("Trainable parameters:", trainable_count)
    print("Total optimizer updates:", total_updates)
    print("Test evaluated:", False)
    print("Resume command uses:", last_checkpoint)


if __name__ == "__main__":
    main()
