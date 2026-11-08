"""Run the remaining locked ConvNeXt confirmation training end to end.

This orchestrator is restart-safe at epoch boundaries. It never evaluates the
test split. It finishes phase A, creates/uses exactly one phase-B preflight,
then finishes phase B for seeds 44, 45, and 46.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from cnn_baseline_common import file_hash
import torch


LOCKED_SEEDS = (44, 45, 46)


def run_command(root: Path, arguments: list[str], dry_run: bool) -> None:
    display = subprocess.list2cmdline(arguments)
    print("\n>", display, flush=True)
    if dry_run:
        return
    completed = subprocess.run(arguments, cwd=root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {display}")


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def is_complete(checkpoint: dict, patience: int, max_epochs: int) -> bool:
    return (
        int(checkpoint["patience_counter"]) >= patience
        or int(checkpoint["completed_epoch"]) >= max_epochs
    )


def finish_phase_a(root: Path, seed: int, python: str, protocol: dict, dry_run: bool) -> None:
    checkpoint_dir = root / "checkpoints" / f"cnn_convnext_linear_probe_seed{seed}"
    last_path = checkpoint_dir / "last.pt"
    arguments = [python, str(root / "scripts" / "train_convnext_linear_probe.py"), "--seed", str(seed)]
    if last_path.is_file():
        checkpoint = load_checkpoint(last_path)
        if checkpoint["seed"] != seed or checkpoint.get("run_role") != "confirmatory":
            raise ValueError(f"Phase-A seed identity differs for seed {seed}")
        if is_complete(checkpoint, protocol["phase_a_patience"], protocol["max_epochs"]):
            print(
                f"Seed {seed} phase A already complete: epoch {checkpoint['completed_epoch']}, "
                f"best epoch {checkpoint['best_epoch']}, F1 {checkpoint['best_validation_macro_f1']:.6f}",
                flush=True,
            )
            return
        arguments += ["--resume", str(last_path)]
    run_command(root, arguments, dry_run)
    if dry_run:
        return
    if not last_path.is_file():
        raise FileNotFoundError(f"Phase-A last checkpoint was not created for seed {seed}")
    checkpoint = load_checkpoint(last_path)
    if not is_complete(checkpoint, protocol["phase_a_patience"], protocol["max_epochs"]):
        raise RuntimeError(f"Phase A returned before its stopping rule for seed {seed}")


def ensure_phase_b_preflight(root: Path, seed: int, python: str, dry_run: bool) -> None:
    pattern = f"cnn_convnext_last_stage_seed{seed}_preflight_*.json"
    paths = sorted((root / "logs").glob(pattern))
    if len(paths) > 1:
        raise RuntimeError(f"Multiple phase-B preflight reports exist for seed {seed}: {paths}")
    if len(paths) == 1:
        report = json.loads(paths[0].read_text(encoding="utf-8"))
        if (
            report["status"] != "phase_b_last_stage_preflight_passed_no_updates"
            or int(report["source_seed"]) != seed
            or report["run_role"] != "confirmatory"
            or report["optimizer_updates"] != 0
            or report["test_evaluated"]
        ):
            raise ValueError(f"Existing phase-B preflight is invalid for seed {seed}")
        print(f"Seed {seed} phase-B preflight already passed: {paths[0].name}", flush=True)
        return
    arguments = [python, str(root / "scripts" / "inspect_convnext_last_stage.py"), "--seed", str(seed)]
    run_command(root, arguments, dry_run)
    if dry_run:
        return
    paths = sorted((root / "logs").glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one phase-B preflight after running seed {seed}")


def finish_phase_b(root: Path, seed: int, python: str, protocol: dict, dry_run: bool) -> None:
    checkpoint_dir = root / "checkpoints" / f"cnn_convnext_last_stage_seed{seed}"
    last_path = checkpoint_dir / "last.pt"
    arguments = [python, str(root / "scripts" / "train_convnext_last_stage.py"), "--seed", str(seed)]
    if last_path.is_file():
        checkpoint = load_checkpoint(last_path)
        if checkpoint["seed"] != seed or checkpoint.get("run_role") != "confirmatory":
            raise ValueError(f"Phase-B seed identity differs for seed {seed}")
        if is_complete(checkpoint, protocol["phase_b_patience"], protocol["max_epochs"]):
            print(
                f"Seed {seed} phase B already complete: epoch {checkpoint['completed_epoch']}, "
                f"best epoch {checkpoint['best_epoch']}, F1 {checkpoint['best_validation_macro_f1']:.6f}",
                flush=True,
            )
            return
        arguments += ["--resume", str(last_path)]
    run_command(root, arguments, dry_run)
    if dry_run:
        return
    if not last_path.is_file():
        raise FileNotFoundError(f"Phase-B last checkpoint was not created for seed {seed}")
    checkpoint = load_checkpoint(last_path)
    if not is_complete(checkpoint, protocol["phase_b_patience"], protocol["max_epochs"]):
        raise RuntimeError(f"Phase B returned before its stopping rule for seed {seed}")


def collect_summary(root: Path, seeds: tuple[int, ...]) -> dict:
    rows = []
    for seed in seeds:
        phase_a = load_checkpoint(root / "checkpoints" / f"cnn_convnext_linear_probe_seed{seed}" / "best.pt")
        phase_b = load_checkpoint(root / "checkpoints" / f"cnn_convnext_last_stage_seed{seed}" / "best.pt")
        rows.append({
            "seed": seed,
            "phase_a_best_epoch": int(phase_a["best_epoch"]),
            "phase_a_validation_macro_f1": float(phase_a["best_validation_macro_f1"]),
            "phase_b_best_epoch": int(phase_b["best_epoch"]),
            "phase_b_validation_macro_f1": float(phase_b["best_validation_macro_f1"]),
            "phase_b_improvement": float(
                phase_b["best_validation_macro_f1"] - phase_a["best_validation_macro_f1"]
            ),
        })
    values = torch.tensor([row["phase_b_validation_macro_f1"] for row in rows], dtype=torch.float64)
    return {
        "status": "locked_confirmation_training_complete_validation_only",
        "seeds": list(seeds),
        "results": rows,
        "phase_b_validation_macro_f1_mean": float(values.mean()),
        "phase_b_validation_macro_f1_sample_std": float(values.std(unbiased=True)),
        "test_evaluated": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    confirmation_path = root / "configs" / "cnn_convnext_multiseed_confirmation.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if tuple(map(int, confirmation["confirmatory_seeds"])) != (43, 44, 45, 46):
        raise ValueError("Locked confirmation seed set differs")
    phase_a_config = json.loads((root / "configs" / "cnn_convnext_tiny_linear_probe.json").read_text(encoding="utf-8"))
    phase_b_config = json.loads((root / "configs" / "cnn_convnext_tiny_last_stage.json").read_text(encoding="utf-8"))
    if file_hash(root / confirmation["phase_a"]["config"]) != confirmation["phase_a"]["config_sha256"]:
        raise ValueError("Locked phase-A config hash differs")
    if file_hash(root / confirmation["phase_b"]["config"]) != confirmation["phase_b"]["config_sha256"]:
        raise ValueError("Locked phase-B config hash differs")
    protocol = {
        "max_epochs": int(phase_a_config["max_epochs"]),
        "phase_a_patience": int(phase_a_config["early_stopping_patience"]),
        "phase_b_patience": int(phase_b_config["early_stopping_patience"]),
    }
    if protocol["max_epochs"] != int(phase_b_config["max_epochs"]):
        raise ValueError("Phase A and B max epochs differ")
    print("Locked remaining confirmation seeds:", list(LOCKED_SEEDS))
    print("Test evaluation: DISABLED")
    print("Restart-safe at completed epoch boundaries:", True)
    for seed in LOCKED_SEEDS:
        print(f"\n========== seed {seed}: phase A ==========" , flush=True)
        finish_phase_a(root, seed, sys.executable, protocol, args.dry_run)
        print(f"\n========== seed {seed}: phase-B preflight ==========" , flush=True)
        ensure_phase_b_preflight(root, seed, sys.executable, args.dry_run)
        print(f"\n========== seed {seed}: phase B ==========" , flush=True)
        finish_phase_b(root, seed, sys.executable, protocol, args.dry_run)
    if args.dry_run:
        print("\nDry run complete; no training command was executed.")
        return
    summary = collect_summary(root, (43, 44, 45, 46))
    now = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    destination = root / "logs" / f"cnn_convnext_confirmation_training_summary_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n========== VALIDATION-ONLY CONFIRMATION SUMMARY ==========")
    for row in summary["results"]:
        print(
            f"seed {row['seed']}: phase A F1={row['phase_a_validation_macro_f1']:.6f}, "
            f"phase B F1={row['phase_b_validation_macro_f1']:.6f}, "
            f"improvement={row['phase_b_improvement']:.6f}"
        )
    print(
        "Phase-B validation Macro-F1 mean +/- sample std:",
        f"{summary['phase_b_validation_macro_f1_mean']:.6f} +/- "
        f"{summary['phase_b_validation_macro_f1_sample_std']:.6f}",
    )
    print("Test evaluated:", False)
    print("Saved summary:", destination)


if __name__ == "__main__":
    main()
