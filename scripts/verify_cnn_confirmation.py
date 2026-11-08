"""Independently verify locked best validation checkpoints for seeds 43-46."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cnn_baseline_common import BaiPatternDataset, build_transforms, load_and_verify_manifest
from cnn_last_stage_common import load_last_stage_config, load_last_stage_model
from train_convnext_linear_probe import evaluate_validation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_last_stage_config(root)
    manifest = load_and_verify_manifest(root, root.parent / "bai_data", config, verify_hashes=True)
    _, eval_transform = build_transforms(config)
    validation = BaiPatternDataset(
        manifest["records"], manifest["split_indices"]["validation"], root.parent / "bai_data", eval_transform
    )
    loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verified = []
    for seed in (43, 44, 45, 46):
        model, _, _, _ = load_last_stage_model(root, config, device, source_seed=seed)
        checkpoint_path = root / "checkpoints" / f"cnn_convnext_last_stage_seed{seed}" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics, _, _, _, _, _, _ = evaluate_validation(
            model, loader, device, manifest["categories"]
        )
        epoch = int(checkpoint["completed_epoch"])
        report_path = root / "logs" / f"cnn_convnext_last_stage_seed{seed}_epoch{epoch:03d}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if metrics != report["validation_metrics"]:
            raise RuntimeError(f"Seed {seed} checkpoint metrics differ from its best-epoch report")
        if checkpoint["best_epoch"] != epoch or checkpoint["seed"] != seed:
            raise RuntimeError(f"Seed {seed} best checkpoint identity differs")
        verified.append({"seed": seed, "best_epoch": epoch, **metrics})
        print(
            f"seed {seed}: epoch {epoch}, accuracy {metrics['accuracy']:.6f}, "
            f"Macro-F1 {metrics['macro_f1']:.6f}, matches report: True"
        )
        del model
    values = np.array([row["macro_f1"] for row in verified], dtype=np.float64)
    print("Validation images:", len(validation))
    print("Macro-F1 mean:", float(values.mean()))
    print("Macro-F1 sample std:", float(values.std(ddof=1)))
    print("Test evaluated:", False)


if __name__ == "__main__":
    main()
