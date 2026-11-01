"""Evaluate frozen EUPE ConvNeXt-Tiny with the shared validation-only cosine 1NN rule."""

from pathlib import Path

from evaluate_dinov3_neighbors import evaluate_validation


if __name__ == '__main__':
    evaluate_validation('baseline_eupe_convnext_tiny.json', 'EUPE ConvNeXt-Tiny', 'eupe', Path(__file__))
