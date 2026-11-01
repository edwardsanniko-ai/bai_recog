"""Evaluate frozen SigLIP 2 using the exact same cosine 1NN code as DINOv3."""

from pathlib import Path

from evaluate_dinov3_neighbors import evaluate_validation


if __name__ == '__main__':
    evaluate_validation('baseline_siglip2.json', 'SigLIP 2 Base', 'siglip2', Path(__file__))
