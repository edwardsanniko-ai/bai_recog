"""Evaluate frozen Qwen 2B with the same validation-only cosine 1NN rule."""

from pathlib import Path

from evaluate_dinov3_neighbors import evaluate_validation


if __name__ == '__main__':
    evaluate_validation('baseline_qwen2b.json', 'Qwen3-VL-Embedding-2B', 'qwen2b', Path(__file__))
