# Bai pattern recognition with DINOv3

This repository contains a new, separate experiment for classifying the six
Bai pattern categories with **DINOv3 ConvNeXt-Tiny**. The earlier Qwen3-VL
reproduction remains outside this repository and is not used as a result here.

## Scope of the first baseline

- Task: six-class whole-image classification.
- Dataset: 620 images, fixed split 434 train / 93 validation / 93 test.
- Backbone: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`.
- First training protocol: frozen DINOv3 features plus a linear classification
  head. Hyperparameters will be changed only after this baseline is recorded.
- Raw images and gated model weights are local-only and are excluded by
  `.gitignore`.

## Reproducibility records

- `data/splits.json`: relative image identifiers, labels and split indices;
  no absolute local paths.
- `configs/baseline_convnext_tiny.json`: planned first-run settings.
- `logs/`: short, append-only experiment records.
- `scripts/`: scripts used to create metadata and run smoke tests.

## Local setup

Run commands from this repository with the project's Python environment. The
DINOv3 Hugging Face weights require accepting the model's access terms and
logging in locally; the token is never committed.

```powershell
python -m pip install -U transformers accelerate pillow scikit-learn
hf auth login
python scripts/make_manifest.py
python scripts/smoke_test_dinov3.py
```

The smoke test loads one local image, prints the feature shape, device, and
whether the output contains finite values. It does not train or upload images.
