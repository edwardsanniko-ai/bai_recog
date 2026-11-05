"""Evaluate both frozen ImageNet-supervised controls with shared cosine 1NN."""

from pathlib import Path

from evaluate_dinov3_neighbors import evaluate_validation


def main():
    entry = Path(__file__)
    evaluate_validation(
        "control_resnet50.json",
        "torchvision ResNet50 ImageNet-1K V2 control",
        "control_resnet50",
        entry,
    )
    print()
    evaluate_validation(
        "control_convnext_tiny.json",
        "torchvision ConvNeXt-Tiny ImageNet-1K V1 control",
        "control_convnext_tiny",
        entry,
    )


if __name__ == "__main__":
    main()
