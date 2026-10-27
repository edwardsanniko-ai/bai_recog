# 白族图案识别：DINOv3 ConvNeXt-Tiny

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
- `configs/baseline_convnext_tiny.json`: selected model and pinned revision;
  training hyperparameters remain unset for step-by-step discussion.
- `data/audit.json`: verified counts and cross-split exact-file duplicate check.
- `logs/`: short, append-only experiment records.
- `scripts/`: scripts used to create metadata and run smoke tests.

## Local setup

当前阶段只完成建档和数据核验，尚未下载权重或训练。
已验证现有环境：PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、
transformers 4.57.3、Pillow 12.3.0、numpy 2.2.5；CUDA 可用，
显卡 NVIDIA GeForce RTX 4060 Laptop GPU。DINOv3ConvNextModel 导入成功。
无需现在升级已有环境。

下一步先完成 GitHub 登录和首次上传。模型阶段需要在
https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
申请权重访问权限，并在本地登录。不要把访问令牌发到聊天或提交到仓库。

以下命令供获得模型权限后的步骤使用，从本仓库目录执行：

```powershell
& '..\.conda-env\python.exe' scripts/make_manifest.py
& '..\.conda-env\python.exe' scripts/smoke_test_dinov3.py
```

The smoke test loads one local image, prints the feature shape, device, and
whether the output contains finite values. It does not train or upload images.

## 实验记录约定

每次实验保存：配置、随机种子、模型版本、代码提交号、环境版本、
每轮训练与验证指标、最佳轮次和失败原因。模型权重与特征缓存留在本地，
后续记录其 SHA256 和备份位置；GitHub 只接收代码和小型结果文件。
当前仓库并未备份任何训练权重，也尚无识别成绩。

只根据训练集和验证集调参。旧测试集成绩此前已看过，后续报告应披露这一点，
不得将它称为从未使用过的全新盲测集。消融实验如实记录，不能预设结果一定下降。

此项目使用 bai_data 中的原图，沿用旧实验的样本划分；不会直接复用旧 Qwen
预处理图像或特征。因此将来与旧实验比较时需要说明预处理差异。

原图文件字节重复检查不能排除近似重复或同源纹样；后续数据审计再检查这些情况。
