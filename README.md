# 白族图案识别：SigLIP 2 Base

This repository contains a new, separate experiment for classifying the six
Bai pattern categories with **SigLIP 2 Base (224)**. The earlier Qwen3-VL
reproduction remains outside this repository and is not used as a result here.

## Scope of the first baseline

- Task: six-class whole-image classification.
- Dataset: 620 images, fixed split 434 train / 93 validation / 93 test.
- Backbone: `google/siglip2-base-patch16-224` (vision encoder only).
- First training protocol: frozen SigLIP 2 features plus a linear classification
  head. Hyperparameters will be changed only after this baseline is recorded.
- Raw images and model weights are local-only and are excluded by
  `.gitignore`.

## Reproducibility records

- `data/splits.json`: relative image identifiers, labels and split indices;
  no absolute local paths.
- `configs/baseline_siglip2.json`: active model and pinned revision;
  training hyperparameters remain unset for step-by-step discussion.
- `data/audit.json`: verified counts and cross-split exact-file duplicate check.
- `logs/`: short, append-only experiment records.
- `scripts/`: scripts used to create metadata and run smoke tests.

## Local setup

当前阶段已完成建档、数据核验、权重下载和单图 GPU 前向，尚未训练。
已验证现有环境：PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、
transformers 4.57.3、Pillow 12.3.0、numpy 2.2.5；CUDA 可用，
显卡 NVIDIA GeForce RTX 4060 Laptop GPU。
无需现在升级已有环境。

GitHub 首次上传已完成。按用户要求改用无需申请权限的 SigLIP 2：
https://huggingface.co/google/siglip2-base-patch16-224
已匿名核实官方配置和权重可访问，不需要 Hugging Face 登录。
DINOv3 配置和脚本仅作历史留痕，已停止使用，且没有进行过训练。

以下命令从本仓库目录执行。单图测试首次运行会自动下载官方权重；
虽然只加载视觉编码器，官方权重文件仍包含图文两部分，下载可能需要等待。

```powershell
& '..\.conda-env\python.exe' scripts/make_manifest.py
& '..\.conda-env\python.exe' scripts/smoke_test_siglip2.py
```

The smoke test loads one local image, prints the feature shape, device, and
whether the output contains finite values. It does not train or upload images.

单图测试通过后，在同一目录运行下一步（全量冻结特征缓存）：

```powershell
& '..\.conda-env\python.exe' scripts/cache_siglip2_features.py
```

预期输出为 (620, 768) 特征和 (620,) 标签。缓存保存在本地 cache/，
核验记录保存在 logs/。全量特征提取不会训练模型或计算分类成绩。

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
