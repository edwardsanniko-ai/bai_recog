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
- Raw images and model weights are local-only and are excluded by
  `.gitignore`.

## Reproducibility records

- `data/splits.json`: relative image identifiers, labels and split indices;
  no absolute local paths.
- `configs/baseline_convnext_tiny.json`: active model and pinned revision;
  training hyperparameters remain unset for step-by-step discussion.
- `data/audit.json`: verified counts and cross-split exact-file duplicate check.
- `logs/`: short, append-only experiment records.
- `scripts/`: scripts used to create metadata and run smoke tests.

## Local setup

当前阶段已完成建档、数据核验，以及 SigLIP 2 权重下载和单图 GPU 前向。
2026-10-30 用户确认获得 Hugging Face DINOv3 访问权限，要求优先回到 DINOv3。
DINOv3 权重已下载，单图 GPU 测试通过：输入 (1, 3, 224, 224)，
输出 (1, 768)、float32、全部有效。两个模型均尚未训练分类器。
已验证现有环境：PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、
transformers 4.57.3、Pillow 12.3.0、numpy 2.2.5；CUDA 可用，
显卡 NVIDIA GeForce RTX 4060 Laptop GPU。
无需现在升级已有环境。

GitHub 首次上传和本机 Hugging Face 登录已完成。新机器首次登录可使用：

```powershell
& '..\.conda-env\python.exe' scripts/login_huggingface.py --clipboard
```

脚本提示后复制令牌，回到终端按回车，不写入代码、日志或聊天。GitHub 登录和 Hugging Face
登录互不替代。模型页面：
https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m

SigLIP 2 的代码、权重及成功记录保留，暂停继续提取和训练。

后续确认本地登录成功后，再从本仓库目录运行 DINOv3 单图测试：

```powershell
& '..\.conda-env\python.exe' scripts/smoke_test_dinov3.py
```

The smoke test loads one local image, prints the feature shape, device, and
whether the output contains finite values. It does not train or upload images.

当前下一步是 DINOv3 全量特征缓存：

```powershell
& '..\.conda-env\python.exe' scripts/cache_dinov3_features.py
```

仅加载本地权重；预期特征 (620, 768)、标签 (620,)，并验证保存后读回一致。
cache/ 中的特征留在本地，logs/ 中的小型记录用于 GitHub 留痕。
保持官方 224×224 快速预处理、模型冻结和 eval 模式；不计算测试集指标。
不要使用 SigLIP 2 的缓存脚本处理 DINOv3：两种模型的特征不能混用。

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
