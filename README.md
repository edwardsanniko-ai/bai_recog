# 白族图案识别：冻结预训练模型对比

This repository contains a new, separate experiment for classifying the six
Bai pattern categories with **DINOv3 ConvNeXt-Tiny**, **SigLIP 2 Base**, and **Qwen3-VL-Embedding-2B** under a fixed cosine 1NN protocol. The earlier Qwen3-VL
reproduction remains outside this repository and is not used as a result here.

## 当前阶段：预训练模型横向筛选，不更新参数

2026-11-01 用户明确：先比较原生预训练模型，再选定 baseline。
本轮不更新视觉编码器，也不训练线性分类器；不使用任何教学更新后的权重。
DINOv3 没有白族六分类输出头，不能直接对六个类别输出概率。
用户已选择最近邻：每张图像特征独立做 L2 归一化，以余弦相似度最高的一张训练图片
作为参照（k=1），采用它的类别；完全相同的最高相似度按最小全局样本编号打破平局。
这类评估使用训练集标签作为参考，是冻结特征的有标签参考分类，不能称为零样本识别。

对可比较的图像编码器采用相同数据划分和判别规则；输入预处理遵循各自官方设置并记录。
以验证集比较/选择模型，测试集不参与筛选或调参。无须优化器、学习率或早停。
此前分类器/损失/单步更新都是教学演示，DINOv3 权重未改变，演示结果不计入对比表。

## Data and model

已完成 DINOv3 的固定协议验证评估：77/93 正确，Accuracy 82.80%，Macro-F1 83.45%，
模型参数更新 0 次，测试集未评估。详见 [模型对比记录](MODEL_COMPARISON.md)。
SigLIP 2 同协议评估也已完成：77/93 正确，Accuracy 82.80%，Macro-F1 82.94%。
两者均未更新模型参数或评估测试集，尚未选定最终 baseline。
用户已选择加入 Qwen3-VL-Embedding-2B，使用图像输入的官方整图嵌入输出，
不加入手工特征，不使用旧 Qwen 教学分类器或中间层融合缓存。该模型目前只通过单图检查，
尚未产生本协议的验证成绩。

- Task: six-class whole-image classification.
- Dataset: 620 images, fixed split 434 train / 93 validation / 93 test.
- Backbone: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`.
- Next candidate: `google/siglip2-base-patch16-224` (vision encoder only).
- Third candidate: `Qwen/Qwen3-VL-Embedding-2B` (image input, official whole-image embedding).
- Protocol: frozen pretrained features + cosine 1NN; no parameter updates.
- Raw images and model weights are local-only and are excluded by
  `.gitignore`.

## Reproducibility records

- `data/splits.json`: relative image identifiers, labels and split indices;
  no absolute local paths.
- `configs/baseline_convnext_tiny.json`: active model, pinned revision and no-update constraint;
  training parameters are inapplicable during screening.
- `data/audit.json`: verified counts and cross-split exact-file duplicate check.
- `logs/`: short, append-only experiment records.
- `scripts/`: scripts used to create metadata and run smoke tests.

## Local setup

当前下一小步：Qwen 2B 单图检查。权重沿用工作区原有的固定本地快照，无需重新下载。

```powershell
& '..\.conda-env\python.exe' scripts/smoke_test_qwen2b.py
```

预期 Feature shape 为 (1, 2048)，All finite / All parameters frozen 均为 True，
Optimizer updates 为 0。采用 bf16 + SDPA、batch=1，要求支持 bf16 的 CUDA 显卡。
官方 helper 已随仓库保留并附带许可，见 [第三方来源](THIRD_PARTY.md)。
本地模型位置在 configs/baseline_qwen2b.json 配置，仓库不分发权重。

Qwen 使用默认通用系统指令 `Represent the user's input.` 和图片，没有类别名提示；
完整冻结的嵌入模型（含语言主干）输出最后一个有效 token 的 2048 维向量，
转 float32 后做 L2 归一化。它不是生成式六分类回答，也不是只截取中间层视觉 token。
采用官方 helper 默认动态缩放 min_pixels=4096、max_pixels=1843200，
与前两款的 224×224 及 float32 推理差异会记录；不为追逐成绩调节这些设置。
后续仍使用相同 434/93/93 划分与余弦 1NN 规则，测试集不参与筛选。

当前阶段已完成建档、数据核验，以及 SigLIP 2 权重下载和单图 GPU 前向。
2026-10-30 用户确认获得 Hugging Face DINOv3 访问权限，要求优先回到 DINOv3。
DINOv3 权重已下载，单图 GPU 测试通过：输入 (1, 3, 224, 224)，
输出 (1, 768)、float32、全部有效。正式分类器训练未开展；仅曾执行一次单样本教学更新。
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

SigLIP 2 已下载的权重和成功记录保留，现恢复全量特征提取，用于无参数更新的横向对比。

SigLIP 2 全量缓存已于 2026-11-01 完成，以下保留提取命令：

```powershell
& '..\.conda-env\python.exe' scripts/cache_siglip2_features.py
```

无需重新下载或登录。预期 (620,768) 特征与 (620,) 标签，
All finite / Readback identical 均为 True，Optimizer updates 为 0。
官方 SigLIP 2 预处理 use_fast=false，与 DINOv3 的快速预处理器各自保持官方设置；
两者均为 224×224。SigLIP 2 的均值/标准差为 0.5，不套用 DINOv3 的归一化参数。
这一步只准备缓存，完成后再做 93 张验证图的 1NN 评估，不产生测试集成绩。

SigLIP 2 已完成的验证集最近邻评估命令（复核用，不更新参数）：

```powershell
& '..\.conda-env\python.exe' scripts/evaluate_siglip2_neighbors.py
```

两个入口调用同一评估函数，固定 k=1 和余弦相似度；SigLIP 2 运行前还会核对
DINOv3 已保存报告的类别顺序、参考/验证编号、数据清单哈希和判别协议。
只使用各模型自己的特征和官方预处理，不混用模型特征。结果分别保存，原结果不覆盖。

后续确认本地登录成功后，再从本仓库目录运行 DINOv3 单图测试：

```powershell
& '..\.conda-env\python.exe' scripts/smoke_test_dinov3.py
```

The smoke test loads one local image, prints the feature shape, device, and
whether the output contains finite values. It does not train or upload images.

DINOv3 全量特征缓存已完成（2026-11-01），如需重新提取可运行：

```powershell
& '..\.conda-env\python.exe' scripts/cache_dinov3_features.py
```

仅加载本地权重；预期特征 (620, 768)、标签 (620,)，并验证保存后读回一致。
cache/ 中的特征留在本地，logs/ 中的小型记录用于 GitHub 留痕。
保持官方 224×224 快速预处理、模型冻结和 eval 模式；不计算测试集指标。
不要使用 SigLIP 2 的缓存脚本处理 DINOv3：两种模型的特征不能混用。

当前选定缓存由 configs/baseline_convnext_tiny.json 的 feature_cache_report 指定，
不会自动选择其他模型或最新文件。DINOv3 已完成的最近邻验证命令（供复核）：

```powershell
& '..\.conda-env\python.exe' scripts/evaluate_dinov3_neighbors.py
```

仅使用 434 张训练图片作参考，预测 93 张验证图片；不评估测试集。
输出 Accuracy、Macro Precision/Recall/F1、Weighted F1 及逐类结果，
并记录逐图预测、最近邻编号和相似度。相似度不是类别概率。
CSV 保存至 results/，报告保存至 logs/，都可提交 GitHub；原图和特征不上传。

## 历史教学步骤（不属于当前评估流程）

以下保留学习记录；更新参数的 demo_dinov3_update.py 已由当前配置禁用。
曾使用下列命令检查未训练线性分类器：

```powershell
& '..\.conda-env\python.exe' scripts/inspect_dinov3_classifier.py
```

脚本先检查缓存哈希、标签和原始划分，再创建 Linear(768, 6)。
默认 seed=42 只用于此前初始化演示；当前不设置训练超参数。
使用一张训练样本检查输出 (1, 6)，不更新参数、不评估验证集或测试集。
演示在 CPU 上运行，无需重新加载视觉模型，自动生成小型日志。

分类器结构检查后，下一小步为单张训练图片的交叉熵演示：

```powershell
& '..\.conda-env\python.exe' scripts/inspect_dinov3_loss.py
```

仍以 seed=42 从头初始化同一线性分类器，显示各类别概率及 -log(真实类别概率)，
与 PyTorch CrossEntropyLoss 核对一致。交叉熵接收原始类别分数，
不要先对其输入做 softmax；softmax 仅用于展示概率。本步骤不更新参数。

交叉熵检查后，单张训练图片的一次参数更新演示：

```powershell
& '..\.conda-env\python.exe' scripts/demo_dinov3_update.py --lr 0.001
```

--lr 为用户指定的学习率，每次启动均从相同 seed=42 初始化，不接续前次演示。
AdamW 的 weight_decay=0 仅用于此前教学；当前禁止运行参数更新步骤。
backward 计算梯度，step 更新参数一次。比较同一训练样本的前后损失，
不能据此推断泛化性能，也不强制要求损失下降。
初始/更新后权重及优化器状态保存在本地 checkpoints/teaching/，
哈希和结果保存至 logs/；教学权重不上传、不用于正式基线初始化。

单步更新之后，检查训练数据如何分批（不创建模型、不更新参数）：

```powershell
& '..\.conda-env\python.exe' scripts/inspect_dinov3_batches.py --batch-size 16
```

将 434 个训练样本打乱后分成 28 批，最后一批保留剩余的 2 个样本。
脚本核对每个训练样本恰好出现一次，且没有验证或测试样本混入。
seed=42 控制此次打乱顺序；日志保存实际顺序和每批数量。
这是此前的数据加载检查；当前不再继续到整轮训练。

## 实验记录约定

每次实验保存：配置、随机种子、模型版本、代码提交号、环境版本、
每轮训练与验证指标、最佳轮次和失败原因。模型权重与特征缓存留在本地，
后续记录其 SHA256 和备份位置；GitHub 只接收代码和小型结果文件。
单步教学权重已保存在本地 checkpoints/teaching/；尚无正式基线训练权重或识别成绩。

只根据训练集和验证集调参。旧测试集成绩此前已看过，后续报告应披露这一点，
不得将它称为从未使用过的全新盲测集。消融实验如实记录，不能预设结果一定下降。

此项目使用 bai_data 中的原图，沿用旧实验的样本划分；不会直接复用旧 Qwen
预处理图像或特征。因此将来与旧实验比较时需要说明预处理差异。

原图文件字节重复检查不能排除近似重复或同源纹样；后续数据审计再检查这些情况。
