# 白族图案识别：冻结预训练模型对比

## 新阶段：可训练 CNN baseline

2026-11-07 开始建立可训练的 ConvNeXt-Tiny CNN baseline。第一阶段只训练
`LayerNorm + Dropout(0.2) + Linear(768,6)` 分类头，ImageNet-1K V1 预训练 backbone
保持冻结；固定 434/93/93 划分，验证 Macro-F1 用于选择 checkpoint，测试集在协议锁定前
不评估。输入采用等比例缩放与白色补边，训练增强保持轻量并禁用水平翻转和随机裁切。
协议见 `configs/cnn_convnext_tiny_linear_probe.json`，训练前检查命令为：

```powershell
& '..\.conda-env\python.exe' scripts/inspect_convnext_linear_probe.py
```

该检查不会创建优化器或更新参数；正式训练脚本将在检查通过后加入。此前冻结特征 + 余弦
1NN 仍作为无训练对照，不与可训练分类器混称为同一实验。

训练循环支持保存完整随机状态并从上一轮继续。第一次只运行第 1 轮：

```powershell
& '..\.conda-env\python.exe' scripts/train_convnext_linear_probe.py --stop-after-epoch 1
```

确认第 1 轮记录后，从第 2 轮继续到早停或最多 50 轮：

```powershell
& '..\.conda-env\python.exe' scripts/train_convnext_linear_probe.py `
  --resume checkpoints/cnn_convnext_linear_probe_seed42/last.pt
```

checkpoint、原图和权重只保留在本地；逐轮小型 JSON 日志和验证集逐图预测可上传 GitHub。

seed 42 第 1 轮检查已完成：训练损失 1.7063，验证 Accuracy 51.61%，验证 Macro-F1
50.37%，分类头已更新、冻结 backbone 未变化、测试集未评估。该数值只是可恢复训练的中间结果，
不是最终 baseline 成绩；详见 `logs/0035_cnn_convnext_linear_probe_seed42_epoch1.md`。

2026-11-07 已修复 CUDA map location 将 CPU/DataLoader RNG 状态错误搬到 GPU 导致的恢复异常；
原 epoch 1 checkpoint 无需重训。修复和无训练验证见 `logs/0036_cnn_resume_rng_device_fix.md`。

阶段 A 的 seed 42 开发运行现已完成：按 patience=10，正式停止于第 17 轮，最佳为第 7 轮，
验证 Accuracy 79.57%、Macro-F1 79.42%，测试集未评估。一次在早停后误恢复产生的第 18 轮
已保留为审计证据但排除出正式训练；恢复入口现会拒绝已早停 checkpoint。完整说明见
`logs/0037_cnn_convnext_linear_probe_seed42_complete.md`。

阶段 B 将从阶段 A 的 seed 42 最佳第 7 轮分类头继续，只解冻 ConvNeXt 的最后一个
`features.7` stage；最后 stage 学习率为 `1e-5`，分类头学习率保持 `3e-4`。训练前先运行：

```powershell
& '..\.conda-env\python.exe' scripts/inspect_convnext_last_stage.py
```

该检查不创建优化器、不反向传播、不更新参数，也不实例化测试集。协议见
`configs/cnn_convnext_tiny_last_stage.json` 和 `logs/0038_cnn_convnext_last_stage_protocol.md`。

参数范围检查已经通过。阶段 B 第一次只运行一轮：

```powershell
& '..\.conda-env\python.exe' scripts/train_convnext_last_stage.py --stop-after-epoch 1
```

阶段 A 最佳 Macro-F1 作为 epoch 0 基准；只有阶段 B 超过该值才更新最佳 checkpoint。
第 1 轮确认后，可通过本地 `checkpoints/cnn_convnext_last_stage_seed42/last.pt` 恢复继续。

阶段 B seed 42 第 1 轮已完成：训练损失 0.7631，验证 Accuracy 77.42%、Macro-F1
77.08%，未超过阶段 A 起点 79.42%，所以最佳轮次仍为 epoch 0；两个可训练参数组均变化，
冻结参数未变化，测试集未评估。详见 `logs/0040_cnn_convnext_last_stage_seed42_epoch1.md`。

阶段 B seed 42 已于第 24 轮正常早停，最佳为第 14 轮：验证 Accuracy 87.10%、Macro-F1
87.32%。独立加载 `best.pt` 复算结果完全一致；第 14 轮后训练损失继续下降但验证成绩未提高，
因此不再恢复该 checkpoint。测试集仍未评估，详见
`logs/0041_cnn_convnext_last_stage_seed42_complete.md`。

两阶段协议现已锁定，进入多种子稳定性确认。seed 42 是用于选择方案的开发运行；seed 43–46
分别从官方权重和各自随机六分类头开始完整训练，主要汇总这四个确认种子的验证均值与样本
标准差。它们使用同一数据划分，只衡量训练随机性，不是四份独立数据。协议见
`configs/cnn_convnext_multiseed_confirmation.json` 和
`logs/0042_cnn_multiseed_confirmation_protocol.md`。

确认运行 seed 43 的阶段 A 第 1 轮已通过：训练损失 1.6981，验证 Accuracy 47.31%、
Macro-F1 41.44%；分类头更新、backbone 保持冻结、测试集未评估，且 checkpoint 与 seed 42
完全隔离。详见 `logs/0043_cnn_seed43_phase_a_epoch1.md`。

seed 43 阶段 A 已于第 18 轮早停，最佳第 8 轮验证 Macro-F1 78.25%；独立复算一致。
阶段 B 无更新检查也已通过，确认加载 seed 43 自己的阶段 A 最佳 checkpoint，测试集仍未评估。
详见 `logs/0044_cnn_seed43_phase_a_complete_and_phase_b_preflight.md`。

seed 43 阶段 B 第 1 轮已通过：验证 Macro-F1 78.13%，略低于本 seed 的阶段 A 起点
78.25%，所以最佳仍为 epoch 0；最后 stage 与分类头均更新，冻结参数未变化，测试集未评估。
详见 `logs/0045_cnn_seed43_phase_b_epoch1.md`。

seed 43 阶段 B 已于第 30 轮早停，最佳第 20 轮：验证 Accuracy 88.17%、Macro-F1
88.38%。独立加载最佳 checkpoint 复算完全一致，测试集未评估。详见
`logs/0046_cnn_seed43_phase_b_complete.md`。

确认运行 seed 44 的阶段 A 第 1 轮已通过：训练损失 1.6721，验证 Accuracy 49.46%、
Macro-F1 46.47%；分类头更新、backbone 保持冻结、测试集未评估，checkpoint 与其他种子
隔离。详见 `logs/0047_cnn_seed44_phase_a_epoch1.md`。

seed 44 阶段 A 已于第 15 轮早停，最佳第 5 轮验证 Macro-F1 76.81%；独立复算一致。
阶段 B 无更新检查确认加载 seed 44 自己的最佳 checkpoint，测试集仍未评估。详见
`logs/0048_cnn_seed44_phase_a_complete_and_phase_b_preflight.md`。

seed 44 阶段 B 第 1 轮验证 Macro-F1 为 78.64%，已超过本 seed 的阶段 A 起点 76.81%；
两个可训练参数组均变化，冻结参数未变化，测试集未评估。当前仍是同一个 ConvNeXt-Tiny
baseline 的多种子确认，详见 `logs/0049_cnn_seed44_phase_b_epoch1.md`。

This repository contains a new, separate experiment for classifying the six
Bai pattern categories with **DINOv3 ConvNeXt-Tiny**, **SigLIP 2 Base**, **Qwen3-VL-Embedding-2B**, and **EUPE ConvNeXt-Tiny**, plus supervised ImageNet controls, under a fixed cosine 1NN protocol. The earlier Qwen3-VL
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
不加入手工特征，不使用旧 Qwen 教学分类器或中间层融合缓存。该模型已完成同协议验证：
73/93 正确，Accuracy 78.49%，Macro-F1 78.77%。
EUPE ConvNeXt-Tiny 同协议验证已完成：77/93 正确，Accuracy 82.80%，Macro-F1 82.75%。
ResNet50 与 ConvNeXt-Tiny ImageNet 控制组也均为 77/93，Macro-F1 分别为 83.26% 和 83.18%。
六款模型均未更新参数；尚未选定最终 baseline。DINOv3 暂列第一，但只比最佳控制组高 0.19 个百分点。

- Task: six-class whole-image classification.
- Dataset: 620 images, fixed split 434 train / 93 validation / 93 test.
- Backbone: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`.
- Next candidate: `google/siglip2-base-patch16-224` (vision encoder only).
- Third candidate: `Qwen/Qwen3-VL-Embedding-2B` (image input, official whole-image embedding).
- Fourth candidate: `facebook/EUPE-ConvNeXt-T` (official whole-image backbone features).
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

## ImageNet 监督预训练控制组

新增两个经典控制组：torchvision ResNet50（ImageNet-1K V2）与
torchvision ConvNeXt-Tiny（ImageNet-1K V1）。它们不是新的主候选，而是用来回答：
相比普通 ImageNet 监督预训练，同规模的新型通用表征模型是否真的更适合白族纹样。

两者均先严格加载官方 1000 类权重，再移除最后的 ImageNet 分类线性层，
分别取 2048 维和 768 维整图向量。原来的 ImageNet 类别分数不参与白族六分类；
所有参数冻结，后续仍使用相同的 434 张训练图、93 张验证图和余弦 1NN。
第一小步只运行一张真实训练图片：

```powershell
& '..\.conda-env\python.exe' scripts/smoke_test_torchvision_controls.py
```

脚本一次检查两个控制组。预期输入均为 `(1,3,224,224)`；ResNet50 特征为
`(1,2048)`，ConvNeXt-Tiny 特征为 `(1,768)`，并显示 `All parameters frozen: True`、
`Optimizer updates: 0`、`Test evaluated: False`。权重已在本地缓存并校验 SHA256，
不会重新下载；本步不预测类别、不计算验证或测试指标。

官方预处理并不完全相同：ResNet50 将短边缩放到 232 后中心裁剪 224，
ConvNeXt-Tiny 将短边缩放到 236 后中心裁剪 224，均采用双线性插值、
ImageNet 均值/标准差和 float32。这个差异属于各自官方权重流程，会保留在报告中。

当前 Conda 环境直接调用 `python.exe` 时可能遗漏 `Library\\bin` 的 DLL 搜索路径；
脚本会在导入 Pillow/torchvision 前注册该目录，因此无需重装 Pillow 或 torchvision。
原图和官方权重仍由 `.gitignore` 排除，不上传 GitHub。

单图检查通过后，下一步一次提取两个控制组的全量特征：

```powershell
& '..\.conda-env\python.exe' scripts/cache_torchvision_control_features.py
```

脚本先完成 ResNet50 的 620 张图片，再完成 ConvNeXt-Tiny。为了使每张图片的特征不受
同一批中其他图片影响，提取批大小固定为 1；开发检查发现当前 CUDA/cuDNN 环境中，
ResNet50 单张与 batch=16 的原始特征最大差约 0.004。固定逐张推理后，每次都使用相同
计算路径，代价只是稍慢。预期两组输出分别为 `(620,2048)` 和 `(620,768)`，并各自保存
本地缓存和小型报告。此步骤不计算分类指标，不评估测试集，不更新参数。

两组全量缓存及验证集评估现已完成。以下命令保留供复核：

```powershell
& '..\.conda-env\python.exe' scripts/evaluate_torchvision_controls.py
```

它直接复用前四款模型的共享评估函数：434 张训练图片作带标签参考，93 张验证图片作查询，
逐图 L2 归一化后用 float64 计算余弦 1NN，精确平局取最小全局参考编号。
脚本先输出 ResNet50 指标，再输出 ConvNeXt-Tiny 指标；两者都只读取本地缓存，
不加载 GPU 模型、不训练参数、不评估测试集。验证 Macro-F1 分别为 83.26% 和 83.18%，
与 DINOv3 的 83.45% 非常接近。完整六模型对比及逐类结果见 MODEL_COMPARISON.md。

当前新增候选：用户选择 EUPE ConvNeXt-Tiny（`facebook/EUPE-ConvNeXt-T`）。
官方权重公开，无需申请权限；代码和权重分别固定版本，使用 FAIR 非商业研究许可。
单张训练图片检查已通过，以下命令保留供复核：

```powershell
& '..\.conda-env\python.exe' scripts/smoke_test_eupe.py
```

预期输入 `(1,3,256,256)`、输出 `(1,768)`，特征全部有效，全部参数冻结，更新次数为 0。
使用官方 256×256 双线性缩放与 ImageNet 归一化，取官方 `x_norm_clstoken` 整图向量；
这和前两款模型的 224×224 输入不同，比较时保留说明。推理采用 float32，关闭 TF32。
已有 PyTorch/torchvision 环境即可运行，无需安装官方分布式训练环境。
权重 SHA256 校验后严格加载全部 backbone 参数；官方 checkpoint 中额外的 8 个蒸馏
projector 张量不参与 backbone 特征提取，与官方 hub 忽略这些额外键的行为一致，日志列出名称。
脚本只检查训练集首张图片并保存日志，不生成六类预测或验证/测试成绩，不更新任何参数。
权重仅保存在本地 `hf_cache/`，不上传 GitHub；第三方代码来源见 `THIRD_PARTY.md`。

EUPE 全量冻结特征已提取完成，以下命令保留供重新提取时使用。

```powershell
& '..\.conda-env\python.exe' scripts/cache_eupe_features.py
```

默认每批 16 张，沿用成功单图检查的官方模型、256×256 预处理及 float32 推理。
只加载本地权重，不下载、不训练。逐张检查原图哈希，保持 620 张原始顺序和 434/93/93 划分。
预期特征 `(620,768)`、标签 `(620,)`，保存到本地 cache/ 后读回核对；小型报告保存到 logs/。
提取所有图片的特征不等于评估测试集：这一步不作类别预测，也不计算任何分类指标。
如需减小显存占用，可显式传 `--batch-size 8`，实际批大小会记录到日志，不自动改变设置。

EUPE 验证集最近邻评估已完成，以下命令保留供复核。

```powershell
& '..\.conda-env\python.exe' scripts/evaluate_eupe_neighbors.py
```

配置固定指向 eupe_cache_20261101_140340_325624.json 对应的本地缓存。
直接复用其他三款模型的评估函数：434 张训练图作参考，93 张验证图作查询，
逐图 L2 归一化、float64 余弦 1NN、完全相同的最高相似度按最小全局参考编号处理。
仅用 CPU 读取缓存，无需重新加载模型；不更新参数，不评估测试集。
预测 CSV 已保存至 results/，指标和可追溯报告保存至 logs/；四模型对比见 MODEL_COMPARISON.md。

Qwen 2B 单图检查已由用户完成，下面保留检查命令。权重沿用固定本地快照，无需重新下载。

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

Qwen 2B 全量冻结特征已保存，以下命令仅供重新提取时参考。

```powershell
& '..\.conda-env\python.exe' scripts/cache_qwen2b_features.py
```

逐张处理原图，使用与成功单图检查相同的模型和预处理，预期 (620,2048) 特征。
保留模型身份、原始标签、类别顺序和固定划分，逐张核对图像 SHA256；
缓存写入后完整读回核对，另记录权重哈希与每张图的实际输入网格。
缓存留在本地 cache/，日志放在 logs/。此步骤不运行分类或更新参数。
如果处理失败，会报告当前样本编号并停止，不静默跳过样本或改变分辨率。

Qwen 2B 验证集评估已完成，以下命令保留供复核。

```powershell
& '..\.conda-env\python.exe' scripts/evaluate_qwen2b_neighbors.py
```

复用前两款模型的同一评估函数，434 张训练图片作参考，预测 93 张验证图片。
配置明确指定缓存报告；运行前检查缓存哈希、2048 维形状、模型版本、标签和划分。
逐图 L2 归一化后使用 float64 余弦 1NN，完全相同的最高相似度取最小参考编号。
仅在 CPU 读取缓存，无需加载 Qwen 权重；不更新参数，不评估测试集。
逐图预测保存至 results/，指标、混淆矩阵与溯源信息保存至 logs/。

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
