# 0029 — ResNet50 / ConvNeXt-Tiny 控制组单图检查准备

日期：2026-11-05（America/New_York）。

用户要求增加 ResNet50 / ConvNeXt-T 控制组。选择 torchvision 0.23.0+cu128 的官方默认权重：

- ResNet50_Weights.IMAGENET1K_V2，监督式 ImageNet-1K 预训练；原模型 25557032 参数，
  权重文件 resnet50-11ad3fa6.pth，102540417 字节，SHA256
  11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca。
- ConvNeXt_Tiny_Weights.IMAGENET1K_V1，监督式 ImageNet-1K 预训练；原模型 28589128 参数，
  权重文件 convnext_tiny-983f1562.pth，114419221 字节，SHA256
  983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d。

两个公开官方权重已下载到本地忽略目录 hf_cache/torch/checkpoints，不上传 GitHub。
配置固定 URL、权重枚举、文件名、完整 SHA256、torchvision 版本、特征维度和评估协议。
模型先由 torchvision builder 严格加载完整 1000 类 checkpoint，确认分类层结构后，
仅将最后的 ImageNet-1000 线性层换成 Identity。ResNet50 取 global average pool 后的
2048 维整图特征；ConvNeXt-Tiny 保留官方 adaptive average pool、LayerNorm 和 Flatten，
取最后线性层之前的 768 维特征。ImageNet 分类头不用于白族标签，也不训练新分类头。

官方权重预处理分别为：ResNet50 短边 resize 232、中心裁剪 224；ConvNeXt-Tiny
短边 resize 236、中心裁剪 224；均为 bilinear antialias、float [0,1]、ImageNet normalize。
推理使用 float32，关闭 CUDA / cuDNN TF32，eval + inference_mode，全部参数冻结。
后续维持 434 张训练参考、93 张验证查询、逐图 L2、float64 余弦 1NN、最小编号平局规则；
测试集不用于筛选，参数更新 0。

环境诊断：直接调用当前 Conda 环境 python.exe 时 Pillow 12.3.0 的 `_imaging` 报 DLL load failed；
文件哈希与 Conda 本地缓存一致，ctypes 可加载，说明 Pillow 文件没有损坏。
根因是直接调用环境 python.exe 时遗漏 `<env>/Library/bin` DLL 搜索目录。
脚本在导入 Pillow/torchvision 前显式注册该目录，不修改或重装环境包。

开发验证：两个官方权重 SHA256 正确，完整模型成功加载并移除最后线性层；
CUDA 合成输入得到 ResNet50 `(1,2048)`、ConvNeXt-Tiny `(1,768)`，float32、有限且非零，
全部参数冻结。没有读取真实数据、生成分类预测、评估验证/测试集或更新参数。
下一步由用户运行 scripts/smoke_test_torchvision_controls.py，回传两个控制组单图输出。
