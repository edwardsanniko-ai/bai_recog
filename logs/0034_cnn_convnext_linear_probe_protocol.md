# 0034 — CNN ConvNeXt-Tiny 线性探针协议固定

日期：2026-11-07（America/New_York）

用户决定启动可训练 CNN baseline。第一阶段使用 torchvision ConvNeXt-Tiny 与 ImageNet-1K V1 官方监督预训练权重，只训练新的六分类头；不把此前冻结特征 + 余弦 1NN 结果误称为可训练 CNN 分类器。

固定数据边界：继续使用 `data/splits.json` 的 434/93/93 划分。训练集用于梯度更新，验证集用于 checkpoint 与超参数选择，测试集在模型结构和超参数锁定前不得实例化或评估。开发阶段先运行 seed 42；协议锁定后再运行 42–46 五个独立训练种子，避免把调参过程中的多次验证观察当作确认性重复。

第一阶段结构为 ConvNeXt-Tiny backbone + 官方 classifier LayerNorm + Dropout(0.2) + Linear(768,6)。backbone 全部冻结，只有 LayerNorm 和六分类线性层可训练。初始参数为 batch size 16、AdamW、head learning rate 3e-4、weight decay 1e-3、CosineAnnealingLR、最多 50 epochs、验证 Macro-F1 早停 patience 10。

输入使用 224×224 等比例缩放和白色居中补边，以减少对稀疏线条、窄条边饰和长宽比的破坏。训练增强只包括 ±5° 仿射旋转、最多 5% 平移、0.95–1.05 缩放和轻微亮度/对比度扰动；禁用水平翻转和 RandomResizedCrop。验证集无随机增强。该预处理与此前冻结 1NN 控制组的官方中心裁剪不同，后续比较时必须披露。

本步骤先增加无参数更新的 preflight：核验全部 620 张图像哈希、划分互斥、第一批张量、分类分数形状以及可训练参数白名单。preflight 通过后才编写并运行正式训练循环。
