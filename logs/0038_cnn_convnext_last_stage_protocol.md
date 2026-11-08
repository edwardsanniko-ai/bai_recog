# 0038 — CNN ConvNeXt-Tiny 阶段 B 最后 stage 微调协议

日期：2026-11-08（America/New_York）

阶段 A seed 42 的线性探针最佳验证 Macro-F1 为 0.7941857298474946，对应第 7 轮。阶段 B 按预定流程使用该 `best.pt` 中的六分类头作为起点，同时保持 backbone 来源为 torchvision 官方 ConvNeXt-Tiny ImageNet-1K V1 权重。

可训练范围固定为 `features.7` 的三个 768 通道 CNBlock，以及 `classifier.0` LayerNorm 和 `classifier.3` 六分类 Linear。`features.0` 至 `features.6`（包括进入最后 stage 前的降采样层）全部冻结。最后 stage 学习率为 1e-5，分类头学习率为 3e-4；两组都使用 AdamW、weight decay 1e-3 和同一 cosine schedule。

数据划分、224×224 等比例白色补边、轻量增强、batch size 16、最多 50 轮、patience 10、验证 Macro-F1 选择以及测试集锁定规则均与阶段 A 相同。开发阶段仍只运行 seed 42；协议锁定后才考虑五种子确认。

配置固定在 `configs/cnn_convnext_tiny_last_stage.json`。阶段 A 最佳 checkpoint 的 SHA256、来源轮次和验证成绩均写入配置，加载时必须完全匹配。第一步只运行 `scripts/inspect_convnext_last_stage.py`：核验全部图片哈希、初始化 checkpoint、参数白名单、参数组计数和一次无梯度前向；不创建优化器、不反向传播、不更新参数、不计算验证指标、不实例化测试集。
