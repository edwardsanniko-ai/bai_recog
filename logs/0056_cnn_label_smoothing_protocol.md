# ConvNeXt-Tiny 阶段 B：轻度 label smoothing 筛选协议

日期：2026-11-08（America/New_York）

B1 提高分辨率的多种子确认未通过，因此模型恢复到原 B0：224 输入、全局平均池化、只训练
`features.7` 和分类头。下一轮只改变交叉熵的 `label_smoothing`：从 `0.0` 改为 `0.05`，
候选记为 L1。

选择规则在训练前锁定：seed 42 的 L1 验证 Macro-F1 至少比 B0 的 `0.873239` 高
`0.005`，才进入 seed 43–46 的配对确认。单 seed 结果不用于宣称稳定提升。类别权重、focal
loss、输入分辨率、池化、增强、解冻范围和优化器设置均不改变。测试集不实例化、不评估。

配置：`configs/cnn_label_smoothing_screen.json`

训练脚本：`scripts/train_cnn_label_smoothing.py`
