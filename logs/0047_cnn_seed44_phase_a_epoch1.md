# 0047 — CNN 确认运行 seed 44 阶段 A 第 1 轮

日期：2026-11-08（America/New_York）

seed 44 按锁定的多种子协议，从官方 ConvNeXt-Tiny ImageNet-1K V1 权重和本 seed 新随机初始化的六分类头开始阶段 A。没有复用 seed 42 或 seed 43 的分类头和 checkpoint。

第 1 轮结果：

- 平均训练损失：1.672129140471533
- 验证 Accuracy：0.4946236559139785（46/93）
- 验证 Macro-F1：0.46474237395290025
- 本轮及累计优化器更新：28
- 分类头参数发生变化：是
- 冻结 backbone 参数保持不变：是
- 测试集评估：否

机器可读报告标记 `seed=44`、`run_role=confirmatory`。`last.pt` 与 `best.pt` 均读回为 seed 44、completed_epoch=1、累计更新 28，且完整保存 Python、NumPy、PyTorch 和 DataLoader 随机状态。逐图验证 CSV 为 93 行；checkpoint 使用独立目录 `checkpoints/cnn_convnext_linear_probe_seed44/`。

第 1 轮尚未收敛，不与其他种子的最终最佳结果比较。下一步只从 seed 44 自己的 `last.pt` 恢复至早停，保持全部锁定参数不变。
