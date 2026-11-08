# 0043 — CNN 确认运行 seed 43 阶段 A 第 1 轮

日期：2026-11-08（America/New_York）

seed 43 作为锁定协议后的确认运行，从官方 ConvNeXt-Tiny ImageNet-1K V1 权重和该 seed 新随机初始化的六分类头开始阶段 A；没有复用 seed 42 的分类头或 checkpoint。

第 1 轮结果：

- 训练图片：434
- 验证图片：93
- 训练 batch：28
- 平均训练损失：1.6980975728979857
- 验证 Accuracy：0.4731182795698925（44/93）
- 验证 Macro-F1：0.41437164739490323
- 本轮及累计更新：28
- 分类头参数发生变化：是
- 冻结 backbone 参数保持不变：是
- 测试集评估：否

机器可读报告的 `seed=43`、`run_role=confirmatory`。`last.pt` 和 `best.pt` 均读回为 seed 43、completed_epoch=1、累计更新 28；Python、NumPy、PyTorch 和 DataLoader 随机状态完整保存。逐图验证 CSV 共 93 行。checkpoint 位于独立的 `checkpoints/cnn_convnext_linear_probe_seed43/`，没有覆盖 seed 42。

seed 43 第 1 轮成绩低于 seed 42 第 1 轮不构成异常：分类头初始化、训练顺序和随机增强不同，而且两者都尚未收敛。下一步从 seed 43 自己的 `last.pt` 继续到早停，期间不修改任何锁定超参数。
