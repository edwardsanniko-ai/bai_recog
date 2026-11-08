# 0045 — CNN 确认运行 seed 43 阶段 B 第 1 轮

日期：2026-11-08（America/New_York）

seed 43 阶段 B 从本 seed 的阶段 A 最佳第 8 轮开始，epoch 0 验证 Macro-F1 为 0.7824825637325638。

第 1 轮结果：

- 平均训练损失：0.7144178547617477
- 验证 Accuracy：0.7849462365591398（73/93）
- 验证 Macro-F1：0.7812788118467999
- 最后 stage 学习率：1e-5
- 分类头学习率：3e-4
- 本轮及累计优化器更新：28
- 最后 stage 参数变化：是
- 分类头参数变化：是
- 冻结参数保持不变：是
- 测试集评估：否

第 1 轮未超过本 seed 的阶段 A 起点，因此 patience=1，最佳阶段 B 轮次仍为 0。`last.pt` 读回为 seed 43、`run_role=confirmatory`、completed_epoch=1、累计更新 28；`best.pt` 为 completed_epoch=0、更新 0。两者记录的阶段 A 初始化 checkpoint SHA256 一致，没有引用 seed 42。

逐图验证 CSV 共 93 行。下一步从 seed 43 阶段 B 的 `last.pt` 恢复至早停，不修改锁定超参数。
