# 0049 — CNN 确认运行 seed 44 阶段 B 第 1 轮

日期：2026-11-08（America/New_York）

当前仍在评估第一个可训练 CNN baseline：torchvision ConvNeXt-Tiny ImageNet-1K V1 的锁定两阶段方案。seed 44 阶段 B 从本 seed 的阶段 A 最佳第 5 轮开始，epoch 0 Macro-F1 为 0.7680746246810365。

第 1 轮结果：

- 平均训练损失：0.839110876283338
- 验证 Accuracy：0.7849462365591398（73/93）
- 验证 Macro-F1：0.7864355388620093
- 最后 stage 学习率：1e-5
- 分类头学习率：3e-4
- 本轮及累计优化器更新：28
- 最后 stage 参数变化：是
- 分类头参数变化：是
- 冻结参数保持不变：是
- 测试集评估：否

本轮超过 seed 44 阶段 A 起点约 1.84 个百分点，因此 best_epoch=1、patience=0。`last.pt` 和 `best.pt` 均读回为 seed 44、`run_role=confirmatory`、completed_epoch=1、累计更新 28；两者绑定相同的 seed 44 阶段 A 初始化 SHA256。逐图验证 CSV 共 93 行。

剩余确认训练为：seed 44 阶段 B 继续至早停；seed 45 的阶段 A 与 B；seed 46 的阶段 A 与 B。它们仍属于同一个 ConvNeXt-Tiny baseline 的多种子稳定性确认，不是五个不同 CNN 模型。四个确认种子完成并汇总验证结果、锁定选择后，才允许评估测试集。
