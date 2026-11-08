# 0040 — CNN ConvNeXt-Tiny 阶段 B seed 42 第 1 轮

日期：2026-11-08（America/New_York）

用户运行阶段 B 正式训练的第 1 轮。训练从阶段 A seed 42 的最佳第 7 轮模型开始，只更新 `features.7` 最后 stage 和六分类头；其余参数保持冻结。

结果：

- 训练图片：434
- 验证图片：93
- batch size：16
- 本轮训练 batch：28
- 平均训练损失：0.763065431112518
- 验证 Accuracy：0.7741935483870968（72/93）
- 验证 Macro-F1：0.7708456530760452
- 最后 stage 学习率：1e-5
- 分类头学习率：3e-4
- 本轮及累计优化器更新：28
- 最后 stage 参数发生变化：是
- 分类头参数发生变化：是
- 冻结参数 SHA256 保持不变：是
- 测试集评估：否

阶段 A 起点 Macro-F1 为 0.7941857298474946。阶段 B 第 1 轮没有超过该值，因此 patience=1，最佳轮次仍为 0；这里的 epoch 0 明确表示阶段 A 初始化模型。`best.pt` 经读回确认为 completed_epoch=0、更新次数 0，`last.pt` 为 completed_epoch=1、累计更新 28，二者没有混淆。

逐图验证 CSV 共 93 行，位于 `results/cnn_convnext_last_stage_seed42_validation_epoch001.csv`；机器可读报告为 `logs/cnn_convnext_last_stage_seed42_epoch001.json`。本地 checkpoint 被 `.gitignore` 排除，不上传 GitHub。下一步从 `last.pt` 恢复，由脚本继续比较每轮成绩与阶段 A 起点及阶段 B 历史最佳值。
