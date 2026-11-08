# 0048 — CNN seed 44 阶段 A 完成与阶段 B 检查

日期：2026-11-08（America/New_York）

## 阶段 A

确认运行 seed 44 正常早停：

- 停止轮次：15
- 停止原因：early stopping
- 最佳轮次：5
- 最佳验证 Accuracy：0.7741935483870968（72/93）
- 最佳验证 Macro-F1：0.7680746246810365
- 正式累计优化器更新：420
- 最佳 checkpoint 对应更新：140
- 全部轮次 backbone 冻结参数保持不变
- 测试集未评估

独立加载 seed 44 `best.pt` 复算 93 张验证图，Accuracy 和 Macro-F1 与第 5 轮报告完全一致。`last.pt` 为 completed_epoch=15、patience=10；`best.pt` 为 completed_epoch=5、patience=0，均标记 `run_role=confirmatory`。

## 阶段 B 无更新检查

运行 `scripts/inspect_convnext_last_stage.py --seed 44`，严格加载 seed 44 自己的阶段 A 最佳 checkpoint：

- 来源阶段 A 最佳轮次：5
- 来源阶段 A Macro-F1：0.7680746246810365
- 第一批输入：`(16,3,224,224)`
- 六分类输出：`(16,6)`
- 最后 stage 可训练参数：14,289,408
- 分类头可训练参数：6,150
- 冻结参数：13,529,184
- 优化器更新：0
- 测试集评估：否

机器可读报告为 `logs/cnn_convnext_last_stage_seed44_preflight_20261108_043746_029632.json`。下一步只运行 seed 44 阶段 B 第 1 轮；epoch 0 基准为本 seed 的 0.7680746246810365。
