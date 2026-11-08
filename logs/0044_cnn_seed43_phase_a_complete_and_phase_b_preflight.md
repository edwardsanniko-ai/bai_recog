# 0044 — CNN seed 43 阶段 A 完成与阶段 B 检查

日期：2026-11-08（America/New_York）

## 阶段 A

确认运行 seed 43 的阶段 A 正常早停：

- 停止轮次：18
- 停止原因：early stopping
- 最佳轮次：8
- 最佳验证 Accuracy：0.7849462365591398（73/93）
- 最佳验证 Macro-F1：0.7824825637325638
- 正式累计优化器更新：504
- 最佳 checkpoint 对应更新：224
- 全部轮次 backbone 冻结参数保持不变
- 测试集未评估

独立加载 `checkpoints/cnn_convnext_linear_probe_seed43/best.pt` 复算 93 张验证图，Accuracy 和 Macro-F1 与第 8 轮报告完全一致。`last.pt` 为 completed_epoch=18、patience=10；`best.pt` 为 completed_epoch=8、patience=0，均标记 seed 43 和 `run_role=confirmatory`。

## 阶段 B 无更新检查

阶段 B 的模型加载与训练脚本已扩展为按 seed 绑定各自的阶段 A 最佳 checkpoint。对于 seed 43，加载器验证了：确认种子属于锁定集合、阶段 A 配置 SHA256、数据清单 SHA256、checkpoint seed/role、最佳轮次及内部 history 一致；没有使用 seed 42 checkpoint。

运行 `scripts/inspect_convnext_last_stage.py --seed 43` 得到：

- 来源阶段 A 最佳轮次：8
- 来源阶段 A Macro-F1：0.7824825637325638
- 第一批输入：`(16,3,224,224)`
- 六分类输出：`(16,6)`
- 最后 stage 可训练参数：14,289,408
- 分类头可训练参数：6,150
- 冻结参数：13,529,184
- 优化器更新：0
- 测试集评估：否

机器可读检查报告为 `logs/cnn_convnext_last_stage_seed43_preflight_20261108_035606_739096.json`。下一步只运行 seed 43 阶段 B 第 1 轮；其 epoch 0 最佳基准为本 seed 的 0.7824825637325638，而不是 seed 42 的成绩。
