# 0039 — CNN ConvNeXt-Tiny 阶段 B 参数范围检查通过

日期：2026-11-08（America/New_York）

运行 `scripts/inspect_convnext_last_stage.py` 前，严格核验全部 620 张原图 SHA256、固定 434/93/93 划分、官方 ImageNet checkpoint 和阶段 A `best.pt`。首次检查因人工抄写 SHA256 时漏掉一个相邻字符而在模型前向前停止；重新读取实际文件哈希并确认 checkpoint 内部为 seed 42、第 7 轮、Macro-F1 0.7941857298474946 后，修正配置并复跑成功。哈希校验没有被放宽。

最终无更新检查结果：

- 阶段 A 来源最佳轮次：7
- 阶段 A 来源验证 Macro-F1：0.7941857298474946
- 第一批输入：`(16,3,224,224)`
- 六分类分数：`(16,6)`
- `features.7` 可训练参数：14,289,408
- 分类头可训练参数：6,150
- 总可训练参数：14,295,558
- 冻结参数：13,529,184
- 可训练参数名称全部属于 `features.7`、`classifier.0` 或 `classifier.3`
- 优化器更新：0
- 测试集评估：否

机器可读报告为 `logs/cnn_convnext_last_stage_preflight_20261107_222111_833366.json`。报告文件名使用运行环境生成的 America/New_York 时间戳；本实验记录按客户端日期 2026-11-08 归档。

正式训练脚本 `scripts/train_convnext_last_stage.py` 已建立。阶段 A 最佳成绩作为阶段 B 的 epoch 0 基准和初始 `best.pt`；只有阶段 B 的验证 Macro-F1 超过该值才覆盖最佳 checkpoint。第一次仍只运行一个阶段 B epoch，检查两组学习率、参数变化和显存/运行稳定性后再恢复继续。
