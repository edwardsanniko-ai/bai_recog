# 0035 — CNN ConvNeXt-Tiny 线性探针 seed 42 第 1 轮

日期：2026-11-07（America/New_York）

正式训练循环 `scripts/train_convnext_linear_probe.py` 已建立。它固定读取 `configs/cnn_convnext_tiny_linear_probe.json`，只实例化训练集和验证集，保存模型、优化器、余弦学习率调度器、DataLoader generator 及 Python/NumPy/PyTorch CPU/CUDA 随机状态，可从上一轮精确继续。checkpoint 位于本地 `checkpoints/`，由 `.gitignore` 排除，不上传 GitHub。

本次运行：

```powershell
& '..\.conda-env\python.exe' scripts/train_convnext_linear_probe.py --stop-after-epoch 1
```

运行前重新核验全部 620 张原图 SHA256 和固定 434/93/93 划分。训练只使用 434 张训练图，batch size 16，共 28 个 batch；验证只使用 93 张验证图。没有实例化或评估测试数据。

第 1 轮结果：

- 平均训练损失：1.7062891452543196
- 验证 Accuracy：0.5161290322580645（48/93）
- 验证 Macro-F1：0.5036881419234361
- 本轮优化器更新：28
- 累计优化器更新：28
- 分类头参数发生变化：是
- 冻结参数 SHA256 保持不变：是
- 当前最佳轮次：1
- 当前最佳验证 Macro-F1：0.5036881419234361
- 停止原因：按计划在第 1 轮保存恢复点，不是早停
- 测试集评估：否

逐图验证输出为 `results/cnn_convnext_linear_probe_seed42_validation_epoch001.csv`，共 93 行。机器可读报告为 `logs/cnn_convnext_linear_probe_seed42_epoch001.json`。本地 `last.pt` 与 `best.pt` 均已生成；独立读回确认 `completed_epoch=1`、`best_epoch=1`、`total_optimizer_updates=28`、配置 SHA256 一致且完整随机状态存在。

第 1 轮的 50.37% Macro-F1 只是随机初始化六分类头开始学习后的中间检查，不与已收敛模型或冻结 1NN 结果作最终排名。下一步从 `last.pt` 继续第 2 轮，保持配置文件不变，以免破坏 checkpoint 的协议哈希。
