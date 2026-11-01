# 0015 — 单步更新完成，准备训练集分批检查

日期：2026-11-01（Asia/Shanghai）

核实用户报告 dinov3_update_demo_20261101_121539_449610.json：
AdamW、lr=0.001、weight_decay=0、seed=42，只对训练图片 237 更新一次。
损失 2.708889 -> 1.091462，真实类别概率 0.066611 -> 0.335725；
backward 后参数未变，step 后改变。这是教学结果，不是正式基线成绩。

下一步 inspect_dinov3_batches.py --batch-size 16：
- 仅将原始 train_idx 的 434 个样本加入 TensorDataset。
- DataLoader 使用 shuffle=true、drop_last=false、num_workers=0，独立随机生成器 seed=42。
- 不创建模型/优化器，不训练，不计算分类指标。
- 验证特征、标签、样本编号保持对应，434 个训练样本各出现一次，无验证/测试样本混入。
- 预期共 28 批，其中前 27 批各 16 个，最后 2 个；实际批次顺序保存为日志。
- 这只是分批演示，学习率等正式参数尚未确定。

开发核验：用户教学 checkpoint 的 SHA256 与报告匹配，安全读回确认更新一次且参数改变。
批次脚本已完整运行，输出 (16,768)、(16,)、28 批、末批 2，覆盖与隔离检查通过，
optimizer_updates=0。开发报告 dinov3_batches_20261101_121831_728956.json。
