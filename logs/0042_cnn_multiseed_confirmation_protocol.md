# 0042 — CNN 两阶段多种子确认协议

日期：2026-11-08（America/New_York）

阶段 B seed 42 达到验证 Macro-F1 0.8732394845298072 后，模型结构、预处理、增强、学习率、weight decay、batch size、scheduler、早停规则和验证指标全部锁定，不再根据后续种子的结果修改。

seed 42 已用于选择阶段 A→B 结构并多次观察验证成绩，因此标记为开发运行，不作为完全独立的确认重复。接下来运行 seed 43、44、45、46。每个种子都必须从官方 ImageNet-1K V1 权重和自己随机初始化的六分类头开始完整训练阶段 A，再使用同一个种子的阶段 A 最佳 checkpoint 初始化阶段 B。禁止让 seed 43–46 共用 seed 42 的分类头或阶段 B checkpoint。

主要稳定性汇总为 seed 43–46 验证 Accuracy 和 Macro-F1 的均值及样本标准差。可以附带展示 seed 42–46 的描述性汇总，但必须明确 seed 42 是开发运行。多种子使用相同 434/93 数据划分，只衡量初始化、shuffle、增强、Dropout 和 stochastic depth 等训练随机性；它们不是新的独立数据样本，不能替代交叉验证或外部数据集。

测试集在四个确认种子全部完成、训练协议不再修改且候选 baseline 选择完成前继续锁定。第一小步只运行 seed 43 的阶段 A 第 1 轮，以核对新随机初始化、checkpoint 隔离和日志命名；通过后从该 seed 的 `last.pt` 恢复至早停。

机器可读协议为 `configs/cnn_convnext_multiseed_confirmation.json`。`scripts/train_convnext_linear_probe.py` 新增 `--seed` 参数，只接受锁定配置中的种子；默认仍为 development seed 42，不改变既有 seed 42 原始结果。
