# ConvNeXt-Tiny 两阶段 CNN：锁定最终测试

日期：2026-11-08（America/New_York）

## 实验身份

- 模型：`torchvision ConvNeXt-Tiny`，`ConvNeXt_Tiny_Weights.IMAGENET1K_V1`。
- 数据划分：固定 434 train / 93 validation / 93 test。
- 阶段 A：冻结 backbone，只训练分类头。
- 阶段 B：从同一 seed 的阶段 A 最佳 checkpoint 出发，只解冻 `features.7` 和分类头。
- 确认种子：43、44、45、46；seed 42 是开发运行，不进入最终均值。
- checkpoint 在测试前锁定；测试不使用 ensemble 或 test-time augmentation。
- 每个 checkpoint 仅对测试集运行一次；测试期间优化器更新为 0。
- 测试结果不得用于回头选择 epoch、模型或超参数。

## 验证集确认

阶段 B 的验证 Macro-F1 为 `0.873328 ± 0.012500`（四个确认种子的均值 ± 样本标准差）。
四个种子的阶段 B 均优于各自的阶段 A。

## 锁定测试结果

| seed | 最佳阶段 B epoch | Test Accuracy | Test Macro-F1 |
|---:|---:|---:|---:|
| 43 | 20 | 0.720430 | 0.720972 |
| 44 | 20 | 0.720430 | 0.723698 |
| 45 | 14 | 0.709677 | 0.714950 |
| 46 | 22 | 0.720430 | 0.722566 |

四种子汇总（均值 ± 样本标准差）：

| 指标 | 结果 |
|---|---:|
| Accuracy | `0.717742 ± 0.005376` |
| Macro Precision | `0.750029 ± 0.005468` |
| Macro Recall | `0.720792 ± 0.005373` |
| Macro-F1 | `0.720547 ± 0.003895` |
| Weighted-F1 | `0.715969 ± 0.004048` |

验证 Macro-F1 均值与测试 Macro-F1 均值相差 `0.152781`（15.28 个百分点）。这是固定划分下的
描述性结果，说明验证成绩没有充分转化到测试集；当前证据不能单独区分过拟合、划分分布差异、
同源纹样分组问题或其他原因，因此不据此修改本模型。

四个种子的逐类平均结果显示，`combined` 最弱：平均 F1 为 `0.462903`，平均召回率为
`0.437500`。其次值得注意的是 `geometric` 平均召回率为 `0.550000`。这些数字只用于
结果解释，不用于测试后调参。

## 独立核验

`scripts/verify_cnn_final_test.py` 不加载模型、不访问图片、不重新推理，只读取最终 JSON 和
四份 CSV，完成以下核验：

- 每份逐图预测均为 93 行，样本顺序与锁定测试索引完全一致；
- checkpoint、锁定配置、验证汇总和逐图 CSV 的 SHA256 与报告一致；
- 概率有限且逐行和为 1，预测类别等于概率最大类别；
- 从 CSV 独立重算的五项指标和混淆矩阵与最终报告完全一致；
- 四种子的均值和样本标准差与最终报告完全一致；
- 核验期间推理次数和优化器更新次数均为 0。

## 可追溯文件

- 锁定规则：`configs/cnn_convnext_final_test.json`
- 一次性测试脚本：`scripts/evaluate_cnn_final_test.py`
- 只读核验脚本：`scripts/verify_cnn_final_test.py`
- 最终报告：`logs/cnn_convnext_final_test_20261108_063602_024373.json`
- 逐图预测：`results/cnn_convnext_seed{43,44,45,46}_test_20261108_063602_024373.csv`

该测试集现已被查看。后续若研究新模型或新超参数，必须披露这一点，不得再将这 93 张图片称为
从未使用过的盲测集；当前 ConvNeXt-Tiny 两阶段 baseline 的结果保持锁定。
