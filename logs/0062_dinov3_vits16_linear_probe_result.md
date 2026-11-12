# DINOv3 ViT-S/16统一冻结线性探测结果

日期：2026-11-11（America/New_York）

## 协议

本次在原六模型统一协议中只增加`facebook/dinov3-vits16-pretrain-lvd1689m`：原生冻结特征、逐样本L2归一化、434张训练图内部固定5折选择LinearSVC的C、全部434张训练图重新拟合、93张验证图评价。没有更新骨干参数，没有评价测试集。

## 七模型验证排名

| 排名 | 冻结表示 | 验证 Macro-F1 |
|---:|---|---:|
| 1 | ImageNet ConvNeXt-Tiny | 0.884835 |
| 2 | SigLIP 2 Base | 0.883387 |
| 3 | Qwen3-VL-Embedding-2B | 0.883102 |
| 4 | EUPE ConvNeXt-Tiny | 0.862064 |
| 5 | DINOv3 ConvNeXt-Tiny | 0.849752 |
| 6 | DINOv3 ViT-S/16 | 0.842847 |
| 7 | ImageNet ResNet50 | 0.783305 |

DINOv3 ViT-S/16选择的`C=1`，训练集内部5折Macro-F1为`0.869582`；固定验证集Accuracy为`0.838710`（78/93），Macro-F1为`0.842847`。它比DINOv3 ConvNeXt-Tiny少正确1张，Macro-F1低`0.006905`；比第一名ImageNet ConvNeXt-Tiny低`0.041988`。

## DINOv3 ViT-S/16逐类结果

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| geometric | 0.882353 | 1.000000 | 0.937500 | 15 |
| animal | 0.684211 | 0.812500 | 0.742857 | 16 |
| antique object | 0.933333 | 0.875000 | 0.903226 | 16 |
| plant | 0.846154 | 0.733333 | 0.785714 | 15 |
| Chinese character | 1.000000 | 0.857143 | 0.923077 | 14 |
| combined | 0.764706 | 0.764706 | 0.764706 | 17 |

主要混淆集中在`animal`与`combined`：3张combined被判为animal，2张animal被判为combined；另有1张animal被判为plant。DINOv3 ViT-S/16与DINOv3 ConvNeXt-Tiny共有12张相同错图；前者独有3张错图，后者独有2张错图。因此两者总体差距仅约一张图片，但错误边界并不完全相同。

相对DINOv3 ConvNeXt-Tiny，ViT-S/16在antique object和Chinese character上F1更高，在geometric与plant上相同，在animal与combined上更低。其最大差距来自animal（低`0.080672`）。

## 解释边界

本轮不支持“DINOv3的Transformer骨干优于DINOv3的CNN骨干”。但这也不是只改变架构的严格单因素实验：ViT-S/16约21.6M参数、384维特征，而DINOv3 ConvNeXt-Tiny约29M参数、768维特征。两者预训练方法和数据来源相近，但容量、特征维度及结构同时不同。

ViT-S/16的训练内部CV分数高于DINOv3 ConvNeXt-Tiny，而固定验证分数略低，说明620张小数据集下结果对具体划分可能敏感。模型筛选仍按预先锁定的验证Macro-F1执行，不能改用训练内部CV分数为ViT-S/16翻盘，也不能据此访问测试集。

完整报告：`logs/frozen_linear_probe_comparison_20261111_225049_614219.json`。逐图预测文件使用相同时间戳保存在`results/`。骨干参数更新次数为0，`test_evaluated=false`。
