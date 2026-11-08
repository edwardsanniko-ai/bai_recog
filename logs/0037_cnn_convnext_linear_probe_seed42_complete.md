# 0037 — CNN ConvNeXt-Tiny 线性探针 seed 42 完成与早停审计

日期：2026-11-08（America/New_York）

## 正式结果

ConvNeXt-Tiny 阶段 A（冻结 backbone、只训练 LayerNorm 和六分类线性层）的 seed 42 开发运行已经完成。按配置的验证 Macro-F1 和 early-stopping patience=10，正式训练应在第 17 轮停止：第 7 轮达到最佳验证 Macro-F1，随后第 8–17 轮连续 10 轮没有提高。

- 正式停止轮次：17
- 正式停止原因：early stopping
- 正式累计优化器更新：476（17×28）
- 最佳轮次：7
- 最佳验证 Accuracy：0.7956989247311828（74/93）
- 最佳验证 Macro-F1：0.7941857298474946
- 最佳 checkpoint：本地 `checkpoints/cnn_convnext_linear_probe_seed42/best.pt`
- 最佳 checkpoint 对应累计更新：196
- 测试集：未实例化、未评估

从 `best.pt` 独立加载模型并重新计算全部 93 张验证图，得到 Accuracy 0.7956989247311828、Macro-F1 0.7941857298474946，与 `logs/cnn_convnext_linear_probe_seed42_epoch007.json` 完全一致。

## 第 18 轮的处理

第 17 轮已经达到 patience=10 后，用户再次运行相同的恢复命令。旧脚本只在每轮末尾检查早停，没有在恢复入口拒绝已经满足早停条件的 checkpoint，因此额外执行了第 18 轮：Macro-F1 0.792543301832518，未超过第 7 轮，也没有覆盖 `best.pt`。

第 18 轮属于早停后的误恢复，保留原始 JSON 和逐图 CSV 作为审计证据，但从正式训练轮数、累计更新次数和模型选择中排除。不能把其 504 次累计更新写成正式阶段 A 的训练预算。

恢复入口已增加保护：若 checkpoint 的 patience 已达到阈值，或已经达到最大 epoch，脚本只报告已完成状态并返回，不执行 forward/backward 或参数更新。使用当前 `last.pt` 回归检查得到：

```text
Checkpoint already reached early stopping after epoch 18.
No training or parameter update was performed.
Best epoch: 7
Best validation Macro-F1: 0.794186
Test evaluated: False
```

## 当前解释

阶段 A 最佳 Macro-F1 79.42%，低于同一 ConvNeXt-Tiny 官方权重在先前“冻结特征 + cosine 1NN”协议下的 83.18%。两者的判别规则和预处理不同：阶段 A 使用可训练线性头、轻量增强及等比例白色补边，1NN 对照使用官方中心裁剪。因此当前只能说明这套阶段 A 配置没有超过先前无训练对照，不能把差异完全归因于分类头或预处理中的某一个因素。

按预定流程，下一步是建立阶段 B：从同一官方 ImageNet 权重开始，解冻 ConvNeXt 最后一个 stage，并使用较小 backbone 学习率；在实施前先单独固定配置和做一次无更新的参数白名单检查。
