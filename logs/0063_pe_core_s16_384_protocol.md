# PE-Core-S16-384候选实验协议

日期：2026-11-16（America/New_York）

## 模型与实现

- 论文模型：Meta `facebook/PE-Core-S16-384`，Perception Encoder Core，ViT-S/16。
- 原生输入：384×384；视觉主干宽度384；12层、6头；约0.02B视觉参数。
- PE-Core在视觉塔上使用8头注意力池化，并投影为512维CLIP图像嵌入。
- Meta官方仓库说明PE已经集成进`timm`；本实验锁定`timm==1.0.22`中的`vit_pe_core_small_patch16_384.fb`和权重仓库`timm/vit_pe_core_small_patch16_384.fb`。
- 权重与代码许可：Apache-2.0。

本轮使用官方最终注意力池化和投影后的512维图像嵌入，而不是自行选择CLS token或平均patch token。统一LinearSVC协议随后仍会逐样本L2归一化。

## 实验边界

- 所有模型参数冻结；不建立或训练白族六分类神经网络头。
- 单图预检和620张特征缓存阶段均不执行分类。
- 后续只在434张训练图内部用固定5折选择LinearSVC的C，用全部434张重拟合，只在93张验证图评价。
- 不评价93张测试图，不用测试集选择模型。
- 原图、权重和特征缓存不上传GitHub；只上传配置、脚本、JSON/Markdown留痕和逐图验证预测。

## 环境处理

新增`timm==1.0.22`。首次导入时发现环境中的未来版`Pillow 12.3.0`无法加载`PIL._imaging` DLL，已回退到稳定版`Pillow 11.3.0`并验证`timm`、Pillow、torchvision可共同导入。没有修改图片、划分或已有特征缓存。
