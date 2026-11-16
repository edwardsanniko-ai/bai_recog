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

## 下载故障与固定方案

首次预检在Hugging Face权重下载前的HEAD请求阶段发生`SSL: UNEXPECTED_EOF_WHILE_READING`，重试5次后失败；模型没有加载，也没有产生实验结果。经公开镜像API核对，同一仓库为公开、非门控、Apache-2.0，精确提交为`255fa76e26fdf8daa47e0d607da46b7051879045`。`model.safetensors`大小为95,147,760 bytes，LFS SHA256为`b07ccd4489dcef7e1d2f3878a8232eb2457ba5b2d2086bd1512def4644cbfdda`。

由于当前`huggingface_hub 0.36.2`不能接受镜像HEAD元数据，改用独立PowerShell脚本从镜像下载这一精确版本，支持断点续传，并在加载前强制核对文件大小和SHA256。预检与缓存脚本此后只离线加载已经校验的本地权重。
