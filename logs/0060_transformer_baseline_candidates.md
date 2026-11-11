# 视觉Transformer候选模型调研

调研日期：2026-11-11（America/New_York）

## 目标与筛选条件

面向620张、6类白族纹样数据集，筛选能够公开获得预训练权重、可冻结提取图像特征、适合RTX 4060 Laptop GPU进行推理的视觉Transformer。候选优先用于与现有模型相同的“冻结特征 + L2归一化 + 训练集内部5折选择C + 线性SVM + 验证集评估”协议。

这里优先比较实用的冻结表示管线，不声称隔离了训练数据、模型规模、输入分辨率等所有因素。

## 第一优先级：建议实际测试

### T1：DINOv3 ViT-S/16

- 架构：Vision Transformer，patch 16。
- 参数量：21M。
- 预训练：language-free DINOv3，LVD-1689M。
- 官方Hugging Face ID：`facebook/dinov3-vits16-pretrain-lvd1689m`。
- 价值：当前已经测试DINOv3 ConvNeXt-Tiny（29M）。换成同一DINOv3方法和同一预训练数据下的ViT-S/16，能较好地隔离“CNN骨干与Transformer骨干”的影响。
- 可行性：规模小于当前DINOv3 ConvNeXt-Tiny；用户已经获得DINOv3 Hugging Face访问权限。
- 论文：Fan et al., *Scaling Language-Free Visual Representation Learning*, ICCV 2025, DOI `10.1109/ICCV51701.2025.00042`。
- 官方实现：`https://github.com/facebookresearch/dinov3`。

### T2：Perception Encoder PE-Core-S16-384

- 架构：ViT-S/16。
- 参数量：官方表记为0.02B，约20M。
- 输入：384px。
- 官方Hugging Face ID：`facebook/PE-Core-S16-384`。
- 预训练方向：通用视觉/图文表示；PE强调不同网络层的视觉嵌入质量，而不是只取最终输出。
- 价值：较高输入分辨率可能更有利于白族纹样中的细线，同时参数规模仍较小。
- 风险：384输入会增加计算量，并且与224输入模型的预处理不完全相同；这属于原生模型管线差异。
- 论文：Bolya et al., *Perception Encoder: The best visual embeddings are not at the output of the network*, NeurIPS 2025, DOI `10.52202/085713-2036`。
- 官方实现：`https://github.com/facebookresearch/perception_models`，Apache-2.0代码许可。

### T3：MetaCLIP 2 ViT-S/16 Worldwide

- 架构：ViT-S/16。
- 输入：224px；官方还提供384px版本。
- 预训练：全球多语言图文数据，官方表列出29B训练样本规模。
- 官方权重名：`ViT-S-16-worldwide@WorldWideCLIP`；官方README提供直接checkpoint链接及Hugging Face collection。
- 价值：与SigLIP 2同属图文对齐路线，但数据整理和训练配方不同，可以检验视觉语言预训练的另一条路线。
- 风险：图文模型不一定优于纯视觉自监督模型；下载的完整CLIP checkpoint可能包含当前任务不需要的文本侧参数。
- 论文：Chuang et al., *Meta CLIP 2: A Worldwide Scaling Recipe*, NeurIPS 2025, DOI `10.52202/085713-1601`，arXiv `2507.22062`。
- 官方实现：`https://github.com/facebookresearch/MetaCLIP`。

### T4：Swin Transformer V2 Tiny

- 架构：分层窗口注意力Transformer；与标准ViT相比，更强调局部窗口与多尺度结构。
- 规模：Tiny级，约28M参数；与ConvNeXt-Tiny体量接近。
- 权重获取：`torchvision`直接提供，当前环境已经安装兼容版本，不需要新增`timm`或`open_clip`。
- 价值：白族纹样同时依赖局部线条和整体布局，Swin的局部窗口和层级特征是有意义的Transformer控制组。
- 局限：论文为2022年，不是最新基础模型；但工程可用性强、结构代表性好。
- 论文：Liu et al., *Swin Transformer V2: Scaling Up Capacity and Resolution*, CVPR 2022, DOI `10.1109/CVPR52688.2022.01170`。

## 第二优先级：第一轮结果好时再测

### PE-Core-B16-224

- ViT-B/16，官方表记0.09B，约90M参数，输入224。
- 官方ID：`facebook/PE-Core-B16-224`。
- 如果PE-Core-S验证表现突出，再用B版本判断扩大模型是否继续收益；不建议一开始同时下载两个规模。

### AIMv2-Large Patch14 224

- 架构：ViT-Large级视觉编码器，patch 14。
- 参数量：0.3B，明显大于本轮首选的小模型。
- 官方ID：`apple/aimv2-large-patch14-224`。
- 预训练：多模态自回归预训练；官方报告冻结trunk具有较强识别性能。
- 优点：2025年的强视觉编码器路线，且官方支持Transformers加载。
- 缺点：下载、显存和特征提取成本显著更高；在620张数据集上，大模型是否带来收益未知。
- 论文：Fini et al., *Multimodal Autoregressive Pre-training of Large Vision Encoders*, CVPR 2025, DOI `10.1109/CVPR52734.2025.00901`，arXiv `2411.14402`。
- 官方实现：`https://github.com/apple-aiml-research/ml-aim`。

### DeiT III Small

- 架构：标准小型ViT，适合作为监督预训练的纯ViT控制组。
- 优点：规模小、论文方法清晰，可以与ImageNet ConvNeXt-Tiny比较CNN和纯ViT的传统迁移特征。
- 缺点：当前环境没有安装`timm`，且模型为2022年；优先级低于可直接测试的SwinV2-Tiny。
- 论文：Touvron et al., *DeiT III: Revenge of the ViT*, ECCV 2022, DOI `10.1007/978-3-031-20053-3_30`。

## 建议测试顺序

1. DINOv3 ViT-S/16：最高信息增益，与现有DINOv3 ConvNeXt-Tiny形成近似受控对比。
2. PE-Core-S16-384：2025年、20M级、面向视觉嵌入且分辨率较高。
3. MetaCLIP 2 ViT-S/16：补充新的视觉语言预训练路线。
4. SwinV2-Tiny：低门槛的层级Transformer控制组。
5. 只有前四个显示明确潜力时，再考虑PE-B或0.3B AIMv2-L。

每个模型按相同步骤进行：单图smoke test、620张冻结特征缓存、读回一致性检查、统一线性SVM验证。选择阶段不访问测试集。

## 当前软件条件

- `transformers 4.57.3`：支持DINOv3，AIMv2官方示例也使用Transformers。
- `torchvision 0.23.0+cu128`：可直接提供SwinV2。
- 当前没有安装`timm`和`open_clip`：PE与MetaCLIP 2可能需要新增依赖，安装前先做版本与磁盘检查。

## 检索来源与限制

- Crossref公开API：核实论文题名、作者、会议、年份和DOI；访问日期2026-11-11。
- GitHub公开API：读取上述官方仓库README，核实模型规格、checkpoint名称和Hugging Face链接；访问日期2026-11-11。
- Hugging Face站点本轮连接超时，但官方GitHub README和Transformers官方文档给出了模型ID。后续实际下载前仍需进行单模型访问预检。
- OpenAlex公共IP额度本轮已耗尽并明确返回429，因此没有将OpenAlex空结果解释为无论文。
- 权重许可可能不同于代码许可；正式发布或商用前应逐个核对model card与许可证。

