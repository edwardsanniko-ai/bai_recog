# 三模型逐图错误视觉审计

日期：2026-11-06（America/New_York）

## 目的与边界

本次审计只检查固定验证集，不评估测试集，也不更新任何模型参数。比较对象为冻结的 DINOv3 ConvNeXt-Tiny（D）、torchvision ResNet50（R）和 torchvision ConvNeXt-Tiny（C）。三者都使用 434 张训练图作为带标签参考、93 张验证图作为查询、L2 归一化后的余弦 1NN；平局时取最小全局参考编号。

脚本：`scripts/analyze_top3_errors.py`  
逐图审计表：`outputs/top3_error_audit/error_audit.csv`  
联系图目录：`outputs/top3_error_audit/`（仅本地派生缩略图，已被项目排除，不上传原图或派生图像）

## 总体统计

| D/R/C 正确模式 | 含义 | 数量 |
|---|---|---:|
| 111 | 三者都正确 | 65 |
| 011 | 仅 DINOv3 错 | 7 |
| 100 | 仅 ResNet50 与 ConvNeXt-T 错 | 6 |
| 000 | 三者都错 | 5 |
| 101 | 仅 ResNet50 错 | 4 |
| 010 | 仅 ResNet50 正确 | 3 |
| 110 | 仅 ConvNeXt-T 错 | 2 |
| 001 | 仅 ConvNeXt-T 正确 | 1 |

因此，至少一个模型错误的验证图为 28/93；三者共同正确的图为 65/93。这里的“共同”只描述这次固定验证划分，不代表模型在更大数据集上具有相同的错误模式。

## 三者都错的 5 张

| 查询编号 | 真实类别 | DINOv3 | ResNet50 | ConvNeXt-T | 联系图中的观察 |
|---:|---|---|---|---|---|
| 134 | animal | combined | plant | plant | 细长横向框饰、主体较小；参考图分别偏向植物边饰或组合边饰 |
| 363 | plant | combined | combined | combined | 黑白高对比的横向植物/花饰条带，和组合纹边饰外观相近 |
| 584 | combined | animal | animal | animal | 组合图中主体是动物，三个最近邻都选到了单独动物纹 |
| 516 | combined | antique object | antique object | antique object | 组合图由器物和边框构成，最近邻抓住了器物/框形而忽略“combined”标签 |
| 410 | plant | animal | combined | combined | 竖向窄条、留白比例高；三个最近邻的类别都受到局部轮廓或版式影响 |

这 5 张说明：组合纹并不一定具有独立、统一的外观；当查询图只显示一个局部元素或边框时，1NN 很容易按视觉近邻而不是数据集标签语义归类。

## DINOv3 独有错误（011，7 张）

ResNet50 与 ConvNeXt-T 在这 7 张上均正确。联系图中可见的固定划分观察包括：

- #466（Chinese character）是稀疏、小尺寸的多个字符/印章样式元素，DINOv3 选到组合纹横条；
- #51、#14（geometric）是重复波纹或窄边框，DINOv3 选到植物卷草类局部；
- #374、#362、#344（plant）是横向植物边饰，DINOv3 的最近邻分别偏向组合或动物边饰；
- #517（combined）含多个动物/花饰元素，DINOv3 选到单独动物纹。

可能原因是查询图的细线、留白和横向布局在 DINOv3 的整图向量中被强调，而类别所依赖的局部语义没有被最近邻规则充分利用。这是视觉解释假设，不是因果结论。

## DINOv3 正确、两个 ImageNet 控制组错误（100，6 张）

这组包括 #289（antique object）、#503（Chinese character）、#57（geometric）、#412（plant）、#178（animal）和 #532（combined）。联系图中 DINOv3 的参考图在轮廓或整体结构上更接近查询图，而两个控制组有时被局部花纹、动物形状或几何边框吸引。例如 #532 的两个控制组分别预测 animal 和 plant，DINOv3 选到 combined 参考图。

这可以作为“DINOv3 在本划分上提供了不同的近邻信息”的证据，但不能据此宣称 DINOv3 普遍优于监督式控制组；总体 Macro-F1 差距很小，尚未做不确定性检验。

## 其他互补错误

- #205、#415、#600、#332（101）：DINOv3 与 ConvNeXt-T 正确，ResNet50 错；错误多为将动物/汉字/组合/植物局部映射到相邻纹样类别。
- #100、#147（110）：DINOv3 与 ResNet50 正确，ConvNeXt-T 分别误判为 combined 和 geometric。
- #127、#590、#519（010）：仅 ResNet50 正确；说明控制组之间也存在互补信息。
- #280（001）：仅 ConvNeXt-T 正确；DINOv3 和 ResNet50 分别选到植物、动物参考图。

## 解释限制

1. 余弦相似度是特征空间的相似度，不是类别概率；错误最近邻的相似度有时高于正确最近邻，不能把它直接当作置信度。
2. 本审计依赖固定的 434/93 划分、官方预处理和 k=1 判别规则；改变裁剪、参考集或 k 值，错误图可能改变。
3. 联系图用于人工核查，不是新的训练数据；原图、缓存和派生缩略图均不纳入 GitHub 提交。

