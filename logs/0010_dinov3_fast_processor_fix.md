# 0010 — 修复 DINOv3 预处理器选择错误

日期：2026-10-30（Asia/Shanghai）

用户运行单图脚本时成功下载官方 preprocessor_config.json，随后报错：
`DINOv3ViTImageProcessorFast does not have a slow version`。
原因是脚本错误设置 use_fast=False，并非此次登录失败。
0009 中关于采用慢速预处理器的方案在这里修正，旧记录保留。

读取固定版本的本地官方预处理配置，确认处理器类型为
DINOv3ViTImageProcessorFast，输入尺寸 224×224，均值
[0.485, 0.456, 0.406]，标准差 [0.229, 0.224, 0.225]。
ConvNeXt 模型使用这个名字的处理器是官方配置，不代表切换了模型架构。

修复：单图脚本设置 use_fast=True，并同步修改输出报告中的该字段。
继续使用官方其他参数，不改变模型、图片或数据划分。
验证仅加载已缓存的预处理配置并处理同一训练图片；
完整模型下载和前向尚待用户重试。

本地验证通过：处理器类型 DINOv3ViTImageProcessorFast，输出
(1, 3, 224, 224)，全部数值有效；验证过程没有联网或加载模型权重。
