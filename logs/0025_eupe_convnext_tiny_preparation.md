# 0025 — EUPE ConvNeXt-Tiny 选型及单图检查准备

日期：2026-11-01（Asia/Shanghai）。

用户指定 https://github.com/facebookresearch/EUPE，随后明确选择 ConvNeXt-Tiny。
维持冻结编码器、无分类器训练、无参数更新、相同 434/93/93 划分及余弦 1NN 规则。
目前只准备单图检查，不计算验证/测试成绩，不选定最终 baseline。

官方身份：Efficient Universal Perception Encoder，Meta，2026，论文 https://arxiv.org/abs/2603.22387。
官方 README 提供 ViT-T/S/B 与 ConvNeXt-T/S/B，ConvNeXt-T 标称约 29M 参数。
官方公开模型 facebook/EUPE-ConvNeXt-T，gated=false，固定 HF revision
d500fe603e80e843ff492f9f1dad77c3158a84b9。
文件 EUPE-ConvNeXt-T.pt，168009287 字节，HF LFS SHA256
72f37eacd46cb7b4378ebcdb17de8710adbe5e8ae3c04448654a0d1b1c912b45。
权重已下载到本地忽略目录 hf_cache/，没有上传。

官方源码 commit f6cdf04d82b97009941a27e9bed3ca17ac7e0366。
原样保留 eupe/models/convnext.py 与完整许可证，来源记录在 THIRD_PARTY.md。
源码和模型使用 FAIR Noncommercial Research License。
Git attributes 保持这两份上游文件的原始字节，防止 Windows 换行转换影响校验。
通过官方 GitHub API 读取 README、hub/backbones.py 与 models/convnext.py，
并用 HF 模型卡与元数据交叉核对。GitHub raw 域名失败时改用官方 API。

实现：直接构造官方 ConvNeXt Tiny，depths=[3,3,9,3]、dims=[96,192,384,768]，
与官方 hub 构造一致。官方 hub 使用 strict=False；本地包装只允许忽略
checkpoint 里已核对的 8 个 projectors.heads.dinov3radio_vitg16 蒸馏投影张量，
对剩余全部 backbone 参数使用 strict=True，并核对共享 LayerNorm 两个名称的值相等。
实际 backbone 参数 27820128，所有参数冻结，eval + inference_mode，无优化器。
使用官方 README 的 ToImage -> Resize((256,256), bilinear, antialias=True) ->
ToDtype(float32,scale=True) -> ImageNet Normalize。
提取 forward_features 的 x_norm_clstoken（全局平均池化后 LayerNorm）作为 768 维整图向量。
采用 float32，关闭 TF32；HF 示例的 bf16 autocast 不是本次推理精度。

开发验证：官方权重 SHA256、完整 backbone 加载、Windows CUDA 推理通过。
仅使用合成灰色图像，输入 (1,3,256,256)、输出 (1,768)，全部有限、非零、参数冻结。
合成单张与两张批处理的最大特征差 2.384185791015625e-06，容差检查通过。
尚未运行真实图片 smoke 脚本；下一步由用户运行 scripts/smoke_test_eupe.py 并回传。
单图报告记录版本、代码/权重/图像哈希、被排除的投影层名称、显存与环境，确保可追溯。
