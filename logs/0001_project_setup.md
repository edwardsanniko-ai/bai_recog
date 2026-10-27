# 0001 — DINOv3 项目初始化

日期：2026-10-27

## 目的

暂停旧的 Qwen3-VL 论文复现实验，建立一个独立的白族图案六分类项目。
第一阶段只使用 DINOv3 ConvNeXt-Tiny，先记录冻结视觉编码器加线性分类头的
baseline，再逐步调参。

## 已确认的数据协议

- 总数：620 张
- 类别：6 类
- 划分：434 train / 93 validation / 93 test
- 划分之间无重复
- 本仓库不上传原始图片

## 本次操作

- GitHub 仓库：`https://github.com/edwardsanniko-ai/bai_recog.git`
- 仓库状态：初始为空仓库，已克隆到本地 `bai_recog/`
- 模型：`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- 已创建：配置、数据划分清单生成脚本、单图 smoke test、`.gitignore`
- 尚未做：模型权重下载、训练、测试集评估

## 下一步

1. 确认 DINOv3 权重访问权限并完成本地登录；
2. 运行 `python scripts/make_manifest.py` 生成不含绝对路径的划分清单；
3. 运行单图 smoke test；
4. 通过 smoke test 后再实现全量特征缓存和冻结线性分类 baseline。
