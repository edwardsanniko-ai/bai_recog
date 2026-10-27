# 0004 — 按用户要求切换为可直接获取的 SigLIP 2

日期：2026-10-27（Asia/Shanghai）

用户认为 DINOv3 权重申请流程繁琐，要求换成可直接获取的模型。
当前唯一基底模型改为 google/siglip2-base-patch16-224（SigLIP 2，2025）。
采用视觉编码器提取特征，之后训练六分类线性头。尚无该模型的分类成绩，
不能预先宣称它优于 DINOv3。训练参数仍待逐项讲解与设置。

## 当次在线核验

- 官方模型 API：gated=false，private=false，标签为 Apache-2.0。
- 固定版本：75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2。
- config.json、preprocessor_config.json 匿名读取成功。
- model.safetensors 匿名 Range 请求返回 HTTP 206，实际读取 8 字节。
- 预处理配置：224×224，均值和标准差均为 [0.5, 0.5, 0.5]。

## 修改与边界

- 新增 SigLIP 2 配置和单张训练图片测试脚本；README 指向新模型。
- 保留并标记 DINOv3 配置为停用，旧记录不覆盖。
- 原图和固定 434/93/93 划分保持不变。
- 测试脚本强制匿名访问、固定版本、关闭远程自定义代码；权重缓存在本地。
- 当前尚未下载完整权重或验证真实前向；下一步由用户执行单图脚本。
- 本地已通过单图脚本 --help 检查，SiglipVisionModel 及所需依赖导入成功；git diff --check 通过。
- 单图成功后自动保存时间、配置、代码版本、图像/清单哈希、特征形状和环境版本。

来源：https://huggingface.co/google/siglip2-base-patch16-224
论文：https://arxiv.org/abs/2502.14786
