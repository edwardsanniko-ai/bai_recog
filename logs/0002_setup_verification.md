# 0002 — 初始化核验与待办

日期：2026-10-27（Asia/Shanghai）

本记录补充并更新 0001 的状态。

- 实际生成 data/splits.json：620 条相对路径、标签和原图 SHA256。
- 验证 620 个本地原图文件均存在，标签对应类别文件夹。
- 验证三个划分共 620 个互异索引，数量为 434 / 93 / 93。
- 跨划分字节完全相同的图片组数：0。尚未做近似重复审计。
- 模型固定为 facebook/dinov3-convnext-tiny-pretrain-lvd1689m，
  官方模型版本为 10d30274b4d445111e2d5bf75ac93bbd94db274b。
- transformers 4.57.3 中的 DINOv3ConvNextModel 实际导入成功；CUDA 可用。
- 初始配置中的训练数值已改为 null，留待逐项教学和用户调参，未运行训练。
- 单图脚本已准备，命令行和导入已检查；权重未下载，真实前向尚未验证。
- 官方模型 API 显示 gated=manual；匿名读取配置返回 HTTP 401，需申请模型权限。
- GitHub 仓库已克隆，本地已提交；推送未完成。禁止交互后捕获到明确错误：
  `Cannot prompt because user interactivity has been disabled` 和
  `unable to get password from user`。需要用户在自己的终端完成 GitHub 身份认证。
- 不含原图、权重或访问令牌。使用 main 分支。旧实验目录未修改。

官方来源：
- https://github.com/facebookresearch/dinov3
- https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m

下一小步：用户在自己的终端执行 git push -u origin main，完成登录和首次上传。
之后再申请/核实 DINOv3 权重访问权限，运行单图测试。
