# 0007 — Hugging Face 登录接口返回 HTTP 400

日期：2026-10-30（Asia/Shanghai）

用户在隐藏输入提示中提交令牌，选择不写入 Git credentials 后，
登录流程访问 https://huggingface.co/api/whoami-v2 返回 HTTP 400 Bad Request。
尚未进入模型下载，不能据此断言令牌无效或 DINOv3 审批未通过。

本地版本：huggingface_hub 0.36.2、requests 2.34.2、urllib3 2.8.0。
暂不升级依赖或更换模型。

准备 scripts/login_huggingface.py：隐藏输入、去除首尾空白、检查基本格式，
再显式传入 login(token=..., add_to_git_credential=False)，避免再次询问 Git 凭据。
失败只输出状态码和错误类型，不转储令牌、请求头或完整异常。
登录成功后的凭据由 Hugging Face 库按其默认机制保存在用户本机缓存，
不写入项目或实验日志。用户尚未运行新脚本，问题尚未确认解决。
