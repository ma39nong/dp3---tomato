# DP3 Epoch20 与工控机框架

本目录是 DP3 模型包：`config/deploy.json`、`checkpoints/epoch20_ema.pt`、`inference.py` 和训练一致的预处理。当前双终端适配工作树是 `/home/user/dp3-epoch20-deployment`，Git 分支 `codex/dp3-epoch20-main`，基于 `/home/user/pi05/main`。其操作说明在 `/home/user/dp3-epoch20-deployment/deploy/dp3/README.md`。

`/home/user/pi05/部署.md` 与它引用的 `deploy/fr3_wuji/README.md` 是工控机接口依据；旧 `/home/user/dp3-epoch20-dp3branch-prototype` 来自独立的 `origin/dp3` 分支，仅供比较，不能作为现场启动入口。

新工作树的 `deploy/dp3/config/preprocess.json` 与 `deploy/dp3/preprocess.py` 是本 bundle 的导出副本，供 ROS Docker 的只读 cam0 深度采集使用。服务启动时核对配置和 EMA SHA256。本机已验证 EMA 加载、点云逐点一致性、54 维状态顺序和 `[4,54]` WebSocket 返回；现场 ROS、相机和机器人仍未验证。主工作树只读影子客户端不发送命令，真实执行入口尚未完成。
