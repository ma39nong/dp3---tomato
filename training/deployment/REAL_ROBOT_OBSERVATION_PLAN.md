# Epoch 20 实机观察：prediction-only 与 rollout 记录

当前只准备和执行 Stage 1–3。**这里没有 DP3 控制 publisher；prediction-only 程序不会向 FR3 或灵巧手发送动作。** Stage 4 的运动控制、急停、Safety Gateway 接入与动作范围由现场有权限的设备负责人另行审核。离线泛化差是已知风险，不以实时输入与训练集完全同分布作为 Stage 1 的通过条件。

## Stage 1：在有 ROS2 和模型文件的现场主机上运行

已另建独立的 [`dp3_deploy/`](../../dp3_deploy/README.md) 现场适配包。它参考了同事的 π0.5 仓库，把 ROS 只读采集和 GPU 推理拆为可选的两个进程；两者通过仅限本机的 Unix socket 传递观测和预测。此包的实际模型契约仍为 **2 帧 × 4096 点、10 Hz、4×54 输出**，不能使用同事 `dp3` 分支的 4 帧 × 1024 点、20 Hz 配置。已在本机用真实 Zarr 样本离线验证模型输出及预处理一致，**未在现场 ROS 图上运行**。

建议现场先运行新包的 prediction-only 入口。若 ROS 与 PyTorch 在同一环境，可直接执行 `python dp3_deploy/ros_prediction_only.py --output /path/to/new.jsonl --save-observations`。若必须分开环境，先在模型环境运行 `python dp3_deploy/policy_server.py --socket /tmp/dp3-policy.sock --device cuda:0`，再在 ROS 环境运行 `python dp3_deploy/ros_prediction_only.py --policy-socket /tmp/dp3-policy.sock --output /path/to/new.jsonl --save-observations`。若 ROS 在容器里，须由现场人员把 socket 路径以同一位置绑定到两个环境并核对文件权限。两进程均无控制 publisher。完整依赖、参数和日志字段见新包 README。相机默认 RELIABLE QoS，但应以现场 depth publisher 的实际 QoS 为准。

原 `training/scripts/deployment_dry_run.py` 可保留作为单环境对照入口；它仍要求现场可读取训练 Zarr 和原工程。以下为其原计划命令。

从项目根目录运行；每次给 `--output` 一个新文件名。现场必须先确认使用的是审核过的机器人源码与原始录制契约，且 `cuda:0` 可用。**原 dry-run 入口的同一个 Python 解释器必须同时能导入 `torch`、`rclpy`、`sensor_msgs` 和项目依赖**；本机 miniforge 环境目前没有 `rclpy`，因此下面是现场依赖齐全时的候选命令，不是已经验证可直接运行的承诺。先在已 source ROS2/机器人工作区的终端执行：

```bash
/home/user/miniforge3/envs/dp3/bin/python -c 'import torch, rclpy, sensor_msgs; print(torch.cuda.is_available())'
```

如果导入失败，先由现场负责人配置兼容的 ROS2 + PyTorch 环境，不改 topic 名或放宽契约检查。

```bash
cd /home/user/dp3/DP3_ROSBAG_ZARR_TRAINING_BUNDLE
/home/user/miniforge3/envs/dp3/bin/python training/scripts/deployment_dry_run.py \
  --dry-run --checkpoint-choice epoch20 --device cuda:0 \
  --max-predictions 100 --max-runtime-s 180 --max-errors 20 \
  --save-observations \
  --output outputs/real_robot_observation/epoch20_site_001.jsonl
```

此程序严格检查 pinned `robot_source_contract.json` 中的六个订阅 topic/type、54 关节顺序、depth/CameraInfo frame、10 Hz 因果时间网格和动态输入的 150 ms 新鲜度。CameraInfo 是机器人录制契约中的静态标定话题，使用最近一条因果有效的内参，不要求每个 10 Hz tick 更新。一个有效预测日志条目记录：网格和源时间戳、EMA checkpoint、输入 shape、点云数量/XYZ/finite ratio、54 维 state、已反归一化的 `[4,54]` 绝对关节目标、完整 16 步采样轨迹、首步相对当前 state 的变化、推理延迟和诊断标志。`--save-observations` 另外保存模型实际使用的两帧 `[2,4096,3]` 点云、`[2,54]` state、两个网格时间戳和预测 chunk 到旁边的 `.observations/` 目录。JSONL 中 `sent_robot_command=false`、`model_action_was_executed=false`。

达到运行时长或错误上限会写入 `stopped` 事件并退出；若退出时预测数为零，应按日志定位输入/时钟/依赖问题，不把它当作模型完成测试。现场核查顺序：

1. topic/type 和 joint-name 必须匹配；不匹配时报 `LIVE CONTRACT MISMATCH`，不自行寻找相近 topic。
2. 日志有连续的 10 Hz 网格点，source header 单调且可在 150 ms 内因果选样；无长期缺帧或重复错误。
3. state、点云和预测均为有限数，输入 shape 正确；点云坐标数量级与训练样本大致相符，允许场景变化造成的分布差。
4. 连续推理、时延、预测相对当前 state 的大小与方向供人工审阅。`safety_flags` 是**离线演示范围诊断**，不是 Safety Gateway 的接受/拒绝结果。

本机没有 `rclpy`/真实 ROS topic，本次未运行 live dry-run。只有现场产生上述日志后才能填写 Stage 1 的实际结果。

## Stage 2–3：按 rollout 保存原始证据与人工阶段结果

机器人侧现有 `rosbag2` 录制链保存 `/teleop/validated_arm_commands`、`/teleop/arm_command_status`、四组 measured state、左右手 command、cam0 深度/图像等。**继续使用现有录制流程**；这里不启动第二条控制链。记录一次 rollout 时，保留原 bag 目录、prediction JSONL 与 `.observations` 快照、视频路径。现场测试之后可在 ROS2 主机上只读提取：

```bash
python3 training/scripts/extract_rollout_rosbag_events.py \
  --bag /path/to/finalized_episode_bag \
  --output /path/to/execution_events.jsonl
```

提取器只读 finalized bag，按固定 joint mapping 输出 Safety Gateway validated arm target（14 维）、左右手**观测到的 command**、四组 measured state 合成的 54 维 feedback、ArmCommandStatus 的 source/session/sequence/faults，以及 cam0 depth 时间戳。手部 command 不等于手部已执行；validated arm target 也不等于机器人已经到达该目标，须结合 measured feedback。现有状态消息没有提供每条 DP3 预测到 Safety Gateway accepted command 的唯一对应 ID，也没有明确的“动作被修正”布尔字段；这两项在有授权的控制适配器提供明确关联前标为 **未验证**，不能从时间相近自行推断。

为一次观察创建记录目录，先用 `prediction_only`；将来经过独立运动审核后，才能用 `dp3_via_safety_gateway` 并填写真实 `ArmCommand.source`：

```bash
/home/user/miniforge3/envs/dp3/bin/python training/scripts/real_rollout_report.py init \
  --session-dir outputs/real_robot_rollouts/site_session_001 \
  --rollout-id rollout_001 --checkpoint-choice epoch20 \
  --control-source prediction_only \
  --prediction-log /absolute/path/epoch20_site_001.jsonl \
  --observation-dir /absolute/path/epoch20_site_001.observations \
  --rosbag-dir /absolute/path/finalized_episode_bag \
  --execution-events /absolute/path/execution_events.jsonl \
  --video-path /absolute/path/to/camera_recording
```

工具生成 `manifest.json` 和 `stages.json`。现场人员在 `stages.json` 对七个阶段分别填写 `success`、`fail`、`human-stop` 或 `safety-stop`；尚不能判断时保持 `pending`。失败/停止必须填写原因。七阶段为接近目标、动作方向、抓取姿态、接触/抓取、交接/后续操作、放置、完整任务。预测-only 的运行不应被标成“机器人执行成功”。

## 实验后汇总

```bash
/home/user/miniforge3/envs/dp3/bin/python training/scripts/real_rollout_report.py summarize \
  --session-dir outputs/real_robot_rollouts/site_session_001 \
  --output outputs/real_robot_rollouts/site_session_001_reports
```

输出 `summary.json` 和分开的 `successful_rollouts.json`、`failed_rollouts.json`、`stopped_rollouts.json`、`pending_rollouts.json` 索引，保留每个失败案例的原始日志路径。汇总包含已标注 rollout 数、完整/逐阶段成功率、human/safety stop、点云与 prediction magnitude、时延和错误。缺失的执行事件或人工标注以 `null`/`pending` 呈现，绝不把“没有证据”计成零次拒绝或一次成功。只有 `control_source=dp3_via_safety_gateway` 且来源身份一致的 status 才用于统计模型源的 safety faults；即使存在 validated action/feedback，当前也不宣称已逐条证实 DP3 预测被执行。Epoch 7 留作后续相同场景对照，第一批仅用 Epoch 20。
