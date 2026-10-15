# DP3 实机部署静态审核（仅申请 Stage 1–3）

审核基线：机器人/原始数据仓库 `ma39nong/Gello_armhand_teleop12` commit `03da40134b97332f6e2d9ad1162dfe287ce28d4e`；模型/最终数据以本工程的 Epoch 20 checkpoint 与 `AB_committed_4096.zarr` 为准。**本方案不会向 FR3 或灵巧手发送 DP3 控制命令。** 本轮没有连接机器人，也没有运行 ROS2 live 验证。后续现场观察到的仓库版本、topic 或类型不同，必须先停止在 prediction-only 阶段并报 `LIVE CONTRACT MISMATCH`。

## 1. 双侧架构与数据契约

机器人侧：Gello/Pico/Vive 等源发布 `teleop_interfaces/msg/ArmCommand` 至 `/teleop/arm_commands`；Safety Gateway 读取双侧实测关节和外力矩，发布 `/teleop/validated_arm_commands` 与唯一的遥操 FR3 硬件命令总线 `/target_robot/joint_commands`；`joint_splitter` 把该总线按左右臂拆成 `/{side}/gello/joint_states`，关节名改为 `fr3_joint1..7`，再由 joint impedance controller / `franka_hardware` / FCI 执行。Wuji 的 command 与 measured state 另有自己的链；本阶段不触及该链的任何写接口。来源：机器人仓库 `ros_ws/src/teleop_core/teleop_core/{contract,safety_gateway,joint_splitter}.py`、`ros_ws/src/franka_fr3_arm_controllers/{config/controllers.yaml,src/joint_impedance_controller.cpp}`。

原始 Harvest 格式以机器人侧 `data_collection/docs/DATA_FORMAT.md`、`data_collection/contract/dataset_contract.yaml` 和 `teleop_core.contract` 为准。它的 `observation.state[108]` 为左臂位置 0–6、左臂速度 7–13、右臂位置 14–20、右臂速度 21–27、左手位置 28–47、左手速度 48–67、右手位置 68–87、右手速度 88–107；`action[54]` 为左臂 0–6、右臂 7–13、左手 14–33、右手 34–53。所有关节量单位 rad；速度块为 rad/s。`observation.engaged[4]` 仅用于采集/转换状态审计，不进入本模型输入。

**DP3 最终 Zarr 不是直接读取 108 维 Harvest feature。** 本工程 `conversion/src/convest/sources/bimanual_rosbag2.py::read_dp3_streams` 直接从 ROS bag 读取四组 measured JointState 的 position/velocity，以及 validated 双臂 action 和左右手 command；`conversion/src/convest/align.py::align` 在 10 Hz 时间轴上因果采样/保持；`conversion/src/convest/formats/dp3_zarr.py::_vectors` 从四组 state 中只取 position，删除四组 velocity 共 54 维，并把原始左右臂、左右手顺序重排成 **左臂 7、左手 20、右臂 7、右手 20**。action 同样重排。最后由 `training/scripts/bimanual_zarr.py` 裁剪并生成最终 Zarr。这里的 raw Harvest 索引是与直接 ROS→DP3 的语义对应关系，不表示程序先写出 108 维 Harvest tensor 再切片。

逐维机器可读映射：[training/deployment/ros_to_dp3_joint_mapping.csv](training/deployment/ros_to_dp3_joint_mapping.csv)。其每行包含 ROS topic 和源关节名、Harvest 108/54 索引、最终 `data/state`/`data/action` 索引、单位和实时来源。最终 DP3 两个 54 维数组的区间是 0–6 左臂、7–26 左手、27–33 右臂、34–53 右手。双臂 action 消息使用 `left/right_fr3v2_joint1..7`，measured state 使用 `left/right_fr3_joint1..7`；映射由机器人侧 `dataset_arm_joint_name()` 对正式 joint tuple 按索引完成，并已经逐维与最终 Zarr 的 `state_names`、`action_names` 比对，不使用字符串相似匹配。Wuji 20 关节顺序由 `WUJI_LEFT_JOINT_NAMES` / `WUJI_RIGHT_JOINT_NAMES` 确定。没有代码中的符号翻转或关节位置单位转换；**PENDING LIVE VERIFICATION**：实际机器人固件关节方向、零位和实时消息值需现场对照遥操读数。

## 2. ROS topic/type 与时序

| 用途 | 正式 topic | ROS type |
|---|---|---|
| 双臂 validated action | `/teleop/validated_arm_commands` | `sensor_msgs/msg/JointState` |
| 左、右 FR3 measured state | `/left/franka/joint_states`、`/right/franka/joint_states` | `sensor_msgs/msg/JointState` |
| 左、右手 command | `/teleop/wuji/left/command`、`/teleop/wuji/right/command` | `sensor_msgs/msg/JointState` |
| 左、右手 measured state | `/teleop/wuji/left/joint_states`、`/teleop/wuji/right/joint_states` | `sensor_msgs/msg/JointState` |
| cam0 depth | `/cam0/depth/image_raw` | `sensor_msgs/msg/Image` |
| cam0 depth intrinsics | `/cam0/depth/camera_info` | `sensor_msgs/msg/CameraInfo` |

前五类见机器人 `dataset_contract.yaml`；depth camera_info 见机器人 `DATA_FORMAT.md` 与 `data_collection/config/record_gello.yaml` 的 `static_topics`，它在原始录制契约中为**可选**，但对当前 DP3 点云构造是**必需**。本工程 `conversion/schemas/harvest_tomato_A_contract.yaml` 将它列为必需 topic。`training/deployment/robot_source_contract.json` 固定了本次审核所用的具体 topic/type/joint 列表与机器人 Git commit。现场不进行模糊 topic 自动选择。

原始 Harvest dataset 行时间轴为 30 Hz，cam0 原生约 20 Hz。DP3 Zarr `target_hz=10`：转换器以每个 bag 可用源数据交集的起点为相位建立 100 ms 网格，对每个状态、深度和 CameraInfo 取 **source header 时间 ≤ 网格时间**的最新样本，允许最大陈旧 150 ms；无效时间片不连接成 episode。每个最终 Zarr frame 是这个 10 Hz 网格上的样本，而非相机原生 20 Hz 帧。对连续时间片 `t-0.1 s`、`t`，模型输入 `[2,4096,3]` 与 `[2,54]`；`n_obs_steps=2`、`horizon=16`、`n_action_steps=4`，`DP3.predict_action` 从长度 16 的反归一化结果中取索引 1:5，故四个 action 对应 `t`、`t+0.1`、`t+0.2`、`t+0.3 s` 的**绝对位置目标**。episode 首尾可能由 `SequenceSampler` 复制边界帧；live 首次推理必须等待两个连续 10 Hz 网格点。

DP3 配置的 `action_policy=sample_hold` 使 active group 的 action 取此前最新 command，代码没有对该 command 单独执行 150 ms 陈旧上限；状态和 status 仍需新鲜。disengaged 时 action 锁存状态切换时的最后有效 target，若无 command 则以 measured position 初始化。此点与原始 Harvest 30 Hz 导出中的 active-action 新鲜度规则不同，应视为本模型标签的实际语义，而不能在 live 端擅自改写目标定义。

prediction-only 适配器已改为缓存带 source header 的消息、在 10 Hz 网格上因果 as-of 选择、拒绝超 150 ms 的数据，并要求两个输入点连续相隔 100 ms。它不再取相机原生 20 Hz 的“最新两帧”。**PENDING LIVE VERIFICATION**：机器人 ROS clock 与各 source header 是否同域；实时网格与 bag 起始相位未必完全一致，需比较时间戳/统计分布；多 topic 消息到达抖动可能导致跳帧，此时不预测。

## 3. 深度 → 点云 → 模型

机器人 raw16 depth 由 `/cam0/depth/image_raw` 提供；当前转换器也能读 `32FC1`。`16UC1` 数值乘 `0.001 m`，`32FC1` 已按米；有效深度 0.05–3.0 m。CameraInfo `K` 投影得到 `(u-cx)z/fx,(v-cy)z/fy,z`，像素按 `pixel_stride=2` 取样。非零畸变系数会报错；没有现场去畸变。转换矩阵是 identity，最终点位于 `cam0_depth_optical_frame`，**没有**变换到 FR3 base 或统一工作台坐标。不同批次相机俯仰变化因此仍属于训练分布风险。

最终 Zarr 的裁剪 AABB：`X[-1.2,1.2] m`、`Y[-0.6,0.3] m`、`Z[0.3,1.45] m`。另有两条穿过相机原点的半空间平面，名称分别是 `keep_right_of_image_column_40` 与 `keep_below_image_row_40`；“40/40”指最初按约第 40 列/行定义的边缘裁剪试验，不是 40 个点，也不是长度 40 cm。真正执行依据为 Zarr `clip_planes` 的单位法向：`[0.7482601,0,0.6634055]` 和 `[0,0.8869718,0.4618236]`，均保留 `normal·XYZ >= 0`。裁剪后以 0.005 m 体素降采样；不足 4096 个体素候选时退回裁剪点；再以 FPS 取固定 4096 点，不足 4096 个裁剪点直接拒绝。最终数组是 `(74340,4096,3)` 的 float32 XYZ。

`GenericZarrDataset.get_normalizer()` 仅以 train episode 拟合 `action`、`agent_pos`（即 Zarr state）、`point_cloud`。Epoch 20 EMA state_dict 内含这些 normalizer buffer。实时输入只做几何/采样，不提前手工归一化；`policy.predict_action()` 内部归一化 observation 并反归一化 action。离线 64 帧抽样参考：训练 XYZ 均值约 `[0.0663,-0.0331,0.9000] m`、标准差 `[0.5048,0.2306,0.2199] m`；验证均值约 `[0.0637,-0.0173,0.8789] m`、标准差 `[0.4874,0.2234,0.1937] m`。**PENDING LIVE VERIFICATION**：现场每次记录 `(4096,3)`、裁剪前后实际点数、XYZ min/max/mean/std 和 finite ratio，与训练抽样比较。

## 4. Epoch 20 推理契约

主候选：`training/data/outputs/harvest-tomato-ab-pre722-40-40-300-from-scratch-4096/checkpoints/epoch=0020.ckpt`；对照：`epoch=0007.ckpt`。`training/scripts/deployment_dry_run.py` 的 `--checkpoint-choice epoch20|epoch7` 用 checkpoint `cfg` 实例化 policy，以 `ema_model` 严格加载权重和 normalizer，设 `cuda:0`、float32、`eval()`，在 `torch.inference_mode()` 中执行。输入 batch 为 point cloud `[1,2,4096,3]`、agent_pos `[1,2,54]`；输出 `[1,4,54]`，单位 rad，已反归一化，含义是 absolute joint target。离线加载已完成；**PENDING LIVE VERIFICATION**：现场 RTX 5090/CUDA 可用性、依赖版本和端到端延迟。预测值目前只记录，**不构造任何 ArmCommand 或 Wuji command publisher**。

## 5. 已知离线风险：不得据此开放 Stage 4

Epoch 20 EMA 全验证集 7470 个序列：DP3 MAE/RMSE `0.05538/0.08874 rad`，hold-current-state 基线 `0.02294/0.05438 rad`。low-motion 2121 序列的 RMSE 为 `0.09052` vs `0.04556`；moving 5349 序列为 `0.08803` vs `0.05751`；强运动上四分位 1548 序列为 `0.09293` vs `0.07543`。分组依据是四步 GT absolute action 相邻差的 RMS，low-motion 阈值来自训练序列的 25% 分位。A/B 验证的 DP3 RMSE 分别 `0.07607/0.09092`，hold 分别 `0.06421/0.05234`。这些是离线 action 回归误差，不是实机任务成功率。

“第一步 0.0574 rad”精确定义：对每个验证序列的第一个未归一化预测 action 与第二个 observation 的 measured state 做逐关节绝对差，然后在 **7470×54** 个数上取均值；不是 54 维欧氏范数。GT action 与同一 state 的对应均值为 `0.01931 rad`。按组（预测 / GT，rad）：左臂 `0.08376/0.00649`；左手 `0.04303/0.01409`；右臂 `0.06884/0.00304`；右手 `0.05850/0.03472`。贡献最大的预测偏移是右臂关节 5 `0.14891`、左臂关节 7 `0.13714`、左臂关节 5 `0.12646`、右手食指 PIP `0.12229`、右手无名指 PIP `0.08561 rad`（均为该维跨序列平均绝对差）。99.88% 的验证序列至少一侧臂某关节的第一目标与当前状态相差超过 0.05 rad；相同样本中 GT 为 15.46%。这是**离线幅值诊断**；Gateway 的 0.05 rad 只在某侧首次激活/重新激活时检查，不是逐周期约束。详见 [第一步对照复核](DP3_STAGE4_INITIAL_DELTA_ANALYSIS.md)。

静态核对显示 checkpoint 的 EMA normalizer、54 维关节顺序和 `action[1:5]` horizon 均与离线评估/当前 dry-run 一致，没有发现能解释上述偏移的明显索引或归一化错误。**PENDING LIVE VERIFICATION**：相机角度/内参、传感器时钟、手部反馈方向和动作语义在真实机器上是否一致。Stage 3 必须逐关节对照真实 measured state 与 prediction，确认数值和时间稳定性后，另行审核 Stage 4；当前不得下发模型 action。

另外，`DP3.conditional_sample()` 每次以 `torch.randn` 初始化扩散轨迹；离线 replay 固定了每个 batch 的随机种子以便复现，当前 live dry-run 不把随机采样视为确定性控制输出。Stage 3 应记录相同或近似静止 observation 下的重复预测波动，分辨采样随机性与传感器/策略抖动。

## 6. 安全链与 Stage Gate

机器人配置 `config/modes/teleop_control.yaml`：Safety Gateway measured state timeout `0.25 s`、source command timeout `0.25 s`、最大关节速度 `0.7 rad/s`、首次目标与 measured state 差上限 `0.05 rad`、nominal dt `0.01 s`；FR3 joint limits 在 `teleop_core/safety.py`；external torque 门限 J1–4 `6 Nm`、J5–7 `3 Nm`。SourceArbiter 要求允许的 source、session ID、严格递增 sequence，并在旧源超时后才允许切换。外力矩缺失/陈旧时 **contact torque gating 会警告并失效开放**，不能声称此门始终有效。下游 joint impedance controller 独立 command watchdog `0.25 s`，超时进入测量位置 hold。真实手也需要单独确认自己的安全链。未来若申请运动测试，必须复用 Safety Gateway 的 ArmCommand 路径和官方/遥操安全限制；不能直接写 `/target_robot/joint_commands` 或 `/{side}/gello/joint_states`。本次没有实现 publisher。

| Stage | 审核范围 | 通行条件 |
|---|---|---|
| 0 Offline replay | 已完成 | Epoch 20/7 可加载，GT/DP3/hold 轨迹和风险已记录 |
| 1 实机 prediction-only | 本次申请 | 只订阅、预测、记录；graph/topic/type/source stamp 不符立即报 `LIVE CONTRACT MISMATCH` |
| 2 实时输入一致性 | 本次申请 | 10 Hz 双帧历史、54 维顺序、深度/内参/裁剪点云与训练样本对照 |
| 3 action 语义/幅值/稳定性 | 本次申请 | 逐关节 absolute target、首步 delta、跨步抖动、推理耗时和异常旗标人工复核 |
| 4 受控运动测试 | **不在本次申请内** | 单独现场授权及控制/停机方案审核后才考虑 |

## 7. prediction-only dry-run 与现场核对

现有入口 `training/scripts/deployment_dry_run.py` 无 ROS publisher、action client 或控制接口。启动示例（只在有 ROS2 和完整模型/Zarr 的只读测试主机上运行；输出选新文件）：

```bash
cd /home/user/dp3/DP3_ROSBAG_ZARR_TRAINING_BUNDLE
/home/user/miniforge3/envs/dp3/bin/python training/scripts/deployment_dry_run.py --dry-run --checkpoint-choice epoch20 --device cuda:0 --output /tmp/dp3_epoch20_prediction_only.jsonl --max-predictions 100
```

现场预先核对：机器人源码 HEAD 是否为本审核 commit；六个订阅 topic 的存在性和 ROS type；joint name 正式集合及顺序；source header 与 ROS clock 同域、单调且新鲜；cam0 depth encoding/CameraInfo frame/K/畸变；每 100 ms 因果样本和点云统计；RTX 5090 显存/延迟；模型第一步与 state 的 54 维差、手臂/手分组及异常旗标。`robot_source_contract.json` 固定话题，不会自动寻找相近 topic；实际配置不同必须停止并重新审核。若运行主机没有 ROS2 包，程序只会报缺依赖，不会尝试机器人动作。

日志每条包含网格与源时间戳、checkpoint、EMA 标识、输入 shape、点云统计、state、归一化数值重构、已反归一化的四步 action、全 horizon、第一目标与当前 state 差、推理延迟、异常旗标和 `sent_robot_command=false`。其中 `normalized_action_reconstructed` 是把已反归一化结果再经 normalizer 映射得到，**不是扩散网络的原始噪声张量**。当前仅记录预测及对示教范围的诊断，不对输出做控制限幅。

## 8. 尚未解决的现场项

- **PENDING LIVE VERIFICATION**：真实机器运行的 Git commit、ROS graph/type、状态消息 joint names、depth CameraInfo 发布频率以及相机实际序列号/俯仰角。
- **PENDING LIVE VERIFICATION**：ROS clock 和 source header 同域；100 ms 在线网格相位与离线 bag 起点不完全相同所造成的观测分布偏差。
- **PENDING LIVE VERIFICATION**：真实机器人输入点云与训练统计、手部零位/符号、推理延迟和缺帧率。
- **PENDING LIVE VERIFICATION**：手部官方运动/停止限制以及未来 Stage 4 的人工 emergency stop 路径；本方案不申请手或 FR3 运动。
- **阻塞 Stage 4**：当前离线 DP3 在 moving subset 仍劣于 hold，第一步臂关节目标相对 GT/实测状态存在广泛偏移；0.05 rad 是 Gateway 首次激活检查，不是逐周期限制。必须先完成 Stage 1–3 诊断并另行申请受控测试。

## 9. 可复核源码

- 机器人侧：[DATA_FORMAT.md](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/data_collection/docs/DATA_FORMAT.md)、[dataset_contract.yaml](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/data_collection/contract/dataset_contract.yaml)、[teleop_core.contract](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/ros_ws/src/teleop_core/teleop_core/contract.py)、[Safety Gateway](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/ros_ws/src/teleop_core/teleop_core/safety_gateway.py)、[safety.py](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/ros_ws/src/teleop_core/teleop_core/safety.py)、[joint_splitter.py](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/ros_ws/src/teleop_core/teleop_core/joint_splitter.py)、[controller watchdog](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/ros_ws/src/franka_fr3_arm_controllers/src/joint_impedance_controller.cpp)。
- 本地模型侧：`conversion/src/convest/{sources/bimanual_rosbag2.py,align.py,formats/dp3_zarr.py}`、`training/scripts/bimanual_zarr.py`、`training/3D-Diffusion-Policy/diffusion_policy_3d/{dataset/generic_zarr_dataset.py,policy/dp3.py}`、`training/scripts/{deployment_offline_replay.py,deployment_dry_run.py}`、最终 Zarr attrs 和 Epoch 20 checkpoint。
