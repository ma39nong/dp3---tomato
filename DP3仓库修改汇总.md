# DP3 仓库修改汇总

更新时间：2026-10-15

## 一、工作区整体情况

基线提交是 `521203e`（`franka-wuji`，与 `origin/main` 当前一致）。原始工作区显示 71 条修改；加上本汇总文档和提交清单后，当前显示 73 条：

- 8 个已有文件被修改，主要是数据集、转换器、训练入口和评估器。
- 63 个新文件，主要是 300 epoch 实验、部署包、离线回放、实机只读验证脚本和审核文档。
- 当前没有提交这些修改；它们都仍在工作区。

这 71 条不是 71 次模型结构修改。没有证据表明 DP3 网络主体被重新设计；主要工作是把数据、训练、评估和部署链路补齐并留下记录。

## 二、实际完成的训练工作

### 数据与转换

- 固定 A/B 数据的 40/40 点云裁剪配置：坐标系是 `cam0_depth_optical_frame`，AABB 为 `X[-1.2,1.2]、Y[-0.6,0.3]、Z[0.3,1.45]`，再做平面裁剪、0.005 m 体素降采样和 FPS 采样到 4096 点。
- 增加 A/B Zarr 合并工具和排除清单支持：`training/scripts/merge_bimanual_zarr.py`、`dp3.sh merge`。
- 转换器增加 episode 上限校验、部分转换数据只取已提交 episode、来源 episode 名称和排除信息写入 attrs，避免把半写入的下一段 bag 送进训练。
- 新增 crop 配置和可视化/点云检查脚本，便于复核裁剪结果。

### 300 epoch 实验

- 新增 `harvest_tomato_300.yaml` 和 `harvest_300.py`，从头训练 300 epoch，使用 EMA、固定 seed 42、batch size 32、点云噪声 0.002 m。
- 使用两阶段 cosine 学习率：warmup 500 steps，前段结束 epoch 50，中间学习率 `1e-5`，最终到 `1e-6`。
- 训练和验证 normalizer 只使用训练 episode 的 action、state 和 point cloud；验证关闭点云噪声。
- 增加固定 A/B bag 验证、每 epoch 指标、梯度/参数诊断、checkpoint 注册和恢复支持。旧的 50 epoch 结果未被覆盖。
- 训练入口增加 `experiment300` 和 `experiment300-resume` 模式。

训练产物的部署主候选是 **Epoch 20 EMA**，Epoch 7 作为对照；两者均为 format-v2 checkpoint，包含 EMA、normalizer、optimizer 和 scheduler。

## 三、离线评估与问题定位

- 评估器兼容旧 checkpoint 和 300 epoch 的顶层 `ema_model` 格式。
- 增加按 horizon、来源 A/B、单个 bag 的误差统计，以及 train-mean 和 hold-current-state 基线。
- 新增 offline replay 和 prediction bias 诊断，核对 normalizer、54 维顺序、action horizon 和 diffusion 随机性。
- 结论：Epoch 20 在 validation 上确实存在泛化误差，不能归因于重复反归一化、维度整体错位或单次 diffusion seed。

关键离线数字：

- 全 validation 7470 条序列、54 维四步 action：DP3 MAE/RMSE 为 `0.05538/0.08874 rad`；hold 基线为 `0.02294/0.05438 rad`。
- moving 子集 DP3 RMSE `0.08803`，hold 为 `0.05751`。
- 随机 64 条 train/validation 序列的首步臂部 MAE 约 `0.01059/0.07592 rad`，说明主要风险是 bag 级 validation 泛化不足。
- 首步预测相对当前 state 的平均偏差约 `0.0574 rad`；这是离线诊断，不是实机成功率，也不是逐周期安全限制。

## 四、部署包做了什么

新增 `dp3_deploy/`，把训练工程中部署所需的部分集中起来：

- `inference.py`：加载 Epoch 20 EMA、normalizer、模型并输出 `[4,54]` absolute joint target。
- `preprocess.py`：深度转 XYZ、裁剪、体素/FPS 采样、54 维状态拼接和两帧历史。
- `deploy.py`、`policy_server.py`、`policy_socket.py`：离线输入和本机预测服务接口。
- `ros_prediction_only.py`：ROS 只读观察适配器，记录预测但不发布机器人命令。
- `config/deploy.json`、joint mapping 和 robot source contract：固定输入 shape、话题、关节顺序、点云参数和 checkpoint 校验。
- `epoch20_ema.pt`：部署所需的 EMA 权重副本。

部署契约是：两帧、10 Hz、点云 `[2,4096,3]`、状态 `[2,54]`，输出 `[4,54]`。输入只做几何预处理，模型内部完成 observation normalize 和 action inverse-normalize。

## 五、实机验证做到哪里

已完成：

- prediction-only 代码、ROS bag 事件提取器和 rollout 汇总工具。
- 日志包含输入 shape、点云统计、状态、预测 action、首步 delta、延迟和 `sent_robot_command=false`。
- 语法检查、关节映射小测试、合成 rollout 汇总测试通过。

尚未完成：

- 本机没有 `rclpy`，没有完成真实 ROS graph、topic/type、CameraInfo、时钟和传感器输入验证。
- 没有连接真实机器人，没有发布 FR3 或 Wuji 命令，没有完成真实 rollout。
- Safety Gateway、手部安全限制和 action 执行关联仍需现场验证。

因此当前状态是 **Stage 0 离线评估完成，Stage 1–3 prediction-only/输入一致性待现场执行，Stage 4 受控运动未开放**。现有代码明确禁止把 DP3 输出直接接到底层控制器。

## 六、汇总后的判断

这批修改完成了“训练结果 → 可独立推理包 → 现场只读验证工具”的工程链路，但没有证明机器人已经能可靠完成任务。当前最需要保留和复核的是：

1. Epoch 20 EMA checkpoint 与部署 bundle 的对应关系。
2. 54 维关节映射和 10 Hz 两帧输入历史。
3. 训练时的点云裁剪/采样参数与现场相机姿态是否一致。
4. validation 泛化误差及首步目标偏差。
5. 现场 ROS 输入和 Safety Gateway 执行证据。

建议后续先把本文件及相关代码作为一次逻辑变更提交，再在机器人主机上完成 Stage 1–3 只读验证；不要直接把当前工作区全部当作“已完成实机部署”。
