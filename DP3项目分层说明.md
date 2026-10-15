# DP3 项目分层说明

项目负责人可以按下面五层管理 DP3。目录已经基本具备这些边界，重点是提交时不要把所有实验文件混在一起。

## 1. 转换数据层

目标：把 ROS bag 转成训练需要的固定格式 Zarr。

保留范围：

- `conversion/`
- `configs/harvest_tomato_AB_40_40.json`
- `conversion/scripts/build_episode_list.py`
- `training/scripts/bimanual_zarr.py`
- `training/scripts/merge_bimanual_zarr.py`
- `dp3.sh` 中的 `preview/crop/verify-final/merge` 命令

关键产物：

- A/B 经过 40/40 裁剪、体素降采样、FPS 采样后的 Zarr。
- 固定 54 维 state/action、4096 点 XYZ、episode manifest。

这一层不应包含 checkpoint、ROS 推理程序或机器人控制代码。

## 2. 训练模型层

目标：从最终 Zarr 训练 DP3，并产生可复现的 checkpoint。

保留范围：

- `training/3D-Diffusion-Policy/`
- `training/scripts/train_bimanual.sh`
- `training/3D-Diffusion-Policy/harvest_300.py`
- `training/3D-Diffusion-Policy/diffusion_policy_3d/config/harvest_tomato_300.yaml`
- `training/3D-Diffusion-Policy/diffusion_policy_3d/model/common/harvest_lr_scheduler.py`

正式实验：

- 300 epoch、seed 42、EMA、训练集拟合 normalizer。
- 主部署候选为 Epoch 20 EMA。
- Epoch 7 只作为离线比较 checkpoint。

这一层不应依赖真实机器人 ROS 环境。

## 3. 评估与审核层

目标：证明模型加载正确，并记录模型风险；不代表实机成功。

保留范围：

- `training/scripts/evaluate_bimanual.py`
- `training/scripts/deployment_offline_replay.py`
- `training/scripts/diagnose_dp3_prediction_bias.py`
- `training/DEPLOYMENT_READINESS.md`
- `DP3_PREDICTION_BIAS_OFFLINE_REVIEW.md`
- `DP3_STAGE4_INITIAL_DELTA_ANALYSIS.md`
- `DP3_REAL_ROBOT_DEPLOYMENT_REVIEW.md`

当前结论：Epoch 20 可以离线推理，但 validation 泛化误差和首步目标偏差仍需现场输入验证；Stage 4 不能根据离线指标直接开放。

## 4. 部署模型层

目标：不依赖训练数据，使用 checkpoint 完成实时 prediction-only 推理。

保留范围：

- `dp3_deploy/README.md`
- `dp3_deploy/config/`
- `dp3_deploy/inference.py`
- `dp3_deploy/preprocess.py`
- `dp3_deploy/action_mapping.py`
- `dp3_deploy/deploy.py`
- `dp3_deploy/build_bundle.py`
- `dp3_deploy/ros_prediction_only.py`
- `dp3_deploy/policy_socket.py`
- `dp3_deploy/policy_server.py`
- `dp3_deploy/diffusion_policy_3d/`

部署接口固定为：两帧 10 Hz 输入，点云 `[2,4096,3]`，状态 `[2,54]`，输出 `[4,54]` absolute joint target。

`dp3_deploy/checkpoints/epoch20_ema.pt` 应作为独立模型制品发布，和源码版本绑定 checksum。

## 5. 实机验证层

目标：在真实 ROS 主机上验证输入、时序和安全链；当前只读，不执行动作。

保留范围：

- `training/scripts/deployment_dry_run.py`
- `training/scripts/extract_rollout_rosbag_events.py`
- `training/scripts/real_rollout_report.py`
- `training/deployment/REAL_ROBOT_OBSERVATION_PLAN.md`
- `training/deployment/robot_source_contract.json`
- `training/deployment/ros_to_dp3_joint_mapping.csv`

当前状态：本机尚未完成 `rclpy`/ROS graph/CameraInfo/真实时钟验证；没有发布 FR3 或 Wuji 命令。

## 建议提交顺序

1. **数据转换提交**：只包含转换器、裁剪配置、Zarr 合并和验证入口。
2. **训练提交**：只包含 300 epoch 实验、normalizer 修复和评估兼容性。
3. **离线评估提交**：包含 replay、bias 诊断和审核报告。
4. **部署提交**：包含 `dp3_deploy` 源码与配置；checkpoint 作为独立制品。
5. **实机验证提交**：包含 prediction-only、ROS bag 事件提取和 rollout 汇总。

`trial`/`full` 兼容脚本、学习率比较、裁剪预览和 PI05 参考资料放在实验归档，不进入以上五个正式提交。
