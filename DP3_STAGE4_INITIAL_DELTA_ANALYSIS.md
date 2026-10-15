# Epoch 20 第一动作与实测状态：离线复核

范围：仅 Epoch 20 EMA、原有 `AB_committed_4096.zarr` 的同一组 7,470 个 validation 序列；没有重训、改模型输出、连接机器人或创建控制 publisher。原始离线结果为 `outputs/deployment_offline_epoch20_final/validation_predictions.npz`。逐维完整数据见 [CSV](training/deployment/epoch20_joint_delta_diagnostics.csv)，汇总见 [JSON](training/deployment/epoch20_joint_delta_diagnostics.json)。CSV 对每个关节和每种比较均给出 mean、median、P90、P95、max 与严格 `>0.05 rad` 的比例。

## 1. 三个变量和共同样本

- `q_t`：第二个 observation 中的 measured state，离线文件的 `hold[:,0,:]`。
- `a_gt(t)`：第一步 demonstration action，离线文件的 `target[:,0,:]`；臂部标签来自 Safety Gateway **已验证输出** `/teleop/validated_arm_commands`，并非原始 `/teleop/arm_commands`。
- `a_pred(t)`：Epoch 20 EMA `predict_action()` 的第一步、已反归一化 action，离线文件的 `prediction[:,0,:]`。

逐样本以 `global_frame` 回查最终 Zarr：`q_t == data/state[global_frame]`、`a_gt(t) == data/action[global_frame]`，全部 7,470 条逐值相等，最大差均为 0；7,470 个 frame 索引互不重复。故以下不是把不同 frame 的 state/action 混在一起。这里的 `t` 指 **DP3 的 10 Hz 网格时间**，各原始 ROS 消息可能是时间戳不晚于 `t` 的最近样本，具体源时间可不同。

## 2. 臂部 14 关节：相同样本上的三个差值

所有表中 delta 均为绝对差，单位 rad；“任一 >0.05”在每条序列的 14 个臂关节中判断。

| 指标 | `|a_gt(t)-q_t|` | `|a_pred(t)-q_t|` | `|a_pred(t)-a_gt(t)|` |
|---|---:|---:|---:|
| 14 关节 × 7,470 样本平均 | 0.00476 | 0.07630 | 0.07679 |
| 14 关节 × 7,470 样本中位数 | 0.00141 | 0.04758 | 0.04773 |
| 每序列最大关节差的中位数 | 0.01614 | 0.22231 | 0.22360 |
| 每序列最大关节差的 P95 | 0.11164 | 0.54535 | 0.56056 |
| 任一臂关节严格 >0.05 的序列比例 | **15.46%** | **99.88%** | **99.96%** |

左臂任一超阈值：GT `13.76%`，DP3 `98.85%`；右臂：GT `1.70%`，DP3 `93.24%`。GT 有 6,315/7,470 条序列的全部 14 个臂关节都在 0.05 rad 以内；DP3 只有 9 条。DP3 每序列平均有 `6.73/14` 个臂关节超过 0.05 rad，中位数为 7 个，`79.71%` 的序列至少 5 个关节超过；所以 **99.88% 并非单个异常关节撑出来**。GT 全臂均未超阈值、而 DP3 至少一个臂关节超阈值的序列占 `84.47%`。

下表列出 14 个臂关节的 **平均绝对差 / 严格 >0.05 的样本比例**。各关节的中位数、P90、P95、最大值和第三种差值的分布完整保存在 CSV；这样不把 14×3×6 个数字挤在一张表内。

| DP3 index / 关节 | `q→GT` | `q→DP3` | `GT→DP3` |
|---|---:|---:|---:|
| 0 left_fr3_joint1 | 0.0026 / 0.0% | 0.0633 / 48.1% | 0.0632 / 47.6% |
| 1 left_fr3_joint2 | 0.0020 / 0.0% | 0.0395 / 24.2% | 0.0399 / 25.1% |
| 2 left_fr3_joint3 | 0.0030 / 0.0% | 0.0837 / 54.3% | 0.0839 / 54.4% |
| 3 left_fr3_joint4 | 0.0025 / 0.0% | 0.0574 / 40.1% | 0.0578 / 41.0% |
| 4 left_fr3_joint5 | 0.0082 / 3.7% | 0.1265 / 68.9% | 0.1283 / 69.9% |
| 5 left_fr3_joint6 | 0.0075 / 2.5% | 0.0788 / 55.6% | 0.0797 / 56.8% |
| 6 left_fr3_joint7 | 0.0196 / 9.8% | 0.1371 / 74.2% | 0.1395 / 73.0% |
| 27 right_fr3_joint1 | 0.0036 / 0.0% | 0.0580 / 39.9% | 0.0585 / 40.1% |
| 28 right_fr3_joint2 | 0.0018 / 0.0% | 0.0330 / 20.0% | 0.0330 / 20.1% |
| 29 right_fr3_joint3 | 0.0038 / 0.0% | 0.0562 / 43.8% | 0.0562 / 44.7% |
| 30 right_fr3_joint4 | 0.0018 / 0.0% | 0.0443 / 23.8% | 0.0446 / 24.2% |
| 31 right_fr3_joint5 | 0.0042 / 0.9% | 0.1489 / 67.6% | 0.1492 / 67.6% |
| 32 right_fr3_joint6 | 0.0022 / 0.2% | 0.0591 / 50.6% | 0.0592 / 50.6% |
| 33 right_fr3_joint7 | 0.0039 / 0.6% | 0.0823 / 61.7% | 0.0823 / 60.8% |

右臂 J5、左臂 J7/J5 贡献较大，应优先逐关节复核；但左/右臂多数关节均有显著比例超阈值，因此不符合“只有少数关节造成 99.88%”的解释。

## 3. 手部单独统计，不套用 FR3 0.05 规则

| 40 个手关节 | `q→GT` | `q→DP3` | `GT→DP3` |
|---|---:|---:|---:|
| 所有手关节平均绝对差 | 0.02441 | 0.05077 | 0.04298 |
| 左手平均绝对差 | 0.01409 | 0.04303 | 0.04122 |
| 右手平均绝对差 | 0.03472 | 0.05850 | 0.04473 |

手部存在更大的 command/measured 差，尤其右手；这不能代替 FR3 的安全判断。0.05 rad 是上述 Safety Gateway **臂部**初始目标参数，不是手部官方阈值。手部逐维分布也在同一 CSV 中，供另行检查。

## 4. 严格时序语义

`conversion/src/convest/align.py::align` 从 bag 可用交集起点建立 10 Hz 网格，对 measured state 与 validated command 分别取 header 时间 `≤` 网格点的最新值。`dp3_zarr.py::_vectors` 在同一个 aligned 行取 state/action，未将 action 移到下一行。`SequenceSampler` 产生长度 16 的序列，配置 `pad_before=1`、`pad_after=3`；训练 dataset 在同一序列的前两行供 observation，`DP3.predict_action()` 取反归一化 action index `n_obs_steps-1=1` 起的四行；离线 replay 的 GT 也取 `batch['action'][:,1:5]`。

```text
10 Hz 网格 t-100 ms : obs[0] = point_cloud/state at row k-1
10 Hz 网格 t        : obs[1] = point_cloud/state at row k = q_t
10 Hz 网格 t        : action[0] = row k 的 validated target = a_gt(t)
10 Hz 网格 t+100 ms : action[1] = row k+1 的 target
10 Hz 网格 t+200 ms : action[2] = row k+2 的 target
10 Hz 网格 t+300 ms : action[3] = row k+3 的 target
```

因此，**action[0] 在监督数据中是最后一个 observation 网格点的当前 target，并未按下一控制周期平移**。各 ROS source message 可早于网格点，且 state 和 validated command 的源 header 不必相同；该 target 在在线推理算完时可能已是历史目标，不能未经单独控制时序设计就视为“下一周期要发的命令”。作为辅助排查，把预测 `action[0]` 与 GT 的 `action[1]` 比较，臂部 MAE `0.07741 rad`，反而略高于与 GT `action[0]` 比较的 `0.07679 rad`；简单错后一帧不能解释主要误差。

## 5. Safety Gateway 的 0.05 rad 真正作用位置

机器人 `teleop_core/safety.py::CommandSafetyGate.validate()` 对每侧单独维护 `self.active[side]`。仅当该侧当前不 active、收到新的 active target 时，计算 `max(abs(target - measured_current))`，若 **大于** `max_initial_delta=0.05 rad` 则丢弃该侧。通过后保存 measured 位置为 `last_output` 并进入 active。持续 active 的后续消息**不再逐周期检查 target 与 measured 的 0.05 差**；它们先通过关节位置限制，再按照 `max_joint_speed × dt` 从该侧 `last_output` 限速，随后可能被 contact torque gate 保持，最后发布 validated target。`dt` 被限制在 `nominal_dt=0.01 s` 至 `0.1 s`。

状态会在该侧退出 active、reset、收到新 source/session、状态缺失/陈旧而 reset，或两侧都无有效 command 时重置；再激活会重新触发首次目标检查。`command_timeout=0.25 s` 属于 source arbitration/接管逻辑；持续 active 的 command 仍走限速/关节限位/contact gate。**不能把 0.05 rad 直接当作每个 DP3 预测目标的逐周期夹紧阈值。** 离线 GT 标签来自 Gateway 的 *validated* topic，即已经过首次检查及后续限速的输出；它通常贴近 measured state 有明确控制链原因，但由于实测跟随延迟，GT 仍可能在 15.46% 序列中出现任一臂关节 `|target-q_t|>0.05`，并不与上述代码矛盾。

## 6. 结论与剩余不确定性

本批数据**主要符合 A**：84.54% 的序列中，GT 的全部 14 个臂关节相对 measured state 均在 0.05 rad 内，而 DP3 几乎总有臂关节超过；`a_pred→a_gt` 的臂部平均误差 `0.07679 rad` 与 `q_t→a_pred` 的 `0.07630 rad` 同量级。**B 不符合当前比例**，但 action/state 的 source header 异步和 Gateway 后处理解释了 GT 偶发较大差值，也提醒我们不能把 0.05 当逐周期规则。**C 不符合“单点主导”**：DP3 每序列中位数有 7 个臂关节超过，而非一两个。

这说明**存在真实的离线第一步臂目标预测误差**，但目前不能仅凭这些统计断言原因是网络能力、扩散采样、训练数据分布或特定 normalizer 缺陷。静态 joint-name 映射、54 维顺序、EMA normalizer 加载和 action index 均与离线评估一致；14 个臂关节预测与 GT 的逐维相关系数 `0.934–0.992`，不支持整体符号反向或左右臂全面错位的简单解释。右臂 J5 有约 `-0.0646 rad` 的平均有符号偏差，值得专门复核。Stage 1–3 应继续检查真实观测分布、逐关节偏差和重复推理稳定性；任何运动测试仍需单独审核，不根据这组阈值分析直接开放。

源码定位：本地 `conversion/src/convest/{align.py,formats/dp3_zarr.py}`、`training/3D-Diffusion-Policy/diffusion_policy_3d/{common/sampler.py,policy/dp3.py}`、`training/scripts/deployment_offline_replay.py`；机器人仓库固定 commit 的 [Safety Gateway 实现](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/ros_ws/src/teleop_core/teleop_core/safety.py)、[Gateway 接入](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/ros_ws/src/teleop_core/teleop_core/safety_gateway.py)与[参数配置](https://github.com/ma39nong/Gello_armhand_teleop12/blob/03da40134b97332f6e2d9ad1162dfe287ce28d4e/config/modes/teleop_control.yaml)。
