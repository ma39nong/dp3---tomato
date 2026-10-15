# DP3 臂部预测偏差：纯离线专项复核

**结论（2026-09-28）：Stage 4 继续阻塞。** Epoch 20 的臂部首步预测在 validation 上确实偏离 GT；本轮数值与源码复核未发现 action 重复反归一化、使用错误 normalizer、54 维顺序整体错位或扩散采样随机性足以解释该误差的证据。最强的新证据是同一 Epoch 20 EMA 模型在随机抽取的 training 序列上臂部首步 MAE 约 0.0106 rad，在随机 validation 序列上约 0.0759 rad，而两组 GT 与当前实测 state 的差值均约 0.0046 rad。这支持**模型对当前 bag 级验证分布的泛化不足**；具体造成泛化差的相机/场景、动作分布与模型训练机制比例，仍需后续离线实验确认。本文没有接入机器人或改动模型输出。

## 数据、复核范围和可重现产物

- 最终训练数据：`outputs/harvest_tomato_AB_pre722_40_40/AB_committed_4096.zarr`；300 epoch 固定 bag 划分：train 66,465 frames，validation 7,875 frames。完整离线 replay 共有 7,470 个有效 validation sequence；采样窗口的长度和边界约束使其少于 validation frame 数。
- Checkpoint：`training/data/outputs/harvest-tomato-ab-pre722-40-40-300-from-scratch-4096/checkpoints/epoch=0007.ckpt` 与 `epoch=0020.ckpt`，均严格加载 `ema_model`，float32、eval、`torch.inference_mode()`、RTX 5090。
- Epoch 20 原全量回放：`outputs/deployment_offline_epoch20_final/validation_predictions.npz`。本轮用同一 `deployment_offline_replay.py`、seed 42、batch size 32 生成 Epoch 7 的 `outputs/deployment_offline_epoch7_matched/validation_predictions.npz`。两个文件的 `global_frame`、`target`、`hold`、`source_bag` 数组逐值相同，故比较的是同一批目标和当前状态。
- 专项程序：`training/scripts/diagnose_dp3_prediction_bias.py`；输出：`outputs/deployment_prediction_bias_diagnostics/summary.json` 和 `arm_joint_diagnostics.csv`。程序只读 Zarr/checkpoint，不引入 ROS 或机器人控制代码。64 个 A/B 与运动程度分层的 validation sequence 用于重复采样；另用 seed 2026 从 train/validation 各随机取 64 个 frame 做相同推理对照。64 条对照是诊断抽样，不代替 7,470 条全量验证结论。
- 本文“臂部首步 MAE”是 14 个臂关节 × 序列的 `mean(abs(pred_action[0]-GT_action[0]))`，单位 rad。它是逐元素平均绝对误差，不是 14 维欧氏范数。

## 1. Normalizer 与模型输出空间

`GenericZarrDataset.__getitem__()` 将最终 Zarr 的原始 `data/action` 直接作为 `batch['action']`，将原始 `data/state` 改名为 `obs.agent_pos`；仅训练时对点云加 0.002 m 高斯噪声，validation 关闭噪声。`GenericZarrDataset.get_normalizer()` 只读取 training episode 的 action、state、point cloud 拟合。`DP3.compute_loss()` 对 observation 和 action 分别 normalize，扩散网络学习 normalized action；`DP3.predict_action()` 对 observation normalize，扩散采样后在函数内部用 action normalizer **执行一次 unnormalize**，再取 action indices `1:5` 返回物理关节目标。`deployment_offline_replay.py` 直接使用这个返回值与原始 `batch['action'][:,1:5]` 比较，没有第二次 unnormalize。

数值复核：对同一 validation sample，前 14 个臂关节的 raw、normalized、inverse 后值记录在 `summary.json`。整条 54 维 action 的 inverse(normalize(GT)) 最大绝对误差 `2.38e-7 rad`；state 为 `2.38e-7`；point cloud 为 `1.19e-7 m`。Epoch 20 checkpoint action/state normalizer 的 scale 和 offset 与重新以 66,465 个 training frame 拟合的结果逐值相同（最大差 0）；Epoch 7 和 Epoch 20 的 action/state/point-cloud normalizer 参数也逐值相同。point-cloud train-only 来源由 dataset 代码确认，未为本轮额外把所有训练点云再加载一遍重新拟合。

**结论：** 当前 checkpoint/离线评估路径上的 normalizer 尺度、维度顺序和反归一化次数可以基本排除为约 0.0768 rad 全局误差的主因。实时 ROS 输入仍需现场单独核对分布；这项离线结论不等于完成 live 验证。

## 2. Target、shape、index 和时间语义

最终 Zarr 的 state/action 都是 54 维，依次是左臂 7、左手 20、右臂 7、右手 20；本文仅取臂部索引 `[0:7,27:34]`。训练样本来自同一个 `SequenceSampler` 的长度 16 窗口：`obs.point_cloud` 为 `[16,4096,3]`、`obs.agent_pos` 为 `[16,54]`，训练 target 为 `[16,54]`；模型只编码前 `n_obs_steps=2` 个 observation。推理返回 `[4,54]`。`n_obs_steps-1=1`，因此返回的首步是扩散轨迹第 1 个 action，不是第 0 个，也不是额外向前挪一帧。离线 replay 的 GT 正是 `batch['action'][:,1:5]`。整个 validation replay 的第一步 GT 与最终 Zarr 的 `data/action[global_frame]`、当前 state 与 `data/state[global_frame]` 逐值相同。

10 Hz 对齐网格对应：`obs[0]=t-100ms`、`obs[1]=t`、`action[0]=t`、后续三个 action 为 `t+100/200/300ms`。这是数据标签的同一网格点语义，**不表示在线推理后可以直接把 action[0] 当作下一控制周期命令**。前次复核中，把首步预测改与 GT 下一步比，臂部 MAE 0.07741 rad，比与同一步 GT 的 0.07679 rad 稍差；简单一帧平移不能解释本问题。

## 3. 全量 validation：Epoch 7 对 Epoch 20

| EMA checkpoint | 序列 | 臂部首步 pred–GT MAE / RMSE | 臂部首步 pred–q MAE | GT–q MAE |
|---|---:|---:|---:|---:|
| Epoch 7 | 7,470 | 0.08272 / 0.12524 | 0.08220 | 0.00476 |
| Epoch 20 | 7,470 | **0.07679 / 0.11690** | 0.07630 | 0.00476 |

Epoch 20 对首步臂部误差略优于 Epoch 7，但两者均明显偏离当前状态附近的 GT。Epoch 20 左臂 pred–GT MAE/RMSE 为 `0.08460/0.12399`，右臂为 `0.06899/0.10935`。按来源分，A validation 的臂部首步 MAE/RMSE 为 `0.05991/0.09363`，B 为 `0.07995/0.12075`（A/B 数量分别 1,176/6,294）。这些是首步臂关节误差；此前整体 54 维、4 步的 Epoch 20 action MAE/RMSE `0.05538/0.08874` 是另一种统计，不能混用。

“系统性偏差”需要再细分：14 个关节总体的**有符号**残差均值约 `-0.00154 rad`，并非所有关节朝同一方向错 `0.0768 rad`。较突出的固定方向偏置是右臂 J5 `-0.06463 rad`、右臂 J7 `+0.03550 rad`；左臂 J5 `-0.00221 rad`、左臂 J7 `-0.00011 rad` 接近零均值，但其平均绝对误差分别是 `0.12833`、`0.13948 rad`，属于较大的**随样本变化的误差**。右臂 J5、左臂 J7、左臂 J5 合计贡献全臂首步平方误差约 58%，其余 11 个关节仍贡献约 42%，所以不是单个关节映射翻转。

完整 14 关节的 train/validation/pred 分布、Epoch 7/20 有符号偏置和抽样扩散标准差见 `arm_joint_diagnostics.csv`；CSV 的偏置/平均绝对误差基于全量 7,470 条，采样标准差基于下述 64 条 × 8 次。

## 4. 扩散采样随机性

对同一组 64 个 validation observation，各 checkpoint 以 8 个 seed 重复采样，且固定 seed 重跑一次。

| EMA | 14 臂关节预测的平均采样标准差 | 8 次 arm RMSE 的均值 ± 标准差 | 同 seed 重跑最大逐值差 |
|---|---:|---:|---:|
| Epoch 7 | 0.000772 rad | 0.113156 ± 0.000346 rad | 0 |
| Epoch 20 | 0.000365 rad | 0.108236 ± 0.000106 rad | 0 |

Epoch 20 的扩散随机波动比全量首步 MAE `0.07679 rad` 小两个数量级。这排除了“主要由一次不走运的 diffusion seed 导致”的解释。此结论限于当前 8 个 seed、64 个 observation、当前采样器配置。

## 5. 训练/验证泛化与输入响应

从 train/validation 各随机抽取 64 个 frame，以相同 Epoch 20 EMA、同一 seed、无点云增强做首步推理：training pred–GT MAE/RMSE `0.01059/0.01671 rad`；validation `0.07592/0.11622 rad`。GT–当前 state 分别为 `0.00462/0.00464 rad`。因此不是“validation 的 GT 本身比 training 更远离状态”导致这组差距。训练记录的 epoch 20 `ema_train_eval_loss=0.000794`、`ema_val_loss=0.014178` 也同方向，但 fixed train subset 只有四个 bag，不能单独当作全 train 估计。

固定 seed 的输入敏感性检查用了 16 个相差明显的 validation observation（A 4、B 12）。全 observation 改变时首步臂预测相对冻结输入变化的平均绝对值 `0.4940 rad`，对应 GT 相对参照 observation 的平均变化 `0.5199 rad`；只改 point cloud 是 `0.0515 rad`，只改 state 是 `0.4842 rad`。冻结全部 observation 再与这些不同 GT 比的 RMSE 为 `0.8691 rad`，使用对应 observation 则为 `0.0955 rad`。模型**确实响应输入**，不是简单输出完全固定的训练均值。交换单一模态会产生训练中未见过的 state/cloud 组合，因此这些数字只能诊断响应性，不能直接量化真实场景中的各模态因果贡献。

分布上，Epoch 20 预测的 14 个臂关节中 13 个在相同 7,470 条 validation sequence 上的标准差小于 GT，标准差比率中位数约 `0.940`；11 个关节的预测均值比 GT 均值更靠近 training 均值。右臂 J5：training 均值 `1.452`、validation GT 均值 `1.575`、prediction 均值 `1.510 rad`；右臂 J7 对应 `1.978/1.922/1.957 rad`。这与一定程度的“向训练分布回缩”一致，但大多数关节的预测均值与验证 GT 均值接近，不能把整个误差归因于常数均值偏置。需要进一步按 bag、姿态区间、相机/任务阶段有真值标签时分层查验。

## 原因排序

| 类别 | 当前判断 | 依据与边界 |
|---|---|---|
| 已确认问题 | Epoch 20 对 validation 臂部首步目标有较大的、跨多关节的预测误差；training 与 validation action 推理误差差距明显 | 7,470 条全量 MAE 0.07679；随机 64/64 train/val 约 0.01059/0.07592；GT–state 均约 0.0046。 |
| 高概率原因 | 模型对 bag 级 validation 分布泛化不足，部分关节向训练分布回缩 | 训练/验证误差与 diffusion loss gap 同方向；B 比 A 差；13/14 关节预测方差缩小；右臂 J5/J7 均值回缩。这里尚不能区分数据覆盖、相机/场景变化和模型结构/训练目标的各自贡献。 |
| 尚未排除 | 个别关节的数据分布异常、相机/场景子群偏移、采集状态与 action 的局部关联弱 | 右臂 J5 有真实有符号偏置；左臂 J5/J7 是高方差误差。缺可靠相机批次真值标签，也未用独立在线观测验证。 |
| 已基本排除为主因 | action/state normalizer 错误、重复/遗漏 unnormalize、全局 54 维映射/首步索引错误、diffusion sampling variance、完全不响应 observation | 代码链、数值 roundtrip、checkpoint 参数对照、逐帧 Zarr 对照、两 checkpoint 对齐、重复采样和模态敏感性均支持。 |

**下一道门槛：** 保持 Stage 4 冻结。可以继续离线按 bag/姿态/关节定位泛化差，并在真实设备上先做 Stage 1–3 的 prediction-only 与输入分布核查。当前不发送 FR3 或灵巧手控制命令，不修改/缩放/clamp DP3 输出。
