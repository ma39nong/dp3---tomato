# ROS2 Bag to DP3 Training Pipeline

一套可复用的离线流水线：将双臂机器人 ROS2 sqlite3 rosbag 转为未裁剪中间
Zarr，交互式确定点云裁剪区域，生成固定点数的最终 Zarr，并训练 DP3。

```text
ROS2 bags
  -> causal time alignment + RGB-D back-projection
  -> intermediate packed point-cloud Zarr
  -> crop preview / crop config
  -> fixed-size point-cloud Zarr
  -> DP3 training + checkpoints
```

## 支持范围

本仓库不是“任意 rosbag 自动猜格式”的工具。内置 source adapter 面向双臂 + 双手：

- 左/右机械臂 `JointState` 状态；
- 左/右机械臂 `JointTrajectory` 动作，或合并的 `JointState` 动作；
- 左/右手 `JointState` 状态和动作；
- 一路已配准深度图 `sensor_msgs/Image` 和 `CameraInfo`；
- 所有消息使用 `header.stamp`，转换采用因果式过去帧对齐；
- 可选自定义机械臂/手健康状态。没有时，转换器按状态流存在性生成健康门控。

话题、关节名、各组维度、采样频率、深度范围和相机外参通过 YAML 配置。
其他机器人拓扑或消息类型需要新增 source adapter，不能只改话题名。

## 1. Clone 与环境

要求：Linux、Git、Python 3.11+、Conda/Miniforge、NVIDIA GPU/驱动。
读取 bag 不需要安装 ROS 2。

```bash
git clone https://github.com/HMJ-max/DP3_ROSBAG_ZARR_TRAINING_BUNDLE.git
cd DP3_ROSBAG_ZARR_TRAINING_BUNDLE

# rosbag -> 中间 Zarr 环境
./dp3.sh setup-conversion

# 最终 Zarr 与 DP3 训练环境
conda env create -f training/environment_dp3.yml
conda activate dp3
TORCH_CUDA=cu128 ./dp3.sh setup-training
```

`TORCH_CUDA` 必须与机器驱动匹配；例如可改成 PyTorch 提供的 `cu126`。
训练命令默认使用当前环境的 `python`，也可设置 `DP3_PYTHON=/path/to/python`。

## 2. 配置自己的数据

复制并修改：

```bash
cp conversion/configs/example_dp3.yaml conversion/configs/my_task.yaml
cp conversion/schemas/example_contract.yaml conversion/schemas/my_robot.yaml
cp conversion/lists/example_episodes.txt conversion/lists/my_episodes.txt
```

至少修改：

1. `source_root`：包含各 episode 目录的路径，每个目录内有 `metadata.yaml` 和 `.db3`。
2. `contract`：指向自己的 contract；修改所有话题和关节名。
3. `state_dim`：`左臂 + 左手 + 右臂 + 右手` 的位置维数总和。
4. `quality_groups`：键是数据组名，值是 episode 清单。
5. `depth_scale_m`、深度范围与 `pixel_stride`。
6. `T_point_from_depth_camera`、坐标系名称和标定版本。

清单支持任意安全的 bag 目录名、`episode1`、`1`、`1-20`，也支持注释。相机固定且训练/部署完全一致时，
可保留相机坐标系；移动相机必须填写真实 camera-to-workcell 外参。不要伪造单位矩阵。

默认示例使用标准消息和独立机械臂动作话题。若 bag 有合并动作话题，在 contract 中删除
`left_arm_action` / `right_arm_action`，增加：

```yaml
validated_arm_action:
  topic: /validated_arm_commands
  type: sensor_msgs/msg/JointState
```

并增加 `validated_left_arm` / `validated_right_arm` 两组源关节名。

## 3. Bag 转中间 Zarr

```bash
./dp3.sh bag-to-zarr \
  --config conversion/configs/my_task.yaml \
  --quality train --resume

./dp3.sh verify-intermediate \
  --config conversion/configs/my_task.yaml \
  --quality train
```

结果位于配置的 `<output_root>/train/`：

```text
dataset_uncropped.zarr/
  data/point_cloud_xyz       float32 [P, 3]
  data/point_cloud_offsets   int64   [T+1]
  data/state                 float32 [T, D]
  data/action                float32 [T, D]
  meta/episode_ends          int64   [E]
manifest.json
episode_manifest.json
conversion_report.json
conversion/
```

中间点云是变长的。`--resume` 只追加未提交的 bag，并检查已经提交的源文件未改变。

## 4. 预览与裁剪点云

以下示例假设中间结果是 `conversion/outputs/intermediate/train`。固定相机坐标系需要显式
传 `--allow-camera-frame`；已转换到工位坐标系时删掉该参数。

```bash
conda activate dp3

./dp3.sh preview conversion/outputs/intermediate/train \
  --allow-camera-frame \
  --output outputs/crop_selector.html \
  --camera-origin

xdg-open outputs/crop_selector.html
```

浏览器中调整 AABB/裁剪平面并下载 JSON，保存为 `configs/my_crop.json`。仓库提供的
`configs/example_crop.json` 只是格式示例，不应直接用于真实数据。

## 5. 中间 Zarr 转最终 Zarr

```bash
./dp3.sh crop \
  conversion/outputs/intermediate/train \
  outputs/my_task_4096.zarr \
  --crop-config configs/my_crop.json \
  --allow-camera-frame \
  --num-points 4096 \
  --expected-dim 54 \
  --voxel-size 0.005 \
  --workers 8

./dp3.sh verify-final outputs/my_task_4096.zarr \
  --num-points 4096 --expected-dim 54
```

每帧先裁剪，再体素降采样，最后使用确定性的 farthest-point sampling 固定为 N 点。
若裁剪后不足 N 点会直接失败，不会静默生成空数据。进度写入
`outputs/my_task_4096.zarr.progress.json`。

## 6. 训练 DP3

先做短流程检查：

```bash
./dp3.sh self-check
./dp3.sh train outputs/my_task_4096.zarr smoke my-task-smoke
```

正式训练默认 50 epoch，并按验证损失保存 checkpoint：

```bash
./dp3.sh train outputs/my_task_4096.zarr full my-task \
  training.num_epochs=300 \
  training.checkpoint_every=30 \
  dataloader.batch_size=32
```

输出位于 `training/data/outputs/my-task/`。W&B 默认离线；需要在线记录时：

```bash
DP3_WANDB_MODE=online ./dp3.sh train outputs/my_task_4096.zarr full my-task
```

数据集的 `target_hz` 决定时序参数：10 Hz 使用 horizon/obs/action = 16/2/4，
20 Hz 使用 32/4/8。其他频率会被拒绝，需先在训练脚本中定义清楚时域长度。

## 数据安全与校验

- rosbag 数据库以 SQLite 只读模式打开。
- 输出采用提交记录和临时目录，失败后不会把半成品标记为可训练。
- 中间/最终校验检查形状、dtype、NaN/Inf、episode 边界、时间单调性及裁剪范围。
- `.gitignore` 排除 bag、Zarr、checkpoint、缓存和本地环境，避免误上传大数据。
- 训练/验证按源 episode 分组，避免同一轨迹切片泄漏到两侧。

## 目录

```text
conversion/                  rosbag 读取、对齐、中间 Zarr、校验
configs/                     点云裁剪配置示例
training/scripts/            点云预览、最终 Zarr、训练/评估脚本
training/3D-Diffusion-Policy DP3 训练代码
dp3.sh                       统一命令入口
```

## License

本仓库集成代码使用 MIT License。DP3 派生代码保留其上游 MIT License；详见
`THIRD_PARTY_NOTICES.md` 和 `training/3D-Diffusion-Policy/LICENSE`。
