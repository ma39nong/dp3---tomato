# Harvest tomato DP3 deployment preparation (read-only)

This preparation uses the completed 300-epoch experiment. It has not moved a
robot. The priority candidate is `epoch=0020.ckpt`; `epoch=0007.ckpt` is the
comparison candidate. Both are format-v2 checkpoints containing raw and EMA
weights, optimizer, scheduler, and train-only normalizer. Deployment inference
loads **EMA**, calls `eval()`, and uses `torch.inference_mode()` on `cuda:0` in
float32. The checkpoint choice is a CLI argument.

## Existing chain and missing interface

- `training/scripts/evaluate_bimanual.py` is an existing offline evaluator.
  Its loader now accepts both the older nested `state_dicts` layout and the
  300-epoch checkpoint's top-level `ema_model` layout. The new replay and
  dry-run loaders use the latter layout directly.
- `training/3D-Diffusion-Policy/diffusion_policy_3d/policy/dp3.py` normalizes
  `point_cloud` and `agent_pos`, samples a 16-step diffusion trajectory, then
  **unnormalizes** actions and returns four target steps beginning at index 1.
- This repository has no existing real-robot command publisher, official joint
  limit table, emergency-stop integration, or deployed ROS2 bridge. The dry-run
  script subscribes only. There is no command-sending mode in it.
- The present computer has no ROS2 `rclpy` installation or live robot topics.
  The real-device dry-run and live/train point-cloud comparison remain pending
  until the robot host is available. Do not treat offline cloud statistics as
  live measurements.

## Training/deployment contract

- Model input: `point_cloud` `[1,2,4096,3]` float32 XYZ in
  `cam0_depth_optical_frame`; `agent_pos` `[1,2,54]` float32 joint positions in
  radians. The training sampler yields 16-frame windows; the policy consumes
  their first two observation steps.
- Recorded source: `/cam0/depth/image_raw` plus
  `/cam0/depth/camera_info`, converted using the actual CameraInfo K matrix,
  valid depth 0.05–3.0 m, integer depth scale 0.001 m, pixel stride 2, identity
  transform. The camera optical frame was **not** transformed to a robot base
  frame. The crop is `x [-1.2,1.2], y [-0.6,0.3], z [0.3,1.45]` m plus the
  two stored 40/40 clipping planes. Voxel downsampling is 0.005 m, then
  farthest-point sampling to exactly 4096 points. The dry-run imports the
  conversion and crop functions rather than defining a second geometry path.
- State/action dimension 54: indexes 0–6 left Franka arm, 7–26 left Wuji hand,
  27–33 right Franka arm, 34–53 right Wuji hand. Exact per-joint topic/name
  mapping is in `outputs/deployment_preparation/joint_mapping.csv`. The
  measured arm state names are `*_fr3_joint*`, while recorded validated arm
  **action source** names are `*_fr3v2_joint*`; Zarr action indexes are renamed
  to the canonical measured names. This semantic mapping must be verified
  against the actual robot command adapter before any action is sent.
- Action semantics: **absolute joint position target in radians**, not delta or
  velocity. The four returned steps correspond to 10 Hz demonstration steps.
  No execution chunk length, interpolation, filter, or robot-control frequency
  has been selected. They require hardware and operator confirmation.
- `policy.predict_action()` already returns unnormalized physical targets.
  The dry-run logs both those values and their normalized representation.
- The checkpoint EMA `state_dict` contains the fitted action, state, and point
  cloud normalizers; they were fitted on training bags only. The live script
  restores them with the EMA model rather than refitting them.

## Offline replay, Epoch 20

Run read-only replay with:

```bash
cd /home/user/dp3/DP3_ROSBAG_ZARR_TRAINING_BUNDLE
/home/user/miniforge3/envs/dp3/bin/python training/scripts/deployment_offline_replay.py \
  --checkpoint training/data/outputs/harvest-tomato-ab-pre722-40-40-300-from-scratch-4096/checkpoints/epoch=0020.ckpt \
  --output outputs/deployment_offline_epoch20_final --device cuda:0
```

The completed report and per-bag plots are in
`outputs/deployment_offline_epoch20_final/`. It evaluated all 7,470 validation
sequences with deterministic EMA inference. Motion is the RMS difference
between adjacent **ground-truth absolute action targets** across each four-step
chunk. Thresholds came from training sequences: bottom quartile <= 0.002382 rad
is low-motion; top quartile >= 0.009466 rad is strong-motion. The remainder of
the validation set above the low-motion threshold is called moving.

| Validation group | Count | DP3 MAE | Hold MAE | DP3 RMSE | Hold RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| All | 7470 | 0.05538 | 0.02294 | 0.08874 | 0.05438 |
| Low-motion | 2121 | 0.05593 | 0.01929 | 0.09052 | 0.04556 |
| Moving | 5349 | 0.05516 | 0.02439 | 0.08803 | 0.05751 |
| Strong-motion | 1548 | 0.05732 | 0.03242 | 0.09293 | 0.07543 |

The hold baseline remains better even among strong-motion sequences. This is
an offline action-imitation metric, not a real-task success measurement; holding
still cannot complete harvest. However, this result is a serious reason to
inspect predictions and run passive live dry-run before any command connection.
The model is **not** simply predicting no motion: its first target differs from
the measured current state by 0.0574 rad mean absolute value, versus 0.0193 rad
for the demonstrated first target. This suggests excess motion or an offset
in some joints, which needs visual and hardware-context review.
Ten of 54 joints have better per-joint RMSE than hold. The highest model RMSE
joint is `right_fr3_joint5` (0.2326 rad versus hold 0.0222 rad). The fraction
of predicted entries outside training demonstration action min/max is 0.000483.

Replay artifacts include an `A` validation bag, a `B` bag, the highest median
motion bag, and the lowest median motion bag. Each has a CSV of all 54 joints
and an HTML plot of representative left/right arm/hand joints. The full
validation prediction/GT/hold arrays are saved in
`validation_predictions.npz` for further inspection. The plots contain only
offline results.

## Passive ROS2 dry-run (prepared; not yet live-tested)

On the robot host, once this repository, checkpoint, final Zarr, ROS2 Humble,
and a compatible Python environment are present, the **prediction-only** entry
point is:

```bash
cd /home/user/dp3/DP3_ROSBAG_ZARR_TRAINING_BUNDLE
python training/scripts/deployment_dry_run.py --dry-run \
  --checkpoint-choice epoch20 \
  --output outputs/deployment_preparation/robot_dry_run_epoch20.jsonl \
  --device cuda:0 --max-predictions 100
```

Use `--checkpoint-choice epoch7` with a new output path for comparison. The
script subscribes to depth, CameraInfo, and four JointState topics from the
recorded contract. It checks header freshness, named joint ordering, depth
frame, 4096 finite points, model/action shape and finiteness. It records the
timestamps, cloud statistics, state, normalized and unnormalized prediction,
first-target delta, latency, and safety flags. It has **no ROS publisher,
action client, or robot command call**. Any live shape/distribution check is
pending until this script actually receives device messages.

Recorded action min/max and consecutive demonstration step bounds are used for
diagnostic flags only. They are not substitutes for official Franka/Wuji joint
limits, the teleoperation controller's safety checks, an emergency stop, and
an explicit command interface. Stage 4 execution must remain separate.

## Stage gates

0. Offline replay: complete for Epoch 20; predictions need operator review.
1. Live prediction-only dry-run: prepared, pending robot host/topics.
2. Compare live cloud/state statistics and camera pose with training data:
   pending live samples. Camera pitch changed across recording batches and
   there is no camera-to-base extrinsic calibration.
3. Operator reviews predicted targets and intervention flags: pending.
4. Controlled motion test: no control adapter exists in this repository; do
   not begin until official limits, emergency-stop path, command semantics,
   control cadence, and execution chunk length are verified.
