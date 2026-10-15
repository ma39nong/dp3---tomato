# Epoch20 EMA prediction bundle

The DP3 serve and read-only observation framework adaptation lives in
`/home/user/dp3-epoch20-deployment`; see
[DP3_EPOCH20_FRAMEWORK_ADAPTER.md](DP3_EPOCH20_FRAMEWORK_ADAPTER.md) for the
directory roles and current validation status. `/home/user/pi05/部署.md` is
the source for the industrial-computer interface; the older independent
`origin/dp3` prototype is not the current deployment entry point.

This directory contains the model code, EMA weights, normalizer state, frozen
preprocessing contract, 54-joint mapping, and read-only inference entry points.
The weights file is a local 975 MiB artifact and is ignored by Git. Copy the
whole directory, including `checkpoints/epoch20_ema.pt`, to the robot computer.
The bundle does not read the training Zarr or the original 3.9 GiB checkpoint
at inference time. Startup verifies the exported EMA file against the SHA256
recorded in `config/deploy.json`.

## Inputs and output

- Input: two consecutive 10 Hz observations, `point_cloud_m [2,4096,3]` in
  `cam0_depth_optical_frame` and `state_rad [2,54]` in the pinned joint order.
- Output: `[4,54]` absolute joint position targets in the physical units of
  the recordings. `predict_action()` applies observation normalization and
  action inverse normalization internally. Do not normalize a second time.
- The programs in this directory do not send any robot command. The four
  predicted targets do not imply an approved execution horizon.

## Environment

Use one Python environment that can import both PyTorch and ROS2 `rclpy` for
live use. The local inference check used Python 3.10, PyTorch 2.7.1+cu128,
torchvision 0.22.1+cu128, diffusers 0.11.1, hydra-core 1.3.2,
omegaconf 2.3.0, einops 0.8.1, NumPy 1.26.4, zarr 2.18.3, dill and
termcolor. Install a PyTorch build appropriate for the target GPU before
installing the other dependencies.

## Prediction from saved observations

```bash
python deploy.py --observation /path/to/two_frames.npz \
  --output /path/to/prediction.json --device cuda:0 --seed 321
```

The NPZ must have `point_cloud_m` and `state_rad`. The output path must be new.
The policy is loaded once per process; use `Predictor` from `inference.py` for
repeated predictions without reloading weights.

## Live read-only observation

```bash
python ros_prediction_only.py --output /path/to/epoch20_prediction.jsonl \
  --device cuda:0 --max-predictions 100 --max-runtime-s 180 \
  --save-observations
```

The ROS adapter checks the fixed source topic/type contract, aligns fresh
messages causally on a 10 Hz grid, builds the two-frame history, saves optional
exact observation snapshots, and logs input statistics, predictions, named
arm/hand target previews and
latency. It contains no publisher or robot command client. The depth/CameraInfo
subscriber defaults to RELIABLE QoS, following the same workcell's π0.5
observation experience; use `--image-qos best_effort` only when the actual depth
publisher requires it. Verify delivery on the live graph.

If the ROS2 environment cannot import the model's PyTorch stack, run the model
in its own Python environment and use a local Unix socket shared with the ROS
environment (and bind mounted if the ROS adapter runs in a container):

```bash
# Model environment, terminal A; no ROS imports
python policy_server.py --socket /tmp/dp3-policy.sock --device cuda:0

# ROS2 environment, terminal B; no PyTorch import
python ros_prediction_only.py --policy-socket /tmp/dp3-policy.sock \
  --output /path/to/epoch20_prediction.jsonl --save-observations
```

The server refuses an existing socket path, restricts socket permissions to
the creating user, and warms the model with a synthetic observation before
accepting requests. The ROS process must be able to access the socket path.
Both programs are prediction-only. This entry point
has not been run against the real ROS graph; verify topic/type, frame,
timestamps, point-cloud statistics and inference latency on site. Actual
movement must use the existing Safety Gateway after review by the authorized
robot operator.

See [PI05_REFERENCE_REVIEW.md](PI05_REFERENCE_REVIEW.md) for the interface
decisions taken from the colleague's repository and the contracts that differ.

## Rebuild provenance

`build_bundle.py` is the one-time export utility; it reads the source
checkpoint and final training Zarr to verify the contract, then saves only
the EMA state dict and deployment manifest. It is not used on the robot.
