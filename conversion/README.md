# Conversion package

`src/convest` reads compatible ROS2 sqlite3 bags without a ROS installation,
aligns required streams causally by `header.stamp`, back-projects registered
depth with `CameraInfo`, and writes a recoverable uncropped DP3 Zarr.

Use the repository root [`README.md`](../README.md) and `./dp3.sh`. The adapter
contract example is `schemas/example_contract.yaml`; task conversion settings
are in `configs/example_dp3.yaml`.
