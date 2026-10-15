#!/usr/bin/env python3
"""Read-only ROS2 observation adapter for the exported Epoch20 EMA policy.

There are no publishers, action clients, or robot command calls in this file.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from action_mapping import load_mapping, preview_action
from preprocess import depth_to_xyz, sample_points, ordered_state, ObservationHistory


ROOT = Path(__file__).resolve().parent
GROUPS = ("left_arm", "left_hand", "right_arm", "right_hand")
REQUIRED = ("cam0_depth", "cam0_depth_info", *(group + "_state" for group in GROUPS))


def stamp_ns(msg) -> int:
    stamp = msg.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value <= 0:
        raise ValueError("Missing ROS source timestamp")
    return value


def decode_depth(msg) -> np.ndarray:
    dtype = {"16uc1": np.dtype("u2"), "32fc1": np.dtype("f4")}.get(msg.encoding.lower())
    if dtype is None:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    if msg.step < msg.width * dtype.itemsize or len(msg.data) != msg.height * msg.step:
        raise ValueError("Depth image byte size/step mismatch")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    return np.frombuffer(rows[:, :msg.width * dtype.itemsize].copy().tobytes(),
                         dtype=dtype).reshape(msg.height, msg.width)


def causal_asof(buffer, target: int, max_age: int):
    return max((item for item in buffer if 0 <= target - item[0] <= max_age),
               key=lambda item: item[0], default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-socket", type=Path,
                        help="Use a separate model process in another Python environment")
    parser.add_argument("--image-qos", choices=("reliable", "best_effort"),
                        default="reliable", help="Match the site's depth publisher QoS")
    parser.add_argument("--max-predictions", type=int, default=100)
    parser.add_argument("--max-runtime-s", type=float, default=180)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--save-observations", action="store_true")
    args = parser.parse_args()
    if args.max_predictions < 1 or args.max_runtime_s <= 0 or args.max_errors < 1:
        parser.error("Limits must be positive")
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    snapshots = args.output.with_name(args.output.stem + ".observations")
    if args.save_observations and snapshots.exists():
        raise FileExistsError(snapshots)
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image, JointState
    except ImportError as exc:
        raise SystemExit("ROS2 rclpy/sensor_msgs unavailable on this host") from exc
    cfg = json.loads((ROOT / "config/deploy.json").read_text())
    action_mapping = load_mapping(ROOT)
    if args.policy_socket:
        from policy_socket import SocketPredictor
        predictor = SocketPredictor(str(args.policy_socket.expanduser().resolve()),
                                    cfg["ema_weights_sha256"])
    else:
        from inference import Predictor
        predictor = Predictor(args.device)
        predictor.warmup()
        if predictor.cfg != cfg:
            raise ValueError("Loaded policy differs from local deployment manifest")
    contract = json.loads((ROOT / "config/robot_source_contract.json").read_text())
    if contract["state_names"] != cfg["state_names"]:
        raise ValueError("Bundle and robot state contracts differ")
    period_ns = int(1_000_000_000 / cfg["target_hz"])
    max_age_ns = 150_000_000
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_observations:
        snapshots.mkdir()

    class ReadOnlyNode(Node):
        def __init__(self):
            super().__init__("harvest_dp3_bundle_prediction_only")
            self.buffers = {key: deque(maxlen=32) for key in REQUIRED}
            self.history = ObservationHistory(period_ns)
            self.log = args.output.open("x", encoding="utf-8")
            self.count = 0
            self.errors = 0
            self.start = time.monotonic()
            self.next_grid_ns = None
            image_qos = QoSProfile(
                depth=10,
                reliability=(ReliabilityPolicy.RELIABLE if args.image_qos == "reliable"
                             else ReliabilityPolicy.BEST_EFFORT),
                durability=DurabilityPolicy.VOLATILE)
            for key in REQUIRED:
                msg_type = (Image if key == "cam0_depth" else CameraInfo if key == "cam0_depth_info"
                            else JointState)
                topic = contract["topics"][key]["topic"]
                self.create_subscription(msg_type, topic,
                                         lambda msg, source=key: self.on_message(source, msg),
                                         image_qos if key in ("cam0_depth", "cam0_depth_info")
                                         else qos_profile_sensor_data)
            self.create_timer(0.01, self.tick)

        def record(self, value):
            self.log.write(json.dumps(value, ensure_ascii=False) + "\n")
            self.log.flush()

        def on_message(self, key, msg):
            try:
                stamp = stamp_ns(msg)
                buffer = self.buffers[key]
                if buffer and stamp <= buffer[-1][0]:
                    raise ValueError(f"Non-increasing {key} timestamp")
                buffer.append((stamp, msg, time.monotonic_ns()))
            except Exception as exc:
                self.error(exc)

        def error(self, exc):
            self.errors += 1
            self.record({"event": "error", "error": str(exc), "sent_robot_command": False})
            if self.errors >= args.max_errors and rclpy.ok():
                self.record({"event": "stopped", "reason": "max_errors"})
                rclpy.shutdown()

        def graph_errors(self):
            graph = dict(self.get_topic_names_and_types())
            return [f"{key}: expected {item['topic']} [{item['type']}], got {graph.get(item['topic'])}"
                    for key, item in contract["topics"].items() if key in REQUIRED
                    and graph.get(item["topic"]) != [item["type"]]]

        def tick(self):
            try:
                if time.monotonic() - self.start >= args.max_runtime_s:
                    self.record({"event": "stopped", "reason": "max_runtime_s",
                                 "prediction_count": self.count})
                    rclpy.shutdown()
                    return
                if any(not self.buffers[key] for key in REQUIRED):
                    return
                now = self.get_clock().now().nanoseconds
                if self.next_grid_ns is None:
                    first = max(self.buffers[key][0][0] for key in REQUIRED)
                    self.next_grid_ns = ((first + period_ns - 1) // period_ns) * period_ns
                if now < self.next_grid_ns + 50_000_000:
                    return
                target = self.next_grid_ns
                self.next_grid_ns += period_ns
                if now - target > max_age_ns:
                    self.history.frames.clear()
                    raise ValueError(f"Missed 10 Hz tick {target}")
                # CameraInfo is recorded as a static calibration topic in the
                # robot source contract; retain the latest causal calibration.
                selected = {key: causal_asof(self.buffers[key], target,
                                             target if key == "cam0_depth_info" else max_age_ns)
                            for key in REQUIRED}
                if any(item is None for item in selected.values()):
                    self.history.frames.clear()
                    raise ValueError(f"No causal fresh sample at {target}")
                received_age_ms = {key: (time.monotonic_ns() - selected[key][2]) / 1e6
                                   for key in REQUIRED}
                if any(age > (200 if key == "cam0_depth" else 150)
                       for key, age in received_age_ms.items() if key != "cam0_depth_info"):
                    self.history.frames.clear()
                    raise ValueError("Selected source was stale at ROS adapter")
                depth_msg = selected["cam0_depth"][1]
                info = selected["cam0_depth_info"][1]
                if depth_msg.header.frame_id != cfg["point_frame"] or info.header.frame_id != cfg["point_frame"]:
                    raise ValueError("Depth/CameraInfo frame mismatch")
                messages = {group: selected[group + "_state"][1] for group in GROUPS}
                state = ordered_state(messages, cfg)
                xyz = depth_to_xyz(decode_depth(depth_msg), info, cfg)
                points, crop_count, voxel_count = sample_points(xyz, cfg)
                obs = self.history.append(points, state, target)
                if obs is None:
                    return
                prediction = predictor.predict(obs["point_cloud"], obs["agent_pos"])
                response_age_ms = (self.get_clock().now().nanoseconds - target) / 1e6
                snapshot = None
                if args.save_observations:
                    snapshot = snapshots / f"prediction_{self.count + 1:06d}.npz"
                    np.savez(snapshot, point_cloud_m=obs["point_cloud"],
                             state_rad=obs["agent_pos"],
                             predicted_action_chunk_rad=prediction["action"])
                self.count += 1
                self.record({"event": "prediction", "timestamp_ns": target,
                             "prediction_index": self.count,
                             "source_timestamps_ns": {key: selected[key][0] for key in REQUIRED},
                             "source_receive_age_ms": received_age_ms,
                             "point_cloud_shape": list(obs["point_cloud"].shape),
                             "state_shape": list(obs["agent_pos"].shape),
                             "point_cloud_xyz_min": points.min(axis=0).tolist(),
                             "point_cloud_xyz_max": points.max(axis=0).tolist(),
                             "point_cloud_xyz_mean": points.mean(axis=0).tolist(),
                             "cropped_points": crop_count, "voxel_candidates": voxel_count,
                             "state_rad": state.tolist(),
                             "predicted_action_chunk_rad": prediction["action"].tolist(),
                             "named_action_preview": preview_action(prediction["action"], action_mapping),
                             "inference_latency_ms": prediction["latency_ms"],
                             "roundtrip_latency_ms": prediction.get("roundtrip_latency_ms"),
                             "observation_age_at_response_ms": response_age_ms,
                             "response_exceeded_200ms": response_age_ms > 200,
                             "observation_snapshot": str(snapshot) if snapshot else None,
                             "sent_robot_command": False})
                if self.count >= args.max_predictions:
                    rclpy.shutdown()
            except Exception as exc:
                self.error(exc)

    rclpy.init()
    node = ReadOnlyNode()
    try:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and node.graph_errors() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not rclpy.ok():
            raise RuntimeError("Read-only adapter stopped before graph verification")
        mismatch = node.graph_errors()
        if mismatch:
            raise RuntimeError("ROS topic/type mismatch: " + "; ".join(mismatch))
        rclpy.spin(node)
    finally:
        node.log.close()
        node.destroy_node()
        if args.policy_socket:
            predictor.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
