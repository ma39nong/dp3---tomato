#!/usr/bin/env python3
"""ROS2 prediction-only probe for harvest DP3. This file has NO publisher/client.

Run only on a machine with the six recorded ROS2 sensor/state topics and rclpy.
It never imports robot control APIs and cannot send an action. Runtime checks
here are diagnostics; official joint limits are still required before execution.
"""
from __future__ import annotations

import argparse
from collections import deque
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import yaml
import zarr


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "3D-Diffusion-Policy"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))
sys.path.insert(0, str(ROOT / "conversion" / "src"))
from convest.formats.dp3_zarr import point_cloud  # noqa: E402
from bimanual_zarr import _sample_points  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
GROUPS = (("left_arm", 0, 7), ("left_hand", 7, 29),
          ("right_arm", 29, 36), ("right_hand", 36, 58))


def decode_depth(msg) -> np.ndarray:
    encoding = msg.encoding.lower()
    dtype = {"16uc1": np.dtype("u2"), "32fc1": np.dtype("f4")}.get(encoding)
    if dtype is None:
        raise ValueError(f"Unsupported depth encoding: {msg.encoding}")
    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    if msg.step < msg.width * dtype.itemsize or len(msg.data) != msg.height * msg.step:
        raise ValueError("Depth image byte size/step mismatch")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    return np.frombuffer(rows[:, :msg.width * dtype.itemsize].copy().tobytes(),
                         dtype=dtype).reshape(msg.height, msg.width)


def header_time_ns(msg) -> int:
    stamp = msg.header.stamp
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    if value <= 0:
        raise ValueError("Missing source header timestamp")
    return value


def causal_asof(messages, target_ns: int, max_age_ns: int):
    """Match the bag converter's latest source stamp <= a 10 Hz grid tick."""
    candidates = (item for item in messages if 0 <= target_ns - item[0] <= max_age_ns)
    return max(candidates, key=lambda item: item[0], default=None)


def ordered_positions(msg, expected_names: list[str]) -> np.ndarray:
    if len(msg.name) != len(set(msg.name)) or len(msg.name) != len(msg.position):
        raise ValueError("Duplicate or misaligned JointState names/positions")
    by_name = dict(zip(msg.name, msg.position))
    missing = set(expected_names) - set(by_name)
    if missing:
        raise ValueError(f"Missing joints: {sorted(missing)}")
    result = np.asarray([by_name[name] for name in expected_names], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite measured joint position")
    return result


def cloud_statistics(points: np.ndarray, original_count: int) -> dict:
    return {"shape": list(points.shape), "actual_points": int(len(points)),
            "cropped_points": int(original_count),
            "xyz_min": points.min(axis=0).tolist(), "xyz_max": points.max(axis=0).tolist(),
            "xyz_mean": points.mean(axis=0).tolist(), "xyz_std": points.std(axis=0).tolist(),
            "finite_ratio": float(np.isfinite(points).mean())}


def checked_prediction(action: np.ndarray, state: np.ndarray, demo_min: np.ndarray,
                       demo_max: np.ndarray, demo_max_step: float,
                       demo_target_state_gap: float) -> list[str]:
    reasons = []
    if action.shape != (4, 58):
        return [f"invalid_action_shape:{action.shape}"]
    if state.shape != (58,):
        return [f"invalid_state_shape:{state.shape}"]
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        return ["nonfinite_action_or_state"]
    if np.any(action < demo_min) or np.any(action > demo_max):
        reasons.append("outside_demonstration_action_range")
    # The command/current-state gap is not the same as a consecutive command
    # step; recorded servo lag can legitimately make it larger.
    if np.max(np.abs(action[0] - state)) > demo_target_state_gap:
        reasons.append("first_target_exceeds_demonstrated_action_state_gap")
    if np.max(np.abs(np.diff(action, axis=0))) > demo_max_step:
        reasons.append("chunk_step_exceeds_demonstration_step_gate")
    return reasons


def load_policy(checkpoint: Path, dataset: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", mmap=True,
                         weights_only=False, pickle_module=dill)
    if payload.get("format_version") != 2 or "ema_model" not in payload:
        raise ValueError("Expected format-v2 checkpoint with EMA weights")
    cfg = payload["cfg"]
    if Path(cfg.task.dataset.zarr_path).resolve() != dataset.resolve():
        raise ValueError("Checkpoint dataset path differs from deployment dataset")
    if (int(cfg.n_obs_steps), int(cfg.n_action_steps),
            tuple(cfg.task.shape_meta.obs.point_cloud.shape)) != (2, 4, (4096, 3)):
        raise ValueError("Checkpoint observation/action contract differs")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["ema_model"], strict=True)
    del payload
    gc.collect()
    policy.to(device).eval()
    if next(policy.parameters()).dtype != torch.float32:
        raise ValueError("Expected float32 EMA policy")
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", required=True)
    parser.add_argument("--checkpoint-choice", choices=("epoch20", "epoch7"), default="epoch20")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-predictions", type=int, default=100)
    parser.add_argument("--max-age-ms", type=float, default=150.0)
    parser.add_argument("--max-runtime-s", type=float, default=180.0)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--save-observations", action="store_true",
                        help="Save the exact two-frame point cloud/state input beside the JSONL log")
    args = parser.parse_args()
    if (args.max_predictions < 1 or args.max_age_ms <= 0 or
            args.max_runtime_s <= 0 or args.max_errors < 1):
        parser.error("max-predictions, max-age-ms, max-runtime-s and max-errors must be positive")
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, CameraInfo, JointState
    except ImportError as error:
        raise SystemExit("ROS2 rclpy/sensor_msgs unavailable; run on the ROS2 robot host") from error

    exp = ROOT / "training/data/outputs/harvest-tomato-ab-pre722-40-40-300-from-scratch-4096"
    checkpoint = exp / "checkpoints" / ("epoch=0020.ckpt" if args.checkpoint_choice == "epoch20"
                                       else "epoch=0007.ckpt")
    dataset = ROOT / "outputs/harvest_tomato_AB_pre722_40_40/AB_committed_4096.zarr"
    root = zarr.open_group(str(dataset), mode="r")
    if root.attrs["point_frame"] != "cam0_depth_optical_frame" or root.attrs["num_points"] != 4096:
        raise ValueError("Unexpected training point-cloud frame/count")
    contract = yaml.safe_load((ROOT / "conversion/schemas/harvest_tomato_A_contract.yaml").read_text())
    robot_contract = json.loads((ROOT / "training/deployment/robot_source_contract.json").read_text())
    for key, source in robot_contract["topics"].items():
        if contract["topics"].get(key, {}).get("topic") != source["topic"] or contract["topics"].get(key, {}).get("type") != source["type"]:
            raise ValueError(f"LIVE CONTRACT MISMATCH: configured topic/type for {key} differs from pinned robot source")
    if (list(root.attrs["state_names"]) != robot_contract["state_names"] or
            list(root.attrs["action_names"]) != robot_contract["action_names"] or
            int(root.attrs["target_hz"]) != robot_contract["dp3_hz"]):
        raise ValueError("LIVE CONTRACT MISMATCH: final Zarr differs from pinned robot source")
    conversion = yaml.safe_load((ROOT / "conversion/configs/harvest_tomato_A.yaml").read_text())
    crop = json.loads((ROOT / "configs/harvest_tomato_AB_40_40.json").read_text())
    if (conversion["source_point_frame"] != root.attrs["point_frame"] or
            conversion["pixel_stride"] != root.attrs["pixel_stride"] or
            conversion["fps"] != root.attrs["target_hz"] or
            conversion["depth_scale_m"] != root.attrs["depth_scale_m"] or
            not np.allclose(conversion["T_point_from_depth_camera"], root.attrs["T_point_from_depth_camera"]) or
            not np.allclose(crop["crop_min"], root.attrs["crop_min"]) or
            not np.allclose(crop["crop_max"], root.attrs["crop_max"]) or
            [p["name"] for p in crop["planes"]] != [p["name"] for p in root.attrs["clip_planes"]]):
        raise ValueError("LIVE CONTRACT MISMATCH: conversion/crop config differs from final Zarr")
    if args.max_age_ms != float(conversion["max_staleness_ms"]):
        raise ValueError("LIVE CONTRACT MISMATCH: max-age-ms must match training conversion")
    names = list(root.attrs["state_names"])
    for group, start, end in GROUPS:
        contract_key = "measured_" + group if group.endswith("arm") else group
        if names[start:end] != contract["source_joint_names"][contract_key]:
            raise ValueError(f"State joint order mismatch: {group}")
    if names != list(root.attrs["action_names"]) or root.attrs["action_semantics"] != "absolute_joint_target":
        raise ValueError("Action contract differs from measured state")
    actions = np.asarray(root["data/action"][:], dtype=np.float32)
    demo_min, demo_max = actions.min(axis=0), actions.max(axis=0)
    demo_target_state_gap = float(np.max(np.abs(actions - root["data/state"][:])))
    device = torch.device(args.device)
    policy = load_policy(checkpoint, dataset, device)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh dry-run log path: {args.output}")
    observation_dir = args.output.with_name(args.output.stem + ".observations")
    if args.save_observations and observation_dir.exists():
        raise FileExistsError(f"Choose a fresh observation directory: {observation_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_observations:
        observation_dir.mkdir()

    class PredictionOnlyNode(Node):
        def __init__(self):
            super().__init__("harvest_dp3_prediction_only")
            self.buffers = {key: deque(maxlen=32) for key in ("depth", "info", *(name for name, _, _ in GROUPS))}
            self.history = deque(maxlen=2)
            self.count = 0
            self.error_count = 0
            self.started_monotonic = time.monotonic()
            self.next_grid_ns = None
            self.period_ns = int(1_000_000_000 / int(root.attrs["target_hz"]))
            self.max_age_ns = int(args.max_age_ms * 1e6)
            self.log = args.output.open("x", encoding="utf-8")
            for group, _, _ in GROUPS:
                topic = contract["topics"][group + "_state"]["topic"]
                self.create_subscription(JointState, topic,
                                         lambda msg, key=group: self._update(key, msg),
                                         qos_profile_sensor_data)
            self.create_subscription(CameraInfo, contract["topics"]["cam0_depth_info"]["topic"],
                                     lambda msg: self._update("info", msg), qos_profile_sensor_data)
            self.create_subscription(Image, contract["topics"]["cam0_depth"]["topic"],
                                     self._depth, qos_profile_sensor_data)
            self.create_timer(0.01, self._tick)

        def _update(self, key, msg):
            self._append(key, msg)

        def _depth(self, msg):
            self._append("depth", msg)

        def _append(self, key, msg):
            try:
                stamp = header_time_ns(msg)
                if self.buffers[key] and stamp <= self.buffers[key][-1][0]:
                    raise ValueError(f"LIVE CONTRACT MISMATCH: non-increasing {key} source timestamp")
                self.buffers[key].append((stamp, msg))
            except Exception as error:
                self._log_error(error)

        def _log_error(self, error):
            self.error_count += 1
            self.log.write(json.dumps({"timestamp_ns": self.get_clock().now().nanoseconds,
                                       "error": str(error), "sent_robot_command": False}) + "\n")
            self.log.flush()
            if self.error_count >= args.max_errors and rclpy.ok():
                self.log.write(json.dumps({"event": "stopped", "reason": "max_errors",
                                           "error_count": self.error_count,
                                           "sent_robot_command": False}) + "\n")
                self.log.flush()
                rclpy.shutdown()

        def check_live_graph(self):
            graph = dict(self.get_topic_names_and_types())
            mismatches = []
            for key in ("left_arm_state", "right_arm_state", "left_hand_state", "right_hand_state",
                        "cam0_depth", "cam0_depth_info"):
                item = robot_contract["topics"][key]
                if graph.get(item["topic"]) != [item["type"]]:
                    mismatches.append(f"{key}: expected {item['topic']} [{item['type']}], got {graph.get(item['topic'])}")
            return mismatches

        def _tick(self):
            try:
                if time.monotonic() - self.started_monotonic >= args.max_runtime_s:
                    self.log.write(json.dumps({"event": "stopped", "reason": "max_runtime_s",
                                               "prediction_count": self.count,
                                               "sent_robot_command": False}) + "\n")
                    self.log.flush()
                    rclpy.shutdown()
                    return
                required = ("depth", "info", *(name for name, _, _ in GROUPS))
                if any(not self.buffers[key] for key in required):
                    return
                now = self.get_clock().now().nanoseconds
                if self.next_grid_ns is None:
                    first_common = max(self.buffers[key][0][0] for key in required)
                    self.next_grid_ns = ((first_common + self.period_ns - 1) // self.period_ns) * self.period_ns
                # Allow delivery jitter, then sample all sources causally on the 10 Hz grid.
                if now < self.next_grid_ns + 50_000_000:
                    return
                target = self.next_grid_ns
                self.next_grid_ns += self.period_ns
                if now - target > self.max_age_ns:
                    self.history.clear()
                    raise ValueError(f"LIVE CONTRACT MISMATCH: missed 10 Hz tick {target}")
                # CameraInfo is a static calibration topic in the recorder;
                # its latest causal message may be older than a 10 Hz tick.
                selected = {key: causal_asof(self.buffers[key], target,
                                             target if key == "info" else self.max_age_ns)
                            for key in required}
                if any(item is None for item in selected.values()):
                    self.history.clear()
                    raise ValueError(f"LIVE CONTRACT MISMATCH: no causal fresh sample at 10 Hz tick {target}")
                timestamps = {key: selected[key][0] for key in required}
                msg = selected["depth"][1]
                if msg.header.frame_id != root.attrs["point_frame"]:
                    raise ValueError(f"LIVE CONTRACT MISMATCH: depth frame {msg.header.frame_id}")
                info = selected["info"][1]
                if info.header.frame_id != msg.header.frame_id:
                    raise ValueError("LIVE CONTRACT MISMATCH: depth and CameraInfo frames differ")
                try:
                    state = np.concatenate([ordered_positions(selected[group][1], names[start:end])
                                            for group, start, end in GROUPS])
                except ValueError as error:
                    raise ValueError(f"LIVE CONTRACT MISMATCH: {error}") from error
                depth = decode_depth(msg)
                xyz = point_cloud(depth, info, conversion)
                points, cropped_count, _ = _sample_points(
                    xyz, np.asarray(root.attrs["crop_min"], dtype=np.float32),
                    np.asarray(root.attrs["crop_max"], dtype=np.float32),
                    root.attrs["clip_planes"], float(root.attrs["voxel_size_m"]), 4096)
                if len(points) != 4096 or not np.isfinite(points).all():
                    raise ValueError("Invalid or insufficient point cloud")
                if self.history and target - self.history[-1][2] != self.period_ns:
                    self.history.clear()
                self.history.append((points, state, target))
                if len(self.history) < 2:
                    return
                obs = {"point_cloud": torch.from_numpy(np.stack([x[0] for x in self.history])[None]).to(device),
                       "agent_pos": torch.from_numpy(np.stack([x[1] for x in self.history])[None]).to(device)}
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode():
                    output = policy.predict_action(obs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latency = time.perf_counter() - started
                action = output["action"][0].cpu().numpy()
                full_action = output["action_pred"][0].cpu().numpy()
                raw = policy.normalizer["action"].normalize(output["action"])[0].cpu().numpy()
                reasons = checked_prediction(action, state, demo_min, demo_max,
                                             float(conversion["max_joint_step_rad"]),
                                             demo_target_state_gap)
                if latency > 1.0 / float(root.attrs["target_hz"]):
                    reasons.append("inference_slower_than_recorded_control_period")
                snapshot_path = None
                if args.save_observations:
                    snapshot_path = observation_dir / f"prediction_{self.count + 1:06d}.npz"
                    np.savez(snapshot_path,
                             point_cloud_m=np.stack([x[0] for x in self.history]),
                             state_rad=np.stack([x[1] for x in self.history]),
                             observation_grid_timestamps_ns=np.asarray(
                                 [x[2] for x in self.history], dtype=np.int64),
                             predicted_action_chunk_rad=action)
                record = {"event": "prediction", "timestamp_ns": target,
                          "prediction_index": self.count + 1,
                          "observation_snapshot": str(snapshot_path) if snapshot_path else None,
                          "prediction_space": "physical_absolute_joint_target_rad",
                          "checkpoint_choice": args.checkpoint_choice,
                          "model_action_was_executed": False,
                          "checkpoint": str(checkpoint),
                          "source_timestamps_ns": timestamps,
                          "weights": "ema_model", "frame": root.attrs["point_frame"],
                          "input_shape": {key: list(value.shape) for key, value in obs.items()},
                          "point_cloud": cloud_statistics(points, cropped_count),
                          "state_names": names, "state_rad": state.tolist(),
                          "normalized_action_reconstructed": raw.tolist(),
                          "unnormalized_full_horizon_action_pred_rad": full_action.tolist(),
                          "unnormalized_action_rad": action.tolist(),
                          "predicted_action_chunk_rad": action.tolist(),
                          "first_action_delta_rad": (action[0] - state).tolist(),
                          "inference_latency_ms": latency * 1000,
                          "safety_flags": reasons, "sent_robot_command": False}
                self.log.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.log.flush()
                self.count += 1
                if self.count >= args.max_predictions:
                    rclpy.shutdown()
            except Exception as error:
                self._log_error(error)

    rclpy.init()
    node = PredictionOnlyNode()
    try:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and node.check_live_graph() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if not rclpy.ok():
            raise RuntimeError("Prediction-only probe stopped before graph verification; inspect JSONL errors")
        mismatch = node.check_live_graph()
        if mismatch:
            raise RuntimeError("LIVE CONTRACT MISMATCH: " + "; ".join(mismatch))
        rclpy.spin(node)
    finally:
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
