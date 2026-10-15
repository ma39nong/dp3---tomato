"""Frozen conversion and point sampling path for deployment observations."""
from __future__ import annotations

from collections import deque
import numpy as np


def depth_to_xyz(depth: np.ndarray, camera_info, cfg: dict) -> np.ndarray:
    if depth.ndim != 2 or (int(camera_info.height), int(camera_info.width)) != depth.shape:
        raise ValueError("Depth/CameraInfo dimensions differ")
    distortion = np.asarray(camera_info.d, dtype=np.float64)
    if distortion.size and not np.allclose(distortion, 0, atol=1e-9):
        raise ValueError("Raw distorted depth is unsupported")
    k = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    if not np.isfinite(k).all() or min(fx, fy) <= 0:
        raise ValueError("Invalid depth intrinsics")
    scale = cfg["depth_scale_m"] if np.issubdtype(depth.dtype, np.integer) else 1.0
    z = depth.astype(np.float32) * np.float32(scale)
    valid = np.isfinite(z) & (z >= cfg["min_depth_m"]) & (z <= cfg["max_depth_m"])
    v, u = np.nonzero(valid)
    stride = int(cfg["pixel_stride"])
    if stride > 1:
        keep = (v % stride == 0) & (u % stride == 0)
        v, u = v[keep], u[keep]
    z = z[v, u]
    if not len(z):
        raise ValueError("Depth frame has no valid points")
    xyz = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z)).astype(np.float32)
    transform = np.asarray(cfg["T_point_from_depth_camera"], dtype=np.float32)
    return (xyz @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def sample_points(points: np.ndarray, cfg: dict) -> tuple[np.ndarray, int, int]:
    crop_min = np.asarray(cfg["crop_min"], dtype=np.float32)
    crop_max = np.asarray(cfg["crop_max"], dtype=np.float32)
    mask = np.all(points >= crop_min, axis=1) & np.all(points <= crop_max, axis=1)
    for plane in cfg["clip_planes"]:
        normal = np.asarray(plane["normal"], dtype=np.float64)
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            raise ValueError("Invalid crop plane normal")
        mask &= (points @ (normal / length) + float(plane.get("offset", 0)) / length
                 >= float(plane.get("margin", 0)))
    cropped = points[mask]
    target = int(cfg["num_points"])
    if len(cropped) < target:
        raise ValueError(f"Only {len(cropped)} cropped points; expected {target}")
    voxel = float(cfg["voxel_size_m"])
    if voxel > 0:
        coordinates = np.floor(cropped / voxel).astype(np.int32)
        _, first = np.unique(coordinates, axis=0, return_index=True)
        candidates = cropped[np.sort(first)]
    else:
        candidates = cropped
    if len(candidates) < target:
        candidates = cropped
    selected = np.empty(target, dtype=np.int64)
    min_distance = np.full(len(candidates), np.inf, dtype=np.float32)
    farthest = int(np.argmax(np.sum((candidates - candidates.mean(axis=0)) ** 2, axis=1)))
    for index in range(target):
        selected[index] = farthest
        delta = candidates - candidates[farthest]
        np.minimum(min_distance, np.einsum("ij,ij->i", delta, delta), out=min_distance)
        farthest = int(np.argmax(min_distance))
    return candidates[selected].astype(np.float32, copy=False), len(cropped), len(candidates)


def ordered_state(messages: dict, cfg: dict) -> np.ndarray:
    names = cfg["state_names"]
    groups = (("left_arm", 0, 7), ("left_hand", 7, 27),
              ("right_arm", 27, 34), ("right_hand", 34, 54))
    parts = []
    for group, start, end in groups:
        msg = messages[group]
        if len(msg.name) != len(set(msg.name)) or len(msg.name) != len(msg.position):
            raise ValueError(f"Duplicate or misaligned {group} JointState")
        positions = dict(zip(msg.name, msg.position))
        missing = set(names[start:end]) - set(positions)
        if missing:
            raise ValueError(f"Missing {group} joints: {sorted(missing)}")
        parts.extend(positions[name] for name in names[start:end])
    result = np.asarray(parts, dtype=np.float32)
    if result.shape != (54,) or not np.isfinite(result).all():
        raise ValueError("Invalid 54-dimensional measured state")
    return result


class ObservationHistory:
    def __init__(self, period_ns: int = 100_000_000):
        self.period_ns = period_ns
        self.frames = deque(maxlen=2)

    def append(self, points: np.ndarray, state: np.ndarray, timestamp_ns: int):
        if points.shape != (4096, 3) or state.shape != (54,):
            raise ValueError("Invalid observation shape")
        if not np.isfinite(points).all() or not np.isfinite(state).all():
            raise ValueError("Nonfinite observation")
        if self.frames and timestamp_ns - self.frames[-1][2] != self.period_ns:
            self.frames.clear()
        self.frames.append((points, state, timestamp_ns))
        if len(self.frames) < 2:
            return None
        return {"point_cloud": np.stack([x[0] for x in self.frames]),
                "agent_pos": np.stack([x[1] for x in self.frames])}
