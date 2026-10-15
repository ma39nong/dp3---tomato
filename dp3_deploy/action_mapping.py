"""Pure DP3 action-to-robot joint-name mapping for logs and review only."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def load_mapping(bundle: Path = ROOT) -> list[dict]:
    bundle = Path(bundle)
    cfg = json.loads((bundle / "config/deploy.json").read_text())
    with (bundle / "config/ros_to_dp3_joint_mapping.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 54 or [int(row["index"]) for row in rows] != list(range(54)):
        raise ValueError("Mapping must contain ordered indices 0..53")
    if [row["joint_name"] for row in rows] != cfg["action_names"]:
        raise ValueError("Mapping differs from checkpoint action order")
    if [row["state_source_joint_name"] for row in rows] != cfg["state_names"]:
        raise ValueError("Mapping differs from measured state order")
    if len(set(row["action_source_joint_name"] for row in rows)) != 54:
        raise ValueError("Duplicate robot action joint names")
    return rows


def preview_action(action_chunk: np.ndarray, mapping: list[dict]) -> list[dict]:
    """Return named absolute targets; the result is not a ROS command."""
    chunk = np.asarray(action_chunk, dtype=np.float32)
    if chunk.shape != (4, 54) or not np.isfinite(chunk).all():
        raise ValueError("Expected finite [4,54] action chunk")
    if len(mapping) != 54:
        raise ValueError("Invalid mapping")
    result = []
    for step in chunk:
        parts = {}
        for side, device in (("left", "arm"), ("left", "hand"),
                             ("right", "arm"), ("right", "hand")):
            rows = [row for row in mapping if row["side"] == side and row["device"] == device]
            expected = 7 if device == "arm" else 20
            if len(rows) != expected:
                raise ValueError(f"Invalid {side} {device} mapping")
            parts[f"{side}_{device}"] = {
                "names": [row["action_source_joint_name"] for row in rows],
                "absolute_targets_rad": [float(step[int(row["index"])]) for row in rows],
            }
        result.append(parts)
    return result
