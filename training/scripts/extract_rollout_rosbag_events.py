#!/usr/bin/env python3
"""Read an existing ROS2 bag into rollout execution events; never publishes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "training/deployment/robot_source_contract.json"
MAPPING = ROOT / "training/deployment/ros_to_dp3_joint_mapping.csv"
STATE_GROUPS = ("left_arm_state", "left_hand_state", "right_arm_state", "right_hand_state")


def stamp_ns(message):
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def mapped_positions(message, rows, column):
    if len(message.name) != len(set(message.name)) or len(message.name) != len(message.position):
        raise ValueError("LIVE CONTRACT MISMATCH: duplicate/misaligned JointState names")
    named = dict(zip(message.name, message.position))
    expected = [row[column] for row in rows]
    if not set(expected).issubset(named):
        raise ValueError(f"LIVE CONTRACT MISMATCH: missing {sorted(set(expected)-set(named))}")
    values = [float(named[name]) for name in expected]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("LIVE CONTRACT MISMATCH: nonfinite joint position")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="Finalized ROS2 bag directory")
    parser.add_argument("--output", type=Path, required=True, help="Fresh execution_events.jsonl")
    args = parser.parse_args()
    if not args.bag.is_dir() or not (args.bag / "metadata.yaml").is_file():
        parser.error("Expected a finalized ROS2 bag with metadata.yaml")
    if args.output.exists():
        parser.error("Choose a new output file; existing event logs are never overwritten")
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        parser.error("A partial extraction already exists; inspect it before retrying")
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError as error:
        raise SystemExit("ROS2 bag libraries unavailable; run on the ROS2 host") from error

    contract = json.loads(CONTRACT.read_text())
    with MAPPING.open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row["index"]))
    if [int(row["index"]) for row in rows] != list(range(54)):
        raise ValueError("DP3 joint mapping is not exactly 54 ordered dimensions")
    topics = contract["topics"]
    selected = {entry["topic"]: (name, entry["type"])
                for name, entry in topics.items()
                if name in (*STATE_GROUPS, "validated_arm_action", "left_hand_action",
                            "right_hand_action", "cam0_depth")}
    # ArmCommandStatus is recorded by the robot-side recorder but is not a
    # training observation. It carries source/session/sequence/faults.
    selected["/teleop/arm_command_status"] = (
        "arm_command_status", "teleop_interfaces/msg/ArmCommandStatus")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag.resolve()), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    actual = {item.name: item.type for item in reader.get_all_topics_and_types()}
    for topic, (_, expected_type) in selected.items():
        if actual.get(topic) != expected_type:
            raise ValueError(f"LIVE CONTRACT MISMATCH: {topic}: expected {expected_type}, got {actual.get(topic)}")
    classes = {topic: get_message(expected_type) for topic, (_, expected_type) in selected.items()}
    state_cache = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("x", encoding="utf-8") as out:
        while reader.has_next():
            topic, serialized, received_ns = reader.read_next()
            if topic not in selected:
                continue
            kind = selected[topic][0]
            message = deserialize_message(serialized, classes[topic])
            timestamp = stamp_ns(message)
            if timestamp <= 0:
                raise ValueError(f"LIVE CONTRACT MISMATCH: no source stamp on {topic}")
            event = None
            if kind in STATE_GROUPS:
                group = kind.removesuffix("_state")
                group_rows = [row for row in rows if row["side"] + "_" + row["device"] == group]
                state_cache[kind] = (timestamp, mapped_positions(message, group_rows,
                                                                  "state_source_joint_name"))
                if len(state_cache) == 4:
                    stamps = [state_cache[name][0] for name in STATE_GROUPS]
                    if max(stamps) - min(stamps) <= 150_000_000:
                        values = [None] * 54
                        for state_kind in STATE_GROUPS:
                            group_name = state_kind.removesuffix("_state")
                            group_rows = [row for row in rows
                                          if row["side"] + "_" + row["device"] == group_name]
                            for row, value in zip(group_rows, state_cache[state_kind][1]):
                                values[int(row["index"])] = value
                        if any(value is None for value in values):
                            raise ValueError("Incomplete 54-dimensional measured feedback")
                        event = {"event": "measured_feedback", "timestamp_ns": max(stamps),
                                 "source_timestamps_ns": dict(zip(STATE_GROUPS, stamps)),
                                 "state_rad": values}
            elif kind == "validated_arm_action":
                arm_rows = [row for row in rows if row["device"] == "arm"]
                event = {"event": "validated_action", "timestamp_ns": timestamp,
                         "arm_target_rad": mapped_positions(message, arm_rows,
                                                            "action_source_joint_name"),
                         "meaning": "Safety Gateway validated arm target; DP3 linkage unverified"}
            elif kind in ("left_hand_action", "right_hand_action"):
                side = "left" if kind.startswith("left") else "right"
                hand_rows = [row for row in rows if row["side"] == side and row["device"] == "hand"]
                event = {"event": "hand_command", "timestamp_ns": timestamp,
                         "side": side, "hand_target_rad": mapped_positions(
                             message, hand_rows, "action_source_joint_name"),
                         "meaning": "Observed hand command; execution not proven"}
            elif kind == "arm_command_status":
                event = {"event": "safety_status", "timestamp_ns": timestamp,
                         "source": str(message.source), "session_id": str(message.session_id),
                         "sequence": int(message.sequence),
                         "accepted_sides": list(message.accepted_sides),
                         "faults": list(message.faults),
                         "corrected": None}
            elif kind == "cam0_depth":
                event = {"event": "camera_timestamp", "timestamp_ns": timestamp,
                         "frame_id": str(message.header.frame_id)}
            if event is not None:
                event["topic"] = topic
                event["rosbag_receive_timestamp_ns"] = int(received_ns)
                out.write(json.dumps(event, ensure_ascii=False) + "\n")
    partial.rename(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
