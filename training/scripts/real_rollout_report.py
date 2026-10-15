#!/usr/bin/env python3
"""Prepare and summarize DP3 real-rollout records. No ROS or control imports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


STAGES = (
    "approach_target", "direction_reasonable", "grasp_pose", "contact_or_grasp",
    "handoff_or_operation", "placement", "complete_task",
)
RESULTS = {"pending", "success", "fail", "human-stop", "safety-stop"}
CHECKPOINTS = {
    "epoch20": "epoch=0020.ckpt",
    "epoch7": "epoch=0007.ckpt",
}
EXPERIMENT = Path(__file__).resolve().parents[2] / (
    "training/data/outputs/harvest-tomato-ab-pre722-40-40-300-from-scratch-4096")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL {path}:{line_no}") from error


def finite_vector(value, length, where):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{where} must have {length} elements")
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        raise ValueError(f"{where} has nonfinite or nonnumeric elements")
    return value


def resolve_reference(rollout_dir: Path, value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (rollout_dir / path).resolve()


def init_rollout(args):
    if not args.rollout_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.rollout_id):
        raise ValueError("rollout-id must contain only letters, numbers, _ or -")
    if args.control_source == "dp3_via_safety_gateway" and not args.model_command_source:
        raise ValueError("A model-command source identity is required for an executed rollout")
    rollout_dir = (args.session_dir / args.rollout_id).resolve()
    rollout_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = EXPERIMENT / "checkpoints" / CHECKPOINTS[args.checkpoint_choice]
    manifest = {
        "schema_version": 1, "rollout_id": args.rollout_id,
        "checkpoint_choice": args.checkpoint_choice,
        "checkpoint": str(checkpoint), "weights": "ema_model",
        "control_source": args.control_source,
        "model_command_source": args.model_command_source,
        "prediction_log": args.prediction_log,
        "observation_dir": args.observation_dir,
        "rosbag_dir": args.rosbag_dir,
        "video_paths": args.video_path or [],
        "execution_events": args.execution_events,
        "note": args.note or "",
    }
    stages = {stage: {"result": "pending", "reason": ""} for stage in STAGES}
    (rollout_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (rollout_dir / "stages.json").write_text(json.dumps(stages, indent=2, ensure_ascii=False) + "\n")
    print(rollout_dir)


def summarize_rollout(rollout_dir: Path):
    manifest = json.loads((rollout_dir / "manifest.json").read_text(encoding="utf-8"))
    stages = json.loads((rollout_dir / "stages.json").read_text(encoding="utf-8"))
    if manifest.get("rollout_id") != rollout_dir.name or manifest.get("schema_version") != 1:
        raise ValueError(f"Invalid rollout manifest: {rollout_dir}")
    if manifest.get("checkpoint_choice") not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint choice: {rollout_dir}")
    if manifest.get("control_source") not in ("prediction_only", "dp3_via_safety_gateway"):
        raise ValueError(f"Unknown control source: {rollout_dir}")
    for stage in STAGES:
        if stage not in stages or stages[stage].get("result") not in RESULTS:
            raise ValueError(f"Missing/invalid stage result {stage}: {rollout_dir}")
        if stages[stage]["result"] in {"fail", "human-stop", "safety-stop"} and not stages[stage].get("reason"):
            raise ValueError(f"Failure/stop needs a reason for {stage}: {rollout_dir}")

    prediction_path = resolve_reference(rollout_dir, manifest.get("prediction_log"))
    execution_path = resolve_reference(rollout_dir, manifest.get("execution_events"))
    rosbag_path = resolve_reference(rollout_dir, manifest.get("rosbag_dir"))
    videos = [resolve_reference(rollout_dir, path) for path in manifest.get("video_paths", [])]
    predictions = []
    errors = []
    dry_run_stops = []
    latency = []
    magnitude = []
    cloud_means = []
    cloud_finite_ratios = []
    cloud_cropped_counts = []
    snapshot_count = 0
    if prediction_path and prediction_path.is_file():
        for event in read_jsonl(prediction_path):
            if "error" in event:
                errors.append(event["error"])
                continue
            if event.get("event") == "stopped":
                dry_run_stops.append(event)
                continue
            if event.get("event", "prediction") != "prediction":
                continue
            if event.get("sent_robot_command") is not False:
                raise ValueError("Prediction log must explicitly state sent_robot_command=false")
            if event.get("checkpoint_choice") not in (None, manifest["checkpoint_choice"]):
                raise ValueError("Prediction checkpoint differs from rollout manifest")
            if event.get("weights") != "ema_model":
                raise ValueError("Prediction did not identify EMA weights")
            action = event.get("predicted_action_chunk_rad", event.get("unnormalized_action_rad"))
            if not isinstance(action, list) or len(action) != 4:
                raise ValueError("Prediction chunk must have four rows")
            first = finite_vector(action[0], 54, "predicted first action")
            state = finite_vector(event.get("state_rad"), 54, "measured state")
            if event.get("input_shape") != {"point_cloud": [1, 2, 4096, 3],
                                             "agent_pos": [1, 2, 54]}:
                raise ValueError("Live input shape differs from the DP3 contract")
            cloud = event.get("point_cloud", {})
            if cloud.get("shape") != [4096, 3] or cloud.get("actual_points") != 4096:
                raise ValueError("Point cloud count/shape differs from the DP3 contract")
            cloud_means.append(finite_vector(cloud.get("xyz_mean"), 3, "point cloud XYZ mean"))
            finite_ratio = float(cloud.get("finite_ratio", 0))
            if not math.isfinite(finite_ratio) or finite_ratio < 1:
                raise ValueError("Point cloud contains nonfinite values")
            cloud_finite_ratios.append(finite_ratio)
            cloud_cropped_counts.append(int(cloud.get("cropped_points", 0)))
            magnitude.extend(abs(a - q) for a, q in zip(first, state))
            latency.append(float(event["inference_latency_ms"]))
            if not math.isfinite(latency[-1]):
                raise ValueError("Nonfinite inference latency")
            snapshot = resolve_reference(rollout_dir, event.get("observation_snapshot"))
            if snapshot:
                if not snapshot.is_file():
                    raise FileNotFoundError(snapshot)
                snapshot_count += 1
            predictions.append(event)

    status_events = []
    validated_events = []
    feedback_events = []
    hand_events = []
    camera_events = []
    if execution_path and execution_path.is_file():
        for event in read_jsonl(execution_path):
            event_type = event.get("event")
            if event_type == "safety_status":
                if (manifest["control_source"] == "dp3_via_safety_gateway"
                        and event.get("source") == manifest.get("model_command_source")):
                    status_events.append(event)
            elif event_type == "validated_action":
                finite_vector(event.get("arm_target_rad"), 14, "validated arm target")
                validated_events.append(event)
            elif event_type == "measured_feedback":
                finite_vector(event.get("state_rad"), 54, "measured feedback")
                feedback_events.append(event)
            elif event_type == "hand_command":
                finite_vector(event.get("hand_target_rad"), 20, "observed hand command")
                hand_events.append(event)
            elif event_type == "camera_timestamp":
                camera_events.append(event)
            else:
                raise ValueError(f"Unknown execution event type {event_type}: {execution_path}")
    faults = [event for event in status_events if event.get("faults")]
    model_status_available = bool(execution_path and execution_path.is_file()
                                  and manifest["control_source"] == "dp3_via_safety_gateway"
                                  and status_events)
    correction_count = (sum(event["corrected"] is True for event in status_events)
                        if status_events and all(isinstance(event.get("corrected"), bool)
                                                 for event in status_events) else None)
    stage_results = {stage: stages[stage]["result"] for stage in STAGES}
    result = stage_results["complete_task"]
    if result == "pending":
        if "safety-stop" in stage_results.values():
            result = "safety-stop"
        elif "human-stop" in stage_results.values():
            result = "human-stop"
        elif "fail" in stage_results.values():
            result = "fail"
    if result == "success" and any(value != "success" for value in stage_results.values()):
        raise ValueError(f"Complete-task success conflicts with an earlier stage: {rollout_dir}")
    return {
        "rollout_id": manifest["rollout_id"], "checkpoint_choice": manifest["checkpoint_choice"],
        "checkpoint": manifest["checkpoint"], "control_source": manifest["control_source"],
        "result": result, "stages": stages, "prediction_log": str(prediction_path) if prediction_path else None,
        "prediction_log_present": bool(prediction_path and prediction_path.is_file()),
        "prediction_count": len(predictions), "prediction_error_count": len(errors),
        "prediction_errors": errors[:10], "dry_run_stops": dry_run_stops,
        "observation_snapshot_count": snapshot_count,
        "prediction_first_step_abs_delta_mean_rad": statistics.mean(magnitude) if magnitude else None,
        "prediction_first_step_abs_delta_max_rad": max(magnitude) if magnitude else None,
        "point_cloud_mean_xyz_m": ([statistics.mean(row[axis] for row in cloud_means)
                                     for axis in range(3)] if cloud_means else None),
        "point_cloud_finite_ratio_min": min(cloud_finite_ratios) if cloud_finite_ratios else None,
        "point_cloud_cropped_count_min": min(cloud_cropped_counts) if cloud_cropped_counts else None,
        "inference_latency_ms_mean": statistics.mean(latency) if latency else None,
        "inference_latency_ms_p95": sorted(latency)[min(len(latency)-1, math.ceil(.95*len(latency))-1)] if latency else None,
        "safety_status_count": len(status_events) if model_status_available else None,
        "safety_rejection_count": len(faults) if model_status_available else None,
        "safety_correction_count": correction_count,
        "validated_arm_action_count": len(validated_events) if execution_path and execution_path.is_file() else None,
        "measured_feedback_count": len(feedback_events) if execution_path and execution_path.is_file() else None,
        "observed_hand_command_count": len(hand_events) if execution_path and execution_path.is_file() else None,
        "camera_timestamp_count": len(camera_events) if execution_path and execution_path.is_file() else None,
        "execution_events": str(execution_path) if execution_path else None,
        "rosbag_dir": str(rosbag_path) if rosbag_path else None,
        "rosbag_present": rosbag_path.is_dir() if rosbag_path else False,
        "video_paths": [str(path) for path in videos],
        "video_present": [path.exists() for path in videos],
        # Matching a particular DP3 prediction to an accepted robot command
        # requires a future authorized bridge to emit an explicit command ID.
        # Merely seeing validated/feedback topics cannot prove DP3 executed.
        "model_action_execution_link_verified": None,
    }


def summarize_session(args):
    session = args.session_dir.resolve()
    rollouts = [summarize_rollout(path.parent) for path in sorted(session.glob("*/manifest.json"))]
    categories = {
        "successful_rollouts": [r for r in rollouts if r["result"] == "success"],
        "failed_rollouts": [r for r in rollouts if r["result"] == "fail"],
        "stopped_rollouts": [r for r in rollouts if r["result"] in ("human-stop", "safety-stop")],
        "pending_rollouts": [r for r in rollouts if r["result"] == "pending"],
    }
    scored = [r for r in rollouts if r["result"] != "pending"]
    prediction_total = sum(r["prediction_count"] for r in rollouts)
    def weighted_prediction_mean(field):
        return (sum(r["prediction_count"] * r[field] for r in rollouts
                    if r["prediction_count"] and r[field] is not None) / prediction_total
                if prediction_total else None)
    stage_rates = {}
    for stage in STAGES:
        values = [r["stages"][stage]["result"] for r in rollouts]
        denominator = sum(v != "pending" for v in values)
        stage_rates[stage] = {
            "success": values.count("success"), "fail": values.count("fail"),
            "human_stop": values.count("human-stop"), "safety_stop": values.count("safety-stop"),
            "pending": values.count("pending"),
            "success_rate": values.count("success") / denominator if denominator else None,
        }
    result = {
        "schema_version": 1, "rollout_count": len(rollouts),
        "scored_rollout_count": len(scored),
        "prediction_count": prediction_total,
        "mean_inference_latency_ms": weighted_prediction_mean("inference_latency_ms_mean"),
        "mean_first_step_abs_delta_rad": weighted_prediction_mean(
            "prediction_first_step_abs_delta_mean_rad"),
        "complete_success_rate": len(categories["successful_rollouts"]) / len(scored) if scored else None,
        "human_stop_count": sum(r["result"] == "human-stop" for r in rollouts),
        "safety_stop_count": sum(r["result"] == "safety-stop" for r in rollouts),
        "stage_rates": stage_rates,
        "safety_rejection_count": sum(r["safety_rejection_count"] for r in rollouts if r["safety_rejection_count"] is not None),
        "safety_rejection_observed_rollouts": sum(r["safety_rejection_count"] is not None for r in rollouts),
        "safety_correction_count": sum(r["safety_correction_count"] for r in rollouts if r["safety_correction_count"] is not None),
        "safety_correction_observed_rollouts": sum(r["safety_correction_count"] is not None for r in rollouts),
        "rollouts": rollouts,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, entries in categories.items():
        (args.output / f"{name}.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rollouts"}, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--session-dir", type=Path, required=True)
    init.add_argument("--rollout-id", required=True)
    init.add_argument("--checkpoint-choice", choices=tuple(CHECKPOINTS), default="epoch20")
    init.add_argument("--control-source", choices=("prediction_only", "dp3_via_safety_gateway"), default="prediction_only")
    init.add_argument("--model-command-source",
                      help="Future authorized ArmCommand.source; never inferred from another source")
    init.add_argument("--prediction-log")
    init.add_argument("--observation-dir")
    init.add_argument("--rosbag-dir")
    init.add_argument("--execution-events")
    init.add_argument("--video-path", action="append")
    init.add_argument("--note")
    init.set_defaults(func=init_rollout)
    summary = sub.add_parser("summarize")
    summary.add_argument("--session-dir", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.set_defaults(func=summarize_session)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
