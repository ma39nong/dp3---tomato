#!/usr/bin/env python3
"""Read-only EMA replay of the 300-epoch harvest experiment on validation bags.

This script has no robot transport or command publisher. It writes predictions
and comparisons beside neither the training data nor the training experiment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch.utils.data import DataLoader
import zarr


ROOT = Path(__file__).resolve().parents[1]
DP3 = ROOT / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3))
OmegaConf.register_new_resolver("eval", eval, replace=True)


GROUPS = {"left_arm": range(0, 7), "left_hand": range(7, 27),
          "right_arm": range(27, 34), "right_hand": range(34, 54)}


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    delta = prediction - target
    return {"count": int(len(delta)), "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta))))}


def motion_score(actions: np.ndarray) -> np.ndarray:
    """RMS change in GT absolute joint targets between adjacent 10 Hz steps."""
    return np.sqrt(np.mean(np.square(np.diff(actions, axis=1)), axis=(1, 2)))


def sampler_targets(dataset, action_steps: int, obs_steps: int) -> np.ndarray:
    """Read action chunks without loading point clouds, honoring sequence padding."""
    indices = np.asarray(dataset.sampler.indices, dtype=np.int64)
    offsets = np.arange(obs_steps - 1, obs_steps - 1 + action_steps)
    frame_indices = indices[:, 0, None] + offsets - indices[:, 2, None]
    frame_indices = np.clip(frame_indices, indices[:, 0, None], indices[:, 1, None] - 1)
    return np.asarray(dataset.replay_buffer["action"][:], dtype=np.float32)[frame_indices]


def cloud_stats(root, frames: np.ndarray) -> dict:
    arrays = [np.asarray(root["data/point_cloud"][int(i)], dtype=np.float32) for i in frames]
    points = np.stack(arrays)
    return {"frames": len(frames), "shape_per_frame": list(points.shape[1:]),
            "xyz_min": points.min(axis=(0, 1)).tolist(),
            "xyz_max": points.max(axis=(0, 1)).tolist(),
            "xyz_mean": points.mean(axis=(0, 1)).tolist(),
            "xyz_std": points.std(axis=(0, 1)).tolist(),
            "finite_ratio": float(np.isfinite(points).mean()),
            "actual_points_per_frame": int(points.shape[1])}


def select_bags(names: np.ndarray, scores: np.ndarray) -> dict[str, str]:
    unique = sorted(set(map(str, names)))
    selected = {}
    for source in ("A", "B"):
        candidates = [bag for bag in unique if bag.startswith(source + "/")]
        if not candidates:
            raise ValueError(f"No {source} validation bag")
        selected[source] = max(candidates, key=lambda bag: int(np.count_nonzero(names == bag)))
    remaining = [bag for bag in unique if bag not in selected.values()]
    if len(remaining) < 2:
        raise ValueError("Need at least four distinct validation bags")
    by_bag = {bag: float(np.median(scores[names == bag])) for bag in remaining}
    selected["moving"] = max(by_bag, key=by_bag.get)
    remaining.remove(selected["moving"])
    selected["low_motion"] = min(remaining, key=by_bag.get)
    return selected


def write_bag_replay(output: Path, label: str, bag: str, sample_indices: np.ndarray,
                     frame_indices: np.ndarray, prediction: np.ndarray, target: np.ndarray,
                     hold: np.ndarray, scores: np.ndarray, names: list[str]) -> dict:
    indices = sample_indices
    csv_path = output / f"{label}_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "global_frame", "gt_motion_score_rad",
                         *[f"gt_{name}" for name in names],
                         *[f"pred_{name}" for name in names],
                         *[f"hold_{name}" for name in names]])
        for i in indices:
            writer.writerow([int(i), int(frame_indices[i]), float(scores[i]),
                             *target[i, 0].tolist(), *prediction[i, 0].tolist(),
                             *hold[i, 0].tolist()])
    figure = make_subplots(rows=4, cols=1, shared_xaxes=True,
                           subplot_titles=list(GROUPS))
    for row, (group, columns) in enumerate(GROUPS.items(), 1):
        joint = max(columns, key=lambda column: float(np.ptp(target[indices, 0, column])))
        for title, values in (("GT", target), ("EMA DP3", prediction), ("hold", hold)):
            figure.add_trace(go.Scatter(x=frame_indices[indices], y=values[indices, 0, joint],
                                        mode="lines", name=f"{group} {title}",
                                        legendgroup=title), row=row, col=1)
        figure.layout.annotations[row - 1].text = f"{group}: {names[joint]}"
    figure.update_layout(title=f"{label}: {bag} (first action of each chunk; rad)",
                         height=1100)
    html_path = output / f"{label}_trajectory.html"
    figure.write_html(html_path, include_plotlyjs="cdn")
    return {"bag": bag, "sequences": len(indices), "csv": str(csv_path),
            "plot": str(html_path), "first_step_dp3": error_metrics(prediction[indices, 0], target[indices, 0]),
            "first_step_hold": error_metrics(hold[indices, 0], target[indices, 0]),
            "median_gt_motion_score_rad": float(np.median(scores[indices]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Replay output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", mmap=True,
                         weights_only=False, pickle_module=dill)
    if payload.get("format_version") != 2 or "ema_model" not in payload:
        raise ValueError("Expected a harvest 300 format-v2 checkpoint with EMA weights")
    cfg = payload["cfg"]
    dataset_path = Path(cfg.task.dataset.zarr_path).resolve(strict=True)
    root = zarr.open_group(str(dataset_path), mode="r")
    if root.attrs.get("action_semantics") != "absolute_joint_target":
        raise ValueError("Unknown action semantics")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["ema_model"], strict=True)
    epoch = int(payload["next_epoch"])
    del payload
    gc.collect()
    device = torch.device(args.device)
    policy.to(device).eval()
    if next(policy.parameters()).dtype != torch.float32:
        raise ValueError("Expected float32 model weights")
    if tuple(cfg.task.shape_meta.obs.point_cloud.shape) != (4096, 3):
        raise ValueError("Checkpoint does not expect 4096 XYZ points")
    if int(cfg.n_obs_steps) != 2 or int(cfg.n_action_steps) != 4:
        raise ValueError("Unexpected observation/action horizon")
    train_actions = sampler_targets(dataset, int(cfg.n_action_steps), int(cfg.n_obs_steps))
    train_scores = motion_score(train_actions)
    train_threshold = float(np.quantile(train_scores, 0.25))
    strong_threshold = float(np.quantile(train_scores, 0.75))
    del train_actions

    loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    predictions, targets, holds = [], [], []
    latency = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            obs = {key: value.to(device) for key, value in batch["obs"].items()}
            if (obs["point_cloud"].shape[1:] != (int(cfg.horizon), 4096, 3)
                    or obs["agent_pos"].shape[1:] != (int(cfg.horizon), 54)):
                raise ValueError("Unexpected observation shape")
            torch.manual_seed(args.seed + 100000 + batch_index)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            prediction = policy.predict_action(obs)["action"]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency.append(time.perf_counter() - start)
            pred = prediction.cpu().numpy()
            if pred.shape[1:] != (4, 54) or not np.isfinite(pred).all():
                raise ValueError("Invalid prediction shape or nonfinite action")
            target = batch["action"][:, 1:5].numpy()
            hold = np.repeat(batch["obs"]["agent_pos"][:, 1:2].numpy(), 4, axis=1)
            predictions.append(pred)
            targets.append(target)
            holds.append(hold)
    prediction, target, hold = (np.concatenate(x) for x in (predictions, targets, holds))
    scores = motion_score(target)
    low = scores <= train_threshold
    strong = scores >= strong_threshold
    indices = np.asarray(validation.sampler.indices, dtype=np.int64)
    ends = np.asarray(validation.replay_buffer.episode_ends[:], dtype=np.int64)
    bag_names = np.asarray(validation.source_episode_names, dtype=object)
    episode_ids = np.searchsorted(ends, indices[:, 0], side="right")
    sample_bags = bag_names[episode_ids]
    frame_indices = np.clip(indices[:, 0] + 1 - indices[:, 2], indices[:, 0], indices[:, 1] - 1)
    selected = select_bags(sample_bags, scores)
    np.savez_compressed(output / "validation_predictions.npz", prediction=prediction,
                        target=target, hold=hold, motion_score_rad=scores,
                        global_frame=frame_indices, source_bag=sample_bags.astype(str))
    joint_names = list(root.attrs["action_names"])
    bag_replays = {label: write_bag_replay(output, label, bag,
                  np.flatnonzero(sample_bags == bag), frame_indices,
                  prediction, target, hold, scores, joint_names)
                  for label, bag in selected.items()}
    train_episodes = np.flatnonzero(dataset.train_mask)
    val_episodes = np.flatnonzero(~dataset.train_mask)
    starts = np.r_[0, ends[:-1]]
    rng = np.random.default_rng(args.seed)
    def sampled_frames(episodes):
        frames = np.concatenate([np.arange(starts[i], ends[i]) for i in episodes])
        return rng.choice(frames, size=min(64, len(frames)), replace=False)
    report = {
        "checkpoint": str(checkpoint), "checkpoint_epoch": epoch,
        "weights": "ema_model", "normalizer": "checkpoint EMA state_dict",
        "dataset": str(dataset_path), "validation_sequences": len(prediction),
        "action_shape": list(prediction.shape[1:]),
        "motion_definition": "RMS consecutive GT absolute-action change over 4 steps and 54 joints, rad",
        "low_motion_threshold_rad": train_threshold,
        "strong_motion_threshold_rad": strong_threshold,
        "threshold_source": "25th percentile of training sequence motion scores",
        "all": {"dp3": error_metrics(prediction, target), "hold": error_metrics(hold, target)},
        "low_motion": {"dp3": error_metrics(prediction[low], target[low]),
                       "hold": error_metrics(hold[low], target[low])},
        "moving": {"dp3": error_metrics(prediction[~low], target[~low]),
                   "hold": error_metrics(hold[~low], target[~low])},
        "strong_motion": {"dp3": error_metrics(prediction[strong], target[strong]),
                          "hold": error_metrics(hold[strong], target[strong])},
        "by_source": {source: {"dp3": error_metrics(prediction[np.char.startswith(sample_bags.astype(str), source + "/")],
                                                  target[np.char.startswith(sample_bags.astype(str), source + "/")]),
                               "hold": error_metrics(hold[np.char.startswith(sample_bags.astype(str), source + "/")],
                                                     target[np.char.startswith(sample_bags.astype(str), source + "/")])}
                      for source in ("A", "B")},
        "prediction_outside_train_action_range_fraction": float(np.mean(
            (prediction < np.asarray(dataset.replay_buffer["action"][:])[np.concatenate([
                np.arange(starts[i], ends[i]) for i in train_episodes])].min(axis=0)) |
            (prediction > np.asarray(dataset.replay_buffer["action"][:])[np.concatenate([
                np.arange(starts[i], ends[i]) for i in train_episodes])].max(axis=0)))),
        "mean_batch_inference_latency_ms": float(np.mean(latency) * 1000),
        "p95_batch_inference_latency_ms": float(np.quantile(latency, 0.95) * 1000),
        "point_cloud_train_sample": cloud_stats(root, sampled_frames(train_episodes)),
        "point_cloud_val_sample": cloud_stats(root, sampled_frames(val_episodes)),
        "selected_bags": bag_replays,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("checkpoint_epoch", "validation_sequences",
                                                    "low_motion_threshold_rad", "all",
                                                    "low_motion", "moving", "strong_motion", "selected_bags")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
