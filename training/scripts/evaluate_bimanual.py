#!/usr/bin/env python3
"""Offline checkpoint audit for the bimanual DP3 dataset."""
from __future__ import annotations

import argparse
import csv
import dill
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import zarr


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "3D-Diffusion-Policy"
sys.path.insert(0, str(PROJECT))

import hydra  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)


GROUPS = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 27),
    "right_arm": slice(27, 34),
    "right_hand": slice(34, 54),
}


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    error = prediction - target
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error": float(np.max(np.abs(error))),
        "per_joint_mae": np.mean(np.abs(error), axis=(0, 1)).tolist(),
        "per_joint_rmse": np.sqrt(np.mean(np.square(error), axis=(0, 1))).tolist(),
        "groups": {
            name: {
                "mae": float(np.mean(np.abs(error[..., columns]))),
                "rmse": float(np.sqrt(np.mean(np.square(error[..., columns])))),
            }
            for name, columns in GROUPS.items()
        },
    }


def training_action_stats(dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = np.asarray(dataset.replay_buffer["action"][:], dtype=np.float32)
    ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    selected = np.concatenate([
        actions[start:end]
        for keep, start, end in zip(dataset.train_mask, starts, ends)
        if keep
    ])
    return selected.mean(axis=0), selected.min(axis=0), selected.max(axis=0)


@torch.inference_mode()
def evaluate_split(policy, dataset, device, batch_size, seed, train_mean,
                   train_min, train_max, ablation_batches):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions, targets, mean_predictions, hold_predictions = [], [], [], []
    pc_deltas, state_deltas = [], []
    start = policy.n_obs_steps - 1
    end = start + policy.n_action_steps

    for batch_index, batch in enumerate(loader):
        obs = {key: value.to(device) for key, value in batch["obs"].items()}
        target = batch["action"][:, start:end].numpy()
        torch.manual_seed(seed + batch_index)
        prediction = policy.predict_action(obs)["action"].cpu().numpy()

        predictions.append(prediction)
        targets.append(target)
        mean_predictions.append(np.broadcast_to(train_mean, target.shape).copy())
        current_state = batch["obs"]["agent_pos"][:, start:start + 1].numpy()
        hold_predictions.append(np.repeat(current_state, policy.n_action_steps, axis=1))

        if batch_index < ablation_batches and len(target) > 1:
            order = torch.roll(torch.arange(len(target), device=device), 1)
            shuffled_pc = dict(obs)
            shuffled_pc["point_cloud"] = obs["point_cloud"][order]
            torch.manual_seed(seed + batch_index)
            pc_prediction = policy.predict_action(shuffled_pc)["action"].cpu().numpy()
            pc_deltas.append(np.abs(pc_prediction - prediction))

            shuffled_state = dict(obs)
            shuffled_state["agent_pos"] = obs["agent_pos"][order]
            torch.manual_seed(seed + batch_index)
            state_prediction = policy.predict_action(shuffled_state)["action"].cpu().numpy()
            state_deltas.append(np.abs(state_prediction - prediction))

    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    result = {
        "sequences": int(len(target)),
        "model": metrics(prediction, target),
        "train_mean_baseline": metrics(np.concatenate(mean_predictions), target),
        "hold_current_state_baseline": metrics(np.concatenate(hold_predictions), target),
        "range_violation_fraction": float(np.mean(
            (prediction < train_min[None, None]) | (prediction > train_max[None, None])
        )),
        "input_sensitivity": {
            "batches": min(ablation_batches, len(loader)),
            "shuffle_point_cloud_mean_abs_action_change": (
                float(np.mean(np.concatenate(pc_deltas))) if pc_deltas else None
            ),
            "shuffle_state_mean_abs_action_change": (
                float(np.mean(np.concatenate(state_deltas))) if state_deltas else None
            ),
        },
    }
    result["improvement_vs_train_mean_percent"] = 100.0 * (
        1.0 - result["model"]["rmse"] / result["train_mean_baseline"]["rmse"]
    )
    result["improvement_vs_hold_state_percent"] = 100.0 * (
        1.0 - result["model"]["rmse"] / result["hold_current_state_baseline"]["rmse"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "validation" / "offline")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation-batches", type=int, default=8)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu",
                         mmap=True, weights_only=False)
    cfg = payload["cfg"]
    if args.dataset:
        OmegaConf.update(cfg, "task.dataset.zarr_path", str(args.dataset.expanduser().resolve()))

    epoch = dill.loads(payload["pickles"]["epoch"])
    global_step = dill.loads(payload["pickles"]["global_step"])
    state_key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][state_key])
    del payload
    gc.collect()

    device = torch.device(args.device)
    policy.to(device).eval()
    finite_parameters = all(
        bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()
    )
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation_dataset = dataset.get_validation_dataset()
    train_mean, train_min, train_max = training_action_stats(dataset)

    report = {
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(epoch),
        "checkpoint_global_step": int(global_step),
        "weights": state_key,
        "finite_parameters": finite_parameters,
        "configured_epochs": int(cfg.training.num_epochs),
        "dataset": str(Path(cfg.task.dataset.zarr_path).resolve()),
        "episode_split": {
            "train": np.flatnonzero(dataset.train_mask).tolist(),
            "validation": np.flatnonzero(~dataset.train_mask).tolist(),
        },
        "shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
        "use_pc_color": bool(cfg.policy.use_pc_color),
    }
    report["train"] = evaluate_split(
        policy, dataset, device, args.batch_size, args.seed,
        train_mean, train_min, train_max, args.ablation_batches,
    )
    report["validation"] = evaluate_split(
        policy, validation_dataset, device, args.batch_size, args.seed + 100000,
        train_mean, train_min, train_max, args.ablation_batches,
    )

    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    data_root = zarr.open_group(str(Path(cfg.task.dataset.zarr_path).resolve()), mode="r")
    names = data_root.attrs.get("action_names", [f"joint_{i}" for i in range(54)])
    with (args.output / "per_joint.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "joint", "train_mae", "train_rmse", "validation_mae", "validation_rmse"])
        for index, name in enumerate(names):
            writer.writerow([
                index, name,
                report["train"]["model"]["per_joint_mae"][index],
                report["train"]["model"]["per_joint_rmse"][index],
                report["validation"]["model"]["per_joint_mae"][index],
                report["validation"]["model"]["per_joint_rmse"][index],
            ])

    print(json.dumps({
        "report": str(report_path),
        "checkpoint_epoch": report["checkpoint_epoch"],
        "train_rmse": report["train"]["model"]["rmse"],
        "validation_rmse": report["validation"]["model"]["rmse"],
        "validation_mean_baseline_rmse": report["validation"]["train_mean_baseline"]["rmse"],
        "validation_hold_baseline_rmse": report["validation"]["hold_current_state_baseline"]["rmse"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
