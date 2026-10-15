#!/usr/bin/env python3
"""Offline EMA action-space diagnostics; no ROS or robot control imports."""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/3D-Diffusion-Policy"))
OmegaConf.register_new_resolver("eval", eval, replace=True)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer  # noqa: E402

ARM = np.r_[0:7, 27:34]
EXPERIMENT = ROOT / "training/data/outputs/harvest-tomato-ab-pre722-40-40-300-from-scratch-4096"
DATASET = ROOT / "outputs/harvest_tomato_AB_pre722_40_40/AB_committed_4096.zarr"
REPLAY = ROOT / "outputs/deployment_offline_epoch20_final/validation_predictions.npz"
REPLAY_EPOCH7 = ROOT / "outputs/deployment_offline_epoch7_matched/validation_predictions.npz"


def stats(values: np.ndarray) -> dict:
    return {"mean": float(values.mean()), "std": float(values.std()),
            "median": float(np.median(values)), "p5": float(np.quantile(values, .05)),
            "p95": float(np.quantile(values, .95))}


def error(pred: np.ndarray, gt: np.ndarray) -> dict:
    delta = pred - gt
    return {"mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.square(delta).mean()))}


def make_split(root):
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    names = list(map(str, root.attrs["source_episode_names"]))
    groups = list(dict.fromkeys(names))
    n_val = min(max(1, round(len(groups) * .1)), len(groups) - 1)
    chosen = np.random.default_rng(42).choice(len(groups), size=n_val, replace=False)
    val_names = {groups[i] for i in chosen}
    val_mask = np.array([n in val_names for n in names])
    bounds = np.r_[0, ends]
    train_frames = np.concatenate([np.arange(bounds[i], bounds[i+1]) for i in range(len(ends)) if not val_mask[i]])
    val_frames = np.concatenate([np.arange(bounds[i], bounds[i+1]) for i in range(len(ends)) if val_mask[i]])
    if (len(train_frames), len(val_frames)) != (66465, 7875):
        raise ValueError("Train/validation split differs from the 300-epoch experiment")
    return ends, train_frames, val_frames


def sample_indices(replay, count=64):
    """Deterministic A/B and motion-stratified validation subset."""
    result = []
    bags = replay["source_bag"].astype(str)
    scores = replay["motion_score_rad"]
    for prefix, n in (("A/", count // 4), ("B/", count - count // 4)):
        candidates = np.flatnonzero(np.char.startswith(bags, prefix))
        ordered = candidates[np.argsort(scores[candidates], kind="stable")]
        result.extend(ordered[np.linspace(0, len(ordered)-1, n, dtype=int)].tolist())
    return np.asarray(result, dtype=np.int64)


def observations(root, ends, frames):
    starts = np.r_[0, ends[:-1]]
    episode = np.searchsorted(ends, frames, side="right")
    previous = np.maximum(frames - 1, starts[episode])
    pairs = np.column_stack((previous, frames))
    unique = np.unique(pairs)
    cloud = {int(i): np.asarray(root["data/point_cloud"][int(i)], dtype=np.float32) for i in unique}
    states = np.asarray(root["data/state"][:], dtype=np.float32)
    obs = {"point_cloud": np.stack([[cloud[int(j)] for j in pair] for pair in pairs]),
           "agent_pos": states[pairs]}
    return {key: torch.from_numpy(value) for key, value in obs.items()}


def load_ema(epoch, device):
    path = EXPERIMENT / "checkpoints" / f"epoch={epoch:04d}.ckpt"
    payload = torch.load(path, map_location="cpu", mmap=True,
                         weights_only=False, pickle_module=dill)
    if payload.get("format_version") != 2 or "ema_model" not in payload:
        raise ValueError(f"Invalid EMA checkpoint: {path}")
    cfg = payload["cfg"]
    if (int(cfg.horizon), int(cfg.n_obs_steps), int(cfg.n_action_steps)) != (16, 2, 4):
        raise ValueError("Unexpected horizon/observation/action-step configuration")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["ema_model"], strict=True)
    del payload
    gc.collect()
    return policy.to(device).eval()


def predict(policy, obs, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        return policy.predict_action(obs)["action"][:, 0].detach().cpu().numpy()


def normalizer_check(policy, obs, gt, train_action, train_state):
    state = obs["agent_pos"]
    point = obs["point_cloud"]
    action = torch.from_numpy(gt).to(state.device)
    norm = policy.normalizer
    a_norm = norm["action"].normalize(action)
    a_back = norm["action"].unnormalize(a_norm)
    s_norm = norm["agent_pos"].normalize(state)
    s_back = norm["agent_pos"].unnormalize(s_norm)
    p_norm = norm["point_cloud"].normalize(point)
    p_back = norm["point_cloud"].unnormalize(p_norm)
    ref = LinearNormalizer()
    ref.fit({"action": train_action, "agent_pos": train_state}, last_n_dims=1)
    output = {"action_raw_first_14": action[0, ARM].tolist(),
              "action_normalized_first_14": a_norm[0, ARM].tolist(),
              "action_inverse_first_14": a_back[0, ARM].tolist(),
              "action_roundtrip_max_abs": float((a_back-action).abs().max()),
              "state_roundtrip_max_abs": float((s_back-state).abs().max()),
              "point_cloud_roundtrip_max_abs": float((p_back-point).abs().max()),
              "checkpoint_vs_train_fit": {}}
    for key in ("action", "agent_pos"):
        output["checkpoint_vs_train_fit"][key] = {
            param: float((norm[key].params_dict[param].cpu() - ref[key].params_dict[param]).abs().max())
            for param in ("scale", "offset")}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("Need at least two repeat samples")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(DATASET), mode="r")
    replay = np.load(REPLAY)
    replay_epoch7 = np.load(REPLAY_EPOCH7)
    for key in ("global_frame", "target", "hold", "source_bag"):
        if not np.array_equal(replay[key], replay_epoch7[key]):
            raise ValueError(f"Epoch 7/20 full-validation replays are not aligned: {key}")
    ends, train_frames, val_frames = make_split(root)
    picked = sample_indices(replay)
    frames = replay["global_frame"][picked]
    gt = replay["target"][picked, 0].astype(np.float32)
    q = replay["hold"][picked, 0].astype(np.float32)
    if not np.array_equal(gt, root["data/action"][frames]) or not np.array_equal(q, root["data/state"][frames]):
        raise ValueError("Replay first action/state differs from final Zarr")
    train_action = np.asarray(root["data/action"][:], dtype=np.float32)[train_frames]
    # Compare prediction and GT distributions on the identical replay frames.
    val_action = replay["target"][:, 0].astype(np.float32)
    train_state = np.asarray(root["data/state"][:], dtype=np.float32)[train_frames]
    obs = observations(root, ends, frames)
    device = torch.device(args.device)
    obs = {key: value.to(device) for key, value in obs.items()}
    names = list(root.attrs["action_names"])
    summary = {"sample_count": len(picked), "sample_global_frames": frames.tolist(),
               "A_samples": int(np.char.startswith(replay["source_bag"][picked].astype(str), "A/").sum()),
               "B_samples": int(np.char.startswith(replay["source_bag"][picked].astype(str), "B/").sum()),
               "repeats": args.repeats, "by_epoch": {}, "distribution": {},
               "full_validation": {}}
    for epoch, full_replay in ((7, replay_epoch7), (20, replay)):
        pred = full_replay["prediction"][:, 0, ARM]
        target = full_replay["target"][:, 0, ARM]
        current = full_replay["hold"][:, 0, ARM]
        summary["full_validation"][epoch] = {
            "n_sequences": len(pred), "arm_first_step_pred_vs_gt": error(pred, target),
            "arm_first_step_pred_vs_state_mae": float(np.abs(pred-current).mean()),
            "arm_first_step_gt_vs_state_mae": float(np.abs(target-current).mean()),
            "arm_signed_bias": (pred-target).mean(axis=0).tolist(),
            "arm_mean_abs_error": np.abs(pred-target).mean(axis=0).tolist()}
    predictions = {}
    norm_params = {}
    for epoch in (7, 20):
        policy = load_ema(epoch, device)
        if next(policy.parameters()).dtype != torch.float32:
            raise ValueError("EMA model is not float32")
        if epoch == 20:
            summary["normalizer"] = normalizer_check(policy, obs, gt, train_action, train_state)
        norm_params[epoch] = {
            key: {p: policy.normalizer[key].params_dict[p].detach().cpu().numpy().copy()
                  for p in ("scale", "offset")}
            for key in ("action", "agent_pos", "point_cloud")}
        samples = np.stack([predict(policy, obs, 3100 + rep) for rep in range(args.repeats)])
        predictions[epoch] = samples
        first = samples[0]
        repeated = predict(policy, obs, 3100)
        spread = samples[:, :, ARM].std(axis=0)
        scores = [error(one[:, ARM], gt[:, ARM])["rmse"] for one in samples]
        summary["by_epoch"][epoch] = {
            "first_seed_arm_error": error(first[:, ARM], gt[:, ARM]),
            "first_seed_arm_delta_from_state_mae": float(np.abs(first[:, ARM]-q[:, ARM]).mean()),
            "repeat_arm_rmse": {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
                                "min": float(np.min(scores)), "max": float(np.max(scores)),
                                "values": list(map(float, scores))},
            "sampling_std_arm_mean": float(spread.mean()),
            "sampling_std_arm_p95": float(np.quantile(spread, .95)),
            "fixed_seed_repeat_max_abs": float(np.abs(first-repeated).max()),
            "first_step_arm_prediction_mean": first[:, ARM].mean(axis=0).tolist(),
            "sampling_std_per_arm_joint": spread.mean(axis=0).tolist(),
        }
        if epoch == 20:
            rng = np.random.default_rng(2026)
            train_val_comparison = {}
            for label, frame_pool in (("train", train_frames), ("validation", val_frames)):
                comparison_frames = np.sort(rng.choice(frame_pool, size=64, replace=False))
                comparison_obs = {key: value.to(device) for key, value in
                                  observations(root, ends, comparison_frames).items()}
                comparison_gt = np.asarray(root["data/action"][comparison_frames],
                                           dtype=np.float32)[:, ARM]
                comparison_q = np.asarray(root["data/state"][comparison_frames],
                                          dtype=np.float32)[:, ARM]
                comparison_pred = predict(policy, comparison_obs, 7600)[:, ARM]
                train_val_comparison[label] = {
                    "n": len(comparison_frames),
                    "pred_vs_gt": error(comparison_pred, comparison_gt),
                    "gt_vs_state_mae": float(np.abs(comparison_gt-comparison_q).mean()),
                    "pred_vs_state_mae": float(np.abs(comparison_pred-comparison_q).mean())}
                del comparison_obs
            summary["random_train_validation_comparison"] = train_val_comparison
            sensitivity_indices = np.linspace(0, len(picked) - 1, 16, dtype=int)
            subset = {key: value[sensitivity_indices] for key, value in obs.items()}
            sensitivity_gt = gt[sensitivity_indices]
            sensitivity_bags = replay["source_bag"][picked[sensitivity_indices]].astype(str)
            frozen = {key: value[:1].expand_as(value).clone() for key, value in subset.items()}
            fixed_all = predict(policy, frozen, 5200)
            varied_all = predict(policy, subset, 5200)
            cloud_only = predict(policy, {"point_cloud": subset["point_cloud"],
                                          "agent_pos": frozen["agent_pos"]}, 5200)
            state_only = predict(policy, {"point_cloud": frozen["point_cloud"],
                                          "agent_pos": subset["agent_pos"]}, 5200)
            summary["sensitivity"] = {
                "reference": "same seed and per-batch noise; 16 diverse A/B observations; outputs compared to frozen first observation",
                "A_samples": int(np.char.startswith(sensitivity_bags, "A/").sum()),
                "B_samples": int(np.char.startswith(sensitivity_bags, "B/").sum()),
                "varied_both_arm_mean_abs_change": float(np.abs(varied_all[:, ARM]-fixed_all[:, ARM]).mean()),
                "varied_cloud_only_arm_mean_abs_change": float(np.abs(cloud_only[:, ARM]-fixed_all[:, ARM]).mean()),
                "varied_state_only_arm_mean_abs_change": float(np.abs(state_only[:, ARM]-fixed_all[:, ARM]).mean()),
                "varied_both_arm_rmse_vs_gt": error(varied_all[:, ARM], sensitivity_gt[:, ARM])["rmse"],
                "frozen_obs_arm_rmse_vs_gt": error(fixed_all[:, ARM], sensitivity_gt[:, ARM])["rmse"],
                "gt_arm_mean_abs_change_vs_reference": float(np.abs(sensitivity_gt[:, ARM]-sensitivity_gt[0, ARM]).mean())}
        del policy
        torch.cuda.empty_cache() if device.type == "cuda" else None
        gc.collect()
    summary["normalizer"]["epoch7_vs_epoch20_max_abs"] = {
        key: {p: float(np.max(np.abs(norm_params[7][key][p] - norm_params[20][key][p])))
              for p in ("scale", "offset")}
        for key in norm_params[7]}
    for label, array in (("train_action", train_action), ("validation_gt_action", val_action),
                         ("epoch20_replay_pred_first", replay["prediction"][:, 0])):
        summary["distribution"][label] = {names[i]: stats(array[:, i]) for i in ARM}
    columns = ["index", "joint_name", "train_mean", "train_std", "train_median", "train_p5", "train_p95",
               "val_gt_mean", "val_gt_std", "val_gt_median", "val_gt_p5", "val_gt_p95",
               "epoch20_pred_mean", "epoch20_pred_std", "epoch20_pred_median", "epoch20_pred_p5", "epoch20_pred_p95",
               "epoch7_signed_bias", "epoch20_signed_bias", "epoch7_mean_abs_error", "epoch20_mean_abs_error",
               "epoch7_sampling_std_64sample", "epoch20_sampling_std_64sample"]
    with (output / "arm_joint_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for j, i in enumerate(ARM):
            row = {"index": int(i), "joint_name": names[i]}
            for prefix, key in (("train", "train_action"), ("val_gt", "validation_gt_action"),
                                ("epoch20_pred", "epoch20_replay_pred_first")):
                row.update({f"{prefix}_{k}": v for k, v in summary["distribution"][key][names[i]].items()})
            for epoch in (7, 20):
                full_replay = replay_epoch7 if epoch == 7 else replay
                residual = full_replay["prediction"][:, 0, i] - full_replay["target"][:, 0, i]
                row[f"epoch{epoch}_signed_bias"] = float(residual.mean())
                row[f"epoch{epoch}_mean_abs_error"] = float(np.abs(residual).mean())
                row[f"epoch{epoch}_sampling_std_64sample"] = float(
                    predictions[epoch][:, :, i].std(axis=0).mean())
            writer.writerow(row)
    summary["same_sample_same_seed_epoch7_vs_epoch20"] = {
        "first_seed_arm_rmse_delta_20_minus_7":
            summary["by_epoch"][20]["first_seed_arm_error"]["rmse"] -
            summary["by_epoch"][7]["first_seed_arm_error"]["rmse"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("sample_count", "normalizer", "full_validation", "by_epoch",
                                                "random_train_validation_comparison", "sensitivity",
                                                "same_sample_same_seed_epoch7_vs_epoch20")}, indent=2))


if __name__ == "__main__":
    main()
