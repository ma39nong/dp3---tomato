#!/usr/bin/env python3
"""One-time export of an EMA-only, Zarr-free prediction bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import dill
from omegaconf import OmegaConf
import torch
import yaml
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    dataset = args.dataset.expanduser().resolve(strict=True)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True,
                         weights_only=False, pickle_module=dill)
    if payload.get("format_version") != 2 or "ema_model" not in payload:
        raise ValueError("Expected format-v2 checkpoint with EMA weights")
    cfg = payload["cfg"]
    if int(cfg.n_obs_steps) != 2 or int(cfg.n_action_steps) != 4:
        raise ValueError("Unexpected model horizons")
    if tuple(cfg.task.shape_meta.obs.point_cloud.shape) != (4096, 3):
        raise ValueError("Unexpected point-cloud shape")
    group = zarr.open_group(str(dataset), mode="r")
    attrs = dict(group.attrs)
    conversion = yaml.safe_load((root.parent / "conversion/configs/harvest_tomato_A.yaml").read_text())
    contract = json.loads((root / "config/robot_source_contract.json").read_text())
    if (attrs["state_names"] != contract["state_names"] or
            attrs["action_names"] != contract["action_names"] or
            attrs["point_frame"] != "cam0_depth_optical_frame" or
            attrs["action_semantics"] != "absolute_joint_target" or
            int(attrs["target_hz"]) != 10 or int(attrs["num_points"]) != 4096):
        raise ValueError("Training data contract differs from pinned robot source")
    if (conversion["pixel_stride"] != attrs["pixel_stride"] or
            conversion["depth_scale_m"] != attrs["depth_scale_m"] or
            conversion["T_point_from_depth_camera"] != attrs["T_point_from_depth_camera"]):
        raise ValueError("Depth conversion differs from final training data")
    policy_cfg = OmegaConf.to_container(cfg.policy, resolve=True)
    manifest = {
        "schema": "harvest_dp3_prediction_bundle/v1",
        "checkpoint_source": str(checkpoint),
        "checkpoint_epoch": int(payload["next_epoch"]),
        "weights": "ema_model",
        "prediction_semantics": "absolute_joint_target_rad",
        "observation_steps": 2,
        "action_steps": 4,
        "state_dim": 54,
        "num_points": 4096,
        "target_hz": 10,
        "point_frame": attrs["point_frame"],
        "pixel_stride": int(attrs["pixel_stride"]),
        "depth_scale_m": float(attrs["depth_scale_m"]),
        "min_depth_m": float(conversion["min_depth_m"]),
        "max_depth_m": float(conversion["max_depth_m"]),
        "T_point_from_depth_camera": attrs["T_point_from_depth_camera"],
        "crop_min": attrs["crop_min"],
        "crop_max": attrs["crop_max"],
        "clip_planes": attrs["clip_planes"],
        "voxel_size_m": float(attrs["voxel_size_m"]),
        "state_names": attrs["state_names"],
        "action_names": attrs["action_names"],
        "policy": policy_cfg,
    }
    weights_path = root / "checkpoints/epoch20_ema.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists() or (root / "config/deploy.json").exists():
        raise FileExistsError("Bundle output exists; remove it explicitly before rebuilding")
    torch.save(payload["ema_model"], weights_path)
    digest = hashlib.sha256()
    with weights_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest["ema_weights_sha256"] = digest.hexdigest()
    (root / "config/deploy.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {manifest['checkpoint_epoch']} EMA tensors to {weights_path}")


if __name__ == "__main__":
    main()
