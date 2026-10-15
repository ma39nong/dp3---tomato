"""EMA inference from a self-contained bundle; no training dataset is loaded."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch


ROOT = Path(__file__).resolve().parent
OmegaConf.register_new_resolver("eval", eval, replace=True)


class Predictor:
    def __init__(self, device: str = "cuda:0", bundle: Path = ROOT):
        self.bundle = Path(bundle).resolve()
        self.cfg = json.loads((self.bundle / "config/deploy.json").read_text())
        if self.cfg["schema"] != "harvest_dp3_prediction_bundle/v1":
            raise ValueError("Unexpected deployment manifest")
        self.device = torch.device(device)
        weights_path = self.bundle / "checkpoints/epoch20_ema.pt"
        expected_hash = self.cfg["ema_weights_sha256"]
        digest = hashlib.sha256()
        with weights_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError("EMA weight SHA256 mismatch")
        policy_cfg = OmegaConf.create(self.cfg["policy"])
        self.policy = hydra.utils.instantiate(policy_cfg)
        weights = torch.load(weights_path,
                             map_location="cpu", mmap=True, weights_only=True)
        self.policy.load_state_dict(weights, strict=True)
        self.policy.to(self.device).eval()
        if next(self.policy.parameters()).dtype != torch.float32:
            raise ValueError("Expected float32 EMA policy")

    def predict(self, point_cloud: np.ndarray, state: np.ndarray,
                seed: int | None = None) -> dict:
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if point_cloud.shape != (2, 4096, 3) or state.shape != (2, 54):
            raise ValueError("Expected point_cloud [2,4096,3] and state [2,54]")
        if not np.isfinite(point_cloud).all() or not np.isfinite(state).all():
            raise ValueError("Nonfinite observation")
        obs = {"point_cloud": torch.from_numpy(point_cloud[None]).to(self.device),
               "agent_pos": torch.from_numpy(state[None]).to(self.device)}
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.policy.predict_action(obs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000
        action = result["action"][0].cpu().numpy()
        if action.shape != (4, 54) or not np.isfinite(action).all():
            raise ValueError("Invalid prediction")
        return {"action": action, "latency_ms": latency_ms,
                "semantics": self.cfg["prediction_semantics"],
                "sent_robot_command": False}

    def warmup(self) -> float:
        """Run a synthetic forward pass before any live observation is accepted."""
        cloud = np.zeros((2, 4096, 3), dtype=np.float32)
        state = np.zeros((2, 54), dtype=np.float32)
        return self.predict(cloud, state)["latency_ms"]
