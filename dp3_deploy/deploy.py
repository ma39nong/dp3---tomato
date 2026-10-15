#!/usr/bin/env python3
"""Prediction-only inference from a two-frame observation NPZ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inference import Predictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True,
                        help="NPZ with point_cloud_m [2,4096,3] and state_rad [2,54]")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.observation) as obs:
        predictor = Predictor(args.device)
        result = predictor.predict(obs["point_cloud_m"], obs["state_rad"], args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"action_rad": result["action"].tolist(),
                                       "semantics": result["semantics"],
                                       "latency_ms": result["latency_ms"],
                                       "sent_robot_command": False}) + "\n")


if __name__ == "__main__":
    main()
