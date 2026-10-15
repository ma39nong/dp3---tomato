#!/usr/bin/env python3
"""Validate the Sharpa-only 58D contract before conversion or training."""
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
contract = json.loads((root / "configs/sharpa_episode68_58d_contract.json").read_text())
left = contract["hand_joint_names"]["left"]
right = contract["hand_joint_names"]["right"]
if len(left) != 22 or len(right) != 22 or len(set(left + right)) != 44:
    raise SystemExit("Sharpa hand contract must contain 22 unique joints per side")
if contract["robot"]["action_dim"] != 58:
    raise SystemExit("Sharpa action dimension must be 58")
print("Sharpa contract OK: 7+22+7+22 = 58D")
