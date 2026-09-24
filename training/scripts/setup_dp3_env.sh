#!/usr/bin/env bash
# Recreate the offline DP3 training env on a new Linux + NVIDIA machine.
#
# Prerequisites:
#   - conda / miniforge
#   - NVIDIA driver new enough for the chosen CUDA wheel
#     (this repo's reference machine uses torch cu128 + driver ~580)
#   - this repository already cloned
#
# Usage (from repo root):
#   conda env create -f environment_dp3.yml   # first time only
#   conda activate dp3
#   bash scripts/setup_dp3_env.sh
#
# Optional: override CUDA wheel channel, e.g. CUDA 12.1 GPUs:
#   TORCH_CUDA=cu121 bash scripts/setup_dp3_env.sh

set -euo pipefail

# This workstation normally has ROS sourced. Do not mix its Python 3.12
# packages into this Python 3.10 training environment.
unset PYTHONPATH
export PYTHONNOUSERSITE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

TORCH_CUDA="${TORCH_CUDA:-cu128}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "[setup_dp3_env] python=$(python -V 2>&1)  TORCH_CUDA=${TORCH_CUDA}"

python -m pip install -U pip setuptools wheel

echo "[setup_dp3_env] installing torch/torchvision from ${TORCH_INDEX}"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "${TORCH_INDEX}"

echo "[setup_dp3_env] installing requirements_dp3.txt"
python -m pip install -r "${ROOT}/requirements_dp3.txt"

echo "[setup_dp3_env] verifying imports"
cd "${ROOT}/3D-Diffusion-Policy"
python - <<'PY'
import torch
import zarr
import hydra
import diffusers
import wandb
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.generic_dp3_dataset import GenericDP3Dataset
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
print("ok")
PY

echo "[setup_dp3_env] done. Activate with: conda activate dp3"
