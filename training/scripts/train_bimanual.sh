#!/usr/bin/env bash
set -euo pipefail
unset PYTHONPATH
export PYTHONNOUSERSITE=1
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DP3_PYTHON:-python}"
dataset="$(realpath -e "${1:?Usage: scripts/train_bimanual.sh DATASET.zarr [smoke|full|STOP_EPOCH] [experiment-name] [full-mode Hydra overrides...]}")"
mode="${2:-smoke}"
read -r point_count state_dim target_hz < <("$python_bin" - "$dataset" <<'PY'
import sys
import zarr

root = zarr.open_group(sys.argv[1], mode="r")
point_cloud = root["data/point_cloud"]
state, action = root["data/state"], root["data/action"]
if root.attrs.get("training_ready") is not True or root.attrs.get("conversion_complete") is not True:
    raise SystemExit("Dataset is not marked conversion_complete/training_ready")
if point_cloud.ndim != 3 or point_cloud.shape[2] != 3:
    raise SystemExit(f"Invalid point_cloud shape: {point_cloud.shape}")
if state.ndim != 2 or action.shape != state.shape or len(point_cloud) != len(state):
    raise SystemExit(f"Incompatible point/state/action shapes: {point_cloud.shape}/{state.shape}/{action.shape}")
print(point_cloud.shape[1], state.shape[1], int(root.attrs.get("target_hz", 10)))
PY
)
case "$target_hz" in
  10) horizon=16; obs_steps=2; action_steps=4 ;;
  20) horizon=32; obs_steps=4; action_steps=8 ;;
  *) echo "Unsupported dataset target_hz=$target_hz; expected 10 or 20" >&2; exit 2 ;;
esac
if [[ "$mode" =~ ^[0-9]+$ ]]; then
  experiment="${3:-dp3-custom}"
else
  experiment="${3:-dp3-custom-$mode}"
fi
run_dir="$root/data/outputs/$experiment"

common=(
  train.py --config-name=dp3 task=generic_robot
  "horizon=$horizon" "n_obs_steps=$obs_steps" "n_action_steps=$action_steps"
  "task.shape_meta.obs.point_cloud.shape=[$point_count,3]"
  "task.shape_meta.obs.agent_pos.shape=[$state_dim]"
  "task.shape_meta.action.shape=[$state_dim]"
  "task.dataset.expected_num_points=$point_count" "task.dataset.expected_dim=$state_dim"
  "task.dataset.zarr_path=$dataset"
  dataloader.batch_size=32 dataloader.num_workers=4
  val_dataloader.batch_size=32 val_dataloader.num_workers=4
  training.device=cuda:0 "logging.mode=${DP3_WANDB_MODE:-offline}"
  "exp_name=$experiment" "hydra.run.dir=$run_dir"
)

echo "Dataset contract: points=$point_count state/action=$state_dim target_hz=$target_hz horizon=$horizon/$obs_steps/$action_steps"

cd "$root/3D-Diffusion-Policy"
if [[ "$mode" == smoke ]]; then
  exec "$python_bin" "${common[@]}" task.dataset.grouped_validation=true \
    task.dataset.point_cloud_noise_std=0.002 optimizer.weight_decay=1e-4 \
    training.num_epochs=1 training.stop_after_epoch=1 training.run_validation=true \
    training.deterministic_validation=true training.resume=false training.max_train_steps=10 \
    training.max_val_steps=2 \
    dataloader.batch_size=8 dataloader.num_workers=0 \
    val_dataloader.batch_size=8 val_dataloader.num_workers=0 checkpoint.save_ckpt=false
elif [[ "$mode" == full ]]; then
  exec "$python_bin" "${common[@]}" \
    task.dataset.grouped_validation=true task.dataset.point_cloud_noise_std=0.002 \
    optimizer.weight_decay=1e-4 \
    training.num_epochs=50 training.run_validation=true \
    training.deterministic_validation=true training.early_stopping_patience=0 \
    training.early_stopping_min_delta=1e-4 training.checkpoint_every=1 \
    training.resume=false checkpoint.save_ckpt=true checkpoint.save_last_ckpt=false \
    checkpoint.topk_until_epoch=20 checkpoint.periodic_every_after_topk=5 \
    checkpoint.topk.monitor_key=val_loss checkpoint.topk.mode=min checkpoint.topk.k=3 \
    "checkpoint.topk.format_str='epoch={epoch:04d}-val_loss={val_loss:.6f}.ckpt'" \
    "${@:4}"
elif [[ "$mode" =~ ^[0-9]+$ ]]; then
  plan_epochs="${DP3_PLAN_EPOCHS:-100}"
  if (( mode < 1 || mode > plan_epochs )); then
    echo "STOP_EPOCH must be between 1 and DP3_PLAN_EPOCHS ($plan_epochs)" >&2
    exit 2
  fi
  resume=false
  if [[ -f "$run_dir/checkpoints/latest.ckpt" ]]; then
    resume=true
  fi
  exec "$python_bin" "${common[@]}" "training.num_epochs=$plan_epochs" \
    "training.stop_after_epoch=$mode" training.run_validation=true \
    "training.resume=$resume" training.checkpoint_every=10 \
    checkpoint.save_ckpt=true checkpoint.topk.k=999
else
  echo "Mode must be smoke, full, or a numeric stop epoch" >&2
  exit 2
fi
