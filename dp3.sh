#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command="${1:-help}"
[[ $# -eq 0 ]] || shift
python_bin="${DP3_PYTHON:-python}"

case "$command" in
  setup-conversion) exec bash "$root/conversion/scripts/setup.sh" "$@" ;;
  setup-training) exec bash "$root/training/scripts/setup_dp3_env.sh" "$@" ;;
  bag-to-zarr) exec "$root/conversion/scripts/dp3" convert "$@" ;;
  verify-intermediate) exec "$root/conversion/scripts/dp3" verify "$@" ;;
  preview) exec "$python_bin" "$root/training/scripts/bimanual_zarr.py" preview "$@" ;;
  crop) exec "$python_bin" "$root/training/scripts/bimanual_zarr.py" convert "$@" ;;
  verify-final) exec "$python_bin" "$root/training/scripts/bimanual_zarr.py" verify "$@" ;;
  self-check) exec "$python_bin" "$root/training/scripts/bimanual_zarr.py" self-check "$@" ;;
  train) exec bash "$root/training/scripts/train_bimanual.sh" "$@" ;;
  *)
    cat <<'EOF'
Usage: ./dp3.sh COMMAND [arguments]

  setup-conversion     create the rosbag conversion environment
  setup-training       install DP3 packages into the active conda environment
  bag-to-zarr          ROS2 bags -> intermediate uncropped Zarr
  verify-intermediate  validate intermediate Zarr
  preview              generate interactive point-cloud crop HTML
  crop                 crop/sample -> final fixed-size Zarr
  verify-final         validate final training Zarr
  self-check           synthetic crop/final-Zarr self-check
  train                train DP3 from final Zarr
EOF
    [[ "$command" == help ]] || exit 2
    ;;
esac
