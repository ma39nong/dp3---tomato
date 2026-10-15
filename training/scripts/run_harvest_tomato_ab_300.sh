#!/usr/bin/env bash
# Run only the new training experiment. Existing conversion and 50-epoch output stay intact.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export DP3_PYTHON="${DP3_PYTHON:-/home/user/miniforge3/envs/dp3/bin/python}"
dataset="$root/outputs/harvest_tomato_AB_pre722_40_40/AB_committed_4096.zarr"
experiment="harvest-tomato-ab-pre722-40-40-300-from-scratch-4096"
mode="${1:-start}"
case "$mode" in
  start) training_mode=experiment300 ;;
  resume) training_mode=experiment300-resume ;;
  *) echo "Usage: $0 [start|resume]" >&2; exit 2 ;;
esac
"$root/dp3.sh" verify-final "$dataset" --num-points 4096 --expected-dim 54
exec "$root/dp3.sh" train "$dataset" "$training_mode" "$experiment"
