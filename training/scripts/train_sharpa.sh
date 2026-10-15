#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="${1:-$root/../outputs/harvest_tomato_sharpa_episode68/dataset.zarr}"
mode="${2:-smoke}"
shift $(( $# > 0 ? 1 : 0 ))
exec "$root/scripts/train_bimanual.sh" "$dataset" "$mode" sharpa-58d "$@"
