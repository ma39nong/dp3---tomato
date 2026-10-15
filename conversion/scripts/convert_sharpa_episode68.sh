#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$root/conversion/scripts/dp3" convert \
  --config "$root/conversion/configs/harvest_tomato_sharpa_episode68.yaml" \
  --quality train "$@"
