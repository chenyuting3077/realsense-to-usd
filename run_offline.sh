#!/usr/bin/env bash
# A run_live.sh session -> <session>/scene.usdc (TSDF mesh, FastGS 3DGS, USD), as 3DGS -> USD in the window.
#   ./run_offline.sh output/live_room
set -eo pipefail
[[ $# -eq 1 ]] || { echo "usage: $0 <session dir>" >&2; exit 1; }
[[ ${CONDA_DEFAULT_ENV:-base} != base ]] || { echo "conda activate the env first (docs/install)" >&2; exit 1; }
exec python "$(dirname "$0")/gs/offline.py" "$1"
