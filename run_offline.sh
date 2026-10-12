#!/usr/bin/env bash
# A run_live.sh session -> 3D Gaussian Splatting (FastGS) + mesh in one USD stage for Isaac Sim.
#
#   ./run_offline.sh output/live_<date>_<time>
#   ITERS=30000 ./run_offline.sh output/live_room       # FastGS steps (default: gs/train_fastgs.sh)
#
# Steps, all on the session's frames/ (the fused frames with FFS depth, ~5 Hz) and its poses:
#   recon/    TSDF mesh + pointcloud.ply (same settings as live.py's final mesh: depth to 4 m,
#             distance-weighted), poses.npy copied from frames/ (export_colmap.py reads it there)
#   colmap/   COLMAP text dataset, every frame (live keeps only ~5 Hz)
#   fastgs/   FastGS model, held-out test renders, results.json (PSNR / SSIM / LPIPS)
#   scene.usdc
# A 3 min session (934 frames) takes ~11 min on Jetson AGX Thor, most of it FastGS.
set -eo pipefail
if [[ $# -ne 1 ]]; then
  echo "usage: $0 <session dir from run_live.sh>" >&2
  exit 1
fi
here=$(cd "$(dirname "$0")" && pwd)
s=$(realpath "$1")
[[ -f $s/frames/poses.npy ]] || { echo "$s/frames/poses.npy not found: not a run_live.sh session?" >&2; exit 1; }
cd "$here"
[[ ${CONDA_DEFAULT_ENV:-base} != base ]] || { echo "conda activate the env first (docs/INSTALL.md)" >&2; exit 1; }

python recon/reconstruct.py "$s/frames" -o "$s/recon" --poses "$s/frames/poses.npy" \
    --depth-dir depth_ffs --depth-max 4 --weighting distance --weight-ref 2
cp "$s/frames/poses.npy" "$s/recon/"
python gs/export_colmap.py "$s/frames" "$s/recon" -o "$s/colmap" --every 1 --depth-dir depth_ffs --depth-max 4
gs/train_fastgs.sh "$s/colmap" "$s/fastgs"
python gs/export_usd.py "$s/recon/mesh.ply" "$s/fastgs/point_cloud.ply" --poses "$s/frames/poses.npy" \
    -o "$s/scene.usdc"
