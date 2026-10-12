#!/usr/bin/env bash
# Train FastGS on a gs/export_colmap.py dataset, render the held-out test views and score them.
#
# Every 8th image is held out (--eval).
# Steps: each training image is seen ~38 times (EPOCHS), rounded to 1000, at least 20k: a 40 s
# recording (529 training images) gets 20k (~3.75 min on Jetson Thor), a 2 min one (1590) 60k
# (~10.5 min, +1.1 dB over 20k on room_1005). ITERS=<n> overrides.
#
# Output: <out>/point_cloud/iteration_<N>/point_cloud.ply (standard 3DGS PLY; <out>/point_cloud.ply
#         links to it), <out>/test/ours_<N>/{renders,gt}/, <out>/results.json (PSNR / SSIM / LPIPS)
#
# Usage (in the activated conda env; needs ~/FastGS with patches/fastgs-thor.patch, see docs/INSTALL.md):
#   gs/train_fastgs.sh output/room/colmap output/room/fastgs
#   ITERS=30000 gs/train_fastgs.sh output/room/colmap output/room/fastgs_30k
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <colmap_dir> <out_dir>" >&2
    exit 1
fi
colmap=$(realpath "$1")
out=$(realpath -m "$2")
n_images=$(find "$colmap/images" -maxdepth 1 -type f | wc -l)
n_train=$(( n_images - (n_images + 7) / 8 ))  # FastGS holds out every 8th, starting at 0
iters=${ITERS:-$(( (${EPOCHS:-38} * n_train + 500) / 1000 * 1000 ))}
(( iters < 20000 )) && [[ -z ${ITERS:-} ]] && iters=20000
echo "FastGS: $n_train training images, $iters steps"
# Densify for the first half, but not past 15k: FastGS prunes Gaussians with opacity < 0.1 every
# 3k steps between 15k and 30k (hard-coded), and an opacity reset (every 3k while densifying)
# falling on such a step removes every Gaussian.
densify_until=$(( iters / 2 < 15000 ? iters / 2 : 15000 ))
root=${FASTGS_ROOT:-$HOME/FastGS}

cd "$root"
export PYTHONPATH="$root"
python train.py -s "$colmap" -i images -m "$out" --eval --quiet \
    --optimizer_type default --densification_interval 500 --highfeature_lr 0.02 --grad_abs_thresh 0.0008 \
    --iterations "$iters" --position_lr_max_steps "$iters" --densify_until_iter "$densify_until" \
    --test_iterations "$iters" --save_iterations "$iters"
python render.py -m "$out" --skip_train --quiet
python metrics.py -m "$out"
ln -sfn "point_cloud/iteration_$iters/point_cloud.ply" "$out/point_cloud.ply"
