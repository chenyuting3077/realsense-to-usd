#!/bin/bash
# The FFS settings of README.md: their engines (~3-5 min each, whole GPU: TensorRT times its kernels while
# building), then latency, ETH3D and D455 depth against the first one (848x480, max_disp 192, TF32).
# Usage: docs/profile/ffs_settings.sh <rec dir>,<rec dir>
set -e
cd "$(dirname "$0")/../.."
export PATH=$HOME/miniforge3/envs/realsense-to-usd/bin:$PATH
for a in "--precision tf32" "" "--max-disp 128 --precision tf32" "--max-disp 128" "--height 384 --width 640 --precision tf32"; do
    python -m recon.export_ffs_trt $a
done
python -m docs.profile.ffs_accuracy "$1" output/ffs/ffs_480x864_it8_v3_fp32_tf32.engine output/ffs/ffs_480x864_it8_v3_amp.engine \
    output/ffs/ffs_480x864_it8_md128_v3_fp32_tf32.engine output/ffs/ffs_480x864_it8_md128_v3_amp.engine \
    output/ffs/ffs_384x640_it8_v3_fp32_tf32.engine
