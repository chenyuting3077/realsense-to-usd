"""FFS engines against the first one: latency of an 848x480 IR pair (as docs/ffs_speed: pre/post-processing
included, mean of 50 after 10 warm-up calls), ETH3D, and on D455 recordings coverage / median / p90
|depth difference| by depth. A rebuild of the first engine's ONNX shows how far two builds of the same
network already differ (TensorRT picks other kernels each build).

Usage: python -m docs.profile.ffs_accuracy <rec dir>[,<rec dir>] <engine> <engine> ...
"""
import glob
import os
import sys

import numpy as np
import torch

from docs.ffs_speed.eval_ffs import BINS, d455_depth, eth3d, gray, latency
from docs.profile.ffs_equal import scale
from recon.ffs import FFS


def main():
    recs, engines = sys.argv[1].split(","), sys.argv[2:]
    pair = [gray(sorted(glob.glob(os.path.join(recs[0], s, "*.png")))[100]) for s in ("ir1_on", "ir2_on")]
    ref = None
    print("| engine | latency | ETH3D EPE | bad-1 | bad-2 | " + " | ".join(f"{lo}–{hi} m" for lo, hi in zip(BINS, BINS[1:]))
          + " |\n|---|---:|---:|---:|---:|" + "---:|" * (len(BINS) - 1))
    for e in engines:
        ffs = FFS(e, scale(e))
        ms = latency(ffs, pair)
        epe, bad1, bad2 = eth3d(ffs)
        z = d455_depth(ffs, recs)
        ref = z if ref is None else ref
        cells = []
        for lo, hi in zip(BINS, BINS[1:]):
            m = (ref >= lo) & (ref < hi)
            d = np.abs(z - ref)[m & (z > 0)] * 1000
            cells.append(f"{(z[m] > 0).mean() * 100:.0f}% / {np.median(d):.0f} / {np.percentile(d, 90):.0f} mm")
        print(f"| {os.path.basename(e)} | {ms:.1f} ms | {epe:.4f} px | {bad1:.3f}% | {bad2:.3f}% | " + " | ".join(cells) + " |",
              flush=True)
        del ffs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
