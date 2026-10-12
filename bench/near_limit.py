"""max_disp 128 at 848x480 cannot match closer than fx * baseline / 128 (~0.32 m) and still outputs a
disparity there. Against the max_disp 192 engine on D455 projector-on pairs (pixels whose match falls
inside the right image in both, as DisparityToColorDepth keeps): what the 128 engine gives where the
disparity is above 128, and how many good pixels a cut at a disparity threshold loses.

Usage: python -m bench.near_limit <engine 192> <engine 128> <rec dir>[,<rec dir>] [every=4]
"""
import glob
import os
import sys

import numpy as np
import torch

from bench.ffs_accuracy import gray
from recon.ffs import FFS


def main():
    e192, e128, recs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    a, b = FFS(e192, 1.0), FFS(e128, 1.0)
    ref, out = [], []
    for rec in recs:
        for p in sorted(glob.glob(os.path.join(rec, "ir1_on", "*.png")))[::every]:
            l, r = gray(p), gray(p.replace("ir1_on", "ir2_on"))
            da, db = a(l, r), b(l, r)
            u = torch.arange(da.shape[1], device=da.device)
            inside = ((u - da >= 0) & (u - db >= 0)).flatten()[::4]  # both matches inside the right image
            ref.append(da.flatten()[::4][inside].cpu().numpy())
            out.append(db.flatten()[::4][inside].cpu().numpy())
    ref, out = np.concatenate(ref), np.concatenate(out)
    near, ok = ref > 128, (ref > 0.5) & (ref <= 128)
    print(f"{near.mean() * 100:.2f}% of the pixels nearer than the 128 px limit")
    print("the 128 engine there: disparity percentiles 1/10/50/90/99:",
          " / ".join(f"{np.percentile(out[near], q):.1f}" for q in (1, 10, 50, 90, 99)))
    print("| cut at disparity > | near pixels removed | good pixels removed (all / 100–128 px) |\n|---:|---:|---:|")
    for t in (110, 115, 118, 120, 122, 124, 126):
        good_hi = ok & (ref > 100)
        print(f"| {t} | {(out[near] > t).mean() * 100:.1f}% | {(out[ok] > t).mean() * 100:.2f}% / "
              f"{(out[good_hi] > t).mean() * 100:.1f}% |")


if __name__ == "__main__":
    main()
