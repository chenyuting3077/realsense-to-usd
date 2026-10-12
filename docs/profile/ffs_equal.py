"""Two FFS engines on the same D455 IR pairs: disparity difference (px) and the engine-alone time of
each (median of 50 after 10 warm-up calls, CUDA events). A faster build of the same model has to give
the same disparity (up to TF32 rounding).

Usage: python -m docs.profile.ffs_equal <engine A> <engine B> <rec dir> [every=8]
"""
import os
import sys

import cv2
import numpy as np
import torch

from recon.ffs import FFS


def scale(engine):
    """recon.ffs's input scale for an engine: 1 for 848x480 (padded to 864), 0.75 for 640x384."""
    return 1.0 if "_480x864_" in engine else 0.75


def engine_ms(ffs, left, right, n=50):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for _ in range(10):
        ffs(left, right)
    t = []
    for _ in range(n):
        ev[0].record(ffs.stream)
        ffs.context.execute_async_v3(ffs.stream.cuda_stream)
        ev[1].record(ffs.stream)
        ffs.stream.synchronize()
        t.append(ev[0].elapsed_time(ev[1]))
    return float(np.median(t))


def main():
    a, b, rec = sys.argv[1:4]
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    fa, fb = (FFS(e, scale(e)) for e in (a, b))
    names = sorted(os.listdir(os.path.join(rec, "ir1_on")))[::every]
    diffs, worst = [], 0.0
    for n in names:
        l, r = (torch.from_numpy(cv2.imread(os.path.join(rec, d, n), cv2.IMREAD_GRAYSCALE)).cuda()
                for d in ("ir1_on", "ir2_on"))
        da, db = fa(l, r), fb(l, r)
        d = (da - db).abs()
        diffs.append(d.flatten()[::16].cpu())
        worst = max(worst, float(d.max()))
    d = torch.cat(diffs).numpy()
    print(f"{len(names)} IR pairs: |disparity A - B| median {np.median(d):.2e}, p99 {np.percentile(d, 99):.2e}, "
          f"max {worst:.2e} px")
    print(f"engine alone: A {engine_ms(fa, l, r):.1f} ms, B {engine_ms(fb, l, r):.1f} ms")


if __name__ == "__main__":
    main()
