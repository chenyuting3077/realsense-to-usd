"""FFS engines against the first one: latency of an 848x480 IR pair (the model call: downscaling,
normalization, padding, inference, disparity back to 848x480; mean of 50 after 10 warm-up calls),
ETH3D two-view (laser-scan ground truth, non-occluded pixels, each pair cut to an 848x480 window and
scored at full resolution), and on D455 recordings (every 4th projector-on IR pair, no ground truth)
coverage / median / p90 |depth difference| against the first engine, by its depth.

Usage: python -m bench.ffs_accuracy <rec dir>[,<rec dir>] <engine> <engine> ...
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from bench.ffs_equal import scale
from recon.ffs import FFS

ETH3D = os.path.expanduser("~/datasets/eth3d")
BINS = [0.2, 0.35, 1, 2, 3, 4, 5]


def gray(path):
    return torch.from_numpy(cv2.imread(path, cv2.IMREAD_GRAYSCALE)).cuda()


def latency(ffs, pair, n=50):
    for i in range(n + 10):
        if i == 10:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        ffs(*pair)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def eth3d(ffs):
    """Mean over the pairs of EPE (px), bad-1 and bad-2 (%)."""
    scores = []
    for d in sorted(glob.glob(os.path.join(ETH3D, "*/"))):
        gt = cv2.imread(d + "disp0GT.pfm", cv2.IMREAD_UNCHANGED)
        h, w = gt.shape
        y0, x0 = max(0, (h - 480) // 2), max(0, (w - 848) // 2)
        win = (slice(y0, y0 + 480), slice(x0, x0 + 848))
        disp = ffs(gray(d + "im0.png")[win].contiguous(), gray(d + "im1.png")[win].contiguous()).cpu().numpy()
        g = gt[win]
        m = (cv2.imread(d + "mask0nocc.png", cv2.IMREAD_GRAYSCALE)[win] == 255) & np.isfinite(g) & (g > 0)
        err = np.abs(disp[m] - g[m])
        scores.append((err.mean(), (err > 1).mean() * 100, (err > 2).mean() * 100))
    return np.mean(scores, axis=0)


def d455_depth(ffs, recs):
    """Depth (m) of every 4th projector-on left IR, all pixels in one array, 0 where none."""
    out = []
    for rec in recs:
        with open(os.path.join(rec, "meta.json")) as f:
            st = json.load(f)["stereo"]
        fb = st["fx"] * np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3])
        for p in sorted(glob.glob(os.path.join(rec, "ir1_on", "*.png")))[::4]:
            disp = ffs(gray(p), gray(p.replace("ir1_on", "ir2_on"))).cpu().numpy()
            ok = (disp > 0.5) & (np.arange(disp.shape[1]) - disp >= 0)  # the match inside the right image
            out.append(np.where(ok, fb / np.maximum(disp, 1e-6), 0).ravel())
    return np.concatenate(out)


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
