"""Faster Fast-FoundationStereo settings vs. the one used before (848x480, max disparity 192), see README.md:
latency of an 848x480 IR pair; ETH3D two-view accuracy (laser-scan ground truth, non-occluded
pixels, each pair cut to an 848x480 window, scored at full resolution); on D455 recordings (every
4th projector-on IR pair, no ground truth) coverage and |depth difference| vs. that setting,
by its depth.

Usage: python -m docs.ffs_speed.eval_ffs output/live_a/rec output/live_b/rec
"""
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from recon.ffs import FFS, engine_path

ETH3D = os.path.expanduser("~/datasets/eth3d")
SETTINGS = [("848x480, max_disp 192 (before)", engine_path(precision="tf32"), 1.0),
            ("636x360 (0.75x), max_disp 192", engine_path(384, 640, precision="tf32"), 0.75),
            ("848x480, max_disp 128", engine_path(max_disp=128, precision="tf32"), 1.0)]
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
    recs = sys.argv[1:]
    pair = [gray(sorted(glob.glob(os.path.join(recs[0], s, "*.png")))[100]) for s in ("ir1_on", "ir2_on")]
    print("| | Latency | fps | ETH3D EPE | bad-1 | bad-2 |\n|---|---:|---:|---:|---:|---:|", flush=True)
    rows, ref = [], None
    for name, engine, scale in SETTINGS:
        ffs = FFS(engine, scale)
        ms = latency(ffs, pair)
        epe, bad1, bad2 = eth3d(ffs)
        print(f"| {name} | {ms:.1f} ms | {1000 / ms:.1f} | {epe:.3f} px | {bad1:.2f}% | {bad2:.2f}% |", flush=True)
        z = d455_depth(ffs, recs)
        ref = z if ref is None else ref
        cells = []
        for lo, hi in zip(BINS, BINS[1:]):
            m = (ref >= lo) & (ref < hi)
            d = np.abs(z - ref)[m & (z > 0)] * 1000
            cells.append(f"{(z[m] > 0).mean() * 100:.0f}% / {np.median(d):.0f} / {np.percentile(d, 90):.0f} mm")
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
        del ffs
        torch.cuda.empty_cache()
    print(f"\nD455 ({len(recs)} recordings): coverage / median / p90 |depth difference| vs. 848x480, max_disp 192, "
          "by its depth\n\n| | " + " | ".join(f"{lo}–{hi} m" for lo, hi in zip(BINS, BINS[1:])) + " |\n|---|"
          + "---:|" * (len(BINS) - 1) + "\n" + "\n".join(rows[1:]))


if __name__ == "__main__":
    main()
