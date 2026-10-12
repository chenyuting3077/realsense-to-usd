"""recon/ffs.py's DisparityToColorDepth against another version of it (a copy of the old ffs.py): the
depth must be identical. Disparities from the FFS engine on every 10th projector-on IR pair.

Usage: python -m docs.profile.warp_equal <old ffs.py> <rec dir>
"""
import importlib.util
import json
import os
import sys
import time

import cv2
import torch

from recon.ffs import FFS, DisparityToColorDepth


def ms(f, disp, n=50):
    for _ in range(5):
        f(disp)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f(disp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    spec = importlib.util.spec_from_file_location("ffs_old", sys.argv[1])
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rec = sys.argv[2]
    with open(os.path.join(rec, "meta.json")) as f:
        meta = json.load(f)
    a, b, ffs = old.DisparityToColorDepth(meta, "cuda"), DisparityToColorDepth(meta, "cuda"), FFS()
    names = sorted(os.listdir(os.path.join(rec, "ir1_on")))[::10]
    same, worst = 0, 0.0
    for n in names:
        disp = ffs(*(torch.from_numpy(cv2.imread(os.path.join(rec, d, n), cv2.IMREAD_GRAYSCALE)).cuda()
                     for d in ("ir1_on", "ir2_on")))
        da, db = a(disp), b(disp)
        same += torch.equal(da, db)
        worst = max(worst, float((da - db).abs().max()))
    print(f"{len(names)} frames: {same} identical, max |difference| {worst} m; "
          f"old {ms(a, disp):.2f} ms, new {ms(b, disp):.2f} ms")


if __name__ == "__main__":
    main()
