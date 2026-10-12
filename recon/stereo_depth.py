"""Depth from the IR stereo pair with Fast-FoundationStereo, aligned to the color camera.

The D455 computes depth on-chip from the same IR pair; a learned stereo matcher
(NVlabs Fast-FoundationStereo) fills most of what the chip leaves empty (screens, dark or
textureless surfaces). For each color/depth frame its own IR pair is used: ir1_on/ir2_on
(projector-on frameset, dot pattern helps matching) for projector-alternating recordings,
else ir1/ir2 when IR and color share the frame numbering.

disparity -> depth = fx * baseline / disparity in the left-IR camera, then forward-warped
into the color camera (each point splatted to its 2x2 pixel neighbourhood, nearest depth
wins). Pixels whose match would fall left of the right image are dropped.

Output: <frames>/<out-dir>/000000.png (uint16, same units as depth/). Use it with
--depth-dir in reconstruct.py / reconstruct_nvblox.py.

Usage (after `source env.sh`; needs ~/Fast-FoundationStereo and its weights, see docs/INSTALL.md):
  python recon/stereo_depth.py output/room/rec                                    # PyTorch
  python recon/stereo_depth.py output/room/rec --engine output/ffs/<name>.engine  # TensorRT (recon/export_ffs_trt.py)
  python recon/reconstruct.py output/room/rec -o output/room/recon_30hz --poses output/room/rec/poses.npy --depth-dir depth_ffs
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

from ffs import FFS
from frames import load_meta


class DisparityToColorDepth:
    """Left-IR disparity (HxW tensor) -> depth in meters in the color camera (0 = none)."""

    def __init__(self, meta, dev):
        st = meta["stereo"]
        self.H, self.W, self.dev = meta["height"], meta["width"], dev
        self.fx_baseline = st["fx"] * float(np.linalg.norm(np.array(st["ir1_from_ir2"])[:3, 3]))
        self.K_ir = torch.tensor([[st["fx"], 0, st["cx"]], [0, st["fy"], st["cy"]], [0, 0, 1]], device=dev)
        self.K_c = torch.tensor([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]], device=dev)
        self.color_from_ir1 = torch.tensor(np.linalg.inv(np.array(st["ir1_from_color"])), device=dev,
                                           dtype=torch.float32)
        self.v, self.u = torch.meshgrid(torch.arange(st["height"], device=dev, dtype=torch.float32),
                                        torch.arange(st["width"], device=dev, dtype=torch.float32), indexing="ij")

    def __call__(self, disp):
        H, W, u, v, K_ir, K_c = self.H, self.W, self.u, self.v, self.K_ir, self.K_c
        ok = (disp > 0.5) & (u - disp >= 0)
        z = self.fx_baseline / disp[ok]
        X = torch.stack([(u[ok] - K_ir[0, 2]) * z / K_ir[0, 0], (v[ok] - K_ir[1, 2]) * z / K_ir[1, 1], z])
        Y = self.color_from_ir1[:3, :3] @ X + self.color_from_ir1[:3, 3:]
        pu, pv = Y[0] / Y[2] * K_c[0, 0] + K_c[0, 2], Y[1] / Y[2] * K_c[1, 1] + K_c[1, 2]
        depth = torch.full((H * W,), torch.inf, device=self.dev)
        for du in (0, 1):
            for dv in (0, 1):
                x, y = torch.floor(pu).long() + du, torch.floor(pv).long() + dv
                m = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (Y[2] > 0)
                depth.scatter_reduce_(0, y[m] * W + x[m], Y[2][m], reduce="amin")
        return torch.where(torch.isinf(depth), 0, depth).reshape(H, W)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", help="frames dir with IR (rec/ of a live session)")
    p.add_argument("--out-dir", default="depth_ffs")
    p.add_argument("--engine", help="TensorRT engine from recon/export_ffs_trt.py (default: PyTorch)")
    p.add_argument("--valid-iters", type=int, default=8, help="PyTorch only (an engine has its own)")
    p.add_argument("--max-disp", type=int, default=192, help="PyTorch only")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp32",
                   help="PyTorch only (fp32 is faster than fp16 autocast on Jetson Thor)")
    p.add_argument("--every", type=int, default=1, help="only every N-th frame (for evaluation)")
    args = p.parse_args()

    meta = load_meta(args.frames)
    st = meta["stereo"]
    n, H, W = len(meta["frames"]), meta["height"], meta["width"]
    if st.get("projector") == "alternate" or os.path.isdir(os.path.join(args.frames, "ir1_on")):
        # each color frame's own IR pair (alternating; or a live session with the projector off,
        # whose color is kept at half the IR rate)
        ir_dirs = ("ir1_on", "ir2_on")
    elif len(st.get("frames", meta["frames"])) == n:
        ir_dirs = ("ir1", "ir2")
    else:
        raise SystemExit("IR frames are not numbered like color frames")
    dev = torch.device("cuda")
    to_color = DisparityToColorDepth(meta, dev)

    ffs = FFS(engine=args.engine, valid_iters=args.valid_iters, max_disp=args.max_disp, precision=args.precision)
    ids = list(range(0, n, args.every))

    out = os.path.join(args.frames, args.out_dir)
    os.makedirs(out, exist_ok=True)
    scale = meta["depth_scale"]

    def read(i):
        paths = [os.path.join(args.frames, d, f"{i:06d}.png") for d in ir_dirs]
        return [cv2.imread(q, cv2.IMREAD_GRAYSCALE) for q in paths] if all(map(os.path.exists, paths)) else None

    t0, missing, valid = time.time(), 0, []
    with ThreadPoolExecutor(4) as pool:
        pending = [pool.submit(read, i) for i in ids[:8]]
        writes = []
        for k, i in enumerate(ids):
            pair = pending[k].result()
            if k + 8 < len(ids):
                pending.append(pool.submit(read, ids[k + 8]))
            if pair is None:  # no IR pair for this frameset: keep the sensor depth
                missing += 1
                writes.append(pool.submit(cv2.imwrite, os.path.join(out, f"{i:06d}.png"),
                                          cv2.imread(os.path.join(args.frames, "depth", f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)))
                continue
            depth = to_color(ffs(*(torch.as_tensor(x, device=dev) for x in pair)))
            d16 = torch.clamp(torch.round(depth * scale), 0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
            valid.append((d16 > 0).mean())
            writes.append(pool.submit(cv2.imwrite, os.path.join(out, f"{i:06d}.png"), d16))
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(ids)} frames, {(k + 1) / (time.time() - t0):.1f} fps")
        for w in writes:
            w.result()
    print(f"{len(ids) - missing} frames from {ir_dirs[0]}/{ir_dirs[1]} ({missing} without an IR pair kept the sensor depth), "
          f"valid depth {np.mean(valid) * 100:.0f}% of pixels, {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
