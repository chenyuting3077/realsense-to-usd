"""Read the color/depth frames of a frames dir (frames/ or rec/ of a live session, see README), the depth filters shared by every
step that consumes them, and the pose helpers shared by the trackers."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta(frames_dir):
    with open(os.path.join(frames_dir, "meta.json")) as f:
        return json.load(f)


def intrinsic_matrix(meta):
    return np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])


def _read_frame(frames_dir, i, depth_dir="depth"):
    name = f"{i:06d}.png"
    color = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, "color", name)), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(os.path.join(frames_dir, depth_dir, name), cv2.IMREAD_UNCHANGED)
    return color, depth


def iter_frames(frames_dir, meta=None, prefetch=16, workers=6, depth_dir="depth", indices=None):
    """Yield (index, color uint8 HxWx3, depth uint16 HxW) in recording order (or of `indices`),
    depth from <frames_dir>/<depth_dir> (e.g. depth_ffs). PNGs are
    decoded ahead on a thread pool (cv2 releases the GIL), so decoding overlaps the caller."""
    meta = meta or load_meta(frames_dir)
    idx = list(range(len(meta["frames"])) if indices is None else indices)
    n = len(idx)
    with ThreadPoolExecutor(workers) as pool:
        pending = [pool.submit(_read_frame, frames_dir, idx[k], depth_dir) for k in range(min(prefetch, n))]
        for k in range(n):
            color, depth = pending[k].result()
            pending[k] = None
            if k + prefetch < n:
                pending.append(pool.submit(_read_frame, frames_dir, idx[k + prefetch], depth_dir))
            yield idx[k], color, depth


def _replicate_pad(x, p):
    """Edge-pad the last two dims of a (C, H, W) or (H, W) tensor by p."""
    y = x if x.dim() == 3 else x[None]
    y = F.pad(y[None], (p, p, p, p), mode="replicate")[0]
    return y if x.dim() == 3 else y[0]


def filter_flying_pixels(d, rel_thresh):
    """Zero out depth pixels on depth discontinuities, where stereo matching smears
    foreground and background together. d: float HxW tensor."""
    h, w = d.shape
    pad = _replicate_pad(d, 1)
    stack = torch.stack([pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                         for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    valid = stack > 0
    dmax = torch.where(valid, stack, 0).amax(0)
    dmin = torch.where(valid, stack, torch.inf).amin(0)
    bad = ~valid.all(0) | ((dmax - dmin) > rel_thresh * d)
    return torch.where(bad, 0, d)


def filter_grazing(d, K, max_deg, step=3):
    """Zero out depth pixels whose local surface is nearly parallel to the viewing ray.
    Stereo depth interpolates smooth ramps across occlusion boundaries; these ramps are
    what smears thin objects in the fused mesh. d: float HxW tensor."""
    h, w = d.shape
    fx, fy, cx, cy = (float(x) for x in (K[0][0], K[1][1], K[0][2], K[1][2]))
    v, u = torch.meshgrid(torch.arange(h, dtype=d.dtype, device=d.device),
                          torch.arange(w, dtype=d.dtype, device=d.device), indexing="ij")
    pts = torch.stack([(u - cx) / fx * d, (v - cy) / fy * d, d])  # 3 x H x W
    pad = _replicate_pad(pts, step)
    du = pad[:, step:step + h, 2 * step:] - pad[:, step:step + h, :w]
    dv = pad[:, 2 * step:, step:step + w] - pad[:, :h, step:step + w]
    n = torch.linalg.cross(du, dv, dim=0)
    cos = (n * pts).sum(0).abs() / torch.clamp(n.norm(dim=0) * pts.norm(dim=0), min=1e-9)
    return torch.where(cos < np.cos(np.deg2rad(max_deg)), 0, d)


def filter_depth(depth, K, edge_thresh, grazing_deg, as_tensor=False):
    """Depth filters on the GPU when there is one. depth: uint16 numpy HxW (or a tensor).
    Returns the same uint16 numpy array type, or with as_tensor the float tensor on DEVICE
    (depth units), which saves a round trip for GPU consumers like nvblox."""
    d = torch.as_tensor(depth.astype(np.float32) if isinstance(depth, np.ndarray) else depth,
                        device=DEVICE).float()
    if edge_thresh > 0:
        d = filter_flying_pixels(d, edge_thresh)
    if grazing_deg > 0:
        d = filter_grazing(d, K, grazing_deg)
    if as_tensor:
        return d
    return d.to(torch.int32).cpu().numpy().astype(np.uint16)


def world_up(meta, first_pose):
    """Up direction in the world of the color poses, from the accelerometer's "up_ir1" (measured at
    the first frame) and that frame's pose; None without it."""
    if not meta.get("up_ir1") or "stereo" not in meta:
        return None
    R_ir1_from_color = np.array(meta["stereo"]["ir1_from_color"])[:3, :3]
    up = np.asarray(first_pose)[:3, :3] @ R_ir1_from_color.T @ np.asarray(meta["up_ir1"])
    return up / np.linalg.norm(up)


class FloorDepth:
    """Depth beyond depth_max is noisy and left out, except on the floor, which a camera held at
    chest height mostly sees beyond it: past ~2.3 m when looking level. There the ray is shallow,
    so the depth error moves a point mostly along the floor and little off it. Pixels beyond
    depth_max (up to floor_max) are kept when they lie within `band` of the floor plane.

    The floor is found in the points within depth_max: a histogram of their heights along `up`
    (2 cm bins) gives its level, the lowest bin with at least FLOOR_SHARE of the largest one at
    least 0.3 m below the camera; a plane is then fitted to the sampled points near that level
    (it may tilt a little against `up`, as a leveling error or drift makes it). update() adds a
    frame, apply() filters one. Depth is a float tensor in meters on the GPU, T the 4x4
    camera-to-world pose. Without `up` (no accelerometer) apply() only cuts at depth_max."""
    BIN, RANGE, FLOOR_SHARE, MIN_POINTS, STEP, SAMPLES, MAX_TILT_DEG = 0.02, 10.0, 0.3, 2000, 4, 200_000, 15

    def __init__(self, K, width, height, up, depth_max, floor_max, band=0.04):
        self.up_np = None if up is None else np.asarray(up, np.float32)
        self.up = None if up is None else torch.tensor(self.up_np, device=DEVICE)
        self.depth_max, self.floor_max, self.band = depth_max, floor_max, band
        v, u = torch.meshgrid(torch.arange(height, device=DEVICE, dtype=torch.float32),
                              torch.arange(width, device=DEVICE, dtype=torch.float32), indexing="ij")
        self.rays = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)
        self.hist = torch.zeros(int(2 * self.RANGE / self.BIN), device=DEVICE)
        self.samples = torch.zeros(0, 3, device=DEVICE)  # world points within depth_max
        self.plane, self.fitted_at, self.updates = None, -1, 0

    def copy(self):
        c = object.__new__(FloorDepth)
        c.__dict__.update(self.__dict__, hist=self.hist.clone(), samples=self.samples.clone())
        return c

    def moved(self, M):
        """A copy for a world moved by the rigid 4x4 M (a re-level): the sampled points moved, the
        height histogram rebuilt from them."""
        c = self.copy()
        if self.up is None:
            return c
        M = torch.as_tensor(np.asarray(M, np.float32), device=DEVICE)
        c.samples = self.samples @ M[:3, :3].T + M[:3, 3]
        c.hist = torch.histc(c.samples @ self.up, len(self.hist), -self.RANGE, self.RANGE) \
            * (float(self.hist.sum()) / max(len(self.samples), 1))
        c.plane = None
        return c

    def _points(self, d, T, sl=(slice(None), slice(None))):
        T = torch.as_tensor(np.asarray(T, np.float32), device=DEVICE)
        return (self.rays[sl] * d[sl][..., None]) @ T[:3, :3].T + T[:3, 3]

    def update(self, d, T):
        if self.up is None:
            return
        s = (slice(None, None, self.STEP), slice(None, None, self.STEP))
        X = self._points(d, T, s)[(d[s] > 0) & (d[s] <= self.depth_max)]
        self.hist += torch.histc(X @ self.up, len(self.hist), -self.RANGE, self.RANGE)
        X = X[torch.randperm(len(X), device=DEVICE)[:2000]]
        self.samples = torch.cat([self.samples, X])
        if len(self.samples) > self.SAMPLES:  # keep a uniform sample of everything seen
            self.samples = self.samples[torch.randperm(len(self.samples), device=DEVICE)[:self.SAMPLES]]
        self.updates += 1

    def _level(self, T):
        cam = float(np.asarray(T)[:3, 3] @ self.up_np)
        h = self.hist[:max(int((cam - 0.3 + self.RANGE) / self.BIN), 0)]
        if len(h) == 0 or h.max() < self.MIN_POINTS:
            return None
        i = int(torch.nonzero(h >= self.FLOOR_SHARE * h.max())[0])
        i += int(h[i:i + 5].argmax())  # the peak of that level
        return (i + 0.5) * self.BIN - self.RANGE

    def floor(self, T):
        """The floor plane (unit normal n, offset c: n . x = c on it), or None while it has not
        been seen below the camera. Refitted after every 10 updates."""
        if self.up is None:
            return None
        if self.plane is not None and self.updates - self.fitted_at < 10:
            return self.plane
        level = self._level(T)
        if level is None:
            return self.plane
        X = self.samples[(self.samples @ self.up - level).abs() < 0.15]
        n, c = self.up, torch.tensor(level, device=DEVICE)
        for thresh in (0.10, 0.05, 0.03):  # least squares, dropping points off the plane
            r = X @ n - c
            P = X[r.abs() < thresh]
            if len(P) < 500:
                return self.plane
            m = P.mean(0)
            n = torch.linalg.svd(P - m, full_matrices=False).Vh[-1]
            n = n if n @ self.up > 0 else -n
            c = n @ m
        if float(n @ self.up) < np.cos(np.radians(self.MAX_TILT_DEG)):
            return self.plane
        self.plane, self.fitted_at = (n, c), self.updates
        return self.plane

    def apply(self, d, T):
        far = d > self.depth_max
        if not bool(far.any()):
            return d
        plane = self.floor(T)
        keep = torch.zeros_like(far)
        if plane is not None:
            n, c = plane
            keep = far & (d <= self.floor_max) & ((self._points(d, T) @ n - c).abs() < self.band)
        return torch.where(far & ~keep, torch.zeros_like(d), d)


def interpolate(poses, t_src, t_dst, max_gap_ns=100_000_000):
    """Poses (4x4 or None) at t_src -> poses at t_dst. Between two tracked neighbours at most
    max_gap_ns apart: slerp rotation, lerp translation. Otherwise the nearest tracked pose
    within max_gap_ns / 2, else None."""
    t_src = np.asarray(t_src)
    ok = np.array([T is not None for T in poses])
    out = []
    for t in t_dst:
        j = int(np.searchsorted(t_src, t))
        a, b = j - 1, j
        if 0 <= a and b < len(t_src) and ok[a] and ok[b] and t_src[b] - t_src[a] <= max_gap_ns:
            w = (t - t_src[a]) / (t_src[b] - t_src[a]) if t_src[b] > t_src[a] else 0.0
            Ra, Rb = Rotation.from_matrix(poses[a][:3, :3]), Rotation.from_matrix(poses[b][:3, :3])
            T = np.eye(4)
            T[:3, :3] = (Ra * Rotation.from_rotvec(w * (Ra.inv() * Rb).as_rotvec())).as_matrix()
            T[:3, 3] = (1 - w) * poses[a][:3, 3] + w * poses[b][:3, 3]
            out.append(T)
            continue
        near = [k for k in (a, b) if 0 <= k < len(t_src) and ok[k] and abs(t_src[k] - t) <= max_gap_ns / 2]
        out.append(poses[min(near, key=lambda k: abs(t_src[k] - t))] if near else None)
    return out


def fill_untracked(poses):
    """Replace None entries with the previous pose (the first valid one before it appears)."""
    out, last = [], next(T for T in poses if T is not None)
    for T in poses:
        last = T if T is not None else last
        out.append(last)
    return np.stack(out)
