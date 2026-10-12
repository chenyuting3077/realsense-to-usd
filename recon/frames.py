"""Frames dirs (numbered PNGs per stream + meta.json), the floor depth filter and pose helpers."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

DEVICE = torch.device("cuda")


def load_meta(frames_dir):
    with open(os.path.join(frames_dir, "meta.json")) as f:
        return json.load(f)


def intrinsic_matrix(meta):
    return np.array([[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]])


def _read_frame(frames_dir, i, depth_dir):
    name = f"{i:06d}.png"
    color = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, "color", name)), cv2.COLOR_BGR2RGB)
    return color, cv2.imread(os.path.join(frames_dir, depth_dir, name), cv2.IMREAD_UNCHANGED)


def iter_frames(frames_dir, meta, depth_dir, prefetch=16):
    """(index, color, depth) in order, decoded ahead on a thread pool."""
    n = len(meta["frames"])
    with ThreadPoolExecutor(6) as pool:
        pending = [pool.submit(_read_frame, frames_dir, k, depth_dir) for k in range(min(prefetch, n))]
        for k in range(n):
            color, depth = pending[k].result()
            pending[k] = None
            if k + prefetch < n:
                pending.append(pool.submit(_read_frame, frames_dir, k + prefetch, depth_dir))
            yield k, color, depth


class FloorDepth:
    """Depth beyond depth_max is cut, except on the floor (up to floor_max): seen from chest height
    the floor is mostly that far, and the shallow ray moves its depth error along the floor.
    The floor level is the lowest strong peak of the point heights along `up` (Z-up world) at least
    0.3 m below the camera, refined by a plane fit. Depth: float tensor in meters, T camera-to-world.
    Without `up` only the cut at depth_max is applied."""
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
        """A copy for the world moved by the rigid 4x4 M (a re-level)."""
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
        self.samples = torch.cat([self.samples, X[torch.randperm(len(X), device=DEVICE)[:2000]]])
        if len(self.samples) > self.SAMPLES:
            self.samples = self.samples[torch.randperm(len(self.samples), device=DEVICE)[:self.SAMPLES]]
        self.updates += 1

    def _level(self, T):
        cam = float(np.asarray(T)[:3, 3] @ self.up_np)
        h = self.hist[:max(int((cam - 0.3 + self.RANGE) / self.BIN), 0)]
        if len(h) == 0 or h.max() < self.MIN_POINTS:
            return None
        i = int(torch.nonzero(h >= self.FLOOR_SHARE * h.max())[0])
        i += int(h[i:i + 5].argmax())
        return (i + 0.5) * self.BIN - self.RANGE

    def floor(self, T):
        """(unit normal n, offset c) with n . x = c on the floor, or None; refitted every 10 updates."""
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
            P = X[(X @ n - c).abs() < thresh]
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
    """Poses (4x4 or None) at t_src -> at t_dst: slerp / lerp between tracked neighbours at most
    max_gap_ns apart, else the nearest tracked pose within max_gap_ns / 2, else None."""
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
    """None -> the previous pose (the first tracked one before it)."""
    out, last = [], next(T for T in poses if T is not None)
    for T in poses:
        last = T if T is not None else last
        out.append(last)
    return np.stack(out)
