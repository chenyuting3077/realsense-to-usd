"""Distance-weighted TSDF (Triton on the GPU) -> mesh: live.py's final mesh, and offline
<frames>/ + poses.npy -> <out>/mesh.ply, pointcloud.ply (the 3DGS initialization).

Usage: python recon/reconstruct.py output/live_room/frames output/live_room/recon
"""
import argparse
import os
import sysconfig

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

from frames import intrinsic_matrix, iter_frames, load_meta

VOXEL, TRUNC_VOX, DEPTH_MAX = 0.01, 4.0, 4.0
WEIGHT_REF = 2.0  # observations within this depth (m) weigh 1, farther ones (WEIGHT_REF / z)^2
MESH_MIN_WEIGHT, MIN_CLUSTER = 3.0, 500
# Open3D's CPU marching cubes segfaults past ~2^31 / 16 voxels: larger grids are meshed in slabs.
MAX_MESH_VOXELS = 100_000_000


def extract_mesh(grid, block_resolution):
    """Marching cubes, slab by slab along the longest axis when the grid is too large. Each slab
    gets one block of overlap; triangles are kept in the slab of their centroid."""
    hm = grid.hashmap()
    n = hm.size()
    if n * block_resolution ** 3 <= MAX_MESH_VOXELS:
        return grid.extract_triangle_mesh(MESH_MIN_WEIGHT).to_legacy()
    idx = hm.active_buf_indices()
    keys = hm.key_tensor()[idx]
    values = [v[idx] for v in hm.value_tensors()]
    k = keys.numpy()
    axis = int(np.argmax(k.max(0) - k.min(0)))
    n_slabs = int(np.ceil(n * block_resolution ** 3 / (0.8 * MAX_MESH_VOXELS)))
    cuts = np.unique(np.quantile(k[:, axis], np.linspace(0, 1, n_slabs + 1)[1:-1]).astype(int))
    bounds = [k[:, axis].min(), *cuts, k[:, axis].max() + 1]
    names = ["tsdf", "weight", "color"]
    attrs = [grid.attribute(a) for a in names]
    mesh = o3d.geometry.TriangleMesh()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = np.nonzero((k[:, axis] >= lo - 1) & (k[:, axis] < hi + 1))[0]
        sub = o3d.t.geometry.VoxelBlockGrid(names, [a.dtype for a in attrs], [a.shape[4:] or [1] for a in attrs],
                                            VOXEL, block_resolution, len(sel), hm.device)
        sel_t = o3c.Tensor(sel.astype(np.int64))
        sub.hashmap().insert(keys[sel_t], [v[sel_t] for v in values])
        part = sub.extract_triangle_mesh(MESH_MIN_WEIGHT).to_legacy()
        c = np.floor(np.asarray(part.vertices)[np.asarray(part.triangles)].mean(1)[:, axis] / (VOXEL * block_resolution))
        part.remove_triangles_by_mask((c < lo) | (c >= hi))
        mesh += part
    return mesh


def clean_mesh(mesh):
    """Merge duplicated vertices, drop pieces under MIN_CLUSTER triangles (stereo noise)."""
    mesh.remove_duplicated_vertices()
    clusters, counts, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(clusters)] < MIN_CLUSTER)
    mesh.remove_unreferenced_vertices()
    return mesh


_KERNEL = None


def _integrate_kernel():
    """One program per (block, 512 voxels): project each voxel into the depth image and update its
    tsdf / weight / color with weight min(1, (weight_ref / z)^2)."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    # Triton's bundled ptxas does not know Thor (sm_110a): use the env's CUDA 13 one
    os.environ.setdefault("TRITON_PTXAS_PATH", os.path.join(sysconfig.get_paths()["purelib"], "nvidia/cu13/bin/ptxas"))
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(keys, buf, depth, color, tsdf, weight, rgb,
               r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
               fx, fy, cx, cy, H, W, depth_scale, voxel_size, trunc, depth_max, weight_ref,
               RES: tl.constexpr, VOXELS: tl.constexpr):
        b = tl.program_id(0)
        local = tl.program_id(1) * VOXELS + tl.arange(0, VOXELS)  # x fastest, as Open3D
        kx = tl.load(keys + b * 3)
        ky = tl.load(keys + b * 3 + 1)
        kz = tl.load(keys + b * 3 + 2)
        px = (kx * RES + local % RES).to(tl.float32) * voxel_size
        py = (ky * RES + (local // RES) % RES).to(tl.float32) * voxel_size
        pz = (kz * RES + local // (RES * RES)).to(tl.float32) * voxel_size
        x = r00 * px + r01 * py + r02 * pz + t0
        y = r10 * px + r11 * py + r12 * pz + t1
        z = r20 * px + r21 * py + r22 * pz + t2
        zs = tl.where(z > 0, z, 1.0)
        u = tl.floor(fx * x / zs + cx + 0.5).to(tl.int32)
        v = tl.floor(fy * y / zs + cy + 0.5).to(tl.int32)
        ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        pix = tl.where(ok, v * W + u, 0)
        d = tl.load(depth + pix, mask=ok, other=0).to(tl.float32) * depth_scale
        sdf = d - z
        ok = ok & (d > 0) & (d <= depth_max) & (sdf >= -trunc)
        sdf = tl.minimum(sdf, trunc) / trunc
        ds = tl.where(d > 0, d, 1.0)
        w = tl.minimum(weight_ref * weight_ref / (ds * ds), 1.0)
        idx = tl.load(buf + b) * (RES * RES * RES) + local
        w_old = tl.load(weight + idx, mask=ok, other=0)
        w_new = w_old + w
        tsdf_old = tl.load(tsdf + idx, mask=ok, other=0)
        tl.store(tsdf + idx, (tsdf_old * w_old + sdf * w) / w_new, mask=ok)
        tl.store(weight + idx, w_new, mask=ok)
        for c in tl.static_range(3):
            obs = tl.load(color + pix * 3 + c, mask=ok, other=0).to(tl.float32)
            old = tl.load(rgb + idx * 3 + c, mask=ok, other=0)
            tl.store(rgb + idx * 3 + c, (old * w_old + obs * w) / w_new, mask=ok)

    _KERNEL = kernel
    return kernel


class WeightedTSDF:
    """Near measurements dominate the far, noisier ones (stereo error grows with z^2). Open3D only
    finds each frame's blocks and keeps the block hash map; tsdf / weight / color are torch tensors
    on the GPU indexed like Open3D's voxel blocks, updated by the Triton kernel."""

    RES = 16
    VOXELS_PER_PROGRAM = 512

    def __init__(self, capacity=1_000_000):
        import torch
        self.torch, self.capacity = torch, capacity
        self.blocks = o3d.t.geometry.VoxelBlockGrid(["tsdf"], [o3c.float32], [[1]], VOXEL, self.RES, 1)
        self.hashmap = o3c.HashMap(capacity, o3c.int32, [3], o3c.int32, [1])
        self.keys = torch.zeros((0, 3), dtype=torch.int32, device="cuda")
        self.tsdf = torch.zeros(0, device="cuda")
        self.weight = torch.zeros(0, device="cuda")
        self.color = torch.zeros((0, 3), device="cuda")

    def _grow(self, n_blocks):
        t, n_old = self.torch, len(self.keys)
        if n_blocks <= n_old:
            return
        n, v = max(n_blocks, int(n_old * 1.5)), self.RES ** 3
        self.keys = t.cat([self.keys, t.zeros((n - n_old, 3), dtype=t.int32, device="cuda")])
        self.tsdf = t.cat([self.tsdf, t.zeros((n - n_old) * v, device="cuda")])
        self.weight = t.cat([self.weight, t.zeros((n - n_old) * v, device="cuda")])
        self.color = t.cat([self.color, t.zeros(((n - n_old) * v, 3), device="cuda")])

    def integrate(self, depth, color, K, T_wc, depth_scale):
        """depth in units of 1 / depth_scale m, color HxWx3 uint8; numpy or torch."""
        t = self.torch
        depth_np = depth.cpu().numpy() if t.is_tensor(depth) else depth
        Tcw = np.linalg.inv(T_wc)
        coords = self.blocks.compute_unique_block_coordinates(
            o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth_np[..., None]))),
            o3c.Tensor(K, o3c.float64), o3c.Tensor(Tcw), depth_scale, DEPTH_MAX, TRUNC_VOX)
        if coords.shape[0] == 0:
            return
        assert self.hashmap.size() + coords.shape[0] <= self.capacity, "TSDF block capacity exceeded"
        self.hashmap.activate(coords)
        buf, _ = self.hashmap.find(coords)
        buf = t.from_numpy(buf.numpy().astype(np.int64)).cuda()
        self._grow(int(buf.max()) + 1)
        keys = t.from_numpy(coords.numpy()).cuda()
        self.keys[buf] = keys
        H, W = depth.shape
        _integrate_kernel()[(len(buf), self.RES ** 3 // self.VOXELS_PER_PROGRAM)](
            keys, buf, t.as_tensor(depth, device="cuda").contiguous(), t.as_tensor(color, device="cuda").contiguous(),
            self.tsdf, self.weight, self.color,
            *Tcw[:3, :3].ravel().tolist(), *Tcw[:3, 3].tolist(),
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), H, W,
            1.0 / depth_scale, VOXEL, VOXEL * TRUNC_VOX, DEPTH_MAX, WEIGHT_REF,
            RES=self.RES, VOXELS=self.VOXELS_PER_PROGRAM)

    def voxel_grid(self):
        """The integrated blocks as a CPU VoxelBlockGrid."""
        act = self.torch.from_numpy(self.hashmap.active_buf_indices().numpy().astype(np.int64)).cuda()
        r = self.RES
        attr = lambda a, c: o3c.Tensor(a.reshape(-1, r, r, r, c)[act].cpu().numpy())  # noqa: E731
        grid = o3d.t.geometry.VoxelBlockGrid(["tsdf", "weight", "color"], [o3c.float32] * 3, [[1], [1], [3]],
                                             VOXEL, r, len(act), o3c.Device("CPU:0"))
        grid.hashmap().insert(o3c.Tensor(self.keys[act].cpu().numpy()),
                              [attr(self.tsdf, 1), attr(self.weight, 1), attr(self.color, 3)])
        return grid

    def mesh(self, grid=None):
        return clean_mesh(extract_mesh(grid or self.voxel_grid(), self.RES))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", help="frames/ of a live session (with poses.npy)")
    p.add_argument("out")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta = load_meta(args.frames)
    K, poses = intrinsic_matrix(meta), np.load(os.path.join(args.frames, "poses.npy"))
    tsdf = WeightedTSDF()
    masks = os.path.join(args.frames, "mask")  # person_mask.py: people (0) stay out of the mesh
    for i, color, depth in iter_frames(args.frames, meta, "depth_ffs"):
        if os.path.isdir(masks):
            depth[cv2.imread(os.path.join(masks, f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) == 0] = 0
        tsdf.integrate(depth, color, K, poses[i], meta["depth_scale"])  # filtered when live.py saved it
    grid = tsdf.voxel_grid()
    mesh = tsdf.mesh(grid)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(args.out, "mesh.ply"), mesh)
    o3d.io.write_point_cloud(os.path.join(args.out, "pointcloud.ply"), grid.extract_point_cloud().to_legacy())
    print(f"mesh: {len(mesh.triangles)} triangles -> {args.out}/mesh.ply")


if __name__ == "__main__":
    main()
