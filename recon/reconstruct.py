"""Reconstruct a scene mesh from a frames dir (frames/ or rec/ of a live session).

Pipeline: color + aligned depth -> depth filtering
  -> camera poses: frame-to-model RGB-D odometry (open3d.t.pipelines.slam), or
     --poses from a tracker (live.py, track_orbslam3.py)
  -> TSDF integration -> marching cubes mesh extraction.

Open3D's frame-to-model tracking needs CUDA (its bilateral filter has no CPU kernel);
the aarch64 Open3D wheels (e.g. Jetson Thor) are CPU-only, so there --poses is required.

Usage (after `source env.sh`):
  python recon/reconstruct.py output/room/frames -o output/room/recon --poses output/room/frames/poses.npy
  # depth up to 4 m, near observations weighted higher (Triton integration on the GPU)
  python recon/reconstruct.py output/room/frames -o output/room/recon --poses output/room/frames/poses.npy \
      --depth-dir depth_ffs --depth-max 4 --weighting distance --weight-ref 2
"""
import argparse
import json
import os
import time

import numpy as np
import open3d as o3d
import open3d.core as o3c

from frames import filter_depth, intrinsic_matrix, iter_frames, load_meta

# Open3D's CPU marching cubes overflows (segfault) once the grid holds more than about
# 2^31 / 16 voxels (~33k blocks of 16^3, e.g. a whole room at 1 cm); larger grids are meshed
# in slabs below this.
MAX_MESH_VOXELS = 100_000_000


def extract_mesh(grid, voxel_size, block_resolution, weight_threshold=3.0):
    """Marching cubes over the TSDF grid, slab by slab when the grid is too large.

    Each slab (along the longest axis, equal block counts) is copied into its own grid with
    one block of overlap on both sides, so the cells across the cut see their neighbours;
    of the triangles in the overlap only those whose centroid lies in the slab are kept,
    and the vertices shared at the cuts are merged afterwards.
    """
    hm = grid.hashmap()
    n = hm.size()
    if n * block_resolution ** 3 <= MAX_MESH_VOXELS:
        return grid.extract_triangle_mesh(weight_threshold).to_legacy()
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
    block_size = voxel_size * block_resolution
    mesh = o3d.geometry.TriangleMesh()
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        sel = np.nonzero((k[:, axis] >= lo - 1) & (k[:, axis] < hi + 1))[0]
        sub = o3d.t.geometry.VoxelBlockGrid(names, [a.dtype for a in attrs], [a.shape[4:] or [1] for a in attrs],
                                            voxel_size, block_resolution, len(sel), grid.hashmap().device)
        sel_t = o3c.Tensor(sel.astype(np.int64))
        sub.hashmap().insert(keys[sel_t], [v[sel_t] for v in values])
        part = sub.extract_triangle_mesh(weight_threshold).to_legacy()
        tri = np.asarray(part.triangles)
        c = np.floor(np.asarray(part.vertices)[tri].mean(1)[:, axis] / block_size)
        part.remove_triangles_by_mask((c < lo) | (c >= hi))
        mesh += part
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh


_KERNEL = None


def _integrate_kernel():
    """Triton kernel: one program per (block, 512 voxels); each voxel is projected into the
    depth image, and its tsdf / weight / color updated with the observation's weight."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
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
        # world corner of the voxel -> camera
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
    """TSDF integration where each observation counts by its distance: weight
    min(1, (weight_ref / z)^2) for depth z, so near measurements dominate the far, noisier
    ones (stereo depth error grows with z^2). Open3D's own integration counts every
    observation 1.

    Open3D (CPU on Thor) only finds the frame's blocks and keeps the block hash map;
    tsdf / weight / color live in torch tensors on the GPU indexed like Open3D's voxel blocks
    (buffer index * 16^3 + x + 16 y + 256 z), updated by one Triton kernel per frame (one
    pass per voxel, as nvblox's integrator), and copied into a VoxelBlockGrid for marching
    cubes at the end.
    """

    RES = 16
    VOXELS_PER_PROGRAM = 512

    def __init__(self, voxel_size, trunc_mult, depth_max, weight_ref, capacity):
        import torch
        self.torch = torch
        self.voxel_size, self.trunc = voxel_size, voxel_size * trunc_mult
        self.trunc_mult, self.depth_max, self.weight_ref = trunc_mult, depth_max, weight_ref
        self.capacity = capacity
        # only the hash map and the block coordinates of a frame are used from this grid
        self.blocks = o3d.t.geometry.VoxelBlockGrid(["tsdf"], [o3c.float32], [[1]], voxel_size, self.RES, 1)
        self.hashmap = o3c.HashMap(capacity, o3c.int32, [3], o3c.int32, [1])
        self.keys = torch.zeros((0, 3), dtype=torch.int32, device="cuda")
        self.tsdf = torch.zeros(0, device="cuda")
        self.weight = torch.zeros(0, device="cuda")
        self.color = torch.zeros((0, 3), device="cuda")

    def _grow(self, n_blocks):
        t = self.torch
        n_old = len(self.keys)
        if n_blocks <= n_old:
            return
        n = max(n_blocks, int(n_old * 1.5))
        v = self.RES ** 3
        self.keys = t.cat([self.keys, t.zeros((n - n_old, 3), dtype=t.int32, device="cuda")])
        self.tsdf = t.cat([self.tsdf, t.zeros((n - n_old) * v, device="cuda")])
        self.weight = t.cat([self.weight, t.zeros((n - n_old) * v, device="cuda")])
        self.color = t.cat([self.color, t.zeros(((n - n_old) * v, 3), device="cuda")])

    def integrate(self, depth_raw, color, K, T_wc, depth_scale):
        """depth_raw: HxW depth in units of 1 / depth_scale m (uint16 or float), color: HxWx3 uint8 RGB;
        numpy arrays or torch tensors (e.g. live.py's frames on the GPU)."""
        t = self.torch
        depth_np = depth_raw.cpu().numpy() if t.is_tensor(depth_raw) else depth_raw
        extrinsic = o3c.Tensor(np.linalg.inv(T_wc))
        coords = self.blocks.compute_unique_block_coordinates(
            o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth_np[..., None]))),
            o3c.Tensor(K, o3c.float64), extrinsic, depth_scale, self.depth_max, self.trunc_mult)
        if coords.shape[0] == 0:
            return
        assert self.hashmap.size() + coords.shape[0] <= self.capacity, "raise --block-count"
        self.hashmap.activate(coords)
        buf, _ = self.hashmap.find(coords)
        buf = t.from_numpy(buf.numpy().astype(np.int64)).cuda()
        self._grow(int(buf.max()) + 1)
        keys = t.from_numpy(coords.numpy()).cuda()
        self.keys[buf] = keys

        H, W = depth_raw.shape
        Tcw = np.linalg.inv(T_wc)
        grid = (len(buf), self.RES ** 3 // self.VOXELS_PER_PROGRAM)
        _integrate_kernel()[grid](
            keys, buf, t.as_tensor(depth_raw, device="cuda").contiguous(), t.as_tensor(color, device="cuda").contiguous(),
            self.tsdf, self.weight, self.color,
            *Tcw[:3, :3].ravel().tolist(), *Tcw[:3, 3].tolist(),
            float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), H, W,
            1.0 / depth_scale, self.voxel_size, self.trunc, self.depth_max, self.weight_ref,
            RES=self.RES, VOXELS=self.VOXELS_PER_PROGRAM)

    def voxel_grid(self):
        """A CPU VoxelBlockGrid holding the integrated blocks, for extract_mesh."""
        act = self.torch.from_numpy(self.hashmap.active_buf_indices().numpy().astype(np.int64)).cuda()
        n, r = len(act), self.RES
        attr = lambda a, c: o3c.Tensor(a.reshape(-1, r, r, r, c)[act].cpu().numpy())  # noqa: E731
        grid = o3d.t.geometry.VoxelBlockGrid(["tsdf", "weight", "color"], [o3c.float32] * 3, [[1], [1], [3]],
                                             self.voxel_size, r, n, o3c.Device("CPU:0"))
        grid.hashmap().insert(o3c.Tensor(self.keys[act].cpu().numpy()),
                              [attr(self.tsdf, 1), attr(self.weight, 1), attr(self.color, 3)])
        return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", help="frames dir")
    p.add_argument("-o", "--out", default="output")
    p.add_argument("--voxel-size", type=float, default=0.01)
    p.add_argument("--depth-max", type=float, default=2.5)
    p.add_argument("--depth-min", type=float, default=0.2)
    p.add_argument("--trunc-mult", type=float, default=4.0)
    p.add_argument("--depth-dir", default="depth",
                   help="depth subdirectory of frames, e.g. depth_ffs")
    p.add_argument("--edge-thresh", type=float, default=0.05,
                   help="drop depth pixels whose 3x3 neighborhood spans more than this "
                        "fraction of their depth (stereo flying pixels); 0 disables")
    p.add_argument("--grazing-deg", type=float, default=0.0,
                   help="drop depth pixels whose surface normal is more than this many "
                        "degrees from the viewing ray (stereo ramps between objects); 0 disables")
    p.add_argument("--block-count", type=int, default=100000,
                   help="initial voxel blocks (Open3D grows it); --weighting distance: the maximum")
    p.add_argument("--weighting", choices=["const", "distance"], default="const",
                   help="per-observation weight: const = 1 (Open3D), distance = min(1, (--weight-ref / z)^2)")
    p.add_argument("--weight-ref", type=float, default=1.0,
                   help="--weighting distance: depth (m) up to which an observation counts 1")
    p.add_argument("--mesh-min-weight", type=float, default=3.0,
                   help="voxels with less total weight are left out of the mesh")
    p.add_argument("--device", default="CUDA:0" if o3c.cuda.is_available() else "CPU:0")
    p.add_argument("--poses", help="Nx4x4 camera-to-world poses to integrate instead of tracking")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = o3c.Device(args.device)
    ext_poses = np.load(args.poses) if args.poses else None
    if ext_poses is None and device.get_type() == o3c.Device.DeviceType.CPU:
        p.error("Open3D frame-to-model tracking does not run on CPU; pass --poses "
                "(e.g. from track_orbslam3.py)")

    meta = load_meta(args.frames)
    depth_scale = meta["depth_scale"]
    K = intrinsic_matrix(meta)
    intrinsic = o3c.Tensor(K, o3c.float64)
    height, width = meta["height"], meta["width"]
    print(f"{len(meta['frames'])} frames {width}x{height} depth_scale={depth_scale:.1f}")
    if ext_poses is not None:
        assert len(ext_poses) == len(meta["frames"]), "--poses does not match this frames dir"
    if args.weighting == "distance":
        if ext_poses is None:
            p.error("--weighting distance needs --poses")
        integrate_weighted(args, meta, K, ext_poses)
        return

    T_frame_to_model = o3c.Tensor(np.identity(4))
    model = o3d.t.pipelines.slam.Model(args.voxel_size, 16, args.block_count,
                                       T_frame_to_model, device)
    input_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)
    raycast_frame = o3d.t.pipelines.slam.Frame(height, width, intrinsic, device)

    poses = []
    t0 = time.time()
    for i, color_np, depth_np in iter_frames(args.frames, meta, depth_dir=args.depth_dir):
        d = filter_depth(depth_np, K, args.edge_thresh, args.grazing_deg)
        depth = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(d[..., None]))).to(device)
        color = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(color_np))).to(device)
        input_frame.set_data_from_image("depth", depth)
        input_frame.set_data_from_image("color", color)

        if ext_poses is not None:
            T_frame_to_model = o3c.Tensor(ext_poses[i])
        elif i > 0:
            try:
                result = model.track_frame_to_model(input_frame, raycast_frame,
                                                    depth_scale, args.depth_max)
                T_frame_to_model = T_frame_to_model @ result.transformation
            except RuntimeError as e:
                print(f"frame {i}: tracking failed ({e}), keeping last pose")

        poses.append(T_frame_to_model.cpu().numpy())
        model.update_frame_pose(i, T_frame_to_model)
        model.integrate(input_frame, depth_scale, args.depth_max, args.trunc_mult)
        if ext_poses is None:  # the raycast is only the tracking target
            model.synthesize_model_frame(raycast_frame, depth_scale, args.depth_min,
                                         args.depth_max, args.trunc_mult, False)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} frames, {(i + 1) / (time.time() - t0):.1f} fps")
    print(f"integrated {len(poses)} frames in {time.time() - t0:.1f}s")

    if ext_poses is None:
        np.save(os.path.join(args.out, "poses.npy"), np.stack(poses))
        with open(os.path.join(args.out, "trajectory.json"), "w") as f:
            json.dump({"intrinsic": np.asarray(K).tolist(),
                       "poses": [T.tolist() for T in poses]}, f)

    save(args, model.voxel_grid)


def integrate_weighted(args, meta, K, poses):
    tsdf = WeightedTSDF(args.voxel_size, args.trunc_mult, args.depth_max, args.weight_ref,
                        max(args.block_count, 1_000_000))
    t0 = time.time()
    for i, color_np, depth_np in iter_frames(args.frames, meta, depth_dir=args.depth_dir):
        d = filter_depth(depth_np, K, args.edge_thresh, args.grazing_deg)
        tsdf.integrate(d, color_np, K, poses[i], meta["depth_scale"])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} frames, {(i + 1) / (time.time() - t0):.1f} fps")
    print(f"integrated {len(poses)} frames in {time.time() - t0:.1f}s ({tsdf.hashmap.size()} blocks)")
    save(args, tsdf.voxel_grid())


def save(args, grid):
    mesh = extract_mesh(grid, args.voxel_size, 16, args.mesh_min_weight)
    mesh.compute_vertex_normals()
    # Drop small floating fragments left by stereo depth noise.
    clusters, counts, _ = mesh.cluster_connected_triangles()
    clusters, counts = np.asarray(clusters), np.asarray(counts)
    mesh.remove_triangles_by_mask(counts[clusters] < 500)
    mesh.remove_unreferenced_vertices()
    out_mesh = os.path.join(args.out, "mesh.ply")
    o3d.io.write_triangle_mesh(out_mesh, mesh)
    print(f"mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles -> {out_mesh}")

    pcd = grid.extract_point_cloud().to_legacy()
    o3d.io.write_point_cloud(os.path.join(args.out, "pointcloud.ply"), pcd)


if __name__ == "__main__":
    main()
