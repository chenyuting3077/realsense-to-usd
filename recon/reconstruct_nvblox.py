"""Integrate a frames dir into an nvblox TSDF and extract a color mesh.

nvblox does not estimate camera poses; they come from track_orbslam3.py (or reconstruct.py's
Open3D odometry on CUDA machines), saved as poses.npy.

Usage (in the activated conda env):
  python recon/reconstruct_nvblox.py output/room/frames output/room/frames/poses.npy -o output/room/recon
"""
import argparse
import os
import time

import numpy as np
import open3d as o3d
import torch
from nvblox_torch.mapper import Mapper
from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType
from nvblox_torch.mapper_params import (MapperParams, MeshIntegratorParams, ProjectiveIntegratorParams,
                                       ViewCalculatorParams)
from nvblox_torch.sensor import Sensor

from frames import FloorDepth, filter_depth, intrinsic_matrix, iter_frames, load_meta, world_up


def add_mapper_args(p):
    """TSDF / mesh options shared with live.py."""
    p.add_argument("--voxel-size", type=float, default=0.01)
    p.add_argument("--depth-max", type=float, default=2.5)
    p.add_argument("--floor-depth-max", type=float, default=4.0,
                   help="on the floor, depth is kept up to this far (frames.FloorDepth; needs the "
                        "accelerometer's up_ir1; <= --depth-max: off)")
    p.add_argument("--floor-band", type=float, default=0.04,
                   help="pixels beyond --depth-max within this height of the floor plane are kept (m)")
    p.add_argument("--max-weight", type=float, default=1e6,
                   help="cap on each voxel's TSDF weight. nvblox's own default (5) keeps only the last few "
                        "observations so a live map can follow moving things; offline, for a static scene, "
                        "every frame is averaged (smoother surface, fewer double layers)")
    p.add_argument("--mesh-min-weight", type=float, default=3.0,
                   help="voxels with less weight are left out of the mesh (nvblox default 1e-4 keeps "
                        "surfaces seen once, i.e. edge noise)")
    p.add_argument("--weighting", default="kInverseSquareWeight",
                   help="per-observation weight: kConstantWeight (as Open3D), kInverseSquareWeight (nvblox default, 1/z^2), ...")
    p.add_argument("--trunc-vox", type=float, default=4.0, help="truncation distance in voxels")
    p.add_argument("--raycast-subsample", type=int, default=4,
                   help="every N-th depth pixel is raycast to find the blocks in view (nvblox default 4)")
    p.add_argument("--free-space-vox", type=float, default=16,
                   help="clear free space only up to this many voxels in front of the measured surface "
                        "(needs patches/nvblox-thor.patch; < 0: along the whole ray, as nvblox does). Clearing "
                        "the whole ray lets noisy views erase surfaces seen well from elsewhere (coverage "
                        "85%% -> 79%% on room_0930b); 16 matches Open3D, which only touches the 16^3 blocks "
                        "around each frame's surface points")
    p.add_argument("--min-cluster", type=int, default=500,
                   help="drop connected pieces with fewer triangles (as reconstruct.py)")
    p.add_argument("--color-sphere-trace", action="store_true",
                   help="find the voxels to color as nvblox does by default (view frustum + sphere tracing the "
                        "TSDF, 7.5 ms per frame) instead of from the depth frame, which is aligned to the color "
                        "camera (1.6 ms; needs patches/nvblox-thor.patch)")
    p.add_argument("--edge-thresh", type=float, default=0.05)
    p.add_argument("--grazing-deg", type=float, default=0.0)


def make_mapper(args, num_maps=1):
    """nvblox mapper with num_maps independent maps (mapper_id 0..num_maps-1) of the same settings."""
    os.environ["NVBLOX_FREE_SPACE_VOX"] = str(args.free_space_vox)  # read by the patched TSDF integrator
    integrator_params = ProjectiveIntegratorParams()
    # The depth is cut at --depth-max beforehand (FloorDepth.apply), except on the floor.
    integrator_params.projective_integrator_max_integration_distance_m = max(args.depth_max, args.floor_depth_max)
    integrator_params.projective_integrator_max_weight = args.max_weight
    integrator_params.projective_integrator_weighting_mode = args.weighting
    integrator_params.projective_integrator_truncation_distance_vox = args.trunc_vox
    mesh_params = MeshIntegratorParams()
    mesh_params.mesh_integrator_min_weight = args.mesh_min_weight
    view_params = ViewCalculatorParams()
    view_params.raycast_subsampling_factor = args.raycast_subsample
    mapper_params = MapperParams()
    mapper_params.set_view_calculator_params(view_params)
    mapper_params.set_projective_integrator_params(integrator_params)
    mapper_params.set_mesh_integrator_params(mesh_params)
    return Mapper(voxel_sizes_m=[args.voxel_size] * num_maps,
                  integrator_types=[ProjectiveIntegratorType.TSDF] * num_maps, mapper_parameters=mapper_params)


def clean_mesh(mesh, min_cluster):
    """Join nvblox's per-block meshes at the seams and drop pieces under min_cluster triangles."""
    mesh.remove_duplicated_vertices()
    if min_cluster > 0:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        clusters, counts = np.asarray(clusters), np.asarray(counts)
        mesh.remove_triangles_by_mask(counts[clusters] < min_cluster)
        mesh.remove_unreferenced_vertices()
    return mesh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames", help="frames dir")
    p.add_argument("poses", help="Nx4x4 camera-to-world poses (poses.npy)")
    p.add_argument("-o", "--out", default="output")
    p.add_argument("--depth-dir", default="depth",
                   help="depth subdirectory of frames, e.g. depth_ffs")
    p.add_argument("--color-every", type=int, default=1,
                   help="integrate color every N-th frame (0: no color); the geometry does not depend on it")
    add_mapper_args(p)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    poses = np.load(args.poses)
    meta = load_meta(args.frames)
    K_np = intrinsic_matrix(meta)
    assert len(poses) == len(meta["frames"]), "poses.npy does not match this frames dir"
    sensor = Sensor.from_camera_matrix(torch.tensor(K_np, dtype=torch.float32),
                                       meta["width"], meta["height"])
    mapper = make_mapper(args)
    up = world_up(meta, poses[0])
    floor = FloorDepth(K_np, meta["width"], meta["height"], up, args.depth_max, args.floor_depth_max,
                       args.floor_band)
    if up is not None and args.floor_depth_max > args.depth_max:  # floor height from every 5th frame first
        for i, _, depth_np in iter_frames(args.frames, meta, depth_dir=args.depth_dir, indices=range(0, len(poses), 5)):
            floor.update(filter_depth(depth_np, K_np, args.edge_thresh, args.grazing_deg,
                                      as_tensor=True) / meta["depth_scale"], poses[i])
        plane = floor.floor(poses[0])
        if plane is None:
            print("floor: not found, depth cut at --depth-max everywhere")
        else:
            n, c = (x.cpu().numpy() for x in plane)
            print(f"floor: {poses[0][:3, 3] @ n - c:.2f} m below the first camera, tilted "
                  f"{np.degrees(np.arccos(min(1.0, n @ up))):.1f} deg from up; depth kept up to "
                  f"{args.floor_depth_max} m on it")

    t0 = time.time()
    for i, color_np, depth_np in iter_frames(args.frames, meta, depth_dir=args.depth_dir):
        # Filtered on the GPU and kept there (float, sensor units) -> meters.
        depth = filter_depth(depth_np, K_np, args.edge_thresh, args.grazing_deg,
                             as_tensor=True) / meta["depth_scale"]
        depth = floor.apply(depth, poses[i])
        pose = torch.from_numpy(poses[i].astype(np.float32))
        mapper.add_depth_frame(depth, pose, sensor)
        if args.color_every > 0 and i % args.color_every == 0:
            mapper.add_color_frame(torch.from_numpy(np.ascontiguousarray(color_np)).cuda(), pose, sensor,
                                   depth_frame=None if args.color_sphere_trace else depth)
    torch.cuda.synchronize()
    print(f"nvblox integrated {len(poses)} frames in {time.time() - t0:.1f}s")

    t0 = time.time()
    mapper.update_color_mesh()
    torch.cuda.synchronize()
    print(f"mesh in {time.time() - t0:.1f}s")
    mesh = clean_mesh(mapper.get_color_mesh().to_open3d(), args.min_cluster)
    out_mesh = os.path.join(args.out, "mesh_nvblox.ply")
    o3d.io.write_triangle_mesh(out_mesh, mesh)
    print(f"mesh -> {out_mesh}")


if __name__ == "__main__":
    main()
