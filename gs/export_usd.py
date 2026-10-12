"""Write the TSDF mesh and the 3D Gaussian Splatting model of one recording into a USD stage.

Both already live in the same world frame: TSDF integration and 3DGS training use the same
SLAM poses (world = first color camera: x right, y down, z forward). The stage is Z-up,
meters; a root transform levels the scene, from the dominant horizontal plane of the mesh
(desk or floor), which keeps the data itself in the pose frame.

  /World/DomeLight  uniform white light; Isaac Sim / Omniverse have no default light, a stage
                    without one renders black
  /World/Scene (xform: pose frame -> Z-up)
      Mesh      UsdGeom.Mesh, vertex colors (displayColor, bound to a UsdPreviewSurface)
      Gaussians UsdVol.ParticleField3DGaussianSplat (OpenUSD 26.03+): positions,
                orientations, linear scales, opacities, degree-3 SH coefficients
      Collision the TSDF mesh decimated to --collision-triangles, invisible, with
                UsdPhysics.CollisionAPI + MeshCollisionAPI (approximation none: the triangles
                themselves, a static collider); the visual Mesh has no physics
/PhysicsScene      gravity -Z, 9.81 m/s^2, for opening the file directly in Isaac Sim. It sits
                   outside the default prim /World, so referencing the asset (Isaac Lab spawns
                   the default prim) does not bring a second physics scene along.

The Gaussians are the visual layer: the mesh is invisible unless --show-mesh (the collider
is always invisible).

Usage (in the activated conda env):
  python gs/export_usd.py output/room/recon/mesh.ply output/room/fastgs/point_cloud.ply \
      --poses output/room/frames/poses.npy -o output/room/scene.usdc
"""
import argparse

# pxr first: open3d bundles an older TBB, and whichever loads first wins the symbols.
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdVol, Vt  # isort: skip
import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def level_transform(mesh, poses):
    """Rotation (4x4) taking the pose frame to Z-up, from the largest plane facing the cameras' up."""
    cam_up = -np.asarray(poses)[:, :3, 1].mean(0)  # camera -y in world
    cam_up /= np.linalg.norm(cam_up)
    o3d.utility.random.seed(0)  # RANSAC: same leveling on every export
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    best = None
    for _ in range(6):
        plane, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=2000)
        # least-squares refit on the inliers (normal = smallest principal axis)
        pts = np.asarray(pcd.points)[inliers]
        n = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][2]
        if abs(n @ cam_up) > np.cos(np.radians(30)):
            best = (np.sign(n @ cam_up) * n, len(inliers))
            break
        pcd = pcd.select_by_index(inliers, invert=True)
    if best is None:
        print("no horizontal plane found, leveling with the mean camera up")
        up = cam_up
    else:
        up = best[0]
        print(f"horizontal plane: {best[1]} vertices, {np.degrees(np.arccos(up @ cam_up)):.1f} deg from the camera up")
    # world x stays roughly x; z becomes up; y completes a right-handed frame
    x = np.array([1.0, 0, 0]) - up[0] * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, up])  # rows: new axes expressed in the pose frame
    return T


def add_mesh(stage, path, mesh):
    m = UsdGeom.Mesh.Define(stage, path)
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.triangles, np.int32)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr([Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())])
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    color = UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray,
                                                  UsdGeom.Tokens.vertex)
    color.Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_colors, np.float32)))

    # Vertex colors through a preview material, so RTX renderers show them too.
    mat = UsdShade.Material.Define(stage, path + "/VertexColor")
    surf = UsdShade.Shader.Define(stage, path + "/VertexColor/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    reader = UsdShade.Shader.Define(stage, path + "/VertexColor/DisplayColor")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        reader.ConnectableAPI(), "result")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return len(v), len(f)


def add_collision(stage, path, mesh, n_triangles):
    """Invisible static triangle-mesh collider, decimated from the visual mesh."""
    if len(mesh.triangles) > n_triangles:
        mesh = mesh.simplify_quadric_decimation(n_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.triangles, np.int32)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr([Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())])
    m.MakeInvisible()  # PhysX collides with invisible prims too
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    return len(f)


def add_gaussians(stage, path, ply):
    v = PlyData.read(ply)["vertex"]
    names = v.data.dtype.names
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)  # w, x, y, z
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    scale = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    opacity = (1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))).astype(np.float32)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    n_rest = sum(n.startswith("f_rest_") for n in names)
    rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], 1).astype(np.float32)
    rest = rest.reshape(len(pos), 3, n_rest // 3).transpose(0, 2, 1)  # PLY is channel-major
    sh = np.concatenate([dc[:, None], rest], 1)  # N x (deg+1)^2 x 3
    degree = int(round(np.sqrt(sh.shape[1]))) - 1

    g = UsdVol.ParticleField3DGaussianSplat.Define(stage, path)
    g.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(pos))
    g.CreateOrientationsAttr(Vt.QuatfArray.FromNumpy(np.ascontiguousarray(rot[:, [1, 2, 3, 0]])))  # Gf: i, j, k, real
    g.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scale))
    g.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(opacity))
    g.CreateRadianceSphericalHarmonicsDegreeAttr(degree)
    g.CreateRadianceSphericalHarmonicsCoefficientsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(sh.reshape(-1, 3))))
    g.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(np.clip(0.5 + SH_C0 * dc, 0, 1)))
    lo, hi = (pos - 3 * scale.max(1, keepdims=True)).min(0), (pos + 3 * scale.max(1, keepdims=True)).max(0)
    g.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    return len(pos), degree, rot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mesh", help="TSDF mesh (recon/reconstruct.py mesh.ply)")
    p.add_argument("gaussians", help="3DGS PLY (FastGS point_cloud.ply)")
    p.add_argument("-o", "--out", required=True, help=".usdc (binary) or .usda")
    p.add_argument("--poses", help="Nx4x4 camera-to-world poses, to level the scene (default: none, keep the pose frame)")
    p.add_argument("--collision-triangles", type=int, default=300_000,
                   help="decimate the collider to this many triangles (0: no collider)")
    p.add_argument("--show-mesh", action="store_true", help="render the mesh too (default: only the Gaussians)")
    p.add_argument("--no-physics-scene", dest="physics_scene", action="store_false",
                   help="leave out /PhysicsScene")
    args = p.parse_args()

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)
    scene = UsdGeom.Xform.Define(stage, "/World/Scene")

    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if args.poses:
        T = level_transform(mesh, np.load(args.poses))
    else:
        # pose frame (y down, z forward) -> Z-up with the first camera level
        T = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    scene.AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))  # USD matrices are row-vector

    nv, nf = add_mesh(stage, "/World/Scene/Mesh", mesh)
    if not args.show_mesh:
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/Scene/Mesh")).MakeInvisible()
    nc = add_collision(stage, "/World/Scene/Collision", mesh, args.collision_triangles) if args.collision_triangles else 0
    if args.physics_scene:
        ps = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
        ps.CreateGravityMagnitudeAttr(9.81)
    ng, degree, rot = add_gaussians(stage, "/World/Scene/Gaussians", args.gaussians)
    stage.GetRootLayer().Save()

    # Round-trip check of the quaternion layout.
    q = UsdVol.ParticleField3DGaussianSplat(stage.GetPrimAtPath("/World/Scene/Gaussians")).GetOrientationsAttr().Get()[0]
    assert np.allclose([q.GetReal(), *q.GetImaginary()], rot[0], atol=1e-6)
    print(f"mesh {nv} vertices / {nf} triangles{' (hidden)' * (not args.show_mesh)}, collider {nc} triangles, "
          f"{ng} Gaussians (SH degree {degree}) -> {args.out}")


if __name__ == "__main__":
    main()
