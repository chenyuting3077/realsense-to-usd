"""TSDF mesh + 3DGS model -> one USD stage for Isaac Sim (Z-up, meters):
  /World/DomeLight
  /World/Scene      leveled by the largest horizontal plane of the mesh
      Mesh          vertex colors, hidden (the Gaussians are the visual layer)
      Collision     the mesh decimated to 300k triangles, hidden, static collider
      Gaussians     UsdVol.ParticleField3DGaussianSplat (OpenUSD 26.03), SH degree 3
  /PhysicsScene     gravity -Z; outside the default prim, so referencing the asset leaves it out

Usage: python -m gs.export_usd <mesh.ply> <3dgs point_cloud.ply> <poses.npy> <out.usdc>
"""
import sys

# pxr first: open3d bundles an older TBB, and whichever loads first wins the symbols
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, UsdVol, Vt  # isort: skip
import numpy as np
import open3d as o3d
from plyfile import PlyData

SH_C0 = 0.28209479177387814
COLLISION_TRIANGLES = 300_000


def level_transform(mesh, poses):
    """Rotation taking the pose frame to Z-up, from the largest plane facing the cameras' up."""
    cam_up = -np.asarray(poses)[:, :3, 1].mean(0)
    cam_up /= np.linalg.norm(cam_up)
    o3d.utility.random.seed(0)
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    up = cam_up
    for _ in range(6):
        _, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=2000)
        pts = np.asarray(pcd.points)[inliers]
        n = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][2]
        if abs(n @ cam_up) > np.cos(np.radians(30)):
            up = np.sign(n @ cam_up) * n
            break
        pcd = pcd.select_by_index(inliers, invert=True)
    x = np.array([1.0, 0, 0]) - up[0] * up
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(up, x), up])
    return T


def define_mesh(stage, path, mesh):
    v, f = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.triangles, np.int32)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr([Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())])
    m.MakeInvisible()
    return m


def add_mesh(stage, path, mesh):
    m = define_mesh(stage, path, mesh)
    mesh.compute_vertex_normals()
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, np.float32)))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex) \
        .Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_colors, np.float32)))
    # vertex colors through a preview material, so RTX renderers show them too
    mat = UsdShade.Material.Define(stage, path + "/VertexColor")
    surf = UsdShade.Shader.Define(stage, path + "/VertexColor/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    reader = UsdShade.Shader.Define(stage, path + "/VertexColor/DisplayColor")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("displayColor")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)


def add_collision(stage, path, mesh):
    if len(mesh.triangles) > COLLISION_TRIANGLES:
        mesh = mesh.simplify_quadric_decimation(COLLISION_TRIANGLES)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    prim = define_mesh(stage, path, mesh).GetPrim()  # PhysX collides with invisible prims too
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)


def add_gaussians(stage, path, ply):
    v = PlyData.read(ply)["vertex"]
    pos = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rot = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)  # w, x, y, z
    rot /= np.linalg.norm(rot, axis=1, keepdims=True)
    scale = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1)).astype(np.float32)
    opacity = (1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))).astype(np.float32)
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1).astype(np.float32)
    n_rest = sum(n.startswith("f_rest_") for n in v.data.dtype.names)
    rest = np.stack([v[f"f_rest_{i}"] for i in range(n_rest)], 1).astype(np.float32)
    rest = rest.reshape(len(pos), 3, n_rest // 3).transpose(0, 2, 1)  # PLY is channel-major
    sh = np.concatenate([dc[:, None], rest], 1)  # N x (deg+1)^2 x 3
    g = UsdVol.ParticleField3DGaussianSplat.Define(stage, path)
    g.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(pos))
    g.CreateOrientationsAttr(Vt.QuatfArray.FromNumpy(np.ascontiguousarray(rot[:, [1, 2, 3, 0]])))  # i, j, k, real
    g.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scale))
    g.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(opacity))
    g.CreateRadianceSphericalHarmonicsDegreeAttr(int(round(np.sqrt(sh.shape[1]))) - 1)
    g.CreateRadianceSphericalHarmonicsCoefficientsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(sh.reshape(-1, 3))))
    g.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(np.clip(0.5 + SH_C0 * dc, 0, 1)))
    r = 3 * scale.max(1, keepdims=True)
    g.CreateExtentAttr([Gf.Vec3f(*(pos - r).min(0).tolist()), Gf.Vec3f(*(pos + r).max(0).tolist())])
    return len(pos)


def main():
    mesh_path, gaussians, poses, out = sys.argv[1:5]
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr(1000.0)  # no default light in Isaac Sim
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    T = level_transform(mesh, np.load(poses))
    UsdGeom.Xform.Define(stage, "/World/Scene").AddTransformOp().Set(Gf.Matrix4d(T.T.tolist()))  # row-vector
    add_mesh(stage, "/World/Scene/Mesh", mesh)
    add_collision(stage, "/World/Scene/Collision", mesh)
    ps = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    n = add_gaussians(stage, "/World/Scene/Gaussians", gaussians)
    stage.GetRootLayer().Save()
    print(f"mesh {len(mesh.triangles)} triangles, {n} Gaussians -> {out}")


if __name__ == "__main__":
    main()
