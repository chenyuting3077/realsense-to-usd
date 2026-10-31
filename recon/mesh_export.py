"""The final mesh (mesh_live.ply: Z up, m, colored) exported for CAD: decimated, in mm with Y up (SolidWorks'
default). STEP holds one planar face per triangle, sharing edges: about 2.5 kB a face, hence fewer faces."""
import shutil

import numpy as np
import open3d as o3d

STL_TRIANGLES = 300_000
STEP_TRIANGLES = 20_000


def cad_mesh(mesh, triangles):
    m = mesh.simplify_quadric_decimation(triangles)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m.vertex_colors = o3d.utility.Vector3dVector()
    return m.rotate(np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]), center=(0, 0, 0)).scale(1000.0, center=(0, 0, 0))


def write_stl(mesh, path):
    m = cad_mesh(mesh, STL_TRIANGLES)
    m.compute_triangle_normals()
    if not o3d.io.write_triangle_mesh(path, m):
        raise OSError(f"cannot write {path}")


def write_step(mesh, path):
    try:
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                             BRepBuilderAPI_MakeVertex, BRepBuilderAPI_MakeWire)
        from OCC.Core.gp import gp_Pnt
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.Interface import Interface_Static
        from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
        from OCC.Core.TopoDS import TopoDS_Shell
    except ImportError:
        raise OSError("STEP needs pythonocc-core (docs/install)") from None
    m = cad_mesh(mesh, STEP_TRIANGLES)
    verts = [BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex() for p in np.asarray(m.vertices).tolist()]
    edges = {}

    def edge(a, b):
        k = (min(a, b), max(a, b))
        if k not in edges:
            edges[k] = BRepBuilderAPI_MakeEdge(verts[k[0]], verts[k[1]]).Edge()
        return edges[k] if a < b else edges[k].Reversed()

    shell, builder = TopoDS_Shell(), BRep_Builder()
    builder.MakeShell(shell)
    for a, b, c in np.asarray(m.triangles).tolist():
        face = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(edge(a, b), edge(b, c), edge(c, a)).Wire(), True)
        if face.IsDone():
            builder.Add(shell, face.Face())
    Interface_Static.SetCVal("write.step.unit", "MM")
    writer = STEPControl_Writer()
    writer.Transfer(shell, STEPControl_AsIs)
    if writer.Write(path) != IFSelect_RetDone:
        raise OSError(f"cannot write {path}")


def export(ply, path):
    """mesh_live.ply -> path by its extension: .ply as it is, .step / .stp, else STL (.stl added if missing).
    Returns the path written."""
    ext = path.lower().rsplit(".", 1)[-1] if "." in path.rsplit("/", 1)[-1] else ""
    if ext == "ply":
        shutil.copyfile(ply, path)
        return path
    if ext not in ("stl", "step", "stp"):
        path += ".stl"
    (write_step if ext in ("step", "stp") else write_stl)(o3d.io.read_triangle_mesh(ply), path)
    return path
