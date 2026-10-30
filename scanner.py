"""main.py's live reconstruction with a chosen RealSense model, and after each session an STL of the final
mesh for SolidWorks.

Usage: python scanner.py --camera d435i -o output/part    (other options as main.py: python scanner.py -h)
Output in -o: everything main.py writes, and mesh_solidworks.stl.
"""
import argparse
import os

import numpy as np
import open3d as o3d

import main

STL_TRIANGLES = 300_000


def write_solidworks_stl(mesh, path):
    """mesh (Z up, m) -> binary STL decimated to STL_TRIANGLES, in mm with Y up (SolidWorks' default)."""
    m = mesh.simplify_quadric_decimation(STL_TRIANGLES)
    m.vertex_colors = o3d.utility.Vector3dVector()
    m.rotate(np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]), center=(0, 0, 0)).scale(1000.0, center=(0, 0, 0))
    m.compute_triangle_normals()
    o3d.io.write_triangle_mesh(path, m)


def after_session(out):
    ply = os.path.join(out, "mesh_live.ply")
    if os.path.exists(ply):
        print("STL for SolidWorks ...", flush=True)
        write_solidworks_stl(o3d.io.read_triangle_mesh(ply), os.path.join(out, "mesh_solidworks.stl"))
        print(f"-> {out}/mesh_solidworks.stl", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", choices=["d455", "d435i"], default="d435i", help="RealSense model to record from")
    main.main(p, after_session)
