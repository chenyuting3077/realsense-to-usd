"""main.py's live reconstruction as a scanner: the RealSense model chosen here or in the window, the session
ends with the final mesh, and Export in the window saves it where the user picks: .stl or .step for
SolidWorks (mm, Y up, decimated, recon/mesh_export.py) or .ply as built.

No rec/ unless --record: about 2.5 GB a minute. Scan close: FFS sees from 0.11 m (D435i, 192 px disparity),
an M4 hole needs voxels of 1.5 mm or less. The live map (nvblox) stays coarser: at 1 mm it took about
1 GB a second while moving and failed to allocate with 46 GB free. A final voxel of 0.5 mm took 39 s and
9.4M triangles for 870 frames; 0.3 mm ran out of memory.

Usage: python scanner.py --camera d435i -o output/part    (other options as main.py: python scanner.py -h)
"""
import argparse

import main
from recon.live_workers import MODELS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", choices=MODELS, default="d435i", help="RealSense model to record from")
    p.add_argument("--voxel", type=float, default=0.0005, help="final mesh voxel in m (main.py: 0.01)")
    p.add_argument("--preview-voxel", type=float, default=0.003, help="live map voxel in m (main.py: 0.01)")
    p.add_argument("--max-depth", type=float, default=0.3, help="depth kept, in m (main.py: 2.5, floor 4)")
    p.add_argument("--record", action="store_true", help="also save rec/ (every frameset, for replay)")
    p.set_defaults(scanner=True, record=False)
    main.main(p)
