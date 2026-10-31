"""main.py's live reconstruction as a scanner: the RealSense model chosen here or in the window, the session
ends with the final mesh, and Export in the window saves it where the user picks: .stl or .step for
SolidWorks (mm, Y up, decimated, recon/mesh_export.py) or .ply as built.

No rec/ unless --record: about 2.5 GB a minute.

Usage: python scanner.py --camera d435i -o output/part    (other options as main.py: python scanner.py -h)
"""
import argparse

import main
from recon.live_workers import MODELS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", choices=MODELS, default="d435i", help="RealSense model to record from")
    p.add_argument("--record", action="store_true", help="also save rec/ (every frameset, for replay)")
    p.set_defaults(scanner=True, record=False)
    main.main(p)
