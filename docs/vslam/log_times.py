"""RTAB-Map's loop closures (Link type 1: found by appearance) and proximity links (type 2: found near the
odometry's position) in rtabmap_<mode>.db, its per-update time
(rtab_<mode>.log: "RTAB-Map=0.0956s"), and OpenVINS' per-frame time (a log with verbosity INFO or DEBUG:
"[TIME]: 0.0067 seconds total"), from a run_session.sh work dir.

    python docs/vslam/log_times.py output/vslam/live_x [OpenVINS debug log]
"""
import glob
import os
import re
import sqlite3
import sys

import numpy as np


def stats(ms):
    return f"median {np.median(ms):6.1f} ms  p90 {np.percentile(ms, 90):6.1f} ms" if len(ms) else "-"


work = sys.argv[1]
for path in sorted(glob.glob(os.path.join(work, "rtab_*.log"))):
    mode = os.path.basename(path)[5:-4]
    ms = np.array([1000 * float(x) for x in re.findall(r"RTAB-Map=([0-9.]+)s", open(path, errors="replace").read())])
    db = sqlite3.connect(os.path.join(work, f"rtabmap_{mode}.db"))
    count = dict(db.execute("select type, count(*) from Link where from_id > to_id group by type"))  # stored both ways
    nodes = db.execute("select count(*) from Node").fetchone()[0]
    print(f"rtabmap {mode:<7} {nodes:4d} nodes  {count.get(1, 0):3d} loop closures + {count.get(2, 0):3d} proximity links"
          f"  per update {stats(ms)}")
for path in sys.argv[2:]:
    txt = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read())
    ms = np.array([1000 * float(x) for x in re.findall(r"\[TIME\]: ([0-9.]+) seconds total", txt)])
    print(f"openvins per frame {stats(ms)} ({len(ms)} frames)")
