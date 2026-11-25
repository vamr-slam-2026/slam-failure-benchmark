"""Run DPVO on a prepared sequence; map its frame indices back to dataset timestamps."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import read_rgb_list, write_tum  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpvo-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    prepared = Path(a.prepared).resolve()
    out = Path(a.out).resolve()
    os.chdir(a.dpvo_dir)
    sys.path.insert(0, str(Path(a.dpvo_dir).resolve()))
    import numpy as np
    from dpvo.config import cfg
    import demo  # systems/DPVO/demo.py

    cfg.merge_from_file(a.config)
    ts, _ = read_rgb_list(prepared)
    # demo.run streams sorted(rgb/*.png)[skip::stride]; prepared names are 000000.png.. in time order
    (poses, tstamps), _ = demo.run(cfg, a.weights, str(prepared / "rgb"), str(prepared / "calib.txt"),
                                   stride=a.stride, skip=0)
    idx = np.asarray(tstamps).astype(int) * a.stride
    write_tum(out, ts[idx], poses)  # poses: tx ty tz qx qy qz qw
    print(f"wrote {len(poses)} poses to {out}")


if __name__ == "__main__":
    main()
