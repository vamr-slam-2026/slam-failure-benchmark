"""Run MASt3R-SLAM on a prepared sequence.

Upstream main.py picks a dataset loader from the path name (tum/euroc/...) and otherwise
falls back to an image folder with fake 30 Hz timestamps. We swap in a loader that reads
prepared/rgb.txt, so the saved keyframe trajectory has the dataset's timestamps, then run
main.py unchanged with the pinhole calibration (prepared/mast3r_calib.yaml).

Output: keyframe poses only (that is what MASt3R-SLAM saves), TUM format.
"""
from __future__ import annotations

import argparse
import os
import runpy
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import read_rgb_list  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mast3r-dir", required=True)
    ap.add_argument("--config", default="config/calib.yaml")
    a = ap.parse_args()

    prepared = Path(a.prepared).resolve()
    out = Path(a.out).resolve()
    save_dir = out.parent / "mast3r_raw"
    os.chdir(a.mast3r_dir)                    # checkpoints/ and config/ are repo-relative
    sys.path.insert(0, str(Path(a.mast3r_dir).resolve()))

    import mast3r_slam.dataloader as dl

    class RGBListDataset(dl.MonocularDataset):
        def __init__(self, dataset_path):
            super().__init__()
            self.dataset_path = Path(dataset_path)
            ts, files = read_rgb_list(self.dataset_path)
            self.rgb_files = files
            self.timestamps = ts

    dl.load_dataset = lambda path: RGBListDataset(path)  # main.py imports it by name at run time

    sys.argv = ["main.py", "--dataset", str(prepared), "--config", a.config,
                "--calib", str(prepared / "mast3r_calib.yaml"), "--no-viz", "--save-as", str(save_dir)]
    runpy.run_path("main.py", run_name="__main__")

    produced = save_dir / f"{prepared.stem}.txt"   # eval.prepare_savedir: logs/<save_as>/<seq>.txt
    if not produced.exists():
        raise SystemExit(f"MASt3R-SLAM produced no trajectory at {produced}")
    shutil.copy(produced, out)
    print(f"copied {produced} -> {out}")


if __name__ == "__main__":
    main()
