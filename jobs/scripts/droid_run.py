"""Run DROID-SLAM (monocular) on a prepared sequence and write a TUM trajectory for every
processed frame. Mirrors systems/DROID-SLAM/demo.py, but streams frames from rgb.txt so the
output carries the dataset timestamps. Images are already undistorted (pinhole calib.txt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import read_rgb_list, write_tum  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--droid-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--buffer", type=int, default=1024)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.droid_dir) / "droid_slam"))
    import cv2
    import numpy as np
    import torch
    from droid import Droid

    prepared = Path(a.prepared)
    ts, files = read_rgb_list(prepared)
    ts, files = ts[::a.stride], files[::a.stride]
    fx, fy, cx, cy = np.loadtxt(prepared / "calib.txt")[:4]

    def stream():
        for t, f in enumerate(files):
            image = cv2.imread(str(f), cv2.IMREAD_COLOR)
            h0, w0 = image.shape[:2]
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1))[: h1 - h1 % 8, : w1 - w1 % 8]
            intr = torch.as_tensor([fx * w1 / w0, fy * h1 / h0, cx * w1 / w0, cy * h1 / h0])
            yield t, torch.as_tensor(image).permute(2, 0, 1)[None], intr

    # defaults from systems/DROID-SLAM/demo.py
    args = argparse.Namespace(
        weights=a.weights, buffer=a.buffer, image_size=None, disable_vis=True, stereo=False, upsample=False,
        beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0, frontend_window=25,
        frontend_radius=2, frontend_nms=1, backend_thresh=22.0, backend_radius=2, backend_nms=3,
        frontend_device="cuda", backend_device="cuda", asynchronous=False)
    torch.multiprocessing.set_start_method("spawn")

    droid = None
    for t, image, intr in stream():
        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        droid.track(t, image, intrinsics=intr)
    poses = droid.terminate(stream())  # (N, 7) tx ty tz qx qy qz qw, camera-to-world
    write_tum(a.out, ts[: len(poses)], poses)
    print(f"wrote {len(poses)} poses to {a.out}")


if __name__ == "__main__":
    main()
