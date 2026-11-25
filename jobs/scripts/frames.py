"""Shared helpers for the system runner wrappers (no heavy dependencies)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_rgb_list(prepared: Path) -> tuple[np.ndarray, list[Path]]:
    """prepared/rgb.txt -> (timestamps [s], absolute image paths), in file order."""
    ts, files = [], []
    for line in (Path(prepared) / "rgb.txt").read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        t, f = s.split()[:2]
        ts.append(float(t))
        files.append(Path(prepared) / f)
    return np.asarray(ts), files


def write_tum(path, timestamps, poses_txyz_qxyzw) -> None:
    """poses: (N, 7) tx ty tz qx qy qz qw (camera-to-world)."""
    poses = np.asarray(poses_txyz_qxyzw, dtype=float)
    assert len(timestamps) == len(poses), (len(timestamps), len(poses))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t, p in zip(timestamps, poses):
            f.write(f"{t:.9f} " + " ".join(f"{v:.9f}" for v in p) + "\n")
