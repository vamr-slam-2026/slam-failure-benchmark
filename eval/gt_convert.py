"""Convert dataset ground truth to TUM format (timestamp[s] tx ty tz qx qy qz qw).

Supported formats:
    euroc      mav0/state_groundtruth_estimate0/data.csv  (ns timestamps, qw first, body=IMU frame)
    tum, bonn  groundtruth.txt, already TUM (copied / re-written)
    tartanair  pose_left.txt or <SEQ>.txt (no timestamps, NED, "tx ty tz qx qy qz qw" per frame)
    uzhfpv     groundtruth.txt ("# id timestamp tx ty tz qx qy qz qw", body=IMU frame)
    lamaria    pGT txt (whitespace/comma separated, timestamp in s or ns, TUM column order)

Body-frame GT (EuRoC, UZH-FPV) can be moved to the camera frame with a 4x4 T_body_cam
(T_W_C = T_W_B @ T_B_C). For EuRoC it is read from mav0/cam0/sensor.yaml (T_BS) when available.

CLI:
    python -m eval.gt_convert --format euroc --input <seq_dir or csv> --output gt_tum.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

# TartanAir: NED (x fwd, y right, z down) -> camera (x right, y down, z fwd).
# Same convention as tartanair_tools/evaluation/transformation.py::ned2cam.
_NED_TO_CAM = np.array(
    [[0, 1, 0, 0],
     [0, 0, 1, 0],
     [1, 0, 0, 0],
     [0, 0, 0, 1]], dtype=float)


# --------------------------------------------------------------------------- helpers
def _load_numeric(path: Path) -> np.ndarray:
    """Load a numeric table, skipping '#' comments and non-numeric header lines.

    Accepts comma or whitespace separated values.
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.replace(",", " ").split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue  # header line such as "timestamp tx ty ..."
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    ncols = {len(r) for r in rows}
    if len(ncols) != 1:
        raise ValueError(f"inconsistent column count {sorted(ncols)} in {path}")
    return np.asarray(rows, dtype=float)


def _to_seconds(t: np.ndarray) -> np.ndarray:
    """Heuristic: timestamps > 1e12 are nanoseconds (1e12 s ~ year 33658)."""
    t = np.asarray(t, dtype=float)
    return t * 1e-9 if np.nanmax(np.abs(t)) > 1e12 else t


def _traj(t, xyz, q_wxyz) -> PoseTrajectory3D:
    q = np.asarray(q_wxyz, dtype=float)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    order = np.argsort(t, kind="stable")
    return PoseTrajectory3D(
        positions_xyz=np.asarray(xyz, dtype=float)[order],
        orientations_quat_wxyz=q[order],
        timestamps=np.asarray(t, dtype=float)[order],
    )


def apply_body_to_cam(traj: PoseTrajectory3D, T_body_cam: np.ndarray | None) -> PoseTrajectory3D:
    """T_W_C = T_W_B @ T_B_C (in place, returns traj)."""
    if T_body_cam is not None:
        traj.transform(np.asarray(T_body_cam, dtype=float), right_mul=True)
    return traj


def load_matrix(path_or_none) -> np.ndarray | None:
    """Load a 4x4 matrix from a .txt (16 numbers) or from a yaml file.

    YAML: EuRoC sensor.yaml (T_BS: {data: [...]}) or Kalibr camchain-imu (cam0.T_cam_imu,
    which is inverted to get T_imu_cam).
    """
    if path_or_none is None:
        return None
    p = Path(path_or_none)
    if p.suffix in (".yaml", ".yml"):
        d = yaml.safe_load(p.read_text())
        if "T_BS" in d:
            return np.asarray(d["T_BS"]["data"], dtype=float).reshape(4, 4)
        if "cam0" in d and "T_cam_imu" in d["cam0"]:
            return np.linalg.inv(np.asarray(d["cam0"]["T_cam_imu"], dtype=float))
        if "T_body_cam" in d:
            return np.asarray(d["T_body_cam"], dtype=float).reshape(4, 4)
        raise ValueError(f"no known extrinsic key in {p}")
    return np.loadtxt(p, dtype=float).reshape(4, 4)


# --------------------------------------------------------------------------- readers
def read_euroc(path, T_body_cam=None) -> PoseTrajectory3D:
    """EuRoC ground truth. `path` = sequence dir (containing mav0/) or the data.csv itself."""
    path = Path(path)
    if path.is_dir():
        mav = path / "mav0" if (path / "mav0").is_dir() else path
        csv = mav / "state_groundtruth_estimate0" / "data.csv"
        sensor = mav / "cam0" / "sensor.yaml"
        if T_body_cam is None and sensor.exists():
            T_body_cam = load_matrix(sensor)
    else:
        csv = path
    d = _load_numeric(csv)
    # #timestamp[ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z, v..., b_w..., b_a...
    traj = _traj(d[:, 0] * 1e-9, d[:, 1:4], d[:, 4:8])
    return apply_body_to_cam(traj, T_body_cam)


def read_tum(path, T_body_cam=None) -> PoseTrajectory3D:
    """TUM RGB-D / Bonn: already 'timestamp tx ty tz qx qy qz qw'."""
    d = _load_numeric(path)
    traj = _traj(d[:, 0], d[:, 1:4], d[:, [7, 4, 5, 6]])
    return apply_body_to_cam(traj, T_body_cam)


def read_tartanair(path, fps: float = 10.0, t0: float = 0.0, ned_to_cam: bool = True) -> PoseTrajectory3D:
    """TartanAir pose file: one 'tx ty tz qx qy qz qw' per image, no timestamps, NED.

    Timestamps are synthesized as t0 + i / fps; scripts/convert/prepare_sequence.py uses the
    same rule for the image list, so estimates and GT share a time base.
    """
    d = _load_numeric(path)
    if d.shape[1] != 7:
        raise ValueError(f"TartanAir pose file must have 7 columns, got {d.shape[1]}")
    t = t0 + np.arange(len(d)) / fps
    traj = _traj(t, d[:, 0:3], d[:, [6, 3, 4, 5]])
    if ned_to_cam:
        # T_cam = M @ T_ned @ M^-1 : change of world basis and of body basis.
        M = _NED_TO_CAM
        poses = [M @ T @ np.linalg.inv(M) for T in traj.poses_se3]
        traj = PoseTrajectory3D(poses_se3=poses, timestamps=traj.timestamps)
    return traj


def read_uzhfpv(path, T_body_cam=None) -> PoseTrajectory3D:
    """UZH-FPV groundtruth.txt: '# id timestamp tx ty tz qx qy qz qw' (id column optional)."""
    d = _load_numeric(path)
    if d.shape[1] == 9:
        d = d[:, 1:]
    elif d.shape[1] != 8:
        raise ValueError(f"unexpected UZH-FPV column count {d.shape[1]}")
    traj = _traj(_to_seconds(d[:, 0]), d[:, 1:4], d[:, [7, 4, 5, 6]])
    return apply_body_to_cam(traj, T_body_cam)


def read_lamaria(path, qw_first: bool = False, T_body_cam=None) -> PoseTrajectory3D:
    """LaMAria pseudo-GT txt. Assumes 8 columns 'timestamp tx ty tz q..' (TUM order unless
    qw_first). Timestamps in ns are detected and converted.

    NOTE: verify the column order against the LaMAria README for the downloaded release.
    """
    d = _load_numeric(path)
    if d.shape[1] < 8:
        raise ValueError(f"LaMAria pGT needs >= 8 columns, got {d.shape[1]}")
    d = d[:, :8]
    q = d[:, 4:8] if qw_first else d[:, [7, 4, 5, 6]]
    traj = _traj(_to_seconds(d[:, 0]), d[:, 1:4], q)
    return apply_body_to_cam(traj, T_body_cam)


FORMATS = ("euroc", "tum", "bonn", "tartanair", "uzhfpv", "lamaria")


def convert(fmt: str, input_path, T_body_cam=None, fps: float = 10.0, qw_first: bool = False) -> PoseTrajectory3D:
    fmt = fmt.lower()
    if fmt == "euroc":
        return read_euroc(input_path, T_body_cam)
    if fmt in ("tum", "bonn"):
        return read_tum(input_path, T_body_cam)
    if fmt == "tartanair":
        return read_tartanair(input_path, fps=fps)
    if fmt == "uzhfpv":
        return read_uzhfpv(input_path, T_body_cam)
    if fmt == "lamaria":
        return read_lamaria(input_path, qw_first=qw_first, T_body_cam=T_body_cam)
    raise ValueError(f"unknown format {fmt!r}, expected one of {FORMATS}")


def write_tum(traj: PoseTrajectory3D, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    file_interface.write_tum_trajectory_file(str(path), traj)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=FORMATS)
    ap.add_argument("--input", required=True, help="GT file (or EuRoC sequence dir)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--T-body-cam", default=None,
                    help="4x4 txt, EuRoC sensor.yaml or Kalibr camchain-imu yaml")
    ap.add_argument("--fps", type=float, default=10.0, help="TartanAir frame rate")
    ap.add_argument("--qw-first", action="store_true", help="LaMAria: quaternion stored as qw qx qy qz")
    a = ap.parse_args(argv)
    traj = convert(a.format, a.input, load_matrix(a.T_body_cam), a.fps, a.qw_first)
    write_tum(traj, a.output)
    print(f"wrote {traj.num_poses} poses to {a.output}")


if __name__ == "__main__":
    main()
