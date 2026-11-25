"""Smoke test for scripts/convert/prepare_sequence.py on tiny fake dataset trees."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from eval.evaluate import load_tum, read_frame_times

_spec = importlib.util.spec_from_file_location(
    "prepare_sequence", Path(__file__).resolve().parents[2] / "scripts" / "convert" / "prepare_sequence.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _img(path, w=64, h=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.random.default_rng(0).random((h, w)) * 255).astype(np.uint8))


def test_prepare_euroc(tmp_path):
    raw = tmp_path / "raw" / "euroc" / "MH_04_difficult" / "mav0"
    ts = [1403636580838555648 + i * 50_000_000 for i in range(3)]
    (raw / "cam0" / "data").mkdir(parents=True)
    (raw / "cam0" / "data.csv").write_text("#timestamp [ns],filename\n" + "".join(f"{t},{t}.png\n" for t in ts))
    for t in ts:
        _img(raw / "cam0" / "data" / f"{t}.png")
    (raw / "cam0" / "sensor.yaml").write_text(yaml.safe_dump({
        "T_BS": {"rows": 4, "cols": 4, "data": [1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]},
        "resolution": [64, 48], "camera_model": "pinhole", "intrinsics": [40.0, 40.0, 32.0, 24.0],
        "distortion_model": "radial-tangential", "distortion_coefficients": [-0.2, 0.05, 0.0, 0.0]}))
    (raw / "state_groundtruth_estimate0").mkdir()
    (raw / "state_groundtruth_estimate0" / "data.csv").write_text(
        "".join(f"{t},{i},0,0,1,0,0,0\n" for i, t in enumerate(ts)))
    entry = dict(dataset="euroc", sequence="MH_04_difficult", path="euroc/MH_04_difficult", format="euroc",
                 images="mav0/cam0/data.csv", gt="mav0/state_groundtruth_estimate0/data.csv",
                 calibration={"source": "sensor_yaml", "file": "mav0/cam0/sensor.yaml"}, T_body_cam="auto", fps=20)
    out = prep.prepare(entry, tmp_path)
    rgb_lines = (out / "rgb.txt").read_text().splitlines()
    assert all(l.startswith("#") for l in rgb_lines[:3]) and not rgb_lines[3].startswith("#")  # mono_tum skips 3
    assert "newWidth" not in (out / "orbslam3.yaml").read_text()
    times = read_frame_times(out / "rgb.txt")
    np.testing.assert_allclose(times, np.array(ts) * 1e-9)
    assert not (out / "rgb" / "000000.png").is_symlink()  # undistorted copy
    assert cv2.imread(str(out / "rgb" / "000002.png"), cv2.IMREAD_UNCHANGED).shape == (48, 64)
    assert len(np.loadtxt(out / "calib.txt")) == 4
    assert "Camera1.k1: 0.0" in (out / "orbslam3.yaml").read_text()
    gt = load_tum(out / "gt_tum.txt")
    np.testing.assert_allclose(gt.positions_xyz[1], [1.1, 0, 0])  # body->cam lever arm applied


def test_prepare_tartanair_and_fpv(tmp_path):
    ta = tmp_path / "raw" / "tartanair" / "mono" / "MH004"
    for i in range(4):
        _img(ta / f"{i:06d}.png")
    (tmp_path / "raw" / "tartanair" / "mono_gt").mkdir(parents=True)
    (tmp_path / "raw" / "tartanair" / "mono_gt" / "MH004.txt").write_text("0 0 0 0 0 0 1\n" * 4)
    e = dict(dataset="tartanair", sequence="MH004", path="tartanair/mono/MH004", format="tartanair",
             images="*.png", gt="../../mono_gt/MH004.txt", fps=10,
             calibration=dict(source="inline", model="pinhole", width=64, height=48, intrinsics=[32, 32, 32, 24]))
    out = prep.prepare(e, tmp_path)
    np.testing.assert_allclose(read_frame_times(out / "rgb.txt"), [0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(load_tum(out / "gt_tum.txt").timestamps, [0, 0.1, 0.2, 0.3])
    assert (out / "rgb" / "000000.png").is_symlink()

    fpv = tmp_path / "raw" / "uzhfpv" / "indoor_forward_9_snapdragon_with_gt"
    for i in range(3):
        _img(fpv / "img" / f"image_0_{i}.png")
    (fpv / "left_images.txt").write_text("# id timestamp image_name\n" +
                                        "".join(f"{i} {100 + i / 30:.6f} img/image_0_{i}.png\n" for i in range(3)))
    (fpv / "groundtruth.txt").write_text("# id timestamp tx ty tz qx qy qz qw\n" +
                                         "".join(f"{i} {100 + i / 30:.6f} {i} 0 0 0 0 0 1\n" for i in range(3)))
    calib_dir = tmp_path / "raw" / "uzhfpv" / "indoor_forward_calib_snapdragon"
    calib_dir.mkdir()
    (calib_dir / "camchain-imucam-x.yaml").write_text(yaml.safe_dump({"cam0": {
        "camera_model": "pinhole", "distortion_model": "equidistant", "resolution": [64, 48],
        "intrinsics": [30.0, 30.0, 32.0, 24.0], "distortion_coeffs": [-0.01, 0.01, 0.0, 0.0],
        "T_cam_imu": np.eye(4).tolist()}}))
    e = dict(dataset="uzhfpv", sequence="indoor_forward_9", path="uzhfpv/indoor_forward_9_snapdragon_with_gt",
             format="uzhfpv", images="left_images.txt", gt="groundtruth.txt", T_body_cam="auto", fps=30,
             calibration={"source": "kalibr", "file": "../indoor_forward_calib_snapdragon/camchain-imucam-*.yaml"})
    out = prep.prepare(e, tmp_path)
    assert yaml.safe_load((out / "meta.yaml").read_text())["calibration_in"]["model"] == "equidistant"
    assert load_tum(out / "gt_tum.txt").num_poses == 3


def test_config_entries_are_complete():
    entries = prep.load_config()
    assert len(entries) == 12
    for e in entries:
        for k in ("dataset", "sequence", "path", "format", "images", "gt", "calibration"):
            assert k in e, (e.get("sequence"), k)
    runs = (Path(prep.REPO) / "jobs" / "runs.csv").read_text().splitlines()[1:]
    seqs = {(e["dataset"], e["sequence"]) for e in entries}
    for line in runs:
        system, dataset, sequence = line.split(",")[:3]
        assert (dataset, sequence) in seqs, line
