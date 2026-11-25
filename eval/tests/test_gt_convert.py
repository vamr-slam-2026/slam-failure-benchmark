import numpy as np
import pytest

from eval import gt_convert as gc
from eval.evaluate import load_tum


def test_euroc_ns_and_qw_first(tmp_path):
    seq = tmp_path / "MH_04_difficult" / "mav0"
    (seq / "state_groundtruth_estimate0").mkdir(parents=True)
    # 90 deg yaw: q = (w, x, y, z) = (cos45, 0, 0, sin45)
    c = np.cos(np.pi / 4)
    (seq / "state_groundtruth_estimate0" / "data.csv").write_text(
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v...\n"
        f"1403636580838555648,1.0,2.0,3.0,{c},0,0,{c},0,0,0\n"
        f"1403636580843555584,1.1,2.0,3.0,{c},0,0,{c},0,0,0\n")
    traj = gc.read_euroc(seq.parent)  # no sensor.yaml -> body frame kept
    assert traj.timestamps[0] == pytest.approx(1403636580.838555648)
    np.testing.assert_allclose(traj.positions_xyz[0], [1, 2, 3])
    np.testing.assert_allclose(traj.orientations_quat_wxyz[0], [c, 0, 0, c])
    out = tmp_path / "gt.txt"
    gc.write_tum(traj, out)
    first = out.read_text().split()
    np.testing.assert_allclose([float(v) for v in first[4:8]], [0, 0, c, c], atol=1e-9)  # qx qy qz qw
    back = load_tum(out)
    np.testing.assert_allclose(back.positions_xyz, traj.positions_xyz)


def test_euroc_body_to_cam(tmp_path):
    seq = tmp_path / "mav0"
    (seq / "state_groundtruth_estimate0").mkdir(parents=True)
    (seq / "cam0").mkdir()
    (seq / "state_groundtruth_estimate0" / "data.csv").write_text("1000000000,0,0,0,1,0,0,0\n")
    (seq / "cam0" / "sensor.yaml").write_text(
        "T_BS:\n  cols: 4\n  rows: 4\n  data: [1,0,0,0.1, 0,1,0,0.2, 0,0,1,0.3, 0,0,0,1]\n")
    traj = gc.read_euroc(tmp_path)
    np.testing.assert_allclose(traj.positions_xyz[0], [0.1, 0.2, 0.3])


def test_tartanair_ned_to_cam(tmp_path):
    # camera moving forward along NED x with identity attitude: in camera convention that's +z
    p = tmp_path / "pose_left.txt"
    p.write_text("0 0 0 0 0 0 1\n1 0 0 0 0 0 1\n")
    traj = gc.read_tartanair(p, fps=10)
    np.testing.assert_allclose(traj.timestamps, [0.0, 0.1])
    np.testing.assert_allclose(traj.positions_xyz[1], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(traj.poses_se3[1][:3, :3], np.eye(3), atol=1e-12)


def test_uzhfpv_with_id_column(tmp_path):
    p = tmp_path / "groundtruth.txt"
    p.write_text("# id timestamp tx ty tz qx qy qz qw\n"
                 "0 1540821844.6 1 2 3 0 0 0 1\n1 1540821844.7 1 2 4 0 0 0 1\n")
    traj = gc.read_uzhfpv(p)
    assert traj.num_poses == 2
    np.testing.assert_allclose(traj.positions_xyz[1], [1, 2, 4])


def test_lamaria_ns_comma(tmp_path):
    p = tmp_path / "pGT.txt"
    p.write_text("timestamp,tx,ty,tz,qx,qy,qz,qw\n2000000000000,1,0,0,0,0,0,1\n1000000000000,0,0,0,0,0,0,1\n")
    traj = gc.read_lamaria(p)
    np.testing.assert_allclose(traj.timestamps, [1000.0, 2000.0])  # sorted, ns -> s
    np.testing.assert_allclose(traj.positions_xyz[1], [1, 0, 0])


def test_tum_cli_roundtrip(tmp_path):
    src = tmp_path / "groundtruth.txt"
    src.write_text("# ground truth\n1305031098.6659 1.3 0.6 1.6 0.66 0.62 -0.29 -0.31\n")
    out = tmp_path / "out" / "gt_tum.txt"
    gc.main(["--format", "tum", "--input", str(src), "--output", str(out)])
    t = load_tum(out)
    assert t.num_poses == 1 and t.timestamps[0] == pytest.approx(1305031098.6659)
