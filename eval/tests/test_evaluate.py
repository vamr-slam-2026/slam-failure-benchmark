import copy
import json

import numpy as np
import pytest
from evo.tools import file_interface

from eval import evaluate as ev
from eval import timeline as tlmod
from eval.tests.synthetic import corrupt, make_gt

JUMP = 350


@pytest.fixture(scope="module")
def gt():
    return make_gt()


@pytest.fixture(scope="module")
def est(gt):
    return corrupt(gt, jump_frame=JUMP)


def test_jump_detected_at_right_frame(gt, est):
    ref_a, est_a, _ = ev.associate_and_align(gt, est)
    spikes = tlmod.detect_spikes(tlmod.rpe_timeline(ref_a, est_a), k=5.0)
    assert len(spikes) == 1, [s.to_dict() for s in spikes]
    s = spikes[0]
    assert s.idx == JUMP
    assert s.t_end == pytest.approx(gt.timestamps[JUMP])
    assert s.t_start == pytest.approx(gt.timestamps[JUMP - 1])


@pytest.mark.parametrize("jump", [60, 200, 540])
def test_jump_detected_at_various_frames(gt, jump):
    e = corrupt(gt, jump_frame=jump, seed=jump)
    res, _, spikes = ev.evaluate(e, gt)
    assert [s.idx for s in spikes] == [jump]
    assert res["spikes"][0]["t_end"] == pytest.approx(gt.timestamps[jump])


def test_no_false_spikes_without_jump(gt):
    e = corrupt(gt, jump_frame=None, seed=3)
    _, _, spikes = ev.evaluate(e, gt)
    assert spikes == []


def test_raw_vs_filtered_metrics(gt, est):
    res, _, _ = ev.evaluate(est, gt)
    raw, filt = res["raw"], res["filtered"]
    assert res["status"] == "ok"
    assert filt["n_spikes"] == 1
    # Sim(3) recovers the global scale (est is ~2.5x and drifts to 1.3x in scale)
    assert 1 / (2.5 * 1.3) < raw["sim3_scale"] < 1 / 2.5
    # jump dominates frame-to-frame RPE; removing it shrinks the RMSE a lot
    assert filt["rpe_trans_frame_rmse"] < raw["rpe_trans_frame_rmse"] / 2
    assert filt["rpe_trans_frame_max"] < raw["rpe_trans_frame_max"] / 5
    assert filt["n_matched"] == raw["n_matched"] - 1
    assert raw["ate_rmse"] > 0.05  # jump + scale drift give a clearly non-zero ATE
    assert raw["rpe_trans_per_m_n"] > 10


def test_perfect_estimate_has_zero_error(gt):
    e = corrupt(gt, jump_frame=None, scale_end=1.0, drift_sigma=0, noise_pos=0, noise_rot_deg=0)
    res, _, _ = ev.evaluate(e, gt, spike_min_trans=1e-6, spike_min_rot_deg=1e-6)
    assert res["raw"]["ate_rmse"] < 1e-9
    assert res["raw"]["sim3_scale"] == pytest.approx(1 / 2.5)


def test_lost_filter_and_coverage(gt, est):
    keep = np.r_[0:100, 160:gt.num_poses]  # 60 frames (3 s) without pose
    e = copy.deepcopy(est)
    e.reduce_to_ids(keep)
    lost = e.timestamps[[10, 20, 30, 40, 50]]
    res, _, _ = ev.evaluate(e, gt, frame_times=gt.timestamps, lost_times=lost, max_gap=0.5)
    assert res["filtered"]["n_dropped_lost"] == 5
    assert res["raw"]["n_matched"] == len(keep)
    assert res["coverage"]["frames"] == pytest.approx(len(keep) / gt.num_poses)
    dur = gt.timestamps[-1] - gt.timestamps[0]
    assert res["coverage"]["time"] == pytest.approx((dur - (160 - 99) / 20.0) / dur, abs=1e-6)


def test_empty_and_unmatched(gt):
    res, _, _ = ev.evaluate(None, gt)
    assert res["status"] == "no_trajectory" and res["raw"] is None
    shifted = copy.deepcopy(gt)
    shifted.timestamps = shifted.timestamps + 1e4
    res, _, _ = ev.evaluate(shifted, gt)
    assert res["status"] == "association_failed"


def test_cli_writes_results_json(tmp_path, gt, est):
    run = tmp_path / "results" / "droid" / "euroc" / "MH_04_difficult"
    run.mkdir(parents=True)
    file_interface.write_tum_trajectory_file(str(run / "traj.txt"), est)
    file_interface.write_tum_trajectory_file(str(tmp_path / "gt.txt"), gt)
    (run / "status.txt").write_text("status=ok\nexit_code=0\nruntime_s=12\n")
    frames = tmp_path / "rgb.txt"
    frames.write_text("".join(f"{t:.6f} rgb/{i:06d}.png\n" for i, t in enumerate(gt.timestamps)))
    ev.main(["--est", str(run / "traj.txt"), "--gt", str(tmp_path / "gt.txt"), "--frames", str(frames), "--plot"])
    res = json.loads((run / "results.json").read_text())
    assert (res["system"], res["dataset"], res["sequence"]) == ("droid", "euroc", "MH_04_difficult")
    assert res["run"]["status"] == "ok"
    assert res["coverage"]["frames"] == pytest.approx(1.0)
    assert [s["t_end"] for s in res["spikes"]] == [pytest.approx(gt.timestamps[JUMP], abs=1e-5)]
    assert (run / "timeline.csv").exists() and (run / "timeline.png").exists()
    lines = (run / "timeline.csv").read_text().splitlines()
    assert len(lines) == gt.num_poses  # header + n-1 pairs
    assert sum(int(l.split(",")[-1]) for l in lines[1:]) == 1
