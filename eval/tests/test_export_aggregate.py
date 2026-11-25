import csv

from evo.tools import file_interface

from eval import aggregate, export_frames
from eval import evaluate as ev
from eval.tests.synthetic import corrupt, make_gt

JUMP = 350


def _make_run(root, system, seq, gt, est=None, status="ok"):
    run = root / "results" / system / "euroc" / seq
    run.mkdir(parents=True)
    if est is not None:
        file_interface.write_tum_trajectory_file(str(run / "traj.txt"), est)
    (run / "status.txt").write_text(f"status={status}\nruntime_s=42\n")
    ev.main(["--est", str(run / "traj.txt"), "--gt", str(root / "gt.txt")])
    return run


def test_export_and_aggregate(tmp_path):
    gt = make_gt()
    file_interface.write_tum_trajectory_file(str(tmp_path / "gt.txt"), gt)
    prep = tmp_path / "prepared" / "euroc" / "MH_04_difficult"
    (prep / "rgb").mkdir(parents=True)
    lines = []
    for i, t in enumerate(gt.timestamps):
        (prep / "rgb" / f"{i:06d}.png").write_bytes(b"png%d" % i)
        lines.append(f"{t:.6f} rgb/{i:06d}.png")
    (prep / "rgb.txt").write_text("# timestamp filename\n" + "\n".join(lines) + "\n")

    run = _make_run(tmp_path, "droid", "MH_04_difficult", gt, corrupt(gt, jump_frame=JUMP))
    _make_run(tmp_path, "orbslam3", "MH_04_difficult", gt, None, status="failed")

    # --- export: exactly the two frames around the jump
    out = tmp_path / "labeling"
    made = export_frames.export_run(run / "results.json", prep / "rgb.txt", out)
    assert len(made) == 1
    imgs = sorted(p.name for p in made[0].glob("*.png"))
    assert imgs == [f"a_{JUMP - 1:06d}_{gt.timestamps[JUMP - 1]:.6f}.png", f"b_{JUMP:06d}_{gt.timestamps[JUMP]:.6f}.png"]
    assert (made[0] / imgs[1]).read_bytes() == b"png%d" % JUMP
    with open(out / "droid" / "euroc" / "MH_04_difficult" / "labels_todo.csv") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == export_frames.LABEL_COLUMNS
    assert float(rows[0]["t_end"]) == float(f"{gt.timestamps[JUMP]:.6f}")

    # --- export with context, via the results-root CLI
    out2 = tmp_path / "labeling2"
    export_frames.main(["--results-root", str(tmp_path / "results"), "--prepared-root", str(tmp_path / "prepared"),
                        "--out", str(out2), "--context", "2"])
    d = next((out2 / "droid" / "euroc" / "MH_04_difficult").glob("spike_00_*"))
    assert len(list(d.glob("*.png"))) == 6

    # --- aggregate
    aggregate.main(["--results-root", str(tmp_path / "results"), "--config", str(tmp_path / "none.yaml")])
    with open(tmp_path / "results" / "summary.csv") as f:
        rows = {r["system"]: r for r in csv.DictReader(f)}
    assert rows["droid"]["status"] == "ok" and float(rows["droid"]["raw.ate_rmse"]) > 0
    assert rows["droid"]["filtered.n_spikes"] == "1"
    assert rows["orbslam3"]["status"] == "no_trajectory" and rows["orbslam3"]["run_status"] == "failed"
    md = (tmp_path / "results" / "summary.md").read_text()
    assert "| droid |" in md and "| orbslam3 | fail |" in md and "MH_04_difficult" in md


def test_labels_template_columns():
    from pathlib import Path
    header = (Path(export_frames.__file__).parent / "labels_template.csv").read_text().strip()
    assert header.split(",") == export_frames.LABEL_COLUMNS
