"""Copy the images around each RPE spike into a folder for manual failure labelling.

For every spike in results.json, the frames closest to t_start and t_end (plus optional
context frames before/after) are copied to
    <out>/<system>/<dataset>/<sequence>/spike_<NN>_t<t_end>/
and a pre-filled labels_todo.csv (columns of eval/labels_template.csv) is written per run.

Frame lists are TUM rgb.txt style ('timestamp relative/path.png'), paths relative to the list.

CLI (one run):
    python -m eval.export_frames --results results/droid/euroc/MH_04_difficult/results.json \
        --frames $DATA_ROOT/prepared/euroc/MH_04_difficult/rgb.txt --out labeling/
CLI (all runs):
    python -m eval.export_frames --results-root results/ --prepared-root $DATA_ROOT/prepared --out labeling/
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

LABEL_COLUMNS = ["system", "sequence", "t_start", "t_end", "failure_type", "cause", "notes"]


def read_frame_list(path) -> tuple[np.ndarray, list[Path]]:
    path = Path(path)
    ts, files = [], []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
        except ValueError:
            continue
        ts.append(t)
        files.append(path.parent / parts[1])
    ts = np.asarray(ts)
    if ts.size and np.max(np.abs(ts)) > 1e12:
        ts = ts * 1e-9
    order = np.argsort(ts, kind="stable")
    return ts[order], [files[i] for i in order]


def nearest(ts: np.ndarray, t: float) -> int:
    i = int(np.searchsorted(ts, t))
    if i == 0:
        return 0
    if i >= ts.size:
        return ts.size - 1
    return i if abs(ts[i] - t) < abs(ts[i - 1] - t) else i - 1


def export_run(results_json, frames, out_root, context: int = 0) -> list[Path]:
    res = json.loads(Path(results_json).read_text())
    system, dataset, sequence = res.get("system"), res.get("dataset"), res.get("sequence")
    run_dir = Path(out_root) / str(system) / str(dataset) / str(sequence)
    spikes = res.get("spikes") or []
    if not spikes:
        return []
    ts, files = read_frame_list(frames)
    if ts.size == 0:
        raise ValueError(f"empty frame list {frames}")
    run_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for n, sp in enumerate(spikes):
        i0, i1 = nearest(ts, sp["t_start"]), nearest(ts, sp["t_end"])
        # frames a (before) / b (after) the jump, plus `context` frames before a and after b
        idx = sorted(set(range(max(0, i0 - context), i0 + 1)) | set(range(i1, min(len(files), i1 + context + 1))))
        d = run_dir / f"spike_{n:02d}_t{sp['t_end']:.3f}"
        d.mkdir(exist_ok=True)
        for i in idx:
            tag = "a" if i == i0 else "b" if i == i1 else "pre" if i < i0 else "post"
            src = files[i]
            shutil.copy2(src, d / f"{tag}_{i:06d}_{ts[i]:.6f}{src.suffix}")
        (d / "spike.json").write_text(json.dumps(sp | {"frame_start": i0, "frame_end": i1}, indent=2))
        made.append(d)
    with open(run_dir / "labels_todo.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS)
        for sp in spikes:
            w.writerow([system, sequence, f"{sp['t_start']:.6f}", f"{sp['t_end']:.6f}", "", "", ""])
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="*", default=[], help="results.json file(s)")
    ap.add_argument("--frames", default=None, help="frame list for a single --results")
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--prepared-root", default=None, help="<root>/<dataset>/<sequence>/rgb.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=int, default=0, help="extra frames before/after each spike")
    a = ap.parse_args(argv)

    jobs = []
    if a.results:
        if a.frames is None and a.prepared_root is None:
            ap.error("--results needs --frames or --prepared-root")
        for r in a.results:
            jobs.append((Path(r), a.frames))
    if a.results_root:
        if a.prepared_root is None:
            ap.error("--results-root needs --prepared-root")
        jobs += [(p, None) for p in sorted(Path(a.results_root).glob("*/*/*/results.json"))]

    total = 0
    for rj, frames in jobs:
        if frames is None:
            res = json.loads(rj.read_text())
            frames = Path(a.prepared_root) / res["dataset"] / res["sequence"] / "rgb.txt"
        if not Path(frames).exists():
            print(f"skip {rj}: no frame list {frames}")
            continue
        made = export_run(rj, frames, a.out, a.context)
        total += len(made)
        print(f"{rj}: {len(made)} spikes exported")
    print(f"{total} spike folders under {a.out}")


if __name__ == "__main__":
    main()
