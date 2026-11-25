"""Collect all results.json into one CSV (long format) and a markdown report
(one system x sequence table per metric).

CLI:
    python -m eval.aggregate --results-root results/ --csv results/summary.csv --md results/summary.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_MD_METRICS = [
    ("raw.ate_rmse", "ATE RMSE raw [m]"),
    ("filtered.ate_rmse", "ATE RMSE filtered [m]"),
    ("raw.rpe_trans_frame_rmse", "RPE trans / frame raw [m]"),
    ("filtered.rpe_trans_frame_rmse", "RPE trans / frame filtered [m]"),
    ("raw.rpe_trans_per_m_rmse", "RPE trans / m raw [m]"),
    ("coverage.frames", "Coverage (frames)"),
    ("coverage.time", "Coverage (time)"),
    ("filtered.n_spikes", "# RPE spikes"),
]

CSV_FIELDS = [
    "system", "dataset", "sequence", "status", "run_status", "runtime_s",
    "coverage.frames", "coverage.time", "coverage.n_est", "coverage.n_frames",
]
for _blk in ("raw", "filtered"):
    CSV_FIELDS += [f"{_blk}.{m}" for m in (
        "ate_rmse", "ate_mean", "ate_median", "ate_max",
        "rpe_trans_frame_rmse", "rpe_rot_frame_deg_rmse",
        "rpe_trans_per_m_rmse", "rpe_rot_per_m_deg_rmse", "sim3_scale", "n_matched")]
CSV_FIELDS += ["filtered.n_spikes", "filtered.n_dropped_lost"]


def get(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def load_all(root) -> list[dict]:
    rows = []
    for p in sorted(Path(root).glob("**/results.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"skip {p}: {e}")
            continue
        r["_path"] = str(p)
        rows.append(r)
    return rows


def flatten(r: dict) -> dict:
    out = {f: get(r, f) for f in CSV_FIELDS}
    run = r.get("run") or {}
    out["run_status"] = run.get("status")
    out["runtime_s"] = run.get("runtime_s")
    return out


def sequence_order(rows, config=None) -> list[str]:
    seqs = sorted({str(r.get("sequence")) for r in rows})
    if config and Path(config).exists():
        import yaml
        cfg = yaml.safe_load(Path(config).read_text()) or {}
        order = [s["sequence"] for s in cfg.get("sequences", [])]
        seqs = [s for s in order if s in seqs] + [s for s in seqs if s not in order]
    return seqs


def fmt(v, status) -> str:
    if v is None:
        return "fail" if status and status != "ok" else "–"
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def markdown(rows, metrics=DEFAULT_MD_METRICS, config=None) -> str:
    systems = sorted({str(r.get("system")) for r in rows})
    seqs = sequence_order(rows, config)
    by = {(str(r.get("system")), str(r.get("sequence"))): r for r in rows}
    out = ["# SLAM failure benchmark – summary", ""]
    for key, title in metrics:
        out += [f"## {title}", "", "| system | " + " | ".join(seqs) + " |",
                "|---|" + "---|" * len(seqs)]
        for s in systems:
            cells = []
            for q in seqs:
                r = by.get((s, q))
                if r is None:
                    cells.append("")
                    continue
                v = get(r, key)
                if key.startswith("coverage.") and isinstance(v, float):
                    cells.append(f"{v:.0%}")
                else:
                    cells.append(fmt(v, r.get("status")))
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--csv", default=None, help="default: <results-root>/summary.csv")
    ap.add_argument("--md", default=None, help="default: <results-root>/summary.md")
    ap.add_argument("--config", default="configs/sequences.yaml", help="for column order")
    a = ap.parse_args(argv)

    root = Path(a.results_root)
    rows = load_all(root)
    if not rows:
        raise SystemExit(f"no results.json under {root}")
    csv_path = Path(a.csv) if a.csv else root / "summary.csv"
    md_path = Path(a.md) if a.md else root / "summary.md"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (str(r.get("system")), str(r.get("dataset")), str(r.get("sequence")))):
            w.writerow(flatten(r))
    md_path.write_text(markdown(rows, config=a.config))
    print(f"{len(rows)} runs -> {csv_path}, {md_path}")


if __name__ == "__main__":
    main()
