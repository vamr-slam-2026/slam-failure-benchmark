"""Per-frame-pair RPE over time, spike detection and plotting.

A spike is a consecutive pose pair whose translational (or rotational) RPE exceeds
k x median of all pairs (and an optional absolute floor). Spikes mark tracking jumps,
relocalisation glitches and map merges, which are the candidates for manual labelling.

CLI:
    python -m eval.timeline --est traj.txt --gt gt_tum.txt --out-dir <dir> [--k 5]
writes <dir>/timeline.csv and <dir>/timeline.png.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from evo.core import metrics
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D


@dataclass
class Timeline:
    """RPE of each consecutive pair (i, i+1) of an associated, aligned trajectory pair."""
    t_start: np.ndarray      # timestamp of pose i
    t_end: np.ndarray        # timestamp of pose i+1
    idx_end: np.ndarray      # index i+1 in the associated trajectory
    trans: np.ndarray        # [m]
    rot_deg: np.ndarray      # [deg]


@dataclass
class Spike:
    idx: int                 # index (in the associated trajectory) of the pose after the jump
    t_start: float
    t_end: float
    trans_err: float
    rot_err_deg: float
    trans_ratio: float       # trans_err / median
    rot_ratio: float

    def to_dict(self):
        return {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in asdict(self).items()}


def rpe_timeline(traj_ref: PoseTrajectory3D, traj_est: PoseTrajectory3D) -> Timeline:
    """Consecutive-pair RPE (delta = 1 frame) using evo. Inputs must be associated (and,
    for meaningful translation errors, Sim(3)-aligned)."""
    out = {}
    for name, rel in (("trans", PoseRelation.translation_part), ("rot", PoseRelation.rotation_angle_deg)):
        rpe = metrics.RPE(rel, delta=1, delta_unit=Unit.frames, all_pairs=False)
        rpe.process_data((traj_ref, traj_est))
        out[name] = rpe.error
        out["ids"] = np.asarray(rpe.delta_ids, dtype=int)
    j = out["ids"]
    ts = traj_ref.timestamps
    return Timeline(t_start=ts[j - 1], t_end=ts[j], idx_end=j, trans=out["trans"], rot_deg=out["rot"])


def spike_thresholds(tl: Timeline, k: float, min_trans: float = 0.0, min_rot_deg: float = 0.0):
    med_t = float(np.median(tl.trans)) if len(tl.trans) else 0.0
    med_r = float(np.median(tl.rot_deg)) if len(tl.rot_deg) else 0.0
    return max(k * med_t, min_trans), max(k * med_r, min_rot_deg), med_t, med_r


def detect_spikes(tl: Timeline, k: float = 5.0, min_trans: float = 0.0, min_rot_deg: float = 0.0,
                  use_rotation: bool = True) -> list[Spike]:
    """Pairs whose RPE exceeds max(k * median, floor). Rotation is tested too unless disabled."""
    thr_t, thr_r, med_t, med_r = spike_thresholds(tl, k, min_trans, min_rot_deg)
    mask = tl.trans > thr_t
    if use_rotation:
        mask |= tl.rot_deg > thr_r
    eps = 1e-12
    return [
        Spike(idx=int(tl.idx_end[i]), t_start=float(tl.t_start[i]), t_end=float(tl.t_end[i]),
              trans_err=float(tl.trans[i]), rot_err_deg=float(tl.rot_deg[i]),
              trans_ratio=float(tl.trans[i] / max(med_t, eps)), rot_ratio=float(tl.rot_deg[i] / max(med_r, eps)))
        for i in np.flatnonzero(mask)
    ]


def write_csv(tl: Timeline, spikes: list[Spike], path) -> None:
    spike_idx = {s.idx for s in spikes}
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "idx", "rpe_trans_m", "rpe_rot_deg", "spike"])
        for i in range(len(tl.trans)):
            w.writerow([f"{tl.t_start[i]:.6f}", f"{tl.t_end[i]:.6f}", int(tl.idx_end[i]),
                        f"{tl.trans[i]:.6g}", f"{tl.rot_deg[i]:.6g}", int(tl.idx_end[i] in spike_idx)])


def plot(tl: Timeline, spikes: list[Spike], path, k: float, min_trans=0.0, min_rot_deg=0.0, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thr_t, thr_r, _, _ = spike_thresholds(tl, k, min_trans, min_rot_deg)
    t0 = tl.t_start[0] if len(tl.t_start) else 0.0
    t = tl.t_end - t0
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for ax, y, thr, lab in ((axes[0], tl.trans, thr_t, "RPE trans [m]"), (axes[1], tl.rot_deg, thr_r, "RPE rot [deg]")):
        ax.plot(t, y, lw=0.8, color="#3a6ea5")
        ax.axhline(thr, ls="--", lw=0.8, color="#888888", label=f"threshold ({k:g}x median)")
        for s in spikes:
            ax.axvline(s.t_end - t0, color="#c0392b", lw=0.8, alpha=0.6)
        ax.set_ylabel(lab)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time since first pose [s]")
    axes[0].set_title(title or f"Per-pair RPE ({len(spikes)} spikes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    from eval.evaluate import load_tum, associate_and_align  # local import: avoid cycle

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", type=float, default=5.0)
    ap.add_argument("--min-trans", type=float, default=0.0)
    ap.add_argument("--min-rot-deg", type=float, default=0.0)
    ap.add_argument("--max-diff", type=float, default=0.02)
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)

    ref, est, _ = associate_and_align(load_tum(a.gt), load_tum(a.est), a.max_diff)
    tl = rpe_timeline(ref, est)
    spikes = detect_spikes(tl, a.k, a.min_trans, a.min_rot_deg)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(tl, spikes, out / "timeline.csv")
    plot(tl, spikes, out / "timeline.png", a.k, a.min_trans, a.min_rot_deg, a.title)
    print(f"{len(spikes)} spikes -> {out / 'timeline.csv'}, {out / 'timeline.png'}")


if __name__ == "__main__":
    main()
