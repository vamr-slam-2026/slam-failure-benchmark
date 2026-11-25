"""Evaluate one estimated trajectory against ground truth (both TUM format).

Pipeline (evo used as a library):
  1. timestamp association (nearest, max_diff; optional interpolation for sparse GT)
  2. Sim(3) Umeyama alignment (monocular -> scale is unobservable)
  3. ATE RMSE (+ mean/median/max), RPE over consecutive frames and per meter
  4. coverage: share of input frames with a pose, and share of sequence time covered
Metrics are reported twice:
  raw       all estimated poses
  filtered  poses flagged lost are dropped, RPE spikes (> k x median of consecutive-pair
            translational/rotational RPE) are detected; spike poses are dropped before
            re-alignment / ATE, and RPE pairs spanning a spike are excluded.

Writes results.json (and timeline.csv next to it).

CLI:
    python -m eval.evaluate --est results/droid/euroc/MH_04_difficult/traj.txt \
        --gt data/prepared/euroc/MH_04_difficult/gt_tum.txt \
        --frames data/prepared/euroc/MH_04_difficult/rgb.txt
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from evo.core import filters, sync
from evo.core import lie_algebra as lie
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from eval import timeline as tlmod

MIN_MATCHES = 3


# --------------------------------------------------------------------------- I/O
def load_tum(path) -> PoseTrajectory3D | None:
    """Read a TUM file; drops duplicate timestamps and non-finite rows, sorts by time.
    Returns None if the file has no valid pose (evo cannot represent empty trajectories)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = s.replace(",", " ").split()
        if len(vals) < 8:
            continue
        try:
            rows.append([float(v) for v in vals[:8]])
        except ValueError:
            continue
    if not rows:
        return None
    d = np.asarray(rows)
    d = d[np.all(np.isfinite(d), axis=1)]
    q = d[:, [7, 4, 5, 6]]
    n = np.linalg.norm(q, axis=1)
    d, q, n = d[n > 1e-9], q[n > 1e-9], n[n > 1e-9]
    _, first = np.unique(d[:, 0], return_index=True)  # sorted unique timestamps
    d, q, n = d[first], q[first], n[first]
    if len(d) == 0:
        return None
    return PoseTrajectory3D(positions_xyz=d[:, 1:4], orientations_quat_wxyz=q / n[:, None], timestamps=d[:, 0])


def read_frame_times(path) -> np.ndarray:
    """Frame list 'timestamp [path]' (TUM rgb.txt style) -> timestamps [s]."""
    ts = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            ts.append(float(s.replace(",", " ").split()[0]))
        except ValueError:
            continue
    ts = np.asarray(ts, dtype=float)
    return ts * 1e-9 if ts.size and np.max(np.abs(ts)) > 1e12 else ts


def read_lost(path) -> np.ndarray:
    """Lost-flag file: 'timestamp' (lost) or 'timestamp flag' (flag != 0 -> lost)."""
    out = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        v = s.replace(",", " ").split()
        try:
            if len(v) == 1 or float(v[1]) != 0:
                out.append(float(v[0]))
        except ValueError:
            continue
    return np.asarray(out, dtype=float)


def read_status(path) -> dict:
    """key=value status log written by jobs/scripts/run_one.sh."""
    d = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip()
    return d


# --------------------------------------------------------------------------- core
def drop_timestamps(traj: PoseTrajectory3D, ts: np.ndarray, tol: float) -> tuple[PoseTrajectory3D, int]:
    if ts.size == 0:
        return traj, 0
    ts = np.sort(ts)
    pos = np.searchsorted(ts, traj.timestamps)
    lo = np.abs(traj.timestamps - ts[np.clip(pos - 1, 0, ts.size - 1)])
    hi = np.abs(traj.timestamps - ts[np.clip(pos, 0, ts.size - 1)])
    drop = np.minimum(lo, hi) <= tol
    if drop.all():
        return None, int(drop.sum())
    t = copy.deepcopy(traj)
    t.reduce_to_ids(np.flatnonzero(~drop))
    return t, int(drop.sum())


def associate_and_align(ref: PoseTrajectory3D, est: PoseTrajectory3D, max_diff: float = 0.02,
                        interpolate: bool = False):
    """Returns (ref_assoc, est_assoc_aligned, umeyama (R, t, s)). Copies, inputs untouched."""
    method = sync.SyncMethod.interpolation if interpolate else sync.SyncMethod.nearest_time
    r, e = sync.associate_trajectories(ref, est, max_diff=max_diff, sync_method=method)
    if r.num_poses < MIN_MATCHES:
        raise sync.SyncException(f"only {r.num_poses} matches (< {MIN_MATCHES})")
    R, t, s = e.align(r, correct_scale=True)
    return r, e, (R, t, s)


def _stats(x: np.ndarray, prefix: str) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {f"{prefix}_{k}": None for k in ("rmse", "mean", "median", "max")} | {f"{prefix}_n": 0}
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_n": int(x.size),
    }


def ate_errors(ref: PoseTrajectory3D, est: PoseTrajectory3D) -> np.ndarray:
    return np.linalg.norm(ref.positions_xyz - est.positions_xyz, axis=1)


def rpe_pairs(ref: PoseTrajectory3D, est: PoseTrajectory3D, pairs, exclude_times=()) -> tuple[np.ndarray, np.ndarray]:
    """Relative pose error for id pairs (i, j): E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j).

    Pairs whose time interval (t_i, t_j] contains any of `exclude_times` are skipped.
    Returns (translation [m], rotation [deg]).
    """
    Q, P, ts = ref.poses_se3, est.poses_se3, ref.timestamps
    ex = np.sort(np.asarray(exclude_times, dtype=float))
    tr, rot = [], []
    for i, j in pairs:
        if ex.size:
            k = np.searchsorted(ex, ts[i], side="right")
            if k < ex.size and ex[k] <= ts[j]:
                continue
        E = lie.relative_se3(lie.relative_se3(Q[i], Q[j]), lie.relative_se3(P[i], P[j]))
        tr.append(np.linalg.norm(E[:3, 3]))
        rot.append(lie.so3_log_angle(lie.so3_from_se3(E), degrees=True))
    return np.asarray(tr), np.asarray(rot)


def metric_block(ref, est, exclude_times=(), delta_m: float = 1.0) -> dict:
    out = _stats(ate_errors(ref, est), "ate")
    t, r = rpe_pairs(ref, est, filters.filter_pairs_by_index(ref.poses_se3, 1), exclude_times)
    out |= _stats(t, "rpe_trans_frame") | _stats(r, "rpe_rot_frame_deg")
    try:
        pairs_m = filters.filter_pairs_by_path(ref.poses_se3, delta_m)
    except Exception:  # noqa: BLE001  (too short / degenerate path)
        pairs_m = []
    t, r = rpe_pairs(ref, est, pairs_m, exclude_times)
    out |= _stats(t, "rpe_trans_per_m") | _stats(r, "rpe_rot_per_m_deg")
    out["n_matched"] = int(ref.num_poses)
    return out


def coverage(est: PoseTrajectory3D, frame_times: np.ndarray | None, gt: PoseTrajectory3D,
             max_gap: float) -> dict:
    """frames: share of input frames with an estimated pose (n_est / n_frames, capped at 1).
    time:   share of the sequence duration covered by est poses whose neighbour gap <= max_gap.
    Keyframe-only outputs (ORB-SLAM3 KeyFrameTrajectory, MASt3R) score low on 'frames' by
    construction; compare 'time' across systems in that case."""
    span_src = frame_times if frame_times is not None and frame_times.size > 1 else gt.timestamps
    duration = float(span_src[-1] - span_src[0]) if span_src.size > 1 else 0.0
    n_est = int(est.num_poses) if est is not None else 0
    cov_frames = None
    if frame_times is not None and frame_times.size:
        cov_frames = min(1.0, n_est / frame_times.size)
    covered = 0.0
    if n_est > 1 and duration > 0:
        gaps = np.diff(est.timestamps)
        covered = float(np.sum(gaps[gaps <= max_gap]))
    return {
        "frames": cov_frames,
        "time": (min(1.0, covered / duration) if duration > 0 else None),
        "n_frames": int(frame_times.size) if frame_times is not None else None,
        "n_est": n_est,
        "duration_s": duration,
        "max_gap_s": max_gap,
    }


def evaluate(est: PoseTrajectory3D, gt: PoseTrajectory3D, *, frame_times=None, lost_times=None,
             max_diff=0.02, interpolate=False, spike_k=5.0, spike_min_trans=0.0, spike_min_rot_deg=0.0,
             spike_use_rotation=True, filter_lost=True, filter_spikes=True, max_gap=0.5, delta_m=1.0):
    """Returns (results dict, filtered timeline or None, spikes)."""
    res = {
        "params": dict(max_diff=max_diff, interpolate=interpolate, spike_k=spike_k,
                       spike_min_trans=spike_min_trans, spike_min_rot_deg=spike_min_rot_deg,
                       spike_use_rotation=spike_use_rotation, filter_lost=filter_lost,
                       filter_spikes=filter_spikes, max_gap=max_gap, delta_m=delta_m),
        "n_gt": int(gt.num_poses),
        "coverage": coverage(est, frame_times, gt, max_gap),
        "raw": None, "filtered": None, "spikes": [],
    }
    if est is None or est.num_poses < MIN_MATCHES:
        res["status"] = "no_trajectory" if est is None else "too_few_poses"
        return res, None, []

    # ---- raw
    try:
        ref_a, est_a, (_, _, s) = associate_and_align(gt, est, max_diff, interpolate)
    except sync.SyncException as e:
        res["status"] = "association_failed"
        res["error"] = str(e)
        return res, None, []
    res["raw"] = metric_block(ref_a, est_a, delta_m=delta_m) | {"sim3_scale": float(s)}

    # ---- filtered
    est_f, n_lost = est, 0
    if filter_lost and lost_times is not None and lost_times.size:
        est_f, n_lost = drop_timestamps(est, lost_times, tol=1e-6)
        if est_f is None:
            res["status"] = "ok_raw_only"
            res["error"] = "filtered: all poses flagged lost"
            return res, None, []
    try:
        ref_f, est_fa, (_, _, s_f) = associate_and_align(gt, est_f, max_diff, interpolate)
    except sync.SyncException as e:
        res["status"] = "ok_raw_only"
        res["error"] = f"filtered: {e}"
        return res, None, []
    tl = tlmod.rpe_timeline(ref_f, est_fa)
    spikes = tlmod.detect_spikes(tl, spike_k, spike_min_trans, spike_min_rot_deg, spike_use_rotation)
    excl = []
    if filter_spikes and spikes:
        excl = [sp.t_end for sp in spikes]  # reference time base (used for RPE pair exclusion)
        est_times = est_fa.timestamps[[sp.idx for sp in spikes]]  # estimate's own time base
        est_s, _ = drop_timestamps(est_f, est_times, tol=max_diff if interpolate else 1e-6)
        try:
            ref_f, est_fa, (_, _, s_f) = associate_and_align(gt, est_s, max_diff, interpolate)
        except sync.SyncException:
            pass  # keep the alignment from before spike removal
    blk = metric_block(ref_f, est_fa, exclude_times=excl, delta_m=delta_m)
    res["filtered"] = blk | {"sim3_scale": float(s_f), "n_dropped_lost": n_lost, "n_spikes": len(spikes)}
    res["spikes"] = [sp.to_dict() for sp in spikes]
    res["status"] = "ok"
    return res, tl, spikes


def infer_ids(est_path: Path) -> dict:
    """results/<system>/<dataset>/<sequence>/traj.txt -> ids (best effort)."""
    p = est_path.resolve().parts
    if len(p) >= 4:
        return {"system": p[-4], "dataset": p[-3], "sequence": p[-2]}
    return {"system": None, "dataset": None, "sequence": None}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def _clean(o):
    """NaN/inf -> None so results.json stays valid JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", required=True, help="estimated trajectory (TUM)")
    ap.add_argument("--gt", required=True, help="ground truth (TUM, see eval.gt_convert)")
    ap.add_argument("--out", default=None, help="results.json path (default: next to --est)")
    ap.add_argument("--frames", default=None, help="frame list (rgb.txt) for coverage")
    ap.add_argument("--lost", default=None, help="lost-flag file ('t' or 't flag' per line)")
    ap.add_argument("--status", default=None, help="run status log (default: status.txt next to --est)")
    ap.add_argument("--system"), ap.add_argument("--dataset"), ap.add_argument("--sequence")
    ap.add_argument("--max-diff", type=float, default=0.02, help="association tolerance [s]")
    ap.add_argument("--interpolate", action="store_true", help="interpolate (for sparse GT, e.g. LaMAria)")
    ap.add_argument("--spike-k", type=float, default=5.0)
    ap.add_argument("--spike-min-trans", type=float, default=0.0, help="absolute floor [m]")
    ap.add_argument("--spike-min-rot-deg", type=float, default=0.0, help="absolute floor [deg]")
    ap.add_argument("--no-rot-spikes", action="store_true", help="detect spikes on translation only")
    ap.add_argument("--no-filter-lost", action="store_true")
    ap.add_argument("--no-filter-spikes", action="store_true")
    ap.add_argument("--max-gap", type=float, default=0.5, help="coverage: max gap between poses [s]")
    ap.add_argument("--delta-m", type=float, default=1.0, help="RPE segment length [m]")
    ap.add_argument("--plot", action="store_true", help="also write timeline.png")
    a = ap.parse_args(argv)

    est_path = Path(a.est)
    out = Path(a.out) if a.out else est_path.parent / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    ids = infer_ids(est_path)
    for k in ("system", "dataset", "sequence"):
        if getattr(a, k):
            ids[k] = getattr(a, k)

    gt = load_tum(a.gt)
    est = load_tum(est_path) if est_path.exists() else None
    frame_times = read_frame_times(a.frames) if a.frames else None
    lost = read_lost(a.lost) if a.lost else None

    res, tl, spikes = evaluate(
        est, gt, frame_times=frame_times, lost_times=lost, max_diff=a.max_diff, interpolate=a.interpolate,
        spike_k=a.spike_k, spike_min_trans=a.spike_min_trans, spike_min_rot_deg=a.spike_min_rot_deg,
        spike_use_rotation=not a.no_rot_spikes, filter_lost=not a.no_filter_lost,
        filter_spikes=not a.no_filter_spikes, max_gap=a.max_gap, delta_m=a.delta_m)
    status_path = Path(a.status) if a.status else est_path.parent / "status.txt"
    res = ids | {"est_file": str(est_path), "gt_file": str(a.gt), "run": read_status(status_path)} | res
    out.write_text(json.dumps(_clean(res), indent=2, default=_json_default))

    if tl is not None:
        tlmod.write_csv(tl, spikes, out.parent / "timeline.csv")
        if a.plot:
            title = " / ".join(str(ids[k]) for k in ("system", "dataset", "sequence"))
            tlmod.plot(tl, spikes, out.parent / "timeline.png", a.spike_k, a.spike_min_trans,
                       a.spike_min_rot_deg, title)
    r = res.get("raw") or {}
    f = res.get("filtered") or {}
    print(f"[{res['status']}] ATE raw={r.get('ate_rmse')} filtered={f.get('ate_rmse')} "
          f"spikes={len(res['spikes'])} coverage={res['coverage']['frames']} -> {out}")


if __name__ == "__main__":
    main()
