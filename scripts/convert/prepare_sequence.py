"""Turn a raw dataset sequence into the common input layout used by every system.

    $DATA_ROOT/prepared/<dataset>/<sequence>/
        rgb/000000.png ...   undistorted pinhole images (symlinks if already pinhole)
        rgb.txt              'timestamp[s] rgb/000000.png'  (TUM style; read by all runners)
        calib.txt            'fx fy cx cy'                  (DROID-SLAM / DPVO)
        mast3r_calib.yaml    width/height/calibration       (MASt3R-SLAM --calib)
        orbslam3.yaml        ORB-SLAM3 monocular settings
        gt_tum.txt           ground truth in TUM format, camera frame where an extrinsic is known
        meta.yaml            source paths and calibration used

CLI:
    python scripts/convert/prepare_sequence.py --all
    python scripts/convert/prepare_sequence.py --dataset euroc --sequence MH_04_difficult
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from eval import gt_convert  # noqa: E402


# --------------------------------------------------------------------------- config
def load_config(path=REPO / "configs" / "sequences.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text())["sequences"]


def find_entry(entries, dataset, sequence) -> dict:
    for e in entries:
        if e["dataset"] == dataset and e["sequence"] == sequence:
            return e
    raise KeyError(f"{dataset}/{sequence} not in sequences.yaml")


def resolve(raw: Path, rel: str) -> Path:
    """Resolve a path relative to the raw sequence dir; globs must match exactly one file."""
    p = (raw / rel)
    if any(c in rel for c in "*?["):
        hits = sorted(glob.glob(str(p)))
        if len(hits) != 1:
            raise FileNotFoundError(f"pattern {p} matched {len(hits)} files")
        return Path(hits[0])
    return p


# --------------------------------------------------------------------------- frames
def _numeric_rows(path: Path):
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        yield s.replace(",", " ").split()


def list_frames(entry: dict, raw: Path) -> list[tuple[float, Path]]:
    fmt, images = entry["format"], entry["images"]
    if fmt in ("euroc", "lamaria"):  # ASL: data.csv '#timestamp [ns],filename'
        csv = raw / images
        return [(int(r[0]) * 1e-9, csv.parent / "data" / r[1]) for r in _numeric_rows(csv)]
    if fmt in ("tum", "bonn"):       # rgb.txt 'timestamp rgb/xxx.png'
        lst = raw / images
        return [(float(r[0]), raw / r[1]) for r in _numeric_rows(lst)]
    if fmt == "uzhfpv":              # left_images.txt '# id timestamp image_name'
        lst = raw / images
        return [(float(r[1]), raw / r[2]) for r in _numeric_rows(lst)]
    if fmt == "tartanair":           # no timestamps: t_i = i / fps (same rule as gt_convert)
        files = sorted(raw.glob(images))
        return [(i / float(entry["fps"]), f) for i, f in enumerate(files)]
    raise ValueError(f"unknown format {fmt}")


# --------------------------------------------------------------------------- calibration
def load_calib(entry: dict, raw: Path) -> dict:
    c = entry["calibration"]
    src = c.get("source", "inline")
    if src == "inline":
        return dict(model=c.get("model", "pinhole"), width=int(c["width"]), height=int(c["height"]),
                    K=list(map(float, c["intrinsics"])), D=list(map(float, c.get("distortion", []))))
    f = resolve(raw, c["file"])
    d = yaml.safe_load(f.read_text())
    if src == "sensor_yaml":         # ASL sensor.yaml
        model = {"radial-tangential": "radtan", "radtan": "radtan", "equidistant": "equidistant",
                 "none": "pinhole"}.get(d.get("distortion_model", "radial-tangential"), "radtan")
        D = list(map(float, d.get("distortion_coefficients", [])))
        if not any(D):
            model = "pinhole"
        return dict(model=model, width=int(d["resolution"][0]), height=int(d["resolution"][1]),
                    K=list(map(float, d["intrinsics"])), D=D, file=str(f))
    if src == "kalibr":
        cam = d["cam0"]
        model = {"radtan": "radtan", "equidistant": "equidistant"}[cam["distortion_model"]]
        return dict(model=model, width=int(cam["resolution"][0]), height=int(cam["resolution"][1]),
                    K=list(map(float, cam["intrinsics"])), D=list(map(float, cam["distortion_coeffs"])),
                    file=str(f))
    raise ValueError(f"unknown calibration source {src}")


def undistort_maps(cal: dict):
    """Returns (K_new 3x3, mapx, mapy) or (K, None, None) for pinhole input."""
    fx, fy, cx, cy = cal["K"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    D = np.asarray(cal["D"], dtype=float)
    size = (cal["width"], cal["height"])
    if cal["model"] == "pinhole" or not np.any(D):
        return K, None, None
    if cal["model"] == "radtan":
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha=0, newImgSize=size)
        mx, my = cv2.initUndistortRectifyMap(K, D, None, K_new, size, cv2.CV_32FC1)
        return K_new, mx, my
    if cal["model"] == "equidistant":
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D[:4], size, np.eye(3), balance=0.0)
        mx, my = cv2.fisheye.initUndistortRectifyMap(K, D[:4], np.eye(3), K_new, size, cv2.CV_32FC1)
        return K_new, mx, my
    raise ValueError(f"unknown distortion model {cal['model']}")


def orbslam3_yaml(K, width, height, fps, n_features=1500) -> str:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return f"""%YAML:1.0
# generated by scripts/convert/prepare_sequence.py (undistorted pinhole input)
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera.width: {width}
Camera.height: {height}
Camera.newWidth: {width}
Camera.newHeight: {height}
Camera.fps: {int(round(fps))}
Camera.RGB: 0
ORBextractor.nFeatures: {n_features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


# --------------------------------------------------------------------------- GT
def body_to_cam(entry: dict, raw: Path, cal: dict):
    spec = entry.get("T_body_cam")
    if not spec:
        return None
    if spec == "auto":
        if "file" not in cal:
            raise ValueError(f"{entry['sequence']}: T_body_cam=auto needs a sensor_yaml/kalibr calibration")
        return gt_convert.load_matrix(cal["file"])
    return gt_convert.load_matrix(resolve(raw, spec))


def prepare(entry: dict, data_root: Path, force: bool = False) -> Path:
    raw = data_root / "raw" / entry["path"]
    out = data_root / "prepared" / entry["dataset"] / entry["sequence"]
    if (out / "rgb.txt").exists() and not force:
        print(f"[skip] {out} exists (use --force)")
        return out
    if not raw.is_dir():
        raise FileNotFoundError(f"raw sequence missing: {raw}")
    frames = list_frames(entry, raw)
    if not frames:
        raise RuntimeError(f"no frames found for {entry['dataset']}/{entry['sequence']}")
    cal = load_calib(entry, raw)
    K_new, mx, my = undistort_maps(cal)

    (out / "rgb").mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (t, src) in enumerate(frames):
        dst = out / "rgb" / f"{i:06d}.png"
        if mx is None:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
        else:
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"cannot read {src}")
            cv2.imwrite(str(dst), cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        lines.append(f"{t:.9f} rgb/{i:06d}.png")
    (out / "rgb.txt").write_text("# timestamp filename\n" + "\n".join(lines) + "\n")

    fx, fy, cx, cy = K_new[0, 0], K_new[1, 1], K_new[0, 2], K_new[1, 2]
    W, H = cal["width"], cal["height"]
    (out / "calib.txt").write_text(f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    (out / "mast3r_calib.yaml").write_text(yaml.safe_dump(
        {"width": W, "height": H, "calibration": [float(fx), float(fy), float(cx), float(cy)]}))
    (out / "orbslam3.yaml").write_text(orbslam3_yaml(K_new, W, H, entry.get("fps", 30)))

    gt_path = resolve(raw, entry["gt"])
    traj = gt_convert.convert(entry["format"], gt_path, body_to_cam(entry, raw, cal), fps=float(entry.get("fps", 10)))
    gt_convert.write_tum(traj, out / "gt_tum.txt")

    (out / "meta.yaml").write_text(yaml.safe_dump({
        "entry": entry, "raw": str(raw), "n_frames": len(frames), "calibration_in": cal,
        "K_out": [float(fx), float(fy), float(cx), float(cy)], "gt_poses": int(traj.num_poses),
    }, sort_keys=False))
    print(f"[ok] {out}: {len(frames)} frames, {traj.num_poses} GT poses, model={cal['model']}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs" / "sequences.yaml"))
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    ap.add_argument("--dataset"), ap.add_argument("--sequence")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if not a.data_root:
        ap.error("--data-root or $DATA_ROOT required")
    entries = load_config(a.config)
    todo = entries if a.all else [find_entry(entries, a.dataset, a.sequence)]
    failed = []
    for e in todo:
        try:
            prepare(e, Path(a.data_root), a.force)
        except Exception as ex:  # noqa: BLE001 - keep going over the other sequences
            print(f"[fail] {e['dataset']}/{e['sequence']}: {ex}")
            failed.append(e["sequence"])
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main()
