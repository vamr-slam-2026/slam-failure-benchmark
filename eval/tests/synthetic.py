"""Synthetic GT / estimate generator for tests."""
from __future__ import annotations

import numpy as np
from evo.core import lie_algebra as lie
from evo.core.trajectory import PoseTrajectory3D


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def make_gt(n: int = 600, rate: float = 20.0, radius: float = 3.0, omega: float = 0.3) -> PoseTrajectory3D:
    """Helix-like GT: ~0.9 m/s, camera yawing along the tangent with a little roll."""
    t = np.arange(n) / rate + 1000.0  # non-zero start, like real datasets
    tau = t - t[0]
    poses = []
    for k in range(n):
        a = omega * tau[k]
        p = np.array([radius * np.cos(a), radius * np.sin(a), 0.5 * np.sin(0.3 * tau[k])])
        R = _rot_z(a + np.pi / 2) @ lie.so3_exp(np.array([0.05 * np.sin(tau[k]), 0, 0]))
        poses.append(lie.se3(R, p))
    return PoseTrajectory3D(poses_se3=poses, timestamps=t)


def corrupt(gt: PoseTrajectory3D, *, jump_frame: int | None = 350, jump_trans: float = 0.5,
            jump_rot_deg: float = 10.0, scale_start: float = 1.0, scale_end: float = 1.3,
            drift_sigma: float = 0.001, noise_pos: float = 0.004, noise_rot_deg: float = 0.05,
            sim3_scale: float = 2.5, seed: int = 0) -> PoseTrajectory3D:
    """Estimate = GT relative motions with drift + linearly changing scale + one jump,
    integrated, expressed in a random Sim(3) frame, plus per-pose noise."""
    rng = np.random.default_rng(seed)
    Q = gt.poses_se3
    n = len(Q)
    scales = np.linspace(scale_start, scale_end, n)
    P = [np.eye(4)]
    for k in range(1, n):
        rel = lie.relative_se3(Q[k - 1], Q[k]).copy()
        rel[:3, 3] = rel[:3, 3] * scales[k] + rng.normal(0, drift_sigma, 3)
        if jump_frame is not None and k == jump_frame:
            J = lie.se3(lie.so3_exp(np.deg2rad(jump_rot_deg) * np.array([0, 0, 1.0])),
                        np.array([jump_trans, 0.0, 0.0]))
            rel = rel @ J
        P.append(P[-1] @ rel)
    # random global Sim(3) frame (monocular: arbitrary origin/orientation/scale)
    R0 = lie.so3_exp(np.array([0.3, -0.2, 1.1]))
    t0 = np.array([5.0, -2.0, 1.0])
    out = []
    for X in P:
        Y = np.eye(4)
        Y[:3, :3] = R0 @ X[:3, :3] @ lie.so3_exp(rng.normal(0, np.deg2rad(noise_rot_deg), 3))
        Y[:3, 3] = sim3_scale * (R0 @ X[:3, 3]) + t0 + rng.normal(0, noise_pos * sim3_scale, 3)
        out.append(Y)
    return PoseTrajectory3D(poses_se3=out, timestamps=gt.timestamps.copy())
