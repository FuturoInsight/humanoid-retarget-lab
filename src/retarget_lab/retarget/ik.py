"""Damped-least-squares IK for the robot arm (position target, vectorised over frames)."""

from __future__ import annotations

import numpy as np

from ..kinematics.fk import chain_fk, jacobian_pos
from ..skeleton import Chain


def solve_arm_ik(
    chain: Chain,
    torso: np.ndarray,
    q0: np.ndarray,
    target: np.ndarray,
    iters: int = 25,
    damping: float = 0.05,
    stay: float = 0.02,
    max_step: float = 0.35,
    limits_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Refine arm joint angles so the tool point reaches `target`.

    chain: arm chain; torso: (N,4,4) torso frame in the base frame; q0: (N, nj) rad initial guess
    (the joint-angle retarget); target: (N,3) in the base frame.

    Each iteration solves the damped, regularised least-squares problem
        (J^T J + (lambda^2 + mu) I) dq = J^T e + mu (q0 - q)
    for every frame at once. `stay` (mu) pulls the solution toward the joint-angle retarget so wrist angles
    and elbow/shoulder redundancy stay close to what the human did and stay temporally smooth. If limits_deg is
    given the joints are projected onto them after each step.
    """
    q = q0.copy()
    n, nj = q.shape
    lam2 = damping**2
    eye = np.eye(nj)
    lim = None if limits_deg is None else np.deg2rad(limits_deg)
    for _ in range(iters):
        frames = chain_fk(chain, q, torso)
        err = target - frames[:, -1, :3, 3]
        jac = jacobian_pos(chain, frames)
        a = np.einsum("nij,nik->njk", jac, jac) + (lam2 + stay) * eye
        b = np.einsum("nij,ni->nj", jac, err) + stay * (q0 - q)
        dq = np.linalg.solve(a, b[..., None])[..., 0]
        norm = np.linalg.norm(dq, axis=1, keepdims=True)
        dq = dq * np.minimum(1.0, max_step / np.maximum(norm, 1e-12))
        q = q + dq
        if lim is not None:
            q = np.clip(q, lim[:, 0], lim[:, 1])
    return q
