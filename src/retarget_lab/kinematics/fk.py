"""Forward kinematics, Euler decomposition and Jacobians for the generic robot (numpy / scipy only).

Everything is vectorised over frames: joint angles are (N, n_joints) in radians unless a name says `_deg`.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ..skeleton import Chain, RobotSkeleton


# ------------------------------------------------------------------ rotation helpers
def axis_rot(axis: np.ndarray, angle) -> np.ndarray:
    """Rodrigues rotation about a fixed unit axis. angle: (N,) rad -> (N, 3, 3)."""
    a = np.asarray(angle, float)
    k = np.asarray(axis, float)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    s, c = np.sin(a)[..., None, None], np.cos(a)[..., None, None]
    return np.eye(3) + s * kx + (1 - c) * (kx @ kx)


def rot_z(a) -> np.ndarray:
    return axis_rot(np.array([0.0, 0.0, 1.0]), a)


def wrap_pi(a) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def min_swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction a to direction b, shape (3, 3)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1.0, 0])
        return axis_rot(perp / np.linalg.norm(perp), np.array(np.pi))
    return axis_rot(v / s, np.array(np.arctan2(s, c)))


# ------------------------------------------------------------------ chain FK
def chain_fk(chain: Chain, q: np.ndarray, base: np.ndarray | None = None) -> np.ndarray:
    """World frames of a serial chain.

    q: (N, nj) rad. base: (N,4,4) parent frame (identity if None).
    Returns (N, nj+1, 4, 4): [:,0] is the chain origin frame; [:,i+1] is the frame after joint i
    (rotation applied, then offset_after). The joint-i rotation centre is frames[:, i, :3, 3].
    """
    q = np.atleast_2d(q)
    n = q.shape[0]
    t0 = np.tile(np.eye(4), (n, 1, 1)) if base is None else base.copy()
    o = np.tile(np.eye(4), (n, 1, 1))
    o[:, :3, 3] = chain.origin
    frames = [t0 @ o]
    for i, j in enumerate(chain.joints):
        step = np.tile(np.eye(4), (n, 1, 1))
        step[:, :3, :3] = axis_rot(j.axis, q[:, i])
        step[:, :3, 3] = step[:, :3, :3] @ j.offset_after
        frames.append(frames[-1] @ step)
    return np.stack(frames, axis=1)


def body_fk(robot: RobotSkeleton, q: np.ndarray) -> dict[str, np.ndarray]:
    """FK for the whole robot. q: (N, n_joints) rad in canonical order -> {chain: frames (N, nj+1, 4, 4)}.

    The torso frame is the waist chain's last frame; arms and head hang off it, legs off the base.
    """
    out: dict[str, np.ndarray] = {}
    for cname, chain in robot.chains.items():
        sl = robot.chain_slice(cname)
        base = out["waist"][:, -1] if chain.parent == "torso" else None
        out[cname] = chain_fk(chain, q[:, sl], base)
    return out


def hand_positions(robot: RobotSkeleton, q: np.ndarray) -> dict[str, np.ndarray]:
    """Tool-point positions in the base frame: {'left': (N,3), 'right': (N,3)}."""
    fk = body_fk(robot, q)
    return {"left": fk["left_arm"][:, -1, :3, 3], "right": fk["right_arm"][:, -1, :3, 3]}


def jacobian_pos(chain: Chain, frames: np.ndarray) -> np.ndarray:
    """Position Jacobian of the chain tip w.r.t. its joint angles. frames: (N, nj+1, 4, 4) -> (N, 3, nj)."""
    tip = frames[:, -1, :3, 3]
    cols = []
    for i, j in enumerate(chain.joints):
        axis_w = frames[:, i, :3, :3] @ j.axis
        cols.append(np.cross(axis_w, tip - frames[:, i, :3, 3]))
    return np.stack(cols, axis=-1)


# ------------------------------------------------------------------ Euler decomposition
def _letters(axes: np.ndarray) -> tuple[str, np.ndarray]:
    """Signed principal axes -> (letters like 'YXZ', signs). Raises if an axis is not principal."""
    letters, signs = "", []
    for a in axes:
        i = int(np.argmax(np.abs(a)))
        if not np.isclose(abs(a[i]), 1.0):
            raise ValueError(f"axis {a} is not a principal axis")
        letters += "XYZ"[i]
        signs.append(np.sign(a[i]))
    return letters, np.array(signs)


def euler_solutions(r: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """Both Euler solutions of R = R(a1,t1) R(a2,t2) R(a3,t3) for signed principal axes (distinct letters).

    r: (N,3,3) -> (N, 2, 3) radians. Solution 0 has the middle unsigned angle in [-90, 90] deg; solution 1 is
    the alternate branch (u1+pi, pi-u2, u3+pi) that matters for joints (e.g. a shoulder roll up to 135 deg)
    whose range exceeds 90 deg. Signed angle t_i = sign_i * u_i because R(-e, t) = R(e, -t).
    """
    letters, signs = _letters(axes)
    u = Rotation.from_matrix(r).as_euler(letters, degrees=False)  # intrinsic: R = R_l1(u1) R_l2(u2) R_l3(u3)
    u_alt = np.stack([u[:, 0] + np.pi, np.pi - u[:, 1], u[:, 2] + np.pi], axis=1)
    return wrap_pi(np.stack([u * signs, u_alt * signs], axis=1))


def compose(axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Product of rotations about the given axes (in order). angles: (N, k) -> (N,3,3)."""
    r = np.tile(np.eye(3), (angles.shape[0], 1, 1))
    for k, a in enumerate(axes):
        r = r @ axis_rot(a, angles[:, k])
    return r


def _euler_first_frame(r0: np.ndarray, axes: np.ndarray, lim: np.ndarray | None) -> np.ndarray:
    cands = euler_solutions(r0[None], axes)[0]
    score = 1e-3 * np.abs(cands).sum(axis=1)
    if lim is not None:
        score = score + (np.maximum(lim[:, 0] - cands, 0) + np.maximum(cands - lim[:, 1], 0)).sum(axis=1)
    return cands[int(np.argmin(score))]


def decompose_tracked(
    r: np.ndarray,
    axes: np.ndarray,
    limits_deg: np.ndarray | None = None,
    iters: int = 12,
    damping: float = 0.15,
) -> np.ndarray:
    """Decompose a sequence of rotations onto signed principal axes with temporal continuity.

    * 3 axes (shoulder, hip): exact. Closed-form Euler (branch chosen inside the limits) for the first frame, then
      damped Gauss-Newton from the previous frame's angles, so the solution never flips branches and, at the
      middle-angle singularity, moves the angles by the minimum amount. `_canonical_branch` finally undoes any
      winding onto the alternate branch.
    * 1-2 axes (waist, head, wrist, elbow, knee, ankle): the rotation has components the joints cannot realise, so
      this is a tracked least-squares fit (minimise the rotation error), which has no gimbal problem and spreads
      the unrealisable residual instead of dropping a branch-dependent third angle.
    Returns (N, k) radians.
    """
    lim = None if limits_deg is None else np.deg2rad(limits_deg)
    n, k = r.shape[0], len(axes)
    exact = k == 3
    out = np.zeros((n, k))
    if exact:
        out[0] = _euler_first_frame(r[0], axes, lim)
    sols = euler_solutions(r, axes) if exact else None
    for t in range(n):
        if t == 0 and exact:
            continue
        q = out[t - 1].copy() if t > 0 else np.zeros(k)
        n_it = iters if t > 0 else 60
        ok = False
        for _ in range(n_it):
            cur = np.eye(3)
            cols = []
            for i in range(k):
                cols.append(cur @ axes[i])
                cur = cur @ axis_rot(axes[i], q[i])
            err = Rotation.from_matrix(r[t] @ cur.T).as_rotvec()
            j = np.stack(cols, axis=1)  # (3, k)
            dq = np.linalg.solve(j.T @ j + damping**2 * np.eye(k), j.T @ err)
            q = q + dq * min(1.0, 0.5 / max(np.linalg.norm(dq), 1e-12))  # cap the step (rad)
            if np.linalg.norm(dq) < 1e-9:
                ok = True
                break
        if exact and not ok:
            cur = compose(axes, q[None])[0]
            if np.linalg.norm(Rotation.from_matrix(r[t] @ cur.T).as_rotvec()) > 1e-4:
                c = sols[t]
                prev = out[t - 1]
                q = prev + wrap_pi(c[int(np.argmin(np.abs(wrap_pi(c - prev)).sum(axis=1)))] - prev)
        out[t] = q
    return _canonical_branch(out, axes, lim) if exact else out


def _canonical_branch(q: np.ndarray, axes: np.ndarray, lim: np.ndarray | None, margin_deg: float = 30.0) -> np.ndarray:
    """Wrap angles to (-pi, pi] and move frames that wound onto the alternate Euler branch back.

    Gauss-Newton tracking is continuous but can run through the singularity onto the alternate branch
    (pitch+180, roll=180-roll, yaw+180), which describes the same rotation with absurd angles. A frame switches to
    the alternate branch only if that cuts the joint-limit excess by more than `margin_deg`, so genuinely
    out-of-limit motion (e.g. bending 100 deg at a 60 deg waist) is left alone instead of being "fixed".
    """
    if lim is None:
        return wrap_pi(q)
    _, signs = _letters(axes)
    cur = wrap_pi(q)
    alt = wrap_pi(np.stack([q[:, 0] + signs[0] * np.pi, signs[1] * np.pi - q[:, 1], q[:, 2] + signs[2] * np.pi], axis=1))

    def excess(a):
        return (np.maximum(lim[:, 0] - a, 0) + np.maximum(a - lim[:, 1], 0)).sum(axis=1)

    use_alt = excess(cur) - excess(alt) > np.deg2rad(margin_deg)
    return np.where(use_alt[:, None], alt, cur)
