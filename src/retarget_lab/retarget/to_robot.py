"""MB-Lab character motion -> generic robot joint angles (pure Python, testable without Blender).

Pipeline (details and axis conventions in docs/conventions.md):
  1. For each human segment compute its world rotation *delta from the T-pose reference* the retarget used
     (`ref_rot`), express it in the robot frame (x forward, y left, z up) and compose it with a fixed correction S
     that maps the robot's zero-pose segment direction onto the reference direction (arms: the robot's arm hangs
     down, the T-pose arm points sideways, so S is a 90 degree swing).
        O_seg(t) = F Q_seg(t) F^T S,   Q_seg(t) = R_seg(t) R_seg(ref)^T
  2. Heading = yaw of the pelvis; the robot base is level (heading only), so the whole trunk tilt lands on the
     2-DoF waist. Each chain is then decomposed onto the robot's joint axes with signed-axis Euler
     decomposition (`kinematics.fk.decompose_tracked`), parent first, using the *robot's* parent orientation so
     truncation errors are absorbed by the child instead of accumulating.
  3. Optional damped-least-squares IK refinement of each arm toward the human hand position scaled by arm length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io import BoneData
from ..kinematics.fk import (
    axis_rot,
    body_fk,
    chain_fk,
    compose,
    decompose_tracked,
    min_swing,
    rot_z,
)
from ..skeleton import RobotSkeleton
from .ik import solve_arm_ik

# Blender world (Z up, character faces -Y, +X left)  ->  robot world (x forward, y left, z up)
F_BR = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
X, Y, Z = np.eye(3)
FULL = np.array([-180.0, 180.0])
# The third (dropped) angle of each decomposition is a residual the robot cannot realise. Giving it a +-30..45 deg
# range lets branch selection prefer the representation with a small residual instead of one that flips it by 180.
RES30, RES45 = np.array([-30.0, 30.0]), np.array([-45.0, 45.0])


@dataclass
class RobotMotion:
    """Result of retargeting one clip (all positions in the robot world frame, metres)."""

    fps: float
    q: np.ndarray  # (N, n_joints) rad, canonical joint order, limits NOT applied
    base_pos: np.ndarray  # (N, 3) base (hip-centre) position
    heading: np.ndarray  # (N,) rad, yaw of the level base frame
    targets: dict[str, np.ndarray]  # {'left'|'right': (N,3)} human hand target in the base frame
    method: str

    @property
    def n_frames(self) -> int:
        return self.q.shape[0]


def _segment_orientations(bones: BoneData, robot: RobotSkeleton) -> dict[str, np.ndarray]:
    hs = robot.human_source
    spec = {"pelvis": (hs["pelvis"], Z, True), "chest": (hs["chest"], Z, True), "head": (hs["head"], Z, True)}
    for side in ("left", "right"):
        for seg, d0, swing in [
            ("upper_arm", -Z, True),
            ("forearm", -Z, True),
            ("hand", -Z, True),
            ("thigh", -Z, True),
            ("shank", -Z, True),
            ("foot", X, False),  # MB-Lab's foot bone points down toward the toes, but the rest foot is flat
        ]:
            spec[f"{side}.{seg}"] = (hs[side][seg], d0, swing)
    out = {}
    for key, (bone, d0, swing) in spec.items():
        b = bones.idx(bone)
        q_world = bones.rot[:, b] @ bones.ref_rot[b].T  # delta from the retarget's T-pose reference
        q_robot = F_BR @ q_world @ F_BR.T
        u_ref = F_BR @ bones.ref_rot[b][:, 1]  # bone direction (local +y) in the T-pose reference
        s = min_swing(d0, u_ref) if swing else np.eye(3)
        out[key] = q_robot @ s
    return out


def _heading(o_pelvis: np.ndarray) -> np.ndarray:
    """Yaw of the pelvis from its left (hip-line) axis, which stays well defined when bending forward."""
    left = o_pelvis[:, :, 1]
    return np.unwrap(np.arctan2(-left[:, 0], left[:, 1]))


def _decompose(rel: np.ndarray, axes: list[np.ndarray], limits: list[np.ndarray]) -> np.ndarray:
    return decompose_tracked(rel, np.array(axes), np.array(limits))


def human_arm_length(bones: BoneData, robot: RobotSkeleton, side: str) -> float:
    """Shoulder -> elbow -> wrist -> palm length of the MB-Lab character at rest (m)."""
    hs = robot.human_source
    i = lambda n: bones.idx(n)  # noqa: E731
    sh, el, wr = (bones.rest_head[i(hs[side][k])] for k in ("upper_arm", "forearm", "hand"))
    palm = bones.rest_head[i(hs["hand_point"][side])]
    return float(np.linalg.norm(el - sh) + np.linalg.norm(wr - el) + np.linalg.norm(palm - wr))


def retarget_joint_angles(bones: BoneData, robot: RobotSkeleton) -> RobotMotion:
    """Joint-angle retarget (rotation decomposition only; no IK, no limits)."""
    n = bones.n_frames
    o = _segment_orientations(bones, robot)
    q = np.zeros((n, len(robot.joint_names)))

    # --- base: level frame with the pelvis heading, positioned at the hip-joint midpoint
    psi = _heading(o["pelvis"])
    h_rot = rot_z(psi)
    hs = robot.human_source
    mid = 0.5 * (bones.pos(hs["left"]["thigh"]) + bones.pos(hs["right"]["thigh"]))
    rest_mid = 0.5 * (
        bones.rest_head[bones.idx(hs["left"]["thigh"])] + bones.rest_head[bones.idx(hs["right"]["thigh"])]
    )
    s_root = robot.hip_height_m / rest_mid[2]
    rel_xy = (mid - mid[0]) @ F_BR.T
    base_pos = np.column_stack([s_root * rel_xy[:, 0], s_root * rel_xy[:, 1], robot.hip_height_m + s_root * (mid[:, 2] - rest_mid[2])])

    # --- waist (yaw, pitch) + dropped roll
    wc = robot.chains["waist"]
    ax = [wc.axes[0], wc.axes[1]]
    lim = [wc.limits_deg[0], wc.limits_deg[1]]
    q[:, robot.chain_slice("waist")] = _decompose(np.swapaxes(h_rot, 1, 2) @ o["chest"], ax, lim)
    torso = chain_fk(wc, q[:, robot.chain_slice("waist")])[:, -1, :3, :3]
    torso_r = h_rot @ torso  # torso orientation in the (heading-rotated) world, as the robot realises it

    # --- head
    hc = robot.chains["head"]
    ax = [hc.axes[0], hc.axes[1]]
    lim = [hc.limits_deg[0], hc.limits_deg[1]]
    q[:, robot.chain_slice("head")] = _decompose(np.swapaxes(torso_r, 1, 2) @ o["head"], ax, lim)

    # --- arms
    for side in ("left", "right"):
        ch = robot.chains[f"{side}_arm"]
        a, lm = ch.axes, ch.limits_deg
        q_sh = _decompose(np.swapaxes(torso_r, 1, 2) @ o[f"{side}.upper_arm"], list(a[:3]), list(lm[:3]))
        o_u = torso_r @ compose(a[:3], q_sh)
        q_el = _decompose(np.swapaxes(o_u, 1, 2) @ o[f"{side}.forearm"], [a[3]], [lm[3]])[:, 0]
        o_f = o_u @ axis_rot(a[3], q_el)
        q_wr = _decompose(np.swapaxes(o_f, 1, 2) @ o[f"{side}.hand"], [a[4], a[5]], [lm[4], lm[5]])
        q[:, robot.chain_slice(f"{side}_arm")] = np.column_stack([q_sh, q_el, q_wr])

    # --- legs
    for side in ("left", "right"):
        ch = robot.chains[f"{side}_leg"]
        a, lm = ch.axes, ch.limits_deg
        q_hip = _decompose(np.swapaxes(h_rot, 1, 2) @ o[f"{side}.thigh"], list(a[:3]), list(lm[:3]))
        o_t = h_rot @ compose(a[:3], q_hip)
        q_kn = _decompose(np.swapaxes(o_t, 1, 2) @ o[f"{side}.shank"], [a[3]], [lm[3]])[:, 0]
        o_s = o_t @ axis_rot(a[3], q_kn)
        q_an = _decompose(np.swapaxes(o_s, 1, 2) @ o[f"{side}.foot"], [a[4], a[5]], [lm[4], lm[5]])
        q[:, robot.chain_slice(f"{side}_leg")] = np.column_stack([q_hip, q_kn, q_an])

    # --- human hand targets in the base frame (scaled by arm length), shoulder from the robot's own torso pose
    fk = body_fk(robot, q)
    targets = {}
    for side in ("left", "right"):
        sh_h = bones.pos(hs[side]["upper_arm"])
        palm_h = bones.pos(hs["hand_point"][side])
        s_arm = robot.arm_length() + robot.dims["hand"]
        s_arm = s_arm / human_arm_length(bones, robot, side)
        vec_b = np.einsum("nji,nj->ni", h_rot, (palm_h - sh_h) @ F_BR.T)  # H^T F (palm - shoulder)
        targets[side] = fk[f"{side}_arm"][:, 0, :3, 3] + s_arm * vec_b
    return RobotMotion(bones.fps, q, base_pos, psi, targets, "joint_angle")


def retarget_ik(robot: RobotSkeleton, motion: RobotMotion, iters: int = 25) -> RobotMotion:
    """Refine both arms with DLS IK so the tool point matches the scaled human hand target."""
    q = motion.q.copy()
    torso = body_fk(robot, motion.q)["waist"][:, -1]
    for side in ("left", "right"):
        sl = robot.chain_slice(f"{side}_arm")
        q[:, sl] = solve_arm_ik(robot.chains[f"{side}_arm"], torso, motion.q[:, sl], motion.targets[side], iters=iters)
    return RobotMotion(motion.fps, q, motion.base_pos, motion.heading, motion.targets, "ik_refined")


def clamp_motion(robot: RobotSkeleton, q: np.ndarray, fps: float, rate_limit: bool = True) -> np.ndarray:
    """What the robot can actually do: clip to position limits, then slew-limit to max joint velocity.

    The slew limiter tracks the clipped command with at most vmax*dt change per frame (the robot lags fast
    motion instead of teleporting), then re-clips so the result still respects the position limits.
    """
    lim = np.deg2rad(robot.limits_deg)
    out = np.clip(q, lim[:, 0], lim[:, 1])
    if rate_limit:
        step = np.deg2rad(robot.max_vel_dps) / fps
        cur = out.copy()
        for t in range(1, q.shape[0]):
            cur[t] = cur[t - 1] + np.clip(out[t] - cur[t - 1], -step, step)
        out = np.clip(cur, lim[:, 0], lim[:, 1])
    return out
