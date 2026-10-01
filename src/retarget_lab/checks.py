"""Kinematics checks K1-K8 and the per-clip verdict.

Every check returns a list of rows with the same keys (`CHECK_COLUMNS`) so they stack into one DataFrame:
check_id, clip_id, joint, frame_start, frame_end, frame_range, severity, value, threshold, message.
Thresholds come from config/thresholds.yaml. Angles are radians internally, degrees in rows/messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from .io import BoneData
from .kinematics.fk import body_fk
from .kinematics.workspace import ReachEnvelope
from .skeleton import RobotSkeleton

CHECK_COLUMNS = [
    "check_id", "clip_id", "joint", "frame_start", "frame_end", "frame_range",
    "severity", "value", "threshold", "message",
]  # fmt: skip


@dataclass
class ClipData:
    """Everything the checks need about one retargeted clip."""

    clip_id: str
    fps: float
    q_raw: np.ndarray  # (N, nj) rad, limits ignored
    q_clamped: np.ndarray  # (N, nj) rad, limits + velocity enforced
    hands_raw: dict[str, np.ndarray]  # FK tool points, base frame
    hands_clamped: dict[str, np.ndarray]
    targets: dict[str, np.ndarray]  # scaled human hand target, base frame
    bones: BoneData | None = None  # for K6 (MB-Lab foot motion)


def row(check_id, clip_id, severity, value, threshold, message, joint="*", seg=(0, 0)):
    return {
        "check_id": check_id, "clip_id": clip_id, "joint": joint,
        "frame_start": int(seg[0]), "frame_end": int(seg[1]),
        "frame_range": f"{int(seg[0])}-{int(seg[1])}",
        "severity": severity, "value": float(value), "threshold": float(threshold), "message": message,
    }  # fmt: skip


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean mask as inclusive (start, end) frame indices."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def longest(mask: np.ndarray) -> tuple[int, int]:
    segs = segments(mask)
    return max(segs, key=lambda s: s[1] - s[0]) if segs else (0, 0)


def joint_velocity_dps(q: np.ndarray, fps: float) -> np.ndarray:
    return np.rad2deg(np.gradient(q, axis=0)) * fps


def _sev(value, warn, fail=None):
    if fail is not None and value > fail:
        return "fail"
    return "warn" if value > warn else "ok"


# ----------------------------------------------------------------------------------------------- K1
def k1_fk_error(d: ClipData, th: dict, ik_method: str) -> list[dict]:
    t = th["k1_fk_error"]
    rows = []
    for side in ("left", "right"):
        err = np.linalg.norm(d.hands_raw[side] - d.targets[side], axis=1) * 100
        sev = "warn" if err.mean() > t["mean_cm_warn"] or err.max() > t["max_cm_warn"] else "ok"
        rows.append(row(
            "K1", d.clip_id, sev, err.mean(), t["mean_cm_warn"],
            f"{side} hand FK vs scaled human target ({ik_method}): mean {err.mean():.2f} cm, max {err.max():.2f} cm",
            joint=f"{side}_hand", seg=(int(err.argmax()),) * 2,
        ))  # fmt: skip
    return rows


# ----------------------------------------------------------------------------------------------- K2
def k2_position_limits(d: ClipData, robot: RobotSkeleton, th: dict) -> list[dict]:
    t = th["k2_position_limits"]
    q = np.rad2deg(d.q_raw)
    lim = robot.limits_deg
    over = (q < lim[:, 0] - t["tolerance_deg"]) | (q > lim[:, 1] + t["tolerance_deg"])
    any_over = over.any(axis=1)
    share = any_over.mean()
    rows = [row("K2", d.clip_id, _sev(share, t["share_warn"], t["share_fail"]), share, t["share_warn"],
                f"{share:.0%} of frames have at least one joint beyond its position limit",
                seg=longest(any_over))]  # fmt: skip
    for j, name in enumerate(robot.joint_names):
        s = over[:, j].mean()
        if s > 0:
            worst = np.maximum(lim[j, 0] - q[:, j], q[:, j] - lim[j, 1]).max()
            rows.append(row("K2", d.clip_id, _sev(s, t["share_warn"], t["share_fail"]), s, t["share_warn"],
                            f"{name}: {s:.0%} of frames beyond [{lim[j, 0]:.0f}, {lim[j, 1]:.0f}] deg "
                            f"(worst overshoot {worst:.0f} deg)", joint=name, seg=longest(over[:, j])))  # fmt: skip
    return rows


# ----------------------------------------------------------------------------------------------- K3
def k3_velocity_limits(d: ClipData, robot: RobotSkeleton, th: dict) -> list[dict]:
    t = th["k3_velocity_limits"]
    v = np.abs(joint_velocity_dps(d.q_raw, d.fps))
    over = v > robot.max_vel_dps
    any_over = over.any(axis=1)
    share = any_over.mean()
    rows = [row("K3", d.clip_id, _sev(share, t["share_warn"], t["share_fail"]), share, t["share_warn"],
                f"{share:.0%} of frames have a joint faster than its velocity limit", seg=longest(any_over))]  # fmt: skip
    for j, name in enumerate(robot.joint_names):
        s = over[:, j].mean()
        if s > 0:
            rows.append(row("K3", d.clip_id, _sev(s, t["share_warn"], t["share_fail"]), s, t["share_warn"],
                            f"{name}: {s:.0%} of frames above {robot.max_vel_dps[j]:.0f} deg/s "
                            f"(peak {v[:, j].max():.0f} deg/s)", joint=name, seg=longest(over[:, j])))  # fmt: skip
    return rows


# ----------------------------------------------------------------------------------------------- K4
def k4_clamp_distortion(d: ClipData, th: dict) -> list[dict]:
    t = th["k4_clamp_distortion"]
    dist = (
        np.maximum(
            np.linalg.norm(d.hands_raw["left"] - d.hands_clamped["left"], axis=1),
            np.linalg.norm(d.hands_raw["right"] - d.hands_clamped["right"], axis=1),
        )
        * 100
    )
    mean, mx = dist.mean(), dist.max()
    if mean > t["mean_cm_fail"] or mx > t["max_cm_fail"]:
        sev = "fail"
    elif mean > t["mean_cm_warn"] or mx > t["max_cm_warn"]:
        sev = "warn"
    else:
        sev = "ok"
    return [row("K4", d.clip_id, sev, mean, t["mean_cm_warn"],
                f"clamping moves the hand by {mean:.1f} cm on average, {mx:.1f} cm at worst",
                seg=(int(dist.argmax()),) * 2)]  # fmt: skip


# ----------------------------------------------------------------------------------------------- K5
def k5_reach(d: ClipData, envelopes: dict[str, ReachEnvelope], th: dict) -> list[dict]:
    t = th["k5_reach_envelope"]
    tol = t["tolerance_cm"] / 100
    per_side = {s: envelopes[s].outside(d.targets[s], tol) for s in ("left", "right")}
    both = per_side["left"] | per_side["right"]
    share = both.mean()
    rows = [row("K5", d.clip_id, _sev(share, t["share_warn"], t["share_fail"]), share, t["share_warn"],
                f"{share:.0%} of frames have a hand target outside the robot's reach envelope",
                seg=longest(both))]  # fmt: skip
    for s, m in per_side.items():
        if m.any():
            far = envelopes[s].distance(d.targets[s]).max() * 100
            rows.append(row("K5", d.clip_id, _sev(m.mean(), t["share_warn"], t["share_fail"]), m.mean(),
                            t["share_warn"], f"{s} hand: {m.mean():.0%} of frames out of reach "
                            f"(up to {far:.0f} cm beyond the envelope)", joint=f"{s}_hand", seg=longest(m)))  # fmt: skip
    return rows


# ----------------------------------------------------------------------------------------------- K6
def foot_contacts(bones: BoneData, th: dict) -> dict[str, dict[str, np.ndarray]]:
    """Per foot: contact mask, horizontal speed (m/s) and floor-relative height of the ankle bone (MB-Lab retarget).

    Contact is decided from height and vertical speed only (horizontal speed is what K6 judges, so using it here would
    be circular). The floor reference is the foot's own 5th-percentile ankle height over the clip.
    """
    t = th["k6_foot_sliding"]
    out = {}
    for side in ("L", "R"):
        ankle = bones.pos(f"foot_{side}")
        height = ankle[:, 2] - np.percentile(ankle[:, 2], 5)
        vz = np.gradient(ankle[:, 2]) * bones.fps
        speed = np.linalg.norm(np.gradient(ankle[:, :2], axis=0), axis=1) * bones.fps
        contact = (height < t["contact_height_cm"] / 100) & (np.abs(vz) < t["contact_max_vz_mps"])
        keep = np.zeros_like(contact)
        for s0, e0 in segments(contact):
            if e0 - s0 + 1 >= t["min_run_frames"]:
                keep[s0 : e0 + 1] = True
        out[side] = {"contact": keep, "speed": speed, "height": height, "ankle": ankle}
    return out


def k6_foot_sliding(d: ClipData, th: dict) -> list[dict]:
    """Share of foot-contact frames in which the planted ankle still moves faster than the slide speed."""
    t = th["k6_foot_sliding"]
    if d.bones is None:
        return []
    fc = foot_contacts(d.bones, th)
    tot_contact = tot_slide = 0
    worst = 0.0
    sliding_any = np.zeros(d.bones.n_frames, bool)
    for f in fc.values():
        slide = f["contact"] & (f["speed"] > t["slide_speed_mps"])
        tot_contact += int(f["contact"].sum())
        tot_slide += int(slide.sum())
        sliding_any |= slide
        if slide.any():
            worst = max(worst, float(f["speed"][slide].max()))
    share = tot_slide / tot_contact if tot_contact else 0.0
    return [row("K6", d.clip_id, "warn" if share > t["share_warn"] else "ok", share, t["share_warn"],
                f"{share:.0%} of foot-contact frames slide faster than {t['slide_speed_mps']} m/s (peak {worst:.2f} m/s)",
                seg=longest(sliding_any))]  # fmt: skip


# ----------------------------------------------------------------------------------------------- K7
def _point_to_vertical_segment(p: np.ndarray, z0: float, z1: float) -> np.ndarray:
    """Distance from points (..., 3) to the segment x=y=0, z in [z0, z1]."""
    dz = np.maximum(np.maximum(z0 - p[..., 2], p[..., 2] - z1), 0.0)
    return np.hypot(np.hypot(p[..., 0], p[..., 1]), dz)


def self_collision_mask(robot: RobotSkeleton, q: np.ndarray) -> dict[str, np.ndarray]:
    """Capsule proxy: forearm and hand capsules vs the torso capsule. Returns per-arm collision masks."""
    c = robot.collision
    fk = body_fk(robot, q)
    torso_inv = np.linalg.inv(fk["waist"][:, -1])
    u = np.linspace(0.0, 1.0, 9)[None, :, None]
    out = {}
    for side in ("left", "right"):
        f = fk[f"{side}_arm"]
        elbow, wrist, tip = f[:, 3, :3, 3], f[:, 4, :3, 3], f[:, 6, :3, 3]

        def to_torso(pts):
            return np.einsum("nij,nkj->nki", torso_inv[:, :3, :3], pts) + torso_inv[:, None, :3, 3]

        fore = to_torso(elbow[:, None] + u * (wrist - elbow)[:, None])
        hand = to_torso(wrist[:, None] + u * (tip - wrist)[:, None])
        z0, z1, rt = c["torso"]["z_from"], c["torso"]["z_to"], c["torso"]["radius"]
        hit_f = (_point_to_vertical_segment(fore, z0, z1) < rt + c["forearm_radius"]).any(axis=1)
        hit_h = (_point_to_vertical_segment(hand, z0, z1) < rt + c["hand_radius"]).any(axis=1)
        out[side] = hit_f | hit_h
    return out


def k7_self_collision(d: ClipData, robot: RobotSkeleton, th: dict) -> list[dict]:
    t = th["k7_self_collision"]
    hits = self_collision_mask(robot, d.q_clamped)
    both = hits["left"] | hits["right"]
    share = both.mean()
    rows = [row("K7", d.clip_id, _sev(share, t["share_warn"], t["share_fail"]), share, t["share_warn"],
                f"{share:.0%} of frames have a hand/forearm inside the torso capsule (executed, clamped motion)",
                seg=longest(both))]  # fmt: skip
    for s, m in hits.items():
        if m.any():
            rows.append(row("K7", d.clip_id, _sev(m.mean(), t["share_warn"], t["share_fail"]), m.mean(),
                            t["share_warn"], f"{s} arm: {m.mean():.0%} of frames in collision",
                            joint=f"{s}_arm", seg=longest(m)))  # fmt: skip
    return rows


# ----------------------------------------------------------------------------------------------- K8
def high_freq_rms_deg(q: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """RMS (deg) of each joint angle after high-pass filtering at cutoff_hz."""
    nyq = fps / 2
    if cutoff_hz >= nyq or q.shape[0] < 16:
        return np.zeros(q.shape[1])
    b, a = butter(4, cutoff_hz / nyq, btype="high")
    hp = filtfilt(b, a, np.rad2deg(q), axis=0)
    return np.sqrt((hp**2).mean(axis=0))


def k8_jitter(d: ClipData, robot: RobotSkeleton, th: dict) -> list[dict]:
    t = th["k8_jitter"]
    rms = high_freq_rms_deg(d.q_raw, d.fps, t["cutoff_hz"])
    worst = int(rms.argmax())
    rows = [row("K8", d.clip_id, "warn" if rms.max() > t["hf_rms_deg_warn"] else "ok", rms.max(),
                t["hf_rms_deg_warn"], f"highest high-frequency (> {t['cutoff_hz']:.0f} Hz) noise: "
                f"{rms[worst]:.2f} deg RMS on {robot.joint_names[worst]}", joint=robot.joint_names[worst])]  # fmt: skip
    return rows


# ----------------------------------------------------------------------------------------------- verdict
def run_checks(d: ClipData, robot: RobotSkeleton, th: dict, envelopes, ik_method: str) -> list[dict]:
    return (
        k1_fk_error(d, th, ik_method)
        + k2_position_limits(d, robot, th)
        + k3_velocity_limits(d, robot, th)
        + k4_clamp_distortion(d, th)
        + k5_reach(d, envelopes, th)
        + k6_foot_sliding(d, th)
        + k7_self_collision(d, robot, th)
        + k8_jitter(d, robot, th)
    )


_RULE_TO_CHECK = {
    "k2_share_fail": ("K2", "fail"), "k3_share_fail": ("K3", "fail"), "k4_fail": ("K4", "fail"),
    "k5_share_fail": ("K5", "fail"), "k7_share_fail": ("K7", "fail"),
    "k2_share_warn": ("K2", "warn"), "k3_share_warn": ("K3", "warn"), "k4_warn": ("K4", "warn"),
    "k5_share_warn": ("K5", "warn"), "k7_share_warn": ("K7", "warn"),
}  # fmt: skip


def verdict(rows: list[dict], th: dict) -> tuple[str, list[str]]:
    """USABLE / USABLE WITH CLAMPING / NOT FEASIBLE from the clip-level (joint == '*') rows, rules in config."""
    sev = {r["check_id"]: r["severity"] for r in rows if r["joint"] == "*" or r["check_id"] == "K4"}
    msg = {r["check_id"]: r["message"] for r in rows if r["joint"] == "*" or r["check_id"] == "K4"}

    def hits(rule_names, level):
        out = []
        for name in rule_names:
            cid, lvl = _RULE_TO_CHECK[name]
            if lvl == level and sev.get(cid) == level:
                out.append(f"{cid}: {msg[cid]}")
        return out

    fails = hits(th["verdict"]["not_feasible_if"], "fail")
    if fails:
        return "NOT FEASIBLE", fails
    warns = hits(th["verdict"]["clamp_if"], "warn")
    if warns:
        return "USABLE WITH CLAMPING", warns
    return "USABLE", []
