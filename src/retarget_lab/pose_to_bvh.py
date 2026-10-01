"""Turn 3D pose landmarks (MediaPipe world landmarks) into a CMU-style BVH that enters the same pipeline.

Landmark positions give bone *directions* but not bone twist, so each joint's world rotation is the minimal swing from
its T-pose direction (taken from a template CMU BVH, whose frame 0 is a T-pose) to the observed direction. The
resulting BVH reuses the template's hierarchy, joint names and offsets, with the template's T-pose as frame 0, so
`scripts/retarget_to_mblab.py` and the rest of the pipeline treat it like any CMU clip.

Pure numpy/scipy: unit-tested with synthetic landmarks (tests/test_pose_to_bvh.py).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import Bvh, forward_kinematics, parse_bvh
from .kinematics.fk import min_swing

# MediaPipe pose landmark indices
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_IDX, R_IDX = 19, 20
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_FOOT, R_FOOT = 23, 24, 25, 26, 27, 28, 31, 32

# For each template joint that has a driving segment: the joint at the far end of that segment.
SEGMENT_END = {
    "LowerBack": "Spine", "Spine": "Spine1", "Spine1": "Neck1", "Neck": "Head", "Neck1": "Head", "Head": "Head_End",
    "LeftShoulder": "LeftArm", "LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand", "LeftHand": "LeftHandIndex1",
    "RightShoulder": "RightArm", "RightArm": "RightForeArm", "RightForeArm": "RightHand", "RightHand": "RightHandIndex1",
    "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "LeftFoot": "LeftToeBase", "LeftToeBase": "LeftToeBase_End",
    "RightUpLeg": "RightLeg", "RightLeg": "RightFoot", "RightFoot": "RightToeBase", "RightToeBase": "RightToeBase_End",
}  # fmt: skip


def mediapipe_to_bvh_space(lm: np.ndarray) -> np.ndarray:
    """MediaPipe world landmarks (x toward the person's left in a selfie view, y down, z away from the camera plane
    toward the camera being negative) -> BVH space (Y up, person faces +Z, +X is the person's left)."""
    return np.stack([lm[..., 0], -lm[..., 1], -lm[..., 2]], axis=-1)


def joint_positions(lm: np.ndarray) -> dict[str, np.ndarray]:
    """Template joint name -> (F, 3) position, built from (F, 33, 3) landmarks already in BVH space."""
    hips = 0.5 * (lm[:, L_HIP] + lm[:, R_HIP])
    neck = 0.5 * (lm[:, L_SH] + lm[:, R_SH])
    head = 0.5 * (lm[:, L_EAR] + lm[:, R_EAR])
    up = neck - hips
    p = {
        "Hips": hips, "LowerBack": hips, "Spine": hips + 0.38 * up, "Spine1": hips + 0.78 * up, "Neck": neck,
        "Neck1": neck + 0.5 * (head - neck), "Head": head, "Head_End": head + 1.1 * (head - neck),
        "LeftShoulder": neck, "RightShoulder": neck,
        "LeftArm": lm[:, L_SH], "LeftForeArm": lm[:, L_EL], "LeftHand": lm[:, L_WR], "LeftHandIndex1": lm[:, L_IDX],
        "RightArm": lm[:, R_SH], "RightForeArm": lm[:, R_EL], "RightHand": lm[:, R_WR], "RightHandIndex1": lm[:, R_IDX],
        "LeftUpLeg": lm[:, L_HIP], "LeftLeg": lm[:, L_KN], "LeftFoot": lm[:, L_AN], "LeftToeBase": lm[:, L_FOOT],
        "RightUpLeg": lm[:, R_HIP], "RightLeg": lm[:, R_KN], "RightFoot": lm[:, R_AN], "RightToeBase": lm[:, R_FOOT],
    }  # fmt: skip
    p["LeftToeBase_End"] = lm[:, L_FOOT] + 0.5 * (lm[:, L_FOOT] - lm[:, L_AN])
    p["RightToeBase_End"] = lm[:, R_FOOT] + 0.5 * (lm[:, R_FOOT] - lm[:, R_AN])
    return p


def _root_frame(left_hip, right_hip, hips, neck) -> np.ndarray:
    """Orthonormal frame (F,3,3), columns x=left, y=up, z=forward, from the hip line and the spine direction."""
    x = left_hip - right_hip
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    up = neck - hips
    y = up - (up * x).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=-1)


def _euler_channels(rot: np.ndarray, channels: list[str]) -> np.ndarray:
    order = "".join(c[0] for c in channels if c.endswith("rotation"))
    return Rotation.from_matrix(rot).as_euler(order, degrees=True)  # intrinsic == BVH left-to-right product


def landmarks_to_bvh(lm_bvh: np.ndarray, template: Bvh, fps: float, out_path: str) -> Bvh:
    """Write a BVH (template hierarchy, template T-pose as frame 0, then one frame per landmark frame)."""
    pos = joint_positions(lm_bvh)
    n = lm_bvh.shape[0]
    t_rot, t_pos = forward_kinematics(template, np.array([0]))
    t_rot, t_pos = t_rot[0], t_pos[0]
    names = template.names
    idx = {nm: i for i, nm in enumerate(names)}

    # world rotation per joint: swing(T-pose segment direction -> observed direction) applied to the T-pose rotation
    world = np.tile(np.eye(3), (n, len(names), 1, 1))
    root = idx["Hips"]
    frame0 = _root_frame(pos["LeftUpLeg"][:1], pos["RightUpLeg"][:1], pos["Hips"][:1], pos["Neck"][:1])[0]
    frame_t = _root_frame(pos["LeftUpLeg"], pos["RightUpLeg"], pos["Hips"], pos["Neck"])
    # the root orientation is relative to the first observed frame, so the person need not start in a T-pose
    world[:, root] = np.einsum("nij,jk,kl->nil", frame_t, frame0.T, t_rot[root])
    for j, nm in enumerate(names):
        if j == root:
            continue
        end = SEGMENT_END.get(nm)
        if end is None or nm.endswith("_End"):
            world[:, j] = world[:, template.parents[j]]
            continue
        d0 = t_pos[idx[end]] - t_pos[j]
        obs = pos[end] - pos[nm]
        for t in range(n):
            # direction relative to the first frame's root frame so a non-T-pose start still maps sensibly
            world[t, j] = min_swing(d0, obs[t]) @ t_rot[j]
    # NOTE: limb directions are absolute (world); only the root uses the first-frame-relative frame above.

    scale = _leg_length(template, t_pos) / max(_leg_length_obs(pos), 1e-6)
    motion = np.zeros((n + 1, template.motion.shape[1]))
    motion[0] = template.motion[0]
    col = 0
    for j, chans in enumerate(template.channels):
        if not chans:
            continue
        p = template.parents[j]
        local = world[:, j] if p < 0 else np.swapaxes(world[:, p], 1, 2) @ world[:, j]
        angles = _euler_channels(local, chans)
        k = 0
        for ch in chans:
            if ch.endswith("position"):
                tr = (pos["Hips"] - pos["Hips"][0]) * scale + t_pos[root]
                motion[1:, col] = tr[:, "XYZ".index(ch[0])]
            else:
                motion[1:, col] = angles[:, k]
                k += 1
            col += 1
    bvh = Bvh(
        names, template.parents, template.offsets, template.channels, motion, 1.0 / fps, template.is_end_site
    )
    write_bvh(bvh, out_path)
    return bvh


def _leg_length(b: Bvh, p: np.ndarray) -> float:
    i = b.index
    return float(
        np.linalg.norm(p[i("LeftUpLeg")] - p[i("LeftLeg")])
        + np.linalg.norm(p[i("LeftLeg")] - p[i("LeftFoot")])
    )


def _leg_length_obs(pos: dict) -> float:
    return float(np.mean(np.linalg.norm(pos["LeftUpLeg"] - pos["LeftLeg"], axis=-1)
                         + np.linalg.norm(pos["LeftLeg"] - pos["LeftFoot"], axis=-1)))  # fmt: skip


def write_bvh(b: Bvh, path: str) -> None:
    lines = ["HIERARCHY"]

    def emit(j: int, depth: int):
        pad = "\t" * depth
        if b.is_end_site[j]:
            lines.append(
                f"{pad}End Site\n{pad}{{\n{pad}\tOFFSET "
                + " ".join(f"{v:.5f}" for v in b.offsets[j])
                + f"\n{pad}}}"
            )
            return
        kind = "ROOT" if b.parents[j] < 0 else "JOINT"
        lines.append(f"{pad}{kind} {b.names[j]}\n{pad}{{")
        lines.append(f"{pad}\tOFFSET " + " ".join(f"{v:.5f}" for v in b.offsets[j]))
        lines.append(f"{pad}\tCHANNELS {len(b.channels[j])} " + " ".join(b.channels[j]))
        for c in b.children(j):
            emit(c, depth + 1)
        lines.append(f"{pad}}}")

    emit(0, 0)
    lines += ["MOTION", f"Frames: {b.n_frames}", f"Frame Time: {b.frame_time:.7f}"]
    lines += [" ".join(f"{v:.5f}" for v in row) for row in b.motion]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def convert_file(landmarks_npy: str, template_bvh: str, fps: float, out_bvh: str) -> Bvh:
    """landmarks_npy: (F, 33, 3) MediaPipe world landmarks as saved by scripts/video_to_pose.py."""
    lm = mediapipe_to_bvh_space(np.load(landmarks_npy))
    return landmarks_to_bvh(lm, parse_bvh(template_bvh), fps, out_bvh)
