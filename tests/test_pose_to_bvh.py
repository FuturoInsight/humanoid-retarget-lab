"""Round trip: CMU joints -> fake MediaPipe landmarks -> BVH -> FK reproduces the joint directions."""

from pathlib import Path

import numpy as np
import pytest

from retarget_lab import pose_to_bvh as P
from retarget_lab.bvh import forward_kinematics, parse_bvh

CLIP = Path(__file__).resolve().parent.parent / "data" / "raw" / "69_72.bvh"
pytestmark = pytest.mark.skipif(
    not CLIP.exists(), reason="CMU clip not downloaded (python -m retarget_lab fetch-data)"
)


def fake_landmarks(b, frames):
    _, p = forward_kinematics(b, frames)
    j = b.index
    lm = np.zeros((len(frames), 33, 3))
    m = {
        P.L_SH: "LeftArm", P.R_SH: "RightArm", P.L_EL: "LeftForeArm", P.R_EL: "RightForeArm", P.L_WR: "LeftHand",
        P.R_WR: "RightHand", P.L_IDX: "LeftHandIndex1", P.R_IDX: "RightHandIndex1", P.L_HIP: "LeftUpLeg",
        P.R_HIP: "RightUpLeg", P.L_KN: "LeftLeg", P.R_KN: "RightLeg", P.L_AN: "LeftFoot", P.R_AN: "RightFoot",
        P.L_FOOT: "LeftToeBase", P.R_FOOT: "RightToeBase", P.L_EAR: "Head", P.R_EAR: "Head", P.NOSE: "Head",
    }  # fmt: skip
    for k, name in m.items():
        lm[:, k] = p[:, j(name)]
    return lm


def test_roundtrip_reproduces_limb_directions(tmp_path):
    template = parse_bvh(str(CLIP))
    frames = np.array([200, 400, 700, 900])
    lm = fake_landmarks(template, frames)
    out = tmp_path / "own.bvh"
    b = P.landmarks_to_bvh(lm, template, 30.0, str(out))
    assert b.n_frames == len(frames) + 1 and out.exists()
    # reload from disk to check the writer too
    b2 = parse_bvh(str(out))
    _, p = forward_kinematics(b2, np.arange(1, len(frames) + 1))
    j = b2.index
    for a, c in [
        ("LeftArm", "LeftForeArm"),
        ("LeftForeArm", "LeftHand"),
        ("RightLeg", "RightFoot"),
        ("LeftUpLeg", "LeftLeg"),
    ]:
        got = p[:, j(c)] - p[:, j(a)]
        want = (
            lm[:, {"LeftForeArm": P.L_EL, "LeftHand": P.L_WR, "RightFoot": P.R_AN, "LeftLeg": P.L_KN}[c]]
            - lm[:, {"LeftArm": P.L_SH, "LeftForeArm": P.L_EL, "RightLeg": P.R_KN, "LeftUpLeg": P.L_HIP}[a]]
        )
        cos = (got * want).sum(-1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1))
        assert cos.min() > 0.995, (a, c, cos)


def test_mediapipe_axes_convention():
    lm = np.zeros((1, 33, 3))
    lm[0, 0] = [0.1, -0.5, -0.2]  # right of image, above the origin (y up is negative), toward the camera
    np.testing.assert_allclose(P.mediapipe_to_bvh_space(lm)[0, 0], [0.1, 0.5, 0.2])
