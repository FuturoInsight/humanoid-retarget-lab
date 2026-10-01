import numpy as np
import pytest
import yaml

from retarget_lab import checks as C
from retarget_lab.kinematics.fk import hand_positions
from retarget_lab.kinematics.workspace import ReachEnvelope
from retarget_lab.retarget.to_robot import clamp_motion

FPS = 30.0


@pytest.fixture(scope="module")
def th():
    with open("config/thresholds.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def envelopes(robot):
    return {s: ReachEnvelope(robot, s, n=60_000) for s in ("left", "right")}


def make_clip(robot, q, targets=None, cid="t"):
    clamped = clamp_motion(robot, q, FPS)
    raw_h, cl_h = hand_positions(robot, q), hand_positions(robot, clamped)
    return C.ClipData(cid, FPS, q, clamped, raw_h, cl_h, targets or raw_h)


def zeros(robot, n=90):
    return np.zeros((n, len(robot.joint_names)))


def test_segments_and_longest():
    m = np.array([0, 1, 1, 0, 1, 1, 1, 0], bool)
    assert C.segments(m) == [(1, 2), (4, 6)]
    assert C.longest(m) == (4, 6)
    assert C.segments(np.zeros(4, bool)) == []


def test_k2_flags_the_offending_joint_and_share(robot, th):
    q = zeros(robot)
    j = robot.joint_names.index("l_elbow_pitch")
    q[45:, j] = np.deg2rad(150)  # limit is 135
    rows = C.k2_position_limits(make_clip(robot, q), robot, th)
    clip_row = rows[0]
    assert clip_row["value"] == pytest.approx(0.5) and clip_row["severity"] == "fail"
    assert clip_row["frame_range"] == "45-89"
    assert [r["joint"] for r in rows[1:]] == ["l_elbow_pitch"]


def test_k2_clean_motion_is_ok(robot, th):
    rows = C.k2_position_limits(make_clip(robot, zeros(robot)), robot, th)
    assert rows[0]["severity"] == "ok" and len(rows) == 1


def test_k3_flags_a_too_fast_joint(robot, th):
    q = zeros(robot)
    j = robot.joint_names.index("l_wrist_pitch")
    q[:, j] = np.deg2rad(40) * np.sin(2 * np.pi * 3.0 * np.arange(90) / FPS)  # 3 Hz, 40 deg: ~750 deg/s peak
    rows = C.k3_velocity_limits(make_clip(robot, q), robot, th)
    assert rows[0]["severity"] in ("warn", "fail") and rows[1]["joint"] == "l_wrist_pitch"


def test_k4_is_zero_when_nothing_is_clamped_and_large_when_it_is(robot, th):
    assert C.k4_clamp_distortion(make_clip(robot, zeros(robot)), th)[0]["value"] == pytest.approx(0, abs=1e-9)
    q = zeros(robot)
    q[:, robot.joint_names.index("l_shoulder_pitch")] = np.deg2rad(179)  # limit 165 -> small distortion
    q[:, robot.joint_names.index("l_shoulder_roll")] = np.deg2rad(179)  # limit 135 -> big distortion
    r = C.k4_clamp_distortion(make_clip(robot, q), th)[0]
    assert r["value"] > 3.0


def test_k5_reach(robot, th, envelopes):
    q = zeros(robot)
    hands = hand_positions(robot, q)
    reachable = dict(hands)  # neutral hand position is inside the envelope
    far = {"left": hands["left"] + [2.0, 0, 0], "right": hands["right"] + [2.0, 0, 0]}
    ok = C.k5_reach(make_clip(robot, q, reachable), envelopes, th)
    bad = C.k5_reach(make_clip(robot, q, far), envelopes, th)
    assert ok[0]["value"] == 0 and bad[0]["value"] == 1 and bad[0]["severity"] == "fail"


def test_k7_detects_arm_across_the_torso_but_not_neutral(robot, th):
    assert C.k7_self_collision(make_clip(robot, zeros(robot)), robot, th)[0]["value"] == 0
    pose = zeros(robot, 1)
    pose[0, robot.joint_names.index("l_shoulder_roll")] = -np.pi / 2  # raw pose: arm swung straight across
    assert C.self_collision_mask(robot, pose)["left"][0]


def test_k8_jitter_separates_noise_from_smooth_motion(robot, th):
    t = np.arange(150) / FPS
    smooth = zeros(robot, 150)
    smooth[:, 5] = np.deg2rad(30) * np.sin(2 * np.pi * 0.5 * t)
    noisy = smooth.copy()
    noisy[:, 5] += np.deg2rad(2.0) * np.sin(2 * np.pi * 13.0 * t)
    assert C.k8_jitter(make_clip(robot, smooth), robot, th)[0]["severity"] == "ok"
    r = C.k8_jitter(make_clip(robot, noisy), robot, th)[0]
    assert r["severity"] == "warn" and r["joint"] == robot.joint_names[5]


def test_k1_reports_fk_error_in_cm(robot, th):
    q = zeros(robot)
    hands = hand_positions(robot, q)
    shifted = {s: v + [0.05, 0, 0] for s, v in hands.items()}
    rows = C.k1_fk_error(make_clip(robot, q, shifted), th, "joint_angle")
    assert rows[0]["value"] == pytest.approx(5.0) and rows[0]["severity"] == "warn"


def test_verdict_rules(th):
    def r(cid, sev):
        return C.row(cid, "t", sev, 0.5, 0.02, f"{cid} message")

    assert C.verdict([r("K2", "ok"), r("K5", "ok")], th)[0] == "USABLE"
    assert C.verdict([r("K2", "warn")], th)[0] == "USABLE WITH CLAMPING"
    v, reasons = C.verdict([r("K2", "warn"), r("K5", "fail")], th)
    assert v == "NOT FEASIBLE" and reasons == ["K5: K5 message"]
