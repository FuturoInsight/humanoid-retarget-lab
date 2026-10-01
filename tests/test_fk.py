"""FK against hand-computed poses of the generic robot (zero pose = arms hanging down)."""

import numpy as np
import pytest

from retarget_lab.kinematics.fk import body_fk, chain_fk, hand_positions


def pose(robot, **deg):
    q = np.zeros((1, len(robot.joint_names)))
    for k, v in deg.items():
        q[0, robot.joint_names.index(k)] = np.deg2rad(v)
    return q


def test_zero_pose_hands_hang_down(robot):
    h = hand_positions(robot, pose(robot))
    # shoulder (0, +-0.20, 0.42) minus upper arm + forearm + hand = 0.66 m
    np.testing.assert_allclose(h["left"][0], [0, 0.20, 0.42 - 0.66], atol=1e-9)
    np.testing.assert_allclose(h["right"][0], [0, -0.20, 0.42 - 0.66], atol=1e-9)


def test_shoulder_flexion_raises_arm_forward(robot):
    h = hand_positions(robot, pose(robot, l_shoulder_pitch=90))
    np.testing.assert_allclose(h["left"][0], [0.66, 0.20, 0.42], atol=1e-9)


def test_shoulder_abduction_is_mirrored(robot):
    h = hand_positions(robot, pose(robot, l_shoulder_roll=90, r_shoulder_roll=90))
    np.testing.assert_allclose(h["left"][0], [0, 0.20 + 0.66, 0.42], atol=1e-9)
    np.testing.assert_allclose(h["right"][0], [0, -0.20 - 0.66, 0.42], atol=1e-9)


def test_elbow_flexion_swings_forearm_forward(robot):
    h = hand_positions(robot, pose(robot, l_elbow_pitch=90))
    # upper arm still down: elbow at z=0.42-0.28=0.14; forearm+hand (0.38) point forward
    np.testing.assert_allclose(h["left"][0], [0.38, 0.20, 0.14], atol=1e-9)


def test_wrist_pitch_only_moves_hand_link(robot):
    h = hand_positions(robot, pose(robot, l_wrist_pitch=90))
    np.testing.assert_allclose(h["left"][0], [0.12, 0.20, 0.42 - 0.54], atol=1e-9)


def test_waist_pitch_tips_the_torso_forward(robot):
    fk = body_fk(robot, pose(robot, waist_pitch=90))
    shoulder = fk["left_arm"][0, 0, :3, 3]
    np.testing.assert_allclose(shoulder, [0.42, 0.20, 0.0], atol=1e-9)


def test_waist_yaw_turns_arms_with_torso(robot):
    h = hand_positions(robot, pose(robot, waist_yaw=90))
    # rotate (0, 0.20, -0.24) about z by 90 deg -> (-0.20, 0, -0.24)
    np.testing.assert_allclose(h["left"][0], [-0.20, 0.0, -0.24], atol=1e-9)


def test_knee_flexion_moves_foot_backward(robot):
    fk = body_fk(robot, pose(robot, l_knee_pitch=90))
    sole = fk["left_leg"][0, -1, :3, 3]
    # hip at (0,0.1,0); thigh straight down 0.40 -> knee at z=-0.40; shank+ankle (0.48) swing backward (-x)
    np.testing.assert_allclose(sole, [-0.48, 0.10, -0.40], atol=1e-9)


def test_chain_fk_is_batched_and_consistent(robot):
    rng = np.random.default_rng(1)
    q = rng.uniform(-1, 1, (5, len(robot.joint_names)))
    batch = hand_positions(robot, q)["left"]
    for i in range(5):
        np.testing.assert_allclose(hand_positions(robot, q[i : i + 1])["left"][0], batch[i], atol=1e-12)


def test_link_lengths_are_preserved_for_any_pose(robot):
    rng = np.random.default_rng(2)
    q = rng.uniform(-1, 1, (20, len(robot.joint_names)))
    fk = chain_fk(robot.chains["left_arm"], q[:, robot.chain_slice("left_arm")])
    # shoulder-yaw frame origin is the shoulder; elbow joint (frame index 3) is upper_arm away
    d = np.linalg.norm(fk[:, 3, :3, 3] - fk[:, 0, :3, 3], axis=1)
    np.testing.assert_allclose(d, robot.dims["upper_arm"], atol=1e-9)


@pytest.mark.parametrize("chain", ["left_arm", "right_arm", "left_leg", "right_leg"])
def test_limits_are_ordered(robot, chain):
    lim = robot.chains[chain].limits_deg
    assert (lim[:, 0] < lim[:, 1]).all()
