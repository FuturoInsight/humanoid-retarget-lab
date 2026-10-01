import numpy as np

from retarget_lab.kinematics.fk import body_fk, hand_positions
from retarget_lab.retarget.ik import solve_arm_ik
from retarget_lab.retarget.to_robot import clamp_motion


def test_ik_reaches_reachable_targets(robot):
    rng = np.random.default_rng(3)
    n = 40
    lim = np.deg2rad(robot.chains["left_arm"].limits_deg)
    q_true = np.zeros((n, len(robot.joint_names)))
    sl = robot.chain_slice("left_arm")
    q_true[:, sl] = lim[:, 0] + rng.uniform(0.15, 0.85, (n, 6)) * (lim[:, 1] - lim[:, 0])
    target = hand_positions(robot, q_true)["left"]
    torso = body_fk(robot, np.zeros_like(q_true))["waist"][:, -1]
    q0 = np.zeros((n, 6))
    q = solve_arm_ik(robot.chains["left_arm"], torso, q0, target, iters=80, stay=1e-4)
    q_full = np.zeros_like(q_true)
    q_full[:, sl] = q
    err = np.linalg.norm(hand_positions(robot, q_full)["left"] - target, axis=1)
    assert np.median(err) < 1e-3 and err.max() < 5e-3


def test_ik_respects_limits_when_asked(robot):
    n = 5
    torso = body_fk(robot, np.zeros((n, len(robot.joint_names))))["waist"][:, -1]
    target = np.tile([0.3, 0.2, 0.2], (n, 1))
    ch = robot.chains["left_arm"]
    q = solve_arm_ik(ch, torso, np.zeros((n, 6)), target, iters=40, limits_deg=ch.limits_deg)
    lim = np.deg2rad(ch.limits_deg)
    assert (q >= lim[:, 0] - 1e-9).all() and (q <= lim[:, 1] + 1e-9).all()


def test_ik_regularisation_keeps_solution_near_initial_guess(robot):
    n = 3
    torso = body_fk(robot, np.zeros((n, len(robot.joint_names))))["waist"][:, -1]
    q0 = np.tile(np.deg2rad([20, 10, 0, 40, 0, 0]), (n, 1))
    target = hand_positions(robot, np.pad(q0, ((0, 0), (4, 18))))["left"]  # already satisfied
    q = solve_arm_ik(robot.chains["left_arm"], torso, q0, target)
    np.testing.assert_allclose(q, q0, atol=1e-6)


def test_clamp_motion_enforces_position_and_velocity(robot):
    fps = 30.0
    q = np.zeros((60, len(robot.joint_names)))
    j = robot.joint_names.index("l_elbow_pitch")
    q[:, j] = np.deg2rad(200.0)  # far beyond the 135 deg limit, instantly
    out = clamp_motion(robot, q, fps)
    assert np.rad2deg(out[:, j]).max() <= 135.0 + 1e-9
    step = np.rad2deg(np.abs(np.diff(out[:, j]))) * fps
    assert step.max() <= robot.max_vel_dps[j] + 1e-6
    assert np.rad2deg(out[-1, j]) == 135.0  # eventually arrives at the limit
