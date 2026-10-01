"""Euler decomposition onto the robot's signed joint axes."""

import numpy as np
import pytest

from retarget_lab.kinematics.fk import compose, decompose_tracked, euler_solutions, min_swing, wrap_pi


@pytest.fixture(params=["left_arm", "right_arm"])
def shoulder_axes(request, robot):
    return robot.chains[request.param].axes[:3]


def test_both_euler_branches_reconstruct_the_rotation(shoulder_axes):
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.4, 1.4, (500, 3))
    r = compose(shoulder_axes, q)
    sols = euler_solutions(r, shoulder_axes)
    for k in (0, 1):
        np.testing.assert_allclose(compose(shoulder_axes, sols[:, k]), r, atol=1e-9)


def test_branch_zero_recovers_angles_in_the_principal_range(shoulder_axes):
    rng = np.random.default_rng(1)
    q = np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-1.4, 1.4, 200), rng.uniform(-3, 3, 200)])
    sols = euler_solutions(compose(shoulder_axes, q), shoulder_axes)
    np.testing.assert_allclose(wrap_pi(sols[:, 0] - q), 0, atol=1e-7)


def test_tracked_decomposition_handles_roll_beyond_90_degrees(robot):
    """Roll up to 135 deg needs the alternate branch: the limit-aware picker must find it."""
    axes = robot.chains["left_arm"].axes[:3]
    lim = robot.chains["left_arm"].limits_deg[:3]
    roll = np.deg2rad(np.linspace(0, 130, 60))
    q = np.column_stack([np.deg2rad(np.full(60, 20.0)), roll, np.deg2rad(np.full(60, -10.0))])
    got = decompose_tracked(compose(axes, q), axes, lim)
    np.testing.assert_allclose(got, q, atol=1e-7)


def test_tracked_decomposition_is_continuous_through_gimbal(robot):
    axes = robot.chains["left_arm"].axes[:3]
    lim = robot.chains["left_arm"].limits_deg[:3]
    t = np.linspace(0, 1, 200)
    q = np.column_stack(
        [np.deg2rad(60 * np.sin(2 * np.pi * t)), np.deg2rad(90 * np.ones(200)), np.deg2rad(30 * t)]
    )
    got = decompose_tracked(compose(axes, q), axes, lim)
    assert np.abs(np.diff(got, axis=0)).max() < np.deg2rad(20)  # no 180 deg flips
    np.testing.assert_allclose(compose(axes, got), compose(axes, q), atol=1e-9)


def test_min_swing_maps_direction_and_is_minimal():
    a, b = np.array([0, 0, -1.0]), np.array([0.5, 0, -0.5])
    r = min_swing(a, b)
    np.testing.assert_allclose(r @ a, b / np.linalg.norm(b), atol=1e-12)
    ang = np.degrees(np.arccos((np.trace(r) - 1) / 2))
    assert ang == pytest.approx(45.0, abs=1e-6)
    np.testing.assert_allclose(min_swing(a, a), np.eye(3))
    np.testing.assert_allclose(min_swing(a, -a) @ a, -a, atol=1e-12)
