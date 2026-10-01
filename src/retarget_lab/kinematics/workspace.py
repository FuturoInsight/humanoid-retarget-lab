"""Reach envelope of a robot arm by Monte Carlo sampling over joint limits (check K5)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..skeleton import RobotSkeleton
from .fk import body_fk


class ReachEnvelope:
    """Samples tool-point positions (base frame) reachable by waist + one arm within joint limits."""

    def __init__(self, robot: RobotSkeleton, side: str, n: int = 300_000, seed: int = 0):
        rng = np.random.default_rng(seed)
        names = robot.joint_names
        lim = np.deg2rad(robot.limits_deg)
        q = np.zeros((n, len(names)))
        active = [robot.chain_slice("waist"), robot.chain_slice(f"{side}_arm")]
        for sl in active:
            # Beta(0.5, 0.5) piles samples near the limits, where the extremes of the workspace live; plain uniform
            # sampling would almost never produce a fully extended arm and under-estimate the envelope.
            u = rng.beta(0.5, 0.5, size=(n, sl.stop - sl.start))
            q[:, sl] = lim[sl, 0] + u * (lim[sl, 1] - lim[sl, 0])
        self.side = side
        self.points = body_fk(robot, q)[f"{side}_arm"][:, -1, :3, 3]
        self.tree = cKDTree(self.points)

    def distance(self, targets: np.ndarray) -> np.ndarray:
        """Distance (m) from each target to the nearest reachable sample."""
        return self.tree.query(targets)[0]

    def outside(self, targets: np.ndarray, tol_m: float) -> np.ndarray:
        return self.distance(targets) > tol_m

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)
