"""Minimal BVH reader + forward kinematics (numpy only).

This module is deliberately numpy-only so Blender's bundled Python can import it
(`scripts/retarget_to_mblab.py` does) while the normal test environment can unit-test it.

Conventions (see docs/conventions.md):
  * BVH space: right-handed, Y up, character faces +Z, +X is the character's left.
  * Blender space: Z up, character faces -Y, +X is the character's left.
  * `BVH_TO_BLENDER` converts the former into the latter. Lengths stay in BVH units
    (CMU "units" are not metres); callers scale by a leg-length ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# x' = x, y' = -z, z' = y  (proper rotation, det = +1)
BVH_TO_BLENDER = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


@dataclass
class Bvh:
    names: list[str]
    parents: list[int]  # -1 for the root
    offsets: np.ndarray  # (J, 3) rest offsets from parent, BVH units
    channels: list[list[str]]  # per joint channel names (End Sites have none)
    motion: np.ndarray  # (F, C) raw channel values (degrees / units)
    frame_time: float
    is_end_site: list[bool] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time

    @property
    def n_frames(self) -> int:
        return self.motion.shape[0]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def children(self, j: int) -> list[int]:
        return [i for i, p in enumerate(self.parents) if p == j]


def parse_bvh(path: str) -> Bvh:
    with open(path, encoding="utf-8", errors="replace") as fh:
        tokens = fh.read().replace("\t", " ").split("\n")
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    is_end: list[bool] = []
    stack: list[int] = []
    i = 0
    while i < len(tokens):
        line = tokens[i].strip()
        i += 1
        if not line:
            continue
        parts = line.split()
        key = parts[0]
        if key in ("ROOT", "JOINT"):
            names.append(parts[1])
            parents.append(stack[-1] if stack else -1)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            is_end.append(False)
            stack.append(len(names) - 1)
        elif line.startswith("End Site"):
            parent = stack[-1]
            names.append(names[parent] + "_End")
            parents.append(parent)
            offsets.append([0.0, 0.0, 0.0])
            channels.append([])
            is_end.append(True)
            stack.append(len(names) - 1)
        elif key == "OFFSET":
            offsets[stack[-1]] = [float(v) for v in parts[1:4]]
        elif key == "CHANNELS":
            channels[stack[-1]] = parts[2:]
        elif key == "}":
            stack.pop()
        elif key == "MOTION":
            break
    n_frames = int(tokens[i].split()[1])
    frame_time = float(tokens[i + 1].split()[2])
    data = np.array([[float(v) for v in tokens[i + 2 + f].split()] for f in range(n_frames)])
    return Bvh(names, parents, np.array(offsets), channels, data, frame_time, is_end)


def _rot(axis: str, deg: np.ndarray) -> np.ndarray:
    """Elementary rotation matrices, shape (F, 3, 3), angles in degrees."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    one, zero = np.ones_like(a), np.zeros_like(a)
    if axis == "X":
        m = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    elif axis == "Y":
        m = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    else:
        m = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
    return np.moveaxis(np.array(m), (0, 1), (-2, -1))


def forward_kinematics(bvh: Bvh, frames: np.ndarray | None = None):
    """World rotations (F, J, 3, 3) and positions (F, J, 3) in BVH space.

    BVH applies the listed rotation channels left to right, i.e. R = R_a1 @ R_a2 @ R_a3.
    """
    motion = bvh.motion if frames is None else bvh.motion[frames]
    n_f, n_j = motion.shape[0], len(bvh.names)
    local_r = np.tile(np.eye(3), (n_f, n_j, 1, 1))
    root_pos = np.zeros((n_f, 3))
    col = 0
    for j, chans in enumerate(bvh.channels):
        rot = np.tile(np.eye(3), (n_f, 1, 1))
        for ch in chans:
            v = motion[:, col]
            col += 1
            if ch.endswith("position"):
                root_pos[:, "XYZ".index(ch[0])] = v
            else:
                rot = rot @ _rot(ch[0], v)
        local_r[:, j] = rot
    world_r = np.zeros_like(local_r)
    world_p = np.zeros((n_f, n_j, 3))
    for j in range(n_j):
        p = bvh.parents[j]
        if p < 0:
            world_r[:, j] = local_r[:, j]
            world_p[:, j] = root_pos + bvh.offsets[j]
        else:
            world_r[:, j] = world_r[:, p] @ local_r[:, j]
            world_p[:, j] = world_p[:, p] + np.einsum("fij,j->fi", world_r[:, p], bvh.offsets[j])
    return world_r, world_p


def to_blender_space(world_r: np.ndarray, world_p: np.ndarray):
    """Re-express BVH-space rotations/positions in Blender space (Z up, faces -Y)."""
    m = BVH_TO_BLENDER
    return m @ world_r @ m.T, world_p @ m.T
