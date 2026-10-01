"""Robot skeleton definition (loaded from config/robot_skeleton.yaml) and hierarchy-diagram helpers."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

CHAIN_ORDER = ["waist", "head", "left_arm", "right_arm", "left_leg", "right_leg"]


@dataclass(frozen=True)
class Joint:
    name: str
    axis: np.ndarray  # unit vector in the parent frame
    lo_deg: float
    hi_deg: float
    max_vel_dps: float
    offset_after: np.ndarray  # translation that follows this joint, in its rotated frame (m)


@dataclass
class Chain:
    name: str
    parent: str  # "base" or "torso"
    origin: np.ndarray
    joints: list[Joint]

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def limits_deg(self) -> np.ndarray:
        return np.array([[j.lo_deg, j.hi_deg] for j in self.joints])

    @property
    def axes(self) -> np.ndarray:
        return np.array([j.axis for j in self.joints])

    @property
    def offsets(self) -> np.ndarray:
        return np.array([j.offset_after for j in self.joints])

    @property
    def max_vel(self) -> np.ndarray:
        return np.array([j.max_vel_dps for j in self.joints])


@dataclass
class RobotSkeleton:
    name: str
    dims: dict
    collision: dict
    chains: dict[str, Chain]
    human_source: dict
    hip_height_m: float
    height_m: float
    raw: dict = field(default_factory=dict)

    @property
    def joint_names(self) -> list[str]:
        """Canonical joint order used for every joint-angle array: waist, head, arms, legs."""
        return [n for c in CHAIN_ORDER for n in self.chains[c].joint_names]

    def joint(self, name: str) -> Joint:
        for c in self.chains.values():
            for j in c.joints:
                if j.name == name:
                    return j
        raise KeyError(name)

    def chain_slice(self, chain: str) -> slice:
        names = self.joint_names
        first = names.index(self.chains[chain].joints[0].name)
        return slice(first, first + len(self.chains[chain].joints))

    @property
    def limits_deg(self) -> np.ndarray:
        return np.array([[self.joint(n).lo_deg, self.joint(n).hi_deg] for n in self.joint_names])

    @property
    def max_vel_dps(self) -> np.ndarray:
        return np.array([self.joint(n).max_vel_dps for n in self.joint_names])

    def arm_length(self) -> float:
        return float(self.dims["upper_arm"] + self.dims["forearm"])

    def dof_table(self) -> list[dict]:
        rows = []
        for cname in CHAIN_ORDER:
            for j in self.chains[cname].joints:
                rows.append(
                    {
                        "chain": cname,
                        "joint": j.name,
                        "axis": " ".join(f"{v:+.0f}" for v in j.axis),
                        "lo_deg": j.lo_deg,
                        "hi_deg": j.hi_deg,
                        "max_vel_dps": j.max_vel_dps,
                    }
                )
        return rows

    def to_json(self) -> dict:
        """Plain-JSON description for the Blender scripts (Blender's Python has no PyYAML)."""
        return {
            "name": self.name,
            "dims": self.dims,
            "joint_names": self.joint_names,
            "chains": {
                c: {
                    "parent": ch.parent,
                    "origin": ch.origin.tolist(),
                    "joints": [
                        {
                            "name": j.name,
                            "axis": j.axis.tolist(),
                            "limits_deg": [j.lo_deg, j.hi_deg],
                            "max_vel_dps": j.max_vel_dps,
                            "offset_after": j.offset_after.tolist(),
                        }
                        for j in ch.joints
                    ],
                }
                for c, ch in self.chains.items()
            },
        }


def _mk_joint(d: dict) -> Joint:
    axis = np.array(d["axis"], float)
    return Joint(
        d["name"],
        axis / np.linalg.norm(axis),
        float(d["limits_deg"][0]),
        float(d["limits_deg"][1]),
        float(d["max_vel_dps"]),
        np.array(d["offset_after"], float),
    )


def _mirror(src: Chain, old: str, new: str, name: str) -> Chain:
    """Mirror across the sagittal plane: y-axis (pitch) joints keep their axis, x/z axes flip."""
    flip_axis = np.array([-1.0, 1.0, -1.0])
    flip_pos = np.array([1.0, -1.0, 1.0])
    joints = [
        Joint(
            j.name.replace(old, new, 1),
            j.axis * flip_axis,
            j.lo_deg,
            j.hi_deg,
            j.max_vel_dps,
            j.offset_after * flip_pos,
        )
        for j in src.joints
    ]
    return Chain(name, src.parent, src.origin * flip_pos, joints)


def load_robot(path: str | Path) -> RobotSkeleton:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    chains: dict[str, Chain] = {}
    for name, c in raw["chains"].items():
        if "mirror_of" in c:
            chains[name] = _mirror(chains[c["mirror_of"]], c["rename"][0], c["rename"][1], name)
        else:
            chains[name] = Chain(
                name, c["parent"], np.array(c["origin"], float), [_mk_joint(j) for j in c["joints"]]
            )
    return RobotSkeleton(
        raw["name"],
        raw["dims"],
        raw["collision"],
        chains,
        raw["human_source"],
        raw["hip_height_m"],
        raw["height_m"],
        raw,
    )


# --------------------------------------------------------------------------- hierarchy diagrams
def _tree_layout(children: dict, root: str):
    """Horizontal tree: x = depth, y = leaf order (parents centred over their children)."""
    pos: dict[str, tuple[int, float]] = {}
    counter = [0]

    def rec(n: str, depth: int) -> float:
        kids = children.get(n, [])
        if not kids:
            y = float(counter[0])
            counter[0] += 1
        else:
            ys = [rec(k, depth + 1) for k in kids]
            y = (ys[0] + ys[-1]) / 2
        pos[n] = (depth, y)
        return y

    rec(root, 0)
    return pos


def hierarchy_svg(nodes: list[tuple[str, str | None]], title: str, highlight: set[str] | None = None) -> str:
    """Render a parent->child hierarchy as a standalone SVG (pure Python, no Graphviz needed).

    nodes: [(name, parent_or_None), ...] in any order that lists parents before children.
    """
    highlight = highlight or set()
    children: dict[str, list[str]] = {}
    roots = []
    for n, p in nodes:
        if p is None:
            roots.append(n)
        else:
            children.setdefault(p, []).append(n)
    pos: dict[str, tuple[int, float]] = {}
    offset = 0.0
    for r in roots:
        sub = _tree_layout(children, r)
        for k, (d, y) in sub.items():
            pos[k] = (d, y + offset)
        offset += max(y for _, y in sub.values()) + 1
    col_w, row_h, pad = 150, 16, 20
    width = (max(d for d, _ in pos.values()) + 1) * col_w + 2 * pad
    height = int((max(y for _, y in pos.values()) + 1) * row_h + 2 * pad + 24)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Consolas, monospace" font-size="11">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{pad}" y="16" font-size="13" font-weight="bold">{html.escape(title)}</text>',
    ]

    def xy(n):
        d, y = pos[n]
        return pad + d * col_w, pad + 24 + y * row_h

    for n, p in nodes:
        if p is not None:
            x1, y1 = xy(p)
            x2, y2 = xy(n)
            parts.append(
                f'<path d="M{x1 + 6},{y1} C{x1 + 40},{y1} {x2 - 40},{y2} {x2 - 6},{y2}" fill="none" '
                f'stroke="#9aa5b1" stroke-width="1"/>'
            )
    for n, _ in nodes:
        x, y = xy(n)
        fill = "#ff6a00" if n in highlight else "#2f3a46"
        parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{fill}"/>')
        parts.append(f'<text x="{x + 7}" y="{y + 4}" fill="#1b232c">{html.escape(n)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def robot_hierarchy_nodes(robot: RobotSkeleton) -> list[tuple[str, str | None]]:
    nodes: list[tuple[str, str | None]] = [("base (pelvis)", None)]
    for cname in CHAIN_ORDER:
        ch = robot.chains[cname]
        parent = "base (pelvis)" if ch.parent == "base" else "waist_pitch"
        for j in ch.joints:
            nodes.append((j.name, parent))
            parent = j.name
    return nodes
