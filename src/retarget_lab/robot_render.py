"""Robot link geometry for rendering: per-frame link endpoints and joint-limit highlight flags.

The Blender side (scripts/build_robot_armature.py, scripts/render_side_by_side.py) only needs, per link, a start
and end point in Blender world coordinates plus a boolean "highlight" per frame, so all kinematics stays here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics.fk import body_fk, rot_z
from .retarget.to_robot import F_BR
from .skeleton import RobotSkeleton

FOOT_LENGTH = 0.16
LINK_RADIUS = {"spine": 0.07, "shoulder_bar": 0.04, "hip_bar": 0.04, "head": 0.06, "foot": 0.035}


@dataclass
class Link:
    name: str
    radius: float
    joints: list[str]  # joints whose limit violation highlights this link
    chain: str
    start: tuple[str, int]  # (chain, frame index in that chain's FK frames)
    end: tuple[str, int] | None = None
    extra: np.ndarray | None = None  # local offset from the start frame (for the foot)


def link_defs(robot: RobotSkeleton) -> list[Link]:
    links = [
        Link(
            "spine",
            LINK_RADIUS["spine"],
            robot.chains["waist"].joint_names,
            "waist",
            ("waist", 0),
            ("head", 0),
        ),
        Link("shoulder_bar", LINK_RADIUS["shoulder_bar"], [], "waist", ("left_arm", 0), ("right_arm", 0)),
        Link("hip_bar", LINK_RADIUS["hip_bar"], [], "base", ("left_leg", 0), ("right_leg", 0)),
    ]
    for cname in ("head", "left_arm", "right_arm", "left_leg", "right_leg"):
        ch = robot.chains[cname]
        group: list[int] = []
        part = 0
        for i, j in enumerate(ch.joints):
            group.append(i)
            if np.linalg.norm(j.offset_after) > 1e-9:  # a group ends where a physical link follows
                tag = {0: "upper", 1: "lower", 2: "end"}[min(part, 2)]
                name = f"{cname}_{tag}" if cname != "head" else "head"
                radius = LINK_RADIUS["head"] if cname == "head" else 0.04 if "arm" in cname else 0.055
                if tag == "end":
                    radius *= 0.8
                links.append(
                    Link(
                        name,
                        radius,
                        [ch.joints[g].name for g in group],
                        cname,
                        (cname, group[0]),
                        (cname, i + 1),
                    )
                )
                group, part = [], part + 1
        if cname.endswith("_leg"):
            links.append(
                Link(f"{cname}_foot", LINK_RADIUS["foot"], ch.joint_names[-2:], cname, (cname, len(ch.joints)), None,
                     np.array([FOOT_LENGTH, 0.0, 0.0]))
            )  # fmt: skip
    return links


def world_endpoints(
    robot: RobotSkeleton,
    q: np.ndarray,
    base_pos: np.ndarray,
    heading: np.ndarray,
    links: list[Link] | None = None,
) -> np.ndarray:
    """(N, L, 2, 3) link start/end points in Blender world coordinates (metres, Z up, robot faces -Y at rest)."""
    links = links or link_defs(robot)
    fk = body_fk(robot, q)
    h = rot_z(heading)
    out = np.zeros((q.shape[0], len(links), 2, 3))
    for k, lk in enumerate(links):
        fs = fk[lk.start[0]][:, lk.start[1]]
        p0 = fs[:, :3, 3]
        if lk.end is not None:
            p1 = fk[lk.end[0]][:, lk.end[1], :3, 3]
        else:
            p1 = p0 + np.einsum("nij,j->ni", fs[:, :3, :3], lk.extra)
        for s, p in enumerate((p0, p1)):
            pw = np.einsum("nij,nj->ni", h, p) + base_pos
            out[:, k, s] = pw @ F_BR  # robot -> Blender: (F_BR^T p) written as p @ F_BR
    return out


def highlight_flags(
    robot: RobotSkeleton, q: np.ndarray, links: list[Link], tol_deg: float = 1.0
) -> np.ndarray:
    """(N, L) True where any joint driving the link is beyond its position limit."""
    qd = np.rad2deg(q)
    lim = robot.limits_deg
    over = (qd < lim[:, 0] - tol_deg) | (qd > lim[:, 1] + tol_deg)
    names = robot.joint_names
    flags = np.zeros((q.shape[0], len(links)), bool)
    for k, lk in enumerate(links):
        for j in lk.joints:
            flags[:, k] |= over[:, names.index(j)]
    return flags


def render_payload(
    robot: RobotSkeleton, q: np.ndarray, base_pos: np.ndarray, heading: np.ndarray, q_flag: np.ndarray
):
    """Arrays for the Blender render: names, radii, rest endpoints (zero pose), per-frame endpoints, highlights."""
    links = link_defs(robot)
    rest = world_endpoints(robot, np.zeros((1, len(robot.joint_names))), np.array([[0.0, 0.0, robot.hip_height_m]]),
                           np.zeros(1), links)[0]  # fmt: skip
    return {
        "names": np.array([lk.name for lk in links]),
        "radius": np.array([lk.radius for lk in links]),
        "rest": rest,
        "endpoints": world_endpoints(robot, q, base_pos, heading, links),
        "highlight": highlight_flags(robot, q_flag, links),
    }
