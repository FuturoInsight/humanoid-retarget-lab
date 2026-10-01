"""End-to-end orchestration: Blender steps (subprocesses) + the pure-Python analysis of every clip.

Blender and the analysis package only exchange files (JSON / NPZ / Parquet), never imports.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import checks as C
from .io import BoneData, bones_npz_to_parquet, load_bones
from .kinematics.fk import hand_positions
from .kinematics.workspace import ReachEnvelope
from .retarget.to_robot import clamp_motion, retarget_ik, retarget_joint_angles
from .skeleton import RobotSkeleton, hierarchy_svg, load_robot, robot_hierarchy_nodes

ROOT = Path(__file__).resolve().parents[2]
RAW_URL = "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/0{subject}/{clip}.bvh"


# ------------------------------------------------------------------------------------------------ context
@dataclass
class Ctx:
    cfg: dict
    robot: RobotSkeleton
    thresholds: dict
    bone_map: dict
    run_dir: Path
    envelopes: dict[str, ReachEnvelope] = field(default_factory=dict)

    def path(self, *parts) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_ctx(config_path: str | Path) -> Ctx:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text())

    def rel(p):
        return ROOT / p

    robot = load_robot(rel(cfg["robot_config"]))
    th = yaml.safe_load(rel(cfg["thresholds"]).read_text())
    bone_map = yaml.safe_load(rel(cfg["bone_map"]).read_text())
    run_dir = ROOT / "outputs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    return Ctx(cfg, robot, th, bone_map, run_dir)


# ------------------------------------------------------------------------------------------------ blender
def run_blender(
    ctx: Ctx, script: str, script_args: list[str], blend: str | Path | None = None, log: str = ""
) -> None:
    exe = ROOT / ctx.cfg["blender"]["exe"]
    env = dict(os.environ, BLENDER_USER_SCRIPTS=str(ROOT / ctx.cfg["blender"]["user_scripts"]))
    cmd = (
        [str(exe), "-b"]
        + ([str(blend)] if blend else [])
        + ["-P", str(ROOT / "scripts" / script), "--"]
        + script_args
    )
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if log:
        (ctx.run_dir / "logs").mkdir(exist_ok=True)
        (ctx.run_dir / "logs" / f"{log}.log").write_text(res.stdout + "\n" + res.stderr)
    bad = res.returncode != 0 or "Traceback" in res.stdout or "Traceback" in res.stderr
    if bad:
        raise RuntimeError(f"Blender step {script} failed:\n{res.stdout[-3000:]}\n{res.stderr[-1500:]}")


def character_blend(ctx: Ctx) -> Path:
    return ROOT / ctx.cfg["character"]["blend"]


def step_character(ctx: Ctx, force: bool = False) -> None:
    blend = character_blend(ctx)
    if blend.exists() and not force:
        return
    blend.parent.mkdir(parents=True, exist_ok=True)
    run_blender(ctx, "create_character.py", ["--character", ctx.cfg["character"]["mblab_character"], "--out", str(blend)],
                log="create_character")  # fmt: skip


def step_fetch_data(ctx: Ctx) -> None:
    for clip in ctx.cfg["clips"]:
        dst = ROOT / clip["bvh"]
        if dst.exists():
            continue
        name = dst.stem  # e.g. 69_72
        url = RAW_URL.format(subject=name.split("_")[0], clip=name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dst)


def step_skeletons(ctx: Ctx) -> dict:
    out = ctx.path("skeletons", "mblab_skeleton.json")
    run_blender(
        ctx, "export_skeleton.py", ["--out", str(out)], blend=character_blend(ctx), log="export_skeleton"
    )
    mb = json.loads(out.read_text())
    svg = hierarchy_svg(
        [(b["name"], b["parent"]) for b in mb["bones"]], f"MB-Lab skeleton hierarchy ({mb['n_bones']} bones)"
    )
    ctx.path("skeletons", "mblab_hierarchy.svg").write_text(svg)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "mblab_hierarchy.svg").write_text(svg)
    rsvg = hierarchy_svg(
        robot_hierarchy_nodes(ctx.robot), f"{ctx.robot.name}: {len(ctx.robot.joint_names)} DoF (generic)"
    )
    ctx.path("skeletons", "robot_hierarchy.svg").write_text(rsvg)
    (ROOT / "docs" / "robot_hierarchy.svg").write_text(rsvg)
    ctx.path("skeletons", "robot_skeleton.json").write_text(json.dumps(ctx.robot.to_json(), indent=1))
    pd.DataFrame(ctx.robot.dof_table()).to_csv(ctx.path("skeletons", "robot_dof_table.csv"), index=False)
    return mb


# ------------------------------------------------------------------------------------------------ per clip
def step_retarget_blender(ctx: Ctx, clip: dict) -> Path:
    """BVH -> MB-Lab (baked .blend) -> per-frame bone transforms (Parquet)."""
    cid = clip["id"]
    bone_map_json = ctx.path("blender", "bone_map.json")
    bone_map_json.write_text(json.dumps(ctx.bone_map))
    blend = ctx.path("blends", f"{cid}.blend")
    run_blender(
        ctx, "retarget_to_mblab.py",
        ["--bvh", str(ROOT / clip["bvh"]), "--bone-map", str(bone_map_json), "--clip-id", cid,
         "--out-blend", str(blend), "--out-meta", str(ctx.path("blender", f"{cid}_retarget.json")),
         "--fps", str(ctx.cfg["fps"]), "--start-sec", str(clip.get("start_sec", 0.0)),
         "--max-sec", str(clip.get("max_sec", ctx.cfg.get("max_sec", 10.0)))],
        blend=character_blend(ctx), log=f"{cid}_retarget",
    )  # fmt: skip
    npz = ctx.path("blender", f"{cid}_bones.npz")
    run_blender(ctx, "export_bone_transforms.py", ["--out", str(npz)], blend=blend, log=f"{cid}_export")
    pq = ctx.path("transforms", f"{cid}_bones.parquet")
    bones_npz_to_parquet(npz, pq)
    return pq


def trajectory_df(robot: RobotSkeleton, q, base_pos, heading, targets, fps) -> pd.DataFrame:
    from .checks import joint_velocity_dps

    n = q.shape[0]
    hands = hand_positions(robot, q)
    cols: dict[str, np.ndarray] = {"frame": np.arange(n), "t": np.arange(n) / fps}
    for i, ax in enumerate("xyz"):
        cols[f"base_{ax}"] = base_pos[:, i]
    cols["heading_deg"] = np.rad2deg(heading)
    qd, vd = np.rad2deg(q), joint_velocity_dps(q, fps)
    for j, name in enumerate(robot.joint_names):
        cols[f"{name}_deg"] = qd[:, j]
    for j, name in enumerate(robot.joint_names):
        cols[f"{name}_vel_dps"] = vd[:, j]
    for side in ("left", "right"):
        for i, ax in enumerate("xyz"):
            cols[f"hand_{side}_{ax}"] = hands[side][:, i]
            cols[f"target_{side}_{ax}"] = targets[side][:, i]
    return pd.DataFrame(cols)


def fk_error_cm(hands, targets) -> float:
    return float(
        np.mean([np.linalg.norm(hands[s] - targets[s], axis=1).mean() for s in ("left", "right")]) * 100
    )


def method_comparison(ctx: Ctx, motion_ja, motion_ik) -> list[dict]:
    """Joint-angle-only vs IK-refined: FK error and joint-limit violations (report section 6)."""
    robot, th = ctx.robot, ctx.thresholds["k2_position_limits"]
    rows = []
    for m in (motion_ja, motion_ik):
        h = hand_positions(robot, m.q)
        qc = clamp_motion(robot, m.q, m.fps)
        hc = hand_positions(robot, qc)
        qd, lim = np.rad2deg(m.q), robot.limits_deg
        over = ((qd < lim[:, 0] - th["tolerance_deg"]) | (qd > lim[:, 1] + th["tolerance_deg"])).any(axis=1)
        maxerr = max(np.linalg.norm(h[s] - m.targets[s], axis=1).max() for s in ("left", "right")) * 100
        rows.append({
            "method": m.method, "fk_err_mean_cm": fk_error_cm(h, m.targets), "fk_err_max_cm": float(maxerr),
            "clamped_fk_err_mean_cm": fk_error_cm(hc, m.targets), "frames_over_limit_share": float(over.mean()),
        })  # fmt: skip
    return rows


def get_envelopes(ctx: Ctx) -> dict[str, ReachEnvelope]:
    if not ctx.envelopes:
        n = int(ctx.cfg.get("workspace_samples", 300_000))
        ctx.envelopes = {s: ReachEnvelope(ctx.robot, s, n=n) for s in ("left", "right")}
    return ctx.envelopes


def analyze_clip(ctx: Ctx, clip: dict, bones: BoneData) -> dict:
    robot, th = ctx.robot, ctx.thresholds
    ja = retarget_joint_angles(bones, robot)
    ik = retarget_ik(robot, ja)
    comparison = method_comparison(ctx, ja, ik)
    chosen = ik if ctx.cfg["retarget_method"] == "ik_refined" else ja
    q_raw = chosen.q
    q_cl = clamp_motion(robot, q_raw, chosen.fps)
    data = C.ClipData(
        clip["id"], chosen.fps, q_raw, q_cl, hand_positions(robot, q_raw), hand_positions(robot, q_cl),
        chosen.targets, bones,
    )  # fmt: skip
    rows = C.run_checks(data, robot, th, get_envelopes(ctx), chosen.method)
    verdict, reasons = C.verdict(rows, th)
    return {
        "motion": chosen, "q_raw": q_raw, "q_clamped": q_cl, "data": data, "rows": rows,
        "verdict": verdict, "reasons": reasons, "comparison": comparison,
        "raw_df": trajectory_df(robot, q_raw, chosen.base_pos, chosen.heading, chosen.targets, chosen.fps),
        "clamped_df": trajectory_df(robot, q_cl, chosen.base_pos, chosen.heading, chosen.targets, chosen.fps),
    }  # fmt: skip


def clip_summary_row(clip: dict, res: dict, robot: RobotSkeleton) -> dict:
    rows = res["rows"]

    def val(cid, joint="*"):
        r = next((r for r in rows if r["check_id"] == cid and r["joint"] == joint), None)
        return r["value"] if r else 0.0

    k1 = [r for r in rows if r["check_id"] == "K1"]
    k2j = sorted((r for r in rows if r["check_id"] == "K2" and r["joint"] != "*"), key=lambda r: -r["value"])
    k3j = sorted((r for r in rows if r["check_id"] == "K3" and r["joint"] != "*"), key=lambda r: -r["value"])
    return {
        "clip_id": clip["id"], "task": clip["task"], "kind": clip.get("kind", ""),
        "frames": res["q_raw"].shape[0], "seconds": res["q_raw"].shape[0] / res["motion"].fps,
        "verdict": res["verdict"], "reasons": " | ".join(res["reasons"]),
        "k1_fk_err_mean_cm": float(np.mean([r["value"] for r in k1])),
        "k2_limit_violation_share": val("K2"), "k3_velocity_violation_share": val("K3"),
        "k4_clamp_distortion_mean_cm": val("K4"), "k5_out_of_reach_share": val("K5"),
        "k6_foot_slide_share": val("K6"), "k7_self_collision_share": val("K7"),
        "k8_jitter_hf_rms_deg": val("K8", next((r["joint"] for r in rows if r["check_id"] == "K8"), "*")),
        "top_position_violators": ", ".join(r["joint"] for r in k2j[:3]),
        "top_velocity_violators": ", ".join(r["joint"] for r in k3j[:3]),
    }  # fmt: skip


def load_clip_bones(ctx: Ctx, cid: str) -> BoneData:
    return load_bones(ctx.path("transforms", f"{cid}_bones.parquet"))


def save_clip_outputs(ctx: Ctx, clip: dict, res: dict) -> None:
    cid = clip["id"]
    res["raw_df"].to_parquet(ctx.path("trajectories", f"{cid}_robot_raw.parquet"), index=False)
    res["clamped_df"].to_parquet(ctx.path("trajectories", f"{cid}_robot_clamped.parquet"), index=False)
    # foot/contact diagnostics for the report
    pd.DataFrame(res["comparison"]).assign(clip_id=cid).to_csv(
        ctx.path("trajectories", f"{cid}_method_comparison.csv"), index=False
    )
