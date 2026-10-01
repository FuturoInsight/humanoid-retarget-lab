"""Retargeting quality report (report.html + report.md) with matplotlib figures.

All numbers in the text come from the run's own data; nothing is typed in by hand.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from . import checks as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
INK, GRID = "#1f2933", "#d9dee4"
ORANGE, STEEL, BLUE, GREEN, AMBER = "#ff5c00", "#6b7785", "#2f6fb3", "#2e9b5f", "#e0a100"
VERDICT_CLASS = {"USABLE": "ok", "USABLE WITH CLAMPING": "warn", "NOT FEASIBLE": "fail"}
DEFAULT_PLOT_JOINTS = ["waist_pitch", "waist_yaw", "l_shoulder_pitch", "r_shoulder_pitch", "l_shoulder_roll",
                       "r_shoulder_roll", "l_elbow_pitch", "r_elbow_pitch"]  # fmt: skip


def _style():
    plt.rcParams.update({
        "font.size": 8.5, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "figure.dpi": 130, "savefig.bbox": "tight", "text.color": INK,
    })  # fmt: skip


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


# ------------------------------------------------------------------------------------------------ figures
def fig_joint_angles(ctx, cid: str, res: dict, out: Path) -> Path:
    robot = ctx.robot
    qd = np.rad2deg(res["q_raw"])
    lim = robot.limits_deg
    names = robot.joint_names
    over = (qd < lim[:, 0] - 1) | (qd > lim[:, 1] + 1)
    ranked = [names[j] for j in np.argsort(-over.mean(axis=0)) if over[:, j].any()]
    chosen = (ranked + [j for j in DEFAULT_PLOT_JOINTS if j not in ranked])[:8]
    t = np.arange(qd.shape[0]) / res["motion"].fps
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 7.2), sharex=True)
    for ax, name in zip(axes.ravel(), chosen, strict=False):
        j = names.index(name)
        ax.axhspan(lim[j, 0], lim[j, 1], color=GREEN, alpha=0.10, lw=0)
        ax.axhline(lim[j, 0], color=GREEN, lw=0.8)
        ax.axhline(lim[j, 1], color=GREEN, lw=0.8)
        ax.plot(t, qd[:, j], color=STEEL, lw=1.0, label="raw (what the human did)")
        qc = np.rad2deg(res["q_clamped"][:, j])
        ax.plot(t, qc, color=BLUE, lw=1.0, label="clamped (what the robot can do)")
        ax.plot(t[over[:, j]], qd[over[:, j], j], ".", color=ORANGE, ms=3, label="past limit")
        ax.set_title(name, loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("deg")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{cid}: joint angles with limit band shaded green", y=1.06, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_reach(ctx, results: dict, out: Path) -> Path:
    env = {s: ctx.envelopes[s] for s in ("left", "right")}
    tol = ctx.thresholds["k5_reach_envelope"]["tolerance_cm"] / 100
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    rng = np.random.default_rng(0)
    pts = np.vstack([env["left"].points, env["right"].points])
    pts = pts[rng.choice(len(pts), 25000, replace=False)]
    views = [
        ("top view (x forward, y left)", 0, 1),
        ("side view (x forward, z up)", 0, 2),
        ("front view (y left, z up)", 1, 2),
    ]
    inside, outside = [], []
    for side in ("left", "right"):
        for res in results.values():
            tg = res["data"].targets[side]
            m = env[side].outside(tg, tol)
            inside.append(tg[~m])
            outside.append(tg[m])
    inside, outside = np.vstack(inside), np.vstack(outside)
    for ax, (title, a, b) in zip(axes, views, strict=True):
        ax.scatter(pts[:, a], pts[:, b], s=1, color=GRID, rasterized=True)
        ax.scatter(inside[::3, a], inside[::3, b], s=3, color=BLUE, alpha=0.5, label="target inside envelope")
        ax.scatter(outside[:, a], outside[:, b], s=5, color=ORANGE, alpha=0.8, label="target out of reach")
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.set_aspect("equal")
        ax.set_xlabel("xyz"[a] + " (m)")
        ax.set_ylabel("xyz"[b] + " (m)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7, markerscale=3)
    fig.suptitle("Reach envelope (grey: Monte Carlo workspace of waist + arm) and the human hand targets", y=1.02,
                 fontsize=10, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_methods(comparison: pd.DataFrame, out: Path) -> Path:
    g = comparison.groupby("method")[
        ["fk_err_mean_cm", "clamped_fk_err_mean_cm", "frames_over_limit_share"]
    ].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    x = np.arange(len(g))
    axes[0].bar(x - 0.18, g.fk_err_mean_cm, 0.36, color=STEEL, label="raw")
    axes[0].bar(x + 0.18, g.clamped_fk_err_mean_cm, 0.36, color=BLUE, label="clamped")
    axes[0].set_xticks(x, g.index)
    axes[0].set_ylabel("mean hand error vs target (cm)")
    axes[0].legend(frameon=False)
    axes[1].bar(x, g.frames_over_limit_share * 100, 0.5, color=ORANGE)
    axes[1].set_xticks(x, g.index)
    axes[1].set_ylabel("frames with a joint past its limit (%)")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Joint-angle-only vs IK-refined retargeting (mean over clips)", y=1.04, fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ analysis helpers
def feasible_mask(ctx, res: dict) -> np.ndarray:
    """Frames the robot could execute as demonstrated: no position/velocity limit hit, target reachable, no collision."""
    robot, th = ctx.robot, ctx.thresholds
    d = res["data"]
    qd, lim = np.rad2deg(d.q_raw), robot.limits_deg
    tol = th["k2_position_limits"]["tolerance_deg"]
    pos = ((qd < lim[:, 0] - tol) | (qd > lim[:, 1] + tol)).any(axis=1)
    vel = (np.abs(C.joint_velocity_dps(d.q_raw, d.fps)) > robot.max_vel_dps).any(axis=1)
    reach_tol = th["k5_reach_envelope"]["tolerance_cm"] / 100
    reach = np.zeros(len(qd), bool)
    for s in ("left", "right"):
        reach |= ctx.envelopes[s].outside(d.targets[s], reach_tol)
    coll = np.zeros(len(qd), bool)
    for m in C.self_collision_mask(robot, d.q_clamped).values():
        coll |= m
    return ~(pos | vel | reach | coll)


def joint_violation_totals(ctx, results: dict) -> pd.DataFrame:
    names, lim = ctx.robot.joint_names, ctx.robot.limits_deg
    tol = ctx.thresholds["k2_position_limits"]["tolerance_deg"]
    pos = np.zeros(len(names))
    vel = np.zeros(len(names))
    total = 0
    for res in results.values():
        d = res["data"]
        qd = np.rad2deg(d.q_raw)
        pos += ((qd < lim[:, 0] - tol) | (qd > lim[:, 1] + tol)).sum(axis=0)
        vel += (np.abs(C.joint_velocity_dps(d.q_raw, d.fps)) > ctx.robot.max_vel_dps).sum(axis=0)
        total += qd.shape[0]
    df = pd.DataFrame(
        {"joint": names, "position_violation_share": pos / total, "velocity_violation_share": vel / total}
    )
    return df.sort_values("position_violation_share", ascending=False)


def takeaways(summary: pd.DataFrame, jv: pd.DataFrame, feasible_share: float) -> list[str]:
    out = []
    n = len(summary)
    nf = (summary.verdict == "NOT FEASIBLE").sum()
    nc = (summary.verdict == "USABLE WITH CLAMPING").sum()
    nu = (summary.verdict == "USABLE").sum()
    out.append(f"<b>Only {feasible_share:.0%} of the analysed frames are executable as demonstrated.</b> Of {n} clips, "
               f"{nu} are usable as is, {nc} need clamping and {nf} are not feasible for this robot. Raw human demonstrations "
               f"should never go into training un-audited.")  # fmt: skip
    top = jv.head(3)
    top_txt = ", ".join(f"{r.joint} ({r.position_violation_share:.0%} of frames)" for r in top.itertuples())
    out.append(f"<b>Violations concentrate in a few joints:</b> {top_txt}. Those are the joints to watch in demonstration review, "
               f"and the first candidates for a hardware or protocol change.")  # fmt: skip
    worst = summary.sort_values("k5_out_of_reach_share", ascending=False).iloc[0]
    out.append(f"<b>Reach is a task-design problem.</b> The worst reach case is {worst.clip_id} ({worst.task}): "
               f"{worst.k5_out_of_reach_share:.0%} of frames have a hand target outside the robot's workspace. "
               f"Protocol fix: mark a reach zone on the table or shelf and have operators keep objects inside it.")  # fmt: skip
    kinds = summary.groupby("kind").k2_limit_violation_share.mean().sort_values(ascending=False)
    kind_txt = ", ".join(f"{k} {v:.0%}" for k, v in kinds.items())
    out.append(
        "<b>Motion type predicts feasibility.</b> Mean share of frames past a position limit by task type: "
        f"{kind_txt}. Task types at the top of that list are where human range exceeds the robot's, so they are "
        "the ones to design around (different object placement, a different robot, or exclude from training)."
    )
    fast = summary.sort_values("k3_velocity_violation_share", ascending=False).iloc[0]
    out.append(
        f"<b>Speed matters as much as range.</b> {fast.clip_id} has {fast.k3_velocity_violation_share:.0%} of frames "
        "above a joint velocity limit. Clamping a fast demonstration makes the robot lag it, which moves where the "
        f"hand ends up (K4: up to {summary.k4_clamp_distortion_mean_cm.max():.1f} cm mean). An operator pacing "
        "guideline is the first protocol change to try; this run does not test whether it works."
    )
    return out


# ------------------------------------------------------------------------------------------------ build
def build_report(
    ctx, clips: list[dict], results: dict, summary: pd.DataFrame, comparison: pd.DataFrame, mblab: dict
):
    _style()
    run = ctx.run_dir
    assets = run / "report_assets"
    assets.mkdir(exist_ok=True)

    masks = {cid: feasible_mask(ctx, res) for cid, res in results.items()}
    total_frames = sum(len(m) for m in masks.values())
    feasible_share = sum(m.sum() for m in masks.values()) / total_frames
    summary = summary.assign(feasible_frame_share=[masks[c].mean() for c in summary.clip_id])
    summary.to_csv(run / "clip_summary.csv", index=False)
    jv = joint_violation_totals(ctx, results)

    figs = {}
    for cid, res in results.items():
        figs[cid] = fig_joint_angles(ctx, cid, res, assets / f"{cid}_joints.png")
    reach_png = fig_reach(ctx, results, assets / "reach_envelope.png")
    methods_png = fig_methods(comparison, assets / "method_comparison.png")

    cmp_mean = comparison.groupby("method").mean(numeric_only=True).reset_index()
    cmp_clip = comparison.pivot(
        index="clip_id", columns="method", values=["fk_err_mean_cm", "frames_over_limit_share"]
    )
    cmp_clip.columns = [f"{a}|{b}" for a, b in cmp_clip.columns]

    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=False)
    tpl = env.get_template("report.html.j2")
    sum_rows = summary.to_dict("records")
    ctxd = {
        "run_id": ctx.cfg["run_id"], "robot": ctx.robot, "mblab": mblab, "summary": sum_rows,
        "n_clips": len(summary), "n_frames": total_frames, "feasible_share": feasible_share,
        "verdict_counts": summary.verdict.value_counts().to_dict(), "joint_viol": jv.head(8).to_dict("records"),
        "dof": ctx.robot.dof_table(), "mblab_svg": (run / "skeletons" / "mblab_hierarchy.svg").read_text(),
        "robot_svg": (run / "skeletons" / "robot_hierarchy.svg").read_text(),
        "figs": {c: _b64(p) for c, p in figs.items()}, "reach": _b64(reach_png), "methods_img": _b64(methods_png),
        "cmp_mean": cmp_mean.to_dict("records"), "cmp_clip": cmp_clip.reset_index().to_dict("records"),
        "takeaways": takeaways(summary, jv, feasible_share), "cls": VERDICT_CLASS,
        "blender": ctx.cfg["blender"], "fps": ctx.cfg["fps"], "method": ctx.cfg["retarget_method"],
        "th": ctx.thresholds, "esc": html.escape, "checks": CHECK_CATALOG,
        "clips": {c["id"]: c for c in clips}, "reasons": {r["clip_id"]: r["reasons"] for r in sum_rows},
    }  # fmt: skip
    (run / "report.html").write_text(tpl.render(**ctxd), encoding="utf-8")
    (run / "report.md").write_text(_markdown(ctxd, figs, reach_png, methods_png), encoding="utf-8")


CHECK_CATALOG = [
    ("K1", "FK verification", "Hand position from the retargeted joint angles + link lengths vs the scaled human hand target (cm)."),
    ("K2", "Joint position limits", "Share of frames where any raw joint angle is beyond its limit; which joints."),
    ("K3", "Joint velocity limits", "Per-joint angular velocity vs the actuator's max velocity."),
    ("K4", "Clamping distortion", "Hand position difference between the raw and the clamped (position + velocity limited) trajectory."),
    ("K5", "Reach envelope", "Monte Carlo workspace of waist + arm; frames where the scaled human hand target is outside it."),
    ("K6", "Foot sliding", "Planted-foot runs of the MB-Lab retarget whose ankle drifts more than a threshold."),
    ("K7", "Self-collision proxy", "Forearm/hand capsules vs the torso capsule on the executed (clamped) motion."),
    ("K8", "Jitter", "High-frequency (above the cutoff) RMS noise in the joint angles."),
]  # fmt: skip


def _markdown(c: dict, figs: dict, reach: Path, methods: Path) -> str:
    L = [f"# Retargeting quality report: run `{c['run_id']}`\n"]
    L.append("## 1. Summary\n")
    L.append(f"- Clips processed: **{c['n_clips']}**; frames analysed: **{c['n_frames']}** at {c['fps']} Hz")
    L.append(f"- Frames executable as demonstrated: **{c['feasible_share']:.0%}**")
    L.append("- Verdicts: " + ", ".join(f"{k}: {v}" for k, v in c["verdict_counts"].items()))
    L.append("- Top violating joints (share of frames past position limit): "
             + ", ".join(f"{r['joint']} {r['position_violation_share']:.0%}" for r in c["joint_viol"][:5]) + "\n")  # fmt: skip
    L.append("## 2. Skeleton overview\n")
    L.append(
        "![MB-Lab hierarchy](skeletons/mblab_hierarchy.svg)\n\n![Robot hierarchy](skeletons/robot_hierarchy.svg)\n"
    )
    L.append("| chain | joint | axis | lo (deg) | hi (deg) | max vel (deg/s) |\n|---|---|---|---|---|---|")
    L += [
        f"| {r['chain']} | {r['joint']} | {r['axis']} | {r['lo_deg']:.0f} | {r['hi_deg']:.0f} | {r['max_vel_dps']:.0f} |"
        for r in c["dof"]
    ]
    L.append("\n## 3. Per-clip verdicts\n")
    L.append(
        "| clip | task | verdict | feasible frames | K2 | K3 | K4 cm | K5 | K7 |\n|---|---|---|---|---|---|---|---|---|"
    )
    for r in c["summary"]:
        L.append(f"| {r['clip_id']} | {r['task']} | **{r['verdict']}** | {r['feasible_frame_share']:.0%} | "
                 f"{r['k2_limit_violation_share']:.0%} | {r['k3_velocity_violation_share']:.0%} | "
                 f"{r['k4_clamp_distortion_mean_cm']:.1f} | {r['k5_out_of_reach_share']:.0%} | {r['k7_self_collision_share']:.0%} |")  # fmt: skip
    L.append("\n## 4. Joint-angle plots\n")
    L += [f"![{cid}](report_assets/{cid}_joints.png)\n" for cid in figs]
    L.append("## 5. Reach envelope\n\n![reach](report_assets/reach_envelope.png)\n")
    L.append("## 6. Retargeting method comparison\n\n![methods](report_assets/method_comparison.png)\n")
    L.append(
        "| method | mean FK error (cm) | clamped FK error (cm) | frames past a limit |\n|---|---|---|---|"
    )
    L += [
        f"| {r['method']} | {r['fk_err_mean_cm']:.2f} | {r['clamped_fk_err_mean_cm']:.2f} | {r['frames_over_limit_share']:.0%} |"
        for r in c["cmp_mean"]
    ]
    L.append("\n## 7. What this means for data collection\n")
    import re

    L += [f"{i + 1}. " + re.sub(r"</?b>", "**", t) for i, t in enumerate(c["takeaways"])]
    L.append("\n## 8. Methodology and limitations\n\nSee `report.html` section 8 and the README.")
    return "\n".join(L) + "\n"
