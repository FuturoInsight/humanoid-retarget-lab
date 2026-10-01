"""Side-by-side renders: drive Blender for the three panels, then stitch, caption and encode MP4 + GIF."""

from __future__ import annotations

import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline import Ctx, run_blender
from .robot_render import render_payload

ORANGE = (255, 92, 0)
INK, BG, MUTED = (236, 240, 244), (24, 28, 34), (150, 160, 172)
VERDICT_COLOR = {"USABLE": (80, 200, 120), "USABLE WITH CLAMPING": (255, 190, 60), "NOT FEASIBLE": ORANGE}
TITLES = ["1  CMU mocap (BVH)", "2  MB-Lab character (retargeted)", "3  Robot skeleton (raw, limits ignored)"]


def _font(size: int, bold: bool = False):
    for name in (
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_frame(
    panels: list[Image.Image],
    clip_id: str,
    frame: int,
    n_frames: int,
    verdict: str,
    n_over: int,
    over_names: str,
):
    w, h = panels[0].size
    head, foot = 34, 56
    canvas = Image.new("RGB", (w * 3, h + head + foot), BG)
    d = ImageDraw.Draw(canvas)
    f_title, f_small = _font(15, True), _font(13)
    for i, p in enumerate(panels):
        canvas.paste(p, (i * w, head))
        d.text((i * w + 12, 8), TITLES[i], fill=INK, font=f_title)
        if i:
            d.line([(i * w, head), (i * w, head + h)], fill=(60, 66, 76), width=2)
    y = head + h + 8
    d.text((12, y), f"{clip_id}   frame {frame + 1}/{n_frames}", fill=INK, font=f_title)
    vc = VERDICT_COLOR.get(verdict, INK)
    d.rounded_rectangle([w * 3 - 250, y - 2, w * 3 - 12, y + 22], radius=6, outline=vc, width=2)
    d.text((w * 3 - 240, y + 2), f"verdict: {verdict}", fill=vc, font=f_small)
    sw = ORANGE if n_over else (90, 96, 106)
    d.rectangle([12, y + 28, 26, y + 42], fill=sw)
    msg = f"{n_over} joint(s) past limit: {over_names}" if n_over else "all joints within limits"
    d.text((34, y + 27), msg, fill=ORANGE if n_over else MUTED, font=f_small)
    return canvas


def render_clip(ctx: Ctx, clip: dict, res: dict, keep_frames: bool = False) -> dict[str, Path]:
    robot = ctx.robot
    cid = clip["id"]
    motion, q = res["motion"], res["q_raw"]
    payload = render_payload(robot, q, motion.base_pos, motion.heading, q)
    robot_npz = ctx.path("render", f"{cid}_robot.npz")
    np.savez_compressed(robot_npz, **payload)
    frames_dir = ctx.run_dir / "render" / f"{cid}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    rc = ctx.cfg["render"]
    run_blender(
        ctx, "render_side_by_side.py",
        ["--mocap", str(ctx.run_dir / "blender" / f"{cid}_retarget_mocap.npz"), "--robot", str(robot_npz),
         "--out-dir", str(frames_dir), "--width", str(rc["width"]), "--height", str(rc["height"]),
         "--step", str(rc["step"]), "--samples", str(rc["samples"])],
        blend=ctx.run_dir / "blends" / f"{cid}.blend", log=f"{cid}_render",
    )  # fmt: skip

    lim = robot.limits_deg
    qd = np.rad2deg(q)
    over = (qd < lim[:, 0] - 1) | (qd > lim[:, 1] + 1)
    n_frames = q.shape[0]
    out = []
    count = len(list(frames_dir.glob("p0_*.png")))
    for k in range(count):
        f = k * rc["step"]
        panels = [Image.open(frames_dir / f"p{i}_{k:04d}.png").convert("RGB") for i in range(3)]
        names = [robot.joint_names[j] for j in np.flatnonzero(over[f])]
        txt = ", ".join(names[:4]) + (" ..." if len(names) > 4 else "")
        out.append(compose_frame(panels, cid, f, n_frames, res["verdict"], len(names), txt))
    fps = ctx.cfg["fps"] / rc["step"]
    mp4 = ctx.path("renders", f"{cid}_side_by_side.mp4")
    arr = [np.asarray(im) for im in out]
    imageio.mimsave(
        mp4,
        arr,
        fps=fps,
        codec="libx264",
        macro_block_size=2,
        pixelformat="yuv420p",
        output_params=["-crf", "27", "-preset", "slow"],
    )
    gif = ctx.path("renders", f"{cid}_side_by_side.gif")
    gif_frames = out[::2]  # the GIF is the 20-second hiring-manager version: half the frames, 45 % size
    small = [im.resize((im.width * 45 // 100, im.height * 45 // 100), Image.LANCZOS).quantize(colors=64, dither=Image.NONE)
             for im in gif_frames]  # fmt: skip
    small[0].save(
        gif, save_all=True, append_images=small[1:], duration=int(2000 / fps), loop=0, optimize=True
    )
    out[len(out) // 2].save(ctx.path("renders", f"{cid}_still.png"))
    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return {"mp4": mp4, "gif": gif}
