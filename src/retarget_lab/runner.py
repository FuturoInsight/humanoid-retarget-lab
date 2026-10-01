"""`run`: the full pipeline for every configured clip."""

from __future__ import annotations

import time

import pandas as pd

from . import pipeline as P
from .checks import CHECK_COLUMNS


def run_all(config: str, clips: list[str] | None = None, skip_render: bool = False, skip_blender: bool = False,
            force_character: bool = False) -> P.Ctx:  # fmt: skip
    ctx = P.load_ctx(config)
    selected = [c for c in ctx.cfg["clips"] if not clips or c["id"] in clips]
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    P.step_fetch_data(ctx)
    if not skip_blender:
        log("character")
        P.step_character(ctx, force=force_character)
        log("skeleton export + hierarchy diagrams")
        mblab = P.step_skeletons(ctx)
    else:
        import json

        mblab = json.loads(ctx.path("skeletons", "mblab_skeleton.json").read_text())

    all_rows, summaries, comparisons, results = [], [], [], {}
    for clip in selected:
        cid = clip["id"]
        if not skip_blender:
            log(f"{cid}: BVH -> MB-Lab -> bone transforms")
            P.step_retarget_blender(ctx, clip)
        log(f"{cid}: robot retarget + checks")
        bones = P.load_clip_bones(ctx, cid)
        res = P.analyze_clip(ctx, clip, bones)
        P.save_clip_outputs(ctx, clip, res)
        all_rows += res["rows"]
        summaries.append(P.clip_summary_row(clip, res, ctx.robot))
        comparisons += [dict(r, clip_id=cid) for r in res["comparison"]]
        results[cid] = res
        log(f"{cid}: {res['verdict']}")
        if not skip_render:
            from .render import render_clip

            log(f"{cid}: rendering")
            render_clip(ctx, clip, res)

    checks_df = pd.DataFrame(all_rows, columns=CHECK_COLUMNS)
    checks_df.to_parquet(ctx.path("check_results.parquet"), index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(ctx.path("clip_summary.csv"), index=False)
    pd.DataFrame(comparisons).to_csv(ctx.path("method_comparison.csv"), index=False)

    from .export import export_lerobot
    from .report import build_report

    log("LeRobot-style export")
    export_lerobot(ctx, selected, results, summary_df)
    log("report")
    build_report(ctx, selected, results, summary_df, pd.DataFrame(comparisons), mblab)
    log("done")
    return ctx
