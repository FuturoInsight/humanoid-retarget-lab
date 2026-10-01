"""Export robot trajectories in a LeRobot-style layout so the Robot Demonstration Data QA Pipeline can ingest them.

    export/meta/info.json      dataset-level description: fps, feature names, totals, source/robot provenance
    export/meta/episodes.csv   one row per episode (= clip): length, task, feasibility verdict and reasons
    export/meta/tasks.csv      task_index -> natural-language task
    export/data/frames.csv     one row per frame: episode_index, frame_index, timestamp, task_index,
                               observation.state.<joint> (deg, clamped = what the robot can execute),
                               action.<joint> (next frame's state, last frame repeats), hand positions, flags

The companion projects' exact schema was not available while building this, so the columns follow LeRobot's
conventions (episode_index / frame_index / timestamp / task_index, observation.state, action); see DECISIONS.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def export_lerobot(ctx, clips: list[dict], results: dict, summary: pd.DataFrame) -> None:
    robot = ctx.robot
    out = ctx.run_dir / "export"
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    tasks = sorted({c["task"] for c in clips})
    task_idx = {t: i for i, t in enumerate(tasks)}
    fps = ctx.cfg["fps"]

    frames, episodes = [], []
    for e, clip in enumerate(clips):
        res = results[clip["id"]]
        q = np.rad2deg(res["q_clamped"])
        raw = np.rad2deg(res["q_raw"])
        n = q.shape[0]
        action = np.vstack([q[1:], q[-1:]])
        df = pd.DataFrame({"episode_index": e, "frame_index": np.arange(n), "timestamp": np.arange(n) / fps,
                           "task_index": task_idx[clip["task"]]})  # fmt: skip
        for j, name in enumerate(robot.joint_names):
            df[f"observation.state.{name}"] = q[:, j]
        for j, name in enumerate(robot.joint_names):
            df[f"action.{name}"] = action[:, j]
        for side in ("left", "right"):
            for ax in "xyz":
                df[f"observation.hand_{side}_{ax}"] = res["clamped_df"][f"hand_{side}_{ax}"].to_numpy()
        df["flag.clamped_frame"] = (np.abs(q - raw) > 0.5).any(axis=1)
        frames.append(df)
        row = summary.loc[summary.clip_id == clip["id"]].iloc[0]
        episodes.append({
            "episode_index": e, "episode_id": clip["id"], "length": n, "task_index": task_idx[clip["task"]],
            "task": clip["task"], "feasibility_verdict": row.verdict, "verdict_reasons": row.reasons,
            "source": clip["bvh"].replace("\\", "/"),
        })  # fmt: skip
    pd.concat(frames).to_csv(out / "data" / "frames.csv", index=False)
    pd.DataFrame(episodes).to_csv(out / "meta" / "episodes.csv", index=False)
    pd.DataFrame({"task_index": range(len(tasks)), "task": tasks}).to_csv(
        out / "meta" / "tasks.csv", index=False
    )
    info = {
        "format": "LeRobot-style (CSV flavour)",
        "robot_type": robot.name,
        "robot_note": "generic humanoid skeleton defined in config/robot_skeleton.yaml; not any real robot",
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": int(sum(len(f) for f in frames)),
        "joint_names": robot.joint_names,
        "joint_limits_deg": {n: [robot.joint(n).lo_deg, robot.joint(n).hi_deg] for n in robot.joint_names},
        "features": {
            "observation.state": {"dtype": "float32", "shape": [len(robot.joint_names)], "unit": "deg",
                                  "names": robot.joint_names},
            "action": {"dtype": "float32", "shape": [len(robot.joint_names)], "unit": "deg",
                       "names": robot.joint_names, "definition": "next-frame observation.state"},
        },
        "source_dataset": "CMU Graphics Lab Motion Capture Database (BVH conversion by B. Hahne)",
        "pipeline": f"retarget_lab {ctx.cfg['retarget_method']}",
    }  # fmt: skip
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
