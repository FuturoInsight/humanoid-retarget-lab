"""Exchange formats between the Blender side (NPZ) and the analysis side (Parquet / DataFrames)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BoneData:
    """Per-frame world transforms of every MB-Lab bone (Blender world space: Z up, faces -Y, metres)."""

    names: list[str]
    parents: np.ndarray
    fps: float
    head: np.ndarray  # (F, B, 3)
    rot: np.ndarray  # (F, B, 3, 3)
    rest_head: np.ndarray  # (B, 3)
    rest_tail: np.ndarray  # (B, 3)
    rest_rot: np.ndarray  # (B, 3, 3)
    ref_rot: (
        np.ndarray
    )  # (B, 3, 3) T-pose reference orientation used by the retarget (rest for unmapped bones)

    @property
    def n_frames(self) -> int:
        return self.head.shape[0]

    def idx(self, name: str) -> int:
        return self.names.index(name)

    def pos(self, name: str) -> np.ndarray:
        return self.head[:, self.idx(name)]


def bones_npz_to_parquet(npz_path: str | Path, out_path: str | Path) -> None:
    """Long-format Parquet: one row per (frame, bone); frame == -1 rows carry the rest pose."""
    d = np.load(npz_path)
    names = [str(n) for n in d["names"]]
    n_f, n_b = d["head"].shape[:2]
    rows = {
        "frame": np.repeat(np.arange(n_f), n_b),
        "bone": np.tile(names, n_f),
        "parent": np.tile([names[p] if p >= 0 else "" for p in d["parents"]], n_f),
    }
    for i, ax in enumerate("xyz"):
        rows[f"h{ax}"] = d["head"][:, :, i].ravel()
    flat = d["rot"].reshape(n_f * n_b, 9)
    for k in range(9):
        rows[f"r{k // 3}{k % 3}"] = flat[:, k]
    rest = {
        "frame": np.full(n_b, -1),
        "bone": names,
        "parent": [names[p] if p >= 0 else "" for p in d["parents"]],
    }
    for i, ax in enumerate("xyz"):
        rest[f"h{ax}"] = d["rest_head"][:, i]
        rest[f"t{ax}"] = d["rest_tail"][:, i]
    rr = d["rest_rot"].reshape(n_b, 9)
    ff = d["ref_rot"].reshape(n_b, 9)
    for k in range(9):
        rest[f"r{k // 3}{k % 3}"] = rr[:, k]
        rest[f"f{k // 3}{k % 3}"] = ff[:, k]
    df = pd.concat([pd.DataFrame(rest), pd.DataFrame(rows)], ignore_index=True)
    df.attrs["fps"] = float(d["fps"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df["fps"] = float(d["fps"])
    df.to_parquet(out_path, index=False)


def load_bones(parquet_path: str | Path) -> BoneData:
    df = pd.read_parquet(parquet_path)
    rest = df[df.frame == -1].reset_index(drop=True)
    names = list(rest.bone)
    parents = np.array([names.index(p) if p else -1 for p in rest.parent])
    body = df[df.frame >= 0].sort_values(["frame"], kind="stable")
    n_f, n_b = int(body.frame.max()) + 1, len(names)
    order = {n: i for i, n in enumerate(names)}
    body = body.assign(_b=body.bone.map(order)).sort_values(["frame", "_b"])
    head = body[["hx", "hy", "hz"]].to_numpy().reshape(n_f, n_b, 3)
    rot = body[[f"r{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(n_f, n_b, 3, 3)
    rrot = rest[[f"r{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(n_b, 3, 3)
    frot = rest[[f"f{i}{j}" for i in range(3) for j in range(3)]].to_numpy().reshape(n_b, 3, 3)
    return BoneData(
        names,
        parents,
        float(df["fps"].iloc[0]),
        head,
        rot,
        rest[["hx", "hy", "hz"]].to_numpy(),
        rest[["tx", "ty", "tz"]].to_numpy(),
        rrot,
        frot,
    )
