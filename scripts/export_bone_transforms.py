"""Export per-frame world-space bone transforms of the animated MB-Lab armature.

Usage: blender -b outputs/x/blends/<clip>.blend -P scripts/export_bone_transforms.py -- --out outputs/x/blender/<clip>_bones.npz

Blender's Python has no pyarrow, so this writes NPZ; the analysis package converts it to Parquet
(outputs/<run>/transforms/<clip>_bones.parquet). Blender world space: Z up, character faces -Y, metres.
For every frame and bone: head position (3) and world rotation matrix (3x3). Rest head/tail/rotation are included
so the analysis code can compute rotation *deltas* from the rest pose without importing bpy.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from _common import find_character_armature, script_args  # noqa: E402

args = script_args(("--out", dict(required=True)))
arm = find_character_armature()
scn = bpy.context.scene
for ob in bpy.data.objects:  # skip skinning the 18k-vertex mesh: only the armature matters here (70 s -> seconds)
    if ob.type == "MESH":
        ob.hide_viewport = True
bones = sorted(arm.data.bones, key=lambda b: len(b.parent_recursive))
names = [b.name for b in bones]
frames = list(range(scn.frame_start, scn.frame_end + 1))

head = np.zeros((len(frames), len(bones), 3))
rot = np.zeros((len(frames), len(bones), 3, 3))
for fi, fr in enumerate(frames):
    scn.frame_set(fr)
    for bi, name in enumerate(names):
        m = np.array(arm.matrix_world @ arm.pose.bones[name].matrix)
        head[fi, bi] = m[:3, 3]
        rot[fi, bi] = m[:3, :3]

rest_head = np.array([np.array(arm.matrix_world @ b.head_local) for b in bones])
rest_tail = np.array([np.array(arm.matrix_world @ b.tail_local) for b in bones])
rest_rot = np.array([np.array(arm.matrix_world.to_3x3() @ b.matrix_local.to_3x3()) for b in bones])
ref_names = list(arm.get("retarget_ref_bones", names))
ref_flat = np.array(arm["retarget_ref_rot"]).reshape(-1, 3, 3) if "retarget_ref_rot" in arm else rest_rot
ref_rot = np.array([ref_flat[ref_names.index(n)] for n in names])
parents = np.array([names.index(b.parent.name) if b.parent else -1 for b in bones])

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(
    args.out,
    names=np.array(names),
    parents=parents,
    fps=np.array(scn.render.fps),
    head=head,
    rot=rot,
    rest_head=rest_head,
    rest_tail=rest_tail,
    rest_rot=rest_rot,
    ref_rot=ref_rot,
)
print(f"BONES {len(frames)} frames x {len(bones)} bones -> {args.out}")
