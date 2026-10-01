"""Export the MB-Lab armature hierarchy to JSON: bone, parent, head/tail rest position, length, depth.

Usage: blender -b assets/mblab_character.blend -P scripts/export_skeleton.py -- --out outputs/x/skeletons/mblab_skeleton.json
Positions are armature-space metres (Blender: Z up, character faces -Y), rest pose.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from _common import find_character_armature, script_args  # noqa: E402

args = script_args(("--out", dict(required=True)))
arm = find_character_armature()
mw = arm.matrix_world

bones = []
for b in arm.data.bones:
    depth, p = 0, b.parent
    while p is not None:
        depth, p = depth + 1, p.parent
    head, tail = mw @ b.head_local, mw @ b.tail_local
    bones.append(
        {
            "name": b.name,
            "parent": b.parent.name if b.parent else None,
            "head": [round(v, 5) for v in head],
            "tail": [round(v, 5) for v in tail],
            "rest_length": round((tail - head).length, 5),
            "depth": depth,
            "deform": b.use_deform,
        }
    )

mesh = next((o for o in bpy.data.objects if o.type == "MESH" and o.parent == arm), None)
doc = {
    "armature": arm.name,
    "blender_version": bpy.app.version_string,
    "units": "metre",
    "axes": "Z up, character faces -Y, +X is the character's left",
    "height_m": round(mesh.dimensions.z, 4) if mesh else None,
    "n_bones": len(bones),
    "bones": bones,
}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w") as fh:
    json.dump(doc, fh, indent=1)
print(f"SKELETON {len(bones)} bones -> {args.out}")
