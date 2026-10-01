"""Create the MB-Lab character headlessly and save it as a .blend (path A in docs/character_setup.md).

Usage: blender -b -P scripts/create_character.py -- --template human_male_base --out assets/mblab_character.blend
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from _common import enable_mblab, script_args  # noqa: E402

args = script_args(
    ("--character", dict(default="m_ca01")),
    ("--out", dict(required=True)),
)

enable_mblab()
scn = bpy.context.scene
scn.mblab_character_name = args.character
result = bpy.ops.mbast.init_character()
assert result == {"FINISHED"}, f"MB-Lab init_character failed: {result}"

# Remove the default cube/light/camera: the render script builds its own stage.
for name in ("Cube", "Light", "Camera"):
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
mesh = next(o for o in bpy.data.objects if o.type == "MESH")
print(f"CHARACTER mesh={mesh.name} armature={arm.name} bones={len(arm.data.bones)} dims={tuple(mesh.dimensions)}")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
