import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from mathutils import Vector
from _common import script_args, find_character_armature
a = script_args(("--out", dict(required=True)), ("--frames", dict(default="1,80,160")))
arm = find_character_armature()
scn = bpy.context.scene
scn.render.engine = "BLENDER_EEVEE"
scn.render.resolution_x, scn.render.resolution_y = 480, 540
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); scn.collection.objects.link(cam); scn.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = 2.4
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); scn.collection.objects.link(sun)
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
for fr in [int(x) for x in a.frames.split(",")]:
    scn.frame_set(fr)
    p = arm.matrix_world @ arm.pose.bones["pelvis"].head
    cam.location = (p.x - 4 * 0.0, p.y - 6, 0.95)  # look along +Y at the character (front view of a -Y facing char => from -Y)
    cam.rotation_euler = (math.radians(90), 0, 0)
    cam.location = (p.x, p.y - 6, 0.95)
    scn.render.filepath = f"{a.out}_front_{fr:04d}.png"
    bpy.ops.render.render(write_still=True)
    cam.location = (p.x + 6, p.y, 0.95); cam.rotation_euler = (math.radians(90), 0, math.radians(90))
    scn.render.filepath = f"{a.out}_side_{fr:04d}.png"
    bpy.ops.render.render(write_still=True)
print("DONE")
