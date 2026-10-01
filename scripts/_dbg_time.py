import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
from _common import script_args
a = script_args(("--engine", dict(default="BLENDER_EEVEE")), ("--samples", dict(type=int, default=8)))
scn = bpy.context.scene
m = next(o for o in bpy.data.objects if o.type=="MESH")
print("MODS", [(x.name,x.type) for x in m.modifiers], "keys", len(m.data.shape_keys.key_blocks) if m.data.shape_keys else 0, "verts", len(m.data.vertices))
for md in m.modifiers:
    if md.type in ("CORRECTIVE_SMOOTH","SUBSURF","DISPLACE"): md.show_render=False; md.show_viewport=False
scn.render.engine = a.engine
scn.render.resolution_x, scn.render.resolution_y = 480, 540
if a.engine == "BLENDER_EEVEE": scn.eevee.taa_render_samples = a.samples
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); scn.collection.objects.link(cam); scn.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = 2.4; cam.location=(0,-6,0.95); cam.rotation_euler=(math.radians(90),0,0)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); scn.collection.objects.link(sun)
for fr in range(1,7):
    t=time.time(); scn.frame_set(fr); t1=time.time()-t
    t=time.time(); scn.render.filepath=f"//dbgt_{fr}.png"; bpy.ops.render.render(write_still=True); print("TIME", a.engine, fr, f"frame_set {t1:.2f}s render {time.time()-t:.2f}s")
