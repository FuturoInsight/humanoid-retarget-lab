"""Render the three panels of one clip: mocap stick figure | MB-Lab character | robot skeleton (Eevee, headless).

Usage:
  blender -b outputs/x/blends/<clip>.blend -P scripts/render_side_by_side.py -- \
      --mocap outputs/x/blender/<clip>_retarget_mocap.npz --robot outputs/x/render/<clip>_robot.npz \
      --out-dir outputs/x/render/<clip>_frames --step 2

Each panel has its own camera that follows its actor; the three actors live 30 m apart on one shared floor, so
the panels never see each other. The python side (retarget_lab.render) stitches panels, adds titles and encodes MP4/GIF.
The robot panel plays the *raw* retargeted motion (limits ignored); links driven by a joint past its limit are
swapped for their safety-orange twin (see build_robot_armature.py).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from _common import find_character_armature, script_args  # noqa: E402
from build_robot_armature import build_robot, make_material, pose_robot, segment_matrix  # noqa: E402

args = script_args(
    ("--mocap", dict(required=True)),
    ("--robot", dict(required=True)),
    ("--out-dir", dict(required=True)),
    ("--width", dict(type=int, default=420)),
    ("--height", dict(type=int, default=540)),
    ("--step", dict(type=int, default=2)),
    ("--samples", dict(type=int, default=12)),
    ("--max-frames", dict(type=int, default=0)),
)
scn = bpy.context.scene
arm = find_character_armature()
mocap = np.load(args.mocap)
robot_payload = np.load(args.robot)
MOCAP_X, ROBOT_X = -30.0, 30.0

# ---------------------------------------------------------------- character: drop the heavy cosmetic modifiers
for ob in bpy.data.objects:
    if ob.type == "MESH":
        for md in ob.modifiers:
            if md.type in ("CORRECTIVE_SMOOTH", "SUBSURF", "DISPLACE"):
                md.show_viewport = md.show_render = False

# ---------------------------------------------------------------- stage: floor, lights, world
world = bpy.data.worlds.new("stage") if scn.world is None else scn.world
scn.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.11, 0.125, 0.145, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8

bpy.ops.mesh.primitive_plane_add(size=400, location=(0, 0, 0))
floor = bpy.context.active_object
floor_mat = bpy.data.materials.new("floor")
floor_mat.use_nodes = True
nt = floor_mat.node_tree
chk = nt.nodes.new("ShaderNodeTexChecker")
chk.inputs["Scale"].default_value = 1.0  # 1 m squares (object coordinates are metres)
chk.inputs["Color1"].default_value = (0.30, 0.33, 0.37, 1)
chk.inputs["Color2"].default_value = (0.21, 0.235, 0.27, 1)
mapping_in = nt.nodes.new("ShaderNodeTexCoord")
nt.links.new(mapping_in.outputs["Object"], chk.inputs["Vector"])
nt.links.new(chk.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.9
floor.data.materials.append(floor_mat)

sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.2
sun.rotation_euler = (math.radians(48), math.radians(8), math.radians(35))
scn.collection.objects.link(sun)
fill = bpy.data.objects.new("fill", bpy.data.lights.new("fill", "SUN"))
fill.data.energy = 1.2
fill.rotation_euler = (math.radians(70), 0, math.radians(-140))
scn.collection.objects.link(fill)

# ---------------------------------------------------------------- mocap stick figure
stick_mat = make_material("mocap_blue", (0.25, 0.55, 0.95, 1), 0.0, 0.5, emission=0.6)
names = [str(n) for n in mocap["names"]]
parents = mocap["parents"]
pos = np.array(mocap["positions"])
toe_idx = [names.index("LeftToeBase"), names.index("RightToeBase")]
pos[:, :, 2] -= pos[:, toe_idx, 2].min()  # lowest foot point of the clip touches the floor
pos[:, :, 0] += MOCAP_X
bones_m = []
for j, p in enumerate(parents):
    if p < 0 or np.linalg.norm(pos[0, j] - pos[0, p]) < 1e-4:
        continue
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.014, depth=1.0)
    cyl = bpy.context.active_object
    cyl.data.materials.append(stick_mat)
    bones_m.append((j, p, cyl))
joint_balls = []
for _ in range(len(names)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.028)
    ball = bpy.context.active_object
    ball.data.materials.append(stick_mat)
    joint_balls.append(ball)


def pose_mocap(f):
    for j, p, cyl in bones_m:
        m, length = segment_matrix(pos[f, p], pos[f, j])
        mid = (pos[f, p] + pos[f, j]) / 2
        gm = Matrix.Translation(mid) @ m.to_3x3().to_4x4() @ Matrix.Rotation(-math.pi / 2, 4, "X")
        cyl.matrix_world = gm @ Matrix.Diagonal(Vector((1, 1, max(length, 1e-6), 1)))
    for j, ball in enumerate(joint_balls):
        ball.matrix_world = Matrix.Translation(pos[f, j])


# ---------------------------------------------------------------- robot
robot = build_robot(robot_payload, x_offset=ROBOT_X)
endpoints = np.array(robot_payload["endpoints"])
highlight = np.array(robot_payload["highlight"])
link_names = [str(n) for n in robot_payload["names"]]

# ---------------------------------------------------------------- cameras that follow each actor
scn.render.engine = "BLENDER_EEVEE"
scn.eevee.taa_render_samples = args.samples
scn.render.resolution_x, scn.render.resolution_y = args.width, args.height
scn.render.image_settings.file_format = "PNG"
scn.view_settings.view_transform = "Standard"


def make_cam(name):
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    cam.data.lens = 55
    scn.collection.objects.link(cam)
    target = bpy.data.objects.new(name + "_target", None)
    scn.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target, con.track_axis, con.up_axis = target, "TRACK_NEGATIVE_Z", "UP_Y"
    return cam, target


panels = [make_cam("cam_mocap"), make_cam("cam_char"), make_cam("cam_robot")]
CAM_OFFSET = Vector((2.5, -4.6, 0.9))  # front-right three-quarter view, character faces -Y

hips_i = names.index("Hips")
hip_bar_i = link_names.index("hip_bar")
n_total = endpoints.shape[0]
frames = list(range(0, n_total, args.step))
if args.max_frames:
    frames = frames[: args.max_frames]
os.makedirs(args.out_dir, exist_ok=True)

for k, f in enumerate(frames):
    scn.frame_set(f + 1)
    pose_mocap(f)
    pose_robot(robot, endpoints[f], highlight[f], x_offset=ROBOT_X)
    centres = [
        Vector(pos[f, hips_i]),
        arm.matrix_world @ arm.pose.bones["pelvis"].head,
        Vector((endpoints[f, hip_bar_i].mean(axis=0))) + Vector((ROBOT_X, 0, 0)),
    ]
    for i, ((cam, target), c) in enumerate(zip(panels, centres, strict=True)):
        target.location = Vector((c.x, c.y, 0.85))
        cam.location = Vector((c.x, c.y, 0.0)) + CAM_OFFSET + Vector((0, 0, 0.4))
        bpy.context.view_layer.update()
        scn.camera = cam
        scn.render.filepath = os.path.join(args.out_dir, f"p{i}_{k:04d}.png")
        bpy.ops.render.render(write_still=True)
print(f"RENDERED {len(frames)} frames x 3 panels -> {args.out_dir}")
