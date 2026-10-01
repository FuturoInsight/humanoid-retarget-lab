"""Build the generic robot skeleton as a Blender armature with capsule geometry (steel + safety-orange overlay).

Standalone:  blender -b -P scripts/build_robot_armature.py -- --payload outputs/x/render/<clip>_robot.npz --out robot.blend
Also imported by render_side_by_side.py (`build_robot`).

The payload (written by retarget_lab.robot_render) holds, per link, the zero-pose start/end points, so the armature
is exactly the robot's zero pose: one bone per link, head at the link start and tail at the link end.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

STEEL = (0.62, 0.66, 0.70, 1.0)
ORANGE = (1.0, 0.36, 0.0, 1.0)  # safety orange


def make_material(name, color, metallic, roughness, emission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Emission Color"].default_value = color
        bsdf.inputs["Emission Strength"].default_value = emission
    return mat


def _link_object(name, kind, radius, material, collection):
    """Unit-length cylinder along +Z (scaled to the link length per frame) or a joint ball of the given radius."""
    if kind == "cyl":
        bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=radius, depth=1.0)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=radius)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    for c in obj.users_collection:
        c.objects.unlink(obj)
    collection.objects.link(obj)
    return obj


def segment_matrix(p0, p1):
    """4x4 with origin at p0 and +Y along p0->p1 (bone convention); roll is arbitrary."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length = np.linalg.norm(d)
    y = d / max(length, 1e-9)
    helper = np.array([0.0, 0.0, 1.0]) if abs(y[2]) < 0.95 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, y)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, p0
    return Matrix(m.tolist()), length


def build_robot(payload, x_offset=0.0, collection=None):
    """Create the armature + geometry. Returns dict(armature, steel={name: obj}, orange={name: obj}, names, rest)."""
    collection = collection or bpy.context.scene.collection
    names = [str(n) for n in payload["names"]]
    radius = payload["radius"]
    rest = np.array(payload["rest"]) + np.array([x_offset, 0, 0])

    arm_data = bpy.data.armatures.new("robot_armature")
    arm = bpy.data.objects.new("robot_armature", arm_data)
    collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for k, n in enumerate(names):
        b = arm_data.edit_bones.new(n)
        b.head, b.tail = Vector(rest[k, 0]), Vector(rest[k, 1])
    bpy.ops.object.mode_set(mode="OBJECT")
    arm_data.display_type = "STICK"

    steel_mat = make_material("robot_steel", STEEL, 0.85, 0.35)
    orange_mat = make_material("robot_safety_orange", ORANGE, 0.1, 0.4, emission=0.35)
    steel, orange = {}, {}
    for k, n in enumerate(names):
        steel[n] = [
            _link_object(f"{n}_steel_cyl", "cyl", radius[k], steel_mat, collection),
            _link_object(f"{n}_steel_ball", "ball", radius[k] * 1.25, steel_mat, collection),
        ]
        orange[n] = [
            _link_object(f"{n}_orange_cyl", "cyl", radius[k] * 1.12, orange_mat, collection),
            _link_object(f"{n}_orange_ball", "ball", radius[k] * 1.4, orange_mat, collection),
        ]
    return {"armature": arm, "steel": steel, "orange": orange, "names": names, "rest": rest}


def pose_robot(robot, endpoints, highlight, x_offset=0.0):
    """Pose the armature from per-link endpoints (L,2,3) and place the capsule geometry on the posed bones."""
    arm = robot["armature"]
    for k, n in enumerate(robot["names"]):
        p0 = np.asarray(endpoints[k, 0]) + np.array([x_offset, 0, 0])
        p1 = np.asarray(endpoints[k, 1]) + np.array([x_offset, 0, 0])
        m, length = segment_matrix(p0, p1)
        arm.pose.bones[n].matrix = m
        mid = (p0 + p1) / 2
        gm = Matrix.Translation(mid) @ Matrix(segment_matrix(p0, p1)[0]).to_3x3().to_4x4()
        # capsules are built along +Z; bone/segment frames put the link along +Y -> rotate Z onto Y
        gm = gm @ Matrix.Rotation(-np.pi / 2, 4, "X")
        gm = gm @ Matrix.Diagonal(Vector((1.0, 1.0, max(length, 1e-6), 1.0)))
        ball = Matrix.Translation(p0)
        for group in (robot["steel"][n], robot["orange"][n]):
            group[0].matrix_world = gm
            group[1].matrix_world = ball
        for o in robot["orange"][n]:
            o.hide_render = not bool(highlight[k])
        for o in robot["steel"][n]:
            o.hide_render = bool(highlight[k])


if __name__ == "__main__":
    from _common import script_args

    args = script_args(("--payload", dict(required=True)), ("--out", dict(required=True)))
    payload = np.load(args.payload)
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    r = build_robot(payload)
    pose_robot(r, payload["rest"], np.zeros(len(r["names"]), bool))
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out))
    print(f"ROBOT armature with {len(r['names'])} bones -> {args.out}")
