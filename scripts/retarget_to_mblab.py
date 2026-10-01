"""Retarget one CMU BVH clip onto the MB-Lab armature and bake it to an action.

Usage:
  blender -b assets/mblab_character.blend -P scripts/retarget_to_mblab.py -- \
      --bvh data/raw/69_72.bvh --bone-map outputs/x/bone_map.json --clip-id cmu_69_72 \
      --out-blend outputs/x/blends/cmu_69_72.blend --out-meta outputs/x/blender/cmu_69_72_retarget.json \
      --fps 30 --start-sec 0 --max-sec 10

How the rest-pose correction works (never a raw rotation copy):
  * The CMU BVH's first frame is a T-pose. The MB-Lab rest pose is an A-pose with different bone roll axes.
  * Per mapped bone we take the source segment's *world-space rotation delta from its T-pose*,
        D(t) = R_src(t) @ R_src(T-pose)^T
    and apply it to the target bone's *T-pose reference orientation*,
        R_dst(t) = D(t) @ R_ref,   R_ref = swing(rest direction -> source T-pose direction) @ R_rest.
    R_ref is the per-bone rest-pose offset: it tilts the A-pose bone into the source's T-pose direction.
  * Root motion: hip translation relative to the T-pose frame, scaled by the leg-length ratio.
All the maths is done in numpy in armature space and written as `matrix_basis` per pose bone, so the result does
not depend on depsgraph evaluation order. Source data comes from retarget_lab.bvh (numpy-only BVH reader).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix  # noqa: E402

from _common import find_character_armature, script_args  # noqa: E402
from retarget_lab.bvh import forward_kinematics, parse_bvh, to_blender_space  # noqa: E402

args = script_args(
    ("--bvh", dict(required=True)),
    ("--bone-map", dict(required=True, help="bone map as JSON (converted from YAML by the CLI)")),
    ("--clip-id", dict(required=True)),
    ("--out-blend", dict(required=True)),
    ("--out-meta", dict(required=True)),
    ("--fps", dict(type=float, default=30.0)),
    ("--start-sec", dict(type=float, default=0.0)),
    ("--max-sec", dict(type=float, default=10.0)),
)


def swing(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-9:
        return np.eye(3)
    k = v / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = np.arctan2(s, c)
    return np.eye(3) + np.sin(ang) * kx + (1 - np.cos(ang)) * (kx @ kx)


def mat_to_quat(m3):
    return Matrix(m3.tolist()).to_quaternion()


with open(args.bone_map) as fh:
    bone_map = json.load(fh)
bvh = parse_bvh(args.bvh)
arm = find_character_armature()
assert np.allclose(np.array(arm.matrix_world), np.eye(4), atol=1e-6), (
    "armature object transform must be identity"
)

# ---- frame selection (decimate 120 Hz -> target fps); frame 0 is the BVH's T-pose reference frame
stride = max(1, int(round(bvh.fps / args.fps)))
first = 1 + int(round(args.start_sec * bvh.fps))
last = min(bvh.n_frames, first + int(round(args.max_sec * bvh.fps)))
sel = np.arange(first, last, stride)
r_all, p_all = to_blender_space(*forward_kinematics(bvh, np.concatenate([[0], sel])))
r0, p0 = r_all[0], p_all[0]
r_src, p_src = r_all[1:], p_all[1:]
n_frames = len(sel)
print(f"RETARGET {args.clip_id}: {n_frames} frames, stride {stride} ({bvh.fps:.0f} -> {args.fps:.0f} Hz)")


def j(name):
    return bvh.index(name)


# ---- target rest data (armature space)
bones = sorted(arm.data.bones, key=lambda b: len(b.parent_recursive))
rest = {b.name: np.array(b.matrix_local) for b in bones}
parent = {b.name: (b.parent.name if b.parent else None) for b in bones}
rest_local = {n: (np.linalg.inv(rest[parent[n]]) @ rest[n] if parent[n] else rest[n]) for n in rest}

# ---- scale: leg-length ratio (hip joint -> ankle joint along the chain)
src_leg = np.mean(
    [
        np.linalg.norm(p0[j(f"{s}UpLeg")] - p0[j(f"{s}Leg")])
        + np.linalg.norm(p0[j(f"{s}Leg")] - p0[j(f"{s}Foot")])
        for s in ("Left", "Right")
    ]
)
dst_leg = np.mean(
    [
        np.linalg.norm(rest[f"thigh_{s}"][:3, 3] - rest[f"calf_{s}"][:3, 3])
        + np.linalg.norm(rest[f"calf_{s}"][:3, 3] - rest[f"foot_{s}"][:3, 3])
        for s in ("L", "R")
    ]
)
scale = float(dst_leg / src_leg)
src_height = float(p0[j("Head_End")][2] - min(p0[j("LeftToeBase")][2], p0[j("RightToeBase")][2]))
print(f"RETARGET scale(leg ratio)={scale:.5f} source_height={src_height * scale:.3f} m (scaled)")

# ---- per-bone rest correction
mapped = bone_map["bones"]
r_ref, src_idx = {}, {}
for bname, spec in mapped.items():
    s, d = j(spec["src"]), j(spec["dir"])
    u_src = p0[d] - p0[s]
    if np.linalg.norm(u_src) < 1e-6:
        raise ValueError(f"bone map: {spec['src']} -> {spec['dir']} has zero length in the BVH T-pose")
    u_dst = rest[bname][:3, 1]  # bone y axis points head -> tail
    r_ref[bname] = swing(u_dst, u_src) @ rest[bname][:3, :3]
    src_idx[bname] = s

# Root motion is driven by the hip-joint midpoint (CMU `Hips` sits above the thigh joints, MB-Lab's `pelvis` head
# sits below them, so copying the pelvis point directly would lift/sink the character when the pelvis rotates).
root_cfg = bone_map["root_translation"]
root_bone = root_cfg["target"]
src_mid = np.mean([p0[j(n)] for n in root_cfg["src"]], axis=0)
src_mid_t = np.mean([p_src[:, j(n)] for n in root_cfg["src"]], axis=0)  # (F, 3)
rest_mid = np.mean([rest[c][:3, 3] for c in root_cfg["children"]], axis=0)
mid_local = np.mean([rest_local[c][:3, 3] for c in root_cfg["children"]], axis=0)  # in the pelvis rest frame

# Store the per-bone T-pose reference orientation on the armature so the analysis side can measure rotation
# deltas from the same reference (unmapped bones: their rest orientation).
ref_all = np.array([r_ref.get(b.name, rest[b.name][:3, :3]) for b in bones])
arm["retarget_ref_bones"] = [b.name for b in bones]
arm["retarget_ref_rot"] = [float(v) for v in ref_all.ravel()]

# ---- solve every frame
act_name = f"{args.clip_id}_retarget"
if arm.animation_data is None:
    arm.animation_data_create()
action = bpy.data.actions.new(act_name)
arm.animation_data.action = action
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
    pb.location = (0, 0, 0)
    pb.rotation_quaternion = (1, 0, 0, 0)

prev_q = {}
for f in range(n_frames):
    world = {}
    basis = {}
    for b in bones:
        n = b.name
        base = (world[parent[n]] @ rest_local[n]) if parent[n] else rest_local[n].copy()
        if n in mapped:
            s = src_idx[n]
            delta = r_src[f, s] @ r0[s].T
            m = np.eye(4)
            m[:3, :3] = delta @ r_ref[n]
            m[:3, 3] = base[:3, 3]
            if n == root_bone:  # place the pelvis so the hip-joint midpoint follows the scaled source
                m[:3, 3] = rest_mid + scale * (src_mid_t[f] - src_mid) - m[:3, :3] @ mid_local
        else:
            m = base
        world[n] = m
        basis[n] = np.linalg.inv(base) @ m
    frame_no = f + 1
    for n, bm in basis.items():
        if n not in mapped:
            continue
        pb = arm.pose.bones[n]
        q = mat_to_quat(bm[:3, :3])
        if n in prev_q and q.dot(prev_q[n]) < 0:
            q.negate()
        prev_q[n] = q.copy()
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=frame_no)
        if n == root_bone:
            pb.location = tuple(bm[:3, 3])
            pb.keyframe_insert("location", frame=frame_no)

# ---- source (mocap) joints, scaled + aligned to the character's pelvis, for the stick-figure panel
src_world = rest_mid[None, None, :] + scale * (p_src - src_mid)  # same placement as the retarget
os.makedirs(os.path.dirname(os.path.abspath(args.out_meta)), exist_ok=True)
np.savez_compressed(
    os.path.splitext(args.out_meta)[0] + "_mocap.npz",
    names=np.array(bvh.names),
    parents=np.array(bvh.parents),
    positions=src_world,
)
meta = {
    "clip_id": args.clip_id,
    "bvh": args.bvh,
    "n_frames": n_frames,
    "fps": args.fps,
    "source_fps": bvh.fps,
    "stride": stride,
    "bvh_frame_first": int(first),
    "bvh_frame_last": int(last),
    "scale_leg_ratio": scale,
    "source_leg_length_units": float(src_leg),
    "target_leg_length_m": float(dst_leg),
    "source_height_scaled_m": src_height * scale,
}
with open(args.out_meta, "w") as fh:
    json.dump(meta, fh, indent=1)

scn = bpy.context.scene
scn.render.fps = int(args.fps)
scn.frame_start, scn.frame_end = 1, n_frames
os.makedirs(os.path.dirname(os.path.abspath(args.out_blend)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out_blend))
print(f"RETARGET saved {args.out_blend}")
