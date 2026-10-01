# Conventions: axes, units, rotation order

Rotation bugs are the main risk in retargeting, so every convention is written down once here and enforced by tests
(`tests/test_bvh.py`, `tests/test_fk.py`, `tests/test_rotation_decomposition.py`).

## Units
* Lengths: **metres** everywhere after import. CMU BVH files use arbitrary "CMU units" (about 5.6 cm); they are scaled
  once, in `scripts/retarget_to_mblab.py`, by the **leg-length ratio** (MB-Lab hip-to-ankle / CMU hip-to-ankle).
* Angles: radians inside the code, **degrees** in configs, tables and parquet columns (`*_deg`, `*_vel_dps`).
* Time: 120 Hz CMU data is decimated to 30 Hz (every 4th frame).

## Coordinate frames
| Frame | Up | Character faces | Character's left |
|---|---|---|---|
| BVH (CMU) | +Y | +Z | +X |
| Blender / MB-Lab | +Z | -Y | +X |
| Robot (REP-103) | +z | +x | +y |

* BVH -> Blender, applied once (`retarget_lab.bvh.BVH_TO_BLENDER`): `x' = x, y' = -z, z' = y` (a proper rotation).
* Blender -> robot (`to_robot.F_BR`): `x_r = -y_b, y_r = x_b, z_r = z_b`. Rotations transform by conjugation `F R F^T`.

## Rotation order
* BVH rotation channels are applied left to right: `R = R_a1 @ R_a2 @ R_a3` (CMU: Z, Y, X). Verified in `test_bvh.py`.
* Robot chains are serial: `T = T_origin * prod_i( Rot(axis_i, q_i) * Trans(offset_after_i) )`. A joint's
  `offset_after` is the link that follows it, expressed in the joint's rotated frame.
* Decomposition uses the same order: `R = Rot(a1,q1) Rot(a2,q2) Rot(a3,q3)` with the signed axes of the chain
  (e.g. shoulder: pitch `-y`, roll `+x`/`-x`, yaw `+z`/`-z`).

## Robot zero pose and sign conventions
Zero pose: standing, arms hanging straight down, legs straight, feet flat, facing +x.

| Joint | Axis (left side) | Positive angle means |
|---|---|---|
| waist_yaw / head_yaw | +z | turn left (counter-clockwise from above) |
| waist_pitch / head_pitch | +y | lean / nod **forward** |
| shoulder_pitch | -y | arm swings **forward** (flexion) |
| shoulder_roll | +x (left), -x (right) | arm swings **out** to the side (abduction) |
| shoulder_yaw, wrist_yaw | +z (left), -z (right) | internal rotation |
| elbow_pitch | -y | elbow **flexion** (forearm forward) |
| wrist_pitch | -y | hand flexes forward |
| hip_pitch | -y | thigh swings **forward** |
| knee_pitch | +y | knee **flexion** (shank backward) |
| ankle_pitch | -y | toes up |

The right side is the left mirrored across the sagittal plane: the pitch axes (y) are shared, the x/z axes flip, so
the **same limits apply to both sides**.

## Segment orientation used by the decomposition
For every human segment: `O(t) = F (R(t) R_ref^T) F^T S`, where `R(t)` is the MB-Lab bone's world rotation,
`R_ref` the T-pose reference orientation the retarget used (saved on the armature as `retarget_ref_rot`) and `S`
the swing that takes the robot's zero-pose segment direction (down for limbs, up for trunk/head) onto the
reference bone direction. `S` is a 90 degree swing for the arms (the reference is a T-pose) and identity for the foot.

## Rest-pose correction (Mocap -> MB-Lab)
Source and target rest poses differ (CMU T-pose vs MB-Lab A-pose, different bone roll axes). For each mapped bone:
`R_dst(t) = [R_src(t) R_src(T-pose)^T] R_ref`, with `R_ref = swing(rest dir -> source T-pose dir) R_rest`.
Root motion: the hip-joint midpoint's displacement from the T-pose frame, times the leg-length ratio.

## Heading and base
The robot base is a **level** frame at the hip-joint midpoint whose yaw is the pelvis heading (from the pelvis
left axis, which stays well defined while bending forward). Trunk tilt beyond the heading is carried by the waist.
