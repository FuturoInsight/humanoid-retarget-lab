# DECISIONS

Every deviation from `humanoid-retarget-lab-spec.md`, and every non-obvious choice, with the reason.

## Environment and versions
| Item | Choice | Why |
|---|---|---|
| Blender | **4.0.2** portable zip in `tools/blender/` (gitignored) | MB-Lab 1.8.1 requires Blender >= 4.0 and its final release was built against 4.0; a newer Blender risks add-on API breakage. |
| MB-Lab | **1.8.1** (tag `1_8_1`, 2024-06-08, the final release; development moved to CharMorph) | Latest release. Installed to `tools/blender_user/addons/mblab`, found through the `BLENDER_USER_SCRIPTS` env var, because a `portable/` folder next to the exe was not picked up by 4.0.2. |
| Character creation | **Scripted** (`bpy.ops.mbast.init_character()` with `scene.mblab_character_name = "m_ca01"`) | It works headless, so no UI path and no committed `.blend` is needed (`docs/character_setup.md` still documents the manual clicks). `assets/mblab_character.blend` is regenerated, not committed. |
| Python (analysis) | **3.14** venv, spec said 3.11 | Only 3.14 was installed on the machine; all dependencies had wheels. `requires-python >= 3.11` is kept in `pyproject.toml`. |
| Graphviz | **Not used**: hierarchy diagrams are drawn by a small pure-Python SVG layout (`skeleton.hierarchy_svg`) | Graphviz was not installed and needs an admin install on Windows. The spec said "e.g. via Graphviz". |
| ffmpeg | `imageio-ffmpeg` (bundled binary) | No system ffmpeg. MP4 via libx264, GIF via Pillow. |

## Pipeline
* **BVH is parsed by my own numpy-only reader** (`retarget_lab.bvh`), not Blender's BVH importer. Reasons: MB-Lab's own release notes call BVH import buggy; the importer rewrites rest poses/bone rolls in ways that are hard to reason about; and a plain FK lets the rotation maths be unit-tested. The Blender script imports that module (it only needs numpy, which Blender bundles). The spec's "import the BVH" is therefore done by a script inside Blender, not by `bpy.ops.import_anim.bvh`.
* **Frame 0 of the CMU files is a T-pose** (added by the Hahne conversion); it is the source reference for the rest-pose correction.
* **Retarget maths in `scripts/retarget_to_mblab.py`** is numpy and writes `matrix_basis` per pose bone, instead of constraints + bake: deterministic and independent of depsgraph evaluation order. The spec's "keep bpy code thin" is met in spirit (rotations are plain numpy) but the retarget script does contain real logic; the robot-side logic is fully in the package.
* **Root motion** follows the hip-joint midpoint (mean of the two thigh joints), not the `Hips`/`pelvis` points. CMU's `Hips` sits above the thigh joints and MB-Lab's `pelvis` head sits below them, so copying the pelvis point lifted/sank the character whenever the pelvis rotated. (Found by comparing exported bone heights against the source skeleton.)
* **Zero-length CMU segments** (`Hips->LowerBack`, `Spine1->Neck`, `Hand->FingerBase`) cannot give a direction; the bone map uses the next non-degenerate child instead (`bone_map_cmu_to_mblab.yaml`).
* **Blender -> Python exchange is NPZ**, converted to Parquet in the analysis environment (`io.bones_npz_to_parquet`). Blender's Python has no pyarrow. Robot config goes the other way as JSON because Blender has no PyYAML.
* **Clips**: 11 CMU clips (spec: 8-12), a window of at most 10 s each (`max_sec`, `start_sec`), decimated 120 -> 30 Hz. Total raw data about 15 MB.
* **Mocap stick-figure panel is scaled by the leg-length ratio**, so its torso/arms are proportioned like the CMU subject, not like the MB-Lab character (CMU subject: shorter torso, longer arms relative to legs). This is visible in the renders and is exactly the proportion mismatch the IK stage compensates for.
* **Robot panel shows the raw (limits ignored) motion** with past-limit links in safety orange; showing the clamped motion would never show a joint past its limit.
* **Cosmetic MB-Lab modifiers (corrective smooth, subsurf, displace) are disabled for rendering**: they took ~4 s per frame and do not affect the skeleton. Eevee (legacy `BLENDER_EEVEE`, 12 samples) renders in about 0.1 s per panel after that.
* **Robot panel geometry**: an armature with one bone per link is posed every frame, and capsule/ball meshes are placed on the posed bones (rather than parented to bones, whose bone-tail parenting offset is easy to get wrong). Highlighting swaps a steel mesh for an orange twin via `hide_render`.

## Robot retargeting
* **Decomposition**: exact Euler for 3-DoF joints (shoulder, hip), but solved by damped Gauss-Newton from the previous frame because closed-form Euler flips branches and degenerates at the shoulder's +-90 degree roll (an arm held out sideways is common). Joints with fewer DoF than the human joint (waist, head, wrist, elbow, knee, ankle) use a tracked least-squares fit instead of dropping a third Euler angle: dropping it made the waist jump 180 degrees when the trunk passed 90 degrees of forward bend. Documented in `kinematics/fk.py`; tests in `tests/test_rotation_decomposition.py`.
* **Segment reference**: rotation deltas are measured from the retarget's own T-pose reference orientation (`ref_rot`, stored on the armature), not the MB-Lab rest pose, so a person standing in a neutral pose gives about-zero robot angles.
* **Base**: level frame at the hip midpoint with the pelvis heading; trunk tilt is carried by the 2-DoF waist.
* **Default trajectories use the IK-refined method**; the joint-angle-only method is computed for the same clips and compared in the report (section 6). IK is damped least squares with a pull toward the joint-angle solution (keeps wrist/elbow redundancy human-like and smooth).
* **Clamped = position clip + velocity slew limit** (the robot lags a too-fast demonstration instead of teleporting). The spec only said "limits enforced".
* **K1** compares FK of the *chosen method's* angles with the scaled human hand target, so with the default IK method it reports the solver residual; the joint-angle-only error is reported in section 6.
* **K6** is measured on the MB-Lab retarget (human side), as in the spec: a "planted" frame is one where the ankle is near its standing height and slow; a planted run that drifts more than a threshold counts as sliding.
* **K7** uses the executed (clamped) motion; capsule proxy covers forearm/hand vs torso only.
* **Verdict thresholds are engineering choices** in `config/thresholds.yaml`, not calibrated against any real robot.

## Export
* The companion projects' exact LeRobot-style schema was not available to me. `export/` follows LeRobot conventions (`meta/info.json`, `meta/episodes.csv`, `meta/tasks.csv`, `data/frames.csv`; `episode_index`, `frame_index`, `timestamp`, `task_index`, `observation.state.*`, `action.*`). The state is the **clamped** (executable) trajectory; `action` is the next frame's state. If the QA pipeline's adapter expects other column names, the mapping is a few lines in `export.py`. **I did not run the QA pipeline's adapter** (it is not in this workspace), so "loads in the QA pipeline's adapter" is untested.

## Not done / partial
* `scripts/video_to_pose.py` (MediaPipe stretch): the landmark -> BVH conversion is implemented and unit-tested with synthetic landmarks; the MediaPipe capture itself was not run (no phone video, and mediapipe was not installed for Python 3.14).
* A single character (`m_ca01`) and a single robot skeleton; no finger/hand retargeting.

## Threshold history (disclosure)
* First full run: almost every clip was NOT FEASIBLE. Causes were partly my definitions, so I changed them before the final run: K6 (the first definition counted the swing foot as "in contact", giving 76-100 % sliding; contact is now height + vertical speed only), K4 fail (mean 8 to 10 cm, max 30 to 100 cm, because the slew-limit lag at fast bends tripped the max), K3 fail share 30 to 35 %, robot waist-pitch limit 60 to 75 deg and velocity 90 to 120 deg/s. The warn thresholds were not touched, which is why no clip is plain USABLE.
* First-frame fit for 1-2 DoF joints now uses several starts: `cmu_26_09` first landed in a local minimum 180 deg off for the right wrist. The 150 deg twist that remains is in the source data.
