# Character setup (MB-Lab)

**Path used: scripted (headless).** MB-Lab's creation flow can be driven from a script, so no UI clicks are needed.

```bash
# Blender 4.0.2 portable + MB-Lab 1.8.1 (final release) in tools/ (gitignored); then:
export BLENDER_USER_SCRIPTS=$PWD/tools/blender_user      # contains addons/mblab
tools/blender/blender.exe -b -P scripts/create_character.py -- --character m_ca01 --out assets/mblab_character.blend
```

`scripts/create_character.py` enables the add-on, sets `scene.mblab_character_name = "m_ca01"` (male, caucasian base),
calls `bpy.ops.mbast.init_character()` (the operator behind the "Create character" button) and saves the file.
`make character` / `python -m retarget_lab character` does the same.

## Reproducing the install
1. Download `blender-4.0.2-windows-x64.zip` from download.blender.org/release/Blender4.0/ and unzip to `tools/blender/`.
2. Download MB-Lab `1_8_1` (github.com/animate1978/MB-Lab/releases) and copy the repo contents to
   `tools/blender_user/addons/mblab/` (folder name must be a valid module name).
3. Run the command above. The scripts set `BLENDER_USER_SCRIPTS` themselves when launched by `retarget_lab`.

## If the scripted path ever breaks (manual fallback)
1. Open Blender 4.0, install `tools/MB-Lab` as an add-on (Edit > Preferences > Add-ons > Install).
2. In the 3D viewport sidebar (N) open the **MB-Lab** tab, choose the character *m_ca01* and press **Create character**.
3. Delete the default cube/camera/light and save as `assets/mblab_character.blend`.

## What you get
Mesh `m_ca01` (about 18k vertices, 1.83 m) skinned to the armature `m_ca01_skeleton` (71 bones; hierarchy in
`docs/mblab_hierarchy.svg`). The armature object transform is identity, which the retarget script asserts. Cosmetic
modifiers (corrective smooth, subsurf, displacement) are switched off for rendering (they cost seconds per frame and
change nothing about the skeleton).
