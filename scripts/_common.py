"""Shared helpers for the Blender-side scripts (run with `blender -b -P scripts/<step>.py -- <args>`).

Blender's bundled Python has numpy but no scipy/pandas/yaml, so these scripts only exchange
JSON / NPZ files with the normal-Python analysis package (never import-coupled to it, except for the
numpy-only `retarget_lab.bvh` reader which is on sys.path).
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def script_args(*specs):
    """Parse the arguments after `--`. specs: (flag, kwargs) pairs."""
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    for flag, kw in specs:
        ap.add_argument(flag, **kw)
    return ap.parse_args(argv)


def enable_mblab():
    import addon_utils

    addon_utils.enable("mblab", default_set=True, persistent=True)


def find_character_armature():
    import bpy

    arms = [o for o in bpy.data.objects if o.type == "ARMATURE" and "root" in o.data.bones]
    if not arms:
        raise RuntimeError("No MB-Lab armature found in this .blend")
    return arms[0]
