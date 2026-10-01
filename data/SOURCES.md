# Data sources

## Human motion capture: CMU Graphics Lab Motion Capture Database (BVH conversion)

* Original data: Carnegie Mellon University Graphics Lab Motion Capture Database, http://mocap.cs.cmu.edu/
  (free for all uses; the site asks that work using it credits "The data used in this project was obtained from
  mocap.cs.cmu.edu. The database was created with funding from NSF EIA-0196217.").
* BVH conversion: Bruce Hahne's MotionBuilder-friendly 2010 re-release (cgspeed.com). His `READMEFIRST.txt` states
  the converted files may be used freely.
* Mirror used to download (a GitHub copy of the same files, with the READMEFIRST):
  https://github.com/una-dinosauria/cmu-mocap  (raw files under `data/0<subject>/<subject>_<trial>.bvh`)

Raw BVH files are not committed (`data/raw/` is gitignored); `python -m retarget_lab fetch-data` re-downloads them.

| Clip | CMU description (from the index) | Used for |
|---|---|---|
| 69_72 | walk forward and pick up object, set down in another place | pick & place |
| 69_73 | walk up to object, lean over, pick up object, set down in another place | bend to pick |
| 69_70 | walk up to object, squat, pick up object, set down in another place | squat to pick |
| 69_69 | walk forward and pick up object, carry back object | carrying |
| 70_01 | carry 5.5lb suitcase | walking with a load |
| 26_09 | bend, pick up | bend to pick |
| 62_20 | closing, moving and opening a box | manipulation |
| 15_06 | lean forward, reach for | reach |
| 14_07 | jump up to grab, reach for, tiptoe | overhead reach |
| 14_01 | boxing | messy / fast |
| 56_02 | vignettes: fists up, wipe window, yawn, stretch, angrily grab, smash against wall | messy / unusual |

Total raw size: about 15 MB.

## Character: MB-Lab 1.8.1 (final release), https://github.com/animate1978/MB-Lab (GPL), Blender 4.0 add-on.
## Software: Blender 4.0.2 (GPL), portable zip from download.blender.org.
