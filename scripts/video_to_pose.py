"""STRETCH: phone video -> MediaPipe Pose 3D landmarks -> CMU-style BVH that enters the same pipeline.

Normal Python (not Blender). Needs the optional extras:  pip install -e ".[stretch]"   (mediapipe, opencv-python)

    python scripts/video_to_pose.py my_task.mp4 --out data/own/my_task.bvh

then add to config/default.yaml:
    - {id: own_my_task, bvh: data/own/my_task.bvh, task: "my demonstration", kind: own}

Status: the landmark -> BVH conversion (src/retarget_lab/pose_to_bvh.py) is unit-tested with synthetic landmarks;
the MediaPipe capture below has NOT been run in this repo's build environment (no phone video was available), so
treat it as a starting point. Single-camera pose is noisy: expect K8 (jitter) and K6 (foot sliding) to flag it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retarget_lab.bvh import parse_bvh  # noqa: E402
from retarget_lab.pose_to_bvh import landmarks_to_bvh, mediapipe_to_bvh_space  # noqa: E402


def extract_landmarks(video: str) -> tuple[np.ndarray, float]:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    with mp.solutions.pose.Pose(model_complexity=2, smooth_landmarks=True) as pose:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_world_landmarks is None:
                frames.append(frames[-1] if frames else np.zeros((33, 3)))
                continue
            frames.append(np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks.landmark]))
    cap.release()
    return np.array(frames), float(fps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--template", default="data/raw/69_72.bvh", help="CMU BVH whose skeleton + T-pose frame to reuse"
    )
    args = ap.parse_args()
    lm, fps = extract_landmarks(args.video)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(Path(args.out).with_suffix(".landmarks.npy"), lm)
    landmarks_to_bvh(mediapipe_to_bvh_space(lm), parse_bvh(args.template), fps, args.out)
    print(f"wrote {args.out}: {len(lm)} frames at {fps:.1f} fps")
