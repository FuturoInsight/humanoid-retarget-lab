import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def robot():
    from retarget_lab.skeleton import load_robot

    return load_robot(ROOT / "config" / "robot_skeleton.yaml")
