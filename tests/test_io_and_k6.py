import numpy as np
import pytest
import yaml

from retarget_lab import checks as C
from retarget_lab.io import BoneData, bones_npz_to_parquet, load_bones


def fake_bones(n=60, slide=False):
    names = ["foot_L", "toes_L", "foot_R", "toes_R"]
    head = np.zeros((n, 4, 3))
    head[:, :, 2] = [0.10, 0.02, 0.10, 0.02]  # both feet planted on the floor
    if slide:
        head[:, 0, 0] = np.linspace(0, 0.5, n)  # planted left ankle creeps 0.5 m in 2 s (0.25 m/s)
    rest_head = np.array([[0, 0, 0.10], [0, 0, 0.02], [0, 0, 0.10], [0, 0, 0.02]], float)
    eye = np.tile(np.eye(3), (4, 1, 1))
    return BoneData(names, np.array([-1, 0, -1, 2]), 30.0, head, np.tile(eye, (n, 1, 1, 1)), rest_head,
                    rest_head + [0, 0.05, 0], eye, eye)  # fmt: skip


@pytest.fixture(scope="module")
def th():
    with open("config/thresholds.yaml") as fh:
        return yaml.safe_load(fh)


def test_k6_planted_feet_do_not_slide(robot, th):
    d = C.ClipData("t", 30.0, None, None, None, None, None, fake_bones())
    assert C.k6_foot_sliding(d, th)[0]["value"] == 0


def test_k6_flags_a_creeping_planted_foot(robot, th):
    d = C.ClipData("t", 30.0, None, None, None, None, None, fake_bones(slide=True))
    r = C.k6_foot_sliding(d, th)[0]
    assert (
        r["severity"] == "warn" and 0.4 < r["value"] < 0.6
    )  # left foot slides at ~0.25 m/s, right foot is planted


def test_bone_parquet_roundtrip(tmp_path):
    n, b = 5, 3
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "b.npz", names=np.array(["a", "b", "c"]), parents=np.array([-1, 0, 1]), fps=np.array(30),
             head=rng.normal(size=(n, b, 3)), rot=rng.normal(size=(n, b, 3, 3)), rest_head=rng.normal(size=(b, 3)),
             rest_tail=rng.normal(size=(b, 3)), rest_rot=rng.normal(size=(b, 3, 3)), ref_rot=rng.normal(size=(b, 3, 3)))  # fmt: skip
    bones_npz_to_parquet(tmp_path / "b.npz", tmp_path / "b.parquet")
    got, src = load_bones(tmp_path / "b.parquet"), np.load(tmp_path / "b.npz")
    assert got.names == ["a", "b", "c"] and got.fps == 30
    for key in ("head", "rot", "rest_head", "rest_tail", "rest_rot", "ref_rot"):
        np.testing.assert_allclose(getattr(got, key), src[key])
    assert list(got.parents) == [-1, 0, 1]
