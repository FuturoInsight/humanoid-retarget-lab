"""BVH reader + FK against a tiny hand-built skeleton, and the BVH -> Blender axis conversion."""

import numpy as np
import pytest

from retarget_lab.bvh import BVH_TO_BLENDER, forward_kinematics, parse_bvh, to_blender_space

BVH = """HIERARCHY
ROOT Hips
{
\tOFFSET 0 0 0
\tCHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
\tJOINT Spine
\t{
\t\tOFFSET 0 2 0
\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\tJOINT Arm
\t\t{
\t\t\tOFFSET 3 0 0
\t\t\tCHANNELS 3 Zrotation Yrotation Xrotation
\t\t\tEnd Site
\t\t\t{
\t\t\t\tOFFSET 1 0 0
\t\t\t}
\t\t}
\t}
}
MOTION
Frames: 2
Frame Time: .0083333
0 0 0 0 0 0 0 0 0 0 0 0
10 0 0 90 0 0 0 0 0 0 0 0
"""


@pytest.fixture()
def bvh(tmp_path):
    p = tmp_path / "t.bvh"
    p.write_text(BVH)
    return parse_bvh(str(p))


def test_parse_structure(bvh):
    assert bvh.names == ["Hips", "Spine", "Arm", "Arm_End"]
    assert bvh.parents == [-1, 0, 1, 2]
    assert bvh.n_frames == 2 and bvh.fps == pytest.approx(120, rel=1e-3)
    np.testing.assert_allclose(bvh.offsets[2], [3, 0, 0])


def test_fk_rest_and_rotated_root(bvh):
    _, p = forward_kinematics(bvh)
    np.testing.assert_allclose(p[0, 3], [4, 2, 0])  # 3 + 1 along x, 2 up
    # frame 1: root moved +10 in x and rotated 90 deg about Z: the (4, 2) chain becomes (-2, 4) before translation
    np.testing.assert_allclose(p[1, 3], [10 - 2, 4, 0], atol=1e-9)


def test_rotation_order_is_z_then_y_then_x(tmp_path):
    p = tmp_path / "o.bvh"
    p.write_text(
        BVH.replace(
            "0 0 0 0 0 0 0 0 0 0 0 0\n10 0 0 90 0 0 0 0 0 0 0 0",
            "0 0 0 90 90 0 0 0 0 0 0 0\n0 0 0 0 0 0 0 0 0 0 0 0",
        )
    )
    b = parse_bvh(str(p))
    r, _ = forward_kinematics(b)
    # R = Rz(90) @ Ry(90): x axis -> Ry sends x to -z; Rz leaves z. So root x axis maps to world -z
    np.testing.assert_allclose(r[0, 0] @ [1, 0, 0], [0, 0, -1], atol=1e-9)


def test_blender_conversion_is_proper_and_maps_axes():
    assert np.linalg.det(BVH_TO_BLENDER) == pytest.approx(1.0)
    up = BVH_TO_BLENDER @ [0, 1, 0]  # BVH up -> Blender Z
    fwd = BVH_TO_BLENDER @ [0, 0, 1]  # BVH forward (+Z) -> Blender -Y
    left = BVH_TO_BLENDER @ [1, 0, 0]  # character's left stays +X
    np.testing.assert_allclose([up, fwd, left], [[0, 0, 1], [0, -1, 0], [1, 0, 0]], atol=1e-12)


def test_to_blender_space_moves_rotations_consistently(bvh):
    r, p = forward_kinematics(bvh)
    rb, pb = to_blender_space(r, p)
    for j in range(len(bvh.names)):
        np.testing.assert_allclose(
            rb[0, j] @ (BVH_TO_BLENDER @ [1, 0, 0]), BVH_TO_BLENDER @ (r[0, j] @ [1, 0, 0]), atol=1e-12
        )
    np.testing.assert_allclose(pb[0, 3], BVH_TO_BLENDER @ p[0, 3])
