from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import torch

from eligsir.contracts.types import CameraIntrinsics, Pose
from eligsir.viewers.rvizsplat_export import (
    CANONICAL_TO_ROS_MAP,
    LINK_FROM_OPTICAL_QUAT_WXYZ,
    RVizSplatSnapshot,
    _quat_multiply_wxyz,
    _quat_wxyz_to_rotmat,
    _rotmat_to_quat_wxyz,
    covariance36_from_scale_quat,
    frustum_line_segments,
    pose_map_camera_link,
    serialize_splat_array_cdr,
    sh_band_rotation_matrices,
    snapshot_gaussians_for_rvizsplat,
    transform_pose_basis,
    transform_snapshot_basis,
)


# ----------------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------------


class _DummyGaussians:
    active_sh_degree = 1

    def __init__(self):
        self._xyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._scales = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self._quats = torch.tensor([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
        self._opacities = torch.tensor([[0.25], [0.75]])
        self.params = {
            "sh0": torch.tensor([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]]),
            "shN": torch.arange(2 * 15 * 3, dtype=torch.float32).reshape(2, 15, 3),
        }

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return self._scales

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._quats, dim=1)

    @property
    def get_opacity(self):
        return self._opacities

    @property
    def get_features(self):
        return torch.cat((self.params["sh0"], self.params["shN"]), dim=1)


class _Point:
    x = y = z = 0.0


class _Quat:
    w = 1.0
    x = y = z = 0.0


class _Pose:
    def __init__(self):
        self.position = _Point()
        self.orientation = _Quat()


class _PoseWithCov:
    def __init__(self):
        self.pose = _Pose()
        self.covariance = [0.0] * 36


class _PoseWithCovStamped:
    def __init__(self):
        self.header = SimpleNamespace(stamp=None, frame_id="")
        self.pose = _PoseWithCov()


class _Splat:
    def __init__(self):
        self.pose = None
        self.opacity = 0
        self.harmonics_deg = 0
        self.spherical_harmonics = []


class _SplatArray:
    def __init__(self):
        self.splats = []


# ----------------------------------------------------------------------------
# Snapshot + transform tests
# ----------------------------------------------------------------------------


def test_snapshot_returns_canonical_untransformed():
    """snapshot_gaussians_for_rvizsplat no longer applies the basis transform;
    the worker thread does it.  Positions are unchanged from the Gaussian model."""
    snap = snapshot_gaussians_for_rvizsplat(_DummyGaussians())
    assert snap.positions.shape == (2, 3)
    assert snap.scales.shape == (2, 3)
    assert snap.quaternions.shape == (2, 4)
    assert snap.spherical_harmonics.shape == (2, 12)
    assert snap.sh_degree == 1
    np.testing.assert_allclose(snap.positions[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(snap.spherical_harmonics[0, :3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(snap.spherical_harmonics[0, 3:6], [0.0, 1.0, 2.0])


def test_snapshot_cap_uses_stable_even_sampling():
    snap = snapshot_gaussians_for_rvizsplat(_DummyGaussians(), max_gaussians=1)
    assert snap.positions.shape == (1, 3)
    np.testing.assert_allclose(snap.positions[0], [1.0, 2.0, 3.0])


def test_covariance36_identity_quat_uses_squared_scales():
    cov = covariance36_from_scale_quat(
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    assert cov.shape == (1, 36)
    np.testing.assert_allclose(cov[0, [0, 7, 14]], [0.01, 0.04, 0.09], rtol=1e-6)


def test_transform_pose_basis_matches_snapshot_position_mapping():
    pose = Pose.from_Rt(
        np.eye(3, dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )
    transformed = transform_pose_basis(pose)
    np.testing.assert_allclose(transformed.translation, [3.0, -1.0, -2.0])
    np.testing.assert_allclose(transformed.rotation[:, 0], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(transformed.rotation[:, 1], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(transformed.rotation[:, 2], [1.0, 0.0, 0.0])


# ``test_splat_snapshot_to_msg_matches_rvizsplat_wire_fields`` lived here until
# the #1076 ROS2 hard cut deleted eligsir/viewers/ros2_rviz.py. The
# gsplat_msgs wire-field contract it pinned is now owned by eligsir-ros2
# (eligsir_ros2/egress/rvizsplat_sink.py); the ROS-free serialization this
# module covers (eligsir/viewers/rvizsplat_export.py) stays in core.


# ----------------------------------------------------------------------------
# Vectorized quat + rotation tests
# ----------------------------------------------------------------------------


def test_rotmat_quat_roundtrip_vectorized():
    """For a batch of random rotations, _quat_wxyz_to_rotmat and
    _rotmat_to_quat_wxyz are mutual inverses up to sign (quaternion sign is
    irrelevant)."""
    rng = np.random.default_rng(42)
    n = 50
    # Build random quats via orthogonal matrices
    A = rng.standard_normal((n, 3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q[0]) < 0:
        Q[0] *= -1
    # Encode each Q to a wxyz quat via the single-element helper
    quats = np.array([_rotmat_to_quat_wxyz(Q[i : i + 1])[0] for i in range(n)])
    # Reconstruct R from quat
    R_recon = _quat_wxyz_to_rotmat(quats)
    # Up to quat sign
    for i in range(n):
        if np.dot(quats[i], _rotmat_to_quat_wxyz(R_recon[i : i + 1])[0]) < 0:
            np.testing.assert_allclose(R_recon[i], Q[i], atol=1e-6)
        else:
            np.testing.assert_allclose(R_recon[i], Q[i], atol=1e-6)


def test_quat_multiply_wxyz_matches_matrix_path():
    """q' = q_B ⊗ q should equal rotating q's matrix by B's matrix."""
    rng = np.random.default_rng(0)
    # Random q_B
    A = rng.standard_normal((1, 3, 3))
    Qb, _ = np.linalg.qr(A)
    if np.linalg.det(Qb[0]) < 0:
        Qb[0] *= -1
    q_B = _rotmat_to_quat_wxyz(Qb)
    # Random q's
    n = 20
    A = rng.standard_normal((n, 3, 3))
    Q, _ = np.linalg.qr(A)
    qs = np.array([_rotmat_to_quat_wxyz(Q[i : i + 1])[0] for i in range(n)])
    # Compose
    q_composed = _quat_multiply_wxyz(np.broadcast_to(q_B, qs.shape).copy(), qs)
    # Matrix path
    R_qs = _quat_wxyz_to_rotmat(qs)
    R_B = _quat_wxyz_to_rotmat(q_B)
    R_combined = np.einsum("nij,njk->nik", np.broadcast_to(R_B, R_qs.shape), R_qs)
    R_from_composed = _quat_wxyz_to_rotmat(q_composed)
    # Match up to sign
    for i in range(n):
        a = R_from_composed[i]
        b = R_combined[i]
        np.testing.assert_allclose(a, b, atol=1e-5)


# ----------------------------------------------------------------------------
# SH rotation tests
# ----------------------------------------------------------------------------


def test_sh_band_rotation_matrices_are_orthogonal():
    """All D_l for l=1..3 must be orthogonal with det=+1."""
    D = sh_band_rotation_matrices(CANONICAL_TO_ROS_MAP, degree=3)
    assert len(D) == 3
    np.testing.assert_allclose(D[0] @ D[0].T, np.eye(3), atol=1e-4)
    np.testing.assert_allclose(D[1] @ D[1].T, np.eye(5), atol=1e-4)
    np.testing.assert_allclose(D[2] @ D[2].T, np.eye(7), atol=1e-4)
    assert abs(np.linalg.det(D[0]) - 1.0) < 1e-3
    assert abs(np.linalg.det(D[1]) - 1.0) < 1e-3
    assert abs(np.linalg.det(D[2]) - 1.0) < 1e-3


def test_sh_rotation_preserves_view_dependent_color():
    """Rotating SH coefficients with the band matrices should produce the
    same radiance under the rotated direction (deg 3, 1e-5)."""
    from eligsir.viewers.rvizsplat_export import _eval_real_sh_basis

    # Build a degree-3 snapshot with random non-zero coefficients
    rng = np.random.default_rng(123)
    coeffs_3by16 = rng.standard_normal((3, 16)).astype(np.float32)
    sh_arr = np.zeros((1, 48), dtype=np.float32)
    for c in range(3):
        for k in range(16):
            sh_arr[0, 3 * k + c] = coeffs_3by16[c, k]

    snap = RVizSplatSnapshot(
        positions=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        quaternions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        opacities=np.array([0.5], dtype=np.float32),
        spherical_harmonics=sh_arr,
        sh_degree=3,
    )

    dirs = rng.standard_normal((100, 3)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def get_colors(snapshot, directions):
        n = snapshot.spherical_harmonics.shape[0]
        cc = (snapshot.sh_degree + 1) ** 2
        # F-order: sh[n, c, k] = sh_flat[n, 3*k + c]
        sh = snapshot.spherical_harmonics[:, : cc * 3].reshape(n, 3, cc, order="F")
        out = np.zeros((n, directions.shape[0], 3))
        for i in range(n):
            Y = _eval_real_sh_basis(directions, snapshot.sh_degree)
            for c in range(3):
                out[i, :, c] = Y @ sh[i, c, :]
        return out

    # Rotating the world by B (positions p → B·p) rotates the radiance field
    # the same way, so the rotated splat viewed from map-direction (B·d) must
    # render the SAME colour as the original splat viewed from canonical
    # direction d:  eval(snap2, B·d) == eval(snap, d).  As row vectors,
    # B·d = d @ B.T.  (The earlier convention used d @ B = Bᵀ·d, which is the
    # inverse rotation — it passed only because the code had the matching
    # inverse bug.  See the transpose in transform_snapshot_basis.)
    orig_colors = get_colors(snap, dirs)[0]
    snap2 = transform_snapshot_basis(snap, CANONICAL_TO_ROS_MAP)
    map_dirs = dirs @ CANONICAL_TO_ROS_MAP.T.astype(np.float32)  # B·d
    new_colors = get_colors(snap2, map_dirs)[0]
    np.testing.assert_allclose(new_colors, orig_colors, atol=1e-5)


def test_sh_basis_matches_rvizsplat_shader_constants():
    """Our SH basis must match the RVizSplat GLSL shader exactly, else the
    Wigner-D rotation (derived from our basis) rotates the wrong coefficients.

    Pinned to gsplat_rviz_plugin/ogre_media/materials/glsl120/splat.vert
    (pinned commit 30311c58): SH_C1, SH_C2_0..4, SH_C3_0..6.  The shader's
    band-1/2/3 polynomials also match _eval_real_sh_basis term-for-term
    (verified: -C1*y, +C1*z, -C1*x; C2_0*xy, C2_1*yz, C2_2*(2zz-xx-yy),
    C2_3*xz, C2_4*(xx-yy); etc.)."""
    from eligsir.viewers.rvizsplat_export import _SH_C1, _SH_C2, _SH_C3

    np.testing.assert_allclose(_SH_C1, 0.4886025119029199)
    np.testing.assert_allclose(
        _SH_C2,
        [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396],
    )
    np.testing.assert_allclose(
        _SH_C3,
        [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435],
    )


def test_sh_rotation_dc_band_invariant():
    """DC (band 0) must NOT change under world rotation."""
    rng = np.random.default_rng(7)
    coeffs = rng.standard_normal((3, 16)).astype(np.float32)
    sh_arr = np.zeros((1, 48), dtype=np.float32)
    for c in range(3):
        for k in range(16):
            sh_arr[0, 3 * k + c] = coeffs[c, k]

    snap = RVizSplatSnapshot(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        quaternions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.array([[0.1, 0.1, 0.1]], dtype=np.float32),
        opacities=np.array([0.5], dtype=np.float32),
        spherical_harmonics=sh_arr,
        sh_degree=3,
    )
    snap2 = transform_snapshot_basis(snap, CANONICAL_TO_ROS_MAP)
    # DC is sh[0:3] in the RGB-interleaved layout (ch=0,1,2 of band 0)
    np.testing.assert_allclose(
        snap2.spherical_harmonics[0, 0:3], snap.spherical_harmonics[0, 0:3], atol=1e-5
    )


# ----------------------------------------------------------------------------
# Frame helpers
# ----------------------------------------------------------------------------


def test_pose_map_camera_link_identity_for_zero_translation():
    """Identity c2w → translation (0,0,0); rotation should be Bᵀ·I·B = I."""
    p = Pose.from_Rt(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    pl = pose_map_camera_link(p)
    np.testing.assert_allclose(pl.translation, [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pl.rotation, np.eye(3), atol=1e-6)


def test_pose_map_camera_link_translation_basis():
    p = Pose.from_Rt(np.eye(3, dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    pl = pose_map_camera_link(p)
    np.testing.assert_allclose(pl.translation, [3.0, -1.0, -2.0], atol=1e-6)


def test_link_from_optical_quat_is_unit_and_consistent():
    """LINK_FROM_OPTICAL_QUAT_WXYZ should be a unit quaternion matching the
    static rotation that maps optical (RDF) to link (FLU)."""
    q = LINK_FROM_OPTICAL_QUAT_WXYZ
    assert q.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-5)
    # Reconstruct the rotation and verify against B
    R = _quat_wxyz_to_rotmat(q.astype(np.float64).reshape(1, 4))[0]
    expected = CANONICAL_TO_ROS_MAP.astype(np.float64)
    np.testing.assert_allclose(R, expected, atol=1e-5)


def test_frustum_line_segments_shape_and_geometry():
    """8 segments: 4 apex→far rays + 4 far-plane edges.  All endpoints
    must be in world frame."""
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    # Camera at world origin, canonical axes (x-right, y-down, z-forward).
    # After transform_pose_basis, the map-frame rotation is B, so
    # optical +z (forward) maps to map +x.
    p = Pose.from_Rt(np.eye(3, dtype=np.float32), np.array([0.0, 0.0, 0.0], dtype=np.float32))
    p_map = transform_pose_basis(p)
    segs = frustum_line_segments(p_map, intr, depth=0.5)
    assert segs.shape == (8, 2, 3)
    # 4 rays share the apex at world origin (camera position in map frame = (0,0,0))
    for i in range(4):
        np.testing.assert_allclose(segs[i, 0], [0.0, 0.0, 0.0], atol=1e-5)
    # The 4 far-plane edges form a closed rectangle: edge 0 start == edge 3 end
    np.testing.assert_allclose(segs[4, 0], segs[7, 1], atol=1e-5)
    # All 4 far corners are at x = depth (since optical +z → map +x via B)
    for i in range(4, 8):
        for endpoint in (0, 1):
            assert abs(segs[i, endpoint, 0] - 0.5) < 1e-4


# ----------------------------------------------------------------------------
# CDR serializer tests
# ----------------------------------------------------------------------------


def _make_snapshot(N: int, sh_degree: int) -> RVizSplatSnapshot:
    K = (sh_degree + 1) ** 2 * 3
    return RVizSplatSnapshot(
        positions=np.arange(N * 3, dtype=np.float32).reshape(N, 3) * 0.1,
        quaternions=np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 1)
        ),
        scales=np.full((N, 3), 0.1, dtype=np.float32),
        opacities=np.full(N, 0.5, dtype=np.float32),
        spherical_harmonics=np.arange(N * K, dtype=np.float32).reshape(N, K),
        sh_degree=sh_degree,
    )


def test_serialize_cdr_empty_returns_8_bytes():
    snap = RVizSplatSnapshot(
        positions=np.zeros((0, 3), dtype=np.float32),
        quaternions=np.zeros((0, 4), dtype=np.float32),
        scales=np.zeros((0, 3), dtype=np.float32),
        opacities=np.zeros((0,), dtype=np.float32),
        spherical_harmonics=np.zeros((0, 0), dtype=np.float32),
        sh_degree=0,
    )
    b = serialize_splat_array_cdr(snap, stamp_sec=0, stamp_nanosec=0, frame_id="map")
    assert len(b) == 8
    # XCDR1 encap (0x00 dummy + 0x01 LE|PLAIN_CDR) + 2 bytes options + 4-byte N=0
    assert b[:4] == b"\x00\x01\x00\x00"
    assert b[4:8] == b"\x00\x00\x00\x00"


def test_serialize_cdr_layout_deg0_odd_k():
    """Deg 0: K=3.  Per the rclpy-verified layout, record-relative offsets
    are: stamp(8) + fid_len_field(4) + "map\\0"(4) + pad(4) + position(24)
    + orientation(32) + covariance(288) + opacity(1) + deg(1) + pad(2)
    + sh_length(4) = 372 fixed + 12 (K=3) = 384.  Total = 8 + 384 = 392.
    pos_off (record-relative) = 20."""
    snap = _make_snapshot(N=1, sh_degree=0)
    b = serialize_splat_array_cdr(snap, stamp_sec=0x1A2B, stamp_nanosec=0x3C4D, frame_id="map")
    # Top-level prefix: XCDR1 encap (4 bytes) + N (4 bytes) = 8 bytes
    assert b[:4] == b"\x00\x01\x00\x00"  # XCDR1: dummy + LE|PLAIN_CDR + options
    n = struct.unpack_from("<I", b, 4)[0]
    assert n == 1
    # Total length: 8 (prefix) + n * (372 + 4*K) = 8 + 384 = 392
    assert len(b) == 8 + 384
    # Record 0 starts at absolute offset 8
    rec = 8
    # Per-record stamp
    assert struct.unpack_from("<i", b, rec)[0] == 0x1A2B
    assert struct.unpack_from("<I", b, rec + 4)[0] == 0x3C4D
    # Per-record frame_id length (INCLUDES NUL terminator) and data
    assert struct.unpack_from("<I", b, rec + 8)[0] == 4
    assert b[rec + 12 : rec + 16].rstrip(b"\x00") == b"map"
    # Position float64 at record-relative 20 = absolute 28
    pos = struct.unpack_from("<3d", b, rec + 20)
    np.testing.assert_allclose(pos, snap.positions[0])
    # Orientation float64 xyzw at rec+44 (20+24=44, absolute 52)
    quat = struct.unpack_from("<4d", b, rec + 44)
    np.testing.assert_allclose(quat, [0.0, 0.0, 0.0, 1.0])  # wxyz(1,0,0,0) → xyzw(0,0,0,1)
    # Opacity at rec + 364 (44+32+288=364, absolute 372)
    assert b[rec + 364] == 127
    # Harmonics deg at rec + 365
    assert b[rec + 365] == 0
    # SH length uint32 at rec + 368
    sh_len = struct.unpack_from("<I", b, rec + 368)[0]
    assert sh_len == 3
    # SH data: K=3 float32 at rec+372
    sh = struct.unpack_from("<3f", b, rec + 372)
    np.testing.assert_allclose(sh, snap.spherical_harmonics[0])


def test_serialize_cdr_layout_deg1_even_k():
    """Deg 1: K=12, fid_len=4 ("map\0").
    Record 0 (abs=8):  p0 = (-(8+12+4 - 4)) % 8 = (-20)%8 = 4  → size = 12+4+4+344+8+48 = 420.
    Record 1 (abs=428): p1 = (-(428+12+4 - 4)) % 8 = (-440)%8 = 0 → size = 12+4+0+344+8+48 = 416.
    Total = 8 + 420 + 416 = 844."""
    snap = _make_snapshot(N=2, sh_degree=1)
    b = serialize_splat_array_cdr(snap, stamp_sec=0, stamp_nanosec=0, frame_id="map")
    n = struct.unpack_from("<I", b, 4)[0]
    assert n == 2
    assert len(b) == 8 + 420 + 416  # = 844
    # Record 1 starts at 8 + 420 = 428
    rec1 = 8 + 420
    assert struct.unpack_from("<I", b, rec1 + 8)[0] == 4
    assert b[rec1 + 12 : rec1 + 16].rstrip(b"\x00") == b"map"
    # p1=0: pos_off=16, quat_off=40, cov_off=72, op_off=360, sh_len_off=364
    sh_len = struct.unpack_from("<I", b, rec1 + 364)[0]
    assert sh_len == 12


def test_serialize_cdr_xyzw_reorder():
    """Snapshot quats are wxyz; CDR stores xyzw.  wxyz=(1,0,0,0) → xyzw=(0,0,0,1)."""
    snap = _make_snapshot(N=1, sh_degree=0)
    # set quaternion to (w=0.5, x=0.5, y=0.5, z=0.5)
    snap.quaternions[0] = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    b = serialize_splat_array_cdr(snap, stamp_sec=0, stamp_nanosec=0, frame_id="map")
    # Record 0 starts at offset 8.  Orientation at rec+44 (record-relative).
    quat = struct.unpack_from("<4d", b, 8 + 44)
    # xyzw order
    np.testing.assert_allclose(quat, [0.5, 0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Golden-bytes regression: the serializer output must remain byte-identical
# across refactors (e.g. the strided-fill vectorization).  Fixtures were
# dumped from the validated serializer; see tests/_rvizsplat_fixtures.py.
# ---------------------------------------------------------------------------
import os  # noqa: E402

import pytest  # noqa: E402

from _rvizsplat_fixtures import (  # noqa: E402
    CASES,
    FRAME_ID,
    STAMP_NANOSEC,
    STAMP_SEC,
    fixture_path,
    make_snapshot,
)


def test_encode_compact_dc_v1_layout():
    from eligsir.viewers.rvizsplat_export import (
        COMPACT_DC_V1_STRIDE,
        encode_compact_dc_v1,
    )

    snap = RVizSplatSnapshot(
        positions=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        quaternions=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        scales=np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
        opacities=np.array([0.25], dtype=np.float32),
        spherical_harmonics=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        sh_degree=0,
    )
    payload = encode_compact_dc_v1(snap)
    assert len(payload) == COMPACT_DC_V1_STRIDE
    np.testing.assert_allclose(np.frombuffer(payload[0:12], dtype="<f4"), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        np.frombuffer(payload[12:24], dtype="<f2").astype(np.float32),
        [4.0, 0.0, 0.0, 9.0, 0.0, 16.0],
    )
    assert list(payload[24:28]) == [128, 128, 128, 64]
    assert payload[28:32] == b"\x00\x00\x00\x00"


def test_split_bytes_chunks_and_tile_keys():
    from eligsir.viewers.rvizsplat_export import split_bytes_chunks, tile_keys_for_positions

    assert split_bytes_chunks(b"abcdef", 2) == [b"ab", b"cd", b"ef"]
    assert split_bytes_chunks(b"", 4) == [b""]

    positions = np.array([[0.0, 0.9, -0.1], [1.0, -1.1, 2.2]], dtype=np.float32)
    keys = tile_keys_for_positions(positions, 1.0)
    np.testing.assert_array_equal(keys, np.array([[0, 0, -1], [1, -2, 2]], dtype=np.int32))


@pytest.mark.parametrize("sh_degree,n", CASES)
def test_serialize_cdr_golden_bytes(sh_degree, n):
    """Serializer output must reproduce the committed golden fixture exactly."""
    path = fixture_path(sh_degree, n)
    assert os.path.exists(path), f"missing fixture {path} (regenerate)"
    with open(path, "rb") as f:
        expected = f.read()
    snap = make_snapshot(sh_degree, n)
    actual = serialize_splat_array_cdr(snap, STAMP_SEC, STAMP_NANOSEC, FRAME_ID)
    assert actual == expected, (
        f"deg{sh_degree} n{n}: {len(actual)} bytes != golden {len(expected)} bytes"
    )


def test_serialize_cdr_empty_array():
    """Empty snapshot → 4-byte encapsulation + uint32 N=0 (8 bytes total)."""
    snap = RVizSplatSnapshot(
        positions=np.zeros((0, 3), np.float32),
        quaternions=np.zeros((0, 4), np.float32),
        scales=np.zeros((0, 3), np.float32),
        opacities=np.zeros((0,), np.float32),
        spherical_harmonics=np.zeros((0, 3), np.float32),
        sh_degree=0,
    )
    b = serialize_splat_array_cdr(snap, STAMP_SEC, STAMP_NANOSEC, FRAME_ID)
    assert b == b"\x00\x01\x00\x00\x00\x00\x00\x00"


def test_rotmat_to_quat_wxyz_all_branches_roundtrip():
    """Vectorized rotmat->quat must handle all 4 Shepperd branches.

    A masking bug previously made the r11-dominant branch (mask2) unreachable,
    so 180-deg-about-Y-like rotations took the wrong (r22) formula. Roundtrip
    R -> quat -> R must hold for every branch, including r11-dominant.
    """
    def _Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

    def _Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    def _Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    mats = [
        np.eye(3),                       # trace>0 branch
        _Rx(np.pi),                      # r00-dominant (diag 1,-1,-1)
        _Ry(np.pi),                      # r11-dominant (diag -1,1,-1)  <- was broken
        _Rz(np.pi),                      # r22-dominant (diag -1,-1,1)
        _Rx(np.pi / 2), _Ry(np.pi / 2), _Rz(np.pi / 2),
    ]
    rng = np.random.default_rng(0)
    for _ in range(200):  # random SO(3) via QR
        q, r = np.linalg.qr(rng.standard_normal((3, 3)))
        q = q @ np.diag(np.sign(np.diag(r)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        mats.append(q)

    R = np.stack(mats).astype(np.float64)
    quats = _rotmat_to_quat_wxyz(R)
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1.0, atol=1e-6)
    R_rt = _quat_wxyz_to_rotmat(quats)
    np.testing.assert_allclose(R_rt, R, atol=1e-5)

    # Explicit guard on the formerly-broken r11-dominant case.
    q_y = _rotmat_to_quat_wxyz(_Ry(np.pi)[None])[0]
    np.testing.assert_allclose(_quat_wxyz_to_rotmat(q_y[None])[0], _Ry(np.pi), atol=1e-5)


def test_drawn_map_orientation_and_covariance_are_consistently_rebased():
    """The published SplatArray's orientation (B·R) and covariance (B·Σ·Bᵀ) must
    describe the SAME ellipsoid in the ROS map frame — i.e. the covariance is
    computed from the *transformed* quaternions, not the canonical ones.

    Guards the egress drawn-map consistency: positions, orientation, and
    covariance all rebase by B together.
    """
    from eligsir.viewers.rvizsplat_export import (
        RVizSplatSnapshot,
        covariance36_from_scale_quat,
        transform_snapshot_basis,
        _quat_wxyz_to_rotmat,
    )
    B = np.asarray(CANONICAL_TO_ROS_MAP, dtype=np.float64)
    rng = np.random.default_rng(1)
    n = 16
    # Random canonical quats (wxyz) + scales.
    q = rng.standard_normal((n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    scales = np.abs(rng.standard_normal((n, 3))).astype(np.float32) + 0.1
    snap = RVizSplatSnapshot(
        positions=rng.standard_normal((n, 3)).astype(np.float32),
        quaternions=q.astype(np.float32),
        scales=scales,
        opacities=np.ones(n, np.float32),
        spherical_harmonics=np.zeros((n, 3), np.float32),
        sh_degree=0,
    )
    # Canonical covariance per splat.
    R_can = _quat_wxyz_to_rotmat(q)
    Sigma_can = np.einsum("nij,nj,nkj->nik", R_can, scales.astype(np.float64) ** 2, R_can)

    snap_x = transform_snapshot_basis(snap, CANONICAL_TO_ROS_MAP)
    cov36 = covariance36_from_scale_quat(snap_x.scales, snap_x.quaternions)
    Sigma_ros = cov36.reshape(n, 6, 6)[:, :3, :3]

    expected = np.einsum("ij,njk,lk->nil", B, Sigma_can, B)  # B·Σ·Bᵀ
    np.testing.assert_allclose(Sigma_ros, expected, atol=1e-5)
