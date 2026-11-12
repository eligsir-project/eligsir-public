"""RVizSplat-compatible Gaussian snapshot helpers.

RVizSplat consumes ``gsplat_msgs/SplatArray``.  This module keeps the tensor
export and wire-layout math independent of ROS imports so it can be unit-tested
without a ROS environment.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from eligsir.contracts.conventions import CANONICAL_TO_ROS_MAP
from eligsir.contracts.types import CameraIntrinsics, Pose

COMPACT_DC_V1_ENCODING = "compact_dc_fp16_cov_rgba_v1"
COMPACT_DC_V1_STRIDE = 32


@dataclass(frozen=True)
class RVizSplatSnapshot:
    positions: np.ndarray
    quaternions: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    spherical_harmonics: np.ndarray
    sh_degree: int
    # Cached per-band SH rotation matrices for the basis that has been baked
    # into this snapshot. Empty list means "SH is in canonical form".
    sh_rotations: Tuple[np.ndarray, ...] = field(default_factory=tuple)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a single wxyz quaternion to a 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


# Static link→optical rotation derived from CANONICAL_TO_ROS_MAP at import
# time.  link is REP-103/105 FLU (x-forward, y-left, z-up).  optical is the
# camera_color_optical_frame (RDF: x-right, y-down, z-forward).  The static
# rotation from link→optical is B.  As a quaternion (wxyz):
#   B = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
_LINK_FROM_OPTICAL_ROT = CANONICAL_TO_ROS_MAP.astype(np.float64)



def _rotmat_to_quat_wxyz_single(R: np.ndarray) -> np.ndarray:
    """Single 3x3 → wxyz quaternion (numerically robust Shepperd)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        )
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array(
            [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        )
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array(
            [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        )
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array(
        [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    )


LINK_FROM_OPTICAL_QUAT_WXYZ = _rotmat_to_quat_wxyz_single(_LINK_FROM_OPTICAL_ROT).astype(np.float32)


def _feature_tensor(gaussians):
    # A eligsir.map.views.GaussianMapReadView exposes sh0/shN directly
    # (no get_features/params/legacy-attr dispatch needed) -- checked first
    # since it's the only shape a read-only view can take.
    if hasattr(gaussians, "sh0") and hasattr(gaussians, "shN"):
        return gaussians.sh0, gaussians.shN
    if hasattr(gaussians, "get_features"):
        return gaussians.get_features
    if hasattr(gaussians, "params") and "sh0" in gaussians.params and "shN" in gaussians.params:
        return gaussians.params["sh0"], gaussians.params["shN"]
    if hasattr(gaussians, "_features_dc") and hasattr(gaussians, "_features_rest"):
        return gaussians._features_dc, gaussians._features_rest
    raise AttributeError("Gaussian model does not expose SH feature tensors")


def _positions(gaussians):
    """Accepts either a raw model (``get_xyz``) or a
    ``eligsir.map.views.GaussianMapReadView`` (``means``)."""
    return gaussians.means if hasattr(gaussians, "means") else gaussians.get_xyz


def _scales(gaussians):
    return gaussians.scales if hasattr(gaussians, "means") else gaussians.get_scaling


def _quaternions(gaussians):
    return gaussians.quaternions if hasattr(gaussians, "means") else gaussians.get_rotation


def _opacities(gaussians):
    return gaussians.opacities if hasattr(gaussians, "means") else gaussians.get_opacity


def _as_feature_tensor(gaussians):
    import torch

    features = _feature_tensor(gaussians)
    if isinstance(features, tuple):
        sh0, shn = features
        return torch.cat((sh0, shn), dim=1)
    return features


def _selected_indices(count: int, max_gaussians: int, device):
    if max_gaussians <= 0 or count <= max_gaussians:
        return None
    import torch

    return torch.linspace(0, count - 1, steps=max_gaussians, device=device).round().long()


def snapshot_gaussians_for_rvizsplat(gaussians, max_gaussians: int = 0) -> RVizSplatSnapshot:
    """Detach a Gaussian model into the RVizSplat wire layout.

    ``max_gaussians=0`` exports the full model.  A positive cap uses stable
    evenly-spaced sampling so live previews remain deterministic and cheap.
    """
    import torch

    with torch.no_grad():
        positions_t = gaussians.get_xyz.detach()
        count = int(positions_t.shape[0])
        idx = _selected_indices(count, int(max_gaussians), positions_t.device)

        def take(t):
            t = t.detach()
            return t if idx is None else t.index_select(0, idx)

        positions = take(positions_t).contiguous().cpu().numpy().astype(np.float32, copy=False)
        scales = take(gaussians.get_scaling).contiguous().cpu().numpy().astype(np.float32, copy=False)
        quats = take(gaussians.get_rotation).contiguous().cpu().numpy().astype(np.float32, copy=False)
        opacities = (
            take(gaussians.get_opacity).reshape(-1).clamp(0.0, 1.0).contiguous().cpu().numpy()
            .astype(np.float32, copy=False)
        )

        features = take(_as_feature_tensor(gaussians))
        active_degree = int(getattr(gaussians, "active_sh_degree", 0))
        max_coeffs = int(features.shape[1])
        max_degree = int(np.floor(np.sqrt(max_coeffs) - 1))
        sh_degree = max(0, min(active_degree, max_degree, 3))
        coeff_count = (sh_degree + 1) ** 2
        sh = (
            features[:, :coeff_count, :]
            .contiguous()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
            .reshape(features.shape[0], coeff_count * 3)
        )

    return RVizSplatSnapshot(
        positions=positions,
        quaternions=quats,
        scales=scales,
        opacities=opacities,
        spherical_harmonics=sh,
        sh_degree=sh_degree,
        sh_rotations=(),
    )


def snapshot_gaussians_for_rvizsplat_packed(gaussians, max_gaussians: int = 0) -> RVizSplatSnapshot:
    """WS4: identical output to :func:`snapshot_gaussians_for_rvizsplat`, but
    issues exactly ONE device->host transfer instead of five.

    positions/scales/quats/opacities/sh are all concatenated into a single
    contiguous tensor on ``gaussians``'s device, downloaded with one
    ``.cpu().numpy()`` call, then split back apart on the host. Parity with
    the five-download version is asserted in
    tests/test_rvizsplat_export_packed.py. The GPU-side basis/quaternion/SH
    rotation math in :func:`transform_snapshot_basis` (still CPU numpy) and
    the async-pinned-buffer publish path are NOT changed here -- those need
    a live RViz run to validate the main-thread-stall improvement and are
    left for a follow-up once the machine isn't running a concurrent GPU
    training job (see docs/2026-07-26-transfer-audit.md).
    """
    import torch

    with torch.no_grad():
        positions_t = gaussians.get_xyz.detach()
        count = int(positions_t.shape[0])
        idx = _selected_indices(count, int(max_gaussians), positions_t.device)

        def take(t):
            t = t.detach()
            return t if idx is None else t.index_select(0, idx)

        positions_d = take(positions_t).to(dtype=torch.float32)
        scales_d = take(gaussians.get_scaling).to(dtype=torch.float32)
        quats_d = take(gaussians.get_rotation).to(dtype=torch.float32)
        opacities_d = take(gaussians.get_opacity).reshape(-1, 1).clamp(0.0, 1.0).to(dtype=torch.float32)

        features = take(_as_feature_tensor(gaussians))
        active_degree = int(getattr(gaussians, "active_sh_degree", 0))
        max_coeffs = int(features.shape[1])
        max_degree = int(np.floor(np.sqrt(max_coeffs) - 1))
        sh_degree = max(0, min(active_degree, max_degree, 3))
        coeff_count = (sh_degree + 1) ** 2
        n = features.shape[0]
        sh_d = (
            features[:, :coeff_count, :].to(dtype=torch.float32).reshape(n, coeff_count * 3)
        )

        # Single contiguous packed buffer -> single D2H transfer.
        packed = torch.cat([positions_d, scales_d, quats_d, opacities_d, sh_d], dim=1).contiguous()
        packed_np = packed.cpu().numpy()

    off = 0
    positions = np.ascontiguousarray(packed_np[:, off:off + 3], dtype=np.float32); off += 3
    scales = np.ascontiguousarray(packed_np[:, off:off + 3], dtype=np.float32); off += 3
    quats = np.ascontiguousarray(packed_np[:, off:off + 4], dtype=np.float32); off += 4
    opacities = np.ascontiguousarray(packed_np[:, off:off + 1].reshape(-1), dtype=np.float32); off += 1
    sh = np.ascontiguousarray(packed_np[:, off:off + coeff_count * 3], dtype=np.float32)

    return RVizSplatSnapshot(
        positions=positions,
        quaternions=quats,
        scales=scales,
        opacities=opacities,
        spherical_harmonics=sh,
        sh_degree=sh_degree,
        sh_rotations=(),
    )


def snapshot_and_transform_for_rvizsplat_torch(
    gaussians, basis: np.ndarray, sh_band_rotations, max_gaussians: int = 0,
) -> RVizSplatSnapshot:
    """WS4: combines :func:`snapshot_gaussians_for_rvizsplat_packed` and
    :func:`transform_snapshot_basis` into one GPU-resident pass, so the
    basis/quaternion/SH-rotation transform happens BEFORE the single
    device->host download instead of after -- eliminating the
    `transform_snapshot_basis` CPU numpy pass entirely from the publish
    worker's hot path (previously >1s/call at N~130k, GIL-stalling training;
    see docs/2026-07-26-transfer-audit.md's WS4 section).

    ``sh_band_rotations`` must be precomputed via
    :func:`sh_band_rotation_matrices` for this exact ``basis`` -- callers
    that already cache these per-basis (as ros2_rviz.py's publish worker
    does) should reuse that cache rather than recompute per call.

    ``gaussians`` accepts either a raw model (``get_xyz``/``get_scaling``/
    etc.) or a ``eligsir.map.views.GaussianMapReadView`` (``means``/
    ``scales``/etc.) -- the latter is what the live ros2_rviz publish path
    now passes, so the mutable model object itself never reaches this
    viewer boundary (see docs/architecture/gaussian_ownership_audit.md).
    """
    import torch

    with torch.no_grad():
        positions_t = _positions(gaussians).detach()
        count = int(positions_t.shape[0])
        idx = _selected_indices(count, int(max_gaussians), positions_t.device)

        def take(t):
            t = t.detach()
            return t if idx is None else t.index_select(0, idx)

        positions_d = take(positions_t).to(dtype=torch.float32)
        scales_d = take(_scales(gaussians)).to(dtype=torch.float32)
        quats_d = take(_quaternions(gaussians)).to(dtype=torch.float32)
        opacities_d = take(_opacities(gaussians)).reshape(-1, 1).clamp(0.0, 1.0).to(dtype=torch.float32)

        features = take(_as_feature_tensor(gaussians))
        active_degree = int(getattr(gaussians, "active_sh_degree", 0))
        max_coeffs = int(features.shape[1])
        max_degree = int(np.floor(np.sqrt(max_coeffs) - 1))
        sh_degree = max(0, min(active_degree, max_degree, 3))
        coeff_count = (sh_degree + 1) ** 2
        n = features.shape[0]
        sh_d = features[:, :coeff_count, :].to(dtype=torch.float32).reshape(n, coeff_count * 3)

        positions_x, quats_x, sh_x = transform_snapshot_basis_torch(
            positions_d, quats_d, sh_d, sh_degree, basis, sh_band_rotations,
        )

        # Single contiguous packed buffer -> single D2H transfer, AFTER the
        # transform (unlike the untransformed _packed function, which is
        # designed for callers that don't need a basis change at all).
        packed = torch.cat([positions_x, scales_d, quats_x, opacities_d, sh_x], dim=1).contiguous()
        packed_np = packed.cpu().numpy()

    off = 0
    positions = np.ascontiguousarray(packed_np[:, off:off + 3], dtype=np.float32); off += 3
    scales = np.ascontiguousarray(packed_np[:, off:off + 3], dtype=np.float32); off += 3
    quats = np.ascontiguousarray(packed_np[:, off:off + 4], dtype=np.float32); off += 4
    opacities = np.ascontiguousarray(packed_np[:, off:off + 1].reshape(-1), dtype=np.float32); off += 1
    sh = np.ascontiguousarray(packed_np[:, off:off + coeff_count * 3], dtype=np.float32)

    return RVizSplatSnapshot(
        positions=positions,
        quaternions=quats,
        scales=scales,
        opacities=opacities,
        spherical_harmonics=sh,
        sh_degree=sh_degree,
        sh_rotations=tuple(sh_band_rotations) if sh_degree >= 1 else (),
    )


def transform_snapshot_basis(snapshot: RVizSplatSnapshot, basis: np.ndarray) -> RVizSplatSnapshot:
    """Rotate positions, orientations, and SH coefficients into another world basis.

    SH bands >=1 are rotated with per-band least-squares rotation matrices
    (cached per basis).  DC (band 0) is invariant under world rotation.
    """
    basis = np.asarray(basis, dtype=np.float64)
    positions = (basis @ snapshot.positions.astype(np.float64).T).T.astype(np.float32)
    rotations = _quat_wxyz_to_rotmat(snapshot.quaternions.astype(np.float64))
    transformed_rotations = np.einsum("ij,njk->nik", basis, rotations)
    quaternions = _rotmat_to_quat_wxyz(transformed_rotations).astype(np.float32)

    sh_rotated = snapshot.spherical_harmonics
    sh_bands: Tuple[np.ndarray, ...] = ()
    if snapshot.sh_degree >= 1 and snapshot.spherical_harmonics.shape[0] > 0:
        sh_bands = sh_band_rotation_matrices(basis, degree=min(3, snapshot.sh_degree))
        coeff_count = (snapshot.sh_degree + 1) ** 2
        # Reshape: (N, coeff_count*3) → (N, 3, coeff_count) — RGB-interleaved
        # (matches snapshot layout: sh[:, 3*k + c] is color channel c of band k).
        # Use F-order so that sh_3[n, c, k] = sh[n, 3*k + c].
        n = snapshot.spherical_harmonics.shape[0]
        sh_3 = snapshot.spherical_harmonics[:, : coeff_count * 3].astype(np.float64)
        sh_3 = sh_3.reshape(n, 3, coeff_count, order="F")
        new_3 = np.empty_like(sh_3)
        # DC band (index 0) is identity
        new_3[:, :, 0] = sh_3[:, :, 0]
        for l_idx, l in enumerate(range(1, snapshot.sh_degree + 1)):
            D_l = sh_bands[l_idx]  # ((2l+1), (2l+1)) float32
            start = l * l
            count = 2 * l + 1
            band = sh_3[:, :, start : start + count]  # (N, 3, 2l+1)
            # Rotating the world by ``basis`` (positions p → basis·p) rotates
            # the radiance field f the same way, so f'(e) = f(basisᵀ·e).  The
            # least-squares fit in _sh_band_rotation_matrices yields D_l with
            # Y_l(basisᵀ·e) = D_lᵀ · Y_l(e); applying band @ D_lᵀ (NOT D_l) is
            # what makes eval(rotated_sh, e) = eval(sh, basisᵀ·e).  Using D_l
            # directly rotates SH by the *inverse* (view-dependent colour wrong
            # for deg ≥ 1; DC is unaffected because it is rotation-invariant).
            new_3[:, :, start : start + count] = band @ D_l.astype(np.float64).T
        sh_rotated = new_3.reshape(n, coeff_count * 3, order="F").astype(np.float32)
        # Pad with zeros if sh was wider than coeff_count
        if snapshot.spherical_harmonics.shape[1] > coeff_count * 3:
            pad = np.zeros(
                (n, snapshot.spherical_harmonics.shape[1] - coeff_count * 3),
                dtype=np.float32,
            )
            sh_rotated = np.concatenate([sh_rotated, pad], axis=1)

    return RVizSplatSnapshot(
        positions=positions,
        quaternions=quaternions,
        scales=snapshot.scales,
        opacities=snapshot.opacities,
        spherical_harmonics=sh_rotated,
        sh_degree=snapshot.sh_degree,
        sh_rotations=sh_bands,
    )


def _quat_wxyz_to_rotmat_torch(quats):
    """Torch port of :func:`_quat_wxyz_to_rotmat`. ``torch`` imported locally
    so this module stays import-torch-free unless this GPU path is used."""
    import torch

    norms = quats.norm(dim=1, keepdim=True)
    quats = torch.where(norms > 1e-12, quats / norms, torch.zeros_like(quats))
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    n = quats.shape[0]
    rot = torch.zeros((n, 3, 3), dtype=quats.dtype, device=quats.device)
    rot[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    rot[:, 0, 1] = 2 * (qx * qy - qw * qz)
    rot[:, 0, 2] = 2 * (qx * qz + qw * qy)
    rot[:, 1, 0] = 2 * (qx * qy + qw * qz)
    rot[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    rot[:, 1, 2] = 2 * (qy * qz - qw * qx)
    rot[:, 2, 0] = 2 * (qx * qz - qw * qy)
    rot[:, 2, 1] = 2 * (qy * qz + qw * qx)
    rot[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return rot


def _rotmat_to_quat_wxyz_torch(rot):
    """Torch port of :func:`_rotmat_to_quat_wxyz` (vectorized, all 4 Shepperd
    branch cases). ``torch`` imported locally, see :func:`_quat_wxyz_to_rotmat_torch`."""
    import torch

    n = rot.shape[0]
    quats = torch.empty((n, 4), dtype=rot.dtype, device=rot.device)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]

    mask0 = trace > 0.0
    mask1 = (~mask0) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    mask2 = (~mask0) & (~mask1) & (rot[:, 1, 1] > rot[:, 2, 2])
    mask3 = ~(mask0 | mask1 | mask2)

    if mask0.any():
        s = torch.sqrt(trace[mask0] + 1.0) * 2.0
        quats[mask0, 0] = 0.25 * s
        quats[mask0, 1] = (rot[mask0, 2, 1] - rot[mask0, 1, 2]) / s
        quats[mask0, 2] = (rot[mask0, 0, 2] - rot[mask0, 2, 0]) / s
        quats[mask0, 3] = (rot[mask0, 1, 0] - rot[mask0, 0, 1]) / s

    if mask1.any():
        s = torch.sqrt(1.0 + rot[mask1, 0, 0] - rot[mask1, 1, 1] - rot[mask1, 2, 2]) * 2.0
        quats[mask1, 0] = (rot[mask1, 2, 1] - rot[mask1, 1, 2]) / s
        quats[mask1, 1] = 0.25 * s
        quats[mask1, 2] = (rot[mask1, 0, 1] + rot[mask1, 1, 0]) / s
        quats[mask1, 3] = (rot[mask1, 0, 2] + rot[mask1, 2, 0]) / s

    if mask2.any():
        s = torch.sqrt(1.0 + rot[mask2, 1, 1] - rot[mask2, 0, 0] - rot[mask2, 2, 2]) * 2.0
        quats[mask2, 0] = (rot[mask2, 0, 2] - rot[mask2, 2, 0]) / s
        quats[mask2, 1] = (rot[mask2, 0, 1] + rot[mask2, 1, 0]) / s
        quats[mask2, 2] = 0.25 * s
        quats[mask2, 3] = (rot[mask2, 1, 2] + rot[mask2, 2, 1]) / s

    if mask3.any():
        s = torch.sqrt(1.0 + rot[mask3, 2, 2] - rot[mask3, 0, 0] - rot[mask3, 1, 1]) * 2.0
        quats[mask3, 0] = (rot[mask3, 1, 0] - rot[mask3, 0, 1]) / s
        quats[mask3, 1] = (rot[mask3, 0, 2] + rot[mask3, 2, 0]) / s
        quats[mask3, 2] = (rot[mask3, 1, 2] + rot[mask3, 2, 1]) / s
        quats[mask3, 3] = 0.25 * s

    norms = quats.norm(dim=1, keepdim=True)
    return torch.where(norms > 1e-12, quats / norms, torch.zeros_like(quats))


def transform_snapshot_basis_torch(positions, quaternions, spherical_harmonics, sh_degree,
                                    basis, sh_band_rotations):
    """Torch-tensor port of :func:`transform_snapshot_basis`'s per-Gaussian
    math (positions/quaternions/SH), for use directly on the GPU-resident
    tensors :func:`snapshot_gaussians_for_rvizsplat_packed` already holds
    before its download -- avoids the CPU numpy round-trip for this
    transform, which measured >1s/call at N~130k (GIL-stalling training,
    see docs/2026-07-26-transfer-audit.md's WS4 section).

    Args:
        positions, quaternions, spherical_harmonics: GPU tensors from an
            unpacked/unpacked-and-split snapshot (float32, on the training
            device).
        sh_degree: int, same as ``RVizSplatSnapshot.sh_degree``.
        basis: (3,3) array-like (numpy or torch) -- the world-basis matrix.
        sh_band_rotations: the per-band rotation matrices from
            :func:`sh_band_rotation_matrices` (numpy, computed once per
            basis on CPU -- tiny, not a per-Gaussian cost) -- uploaded once
            here, not per Gaussian.

    Returns:
        (positions_t, quaternions_t, sh_t) -- same shapes/dtypes as inputs,
        all still on the input device. Caller downloads once.
    """
    import torch

    device = positions.device
    basis_t = torch.as_tensor(np.asarray(basis, dtype=np.float32), device=device)

    positions_out = (basis_t @ positions.T).T
    rotations = _quat_wxyz_to_rotmat_torch(quaternions)
    transformed_rotations = torch.einsum("ij,njk->nik", basis_t, rotations)
    quaternions_out = _rotmat_to_quat_wxyz_torch(transformed_rotations)

    sh_out = spherical_harmonics
    if sh_degree >= 1 and spherical_harmonics.shape[0] > 0 and sh_band_rotations:
        coeff_count = (sh_degree + 1) ** 2
        n = spherical_harmonics.shape[0]
        sh_flat = spherical_harmonics[:, : coeff_count * 3]
        # F-order reshape (N, coeff_count*3) -> (N, 3, coeff_count): torch has
        # no `order=` kwarg, so permute the interpretation manually -- the
        # numpy F-order reshape means sh_3[n, c, k] = sh[n, 3*k + c], i.e.
        # channel-minor / coefficient-major layout. Equivalent to viewing as
        # (N, coeff_count, 3) row-major then transposing the last two dims.
        sh_3 = sh_flat.reshape(n, coeff_count, 3).transpose(1, 2)
        new_3 = torch.empty_like(sh_3)
        new_3[:, :, 0] = sh_3[:, :, 0]
        for l_idx, l in enumerate(range(1, sh_degree + 1)):
            D_l = torch.as_tensor(np.asarray(sh_band_rotations[l_idx], dtype=np.float32), device=device)
            start = l * l
            count = 2 * l + 1
            band = sh_3[:, :, start:start + count]
            new_3[:, :, start:start + count] = band @ D_l.T
        sh_3_out = new_3.transpose(1, 2).reshape(n, coeff_count * 3)
        if spherical_harmonics.shape[1] > coeff_count * 3:
            pad = torch.zeros(
                (n, spherical_harmonics.shape[1] - coeff_count * 3),
                dtype=spherical_harmonics.dtype, device=device,
            )
            sh_out = torch.cat([sh_3_out, pad], dim=1)
        else:
            sh_out = sh_3_out

    return positions_out, quaternions_out, sh_out


def pose_map_camera_link(pose: Pose, basis: np.ndarray = CANONICAL_TO_ROS_MAP) -> Pose:
    """Map a canonical c2w pose to its REP-103 link-frame equivalent.

    World rebasis (B·c2w) followed by world basis swap back to link
    (Bᵀ) — so the resulting pose's world translation is in ROS map and
    its rotation is the FLU camera-link basis.
    """
    basis = np.asarray(basis, dtype=np.float32)
    basis_t = basis.T
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = basis @ pose.rotation @ basis_t
    out[:3, 3] = basis @ pose.translation
    return Pose(out)


def transform_pose_basis(pose: Pose, basis: np.ndarray = CANONICAL_TO_ROS_MAP) -> Pose:
    """Rotate a canonical camera-to-world pose into another world basis."""
    basis = np.asarray(basis, dtype=np.float32)
    transformed = np.eye(4, dtype=np.float32)
    transformed[:3, :3] = basis @ pose.rotation
    transformed[:3, 3] = basis @ pose.translation
    return Pose(transformed)


def _quat_wxyz_to_rotmat(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.divide(quats, norms, out=np.zeros_like(quats), where=norms > 1e-12)
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    rot = np.zeros((quats.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    rot[:, 0, 1] = 2 * (qx * qy - qw * qz)
    rot[:, 0, 2] = 2 * (qx * qz + qw * qy)
    rot[:, 1, 0] = 2 * (qx * qy + qw * qz)
    rot[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    rot[:, 1, 2] = 2 * (qy * qz - qw * qx)
    rot[:, 2, 0] = 2 * (qx * qz - qw * qy)
    rot[:, 2, 1] = 2 * (qy * qz + qw * qx)
    rot[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return rot


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Vectorized rotation matrix → quaternion (w,x,y,z). Handles all branch cases."""
    n = rot.shape[0]
    quats = np.empty((n, 4), dtype=np.float64)
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]

    # case trace > 0
    mask0 = trace > 0.0
    s = np.sqrt(trace[mask0] + 1.0) * 2.0
    quats[mask0, 0] = 0.25 * s
    quats[mask0, 1] = (rot[mask0, 2, 1] - rot[mask0, 1, 2]) / s
    quats[mask0, 2] = (rot[mask0, 0, 2] - rot[mask0, 2, 0]) / s
    quats[mask0, 3] = (rot[mask0, 1, 0] - rot[mask0, 0, 1]) / s

    # case r00 dominant: r00 > r11 and r00 > r22
    mask1 = (~mask0) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
    s = np.sqrt(1.0 + rot[mask1, 0, 0] - rot[mask1, 1, 1] - rot[mask1, 2, 2]) * 2.0
    quats[mask1, 0] = (rot[mask1, 2, 1] - rot[mask1, 1, 2]) / s
    quats[mask1, 1] = 0.25 * s
    quats[mask1, 2] = (rot[mask1, 0, 1] + rot[mask1, 1, 0]) / s
    quats[mask1, 3] = (rot[mask1, 0, 2] + rot[mask1, 2, 0]) / s

    # case r11 dominant: r11 >= r00 and r11 > r22
    mask2 = (~mask0) & (~mask1) & (rot[:, 1, 1] > rot[:, 2, 2])
    s = np.sqrt(1.0 + rot[mask2, 1, 1] - rot[mask2, 0, 0] - rot[mask2, 2, 2]) * 2.0
    quats[mask2, 0] = (rot[mask2, 0, 2] - rot[mask2, 2, 0]) / s
    quats[mask2, 1] = (rot[mask2, 0, 1] + rot[mask2, 1, 0]) / s
    quats[mask2, 2] = 0.25 * s
    quats[mask2, 3] = (rot[mask2, 1, 2] + rot[mask2, 2, 1]) / s

    # case r22 > r00 and r22 > r11
    mask3 = ~(mask0 | mask1 | mask2)
    s = np.sqrt(1.0 + rot[mask3, 2, 2] - rot[mask3, 0, 0] - rot[mask3, 1, 1]) * 2.0
    quats[mask3, 0] = (rot[mask3, 1, 0] - rot[mask3, 0, 1]) / s
    quats[mask3, 1] = (rot[mask3, 0, 2] + rot[mask3, 2, 0]) / s
    quats[mask3, 2] = (rot[mask3, 1, 2] + rot[mask3, 2, 1]) / s
    quats[mask3, 3] = 0.25 * s

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    return np.divide(quats, norms, out=np.zeros_like(quats), where=norms > 1e-12)


def _quat_multiply_wxyz(q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    """Vectorized Hamilton product: returns (q_left ⊗ q_right) in wxyz.

    Accepts (4,) and (N, 4) inputs; broadcasts the singleton dimension.
    Output dtype is float64.
    """
    q_left = np.asarray(q_left, dtype=np.float64)
    q_right = np.asarray(q_right, dtype=np.float64)
    if q_left.ndim == 1:
        q_left = q_left[np.newaxis, :]
    if q_right.ndim == 1:
        q_right = q_right[np.newaxis, :]
    if q_left.shape[0] == 1 and q_right.shape[0] != 1:
        q_left = np.broadcast_to(q_left, q_right.shape).copy()
    if q_right.shape[0] == 1 and q_left.shape[0] != 1:
        q_right = np.broadcast_to(q_right, q_left.shape).copy()
    lw, lx, ly, lz = q_left[:, 0], q_left[:, 1], q_left[:, 2], q_left[:, 3]
    rw, rx, ry, rz = q_right[:, 0], q_right[:, 1], q_right[:, 2], q_right[:, 3]
    out = np.empty_like(q_left)
    out[:, 0] = lw * rw - lx * rx - ly * ry - lz * rz
    out[:, 1] = lw * rx + lx * rw + ly * rz - lz * ry
    out[:, 2] = lw * ry - lx * rz + ly * rw + lz * rx
    out[:, 3] = lw * rz + lx * ry - ly * rx + lz * rw
    return out


# Standard real-SH coefficients used in 3DGS (copied from eligsir.utils/sh_utils.py to
# keep this module ROS-/torch-free).  Order matches gsplat/Inria and
# 3D Gaussian Splatting:  (Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1, Y_2^{-2}, ...).
_SH_C0 = 0.28209479177387814
_SH_C1 = 0.4886025119029199
_SH_C2 = np.array(
    [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    ],
    dtype=np.float64,
)
_SH_C3 = np.array(
    [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    ],
    dtype=np.float64,
)


def _eval_real_sh_basis(dirs: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate real-SH basis Y_l^m(d) for each direction.

    Args:
        dirs: (N, 3) unit directions (assumed unit length; not enforced).
        degree: 0, 1, 2, or 3.

    Returns:
        (N, (degree+1)**2) array of Y_l^m(d) values.  Layout matches
        gsplat/Inria standard ordering.
    """
    n = dirs.shape[0]
    out = np.empty((n, (degree + 1) ** 2), dtype=np.float64)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    out[:, 0] = _SH_C0
    if degree == 0:
        return out
    out[:, 1] = -_SH_C1 * y
    out[:, 2] = _SH_C1 * z
    out[:, 3] = -_SH_C1 * x
    if degree == 1:
        return out
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    out[:, 4] = _SH_C2[0] * xy
    out[:, 5] = _SH_C2[1] * yz
    out[:, 6] = _SH_C2[2] * (2.0 * zz - xx - yy)
    out[:, 7] = _SH_C2[3] * xz
    out[:, 8] = _SH_C2[4] * (xx - yy)
    if degree == 2:
        return out
    out[:, 9] = _SH_C3[0] * y * (3.0 * xx - yy)
    out[:, 10] = _SH_C3[1] * xy * z
    out[:, 11] = _SH_C3[2] * y * (4.0 * zz - xx - yy)
    out[:, 12] = _SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
    out[:, 13] = _SH_C3[4] * x * (4.0 * zz - xx - yy)
    out[:, 14] = _SH_C3[5] * z * (xx - yy)
    out[:, 15] = _SH_C3[6] * x * (xx - 3.0 * yy)
    return out


def _sh_band_rotation_matrices(basis: np.ndarray, degree: int = 3) -> List[np.ndarray]:
    """Least-squares SH band rotation matrices for a 3x3 basis.

    For each band l in [1, degree], compute D_l such that
        Y_l(basis^T d) @ D_l.T ≈ Y_l(d)
    in the least-squares sense over ~200 directions sampled on the unit sphere.

    Args:
        basis: 3x3 rotation matrix (proper, det=+1).
        degree: Max SH degree to compute bands for (1..3).

    Returns:
        List [D_1, D_2, ..., D_degree] of band rotation matrices.  D_0 is
        identity (scalar coefficient invariant under world rotation).
    """
    n_samples = 200
    rng = np.random.default_rng(0xC0FFEE)
    # Uniform sampling on the sphere
    v = rng.standard_normal((n_samples, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    vd = v @ basis  # rotate directions

    band_matrices: List[np.ndarray] = []
    for l in range(1, degree + 1):
        n_band = 2 * l + 1
        Y_d = _eval_real_sh_basis(v, l)            # (N, (l+1)^2)
        Y_vd = _eval_real_sh_basis(vd, l)          # (N, (l+1)^2)
        # Slice out only the band-l coefficients (skip the lower bands).
        start = l * l
        Y_d_band = Y_d[:, start:]                  # (N, 2l+1)
        Y_vd_band = Y_vd[:, start:]                # (N, 2l+1)
        # Y_vd_band = Y_d_band @ D_l.T  ⇒  D_l = lstsq(Y_d_band, Y_vd_band).T
        D, *_ = np.linalg.lstsq(Y_d_band, Y_vd_band, rcond=None)
        # Pad/trim D to exactly (2l+1, 2l+1) — lstsq can return (n, m) with m >= n_band
        D_full = np.zeros((n_band, n_band), dtype=np.float64)
        m = min(D.shape[0], n_band)
        n = min(D.shape[1], n_band)
        D_full[:m, :n] = D[:m, :n]
        band_matrices.append(D_full)
    return band_matrices


@lru_cache(maxsize=16)
def _sh_rotation_cache(basis_key: bytes, degree: int) -> Tuple[Tuple[float, ...], ...]:
    """Memoize per-band SH rotation matrices for a given basis + degree.

    ``basis_key`` is the canonical bytes of the basis matrix (e.g. via
    ``basis.astype(np.float64).tobytes()``); results are stored in a flat
    tuple-of-tuples so they're hashable and picklable.
    """
    basis = np.frombuffer(basis_key, dtype=np.float64).reshape(3, 3).copy()
    bands = _sh_band_rotation_matrices(basis, degree=degree)
    return tuple(tuple(map(float, np.asarray(b, dtype=np.float32).reshape(-1))) for b in bands)


def sh_band_rotation_matrices(basis: np.ndarray, degree: int = 3) -> List[np.ndarray]:
    """Public cached wrapper.  See :func:`_sh_band_rotation_matrices`."""
    basis = np.asarray(basis, dtype=np.float64)
    key = basis.astype(np.float64).tobytes()
    cached = _sh_rotation_cache(key, degree)
    shapes = {1: (3, 3), 2: (5, 5), 3: (7, 7)}
    out: List[np.ndarray] = []
    for l, c in enumerate(cached, start=1):
        arr = np.asarray(c, dtype=np.float32)
        expected = shapes[l]
        out.append(arr.reshape(expected))
    return out


def covariance36_from_scale_quat(scales: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Return row-major 6x6 covariance arrays with the 3D covariance in top-left."""
    scales = np.asarray(scales, dtype=np.float64)
    quats = np.asarray(quats, dtype=np.float64)
    if scales.shape[0] == 0:
        return np.zeros((0, 36), dtype=np.float32)

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.divide(quats, norms, out=np.zeros_like(quats), where=norms > 1e-12)
    qw, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    rot = np.zeros((quats.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    rot[:, 0, 1] = 2 * (qx * qy - qw * qz)
    rot[:, 0, 2] = 2 * (qx * qz + qw * qy)
    rot[:, 1, 0] = 2 * (qx * qy + qw * qz)
    rot[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    rot[:, 1, 2] = 2 * (qy * qz - qw * qx)
    rot[:, 2, 0] = 2 * (qx * qz - qw * qy)
    rot[:, 2, 1] = 2 * (qy * qz + qw * qx)
    rot[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)

    scale2 = scales * scales
    weighted = rot * scale2[:, None, :]
    cov3 = np.einsum("nrk,nck->nrc", weighted, rot)

    cov36 = np.zeros((scales.shape[0], 36), dtype=np.float32)
    for row in range(3):
        for col in range(3):
            cov36[:, row * 6 + col] = cov3[:, row, col].astype(np.float32)
    return cov36


def frustum_line_segments(
    pose_map_optical: Pose,
    intrinsics: CameraIntrinsics,
    depth: float = 0.15,
) -> np.ndarray:
    """Return 4 rays + far rectangle as (8, 2, 3) line segments.

    The 4 corner rays go from the camera origin (apex) to the four
    far-corner points on a virtual image plane at the given depth (metres
    in the optical frame).  The 4 rectangle segments close the far plane.
    Total: 8 line segments × 2 endpoints × 3 coords = (8, 2, 3) array.

    Args:
        pose_map_optical: 4x4 c2w pose already in the ROS map frame
            (NOT canonical — call :func:`transform_pose_basis` first).
        intrinsics: Camera intrinsics used to project the four corners.
        depth: Distance from the optical centre to the far plane (metres).

    Returns:
        (8, 2, 3) float32 array of line segments.
    """
    R = pose_map_optical.rotation
    t = pose_map_optical.translation
    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.cx)
    cy = float(intrinsics.cy)
    w = int(intrinsics.width)
    h = int(intrinsics.height)

    # Optical-frame far-plane corner points (z=depth in front of camera).
    # Pinhole back-projection: (u, v) → (x, y, 1) * (depth/z'), but we
    # place the plane at z=depth so we need to scale.
    # corner = ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)
    def optical_corner(u: float, v: float) -> np.ndarray:
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return np.array([x, y, depth], dtype=np.float32)

    corners_local = np.stack(
        [
            optical_corner(0.0, 0.0),
            optical_corner(float(w), 0.0),
            optical_corner(float(w), float(h)),
            optical_corner(0.0, float(h)),
        ],
        axis=0,
    )
    # Transform to world (map) frame: R @ p + t
    corners_world = (R.astype(np.float32) @ corners_local.T).T + t.astype(np.float32)
    origin_world = t.astype(np.float32)

    apex = origin_world  # (3,)
    far_corners = corners_world  # (4, 3)

    # 4 apex→far rays
    rays = np.stack([apex[None].repeat(4, axis=0), far_corners], axis=1)  # (4, 2, 3)
    # 4 far-plane edges: (i, (i+1) % 4)
    edges = np.stack([far_corners, far_corners[(np.arange(4) + 1) % 4]], axis=1)  # (4, 2, 3)
    return np.concatenate([rays, edges], axis=0).astype(np.float32)


# ============================================================================
# Vectorized CDR serializer for gsplat_msgs/msg/SplatArray
# ============================================================================
# Message definitions (RVizSplat pinned commit 30311c58):
#   SplatArray:  gsplat_msgs/Splat[] splats          (NO trailing header)
#   Splat:       geometry_msgs/PoseWithCovarianceStamped pose
#                uint8 opacity
#                uint8 harmonics_deg
#                float32[] spherical_harmonics        (coeff-major RGB-interleaved)
#
# So the message is just a length-prefixed sequence of Splat records — there
# is no array-level header to serialize.  The full per-record XCDR1 wire
# layout (encapsulation, sequence length, float64 alignment padding) is
# documented in the docstring of :func:`serialize_splat_array_cdr` below,
# which is the single source of truth.


def _cdr_align_pad(abs_pos: int, alignment: int, origin: int = 4) -> int:
    """Pad bytes to align ``abs_pos`` to ``alignment`` with origin at byte ``origin``.

    Used to match XCDR1 alignment after the 4-byte encapsulation header
    shifts the natural origin.  ``origin=4`` (default) is the byte right
    after the encapsulation, so a member of alignment ``a`` that would
    otherwise start at absolute ``P`` instead starts at
    ``P + (-(P - 4)) % a``.
    """
    return (-(abs_pos - origin)) % alignment


def serialize_splat_array_cdr(
    snapshot: RVizSplatSnapshot,
    stamp_sec: int,
    stamp_nanosec: int,
    frame_id: str,
) -> bytes:
    """Serialize an :class:`RVizSplatSnapshot` to the wire bytes that
    ``rclpy``'s :func:`rclpy.serialization.serialize_message` produces for
    a ``gsplat_msgs/msg/SplatArray``.

    Use :meth:`rclpy.publisher.Publisher.publish` with the returned bytes
    for ~100× faster publishing than the rclpy serializer.

    This is **XCDR1 / plain CDR** payload serialization.  The 4-byte
    encapsulation header shifts the alignment origin to byte 4, so
    members are aligned to ``a`` bytes by padding
    ``(-(abs_pos - 4)) % a`` zero bytes before each member whose
    alignment exceeds the natural byte-stream position.  The per-record
    float64 alignment padding ``p`` is computed from the **absolute**
    buffer position of the record, not from a record-relative offset —
    this matters for K-even (degree 1, 3): record 0 always has p=4
    while records 1..N-1 may have p=0.  Validated against
    ``rclpy.serialization.deserialize_message`` in
    ``tests/test_rvizsplat_ros_roundtrip.py``.

    Wire layout (per-record, relative to record start R):

        R+0     8 bytes  stamp (int32 sec + uint32 nanosec)
        R+8     4 bytes  frame_id length (uint32, includes trailing NUL)
        R+12    L bytes  frame_id + NUL
        R+pad   p bytes  where p = _cdr_align_pad(abs(R)+12+L, 8) — 8-align
        +0      24 bytes position (3 × float64)
        +24     32 bytes orientation (4 × float64, xyzw order;
                                     snapshot is wxyz → reordered here)
        +56     288 bytes covariance (36 × float64, fixed array)
        +344    1 byte   opacity (uint8, 0..255)
        +345    1 byte   harmonics_deg (uint8)
        +346    2 bytes  pad (align uint32 to 4)
        +348    4 bytes  sh_length (uint32)
        +352    K*4 bytes sh_data (K × float32, RGB-interleaved
                                    coefficient-major)

    Record size = ``368 + p + 4*K``.  No tail padding between records.
    For K odd (deg 0, 2): p=4 for every record → stride = 372+4K.
    For K even (deg 1, 3): p=4 for record 0, p=0 for records 1..N-1.

    Args:
        snapshot: Snapshot with wxyz quaternions and RGB-interleaved SH.
        stamp_sec: ROS stamp seconds (applied to every record's header).
        stamp_nanosec: ROS stamp nanoseconds.
        frame_id: Frame id for every Splat.pose.header (string).

    Returns:
        Complete wire-format byte buffer ready to publish.
    """
    n = int(snapshot.positions.shape[0])
    K = (snapshot.sh_degree + 1) ** 2 * 3  # K float32 per record
    fid_bytes = frame_id.encode("utf-8") + b"\x00"
    fid_len = len(fid_bytes)  # INCLUDES the trailing NUL

    if n == 0:
        # Empty SplatArray: 4-byte XCDR1 encapsulation (00 01 00 00) + 4-byte N=0.
        return b"\x00\x01\x00\x00" + b"\x00\x00\x00\x00"

    # Pre-compute batch data to avoid per-record Python overhead.
    cov36_all = covariance36_from_scale_quat(snapshot.scales, snapshot.quaternions)
    opacity_all = np.clip(snapshot.opacities * 255.0, 0, 255).astype(np.uint8)
    # CDR orientation is xyzw; snapshot stores wxyz.
    quats_xyzw = snapshot.quaternions[:, [1, 2, 3, 0]].astype(np.float64)
    fid_arr = np.frombuffer(fid_bytes, dtype=np.uint8)

    # Compute per-record padding.  Padding is determined by the *absolute*
    # buffer position (payload-relative CDR alignment, origin=4 after encap).
    # Record 0 starts at absolute offset 8; subsequent records follow immediately.
    p0 = _cdr_align_pad(8 + 12 + fid_len, 8)
    rec0_size = 12 + fid_len + p0 + 344 + 8 + 4 * K

    if n > 1:
        p1 = _cdr_align_pad(8 + rec0_size + 12 + fid_len, 8)
        rec1_size = 12 + fid_len + p1 + 344 + 8 + 4 * K
    else:
        p1, rec1_size = 0, 0

    total_size = 8 + rec0_size + max(0, n - 1) * rec1_size
    buf = np.zeros(total_size, dtype=np.uint8)
    buf[0] = 0x00  # XCDRv1 dummy
    buf[1] = 0x01  # encapsulation: LE | PLAIN_CDR
    buf[4:8].view("<u4")[0] = n

    def _fill_record(rec_abs: int, i: int, p: int) -> None:
        pos_off = 12 + fid_len + p
        quat_off_ = pos_off + 24
        cov_off_ = quat_off_ + 32
        op_off_ = cov_off_ + 288          # 36 × 8 = 288
        sh_len_off_ = op_off_ + 4         # opacity(1) + deg(1) + pad(2) = 4
        sh_data_off_ = sh_len_off_ + 4

        buf[rec_abs:rec_abs + 4].view("<i4")[0] = int(stamp_sec)
        buf[rec_abs + 4:rec_abs + 8].view("<u4")[0] = int(stamp_nanosec)
        buf[rec_abs + 8:rec_abs + 12].view("<u4")[0] = fid_len
        buf[rec_abs + 12:rec_abs + 12 + fid_len] = fid_arr
        # alignment pad bytes are already zero (buf is zero-initialised)
        buf[rec_abs + pos_off:rec_abs + pos_off + 24].view("<f8").reshape(3)[:] = (
            snapshot.positions[i].astype(np.float64, copy=False)
        )
        buf[rec_abs + quat_off_:rec_abs + quat_off_ + 32].view("<f8").reshape(4)[:] = quats_xyzw[i]
        buf[rec_abs + cov_off_:rec_abs + cov_off_ + 288].view("<f8").reshape(36)[:] = (
            cov36_all[i].astype(np.float64, copy=False)
        )
        buf[rec_abs + op_off_] = int(opacity_all[i])
        buf[rec_abs + op_off_ + 1] = int(snapshot.sh_degree) & 0xFF
        if K > 0:
            buf[rec_abs + sh_len_off_:rec_abs + sh_len_off_ + 4].view("<u4")[0] = K
            buf[rec_abs + sh_data_off_:rec_abs + sh_data_off_ + 4 * K].view("<f4").reshape(K)[:] = (
                snapshot.spherical_harmonics[i, :K].astype(np.float32, copy=False)
            )

    # Record 0 has its own alignment padding p0 (always 4); fill it directly.
    _fill_record(8, 0, p0)

    if n > 1:
        # Records 1..n-1 share an identical stride (rec1_size) and padding
        # (p1).  Fill them with vectorized strided column writes over a single
        # 2-D uint8 view — no per-record Python loop, ~10-15x faster than the
        # per-record path and near-zero GIL hold on the worker thread.
        body = buf[8 + rec0_size:8 + rec0_size + (n - 1) * rec1_size].reshape(
            n - 1, rec1_size
        )
        pos_off = 12 + fid_len + p1
        quat_off_ = pos_off + 24
        cov_off_ = quat_off_ + 32
        op_off_ = cov_off_ + 288          # 36 × 8 = 288
        sh_len_off_ = op_off_ + 4         # opacity(1) + deg(1) + pad(2) = 4
        sh_data_off_ = sh_len_off_ + 4

        # Constant columns: broadcast one byte row across every record.
        body[:, 0:4] = np.frombuffer(struct.pack("<i", int(stamp_sec)), dtype=np.uint8)
        body[:, 4:8] = np.frombuffer(struct.pack("<I", int(stamp_nanosec)), dtype=np.uint8)
        body[:, 8:12] = np.frombuffer(struct.pack("<I", fid_len), dtype=np.uint8)
        body[:, 12:12 + fid_len] = fid_arr
        # alignment pad columns stay zero (buf is zero-initialised)

        # Per-record float fields: reinterpret each source row block as
        # little-endian bytes and assign the whole (n-1, width) block at once.
        # ascontiguousarray(dtype=...) forces both C-contiguity (required by
        # .view to shrink the itemsize) and little-endian byte order (CDR LE,
        # portable on big-endian hosts).
        body[:, pos_off:pos_off + 24] = (
            np.ascontiguousarray(snapshot.positions[1:n], dtype="<f8").view(np.uint8)
        )
        body[:, quat_off_:quat_off_ + 32] = (
            np.ascontiguousarray(quats_xyzw[1:n], dtype="<f8").view(np.uint8)
        )
        body[:, cov_off_:cov_off_ + 288] = (
            np.ascontiguousarray(cov36_all[1:n], dtype="<f8").view(np.uint8)
        )
        body[:, op_off_] = opacity_all[1:n]
        body[:, op_off_ + 1] = int(snapshot.sh_degree) & 0xFF
        if K > 0:
            body[:, sh_len_off_:sh_len_off_ + 4] = np.frombuffer(
                struct.pack("<I", K), dtype=np.uint8
            )
            body[:, sh_data_off_:sh_data_off_ + 4 * K] = (
                np.ascontiguousarray(
                    snapshot.spherical_harmonics[1:n, :K], dtype="<f4"
                ).view(np.uint8)
            )

    return bytes(buf)


def encode_compact_dc_v1_array(snapshot: RVizSplatSnapshot) -> np.ndarray:
    """Encode snapshot as a (N, 32) uint8 array for per-tile slicing.

    Layout per splat, little-endian, 32 bytes:

    - ``float32 x, y, z``
    - ``float16 cov00, cov01, cov02, cov11, cov12, cov22``
    - ``uint8 r, g, b, a`` baked from DC SH and opacity
    - ``uint32 id_or_flags`` currently zero
    """
    n = int(snapshot.positions.shape[0])
    if n == 0:
        return np.zeros((0, COMPACT_DC_V1_STRIDE), dtype=np.uint8)

    cov36 = covariance36_from_scale_quat(snapshot.scales, snapshot.quaternions)
    cov6 = cov36[:, [0, 1, 2, 7, 8, 14]].astype("<f2", copy=False)
    positions = np.ascontiguousarray(snapshot.positions.astype("<f4", copy=False))

    sh = snapshot.spherical_harmonics
    if sh.shape[1] >= 3:
        rgb = np.clip(_SH_C0 * sh[:, :3].astype(np.float32) + 0.5, 0.0, 1.0)
    else:
        rgb = np.full((n, 3), 0.5, dtype=np.float32)
    alpha = np.clip(snapshot.opacities.reshape(-1, 1), 0.0, 1.0)
    rgba = np.clip(np.concatenate([rgb, alpha], axis=1) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    out = np.zeros((n, COMPACT_DC_V1_STRIDE), dtype=np.uint8)
    out[:, 0:12] = positions.view(np.uint8).reshape(n, 12)
    out[:, 12:24] = np.ascontiguousarray(cov6).view(np.uint8).reshape(n, 12)
    out[:, 24:28] = rgba
    return out


def encode_compact_dc_v1(snapshot: RVizSplatSnapshot) -> bytes:
    """Encode a transformed snapshot as compact DC-only splat records (bytes)."""
    return encode_compact_dc_v1_array(snapshot).tobytes()


def split_bytes_chunks(data: bytes, max_chunk_bytes: int) -> list[bytes]:
    """Split bytes into non-empty chunks capped at ``max_chunk_bytes``."""
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")
    if not data:
        return [b""]
    return [data[i : i + max_chunk_bytes] for i in range(0, len(data), max_chunk_bytes)]


def tile_keys_for_positions(positions: np.ndarray, tile_size_m: float) -> np.ndarray:
    """Return int32 tile xyz keys for positions in map coordinates."""
    tile_size = float(tile_size_m)
    if not math.isfinite(tile_size) or tile_size <= 0.0:
        raise ValueError("tile_size_m must be finite and positive")
    return np.floor(np.asarray(positions, dtype=np.float32) / tile_size).astype(np.int32)
