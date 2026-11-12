"""Frame-integration geometry/masking helpers, extracted from ``train_streaming.py``.

Wave 1 consolidation plan step 10 prerequisite (see
the mapping integration ownership boundary
step 10 and ``docs/architecture/mapping_coordinator_plan.md``).

This cluster (depth-mask sampling, occupancy-render masking, normal
estimation, KNN smoothness regularization) was the blocker that stopped
step 8 from extracting ``insert_gaussians_from_frame``,
``_run_rolling_seed``, and ``_run_submap_stitch`` out of
``train_streaming.py``: all three call into these functions, and they
could not move to ``eligsir/mapping/integration/`` while the helpers
stayed behind (a circular import). Moving the helpers here first — with no
math change, verbatim except for import adjustments — unblocks that
extraction for a future pass building the real ``RGBDIntegrator``/
``BootstrapIntegrator``/``SubmapIntegrator`` implementations.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch


def finite_diff_normal_denominators(fx: float, fy: float, s: int) -> Tuple[float, float]:
    """Denominators for the central-difference surface-normal estimate used by
    ``insert_gaussians_from_frame``'s three near-identical branches
    (gpu_resident / use_gpu / numpy).

    ``s`` is the pixel half-offset the central difference is taken over
    (``dz = z[x+s] - z[x-s]``), currently always equal to
    ``--streaming_depth_stride``. The normal-tilt estimate is
    ``nx = -dzdx / denom_x``, ``ny = -dzdy / denom_y``, with ``nz = 1`` before
    normalization -- so scaling ``denom_x``/``denom_y`` by the same factor the
    finite-difference span grows by keeps the estimated tilt (and therefore
    the grazing-angle rejection test ``nz > cos(max_view_angle)``) invariant
    to ``s``, which is the physically correct behavior: a central difference
    taken over a wider baseline shouldn't itself report a steeper surface.

    FIXED (docs/architecture/2026-08-29-staged-coarse-to-fine-depth-insertion.md
    §4.2): the previous formula was ``fx * (2.0 * s / depth_stride)``, and
    every call site passed ``s == depth_stride``, so the ratio was identically
    1 and the denominator silently collapsed to the constant ``2 * fx`` --
    independent of ``s``/stride entirely. That made the estimated normal tilt
    scale linearly with stride while its normalizing divisor stayed fixed, a
    stride-dependent bias in the grazing-rejection test that was invisible
    only because stride was a fixed constant for the whole run. A schedule
    that varies stride over training un-freezes that bias into a confound.
    The fix removes the self-cancelling ``/ depth_stride`` term so the
    denominator scales with the actual finite-difference span ``2 * s``.
    """
    s = float(s)
    return float(fx) * 2.0 * s, float(fy) * 2.0 * s


def _load_depth_meters(frame) -> Optional[np.ndarray]:
    """Load and convert a frame's depth image to float32 metres array, or None on failure."""
    from eligsir.utils.rgbd_frames import depth_to_meters
    from eligsir.io.sources.decode_cache import load_frame_depth_np
    if frame is None:
        return None
    if frame._depth_np is None and frame.depth_path is None and frame._depth_bytes is None:
        return None
    if frame._depth_np is None and frame.depth_path is not None and not os.path.exists(frame.depth_path) and frame._depth_bytes is None:
        return None
    try:
        depth_raw = load_frame_depth_np(frame)
        if depth_raw is None:
            return None
        return depth_to_meters(depth_raw, frame.depth_scale).astype(np.float32)
    except Exception:
        return None


def _resize_nearest_np(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D array with nearest-neighbour indexing without dtype changes."""
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if arr.shape[:2] == (target_h, target_w):
        return arr
    src_h, src_w = arr.shape[:2]
    ys = np.clip(np.round(np.linspace(0, src_h - 1, target_h)).astype(np.int32), 0, src_h - 1)
    xs = np.clip(np.round(np.linspace(0, src_w - 1, target_w)).astype(np.int32), 0, src_w - 1)
    return arr[ys[:, None], xs[None, :]]


def _sample_stable_depth_mask(
    z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    min_depth: float,
    max_depth: float,
    *,
    erode_px: int,
    window: int,
    min_valid_ratio: float,
    max_range: float,
    median_thresh: float,
) -> np.ndarray:
    """Return per-sampled-pixel mask for locally stable depth measurements."""
    sample_shape = xs.shape
    if erode_px <= 0 and window <= 1 and min_valid_ratio <= 0 and max_range <= 0 and median_thresh <= 0:
        return np.ones(sample_shape, dtype=bool)

    h, w = z.shape
    local_depths = []
    radius = max(0, int(window) // 2)
    if radius > 0:
        for dy in range(-radius, radius + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            for dx in range(-radius, radius + 1):
                xx = np.clip(xs + dx, 0, w - 1)
                local_depths.append(z[yy, xx])
    else:
        local_depths.append(z[ys, xs])

    local = np.stack(local_depths, axis=0).astype(np.float32, copy=False)
    local_valid = np.isfinite(local) & (local > min_depth) & (local < max_depth)
    stable = np.ones(sample_shape, dtype=bool)

    if min_valid_ratio > 0:
        stable &= local_valid.mean(axis=0) >= float(min_valid_ratio)

    if max_range > 0:
        local_min = np.where(local_valid, local, np.inf).min(axis=0)
        local_max = np.where(local_valid, local, -np.inf).max(axis=0)
        stable &= np.isfinite(local_min) & np.isfinite(local_max) & ((local_max - local_min) <= float(max_range))

    if median_thresh > 0:
        local_nan = np.where(local_valid, local, np.nan)
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            local_median = np.nanmedian(local_nan, axis=0)
        stable &= np.isfinite(local_median) & (np.abs(z[ys, xs] - local_median) <= float(median_thresh))

    erode_px = max(0, int(erode_px))
    if erode_px > 0:
        valid_hw = np.isfinite(z) & (z > min_depth) & (z < max_depth)
        eroded = np.ones(sample_shape, dtype=bool)
        for dy in range(-erode_px, erode_px + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            for dx in range(-erode_px, erode_px + 1):
                xx = np.clip(xs + dx, 0, w - 1)
                eroded &= valid_hw[yy, xx]
        stable &= eroded

    return stable


def _sample_stable_depth_mask_gpu(
    z_t: "torch.Tensor",
    xs_t: "torch.Tensor",
    ys_t: "torch.Tensor",
    min_depth: float,
    max_depth: float,
    *,
    erode_px: int,
    window: int,
    min_valid_ratio: float,
    max_range: float,
    median_thresh: float,
) -> "torch.Tensor":
    """GPU port of _sample_stable_depth_mask (the dominant per-frame CPU cost).

    Runs the window gather-stack + reductions on-device over GPU-resident depth,
    so the 25-plane stack over the strided sample grid (the single biggest numpy
    cost on dense sensors, e.g. Orbbec 1280x720 @ stride 4) never touches CPU.
    Returns a bool tensor on z_t.device; the caller copies the small result back.

    Faithful to the numpy version within float tolerance. One known approximation:
    torch.nanmedian returns the lower-middle value for even windows where
    numpy.nanmedian averages the two middle — a sub-mm tie-break difference on a
    filter mask, well inside the gpu_frame_prep "float32 tolerance" contract.
    """
    sample_shape = xs_t.shape
    device = z_t.device
    if erode_px <= 0 and window <= 1 and min_valid_ratio <= 0 and max_range <= 0 and median_thresh <= 0:
        return torch.ones(sample_shape, dtype=torch.bool, device=device)

    H, W = int(z_t.shape[0]), int(z_t.shape[1])
    ys_i = ys_t.to(torch.int64)
    xs_i = xs_t.to(torch.int64)
    radius = max(0, int(window) // 2)

    planes = []
    if radius > 0:
        for dy in range(-radius, radius + 1):
            yy = (ys_i + dy).clamp_(0, H - 1)
            for dx in range(-radius, radius + 1):
                xx = (xs_i + dx).clamp(0, W - 1)
                planes.append(z_t[yy, xx])
    else:
        planes.append(z_t[ys_i, xs_i])

    local = torch.stack(planes, dim=0)  # (K, gh, gw)
    local_valid = torch.isfinite(local) & (local > min_depth) & (local < max_depth)
    stable = torch.ones(sample_shape, dtype=torch.bool, device=device)

    if min_valid_ratio > 0:
        stable &= local_valid.to(torch.float32).mean(dim=0) >= float(min_valid_ratio)

    if max_range > 0:
        local_min = torch.where(local_valid, local, torch.full_like(local, float("inf"))).amin(dim=0)
        local_max = torch.where(local_valid, local, torch.full_like(local, float("-inf"))).amax(dim=0)
        stable &= torch.isfinite(local_min) & torch.isfinite(local_max) & ((local_max - local_min) <= float(max_range))

    if median_thresh > 0:
        local_nan = torch.where(local_valid, local, torch.full_like(local, float("nan")))
        local_median = torch.nanmedian(local_nan, dim=0).values
        center = z_t[ys_i, xs_i]
        stable &= torch.isfinite(local_median) & ((center - local_median).abs() <= float(median_thresh))

    erode_px = max(0, int(erode_px))
    if erode_px > 0:
        valid_hw = torch.isfinite(z_t) & (z_t > min_depth) & (z_t < max_depth)
        eroded = torch.ones(sample_shape, dtype=torch.bool, device=device)
        for dy in range(-erode_px, erode_px + 1):
            yy = (ys_i + dy).clamp(0, H - 1)
            for dx in range(-erode_px, erode_px + 1):
                xx = (xs_i + dx).clamp(0, W - 1)
                eroded &= valid_hw[yy, xx]
        stable &= eroded

    return stable


def _sample_temporal_depth_mask(
    z_v: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    frame,
    prev_frame,
    prev_depth_meters: Optional[np.ndarray],
    d_fx: float,
    d_fy: float,
    d_cx: float,
    d_cy: float,
    *,
    thresh: float,
) -> Optional[np.ndarray]:
    """Check current depth candidates against the previous inserted frame."""
    if prev_frame is None or prev_depth_meters is None or thresh <= 0:
        return None
    if getattr(frame, "c2w", None) is None or getattr(prev_frame, "c2w", None) is None:
        return None
    try:
        c2w = np.asarray(frame.c2w, dtype=np.float32)
        prev_c2w = np.asarray(prev_frame.c2w, dtype=np.float32)
        if c2w.shape != (4, 4) or prev_c2w.shape != (4, 4):
            return None
        if not np.isfinite(c2w).all() or not np.isfinite(prev_c2w).all():
            return None

        prev_h, prev_w = int(prev_frame.height), int(prev_frame.width)
        prev_depth = _resize_nearest_np(np.asarray(prev_depth_meters, dtype=np.float32), (prev_h, prev_w))

        x_c = (xs.astype(np.float32) - float(d_cx)) / float(d_fx) * z_v.astype(np.float32)
        y_c = (ys.astype(np.float32) - float(d_cy)) / float(d_fy) * z_v.astype(np.float32)
        pts_cam = np.stack([x_c, y_c, z_v.astype(np.float32)], axis=1)
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

        prev_w2c = np.linalg.inv(prev_c2w)
        pts_prev = (prev_w2c[:3, :3] @ pts_world.T).T + prev_w2c[:3, 3]
        z_prev = pts_prev[:, 2]

        p_fx = prev_frame.depth_fx if prev_frame.depth_fx is not None else prev_frame.fx
        p_fy = prev_frame.depth_fy if prev_frame.depth_fy is not None else prev_frame.fy
        p_cx = prev_frame.depth_cx if prev_frame.depth_cx is not None else prev_frame.cx
        p_cy = prev_frame.depth_cy if prev_frame.depth_cy is not None else prev_frame.cy
        # Scale prev-frame depth intrinsics if they are at a different resolution than the
        # color image (same correction as the main backprojection path in insert_gaussians_from_frame).
        if prev_frame.depth_fx is not None:
            _pdw = prev_frame.depth_width or prev_w
            _pdh = prev_frame.depth_height or prev_h
            if _pdw != prev_w or _pdh != prev_h:
                p_fx = p_fx * prev_w / _pdw
                p_fy = p_fy * prev_h / _pdh
                p_cx = p_cx * prev_w / _pdw
                p_cy = p_cy * prev_h / _pdh
        u = np.round((pts_prev[:, 0] / np.maximum(z_prev, 1e-6)) * float(p_fx) + float(p_cx)).astype(np.int32)
        v = np.round((pts_prev[:, 1] / np.maximum(z_prev, 1e-6)) * float(p_fy) + float(p_cy)).astype(np.int32)

        ok = (z_prev > 0) & (u >= 0) & (u < prev_w) & (v >= 0) & (v < prev_h)
        out = np.zeros(z_v.shape[0], dtype=bool)
        if ok.any():
            lookup = prev_depth[v[ok], u[ok]]
            out[ok] = np.isfinite(lookup) & (lookup > 0) & (np.abs(z_prev[ok] - lookup) <= float(thresh))
        return out
    except Exception:
        return None


def _sample_temporal_depth_mask_gpu(
    z_v_t: "torch.Tensor",
    xs_t: "torch.Tensor",
    ys_t: "torch.Tensor",
    frame,
    prev_frame,
    prev_depth_meters: Optional[np.ndarray],
    d_fx: float,
    d_fy: float,
    d_cx: float,
    d_cy: float,
    *,
    thresh: float,
) -> Optional["torch.Tensor"]:
    """WS2 M1: torch port of :func:`_sample_temporal_depth_mask`, operating on
    the full DENSE strided grid (same shape as ``z_v_t``/``xs_t``/``ys_t``)
    instead of an already-``valid``-masked flat array.

    Per the mask-late design (docs/2026-06-30-gpu-resident-ingestion-plan.md),
    every dense stage is computed over the full fixed-size grid and combined
    with a single final AND -- since this check is purely per-pixel
    (independent of any other pixel/mask), evaluating it everywhere and
    ANDing the result into ``valid`` afterward is exactly equivalent to only
    evaluating it where ``valid`` was already true (bitwise AND is
    commutative/associative), so this changes nothing about which pixels are
    ultimately accepted -- only removes the variable-length ``z_v[valid]``
    gather. Returns ``None`` in exactly the same "not applicable" cases as
    the numpy version (missing prev_frame/prev_depth_meters, non-finite or
    wrong-shape c2w, thresh<=0, or any exception during the computation) --
    contract preserved so callers can fall back identically.

    ``torch.linalg.inv`` on the previous camera's c2w is done in float64 then
    cast back to float32 (per the design doc's temporal-port note) to keep
    the matrix inverse numerically close to numpy's float64 default
    ``np.linalg.inv``; a handful of round-half-to-even boundary pixels may
    disagree with the numpy path at the sub-mm threshold-comparison level,
    which is why callers should tolerate ~99.9% agreement here rather than
    bitwise equality (see test_temporal_depth_mask_gpu_parity.py).
    """
    if prev_frame is None or prev_depth_meters is None or thresh <= 0:
        return None
    if getattr(frame, "c2w", None) is None or getattr(prev_frame, "c2w", None) is None:
        return None
    try:
        device = z_v_t.device
        c2w_np = np.asarray(frame.c2w, dtype=np.float32)
        prev_c2w_np = np.asarray(prev_frame.c2w, dtype=np.float32)
        if c2w_np.shape != (4, 4) or prev_c2w_np.shape != (4, 4):
            return None
        if not np.isfinite(c2w_np).all() or not np.isfinite(prev_c2w_np).all():
            return None

        prev_h, prev_w = int(prev_frame.height), int(prev_frame.width)
        prev_depth_np = _resize_nearest_np(
            np.asarray(prev_depth_meters, dtype=np.float32), (prev_h, prev_w)
        )
        prev_depth_t = torch.from_numpy(np.ascontiguousarray(prev_depth_np)).to(
            device=device, dtype=torch.float32
        )

        c2w = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)
        prev_c2w = torch.from_numpy(prev_c2w_np).to(device=device, dtype=torch.float32)
        # float64 inverse then cast to f32, per the design doc's temporal-port note.
        prev_w2c = torch.linalg.inv(prev_c2w.to(torch.float64)).to(torch.float32)

        z_v = z_v_t.to(torch.float32)
        x_c = (xs_t.to(torch.float32) - float(d_cx)) / float(d_fx) * z_v
        y_c = (ys_t.to(torch.float32) - float(d_cy)) / float(d_fy) * z_v
        pts_cam = torch.stack([x_c, y_c, z_v], dim=-1)  # (Gh, Gw, 3)

        R = c2w[:3, :3]
        t = c2w[:3, 3]
        pts_world = torch.einsum("ij,...j->...i", R, pts_cam) + t

        R_pw = prev_w2c[:3, :3]
        t_pw = prev_w2c[:3, 3]
        pts_prev = torch.einsum("ij,...j->...i", R_pw, pts_world) + t_pw
        z_prev = pts_prev[..., 2]

        p_fx = prev_frame.depth_fx if prev_frame.depth_fx is not None else prev_frame.fx
        p_fy = prev_frame.depth_fy if prev_frame.depth_fy is not None else prev_frame.fy
        p_cx = prev_frame.depth_cx if prev_frame.depth_cx is not None else prev_frame.cx
        p_cy = prev_frame.depth_cy if prev_frame.depth_cy is not None else prev_frame.cy
        if prev_frame.depth_fx is not None:
            _pdw = prev_frame.depth_width or prev_w
            _pdh = prev_frame.depth_height or prev_h
            if _pdw != prev_w or _pdh != prev_h:
                p_fx = p_fx * prev_w / _pdw
                p_fy = p_fy * prev_h / _pdh
                p_cx = p_cx * prev_w / _pdw
                p_cy = p_cy * prev_h / _pdh

        z_prev_safe = torch.clamp(z_prev, min=1e-6)
        u = torch.round((pts_prev[..., 0] / z_prev_safe) * float(p_fx) + float(p_cx)).to(torch.int64)
        v = torch.round((pts_prev[..., 1] / z_prev_safe) * float(p_fy) + float(p_cy)).to(torch.int64)

        ok = (z_prev > 0) & (u >= 0) & (u < prev_w) & (v >= 0) & (v < prev_h)
        u_c = torch.clamp(u, 0, prev_w - 1)
        v_c = torch.clamp(v, 0, prev_h - 1)
        lookup = prev_depth_t[v_c, u_c]
        out = ok & torch.isfinite(lookup) & (lookup > 0) & ((z_prev - lookup).abs() <= float(thresh))
        return out
    except Exception:
        return None


def _quaternion_from_normal(normal: np.ndarray) -> np.ndarray:
    """Compute wxyz quaternions rotating [0,0,1] to the given normals."""
    z_axis = np.array([0, 0, 1.0])
    cross = np.cross(z_axis, normal)           # [N, 3]
    w = 1.0 + np.sum(z_axis * normal, axis=1)  # [N]
    q = np.stack([w, cross[:, 0], cross[:, 1], cross[:, 2]], axis=1)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    # Degenerate case: normal ≈ [0,0,-1] → 180° rotation around x-axis
    bad = (nrm.squeeze(-1) < 1e-6)
    if bad.any():
        q[bad] = np.array([0.0, 1.0, 0.0, 0.0])
        nrm[bad] = 1.0
    return q / nrm


def _gpu_render_occupancy_masks(
    render_alpha,
    render_depth,
    xs_t,
    ys_t,
    z_v_t,
    h,
    w,
    insertion_policy,
):
    """GPU mirror of the alpha+depth occupancy masking block.

    Gathers the (already-CUDA) full-res render_alpha / render_depth tensors at
    the strided sensor grid coords using the SAME rounding/clip mapping as the
    numpy path (lines 889-927 of insert_gaussians_from_frame), so no
    .cpu().numpy() of the full render tensors is required.

    `xs_t`, `ys_t`, `z_v_t` are the strided sensor grid (2D float tensors on the
    render device).  Returns (low_alpha, no_rend_depth, sensor_closer,
    score_grid) as 2D bool/float tensors on the same device.
    """
    device = xs_t.device
    sh2d = xs_t.shape
    low_alpha = torch.zeros(sh2d, dtype=torch.bool, device=device)
    no_rend_depth = torch.ones(sh2d, dtype=torch.bool, device=device)  # default "no depth"
    sensor_closer = torch.zeros(sh2d, dtype=torch.bool, device=device)
    score_grid = torch.zeros(sh2d, dtype=torch.float32, device=device)

    if render_alpha is not None:
        alpha_t = render_alpha.detach().squeeze()  # [Hr, Wr]
        Hr, Wr = int(alpha_t.shape[0]), int(alpha_t.shape[1])
        xs_r = torch.clamp(
            torch.round(xs_t / max(w - 1, 1) * (Wr - 1)).to(torch.int64), 0, Wr - 1
        )
        ys_r = torch.clamp(
            torch.round(ys_t / max(h - 1, 1) * (Hr - 1)).to(torch.int64), 0, Hr - 1
        )
        alpha_v = alpha_t[ys_r, xs_r]
        alpha_thresh = float(getattr(insertion_policy, "alpha_threshold", getattr(insertion_policy, "streaming_insert_alpha_threshold", 0.3)))
        low_alpha = alpha_v < alpha_thresh
        score_grid = score_grid + torch.clamp(alpha_thresh - alpha_v, min=0.0).to(
            torch.float32
        ) * float(getattr(insertion_policy, "score_alpha_weight", getattr(insertion_policy, "streaming_insert_score_alpha_weight", 1.0)))

    if render_depth is not None:
        rend_d_t = render_depth.detach().squeeze()
        Hr_d, Wr_d = int(rend_d_t.shape[0]), int(rend_d_t.shape[1])
        xs_rd = torch.clamp(
            torch.round(xs_t / max(w - 1, 1) * (Wr_d - 1)).to(torch.int64), 0, Wr_d - 1
        )
        ys_rd = torch.clamp(
            torch.round(ys_t / max(h - 1, 1) * (Hr_d - 1)).to(torch.int64), 0, Hr_d - 1
        )
        rend_d_v = rend_d_t[ys_rd, xs_rd]
        consistency_thresh = float(getattr(insertion_policy, "occupancy_distance_m", getattr(insertion_policy, "streaming_depth_consistency_thresh", 0.05)))
        no_rend_depth = rend_d_v <= 0
        sensor_closer = (rend_d_v - z_v_t) > consistency_thresh
        score_grid = score_grid + no_rend_depth.to(torch.float32) * float(
            getattr(insertion_policy, "score_no_depth_weight", getattr(insertion_policy, "streaming_insert_score_no_depth_weight", 1.0))
        )
        score_grid = score_grid + torch.clamp(rend_d_v - z_v_t, min=0.0).to(
            torch.float32
        ) * float(getattr(insertion_policy, "score_sensor_closer_weight", getattr(insertion_policy, "streaming_insert_score_sensor_closer_weight", 2.0)))

    return low_alpha, no_rend_depth, sensor_closer, score_grid


def _kornia_normal_map(z_t, d_fx, d_fy, d_cx, d_cy):
    """Full-resolution camera-frame surface normals via kornia.depth_to_normals.

    `z_t` is the (H, W) float32 depth map (meters) on the render device. Returns
    a (3, H, W) normalized normal tensor on the same device. kornia computes
    normals from the full-res 3D-point spatial gradient (Sobel), so the result is
    NOT bit-identical to the strided central-difference finite_diff path — it is
    the alternative the kornia A/B evaluates. Sign convention differs from the
    finite_diff path, so callers should use |nz| for grazing tests and flip to a
    consistent hemisphere before building surfel quaternions.
    """
    import kornia
    device = z_t.device
    K = torch.tensor(
        [[float(d_fx), 0.0, float(d_cx)],
         [0.0, float(d_fy), float(d_cy)],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    ).unsqueeze(0)
    depth_b = z_t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    normals = kornia.geometry.depth.depth_to_normals(depth_b, K)  # (1, 3, H, W)
    return normals[0]


def _quat_wxyz_to_rotmat(quats):
    """[N,4] wxyz (unnormalised ok) -> [N,3,3] rotation matrices."""
    q = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def _gaussian_min_axis_normals(quats, scales):
    """Per-Gaussian surface normal = world direction of the shortest scale axis
    (the surfel's thin axis). quats [N,4] wxyz, scales [N,3] -> [N,3] unit normals."""
    R = _quat_wxyz_to_rotmat(quats)                     # [N,3,3]
    ax = torch.argmin(scales, dim=-1)                   # [N]
    col = torch.gather(R, 2, ax.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)  # [N,3]
    return col / col.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def knn_smoothness_reg(gaussians, args, iteration):
    """Amortized KNN scale-smoothness regularizer — KNN in the per-iteration hot
    path WITHOUT the per-iteration search cost (flag-gated; --knn_reg_weight, off
    by default).

    The expensive neighbour SEARCH is amortized: the k-NN graph is rebuilt only
    every --knn_reg_interval iters (Gaussians drift slowly, so a slightly stale
    graph is fine) and cached on the model; it is also rebuilt whenever the
    Gaussian count changes. Each iteration the only cost is a gather on the cached
    DETACHED indices + an elementwise loss (~0.4 ms at ~22k Gaussians, profiled),
    so it/s is barely affected. Penalizes a Gaussian's scale deviating from its
    neighbours' mean scale (local scale coherence → fewer outlier/floater splats).
    Gradients flow through the scales only — never the non-differentiable search.
    """
    w = float(getattr(args, "knn_reg_weight", 0.0))
    if w <= 0.0:
        return 0.0
    k = int(getattr(args, "knn_reg_k", 8))
    interval = max(1, int(getattr(args, "knn_reg_interval", 100)))
    means = gaussians.get_xyz
    N = means.shape[0]
    idx = getattr(gaussians, "_knn_reg_idx", None)
    last = getattr(gaussians, "_knn_reg_iter", -(10 ** 9))
    cached_n = getattr(gaussians, "_knn_reg_n", -1)
    if N <= k:
        gaussians._knn_reg_idx = None
        gaussians._knn_reg_n = N
        gaussians._knn_reg_iter = iteration
        return 0.0
    if idx is None or cached_n != N or (iteration - last) >= interval:
        from eligsir.utils.rocm_knn_fallback import knn_self
        with torch.no_grad():
            _, idx = knn_self(means.detach(), k)       # cross-set search, amortized
        idx = idx.clamp_min(0)                          # -1 (missing) -> 0, harmless
        gaussians._knn_reg_idx = idx
        gaussians._knn_reg_n = N
        gaussians._knn_reg_iter = iteration
    if idx is None or idx.shape[0] != N:
        return 0.0
    s = gaussians.get_scaling                           # [N,3] differentiable
    target = s[idx].mean(dim=1)                          # [N,3] neighbour-mean (detached idx)
    return w * (s - target).abs().mean()
