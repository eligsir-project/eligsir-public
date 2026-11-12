import math
import torch
from eligsir.utils.taming_stats import (
    TAMING_STATS_BACKEND,
    compute_camera_depths,
    compute_exact_taming_stats,
)

SUPPORTED_RASTERIZE_MODES = frozenset({"classic", "antialiased"})


def validate_rasterize_mode(value):
    if not isinstance(value, str):
        raise TypeError("rasterize_mode must be a string")
    if value not in SUPPORTED_RASTERIZE_MODES:
        raise ValueError("rasterize_mode must be exactly one of: antialiased, classic")
    return value


def _fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2.0))


def _expand_packed_stat(meta, key, ids, num_points, device, dtype=torch.float32, reduce_radii=False):
    values = meta.get(key, None)
    out = torch.zeros(num_points, device=device, dtype=dtype)
    if values is None or ids is None:
        return out
    values = values.detach()
    if reduce_radii and values.dim() > 1:
        values = values.max(dim=-1).values
    out[ids] = values.to(device=device, dtype=dtype)
    return out

def _camera_tensors(viewpoint_camera, device):
    gsplat = getattr(viewpoint_camera, "gsplat", None)
    if gsplat is not None:
        target = torch.device(device)
        if (
            gsplat.viewmat.device == target
            and gsplat.K.device == target
            and gsplat.camera_center.device == target
            and gsplat.viewmat.dtype == torch.float32
            and gsplat.K.dtype == torch.float32
            and gsplat.camera_center.dtype == torch.float32
            and gsplat.viewmat.is_contiguous()
            and gsplat.K.is_contiguous()
            and gsplat.camera_center.is_contiguous()
        ):
            return gsplat.viewmat, gsplat.K, gsplat.camera_center
        viewmat = gsplat.viewmat.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        K = gsplat.K.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        cam_center = gsplat.camera_center.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        return viewmat, K, cam_center

    if hasattr(viewpoint_camera, "_gsplat_viewmat"):
        viewmat = viewpoint_camera._gsplat_viewmat.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        K = viewpoint_camera._gsplat_K.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        cam_center = viewpoint_camera._gsplat_camera_center.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
        return viewmat, K, cam_center

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(device, non_blocking=True).contiguous()
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    fx = getattr(viewpoint_camera, "fx", None)
    fy = getattr(viewpoint_camera, "fy", None)
    cx = getattr(viewpoint_camera, "cx", None)
    cy = getattr(viewpoint_camera, "cy", None)
    fx = float(fx) if fx is not None else _fov2focal(viewpoint_camera.FoVx, W)
    fy = float(fy) if fy is not None else _fov2focal(viewpoint_camera.FoVy, H)
    cx = float(cx) if cx is not None else W / 2.0
    cy = float(cy) if cy is not None else H / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        device=device, dtype=torch.float32,
    ).contiguous()
    cam_center = viewpoint_camera.camera_center.to(device=device, dtype=torch.float32).view(1, 3).contiguous()
    return viewmat, K, cam_center


def _warn_if_clamp_mode_inert(pipe):
    """sh_color_clamp_mode has no effect when sh_impl='gsplat' -- native
    gsplat's own CUDA SH kernel does its own clamping internally with no
    hook for _apply_sh_clamp. Warn once per call site if a user has set a
    non-default clamp mode alongside sh_impl='gsplat', since today this is
    silently ignored with no error and no other signal."""
    sh_impl = str(getattr(pipe, "sh_impl", "python")).lower()
    clamp_mode = str(getattr(pipe, "sh_color_clamp_mode", "clamped")).lower()
    if sh_impl == "gsplat" and clamp_mode != "clamped":
        import warnings
        warnings.warn(
            f"sh_color_clamp_mode='{clamp_mode}' has no effect when sh_impl='gsplat' "
            "(native gsplat's SH kernel does its own clamping internally).",
            stacklevel=2,
        )


def _python_sh_colors(pc, means, cam_center, sparse_grad, pipe, update_sh_rest, compiled):
    clamp_mode = str(getattr(pipe, "sh_color_clamp_mode", "clamped")).lower()
    if clamp_mode not in {"clamped", "lower_only"}:
        raise ValueError(
            f"Unsupported sh_color_clamp_mode '{clamp_mode}'. Expected 'clamped' or 'lower_only'."
        )

    if pc.active_sh_degree == 0:
        from eligsir.utils.compiled_kernels import _apply_sh_clamp
        return _apply_sh_clamp(pc._features_dc[:, 0, :], clamp_mode)

    features_rest = pc._features_rest
    if not update_sh_rest:
        features_rest = features_rest.detach()

    _cc = cam_center.to(dtype=means.dtype) if cam_center.dtype != means.dtype else cam_center
    dir_source = means.detach() if (sparse_grad and getattr(pipe, "sparse_mode_detach_sh_dir", True)) else means
    dir_pp = dir_source - _cc
    dir_pp_normalized = torch.nn.functional.normalize(dir_pp, dim=1, eps=1e-8)

    from eligsir.utils.compiled_kernels import sh_to_rgb, sh_to_rgb_eager
    if compiled:
        return sh_to_rgb(pc.active_sh_degree, pc._features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)
    return sh_to_rgb_eager(pc.active_sh_degree, pc._features_dc, features_rest, dir_pp_normalized, clamp_mode=clamp_mode)


def _gsplat_sh_coeffs(pc, update_sh_rest):
    features_rest = pc._features_rest if update_sh_rest else pc._features_rest.detach()
    return torch.cat((pc._features_dc, features_rest), dim=1).contiguous()


def _compute_splat_normals_camera(quats, scales, viewmat):
    """Per-Gaussian surface normals in camera space, as raw [-1, 1] unit vectors.

    Surface normal = shortest-scale axis of each Gaussian, transformed to camera space
    and flipped to face the camera. Output: [N, 3] in [-1, 1] (caller remaps to [0,1]
    after rasterization so compositing operates on signed normals — see render()).
    """
    # Quaternions reaching here may be raw (unnormalised) — the gsplat sparse-grad
    # path passes pc._rotation directly. The wxyz→rotmat formula below assumes a
    # unit quaternion, so normalise first to avoid a |q|² scale on the axes.
    quats = torch.nn.functional.normalize(quats, dim=-1, eps=1e-8)

    N = quats.shape[0]
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    # Quaternion (wxyz) → rotation matrix [N, 3, 3]
    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - z*w),  2*(x*z + y*w),
        2*(x*y + z*w),  1 - 2*(x*x + z*z),  2*(y*z - x*w),
        2*(x*z - y*w),  2*(y*z + x*w),  1 - 2*(x*x + y*y),
    ], dim=1).reshape(N, 3, 3)

    # Shortest scale axis = surface normal in world space
    min_idx = scales.argmin(dim=1)  # [N]
    normals_world = R[torch.arange(N, device=quats.device), :, min_idx]  # [N, 3]

    # Transform to camera space: R_w2c = viewmat[:3, :3]
    R_w2c = viewmat[:3, :3]  # [3, 3]
    normals_cam = (R_w2c @ normals_world.T).T  # [N, 3]

    # Flip normals to face camera (z < 0 in OpenCV/gsplat cam convention → facing camera)
    flip = torch.sign(-normals_cam[:, 2:3])  # -1 or 1; 0 treated as 1 below
    flip = torch.where(flip == 0, torch.ones_like(flip), flip)
    normals_cam = normals_cam * flip

    return normals_cam  # raw [-1, 1]; remapped after compositing in render()


def _encode_normals(normals_raw, valid_mask):
    """Remap raw [-1,1] camera-space normals [3,H,W] → [0,1] for visualisation.

    Pixels where ``valid_mask`` is False are set to neutral 0.5 gray (zero normal).
    """
    vis = (normals_raw * 0.5 + 0.5).clamp(0.0, 1.0)
    vis = torch.where(valid_mask, vis, torch.full_like(vis, 0.5))
    return vis.contiguous()


def _normals_from_depth(depth, K):
    """Surface normals from a rendered depth map (gsplat 2DGS ``surf_normals`` equivalent).

    Backprojects the [1, H, W] camera-space z-depth (metres) to camera-space points
    using intrinsics ``K``, then takes the normalised cross product of the horizontal
    and vertical point gradients. Output: raw [-1, 1] normals [3, H, W] (camera space,
    flipped to face the camera) plus a [1, H, W] bool validity mask.
    """
    z = depth[0]  # [H, W]
    H, W = z.shape
    device = z.device
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    vv, uu = torch.meshgrid(
        torch.arange(H, device=device, dtype=z.dtype),
        torch.arange(W, device=device, dtype=z.dtype),
        indexing="ij",
    )
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    P = torch.stack([x, y, z], dim=0)  # [3, H, W]

    normals = torch.zeros_like(P)
    # Central differences over the interior; cross(dP/du, dP/dv) → surface normal.
    dPdu = P[:, 1:-1, 2:] - P[:, 1:-1, :-2]   # [3, H-2, W-2]
    dPdv = P[:, 2:, 1:-1] - P[:, :-2, 1:-1]   # [3, H-2, W-2]
    normals[:, 1:-1, 1:-1] = torch.cross(dPdu, dPdv, dim=0)
    normals = torch.nn.functional.normalize(normals, dim=0, eps=1e-8)

    # Flip to face camera (z < 0 → facing camera), matching the splat-axis convention.
    flip = torch.sign(-normals[2:3])
    flip = torch.where(flip == 0, torch.ones_like(flip), flip)
    normals = normals * flip

    # Valid where depth is positive and the normal is well-defined (non-zero).
    valid = (z[None] > 0) & (normals.abs().sum(dim=0, keepdim=True) > 1e-6)
    return normals, valid


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, pixel_weights=None,
           return_taming_stats=False, update_sh_rest=True, render_depth=False,
           render_normals=False):
    """
    Render the scene using gsplat (ROCm-compatible backend).

    Background tensor (bg_color) must be on GPU!
    """
    from gsplat.rendering import rasterization

    device = pc.get_xyz.device
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)

    means = pc.get_xyz.contiguous()
    scales = pc.get_scaling.contiguous()
    sparse_grad = bool(getattr(pipe, "gsplat_sparse_grad", False))
    # gsplat sparse gradients cannot backpropagate through PyTorch's dense
    # quaternion normalization. The accelerated path keeps raw rotations
    # normalized after optimizer steps and passes them directly.
    quats = (pc._rotation if sparse_grad else pc.get_rotation).contiguous()
    opacities = pc.get_opacity.squeeze(-1).contiguous()

    if scaling_modifier != 1.0:
        scales = scales * scaling_modifier

    # Camera convention mapping
    # -------------------------------------------------------------------------
    # Inria's codebase stores world_view_transform as the *transpose* of the
    # actual world-to-camera matrix. gsplat expects the actual W2C matrix.
    #
    # Use cached camera tensors prepared by scene.cameras.prepare_camera_for_render
    # and keep a fallback for MiniCam / interactive paths.
    viewmat, K, cam_center = _camera_tensors(viewpoint_camera, device)

    viewmats = viewmat[None].contiguous()
    Ks = K[None].contiguous()

    # Background handling.
    # RGB mode: pass flat [3] tensor — gsplat packed mode accepts this.
    # RGB+D / RGB+ED mode: gsplat's Python-level code does cat([backgrounds, zeros([C,1])],
    # dim=-1) which requires 2D input, but the CUDA kernel then asserts 1D. These
    # two requirements are contradictory in packed+C=1 mode (gsplat bug). Workaround:
    # pass bg=None for the rasterization call and manually composite afterward.
    if render_depth:
        # RGB+ED (not RGB+D): channel 4 is the alpha-normalised "expected" depth
        # (ED = D / alpha), not the raw alpha-composited accumulation. D is
        # reconstructed below as D = ED * alpha — exact per a real-kernel probe
        # (2026-08-11): max abs diff vs a direct RGB+D render stayed at float32
        # noise (~1e-8) across every alpha value gsplat doesn't hard-cull below
        # its ~1/255 per-splat contribution threshold. gsplat's own ED divide
        # IS clamped (render_colors[..., -1:] / render_alphas.clamp(min=1e-10)),
        # but that clamp floor is never reached in practice here because
        # gsplat's contribution-culling threshold (~1/255) is reached first —
        # no fallback branch needed.
        effective_render_mode = "RGB+ED"
        bg = None  # composited manually after rasterization
    else:
        effective_render_mode = getattr(pipe, "render_mode", "RGB")
        bg = bg_color.contiguous() if bg_color is not None else None

    # Colors / SH handling
    # -------------------------------------------------------------------------
    # Python-side SH evaluation to bypass gsplat ROCm SH backward kernel.
    # Optimized: avoid pc.get_features round-trip, slice only active coeffs,
    # and special-case degree 0 to skip eval_sh entirely.
    # For deg >= 1, the full SH pipeline is dispatched through the compile
    # registry (utils.compiled_kernels.sh_to_rgb) for potential torch.compile
    # acceleration. Features rest is passed as full [N, 15, 3] so that function
    # input shapes remain stable across SH degree changes.
    sh_impl = str(getattr(pipe, "sh_impl", "python")).lower()
    _warn_if_clamp_mode_inert(pipe)
    if override_color is not None:
        colors = override_color.contiguous()
        sh_degree = None
    elif sh_impl == "gsplat":
        colors = _gsplat_sh_coeffs(pc, update_sh_rest=update_sh_rest)
        sh_degree = pc.active_sh_degree
    else:
        if sh_impl != "python":
            raise ValueError(
                f"Unsupported SH implementation '{sh_impl}'. "
                "Expected 'python' or 'gsplat'."
            )
        colors = _python_sh_colors(
            pc, means, cam_center, sparse_grad, pipe, update_sh_rest,
            compiled=bool(getattr(pipe, "compile_sh", False)),
        )
        sh_degree = None

    # tile_size=8 causes NaN gradients on ROCm/gfx1151 with wave32-patched gsplat.
    # Default is 16. See docs/ROCM_PODMAN.md.
    tile_size = getattr(pipe, "tile_size", 16)
    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        near_plane=getattr(pipe, "near_plane", 0.01),
        far_plane=getattr(pipe, "far_plane", 1e10),
        radius_clip=getattr(pipe, "radius_clip", 0.0),
        eps2d=getattr(pipe, "eps2d", 0.3),
        sh_degree=sh_degree,
        packed=True,
        tile_size=tile_size,
        backgrounds=bg,
        render_mode=effective_render_mode,
        sparse_grad=sparse_grad,
        absgrad=getattr(pipe, "absgrad", False),
        rasterize_mode=validate_rasterize_mode(
            getattr(pipe, "rasterize_mode", "classic")
        ),
    )

    # gsplat returns [C, H, W, channels]; Inria backend returns [channels, H, W]
    rendered = render_colors[0].permute(2, 0, 1).contiguous()
    image = rendered[:3] if rendered.shape[0] >= 3 else rendered

    # When render_depth=True we pass bg=None to avoid a gsplat packed+RGB+D bug.
    # Manually composite the background onto the RGB channels using the alpha map.
    if render_depth and bg_color is not None:
        alpha = render_alphas[0].permute(2, 0, 1)  # [1, H, W]
        image = image + bg_color[:, None, None] * (1.0 - alpha)
        rendered = torch.cat([image, rendered[3:]], dim=0)  # rebuild [4, H, W]

    # Expand packed metadata back to full [N] arrays for compatibility
    N = means.shape[0]
    radii = torch.zeros(N, device=device, dtype=torch.int32)
    is_used = torch.zeros(N, device=device, dtype=torch.bool)

    if "gaussian_ids" in meta and meta["gaussian_ids"] is not None:
        ids = meta["gaussian_ids"]
        tile_radii = meta["radii"]  # [M, 2] — (x_radius, y_radius) per gaussian
        # Inria backend returns a single scalar radius = max of x/y extent
        radii_scalar = tile_radii.max(dim=-1).values.to(torch.int32)  # [M]
        radii[ids] = radii_scalar
        is_used[ids] = True
    else:
        radii = meta.get("radii", torch.zeros(N, device=device, dtype=torch.int32))
        is_used = radii > 0

    # Dummy screenspace_points for API parity. NOT connected to rasterisation —
    # gradient-based densification strategies must not read viewspace_points.grad.
    screenspace_points = torch.zeros_like(means, dtype=means.dtype, requires_grad=False, device=device)

    visibility_filter = radii > 0

    result = {
        "render": image,
        # Placeholder; when render_depth=True this is corrected below so
        # channel 3 is restored to D (reconstructed), preserving
        # render_full's pre-ED-switch contract for any consumer reading the
        # depth channel directly instead of via result["rendered_depth"].
        "render_full": rendered,
        "viewspace_points": screenspace_points,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "is_used": is_used,
        "alpha": render_alphas[0],
        "meta": meta,
        "sparse_grad": sparse_grad,
    }
    if render_depth:
        # rendered[3:4] is now the alpha-normalised "expected" (ED) depth in
        # camera space (metres), since effective_render_mode is "RGB+ED"
        # above. Reconstruct the original accumulated "D" depth as
        # D = ED * alpha (exact, per the raw-divide finding above) so every
        # existing consumer of "rendered_depth" (free-space loss, the
        # depth-residual diagnostic, utils/comparison_report.py,
        # geometry-metric tests) sees unchanged values. The raw ED is
        # exposed separately as "rendered_depth_ed" for the new, flag-gated
        # depth-loss experiment (--streaming_depth_loss_quantity ed).
        #
        # NOTE: result["alpha"] (= render_alphas[0]) is [H, W, 1], NOT
        # [1, H, W] — confirmed empirically 2026-08-11 (a real-kernel probe
        # showed render_alphas[0].shape == [64, 64, 1]); existing consumers
        # (e.g. objective.py's free-space loss) already permute it
        # defensively before use. Do not reuse result["alpha"] directly here
        # (an in-place [1,H,W] * [H,W,1] multiply would silently broadcast
        # to the wrong shape whenever H == W) — build a correctly-permuted
        # [1, H, W] alpha map locally instead, matching the pattern used for
        # normals a few lines below.
        _depth_alpha_map = render_alphas[0].permute(2, 0, 1)  # [1, H, W]
        result["rendered_depth_ed"] = rendered[3:4]  # [1, H, W], ED (alpha-normalised)
        result["rendered_depth"] = rendered[3:4] * _depth_alpha_map  # [1, H, W], reconstructed D
        # Restore render_full's depth channel to D (was raw ED from the
        # RGB+ED rasterization call above) so this key's contract is
        # unchanged for any consumer reading channel 3 directly, not just
        # ones using result["rendered_depth"].
        result["render_full"] = torch.cat([rendered[:3], result["rendered_depth"]], dim=0)

    if render_normals:
        # Per-splat shortest-axis normals as raw [-1,1] camera-space colors. Composite
        # over a ZERO background (not 0.5) and re-normalise afterwards so the alpha
        # blend operates on signed unit vectors rather than [0,1]-encoded values —
        # this keeps the result a unit normal and avoids muddy edges/floaters.
        # render_mode="RGB+ED" gives the *expected* (alpha-normalised) depth in the
        # same pass — metrically correct per pixel for backprojection, unlike the
        # accumulated "D" depth which is underestimated at partially-covered pixels.
        # Per the packed+RGB+D gsplat bug (see background handling above), pass
        # backgrounds=None and rely on the composite-over-zero default.
        normals_raw = _compute_splat_normals_camera(quats.detach(), scales.detach(), viewmat.detach())
        normals_out, normals_alpha, _ = rasterization(
            means=means.detach(),
            quats=quats.detach(),
            scales=scales.detach(),
            opacities=opacities.detach(),
            colors=normals_raw,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=getattr(pipe, "near_plane", 0.01),
            far_plane=getattr(pipe, "far_plane", 1e10),
            radius_clip=getattr(pipe, "radius_clip", 0.0),
            eps2d=getattr(pipe, "eps2d", 0.3),
            sh_degree=None,
            packed=True,
            tile_size=tile_size,
            backgrounds=None,
            render_mode="RGB+ED",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=validate_rasterize_mode(
                getattr(pipe, "rasterize_mode", "classic")
            ),
        )
        normals_full = normals_out[0].permute(2, 0, 1).contiguous()  # [4, H, W] (3 normal + depth)
        alpha_map = normals_alpha[0].permute(2, 0, 1)                # [1, H, W]
        covered = alpha_map > 1e-3

        # Splat-axis normals (Change C): re-normalise composited vector, then encode.
        splat_normals_raw = torch.nn.functional.normalize(normals_full[:3], dim=0, eps=1e-8)
        result["rendered_normals"] = _encode_normals(splat_normals_raw, covered)

        # Depth-derived surface normals (gsplat 2DGS surf_normals equivalent).
        depth_map = normals_full[3:4]  # [1, H, W] camera-space z (metres)
        surf_raw, surf_valid = _normals_from_depth(depth_map, K)
        result["rendered_surf_normals"] = _encode_normals(surf_raw, surf_valid & covered)

        # Raw (signed, un-encoded) outputs for quantitative geometry metrics
        # (utils/scene_geometry_metrics): splat-axis vs depth normal agreement, and
        # the expected (RGB+ED) depth for sensor-depth accuracy. K is exposed so the
        # report can derive sensor-depth normals in the same camera convention.
        result["rendered_normals_raw"] = splat_normals_raw            # [3,H,W], signed unit
        result["rendered_surf_normals_raw"] = surf_raw                # [3,H,W], signed unit
        result["rendered_normals_valid"] = (surf_valid & covered)     # [1,H,W] bool
        result["rendered_normals_depth"] = depth_map                  # [1,H,W] expected z (m)
        result["rendered_normals_K"] = K                              # [3,3] intrinsics

    if return_taming_stats or pixel_weights is not None:
        radii_float = radii.to(dtype=torch.float32)
        depths = meta.get("depths", None)
        if depths is None:
            depths = meta.get("depth", None)
        if depths is not None and depths.shape[0] != N:
            full_depths = torch.zeros(N, device=device, dtype=torch.float32)
            if "gaussian_ids" in meta and meta["gaussian_ids"] is not None:
                full_depths[meta["gaussian_ids"]] = depths.detach().to(dtype=torch.float32)
            depths = full_depths
        elif depths is None:
            depths = compute_camera_depths(means.detach(), viewmat.detach()).to(dtype=torch.float32)
        else:
            depths = depths.detach().to(dtype=torch.float32)

        exact_stats = compute_exact_taming_stats(meta, pixel_weights, N, W, H, device)
        result.update({
            "gaussian_depths": depths,
            "gaussian_radii": radii_float,
            **exact_stats,
            "taming_stats_exact": True,
            "taming_stats_backend": TAMING_STATS_BACKEND,
            "taming_stats_approximation": None,
        })

    return result


def render_batch(viewpoint_cameras, pc, pipe, bg_color: torch.Tensor,
                 chunk_size: int = 32) -> list:
    """Render multiple cameras via batched gsplat rasterization.

    All cameras must have the same image dimensions. Falls back to sequential
    render() calls if cameras have mismatched sizes or on any error.
    Returns a list of (3, H, W) float tensors clamped [0, 1].

    Issue #1326: this function must call gsplat's rasterization() with
    packed=False, not packed=True. Confirmed on real GPU hardware (r9700,
    gfx1201) that gsplat's packed=True forward kernel non-deterministically
    corrupts every camera slot beyond index 0 in a C>1 batch when `colors`
    is a per-camera-varying (C, N, 3) tensor with sh_degree=None -- exactly
    the shape this function's sh_impl="python" branch builds (colors_list
    stacked per camera). Repeated calls with byte-identical inputs produced
    different, sometimes wildly out-of-range (e.g. >100 pre-clamp) results
    for the same camera slot, consistent with the kernel reading
    uninitialized/reused GPU memory rather than a deterministic logic bug.
    packed=False reproduced the correct per-camera reference render
    (verified against render()'s own single-camera path) across 8 repeated
    attempts, with and without torch.cuda.empty_cache() in between. This
    path always runs with sparse_grad=False (see below), so packed=False
    costs no functionality here -- sparse_grad is only effective under
    packed=True and this function never uses it.
    """
    if not viewpoint_cameras:
        return []
    from gsplat.rendering import rasterization

    if len(viewpoint_cameras) == 1:
        with torch.no_grad():
            return [render(viewpoint_cameras[0], pc, pipe, bg_color)["render"].clamp(0, 1)]

    device = pc.get_xyz.device
    W = int(viewpoint_cameras[0].image_width)
    H = int(viewpoint_cameras[0].image_height)

    # Verify all cameras share H, W — fall back if not
    for cam in viewpoint_cameras:
        if int(cam.image_width) != W or int(cam.image_height) != H:
            with torch.no_grad():
                return [render(cam, pc, pipe, bg_color)["render"].clamp(0, 1)
                        for cam in viewpoint_cameras]

    means = pc.get_xyz.contiguous()
    scales = pc.get_scaling.contiguous()
    quats = pc.get_rotation.contiguous()
    opacities = pc.get_opacity.squeeze(-1).contiguous()
    bg = bg_color.contiguous() if bg_color is not None else None

    sh_impl = str(getattr(pipe, "sh_impl", "python")).lower()
    tile_size = getattr(pipe, "tile_size", 16)

    results = []
    for start in range(0, len(viewpoint_cameras), chunk_size):
        chunk = viewpoint_cameras[start:start + chunk_size]
        C = len(chunk)

        viewmats_list = []
        Ks_list = []

        if sh_impl == "gsplat":
            colors = _gsplat_sh_coeffs(pc, update_sh_rest=False)
            sh_degree = pc.active_sh_degree
            for cam in chunk:
                vm, K, _ = _camera_tensors(cam, device)
                viewmats_list.append(vm)
                Ks_list.append(K)
        else:
            colors_list = []
            sh_degree = None
            compiled = bool(getattr(pipe, "compile_sh", False))
            for cam in chunk:
                vm, K, cam_center = _camera_tensors(cam, device)
                viewmats_list.append(vm)
                Ks_list.append(K)
                c = _python_sh_colors(pc, means, cam_center, False, pipe, False, compiled)
                colors_list.append(c)
            colors = torch.stack(colors_list, dim=0)  # (C, N, 3)

        viewmats = torch.stack(viewmats_list, dim=0).contiguous()  # (C, 4, 4)
        Ks = torch.stack(Ks_list, dim=0).contiguous()              # (C, 3, 3)

        # Issue #1326: packed=False, not packed=True -- see the module-level
        # docstring above for the root cause. Unpacked mode requires a
        # per-camera background of shape (C, channels), not the flat (3,)
        # tensor packed mode accepted.
        bg_chunk = bg.unsqueeze(0).expand(C, -1).contiguous() if bg is not None else None

        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
            near_plane=getattr(pipe, "near_plane", 0.01),
            far_plane=getattr(pipe, "far_plane", 1e10),
            radius_clip=getattr(pipe, "radius_clip", 0.0),
            eps2d=getattr(pipe, "eps2d", 0.3),
            sh_degree=sh_degree,
            packed=False,
            tile_size=tile_size,
            backgrounds=bg_chunk,
            render_mode="RGB",
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=validate_rasterize_mode(
                getattr(pipe, "rasterize_mode", "classic")
            ),
        )

        for c_idx in range(C):
            img = render_colors[c_idx].permute(2, 0, 1).contiguous().clamp(0, 1)
            results.append(img)

    return results
