"""Canonical per-work-plan supervision resolution materialization."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as functional

from eligsir.contracts.control import ResolutionTier

#: Selectable RGB/depth resampling strategies for materialize_resolution_tier().
#: "legacy" is the only strategy that existed before 2026-08-29 and remains the
#: default -- passing no resampling_mode/streaming_resolution_resampling_mode
#: reproduces the exact pre-existing behavior byte-for-byte.
RESAMPLING_MODES = ("legacy", "masked", "ray_preserving")

#: Selectable coarse-tier sensor-depth downsampling strategies for the
#: depth-supervision loss (issue #1833), independent of RESAMPLING_MODES.
#: Only consulted by the "legacy"/"masked" area/nearest resize branch (not
#: "ray_preserving", which never resizes -- strided decimation already picks
#: a single native ray per output pixel, so there is nothing to average).
#: "nearest" is the default and reproduces the exact pre-#1833 behavior
#: (F.interpolate(mode="nearest")) byte-for-byte.
DEPTH_DOWNSAMPLE_MODES = ("nearest", "valid_area")


@dataclass(frozen=True, slots=True)
class _NativeCameraState:
    rgb: torch.Tensor
    depth: torch.Tensor | None
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    score_provenance: object | None


def _native_state(camera) -> _NativeCameraState:
    state = getattr(camera, "_canonical_native_resolution_state", None)
    if state is not None:
        return state
    rgb = getattr(camera, "original_image", None)
    if not isinstance(rgb, torch.Tensor) or rgb.ndim != 3:
        raise ValueError("resolution materialization requires CHW camera.original_image")
    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    if (int(getattr(camera, "image_width", width)), int(getattr(camera, "image_height", height))) != (width, height):
        raise ValueError("camera image dimensions disagree with original_image")
    values = tuple(getattr(camera, name, None) for name in ("fx", "fy", "cx", "cy"))
    if not all(value is not None for value in values):
        raise ValueError("resolution materialization requires camera intrinsics")
    depth = getattr(camera, "_sensor_depth_cache", None)
    if depth is not None and not isinstance(depth, torch.Tensor):
        raise TypeError("camera depth cache must be a tensor when present")
    score_provenance = getattr(camera, "score_provenance", None)
    if score_provenance is not None:
        # `score_provenance` is captured here alongside `rgb` and later treated as
        # describing it (issue #1259). A stale provenance left over from a prior
        # tier/materialization of this SAME camera object (e.g. a cold-restore path
        # that reset `original_image` without refreshing `score_provenance`) would
        # otherwise be silently paired with these fresh native pixels and only fail
        # much later, deep inside a scoring call, with no link back to this pairing.
        # Fail here, at the pairing site, instead.
        canonical = getattr(score_provenance, "canonical_score_domain", None)
        recorded_resolution = getattr(canonical, "resolution", None)
        recorded_dtype = getattr(canonical, "dtype", None)
        actual_dtype = str(rgb.dtype).replace("torch.", "")
        if recorded_resolution is not None and tuple(recorded_resolution) != (height, width):
            raise ValueError(
                "camera.score_provenance resolution "
                f"{tuple(recorded_resolution)} does not match native original_image "
                f"{(height, width)} -- stale provenance from a prior materialization"
            )
        if recorded_dtype is not None and recorded_dtype != actual_dtype:
            raise ValueError(
                f"camera.score_provenance dtype {recorded_dtype!r} does not match "
                f"native original_image dtype {actual_dtype!r} -- stale provenance "
                "from a prior materialization"
            )
    state = _NativeCameraState(
        rgb,
        depth,
        width,
        height,
        *(float(value) for value in values),
        getattr(camera, "score_provenance", None),
    )
    camera._canonical_native_resolution_state = state
    return state


def _resize(tensor: torch.Tensor, *, height: int, width: int, mode: str) -> torch.Tensor:
    original_dims = tensor.ndim
    if original_dims == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif original_dims == 3:
        tensor = tensor.unsqueeze(0)
    else:
        raise ValueError("supervision tensors must be HW or CHW")
    result = functional.interpolate(tensor, size=(height, width), mode=mode)
    return result.squeeze(0) if original_dims == 3 else result.squeeze(0).squeeze(0)


def _strided_decimate(tensor: torch.Tensor, *, height: int, width: int, factor: int) -> torch.Tensor:
    """Zero-phase strided decimation: out[..., i, j] = tensor[..., factor*i, factor*j].

    No filtering, no interpolation -- every output sample IS a native sample,
    picked at offset 0 so the sampled coordinate set nests across tiers
    (every quarter-tier coordinate is also a half-tier coordinate and a
    native-grid coordinate; see ``materialize_resolution_tier``'s docstring).
    """
    if tensor.ndim == 2:
        return tensor[: height * factor : factor, : width * factor : factor]
    if tensor.ndim == 3:
        return tensor[:, : height * factor : factor, : width * factor : factor]
    raise ValueError("supervision tensors must be HW or CHW")


def _resize_depth_valid_area(tensor: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Area-average finite, positive depth samples within each output footprint.

    Uses the same adaptive block partitioning ``F.interpolate(mode="area")``
    (equivalently ``F.adaptive_avg_pool2d``) uses for RGB, so a coarse-tier
    output pixel averages exactly the native pixels the RGB resize also
    blends -- but only the ones with finite, positive sensor depth, instead
    of picking one arbitrary native pixel via ``mode="nearest"``. The
    per-block pixel count cancels between the numerator and denominator
    pooling calls (both use the identical output size on the identical input
    size), leaving exactly ``sum(valid depth) / count(valid)`` regardless of
    whether native dimensions divide evenly by the output size. Blocks with
    no valid pixel at all are exactly ``0.0`` (never NaN/inf).
    """
    valid = _valid_depth_mask(tensor)
    numerator = _resize(
        torch.where(valid, tensor, 0.0), height=height, width=width, mode="area"
    )
    denominator = _resize(valid.to(tensor.dtype), height=height, width=width, mode="area")
    return numerator / denominator.masked_fill(denominator == 0, 1)


def _resize_depth(
    tensor: torch.Tensor, *, height: int, width: int, depth_downsample: str
) -> torch.Tensor:
    """Dispatch depth resize to ``mode="nearest"`` or #1833's valid-area average."""
    if depth_downsample not in DEPTH_DOWNSAMPLE_MODES:
        raise ValueError(
            f"depth_downsample must be one of {DEPTH_DOWNSAMPLE_MODES!r}, got {depth_downsample!r}"
        )
    if depth_downsample == "valid_area":
        return _resize_depth_valid_area(tensor, height=height, width=width)
    return _resize(tensor, height=height, width=width, mode="nearest")


def _valid_depth_mask(depth: torch.Tensor) -> torch.Tensor:
    """Canonical invalid-depth convention used across this codebase.

    Matches ``eligsir.mapping.geometry.eval_masks.invalid_depth_mask``'s
    validity predicate (``isfinite(depth) & depth > 0``), inverted.
    """
    return torch.isfinite(depth) & (depth > 0)


def _discontinuity_confidence_weight(
    native_depth_hw: torch.Tensor,
    *,
    out_height: int,
    out_width: int,
    factor: int,
    abs_floor_m: float,
    rel_frac: float,
) -> torch.Tensor:
    """Per-output-pixel depth confidence weight in (0, 1], 1.0 = no discontinuity.

    For each output pixel's ``factor``x``factor`` native footprint, computes
    ``delta_z = max(valid depths) - min(valid depths)`` (invalid/zero/non-finite
    samples excluded, see ``_valid_depth_mask``) and a distance-scaled
    threshold ``threshold(z) = max(abs_floor_m, rel_frac * z)`` where ``z`` is
    the footprint's nearest (minimum) valid depth. Footprints with fewer than
    two valid samples cannot exhibit a discontinuity and get weight 1.0.
    A footprint at or under threshold gets weight 1.0; beyond it the weight
    decays smoothly as ``threshold / delta_z`` -- never a hard binary cliff.
    """
    cropped = native_depth_hw[: out_height * factor, : out_width * factor].to(torch.float32)
    # Minimum-size tiers can have a final footprint smaller than ``factor``.
    # Pad only that missing extent as invalid samples so the one output pixel
    # still measures the real available native footprint instead of failing a
    # fixed-factor reshape.
    padded = torch.zeros(
        (out_height * factor, out_width * factor), dtype=cropped.dtype, device=cropped.device
    )
    padded[: cropped.shape[0], : cropped.shape[1]] = cropped
    cropped = padded
    valid = _valid_depth_mask(cropped)
    blocks = cropped.reshape(out_height, factor, out_width, factor).permute(0, 2, 1, 3)
    blocks = blocks.reshape(out_height, out_width, factor * factor)
    valid_blocks = valid.reshape(out_height, factor, out_width, factor).permute(0, 2, 1, 3)
    valid_blocks = valid_blocks.reshape(out_height, out_width, factor * factor)

    pos_inf = torch.full_like(blocks, float("inf"))
    neg_inf = torch.full_like(blocks, float("-inf"))
    max_z = torch.where(valid_blocks, blocks, neg_inf).amax(dim=-1)
    min_z = torch.where(valid_blocks, blocks, pos_inf).amin(dim=-1)
    n_valid = valid_blocks.sum(dim=-1)
    has_discontinuity_evidence = n_valid >= 2

    delta_z = torch.where(has_discontinuity_evidence, max_z - min_z, torch.zeros_like(max_z))
    z_ref = torch.where(has_discontinuity_evidence, min_z, torch.zeros_like(min_z))
    threshold = torch.clamp(rel_frac * z_ref, min=abs_floor_m)

    excess = torch.clamp(delta_z - threshold, min=0.0)
    weight = torch.where(
        has_discontinuity_evidence,
        threshold / (threshold + excess),
        torch.ones_like(delta_z),
    )
    return weight.contiguous()


def resolution_tier_geometry(
    *,
    native_width: int,
    native_height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tier: ResolutionTier,
    resampling_mode: str = "legacy",
) -> tuple[int, int, float, float, float, float]:
    """Return the canonical tier dimensions and CPU-side camera scalars.

    These immutable scalars are sealed into a prepared tier before READY;
    consumers therefore never inspect device K values to rebuild calibration.
    """

    if not isinstance(tier, ResolutionTier):
        raise TypeError("tier must be ResolutionTier")
    if resampling_mode not in RESAMPLING_MODES:
        raise ValueError(
            f"resampling_mode must be one of {RESAMPLING_MODES!r}, got {resampling_mode!r}"
        )
    if native_width < 1 or native_height < 1:
        raise ValueError("native tier geometry must be non-empty")
    factor = tier.downsample_factor
    if factor == 1:
        return native_width, native_height, fx, fy, cx, cy
    width = max(1, native_width // factor)
    height = max(1, native_height // factor)
    if resampling_mode == "ray_preserving":
        scale = float(factor)
        return width, height, fx / scale, fy / scale, cx / scale, cy / scale
    scale_x, scale_y = width / native_width, height / native_height
    return (
        width,
        height,
        fx * scale_x,
        fy * scale_y,
        (cx + 0.5) * scale_x - 0.5,
        (cy + 0.5) * scale_y - 0.5,
    )


def materialize_resolution_tensors(
    rgb_hwc: torch.Tensor,
    depth_hw: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    tier: ResolutionTier,
    *,
    resampling_mode: str = "legacy",
    discontinuity_abs_floor_m: float = 0.05,
    discontinuity_rel_frac: float = 0.02,
    depth_downsample: str = "nearest",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Derive one device-ready tier using the canonical camera semantics.

    The prepared-source path owns HWC RGB and HW metric depth, while canonical
    cameras use CHW RGB.  This is the only representation difference: pixel
    resampling and intrinsics follow :func:`materialize_resolution_tier`
    exactly, so preparation can move that work before READY without changing
    what a planned tier means.
    """

    if not isinstance(tier, ResolutionTier):
        raise TypeError("tier must be ResolutionTier")
    if resampling_mode not in RESAMPLING_MODES:
        raise ValueError(
            f"resampling_mode must be one of {RESAMPLING_MODES!r}, got {resampling_mode!r}"
        )
    if depth_downsample not in DEPTH_DOWNSAMPLE_MODES:
        raise ValueError(
            f"depth_downsample must be one of {DEPTH_DOWNSAMPLE_MODES!r}, got {depth_downsample!r}"
        )
    if not isinstance(rgb_hwc, torch.Tensor) or rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3:
        raise ValueError("prepared RGB must be HWC with three channels")
    if not isinstance(depth_hw, torch.Tensor) or depth_hw.ndim != 2:
        raise ValueError("prepared depth must be HW")
    if tuple(depth_hw.shape) != tuple(rgb_hwc.shape[:2]):
        raise ValueError("prepared depth must match RGB geometry")
    if (
        not isinstance(camera_intrinsics, torch.Tensor)
        or tuple(camera_intrinsics.shape) != (3, 3)
        or camera_intrinsics.device != rgb_hwc.device
        or depth_hw.device != rgb_hwc.device
    ):
        raise ValueError("prepared RGB, depth and intrinsics must share one device")

    native_height, native_width = (int(dim) for dim in rgb_hwc.shape[:2])
    rgb_chw = rgb_hwc.permute(2, 0, 1)
    factor = tier.downsample_factor
    confidence = None
    if resampling_mode == "ray_preserving" and factor != 1:
        width = max(1, native_width // factor)
        height = max(1, native_height // factor)
        rgb_chw = _strided_decimate(rgb_chw, height=height, width=width, factor=factor)
        depth = _strided_decimate(depth_hw, height=height, width=width, factor=factor)
        scale_x = scale_y = 1.0 / float(factor)
        principal_offset = 0.0
    elif factor == 1:
        width, height = native_width, native_height
        depth = depth_hw
        scale_x = scale_y = 1.0
        principal_offset = 0.0
    else:
        width = max(1, native_width // factor)
        height = max(1, native_height // factor)
        rgb_chw = _resize(rgb_chw, height=height, width=width, mode="area")
        depth = _resize_depth(depth_hw, height=height, width=width, depth_downsample=depth_downsample)
        scale_x, scale_y = width / native_width, height / native_height
        principal_offset = 0.5
        if resampling_mode == "masked":
            confidence = _discontinuity_confidence_weight(
                depth_hw,
                out_height=height,
                out_width=width,
                factor=factor,
                abs_floor_m=float(discontinuity_abs_floor_m),
                rel_frac=float(discontinuity_rel_frac),
            )

    intrinsics = camera_intrinsics.clone()
    intrinsics[0, 0] *= scale_x
    intrinsics[1, 1] *= scale_y
    if principal_offset == 0.0:
        intrinsics[0, 2] *= scale_x
        intrinsics[1, 2] *= scale_y
    else:
        intrinsics[0, 2] = (intrinsics[0, 2] + principal_offset) * scale_x - principal_offset
        intrinsics[1, 2] = (intrinsics[1, 2] + principal_offset) * scale_y - principal_offset
    return rgb_chw.permute(1, 2, 0).contiguous(), depth.contiguous(), intrinsics, confidence


def materialize_resolution_tier(
    camera,
    tier: ResolutionTier,
    *,
    resampling_mode: str = "legacy",
    discontinuity_abs_floor_m: float = 0.05,
    discontinuity_rel_frac: float = 0.02,
    depth_downsample: str = "nearest",
) -> None:
    """Bind ``camera`` to a tier derived solely from its canonical native view.

    This is deliberately per mapping observation.  It changes no source packet,
    admission state, or insertion budget, and it invalidates renderer camera
    tensors so their intrinsics are rebuilt with the same geometry as RGB/depth.

    ``resampling_mode`` selects between three independently selectable
    strategies (default ``"legacy"``, zero behavior change from before this
    parameter existed):

    * ``"legacy"``: RGB via ``F.interpolate(mode="area")``, depth via
      ``F.interpolate(mode="nearest")``, resampled independently. This is
      exactly the behavior that existed before 2026-08-29.
    * ``"masked"``: identical area/nearest resampling as ``"legacy"``, plus a
      per-pixel depth-discontinuity confidence weight (see
      ``_discontinuity_confidence_weight``) attached to the camera as
      ``camera._resolution_depth_confidence_weight``. Consumed only by the
      depth-supervision loss (``eligsir.optimization.gaussian.
      depth_loss_weighting``); the RGB/photometric loss is unaffected -- area
      averaging is legitimate anti-aliasing for photometric supervision, the
      problem is only a mixed-surface RGB pixel silently supervising depth
      toward an invented value.
    * ``"ray_preserving"``: replaces the resampling itself with zero-phase
      strided decimation, ``tier_rgb[i, j] = native_rgb[factor*i, factor*j]``
      and identically for depth. Every low-resolution supervision pixel is
      then an EXACT single native camera ray (not a blend of several), so
      the RGB/depth mismatch ``"masked"`` guards against cannot occur in the
      first place. Offset is zero for every tier (not ``factor/2`` or any
      other phase) so the sampled coordinate sets NEST: every quarter-tier
      coordinate is also a half-tier coordinate and a native-grid coordinate.
      This makes increasing resolution (quarter -> half -> native) strictly
      ADD supervision rays rather than replace them with different ones.
      Intrinsics use the matching zero-phase analytic formula
      ``f' = f / s``, ``c' = c / s`` -- NOT the pixel-center-aware
      ``(c + 0.5) * s - 0.5`` formula the area-resize legacy/masked paths
      use, which assumes a pixel-center-aligned continuous resample rather
      than an exact discrete pick.

      PREREQUISITE (not verified by this function): ray-preserving decimation
      is only correct when native RGB pixel (u, v) and native depth pixel
      (u, v) already represent the same physical ray -- i.e. RGB and depth
      are co-registered at native resolution before this function runs. This
      holds for TUM (single calibrated RGB-D sensor) and for a validated
      Orbbec source (hardware ``align_target_stream`` registration, see
      CLAUDE.md's live-camera hard constraint). It does NOT hold for
      ScanNet's raw-sensor path when its depth sensor's native resolution
      differs from the color camera's (``StreamingRGBDFrame.depth_fx`` set):
      ``eligsir/mapping/integration/depth_helpers.py``'s
      ``_get_sensor_depth`` (see its docstring/comments around lines 236-268)
      reconciles that shape mismatch with a plain
      ``F.interpolate(mode="nearest")`` reshape onto the color grid, NOT a
      geometric depth-to-color reprojection (unlike
      ``scene/streaming_scene.py``'s point-cloud backprojection path around
      line 1526, which does reproject through ``depth_fx``/``depth_cx``).
      Shape reconciliation is not proof of registration: on that path,
      native depth (u, v) after the nearest-neighbor reshape does not
      necessarily correspond to the same ray as native RGB (u, v). This
      function detects that case from ``camera._streaming_frame.depth_fx``
      (set and differing from the color camera's ``fx``) and falls back to
      ``"legacy"`` resampling instead of silently mis-supervising depth
      against the wrong native ray (issue #1260). This is a coarse signal --
      a source could set ``depth_fx`` yet still be genuinely registered, or
      leave it unset yet be unregistered by some other means -- but it
      catches ScanNet's raw-sensor case, the only one so far confirmed
      unregistered on this path.
    """

    if not isinstance(tier, ResolutionTier):
        raise TypeError("tier must be ResolutionTier")
    if resampling_mode not in RESAMPLING_MODES:
        raise ValueError(
            f"resampling_mode must be one of {RESAMPLING_MODES!r}, got {resampling_mode!r}"
        )
    if depth_downsample not in DEPTH_DOWNSAMPLE_MODES:
        raise ValueError(
            f"depth_downsample must be one of {DEPTH_DOWNSAMPLE_MODES!r}, got {depth_downsample!r}"
        )
    if resampling_mode == "ray_preserving":
        # Issue #1260: ray-preserving decimation is only correct when native
        # RGB (u, v) and native depth (u, v) already represent the same
        # physical ray. `depth_fx` set (and differing from the color `fx`)
        # means the source's depth sensor has its own native resolution that
        # `_get_sensor_depth` reconciles onto the color grid with a plain
        # nearest-neighbor reshape (see this function's docstring) -- not a
        # geometric depth-to-color reprojection. Detect that case here and
        # fall back to "legacy" rather than silently mis-supervising depth
        # against the wrong ray.
        frame = getattr(camera, "_streaming_frame", None)
        depth_fx = getattr(frame, "depth_fx", None)
        color_fx = getattr(frame, "fx", None)
        if depth_fx is not None and (color_fx is None or float(depth_fx) != float(color_fx)):
            if not getattr(camera, "_ray_preserving_unregistered_depth_logged", False):
                print(
                    "[resolution_materialization] ray_preserving requested for a "
                    "source whose depth sensor is not registered to its color "
                    f"camera (depth_fx={depth_fx!r} != fx={color_fx!r}); falling "
                    "back to legacy resampling for this camera to avoid mis-"
                    "supervising depth against the wrong native ray. See issue "
                    "#1260.",
                    flush=True,
                )
                camera._ray_preserving_unregistered_depth_logged = True
            resampling_mode = "legacy"
    existing_native = getattr(camera, "_canonical_native_resolution_state", None)
    current_rgb = getattr(camera, "original_image", None)
    current_depth = getattr(camera, "_sensor_depth_cache", None)
    if (
        existing_native is not None
        and getattr(camera, "_canonical_resolution_binding_native_state", None)
        is existing_native
        and getattr(camera, "_canonical_resolution_tier", None) is tier
        and getattr(camera, "_canonical_resolution_resampling_mode", None) == resampling_mode
        and getattr(camera, "_canonical_resolution_depth_downsample", None) == depth_downsample
        and isinstance(current_rgb, torch.Tensor)
        and (
            existing_native.depth is None
            or isinstance(current_depth, torch.Tensor)
        )
    ):
        return
    native = _native_state(camera)
    factor = tier.downsample_factor
    discontinuity_weight = None
    if resampling_mode == "ray_preserving" and factor != 1:
        width = max(1, native.width // factor)
        height = max(1, native.height // factor)
        scale = float(factor)
        rgb = _strided_decimate(native.rgb, height=height, width=width, factor=factor)
        depth = (
            None
            if native.depth is None
            else _strided_decimate(native.depth, height=height, width=width, factor=factor)
        )
        fx_new = native.fx / scale
        fy_new = native.fy / scale
        cx_new = native.cx / scale
        cy_new = native.cy / scale
    elif factor == 1:
        width, height = native.width, native.height
        rgb, depth = native.rgb, native.depth
        fx_new, fy_new, cx_new, cy_new = native.fx, native.fy, native.cx, native.cy
    else:
        width = max(1, native.width // factor)
        height = max(1, native.height // factor)
        scale_x, scale_y = width / native.width, height / native.height
        rgb = _resize(native.rgb, height=height, width=width, mode="area")
        depth = (
            None
            if native.depth is None
            else _resize_depth(native.depth, height=height, width=width, depth_downsample=depth_downsample)
        )
        fx_new = native.fx * scale_x
        fy_new = native.fy * scale_y
        cx_new = (native.cx + 0.5) * scale_x - 0.5
        cy_new = (native.cy + 0.5) * scale_y - 0.5
        if resampling_mode == "masked" and native.depth is not None:
            native_depth_hw = native.depth
            if native_depth_hw.ndim == 3:
                native_depth_hw = native_depth_hw.squeeze(0)
            discontinuity_weight = _discontinuity_confidence_weight(
                native_depth_hw,
                out_height=height,
                out_width=width,
                factor=factor,
                abs_floor_m=float(discontinuity_abs_floor_m),
                rel_frac=float(discontinuity_rel_frac),
            )
    camera.original_image = rgb.contiguous()
    if native.score_provenance is not None:
        from eligsir.utils.image_provenance import TransformStep, TransformType
        from eligsir.utils.score_contract import attach_score_provenance

        provenance = native.score_provenance
        if factor != 1:
            provenance = replace(
                provenance,
                canonical_score_domain=replace(
                    provenance.canonical_score_domain,
                    resolution=(height, width),
                    is_declared_transform=True,
                ),
            ).add_step(
                TransformStep(
                    transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
                    description=(
                        f"{resampling_mode} resize for canonical mapping supervision "
                        f"tier {tier.value}"
                    ),
                    target_resolution=(height, width),
                )
            )
        camera.score_provenance = provenance
        attach_score_provenance(camera.original_image, provenance)
    if native.depth is not None:
        camera._sensor_depth_cache = depth.contiguous()
    camera._resolution_depth_confidence_weight = discontinuity_weight
    camera.image_width = camera.width = width
    camera.image_height = camera.height = height
    camera.fx = fx_new
    camera.fy = fy_new
    camera.cx = cx_new
    camera.cy = cy_new
    camera.gsplat = None
    camera._canonical_resolution_tier = tier
    camera._canonical_resolution_binding_native_state = native
    camera._canonical_resolution_resampling_mode = resampling_mode
    camera._canonical_resolution_depth_downsample = depth_downsample


__all__ = [
    "materialize_resolution_tensors",
    "materialize_resolution_tier",
    "resolution_tier_geometry",
    "RESAMPLING_MODES",
    "DEPTH_DOWNSAMPLE_MODES",
]
