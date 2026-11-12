"""Gaussian objectives with explicit inputs and ordered numerical finalization.

``assemble_loss`` preserves B1/legacy RGB measurements. Canonical true-camera
work requests only optimization/evidence terms; unmeasured components are absent.
``prepare_loss`` allows one status readback per render group without coupling
the views' autograd graphs. Reuse ``nan_skip_count_holder`` across iterations
to retain the first-five-failures diagnostic limit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from eligsir.mapping.integration.depth_helpers import (
    _get_resident_sensor_depth,
    _get_sensor_depth,
    _get_surface_prior,
)
from eligsir.mapping.integration.geometry_helpers import knn_smoothness_reg
from eligsir.mapping.integration.loss_diagnostics import GuardedLossAccumulator
from eligsir.mapping.geometry.appearance_only import run_appearance_only_update
from eligsir.mapping.geometry.free_space_consensus import FreeSpaceConsensusIndex
from eligsir.mapping.geometry.material_risk import geometry_reliability_map, material_risk_mask
from eligsir.mapping.geometry.ray_classifier import classify_rays
from eligsir.utils.ingest_phase_profiler import PhaseSegments as _PhaseSegments
from eligsir.optimization.gaussian.contracts import RenderOutputs
from eligsir.optimization.gaussian.depth_loss_weighting import (
    DEPTH_LOSS_QUANTITIES,
    combine_depth_loss_weights,
    masked_depth_error,
    weighted_depth_error_mean,
)
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
from eligsir.optimization.gaussian.losses import (
    _streaming_young_isotropy_reg,
    _streaming_young_opacity_reg,
)
from eligsir.utils.depth_alpha_diagnostics import compute_alpha_bin_residuals, log_alpha_bin_residuals_to_tb
from eligsir.utils.surface_losses import compute_surface_anchor_losses

# l1_loss / ssim: same helpers train_streaming.py imports for the base
# photometric loss. Imported lazily-at-module-scope here (not inside the
# function) to match train_streaming.py's own top-level import style.
from eligsir.utils.loss_utils import l1_loss, ssim  # noqa: E402


def _is_prepared_camera(viewpoint_cam: Any) -> bool:
    """Check if a camera is from the prepared path.

    Prepared cameras bound via ready_camera_binder.bind_ready_camera() have
    a `_source_identity` attribute set to a SourceIdentity. Legacy cameras
    may have a `gsplat` attribute (set to a GsplatCameraTensors or None by
    prepare_camera_for_render()), so hasattr() is not reliable; we check
    the _source_identity marker instead.
    """
    return getattr(viewpoint_cam, "_source_identity", None) is not None


def _surface_confidence_enabled(args: Any) -> bool:
    """Check if surface-prior confidence loss is enabled.

    Shared predicate used by both preflight validation and loss assembly body.
    """
    return bool(getattr(args, "surface_confidence_depth_loss", False))


def _ray_semantics_inputs_enabled(args: Any) -> bool:
    """Check if ray-semantics inputs are enabled (or appearance-only which depends on RS).

    Shared predicate used by both preflight validation and loss assembly body.
    Ray-semantics and appearance-only both require the ray-semantics sensor depth.
    """
    _rs_enabled = bool(getattr(args, "streaming_ray_semantics_enabled", False))
    _ao_enabled = bool(getattr(args, "streaming_appearance_only_enabled", False))
    return _rs_enabled or _ao_enabled


def _free_space_enabled(args: Any, use_depth_loss: bool) -> bool:
    """Check if free-space loss is enabled (static part; runtime render_pkg keys are checked separately).

    Shared predicate used by both preflight validation and loss assembly body.
    Free-space requires: weight > 0 AND use_depth_loss=True. The render_pkg key checks
    ("alpha" in render_pkg, "rendered_depth" in render_pkg) are runtime-only and checked
    in the loss body; preflight cannot inspect render_pkg but accepts the static condition
    and may warn (fail-closed) on configs that won't actually use the auxiliary.
    """
    free_space_weight = float(getattr(args, "streaming_free_space_loss_weight", 0.0))
    return free_space_weight > 0 and use_depth_loss


def _legacy_to_device(tensor: torch.Tensor, viewpoint_cam: Any) -> torch.Tensor:
    """Transfer tensor to cuda only on legacy path; prepared path tensors are already device-resident.

    Issue #1686: on the prepared path, auxiliaries must be device-resident before
    loss assembly. This helper skips the transfer for prepared cameras and only
    performs H2D on legacy cameras for backward compatibility.

    Args:
        tensor: The tensor to transfer (or pass through).
        viewpoint_cam: The camera object (determines prepared vs legacy path).

    Returns:
        The tensor itself (prepared path) or transferred to cuda (legacy path).
    """
    if _is_prepared_camera(viewpoint_cam):
        return tensor
    else:
        return tensor.to(device="cuda", non_blocking=True)


def preflight_prepared_objective_auxiliaries(
    viewpoint_cam: Any, args: Any, device: torch.device | str, use_depth_loss: bool = False
) -> None:
    """Check enabled prepared auxiliaries without materializing or resizing them.

    Missing/stale caches with a source fail closed; explicitly absent inputs
    remain optional. Legacy cameras bypass this check. Free-space's dynamic
    alpha/rendered-depth requirements stay in the loss body.
    """
    if not _is_prepared_camera(viewpoint_cam):
        return
    device = torch.device(device)

    # One confidence check covers both depth and free-space consumers.
    if _surface_confidence_enabled(args):
        _surface_prior = _get_surface_prior(viewpoint_cam, args, resident_only=True)
        if _surface_prior is not None and _surface_prior.confidence is not None:
            if _surface_prior.confidence.device != device:
                raise ValueError(
                    f"prepared camera auxiliary 'surface_confidence' must be device-resident "
                    f"(expected {device}, got {_surface_prior.confidence.device})"
                )

    ray_inputs = _ray_semantics_inputs_enabled(args)
    if ray_inputs or _free_space_enabled(args, use_depth_loss):
        sensor_depth = _get_resident_sensor_depth(viewpoint_cam)
        if sensor_depth is not None and sensor_depth.device != device:
            name = "ray_semantics_sensor_depth" if ray_inputs else "free_space_sensor_depth"
            raise ValueError(
                f"prepared camera auxiliary '{name}' must be device-resident "
                f"(expected {device}, got {sensor_depth.device})"
            )


@dataclass
class LossAssemblyInputs:
    """Objective inputs, optional measurements, and per-run diagnostics."""

    # Core objects
    gaussians: Any
    viewpoint_cam: Any
    args: Any
    opt: Any
    pipe: Any
    render_outputs: RenderOutputs
    image: torch.Tensor
    iteration: int
    n_frames_ingested: int
    # Canonical behavior clock; artifact ``iteration`` remains available for
    # diagnostics and legacy offline callers.
    optimizer_update_id: int | None = None
    tb_writer: Any = None
    streaming_scene: Any = None
    seed_fade_monitor: Any = None
    # Wave 3, PR W3-08: needed for the (default-off) appearance-only
    # auxiliary render (eligsir/mapping/geometry/appearance_only.py).
    # Optional/default None so every pre-existing call site (tests,
    # engine.py before this PR) keeps working unchanged.
    background: Any = None

    # Pre-resolved config flags/values (mirrors the local variables the
    # inline block read; each is a plain `getattr(args, ..., default)` at
    # the original call site, hoisted here so the caller controls exactly
    # when they're resolved).
    sparse_active_set: bool = False
    rgb_loss_enabled: bool = True
    rgb_measurements: tuple[bool, bool] = (True, True)  # L1, SSIM; legacy default
    use_depth_loss: bool = False
    use_energy_mcmc: bool = False
    depth_loss_type: str = "l1"
    depth_loss_weight: float = 0.0
    scalar_log_interval: int = 10

    _anchor_loss_weight: float = 0.0
    _anchor_decay_steps: int = 500
    _bootstrap_anchor_weight: float = 0.0
    _bootstrap_anchor_decay: int = 2000
    _insertion_anchor_weight: float = 1.0
    _insertion_anchor_decay: int = 2000
    _lifecycle_enabled: bool = False
    _mature_anchor_xyz_w: float = 0.0
    _mature_anchor_scale_w: float = 0.0
    _mature_anchor_opacity_w: float = 0.0

    # Issue #1670 Phase A (`--streaming_objective_step_regularizers`): the
    # caller (true_batch_executor.py) has already computed this view's
    # contribution to the per-Gaussian regularizers (active_reg[+young]) as
    # part of a once-per-step batched evaluation across the whole group --
    # skip ONLY the `else:` branch of `_energy_loss_enabled_main` in the
    # `m_loss_regs_gaussian` segment below (the active_reg_core[_masked] +
    # _streaming_young_opacity_reg / plain active_reg_core additions).
    # Everything else in that segment (the `use_energy_mcmc` eff-count/
    # entropy block, young_isotropy) is untouched. Default False: this
    # flag's own effect on assemble_loss is unchanged/byte-identical when
    # off; it says nothing about true_batch_executor.py's separate
    # multi-group recombination, which is value-preserving but not
    # byte-identical regardless of this flag (see `_combine_group_
    # aggregates`'s docstring).
    skip_gaussian_regularizers: bool = False

    precomputed_pixel_terms: "PixelTermsView | None" = None

    # Mutable per-run counter, see module docstring re: `_nan_skip_count`.
    nan_skip_count_holder: list = field(default_factory=lambda: [0])


def required_rgb_terms(
    opt: Any, rgb_loss_enabled: bool, measurements: tuple[bool, bool],
) -> tuple[bool, bool]:
    """Union explicit measurements with nonzero optimization terms."""
    return (
        measurements[0] or (rgb_loss_enabled and opt.lambda_dssim != 1.0),
        measurements[1] or (rgb_loss_enabled and opt.lambda_dssim != 0.0),
    )


# ---------------------------------------------------------------------------
# Wave 3, PR W3-08 (plan section 13): tri-state ray semantics integration.
#
# ``_get_ray_consensus_index``/``_classify_view_rays`` are thin adapters
# between the streaming loop's per-iteration objects (viewpoint_cam, args,
# streaming_scene) and the layout-agnostic
# ``eligsir.mapping.geometry.*`` classifier package. Kept here (rather
# than inside the geometry package itself) because they're the only part of
# this integration that knows about this repo's specific ``Camera``
# (``fx``/``fy``/``cx``/``cy``/``world_view_transform``) and
# ``StreamingScene`` conventions.
# ---------------------------------------------------------------------------

_RS_WARNED_LEGACY_CARVE_COMBO = False
_DEPTH_LOSS_STARTUP_LOGGED = False
_DEPTH_LOSS_ED_ALPHA_GATE_WARNED = False
_SUPERVISION_TILE_PROBE_WARNED = False


def _get_ray_consensus_index(streaming_scene, args) -> FreeSpaceConsensusIndex:
    """Return the run-lifetime consensus index, creating it once and caching
    it on ``streaming_scene`` (mirrors how ``OccupancyIndex``/other stores
    are owned — see ``eligsir/mapping/stores/``)."""
    index = getattr(streaming_scene, "_ray_free_space_consensus", None)
    if index is None:
        index = FreeSpaceConsensusIndex(
            voxel_size_m=float(getattr(args, "streaming_ray_semantics_voxel_size_m", 0.05)),
            min_baseline_m=float(getattr(args, "streaming_ray_semantics_min_baseline_m", 0.03)),
        )
        try:
            streaming_scene._ray_free_space_consensus = index
        except Exception:
            pass  # streaming_scene may be a lightweight stand-in (e.g. in tests)
    return index


def _classify_view_rays(viewpoint_cam, sensor_depth_hw: torch.Tensor, args, streaming_scene, iteration: int):
    """Classify one view's rays via
    :func:`eligsir.mapping.geometry.ray_classifier.classify_rays`.

    Returns ``None`` when the camera doesn't expose the intrinsics/pose this
    needs (fails open rather than crashing training — matches this file's
    existing convention of feature-gated blocks degrading silently rather
    than raising).
    """
    for attr in ("fx", "fy", "cx", "cy", "world_view_transform"):
        # Camera.__init__ defaults fx/fy/cx/cy to None (e.g. COLMAP-style
        # sources that only set FoVx/FoVy) -- hasattr() alone is True for a
        # None-valued attribute, so this must also reject None explicitly or
        # float(None) below raises and defeats the "fails open" contract.
        if getattr(viewpoint_cam, attr, None) is None:
            return None
    w2c = viewpoint_cam.world_view_transform.transpose(0, 1)
    c2w = torch.inverse(w2c)
    consensus_index = _get_ray_consensus_index(streaming_scene, args)
    view_uid = int(getattr(viewpoint_cam, "uid", id(viewpoint_cam)))
    pose_confidence = float(getattr(viewpoint_cam, "pose_confidence", 1.0) or 1.0)
    result = classify_rays(
        sensor_depth_hw,
        c2w=c2w,
        fx=float(viewpoint_cam.fx), fy=float(viewpoint_cam.fy),
        cx=float(viewpoint_cam.cx), cy=float(viewpoint_cam.cy),
        consensus_index=consensus_index,
        min_views=int(getattr(args, "streaming_ray_semantics_min_views", 3)),
        # No getattr fallback: streaming_max_depth is an unconditional
        # StreamingParams default, always present on the real parsed args
        # (issue #368 -- old getattr(..., 5.0) fallback drifted from the real
        # default of 8.0; removed rather than re-stated).
        max_depth_m=float(args.streaming_max_depth),
        view_uid=view_uid,
        pose_confidence=pose_confidence,
        min_pose_confidence=float(getattr(args, "streaming_ray_semantics_min_pose_confidence", 0.0)),
    )
    return result


# Legacy/test helper kept for the exact work-clock unit surface. The canonical
# objective no longer uses it to decide whether depth math executes.
def _materialize_depth_valid_sample_count(valid: Any) -> int:
    return int(valid.count_nonzero().item())


def _accepted_depth_valid_sample_count(
    count: int, *, depth_loss_finite: bool
) -> int:
    return count if depth_loss_finite else 0


def _gt_image_validated(viewpoint_cam: Any, gt_image: torch.Tensor) -> torch.Tensor:
    """Return the cached, finite ground truth image used by the RGB loss."""
    if (
        getattr(viewpoint_cam, "_gt_finite_validated_id", None) != id(gt_image)
        or getattr(viewpoint_cam, "_gt_finite_validated_version", None)
        != int(gt_image._version)
    ):
        if not bool(torch.isfinite(gt_image).all()):
            view_id = getattr(viewpoint_cam, "image_name", getattr(viewpoint_cam, "uid", "?"))
            print(
                f"[gt-sanitize] non-finite ground-truth pixels in view={view_id} "
                "— sanitizing cached original_image in place (one-time).",
                flush=True,
            )
            gt_image = torch.nan_to_num(gt_image, nan=0.0, posinf=1.0, neginf=0.0)
            viewpoint_cam.original_image = gt_image.detach()
            from eligsir.utils.score_contract import declare_camera_score_provenance

            declare_camera_score_provenance(viewpoint_cam, context="loss_nan_sanitization")
        gt_image = viewpoint_cam.original_image
        viewpoint_cam._gt_finite_validated = True
        viewpoint_cam._gt_finite_validated_id = id(gt_image)
        viewpoint_cam._gt_finite_validated_version = int(gt_image._version)
    return gt_image


def _evwb_config(args: Any) -> tuple[float, float, float, str] | None:
    """Return the opt-in EV/WB configuration, validating its scientific gauge."""
    if not bool(getattr(args, "streaming_train_evwb_enabled", False)):
        return None
    reference = str(getattr(args, "streaming_train_evwb_reference_source", "") or "")
    margin = float(getattr(args, "streaming_train_evwb_display_margin", 0.00392156862745098))
    prior = float(getattr(args, "streaming_train_evwb_prior_weight", 0.0))
    max_ev = float(getattr(args, "streaming_train_evwb_max_ev", 2.0))
    max_wb = float(getattr(args, "streaming_train_evwb_max_log_wb", 1.0))
    if not reference:
        raise ValueError("streaming_train_evwb_enabled requires an exact reference source identity")
    if not all(math.isfinite(value) for value in (margin, prior, max_ev, max_wb)):
        raise ValueError("streaming train EV/WB bounds and prior must be finite")
    if not (0.0 < margin < 0.5) or prior < 0.0 or max_ev <= 0.0 or max_wb <= 0.0:
        raise ValueError("invalid streaming train EV/WB display margin, prior, or bounds")
    return margin, prior, max_ev, reference


def _evwb_state_for_camera(scene: Any, camera: Any, image: torch.Tensor, config: tuple[float, float, float, str]):
    """Borrow admission-owned training state, independently of pixel residency."""
    database = getattr(scene, "_observation_db", None)
    source = getattr(camera, "_source_identity", None)
    if database is None or source is None:
        return None
    frame_id = database.frame_id_for_source(source)
    record = None if frame_id is None else database.get_frame(frame_id)
    if record is None or record.source_identity == config[3]:
        return None
    state = record.photometric_evwb
    if state is None:
        return None  # Admission allocates state only for eligible training sources.
    if state.device != image.device:
        raise RuntimeError("train EV/WB state device disagrees with READY camera")
    return state


def _display_referred_evwb(
    image: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    *,
    display_margin: float,
    prior_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply EV + zero-sum two-coordinate log-WB only on safe display pixels.

    The renderer and cached target remain display-referred sRGB. This helper
    linearizes a finite unclipped render locally, applies the three identifiable
    parameters, and encodes it back before the existing RGB objective.
    """
    safe_image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    safe_image = safe_image.clamp(display_margin, 1.0 - display_margin)
    finite_state = torch.isfinite(state).all()
    max_component = math.log2(torch.finfo(state.dtype).max) / 3.0
    state_supported = finite_state & (state.abs().amax() < max_component)
    # Reject the whole vector before nonlinear work: an unused overflowing
    # channel can otherwise poison backward even when the output is masked.
    safe_state = torch.where(state_supported, state, torch.zeros_like(state))
    linear = torch.where(
        safe_image <= 0.04045,
        safe_image / 12.92,
        ((safe_image + 0.055) / 1.055).pow(2.4),
    )
    log_gains = torch.stack((
        safe_state[0] + safe_state[1],
        safe_state[0] + safe_state[2],
        safe_state[0] - safe_state[1] - safe_state[2],
    )).view(3, 1, 1)
    corrected_linear = linear * torch.exp2(log_gains)
    # Keep the unused high branch differentiable when the gain product underflows.
    corrected = torch.where(
        corrected_linear <= 0.0031308,
        12.92 * corrected_linear,
        1.055 * corrected_linear.clamp_min(0.0031308).pow(1.0 / 2.4) - 0.055,
    )
    trusted = (
        torch.isfinite(image).all(dim=0)
        & torch.isfinite(target).all(dim=0)
        & (image > display_margin).all(dim=0)
        & (image < 1.0 - display_margin).all(dim=0)
        & (target > display_margin).all(dim=0)
        & (target < 1.0 - display_margin).all(dim=0)
        & torch.isfinite(corrected).all(dim=0)
        & (corrected > display_margin).all(dim=0)
        & (corrected < 1.0 - display_margin).all(dim=0)
        & state_supported
    )
    # Unsafe samples keep the unmodified render so the primary non-finite
    # guard remains authoritative instead of silently masking bad renderer data.
    corrected = torch.where(trusted.unsqueeze(0), corrected, image)
    return corrected, prior_weight * trusted.any().to(dtype=image.dtype) * safe_state.square().sum()


def assemble_loss(inputs: LossAssemblyInputs) -> LossBreakdown:
    """Single-view objective API; grouped execution defers the same finalization."""
    accumulator, loss, finalize = prepare_loss(inputs)
    return finalize(*accumulator.resolve(loss, inputs.iteration))


def prepare_loss(inputs: LossAssemblyInputs):
    """Construct one view's terms; return its accumulator, base loss and finalizer.

    The executor owns only C such captures until that group's numerical gate.
    Diagnostic construction retains its cadence; accepted-term callbacks and
    NaN counters run in view order only when the finalizer is reached.
    """
    gaussians = inputs.gaussians
    viewpoint_cam = inputs.viewpoint_cam
    args = inputs.args
    opt = inputs.opt
    render_pkg = inputs.render_outputs.payload
    image = inputs.image
    iteration = inputs.iteration
    optimizer_update_id = (
        iteration
        if inputs.optimizer_update_id is None
        else int(inputs.optimizer_update_id)
    )
    n_frames_ingested = inputs.n_frames_ingested
    tb_writer = inputs.tb_writer
    streaming_scene = inputs.streaming_scene
    seed_fade_monitor = inputs.seed_fade_monitor
    background = inputs.background

    sparse_active_set = inputs.sparse_active_set
    use_depth_loss = inputs.use_depth_loss
    use_energy_mcmc = inputs.use_energy_mcmc
    depth_loss_type = inputs.depth_loss_type
    depth_loss_weight = inputs.depth_loss_weight
    scalar_log_interval = inputs.scalar_log_interval
    skip_gaussian_regularizers = inputs.skip_gaussian_regularizers
    precomputed_pixel_terms = inputs.precomputed_pixel_terms

    # Issue #1686: validate that prepared cameras have device-resident auxiliaries.
    # Legacy cameras skip this check; the later .to() sites handle H2D for them.
    device = image.device
    preflight_prepared_objective_auxiliaries(viewpoint_cam, args, device, use_depth_loss)

    _anchor_loss_weight = inputs._anchor_loss_weight
    _anchor_decay_steps = inputs._anchor_decay_steps
    # NOTE: `inputs._bootstrap_anchor_*` / `inputs._insertion_anchor_*` are
    # deliberately NOT unpacked here any more -- their only consumer in this
    # module was the duplicate TB diagnostics block removed for issue #905 (see
    # the note further down). The dataclass fields stay on LossAssemblyInputs so
    # existing callers/tests keep constructing it unchanged.
    _lifecycle_enabled = inputs._lifecycle_enabled
    _mature_anchor_xyz_w = inputs._mature_anchor_xyz_w
    _mature_anchor_scale_w = inputs._mature_anchor_scale_w
    _mature_anchor_opacity_w = inputs._mature_anchor_opacity_w

    # PR #39 review fix (P1): kept as detached GPU tensors, NOT floats, here.
    # `float(tensor)` forces a GPU->CPU host sync; doing that unconditionally
    # for every component on every step (as opposed to only when a caller
    # actually needs a Python float, e.g. TB logging/JSON export) serializes
    # the forward->backward pipeline -- a new host sync the original inline
    # train_streaming.py code never had at this point. See
    # `LossBreakdown.materialized_components()`.
    components: dict[str, Any] = {}
    # A true-camera group shares one readback; B1 resolves the same packet alone.
    _guard_acc = GuardedLossAccumulator()

    # Sub-segments of m_loss_assembly for the phase/torch profilers (launch
    # attribution per term family); sequential, closed before the gate.
    _seg = _PhaseSegments()
    _seg.begin("m_loss_rgb")
    # ---- Loss ---------------------------------------------------------
    need_l1, need_ssim = required_rgb_terms(opt, inputs.rgb_loss_enabled, inputs.rgb_measurements)
    photometric_evwb_state = None
    if precomputed_pixel_terms is None:
        evwb = _evwb_config(args)
        # The opt-in path never rewrites cached display targets. Its local
        # support gate leaves unsafe values visible to the existing loss guard.
        gt_image = (
            viewpoint_cam.original_image
            if evwb is not None
            else _gt_image_validated(viewpoint_cam, viewpoint_cam.original_image)
        )
        if evwb is not None and inputs.rgb_loss_enabled:
            photometric_evwb_state = _evwb_state_for_camera(
                inputs.streaming_scene, inputs.viewpoint_cam, image, evwb
            )
            if photometric_evwb_state is not None:
                image, _evwb_prior = _display_referred_evwb(
                    image,
                    gt_image,
                    photometric_evwb_state,
                    display_margin=evwb[0],
                    prior_weight=evwb[1],
                )
        Ll1 = l1_loss(image, gt_image) if need_l1 else None
        _ssim = ssim(image, gt_image) if need_ssim else None
    else:
        gt_image = viewpoint_cam.original_image
        Ll1 = precomputed_pixel_terms.Ll1
        _ssim = precomputed_pixel_terms.ssim
        photometric_evwb_state = (
            precomputed_pixel_terms.photometric_evwb_state if inputs.rgb_loss_enabled else None
        )
        _evwb_prior = precomputed_pixel_terms.photometric_evwb_prior
        if (need_l1 and Ll1 is None) or (need_ssim and _ssim is None):
            raise ValueError("precomputed terms omit a requested RGB measurement")
    if inputs.rgb_loss_enabled:
        loss = (1.0 - opt.lambda_dssim) * Ll1 if opt.lambda_dssim != 1.0 else 0.0
        if opt.lambda_dssim != 0.0:
            loss = loss + opt.lambda_dssim * (1.0 - _ssim)
    else:
        loss = image.sum() * 0.0
    if photometric_evwb_state is not None:
        loss = loss + _evwb_prior
    for name, term in (("Ll1", Ll1), ("ssim", _ssim)):
        if term is not None:
            components[name] = term.detach()

    # ---- Supervision-tile evidence probe (#310 items 4-5 / #611 Phase 0) ----
    # DIAGNOSTIC ONLY, default OFF (`--supervision_tile_probe_grid 0`). Nothing
    # reads this back into training: it does not touch `loss`, no scheduler or
    # allocator may consume it, and with the flag at its default this branch is
    # a single integer comparison.
    #
    # Cost discipline (plan doc section 6.2): no extra rasterization, and the
    # ONE host transfer is pinned to an iteration that already synchronises for
    # scalar logging. RGB only for now -- the depth residual/valid tensors live
    # inside the nested depth-loss branch further down, and threading them out
    # is a refactor rather than a probe (P1, alongside the #314 masks).
    _tile_probe_grid = int(getattr(args, "supervision_tile_probe_grid", 0) or 0)
    if (
        _tile_probe_grid > 0
        and tb_writer is not None
        and iteration % scalar_log_interval == 0
    ):
        try:
            from eligsir.mapping.supervision.tile_lattice import (
                SupervisionTileLattice,
                collect_tile_evidence,
            )

            _tile_lattice = SupervisionTileLattice.square(
                _tile_probe_grid,
                width=int(image.shape[-1]),
                height=int(image.shape[-2]),
            )
            _tile_evidence = collect_tile_evidence(
                image=image, gt_image=gt_image, lattice=_tile_lattice
            )
            for _tk, _tv in _tile_evidence.as_scalar_dict().items():
                tb_writer.add_scalar(_tk, _tv, iteration)
        except Exception as _tile_exc:  # pragma: no cover - diagnostics must never fail a run
            global _SUPERVISION_TILE_PROBE_WARNED
            if not _SUPERVISION_TILE_PROBE_WARNED:
                print(
                    f"[supervision_tile] probe disabled after error: {_tile_exc!r}",
                    flush=True,
                )
                _SUPERVISION_TILE_PROBE_WARNED = True

    _seg.begin("m_loss_regs_gaussian")
    # Phase 5: differentiable energy loss flag (replaces reg call sites when enabled)
    _energy_loss_enabled_main = getattr(args, "energy_loss_enabled", False)
    _p5_result = None
    if _energy_loss_enabled_main:
        from eligsir.utils.differentiable_energy_loss import compute_differentiable_loss
        if sparse_active_set:
            _p5_visible = render_pkg["visibility_filter"].detach()
            _p5_provisional = gaussians.provisional.detach()
            _p5_active = _p5_visible | _p5_provisional
        else:
            _p5_active = None
        _p5_result = compute_differentiable_loss(
            gaussians=gaussians,
            iteration=optimizer_update_id,
            cap_max=args.cap_max,
            active_mask=_p5_active,
            include_effective_count=use_energy_mcmc,
            w_opacity_reg=args.opacity_reg,
            w_scale_reg=args.scale_reg,
            w_opacity_entropy=args.lambda_opacity_entropy,
            w_effective_count=args.lambda_eff_count,
            w_anisotropy_reg=float(args.anisotropy_reg),
            anisotropy_soft_limit=float(getattr(args, "anisotropy_soft_limit", 10.0)),
            w_needle_reg=float(getattr(args, "needle_reg_weight", 0.0)),
            needle_threshold=float(getattr(args, "needle_reg_threshold", 0.3)),
            scale_reg_mode=str(getattr(args, "scale_reg_mode", "l1")),
            max_iterations=opt.iterations,
        )
        loss = loss + _p5_result.total
        components["energy_total"] = _p5_result.total.detach()
        if tb_writer:
            for _k, _v in _p5_result.to_tensorboard_dict().items():
                tb_writer.add_scalar(_k, _v, iteration)
    else:
        # Regularisation: local (visible) or global depending on active-set mode
        from eligsir.utils.compiled_kernels import active_reg_core, active_reg_core_masked
        _w_aniso = float(args.anisotropy_reg)
        _young_exempt_iters = int(args.streaming_young_reg_exempt_iters)
        _maturity_ramp = bool(args.streaming_maturity_reg_ramp)
        _recency_reg_on = float(getattr(args, "streaming_recency_reg_horizon", 0)) > 0
        _opacity_reg_custom = _young_exempt_iters > 0 or _maturity_ramp or _recency_reg_on
        if skip_gaussian_regularizers:
            # Issue #1670 Phase A: the caller already accounted for this
            # view's active_reg[+young] contribution in a once-per-step
            # batched evaluation (true_batch_executor.py + assemble_step_
            # regularizers) -- do not add it again here.
            pass
        elif sparse_active_set:
            # Step 10: Refine active set to include provisional/new points
            _visible = render_pkg["visibility_filter"].detach()
            _provisional = gaussians.provisional.detach()
            _active = _visible | _provisional

            # Issue #1670: full-population fixed-shape masked math instead of
            # a caller-side `tensor[_active]` dynamic boolean compaction of
            # opacity/scaling — see active_reg_core_masked's docstring for
            # the exact parity/gradient argument.
            if _opacity_reg_custom:
                _guard_acc.add(active_reg_core_masked(
                    gaussians.get_opacity,
                    gaussians.get_scaling,
                    _active,
                    w_opacity=0.0,
                    w_scale=args.scale_reg,
                    w_aniso=_w_aniso,
                ) + _streaming_young_opacity_reg(
                    gaussians, args, optimizer_update_id, _active,
                    _young_exempt_iters, _maturity_ramp,
                ), "active_reg+young")
            else:
                _guard_acc.add(active_reg_core_masked(
                    gaussians.get_opacity,
                    gaussians.get_scaling,
                    _active,
                    w_opacity=args.opacity_reg,
                    w_scale=args.scale_reg,
                    w_aniso=_w_aniso,
                ), "active_reg")
        else:
            if _opacity_reg_custom:
                _guard_acc.add(active_reg_core(
                    gaussians.get_opacity,
                    gaussians.get_scaling,
                    w_opacity=0.0,
                    w_scale=args.scale_reg,
                    w_aniso=_w_aniso,
                ) + _streaming_young_opacity_reg(
                    gaussians, args, optimizer_update_id, None,
                    _young_exempt_iters, _maturity_ramp,
                ), "active_reg+young")
            else:
                _guard_acc.add(active_reg_core(
                    gaussians.get_opacity,
                    gaussians.get_scaling,
                    w_opacity=args.opacity_reg,
                    w_scale=args.scale_reg,
                    w_aniso=_w_aniso,
                ), "active_reg")

        # Energy-guided losses (full path, same as offline when enabled)
        if use_energy_mcmc:
            from eligsir.utils.energy_mcmc import (
                compute_effective_count_loss,
                compute_opacity_entropy_loss,
                compute_gaussian_utility,
            )
            # issue #282: observe-only handling + applied-scale recording now
            # live inside the monitor (see effective_suppression_scale). The
            # old ternary read args.streaming_seed_fade_disabled here, which
            # left the monitor unaware that its output was being discarded --
            # hence a report that claimed controller_engaged on runs where no
            # suppression was ever applied.
            _fade_scale = (
                seed_fade_monitor.effective_suppression_scale()
                if seed_fade_monitor is not None else 1.0
            )
            _energy_interval = getattr(args, "sparse_energy_global_interval", 500)
            _run_energy = (
                not sparse_active_set
                or _energy_interval <= 0
                or optimizer_update_id % _energy_interval == 0
            )
            if _run_energy:
                _op_in = gaussians._opacity
                if getattr(args, "streaming_count_loss_local", False):
                    _vis = render_pkg["visibility_filter"].detach()
                    _prov = gaussians.provisional.detach() if hasattr(gaussians, "provisional") else torch.zeros_like(_vis)
                    _active_op = _vis | _prov
                    if _active_op.shape[0] == _op_in.shape[0]:
                        _op_in = torch.where(
                            _active_op.unsqueeze(-1), _op_in, _op_in.detach()
                        )
                _anneal_clock = getattr(opt, "_anneal_clock", None)
                _eff_horizon = _anneal_clock.horizon if _anneal_clock is not None else opt.iterations
                L_eff, N_eff, N_target = compute_effective_count_loss(
                    _op_in,
                    iteration=optimizer_update_id,
                    cap_max=args.cap_max,
                    max_iterations=_eff_horizon,
                    target_splat_end=args.mcmc_target_splat_end,
                    q_start=getattr(args, "mcmc_target_q_start", 0.05),
                    q_end=getattr(args, "mcmc_target_q_end", 0.85),
                    tau_N=getattr(args, "mcmc_target_tau", 0.45),
                    target_capacity=int(getattr(args, "streaming_target_capacity", 0)),
                )
                _guard_acc.add(_fade_scale * args.lambda_eff_count * L_eff, "eff_count")
                _guard_acc.add(_fade_scale * args.lambda_opacity_entropy * compute_opacity_entropy_loss(_op_in), "opacity_entropy")

    # ---- Decaying isotropy reg on immature (provisional+young) Gaussians ----
    if float(getattr(args, "streaming_young_isotropy_weight", 0.0)) > 0.0:
        _iso_loss = _streaming_young_isotropy_reg(gaussians, args, n_frames_ingested)
        if _iso_loss is not None:
            _guard_acc.add(_iso_loss, "young_isotropy")

    _seg.begin("m_loss_depth")
    # ---- Depth supervision loss ----------------------------------------
    _depth_loss_val = None
    _depth_valid_sample_count = 0
    global _DEPTH_LOSS_STARTUP_LOGGED, _DEPTH_LOSS_ED_ALPHA_GATE_WARNED
    if (
        use_depth_loss
        and "rendered_depth" in render_pkg
        and precomputed_pixel_terms is not None
        and precomputed_pixel_terms.depth is not None
    ):
        depth_quantity = str(getattr(args, "streaming_depth_loss_quantity", "d")).lower()
        alpha_gate = float(getattr(args, "streaming_depth_loss_alpha_gate", 0.0))
        if not _DEPTH_LOSS_STARTUP_LOGGED:
            print(f"[DepthLoss] quantity={depth_quantity} alpha_gate={alpha_gate}", flush=True)
            _DEPTH_LOSS_STARTUP_LOGGED = True
        if depth_quantity == "ed" and alpha_gate == 0.0 and not _DEPTH_LOSS_ED_ALPHA_GATE_WARNED:
            print(
                "[DepthLoss] WARNING: quantity='ed' with alpha_gate=0.0 is "
                "numerically unstable at low alpha -- consider setting "
                "--streaming_depth_loss_alpha_gate",
                flush=True,
            )
            _DEPTH_LOSS_ED_ALPHA_GATE_WARNED = True
        L_depth = precomputed_pixel_terms.depth
        _guard_acc.depth_valid_sample_count_device = precomputed_pixel_terms.depth_valid_count

        def _on_precomputed_depth_resolved(is_finite: bool, _L_depth=L_depth) -> None:
            nonlocal _depth_loss_val
            resolved = _L_depth if is_finite else torch.zeros((), device=_L_depth.device)
            _depth_loss_val = resolved.detach()
            components["depth"] = resolved.detach()
            if tb_writer is not None and iteration % scalar_log_interval == 0:
                tb_writer.add_scalar(
                    "streaming/depth_loss_raw_magnitude",
                    float(resolved.detach().item()),
                    iteration,
                )

        _guard_acc.add(
            depth_loss_weight * L_depth,
            "depth_loss",
            check_term=L_depth,
            on_resolved=_on_precomputed_depth_resolved,
            gates_depth_valid_sample_count=True,
        )
    elif use_depth_loss and "rendered_depth" in render_pkg:
        _surface_prior = _get_surface_prior(viewpoint_cam, args) if getattr(args, "surface_confidence_depth_loss", False) else None
        sensor_d_cpu = _surface_prior.depth if _surface_prior is not None else _get_sensor_depth(
            viewpoint_cam, image.shape[1], image.shape[2]
        )
        if sensor_d_cpu is not None:
            # Issue #1686: prepared path forbids H2D transfer here; tensors must already
            # be device-resident (validated at preflight above). Legacy path still
            # performs the transfer for backward compat.
            sensor_d = _legacy_to_device(sensor_d_cpu, viewpoint_cam)
            prior_conf = None
            if _surface_prior is not None:
                prior_conf = _legacy_to_device(_surface_prior.confidence, viewpoint_cam)
            _depth_quantity = str(getattr(args, "streaming_depth_loss_quantity", "d")).lower()
            _alpha_gate = float(getattr(args, "streaming_depth_loss_alpha_gate", 0.0))
            if _depth_quantity not in DEPTH_LOSS_QUANTITIES:
                raise ValueError(
                    f"Unrecognized --streaming_depth_loss_quantity={_depth_quantity!r}; "
                    f"expected one of {DEPTH_LOSS_QUANTITIES!r}."
                )
            if not math.isfinite(_alpha_gate) or not (0.0 <= _alpha_gate < 1.0):
                raise ValueError(
                    f"Invalid --streaming_depth_loss_alpha_gate={_alpha_gate!r}; "
                    f"expected a finite value in [0.0, 1.0)."
                )
            if not _DEPTH_LOSS_STARTUP_LOGGED:
                print(f"[DepthLoss] quantity={_depth_quantity} alpha_gate={_alpha_gate}", flush=True)
                _DEPTH_LOSS_STARTUP_LOGGED = True
            if _depth_quantity == "ed" and _alpha_gate == 0.0 and not _DEPTH_LOSS_ED_ALPHA_GATE_WARNED:
                print(
                    "[DepthLoss] WARNING: quantity='ed' with alpha_gate=0.0 is "
                    "numerically unstable at low alpha -- consider setting "
                    "--streaming_depth_loss_alpha_gate",
                    flush=True,
                )
                _DEPTH_LOSS_ED_ALPHA_GATE_WARNED = True
            _depth_quantity_uses_ed = _depth_quantity in ("ed", "ed_alpha_weighted")
            if _depth_quantity_uses_ed:
                if "rendered_depth_ed" not in render_pkg:
                    raise RuntimeError(
                        f"streaming_depth_loss_quantity={_depth_quantity!r} was requested "
                        "but render_pkg has no 'rendered_depth_ed' key -- the renderer did "
                        "not produce it (check render_depth=True was passed). Refusing to "
                        "silently fall back to D, since that would silently corrupt an "
                        "experiment comparison."
                    )
                rend_d = render_pkg["rendered_depth_ed"]  # [1, H, W], alpha-normalised
            else:
                rend_d = render_pkg["rendered_depth"]  # [1, H, W]
            valid = (sensor_d > 0) & (rend_d.detach() > 0)
            # render_pkg["alpha"] is [H, W, 1], not [1, H, W] -- permute
            # before comparing against/broadcasting with `valid`/`rend_d`
            # (see gsplat_backend.py's NOTE on this same pitfall; a bare
            # [1,H,W] & [H,W,1] broadcast silently misaligns whenever H == W
            # and errors otherwise). Fetched once and reused for both the
            # optional hard gate below and issue #1833's per-pixel alpha
            # weighting.
            _alpha_map = None
            if _alpha_gate > 0.0 or _depth_quantity == "ed_alpha_weighted":
                _alpha_map = render_pkg.get("alpha")
                if _alpha_map is None:
                    raise RuntimeError(
                        (
                            "streaming_depth_loss_alpha_gate > 0.0"
                            if _alpha_gate > 0.0
                            else "streaming_depth_loss_quantity='ed_alpha_weighted'"
                        )
                        + " was requested but render_pkg has no 'alpha' key -- refusing to "
                        "silently apply no gate/weight at all, since that would silently "
                        "corrupt an experiment comparison (same failure mode as the "
                        "ed-without-rendered_depth_ed guard above)."
                    )
                _alpha_map = _alpha_map.detach()
                if _alpha_map.dim() == 3 and _alpha_map.shape[-1] == 1:
                    _alpha_map = _alpha_map.permute(2, 0, 1)
                if _alpha_gate > 0.0:
                    valid = valid & (_alpha_map > _alpha_gate)
            # Exact accepted count shares the term's numerical-status packet (#1591).
            _guard_acc.depth_valid_sample_count_device = valid.count_nonzero()
            depth_err = masked_depth_error(
                rend_d,
                sensor_d,
                valid,
                loss_type=depth_loss_type,
            )
            # Issue #1833: weight each valid pixel's residual by its rendered
            # alpha (already detached above), so low-coverage pixels
            # contribute proportionally less instead of either being hard-
            # gated out or supervised at full weight -- and so the loss
            # cannot be reduced merely by inflating opacity, since the
            # weight itself carries no gradient. Combined multiplicatively
            # with the existing confidence/discontinuity weights below and
            # normalised by weighted_depth_error_mean's weight-sum
            # denominator (not the valid-pixel count).
            _alpha_weight = _alpha_map if _depth_quantity == "ed_alpha_weighted" else None
            _disc_weight = getattr(viewpoint_cam, "_resolution_depth_confidence_weight", None)
            if _disc_weight is not None:
                # Issue #1686 site 5: _resolution_depth_confidence_weight is prepared on device
                # in the prepared path (eligsir/runtime/ready_camera_binder.py:256 assigns
                # prepared.depth_confidence.data, which is device-resident). The .to(device=...)
                # call targets rend_d.device (not hardcoded "cuda"), making it a device→device
                # no-op in the prepared path. Legacy path creates weight from CPU-resident
                # native_depth_hw (eligsir/mapping/resolution_materialization.py:463),
                # so the transfer is needed for backward compat. No preflight check needed.
                _disc_weight = _disc_weight.to(device=rend_d.device, non_blocking=True)
                if _disc_weight.dim() == 2:
                    _disc_weight = _disc_weight.unsqueeze(0)
            combined_weights = combine_depth_loss_weights(prior_conf, _disc_weight, _alpha_weight)
            L_depth = weighted_depth_error_mean(depth_err, combined_weights, valid)

            # Replay diagnostics only after this view's finite decision. Scalar
            # logging keeps its existing cadence, outside the normal status gate.
            def _on_depth_resolved(is_finite: bool, _L_depth=L_depth) -> None:
                nonlocal _depth_loss_val
                _resolved_depth = _L_depth if is_finite else torch.zeros((), device=_L_depth.device)
                # Keep diagnostics device-resident on the normal path.  The
                # non-finite error reporter above may materialize it, but a
                # successful optimization step must not synchronize here.
                _depth_loss_val = _resolved_depth.detach()
                components["depth"] = _resolved_depth.detach()
                if tb_writer is not None and iteration % scalar_log_interval == 0:
                    tb_writer.add_scalar(
                        "streaming/depth_loss_raw_magnitude",
                        float(_resolved_depth.detach().item()),
                        iteration,
                    )
                    if bool(getattr(args, "streaming_depth_alpha_bin_diagnostic", False)):
                        # Training-time D/ED-vs-alpha instrumentation (2026-08-11, see
                        # docs/architecture/2026-08-10-debt-controlled-sparse-assimilation-plan.md
                        # section 4.5 step 0). The frozen-checkpoint diagnostic found almost no
                        # low-alpha population on final test-holdout renders; this answers whether
                        # that's also true on ACTUAL training cameras during active map expansion,
                        # which is the population the depth loss is actually computed against.
                        # rendered_depth_ed is guaranteed present here regardless of
                        # streaming_depth_loss_quantity -- render_depth=use_depth_loss (the
                        # renderer call site) is exactly the condition gating this whole block.
                        _diag_alpha_map = render_pkg.get("alpha")
                        if _diag_alpha_map is not None:
                            _diag_alpha_map = _diag_alpha_map.detach()
                            if _diag_alpha_map.dim() == 3 and _diag_alpha_map.shape[-1] == 1:
                                _diag_alpha_map = _diag_alpha_map.permute(2, 0, 1)
                            _diag_stats = compute_alpha_bin_residuals(
                                render_pkg["rendered_depth"],
                                render_pkg["rendered_depth_ed"],
                                sensor_d,
                                _diag_alpha_map,
                            )
                            log_alpha_bin_residuals_to_tb(tb_writer, _diag_stats, iteration)

            _guard_acc.add(
                depth_loss_weight * L_depth,
                "depth_loss",
                check_term=L_depth,
                on_resolved=_on_depth_resolved,
                gates_depth_valid_sample_count=True,
            )

    _seg.begin("m_loss_anchor")
    # ---- H9: Anchor loss for young provisional splats ------------------
    if _anchor_loss_weight > 0 and hasattr(gaussians, "anchor_xyz"):
        with torch.no_grad():
            age_iters = (
                optimizer_update_id - gaussians.anchor_iter.long()
            ).clamp(min=0)
            young_mask = (age_iters < _anchor_decay_steps) & gaussians.provisional
        if young_mask.any():
            decay = (1.0 - age_iters[young_mask].float() / _anchor_decay_steps).clamp(0.0, 1.0)
            dxyz = gaussians.get_xyz[young_mask] - gaussians.anchor_xyz[young_mask]
            L_anchor = (decay.unsqueeze(1) * dxyz.pow(2)).sum(dim=1).mean()
            # Issue #1670 Part B: this is an UNCONDITIONAL addition (never
            # finite-gated) that sits between two batches of guarded terms
            # (active_reg/eff_count/opacity_entropy/young_isotropy/depth
            # before it, soft_anchor/free_space/appearance_only/surface/
            # knn_reg after it) -- queue it through the accumulator, at
            # this exact position, rather than adding directly to `loss`,
            # so `resolve()`'s single replay reproduces the ORIGINAL
            # interleaved summation order bit-for-bit instead of just
            # summing the same terms in a different (still mathematically
            # equivalent, but not bit-identical) order.
            _guard_acc.add_unconditional(_anchor_loss_weight * L_anchor)
            if tb_writer is not None and iteration % scalar_log_interval == 0:
                components["anchor"] = L_anchor.detach()

    # Anchor/bubble TB diagnostics deliberately do NOT live here — see issue #905.
    #
    # A copy of that block (streaming/bootstrap_anchor/*, streaming/insertion_anchor/*,
    # streaming/bubble/*) was carried into this module by the W3-02 extraction
    # (10e7f8d5, 2026-08-07) while the original stayed in train_streaming.py. Once
    # PR #884 (bec78819) hoisted the train_streaming.py copy out of the inline `else:`
    # arm so BOTH optimizer arms reach it, engine-path runs executed both copies and
    # wrote all seven tags twice per sampled iteration under one step index. The two
    # writes are taken at different points in the iteration, so for the recomputed
    # `drift_mean_m` they disagreed on 39 of 40 steps (measured on i879engtel_a_engine)
    # — the stored series was an interleaving of a mid-loss and a post-loss reading of
    # the same quantity.
    #
    # `assemble_loss` is called only from engine.py (and it calls it more than once per
    # step for extra views), whereas train_streaming.py's hoisted block runs exactly
    # once per sampled iteration on both arms — so the train_streaming.py copy is the
    # correct single writer and this one is removed. These are TB-only diagnostics;
    # neither writer ever fed a loss, so no quality number changes.
    #
    # Guarded by tests/test_issue905_single_telemetry_writer.py.

    # ---- Mature-Gaussian soft-anchor anti-fade losses (Component C) -----
    if _lifecycle_enabled and _mature_anchor_xyz_w + _mature_anchor_scale_w + _mature_anchor_opacity_w > 0 and hasattr(gaussians, "lifecycle_state"):
        with torch.no_grad():
            mature_mask = (gaussians.lifecycle_state >= 2)  # MATURE or FROZEN
        if mature_mask.any():
            L_ma = torch.tensor(0.0, device="cuda")
            if _mature_anchor_xyz_w > 0 and hasattr(gaussians, "anchor_xyz"):
                dxyz = gaussians.get_xyz[mature_mask] - gaussians.anchor_xyz[mature_mask]
                L_ma = L_ma + _mature_anchor_xyz_w * dxyz.pow(2).sum(dim=1).mean()
            if _mature_anchor_scale_w > 0 and hasattr(gaussians, "anchor_scale_log"):
                d_scale = gaussians.raw_scales[mature_mask] - gaussians.anchor_scale_log[mature_mask]
                L_ma = L_ma + _mature_anchor_scale_w * d_scale.pow(2).mean()
            if _mature_anchor_opacity_w > 0 and hasattr(gaussians, "anchor_opacity_logit"):
                d_op = gaussians.raw_opacities[mature_mask] - gaussians.anchor_opacity_logit[mature_mask]
                L_ma = L_ma + _mature_anchor_opacity_w * d_op.pow(2).mean()
            # Issue #1670 Part B: same reasoning as the H9 anchor addition
            # above -- unconditional, interleaved between guarded batches,
            # queued at this exact position so resolve()'s replay preserves
            # the original bit-exact summation order.
            _guard_acc.add_unconditional(L_ma)
            components["mature_anchor"] = L_ma.detach()

    # ---- 2.4: distance-weighted soft anchor (bubble-as-local-bounds) ----
    _sa_lam = float(getattr(args, "streaming_soft_anchor_weight", 0.0))
    if (
        _sa_lam > 0
        and _lifecycle_enabled
        and hasattr(gaussians, "lifecycle_state")
        and hasattr(gaussians, "anchor_xyz")
        and getattr(args, "_bubble_center", None) is not None
    ):
        _sa_R = float(getattr(args, "streaming_bubble_radius", 5.0))
        with torch.no_grad():
            _sa_anchored = gaussians.lifecycle_state >= 2
        if _sa_anchored.any() and _sa_R > 0:
            _sa_center = args._bubble_center.to(gaussians.get_xyz.device)
            _sa_xyz = gaussians.get_xyz[_sa_anchored]
            with torch.no_grad():
                _sa_d = torch.linalg.norm(_sa_xyz.detach() - _sa_center, dim=1)
                _sa_w = torch.sigmoid((_sa_d - _sa_R) / (0.2 * _sa_R)) * (_sa_d > _sa_R)
            _sa_terms = (
                (_sa_xyz - gaussians.anchor_xyz[_sa_anchored]).pow(2).sum(dim=1) / (_sa_R * _sa_R)
            )
            if hasattr(gaussians, "anchor_scale_log"):
                _sa_terms = _sa_terms + (
                    gaussians.raw_scales[_sa_anchored] - gaussians.anchor_scale_log[_sa_anchored]
                ).pow(2).sum(dim=1)
            if hasattr(gaussians, "anchor_opacity_logit"):
                _sa_terms = _sa_terms + (
                    gaussians.raw_opacities[_sa_anchored] - gaussians.anchor_opacity_logit[_sa_anchored]
                ).pow(2).squeeze(-1)
            _sa_eff_lam = _sa_lam
            _sa_idle_steps = int(getattr(args, "streaming_consolidation_idle_steps", 500))
            if getattr(streaming_scene, "_steps_since_new_frame", 0) > _sa_idle_steps:
                _sa_eff_lam = _sa_lam * float(getattr(args, "streaming_soft_anchor_idle_mult", 0.3))
            L_sa = _sa_eff_lam * (_sa_w * _sa_terms).mean()

            def _on_soft_anchor_resolved(is_finite: bool, _L_sa=L_sa) -> None:
                if is_finite:
                    components["soft_anchor"] = _L_sa.detach()

            _guard_acc.add(L_sa, name=None, on_resolved=_on_soft_anchor_resolved)

    _seg.begin("m_loss_freespace")
    # ---- Tri-state ray semantics classification (Wave 3, PR W3-08) ------
    # Computed once per view when either the certified-free-space carving
    # replacement or the appearance-only auxiliary path is enabled (both
    # default OFF — see arguments/__init__.py), shared by both consumers
    # below so the view is only classified once. `_get_sensor_depth` caches
    # its decode on the camera object, so this is not a second I/O cost even
    # though the (also-gated) legacy free-space block below fetches sensor
    # depth again independently.
    _rs_enabled = bool(getattr(args, "streaming_ray_semantics_enabled", False))
    _ao_enabled = bool(getattr(args, "streaming_appearance_only_enabled", False))
    _rs_classification = None
    _rs_sensor_d_hw = None
    if _rs_enabled or _ao_enabled:
        _rs_sensor_d = _get_sensor_depth(viewpoint_cam, image.shape[1], image.shape[2])
        if _rs_sensor_d is not None:
            # Issue #1686: prepared path forbids H2D transfer here; tensors must already
            # be device-resident (validated at preflight above). Legacy path still
            # performs the transfer for backward compat.
            _rs_sensor_d = _legacy_to_device(_rs_sensor_d, viewpoint_cam)
            _rs_sensor_d_hw = _rs_sensor_d.squeeze(0)
            _rs_classification = _classify_view_rays(viewpoint_cam, _rs_sensor_d_hw, args, streaming_scene, iteration)
            if _rs_classification is not None and tb_writer is not None and iteration % scalar_log_interval == 0:
                for _rs_k, _rs_v in _rs_classification.counts.as_dict().items():
                    tb_writer.add_scalar(f"streaming/ray_semantics/{_rs_k}", _rs_v, iteration)

    # ---- Free-space / floater loss -------------------------------------
    _free_loss_val = None
    free_space_weight = getattr(args, "streaming_free_space_loss_weight", 0.0)
    if free_space_weight > 0 and use_depth_loss and "alpha" in render_pkg and "rendered_depth" in render_pkg:
        _surface_prior_fs = _get_surface_prior(viewpoint_cam, args) if getattr(args, "surface_confidence_depth_loss", False) else None
        sensor_d_fs = _surface_prior_fs.depth if _surface_prior_fs is not None else _get_sensor_depth(viewpoint_cam, image.shape[1], image.shape[2])
        if sensor_d_fs is not None:
            # Issue #1686: prepared path forbids H2D transfer here; tensors must already
            # be device-resident (validated at preflight above). Legacy path still
            # performs the transfer for backward compat.
            sensor_d_fs = _legacy_to_device(sensor_d_fs, viewpoint_cam)
            rend_d_fs = render_pkg["rendered_depth"]         # [1, H, W]
            rend_a_raw = render_pkg["alpha"]
            if rend_a_raw.dim() == 3 and rend_a_raw.shape[-1] == 1:
                rend_a_fs = rend_a_raw.permute(2, 0, 1)     # [1, H, W]
            else:
                rend_a_fs = rend_a_raw.unsqueeze(0) if rend_a_raw.dim() == 2 else rend_a_raw
            floater = (
                (sensor_d_fs > 0)
                & (rend_d_fs.detach() > 0)
                & (rend_a_fs.detach() > 0.2)
                & (rend_d_fs.detach() < sensor_d_fs - 0.05)
            )
            if _rs_enabled and _rs_classification is not None:
                # Deliverable 3: current carving remains available ONLY as an
                # explicit legacy ablation — ray semantics takes precedence
                # over streaming_carve_invalid_depth when both are set (never
                # silently double-apply), logged once so a misconfigured run
                # is obvious in train.log rather than silently wrong.
                if getattr(args, "streaming_carve_invalid_depth", False):
                    global _RS_WARNED_LEGACY_CARVE_COMBO
                    if not _RS_WARNED_LEGACY_CARVE_COMBO:
                        print(
                            "[W3-08] streaming_ray_semantics_enabled=True AND "
                            "streaming_carve_invalid_depth=True are both set; ray "
                            "semantics takes precedence, legacy carving is skipped "
                            "for this run (they are not meant to run combined).",
                            flush=True,
                        )
                        _RS_WARNED_LEGACY_CARVE_COMBO = True
                # No getattr fallback: streaming_max_depth is an unconditional
                # StreamingParams default, always present on the real parsed args
                # (issue #368 -- old getattr(..., 5.0) fallback drifted from the real
                # default of 8.0; removed rather than re-stated).
                _cv_max_d = float(args.streaming_max_depth)
                _certified = (_rs_classification.kind_map == 1).unsqueeze(0)  # CERTIFIED_FREE_SPACE
                _cv = (
                    _certified
                    & (rend_d_fs.detach() > 0)
                    & (rend_a_fs.detach() > 0.2)
                    & (rend_d_fs.detach() < _cv_max_d - 0.05)
                )
                if _cv.any():
                    floater = floater | _cv
                    sensor_d_fs = torch.where(_cv, torch.full_like(sensor_d_fs, _cv_max_d), sensor_d_fs)
            elif getattr(args, "streaming_carve_invalid_depth", False):
                # No getattr fallback: streaming_max_depth is an unconditional
                # StreamingParams default, always present on the real parsed args
                # (issue #368 -- old getattr(..., 5.0) fallback drifted from the real
                # default of 8.0; removed rather than re-stated).
                _cv_max_d = float(args.streaming_max_depth)
                _cv_k = int(getattr(args, "streaming_carve_erode_px", 15))
                _inv = (sensor_d_fs <= 0).float().unsqueeze(0)
                _inv_core = torch.nn.functional.avg_pool2d(
                    _inv, kernel_size=2 * _cv_k + 1, stride=1, padding=_cv_k
                ).squeeze(0) > 0.999
                _cv = (
                    _inv_core
                    & (rend_d_fs.detach() > 0)
                    & (rend_a_fs.detach() > 0.2)
                    & (rend_d_fs.detach() < _cv_max_d - 0.05)
                )
                if _cv.any():
                    floater = floater | _cv
                    sensor_d_fs = torch.where(_cv, torch.full_like(sensor_d_fs, _cv_max_d), sensor_d_fs)
            if floater.any():
                free_err = rend_a_fs[floater] * (sensor_d_fs[floater] - rend_d_fs[floater]).detach()
                if _surface_prior_fs is not None:
                    # Issue #1686: prepared path forbids H2D transfer here; tensors must already
                    # be device-resident (validated at preflight above). Legacy path still
                    # performs the transfer for backward compat.
                    conf = _legacy_to_device(_surface_prior_fs.confidence, viewpoint_cam)[floater].clamp_min(1e-4)
                    L_free = (free_err * conf).sum() / conf.sum()
                else:
                    L_free = free_err.mean()
                # Issue #1670 Part B: deferred to the single accounting
                # boundary, same pattern as the depth-loss guard above.
                def _on_free_resolved(is_finite: bool, _L_free=L_free) -> None:
                    nonlocal _free_loss_val
                    _resolved_free = _L_free if is_finite else torch.zeros((), device=_L_free.device)
                    # See the depth-loss note above: only materialize this
                    # when emitting an exceptional diagnostic or a reporting
                    # boundary.
                    _free_loss_val = _resolved_free.detach()
                    if tb_writer is not None and iteration % scalar_log_interval == 0:
                        components["free_space"] = _resolved_free.detach()

                _guard_acc.add(
                    free_space_weight * L_free,
                    "free_space_loss",
                    check_term=L_free,
                    on_resolved=_on_free_resolved,
                )

    # ---- Appearance-only auxiliary update (Wave 3, PR W3-08, plan 13.5) --
    # Independent of the depth/geometry loss above (deliverable 6:
    # streaming_depth_loss_weight vs. streaming_appearance_only_weight are
    # unrelated flags). Throttled to every streaming_appearance_only_interval
    # iterations — see appearance_only.py's module docstring for the honest
    # scope note on why this is interval-throttled rather than a genuine
    # reduced-resolution render.
    _ao_interval = max(1, int(getattr(args, "streaming_appearance_only_interval", 20)))
    if (
        _ao_enabled
        and _rs_classification is not None
        and _rs_sensor_d_hw is not None
        and background is not None
        and optimizer_update_id % _ao_interval == 0
    ):
        _ao_weight = float(getattr(args, "streaming_appearance_only_weight", 0.0))
        if _ao_weight > 0:
            _reliability = geometry_reliability_map(
                _rs_classification.kind_map, _rs_sensor_d_hw, gt_image.detach(),
                pose_confidence=float(getattr(viewpoint_cam, "pose_confidence", 1.0) or 1.0),
            )
            _risk_mask = material_risk_mask(
                _rs_classification.kind_map, _reliability,
                reliability_thresh=float(getattr(args, "streaming_appearance_only_reliability_thresh", 0.35)),
            )
            if bool(_risk_mask.any()):
                from eligsir.mapping.renderer.gsplat_backend import render as _gsplat_render
                L_ao = run_appearance_only_update(
                    _gsplat_render, gaussians, viewpoint_cam, inputs.pipe, background,
                    gt_image, _risk_mask, weight=_ao_weight,
                )
                if L_ao is not None:
                    def _on_ao_resolved(is_finite: bool, _L_ao=L_ao) -> None:
                        if is_finite:
                            components["appearance_only"] = _L_ao.detach()
                            if tb_writer is not None and iteration % scalar_log_interval == 0:
                                tb_writer.add_scalar(
                                    "streaming/appearance_only/risk_pixel_frac",
                                    float(_risk_mask.float().mean()),
                                    iteration,
                                )

                    _guard_acc.add(L_ao, name=None, on_resolved=_on_ao_resolved)

    _surface_loss_result = None
    if bool(getattr(args, "surface_loss_enabled", False)):
        _surface_loss_result = compute_surface_anchor_losses(
            gaussians,
            min_confidence=float(getattr(args, "surface_loss_min_confidence", 0.1)),
            normal_weight=float(getattr(args, "surface_anchor_normal_weight", 0.0)),
            tangent_weight=float(getattr(args, "surface_anchor_tangent_weight", 0.0)),
            normal_align_weight=float(getattr(args, "surface_normal_align_weight", 0.0)),
            thinness_weight=float(getattr(args, "surface_thinness_weight", 0.0)),
        )
        _guard_acc.add(_surface_loss_result.total, "surface")
        components["surface"] = _surface_loss_result.total.detach()
        if tb_writer is not None and iteration % scalar_log_interval == 0:
            for _k, _v in _surface_loss_result.to_tensorboard_dict().items():
                tb_writer.add_scalar(_k, _v, iteration)

    _seg.begin("m_loss_knn")
    # ---- Hot-path KNN scale-smoothness regularizer (amortized; flag-gated) --
    _knn_reg = knn_smoothness_reg(gaussians, args, optimizer_update_id)
    if not isinstance(_knn_reg, float):
        _guard_acc.add(_knn_reg, "knn_reg")
        components["knn_smoothness"] = _knn_reg.detach()
        if tb_writer is not None and iteration % scalar_log_interval == 0:
            tb_writer.add_scalar("knn_reg/scale_smoothness", float(_knn_reg.detach()), iteration)

    _seg.close()
    def finalize(loss, _loss_finite):
        nonlocal _depth_valid_sample_count
        if _guard_acc.resolved_depth_valid_sample_count is not None:
            _depth_valid_sample_count = _guard_acc.resolved_depth_valid_sample_count
        if not _loss_finite:
            inputs.nan_skip_count_holder[0] += 1
            _nan_skip_count = inputs.nan_skip_count_holder[0]
            if _nan_skip_count <= 5:
                _e_txt = ""
                if _energy_loss_enabled_main:
                    try:
                        _e_txt = f" energy_total={float(_p5_result.total.detach()):.4g}"
                    except Exception:
                        _e_txt = " energy_total=<err>"
                _vp_id = getattr(viewpoint_cam, "image_name", getattr(viewpoint_cam, "uid", "?"))
                _dv = _depth_loss_val
                _fv = _free_loss_val
                rgb_diagnostics = " ".join(
                    f"{name}={float(term.detach()):.4g}" if term is not None else f"{name}=not_measured"
                    for name, term in (("Ll1", Ll1), ("ssim", _ssim))
                )
                print(
                    f"[nan-guard] step={iteration} SKIPPING step — non-finite loss."
                    f" {rgb_diagnostics}{_e_txt}"
                    f" depth_val={_dv} free_val={_fv}"
                    f" color_finite={bool(torch.isfinite(image).all())}"
                    f" gt_finite={bool(torch.isfinite(gt_image).all())}"
                    f" N={gaussians.get_xyz.shape[0]} view={_vp_id}",
                    flush=True,
                )
            if tb_writer is not None:
                tb_writer.add_scalar("streaming/nan_skip", 1.0, iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)

        return LossBreakdown(
            total=loss.detach(), components=components, total_tensor=loss,
            loss_finite=_loss_finite,
            depth_valid_sample_count=_depth_valid_sample_count,
            photometric_evwb_state=photometric_evwb_state,
        )

    return _guard_acc, loss, finalize
