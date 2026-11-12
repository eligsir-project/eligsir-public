"""Canonical scientific score entry points.

Every repository call to ``utils.image_utils.psnr`` delegates here.  Metric
callers therefore operate on in-memory floating tensors with matching shapes;
serialization/visualization artifacts are rejected before a scalar can enter a
paper-bearing report.  Callers that own explicit lineage (for example a camera
GT tensor) may additionally pass :class:`ImageProvenance`, which is validated
against the same canonical score domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from eligsir.utils.image_provenance import (
    CanonicalScoreDomain,
    ChannelLayout,
    ColorOrder,
    EncodingFormat,
    ImageProvenance,
    ImageSourceDescriptor,
    TransformStep,
    TransformType,
    assert_scorable,
    detect_multires_domain_mismatch,
    provenance_for_canonical_float,
    provenance_to_report_dict,
)


@dataclass(frozen=True, slots=True)
class ScorePairAudit:
    dtype: str
    resolution: tuple[int, int]
    gt_provenance_explicit: bool
    render_provenance_explicit: bool
    contract_version: int = 1


@dataclass(frozen=True, slots=True)
class CanonicalPairScores:
    """Metric values for one already-canonical render/GT pair.

    The caller supplies SSIM/LPIPS functions so it retains model caching and
    device policy.  ``None`` means that metric was not requested on this path.
    """

    psnr: torch.Tensor
    ssim: torch.Tensor | None
    lpips: torch.Tensor | None


_PROVENANCE_ATTR = "_eligsir_score_provenance"
_VALIDATED_VERSION_ATTR = "_eligsir_score_validated_version"
SCORE_CLAMP_MIN = 0.0
SCORE_CLAMP_MAX = 1.0


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


def _validate_canonical_tensor(tensor: torch.Tensor, *, context: str) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError("scientific metric input must be a torch.Tensor")
    if not torch.is_floating_point(tensor):
        raise ValueError(
            "scientific metric input must be floating point; integer/JPEG/PNG/video "
            "presentation artifacts are not score inputs"
        )
    if tensor.ndim not in (3, 4) or int(tensor.shape[-3]) not in (1, 3):
        raise ValueError(
            f"scientific image metric input ({context}) must be CHW or NCHW "
            f"with 1 or 3 channels, got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[-2]) <= 0 or int(tensor.shape[-1]) <= 0:
        raise ValueError(f"scientific image metric input ({context}) has empty spatial dimensions")
    version = _tensor_version(tensor)
    if (
        version is not None
        and getattr(tensor, _VALIDATED_VERSION_ATTR, None) == version
    ):
        return
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0).any()) or bool((tensor > 1).any()):
        raise ValueError(
            f"scientific image metric input ({context}) must contain finite values in [0, 1]"
        )


def declare_canonical_score_tensor(
    tensor: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, ImageProvenance]:
    """Attach provenance at a trusted boundary that owns canonicalization.

    This is deliberately separate from metric validation: the scorer never
    calls it on an unattributed tensor and therefore cannot invent lineage.
    """
    if torch.is_floating_point(tensor) and tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.to(torch.float32)
    _validate_canonical_tensor(tensor, context=context)
    dtype = str(tensor.dtype).replace("torch.", "")
    shape = tuple(int(x) for x in tensor.shape[-3:])
    provenance = provenance_for_canonical_float(
        dtype,
        shape,
        (int(tensor.shape[-2]), int(tensor.shape[-1])),
    )
    setattr(tensor, _PROVENANCE_ATTR, provenance)
    version = _tensor_version(tensor)
    if version is not None:
        setattr(tensor, _VALIDATED_VERSION_ATTR, version)
    return tensor, provenance


def attach_score_provenance(
    tensor: torch.Tensor, provenance: ImageProvenance
) -> torch.Tensor:
    """Carry already-declared lineage across a trusted tensor boundary."""
    setattr(tensor, _PROVENANCE_ATTR, provenance)
    return tensor


def apply_declared_score_clamp(
    tensor: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Transform raw renderer RGB into the canonical [0, 1] score domain.

    SH/rasterization output is a finite floating radiance-like tensor and is
    not required to be upper-bounded by one.  The raw tensor is left untouched
    for loss/backprop.  Only the returned metric copy is clamped.  The returned
    record makes both sides of the boundary observable so a score can never
    hide substantial renderer overshoot. Non-finite input is rejected before
    clamping so saturation can never conceal it.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("score clamp input must be a torch.Tensor")
    if not torch.is_floating_point(tensor):
        raise ValueError("score clamp input must be floating point")
    if tensor.ndim not in (3, 4) or int(tensor.shape[-3]) not in (1, 3):
        raise ValueError(
            f"score clamp input ({context}) must be CHW or NCHW with 1 or 3 channels"
        )
    if tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.to(torch.float32)

    if int(tensor.shape[-2]) <= 0 or int(tensor.shape[-1]) <= 0:
        raise ValueError(f"score clamp input ({context}) has empty spatial dimensions")

    upper_excess = torch.clamp_min(tensor - SCORE_CLAMP_MAX, 0.0)
    stats = torch.stack(
        (
            torch.count_nonzero(~torch.isfinite(tensor)).to(torch.float64),
            torch.count_nonzero(tensor < SCORE_CLAMP_MIN).to(torch.float64),
            torch.count_nonzero(tensor > SCORE_CLAMP_MAX).to(torch.float64),
            upper_excess.sum().to(torch.float64),
            upper_excess.amax().to(torch.float64),
            tensor.amin().to(torch.float64),
            tensor.amax().to(torch.float64),
        )
    )
    (
        nonfinite,
        n_below,
        n_above,
        upper_excess_sum,
        max_upper_excess,
        value_min,
        value_max,
    ) = stats.detach().cpu().tolist()
    if int(nonfinite):
        raise ValueError(
            f"score tensor ({context}) must contain finite values in [0, 1]"
        )

    n = int(tensor.numel())
    n_below = int(n_below)
    n_above = int(n_above)
    changed = n_below + n_above
    mean_upper_excess = upper_excess_sum / n_above if n_above else 0.0
    clamped = tensor.clamp(SCORE_CLAMP_MIN, SCORE_CLAMP_MAX)
    clamped_version = _tensor_version(clamped)
    if clamped_version is not None:
        setattr(clamped, _VALIDATED_VERSION_ATTR, clamped_version)

    dtype = str(clamped.dtype).replace("torch.", "")
    shape = tuple(int(x) for x in clamped.shape[-3:])
    resolution = (int(clamped.shape[-2]), int(clamped.shape[-1]))
    source = ImageSourceDescriptor(
        dtype=dtype,
        shape=shape,
        layout=ChannelLayout.CHW,
        color_order=ColorOrder.RGB if shape[0] == 3 else ColorOrder.GRAYSCALE,
        encoding=EncodingFormat.RAW,
        resolution=resolution,
        domain=f"{dtype}_renderer_raw",
    )
    canonical = CanonicalScoreDomain(
        dtype=dtype,
        domain=f"{dtype}_0_1",
        resolution=resolution,
        is_declared_transform=True,
    )
    provenance = ImageProvenance(
        source=source,
        canonical_score_domain=canonical,
        steps=(
            TransformStep(
                transform_type=TransformType.DECLARED_SCORE_TRANSFORM,
                description="clamp raw renderer output to the canonical score domain",
                target_domain=f"{dtype}_0_1",
            ),
        ),
    )
    attach_score_provenance(clamped, provenance)
    record = {
        "applied": True,
        "context": context,
        "range": [SCORE_CLAMP_MIN, SCORE_CLAMP_MAX],
        "nonfinite_policy": "rejected before clamp",
        "samples_nonfinite": 0,
        "samples_changed": int(changed),
        "samples_below_min": n_below,
        "samples_above_max": n_above,
        "samples_total": n,
        "fraction_changed": (changed / n) if n else 0.0,
        "fraction_below_min": (n_below / n) if n else 0.0,
        "fraction_above_max": (n_above / n) if n else 0.0,
        "mean_excess_above_max": mean_upper_excess,
        "max_excess_above_max": max_upper_excess,
        "value_min": float(value_min),
        "value_max": float(value_max),
    }
    return clamped, record


def require_canonical_score_tensor(
    tensor: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Require an already-canonical score tensor without modifying samples.

    Ground truth reaches scoring through this path.  A camera/input pipeline
    that produces values outside [0, 1], non-finite samples, or missing/mismatched
    provenance is a correctness failure and must not be silently repaired by the
    same transform used for renderer predictions.
    """
    _validate_canonical_tensor(tensor, context=context)
    provenance = _tensor_provenance(tensor)
    if provenance is None:
        raise ValueError(
            f"canonical score tensor requires explicit score provenance ({context})"
        )
    assert_scorable(provenance, context=context)
    _validate_provenance_matches_tensor(provenance, tensor, context=context)
    n = int(tensor.numel())
    record = {
        "applied": False,
        "context": context,
        "range": [SCORE_CLAMP_MIN, SCORE_CLAMP_MAX],
        "nonfinite_policy": "rejected before scoring",
        "samples_nonfinite": 0,
        "samples_changed": 0,
        "samples_below_min": 0,
        "samples_above_max": 0,
        "samples_total": n,
        "fraction_changed": 0.0,
        "fraction_below_min": 0.0,
        "fraction_above_max": 0.0,
        "mean_excess_above_max": 0.0,
        "max_excess_above_max": 0.0,
        "value_min": float(tensor.min().item()) if n else 0.0,
        "value_max": float(tensor.max().item()) if n else 0.0,
    }
    return tensor, record


def canonical_score_batch(tensor: torch.Tensor, *, context: str) -> torch.Tensor:
    """Return one canonical score tensor as NCHW without losing provenance."""
    tensor, _ = require_canonical_score_tensor(tensor, context=context)
    if tensor.ndim == 4:
        return tensor
    provenance = _tensor_provenance(tensor)
    batched = tensor.unsqueeze(0)
    attach_score_provenance(batched, provenance)
    version = _tensor_version(batched)
    if version is not None:
        setattr(batched, _VALIDATED_VERSION_ATTR, version)
    return batched


def apply_declared_score_clamp_pair(
    render: torch.Tensor,
    gt: torch.Tensor,
    *,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, Any]]]:
    """Canonicalize a renderer prediction while strictly validating its GT.

    The asymmetric contract is intentional: finite renderer overshoot is legal
    and gets an explicit score-domain transform, while GT is owned by the input
    canonicalization boundary and must already be valid [0, 1] data.
    """
    render_score, render_record = apply_declared_score_clamp(
        render,
        context=f"{context}:render",
    )
    gt_score, gt_record = require_canonical_score_tensor(
        gt,
        context=f"{context}:gt",
    )
    audit = validate_score_pair(
        render_score,
        gt_score,
        context=context,
    )
    render_provenance = _tensor_provenance(render_score)
    gt_provenance = _tensor_provenance(gt_score)
    if render_provenance is None or gt_provenance is None:  # validate_score_pair fails first
        raise ValueError(f"score provenance disappeared after validation ({context})")
    return render_score, gt_score, {
        "render": render_record,
        "gt": gt_record,
        "score_provenance": {
            "contract_version": audit.contract_version,
            "dtype": audit.dtype,
            "domain": gt_provenance.canonical_score_domain.domain,
            "resolution": audit.resolution,
            "render": provenance_to_report_dict(render_provenance),
            "gt": provenance_to_report_dict(gt_provenance),
        },
    }


def apply_declared_camera_grid_score_clamp_pair(
    render: torch.Tensor,
    gt: torch.Tensor,
    *,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, Any]]]:
    """Score a render produced directly on one declared camera score grid.

    A resolution-tier camera carries the authoritative grid transform in its
    GT provenance.  The renderer consumes that already-transformed camera and
    therefore emits samples on the same grid; it does not resize its output.
    This trusted boundary copies only that declared grid identity onto the
    renderer lineage before the ordinary strict pair validation.  Generic
    scoring remains fail-closed and never infers a matching grid from shape.
    """

    render_score, render_record = apply_declared_score_clamp(
        render,
        context=f"{context}:render",
    )
    gt_score, gt_record = require_canonical_score_tensor(
        gt,
        context=f"{context}:gt",
    )
    render_provenance = _tensor_provenance(render_score)
    gt_provenance = _tensor_provenance(gt_score)
    if render_provenance is None or gt_provenance is None:
        raise ValueError(f"score provenance disappeared before camera-grid validation ({context})")
    if render_score.shape != gt_score.shape or render_score.device != gt_score.device:
        # Keep the ordinary pair diagnostics and rejection semantics.
        validate_score_pair(render_score, gt_score, context=context)

    grid_steps = tuple(
        step
        for step in gt_provenance.steps
        if step.transform_type is TransformType.DECLARED_SCORE_TRANSFORM
        and step.target_resolution is not None
    )
    if grid_steps:
        render_provenance = ImageProvenance(
            source=render_provenance.source,
            canonical_score_domain=render_provenance.canonical_score_domain,
            steps=grid_steps + render_provenance.steps,
        )
        attach_score_provenance(render_score, render_provenance)

    audit = validate_score_pair(render_score, gt_score, context=context)
    return render_score, gt_score, {
        "render": render_record,
        "gt": gt_record,
        "camera_grid": {
            "source": "authoritative_camera_score_provenance",
            "declared": bool(grid_steps),
            "resolution": audit.resolution,
            "transforms": [
                {
                    "description": step.description,
                    "target_resolution": step.target_resolution,
                }
                for step in grid_steps
            ],
        },
        "score_provenance": {
            "contract_version": audit.contract_version,
            "dtype": audit.dtype,
            "domain": gt_provenance.canonical_score_domain.domain,
            "resolution": audit.resolution,
            "render": provenance_to_report_dict(render_provenance),
            "gt": provenance_to_report_dict(gt_provenance),
        },
    }


def declare_camera_score_provenance(camera: Any, *, context: str) -> None:
    """Declare canonical GT lineage at a camera materialization boundary."""
    camera.original_image, camera.score_provenance = declare_canonical_score_tensor(
        camera.original_image,
        context=context,
    )


def reattach_camera_score_provenance(camera: Any) -> None:
    """Preserve existing camera lineage across copies without inventing it."""
    provenance = camera_provenance(camera)
    if provenance is not None:
        attach_score_provenance(camera.original_image, provenance)


def _tensor_provenance(tensor: torch.Tensor) -> ImageProvenance | None:
    return getattr(tensor, _PROVENANCE_ATTR, None)


def _validate_provenance_matches_tensor(
    provenance: ImageProvenance, tensor: torch.Tensor, *, context: str
) -> None:
    """Reject a lineage descriptor that does not describe the scored tensor."""
    dtype = str(tensor.dtype).replace("torch.", "")
    resolution = (int(tensor.shape[-2]), int(tensor.shape[-1]))
    canonical = provenance.canonical_score_domain
    if (
        canonical.dtype != dtype
        or canonical.domain != f"{dtype}_0_1"
        or tuple(canonical.resolution) != resolution
    ):
        raise ValueError(
            f"canonical score provenance does not match tensor dtype/domain/resolution ({context})"
        )


def validate_score_pair(
    render: torch.Tensor,
    gt: torch.Tensor,
    *,
    render_provenance: ImageProvenance | None = None,
    gt_provenance: ImageProvenance | None = None,
    context: str = "scientific_metric",
) -> ScorePairAudit:
    """Fail closed unless render and GT occupy one canonical score domain."""
    if not torch.is_tensor(render) or not torch.is_tensor(gt):
        raise TypeError("scientific metric inputs must both be torch tensors")
    if render.shape != gt.shape:
        raise ValueError(
            f"score-domain shape mismatch: render={tuple(render.shape)} gt={tuple(gt.shape)}"
        )
    _validate_canonical_tensor(render, context=f"{context}:render")
    _validate_canonical_tensor(gt, context=f"{context}:gt")
    if render.device != gt.device:
        raise ValueError(
            f"score-domain device mismatch: render={render.device} gt={gt.device}; "
            "materialize the pair deliberately before scoring"
        )
    rprov = render_provenance or _tensor_provenance(render)
    gprov = gt_provenance or _tensor_provenance(gt)
    if rprov is None or gprov is None:
        missing = []
        if rprov is None:
            missing.append("render")
        if gprov is None:
            missing.append("gt")
        raise ValueError(
            "scientific metric inputs require explicit score provenance from a "
            f"trusted boundary; missing {', '.join(missing)} ({context})"
        )
    assert_scorable(rprov, context=f"{context}:render")
    assert_scorable(gprov, context=f"{context}:gt")
    _validate_provenance_matches_tensor(rprov, render, context=f"{context}:render")
    _validate_provenance_matches_tensor(gprov, gt, context=f"{context}:gt")
    mismatch = detect_multires_domain_mismatch(gprov, rprov)
    render_has_score_clamp = any(
        step.transform_type is TransformType.DECLARED_SCORE_TRANSFORM
        and step.target_domain == rprov.canonical_score_domain.domain
        for step in rprov.steps
    )
    if (
        mismatch is not None
        and mismatch.startswith("Transform mismatch: GT is_declared_transform=")
        and render_has_score_clamp
        and rprov.canonical_score_domain.resolution
        == gprov.canonical_score_domain.resolution
        and rprov.canonical_score_domain.domain
        == gprov.canonical_score_domain.domain
    ):
        # Prediction-only score clamping deliberately has no GT counterpart:
        # GT must already be canonical and is never sample-modified. This is
        # not the multires mismatch that the generic provenance helper guards.
        mismatch = None
    if mismatch:
        raise ValueError(f"score-domain mismatch ({context}): {mismatch}")
    return ScorePairAudit(
        dtype=str(render.dtype).replace("torch.", ""),
        resolution=(int(render.shape[-2]), int(render.shape[-1])),
        gt_provenance_explicit=gt_provenance is not None,
        render_provenance_explicit=render_provenance is not None,
    )


def _psnr_raw(render: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Historical Graphdeco PSNR formula, kept byte-for-byte in tensor math."""
    mse = (((render - gt)) ** 2).view(render.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def _scientific_psnr_validated(render: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Historical PSNR arithmetic after the canonical-pair preflight."""
    if render.dtype not in (torch.float32, torch.float64):
        render = render.to(torch.float32)
    if gt.dtype != render.dtype:
        gt = gt.to(render.dtype)
    return _psnr_raw(render, gt)


def scientific_psnr(
    render: torch.Tensor,
    gt: torch.Tensor,
    *,
    render_provenance: ImageProvenance | None = None,
    gt_provenance: ImageProvenance | None = None,
    context: str = "psnr",
):
    """PSNR on canonical in-memory tensors only.

    Half/bfloat tensors are promoted to float32 after validation. Float32 and
    float64 preserve their native dtype, so established float32 paper numbers
    retain the same arithmetic as the historical implementation.
    """
    validate_score_pair(
        render,
        gt,
        render_provenance=render_provenance,
        gt_provenance=gt_provenance,
        context=context,
    )
    return _scientific_psnr_validated(render, gt)


def score_canonical_pair(
    render: torch.Tensor,
    gt: torch.Tensor,
    *,
    ssim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    lpips_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    context: str = "canonical_pair_metrics",
) -> CanonicalPairScores:
    """Score one validated tensor pair through the shared scientific engine.

    This deliberately does not instantiate or cache LPIPS, choose a device, or
    alter tensor layout.  Those are caller-owned runtime concerns.  It merely
    makes the contract preflight and PSNR/SSIM/LPIPS sequencing identical for
    inline and post-hoc evaluation.
    """
    validate_score_pair(render, gt, context=context)
    return CanonicalPairScores(
        psnr=_scientific_psnr_validated(render, gt),
        ssim=ssim_fn(render, gt) if ssim_fn is not None else None,
        lpips=lpips_fn(render, gt) if lpips_fn is not None else None,
    )


def camera_provenance(camera: Any) -> ImageProvenance | None:
    """Return camera-attached scientific GT provenance when available."""
    return getattr(camera, "score_provenance", None)


def score_tensor_from_camera(camera: Any, *, device: Any = None) -> torch.Tensor:
    """Materialize camera GT while preserving its trusted score provenance."""
    provenance = camera_provenance(camera)
    if provenance is None:
        raise ValueError("camera has no trusted score provenance")
    tensor = camera.original_image[:3]
    if device is not None:
        tensor = tensor.to(device, non_blocking=True)
    attach_score_provenance(tensor, provenance)
    version = _tensor_version(tensor)
    if version is not None:
        setattr(tensor, _VALIDATED_VERSION_ATTR, version)
    return tensor


__all__ = [
    "ScorePairAudit",
    "CanonicalPairScores",
    "attach_score_provenance",
    "apply_declared_score_clamp",
    "apply_declared_camera_grid_score_clamp_pair",
    "apply_declared_score_clamp_pair",
    "camera_provenance",
    "canonical_score_batch",
    "declare_camera_score_provenance",
    "declare_canonical_score_tensor",
    "reattach_camera_score_provenance",
    "require_canonical_score_tensor",
    "score_tensor_from_camera",
    "scientific_psnr",
    "score_canonical_pair",
    "validate_score_pair",
]
