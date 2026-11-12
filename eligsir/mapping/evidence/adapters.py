"""Compatibility adapters: convert legacy Wave-3 InputEvidence types to canonical FrameEvidence.

This module builds canonical :class:`FrameEvidence` envelopes from two
incompatible legacy ``InputEvidence`` dataclasses:

1. :class:`eligsir.mapping.evidence.contracts.InputEvidence` — the older
   frame-input-quality type (pose/image/depth/geometry reliability).
2. :class:`eligsir.mapping.candidates.contracts.InputEvidence` — the
   prepared-candidate image-quality type (blur/exposure/valid-depth).

Both are consumed as transient dataclasses; neither is persisted to disk or
used as a stable schema. The canonical :class:`FrameEvidence` is what feeds
admission scoring, replay, and evidence-based policy decisions (issues #821,
#803, #292). The adapters make a hard distinction:

- ``AVAILABLE`` = the legacy type provides a value (converted as-is).
- ``NOT_REQUESTED`` = the legacy type lacks this signal entirely (caller
  didn't measure it, and the original dataclass has no field for it).
- ``NOT_SUPPORTED`` = the legacy type has structural support but the value
  is explicitly unavailable/unevaluated (a timestamp passed, a computation
  skipped).

The module ensures that missing does NOT equal zero — a frame without a
valid-depth measurement carries ``NOT_REQUESTED`` availability, not
``depth_valid_fraction=0.0``. This preserves the distinction crucial to
issue #822: "no evidence" vs. "evidence of zero" are not the same.

See :class:`eligsir.mapping.evidence.canonical.EvidenceAvailability`
for the full vocabulary.
"""

from __future__ import annotations

from typing import Optional

from eligsir.mapping.evidence.canonical import (
    EvidenceAvailability,
    EvidenceIdentity,
    FrameEvidence,
    GaussianProbeEvidence,
    ScalarEvidence,
    SourceReliabilityEvidence,
)


# Table-driven helper: common NOT_REQUESTED fields for SourceReliabilityEvidence
# Both adapters use this to avoid duplication of the motion/timing/tracking signals
# that neither legacy InputEvidence type provides.
def _source_reliability_template() -> dict:
    """Return the common template of NOT_REQUESTED signals for both legacy types."""
    return {
        "source_elapsed_s": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="s"),
        "translation_delta_m": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="m"),
        "rotation_delta_deg": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="deg"),
        "translation_velocity_m_per_s": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="m/s"),
        "rotation_velocity_deg_per_s": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED, unit="deg/s"),
    }


def from_mapping_input_evidence(
    legacy: eligsir.mapping.evidence.contracts.InputEvidence,
    *,
    source_id: str = "mapping-evidence",
    source_ordinal: int = 0,
) -> FrameEvidence:
    """Convert :class:`eligsir.mapping.evidence.contracts.InputEvidence`
    to canonical :class:`FrameEvidence`.

    The legacy type from ``eligsir.mapping.evidence.contracts`` carries
    pose/image/depth/geometry reliability metrics produced during training.
    It lacks motion, timing, and render-probe data — all marked as
    ``NOT_REQUESTED``.

    Args:
        legacy: The legacy InputEvidence from evidence.contracts.
        source_id: Identifier for the evidence source (e.g., "mapping-evidence").
        source_ordinal: Source index (frame number or other identifier).

    Returns:
        A canonical :class:`FrameEvidence` with fields populated from the
        legacy type and missing fields marked ``NOT_REQUESTED``.
    """
    # Import here to avoid circular imports
    import eligsir.mapping.evidence.contracts

    # Build identity from the legacy meta field
    identity = EvidenceIdentity(
        source_id=source_id,
        source_ordinal=int(source_ordinal),
        timestamp_ns=0,  # Legacy type has no timestamp
        map_version=int(legacy.meta.map_version),
        topology_version=int(legacy.meta.topology_version),
        resolution_bucket=str(legacy.meta.resolution_bucket),
    )

    # Build source reliability evidence from template + available fields
    source_dict = _source_reliability_template()
    source_dict.update({
        # Image quality signals not in legacy type
        "blur_laplacian": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED),
        "exposure_luma": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED),
        "white_clip_fraction": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED),
        "black_clip_fraction": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED),
        # Depth validity is available
        "depth_valid_fraction": ScalarEvidence(
            value=float(legacy.depth_valid_fraction),
            availability=EvidenceAvailability.AVAILABLE,
        ),
        # Pose/tracking confidence available
        "tracking_confidence": ScalarEvidence(
            value=float(legacy.pose_confidence),
            availability=EvidenceAvailability.AVAILABLE,
        ),
    })
    source = SourceReliabilityEvidence(**source_dict)

    # Legacy type has no render probe or Gaussian evidence
    probe = GaussianProbeEvidence()

    # Legacy type doesn't provide rejection reasons
    return FrameEvidence(
        identity=identity,
        source=source,
        probe=probe,
        rgb_eligible=True,
        depth_eligible=float(legacy.depth_valid_fraction) > 0.0,
        hard_reject_reasons=(),
        soft_reject_reasons=(),
        escalations=(),
    )


def from_candidate_input_evidence(
    legacy: eligsir.mapping.candidates.contracts.InputEvidence,
    *,
    source_index: int = 0,
    timestamp_ns: int = 0,
    source_id: str = "prepared-candidate",
) -> FrameEvidence:
    """Convert :class:`eligsir.mapping.candidates.contracts.InputEvidence`
    to canonical :class:`FrameEvidence`.

    The legacy type from ``eligsir.mapping.candidates.contracts`` carries
    image quality metrics (blur, exposure, clipping) and a frame-level
    ``quality_weight`` computed during CPU-side preparation. It has no motion,
    timing (except through caller timestamp_ns), or pose confidence — all
    marked as ``NOT_REQUESTED``. The ``reject_reason`` field is threaded
    into ``hard_reject_reasons``.

    Args:
        legacy: The legacy InputEvidence from candidates.contracts.
        source_index: Frame/source ordinal (typically frame ID from the source).
        timestamp_ns: Frame timestamp in nanoseconds.
        source_id: Identifier for the evidence source (e.g., "prepared-candidate").

    Returns:
        A canonical :class:`FrameEvidence` with fields populated from the
        legacy type and missing fields marked ``NOT_REQUESTED``.
    """
    # Import here to avoid circular imports
    import eligsir.mapping.candidates.contracts

    identity = EvidenceIdentity(
        source_id=source_id,
        source_ordinal=int(source_index),
        timestamp_ns=int(timestamp_ns),
    )

    # Build source reliability evidence from template + available fields
    source_dict = _source_reliability_template()
    source_dict.update({
        # Image quality signals are available
        "blur_laplacian": ScalarEvidence(
            value=float(legacy.blur_laplacian),
            availability=EvidenceAvailability.AVAILABLE,
        ),
        "exposure_luma": ScalarEvidence(
            value=float(legacy.exposure_luma),
            availability=EvidenceAvailability.AVAILABLE,
        ),
        "white_clip_fraction": ScalarEvidence(
            value=float(legacy.white_clip_ratio),
            availability=EvidenceAvailability.AVAILABLE,
        ),
        "black_clip_fraction": ScalarEvidence(
            value=float(legacy.black_clip_ratio),
            availability=EvidenceAvailability.AVAILABLE,
        ),
        # Depth validity is available
        "depth_valid_fraction": ScalarEvidence(
            value=float(legacy.valid_depth_fraction),
            availability=EvidenceAvailability.AVAILABLE,
        ),
        # Pose/tracking confidence not available in legacy type
        "tracking_confidence": ScalarEvidence.missing(EvidenceAvailability.NOT_REQUESTED),
    })
    source = SourceReliabilityEvidence(**source_dict)

    # Legacy type has no render probe or Gaussian evidence
    probe = GaussianProbeEvidence()

    # Thread reject_reason into hard_reject_reasons
    hard_rejects = ()
    if legacy.reject_reason is not None:
        hard_rejects = (str(legacy.reject_reason),)

    # quality_weight is not a canonical field, so we don't need to handle it
    # (it's prepfilter-specific and belongs in decision logic, not evidence)
    return FrameEvidence(
        identity=identity,
        source=source,
        probe=probe,
        rgb_eligible=not bool(hard_rejects),
        depth_eligible=(not bool(hard_rejects) and float(legacy.valid_depth_fraction) > 0.0),
        hard_reject_reasons=hard_rejects,
        soft_reject_reasons=(),
        escalations=(),
    )
