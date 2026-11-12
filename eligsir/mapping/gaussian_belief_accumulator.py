"""Device-native belief accumulation for one completed canonical true-B4 step.

The reducer in ``true_batch_executor`` owns measurement production.  This
module only validates and folds its full-row E0 tensors into already-registered
Gaussian row state.  Renderer incidence is repeated measurement opportunity,
not independent physical-view proof.  Dedicated live-policy support counts
therefore come only from validated ``SourceIdentity`` values, never lifecycle
sidecars.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch

from eligsir.contracts.runtime import SourceIdentity
from eligsir.map.row_state import GaussianRowStateRegistry
from eligsir.optimization.gaussian.supervision_evidence import (
    SupervisionEvidenceBatch,
)


class GaussianBeliefAccumulationStatus(str, Enum):
    UPDATED = "updated"
    NOT_COMPLETED_TRUE_B4 = "not_completed_true_b4"
    MISSING_EVIDENCE = "missing_evidence"
    STALE_SNAPSHOT = "stale_snapshot"
    INVALID_EVIDENCE = "invalid_evidence"
    SOURCE_LEDGER_SATURATED = "source_ledger_saturated"
    INVALID_ROW_REGISTRY = "invalid_row_registry"


_REGISTERED_FIELDS = {
    "population_policy_exposure_count": "belief.population_policy_exposure_count",
    "population_policy_support_count": "belief.population_policy_support_count",
    "population_policy_source_seen": "belief.population_policy_source_seen",
    "last_exposed_update": "belief.last_exposed_update",
    "visibility_ema": "render.visibility_ema",
    "utility_ema": "provenance.utility_ema",
    "expected_marginal_gain": "belief.expected_marginal_gain",
}

# A source is retained for the lifetime of the current model instance.  Slots
# never evict: losing an identity-to-row association would make a later replay
# indistinguishable from a new physical view.  B4 is deliberately small, so a
# compact fixed table is ample while still providing a hard fail-closed bound.
_SOURCE_LEDGER_CAPACITY = 256


def ensure_population_policy_e0_buffers(gaussians: Any) -> None:
    """Register the live-policy's own E0 counters without enabling a veto.

    These counts deliberately do not alias ``resolution_exposure.exposure_count``
    or the legacy lifecycle ``support_count`` sidecar.  A live policy epoch
    must either have these bounded row buffers or abstain; it must never borrow
    an unrelated feature's history.
    """

    xyz = getattr(gaussians, "get_xyz", None)
    if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2:
        raise ValueError("population policy E0 buffers require Gaussian rows")
    n, device = xyz.shape[0], xyz.device
    for name, dtype in (
        ("population_policy_exposure_count", torch.int32),
        ("population_policy_support_count", torch.int16),
    ):
        value = getattr(gaussians, name, None)
        if value is None:
            setattr(gaussians, name, torch.zeros(n, dtype=dtype, device=device))
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.shape[0] != n
            or value.device != device
            or value.dtype != dtype
        ):
            raise ValueError(f"invalid population policy E0 buffer: {name}")
    seen = getattr(gaussians, "population_policy_source_seen", None)
    if seen is None:
        setattr(
            gaussians,
            "population_policy_source_seen",
            torch.zeros((n, _SOURCE_LEDGER_CAPACITY), dtype=torch.bool, device=device),
        )
    elif (
        not isinstance(seen, torch.Tensor)
        or seen.shape != (n, _SOURCE_LEDGER_CAPACITY)
        or seen.device != device
        or seen.dtype is not torch.bool
    ):
        raise ValueError("invalid population policy E0 buffer: population_policy_source_seen")
    slots = getattr(gaussians, "_population_policy_source_slot_by_fingerprint", None)
    if slots is None:
        setattr(gaussians, "_population_policy_source_slot_by_fingerprint", {})
    elif not isinstance(slots, dict) or len(slots) > _SOURCE_LEDGER_CAPACITY:
        raise ValueError("invalid population policy E0 source ledger")
    identities = getattr(gaussians, "_population_policy_source_identity_by_fingerprint", None)
    if identities is None:
        setattr(gaussians, "_population_policy_source_identity_by_fingerprint", {})
    elif not isinstance(identities, dict) or set(identities) != set(slots):
        raise ValueError("invalid population policy E0 source ledger")


def _exact_nonnegative(value: Any) -> bool:
    return type(value) is int and value >= 0


def _completed_true_b4(result: Any) -> bool:
    return (
        getattr(result, "true_camera_batch", None) is True
        and getattr(result, "step_allowed", None) is True
        and getattr(result, "batch_size", None) == 4
        and getattr(result, "supervision_view_count", None) == 4
        and getattr(result, "skipped_view_count", None) == 0
        and getattr(result, "optimizer_step_count", None) == 1
        and getattr(result, "camera_batch_rasterizer_calls", None) == 1
    )


def _registered_rows(gaussians: Any, *, n: int, device: torch.device):
    builder = getattr(gaussians, "build_row_state_registry", None)
    if not callable(builder):
        return None
    try:
        registry = builder()
    except Exception:
        return None
    if not isinstance(registry, GaussianRowStateRegistry):
        return None
    descriptors = {descriptor.name: descriptor for descriptor in registry.descriptors}
    if not set(_REGISTERED_FIELDS.values()) <= registry.names:
        return None

    rows: dict[str, torch.Tensor] = {}
    for attribute, registered_name in _REGISTERED_FIELDS.items():
        value = getattr(gaussians, attribute, None)
        descriptor = descriptors.get(registered_name)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim not in (1, 2)
            or value.shape[0] != n
            or value.device != device
            or descriptor is None
            or descriptor.dtype != value.dtype
            or descriptor.shape_suffix != tuple(value.shape[1:])
        ):
            return None
        if attribute not in {"visibility_ema", "population_policy_source_seen"} and value.ndim != 1:
            return None
        if attribute == "visibility_ema" and tuple(value.shape[1:]) not in ((), (1,)):
            return None
        if attribute == "population_policy_source_seen" and value.shape[1:] != (_SOURCE_LEDGER_CAPACITY,):
            return None
        rows[attribute] = value

    if (
        rows["population_policy_exposure_count"].dtype.is_floating_point
        or rows["population_policy_support_count"].dtype.is_floating_point
    ):
        return None
    if rows["last_exposed_update"].dtype.is_floating_point:
        return None
    if rows["population_policy_source_seen"].dtype is not torch.bool:
        return None
    if any(
        not rows[name].dtype.is_floating_point
        for name in (
            "visibility_ema", "utility_ema",
            "expected_marginal_gain",
        )
    ):
        return None
    return rows


def _current_versions_match(
    snapshot: Any,
    current_versions: Any,
    *,
    map_after: int,
    parameter_after: int,
) -> bool:
    """Require the exact post-step state before changing row evidence."""

    names = (
        # Fixed-topology optimization advances visible map content alongside
        # parameter state.  E0 remains stamped to the pre-render content, so
        # compare the current owner to the completed result rather than
        # incorrectly requiring the pre-step snapshot to still be current.
        ("map_content", map_after),
        ("topology", snapshot.topology),
        ("coordinate_frame", snapshot.coordinate_frame),
        ("schema", snapshot.schema),
        ("optimizer_update", snapshot.optimizer_update),
        ("surface_evidence", snapshot.surface_evidence),
        ("policy", snapshot.policy),
    )
    return all(
        _exact_nonnegative(getattr(current_versions, name, None))
        and getattr(current_versions, name) == expected
        for name, expected in names
    ) and (
        _exact_nonnegative(getattr(current_versions, "parameter", None))
        and current_versions.parameter == parameter_after
    )


def accumulate_true_b4_belief(
    gaussians: Any, result: Any, *, current_versions: Any | None
) -> GaussianBeliefAccumulationStatus:
    """Fold full-row E0 into registered belief tensors without host sync.

    All metadata, version, device, shape, and registry checks finish before any
    row is written.  Tensor content stays on-device throughout; there is no
    render, transfer, ``item()``, CPU conversion, or independent-view update
    that is not bound to validated physical source identities.
    """

    if not _completed_true_b4(result):
        return GaussianBeliefAccumulationStatus.NOT_COMPLETED_TRUE_B4
    evidence = getattr(result, "evidence", None)
    if not isinstance(evidence, SupervisionEvidenceBatch):
        return GaussianBeliefAccumulationStatus.MISSING_EVIDENCE
    snapshot = evidence.snapshot_token
    map_before = getattr(result, "map_version_before", None)
    map_after = getattr(result, "map_version_after", None)
    topology = getattr(result, "topology_version", None)
    parameter_before = getattr(result, "parameter_version_before", None)
    parameter_after = getattr(result, "parameter_version_after", None)
    if not all(
        _exact_nonnegative(value)
        for value in (map_before, map_after, topology, parameter_before, parameter_after)
    ):
        return GaussianBeliefAccumulationStatus.STALE_SNAPSHOT
    if (
        map_after - map_before not in (0, 1)
        or snapshot.map_content != map_before
        or snapshot.topology != topology
        or snapshot.parameter != parameter_before
        or parameter_after < parameter_before
        or not _current_versions_match(
            snapshot,
            current_versions,
            map_after=map_after,
            parameter_after=parameter_after,
        )
    ):
        return GaussianBeliefAccumulationStatus.STALE_SNAPSHOT

    xyz = getattr(gaussians, "get_xyz", None)
    if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2:
        return GaussianBeliefAccumulationStatus.INVALID_ROW_REGISTRY
    n, device = xyz.shape[0], xyz.device
    incidence = evidence.view_incidence
    opportunity = evidence.meaningful_measurement_opportunity
    contribution = evidence.render_contribution
    gradient_utility = evidence.gradient_utility_evidence
    if (
        evidence.gaussian_ids is not None
        or evidence.gaussian_count != n
        or len(evidence.modalities) != 4
        or len(evidence.source_identities) != 4
        or len(set(evidence.source_identities)) != 4
        or not all(isinstance(identity, SourceIdentity) for identity in evidence.source_identities)
        or not isinstance(incidence, torch.Tensor)
        or incidence.shape != (n, 4)
        or incidence.dtype is not torch.bool
    ):
        return GaussianBeliefAccumulationStatus.INVALID_EVIDENCE
    for value in (opportunity, contribution, gradient_utility):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or value.shape[0] != n
            or value.device != device
            or not value.dtype.is_floating_point
        ):
            return GaussianBeliefAccumulationStatus.INVALID_EVIDENCE
    if incidence.device != device:
        return GaussianBeliefAccumulationStatus.INVALID_EVIDENCE

    # Allocate each physical identity one non-evicting slot before touching a
    # row.  This permits a later B4 to contain a mixture of prior and new
    # cameras, while rejecting a saturated ledger rather than forgetting a
    # prior view and fabricating independent support on replay.
    source_slots = getattr(gaussians, "_population_policy_source_slot_by_fingerprint", None)
    source_identities = getattr(
        gaussians, "_population_policy_source_identity_by_fingerprint", None
    )
    if not isinstance(source_slots, dict) or not isinstance(source_identities, dict):
        return GaussianBeliefAccumulationStatus.INVALID_EVIDENCE
    pending_slots: dict[str, int] = {}
    pending_identities: dict[str, Any] = {}
    next_slot = len(source_slots)
    for identity in evidence.source_identities:
        fingerprint = identity.fingerprint()
        slot = source_slots.get(fingerprint)
        if slot is None:
            if next_slot >= _SOURCE_LEDGER_CAPACITY:
                return GaussianBeliefAccumulationStatus.SOURCE_LEDGER_SATURATED
            slot = next_slot
            next_slot += 1
            pending_slots[fingerprint] = slot
            pending_identities[fingerprint] = identity
        elif (
            type(slot) is not int
            or not 0 <= slot < _SOURCE_LEDGER_CAPACITY
            or source_identities.get(fingerprint) != identity
        ):
            return GaussianBeliefAccumulationStatus.INVALID_EVIDENCE
    slot_indices = [
        source_slots[identity.fingerprint()]
        if identity.fingerprint() in source_slots
        else pending_slots[identity.fingerprint()]
        for identity in evidence.source_identities
    ]

    rows = _registered_rows(gaussians, n=n, device=device)
    if rows is None:
        return GaussianBeliefAccumulationStatus.INVALID_ROW_REGISTRY

    # Reconstruct deterministic renderer-derived fields from incidence instead
    # of trusting caller-owned floating payload values.  Row-local equality
    # masks reject malformed/non-finite input without a device-to-host
    # reduction; invalid rows retain their prior belief unchanged.
    rgb_evidence = torch.tensor(
        [modality.rgb_evidence_enabled for modality in evidence.modalities],
        dtype=torch.float32,
        device=device,
    )
    expected_opportunity = incidence.to(dtype=torch.float32).matmul(rgb_evidence)
    expected_contribution = expected_opportunity / 4.0
    valid_gradient = torch.isfinite(gradient_utility) & (gradient_utility >= 0)
    valid_rows = (
        torch.isfinite(opportunity)
        & torch.isfinite(contribution)
        & valid_gradient
        & (opportunity == expected_opportunity.to(dtype=opportunity.dtype))
        & (contribution == expected_contribution.to(dtype=contribution.dtype))
    )
    exposed = valid_rows & (expected_opportunity > 0)
    exposure_next = torch.where(
        exposed,
        rows["population_policy_exposure_count"] + expected_opportunity.to(
            dtype=rows["population_policy_exposure_count"].dtype
        ),
        rows["population_policy_exposure_count"],
    )
    # Each incidence column has already been attested as one distinct physical
    # ``SourceIdentity``.  Count only identities for cameras where this row
    # was actually incident; a global B4 source count would fabricate support
    # for cameras that did not observe the row.
    seen_columns = rows["population_policy_source_seen"].index_select(
        1, torch.tensor(
            slot_indices,
            dtype=torch.int64,
            device=device,
        )
    )
    new_incidence = incidence & ~seen_columns
    independent_sources = new_incidence.sum(dim=1).to(dtype=torch.int32)
    support_limit = torch.iinfo(rows["population_policy_support_count"].dtype).max
    support_with_delta = (
        rows["population_policy_support_count"].to(dtype=torch.int32)
        + independent_sources
    ).clamp_max(support_limit).to(dtype=rows["population_policy_support_count"].dtype)
    support_next = torch.where(
        exposed,
        support_with_delta,
        rows["population_policy_support_count"],
    )
    update_id = torch.full_like(
        rows["last_exposed_update"], snapshot.optimizer_update
    )
    last_exposed_next = torch.where(
        exposed, update_id, rows["last_exposed_update"]
    )
    visibility = rows["visibility_ema"]
    visibility_vector = visibility if visibility.ndim == 1 else visibility.squeeze(1)
    visibility_next = torch.where(
        exposed,
        visibility_vector * 0.5
        + expected_contribution.to(dtype=visibility_vector.dtype) * 0.5,
        visibility_vector,
    )
    utility = torch.where(
        valid_gradient,
        gradient_utility.detach().clamp_min(0),
        torch.zeros_like(gradient_utility),
    )
    # A per-row saturating transform keeps the score bounded without a full-N
    # normalization reduction.  It does not compare rows or fabricate support.
    utility_score = utility / (utility + torch.finfo(utility.dtype).eps)
    measured_gain = utility_score * expected_contribution.to(
        dtype=utility_score.dtype
    )
    utility_next = torch.where(
        exposed,
        torch.maximum(
            rows["utility_ema"] * 0.5,
            utility_score.to(dtype=rows["utility_ema"].dtype),
        ),
        rows["utility_ema"],
    )
    gain_next = torch.where(
        exposed,
        torch.maximum(
            rows["expected_marginal_gain"] * 0.5,
            measured_gain.to(dtype=rows["expected_marginal_gain"].dtype),
        ),
        rows["expected_marginal_gain"],
    )

    with torch.no_grad():
        rows["population_policy_exposure_count"].copy_(exposure_next)
        rows["population_policy_support_count"].copy_(support_next)
        # Seen state is row-local: only a row that supplied valid evidence may
        # consume an identity slot.  CREATE/RELOCATE row defaults reset this
        # registered buffer, so fresh hypotheses never inherit donor support.
        seen_next = rows["population_policy_source_seen"].clone()
        seen_next[:, torch.tensor(
            slot_indices,
            dtype=torch.int64, device=device,
        )] |= incidence & exposed.unsqueeze(1)
        rows["population_policy_source_seen"].copy_(seen_next)
        rows["last_exposed_update"].copy_(last_exposed_next)
        rows["visibility_ema"].copy_(
            visibility_next if visibility.ndim == 1 else visibility_next.unsqueeze(1)
        )
        rows["utility_ema"].copy_(utility_next)
        rows["expected_marginal_gain"].copy_(gain_next)
    source_slots.update(pending_slots)
    source_identities.update(pending_identities)
    return GaussianBeliefAccumulationStatus.UPDATED


__all__ = [
    "GaussianBeliefAccumulationStatus",
    "accumulate_true_b4_belief",
    "ensure_population_policy_e0_buffers",
]
