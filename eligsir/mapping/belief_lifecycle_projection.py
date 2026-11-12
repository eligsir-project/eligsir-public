"""Diagnostic projection of legacy lifecycle labels from ``GaussianBelief``.

This is a migration/telemetry adapter, not a lifecycle owner.  It derives a
small, device-resident view for parity reporting while the old lifecycle and
MCMC controls remain authoritative.  In particular, missing measurement is
not treated as contradictory evidence and none of the returned masks are
written back to a Gaussian model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch

from eligsir.utils.gaussian_belief import GaussianBelief


class DiagnosticLifecycle(IntEnum):
    """Non-authoritative labels used only for migration diagnostics."""

    PROVISIONAL = 0
    YOUNG = 1
    ESTABLISHED = 2

    @property
    def code(self) -> int:
        return int(self)


@dataclass(frozen=True)
class DiagnosticLifecycleProjection:
    """Device-resident masks and parity views for one belief snapshot."""

    state: torch.Tensor
    birth_grace: torch.Tensor
    no_measurement: torch.Tensor
    negative_evidence: torch.Tensor
    established: torch.Tensor
    legacy_established: torch.Tensor | None = None
    established_agreement: torch.Tensor | None = None


def _validate_vector(name: str, value: torch.Tensor, *, n: int, device: torch.device) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"belief field {name} must be a rank-1 tensor")
    if value.shape[0] != n or value.device != device:
        raise ValueError(f"belief field {name} must have shape [{n}] on {device}")


def project_diagnostic_lifecycle(
    belief: GaussianBelief,
    *,
    optimizer_update: int,
    min_meaningful_exposures: int = 1,
    min_independent_support_views: int = 1,
    min_surface_confidence: float = 0.75,
    birth_grace_updates: int = 8,
    legacy_state: torch.Tensor | None = None,
) -> DiagnosticLifecycleProjection:
    """Derive diagnostic lifecycle labels from evidence, without side effects.

    ``PROVISIONAL`` means either a short birth grace is active or the Gaussian
    lacks both meaningful exposure and independent supporting views.  ``YOUNG``
    means it is measurable but its surface confidence is not established.
    ``ESTABLISHED`` requires measurable evidence and surface confidence.  A
    contradiction is reported independently; it never turns missing evidence
    into negative evidence.

    All masks remain on the belief tensor's device.  The function performs no
    tensor-to-host conversion or synchronizing reduction.
    """
    if not isinstance(belief, GaussianBelief):
        raise TypeError("belief must be GaussianBelief")
    if type(optimizer_update) is not int or optimizer_update < 0:
        raise ValueError("optimizer_update must be a non-negative exact integer")
    for name, value in (
        ("min_meaningful_exposures", min_meaningful_exposures),
        ("min_independent_support_views", min_independent_support_views),
        ("birth_grace_updates", birth_grace_updates),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative exact integer")
    if not isinstance(min_surface_confidence, (float, int)):
        raise TypeError("min_surface_confidence must be numeric")
    if not 0.0 <= float(min_surface_confidence) <= 1.0:
        raise ValueError("min_surface_confidence must be in [0, 1]")
    if belief.n < 0 or type(belief.n) is not int:
        raise ValueError("belief.n must be a non-negative exact integer")

    birth = belief.provenance.birth_update
    device = birth.device
    n = belief.n
    fields = {
        "birth_update": birth,
        "meaningful_exposure_count": belief.observability.meaningful_exposure_count,
        "independent_support_views": belief.observability.independent_support_views,
        "p_surface_consistent": belief.geometry.p_surface_consistent,
        "contradiction_evidence": belief.existence.contradiction_evidence,
    }
    for name, value in fields.items():
        if value is None:
            raise ValueError(f"belief field {name} is required")
        _validate_vector(name, value, n=n, device=device)
    if legacy_state is not None:
        if legacy_state.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("legacy_state must be an integer tensor")
        _validate_vector("legacy_state", legacy_state, n=n, device=device)

    # ``None`` means measurement accounting is unavailable, not that the row
    # was contradicted.  The accessor normally provides this field when the
    # exposure feature is enabled; fail closed if it is absent.
    measurable = (
        (belief.observability.meaningful_exposure_count >= min_meaningful_exposures)
        & (belief.observability.independent_support_views >= min_independent_support_views)
    )
    no_measurement = ~measurable
    age = optimizer_update - birth
    birth_grace = (age >= 0) & (age < birth_grace_updates)
    negative_evidence = belief.existence.contradiction_evidence > 0
    established = measurable & ~birth_grace & (
        belief.geometry.p_surface_consistent >= min_surface_confidence
    )
    state = torch.full((n,), DiagnosticLifecycle.YOUNG.code, dtype=torch.int8, device=device)
    state = torch.where(measurable & ~birth_grace, state, torch.full_like(state, DiagnosticLifecycle.PROVISIONAL.code))
    state = torch.where(established, torch.full_like(state, DiagnosticLifecycle.ESTABLISHED.code), state)

    legacy_established = None
    established_agreement = None
    if legacy_state is not None:
        legacy_established = legacy_state >= 2
        established_agreement = legacy_established == established
    return DiagnosticLifecycleProjection(
        state=state,
        birth_grace=birth_grace,
        no_measurement=no_measurement,
        negative_evidence=negative_evidence,
        established=established,
        legacy_established=legacy_established,
        established_agreement=established_agreement,
    )


__all__ = ["DiagnosticLifecycle", "DiagnosticLifecycleProjection", "project_diagnostic_lifecycle"]
