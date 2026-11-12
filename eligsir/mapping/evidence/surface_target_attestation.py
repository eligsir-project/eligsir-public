"""CPU-safe result of target validation at the population mutation fence."""

from __future__ import annotations

from dataclasses import dataclass

from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
)


@dataclass(frozen=True, slots=True)
class SurfaceTargetQualificationAttestation:
    """One immutable target-validation result for an exact post-Adam snapshot."""

    snapshot: EvidenceSnapshotToken
    parameter_version: int
    qualified: bool

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, EvidenceSnapshotToken):
            raise TypeError("snapshot must be an EvidenceSnapshotToken")
        if type(self.parameter_version) is not int or self.parameter_version < 0:
            raise ValueError("parameter_version must be a non-negative exact integer")
        if self.parameter_version != self.snapshot.parameter:
            raise ValueError("parameter_version must match snapshot.parameter")
        if type(self.qualified) is not bool:
            raise TypeError("qualified must be bool")

    def trace(self) -> dict[str, object]:
        """Return the scalar-only form retained in a decision trace."""

        return {
            "qualified": self.qualified,
            "snapshot": {
                "map": self.snapshot.map_content,
                "topology": self.snapshot.topology,
                "coordinate": self.snapshot.coordinate_frame,
                "schema": self.snapshot.schema,
                "optimizer": self.snapshot.optimizer_update,
                "surface": self.snapshot.surface_evidence,
                "policy": self.snapshot.policy,
                "parameter": self.parameter_version,
            },
        }


__all__ = ["SurfaceTargetQualificationAttestation"]
