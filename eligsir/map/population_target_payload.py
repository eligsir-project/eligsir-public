"""Concrete initialization payload for an atomic population mutation plan.

Population mutation is deliberately not expressed as a historical sequence of
``prune`` / ``insert`` / ``relocate`` calls.  At the one attributed host
mutation fence, the policy hands the arbiter a bounded, immutable description
of every target row.  The backend gets this complete payload in one call and
must apply it alongside the exact row-state transform.

This is a host-fence contract, not a new GPU hot path.  Candidate generation
remains device-resident; only the already-authorized bounded target selection
crosses the cardinality fence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PopulationTargetOpaqueField:
    """One explicitly typed non-core initialization field.

    Core geometric and appearance initialization is intentionally not opaque
    (position, colour, scale, rotation, opacity, SH).  A backend-specific
    value such as a lineage seed or renderer feature must instead be declared
    here, with its exact dtype and row suffix, so it cannot silently become a
    new post-hoc sidecar.
    """

    name: str
    dtype: torch.dtype
    shape_suffix: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("opaque field name must be non-empty")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("opaque field dtype must be a torch.dtype")
        if not isinstance(self.shape_suffix, tuple) or any(
            type(dim) is not int or dim < 0 for dim in self.shape_suffix
        ):
            raise ValueError("opaque field shape_suffix must contain non-negative integers")


@dataclass(frozen=True)
class PopulationTargetPayloadSchema:
    """Versioned, exact schema for target-row initialization payloads."""

    opaque_fields: tuple[PopulationTargetOpaqueField, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("target payload schema version must be a positive integer")
        if not isinstance(self.opaque_fields, tuple) or not all(
            isinstance(field, PopulationTargetOpaqueField) for field in self.opaque_fields
        ):
            raise TypeError("opaque_fields must be a tuple of PopulationTargetOpaqueField")
        names = tuple(field.name for field in self.opaque_fields)
        if len(set(names)) != len(names):
            raise ValueError("opaque field names must be unique")


@dataclass(frozen=True)
class PopulationTargetPayloadBatch:
    """Bounded complete initialization for CREATE and RELOCATE target rows.

    Rows are ordered exactly as ``PopulationMutationPlan.target_payload``:
    relocation targets first, then CREATE targets.  All values are CPU-resident
    because this object is only valid at the explicit
    ``MUTATION_CARDINALITY`` fence.  Backends must consume the full batch in a
    single ``apply_population_plan`` transaction; no target may be decomposed
    into ordinary legacy mutation classes.
    """

    schema: PopulationTargetPayloadSchema
    positions: torch.Tensor
    colors: torch.Tensor
    raw_scales: torch.Tensor
    quaternions: torch.Tensor
    opacity_logits: torch.Tensor
    sh_rest: torch.Tensor
    opaque_values: tuple[torch.Tensor, ...]
    max_targets: int

    def __post_init__(self) -> None:
        if not isinstance(self.schema, PopulationTargetPayloadSchema):
            raise TypeError("schema must be a PopulationTargetPayloadSchema")
        if type(self.max_targets) is not int or self.max_targets < 0:
            raise ValueError("max_targets must be a non-negative integer")
        core = {
            "positions": self.positions,
            "colors": self.colors,
            "raw_scales": self.raw_scales,
            "quaternions": self.quaternions,
            "opacity_logits": self.opacity_logits,
            "sh_rest": self.sh_rest,
        }
        if any(not isinstance(value, torch.Tensor) for value in core.values()):
            raise TypeError("all core target payload fields must be torch tensors")
        if any(value.device.type != "cpu" for value in core.values()):
            raise ValueError("target payload is valid only on CPU at MUTATION_CARDINALITY fence")
        count = self.positions.shape[0] if self.positions.ndim == 2 else -1
        expected_shapes = {
            "positions": (count, 3),
            "colors": (count, 3),
            "raw_scales": (count, 3),
            "quaternions": (count, 4),
            "opacity_logits": (count, 1),
        }
        for name, shape in expected_shapes.items():
            value = core[name]
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not value.is_floating_point():
                raise TypeError(f"{name} must use a floating-point dtype")
        if self.sh_rest.ndim < 2 or self.sh_rest.shape[0] != count:
            raise ValueError("sh_rest must have aligned rank >= 2 rows")
        if not self.sh_rest.is_floating_point():
            raise TypeError("sh_rest must use a floating-point dtype")
        if count > self.max_targets:
            raise ValueError("target payload batch exceeds configured bound")
        if not isinstance(self.opaque_values, tuple) or any(
            not isinstance(value, torch.Tensor) for value in self.opaque_values
        ):
            raise TypeError("opaque_values must be a tuple of tensors")
        if len(self.opaque_values) != len(self.schema.opaque_fields):
            raise ValueError("opaque values must exactly match the payload schema")
        for spec, value in zip(self.schema.opaque_fields, self.opaque_values):
            if value.device.type != "cpu":
                raise ValueError(f"opaque value {spec.name} must be CPU-resident")
            if value.dtype != spec.dtype:
                raise TypeError(f"opaque value {spec.name} dtype does not match schema")
            if value.shape != (count,) + spec.shape_suffix:
                raise ValueError(f"opaque value {spec.name} shape does not match schema")

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def validate_debug(self) -> None:
        """Synchronously validate numeric payload contents for diagnostics/tests."""

        for name in (
            "positions", "colors", "raw_scales", "quaternions", "opacity_logits", "sh_rest"
        ):
            if not bool(torch.isfinite(getattr(self, name)).all()):
                raise ValueError(f"{name} must contain only finite values")


__all__ = [
    "PopulationTargetOpaqueField",
    "PopulationTargetPayloadBatch",
    "PopulationTargetPayloadSchema",
]
