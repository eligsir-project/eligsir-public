"""Pure, bounded contract for one atomic Gaussian population mutation fence.

The plan is deliberately an immutable description.  It does not resize model
or optimizer tensors; the arbiter consumes its exact row transform and applies
all registered row state in one transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken
from eligsir.map.row_state import GaussianRowStateRegistry, PopulationRowTransform, RowStateDescriptor


class MutationAction(Enum):
    NONE = "none"
    CREATE = "create"
    RELOCATE = "relocate"
    RETIRE = "retire"


_INT_DTYPES = {torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64}


@dataclass(frozen=True)
class PopulationMutationPlan:
    """Bounded typed operations for a single mutation epoch.

    ``relocate_donor_rows`` and ``relocate_target_rows`` are paired tensors:
    each donor slot is reused for a target carrying a *new* hypothesis UID.
    ``target_payload`` contains one row per RELOCATE target followed by one
    row per CREATE target (the target row feature schema is owned by the
    caller), and is borrowed device storage.
    """

    snapshot_token: EvidenceSnapshotToken
    next_snapshot_token: EvidenceSnapshotToken
    row_transform: PopulationRowTransform
    survivor_hypothesis_uids: tuple[str, ...]
    mutation_fence_reason: str
    mutation_fence_snapshot: EvidenceSnapshotToken
    capacity: int
    old_population: int
    new_population: int
    retire_rows: torch.Tensor
    relocate_donor_rows: torch.Tensor
    relocate_target_rows: torch.Tensor
    create_rows: torch.Tensor
    target_payload: torch.Tensor
    donor_hypothesis_uids: tuple[str, ...]
    relocated_hypothesis_uids: tuple[str, ...]
    created_hypothesis_uids: tuple[str, ...]
    reset_fields: frozenset[str]
    max_actions: int
    row_state_registry_snapshot: tuple[RowStateDescriptor, ...] = ()
    validation_boundary: str = "HOST_MUTATION_FENCE"

    REQUIRED_RESET_FIELDS = frozenset({
        "parameters", "optimizer", "belief", "lineage", "anchors",
    })

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_token, EvidenceSnapshotToken) or not isinstance(self.next_snapshot_token, EvidenceSnapshotToken):
            raise TypeError("snapshot tokens are required")
        if self.mutation_fence_reason != "MUTATION_CARDINALITY":
            raise ValueError("plan requires an attributed MUTATION_CARDINALITY fence")
        if self.mutation_fence_snapshot != self.snapshot_token:
            raise ValueError("mutation fence snapshot must exactly match plan snapshot")
        if not isinstance(self.row_transform, PopulationRowTransform):
            raise TypeError("row_transform is required")
        if not isinstance(self.row_state_registry_snapshot, tuple) or not all(isinstance(d, RowStateDescriptor) for d in self.row_state_registry_snapshot):
            raise TypeError("row_state_registry_snapshot must be a tuple of concrete descriptors")
        if len({descriptor.name for descriptor in self.row_state_registry_snapshot}) != len(self.row_state_registry_snapshot):
            raise ValueError("row_state_registry_snapshot must not contain duplicate field names")
        if self.action_count and not self.row_state_registry_snapshot:
            raise ValueError("mutation plan requires a concrete row-state registry snapshot")
        if self.validation_boundary != "HOST_MUTATION_FENCE":
            raise ValueError("plan validation boundary must be HOST_MUTATION_FENCE")
        for name in ("capacity", "old_population", "new_population", "max_actions"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.capacity < self.new_population:
            raise ValueError("new population exceeds capacity")
        if self.max_actions < 0:
            raise ValueError("max_actions must be non-negative")
        if self.row_transform.old_population != self.old_population or self.row_transform.new_population != self.new_population:
            raise ValueError("row transform population does not match mutation plan")
        if not isinstance(self.survivor_hypothesis_uids, tuple):
            raise TypeError("survivor_hypothesis_uids must be a tuple")
        if not all(isinstance(uid, str) and uid for uid in self.survivor_hypothesis_uids):
            raise ValueError("survivor hypothesis UIDs must be non-empty strings")
        if not isinstance(self.reset_fields, frozenset) or not all(isinstance(x, str) and x for x in self.reset_fields):
            raise TypeError("reset_fields must be a frozenset of names")
        if self.action_count and self.reset_fields != self.REQUIRED_RESET_FIELDS:
            raise ValueError("atomic mutation plan must declare exact reset fields")
        if not isinstance(self.target_payload, torch.Tensor) or self.target_payload.ndim != 2:
            raise ValueError("target_payload must be a rank-2 tensor")
        tensors = (self.retire_rows, self.relocate_donor_rows, self.relocate_target_rows, self.create_rows)
        for name, rows in zip(("retire_rows", "relocate_donor_rows", "relocate_target_rows", "create_rows"), tensors):
            if not isinstance(rows, torch.Tensor) or rows.ndim != 1 or rows.dtype not in _INT_DTYPES:
                raise TypeError(f"{name} must be a one-dimensional integer tensor")
            if isinstance(rows, torch.Tensor) and rows.device.type != "cpu":
                raise ValueError("plan row tensors must be CPU-resident after MUTATION_CARDINALITY fence")
        if self.target_payload.device.type != "cpu":
            raise ValueError("plan target payload must be CPU-resident after MUTATION_CARDINALITY fence")
        if len(self.donor_hypothesis_uids) != self.relocate_donor_rows.numel():
            raise ValueError("donor hypothesis UID count must match relocation pairs")
        if len(self.relocated_hypothesis_uids) != self.relocate_donor_rows.numel():
            raise ValueError("relocated hypothesis UID count must match relocation pairs")
        if len(self.created_hypothesis_uids) != self.create_rows.numel():
            raise ValueError("created hypothesis UID count must match create rows")
        if not all(isinstance(uid, str) and uid for uid in (*self.donor_hypothesis_uids, *self.relocated_hypothesis_uids, *self.created_hypothesis_uids)):
            raise ValueError("hypothesis UIDs must be non-empty strings")
        expected_payload_rows = self.relocate_target_rows.numel() + self.create_rows.numel()
        if self.target_payload.shape[0] != expected_payload_rows:
            raise ValueError("target payload rows must match relocation and create targets")
        self._validate_versions()
        self._validate_rows()
        self.row_transform.validate(survivor_hypothesis_uids=self.survivor_hypothesis_uids)
        self._validate_transform_correspondence()

    @property
    def action_count(self) -> int:
        return int(self.retire_rows.numel() + self.relocate_donor_rows.numel() + self.create_rows.numel())

    @property
    def actions(self) -> tuple[MutationAction, ...]:
        if self.action_count == 0:
            return (MutationAction.NONE,)
        return (MutationAction.RETIRE,) * int(self.retire_rows.numel()) + (MutationAction.RELOCATE,) * int(self.relocate_donor_rows.numel()) + (MutationAction.CREATE,) * int(self.create_rows.numel())

    @classmethod
    def noop(cls, *, snapshot_token: EvidenceSnapshotToken, population: int, capacity: int) -> "PopulationMutationPlan":
        return cls(snapshot_token=snapshot_token, next_snapshot_token=snapshot_token, capacity=capacity,
                   row_transform=PopulationRowTransform(old_population=population, new_population=population,
                       survivor_old_indices=torch.arange(population, dtype=torch.int64),
                       retired_old_indices=torch.empty(0, dtype=torch.int64), created_rows=(), relocated_rows=(), covered_fields=frozenset()),
                   survivor_hypothesis_uids=tuple(f"survivor-{i}" for i in range(population)),
                   mutation_fence_reason="MUTATION_CARDINALITY", mutation_fence_snapshot=snapshot_token,
                   old_population=population, new_population=population,
                   retire_rows=torch.empty(0, dtype=torch.int64), relocate_donor_rows=torch.empty(0, dtype=torch.int64),
                   relocate_target_rows=torch.empty(0, dtype=torch.int64), create_rows=torch.empty(0, dtype=torch.int64),
                   target_payload=torch.empty((0, 0)), donor_hypothesis_uids=(), relocated_hypothesis_uids=(), created_hypothesis_uids=(),
                   reset_fields=frozenset(), max_actions=0)

    def has_action(self, action: MutationAction) -> bool:
        return action in self.actions

    def validate(self, *, current_snapshot: EvidenceSnapshotToken) -> None:
        if current_snapshot != self.snapshot_token:
            raise ValueError("mutation plan snapshot is stale or incompatible")
        if self.action_count and self.reset_fields != self.REQUIRED_RESET_FIELDS:
            raise ValueError("atomic mutation plan must declare exact reset fields")

    @staticmethod
    def snapshot_registry(registry: GaussianRowStateRegistry) -> tuple[RowStateDescriptor, ...]:
        """Freeze concrete registry descriptors at the mutation boundary."""
        if not isinstance(registry, GaussianRowStateRegistry):
            raise TypeError("registry must be a GaussianRowStateRegistry")
        return tuple(registry.descriptors)

    def _validate_versions(self) -> None:
        before, after = self.snapshot_token, self.next_snapshot_token
        if self.action_count == 0:
            if after != before:
                raise ValueError("NONE plan cannot advance versions")
            return
        if after.schema != before.schema or after.coordinate_frame != before.coordinate_frame or after.surface_evidence != before.surface_evidence or after.policy != before.policy:
            raise ValueError("mutation plan changes incompatible snapshot versions")
        for name in ("map_content", "topology", "optimizer_update", "parameter"):
            if getattr(after, name) <= getattr(before, name):
                raise ValueError(f"mutation must advance {name} version")

    def _validate_rows(self) -> None:
        # Host-only validation after the single attributed
        # MUTATION_CARDINALITY fence. Device hot paths must provide bounded
        # tensors from device-side masks/reductions; these conversions are
        # never an implicit synchronization.
        groups = (self.retire_rows, self.relocate_donor_rows)
        all_donors = torch.cat(groups) if any(group.numel() for group in groups) else torch.empty(0, dtype=torch.int64)
        if all_donors.numel() and torch.unique(all_donors).numel() != all_donors.numel():
            raise ValueError("donor/retire rows must be unique")
        if all_donors.numel() and (int(all_donors.min()) < 0 or int(all_donors.max()) >= self.old_population):
            raise ValueError("donor row is outside old population")
        targets = self.relocate_target_rows
        if targets.numel() and (torch.unique(targets).numel() != targets.numel() or int(targets.min()) < 0 or int(targets.max()) >= self.new_population):
            raise ValueError("relocation target rows must be unique and in new population")
        creates = self.create_rows
        if creates.numel() and (torch.unique(creates).numel() != creates.numel() or int(creates.min()) < 0 or int(creates.max()) >= self.new_population):
            raise ValueError("create rows must be unique and in new population")
        if set(targets.tolist()).intersection(creates.tolist()):
            raise ValueError("relocation targets and create rows must be disjoint")
        uids = (*self.relocated_hypothesis_uids, *self.created_hypothesis_uids)
        if len(set(uids)) != len(uids):
            raise ValueError("new hypothesis UIDs must be unique")
        if any(old == new for old, new in zip(self.donor_hypothesis_uids, self.relocated_hypothesis_uids)):
            raise ValueError("relocation requires a fresh hypothesis UID")
        if self.action_count > self.max_actions:
            raise ValueError("mutation plan exceeds bounded action budget")
        if self.old_population == self.new_population and self.retire_rows.numel() and self.create_rows.numel():
            if self.relocate_donor_rows.numel() < min(self.retire_rows.numel(), self.create_rows.numel()):
                raise ValueError("constant-capacity retire/create must be paired as relocation")

    def _validate_transform_correspondence(self) -> None:
        transform = self.row_transform
        registry_fields = frozenset(descriptor.name for descriptor in self.row_state_registry_snapshot)
        if self.action_count and transform.covered_fields != registry_fields:
            raise ValueError("row transform must cover exactly the concrete registry snapshot")
        required = self.REQUIRED_RESET_FIELDS
        if self.action_count and required.issubset(registry_fields):
            # Compatibility/reference registries used during migration expose
            # the five coarse reset categories literally.
            if not required.issubset(transform.covered_fields):
                raise ValueError("row transform must cover all atomic reset fields")
        elif self.action_count:
            # The production gsplat registry is deliberately concrete
            # (``params.means``, ``optimizer.means.exp_avg``,
            # ``provenance.gaussian_uid``, ...).  Requiring made-up sidecar
            # names here would make the atomic plan impossible to apply.  Its
            # descriptor families must nevertheless cover every semantic reset
            # domain, including lineage/provenance and surface anchors.
            families = {descriptor.family for descriptor in self.row_state_registry_snapshot}
            required_families = {"params", "optimizer", "belief", "provenance", "surface"}
            if not required_families.issubset(families):
                missing = sorted(required_families - families)
                raise ValueError(
                    "concrete row-state registry does not cover all atomic reset domains; "
                    f"missing={missing}"
                )
        if len(self.survivor_hypothesis_uids) != transform.survivor_old_indices.numel():
            raise ValueError("survivor hypothesis UID count must match survivor rows")
        if len(set(self.survivor_hypothesis_uids)) != len(self.survivor_hypothesis_uids):
            raise ValueError("survivor hypothesis UIDs must be unique")
        retired = sorted(self.retire_rows.tolist())
        relocated_donors = sorted(self.relocate_donor_rows.tolist())
        transform_retired = sorted(transform.retired_old_indices.tolist())
        if sorted(retired + relocated_donors) != transform_retired:
            raise ValueError("plan retire/relocate donors must exactly match row transform retirees")
        transform_relocations = sorted(transform.relocated_rows, key=lambda row: row.row_index)
        planned_relocations = sorted(zip(self.relocate_target_rows.tolist(), self.relocate_donor_rows.tolist(), self.donor_hypothesis_uids, self.relocated_hypothesis_uids))
        if any(row.donor_row_index is None for row in transform_relocations):
            raise ValueError("row transform relocation must declare donor_row_index")
        actual_relocations = sorted((row.row_index, row.donor_row_index, row.old_hypothesis_uid, row.new_hypothesis_uid) for row in transform_relocations)
        if actual_relocations != planned_relocations:
            raise ValueError("plan relocation rows and identities must exactly match row transform")
        transform_creates = sorted(transform.created_rows, key=lambda row: row.row_index)
        planned_creates = sorted(zip(self.create_rows.tolist(), self.created_hypothesis_uids))
        actual_creates = sorted((row.row_index, row.hypothesis_uid) for row in transform_creates)
        if actual_creates != planned_creates:
            raise ValueError("plan create rows and identities must exactly match row transform")


__all__ = ["MutationAction", "PopulationMutationPlan"]
