"""Typed structural mutation boundary for the live Gaussian map.

All ordinary production writes must enter through one transaction. The
transaction distinguishes map-content changes from topology and coordinate-
frame changes, and pose corrections update every supplied world-frame anchor
atomically with the Gaussian means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol

import torch

if TYPE_CHECKING:
    from scene.gsplat_model import GsplatGaussianModel
    from eligsir.map.population_mutation_plan import PopulationMutationPlan
    from eligsir.map.population_target_payload import PopulationTargetPayloadBatch
    from eligsir.map.row_state import RowStateDescriptor


def _validate_mask(
    mask: "torch.Tensor",
    population: int,
    name: str = "mask",
    *,
    device: Optional["torch.device"] = None,
) -> None:
    """Validate structural mask facts without extracting a device scalar."""
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if device is not None and mask.device != device:
        raise ValueError(f"{name} must be on model device {device}, got {mask.device}")
    if mask.dtype == torch.bool:
        if mask.numel() != population:
            raise ValueError(
                f"boolean {name} length {mask.numel()} != population {population}"
            )
    elif mask.dtype not in {
        torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64
    }:
        raise TypeError(f"{name} must be boolean or integer indices")


def _validate_insertion_batch(batch: "GaussianInsertionBatch") -> None:
    """Validate every aligned insertion field before model mutation begins."""
    if not isinstance(batch.points, torch.Tensor) or batch.points.ndim != 2 or batch.points.shape[1] != 3:
        raise ValueError("points must be a (N, 3) tensor")
    count = batch.points.shape[0]
    if not isinstance(batch.colors, torch.Tensor) or batch.colors.shape != (count, 3):
        raise ValueError(f"colors must have shape ({count}, 3)")
    if batch.colors.device != batch.points.device:
        raise ValueError("colors and points must be on the same device")
    exact_shapes = {
        "normals": (count, 3),
        "scales": (count, 3),
        "rotations": (count, 4),
    }
    for name, shape in exact_shapes.items():
        value = getattr(batch, name)
        if value is not None and (
            not isinstance(value, torch.Tensor) or value.shape != shape
        ):
            raise ValueError(f"{name} must have shape {shape}")
    for name in ("opacities_raw", "sh_rest", "confidence", "source"):
        value = getattr(batch, name)
        if value is not None and (
            not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != count
        ):
            raise ValueError(f"{name} must have {count} aligned rows")
    for name in (
        "normals", "scales", "rotations", "opacities_raw", "sh_rest",
        "confidence", "source",
    ):
        value = getattr(batch, name)
        if value is not None and value.device != batch.points.device:
            raise ValueError(
                f"{name} must be on points device {batch.points.device}, got {value.device}"
            )


def mutation_transaction_for_model(
    gaussians,
    map_version: int = 0,
    topology_version: int = 0,
    coordinate_frame_version: int = 0,
    schema_version: int = 0,
) -> "GaussianMutationTransaction":
    """Construct the layout-appropriate transaction for a Gaussian model."""
    if hasattr(gaussians, "params"):
        return GsplatMutationTransaction(
            gaussians,
            map_version,
            topology_version,
            coordinate_frame_version,
            schema_version,
        )
    from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMutationTransaction

    return LegacyGraphdecoMutationTransaction(
        gaussians,
        map_version,
        topology_version,
        coordinate_frame_version,
        schema_version,
    )


@dataclass(frozen=True)
class GaussianInsertionBatch:
    """Device-native insertion bundle consumed by the sole mutation owner."""

    points: "torch.Tensor"
    colors: "torch.Tensor"
    init_scale: float = 0.01
    init_opacity: float = 0.5
    use_knn_scale: bool = False
    normals: Optional["torch.Tensor"] = None
    scales: Optional["torch.Tensor"] = None
    rotations: Optional["torch.Tensor"] = None
    is_provisional: bool = False
    birth_frame: int = 0
    birth_iter: int = 0
    opacities_raw: Optional["torch.Tensor"] = None
    sh_rest: Optional["torch.Tensor"] = None
    confidence: Optional["torch.Tensor"] = None
    source: Optional["torch.Tensor"] = None


@dataclass(frozen=True)
class RelocationPlan:
    """In-place parameter relocation preserving Gaussian identity/order."""

    mask: "torch.Tensor"
    new_means: Optional["torch.Tensor"] = None
    new_raw_scales: Optional["torch.Tensor"] = None
    new_raw_opacities: Optional["torch.Tensor"] = None
    new_quaternions: Optional["torch.Tensor"] = None


@dataclass(frozen=True)
class SubmapCorrection:
    """Atomic world-coordinate correction for a selected Gaussian set.

    ``new_anchor_xyz`` and ``new_anchor_normal`` are explicit rather than
    inferred inside the transaction because non-rigid corrections and rigid
    SE(3) corrections have different rules. A caller correcting a model that
    owns populated anchor buffers must supply their corrected values; the
    transaction fails closed rather than leaving them stale.
    """

    mask: "torch.Tensor"
    new_positions: "torch.Tensor"
    new_anchor_xyz: Optional["torch.Tensor"] = None
    new_anchor_normal: Optional["torch.Tensor"] = None


@dataclass(frozen=True)
class TopologyChange:
    """Version/result record returned by a committed mutation transaction."""

    map_version: int
    topology_version: int
    coordinate_frame_version: int = 0
    schema_version: int = 0
    inserted: int = 0
    pruned: int = 0
    relocated: int = 0
    corrected: int = 0
    content_changed: bool = False

    @property
    def map_content_version(self) -> int:
        """Explicit name for the legacy ``map_version`` field."""
        return self.map_version


class GaussianMutationTransaction(Protocol):
    """Mutation transaction interface implemented by each storage layout."""

    def insert(self, batch: GaussianInsertionBatch) -> int:
        """Append a device-native insertion batch and return rows added."""
        ...

    def prune(self, mask: "torch.Tensor") -> int:
        """Remove selected rows and return the shape-derived row-count delta."""
        ...

    def relocate(self, plan: RelocationPlan) -> None:
        """Apply identity-preserving in-place parameter relocation."""
        ...

    def apply_pose_correction(self, correction: SubmapCorrection) -> None:
        """Apply an atomic coordinate correction to means and anchors."""
        ...

    def update_opacity_logits(self, mask: "torch.Tensor", value) -> None:
        """Update raw opacity logits for selected rows."""
        ...

    def commit(self) -> TopologyChange:
        """Close the transaction and return version/counter effects."""
        ...


class PopulationPlanMutationTransaction(Protocol):
    """Optional one-call capability for the #1365 atomic plan migration.

    This is intentionally separate from :class:`GaussianMutationTransaction`:
    legacy layout transactions remain supported while the real row-state
    backend migration is incomplete.  The arbiter requires this capability
    for a ``PopulationMutationProposal`` and otherwise fails closed.
    """

    def apply_population_plan(
        self,
        plan: "PopulationMutationPlan",
        target_payload: "PopulationTargetPayloadBatch",
        row_state_registry_snapshot: tuple["RowStateDescriptor", ...],
    ) -> None:
        """Apply exactly one complete population plan without decomposition."""
        ...

    def validate_population_plan(
        self,
        plan: "PopulationMutationPlan",
        target_payload: "PopulationTargetPayloadBatch",
        row_state_registry_snapshot: tuple["RowStateDescriptor", ...],
    ) -> None:
        """Validate the exact plan against live backend storage before apply."""
        ...

    def current_population_plan_registry_snapshot(
        self,
    ) -> tuple["RowStateDescriptor", ...]:
        """Attest the concrete live row-state registry inventory."""
        ...


def _anchor_buffer_is_populated(model, name: str, population: int) -> bool:
    """Return whether an aligned anchor buffer exists for the whole population."""
    value = getattr(model, name, None)
    return value is not None and getattr(value, "shape", (0,))[0] == population and population > 0


class GsplatMutationTransaction:
    """Sole raw-write owner for :class:`GsplatGaussianModel`."""

    def __init__(
        self,
        model: "GsplatGaussianModel",
        map_version: int,
        topology_version: int,
        coordinate_frame_version: int = 0,
        schema_version: int = 0,
        population_hypothesis_uids: tuple[str, ...] | None = None,
        population_next_hypothesis_uid: int | None = None,
    ):
        """Bind model storage and the versions this transaction starts from."""
        self._model = model
        self._map_version = int(map_version)
        self._topology_version = int(topology_version)
        self._coordinate_frame_version = int(coordinate_frame_version)
        self._schema_version = int(schema_version)
        self._inserted = 0
        self._pruned = 0
        self._relocated = 0
        self._corrected = 0
        self._content_dirty = False
        self._topology_dirty = False
        self._coordinate_dirty = False
        self._committed = False
        self._population_plan_next_snapshot = None
        self._population_hypothesis_uids = population_hypothesis_uids
        self._population_next_hypothesis_uid = population_next_hypothesis_uid
        self._population_plan_next_identity_snapshot = None

    def _ensure_open(self) -> None:
        """Fail closed if a caller tries to reuse a committed transaction."""
        if self._committed:
            raise RuntimeError("mutation transaction has already been committed")

    def _reject_legacy_population_writer(self, writer: str) -> None:
        """Keep live population mutations inside one complete plan transform."""

        if getattr(self._model, "_population_plan_ownership_enabled", False):
            raise RuntimeError(
                "live population ownership requires PopulationMutationPlan; "
                f"legacy transaction writer {writer} is unavailable"
            )

    def insert(self, batch: GaussianInsertionBatch) -> int:
        """Append ``batch`` through the model's aligned insertion primitive."""
        self._ensure_open()
        self._reject_legacy_population_writer("insert")
        added = int(
            self._model.add_points_as_gaussians(
                points=batch.points,
                colors=batch.colors,
                init_scale=batch.init_scale,
                init_opacity=batch.init_opacity,
                use_knn_scale=batch.use_knn_scale,
                normals=batch.normals,
                scales=batch.scales,
                rotations=batch.rotations,
                is_provisional=batch.is_provisional,
                birth_frame=batch.birth_frame,
                birth_iter=batch.birth_iter,
                opacities_raw=batch.opacities_raw,
                sh_rest=batch.sh_rest,
                confidence=batch.confidence,
                source=batch.source,
            )
        )
        self._inserted += added
        if added > 0:
            self._content_dirty = True
            self._topology_dirty = True
        return added

    def validate_insert(self, batch: GaussianInsertionBatch) -> None:
        _validate_insertion_batch(batch)

    def validate_prune(self, mask: "torch.Tensor") -> None:
        means = self._model.params["means"]
        _validate_mask(mask, int(means.shape[0]), device=means.device)

    def validate_relocate(self, plan: RelocationPlan) -> None:
        population = int(self._model.params["means"].shape[0])
        means = self._model.params["means"]
        _validate_mask(plan.mask, population, "relocation mask", device=means.device)
        updates = (
            ("new_means", "means"),
            ("new_raw_scales", "scales"),
            ("new_raw_opacities", "opacities"),
            ("new_quaternions", "quats"),
        )
        row_counts = set()
        for name, parameter_name in updates:
            value = getattr(plan, name)
            if value is None:
                continue
            target = self._model.params[parameter_name]
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                raise ValueError(f"{name} must be a row-aligned tensor")
            if value.device != target.device or value.shape[1:] != target.shape[1:]:
                raise ValueError(f"{name} must match {parameter_name} device and row shape")
            row_counts.add(value.shape[0])
        if len(row_counts) > 1:
            raise ValueError("relocation update tensors must have equal row counts")
        if plan.mask.dtype != torch.bool and row_counts and row_counts != {plan.mask.numel()}:
            raise ValueError("relocation update rows must match index count")

    def validate_pose_correction(self, correction: SubmapCorrection) -> None:
        population = int(self._model.params["means"].shape[0])
        means = self._model.params["means"]
        _validate_mask(
            correction.mask, population, "pose correction mask", device=means.device
        )
        if not isinstance(correction.new_positions, torch.Tensor) or correction.new_positions.ndim != 2:
            raise ValueError("new_positions must be a row-aligned tensor")
        if (
            correction.new_positions.device != means.device
            or correction.new_positions.shape[1:] != means.shape[1:]
        ):
            raise ValueError("new_positions must match means device and row shape")
        if (
            correction.mask.dtype != torch.bool
            and correction.new_positions.shape[0] != correction.mask.numel()
        ):
            raise ValueError("new_positions rows must match pose correction index count")
        has_anchor_xyz = _anchor_buffer_is_populated(self._model, "anchor_xyz", population)
        has_anchor_normal = _anchor_buffer_is_populated(self._model, "anchor_normal", population)
        if has_anchor_xyz and correction.new_anchor_xyz is None:
            raise ValueError("pose correction must update populated anchor_xyz atomically")
        if has_anchor_normal and correction.new_anchor_normal is None:
            raise ValueError("pose correction must update populated anchor_normal atomically")
        for name, target in (
            ("new_anchor_xyz", getattr(self._model, "anchor_xyz", None)),
            ("new_anchor_normal", getattr(self._model, "anchor_normal", None)),
        ):
            value = getattr(correction, name)
            if value is None:
                continue
            if (
                not isinstance(value, torch.Tensor)
                or value.device != means.device
                or value.shape[0] != correction.new_positions.shape[0]
                or target is None
                or value.shape[1:] != target.shape[1:]
            ):
                raise ValueError(f"{name} must match selected anchor rows and device")

    def validate_opacity_update(self, mask: "torch.Tensor", value) -> None:
        means = self._model.params["means"]
        _validate_mask(
            mask, int(means.shape[0]), "opacity mask", device=means.device
        )
        target = self._model.params["opacities"]
        if not isinstance(value, torch.Tensor) or value.device != target.device:
            raise ValueError("opacity value must be a tensor on the model device")
        if value.ndim == 0 or value.shape[1:] != target.shape[1:]:
            raise ValueError("opacity value must match raw opacity row shape")
        if mask.dtype != torch.bool and value.shape[0] != mask.numel():
            raise ValueError("opacity value rows must match index count")

    def current_population_plan_registry_snapshot(self):
        """Attest the live, complete gsplat row-state inventory.

        The returned descriptors are built from the model immediately at the
        mutation fence.  A proposal's copied tuple is never accepted as proof
        of live storage coverage.
        """

        from eligsir.map.population_mutation_plan import PopulationMutationPlan

        registry = self._model.build_row_state_registry()
        return PopulationMutationPlan.snapshot_registry(registry)

    def validate_population_plan(self, plan, target_payload, row_state_registry_snapshot) -> None:
        """Preflight an atomic population plan without mutating any row state."""

        from eligsir.map.population_mutation_plan import PopulationMutationPlan
        from eligsir.map.population_target_payload import PopulationTargetPayloadBatch
        from eligsir.map.row_state import GaussianRowStateRegistry

        self._ensure_open()
        if not isinstance(plan, PopulationMutationPlan):
            raise TypeError("plan must be a PopulationMutationPlan")
        if not isinstance(target_payload, PopulationTargetPayloadBatch):
            raise TypeError("target_payload must be a PopulationTargetPayloadBatch")
        if not isinstance(row_state_registry_snapshot, tuple):
            raise TypeError("row_state_registry_snapshot must be a tuple")
        registry = self._model.build_row_state_registry()
        if not isinstance(registry, GaussianRowStateRegistry):
            raise TypeError("gsplat model did not provide a GaussianRowStateRegistry")
        authoritative = PopulationMutationPlan.snapshot_registry(registry)
        if row_state_registry_snapshot != authoritative:
            raise ValueError("population-plan registry snapshot does not match live gsplat storage")
        if plan.row_state_registry_snapshot != authoritative:
            raise ValueError("plan registry snapshot does not match live gsplat storage")
        population = int(self._model.params["means"].shape[0])
        if plan.old_population != population:
            raise ValueError("population mutation plan old population does not match gsplat storage")
        registry.validate_transform(
            plan.row_transform,
            survivor_hypothesis_uids=plan.survivor_hypothesis_uids,
        )
        if plan.target_payload.shape != target_payload.positions.shape or not torch.equal(
            plan.target_payload, target_payload.positions
        ):
            raise ValueError("population plan target payload does not match typed target batch")
        if target_payload.count != int(plan.relocate_target_rows.numel() + plan.create_rows.numel()):
            raise ValueError("typed target batch count does not match population plan targets")
        _validate_live_population_plan_identities(
            self._model,
            plan,
            self._population_hypothesis_uids,
            self._population_next_hypothesis_uid,
        )
        target_payload.validate_debug()
        _validate_population_payload_shapes(self._model, target_payload)

    def apply_population_plan(self, plan, target_payload, row_state_registry_snapshot) -> None:
        """Apply exactly one complete row transform without legacy composition.

        All replacements are materialized before the first model assignment.
        The only failure-recovery path restores the exact parameter, optimizer,
        buffer, lineage, anchor, and belief references captured at entry.
        """

        self.validate_population_plan(plan, target_payload, row_state_registry_snapshot)
        registry = self._model.build_row_state_registry()
        staged = _stage_population_row_transform(self._model, registry, plan, target_payload)
        next_identity_snapshot = _population_identity_after_plan(
            plan,
            self._population_hypothesis_uids,
            self._population_next_hypothesis_uid,
        )
        _install_population_row_transform(self._model, staged)
        self._inserted += int(plan.create_rows.numel())
        self._pruned += int(plan.retire_rows.numel())
        self._relocated += int(plan.relocate_donor_rows.numel())
        if plan.action_count:
            self._content_dirty = True
            self._topology_dirty = True
        self._population_plan_next_snapshot = plan.next_snapshot_token
        self._population_plan_next_identity_snapshot = next_identity_snapshot
        self._population_plan_expected_population = plan.new_population
        self._population_plan_registry_snapshot = row_state_registry_snapshot

    def prune(self, mask: "torch.Tensor") -> int:
        """Prune selected rows and mark content/topology dirty."""
        self._ensure_open()
        self._reject_legacy_population_writer("prune")
        before = int(self._model.params["means"].shape[0])
        if mask.numel() == 0:
            return 0
        self._model.prune_points(mask)
        count = before - int(self._model.params["means"].shape[0])
        if count == 0:
            return 0
        self._pruned += count
        self._content_dirty = True
        self._topology_dirty = True
        return count

    def relocate(self, plan: RelocationPlan) -> None:
        """Apply same-row relocation and mark content, not topology, dirty."""
        self._ensure_open()
        self._reject_legacy_population_writer("relocate")
        updates = (
            plan.new_means, plan.new_raw_scales, plan.new_raw_opacities,
            plan.new_quaternions,
        )
        first_update = next((value for value in updates if value is not None), None)
        if first_update is None:
            return
        count = (
            int(plan.mask.numel())
            if plan.mask.dtype != torch.bool
            else int(first_update.shape[0])
        )
        if count == 0:
            return
        named_updates = (
            ("means", plan.new_means), ("scales", plan.new_raw_scales),
            ("opacities", plan.new_raw_opacities), ("quats", plan.new_quaternions),
        )
        originals = {
            name: self._model.params[name].data[plan.mask].clone()
            for name, value in named_updates if value is not None
        }
        wrote = False
        try:
            for name, value in named_updates:
                if value is not None:
                    self._model.params[name].data[plan.mask] = value
                    wrote = True
        except Exception:
            for name, original in originals.items():
                self._model.params[name].data[plan.mask] = original
            raise
        if wrote:
            self._relocated += count
            self._content_dirty = True

    def apply_pose_correction(self, correction: SubmapCorrection) -> None:
        """Move means and every populated world-frame anchor atomically."""
        self._ensure_open()
        count = int(correction.new_positions.shape[0])
        if count == 0:
            return
        population = int(self._model.params["means"].shape[0])
        has_anchor_xyz = _anchor_buffer_is_populated(self._model, "anchor_xyz", population)
        has_anchor_normal = _anchor_buffer_is_populated(self._model, "anchor_normal", population)
        if has_anchor_xyz and correction.new_anchor_xyz is None:
            raise ValueError("pose correction must update populated anchor_xyz atomically")
        if has_anchor_normal and correction.new_anchor_normal is None:
            raise ValueError("pose correction must update populated anchor_normal atomically")

        original_means = self._model.params["means"].data[correction.mask].clone()
        original_xyz = self._model.anchor_xyz[correction.mask].clone() if has_anchor_xyz else None
        original_normal = self._model.anchor_normal[correction.mask].clone() if has_anchor_normal else None
        try:
            self._model.params["means"].data[correction.mask] = correction.new_positions
            if has_anchor_xyz:
                self._model.anchor_xyz[correction.mask] = correction.new_anchor_xyz
            if has_anchor_normal:
                self._model.anchor_normal[correction.mask] = correction.new_anchor_normal
        except Exception:
            self._model.params["means"].data[correction.mask] = original_means
            if has_anchor_xyz:
                self._model.anchor_xyz[correction.mask] = original_xyz
            if has_anchor_normal:
                self._model.anchor_normal[correction.mask] = original_normal
            raise
        self._corrected += count
        self._content_dirty = True
        self._coordinate_dirty = True

    def update_opacity_logits(self, mask: "torch.Tensor", value) -> None:
        """Apply a same-topology raw opacity update to selected rows."""
        self._ensure_open()
        if mask.numel() == 0:
            return
        self._model.params["opacities"].data[mask] = value
        self._content_dirty = True

    def commit(self) -> TopologyChange:
        """Close the transaction and derive independent version increments."""
        self._ensure_open()
        self._committed = True
        next_snapshot = self._population_plan_next_snapshot
        return TopologyChange(
            map_version=(
                next_snapshot.map_content
                if next_snapshot is not None
                else self._map_version + int(self._content_dirty)
            ),
            topology_version=(
                next_snapshot.topology
                if next_snapshot is not None
                else self._topology_version + int(self._topology_dirty)
            ),
            coordinate_frame_version=(
                next_snapshot.coordinate_frame
                if next_snapshot is not None
                else self._coordinate_frame_version + int(self._coordinate_dirty)
            ),
            schema_version=(
                next_snapshot.schema if next_snapshot is not None else self._schema_version
            ),
            inserted=self._inserted,
            pruned=self._pruned,
            relocated=self._relocated,
            corrected=self._corrected,
            content_changed=self._content_dirty,
        )


def _validate_population_payload_shapes(model, payload) -> None:
    """Reject target tensors that cannot initialize the live gsplat layout."""

    params = model.params
    expected = {
        "means": payload.positions,
        "scales": payload.raw_scales,
        "quats": payload.quaternions,
        "opacities": payload.opacity_logits.reshape(-1),
        "shN": payload.sh_rest,
    }
    for name, value in expected.items():
        target = params[name]
        if value.shape[0] != payload.count or tuple(value.shape[1:]) != tuple(target.shape[1:]):
            raise ValueError(f"population target {name} does not match live gsplat row shape")
        if value.dtype != target.dtype:
            raise TypeError(f"population target {name} dtype does not match live gsplat parameter")
    sh0 = params["sh0"]
    if tuple(sh0.shape[1:]) != (1, 3) or payload.colors.dtype != sh0.dtype:
        raise ValueError("population target colors do not match live gsplat SH0 layout")


def _stage_population_row_transform(model, registry, plan, payload):
    """Build a complete replacement set before any live-state assignment."""

    from eligsir.utils.sh_utils import RGB2SH

    survivors = plan.row_transform.survivor_old_indices.to(
        device=model.params["means"].device, dtype=torch.long
    )
    count = payload.count
    device = model.params["means"].device
    target_rows = _target_payload_rows(model, payload)
    minted_uid_values = tuple(
        _parse_gsplat_uid(uid)
        for uid in (*plan.relocated_hypothesis_uids, *plan.created_hypothesis_uids)
    )
    minted_uids = torch.tensor(
        minted_uid_values,
        dtype=getattr(model, "gaussian_uid").dtype,
        device=device,
    )
    target_output_rows = torch.cat(
        (plan.relocate_target_rows, plan.create_rows)
    ).to(device=device, dtype=torch.long)
    tail_by_name = {
        "params.means": target_rows["means"],
        "params.scales": target_rows["scales"],
        "params.quats": target_rows["quats"],
        "params.opacities": target_rows["opacities"],
        "params.sh0": RGB2SH(payload.colors.to(device=device, dtype=model.params["sh0"].dtype)).unsqueeze(1),
        "params.shN": target_rows["shN"],
    }
    replacements = {}
    for descriptor in registry.descriptors:
        name = descriptor.name
        if name.startswith("params."):
            old = model.params[name.removeprefix("params.")]
        elif name.startswith("optimizer."):
            _, param_name, state_name = name.split(".", 2)
            old = model.optimizers[param_name].state[model.params[param_name]][state_name]
        else:
            _, attribute = name.split(".", 1)
            old = getattr(model, attribute)
        if name.startswith("optimizer."):
            tail = torch.zeros((count,) + tuple(old.shape[1:]), dtype=old.dtype, device=old.device)
        elif name in tail_by_name:
            tail = tail_by_name[name]
        else:
            tail = _fresh_state_rows(
                descriptor,
                old,
                count=count,
                target_rows=target_rows,
                minted_uids=minted_uids,
            )
        replacements[name] = _scatter_population_rows(
            old.detach(),
            survivors=survivors,
            target_output_rows=target_output_rows,
            target_values=tail.detach(),
            new_population=plan.new_population,
        )
    return {
        "replacements": replacements,
        "next_gaussian_uid": max(
            [
                int(getattr(model, "_next_gaussian_uid", 0)),
                *(uid + 1 for uid in minted_uid_values),
            ]
        ),
    }


def _parse_gsplat_uid(value) -> int:
    """Decode the gsplat lineage representation of a plan identity exactly."""

    if not isinstance(value, str) or not value or not value.isdecimal():
        raise ValueError("gsplat population-plan hypothesis identities must be decimal UID strings")
    parsed = int(value)
    if str(parsed) != value:
        raise ValueError("gsplat population-plan hypothesis identities must be canonical decimal UIDs")
    return parsed


def _validate_live_population_plan_identities(
    model,
    plan,
    hypothesis_uids: tuple[str, ...] | None,
    next_hypothesis_uid: int | None,
) -> None:
    """Attest plan identities against the backend-owned host ledger."""

    population = int(model.params["means"].shape[0])
    live_values = _validate_host_population_identity_snapshot(
        hypothesis_uids, next_hypothesis_uid, population=population,
    )
    survivor_rows = plan.row_transform.survivor_old_indices.tolist()
    donor_rows = plan.relocate_donor_rows.tolist()
    actual_survivors = tuple(hypothesis_uids[row] for row in survivor_rows)
    actual_donors = tuple(hypothesis_uids[row] for row in donor_rows)
    if actual_survivors != plan.survivor_hypothesis_uids:
        raise ValueError("survivor hypothesis identity inventory does not match live gsplat rows")
    if actual_donors != plan.donor_hypothesis_uids:
        raise ValueError("donor hypothesis identity inventory does not match live gsplat rows")
    minted = tuple(
        _parse_gsplat_uid(uid)
        for uid in (*plan.relocated_hypothesis_uids, *plan.created_hypothesis_uids)
    )
    if len(set(minted)) != len(minted):
        raise ValueError("gsplat population-plan fresh hypothesis identities must be unique")
    if set(minted).intersection(live_values):
        raise ValueError("gsplat population-plan fresh hypothesis identity collides with live inventory")


def _validate_host_population_identity_snapshot(
    hypothesis_uids: tuple[str, ...] | None,
    next_hypothesis_uid: int | None,
    *,
    population: int,
) -> tuple[int, ...]:
    if (
        not isinstance(hypothesis_uids, tuple)
        or len(hypothesis_uids) != population
        or type(next_hypothesis_uid) is not int
        or next_hypothesis_uid < 0
        or any(
            not isinstance(uid, str)
            or not uid
            or not uid.isdecimal()
            or str(int(uid)) != uid
            for uid in hypothesis_uids
        )
    ):
        raise ValueError("backend population hypothesis identity ledger is unavailable or malformed")
    parsed = tuple(int(uid) for uid in hypothesis_uids)
    if len(set(parsed)) != len(parsed) or (parsed and next_hypothesis_uid <= max(parsed)):
        raise ValueError("backend population hypothesis identity ledger is not unique and monotonic")
    return parsed


def _population_identity_after_plan(
    plan,
    hypothesis_uids: tuple[str, ...] | None,
    next_hypothesis_uid: int | None,
) -> tuple[tuple[str, ...], int]:
    live_values = _validate_host_population_identity_snapshot(
        hypothesis_uids, next_hypothesis_uid, population=plan.old_population,
    )
    result: list[str | None] = [None] * plan.new_population
    survivor_rows = plan.row_transform.survivor_old_indices.tolist()
    for output_row, old_row in enumerate(survivor_rows):
        result[output_row] = hypothesis_uids[old_row]
    for relocation in plan.row_transform.relocated_rows:
        result[relocation.row_index] = str(relocation.new_hypothesis_uid)
    for creation in plan.row_transform.created_rows:
        result[creation.row_index] = str(creation.hypothesis_uid)
    if any(uid is None for uid in result):
        raise ValueError("population row transform did not produce a complete identity ledger")
    resolved = tuple(result)
    parsed = tuple(_parse_gsplat_uid(uid) for uid in resolved)
    if len(set(parsed)) != len(parsed):
        raise ValueError("population row transform produced duplicate hypothesis identities")
    return resolved, max([next_hypothesis_uid, *(uid + 1 for uid in parsed)])


def _scatter_population_rows(
    old: torch.Tensor,
    *,
    survivors: torch.Tensor,
    target_output_rows: torch.Tensor,
    target_values: torch.Tensor,
    new_population: int,
) -> torch.Tensor:
    """Build exact output order; payload order never implies destination order."""

    result = torch.empty(
        (new_population,) + tuple(old.shape[1:]), dtype=old.dtype, device=old.device
    )
    survivor_count = int(survivors.numel())
    if survivor_count:
        result[:survivor_count] = old[survivors]
    if target_values.shape[0]:
        result[target_output_rows] = target_values
    return result


def _target_payload_rows(model, payload):
    """Move the bounded host-fence payload once into the model's device layout."""

    params = model.params
    return {
        "means": payload.positions.to(device=params["means"].device, dtype=params["means"].dtype),
        "scales": payload.raw_scales.to(device=params["scales"].device, dtype=params["scales"].dtype),
        "quats": payload.quaternions.to(device=params["quats"].device, dtype=params["quats"].dtype),
        "opacities": payload.opacity_logits.to(device=params["opacities"].device, dtype=params["opacities"].dtype).reshape(-1),
        "shN": payload.sh_rest.to(device=params["shN"].device, dtype=params["shN"].dtype),
    }


def _fresh_state_rows(descriptor, old, *, count, target_rows, minted_uids):
    """Initialize a fresh latent hypothesis; never carry donor evidence over."""

    shape = (count,) + tuple(old.shape[1:])
    name = descriptor.name.rsplit(".", 1)[-1]
    if name in {"anchor_xyz", "birth_anchor_xyz"}:
        return target_rows["means"].to(dtype=old.dtype)
    if name == "anchor_scale_log":
        return target_rows["scales"].to(dtype=old.dtype)
    if name == "anchor_opacity_logit":
        return target_rows["opacities"].to(dtype=old.dtype)
    if name in {"gaussian_uid", "root_uid"}:
        return minted_uids.to(dtype=old.dtype)
    if name == "parent_uid":
        return torch.full(shape, -1, dtype=old.dtype, device=old.device)
    if name == "anchor_normal":
        value = torch.zeros(shape, dtype=old.dtype, device=old.device)
        if count:
            value[..., 2] = 1
        return value
    if name in {"p_exist"}:
        return torch.ones(shape, dtype=old.dtype, device=old.device)
    if descriptor.default == "sentinel_minus_one" or name in {"last_reseed_iter"}:
        return torch.full(shape, -1, dtype=old.dtype, device=old.device)
    if descriptor.default == "nan_vector":
        return torch.full(shape, float("nan"), dtype=old.dtype, device=old.device)
    return torch.zeros(shape, dtype=old.dtype, device=old.device)


def _install_population_row_transform(model, staged) -> None:
    """Install prebuilt replacements and restore exact references on failure."""

    replacements = staged["replacements"]
    params_before = {name: param for name, param in model.params.items()}
    optimizer_before = {}
    buffers_before = {}
    next_uid_before = getattr(model, "_next_gaussian_uid", None)
    for name, old_param in params_before.items():
        optimizer = model.optimizers[name]
        group = optimizer.param_groups[0]
        optimizer_before[name] = (group, old_param, optimizer.state.get(old_param))
    for name in replacements:
        if name.startswith("params.") or name.startswith("optimizer."):
            continue
        _, attribute = name.split(".", 1)
        buffers_before[attribute] = getattr(model, attribute)
    installed_params = {}
    try:
        for name, old_param in params_before.items():
            replacement = replacements[f"params.{name}"]
            new_param = torch.nn.Parameter(replacement.detach().requires_grad_(True))
            optimizer = model.optimizers[name]
            group, _, old_state = optimizer_before[name]
            new_state = {}
            if old_state is not None:
                for key, value in old_state.items():
                    descriptor_name = f"optimizer.{name}.{key}"
                    new_state[key] = replacements[descriptor_name] if descriptor_name in replacements else value
                del optimizer.state[old_param]
                optimizer.state[new_param] = new_state
            group["params"][0] = new_param
            model.params[name] = new_param
            installed_params[name] = new_param
        for descriptor_name, replacement in replacements.items():
            if descriptor_name.startswith("params.") or descriptor_name.startswith("optimizer."):
                continue
            _, attribute = descriptor_name.split(".", 1)
            setattr(model, attribute, replacement)
        model._next_gaussian_uid = staged["next_gaussian_uid"]
        model.attest_live_population_rows()
    except Exception:
        for name, new_param in installed_params.items():
            optimizer = model.optimizers[name]
            group, old_param, old_state = optimizer_before[name]
            group["params"][0] = old_param
            model.params[name] = old_param
            optimizer.state.pop(new_param, None)
            if old_state is not None:
                optimizer.state[old_param] = old_state
        for attribute, original in buffers_before.items():
            setattr(model, attribute, original)
        if next_uid_before is not None:
            model._next_gaussian_uid = next_uid_before
        raise
