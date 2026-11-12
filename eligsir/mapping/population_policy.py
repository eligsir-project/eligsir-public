"""Shadow, bounded Gaussian population-control policy.

The policy is deliberately read-only.  It turns a device-resident
``GaussianBelief`` snapshot into action codes and optional target indices;
the scheduler owns work tokens and frame selection, while the mutation arbiter
owns all writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import math
from numbers import Real
from typing import Optional

import torch

from eligsir.utils.gaussian_belief import GaussianBelief
from eligsir.mapping.evidence.evidence_demand import (
    EvidenceDemand,
    EvidenceDemandBatch,
    EvidenceDemandKind,
    EvidenceDemandModality,
)
from eligsir.mapping.evidence.surface_targets import SurfaceTargetSnapshot
from eligsir.mapping.population_control_contracts import MutationWorkToken


_POLICY_SELECTION_CAPABILITY = object()


class PopulationAction(IntEnum):
    NONE = 0
    ABSTAIN = 1
    CORRECT = 2
    EXPLORE = 3
    CREATE = 4
    RELOCATE = 5
    RETIRE = 6

    @property
    def code(self) -> int:
        return int(self)


@dataclass
class PopulationEvidence:
    """E0/E1 policy inputs, all shaped ``[N]`` on one device."""

    target_gain: torch.Tensor
    depth_residual: torch.Tensor
    free_space_conflict: torch.Tensor
    independent_view_count: torch.Tensor | None = None
    target_conflict: torch.Tensor | None = None

    @classmethod
    def empty(cls, n: int, *, device: str | torch.device = "cpu") -> "PopulationEvidence":
        zeros = torch.zeros(n, device=device, dtype=torch.float32)
        return cls(zeros, zeros.clone(), zeros.clone(), None)

    def validate_finite(self) -> None:
        """Explicit construction/debug-boundary validation.

        This is intentionally not called from every policy epoch: reductions
        used to check tensor contents can introduce the very host/device sync
        the shadow contract is designed to avoid.
        """
        for name, value in (
            ("target_gain", self.target_gain),
            ("depth_residual", self.depth_residual),
            ("free_space_conflict", self.free_space_conflict),
        ):
            if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                raise TypeError(f"{name} must use a floating-point dtype")
            torch._assert(torch.isfinite(value).all(), f"{name} contains non-finite values")
        if self.target_conflict is not None:
            if (
                not isinstance(self.target_conflict, torch.Tensor)
                or not self.target_conflict.dtype.is_floating_point
            ):
                raise TypeError("target_conflict must use a floating-point dtype")
            torch._assert(torch.isfinite(self.target_conflict).all(), "target_conflict contains non-finite values")
        if self.independent_view_count is not None:
            if (
                not isinstance(self.independent_view_count, torch.Tensor)
                or self.independent_view_count.dtype.is_floating_point
            ):
                raise TypeError("independent_view_count must use an integer dtype")


@dataclass(frozen=True)
class PopulationDecision:
    """Device-resident policy output; no model state is mutated."""

    actions: torch.Tensor
    target_indices: torch.Tensor
    evidence_demands: EvidenceDemandBatch
    demand_masks: tuple[torch.Tensor, ...]
    structural_selection: "PopulationStructuralSelection | None" = None
    _structural_attestation: "_StructuralSelectionAttestation | None" = field(
        default=None, repr=False, compare=False,
    )

    @classmethod
    def policy_output(
        cls,
        *,
        actions: torch.Tensor,
        target_indices: torch.Tensor,
        evidence_demands: EvidenceDemandBatch,
        demand_masks: tuple[torch.Tensor, ...],
        structural_selection: "PopulationStructuralSelection",
    ) -> "PopulationDecision":
        """Construct the only decision shape accepted by the mutation fence."""

        if (
            not structural_selection._has_policy_capability()
            or structural_selection._policy_actions is not actions
            or structural_selection._policy_target_indices is not target_indices
        ):
            raise ValueError("structural selection was not produced by GaussianPopulationPolicy")
        return cls(
            actions=actions, target_indices=target_indices,
            evidence_demands=evidence_demands, demand_masks=demand_masks,
            structural_selection=structural_selection,
            _structural_attestation=_StructuralSelectionAttestation(
                actions=actions, target_indices=target_indices,
                selection=structural_selection,
            ),
        )

    def has_authentic_structural_selection(self) -> bool:
        """Check object-identity binding without reducing device tensors."""

        attestation = self._structural_attestation
        return (
            isinstance(attestation, _StructuralSelectionAttestation)
            and attestation.actions is self.actions
            and attestation.target_indices is self.target_indices
            and attestation.selection is self.structural_selection
            and self.structural_selection is not None
            and self.structural_selection._has_policy_capability()
            and self.structural_selection._policy_target_indices is self.target_indices
        )

    def bind_surface_target_ids(
        self,
        surface_target_ids: torch.Tensor,
        *,
        target_snapshot: SurfaceTargetSnapshot,
        structural_action: PopulationAction | None = None,
    ) -> "PopulationDecision":
        """Return an authenticated decision with a matcher-bound target order.

        ``structural_action=None`` is the production arbitrary-action binding:
        the already-authenticated policy action tensor decides on-device which
        CREATE/RELOCATE rows receive target IDs, while RETIRE rows preserve
        their ``-1`` sentinel.  The matcher supplies geometry identities only;
        it does not make a second population decision or read an action scalar.
        """

        if not self.has_authentic_structural_selection() or self.structural_selection is None:
            raise ValueError("only an authenticated policy decision may bind surface target IDs")
        selection = self.structural_selection
        assert selection is not None
        if (
            surface_target_ids.shape != selection.surface_target_ids.shape
            or not isinstance(target_snapshot, SurfaceTargetSnapshot)
            or structural_action not in {
                None,
                PopulationAction.CREATE, PopulationAction.RELOCATE, PopulationAction.RETIRE,
            }
        ):
            raise ValueError("surface target IDs must match the complete bounded selection envelope")
        # Class agreement remains device-resident.  It joins the existing
        # K-bounded mutation envelope and is reduced only after its one
        # attributed cardinality transfer; binding itself must not sync GPU.
        safe_rows = selection.source_rows.clamp(min=0, max=max(int(self.actions.shape[0]) - 1, 0))
        existing = self.target_indices.index_select(0, safe_rows) if self.actions.numel() else self.target_indices
        target_bearing = (
            (selection.action_codes == PopulationAction.CREATE.code)
            | (selection.action_codes == PopulationAction.RELOCATE.code)
            if structural_action is None
            else selection.action_codes == structural_action.code
        )
        selection_target_ids = torch.where(
            selection.valid_mask & target_bearing,
            surface_target_ids,
            selection.surface_target_ids,
        )
        bound_values = torch.where(
            selection.valid_mask & target_bearing, selection_target_ids, existing
        )
        bound_target_indices = self.target_indices.scatter(0, safe_rows, bound_values)
        return PopulationDecision.policy_output(
            actions=self.actions, target_indices=bound_target_indices,
            evidence_demands=self.evidence_demands, demand_masks=self.demand_masks,
            structural_selection=selection.with_surface_target_ids(
                selection_target_ids,
                target_snapshot=target_snapshot,
                target_binding_action=structural_action,
                policy_target_indices=bound_target_indices,
            ),
        )


@dataclass(frozen=True)
class _StructuralSelectionAttestation:
    """Private capability linking borrowed selection tensors to policy output."""

    actions: torch.Tensor
    target_indices: torch.Tensor
    selection: "PopulationStructuralSelection"


@dataclass(frozen=True)
class PopulationStructuralSelection:
    """Bounded structural candidates prepared on-device for one fence.

    ``surface_target_ids`` starts as ``-1`` in the generic population policy:
    a later surface-target matcher owns binding an action to a concrete
    geometry hypothesis.  It must produce a new selection with exact IDs
    before the host-fence adapter will construct a proposal.  Padded entries
    have ``valid_mask=False`` and are never materialized by the adapter.
    """

    source_rows: torch.Tensor
    action_codes: torch.Tensor
    surface_target_ids: torch.Tensor
    valid_mask: torch.Tensor
    max_actions: int
    _policy_capability: object | None = field(default=None, repr=False, compare=False)
    _policy_actions: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _policy_target_indices: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _bound_target_snapshot: SurfaceTargetSnapshot | None = field(
        default=None, repr=False, compare=False,
    )
    # Private K-sized snapshots are the trust boundary for the mutable device
    # borrows above.  The adapter compares them on device before the sole
    # fence readback, so coordinated public-tensor mutation cannot rewrite a
    # policy decision after authentication.
    _attested_source_rows: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _attested_action_codes: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _attested_target_ids: torch.Tensor | None = field(default=None, repr=False, compare=False)
    _target_binding_action: PopulationAction | None = field(default=None, repr=False, compare=False)
    _binding_action_consistency: torch.Tensor | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.max_actions) is not int or self.max_actions < 0:
            raise ValueError("max_actions must be a non-negative integer")
        values = (self.source_rows, self.action_codes, self.surface_target_ids, self.valid_mask)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("structural selection fields must be tensors")
        if any(value.ndim != 1 or value.shape[0] != self.max_actions for value in values):
            raise ValueError("structural selection fields must have shape [max_actions]")
        if len({value.device for value in values}) != 1:
            raise ValueError("structural selection fields must share one device")
        if self.source_rows.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("source_rows must be an integer tensor")
        if self.action_codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("action_codes must be an integer tensor")
        if self.surface_target_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("surface_target_ids must be an integer tensor")
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must be a bool tensor")

    def with_surface_target_ids(
        self,
        surface_target_ids: torch.Tensor,
        *,
        target_snapshot: SurfaceTargetSnapshot,
        target_binding_action: PopulationAction | None,
        policy_target_indices: torch.Tensor | None = None,
    ) -> "PopulationStructuralSelection":
        """Bind exact target IDs without changing the policy-selected rows."""

        if (
            not isinstance(surface_target_ids, torch.Tensor)
            or surface_target_ids.shape != self.surface_target_ids.shape
            or surface_target_ids.device != self.surface_target_ids.device
            or surface_target_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
        ) or (
            not isinstance(target_snapshot, SurfaceTargetSnapshot)
            or target_binding_action not in {
                None,
                PopulationAction.CREATE, PopulationAction.RELOCATE, PopulationAction.RETIRE,
            }
        ):
            raise ValueError("surface_target_ids must match the bounded selection")
        return PopulationStructuralSelection(
            source_rows=self.source_rows,
            action_codes=self.action_codes,
            surface_target_ids=surface_target_ids,
            valid_mask=self.valid_mask,
            max_actions=self.max_actions,
            _policy_capability=self._policy_capability,
            _policy_actions=self._policy_actions,
            _policy_target_indices=(
                self._policy_target_indices
                if policy_target_indices is None
                else policy_target_indices
            ),
            _bound_target_snapshot=target_snapshot,
            _attested_source_rows=self.source_rows.detach().clone(),
            _attested_action_codes=self.action_codes.detach().clone(),
            _attested_target_ids=surface_target_ids.detach().clone(),
            _target_binding_action=target_binding_action,
            _binding_action_consistency=(
                (~self.valid_mask)
                | (
                    (
                        (self._attested_action_codes == PopulationAction.CREATE.code)
                        | (
                            self._attested_action_codes
                            == PopulationAction.RELOCATE.code
                        )
                        | (
                            (self._attested_action_codes == PopulationAction.RETIRE.code)
                            & (surface_target_ids == -1)
                        )
                    )
                    if target_binding_action is None
                    else self._attested_action_codes == target_binding_action.code
                )
            ),
        )

    def _has_policy_capability(self) -> bool:
        return self._policy_capability is _POLICY_SELECTION_CAPABILITY


class GaussianPopulationPolicy:
    """Small deterministic shadow policy with one ordinary decision owner."""

    def __init__(
        self,
        *,
        min_meaningful_exposures: int = 1,
        min_independent_views: int = 1,
        correction_depth_threshold: float = 0.5,
        exploration_uncertainty_threshold: float = 0.5,
        retirement_existence_threshold: float = 0.2,
        target_gain_threshold: float = 0.5,
    ) -> None:
        if (
            isinstance(min_meaningful_exposures, bool)
            or not isinstance(min_meaningful_exposures, int)
            or min_meaningful_exposures < 0
            or isinstance(min_independent_views, bool)
            or not isinstance(min_independent_views, int)
            or min_independent_views < 0
        ):
            raise ValueError("minimum evidence counts must be nonnegative integers")
        thresholds = (
            correction_depth_threshold,
            exploration_uncertainty_threshold,
            retirement_existence_threshold,
            target_gain_threshold,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
            for value in thresholds
        ):
            raise ValueError("policy thresholds must be finite real numbers")
        self.min_meaningful_exposures = min_meaningful_exposures
        self.min_independent_views = min_independent_views
        self.correction_depth_threshold = correction_depth_threshold
        self.exploration_uncertainty_threshold = exploration_uncertainty_threshold
        self.retirement_existence_threshold = retirement_existence_threshold
        self.target_gain_threshold = target_gain_threshold

    @staticmethod
    def _check_vector(name: str, value: torch.Tensor, n: int, device: torch.device) -> None:
        if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != n:
            raise ValueError(f"{name} must be a tensor shaped [N]")
        if value.device != device:
            raise ValueError(f"{name} must share the belief device")
        if not value.dtype.is_floating_point and name not in {
            "meaningful_exposure_count", "independent_support_views",
            "independent_view_count",
        }:
            raise TypeError(f"{name} must use a floating-point dtype")

    @staticmethod
    def _demands(
        versions: Optional[dict[str, int]], epoch: int
    ) -> EvidenceDemandBatch:
        if versions is None:
            return EvidenceDemandBatch(())
        common = dict(
            map_version=versions["map"], topology_version=versions["topology"],
            coordinate_version=versions["coordinate"], schema_version=versions["schema"],
            optimizer_version=versions["optimizer"], surface_version=versions["surface"],
            policy_version=versions["policy"], priority=1.0, expiry_epoch=epoch + 1,
        )
        return EvidenceDemandBatch((
            EvidenceDemand(
                kind=EvidenceDemandKind.INDEPENDENT_VIEW,
                requested_modality=EvidenceDemandModality.DEPTH,
                target_ids=(), region_id="underexposed_gaussians", cell_id=None, **common,
            ),
            EvidenceDemand(
                kind=EvidenceDemandKind.DEPTH_SURFACE,
                requested_modality=EvidenceDemandModality.DEPTH,
                target_ids=(), region_id="uncertain_geometry", cell_id=None, **common,
            ),
        ))

    @staticmethod
    def _structural_selection(
        actions: torch.Tensor,
        action_score: torch.Tensor,
        target_indices: torch.Tensor,
        *,
        max_actions: int,
    ) -> PopulationStructuralSelection:
        """Keep a fixed bounded candidate envelope on device.

        ``topk`` yields a fixed K envelope with a device-side valid mask;
        this avoids exporting the full [N] policy vector or performing dynamic
        ``nonzero`` compaction merely to reach a mutation fence.
        """

        count = min(max_actions, int(actions.shape[0]))
        if count == 0:
            empty_i64 = torch.empty((0,), dtype=torch.int64, device=actions.device)
            empty_i8 = torch.empty((0,), dtype=torch.int8, device=actions.device)
            return PopulationStructuralSelection(
                source_rows=empty_i64,
                action_codes=empty_i8,
                surface_target_ids=empty_i64.clone(),
                valid_mask=torch.empty((0,), dtype=torch.bool, device=actions.device),
                max_actions=0,
                _policy_capability=_POLICY_SELECTION_CAPABILITY,
                _policy_actions=actions,
                _policy_target_indices=target_indices,
                _attested_source_rows=empty_i64.clone(),
                _attested_action_codes=empty_i8.clone(),
                _attested_target_ids=empty_i64.clone(),
                _binding_action_consistency=torch.empty((0,), dtype=torch.bool, device=actions.device),
            )
        structural = (
            (actions == PopulationAction.CREATE.code)
            | (actions == PopulationAction.RELOCATE.code)
            | (actions == PopulationAction.RETIRE.code)
        )
        ranks = torch.where(structural, action_score, torch.full_like(action_score, float("-inf")))
        values, source_rows = torch.topk(ranks, k=count, largest=True, sorted=True)
        valid_mask = torch.isfinite(values)
        return PopulationStructuralSelection(
            source_rows=source_rows.to(dtype=torch.int64),
            action_codes=actions.index_select(0, source_rows).to(dtype=torch.int8),
            surface_target_ids=torch.full((count,), -1, dtype=torch.int64, device=actions.device),
            valid_mask=valid_mask,
            max_actions=count,
            _policy_capability=_POLICY_SELECTION_CAPABILITY,
            _policy_actions=actions,
            _policy_target_indices=target_indices,
            _attested_source_rows=source_rows.detach().clone().to(dtype=torch.int64),
            _attested_action_codes=actions.index_select(0, source_rows).detach().clone().to(dtype=torch.int8),
            _attested_target_ids=torch.full((count,), -1, dtype=torch.int64, device=actions.device),
            _binding_action_consistency=torch.ones((count,), dtype=torch.bool, device=actions.device),
        )

    def decide(
        self,
        belief: GaussianBelief,
        token: MutationWorkToken,
        evidence: PopulationEvidence,
        *,
        capacity: Optional[int] = None,
        snapshot_versions: Optional[dict[str, int]] = None,
    ) -> PopulationDecision:
        """Score one snapshot using bounded tensor operations only.

        ``capacity`` is a hard population limit.  At the limit, a useful target
        is paired with a donor and emitted as one RELOCATE action; the policy
        never emits an unpaired RETIRE plus CREATE for that case.
        """
        n = belief.n
        device = belief.existence.p_exist.device
        for name, value in (
            ("p_exist", belief.existence.p_exist),
            ("p_surface_consistent", belief.geometry.p_surface_consistent),
            ("surface_position_uncertainty", belief.geometry.surface_position_uncertainty),
            ("redundancy", belief.representation.redundancy),
            ("expected_marginal_gain", belief.representation.expected_marginal_gain),
            ("target_gain", evidence.target_gain),
            ("depth_residual", evidence.depth_residual),
            ("free_space_conflict", evidence.free_space_conflict),
        ):
            self._check_vector(name, value, n, device)
        native_independent = evidence.independent_view_count
        if native_independent is not None:
            self._check_vector("independent_view_count", native_independent, n, device)
        target_conflict = evidence.target_conflict
        if target_conflict is not None:
            self._check_vector("target_conflict", target_conflict, n, device)
        if n == 0:
            empty_actions = torch.empty((0,), dtype=torch.int8, device=device)
            empty_target_indices = torch.empty((0,), dtype=torch.int64, device=device)
            empty_selection = self._structural_selection(
                empty_actions,
                torch.empty((0,), dtype=torch.float32, device=device),
                empty_target_indices,
                max_actions=token.units,
            )
            return PopulationDecision.policy_output(
                actions=empty_actions,
                target_indices=empty_target_indices,
                evidence_demands=EvidenceDemandBatch(()),
                demand_masks=(), structural_selection=empty_selection,
            )
        if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0
        ):
            raise ValueError("capacity must be nonnegative")

        versions: Optional[dict[str, int]] = None
        if snapshot_versions is not None:
            required = {"map", "topology", "coordinate", "schema", "optimizer", "surface", "policy"}
            if set(snapshot_versions) != required:
                raise ValueError("snapshot_versions must contain the seven required versions")
            versions = dict(snapshot_versions)
            for name, value in versions.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} version must be a nonnegative integer")

        actions = torch.full((n,), PopulationAction.NONE.code, dtype=torch.int8, device=device)
        target_indices = torch.full((n,), -1, dtype=torch.int64, device=device)
        exposure = belief.observability.meaningful_exposure_count
        if exposure is None:
            exposure = torch.zeros(n, dtype=torch.int32, device=device)
        self._check_vector("meaningful_exposure_count", exposure, n, device)
        independent = belief.observability.independent_support_views
        self._check_vector("independent_support_views", independent, n, device)
        eligible = (exposure >= self.min_meaningful_exposures) & (
            independent >= self.min_independent_views
        )
        underexposed = ~eligible
        uncertain_geometry = eligible & (
            belief.geometry.surface_position_uncertainty >= self.exploration_uncertainty_threshold
        )
        demands = self._demands(versions, token.epoch)
        if token.units == 0:
            actions.fill_(PopulationAction.ABSTAIN.code)
            return PopulationDecision.policy_output(
                actions=actions, target_indices=target_indices,
                evidence_demands=demands, demand_masks=(underexposed, uncertain_geometry),
                structural_selection=self._structural_selection(
                    actions, torch.zeros_like(evidence.target_gain), target_indices,
                    max_actions=token.units,
                ),
            )
        actions = torch.where(
            ~eligible,
            torch.full_like(actions, PopulationAction.ABSTAIN.code),
            actions,
        )
        actions = torch.where(
            eligible & (evidence.depth_residual >= self.correction_depth_threshold) &
            (belief.geometry.p_surface_consistent >= self.retirement_existence_threshold),
            torch.full_like(actions, PopulationAction.CORRECT.code), actions,
        )
        actions = torch.where(
            eligible & (actions == PopulationAction.NONE.code) &
            (belief.geometry.surface_position_uncertainty >= self.exploration_uncertainty_threshold),
            torch.full_like(actions, PopulationAction.EXPLORE.code), actions,
        )
        retire = eligible & (belief.existence.p_exist <= self.retirement_existence_threshold)
        retire |= eligible & (evidence.free_space_conflict > 0)
        actions = torch.where(
            retire & (actions == PopulationAction.NONE.code),
            torch.full_like(actions, PopulationAction.RETIRE.code), actions,
        )

        target = evidence.target_gain.maximum(belief.representation.expected_marginal_gain)
        # Native depth-target conflict is local to the proposed target.  It
        # can disqualify that target but is never row-level free-space proof
        # for retiring unrelated existing Gaussians.
        if target_conflict is not None:
            target = (target - target_conflict.clamp_min(0)).clamp_min(0)
        target_views = (
            independent if native_independent is None else native_independent
        )
        demand = (
            eligible
            & (target_views >= self.min_independent_views)
            & (target >= self.target_gain_threshold)
        )
        if capacity is not None and n >= capacity:
            # The index tensors remain on-device; this is one paired mutation candidate.
            donor_score = torch.where(eligible, belief.representation.redundancy, torch.zeros_like(target))
            donor = torch.argmax(donor_score)
            target_mask = demand & (torch.arange(n, device=device) != donor)
            target_index = torch.argmax(torch.where(target_mask, target, torch.zeros_like(target)))
            has_target = torch.any(target_mask)
            paired = has_target & eligible[donor]
            actions = torch.where(
                paired & (torch.arange(n, device=device) == donor),
                torch.full_like(actions, PopulationAction.RELOCATE.code), actions,
            )
            target_indices = torch.where(
                paired & (torch.arange(n, device=device) == donor),
                target_index.expand(n), target_indices,
            )
            actions = torch.where(
                paired & (torch.arange(n, device=device) != donor) &
                (actions == PopulationAction.CREATE.code),
                torch.full_like(actions, PopulationAction.NONE.code), actions,
            )
            # At capacity, retirement is not an independent structural action:
            # capacity recycling must be represented by the single paired
            # relocation above.
            actions = torch.where(
                actions == PopulationAction.RETIRE.code,
                torch.full_like(actions, PopulationAction.NONE.code),
                actions,
            )
        elif capacity is None or n < capacity:
            remaining = n if capacity is None else capacity - n
            create_count = min(token.units, remaining, n)
            create_indices = torch.topk(target, k=create_count, largest=True, sorted=False).indices
            create_slots = torch.zeros(n, dtype=torch.bool, device=device).scatter_(
                0, create_indices, True
            )
            actions = torch.where(
                demand & create_slots & (actions == PopulationAction.NONE.code),
                torch.full_like(actions, PopulationAction.CREATE.code), actions,
            )

        # Enforce the scheduler's neutral work budget over every emitted action.
        # Structural actions are already paired in capacity mode; ranking here
        # merely truncates the finite action set deterministically.
        action_score = torch.where(
            actions == PopulationAction.CORRECT.code, evidence.depth_residual,
            torch.where(
                actions == PopulationAction.EXPLORE.code,
                belief.geometry.surface_position_uncertainty,
                torch.where(
                    actions == PopulationAction.RETIRE.code,
                    belief.representation.redundancy + evidence.free_space_conflict,
                    torch.where(
                        actions == PopulationAction.RELOCATE.code,
                        target,
                        torch.where(actions == PopulationAction.CREATE.code, target, torch.zeros_like(target)),
                    ),
                ),
            ),
        )
        # A mutation fence accepts one homogeneous structural plan.  When
        # independent evidence makes CREATE and RETIRE attractive in the same
        # epoch, select the better bounded action class on-device rather than
        # delegating a mixed decision to the adapter.  Constant-capacity
        # RELOCATE was already canonicalized above and participates as its
        # own single class.
        structural_codes = torch.tensor(
            [PopulationAction.CREATE.code, PopulationAction.RELOCATE.code, PopulationAction.RETIRE.code],
            dtype=actions.dtype, device=device,
        )
        class_scores = torch.stack(tuple(
            torch.max(torch.where(actions == code, action_score, torch.full_like(action_score, float("-inf"))))
            for code in structural_codes
        ))
        selected_structural_code = structural_codes[torch.argmax(class_scores)]
        is_structural = (actions == PopulationAction.CREATE.code) | (
            actions == PopulationAction.RELOCATE.code
        ) | (actions == PopulationAction.RETIRE.code)
        actions = torch.where(
            is_structural & (actions != selected_structural_code),
            torch.full_like(actions, PopulationAction.NONE.code), actions,
        )
        action_score = torch.where(
            actions >= PopulationAction.CORRECT.code,
            action_score,
            torch.zeros_like(action_score),
        )
        keep_count = min(token.units, n)
        keep_indices = torch.topk(action_score, k=keep_count, largest=True, sorted=False).indices
        keep = torch.zeros(n, dtype=torch.bool, device=device).scatter_(0, keep_indices, True)
        actionable = actions >= PopulationAction.CORRECT.code
        actions = torch.where(actionable & ~keep, torch.full_like(actions, PopulationAction.ABSTAIN.code), actions)
        target_indices = torch.where(actions == PopulationAction.RELOCATE.code, target_indices, torch.full_like(target_indices, -1))

        # Demand descriptors are bounded and versioned; masks keep applicability
        # device-resident so the scheduler can ignore empty categories without
        # forcing a policy-epoch synchronization or selecting a frame here.
        return PopulationDecision.policy_output(
            actions=actions, target_indices=target_indices,
            evidence_demands=demands, demand_masks=(underexposed, uncertain_geometry),
            structural_selection=self._structural_selection(
                actions, action_score, target_indices, max_actions=token.units,
            ),
        )
