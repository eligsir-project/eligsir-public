"""Read-only runtime bridge for the M5 Gaussian population-policy shadow.

The bridge deliberately sits *after* a completed fixed-topology transaction.
It consumes the transaction's E0 ``SupervisionEvidenceBatch`` together with a
current ``GaussianBelief`` view, publishes a bounded device-resident policy
observation, and stops there.  In particular it never submits a proposal to
the mutation arbiter, changes legacy MCMC cadence, updates lifecycle state, or
enqueues a demand into the live scheduler.  Those are separate migration
steps, and keeping this boundary observational makes the first live wiring
safe to enable for telemetry.

Evidence is bound to the pre-Adam parameter snapshot.  The bridge therefore
checks it against the result's ``*_before`` versions, then independently
checks that the supplied current snapshot is exactly the result's ``*_after``
state.  A stale result is dropped rather than remapped or applied to a newer
map.  The host only examines tensor metadata (shape/device/dtype); it never
reduces, materializes, or copies a device tensor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from eligsir.mapping.population_policy import (
    GaussianPopulationPolicy,
    MutationWorkToken,
    PopulationAction,
    PopulationDecision,
    PopulationEvidence,
)
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
)
from eligsir.utils.gaussian_belief import GaussianBelief


class PopulationShadowDropReason(str, Enum):
    """Why an observational shadow epoch emitted no policy decision."""

    DISABLED = "disabled"
    NO_SUPERVISION_EVIDENCE = "no_supervision_evidence"
    MISSING_WORK_TOKEN = "missing_work_token"
    MISSING_CURRENT_VERSIONS = "missing_current_versions"
    MISSING_BELIEF = "missing_belief"
    STALE_EVIDENCE_SNAPSHOT = "stale_evidence_snapshot"
    STALE_CURRENT_SNAPSHOT = "stale_current_snapshot"
    INVALID_E0_ACCUMULATOR = "invalid_e0_accumulator"
    INVALID_BELIEF_SHAPE = "invalid_belief_shape"
    INVALID_EVIDENCE_SHAPE = "invalid_evidence_shape"


# These are deliberately integer codes in the asynchronous summary.  The
# policy thread never reads a device scalar merely to name a telemetry event;
# a consumer may materialize this one-element code under its own boundary.
NATIVE_TARGET_NO_TARGET = 0
NATIVE_TARGET_TARGET_UNQUALIFIED = 1
NATIVE_TARGET_SUPPORT_OK = 2


@dataclass(frozen=True)
class PopulationShadowVersions:
    """All exact versions needed to accept one shadow decision epoch."""

    map_content: int
    topology: int
    coordinate_frame: int
    schema: int
    optimizer_update: int
    surface_evidence: int
    policy: int
    parameter: int

    def __post_init__(self) -> None:
        for name in (
            "map_content",
            "topology",
            "coordinate_frame",
            "schema",
            "optimizer_update",
            "surface_evidence",
            "policy",
            "parameter",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def policy_versions(self) -> dict[str, int]:
        """Return the policy's seven-version evidence-demand snapshot."""

        return {
            "map": self.map_content,
            "topology": self.topology,
            "coordinate": self.coordinate_frame,
            "schema": self.schema,
            "optimizer": self.optimizer_update,
            "surface": self.surface_evidence,
            "policy": self.policy,
        }


@dataclass
class _AdapterRejectionState:
    """Bounded mutable telemetry state shared by one decision handoff."""

    rejected: bool = False
    telemetry_retained: bool = False
    telemetry_delivered: bool = False
    plan_drop_reason: str | None = None


@dataclass(frozen=True)
class _PopulationShadowEvent:
    """One internal bounded, device-resident policy observation.

    ``decision`` carries borrowed device tensors.  A CPU telemetry consumer may
    choose to summarize it asynchronously, but the bridge itself deliberately
    does not call ``.item()``, ``.cpu()``, or any reduction.
    """

    transaction_id: int | str | None
    token: MutationWorkToken | None
    versions: PopulationShadowVersions | None
    decision: PopulationDecision | None
    drop_reason: PopulationShadowDropReason | None
    accumulator_status: str | None = None
    e0_exposure_count: torch.Tensor | None = None
    e0_support_count: torch.Tensor | None = None
    target_gain: torch.Tensor | None = None
    stale_current_mismatches: tuple[tuple[str, int, int], ...] = ()
    decision_token: int | None = None
    rejection_state: _AdapterRejectionState | None = None
    native_target_count: int | None = 0
    native_target_independent_view_count: torch.Tensor | None = None


@dataclass(frozen=True)
class PopulationShadowSummary:
    """Owned CPU summary emitted only by ``drain_async_summaries``.

    ``action_counts_cpu`` is a new, compact seven-element CPU tensor.  It is
    intentionally not converted to Python scalars here, so a control-plane
    consumer can synchronize/attribute it under its own explicit telemetry
    boundary.  No borrowed Gaussian or device decision tensor escapes.
    """

    transaction_id: int | str | None
    epoch: int | None
    units: int | None
    versions: PopulationShadowVersions | None
    demand_count: int
    action_counts_cpu: torch.Tensor | None
    drop_reason: PopulationShadowDropReason | None
    # [exposure nonzero count, exposure max, support nonzero count, support
    # max, target-gain nonzero count, target-gain max].  This is intentionally
    # compact and leaves device-to-host synchronization to the consumer.
    e0_input_counts_maxima_cpu: torch.Tensor | None = None
    accumulator_status: str | None = None
    selection_rejected_cpu: torch.Tensor | None = None
    adapter_rejected: bool | None = None
    # One scalar code: 0=no qualified native target, 1=present target is
    # unqualified for true B4, 2=qualified target.  The code
    # is copied asynchronously; no full target array crosses the boundary.
    native_target_count: int | None = 0
    native_target_status_cpu: torch.Tensor | None = None
    # Explicit name for the action histogram used to explain selection.  Keep
    # ``action_counts_cpu`` as the compatibility spelling; both refer to the
    # same bounded seven-element tensor.
    selected_action_histogram_cpu: torch.Tensor | None = None
    adapter_plan_drop_reason: str | None = None
    # Stable in-process key for correlating a late mutation-fence result with
    # the earlier shadow observation.  ``None`` denotes a dropped/non-policy
    # event with no decision handoff.
    decision_token: int | None = None
    # ``(axis, expected, current)`` triples are emitted only for a stale
    # current snapshot.  The tuple is bounded by the eight version axes.
    stale_current_mismatches: tuple[tuple[str, int, int], ...] = ()
    compact_packed_readback: Any = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class PopulationActivationObservation:
    """One accepted policy result handed to the live mutation boundary.

    This remains an in-process, borrowed-device handoff.  It deliberately has
    no CPU summaries and is consumed exactly once by the production mutation
    tail.  The separate telemetry queue retains its own event, so enabling
    actuation cannot make the shadow observability stream disappear.
    """

    transaction_id: int | str | None
    token: MutationWorkToken
    # Preserve the borrowed immutable E0 provenance separately from the
    # post-Adam action fence.  They must never be conflated: the origin
    # authenticates the reused geometry/target object, while ``versions`` is
    # the only version vector that can authorize a structural write.
    origin_snapshot: EvidenceSnapshotToken
    versions: PopulationShadowVersions
    decision: PopulationDecision
    decision_token: int = 0
    rejection_state: _AdapterRejectionState = field(
        default_factory=_AdapterRejectionState, repr=False, compare=False
    )


def _vector_on(
    value: Any, *, n: int, device: torch.device, floating: bool = True
) -> bool:
    """Check tensor metadata only; never inspect tensor values on the host."""

    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 1
        and value.shape[0] == n
        and value.device == device
        and (not floating or value.dtype.is_floating_point)
    )


def _result_version(result: Any, name: str) -> int | None:
    """Return an exact non-negative result version or fail closed."""

    value = getattr(result, name, None)
    return value if type(value) is int and value >= 0 else None


def _compact_async_cpu(
    value: torch.Tensor,
    *,
    transfer_owner: Any,
    transfer_events: list[Any],
) -> torch.Tensor | None:
    """Submit one compact copy only through the exact owner/device path.

    CUDA telemetry is optional: without the composition owner, dropping the
    field is safer than creating an unaccounted host transfer.  CPU tensors
    remain directly materializable for deterministic tests and CPU runtimes.
    """

    if value.device.type == "cuda":
        try:
            owner_device = torch.device(getattr(transfer_owner, "device", "cpu"))
            stream = getattr(transfer_owner, "transfer_stream", None)
        except (TypeError, RuntimeError, ValueError):
            return None
        if owner_device != value.device or stream is None:
            return None
        try:
            host = torch.empty_like(value, device="cpu", pin_memory=True)
            transfer_events.append(stream.d2h_into(value.detach(), host))
        except (RuntimeError, TypeError, ValueError):
            return None
        return host
    try:
        return value.detach().to(device="cpu", non_blocking=True)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


class PopulationShadowRuntime:
    """Own one read-only ``GaussianPopulationPolicy`` at runtime.

    ``max_pending_telemetry`` bounds retained observations.  No retained event
    is forwarded to the real evidence-demand resolver: publishing the policy's
    intent is intentionally not the same thing as giving the scheduler a new
    decision input during M5 shadow operation.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        policy: GaussianPopulationPolicy | None = None,
        max_pending_telemetry: int = 16,
        transfer_owner: Any = None,
        full_diagnostics: bool = False,
    ) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if type(max_pending_telemetry) is not int or max_pending_telemetry <= 0:
            raise ValueError("max_pending_telemetry must be a positive exact integer")
        if type(full_diagnostics) is not bool:
            raise TypeError("full_diagnostics must be bool")
        if policy is not None and not isinstance(policy, GaussianPopulationPolicy):
            raise TypeError("policy must be GaussianPopulationPolicy or None")
        self.enabled = enabled
        self.policy = policy if policy is not None else GaussianPopulationPolicy()
        self.max_pending_telemetry = max_pending_telemetry
        self.transfer_owner = transfer_owner
        self.full_diagnostics = full_diagnostics
        self._pending: deque[_PopulationShadowEvent] = deque()
        # Demand descriptors are CPU-only, versioned policy output.  Keep a
        # separate bounded handoff so telemetry draining is never a scheduler
        # prerequisite and no policy tensor needs a host scalar decision.
        self._pending_evidence_demands = deque()
        self._activation_pending: deque[PopulationActivationObservation] = deque()
        self._next_decision_token = 0
        self.telemetry_overflow_drops = 0
        self.last_stale_current_mismatches: tuple[tuple[str, int, int], ...] = ()

    @property
    def pending_event_count(self) -> int:
        """Return the bounded queue size without exposing borrowed tensors."""

        return len(self._pending)

    def drain_async_summaries(self) -> tuple[PopulationShadowSummary, ...]:
        """Drain compact summaries outside the optimization transaction.

        This is the only public telemetry egress.  It does one bounded action
        histogram plus six E0 input counters per accepted event and transfers
        those compact values to
        CPU with ``non_blocking=True``; callers that materialize those values
        must do so under their own attributed consumer boundary.
        """

        events = tuple(self._pending)
        self._pending.clear()
        summaries: list[PopulationShadowSummary] = []
        for event in events:
            decision = event.decision
            action_counts_cpu = None
            e0_input_counts_maxima_cpu = None
            selection_rejected_cpu = None
            native_target_status_cpu = None
            demand_count = 0
            compact_packed_readback = None
            if decision is not None:
                def compact(value: torch.Tensor | None) -> torch.Tensor | None:
                    if value is None:
                        return None
                    return _compact_async_cpu(
                        value,
                        transfer_owner=self.transfer_owner,
                        transfer_events=[],
                    )

                selection = decision.structural_selection
                if selection is None:
                    action_codes = None
                    valid_mask = None
                else:
                    action_codes = selection.action_codes
                    valid_mask = selection.valid_mask
                action_counts = (
                    None
                    if action_codes is None
                    else torch.bincount(
                        action_codes[valid_mask].to(dtype=torch.int64), minlength=7
                    )
                )
                if action_counts is not None and action_counts.device.type == "cuda":
                    owner = self.transfer_owner
                    try:
                        owner_device = torch.device(getattr(owner, "device", "cpu"))
                        if (
                            owner_device == action_counts.device
                            and callable(getattr(owner, "readback_mutation_cardinality", None))
                        ):
                            native_status = torch.zeros(1, dtype=torch.uint8, device=action_counts.device)
                            if event.native_target_count > 0 and event.native_target_independent_view_count is not None:
                                has_support = torch.any(event.native_target_independent_view_count >= 2)
                                native_status = torch.where(
                                    has_support,
                                    torch.tensor(NATIVE_TARGET_SUPPORT_OK, dtype=torch.uint8, device=action_counts.device),
                                    torch.tensor(NATIVE_TARGET_TARGET_UNQUALIFIED, dtype=torch.uint8, device=action_counts.device),
                                ).reshape(1)
                            elif event.native_target_count > 0:
                                native_status.fill_(NATIVE_TARGET_TARGET_UNQUALIFIED)
                            selected_rejected = (
                                (
                                    action_counts[PopulationAction.CREATE.code]
                                    + action_counts[PopulationAction.RELOCATE.code]
                                    + action_counts[PopulationAction.RETIRE.code]
                                    == 0
                                )
                                .to(dtype=torch.uint8)
                                .reshape(1)
                            )
                            packed = torch.cat(
                                (action_counts.clamp(max=255).to(torch.uint8), native_status, selected_rejected)
                            ).contiguous()
                            compact_packed_readback = owner.readback_mutation_cardinality(packed)
                            action_counts_cpu = None
                        else:
                            action_counts_cpu = None
                    except (RuntimeError, TypeError, ValueError, AttributeError):
                        action_counts_cpu = None
                else:
                    action_counts_cpu = compact(action_counts)
                demand_count = len(decision.evidence_demands.demands)
                if self.full_diagnostics and (
                    event.e0_exposure_count is not None
                    and event.e0_support_count is not None
                    and event.target_gain is not None
                ):
                    exposure = event.e0_exposure_count.to(dtype=torch.float32)
                    support = event.e0_support_count.to(dtype=torch.float32)
                    target_gain = event.target_gain
                    e0_input_counts_maxima = torch.stack(
                        (
                            torch.count_nonzero(exposure).to(torch.float32),
                            exposure.amax(),
                            torch.count_nonzero(support).to(torch.float32),
                            support.amax(),
                            torch.count_nonzero(target_gain).to(torch.float32),
                            target_gain.amax(),
                        )
                    )
                    e0_input_counts_maxima_cpu = compact(e0_input_counts_maxima)
                if action_counts is not None and action_counts_cpu is not None:
                    selection_rejected = (
                        (action_counts[PopulationAction.CREATE.code]
                         + action_counts[PopulationAction.RELOCATE.code]
                         + action_counts[PopulationAction.RETIRE.code]) == 0
                    )
                    selection_rejected_cpu = compact(selection_rejected)
                if compact_packed_readback is not None:
                    # The packed owner readback already contains the sole
                    # native-support byte for this epoch.  Do not submit a
                    # second status-only D2H here.
                    native_target_status_cpu = None
                elif event.native_target_count <= 0:
                    native_target_status_cpu = torch.tensor(
                            [NATIVE_TARGET_NO_TARGET], dtype=torch.int8
                    )
                elif (
                    event.native_target_independent_view_count is None
                    or not isinstance(event.native_target_independent_view_count, torch.Tensor)
                ):
                    native_target_status_cpu = torch.tensor(
                        [NATIVE_TARGET_NO_TARGET], dtype=torch.int8
                    )
                else:
                    # Production native targets are K=1, but keep this
                    # reduction bounded by the target provider's configured K
                    # rather than assuming a full Gaussian-sized payload.
                    independent = event.native_target_independent_view_count
                    has_support = torch.any(independent >= 2)
                    status = torch.where(
                        has_support,
                        torch.tensor(
                            NATIVE_TARGET_SUPPORT_OK,
                            dtype=torch.int8,
                            device=independent.device,
                        ),
                        torch.tensor(
                            NATIVE_TARGET_TARGET_UNQUALIFIED,
                            dtype=torch.int8,
                            device=independent.device,
                        ),
                    )
                    native_target_status_cpu = compact(status.reshape(1))
            summaries.append(
                PopulationShadowSummary(
                    transaction_id=event.transaction_id,
                    epoch=None if event.token is None else event.token.epoch,
                    units=None if event.token is None else event.token.units,
                    versions=event.versions,
                    demand_count=demand_count,
                    action_counts_cpu=action_counts_cpu,
                    drop_reason=event.drop_reason,
                    e0_input_counts_maxima_cpu=e0_input_counts_maxima_cpu,
                    accumulator_status=event.accumulator_status,
                    selection_rejected_cpu=selection_rejected_cpu,
                    native_target_count=event.native_target_count,
                    native_target_status_cpu=native_target_status_cpu,
                    selected_action_histogram_cpu=action_counts_cpu,
                    adapter_plan_drop_reason=(
                        None
                        if event.rejection_state is None
                        else event.rejection_state.plan_drop_reason
                    ),
                    adapter_rejected=(
                        True
                        if (
                            event.rejection_state is not None
                            and event.rejection_state.rejected
                        )
                        else None
                    ),
                    decision_token=event.decision_token,
                    stale_current_mismatches=event.stale_current_mismatches,
                    compact_packed_readback=compact_packed_readback,
                )
            )
            if event.rejection_state is not None:
                event.rejection_state.telemetry_retained = False
                event.rejection_state.telemetry_delivered = True
        return tuple(summaries)

    def _publish(self, event: _PopulationShadowEvent) -> None:
        if len(self._pending) >= self.max_pending_telemetry:
            evicted = self._pending.popleft()
            if evicted.rejection_state is not None:
                evicted.rejection_state.telemetry_retained = False
            self.telemetry_overflow_drops += 1
        if event.rejection_state is not None:
            event.rejection_state.telemetry_retained = True
        self._pending.append(event)

    def take_latest_activation_observation(
        self,
    ) -> PopulationActivationObservation | None:
        """Consume the newest accepted policy result for one live fence.

        A B=1 plan can contain several fixed-topology transactions but has
        one structural tail.  Older observations therefore cannot safely be
        applied after a later Adam update; retaining only the newest result is
        the fail-closed choice.  This does not affect asynchronous telemetry.
        """

        if not self._activation_pending:
            return None
        latest = self._activation_pending.pop()
        self._activation_pending.clear()
        return latest

    def record_adapter_rejection(
        self,
        observation: PopulationActivationObservation,
        plan_drop_reason: str | Enum | None = None,
    ) -> None:
        """Record a host-fence adapter rejection without changing policy output."""

        if not isinstance(observation, PopulationActivationObservation):
            return
        state = observation.rejection_state
        state.rejected = True
        if plan_drop_reason is not None:
            reason = getattr(plan_drop_reason, "value", plan_drop_reason)
            if not isinstance(reason, str) or not reason or len(reason) > 96:
                raise ValueError("plan_drop_reason must be a non-empty bounded string")
            state.plan_drop_reason = reason
        if not state.telemetry_retained:
            # If the original event was drained or evicted before the fence
            # rejected it, publish one bounded late-rejection record.  Stable
            # decision_token attribution avoids id() reuse and no global set
            # outlives the event/activation handoff.
            self._publish(
                _PopulationShadowEvent(
                    transaction_id=observation.transaction_id,
                    token=observation.token,
                    versions=observation.versions,
                    # The original decision was already drained/evicted.  A
                    # late fence record must not recreate selection or target
                    # classifications; correlate it to the original summary
                    # with decision_token instead.
                    decision=None,
                    drop_reason=None,
                    decision_token=observation.decision_token,
                    rejection_state=state,
                    native_target_count=None,
                )
            )

    def take_latest_evidence_demands(self):
        """Consume bounded CPU demand descriptors for the scheduler lane.

        This intentionally returns only immutable ``EvidenceDemand`` records;
        action tensors stay inside the policy event/telemetry path.
        """

        demands = tuple(self._pending_evidence_demands)
        self._pending_evidence_demands.clear()
        return demands

    def _drop(
        self,
        reason: PopulationShadowDropReason,
        *,
        evidence: SupervisionEvidenceBatch | None = None,
        token: MutationWorkToken | None = None,
        versions: PopulationShadowVersions | None = None,
        accumulator_status: str | None = None,
        stale_current_mismatches: tuple[tuple[str, int, int], ...] = (),
    ) -> PopulationShadowDropReason:
        self._publish(
            _PopulationShadowEvent(
                transaction_id=None if evidence is None else evidence.transaction_id,
                token=token,
                versions=versions,
                decision=None,
                drop_reason=reason,
                accumulator_status=accumulator_status,
                stale_current_mismatches=stale_current_mismatches,
            )
        )
        return reason

    @staticmethod
    def _current_mismatches(
        token: EvidenceSnapshotToken,
        versions: PopulationShadowVersions,
        result: Any,
    ) -> tuple[tuple[str, int, int], ...]:
        """Return bounded, host-metadata-only attribution for current staleness."""

        expected = (
            ("map", _result_version(result, "map_version_after"), versions.map_content),
            ("topology", _result_version(result, "topology_version"), versions.topology),
            ("coordinate", token.coordinate_frame, versions.coordinate_frame),
            ("schema", token.schema, versions.schema),
            ("optimizer_update", token.optimizer_update, versions.optimizer_update),
            ("parameter", _result_version(result, "parameter_version_after"), versions.parameter),
            ("surface", token.surface_evidence, versions.surface_evidence),
            ("policy", token.policy, versions.policy),
        )
        return tuple(
            (axis, expected_value, current_value)
            for axis, expected_value, current_value in expected
            if expected_value is not None and expected_value != current_value
        )

    @staticmethod
    def _pre_adam_matches(
        token: EvidenceSnapshotToken, result: Any
    ) -> bool:
        """Accept E0 only when it describes this result's render snapshot."""

        return (
            token.map_content == _result_version(result, "map_version_before")
            and token.topology == _result_version(result, "topology_version")
            and token.parameter == _result_version(result, "parameter_version_before")
        )

    @staticmethod
    def _current_matches(
        token: EvidenceSnapshotToken,
        versions: PopulationShadowVersions,
        result: Any,
    ) -> bool:
        """Reject a completed result once any relevant current state advanced."""

        return (
            versions.map_content == _result_version(result, "map_version_after")
            and versions.topology == _result_version(result, "topology_version")
            and versions.parameter == _result_version(result, "parameter_version_after")
            and versions.coordinate_frame == token.coordinate_frame
            and versions.schema == token.schema
            and versions.optimizer_update == token.optimizer_update
            and versions.surface_evidence == token.surface_evidence
            and versions.policy == token.policy
        )

    @staticmethod
    def _belief_is_valid(belief: GaussianBelief) -> bool:
        if not isinstance(belief, GaussianBelief) or type(belief.n) is not int or belief.n < 0:
            return False
        n = belief.n
        device = belief.existence.p_exist.device if isinstance(belief.existence.p_exist, torch.Tensor) else None
        if device is None:
            return False
        required = (
            belief.existence.p_exist,
            belief.existence.contradiction_evidence,
            belief.geometry.p_surface_consistent,
            belief.geometry.surface_position_uncertainty,
            belief.representation.redundancy,
            belief.representation.expected_marginal_gain,
            belief.representation.depth_residual_ema,
        )
        if not all(_vector_on(value, n=n, device=device) for value in required):
            return False
        exposure = belief.observability.meaningful_exposure_count
        if exposure is not None and not _vector_on(exposure, n=n, device=device, floating=False):
            return False
        return _vector_on(
            belief.observability.independent_support_views,
            n=n,
            device=device,
            floating=False,
        )

    @staticmethod
    def _adapt_evidence(
        evidence: SupervisionEvidenceBatch,
        belief: GaussianBelief,
        native_targets: Any = None,
    ) -> PopulationEvidence | None:
        """Adapt full-row E0 values without remapping compact row identities.

        Packed attribution needs the atomic row registry before it can safely
        address current rows.  Until then M5 accepts only an empty E0
        per-Gaussian payload (using existing belief EMAs) or explicitly full
        ``[N]`` row vectors with no ``gaussian_ids`` indirection.  Any other
        shape is deliberately stale-dropped, never padded or guessed.
        """

        n = belief.n
        device = belief.existence.p_exist.device
        # Even an E0 field that this initial adapter does not score must be
        # resident alongside the belief.  Accepting a CPU (or other-device)
        # per-view payload here would quietly establish a future transfer path.
        for name in (
            "rgb_residual_summary",
            "depth_residual_summary",
            "rgb_valid_pixels",
            "depth_valid_pixels",
            "alpha_support_summary",
            "intrinsic_reliability",
            "gaussian_ids",
            "view_incidence",
            "meaningful_measurement_opportunity",
            "support_evidence",
            "contradiction_evidence",
            "render_contribution",
            "gradient_utility_evidence",
        ):
            value = getattr(evidence, name)
            if value is not None and value.device != device:
                return None
        if evidence.gaussian_ids is not None:
            return None
        if evidence.gaussian_count not in (0, n):
            return None
        for name in (
            "support_evidence",
            "contradiction_evidence",
            "render_contribution",
            "gradient_utility_evidence",
            "meaningful_measurement_opportunity",
        ):
            value = getattr(evidence, name)
            if value is not None and not _vector_on(value, n=n, device=device):
                return None
        adapted = PopulationEvidence(
            target_gain=(
                evidence.gradient_utility_evidence
                if evidence.gradient_utility_evidence is not None
                else belief.representation.expected_marginal_gain
            ),
            depth_residual=belief.representation.depth_residual_ema,
            free_space_conflict=(
                evidence.contradiction_evidence
                if evidence.contradiction_evidence is not None
                else belief.existence.contradiction_evidence
            ),
        )
        if native_targets is None:
            return adapted
        from eligsir.mapping.evidence.surface_targets import SurfaceTargetBatch

        if (
            not isinstance(native_targets, SurfaceTargetBatch)
            or native_targets.count != 1
            or native_targets.device != device
            or native_targets.snapshot.map_content != evidence.snapshot_token.map_content
            or native_targets.snapshot.topology != evidence.snapshot_token.topology
            or native_targets.snapshot.coordinate_frame != evidence.snapshot_token.coordinate_frame
            or native_targets.snapshot.schema != evidence.snapshot_token.schema
            or native_targets.snapshot.surface_evidence != evidence.snapshot_token.surface_evidence
        ):
            return None
        support = native_targets.support_evidence
        conflict = native_targets.conflict_evidence
        independent = native_targets.independent_view_count
        if (
            support.shape != (1,)
            or conflict.shape != (1,)
            or independent.shape != (1,)
            or not support.dtype.is_floating_point
            or not conflict.dtype.is_floating_point
            or independent.dtype.is_floating_point
        ):
            return None
        # The bounded native target is an input to scoring only.  Its gain,
        # conflict, and physical-view count stay device resident and are never
        # interpreted as a topology write at this boundary.
        valid = (
            torch.isfinite(support)
            & torch.isfinite(conflict)
            & (support > 0)
            & (conflict >= 0)
            & (independent >= 2)
        )
        target_gain = support / independent.to(dtype=support.dtype).clamp_min(1)
        return PopulationEvidence(
            target_gain=torch.maximum(adapted.target_gain, target_gain.expand(n)),
            depth_residual=adapted.depth_residual,
            # ``conflict`` belongs to the one depth target, not every current
            # Gaussian row.  Preserve row conflict solely from full-row E0;
            # target-local conflict is scored below and cannot emit RETIRE.
            free_space_conflict=adapted.free_space_conflict,
            independent_view_count=torch.where(
                valid,
                independent,
                torch.zeros_like(independent),
            ).expand(n),
            target_conflict=torch.where(
                valid, conflict, torch.zeros_like(conflict)
            ).expand(n),
        )

    def observe(
        self,
        *,
        result: Any,
        belief: GaussianBelief | None,
        token: MutationWorkToken | None,
        current_versions: PopulationShadowVersions | None,
        capacity: int | None,
        native_targets: Any = None,
    ) -> PopulationShadowDropReason | None:
        """Run one strictly observational shadow-policy epoch.

        The caller owns cadence/budget through ``token``.  This method neither
        manufactures a token nor turns a decision into a mutation proposal.
        """

        self.last_stale_current_mismatches = ()
        if not self.enabled:
            # A disabled flag must be indistinguishable from no bridge at all:
            # not even a telemetry callback or retained record is permitted.
            return PopulationShadowDropReason.DISABLED
        evidence = getattr(result, "evidence", None)
        if not isinstance(evidence, SupervisionEvidenceBatch):
            return self._drop(PopulationShadowDropReason.NO_SUPERVISION_EVIDENCE)
        accumulator_status = getattr(
            result, "population_policy_e0_accumulator_status", None
        )
        if accumulator_status is not None and accumulator_status != "updated":
            return self._drop(
                PopulationShadowDropReason.INVALID_E0_ACCUMULATOR,
                evidence=evidence,
                accumulator_status=accumulator_status,
            )
        if not isinstance(token, MutationWorkToken):
            return self._drop(
                PopulationShadowDropReason.MISSING_WORK_TOKEN, evidence=evidence
            )
        if not isinstance(current_versions, PopulationShadowVersions):
            return self._drop(
                PopulationShadowDropReason.MISSING_CURRENT_VERSIONS,
                evidence=evidence,
                token=token,
            )
        if not isinstance(belief, GaussianBelief):
            return self._drop(
                PopulationShadowDropReason.MISSING_BELIEF,
                evidence=evidence,
                token=token,
                versions=current_versions,
            )
        snapshot = evidence.snapshot_token
        if not self._pre_adam_matches(snapshot, result):
            return self._drop(
                PopulationShadowDropReason.STALE_EVIDENCE_SNAPSHOT,
                evidence=evidence,
                token=token,
                versions=current_versions,
            )
        if not self._current_matches(snapshot, current_versions, result):
            mismatches = self._current_mismatches(snapshot, current_versions, result)
            self.last_stale_current_mismatches = mismatches
            return self._drop(
                PopulationShadowDropReason.STALE_CURRENT_SNAPSHOT,
                evidence=evidence,
                token=token,
                versions=current_versions,
                stale_current_mismatches=mismatches,
            )
        if not self._belief_is_valid(belief):
            return self._drop(
                PopulationShadowDropReason.INVALID_BELIEF_SHAPE,
                evidence=evidence,
                token=token,
                versions=current_versions,
            )
        population_evidence = self._adapt_evidence(evidence, belief, native_targets)
        if population_evidence is None:
            return self._drop(
                PopulationShadowDropReason.INVALID_EVIDENCE_SHAPE,
                evidence=evidence,
                token=token,
                versions=current_versions,
            )
        decision = self.policy.decide(
            belief,
            token,
            population_evidence,
            capacity=capacity,
            snapshot_versions=current_versions.policy_versions(),
        )
        decision_token = self._next_decision_token
        self._next_decision_token += 1
        rejection_state = _AdapterRejectionState()
        native_target_count = 0
        native_target_independent = None
        if native_targets is not None:
            raw_count = getattr(native_targets, "count", 0)
            # ``SurfaceTargetBatch.count`` is an integer shape fact.  Do not
            # coerce arbitrary provider metadata here: a tensor-valued count
            # would turn this supposedly metadata-only path into a readback.
            if type(raw_count) is int:
                native_target_count = max(0, raw_count)
            candidate_independent = getattr(
                native_targets, "independent_view_count", None
            )
            if isinstance(candidate_independent, torch.Tensor):
                native_target_independent = candidate_independent
        self._publish(
            _PopulationShadowEvent(
                transaction_id=evidence.transaction_id,
                token=token,
                versions=current_versions,
                decision=decision,
                drop_reason=None,
                accumulator_status=(
                    getattr(result, "population_policy_e0_accumulator_status", None)
                    if isinstance(
                        getattr(result, "population_policy_e0_accumulator_status", None), str
                    )
                    else None
                ),
                e0_exposure_count=belief.observability.meaningful_exposure_count,
                e0_support_count=belief.observability.independent_support_views,
                target_gain=population_evidence.target_gain,
                decision_token=decision_token,
                rejection_state=rejection_state,
                native_target_count=native_target_count,
                native_target_independent_view_count=native_target_independent,
            )
        )
        self._activation_pending.append(
            PopulationActivationObservation(
                transaction_id=evidence.transaction_id,
                token=token,
                origin_snapshot=snapshot,
                versions=current_versions,
                decision=decision,
                decision_token=decision_token,
                rejection_state=rejection_state,
            )
        )
        if len(self._activation_pending) > self.max_pending_telemetry:
            self._activation_pending.popleft()
        for demand in decision.evidence_demands:
            if len(self._pending_evidence_demands) >= self.max_pending_telemetry:
                self._pending_evidence_demands.popleft()
            self._pending_evidence_demands.append(demand)
        return None


__all__ = [
    "NATIVE_TARGET_NO_TARGET",
    "NATIVE_TARGET_TARGET_UNQUALIFIED",
    "NATIVE_TARGET_SUPPORT_OK",
    "PopulationShadowDropReason",
    "PopulationActivationObservation",
    "PopulationShadowRuntime",
    "PopulationShadowSummary",
    "PopulationShadowVersions",
]
