"""Fail-closed M6 surface targets from completed supervision evidence.

This is the production bridge between one completed B1/B4 supervision
transaction and live population activation.  It deliberately consumes only
the already-rendered geometry leases and already-resident source payloads;
the injected reducer owns depth/surface estimation, while this module owns
the bounded version/source/tier fence and caches only the latest exact batch.

It neither renders, uploads, synchronizes, selects a population action, nor
mutates a Gaussian row.  Missing or stale evidence clears the cache so live
activation ABSTAINS rather than reusing a previous transaction's target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from eligsir.contracts import SourceIdentity
from eligsir.mapping.evidence.depth_surface_adapter import (
    DepthSurfaceEvidenceInput,
    adapt_depth_surface_evidence,
)
from eligsir.mapping.evidence.surface_targets import (
    SurfaceTargetBatch,
    SurfaceTargetSnapshot,
)
from eligsir.mapping.integration.depth_render_reuse import (
    DepthInsertionRenderReuse,
    GeometryRenderLease,
    bind_population_target_render_reuse,
)
from eligsir.mapping.population_policy import (
    MutationWorkToken,
    PopulationDecision,
)
from eligsir.optimization.gaussian.supervision_evidence import (
    EvidenceSnapshotToken,
    SupervisionEvidenceBatch,
)
from eligsir.runtime.prepared_source import PreparedSourceLease


SurfaceReducer = Callable[
    [DepthInsertionRenderReuse, SupervisionEvidenceBatch, SourceIdentity],
    DepthSurfaceEvidenceInput | None,
]
TargetBinder = Callable[
    [
        PopulationDecision,
        SurfaceTargetBatch,
        EvidenceSnapshotToken,
        MutationWorkToken,
    ],
    tuple[PopulationDecision, SurfaceTargetBatch] | None,
]


@dataclass(frozen=True, slots=True)
class SurfaceTargetObservation:
    """Bounded telemetry for one exact target-observation attempt."""

    source_count: int
    target_count: int
    source_h2d_ops: int
    extra_rasterizer_calls: int

    def __post_init__(self) -> None:
        for name in (
            "source_count",
            "target_count",
            "source_h2d_ops",
            "extra_rasterizer_calls",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact integer")


@dataclass(frozen=True, slots=True)
class _CachedSurfaceTargets:
    """One short-lived batch tied to the supervision evidence token."""

    evidence_snapshot: EvidenceSnapshotToken
    targets: SurfaceTargetBatch


def _native_image_2d(value: torch.Tensor, height: int, width: int) -> torch.Tensor | None:
    """Return an exact native HxW view without transfer or materialization."""

    if tuple(value.shape) == (height, width):
        return value
    if tuple(value.shape) == (height, width, 1):
        return value[..., 0]
    if tuple(value.shape) == (1, height, width):
        return value[0]
    return None


def reduce_native_depth_surface_evidence(
    reuse: DepthInsertionRenderReuse,
    evidence: SupervisionEvidenceBatch,
    source: SourceIdentity,
) -> DepthSurfaceEvidenceInput | None:
    """Reduce one resident native RGB-D/render borrow to one surface target.

    Live mutation authorization is currently bounded to one action per epoch,
    so this reducer deliberately emits at most one target.  All ranking,
    perspective backprojection, and normal estimation remain on the existing
    tensor device; a normal needs distinct valid rays in both image axes.
    Invalid/empty geometry is represented by a non-finite payload which the
    existing attributed mutation fence rejects; no scalar readback is
    introduced merely to discover abstention.
    """

    if not isinstance(reuse, DepthInsertionRenderReuse):
        return None
    if not isinstance(evidence, SupervisionEvidenceBatch):
        return None
    if not isinstance(source, SourceIdentity):
        return None
    depth = reuse.depth_meters.detach()
    height, width = tuple(int(value) for value in depth.shape)
    if height <= 0 or width <= 0:
        return None
    alpha = _native_image_2d(reuse.alpha.detach(), height, width)
    rendered_depth = _native_image_2d(
        reuse.rendered_depth.detach(), height, width
    )
    intrinsics = reuse.camera_intrinsics
    if (
        alpha is None
        or rendered_depth is None
        or not isinstance(intrinsics, torch.Tensor)
        or intrinsics.shape != (3, 3)
        or intrinsics.device != depth.device
    ):
        return None

    intrinsics = intrinsics.detach().to(dtype=torch.float32)
    c2w = reuse.c2w.detach().to(dtype=torch.float32)
    sensor = depth.to(dtype=torch.float32)
    rendered = rendered_depth.to(dtype=torch.float32)
    support = alpha.to(dtype=torch.float32).clamp(min=0.0, max=1.0)
    conflict = (rendered - sensor).clamp_min(0.0)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    camera_valid = (
        torch.isfinite(intrinsics).all()
        & torch.isfinite(c2w).all()
        & (fx > 0)
        & (fy > 0)
    )
    valid = (
        torch.isfinite(sensor)
        & torch.isfinite(rendered)
        & torch.isfinite(support)
        & (sensor > 0)
        & camera_valid
    )
    # A map surface in front of the measured depth is occluded, rather than
    # negative support for a new target.  Likewise an already-covered,
    # depth-consistent point has no surface-target opportunity.  Preserve
    # abstention as an invalid device candidate; the mutation fence is the
    # sole owner that may turn that compact condition into a host decision.
    depth_tolerance = torch.maximum(sensor * 0.01, torch.full_like(sensor, 1e-3))
    candidate = (support < 1.0) | (conflict > depth_tolerance)
    candidate_valid = valid & candidate
    gain = (1.0 - support) + conflict
    scores = torch.where(candidate_valid, gain, torch.full_like(gain, float("-inf")))
    selected = torch.topk(scores.reshape(-1), k=1, largest=True, sorted=True).indices
    selected_valid = candidate_valid.reshape(-1).index_select(0, selected)
    z = sensor.reshape(-1).index_select(0, selected)
    ys_i = torch.div(selected, width, rounding_mode="floor")
    xs_i = selected.remainder(width)

    def backproject(
        cols: torch.Tensor, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_z = sensor[rows, cols]
        return (
            torch.stack(
                (
                    (cols.to(dtype=torch.float32) - cx) / fx * sample_z,
                    (rows.to(dtype=torch.float32) - cy) / fy * sample_z,
                    sample_z,
                ),
                dim=1,
            ),
            sample_z,
        )

    points_camera, _ = backproject(xs_i, ys_i)
    positions = points_camera @ c2w[:3, :3].transpose(0, 1) + c2w[:3, 3]
    xp = (xs_i + 1).clamp(max=width - 1)
    xm = (xs_i - 1).clamp(min=0)
    yp = (ys_i + 1).clamp(max=height - 1)
    ym = (ys_i - 1).clamp(min=0)
    left, left_z = backproject(xm, ys_i)
    right, right_z = backproject(xp, ys_i)
    up, up_z = backproject(xs_i, ym)
    down, down_z = backproject(xs_i, yp)

    def valid_depth(sample_z: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(sample_z) & (sample_z > 0) & camera_valid

    left_valid = (xs_i > 0) & valid_depth(left_z)
    right_valid = (xs_i < width - 1) & valid_depth(right_z)
    up_valid = (ys_i > 0) & valid_depth(up_z)
    down_valid = (ys_i < height - 1) & valid_depth(down_z)
    tangent_x_camera = torch.where(
        (left_valid & right_valid).unsqueeze(1),
        right - left,
        torch.where(right_valid.unsqueeze(1), right - points_camera, points_camera - left),
    )
    tangent_y_camera = torch.where(
        (up_valid & down_valid).unsqueeze(1),
        down - up,
        torch.where(down_valid.unsqueeze(1), down - points_camera, points_camera - up),
    )

    def normalize_nonzero(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = vector.abs().amax(dim=1, keepdim=True)
        finite_nonzero = (
            torch.isfinite(vector).all(dim=1, keepdim=True) & (scale > 0)
        )
        scaled = vector / torch.where(finite_nonzero, scale, torch.ones_like(scale))
        length = torch.linalg.vector_norm(scaled, dim=1, keepdim=True)
        valid = finite_nonzero & torch.isfinite(length) & (length > 0)
        return (
            scaled / torch.where(valid, length, torch.ones_like(length)),
            valid.squeeze(1),
        )

    tangent_x_unit, tangent_x_valid = normalize_nonzero(tangent_x_camera)
    tangent_y_unit, tangent_y_valid = normalize_nonzero(tangent_y_camera)
    normals_camera, normal_valid = normalize_nonzero(
        torch.linalg.cross(tangent_x_unit, tangent_y_unit, dim=1)
    )
    neighborhood_valid = (
        (left_valid | right_valid)
        & (up_valid | down_valid)
        & tangent_x_valid
        & tangent_y_valid
        & normal_valid
    )
    normals = torch.nn.functional.normalize(
        normals_camera @ c2w[:3, :3].transpose(0, 1), dim=1, eps=1e-8
    )
    tangent_x = (z / fx).abs().clamp_min(1e-5)
    tangent_y = (z / fy).abs().clamp_min(1e-5)
    selected_conflict = conflict.reshape(-1).index_select(0, selected)
    normal_uncertainty = (
        torch.maximum(tangent_x, tangent_y) * 0.25 + selected_conflict * 0.5
    ).clamp_min(1e-5)
    uncertainty = torch.stack(
        (tangent_x, tangent_y, normal_uncertainty), dim=1
    )

    selected_usable = selected_valid & neighborhood_valid
    usable = selected_usable.unsqueeze(1)
    invalid = torch.full_like(positions, float("nan"))
    positions = torch.where(usable, positions, invalid)
    normals = torch.where(usable, normals, invalid)
    uncertainty = torch.where(usable, uncertainty, invalid)
    # Surface occupancy selects the opportunity; a valid measured depth is
    # the independent support signal.  An uncovered pixel therefore remains
    # positively supported instead of being mistaken for missing evidence.
    selected_support = torch.ones_like(z)
    selected_conflict = torch.where(
        selected_usable, selected_conflict, torch.zeros_like(selected_conflict)
    )
    selected_support = torch.where(
        selected_usable, selected_support, torch.zeros_like(selected_support)
    )
    counts = selected_usable.to(dtype=torch.int32)
    source_key = int(source.fingerprint()[:8], 16) & 0x7FFFFFFF
    target_ids = selected.to(dtype=torch.int64) + source_key * (height * width)
    return DepthSurfaceEvidenceInput(
        positions=positions,
        normals=normals,
        uncertainty=uncertainty,
        support_evidence=selected_support,
        conflict_evidence=selected_conflict,
        independent_view_count=counts,
        repeated_exposure_count=counts.clone(),
        target_ids=target_ids,
    )


def bind_policy_surface_targets(
    decision: PopulationDecision,
    targets: SurfaceTargetBatch,
    _current_snapshot: EvidenceSnapshotToken,
    token: MutationWorkToken,
) -> tuple[PopulationDecision, SurfaceTargetBatch] | None:
    """Bind exact target IDs without choosing a population action.

    The authenticated policy envelope remains the sole action decision.  Its
    device action codes decide whether the supplied target belongs to CREATE,
    RELOCATE, or is ignored for RETIRE; this function performs no host scalar
    branch and returns the provider's exact target batch object.
    """

    if (
        not isinstance(decision, PopulationDecision)
        or not decision.has_authentic_structural_selection()
        or decision.structural_selection is None
        or not isinstance(targets, SurfaceTargetBatch)
        or not isinstance(token, MutationWorkToken)
        or targets.count != decision.structural_selection.max_actions
        or targets.count != token.units
    ):
        return None
    try:
        bound = decision.bind_surface_target_ids(
            targets.target_ids,
            target_snapshot=targets.snapshot,
            structural_action=None,
        )
    except (TypeError, ValueError):
        return None
    return bound, targets


class ProductionSurfaceTargetProvider:
    """Bind resident B1/B4 geometry evidence to one live-policy target batch.

    ``reducer`` receives borrowed normal-render/source values and returns
    already-reduced device depth evidence.  ``binder`` is the sole target-ID
    matcher: it may only derive the supplied policy decision by pairing it
    with this exact batch.  The production runtime independently verifies that
    derivative before a population plan can reach the mutation arbiter.
    """

    def __init__(
        self,
        *,
        reducer: SurfaceReducer,
        binder: TargetBinder,
        max_targets: int,
    ) -> None:
        if not callable(reducer):
            raise TypeError("reducer must be callable")
        if not callable(binder):
            raise TypeError("binder must be callable")
        if type(max_targets) is not int or max_targets <= 0:
            raise ValueError("max_targets must be a positive exact integer")
        self._reducer = reducer
        self._binder = binder
        self._max_targets = max_targets
        self._cached: _CachedSurfaceTargets | None = None
        self.last_observation = SurfaceTargetObservation(0, 0, 0, 0)

    @staticmethod
    def _surface_snapshot(token: EvidenceSnapshotToken) -> SurfaceTargetSnapshot:
        return SurfaceTargetSnapshot(
            map_content=token.map_content,
            topology=token.topology,
            coordinate_frame=token.coordinate_frame,
            schema=token.schema,
            surface_evidence=token.surface_evidence,
        )

    def _clear(self, *, source_count: int = 0) -> None:
        self._cached = None
        self.last_observation = SurfaceTargetObservation(source_count, 0, 0, 0)

    @staticmethod
    def _is_fixed_topology_post_step_transition(
        origin: EvidenceSnapshotToken, action_fence: EvidenceSnapshotToken
    ) -> bool:
        """Accept only the one-step E0-to-action-fence version transition."""

        map_advance = action_fence.map_content - origin.map_content
        parameter_advance = action_fence.parameter - origin.parameter
        return (
            map_advance in (0, 1)
            and parameter_advance == 1
            and action_fence.topology == origin.topology
            and action_fence.coordinate_frame == origin.coordinate_frame
            and action_fence.schema == origin.schema
            and action_fence.optimizer_update == origin.optimizer_update
            and action_fence.surface_evidence == origin.surface_evidence
            and action_fence.policy == origin.policy
        )

    @classmethod
    def _fence_stamped_targets(
        cls, targets: SurfaceTargetBatch, action_fence: EvidenceSnapshotToken
    ) -> SurfaceTargetBatch:
        """Wrap the exact cached tensors in the post-step action-fence snapshot."""

        return SurfaceTargetBatch(
            snapshot=cls._surface_snapshot(action_fence),
            positions=targets.positions,
            normals=targets.normals,
            uncertainty=targets.uncertainty,
            support_evidence=targets.support_evidence,
            conflict_evidence=targets.conflict_evidence,
            independent_view_count=targets.independent_view_count,
            repeated_exposure_count=targets.repeated_exposure_count,
            target_ids=targets.target_ids,
            source_modality=targets.source_modality,
            max_targets=targets.max_targets,
        )

    @staticmethod
    def _reprojection_support(
        positions: torch.Tensor,
        uncertainty: torch.Tensor,
        reuse: DepthInsertionRenderReuse,
    ) -> torch.Tensor:
        """Return one device support bit per anchor from one physical view."""

        sensor = reuse.depth_meters.detach().to(dtype=torch.float32)
        intrinsics = reuse.camera_intrinsics
        if not isinstance(intrinsics, torch.Tensor):
            return torch.zeros(
                positions.shape[0], dtype=torch.bool, device=positions.device
            )
        intrinsics = intrinsics.detach().to(dtype=torch.float32)
        c2w = reuse.c2w.detach().to(dtype=torch.float32)
        height, width = tuple(int(value) for value in sensor.shape)
        rotation = c2w[:3, :3]
        translation = c2w[:3, 3]
        points_camera = (positions - translation) @ rotation
        z_camera = points_camera[:, 2]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        xs = fx * points_camera[:, 0] / z_camera + cx
        ys = fy * points_camera[:, 1] / z_camera + cy
        camera_valid = (
            torch.isfinite(intrinsics).all()
            & torch.isfinite(c2w).all()
            & (fx > 0)
            & (fy > 0)
        )
        projected_valid = (
            torch.isfinite(points_camera).all(dim=1)
            & torch.isfinite(xs)
            & torch.isfinite(ys)
            & (z_camera > 0)
            & (xs >= 0)
            & (xs <= width - 1)
            & (ys >= 0)
            & (ys <= height - 1)
            & camera_valid
        )

        offsets = torch.arange(-1, 2, dtype=torch.int64, device=positions.device)
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        center_x = torch.round(xs).to(dtype=torch.int64).unsqueeze(1)
        center_y = torch.round(ys).to(dtype=torch.int64).unsqueeze(1)
        sample_x = (center_x + offset_x.reshape(1, -1)).clamp(0, width - 1)
        sample_y = (center_y + offset_y.reshape(1, -1)).clamp(0, height - 1)
        sample_depth = sensor[sample_y, sample_x]
        sample_valid = torch.isfinite(sample_depth) & (sample_depth > 0)
        sample_x_float = sample_x.to(dtype=torch.float32)
        sample_y_float = sample_y.to(dtype=torch.float32)
        sample_camera = torch.stack(
            (
                (sample_x_float - cx) / fx * sample_depth,
                (sample_y_float - cy) / fy * sample_depth,
                sample_depth,
            ),
            dim=2,
        )
        sample_world = sample_camera @ rotation.transpose(0, 1) + translation
        separation = torch.linalg.vector_norm(
            sample_world - positions.unsqueeze(1), dim=2
        )
        depth_tolerance = torch.maximum(
            sample_depth * 0.01, torch.full_like(sample_depth, 1e-3)
        )
        sample_radius = torch.sqrt(
            (sample_depth / fx).square()
            + (sample_depth / fy).square()
            + depth_tolerance.square()
        )
        anchor_radius = torch.linalg.vector_norm(uncertainty, dim=1).unsqueeze(1)
        matches = (
            sample_valid
            & projected_valid.unsqueeze(1)
            & (separation <= anchor_radius + sample_radius)
        )
        return torch.any(matches, dim=1)

    @staticmethod
    def _aggregate_distinct_views(
        batches: tuple[SurfaceTargetBatch, ...],
        reuses: tuple[DepthInsertionRenderReuse, ...],
        *,
        max_targets: int,
    ) -> SurfaceTargetBatch | None:
        """Form one K=1 candidate from bounded cross-view reprojection.

        Source identity is deliberately checked by ``observe_evidence`` before
        this method is called.  Each per-view top-1 remains only an anchor:
        physical support comes from reprojecting that anchor into the other
        resident native depth images, never from comparing unrelated top-1
        pixels.  Numeric consistency and selection stay on the resident device.
        """

        if len(batches) < 2 or len(reuses) != len(batches) or max_targets != 1:
            return None
        snapshot = batches[0].snapshot
        device = batches[0].device
        if any(
            batch.snapshot != snapshot or batch.device != device or batch.count != 1
            for batch in batches
        ) or any(
            reuse.depth_meters.device != device for reuse in reuses
        ):
            return None

        positions = torch.cat(tuple(batch.positions for batch in batches), dim=0)
        normals = torch.cat(tuple(batch.normals for batch in batches), dim=0)
        uncertainty = torch.cat(tuple(batch.uncertainty for batch in batches), dim=0)
        support = torch.cat(tuple(batch.support_evidence for batch in batches), dim=0)
        conflict = torch.cat(tuple(batch.conflict_evidence for batch in batches), dim=0)
        target_ids = torch.cat(tuple(batch.target_ids for batch in batches), dim=0)

        anchor_valid = (
            torch.isfinite(positions).all(dim=1)
            & torch.isfinite(normals).all(dim=1)
            & torch.isfinite(uncertainty).all(dim=1)
            & torch.isfinite(support)
            & torch.isfinite(conflict)
            & (support > 0)
            & (conflict >= 0)
            & (uncertainty >= 0).all(dim=1)
        )
        support_by_view = torch.stack(
            tuple(
                ProductionSurfaceTargetProvider._reprojection_support(
                    positions, uncertainty, reuse
                )
                for reuse in reuses
            ),
            dim=1,
        )
        support_by_view &= anchor_valid.unsqueeze(1)
        support_count = support_by_view.sum(dim=1).to(dtype=torch.int32)
        maximum_support = support_count.max()
        opportunity = support + conflict
        ranked_opportunity = torch.where(
            support_count == maximum_support,
            opportunity,
            torch.full_like(opportunity, float("-inf")),
        )
        selected = torch.argmax(ranked_opportunity).reshape(1)
        selected_position = positions.index_select(0, selected)
        selected_normal = normals.index_select(0, selected)
        selected_uncertainty = uncertainty.index_select(0, selected)
        selected_conflict = conflict.index_select(0, selected)
        selected_count = support_count.index_select(0, selected)
        qualified = selected_count >= 2
        target_mask = qualified.reshape(1, 1)
        scalar_mask = qualified.reshape(1)
        invalid_geometry = torch.full_like(selected_position, float("nan"))
        selected_id = target_ids.index_select(0, selected)
        zero_count = torch.zeros_like(selected_count)
        return SurfaceTargetBatch(
            snapshot=snapshot,
            positions=torch.where(target_mask, selected_position, invalid_geometry),
            normals=torch.where(target_mask, selected_normal, invalid_geometry),
            uncertainty=torch.where(target_mask, selected_uncertainty, invalid_geometry),
            support_evidence=torch.where(
                scalar_mask,
                selected_count.to(dtype=support.dtype),
                torch.zeros((1,), dtype=support.dtype, device=device),
            ),
            conflict_evidence=torch.where(
                scalar_mask,
                selected_conflict,
                torch.zeros((1,), dtype=conflict.dtype, device=device),
            ),
            independent_view_count=torch.where(
                scalar_mask, selected_count, zero_count
            ),
            repeated_exposure_count=torch.where(
                scalar_mask,
                selected_count,
                zero_count,
            ),
            # Keep the deterministic policy pairing intact even for an
            # unqualified candidate.  The mutation fence then attributes the
            # abstention to its one packed readback instead of failing early
            # through a target-ID side channel.
            target_ids=selected_id,
            source_modality=torch.full((1,), 4, dtype=torch.int8, device=device),
            max_targets=1,
        )

    def observe_evidence(
        self,
        *,
        evidence: SupervisionEvidenceBatch | None,
        source_leases: Mapping[SourceIdentity, PreparedSourceLease],
    ) -> SurfaceTargetBatch | None:
        """Reduce only exact native geometry leases from this transaction.

        A failed source or an unavailable/asynchronous surface reducer clears
        the complete cache.  Partial multi-view targets would otherwise make
        B4 target provenance depend on evaluation order rather than one
        complete evidence transaction.
        """

        self._clear()
        if not isinstance(evidence, SupervisionEvidenceBatch):
            return None
        if not isinstance(source_leases, Mapping):
            raise TypeError("source_leases must be a mapping")
        token = evidence.snapshot_token
        if not isinstance(token, EvidenceSnapshotToken):
            return None
        leases = evidence.geometry_render_leases
        if not leases:
            return None
        if any(not isinstance(lease, GeometryRenderLease) for lease in leases):
            return None
        expected_sources = tuple(
            source
            for source, modality, eligibility in zip(
                evidence.source_identities,
                evidence.modalities,
                evidence.modality_eligibility,
            )
            if (
                modality.geometry_evidence_enabled
                and eligibility.geometry_population_target
            )
        )
        target_sources = set(expected_sources)
        population_leases = tuple(
            lease for lease in leases if lease.source_identity in target_sources
        )
        actual_sources = tuple(lease.source_identity for lease in population_leases)
        # A true B4 result can have a non-primary view skipped after the
        # rasterizer returns (for example a non-finite loss).  That partial
        # evidence is useful for optimization telemetry but never a complete
        # geometry transaction.  Require the exact eligible identity set and
        # one lease per source before the reducer can see any payload.
        if (
            not 2 <= len(expected_sources) <= 4
            or len(set(expected_sources)) != len(expected_sources)
            or len(actual_sources) != len(expected_sources)
            or len(set(actual_sources)) != len(actual_sources)
            or set(actual_sources) != set(expected_sources)
        ):
            return None
        eligible_by_source = {
            source: eligibility
            for source, modality, eligibility in zip(
                evidence.source_identities,
                evidence.modalities,
                evidence.modality_eligibility,
            )
            if (
                modality.geometry_evidence_enabled
                and eligibility.geometry_population_target
            )
        }
        if any(
            render_lease.modality_eligibility != eligible_by_source.get(
                render_lease.source_identity
            )
            for render_lease in population_leases
        ):
            return None
        batches: list[SurfaceTargetBatch] = []
        reuses: list[DepthInsertionRenderReuse] = []
        source_count = 0
        lease_by_source = {lease.source_identity: lease for lease in population_leases}
        for source in expected_sources:
            render_lease = lease_by_source[source]
            source_lease = source_leases.get(source)
            try:
                reuse, _reason = bind_population_target_render_reuse(
                    render_lease=render_lease,
                    source_lease=source_lease,
                    source=source,
                    expected_snapshot=token,
                    expected_resolution_tier="native",
                )
            except (TypeError, ValueError):
                self._clear(source_count=source_count)
                return None
            if reuse is None:
                self._clear(source_count=source_count)
                return None
            source_count += 1
            try:
                reduced = self._reducer(reuse, evidence, source)
            except Exception:
                self._clear(source_count=source_count)
                return None
            if not isinstance(reduced, DepthSurfaceEvidenceInput):
                self._clear(source_count=source_count)
                return None
            if reduced.device != reuse.alpha.device:
                self._clear(source_count=source_count)
                return None
            try:
                batch = adapt_depth_surface_evidence(
                    reduced,
                    snapshot=self._surface_snapshot(token),
                    max_targets=self._max_targets,
                )
            except (TypeError, ValueError):
                self._clear(source_count=source_count)
                return None
            batches.append(batch)
            reuses.append(reuse)
        targets = self._aggregate_distinct_views(
            tuple(batches), tuple(reuses), max_targets=self._max_targets
        )
        if targets is None:
            self._clear(source_count=source_count)
            return None
        self._cached = _CachedSurfaceTargets(token, targets)
        self.last_observation = SurfaceTargetObservation(
            source_count=source_count,
            target_count=targets.count,
            source_h2d_ops=0,
            extra_rasterizer_calls=0,
        )
        return targets

    def __call__(
        self,
        decision: PopulationDecision,
        origin_snapshot: EvidenceSnapshotToken,
        action_fence: EvidenceSnapshotToken,
        token: MutationWorkToken,
    ) -> tuple[PopulationDecision, SurfaceTargetBatch] | None:
        """Rebind one exact E0 target batch to its post-Adam action fence.

        ``origin_snapshot`` is the immutable supervision/render provenance;
        ``action_fence`` is the separately validated post-Adam vector that
        authorizes actuation.  The provider does not rewrite either object or
        any target tensor.  It only verifies that the cached target came from
        this exact E0 origin and that every non-parameter version survived to
        the action fence unchanged.
        """

        cached = self._cached
        if not isinstance(decision, PopulationDecision):
            return None
        if not isinstance(origin_snapshot, EvidenceSnapshotToken):
            return None
        if not isinstance(action_fence, EvidenceSnapshotToken):
            return None
        if not isinstance(token, MutationWorkToken):
            return None
        if cached is None:
            return None
        if cached.evidence_snapshot is not origin_snapshot:
            self._clear(source_count=self.last_observation.source_count)
            return None
        if cached.targets.snapshot != self._surface_snapshot(origin_snapshot):
            self._clear(source_count=self.last_observation.source_count)
            return None
        if not self._is_fixed_topology_post_step_transition(
            origin_snapshot, action_fence
        ):
            self._clear(source_count=self.last_observation.source_count)
            return None
        targets = self._fence_stamped_targets(cached.targets, action_fence)
        try:
            bound = self._binder(decision, targets, action_fence, token)
        except Exception:
            return None
        if not isinstance(bound, tuple) or len(bound) != 2:
            return None
        bound_decision, bound_targets = bound
        # The matcher cannot smuggle a second target pool or a stale clone
        # into activation.  It receives the authoritative borrowed batch and
        # must hand that exact object back to the already-hardened live fence.
        if not isinstance(bound_decision, PopulationDecision) or bound_targets is not targets:
            return None
        return bound_decision, targets


__all__ = [
    "bind_policy_surface_targets",
    "ProductionSurfaceTargetProvider",
    "reduce_native_depth_surface_evidence",
    "SurfaceReducer",
    "SurfaceTargetObservation",
    "TargetBinder",
]
