"""Owner-safe backend adapter for the legacy Graphdeco Gaussian layout.

Like the gsplat adapter, this class is canonical per live model: repeated
construction returns the same owner and therefore cannot reset map/topology
version state in a parallel wrapper.
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Any

import torch

from eligsir.map.contracts import CheckpointBlob, GaussianOptimizationSession
from eligsir.map.mutation import (
    GaussianInsertionBatch,
    TopologyChange,
    _anchor_buffer_is_populated,
    _validate_insertion_batch,
    _validate_mask,
)
from eligsir.map.snapshots import MapSnapshot, SnapshotRequest
from eligsir.map.views import (
    GaussianMapOutputSample,
    GaussianMapReadView,
    read_view_from_model,
    sample_output_from_model,
)

if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel


class _LegacyOptimizationSession:
    """Fixed-topology optimizer facade for the canonical legacy backend."""

    def __init__(self, backend: "LegacyGraphdecoMapBackend"):
        """Bind the canonical owner and start a local parameter-step counter."""
        self._backend = backend
        self._model = backend._model
        self._parameter_version = 0

    @property
    def model(self):
        """Return the optimizer-owned live model."""
        return self._model

    @property
    def parameter_version(self) -> int:
        """Return optimizer steps issued through this session."""
        return self._parameter_version

    def zero_grad(self) -> None:
        """Clear gradients through the legacy model's canonical optimizer."""
        self._model.optimizer.zero_grad(set_to_none=True)

    def step(self, visibility=None) -> None:
        """Apply one fixed-topology step and advance visible-content versions."""
        with self._backend.state_lock:
            if visibility is not None:
                self._model.optimizer.step(visibility)
            else:
                self._model.optimizer.step()
            self._parameter_version += 1
            self._backend._map_content_version += 1

    def update_learning_rate(self, iteration: int) -> float | None:
        """Delegate learning-rate scheduling to the model."""
        return self._model.update_learning_rate(iteration)


class LegacyGraphdecoMutationTransaction:
    """Sole transaction writer for the legacy raw-tensor layout."""

    def __init__(
        self,
        model: "GaussianModel",
        map_version: int,
        topology_version: int,
        coordinate_frame_version: int = 0,
        schema_version: int = 0,
    ):
        """Bind legacy storage and the versions this transaction starts from."""
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

    def _ensure_open(self) -> None:
        """Reject reuse after the transaction has been committed."""
        if self._committed:
            raise RuntimeError("mutation transaction has already been committed")

    def insert(self, batch: GaussianInsertionBatch) -> int:
        """Append an aligned insertion batch through the model primitive."""
        self._ensure_open()
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
                opacities_raw=batch.opacities_raw,
                sh_rest=batch.sh_rest,
                confidence=batch.confidence,
                source=batch.source,
            )
        )
        self._inserted += added
        if added:
            self._content_dirty = True
            self._topology_dirty = True
        return added

    def validate_insert(self, batch: GaussianInsertionBatch) -> None:
        _validate_insertion_batch(batch)

    def validate_prune(self, mask: "torch.Tensor") -> None:
        _validate_mask(mask, int(self._model._xyz.shape[0]), device=self._model._xyz.device)

    def validate_relocate(self, plan) -> None:
        _validate_mask(plan.mask, int(self._model._xyz.shape[0]), "relocation mask", device=self._model._xyz.device)
        targets = ((plan.new_means, self._model._xyz), (plan.new_raw_scales, self._model._scaling),
                   (plan.new_raw_opacities, self._model._opacity), (plan.new_quaternions, self._model._rotation))
        rows = set()
        for value, target in targets:
            if value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.device != target.device or value.shape[1:] != target.shape[1:]:
                raise ValueError("legacy relocation value must match parameter row shape/device")
            rows.add(value.shape[0])
        if len(rows) > 1 or (plan.mask.dtype != torch.bool and rows and rows != {plan.mask.numel()}):
            raise ValueError("legacy relocation rows must match selection")

    def validate_pose_correction(self, correction) -> None:
        target = self._model._xyz
        _validate_mask(correction.mask, int(target.shape[0]), "pose correction mask", device=target.device)
        if not isinstance(correction.new_positions, torch.Tensor) or correction.new_positions.device != target.device or correction.new_positions.shape[1:] != target.shape[1:]:
            raise ValueError("new_positions must match legacy means row shape/device")
        for name, anchor in (("new_anchor_xyz", getattr(self._model, "anchor_xyz", None)),
                             ("new_anchor_normal", getattr(self._model, "anchor_normal", None))):
            value = getattr(correction, name)
            if value is not None and (anchor is None or value.device != target.device or value.shape[0] != correction.new_positions.shape[0] or value.shape[1:] != anchor.shape[1:]):
                raise ValueError(f"{name} must match selected legacy anchor rows/device")

    def validate_opacity_update(self, mask, value) -> None:
        target = self._model._opacity
        _validate_mask(mask, int(target.shape[0]), "opacity mask", device=target.device)
        if not isinstance(value, torch.Tensor) or value.device != target.device or value.ndim == 0 or value.shape[1:] != target.shape[1:]:
            raise ValueError("opacity value must match legacy opacity row shape/device")

    def prune(self, mask: "torch.Tensor") -> None:
        """Remove selected rows and mark content/topology dirty."""
        self._ensure_open()
        before = int(self._model._xyz.shape[0])
        self._model.prune_points(mask)
        count = before - int(self._model._xyz.shape[0])
        if count < 0:
            raise RuntimeError("legacy prune unexpectedly increased population")
        if count == 0:
            return
        self._pruned += count
        self._content_dirty = True
        self._topology_dirty = True

    def relocate(self, plan) -> None:
        """Apply identity-preserving raw relocation for selected rows."""
        self._ensure_open()
        updates = (
            ("_xyz", plan.new_means), ("_scaling", plan.new_raw_scales),
            ("_opacity", plan.new_raw_opacities), ("_rotation", plan.new_quaternions),
        )
        originals = {
            name: getattr(self._model, name).data[plan.mask].clone()
            for name, value in updates if value is not None
        }
        written: list[str] = []
        count = 0
        try:
            for name, value in updates:
                if value is None:
                    continue
                target = getattr(self._model, name).data
                target[plan.mask] = value.reshape(target[plan.mask].shape)
                written.append(name)
                count = int(value.shape[0])
        except Exception:
            for name in reversed(written):
                getattr(self._model, name).data[plan.mask] = originals[name]
            raise
        wrote = bool(written)
        if wrote and count:
            self._relocated += count
            self._content_dirty = True

    def apply_pose_correction(self, correction) -> None:
        """Move xyz and every populated world-frame anchor atomically."""
        self._ensure_open()
        count = int(correction.new_positions.shape[0])
        if not count:
            return
        population = int(self._model._xyz.shape[0])
        has_anchor_xyz = _anchor_buffer_is_populated(self._model, "anchor_xyz", population)
        has_anchor_normal = _anchor_buffer_is_populated(self._model, "anchor_normal", population)
        if has_anchor_xyz and correction.new_anchor_xyz is None:
            raise ValueError("pose correction must update populated anchor_xyz atomically")
        if has_anchor_normal and correction.new_anchor_normal is None:
            raise ValueError("pose correction must update populated anchor_normal atomically")
        original_xyz = self._model._xyz.data[correction.mask].clone()
        original_anchor_xyz = self._model.anchor_xyz[correction.mask].clone() if has_anchor_xyz else None
        original_anchor_normal = self._model.anchor_normal[correction.mask].clone() if has_anchor_normal else None
        wrote_xyz = wrote_anchor_xyz = False
        try:
            self._model._xyz.data[correction.mask] = correction.new_positions
            wrote_xyz = True
            if has_anchor_xyz:
                self._model.anchor_xyz[correction.mask] = correction.new_anchor_xyz
                wrote_anchor_xyz = True
            if has_anchor_normal:
                self._model.anchor_normal[correction.mask] = correction.new_anchor_normal
        except Exception:
            if wrote_anchor_xyz:
                self._model.anchor_xyz[correction.mask] = original_anchor_xyz
            if wrote_xyz:
                self._model._xyz.data[correction.mask] = original_xyz
            if has_anchor_normal and original_anchor_normal is not None:
                # This assignment is harmless if the failing write never landed
                # and restores the exact prior value if a backend failed after it.
                self._model.anchor_normal[correction.mask] = original_anchor_normal
            raise
        self._corrected += count
        self._content_dirty = True
        self._coordinate_dirty = True

    def update_opacity_logits(self, mask: "torch.Tensor", value) -> None:
        """Apply a same-topology opacity-logit update to selected rows."""
        self._ensure_open()
        if mask.numel() == 0:
            return
        self._model._opacity.data[mask] = value
        self._content_dirty = True

    def commit(self) -> TopologyChange:
        """Close the transaction and derive independent version increments."""
        self._ensure_open()
        self._committed = True
        return TopologyChange(
            map_version=self._map_version + int(self._content_dirty),
            topology_version=self._topology_version + int(self._topology_dirty),
            coordinate_frame_version=(
                self._coordinate_frame_version + int(self._coordinate_dirty)
            ),
            schema_version=self._schema_version,
            inserted=self._inserted,
            pruned=self._pruned,
            relocated=self._relocated,
            corrected=self._corrected,
            content_changed=self._content_dirty,
        )


class LegacyGraphdecoMapBackend:
    """Canonical map backend for one legacy Graphdeco model."""

    _MODEL_OWNER_ATTR = "_eligsir_map_backend"

    def __new__(cls, model: "GaussianModel"):
        """Return the model's existing backend owner when already bound."""
        existing = getattr(model, cls._MODEL_OWNER_ATTR, None)
        if isinstance(existing, cls):
            return existing
        instance = super().__new__(cls)
        setattr(model, cls._MODEL_OWNER_ATTR, instance)
        return instance

    def __init__(self, model: "GaussianModel"):
        """Initialize canonical version and mutation state exactly once."""
        if getattr(self, "_owner_initialized", False):
            if self._model is not model:
                raise RuntimeError("canonical LegacyGraphdecoMapBackend cannot be rebound")
            return
        self._owner_initialized = True
        self._state_lock = RLock()
        self._model = model
        self._map_content_version = 0
        self._topology_version = 0
        self._coordinate_frame_version = 0
        self._schema_version = 0
        self._cumulative_inserted = 0
        self._cumulative_pruned = 0

    @property
    def population_count(self) -> int:
        """Return the owned model's row count in O(1), without a read view."""
        return int(self._model.get_xyz.shape[0])

    def read_view(self) -> GaussianMapReadView:
        """Return a logical read view stamped with current versions."""
        return read_view_from_model(
            self._model, self._map_content_version, self._topology_version
        )

    def sample_output(
        self, *, max_gaussians: int, parameter_version: int
    ) -> GaussianMapOutputSample:
        """Capture one bounded owned sample at the current version boundary."""
        with self._state_lock:
            return sample_output_from_model(
                self._model,
                max_gaussians=max_gaussians,
                map_content_version=self._map_content_version,
                topology_version=self._topology_version,
                parameter_version=parameter_version,
            )

    @property
    def state_lock(self):
        """Fence output sampling against canonical parameter/topology writers."""
        return self._state_lock

    @property
    def map_content_version(self) -> int:
        """Return the visible-content mutation/optimization version."""
        return self._map_content_version

    @property
    def map_version(self) -> int:
        """Return the compatibility alias for content version."""
        return self._map_content_version

    @property
    def _map_version(self) -> int:
        """Historical private alias retained during the content-version cutover."""
        return self._map_content_version

    @_map_version.setter
    def _map_version(self, value: int) -> None:
        self._map_content_version = int(value)

    @property
    def topology_version(self) -> int:
        """Return the identity/count/order topology version."""
        return self._topology_version

    @property
    def coordinate_frame_version(self) -> int:
        """Return the map/world coordinate-frame correction version."""
        return self._coordinate_frame_version

    @property
    def schema_version(self) -> int:
        """Return the storage schema/layout version."""
        return self._schema_version

    @property
    def cumulative_inserted(self) -> int:
        """Return rows inserted through the canonical backend."""
        return self._cumulative_inserted

    @property
    def cumulative_pruned(self) -> int:
        """Return rows pruned through the canonical backend."""
        return self._cumulative_pruned

    def record_external_mutation(self, *, inserted: int = 0, pruned: int = 0) -> None:
        """Record a hard-safety external structural mutation."""
        inserted = max(int(inserted), 0)
        pruned = max(int(pruned), 0)
        if inserted == 0 and pruned == 0:
            return
        self._cumulative_inserted += inserted
        self._cumulative_pruned += pruned
        self._topology_version += 1
        self._map_content_version += 1

    def record_delegated_mutation(
        self, *, inserted: int = 0, pruned: int = 0,
        content_changed: bool = True, topology_changed: bool = True,
        relocated: int = 0, corrected: int = 0,
    ):
        """Record delegated content/topology effects without net-count inference."""
        from eligsir.map.mutation import TopologyChange

        inserted = max(int(inserted), 0)
        pruned = max(int(pruned), 0)
        relocated = max(int(relocated), 0)
        corrected = max(int(corrected), 0)
        structural = bool(topology_changed or inserted or pruned)
        self._cumulative_inserted += inserted
        self._cumulative_pruned += pruned
        self._topology_version += int(structural)
        self._map_content_version += int(bool(content_changed or structural))
        return TopologyChange(
            map_version=self._map_content_version,
            topology_version=self._topology_version,
            coordinate_frame_version=self._coordinate_frame_version,
            schema_version=self._schema_version,
            inserted=inserted,
            pruned=pruned,
            relocated=relocated,
            corrected=corrected,
            content_changed=bool(content_changed or structural),
        )

    def create_snapshot(self, request: SnapshotRequest) -> MapSnapshot:
        """Create a packed host snapshot of selected Gaussian fields."""
        import numpy as np

        view = self.read_view()
        sh = self._model.get_features if request.include_sh else view.sh0
        fields = [view.means, view.scales, view.quaternions, view.opacities, sh]
        n = int(view.means.shape[0])
        if request.max_gaussians is not None:
            n = min(n, max(int(request.max_gaussians), 0))
            fields = [x[:n] for x in fields]
        shapes = [tuple(x.shape[1:]) for x in fields]
        widths = [int(x[0].numel()) if n else int(np.prod(s)) for x, s in zip(fields, shapes)]
        packed = torch.cat([x.reshape(n, -1) for x in fields], dim=1).detach().to("cpu").numpy()
        out = []
        start = 0
        for shape, width in zip(shapes, widths):
            out.append(packed[:, start : start + width].reshape((n,) + shape))
            start += width
        return MapSnapshot(
            positions=out[0],
            scales=out[1],
            quaternions=out[2],
            opacities=out[3],
            spherical_harmonics=out[4],
            map_version=self._map_content_version,
            purpose=request.purpose,
            extra={"packed_d2h": True, "packed_fields": 5},
        )

    def begin_optimization(self) -> GaussianOptimizationSession:
        """Open a fixed-topology optimizer session for the live model."""
        return _LegacyOptimizationSession(self)

    def begin_mutation(self) -> LegacyGraphdecoMutationTransaction:
        """Open a mutation transaction at current backend versions."""
        return LegacyGraphdecoMutationTransaction(
            self._model,
            self._map_content_version,
            self._topology_version,
            self._coordinate_frame_version,
            self._schema_version,
        )

    def commit_mutation(self, transaction: LegacyGraphdecoMutationTransaction):
        """Commit transaction results into the canonical version state."""
        change = transaction.commit()
        self._map_content_version = change.map_content_version
        self._topology_version = change.topology_version
        self._coordinate_frame_version = change.coordinate_frame_version
        self._schema_version = change.schema_version
        self._cumulative_inserted += int(change.inserted)
        self._cumulative_pruned += int(change.pruned)
        return change

    def checkpoint_state(self) -> CheckpointBlob:
        """Capture legacy model payload together with backend version counters."""
        return CheckpointBlob(
            layout="legacy_graphdeco",
            payload=self._model.capture(),
            map_version=self._map_content_version,
            topology_version=self._topology_version,
            coordinate_frame_version=self._coordinate_frame_version,
            schema_version=self._schema_version,
        )

    def restore_checkpoint(self, blob: CheckpointBlob, training_args: Any) -> None:
        """Restore a legacy checkpoint and its canonical version state."""
        if blob.layout != "legacy_graphdeco":
            raise ValueError(
                f"LegacyGraphdecoMapBackend cannot restore a '{blob.layout}' checkpoint blob."
            )
        self._model.restore(blob.payload, training_args)
        self._map_content_version = blob.map_content_version
        self._topology_version = blob.topology_version
        self._coordinate_frame_version = blob.coordinate_frame_version
        self._schema_version = blob.schema_version
