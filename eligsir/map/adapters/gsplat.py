"""Owner-safe gsplat backend adapter.

Exactly one backend instance owns version/cumulative mutation state for a live
model. Constructing an adapter for the same model again returns that canonical
owner instead of silently resetting version counters in a parallel wrapper.
"""

from __future__ import annotations

from threading import RLock
from typing import TYPE_CHECKING, Any

from eligsir.map.contracts import CheckpointBlob, GaussianOptimizationSession
from eligsir.map.mutation import GsplatMutationTransaction
from eligsir.map.snapshots import MapSnapshot, SnapshotRequest
from eligsir.map.views import (
    GaussianMapOutputSample,
    GaussianMapReadView,
    read_view_from_model,
    sample_output_from_model,
)
from eligsir.optimization.gaussian.supervision_evidence import EvidenceSnapshotToken

if TYPE_CHECKING:
    from scene.gsplat_model import GsplatGaussianModel


class _GsplatOptimizationSession:
    """Fixed-topology optimizer facade over one canonical gsplat backend."""

    def __init__(self, backend: "GsplatMapBackend | GsplatGaussianModel"):
        """Bind a canonical owner, accepting the historical model-only facade."""
        self._backend = backend if isinstance(backend, GsplatMapBackend) else None
        self._model = backend._model if self._backend is not None else backend
        self._parameter_version = 0

    @property
    def model(self):
        """Return the optimizer-owned live model."""
        return self._model

    @property
    def parameter_version(self) -> int:
        """Return the number of optimizer steps issued through this session."""
        return self._parameter_version

    def capture_branch_state(self) -> int:
        """Capture the session-local version absent from the map checkpoint."""
        return int(self._parameter_version)

    def restore_branch_state(self, parameter_version: int) -> None:
        """Restore the parameter fence paired with a model branch restore."""
        if isinstance(parameter_version, bool) or not isinstance(parameter_version, int):
            raise TypeError("parameter_version must be an int")
        self._parameter_version = parameter_version

    def restore_live_branch_state(self, parameter_version: int) -> None:
        """Restore accepted optimizer values without reusing a rejected fence."""

        if isinstance(parameter_version, bool) or not isinstance(parameter_version, int):
            raise TypeError("parameter_version must be an int")
        self._parameter_version = max(self._parameter_version, parameter_version) + 1

    def zero_grad(self) -> None:
        """Clear all optimizer gradients using the model's canonical optimizer."""
        self._model.optimizer.zero_grad(set_to_none=True)

    def step(self, visibility=None) -> None:
        """Apply one fixed-topology step and advance visible-content versions.

        Optimization changes means/SH/opacity/scale values without changing row
        identity/count/order or the map/world coordinate frame.  Therefore only
        ``map_content_version`` advances here; topology/coordinate/schema clocks
        remain unchanged.  Keeping this on the canonical backend makes evidence
        snapshots and replay fingerprints observe optimizer-written state rather
        than treating all fixed-topology training steps as one frozen map.
        """
        lock = None if self._backend is None else self._backend.state_lock
        if lock is None:
            self._model.optimizer.step(visibility)
            self._parameter_version += 1
        else:
            with lock:
                self._model.optimizer.step(visibility)
                self._parameter_version += 1
                self._backend._map_content_version += 1


    def update_learning_rate(self, iteration: int) -> float | None:
        """Delegate learning-rate scheduling to the model."""
        return self._model.update_learning_rate(iteration)


class GsplatMapBackend:
    """Canonical live-state owner around one ``GsplatGaussianModel``."""

    _MODEL_OWNER_ATTR = "_eligsir_map_backend"

    def __new__(cls, model: "GsplatGaussianModel"):
        """Return the already-bound backend when the model has one."""
        existing = getattr(model, cls._MODEL_OWNER_ATTR, None)
        if isinstance(existing, cls):
            return existing
        instance = super().__new__(cls)
        setattr(model, cls._MODEL_OWNER_ATTR, instance)
        return instance

    def __init__(self, model: "GsplatGaussianModel"):
        """Initialize canonical ownership exactly once per live model."""
        if getattr(self, "_owner_initialized", False):
            if self._model is not model:
                raise RuntimeError("canonical GsplatMapBackend cannot be rebound to another model")
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
        # Installed only by the explicit population-plan fence.  Until an
        # owner binds all eight versions, the backend fails closed rather than
        # reconstructing a partial token from its older counters.
        self._population_plan_snapshot: EvidenceSnapshotToken | None = None
        self._population_hypothesis_uids: tuple[str, ...] | None = None
        self._population_next_hypothesis_uid: int | None = None

    @property
    def population_count(self) -> int:
        """Return the owned model's row count in O(1), without a read view."""
        return int(self._model.get_xyz.shape[0])

    def read_view(self) -> GaussianMapReadView:
        """Return a read-only logical view stamped with current map versions."""
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
        """Compatibility alias for :attr:`map_content_version`."""
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
        """Return the world/map coordinate-frame correction version."""
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
        """Record only hard-safety external count/order mutations."""
        inserted = max(int(inserted), 0)
        pruned = max(int(pruned), 0)
        if inserted == 0 and pruned == 0:
            return
        self._cumulative_inserted += inserted
        self._cumulative_pruned += pruned
        self._topology_version += 1
        self._map_content_version += 1
        self._invalidate_population_identity_snapshot()

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
        if structural:
            self._invalidate_population_identity_snapshot()
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
        """Create a host snapshot with one packed device-to-host transfer."""
        import numpy as np
        import torch

        view = self.read_view()
        sh = self._model.get_features if request.include_sh else view.sh0
        fields = [view.means, view.scales, view.quaternions, view.opacities, sh]
        n = int(view.means.shape[0])
        if request.max_gaussians is not None:
            n = min(n, max(int(request.max_gaussians), 0))
            fields = [x[:n] for x in fields]

        shapes = [tuple(x.shape[1:]) for x in fields]
        widths = [int(x[0].numel()) if n else int(np.prod(s)) for x, s in zip(fields, shapes)]
        flat = [x.reshape(n, -1) for x in fields]
        packed = torch.cat(flat, dim=1).detach().to("cpu").numpy()

        out = []
        start = 0
        for shape, width in zip(shapes, widths):
            part = packed[:, start : start + width]
            out.append(part.reshape((n,) + shape))
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
        """Open a fixed-topology optimizer session for this live model."""
        return _GsplatOptimizationSession(self)

    def begin_mutation(self) -> GsplatMutationTransaction:
        """Open a mutation transaction at the backend's current versions."""
        return GsplatMutationTransaction(
            self._model,
            self._map_content_version,
            self._topology_version,
            self._coordinate_frame_version,
            self._schema_version,
            population_hypothesis_uids=self._population_hypothesis_uids,
            population_next_hypothesis_uid=self._population_next_hypothesis_uid,
        )

    def install_population_plan_snapshot(self, snapshot: EvidenceSnapshotToken) -> None:
        """Bind the exact current all-version token before plan actuation.

        This is an explicit integration boundary: callers that only have the
        historical four backend counters cannot silently manufacture the
        optimizer/parameter/surface/policy parts of an atomic plan snapshot.
        """

        if not isinstance(snapshot, EvidenceSnapshotToken):
            raise TypeError("population-plan snapshot must be an EvidenceSnapshotToken")
        with self._state_lock:
            if (
                snapshot.map_content != self._map_content_version
                or snapshot.topology != self._topology_version
                or snapshot.coordinate_frame != self._coordinate_frame_version
                or snapshot.schema != self._schema_version
            ):
                raise ValueError("population-plan snapshot does not match current backend versions")
            self._population_plan_snapshot = snapshot
            # The snapshot is the explicit boundary at which live ownership
            # becomes active.  Before it, compatibility/reproduction writers
            # remain available; after it, only a complete population plan may
            # change population rows.
            self._model._population_plan_ownership_enabled = True

    def current_population_plan_snapshot(self) -> EvidenceSnapshotToken:
        """Return the backend's authoritative complete plan token or fail closed."""

        snapshot = self._population_plan_snapshot
        if snapshot is None:
            raise RuntimeError("population-plan snapshot has not been installed")
        if (
            snapshot.map_content != self._map_content_version
            or snapshot.topology != self._topology_version
            or snapshot.coordinate_frame != self._coordinate_frame_version
            or snapshot.schema != self._schema_version
        ):
            raise RuntimeError("population-plan snapshot is stale relative to backend versions")
        return snapshot

    def current_population_plan_registry_snapshot(self):
        """Attest the exact live gsplat row-state descriptor inventory."""

        from eligsir.map.population_mutation_plan import PopulationMutationPlan

        with self._state_lock:
            return PopulationMutationPlan.snapshot_registry(
                self._model.build_row_state_registry()
            )

    def install_population_identity_snapshot_from_bootstrap(self) -> None:
        """Install the host identity ledger from authoritative bootstrap scalars."""

        import torch
        from eligsir.map.row_state import GaussianRowStateRegistry

        with self._state_lock:
            registry = self._model.build_row_state_registry()
            if not isinstance(registry, GaussianRowStateRegistry):
                raise TypeError("population-plan row registry is not canonical")
            descriptors = {
                descriptor.name: descriptor for descriptor in registry.descriptors
            }
            descriptor = descriptors.get("provenance.gaussian_uid")
            uids = getattr(self._model, "gaussian_uid", None)
            means = getattr(self._model, "get_xyz", None)
            next_uid = getattr(self._model, "_next_gaussian_uid", None)
            seed_range = getattr(self._model, "_seed_uid_range", None)
            if (
                descriptor is None
                or descriptor.shape_suffix != ()
                or descriptor.dtype is not torch.int64
                or not isinstance(uids, torch.Tensor)
                or uids.ndim != 1
                or uids.dtype is not torch.int64
                or not isinstance(means, torch.Tensor)
                or means.ndim != 2
                or uids.shape[0] != means.shape[0]
                or uids.device != means.device
                or type(next_uid) is not int
                or next_uid < 0
                or not isinstance(seed_range, tuple)
                or len(seed_range) != 2
                or any(type(value) is not int for value in seed_range)
                or seed_range[0] < 0
                or seed_range[1] != next_uid
                or seed_range[1] - seed_range[0] != means.shape[0]
            ):
                raise ValueError("population-plan bootstrap identity scalars are malformed")
            self._population_hypothesis_uids = tuple(
                str(uid) for uid in range(seed_range[0], seed_range[1])
            )
            self._population_next_hypothesis_uid = next_uid

    def _invalidate_population_identity_snapshot(self) -> None:
        self._population_hypothesis_uids = None
        self._population_next_hypothesis_uid = None

    def current_population_plan_identity_snapshot(self):
        """Return the exact registered host identity ledger without device reads."""

        import torch

        from eligsir.map.row_state import GaussianRowStateRegistry

        with self._state_lock:
            registry = self._model.build_row_state_registry()
            if not isinstance(registry, GaussianRowStateRegistry):
                raise TypeError("population-plan row registry is not canonical")
            descriptors = {
                descriptor.name: descriptor for descriptor in registry.descriptors
            }
            descriptor = descriptors.get("provenance.gaussian_uid")
            uids = getattr(self._model, "gaussian_uid", None)
            means = getattr(self._model, "get_xyz", None)
            ledger = self._population_hypothesis_uids
            next_uid = self._population_next_hypothesis_uid
            if (
                descriptor is None
                or descriptor.shape_suffix != ()
                or descriptor.dtype is not torch.int64
                or not isinstance(uids, torch.Tensor)
                or uids.ndim != 1
                or uids.dtype is not torch.int64
                or not isinstance(means, torch.Tensor)
                or means.ndim != 2
                or uids.shape[0] != means.shape[0]
                or uids.device != means.device
                or not isinstance(ledger, tuple)
                or len(ledger) != means.shape[0]
                or type(next_uid) is not int
                or any(
                    not isinstance(uid, str)
                    or not uid.isdecimal()
                    or str(int(uid)) != uid
                    for uid in ledger
                )
            ):
                raise ValueError("population-plan host UID ledger is unavailable or malformed")
            parsed = tuple(int(uid) for uid in ledger)
            if len(set(parsed)) != len(parsed) or (parsed and next_uid <= max(parsed)):
                raise ValueError("population-plan host UID ledger is not unique and monotonic")
            if getattr(self._model, "_next_gaussian_uid", None) != next_uid:
                raise ValueError("population-plan host UID watermark is stale")
            return registry, ledger, next_uid

    def commit_mutation(self, transaction: GsplatMutationTransaction):
        """Commit transaction results into the canonical version state."""
        with self._state_lock:
            next_snapshot = getattr(transaction, "_population_plan_next_snapshot", None)
            next_identity = getattr(
                transaction, "_population_plan_next_identity_snapshot", None
            )
            if next_snapshot is not None:
                if (
                    not isinstance(next_identity, tuple)
                    or len(next_identity) != 2
                    or not isinstance(next_identity[0], tuple)
                    or type(next_identity[1]) is not int
                    or len(next_identity[0]) != self.population_count
                    or any(
                        not isinstance(uid, str)
                        or not uid.isdecimal()
                        or str(int(uid)) != uid
                        for uid in next_identity[0]
                    )
                ):
                    raise RuntimeError("population-plan commit omitted its identity ledger")
                parsed = tuple(int(uid) for uid in next_identity[0])
                if (
                    len(set(parsed)) != len(parsed)
                    or (parsed and next_identity[1] <= max(parsed))
                    or getattr(self._model, "_next_gaussian_uid", None)
                    != next_identity[1]
                ):
                    raise RuntimeError("population-plan commit identity ledger is malformed")
            elif next_identity is not None:
                raise RuntimeError("non-population mutation carried a population identity ledger")
            change = transaction.commit()
            if next_snapshot is not None:
                if (
                    next_snapshot.map_content != change.map_content_version
                    or next_snapshot.topology != change.topology_version
                    or next_snapshot.coordinate_frame != change.coordinate_frame_version
                    or next_snapshot.schema != change.schema_version
                ):
                    raise RuntimeError("population-plan commit versions do not match transaction result")
            self._map_content_version = change.map_content_version
            self._topology_version = change.topology_version
            self._coordinate_frame_version = change.coordinate_frame_version
            self._schema_version = change.schema_version
            if next_snapshot is not None:
                self._population_plan_snapshot = next_snapshot
                self._population_hypothesis_uids = next_identity[0]
                self._population_next_hypothesis_uid = next_identity[1]
                self._attest_population_plan_post_commit(transaction)
            elif change.topology_version != transaction._topology_version:
                self._invalidate_population_identity_snapshot()
            self._cumulative_inserted += int(change.inserted)
            self._cumulative_pruned += int(change.pruned)
            return change

    def _attest_population_plan_post_commit(
        self, transaction: GsplatMutationTransaction
    ) -> None:
        """Verify the committed plan still owns one complete row population."""

        from eligsir.map.population_mutation_plan import PopulationMutationPlan

        expected_population = getattr(
            transaction, "_population_plan_expected_population", None
        )
        expected_registry = getattr(
            transaction, "_population_plan_registry_snapshot", None
        )
        if type(expected_population) is not int or expected_population < 0:
            raise RuntimeError("population-plan commit omitted expected population")
        if not isinstance(expected_registry, tuple):
            raise RuntimeError("population-plan commit omitted registry attestation")
        registry = self._model.attest_live_population_rows()
        live_registry = PopulationMutationPlan.snapshot_registry(registry)
        if live_registry != expected_registry:
            raise RuntimeError(
                "population-plan post-commit registry attestation does not match plan"
            )
        if self.population_count != expected_population:
            raise RuntimeError(
                "population-plan post-commit population does not match plan"
            )

    def checkpoint_state(self) -> CheckpointBlob:
        """Capture model payload together with every backend version counter."""
        return CheckpointBlob(
            layout="gsplat",
            payload=self._model.capture(),
            map_version=self._map_content_version,
            topology_version=self._topology_version,
            coordinate_frame_version=self._coordinate_frame_version,
            schema_version=self._schema_version,
        )

    def restore_checkpoint(self, blob: CheckpointBlob, training_args: Any) -> None:
        """Restore a gsplat checkpoint and its canonical version state."""
        if blob.layout != "gsplat":
            raise ValueError(
                f"GsplatMapBackend cannot restore a '{blob.layout}' checkpoint blob."
            )
        self._model.restore(blob.payload, training_args)
        self._map_content_version = blob.map_content_version
        self._topology_version = blob.topology_version
        self._coordinate_frame_version = blob.coordinate_frame_version
        self._schema_version = blob.schema_version
        self._invalidate_population_identity_snapshot()

    def restore_live_checkpoint(self, blob: CheckpointBlob, training_args: Any) -> None:
        """Restore accepted values while minting a new live version boundary."""

        with self._state_lock:
            attempted = (
                self._map_content_version,
                self._topology_version,
                self._coordinate_frame_version,
                self._schema_version,
            )
            self.restore_checkpoint(blob, training_args)
            self._map_content_version = max(attempted[0], blob.map_content_version) + 1
            self._topology_version = max(attempted[1], blob.topology_version) + 1
            self._coordinate_frame_version = max(
                attempted[2], blob.coordinate_frame_version
            ) + 1
            self._schema_version = max(attempted[3], blob.schema_version) + 1
