import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.map.contracts import CheckpointBlob
from eligsir.mapping.branching import AttributeStateProvider, BranchableMapperRuntime


class FakeBackend:
    def __init__(self):
        self.value = 1.0
        self._map_version = 2
        self._topology_version = 3
    @property
    def map_version(self): return self._map_version
    @property
    def topology_version(self): return self._topology_version
    def checkpoint_state(self):
        return CheckpointBlob("fake", copy.deepcopy(self.value), self._map_version, self._topology_version)
    def restore_checkpoint(self, blob, training_args):
        assert training_args.token == "args"
        self.value = copy.deepcopy(blob.payload)
        self._map_version = blob.map_version
        self._topology_version = blob.topology_version

    def restore_live_checkpoint(self, blob, training_args):
        previous_map = self._map_version
        previous_topology = self._topology_version
        self.restore_checkpoint(blob, training_args)
        self._map_version = max(previous_map, blob.map_version) + 1
        self._topology_version = max(previous_topology, blob.topology_version) + 1


def make_runtime():
    backend = FakeBackend()
    scheduler = SimpleNamespace(tick=4, history=[1, 2])
    provider = AttributeStateProvider(scheduler, ("tick", "history"), "scheduler")
    def ingest(observation):
        sample = random.random() + float(np.random.random()) + float(torch.rand(()))
        backend.value += float(observation) + sample
        backend._map_version += 1
        scheduler.tick += 1
        scheduler.history.append(float(observation))
        return {"sample": sample}
    def evaluate():
        return {"quality": backend.value, "tick": scheduler.tick}
    runtime = BranchableMapperRuntime(
        backend=backend,
        training_args=SimpleNamespace(token="args"),
        ingest_fn=ingest,
        evaluate_fn=evaluate,
        state_providers=(provider,),
        evaluation_id="views:0,4,8",
    )
    return runtime, backend, scheduler


def test_capture_mutate_restore_reproduces_state_and_rng():
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    runtime, backend, scheduler = make_runtime()
    state = runtime.checkpoint_state()
    first = runtime.ingest(2.0)
    first_quality = runtime.evaluate()["quality"]
    runtime.restore_state(state)
    assert runtime.state_fingerprint() == state.fingerprint
    second = runtime.ingest(2.0)
    assert second["sample"] == pytest.approx(first["sample"], abs=0.0)
    assert runtime.evaluate()["quality"] == pytest.approx(first_quality, abs=0.0)
    assert scheduler.tick == 5
    assert backend.map_version == 3


def test_repeated_restore_and_readonly_evaluation():
    runtime, _, scheduler = make_runtime()
    state = runtime.checkpoint_state()
    scheduler.tick = 99
    runtime.restore(state); assert scheduler.tick == 4
    scheduler.tick = 88
    runtime.restore(state); assert scheduler.tick == 4
    metrics = runtime.evaluate_fixed_views(verify_read_only=True)
    assert metrics["quality"] == 1.0
    assert scheduler.tick == 4


def test_live_restore_reinstates_values_without_reusing_candidate_versions():
    runtime, backend, scheduler = make_runtime()
    accepted = runtime.checkpoint_state()
    runtime.ingest(2.0)
    rejected_map_version = backend.map_version
    rejected_topology_version = backend.topology_version

    runtime.restore_live_state(accepted)

    assert backend.value == 1.0
    assert scheduler.tick == 4
    assert backend.map_version > rejected_map_version
    assert backend.topology_version > rejected_topology_version


def test_restored_branch_always_restores_caller_state():
    runtime, backend, scheduler = make_runtime()
    branch = runtime.checkpoint_state()
    backend.value = 10.0; scheduler.tick = 20
    caller = runtime.checkpoint_state()
    with pytest.raises(RuntimeError):
        with runtime.restored_branch(branch):
            runtime.ingest(1.0)
            raise RuntimeError("boom")
    assert runtime.state_fingerprint() == caller.fingerprint
    assert backend.value == 10.0
    assert scheduler.tick == 20
