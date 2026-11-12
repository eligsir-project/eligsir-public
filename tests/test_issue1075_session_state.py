"""Contract tests for the streaming session state container (#1075, owner A).

Pins the partition that keeps a Gaussian model from being deepcopied off the
GPU, and the declared-registry facts the integration owner will build against.
No line of ``streaming_training()`` is touched.
"""

from __future__ import annotations

import pytest

from eligsir.mapping.branching import (
    AttributeStateProvider,
    BranchComponentState,
    CallableStateProvider,
)
from eligsir.runtime.session_state import (
    STREAMING_SLOTS,
    BackendOwnedSlotError,
    CaptureAPI,
    SessionState,
    SlotClass,
    SlotError,
    SlotSpec,
)


def _state(**values) -> SessionState:
    return SessionState(values=values)


# -- the trap this container exists to prevent ----------------------------


def test_backend_owned_slot_refuses_provider_capture():
    """AttributeStateProvider deepcopies what it is pointed at; over `gaussians`
    that is a silent multi-GB GPU->CPU copy per branch point."""
    state = _state(gaussians=object())

    with pytest.raises(BackendOwnedSlotError, match="backend-owned"):
        state.state_providers(["gaussians"])


@pytest.mark.parametrize(
    "name", ["gaussians", "background", "_deferred_stale_mask", "viewpoint_cam"]
)
def test_every_backend_owned_slot_is_refused(name):
    state = _state()
    with pytest.raises(BackendOwnedSlotError):
        state.state_providers([name])


def test_backend_owned_slots_are_excluded_from_the_default_provider_set():
    """The guard must hold on the bulk path too, not only when named."""
    state = _state()
    providers = state.state_providers()

    captured = set()
    for provider in providers:
        if isinstance(provider, AttributeStateProvider):
            captured.update(provider.attributes)
    for spec in STREAMING_SLOTS:
        if spec.slot_class is SlotClass.BACKEND_OWNED:
            assert spec.name not in captured


def test_a_backend_owned_slot_cannot_declare_a_capture_api():
    with pytest.raises(ValueError, match="must not declare a capture API"):
        SlotSpec(
            "x",
            SlotClass.BACKEND_OWNED,
            "s",
            reason="r",
            capture_api=CaptureAPI("state_dict", "load_state_dict"),
        )


def test_a_backend_owned_slot_must_state_why():
    with pytest.raises(ValueError, match="requires a reason"):
        SlotSpec("x", SlotClass.BACKEND_OWNED, "s")


# -- the registry's real content ------------------------------------------


def test_registry_declares_the_high_capture_collaborators():
    """`args` (16 closures) and `gaussians` (8) are the two most-captured names
    in the AST analysis; a registry missing either is not modelling the loop."""
    names = {spec.name for spec in STREAMING_SLOTS}
    for expected in ("args", "gaussians", "streaming_scene", "_cadence", "tb_writer"):
        assert expected in names


def test_seed_fade_monitor_is_the_only_declared_capture_api():
    """Of ~28 NATIVE_CAPTURE collaborators exactly one has a real round-trippable
    pair. If this ever fails, a reporting projection was mistaken for capture."""
    with_api = [
        spec.name
        for spec in STREAMING_SLOTS
        if spec.slot_class is SlotClass.NATIVE_CAPTURE and spec.capture_api is not None
    ]
    assert with_api == ["seed_fade_monitor"]


def test_every_uncapturable_native_slot_states_a_reason():
    for spec in STREAMING_SLOTS:
        if spec.slot_class is SlotClass.NATIVE_CAPTURE and spec.capture_api is None:
            assert spec.reason, f"{spec.name} must say why it cannot be captured"


def test_cross_thread_collaborators_are_native_not_bookkeeping():
    """Type does not determine the bucket: `_progress_frames` is a plain list but
    is appended from a daemon thread while the main loop reads it."""
    state = _state()
    for name in ("_progress_frames", "_pv_cpu_queue", "_async_state"):
        assert state.spec(name).slot_class is SlotClass.NATIVE_CAPTURE


def test_partition_control_roles_is_not_bookkeeping():
    """A small dataclass -- but it holds camera references, hence CUDA tensors."""
    assert (
        _state().spec("_partition_control_roles").slot_class
        is SlotClass.NATIVE_CAPTURE
    )


def test_loop_invariant_config_is_declared_but_not_captured():
    state = _state()
    for name in ("args", "pipe"):
        spec = state.spec(name)
        assert spec.slot_class is SlotClass.BOOKKEEPING
        assert spec.captured is False


def test_closure_rebound_slots_are_recorded():
    """The three `nonlocal`-written names need the container as their holder."""
    rebound = {s.name for s in STREAMING_SLOTS if s.rebound_by_closure}
    assert rebound == {
        "_deferred_stale_mask",
        "_deferred_stale_count",
        "_partition_control_roles",
    }
    assert all(s.rebound_in_loop for s in STREAMING_SLOTS if s.rebound_by_closure)


# -- providers actually satisfy the existing contract ---------------------


def test_bookkeeping_is_one_grouped_provider_that_round_trips():
    state = _state(iteration=100, total_inserted=7, n_frames_ingested=3)
    (provider,) = [
        p for p in state.state_providers() if isinstance(p, AttributeStateProvider)
    ]
    # RuntimeStateProvider is a non-runtime_checkable Protocol, so conformance
    # is asserted structurally -- which is also how branching.py consumes it.
    assert isinstance(provider.name, str)
    assert callable(provider.capture_branch_state)
    assert callable(provider.restore_branch_state)

    captured = provider.capture_branch_state()
    assert isinstance(captured, BranchComponentState)

    state.iteration = 999
    state.total_inserted = 0
    provider.restore_branch_state(captured)

    assert state.iteration == 100
    assert state.total_inserted == 7


def test_native_capture_provider_uses_the_collaborators_own_methods():
    class _Monitor:
        def __init__(self):
            self.value = 1

        def state_dict(self):
            return {"value": self.value}

        def load_state_dict(self, payload):
            self.value = payload["value"]

    monitor = _Monitor()
    state = _state(seed_fade_monitor=monitor)
    (provider,) = state.state_providers(["seed_fade_monitor"])
    assert isinstance(provider, CallableStateProvider)

    captured = provider.capture_branch_state()
    monitor.value = 42
    provider.restore_branch_state(captured)

    assert monitor.value == 1


def test_provider_binding_is_deferred_until_capture_time():
    """Slots are filled across ~4000 lines of setup, so a container must be
    constructible before its collaborators exist."""
    state = _state()  # seed_fade_monitor is unset
    (provider,) = state.state_providers(["seed_fade_monitor"])  # no raise

    class _Monitor:
        def state_dict(self):
            return {"ok": True}

        def load_state_dict(self, payload):
            self.restored = payload

    state.seed_fade_monitor = _Monitor()
    assert provider.capture_branch_state().payload == {"ok": True}


def test_asking_for_an_uncapturable_slot_by_name_fails_loudly():
    """Skipping is silent only when declared; an explicit request must not
    quietly return nothing and read as success."""
    state = _state()
    with pytest.raises(SlotError, match="no capture API"):
        state.state_providers(["streaming_scene"])


def test_uncapturable_enumerates_the_real_gap():
    gaps = dict(_state().uncapturable())
    assert "streaming_scene" in gaps and gaps["streaming_scene"]
    assert "seed_fade_monitor" not in gaps  # it has a real pair
    assert "gaussians" not in gaps  # covered by the backend checkpoint


# -- slot discipline ------------------------------------------------------


def test_undeclared_slot_assignment_is_refused():
    """Silent slot creation is how the partition would stop covering a real
    collaborator."""
    state = _state()
    with pytest.raises(SlotError, match="undeclared slot"):
        state.some_new_thing = object()


def test_undeclared_slot_read_is_an_attribute_error():
    with pytest.raises(AttributeError):
        _state().some_new_thing


def test_undeclared_initial_value_is_refused():
    with pytest.raises(SlotError, match="not a declared slot"):
        SessionState(values={"nope": 1})


def test_declared_slots_default_to_none_and_are_writable():
    state = _state()
    assert state.streaming_scene is None
    sentinel = object()
    state.streaming_scene = sentinel
    assert state.streaming_scene is sentinel


def test_duplicate_slot_declaration_is_refused():
    spec = SlotSpec("dup", SlotClass.BOOKKEEPING, "s")
    with pytest.raises(ValueError, match="duplicate slot"):
        SessionState(specs=(spec, spec))


def test_registry_has_no_duplicate_names():
    names = [spec.name for spec in STREAMING_SLOTS]
    assert len(names) == len(set(names))
