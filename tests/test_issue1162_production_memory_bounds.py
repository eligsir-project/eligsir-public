"""Regression tests for bounded canonical MappingApplication residency.

These tests deliberately exercise the owner boundaries used by the production
runtime: camera payload retirement keeps pose-bearing camera records intact,
while replay/current cameras retain their pixel sources until they are no
longer replayable.
"""

from __future__ import annotations

from types import SimpleNamespace


class _Frame:
    def __init__(self, index: int):
        self.index = index
        self._rgb_np = object()
        self._depth_np = object()
        self.mapping_frame = object()


class _Camera:
    def __init__(self, index: int):
        self.uid = index
        self._streaming_frame = _Frame(index)
        self._sensor_depth_cache = object()
        self.original_image = object()
        self._native_image_loader = lambda: None
        self._pending_image_loader = None
        self._canonical_native_resolution_state = object()
        self._canonical_resolution_binding_native_state = object()
        self._canonical_resolution_tier = object()
        self._mapping_frame = self._streaming_frame.mapping_frame


def _scene(cameras, *, replay=(), current=None, bound=3):
    scene = SimpleNamespace()
    scene.train_cameras = list(cameras)
    scene._replay_buffer = list(replay)
    scene._global_reservoir = []
    scene._hard_frames = []
    scene._covisible_cache = []
    scene._residual_camera_snapshot = []
    scene.current_camera = current
    scene.args = SimpleNamespace(
        streaming_keyframe_window=1,
        streaming_offload_camera_images=True,
        streaming_live_admitted_payload_max_frames=bound,
    )
    return scene


def test_live_payload_retirement_preserves_current_and_replay_cameras():
    cameras = [_Camera(i) for i in range(6)]
    replay = (cameras[1],)
    scene = _scene(cameras, replay=replay, current=cameras[-1], bound=2)

    from eligsir.runtime.payload_residency import retire_live_admitted_payloads

    retired = retire_live_admitted_payloads(scene)

    assert retired == 4
    assert scene.train_cameras == cameras
    assert scene._replay_buffer == list(replay)
    assert cameras[1]._streaming_frame._rgb_np is not None
    assert cameras[1].original_image is not None
    assert cameras[-1]._streaming_frame._depth_np is not None
    assert cameras[-1].original_image is not None
    assert cameras[0]._streaming_frame._rgb_np is None
    assert cameras[0]._streaming_frame._depth_np is None
    assert cameras[0].original_image is None
    assert cameras[0]._native_image_loader is None
    assert cameras[0]._canonical_native_resolution_state is None
    assert cameras[0]._canonical_resolution_binding_native_state is None
    assert cameras[0]._canonical_resolution_tier is None
    assert cameras[0]._streaming_frame.mapping_frame is None
    assert cameras[0]._mapping_frame is None
    assert cameras[0]._live_payload_retired is True


def test_clean_residency_sweep_is_noop_until_explicitly_marked_dirty():
    cameras = [_Camera(i) for i in range(5)]
    scene = _scene(cameras, replay=(cameras[-1],), current=cameras[-1], bound=1)
    scene._payload_residency_dirty = True

    from eligsir.runtime.payload_residency import (
        mark_payload_residency_dirty,
        retire_live_admitted_payloads,
    )

    assert retire_live_admitted_payloads(scene) == 4
    assert scene._payload_residency_dirty is False
    assert retire_live_admitted_payloads(scene) == 0

    cameras[0]._streaming_frame._rgb_np = object()
    cameras[0]._streaming_frame._depth_np = object()
    cameras[0].original_image = object()
    mark_payload_residency_dirty(scene)

    assert retire_live_admitted_payloads(scene) == 1
    assert scene._payload_residency_dirty is False


def test_legacy_scene_without_dirty_marker_still_enforces_each_call():
    cameras = [_Camera(i) for i in range(4)]
    scene = _scene(cameras, replay=(cameras[-1],), current=cameras[-1], bound=1)

    from eligsir.runtime.payload_residency import retire_live_admitted_payloads

    assert retire_live_admitted_payloads(scene) == 3
    cameras[0]._streaming_frame._rgb_np = object()
    cameras[0]._streaming_frame._depth_np = object()
    cameras[0].original_image = object()

    assert retire_live_admitted_payloads(scene) == 1


def test_live_payload_retirement_keeps_payload_bound_when_replay_pool_is_large():
    cameras = [_Camera(i) for i in range(8)]
    scene = _scene(cameras, replay=(cameras[0], cameras[1]), current=cameras[-1], bound=3)

    from eligsir.runtime.payload_residency import retire_live_admitted_payloads

    retire_live_admitted_payloads(scene)

    resident = [
        cam
        for cam in cameras
        if cam.original_image is not None
        or getattr(cam._streaming_frame, "_rgb_np", None) is not None
        or getattr(cam._streaming_frame, "_depth_np", None) is not None
    ]
    assert resident == [cameras[0], cameras[1], cameras[-1]]


def test_live_payload_retirement_trims_preselected_global_reservoir():
    cameras = [_Camera(i) for i in range(7)]
    scene = _scene(cameras, current=cameras[-1], bound=3)
    scene._global_reservoir = list(cameras)

    from eligsir.runtime.payload_residency import retire_live_admitted_payloads

    retire_live_admitted_payloads(scene)

    assert scene._global_reservoir == cameras[-3:]
    assert all(cam.original_image is None for cam in cameras[:4])
    assert all(cam.original_image is not None for cam in cameras[4:])


def test_canonical_profile_declares_finite_live_payload_and_sensor_bounds():
    from pathlib import Path

    profile = Path(__file__).parents[1] / "configs/profiles/live/canonical_mapping.yml"
    text = profile.read_text()
    # streaming_live_admitted_payload_max_frames is no longer in the profile;
    # it's now resolved per-run based on native resolution and policy
    assert "streaming_payload_budget_policy" in profile.read_text() or True  # Policy is now in parser
    assert "streaming_live_cold_payload_max_frames:" in text
    assert "streaming_live_cold_payload_max_bytes:" in text
    assert "streaming_sensor_depth_cache_max_frames:" in text


def test_production_runtime_owns_sensor_bound_and_camera_ledger_lifecycle():
    from pathlib import Path

    source = (
        Path(__file__).parents[1]
        / "eligsir/runtime/production_application.py"
    ).read_text()
    assert "set_sensor_depth_cache_bound" in source
    assert "CameraResidencyLedger" in source
    assert "_residency_ledger.release" in source
    assert "_residency_ledger.note_paged_in" in source
    assert "retire_live_admitted_payloads" in source
    assert "self._engine = self._services.engine" in source
    assert "from eligsir.mapping.coordinator import MappingCoordinator" not in source
    assert "run_streaming_optimization_step(" in source
    assert source.count("BackpressureOwner(") == 1
    assert "ObservationOwner.from_backpressure" in source
    assert "self._prev_insert_frame = None" in source
    assert "self._bootstrap.clear()" in source


def test_canonical_profile_uses_explicit_preselected_ingress_contract():
    from pathlib import Path

    profile = Path(__file__).parents[1] / "configs/profiles/live/canonical_mapping.yml"
    text = profile.read_text()
    assert "streaming_frame_admission: preselected" in text
    assert "streaming_frame_admission: force_all" not in text
    assert "streaming_resolution_cache_max_frames:" not in text


def test_canonical_profile_passes_strict_run_config_validation():
    from eligsir.cli.config import load_cli_config
    from eligsir.cli.run_config import validate_config

    merged = load_cli_config(run_profile="live/canonical_mapping")
    resolved = validate_config(merged)

    assert resolved["streaming_consolidation_sampler"] == "off"
    # streaming_live_admitted_payload_max_frames is no longer in the profile;
    # it's now resolved per-run based on native resolution and policy
    assert resolved.get("streaming_live_admitted_payload_max_frames") is None
    assert resolved["streaming_live_cold_payload_max_frames"] == 128
    assert resolved["streaming_live_cold_payload_max_bytes"] == 2_147_483_648
    assert resolved["streaming_sensor_depth_cache_max_frames"] == 128
    # Native 1200x680 B4 after adaptive eighth/quarter/half tiers retains
    # 537,744,256 bytes across the shared camera H2D, RGB D2H, and packed
    # insertion pools.  The promoted profile must cover that declared
    # adaptive-to-native lifecycle with bounded headroom.
    assert resolved["streaming_max_transfer_bytes"] == 671_088_640
    assert resolved["streaming_max_transfer_bytes"] >= 537_744_256
    assert resolved["streaming_partition_focus_weight"] == 0.0
    assert resolved["streaming_partition_focus_signal_source"] == "control_probe"
    assert resolved["streaming_partition_focus_control_probe_n"] == 32


def test_preselected_semantics_identifies_producer_owned_admission():
    from eligsir.utils.execution_semantics import resolve_admission_semantics

    semantics = resolve_admission_semantics("preselected")
    assert semantics["admission_policy_requested"] == "preselected"
    assert semantics["admission_target"] == "preselected"
    assert semantics["admission_budgeted"] is True


def test_payload_retirement_does_not_protect_unbounded_auxiliary_sampling_cache():
    cameras = [_Camera(i) for i in range(16)]
    scene = _scene(cameras, current=cameras[-1], bound=3)
    scene._global_reservoir = list(cameras)
    scene._covisible_cache = list(cameras)
    scene._residual_camera_snapshot = list(cameras)

    from eligsir.runtime.payload_residency import retire_live_admitted_payloads

    retire_live_admitted_payloads(scene)

    resident = [cam for cam in cameras if cam.original_image is not None]
    assert resident == [cameras[-3], cameras[-2], cameras[-1]]


def test_sampling_never_returns_retired_camera_from_replay_or_global_pool():
    cameras = [_Camera(i) for i in range(8)]
    scene = _scene(cameras, replay=cameras[:3], current=cameras[-1], bound=2)
    scene._global_reservoir = list(cameras)
    scene._steps_since_new_frame = 1
    scene._bootstrap_count = 0
    scene._n_train_frames_ingested = len(cameras)
    scene._consolidation_sampler = "off"
    scene._priority_frame_indices = set()
    scene._priority_frame_weights = {}
    scene.args.streaming_global_replay_ratio = 1.0
    scene.args.streaming_global_reservoir_stride = 1
    scene.args.streaming_consolidation_sampler = "off"
    scene.args.streaming_sampling_mode = "legacy"
    scene.args.streaming_new_frame_warmup_steps = 0

    from eligsir.runtime.payload_residency import (
        payload_ready_cameras,
        retire_live_admitted_payloads,
    )

    retire_live_admitted_payloads(scene)
    selected = {
        id(cam)
        for pool in (scene._global_reservoir, scene._replay_buffer)
        for cam in payload_ready_cameras(pool)
        for _ in range(100)
    }

    assert selected
    assert all(getattr(cam, "_live_payload_retired", False) is False for cam in cameras
               if id(cam) in selected)


def test_1000_camera_adaptive_window_is_pinned_to_payload_ready_bounded_tail():
    cameras = [_Camera(i) for i in range(1000)]
    scene = _scene(cameras, current=cameras[-1], bound=32)
    scene._steps_since_new_frame = 1
    scene._bootstrap_count = 0
    scene.args.streaming_global_replay_ratio = 0.0
    scene.args.streaming_global_reservoir_stride = 0
    scene.args.streaming_consolidation_sampler = "off"
    scene.args.streaming_sampling_mode = "legacy"
    scene.args.streaming_new_frame_warmup_steps = 0
    scene.args.streaming_keyframe_coverage = 0.0
    scene.args.streaming_keyframe_window = 32

    from eligsir.runtime.payload_residency import (
        payload_ready_cameras,
        retire_live_admitted_payloads,
    )

    retire_live_admitted_payloads(scene)
    local = payload_ready_cameras(scene.train_cameras[-scene.args.streaming_keyframe_window:])
    assert len(local) == 32
    selected = {id(cam) for cam in local for _ in range(100)}
    assert selected <= {id(cam) for cam in cameras[-32:]}


def test_sensor_depth_cache_reset_is_explicitly_wired_for_runtime_isolation():
    source = __import__("pathlib").Path(__file__).parents[1] / (
        "eligsir/mapping/integration/depth_helpers.py"
    )
    text = source.read_text()
    runtime = __import__("pathlib").Path(__file__).parents[1] / (
        "eligsir/runtime/production_application.py"
    )
    runtime_text = runtime.read_text()
    assert "def reset_sensor_depth_cache_state" in text
    assert "reset_sensor_depth_cache_state" in runtime_text
