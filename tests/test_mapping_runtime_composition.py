from __future__ import annotations

from types import SimpleNamespace
import inspect
import sys

import numpy as np
import pytest

from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.materialize import NumpyBackend
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.application import create_mapping_application
from eligsir.runtime.config import (
    DepthInsertionPolicy,
    DepthLossPolicy,
    IngressTelemetrySnapshot,
    MappingRuntimeConfig,
    MappingRuntimeDependencies,
    RuntimeBinding,
)
from eligsir.contracts.control import (
    DepthInsertionPlan,
    DispositionKind,
    ModalityEligibility,
    ObservationDisposition,
    ResidencyIntent,
)
from eligsir.tracking.providers import ExternalTrackingProvider, observation_from_pose_matrix


def _packet() -> SensorPacket:
    source = SourceIdentity("dataset", 0, 0, 1_000_000_000)
    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8), encoding="rgb8"),
        depth=BufferRef.from_numpy(
            np.ones((2, 2), dtype=np.float32), encoding="32FC1", units="m"
        ),
        intrinsics=CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0, width=2, height=2),
        tracking=observation_from_pose_matrix(source, np.eye(4), provider="external"),
    )


def test_config_dependency_composition_is_dataset_free_and_constructs_one_application():
    mapped = []
    ticks = []
    config = MappingRuntimeConfig(
        model_layout="legacy",
        renderer="gsplat",
        numerical_mode="selective_adam",
        depth_loss=DepthLossPolicy(weight=0.25),
        depth_insertion=DepthInsertionPolicy(min_depth_m=0.1, max_depth_m=5.0),
    )
    ingress = []
    dependencies = MappingRuntimeDependencies(
        runtime_factory=lambda received: RuntimeBinding(
            mapper_sink=lambda frame, observation: mapped.append((received, frame, observation)),
            optimizer_tick=lambda: ticks.append(1),
        ),
        tracking_provider=ExternalTrackingProvider(),
        ingress_telemetry_provider=lambda packet, observation: (
            ingress.append(packet.source) or IngressTelemetrySnapshot()
        ),
    )

    application = create_mapping_application(config=config, dependencies=dependencies)
    outcome = application.session.submit(_packet())
    application.tick()

    assert outcome.accepted
    assert mapped[0][0] is config
    assert mapped[0][1] is outcome.frame
    assert ticks == [1]
    assert ingress == [_packet().source]
    assert not hasattr(config, "dataset")


def test_runtime_config_rejects_process_owned_or_non_scalar_configuration():
    with pytest.raises(TypeError, match="scalar"):
        MappingRuntimeConfig(
            model_layout="gsplat",
            renderer="gsplat",
            numerical_mode="adam",
            static_options={"dataset": SimpleNamespace()},
        )


def test_canonical_runtime_constructor_accepts_only_config_and_backend_compatibility():
    from eligsir.runtime.production_application import (
        CanonicalMappingRuntime,
    )

    parameters = tuple(inspect.signature(CanonicalMappingRuntime).parameters)
    assert parameters == ("config", "compatibility")


def test_backend_compatibility_does_not_carry_raw_runtime_args():
    from dataclasses import fields

    from eligsir.runtime.production_application import MappingBackendCompatibility

    assert {field.name for field in fields(MappingBackendCompatibility)} == {
        "dataset",
        "opt",
        "pipe",
    }


def test_runtime_config_freezes_canonical_device_capabilities():
    from eligsir.runtime.config import CanonicalDeviceCapabilities

    capabilities = CanonicalDeviceCapabilities(
        accelerator="cuda",
        gpu_frame_prep=True,
        gpu_frame_resident=True,
        tensor_candidate_path=True,
    )
    config = MappingRuntimeConfig(
        model_layout="gsplat",
        renderer="gsplat",
        numerical_mode="selective_adam",
        device_capabilities=capabilities,
    )

    assert config.device_capabilities is capabilities
    with pytest.raises((AttributeError, TypeError)):
        capabilities.gpu_frame_prep = False


def test_runtime_config_freezes_pipeline_and_optimizer_policies():
    from eligsir.runtime.config import (
        CanonicalOptimizationPolicy,
        CanonicalPipelinePolicy,
    )

    optimization = CanonicalOptimizationPolicy({"iterations": 100})
    pipeline = CanonicalPipelinePolicy({"sh_impl": "gsplat"})
    config = MappingRuntimeConfig(
        model_layout="gsplat",
        renderer="gsplat",
        numerical_mode="selective_adam",
        optimization=optimization,
        pipeline=pipeline,
    )

    assert config.optimization.iterations == 100
    assert config.pipeline.sh_impl == "gsplat"
    with pytest.raises(TypeError):
        config.pipeline.values["sh_impl"] = "python"


def test_canonical_runtime_has_no_raw_args_behavior_authority():
    from pathlib import Path

    source = Path("eligsir/runtime/production_application.py").read_text()
    forbidden = (
        "compatibility.args",
        "self.args",
        "backpressure_config_from_args",
        "keyframe_admitter_proposal_provider_from_args",
        "validate_accelerator_live_insertion_configuration(args)",
        "self._backend_compatibility",
        "self.dataset =",
        "self.opt = opt",
        "self.pipe = pipe",
    )
    assert not [text for text in forbidden if text in source]
    assert source.count("compatibility.opt") == 1
    assert "self.opt = config.optimization" in source
    assert "self.pipe = config.pipeline" in source
    assert "optimizer_update_permitted=True" not in source
    assert "optimizer_update_permitted=optimizer_update_permitted" in source
    assert "optimizer_iteration_limit=2**63 - 1" not in source


def test_canonical_resolver_denies_retired_engine_key(monkeypatch):
    from eligsir.cli.config import ConfigError
    from eligsir.runtime import production_application

    monkeypatch.setattr(
        production_application,
        "load_cli_config",
        lambda **_kwargs: {"streaming_optimization_engine": False},
    )
    with pytest.raises(ConfigError, match="streaming_optimization_engine"):
        production_application._resolve_parameters("ignored", None)


def test_causal_outer_resolver_creates_source_config_and_one_runtime_binding(monkeypatch):
    from eligsir.runtime import causal_entry

    events = []
    source = object()
    config = MappingRuntimeConfig(
        model_layout="gsplat", renderer="gsplat", numerical_mode="adam"
    )

    class _Compatibility:
        def __init__(self, **values):
            self.values = values

    class _Runtime:
        observation_owner = object()

        def __init__(self, received_config, compatibility):
            events.append(("runtime", received_config, compatibility))

        def map_frame(self, *_): pass
        def tick(self): pass
        def mark_source_complete(self): pass
        def flush(self): pass
        def close(self): pass
        def control_snapshot(self, *_): pass
        def record_disposition(self, *_): pass

    monkeypatch.setattr(
        causal_entry,
        "make_frame_source",
        lambda source_path, args: events.append(("source", source_path, args)) or source,
    )
    monkeypatch.setattr(
        causal_entry, "mapping_runtime_config_from_args", lambda args: config
    )
    monkeypatch.setitem(
        sys.modules,
        "eligsir.runtime.production_application",
        SimpleNamespace(
            CanonicalMappingRuntime=_Runtime,
            MappingBackendCompatibility=_Compatibility,
        ),
    )
    args = object()
    result = causal_entry.resolve_causal_mapping_composition(
        dataset=SimpleNamespace(source_path="capture"),
        opt=object(),
        pipe=object(),
        args=args,
    )

    assert result.source is source
    assert result.config is config
    assert result.binding.mapper_sink.__self__.__class__ is _Runtime
    assert [event[0] for event in events] == ["source", "runtime"]


def test_causal_prepared_options_route_dispositions_to_prepared_runtime_hook():
    from eligsir.runtime.causal_entry import (
        CausalMappingComposition,
        causal_mapping_application_options,
    )

    calls = []

    class _Runtime:
        def map_frame(self, *_): pass
        def tick(self): pass
        def mark_source_complete(self): pass
        def flush(self): pass
        def close(self): pass
        def control_snapshot(self, *_): pass
        def record_disposition(self, value): calls.append(("legacy", value))
        def record_prepared_disposition(self, value):
            calls.append(("prepared", value))
        def wait_for_progress(self, *_, **__): pass

    runtime = _Runtime()
    capture = lambda packet: packet
    binding = RuntimeBinding(
        mapper_sink=runtime.map_frame,
        optimizer_tick=runtime.tick,
        source_complete_hook=runtime.mark_source_complete,
        flush_hook=runtime.flush,
        close_hook=runtime.close,
        observation_owner=object(),
        control_snapshot_provider=runtime.control_snapshot,
        disposition_sink=runtime.record_disposition,
        packet_capture=capture,
    )
    prepared = SimpleNamespace(
        mark_source_complete=runtime.mark_source_complete,
        handoff=object(),
        bridge=object(),
        tick=runtime.tick,
        wait_for_progress=runtime.wait_for_progress,
    )
    options = causal_mapping_application_options(
        CausalMappingComposition(
            source=object(),
            config=SimpleNamespace(),
            binding=binding,
            runtime=runtime,
            prepared_ready_path=prepared,
        ),
        args=SimpleNamespace(),
        ingress_telemetry=None,
    )

    marker = object()
    options.disposition_sink(marker)
    assert calls == [("prepared", marker)]
    assert options.packet_capture is capture

    legacy = causal_mapping_application_options(
        CausalMappingComposition(
            source=object(),
            config=SimpleNamespace(),
            binding=binding,
            runtime=runtime,
            prepared_ready_path=None,
        ),
        args=SimpleNamespace(causal_prepared_ready_b4=False),
        ingress_telemetry=None,
    )
    assert legacy.mapper_sink.__self__ is runtime
    assert legacy.packet_capture is capture


def test_depth_policies_carry_static_method_semantics_not_tick_budgets():
    loss = DepthLossPolicy(loss_type="huber", weight=0.25, robust_delta=0.1)
    insertion = DepthInsertionPolicy(
        min_depth_m=0.1,
        max_depth_m=5.0,
        temporal_tolerance_m=0.02,
        temporal_relative_tolerance=0.05,
        edge_threshold=0.2,
        max_view_angle_deg=70.0,
        occupancy_distance_m=0.03,
        scale_mode="knn",
    )

    assert loss.loss_type == "huber"
    assert insertion.scale_mode == "knn"
    assert not hasattr(insertion, "seed_budget")


def test_configured_factory_rejects_legacy_options_or_production_kwargs():
    config = MappingRuntimeConfig(
        model_layout="gsplat", renderer="gsplat", numerical_mode="adam"
    )
    dependencies = MappingRuntimeDependencies(
        runtime_factory=lambda _: RuntimeBinding(
            mapper_sink=lambda *_: None,
            optimizer_tick=lambda: None,
        ),
        tracking_provider=ExternalTrackingProvider(),
        ingress_telemetry_provider=lambda packet, observation: IngressTelemetrySnapshot(),
    )

    with pytest.raises(TypeError, match="cannot be combined"):
        create_mapping_application(
            config=config, dependencies=dependencies, profile="live/canonical_mapping"
        )


def test_production_runtime_binding_derives_every_actuator_from_one_runtime_instance(tmp_path, monkeypatch):
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import eligsir.runtime.runtime_binding; "
            "assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr

    from eligsir.runtime import output_staging
    from eligsir.runtime.runtime_binding import runtime_binding_from_runtime

    monkeypatch.setattr(output_staging, "is_network_filesystem", lambda _path: False)
    monkeypatch.setattr(output_staging, "_is_tmpfs_mount", lambda _path: False)

    class _Runtime:
        observation_owner = object()

        def map_frame(self, *_): pass
        def tick(self): pass
        def mark_source_complete(self): pass
        def flush(self): pass
        def close(self): pass
        def control_snapshot(self, *_): pass
        def record_disposition(self, *_): pass
        def capture_packet_for_history(self, packet): return packet

        def configure_prepared_cold_payload_spill(self, root):
            self.spill_root = root

    runtime = _Runtime()
    model_path = tmp_path / "model"
    model_path.mkdir()
    binding = runtime_binding_from_runtime(
        runtime,
        args=SimpleNamespace(model_path=str(model_path), output_staging_root=None),
    )

    assert binding.mapper_sink.__self__ is runtime
    assert binding.optimizer_tick.__self__ is runtime
    assert binding.source_complete_hook.__self__ is runtime
    assert binding.control_snapshot_provider.__self__ is runtime
    assert binding.disposition_sink.__self__ is runtime
    assert binding.packet_capture.__self__ is runtime
    # The ordinary B4-off runtime remains CPU materialization; a device-only
    # packet needs a separately configured transfer/materialization contract.
    assert isinstance(binding.materialization_backend, NumpyBackend)
    assert runtime.spill_root.startswith(str(model_path / "cold_payloads"))


def test_modality_eligibility_and_depth_insertion_plan_are_identity_bearing_and_immutable():
    source = _packet().source
    eligibility = ModalityEligibility(
        rgb_supervision=True,
        rgb_quality="qualified",
        depth_supervision=False,
        depth_quality="missing",
        depth_insertion=True,
    )
    disposition = ObservationDisposition(
        source=source,
        snapshot_id=3,
        kind=DispositionKind.ADMIT,
        observe=True,
        mapping_candidate=True,
        admit_supervision=True,
        allow_geometry_insertion=True,
        residency=ResidencyIntent.HOT,
        reason="rgb and geometry are independently eligible",
        modality_eligibility=eligibility,
    )
    plan = DepthInsertionPlan(
        source=source,
        enabled=True,
        sample_stride=2,
        candidate_budget=32,
        seed_budget=8,
        voxel_size_m=0.02,
        minimum_confidence=0.5,
    )

    assert disposition.modality_eligibility == eligibility
    assert plan.source == source
    assert plan.max_seeds == 8
    with pytest.raises((AttributeError, TypeError)):
        plan.seed_budget = 9


def test_enabled_depth_insertion_plan_requires_an_eligible_source_identity():
    with pytest.raises(ValueError, match="source"):
        DepthInsertionPlan(
            source=None,
            enabled=True,
            sample_stride=1,
            candidate_budget=1,
            seed_budget=1,
            voxel_size_m=0.02,
            minimum_confidence=0.0,
        )


def test_disposition_rejects_geometry_authorization_without_modality_eligibility():
    source = _packet().source
    with pytest.raises(ValueError, match="modality"):
        ObservationDisposition(
            source=source,
            snapshot_id=3,
            kind=DispositionKind.ADMIT,
            observe=True,
            mapping_candidate=True,
            admit_supervision=True,
            allow_geometry_insertion=True,
            residency=ResidencyIntent.HOT,
            reason="inconsistent compatibility fields",
            modality_eligibility=ModalityEligibility(
                rgb_supervision=True,
                rgb_quality="qualified",
                depth_supervision=False,
                depth_quality="missing",
                depth_insertion=False,
            ),
        )
