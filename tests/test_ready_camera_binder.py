"""CPU contract tests for the opt-in ready-lease camera binder."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    ModalityEligibility,
    SourceIdentity,
)
from eligsir.runtime.prepared_ingress import PreparedIngressControlToken
from eligsir.runtime.prepared_source import PreparedSourceLease, PreparedTierPayload
from eligsir.runtime.ready_camera_binder import bind_ready_camera
from eligsir.runtime.ready_only_tick import ReadySourceBinding, ReadySourceVersions
from eligsir.runtime.production_application import CanonicalMappingRuntime


class _Event:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.queries = 0

    def query(self) -> bool:
        self.queries += 1
        return self.ready


def _source() -> SourceIdentity:
    return SourceIdentity("ready-camera", 3, 17, 1_700_000_017)


def _payload(value: torch.Tensor, *, encoding: str, units: str | None = None) -> BufferRef:
    return BufferRef(
        data=value,
        spec=BufferSpec(
            shape=tuple(value.shape),
            dtype=str(value.dtype).removeprefix("torch."),
            memory_domain=MemoryDomain.OTHER_DEVICE,
            encoding=encoding,
            units=units,
            device=str(value.device),
            contiguous=value.is_contiguous(),
        ),
        ownership=BufferOwnership.BORROWED,
        owner_token=object(),
    )


def _lease(
    *, ready: bool = True, metadata: dict | None = None, source: SourceIdentity | None = None
) -> PreparedSourceLease:
    source = _source() if source is None else source
    rgb = torch.zeros((4, 5, 3), dtype=torch.float32)
    depth = torch.ones((4, 5), dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    K = torch.tensor(((8.0, 0.0, 2.0), (0.0, 9.0, 1.5), (0.0, 0.0, 1.0)))
    token = PreparedIngressControlToken(8, source, None, None)
    return PreparedSourceLease(
        source_identity=source,
        modality_eligibility=ModalityEligibility(
            rgb_supervision=True,
            depth_supervision=True,
            depth_insertion=True,
            geometry_population_target=True,
        ),
        source_metadata={
            "source_id": source.source_id,
            "stream_epoch": source.stream_epoch,
            "source_ordinal": source.source_ordinal,
            "timestamp_ns": source.timestamp_ns,
            "rgb_shape": (4, 5, 3),
            "depth_shape": (4, 5),
            "depth_units": "m",
            "depth_aligned_to_rgb": True,
            "camera_intrinsics": (
                8.0, 9.0, 2.0, 1.5, 5, 4,
                8.0, 9.0, 2.0, 1.5, 5, 4,
                1000.0, "", "", "", None, "K",
            ),
            "pose_c2w": tuple(tuple(float(item) for item in row) for row in c2w.tolist()),
            "calibration_attestation": (
                "prepared-calibration-v1",
                tuple(tuple(float(item) for item in row) for row in c2w.tolist()),
                ((8.0, 0.0, 2.0), (0.0, 9.0, 1.5), (0.0, 0.0, 1.0)),
            ),
            **(metadata or {}),
        },
        tier_preparation_available=("native",),
        borrowed_device_payloads={
            "rgb": _payload(rgb, encoding="rgb_float32_hwc"),
            "depth_meters": _payload(depth, encoding="depth_meters", units="m"),
            "c2w": _payload(c2w, encoding="c2w"),
            "camera_intrinsics": _payload(K, encoding="camera_intrinsics"),
        },
        device="cpu",
        ready_event=_Event(ready),
        control_token=token,
    )


def _binding(
    *, ready: bool = True, metadata: dict | None = None, source: SourceIdentity | None = None
) -> ReadySourceBinding:
    return ReadySourceBinding(
        _lease(ready=ready, metadata=metadata, source=source),
        ReadySourceVersions(7, 11, 13),
    )


def test_ready_binder_requires_authoritative_native_metadata() -> None:
    binding = _binding(metadata={"depth_aligned_to_rgb": False})

    with pytest.raises(ValueError, match="aligned"):
        bind_ready_camera(binding)


def test_ready_binder_reuses_device_payloads_without_cold_materialization_or_h2d(monkeypatch) -> None:
    binding = _binding()
    cold = SimpleNamespace(calls=0)

    def forbidden(*_args, **_kwargs):
        cold.calls += 1
        raise AssertionError("ready binder entered a cold or transfer path")

    monkeypatch.setattr("scene.cameras.Camera.materialize_pending_image", forbidden)
    monkeypatch.setattr("scene.cameras.Camera.rematerialize_native_image", forbidden)

    bound = bind_ready_camera(binding)

    assert cold.calls == 0
    assert bound.camera.original_image.data_ptr() == binding.lease.payloads["rgb"].data.data_ptr()
    assert bound.camera._sensor_depth_cache is binding.lease.payloads["depth_meters"].data
    assert bound.camera.gsplat.K is binding.lease.payloads["camera_intrinsics"].data


@pytest.mark.parametrize(
    ("tier", "shape", "intrinsics"),
    (
        ("half", (2, 2), (3.2, 4.5, 0.5, 0.5)),
        ("eighth", (1, 1), (1.6, 2.25, 0.0, 0.0)),
    ),
)
def test_ready_binder_consumes_the_exact_prepared_tier_without_materialization(
    monkeypatch, tier, shape, intrinsics
) -> None:
    binding = _binding()
    native = PreparedTierPayload(
        name="native",
        rgb=binding.lease.payloads["rgb"],
        depth=binding.lease.payloads["depth_meters"],
        camera_intrinsics=binding.lease.payloads["camera_intrinsics"],
        width=5,
        height=4,
        intrinsics=(8.0, 9.0, 2.0, 1.5),
    )
    height, width = shape
    tier_rgb = torch.zeros((height, width, 3), dtype=torch.float32)
    tier_depth = torch.ones((height, width), dtype=torch.float32)
    tier_K = torch.tensor(
        ((intrinsics[0], 0.0, intrinsics[2]), (0.0, intrinsics[1], intrinsics[3]), (0.0, 0.0, 1.0))
    )
    prepared = PreparedTierPayload(
        name=tier,
        rgb=_payload(tier_rgb, encoding="rgb_float32_hwc"),
        depth=_payload(tier_depth, encoding="depth_meters", units="m"),
        camera_intrinsics=_payload(tier_K, encoding="camera_intrinsics"),
        width=width,
        height=height,
        intrinsics=intrinsics,
        projection_matrix=_payload(torch.eye(4), encoding="projection_matrix"),
        full_proj_transform=_payload(torch.eye(4), encoding="full_proj_transform"),
    )
    common_geometry = {
        "world_view_transform": _payload(torch.eye(4), encoding="world_view_transform"),
        "camera_center": _payload(torch.zeros(3), encoding="camera_center"),
        "gsplat_viewmat": _payload(torch.eye(4), encoding="gsplat_viewmat"),
        "gsplat_camera_center": _payload(torch.zeros((1, 3)), encoding="gsplat_camera_center"),
    }
    lease = dataclasses.replace(
        binding.lease,
        tier_preparation_available=("native", tier),
        tier_payloads={"native": native, tier: prepared},
        borrowed_device_payloads={**binding.lease.payloads, **common_geometry},
    )
    monkeypatch.setattr(
        "eligsir.mapping.resolution_materialization.materialize_resolution_tier",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tick-local tier materialization")),
    )
    monkeypatch.setattr(
        "scene.cameras.torch.linalg.inv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tick-local camera inverse")),
    )

    bound = bind_ready_camera(ReadySourceBinding(lease, binding.versions), tier=tier)

    assert bound.prepared_tier == tier
    assert (bound.camera.width, bound.camera.height) == (width, height)
    assert bound.camera.original_image.data_ptr() == tier_rgb.permute(2, 0, 1).data_ptr()
    assert bound.camera.gsplat.K is tier_K


def test_ready_binder_rejects_prepared_tier_without_sealed_geometry() -> None:
    binding = _binding()
    native = PreparedTierPayload(
        name="native",
        rgb=binding.lease.payloads["rgb"],
        depth=binding.lease.payloads["depth_meters"],
        camera_intrinsics=binding.lease.payloads["camera_intrinsics"],
        width=5,
        height=4,
        intrinsics=(8.0, 9.0, 2.0, 1.5),
    )
    lease = dataclasses.replace(binding.lease, tier_payloads={"native": native})

    with pytest.raises(ValueError, match="sealed before READY"):
        bind_ready_camera(ReadySourceBinding(lease, binding.versions))


def test_ready_binder_preserves_control_token_and_jit_versions_without_envelope() -> None:
    binding = _binding()

    bound = bind_ready_camera(binding)

    assert bound.source_identity is binding.lease.source_identity
    assert bound.control_token is binding.lease.control_token
    assert bound.versions == ReadySourceVersions(7, 11, 13)
    assert bound.metadata.intrinsics.width == 5
    assert bound.metadata.pose_c2w == binding.lease.source_metadata["pose_c2w"]
    assert not hasattr(bound, "envelope")
    assert "GpuWorkEnvelope" not in type(bound).__annotations__


def test_ready_binder_uses_a_deterministic_full_identity_uid_not_source_ordinal() -> None:
    sources = tuple(
        SourceIdentity(f"camera-{index}", index, 17, 1_700_000_017)
        for index in range(4)
    )

    bound = tuple(bind_ready_camera(_binding(source=source)) for source in sources)
    repeated_first = bind_ready_camera(_binding(source=sources[0]))

    assert len({item.camera.uid for item in bound}) == 4
    assert bound[0].camera.uid == repeated_first.camera.uid


def test_ready_binder_rejects_identity_or_payload_shape_mismatch() -> None:
    binding = _binding(metadata={"source_ordinal": 18})

    with pytest.raises(ValueError, match="source metadata"):
        bind_ready_camera(binding)

    binding = _binding(metadata={"depth_shape": (5, 4)})
    with pytest.raises(ValueError, match="depth shape"):
        bind_ready_camera(binding)


@pytest.mark.parametrize("depth_scale", [True, math.nan, math.inf, -math.inf, 0.0])
def test_ready_binder_rejects_nonfinite_or_nonpositive_depth_scale(depth_scale) -> None:
    intrinsics = list(_lease().source_metadata["camera_intrinsics"])
    intrinsics[12] = depth_scale

    with pytest.raises(ValueError, match="depth_scale"):
        bind_ready_camera(_binding(metadata={"camera_intrinsics": tuple(intrinsics)}))


def test_ready_binder_rejects_calibration_attestation_mismatch() -> None:
    binding = _binding(
        metadata={
            "calibration_attestation": (
                "prepared-calibration-v1",
                (
                    (1.0, 0.0, 0.0, 0.25),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                ((8.0, 0.0, 2.0), (0.0, 9.0, 1.5), (0.0, 0.0, 1.0)),
            )
        }
    )

    with pytest.raises(ValueError, match="calibration attestation"):
        bind_ready_camera(binding)


def test_ready_binder_keeps_c2w_payload_device_resident_for_preparation_attestation() -> None:
    binding = _binding()
    altered = binding.lease.payloads["c2w"].data.clone()
    altered[0, 3] = 0.25
    lease = dataclasses.replace(
        binding.lease,
        borrowed_device_payloads={
            **binding.lease.payloads,
            "c2w": _payload(altered, encoding="c2w"),
        },
    )

    assert bind_ready_camera(ReadySourceBinding(lease, binding.versions)).camera is not None


def test_ready_binder_keeps_K_payload_device_resident_for_preparation_attestation() -> None:
    binding = _binding()
    altered = binding.lease.payloads["camera_intrinsics"].data.clone()
    altered[0, 0] = 8.25
    lease = dataclasses.replace(
        binding.lease,
        borrowed_device_payloads={
            **binding.lease.payloads,
            "camera_intrinsics": _payload(altered, encoding="camera_intrinsics"),
        },
    )

    assert bind_ready_camera(ReadySourceBinding(lease, binding.versions)).camera is not None


def test_ready_binder_never_reduces_device_calibration_to_a_host_boolean(monkeypatch) -> None:
    binding = _binding()
    monkeypatch.setattr(
        torch, "allclose",
        lambda *_args, **_kwargs: pytest.fail("binder performed a host-sync comparison"),
    )

    assert bind_ready_camera(binding).camera is not None


def test_ready_binder_does_not_repoll_selected_readiness_event() -> None:
    binding = _binding()
    event = binding.lease.ready_event

    assert event.queries == 0
    assert bind_ready_camera(binding).camera is not None
    assert event.queries == 0


def test_runtime_hook_binds_camera_only_after_sampling_current_versions() -> None:
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._ready_only_tick_selector = object()
    binding = _binding()
    runtime.bind_ready_only_sources = lambda _selection: (binding,)

    cameras = runtime.bind_ready_only_cameras(object())

    assert cameras[0].versions == ReadySourceVersions(7, 11, 13)
    assert not hasattr(cameras[0], "envelope")
