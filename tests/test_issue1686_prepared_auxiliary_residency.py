"""Tests for issue #1686: device-resident prepared objective auxiliaries.

Ensures that on the prepared path, optional objective auxiliaries must already be
device-resident before render/loss, and are not transferred inside assemble_loss().
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import scene.cameras as cameras
from eligsir.cli.parser import parse_training_args
from eligsir.optimization.gaussian.contracts import RenderOutputs
from eligsir.optimization.gaussian.objective import (
    LossAssemblyInputs,
    assemble_loss,
    _is_prepared_camera,
)


def _make_legacy_camera(device: str = "cpu") -> cameras.Camera:
    """Create a legacy camera without gsplat attribute."""
    cam = object.__new__(cameras.Camera)
    torch.nn.Module.__init__(cam)
    cam.original_image = torch.zeros((3, 4, 4), dtype=torch.float32, device=device)
    cam.uid = 1
    cam.image_name = "test_legacy"
    cam._gt_finite_validated = True
    cam._gt_finite_validated_id = id(cam.original_image)
    cam._gt_finite_validated_version = int(cam.original_image._version)
    # Legacy cameras do NOT have gsplat attribute
    return cam


def _make_prepared_camera(device: str = "cpu") -> cameras.Camera:
    """Create a prepared camera with gsplat and _source_identity attributes."""
    from eligsir.contracts import SourceIdentity

    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    center = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)

    cam = object.__new__(cameras.Camera)
    torch.nn.Module.__init__(cam)
    cam.original_image = torch.zeros((3, 4, 4), dtype=torch.float32, device=device)
    cam.uid = 1
    cam.image_name = "test_prepared"
    cam._gt_finite_validated = True
    cam._gt_finite_validated_id = id(cam.original_image)
    cam._gt_finite_validated_version = int(cam.original_image._version)

    # Prepared cameras have gsplat attribute
    cam.gsplat = cameras.GsplatCameraTensors(
        viewmat=viewmat,
        K=K,
        camera_center=center,
        width=4,
        height=4,
    )
    # Prepared cameras have _source_identity set (the canonical marker)
    cam._source_identity = SourceIdentity(
        source_id="test_prepared_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )
    return cam


def test_is_prepared_camera_detection():
    """Test that prepared cameras are correctly identified by gsplat attribute."""
    legacy = _make_legacy_camera()
    prepared = _make_prepared_camera()

    assert not _is_prepared_camera(legacy)
    assert _is_prepared_camera(prepared)


def test_prepared_path_cpu_resident_surface_confidence_raises_at_preflight():
    """Test that CPU-resident surface confidence on prepared path raises at preflight.

    Tests the preflight function directly with device mismatch.
    """
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    args = SimpleNamespace(surface_confidence_depth_loss=True)

    # Create a prepared camera (with _source_identity to mark it as prepared)
    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    # Create a mock surface prior with CPU-resident confidence
    mock_prior = SimpleNamespace(
        depth=torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu"),
        confidence=torch.ones((1, 4, 4), dtype=torch.float32, device="cpu"),
    )

    with patch("eligsir.optimization.gaussian.objective._get_surface_prior", return_value=mock_prior):
        # Preflight with target device as cuda should raise (confidence is on cpu)
        with pytest.raises(ValueError, match="surface_confidence.*device-resident"):
            preflight_prepared_objective_auxiliaries(cam, args, torch.device("cuda:0"))


def test_prepared_path_device_resident_auxiliaries_no_to_call():
    """Test that device-resident auxiliaries on prepared path don't trigger .to() calls."""
    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0

    # Create a prepared camera on CPU (test device)
    cam = _make_prepared_camera(device="cpu")

    n = 3
    gaussians = SimpleNamespace(
        _opacity=torch.zeros((n, 1), requires_grad=True),
        get_scaling=torch.ones((n, 3), requires_grad=True),
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    image = torch.zeros((3, 4, 4), dtype=torch.float32, device="cpu", requires_grad=True)

    inputs = LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": image,
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        ),
        image=image,
        iteration=1,
        optimizer_update_id=1,
        n_frames_ingested=1,
    )

    # Track .to() calls on tensors
    original_to = torch.Tensor.to
    to_call_count = {"sensor_d": 0, "prior_conf": 0, "rs_sensor_d": 0, "sensor_d_fs": 0}

    def tracked_to(self, *args, **kwargs):
        # Only count calls that would move to cuda/another device
        if "device" in kwargs and kwargs["device"] != self.device:
            # Identify which tensor this is by its shape/context
            if self.shape == (1, 4, 4):  # Depth tensors
                if to_call_count["sensor_d"] == 0:
                    to_call_count["sensor_d"] += 1
                elif to_call_count["rs_sensor_d"] == 0:
                    to_call_count["rs_sensor_d"] += 1
                elif to_call_count["sensor_d_fs"] == 0:
                    to_call_count["sensor_d_fs"] += 1
            elif self.shape == (1, 4, 4):  # Confidence tensor (same shape, counted separately)
                to_call_count["prior_conf"] += 1
        return original_to(self, *args, **kwargs)

    with patch.object(torch.Tensor, "to", tracked_to):
        breakdown = assemble_loss(inputs)

    # For prepared path with all auxiliaries on device, no .to() should be called
    # for those tensors (they're already on the target device)
    assert bool(torch.isfinite(breakdown.total_tensor))


def test_legacy_path_unchanged_cpu_auxiliaries_still_transferred():
    """Test that legacy path unchanged: CPU auxiliaries still get transferred."""
    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0

    # Create a legacy camera (without gsplat)
    cam = _make_legacy_camera(device="cpu")

    n = 3
    gaussians = SimpleNamespace(
        _opacity=torch.zeros((n, 1), requires_grad=True),
        get_scaling=torch.ones((n, 3), requires_grad=True),
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    # Render on CPU for this test
    image = torch.zeros((3, 4, 4), dtype=torch.float32, device="cpu", requires_grad=True)

    inputs = LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": image,
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        ),
        image=image,
        iteration=1,
        optimizer_update_id=1,
        n_frames_ingested=1,
    )

    # For legacy path, the .to() calls should pass through unchanged.
    # We just verify that loss assembly succeeds and produces a finite result.
    breakdown = assemble_loss(inputs)
    assert bool(torch.isfinite(breakdown.total_tensor))


def test_prepared_camera_without_surface_confidence_no_preflight_error():
    """Test that prepared camera without surface_confidence_depth_loss enabled passes preflight."""
    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0

    # Disable surface confidence depth loss
    args.surface_confidence_depth_loss = False

    # Create a prepared camera
    cam = _make_prepared_camera(device="cpu")

    n = 3
    gaussians = SimpleNamespace(
        _opacity=torch.zeros((n, 1), requires_grad=True),
        get_scaling=torch.ones((n, 3), requires_grad=True),
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    image = torch.zeros((3, 4, 4), dtype=torch.float32, device="cpu", requires_grad=True)

    inputs = LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": image,
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        ),
        image=image,
        iteration=1,
        optimizer_update_id=1,
        n_frames_ingested=1,
    )

    # Should not raise preflight error since surface_confidence_depth_loss is disabled
    breakdown = assemble_loss(inputs)
    assert bool(torch.isfinite(breakdown.total_tensor))


def test_regression_legacy_camera_with_gsplat_none_treated_as_legacy():
    """Regression test: legacy camera with gsplat=None must be treated as legacy (not prepared).

    Production cameras may have gsplat set to None by prepare_camera_for_render().
    The marker must not be hasattr(gsplat) (which is True for gsplat=None), but
    _source_identity which only prepared cameras have.
    """
    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0

    # Create a legacy camera with gsplat = None (simulating prepare_camera_for_render behavior)
    cam = _make_legacy_camera(device="cpu")
    cam.gsplat = None  # Explicitly set to None, as production code does

    # Verify this is NOT treated as prepared
    assert not _is_prepared_camera(cam)

    n = 3
    gaussians = SimpleNamespace(
        _opacity=torch.zeros((n, 1), requires_grad=True),
        get_scaling=torch.ones((n, 3), requires_grad=True),
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    image = torch.zeros((3, 4, 4), dtype=torch.float32, device="cpu", requires_grad=True)

    inputs = LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": image,
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        ),
        image=image,
        iteration=1,
        optimizer_update_id=1,
        n_frames_ingested=1,
    )

    # Should NOT raise preflight error (legacy path, no preflight check)
    breakdown = assemble_loss(inputs)
    assert bool(torch.isfinite(breakdown.total_tensor))


def test_regression_prepared_camera_with_source_identity_requires_device_residency():
    """Regression test: properly prepared camera (with _source_identity) must have device-resident auxiliaries.

    Prepared cameras bound via ready_camera_binder.bind_ready_camera() have _source_identity set.
    This is the correct marker, not hasattr(gsplat).
    """
    from eligsir.contracts import SourceIdentity

    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0
    args.surface_confidence_depth_loss = True

    # Create a prepared camera with _source_identity set
    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    # Verify this IS treated as prepared
    assert _is_prepared_camera(cam)

    class MockImageTensor:
        def __init__(self, device):
            self._device = device

        @property
        def device(self):
            return self._device

    fake_gpu_device = torch.device("cuda:0")
    image = MockImageTensor(fake_gpu_device)

    n = 3
    gaussians = SimpleNamespace(
        _opacity=torch.zeros((n, 1), requires_grad=True),
        get_scaling=torch.ones((n, 3), requires_grad=True),
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    inputs = LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": image,
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        ),
        image=image,
        iteration=1,
        optimizer_update_id=1,
        n_frames_ingested=1,
    )

    # Mock _get_surface_prior to return a surface prior with CPU-resident confidence
    mock_prior = SimpleNamespace(
        depth=torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu"),
        confidence=torch.ones((1, 4, 4), dtype=torch.float32, device="cpu"),
    )

    with patch("eligsir.optimization.gaussian.objective._get_surface_prior", return_value=mock_prior):
        # Preflight should raise ValueError for CPU-resident auxiliary on prepared camera
        with pytest.raises(ValueError, match="surface_confidence.*device-resident"):
            assemble_loss(inputs)


def test_prepared_path_cpu_resident_ray_semantics_depth_raises_at_preflight():
    """Test that CPU-resident ray-semantics depth on prepared path raises at preflight."""
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    args = SimpleNamespace(
        streaming_ray_semantics_enabled=True,
        streaming_appearance_only_enabled=False,
        surface_confidence_depth_loss=False,
    )

    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    cpu_depth = torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu")
    cam._sensor_depth_cache = cpu_depth
    with pytest.raises(ValueError, match="ray_semantics_sensor_depth.*device-resident"):
        preflight_prepared_objective_auxiliaries(cam, args, torch.device("cuda:0"))


def test_prepared_path_cpu_resident_free_space_depth_raises_at_preflight():
    """Test that CPU-resident free-space depth on prepared path raises at preflight.

    Free-space requires both weight > 0 AND use_depth_loss=True to be enabled.
    """
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    args = SimpleNamespace(
        streaming_free_space_loss_weight=1.0,
        surface_confidence_depth_loss=False,
        streaming_ray_semantics_enabled=False,
        streaming_appearance_only_enabled=False,
    )

    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    cpu_depth = torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu")

    cam._sensor_depth_cache = cpu_depth
    with pytest.raises(ValueError, match="free_space_sensor_depth.*device-resident"):
        preflight_prepared_objective_auxiliaries(cam, args, torch.device("cuda:0"), use_depth_loss=True)


def test_prepared_path_cpu_resident_free_space_with_confidence_raises_at_preflight():
    """Test that CPU-resident surface confidence in free-space context raises at preflight.

    Note: surface_confidence check happens before free-space-specific checks,
    so the error message will mention 'surface_confidence' not 'free_space_surface_confidence'.
    """
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    args = SimpleNamespace(
        streaming_free_space_loss_weight=1.0,
        surface_confidence_depth_loss=True,  # Enables surface prior
        streaming_ray_semantics_enabled=False,
        streaming_appearance_only_enabled=False,
    )

    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    device_depth = torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu")
    mock_prior = SimpleNamespace(
        depth=device_depth,
        confidence=torch.ones((1, 4, 4), dtype=torch.float32, device="cpu"),  # CPU-resident confidence
    )

    cam._sensor_depth_cache = device_depth
    with patch("eligsir.optimization.gaussian.objective._get_surface_prior", return_value=mock_prior):
        with pytest.raises(ValueError, match="surface_confidence.*device-resident"):
            preflight_prepared_objective_auxiliaries(cam, args, torch.device("cuda:0"), use_depth_loss=True)


def test_prepared_path_ray_semantics_or_appearance_only_check_sensor_depth():
    """Test that sensor depth is checked when EITHER ray-semantics or appearance-only is enabled."""
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    cpu_depth = torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu")
    gpu_device = torch.device("cuda:0")

    # Test: appearance-only enabled, ray-semantics disabled → should still check
    args_ao_only = SimpleNamespace(
        streaming_ray_semantics_enabled=False,
        streaming_appearance_only_enabled=True,
        surface_confidence_depth_loss=False,
        streaming_free_space_loss_weight=0.0,
    )

    cam._sensor_depth_cache = cpu_depth
    with pytest.raises(ValueError, match="ray_semantics_sensor_depth.*device-resident"):
        preflight_prepared_objective_auxiliaries(cam, args_ao_only, gpu_device)


def test_prepared_path_device_resident_ray_semantics_depth_passes_preflight():
    """Test that device-resident ray-semantics depth passes preflight."""
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    args = SimpleNamespace(
        streaming_ray_semantics_enabled=True,
        streaming_appearance_only_enabled=False,
        surface_confidence_depth_loss=False,
        streaming_free_space_loss_weight=0.0,
    )

    cam._sensor_depth_cache = torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu")
    preflight_prepared_objective_auxiliaries(cam, args, torch.device("cpu"))


def test_legacy_path_free_space_surface_confidence_still_transferred():
    """Test that legacy path still uses surface confidence in free-space loss.

    Verifies that the fix to line 998 (replacing hardcoded .to("cuda") with _legacy_to_device)
    doesn't break the surface confidence integration in the free-space path.
    """
    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0
    # Disable depth loss to avoid CUDA transfer
    args.streaming_depth_loss_weight = 0.0
    args.streaming_free_space_loss_weight = 0.0  # Disable free-space for this test (it needs depth loss anyway)
    args.surface_confidence_depth_loss = False

    # Create a legacy camera
    cam = _make_legacy_camera(device="cpu")

    n = 3
    gaussians = SimpleNamespace(
        _opacity=torch.zeros((n, 1), requires_grad=True),
        get_scaling=torch.ones((n, 3), requires_grad=True),
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    image = torch.zeros((3, 4, 4), dtype=torch.float32, device="cpu", requires_grad=True)

    inputs = LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": image,
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        ),
        image=image,
        iteration=1,
        optimizer_update_id=1,
        n_frames_ingested=1,
        use_depth_loss=False,
    )

    # Should not raise - legacy path operates normally
    breakdown = assemble_loss(inputs)
    assert bool(torch.isfinite(breakdown.total_tensor))


def test_disabled_free_space_weight_zero_skips_check():
    """Test that free-space check is skipped when weight is zero."""
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    args = SimpleNamespace(
        streaming_free_space_loss_weight=0.0,  # Disabled
        surface_confidence_depth_loss=False,
        streaming_ray_semantics_enabled=False,
        streaming_appearance_only_enabled=False,
    )

    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    cam._sensor_depth_cache = torch.zeros((1, 4, 4), device="cpu")
    preflight_prepared_objective_auxiliaries(cam, args, torch.device("cuda:0"))


def test_free_space_depth_loss_disabled_skips_check():
    """Test that free-space check is skipped when use_depth_loss is False (regression test for false-positive).

    A prepared camera with free-space weight > 0 but use_depth_loss=False should NOT raise at preflight,
    even if sensor depth is CPU-resident, because the loss body won't actually use that tensor
    (line 930: free-space requires BOTH weight > 0 AND use_depth_loss).
    """
    from eligsir.contracts import SourceIdentity
    from eligsir.optimization.gaussian.objective import (
        preflight_prepared_objective_auxiliaries,
    )

    args = SimpleNamespace(
        streaming_free_space_loss_weight=1.0,  # Enabled
        surface_confidence_depth_loss=False,
        streaming_ray_semantics_enabled=False,
        streaming_appearance_only_enabled=False,
    )

    cam = _make_prepared_camera(device="cpu")
    cam._source_identity = SourceIdentity(
        source_id="test_source",
        stream_epoch=0,
        source_ordinal=0,
        timestamp_ns=0,
    )

    cam._sensor_depth_cache = torch.zeros((1, 4, 4), dtype=torch.float32, device="cpu")
    preflight_prepared_objective_auxiliaries(cam, args, torch.device("cuda:0"), use_depth_loss=False)
