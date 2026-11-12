import math
from types import SimpleNamespace

import pytest
import torch

from eligsir.mapping.motion_depth_confidence import motion_depth_confidence


def _intrinsics(*, fx=10.0, fy=10.0, cx=0.0, cy=0.0):
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _translation(*, x=0.0, y=0.0, z=0.0):
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([x, y, z])
    return transform


def test_translation_weights_follow_depth_not_one_frame_scalar():
    depth = torch.tensor([[1.0, 2.0]])
    intrinsics = torch.diag(torch.tensor([10.0, 10.0, 1.0]))
    relative = torch.eye(4)
    relative[0, 3] = 0.1

    actual = motion_depth_confidence(depth, intrinsics, relative)

    torch.testing.assert_close(actual, torch.tensor([[0.5, 0.8]]))


def test_identity_and_invalid_depth_are_neutral_without_mutating_inputs():
    depth = torch.tensor([[1.0, 0.0, -1.0, math.nan, math.inf]])
    intrinsics = _intrinsics()
    relative = _translation()
    originals = tuple(item.clone() for item in (depth, intrinsics, relative))

    actual = motion_depth_confidence(depth, intrinsics, relative)

    torch.testing.assert_close(actual, torch.ones_like(depth))
    assert actual.shape == depth.shape
    assert actual.dtype == depth.dtype
    assert torch.isfinite(actual).all()
    for actual_input, original in zip((depth, intrinsics, relative), originals, strict=True):
        torch.testing.assert_close(actual_input, original, equal_nan=True)


def test_rotation_reprojects_each_pixel_in_the_selected_tier_geometry():
    depth = torch.ones((1, 2))
    intrinsics = _intrinsics()
    relative = _translation()
    relative[:2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])

    actual = motion_depth_confidence(depth, intrinsics, relative)

    torch.testing.assert_close(actual, torch.tensor([[1.0, 1.0 / 3.0]]))


def test_native_and_half_tiers_use_their_own_intrinsics():
    native = motion_depth_confidence(torch.ones((2, 4)), _intrinsics(fx=10.0, fy=10.0), _translation(x=0.1))
    half = motion_depth_confidence(torch.ones((1, 2)), _intrinsics(fx=5.0, fy=5.0), _translation(x=0.1))

    torch.testing.assert_close(native, torch.full((2, 4), 0.5))
    torch.testing.assert_close(half, torch.full((1, 2), 0.8))


def test_equal_axis_tolerance_preserves_scalar_operator_results():
    depth = torch.tensor([[1.0, 2.0]])
    intrinsics = _intrinsics()
    relative = _translation(x=0.1, y=0.1)

    torch.testing.assert_close(
        motion_depth_confidence(depth, intrinsics, relative, tau_px=0.25),
        motion_depth_confidence(depth, intrinsics, relative, tau_px=(0.25, 0.25)),
    )


def test_behind_camera_projection_and_invalid_source_matrices_are_neutral():
    depth = torch.ones((1, 2))
    behind = motion_depth_confidence(depth, _intrinsics(), _translation(z=-2.0))
    bad_transform = _translation(x=0.1)
    bad_transform[0, 0] = math.nan
    infinite_transform = _translation(x=0.1)
    infinite_transform[0, 0] = math.inf
    bad_intrinsics = _intrinsics()
    bad_intrinsics[1, 1] = math.inf

    torch.testing.assert_close(behind, torch.ones_like(depth))
    torch.testing.assert_close(
        motion_depth_confidence(depth, _intrinsics(), bad_transform), torch.ones_like(depth)
    )
    torch.testing.assert_close(
        motion_depth_confidence(depth, _intrinsics(), infinite_transform), torch.ones_like(depth)
    )
    torch.testing.assert_close(
        motion_depth_confidence(depth, bad_intrinsics, _translation(x=0.1)), torch.ones_like(depth)
    )


@pytest.mark.parametrize(
    ("axis", "focal"), (("x", 0.0), ("x", -10.0), ("y", 0.0), ("y", -10.0))
)
def test_finite_nonpositive_focals_neutralize_nonidentity_motion(axis, focal):
    depth = torch.ones((1, 2)) if axis == "x" else torch.ones((2, 1))
    intrinsics = _intrinsics(**({"fx": focal} if axis == "x" else {"fy": focal}))
    relative = _translation(**{axis: 0.1})

    actual = motion_depth_confidence(depth, intrinsics, relative)

    torch.testing.assert_close(actual, torch.ones_like(depth))


def test_nonfinite_reprojection_and_weight_floor_are_neutral_or_bounded():
    depth = torch.full((1, 2), torch.finfo(torch.float32).max)
    neutral = motion_depth_confidence(
        depth, _intrinsics(), _translation(z=torch.finfo(torch.float32).max)
    )
    bounded = motion_depth_confidence(
        torch.ones((1, 2)), _intrinsics(), _translation(x=0.1), w_min=0.75
    )

    torch.testing.assert_close(neutral, torch.ones_like(depth))
    torch.testing.assert_close(bounded, torch.full((1, 2), 0.75))
    assert bool(((bounded >= 0.75) & (bounded <= 1.0)).all())


@pytest.mark.parametrize("shape", ((0, 2), (2, 0)))
def test_empty_spatial_tensors_preserve_shape_and_dtype(shape):
    depth = torch.empty(shape)

    actual = motion_depth_confidence(depth, _intrinsics(), _translation(x=0.1))

    assert actual.shape == depth.shape
    assert actual.dtype == depth.dtype
    assert actual.numel() == 0


@pytest.mark.parametrize(
    ("depth", "intrinsics", "relative", "kwargs"),
    [
        (torch.ones((1, 2, 1)), _intrinsics(), _translation(), {}),
        (torch.ones((1, 2)), torch.eye(2), _translation(), {}),
        (torch.ones((1, 2)), _intrinsics(), torch.eye(3), {}),
        (torch.ones((1, 2), dtype=torch.int64), _intrinsics(), _translation(), {}),
        (torch.ones((1, 2)), _intrinsics().double(), _translation(), {}),
        (torch.ones((1, 2)), _intrinsics(), _translation().double(), {}),
        (torch.ones((1, 2)), _intrinsics(), _translation(), {"tau_px": 0.0}),
        (torch.ones((1, 2)), _intrinsics(), _translation(), {"tau_px": math.nan}),
        (torch.ones((1, 2)), _intrinsics(), _translation(), {"w_min": -0.1}),
        (torch.ones((1, 2)), _intrinsics(), _translation(), {"w_min": 1.1}),
    ],
)
def test_structural_and_scalar_contract_errors_raise_value_error(depth, intrinsics, relative, kwargs):
    with pytest.raises(ValueError):
        motion_depth_confidence(depth, intrinsics, relative, **kwargs)


def test_device_mismatch_is_rejected_before_tensor_computation():
    with pytest.raises(ValueError):
        motion_depth_confidence(
            torch.ones((1, 2)), torch.eye(3, device="meta"), _translation()
        )


@pytest.mark.parametrize(
    ("dtype", "tau_px"),
    ((torch.float16, 1e-4), (torch.float32, 1e-23), (torch.float64, 1e-200)),
)
def test_identity_stays_finite_when_tau_square_would_underflow(dtype, tau_px):
    depth = torch.ones((1, 1), dtype=dtype)

    actual = motion_depth_confidence(
        depth, _intrinsics().to(dtype), _translation().to(dtype), tau_px=tau_px
    )

    torch.testing.assert_close(actual, torch.ones_like(depth))
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("dtype", (torch.float16, torch.float32, torch.float64))
def test_identity_accepts_the_smallest_supported_working_dtype_tau(dtype):
    depth = torch.ones((1, 1), dtype=dtype)

    actual = motion_depth_confidence(
        depth,
        _intrinsics().to(dtype),
        _translation().to(dtype),
        tau_px=torch.finfo(dtype).tiny,
    )

    torch.testing.assert_close(actual, torch.ones_like(depth))


class _FloatSpy:
    def __init__(self):
        self.called = False

    def __float__(self):
        self.called = True
        raise AssertionError("configuration scalar conversion must not run")


@pytest.mark.parametrize("name", ("tau_px", "w_min"))
def test_non_numeric_scalar_configuration_is_rejected_before_float_conversion(name):
    spy = _FloatSpy()

    with pytest.raises(ValueError):
        motion_depth_confidence(torch.ones((1, 1)), _intrinsics(), _translation(), **{name: spy})

    assert not spy.called


@pytest.mark.parametrize("name", ("tau_px", "w_min"))
def test_tensor_scalar_configuration_is_rejected_without_readback(name):
    with pytest.raises(ValueError):
        motion_depth_confidence(
            torch.ones((1, 1)), _intrinsics(), _translation(), **{name: torch.tensor(1.0)}
        )


def _depth_terms(rendered_depth, weight_map=None):
    from eligsir.optimization.gaussian.batched_pixel_terms import (
        assemble_batched_pixel_terms,
    )

    return assemble_batched_pixel_terms(
        renders=torch.zeros((1, 3, 1, 2)),
        gts=torch.zeros((1, 3, 1, 2)),
        rendered_depth=rendered_depth,
        sensor_depth=torch.ones((1, 1, 1, 2)),
        weight_maps=() if weight_map is None else (weight_map,),
        required_rgb=(False, False),
    )[0].depth


@pytest.mark.parametrize(
    ("weight_map", "expected_loss", "expected_gradient"),
    [
        (None, 50.5, (0.5, 0.5)),
        (torch.full((1, 1, 1, 2), 0.25), 50.5, (0.5, 0.5)),
        (torch.tensor([[[[0.01, 1.0]]]]), 1.980198, (0.00990099, 0.990099)),
    ],
)
def test_real_depth_objective_normalizes_spatial_confidence_and_actuates_gradients(
    weight_map, expected_loss, expected_gradient
):
    rendered_depth = torch.tensor([[[[101.0, 2.0]]]], requires_grad=True)

    loss = _depth_terms(rendered_depth, weight_map)
    (gradient,) = torch.autograd.grad(loss, rendered_depth)

    torch.testing.assert_close(loss, torch.tensor(expected_loss), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        gradient.flatten(), torch.tensor(expected_gradient), atol=1e-6, rtol=1e-6
    )


def test_real_depth_objective_changes_when_spatial_confidence_is_swapped():
    rendered_depth = torch.tensor([[[[101.0, 2.0]]]], requires_grad=True)
    selected = torch.tensor([[[[0.01, 1.0]]]])

    loss = _depth_terms(rendered_depth, selected)
    shuffled_loss = _depth_terms(rendered_depth, selected.flip(-1))

    torch.testing.assert_close(loss, torch.tensor(1.980198), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(shuffled_loss, torch.tensor(99.019802), atol=1e-6, rtol=1e-6)


def test_operator_generated_confidence_map_changes_the_real_depth_objective_when_shuffled():
    generated = motion_depth_confidence(
        torch.tensor([[1.0, 2.0]]), _intrinsics(), _translation(x=0.1)
    ).reshape(1, 1, 1, 2)
    rendered_depth = torch.tensor([[[[101.0, 2.0]]]])

    selected_loss = _depth_terms(rendered_depth, generated)
    shuffled_loss = _depth_terms(rendered_depth, generated.flip(-1))

    torch.testing.assert_close(generated.flatten(), torch.tensor([0.5, 0.8]))
    torch.testing.assert_close(selected_loss, torch.tensor(39.076923), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(shuffled_loss, torch.tensor(61.923077), atol=1e-6, rtol=1e-6)


def test_prepared_two_camera_batch_consumes_all_resident_motion_confidence_maps():
    from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
    from eligsir.optimization.gaussian.true_batch_executor import _resolve_batched_pixel_terms

    def terms(weights):
        cameras = [
            SimpleNamespace(
                original_image=torch.zeros((3, 1, 2)), _sensor_depth_cache=torch.ones((1, 2)),
                _resolution_depth_confidence_weight=weight, uid=index,
            )
            for index, weight in enumerate(weights)
        ]
        depth = torch.tensor([[[[101.0, 2.0]]], [[[3.0, 101.0]]]], requires_grad=True)
        resolved = _resolve_batched_pixel_terms(
            args=SimpleNamespace(streaming_objective_batched_pixel_terms=True),
            group=tuple(SimpleNamespace(view=SimpleNamespace(camera=camera)) for camera in cameras),
            group_modalities=(
                SimpleNamespace(rgb_loss_enabled=True, depth_loss_enabled=True),
                SimpleNamespace(rgb_loss_enabled=True, depth_loss_enabled=True),
            ),
            batch=TrueCameraBatchResult(
                renders=torch.zeros((2, 3, 1, 2)), alphas=torch.ones((2, 1, 1, 2)),
                rendered_depth=depth, rendered_depth_ed=None, meta={}, camera_count=2, sparse_grad=False,
            ),
            iteration=1, tb_writer=None, scalar_log_interval=1, depth_loss_type="l1",
            opt=SimpleNamespace(lambda_dssim=1.0), rgb_measurements=(False, False),
        )
        assert resolved is not None and resolved[0].depth is not None
        return resolved[0].depth, torch.autograd.grad(resolved[0].depth, depth)[0]

    neutral_loss, neutral_gradient = terms((torch.ones((1, 2)), torch.ones((1, 2))))
    weighted_loss, weighted_gradient = terms((torch.tensor(((0.01, 1.0),)), torch.ones((1, 2))))

    assert weighted_loss < neutral_loss
    assert not torch.equal(weighted_gradient, neutral_gradient)
