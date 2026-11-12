import numpy as np
import pytest
import torch


def test_affine_fit_and_apply_preserve_raw_input():
    from eligsir.exposure_opt import fit_affine
    render = torch.tensor([[[1., 2.]], [[2., 4.]], [[3., 6.]]])
    original = render.clone()
    gt = render * torch.tensor([2., 3., 4.]).view(3, 1, 1) + 1.
    gain, bias = fit_affine(render, gt)
    torch.testing.assert_close(gain, torch.tensor([2., 3., 4.]))
    torch.testing.assert_close(bias, torch.ones(3))
    torch.testing.assert_close(render, original)


def test_propagation_interpolates_pose_and_exposure_with_gap_guard():
    from eligsir.mapping.post_mapping_pose_align_propagate import (
        interpolate_delta_windowed, interpolate_exposure_windowed,
    )
    deltas = {0: np.zeros(6), 10: np.array([0., 0., 0., 10., 0., 0.])}
    exposure = {0: (np.ones(3), np.zeros(3)), 10: (np.full(3, 3.), np.full(3, 2.))}
    np.testing.assert_allclose(interpolate_delta_windowed(deltas, 5, 8, 1)[3:], [5., 0., 0.])
    gain, bias = interpolate_exposure_windowed(exposure, 5, 8, 1)
    np.testing.assert_allclose(gain, 2.)
    np.testing.assert_allclose(bias, 1.)
    assert interpolate_delta_windowed(deltas, 30, 8, 1) is None


def test_global_entrypoints_ignore_locality_and_gap():
    from eligsir.mapping.post_mapping_pose_align_propagate import PropagateHoldoutConfig, propagated_holdout_deltas, propagated_holdout_exposure
    deltas={0:np.zeros(6),10:np.array([0,0,.2,10,0,0]),100:np.array([0,0,.8,100,0,0])}
    cfg=PropagateHoldoutConfig("global",max_gap=8,propagate_exposure=True)
    assert propagated_holdout_deltas(deltas,[50],cfg)[50][3] == pytest.approx(110/3)
    exp={0:(np.ones(3),np.zeros(3)),100:(np.full(3,5.),np.full(3,4.))}
    gain,bias=propagated_holdout_exposure(exp,[50],cfg)[50]
    np.testing.assert_allclose(gain,3.);np.testing.assert_allclose(bias,2.)


def test_widened_window_uses_nearest_before_and_after_reference_frames():
    """Regression for the audited 8064 window selection, not a helper self-test."""
    from eligsir.mapping.post_mapping_pose_align_propagate import (
        interpolate_delta_windowed,
        interpolate_exposure_windowed,
    )

    deltas = {
        0: np.array([0., 0., 0., 0., 0., 0.]),
        9: np.array([0., 0., .1, 9., 0., 0.]),
        10: np.array([0., 0., .2, 10., 0., 0.]),
        11: np.array([0., 0., .3, 11., 0., 0.]),
        20: np.array([0., 0., .4, 20., 0., 0.]),
    }
    exposure = {
        ordinal: (np.full(3, delta[3]), np.full(3, -delta[3]))
        for ordinal, delta in deltas.items()
    }

    # The immutable 8064 implementation selects 10, 11, 20: the last two
    # valid frames before ordinal 15 and the first valid frame after it.
    # Its Gaussian weights yield this independent hand-recorded value.
    expected_translation = 11.086940330212366
    propagated = interpolate_delta_windowed(deltas, 15, 20, window=2)
    gain, bias = interpolate_exposure_windowed(exposure, 15, 20, window=2)

    assert propagated[3] == pytest.approx(expected_translation)
    np.testing.assert_allclose(gain, expected_translation)
    np.testing.assert_allclose(bias, -expected_translation)


def test_width_one_preserves_camera_local_slerp_and_right_composition():
    from eligsir.mapping.post_mapping_pose_align_propagate import (
        apply_delta6_to_c2w,
        delta6_to_matrix,
        interpolate_delta_windowed,
    )

    # A 180-degree turn cannot be checked by averaging axis-angle vectors;
    # the midpoint must be the camera-local 90-degree rotation produced by
    # quaternion SLERP.  The translated camera also checks right composition.
    deltas = {
        0: np.zeros(6),
        10: np.array([0., 0., np.pi, 2., 0., 0.]),
    }
    midpoint = interpolate_delta_windowed(deltas, 5, max_gap=8, window=1)
    expected_rotation = delta6_to_matrix([0., 0., np.pi / 2, 0., 0., 0.])[:3, :3]
    np.testing.assert_allclose(delta6_to_matrix(midpoint)[:3, :3], expected_rotation, atol=1e-7)

    camera = np.eye(4)
    camera[:3, 3] = [1., 2., 3.]
    np.testing.assert_allclose(
        apply_delta6_to_c2w(camera, midpoint),
        camera @ delta6_to_matrix(midpoint),
        atol=1e-12,
    )


def test_propagation_entrypoints_cover_empty_singleton_and_global_copy_semantics():
    from eligsir.mapping.post_mapping_pose_align_propagate import (
        PropagateHoldoutConfig,
        propagated_holdout_deltas,
        propagated_holdout_exposure,
    )

    config = PropagateHoldoutConfig(mode="global", propagate_exposure=True)
    assert propagated_holdout_deltas({}, [1], config) == {}
    assert propagated_holdout_exposure({}, [1], config) == {}

    delta = np.array([0., 0., np.pi - 1e-4, 3., -2., 1.])
    exposure = (np.array([1.5, 2., 2.5]), np.array([.1, .2, .3]))
    propagated_delta = propagated_holdout_deltas({7: delta}, [1, 999], config)
    propagated_exposure = propagated_holdout_exposure({7: exposure}, [1, 999], config)
    for ordinal in (1, 999):
        # The audited 8064 quaternion/log-map round-trip intentionally
        # defines this near-pi numeric result; preserve that public behavior
        # rather than silently substituting a different SO(3) implementation.
        np.testing.assert_allclose(
            propagated_delta[ordinal],
            [0., 0., 3.140236807976505, 3., -2., 1.],
            atol=1e-7,
        )
        np.testing.assert_allclose(propagated_exposure[ordinal][0], exposure[0])
        np.testing.assert_allclose(propagated_exposure[ordinal][1], exposure[1])
