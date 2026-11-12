import math
from types import SimpleNamespace

import pytest
import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.mapping.resolution_materialization import (
    RESAMPLING_MODES,
    materialize_resolution_tensors,
    materialize_resolution_tier,
)


def _camera():
    return SimpleNamespace(
        original_image=torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16),
        _sensor_depth_cache=torch.ones((1, 8, 16), dtype=torch.float32),
        image_width=16,
        image_height=8,
        width=16,
        height=8,
        fx=80.0,
        fy=40.0,
        cx=7.5,
        cy=3.5,
        gsplat=object(),
    )


def test_materialize_resolution_tier_scales_rgb_depth_and_intrinsics_from_native():
    camera = _camera()

    materialize_resolution_tier(camera, ResolutionTier.QUARTER)

    assert tuple(camera.original_image.shape) == (3, 2, 4)
    assert tuple(camera._sensor_depth_cache.shape) == (1, 2, 4)
    assert (camera.image_width, camera.image_height) == (4, 2)
    assert (camera.width, camera.height) == (4, 2)
    assert (camera.fx, camera.fy, camera.cx, camera.cy) == (20.0, 10.0, 1.5, 0.5)
    assert camera.gsplat is None


def test_materialize_resolution_tier_rebinds_from_native_not_an_already_downsampled_view():
    camera = _camera()

    materialize_resolution_tier(camera, ResolutionTier.EIGHTH)
    materialize_resolution_tier(camera, ResolutionTier.HALF)

    assert tuple(camera.original_image.shape) == (3, 4, 8)
    assert tuple(camera._sensor_depth_cache.shape) == (1, 4, 8)
    assert (camera.image_width, camera.image_height) == (8, 4)
    assert (camera.fx, camera.fy, camera.cx, camera.cy) == (40.0, 20.0, 3.5, 1.5)


def test_same_resolution_tier_reuses_current_binding_without_resize_or_camera_rebuild(
    monkeypatch,
):
    import eligsir.mapping.resolution_materialization as resolution_materialization

    camera = _camera()
    materialize_resolution_tier(camera, ResolutionTier.QUARTER)
    rgb = camera.original_image
    depth = camera._sensor_depth_cache
    prepared_gsplat = object()
    camera.gsplat = prepared_gsplat
    monkeypatch.setattr(
        resolution_materialization,
        "_resize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("same-tier materialization resized pixels")
        ),
    )

    materialize_resolution_tier(camera, ResolutionTier.QUARTER)

    assert camera.original_image is rgb
    assert camera._sensor_depth_cache is depth
    assert camera.gsplat is prepared_gsplat


def test_resolution_tier_declares_score_resize_and_restores_native_lineage():
    from eligsir.utils.score_contract import (
        declare_camera_score_provenance,
        score_tensor_from_camera,
        validate_score_pair,
    )

    camera = _camera()
    camera.original_image = torch.full((3, 8, 16), 0.5)
    declare_camera_score_provenance(camera, context="test_native_camera")

    materialize_resolution_tier(camera, ResolutionTier.QUARTER)
    quarter = score_tensor_from_camera(camera)
    audit = validate_score_pair(quarter, quarter.clone(), gt_provenance=camera.score_provenance)

    assert audit.resolution == (2, 4)
    assert camera.score_provenance.source.resolution == (8, 16)
    assert camera.score_provenance.canonical_score_domain.resolution == (2, 4)
    assert camera.score_provenance.steps[-1].target_resolution == (2, 4)

    materialize_resolution_tier(camera, ResolutionTier.NATIVE)
    native = score_tensor_from_camera(camera)
    audit = validate_score_pair(native, native.clone(), gt_provenance=camera.score_provenance)

    assert audit.resolution == (8, 16)
    assert camera.score_provenance.steps == ()


# ---------------------------------------------------------------------------
# resampling_mode selection: legacy (default) / masked / ray_preserving
# ---------------------------------------------------------------------------


def _camera_with_distinct_pixels():
    """8x16 native RGB/depth where every pixel has a unique value.

    Distinct per-pixel values let ray-identity assertions distinguish "this
    output pixel IS native[s*i, s*j]" from an interpolated/averaged value.
    """
    return SimpleNamespace(
        original_image=torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16),
        _sensor_depth_cache=torch.arange(8 * 16, dtype=torch.float32).reshape(1, 8, 16) + 1.0,
        image_width=16,
        image_height=8,
        width=16,
        height=8,
        fx=80.0,
        fy=40.0,
        cx=7.5,
        cy=3.5,
        gsplat=object(),
    )


def test_legacy_resampling_mode_is_byte_identical_to_no_mode_argument():
    """Passing resampling_mode='legacy' explicitly must match omitting it entirely."""
    camera_default = _camera_with_distinct_pixels()
    camera_explicit = _camera_with_distinct_pixels()

    materialize_resolution_tier(camera_default, ResolutionTier.QUARTER)
    materialize_resolution_tier(camera_explicit, ResolutionTier.QUARTER, resampling_mode="legacy")

    assert torch.equal(camera_default.original_image, camera_explicit.original_image)
    assert torch.equal(camera_default._sensor_depth_cache, camera_explicit._sensor_depth_cache)
    assert (camera_default.fx, camera_default.fy, camera_default.cx, camera_default.cy) == (
        camera_explicit.fx,
        camera_explicit.fy,
        camera_explicit.cx,
        camera_explicit.cy,
    )
    assert getattr(camera_default, "_resolution_depth_confidence_weight", None) is None
    assert getattr(camera_explicit, "_resolution_depth_confidence_weight", None) is None


@pytest.mark.parametrize("mode", RESAMPLING_MODES)
@pytest.mark.parametrize("tier", (ResolutionTier.NATIVE, ResolutionTier.HALF, ResolutionTier.QUARTER))
def test_device_tensor_tier_materialization_matches_canonical_camera_semantics(mode, tier):
    """Prepared tiers must be exactly the canonical camera tiers, not an approximation."""

    camera = _camera_with_distinct_pixels()
    native_rgb_hwc = camera.original_image.permute(1, 2, 0).contiguous()
    native_depth_hw = camera._sensor_depth_cache.squeeze(0)
    K = torch.tensor(
        ((camera.fx, 0.0, camera.cx), (0.0, camera.fy, camera.cy), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
    )

    materialize_resolution_tier(camera, tier, resampling_mode=mode)
    rgb, depth, intrinsics, confidence = materialize_resolution_tensors(
        native_rgb_hwc,
        native_depth_hw,
        K,
        tier,
        resampling_mode=mode,
    )

    assert torch.equal(rgb.permute(2, 0, 1), camera.original_image)
    assert torch.equal(depth, camera._sensor_depth_cache.squeeze(0))
    assert torch.equal(
        intrinsics,
        torch.tensor(
            ((camera.fx, 0.0, camera.cx), (0.0, camera.fy, camera.cy), (0.0, 0.0, 1.0)),
            dtype=torch.float32,
        ),
    )
    expected_confidence = getattr(camera, "_resolution_depth_confidence_weight", None)
    if expected_confidence is None:
        assert confidence is None
    else:
        assert torch.equal(confidence, expected_confidence)


def test_ray_preserving_mode_samples_exact_native_pixels_no_interpolation():
    for tier, factor in ((ResolutionTier.HALF, 2), (ResolutionTier.QUARTER, 4)):
        camera = _camera_with_distinct_pixels()
        native_rgb = camera.original_image.clone()
        native_depth = camera._sensor_depth_cache.clone()

        materialize_resolution_tier(camera, tier, resampling_mode="ray_preserving")

        out_h, out_w = camera.original_image.shape[1], camera.original_image.shape[2]
        assert out_h == 8 // factor and out_w == 16 // factor
        for i in range(out_h):
            for j in range(out_w):
                assert torch.equal(
                    camera.original_image[:, i, j], native_rgb[:, i * factor, j * factor]
                )
                assert camera._sensor_depth_cache[0, i, j].item() == native_depth[0, i * factor, j * factor].item()


def test_ray_preserving_mode_nests_quarter_coordinates_inside_half_and_native():
    half_factor, quarter_factor = 2, 4
    native_h, native_w = 8, 16
    half_h, half_w = native_h // half_factor, native_w // half_factor
    quarter_h, quarter_w = native_h // quarter_factor, native_w // quarter_factor

    half_native_rows = {i * half_factor for i in range(half_h)}
    half_native_cols = {j * half_factor for j in range(half_w)}
    quarter_native_rows = {i * quarter_factor for i in range(quarter_h)}
    quarter_native_cols = {j * quarter_factor for j in range(quarter_w)}

    assert quarter_native_rows.issubset(half_native_rows)
    assert quarter_native_cols.issubset(half_native_cols)
    assert quarter_native_rows.issubset(set(range(native_h)))
    assert quarter_native_cols.issubset(set(range(native_w)))

    # And the actual indexing code (not just the arithmetic above) produces
    # tensors consistent with this nesting for a real camera.
    camera_half = _camera_with_distinct_pixels()
    camera_quarter = _camera_with_distinct_pixels()
    materialize_resolution_tier(camera_half, ResolutionTier.HALF, resampling_mode="ray_preserving")
    materialize_resolution_tier(camera_quarter, ResolutionTier.QUARTER, resampling_mode="ray_preserving")
    for qi in range(quarter_h):
        for qj in range(quarter_w):
            hi, hj = qi * quarter_factor // half_factor, qj * quarter_factor // half_factor
            assert torch.equal(
                camera_quarter.original_image[:, qi, qj], camera_half.original_image[:, hi, hj]
            )


def test_ray_preserving_mode_uses_zero_phase_analytic_intrinsics():
    camera = _camera_with_distinct_pixels()

    materialize_resolution_tier(camera, ResolutionTier.QUARTER, resampling_mode="ray_preserving")

    # f' = f/s, c' = c/s -- NOT the legacy (c+0.5)*s-0.5 pixel-center formula.
    assert camera.fx == 80.0 / 4
    assert camera.fy == 40.0 / 4
    assert camera.cx == 7.5 / 4
    assert camera.cy == 3.5 / 4


def test_ray_preserving_downgrades_to_legacy_for_unregistered_depth():
    """Issue #1260: depth_fx != fx means the depth sensor's native resolution
    was reconciled onto the color grid with a plain nearest-neighbor reshape,
    not a geometric reprojection -- ray_preserving's same-ray assumption does
    not hold, so it must fall back to legacy resampling instead of silently
    mis-supervising depth against the wrong native ray."""
    camera = _camera_with_distinct_pixels()
    camera._streaming_frame = SimpleNamespace(fx=camera.fx, depth_fx=camera.fx / 2.0)
    legacy_camera = _camera_with_distinct_pixels()

    materialize_resolution_tier(camera, ResolutionTier.QUARTER, resampling_mode="ray_preserving")
    materialize_resolution_tier(legacy_camera, ResolutionTier.QUARTER, resampling_mode="legacy")

    assert torch.equal(camera.original_image, legacy_camera.original_image)
    assert (camera.fx, camera.fy, camera.cx, camera.cy) == (
        legacy_camera.fx,
        legacy_camera.fy,
        legacy_camera.cx,
        legacy_camera.cy,
    )
    assert camera._canonical_resolution_resampling_mode == "legacy"


def test_ray_preserving_stays_ray_preserving_for_registered_depth():
    """depth_fx unset, or equal to the color fx, means the source is either a
    single calibrated RGB-D sensor or already registered -- ray_preserving
    must run normally, not silently downgrade."""
    for depth_fx in (None, 80.0):
        camera = _camera_with_distinct_pixels()
        camera._streaming_frame = SimpleNamespace(fx=camera.fx, depth_fx=depth_fx)

        materialize_resolution_tier(camera, ResolutionTier.QUARTER, resampling_mode="ray_preserving")

        assert camera._canonical_resolution_resampling_mode == "ray_preserving"
        # f' = f/s zero-phase intrinsics, not legacy's (c+0.5)*s-0.5.
        assert camera.cx == 7.5 / 4


def test_masked_resampling_mode_keeps_legacy_area_nearest_pixels():
    camera_legacy = _camera_with_distinct_pixels()
    camera_masked = _camera_with_distinct_pixels()

    materialize_resolution_tier(camera_legacy, ResolutionTier.QUARTER, resampling_mode="legacy")
    materialize_resolution_tier(camera_masked, ResolutionTier.QUARTER, resampling_mode="masked")

    assert torch.equal(camera_legacy.original_image, camera_masked.original_image)
    assert torch.equal(camera_legacy._sensor_depth_cache, camera_masked._sensor_depth_cache)
    assert (camera_legacy.fx, camera_legacy.fy, camera_legacy.cx, camera_legacy.cy) == (
        camera_masked.fx,
        camera_masked.fy,
        camera_masked.cx,
        camera_masked.cy,
    )
    assert getattr(camera_legacy, "_resolution_depth_confidence_weight", None) is None
    assert camera_masked._resolution_depth_confidence_weight is not None


def _masked_camera(footprint_00, fill_value=2.0):
    """4x4 native RGB/depth, HALF tier -> 2x2 output.

    ``footprint_00`` is the (2, 2) native depth block feeding output pixel
    (0, 0); every other output pixel's footprint is filled uniformly with
    ``fill_value`` (never itself a boundary) so only pixel (0, 0) is under
    test.
    """
    depth = torch.full((4, 4), float(fill_value), dtype=torch.float32)
    depth[0, 0] = footprint_00[0][0]
    depth[0, 1] = footprint_00[0][1]
    depth[1, 0] = footprint_00[1][0]
    depth[1, 1] = footprint_00[1][1]
    return SimpleNamespace(
        original_image=torch.zeros((3, 4, 4), dtype=torch.float32),
        _sensor_depth_cache=depth.unsqueeze(0),
        image_width=4,
        image_height=4,
        width=4,
        height=4,
        fx=40.0,
        fy=40.0,
        cx=1.5,
        cy=1.5,
        gsplat=object(),
    )


def test_masked_mode_flags_footprint_with_large_depth_gap_as_boundary():
    # Native footprint for output pixel (0,0) straddles two clusters: 1.0m
    # and 3.0m -- exactly the "2x2 native block spanning a foreground/
    # background boundary" scenario this mask exists to catch.
    camera = _masked_camera([[1.0, 3.0], [1.0, 3.0]])

    materialize_resolution_tier(camera, ResolutionTier.HALF, resampling_mode="masked")

    weight = camera._resolution_depth_confidence_weight
    assert tuple(weight.shape) == (2, 2)
    # 2m gap at ~1m range is far beyond both the abs floor (0.05m default) and
    # the relative fraction (0.02 * 1.0m = 0.02m) -- must be heavily downweighted.
    assert weight[0, 0].item() < 0.2
    # The uniform-fill footprints must NOT be flagged.
    assert weight[0, 1].item() == 1.0
    assert weight[1, 0].item() == 1.0
    assert weight[1, 1].item() == 1.0


def test_masked_mode_does_not_flag_footprint_within_threshold():
    camera = _masked_camera([[2.00, 2.01], [2.00, 2.01]])

    materialize_resolution_tier(camera, ResolutionTier.HALF, resampling_mode="masked")

    weight = camera._resolution_depth_confidence_weight
    # delta_z=0.01m << threshold (abs floor 0.05m) -- must not be downweighted.
    assert weight[0, 0].item() == 1.0


def test_masked_mode_excludes_invalid_depth_samples_from_delta_z():
    # Footprint has one real 1.0m sample and three invalid (zero) samples --
    # a naive computation that treats 0.0 as a real depth would see a fake
    # 1.0m "discontinuity" against nothing; the real answer is "no evidence".
    camera = _masked_camera([[1.0, 0.0], [0.0, 0.0]])

    materialize_resolution_tier(camera, ResolutionTier.HALF, resampling_mode="masked")

    weight = camera._resolution_depth_confidence_weight
    assert weight[0, 0].item() == 1.0


def test_all_resampling_modes_produce_shape_and_dtype_parity():
    shapes = {}
    for mode in RESAMPLING_MODES:
        camera = _camera_with_distinct_pixels()
        materialize_resolution_tier(camera, ResolutionTier.QUARTER, resampling_mode=mode)
        shapes[mode] = (
            tuple(camera.original_image.shape),
            camera.original_image.dtype,
            tuple(camera._sensor_depth_cache.shape),
            camera._sensor_depth_cache.dtype,
        )
    assert len(set(shapes.values())) == 1


def test_invalid_resampling_mode_raises():
    import pytest

    camera = _camera_with_distinct_pixels()
    with pytest.raises(ValueError):
        materialize_resolution_tier(camera, ResolutionTier.QUARTER, resampling_mode="bogus")


# ---------------------------------------------------------------------------
# Issue #1833: depth_downsample selection for the depth-supervision loss's
# sensor depth ("nearest" default / "valid_area" validity-aware average),
# independent of resampling_mode.
# ---------------------------------------------------------------------------


def _camera_with_sparse_depth_holes():
    """8x16 native RGB (unique per-pixel values, unused by these tests) with a
    depth map that is exactly divisible into QUARTER's 4x4-pixel native
    footprints (2 output rows x 4 output cols), so each output pixel's
    valid-area average can be hand-computed as a simple mean over a known,
    non-overlapping 4x4 block:

    * block (0, 0) (rows 0-3, cols 0-3): two valid pixels, depth 2.0 and 4.0
      at (0, 0) and (1, 1) -- mean of valid = 3.0.
    * block (0, 1) (rows 0-3, cols 4-7): no valid pixels at all -- 0.0, not
      NaN/inf.
    * every other block: exactly one valid pixel at its native (0, 0) corner
      -- mean of valid equals that single value, and (because it sits at the
      block's nearest-neighbor sample point) also equals what "nearest"
      downsampling already produces, so those blocks double as a
      nearest-vs-valid_area regression check.
    """
    depth = torch.zeros((1, 8, 16), dtype=torch.float32)
    depth[0, 0, 0] = 2.0
    depth[0, 1, 1] = 4.0
    # block (0,1) rows 0-3 cols 4-7 -- deliberately left all-zero (no valid pixel).
    depth[0, 0, 8] = 5.0  # block (0,2)'s native (0,0) corner
    depth[0, 4, 0] = 6.0  # block (1,0)'s native (0,0) corner
    depth[0, 4, 4] = 7.0  # block (1,1)'s native (0,0) corner
    depth[0, 4, 8] = 8.0  # block (1,2)'s native (0,0) corner
    depth[0, 0, 12] = 9.0  # block (0,3)'s native (0,0) corner
    depth[0, 4, 12] = 10.0  # block (1,3)'s native (0,0) corner
    return SimpleNamespace(
        original_image=torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16),
        _sensor_depth_cache=depth,
        image_width=16,
        image_height=8,
        width=16,
        height=8,
        fx=80.0,
        fy=40.0,
        cx=7.5,
        cy=3.5,
        gsplat=object(),
    )


def test_depth_downsample_default_omitted_matches_explicit_nearest():
    """Zero behavior change: omitting depth_downsample must match 'nearest' exactly."""
    camera_default = _camera_with_sparse_depth_holes()
    camera_explicit = _camera_with_sparse_depth_holes()

    materialize_resolution_tier(camera_default, ResolutionTier.QUARTER)
    materialize_resolution_tier(
        camera_explicit, ResolutionTier.QUARTER, depth_downsample="nearest"
    )

    assert torch.equal(camera_default._sensor_depth_cache, camera_explicit._sensor_depth_cache)


def test_depth_downsample_valid_area_averages_only_valid_pixels_per_block():
    camera = _camera_with_sparse_depth_holes()

    materialize_resolution_tier(camera, ResolutionTier.QUARTER, depth_downsample="valid_area")

    out = camera._sensor_depth_cache
    assert tuple(out.shape) == (1, 2, 4)
    assert math.isclose(float(out[0, 0, 0]), 3.0, rel_tol=1e-6)  # (2.0 + 4.0) / 2
    assert float(out[0, 0, 1]) == 0.0  # no valid pixel in this block -- exactly 0, not NaN
    assert math.isclose(float(out[0, 0, 2]), 5.0, rel_tol=1e-6)
    assert math.isclose(float(out[0, 1, 0]), 6.0, rel_tol=1e-6)
    assert math.isclose(float(out[0, 1, 1]), 7.0, rel_tol=1e-6)
    assert math.isclose(float(out[0, 1, 2]), 8.0, rel_tol=1e-6)
    assert math.isclose(float(out[0, 0, 3]), 9.0, rel_tol=1e-6)
    assert math.isclose(float(out[0, 1, 3]), 10.0, rel_tol=1e-6)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("prepared", [False, True])
def test_valid_area_excludes_nonfinite_depth_without_poisoning_valid_neighbors(invalid, prepared):
    camera = _camera_with_sparse_depth_holes()
    camera._sensor_depth_cache[0, 2, 2] = invalid
    camera._sensor_depth_cache[0, :4, 4:8] = invalid
    if prepared:
        _, out, _, _ = materialize_resolution_tensors(
            rgb_hwc=camera.original_image.permute(1, 2, 0),
            depth_hw=camera._sensor_depth_cache[0],
            camera_intrinsics=torch.tensor([[80., 0., 7.5], [0., 40., 3.5], [0., 0., 1.]]),
            tier=ResolutionTier.QUARTER,
            depth_downsample="valid_area",
        )
    else:
        materialize_resolution_tier(camera, ResolutionTier.QUARTER, depth_downsample="valid_area")
        out = camera._sensor_depth_cache[0]

    expected = torch.tensor([[3., 0., 5., 9.], [6., 7., 8., 10.]])
    torch.testing.assert_close(out, expected)
    assert torch.isfinite(out).all()


def test_depth_downsample_valid_area_differs_from_nearest_when_holes_present():
    """Block (0, 0) is the discriminating case: F.interpolate(mode="nearest")
    downsampling native 8x16 to 2x4 (scale 4) samples output (0, 0) at exactly
    native (0, 0) -- which this fixture sets to 2.0, ignoring the OTHER valid
    pixel in that same footprint at native (1, 1) (4.0). valid_area instead
    averages every valid pixel in the footprint: (2.0 + 4.0) / 2 = 3.0.
    """
    camera_nearest = _camera_with_sparse_depth_holes()
    camera_valid_area = _camera_with_sparse_depth_holes()

    materialize_resolution_tier(camera_nearest, ResolutionTier.QUARTER, depth_downsample="nearest")
    materialize_resolution_tier(
        camera_valid_area, ResolutionTier.QUARTER, depth_downsample="valid_area"
    )

    assert math.isclose(float(camera_nearest._sensor_depth_cache[0, 0, 0]), 2.0, rel_tol=1e-6)
    assert math.isclose(float(camera_valid_area._sensor_depth_cache[0, 0, 0]), 3.0, rel_tol=1e-6)
    assert not torch.equal(
        camera_nearest._sensor_depth_cache, camera_valid_area._sensor_depth_cache
    )


def test_depth_downsample_valid_area_all_valid_matches_plain_area_mean():
    """With no holes at all, valid_area must reduce to the ordinary area mean
    per block (every pixel counts), since the validity mask is then all-ones
    and cancels out of sum(d*1)/sum(1) = mean(d).
    """
    depth = torch.arange(8 * 16, dtype=torch.float32).reshape(1, 8, 16) + 1.0  # all > 0
    camera = SimpleNamespace(
        original_image=torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16),
        _sensor_depth_cache=depth.clone(),
        image_width=16,
        image_height=8,
        width=16,
        height=8,
        fx=80.0,
        fy=40.0,
        cx=7.5,
        cy=3.5,
        gsplat=object(),
    )

    materialize_resolution_tier(camera, ResolutionTier.QUARTER, depth_downsample="valid_area")

    expected = torch.nn.functional.adaptive_avg_pool2d(depth.unsqueeze(0), output_size=(2, 4)).squeeze(0)
    assert torch.allclose(camera._sensor_depth_cache, expected, atol=1e-5)


def test_depth_downsample_ignored_at_native_tier_and_ray_preserving():
    """Native (factor==1) and ray_preserving never resize depth at all, so
    depth_downsample is a no-op for both -- confirms it doesn't silently
    invent a resize where the codebase intentionally has none."""
    camera_native_nearest = _camera_with_sparse_depth_holes()
    camera_native_valid_area = _camera_with_sparse_depth_holes()
    materialize_resolution_tier(camera_native_nearest, ResolutionTier.NATIVE, depth_downsample="nearest")
    materialize_resolution_tier(
        camera_native_valid_area, ResolutionTier.NATIVE, depth_downsample="valid_area"
    )
    assert torch.equal(
        camera_native_nearest._sensor_depth_cache, camera_native_valid_area._sensor_depth_cache
    )

    camera_rp_nearest = _camera_with_sparse_depth_holes()
    camera_rp_valid_area = _camera_with_sparse_depth_holes()
    materialize_resolution_tier(
        camera_rp_nearest, ResolutionTier.QUARTER, resampling_mode="ray_preserving", depth_downsample="nearest"
    )
    materialize_resolution_tier(
        camera_rp_valid_area, ResolutionTier.QUARTER, resampling_mode="ray_preserving", depth_downsample="valid_area"
    )
    assert torch.equal(
        camera_rp_nearest._sensor_depth_cache, camera_rp_valid_area._sensor_depth_cache
    )


def test_invalid_depth_downsample_raises():
    import pytest

    camera = _camera_with_distinct_pixels()
    with pytest.raises(ValueError):
        materialize_resolution_tier(camera, ResolutionTier.QUARTER, depth_downsample="bogus")
