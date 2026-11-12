"""Issue #570: the canonical preview event schemas.

The headline requirement is that a training pair whose target and render came
from *different cameras* must be **unconstructible**, not merely detectable.
These tests are written to fail against a stubbed/no-op implementation and
against the naive single-``camera_id`` schema (see
``test_naive_single_camera_id_schema_cannot_express_the_fixture``).
"""
from __future__ import annotations

import numpy as np
import pytest

from eligsir.viewers.events import (
    PREVIEW_SCHEMA_VERSION,
    CameraRole,
    CapturedImage,
    ImageKind,
    MapPreviewSnapshot,
    MappingPhase,
    MetricsPreview,
    PreviewChannel,
    PreviewSchemaError,
    RawInputPreview,
    TrainingPairPreview,
    TrajectoryPreview,
    channel_for_event,
)


def _img(camera_id=7, kind=ImageKind.TARGET, shape=(4, 6, 3), source_frame_id=None):
    return CapturedImage(
        np.zeros(shape, dtype=np.uint8), camera_id, kind, source_frame_id
    )


def _pair(**over):
    kwargs = dict(
        iteration=100,
        wall_time_s=1.5,
        target=_img(kind=ImageKind.TARGET),
        render=_img(kind=ImageKind.RENDER),
        optimizer_step_id=42,
        supervision_view_count=3,
        camera_role=CameraRole.REPLAY,
        phase=MappingPhase.ACTIVE_MAPPING,
        n_gaussians=12345,
    )
    kwargs.update(over)
    return TrainingPairPreview(**kwargs)


# --------------------------------------------------------------------------
# THE headline contract
# --------------------------------------------------------------------------
def test_mismatched_camera_identity_is_rejected():
    with pytest.raises(PreviewSchemaError, match="camera identity mismatch"):
        _pair(
            target=_img(camera_id=731, kind=ImageKind.TARGET),
            render=_img(camera_id=122, kind=ImageKind.RENDER),
        )


def test_matched_camera_identity_is_accepted_and_exposed():
    pair = _pair(
        target=_img(camera_id=731, kind=ImageKind.TARGET, source_frame_id=900),
        render=_img(camera_id=731, kind=ImageKind.RENDER, source_frame_id=900),
    )
    assert pair.camera_id == 731
    assert pair.source_frame_id == 900
    assert pair.resolution == (6, 4)
    assert pair.schema_version == PREVIEW_SCHEMA_VERSION


def test_mismatched_resolution_is_rejected():
    with pytest.raises(PreviewSchemaError, match="resolution mismatch"):
        _pair(
            target=_img(kind=ImageKind.TARGET, shape=(4, 6, 3)),
            render=_img(kind=ImageKind.RENDER, shape=(8, 12, 3)),
        )


def test_mismatched_source_frame_id_is_rejected():
    with pytest.raises(PreviewSchemaError, match="source_frame_id mismatch"):
        _pair(
            target=_img(kind=ImageKind.TARGET, source_frame_id=731),
            render=_img(kind=ImageKind.RENDER, source_frame_id=122),
        )


def test_swapped_halves_are_rejected_even_with_matching_camera_id():
    """Same camera, right shapes -- only the roles are swapped."""
    with pytest.raises(PreviewSchemaError, match="expected 'target'"):
        _pair(target=_img(kind=ImageKind.RENDER), render=_img(kind=ImageKind.RENDER))
    with pytest.raises(PreviewSchemaError, match="expected 'render'"):
        _pair(target=_img(kind=ImageKind.TARGET), render=_img(kind=ImageKind.TARGET))


def test_bare_arrays_are_rejected_as_pair_halves():
    """A bare array carries no identity, so it could never be checked."""
    with pytest.raises(PreviewSchemaError, match="must be a CapturedImage"):
        _pair(target=np.zeros((4, 6, 3), dtype=np.uint8))


def test_naive_single_camera_id_schema_cannot_express_the_fixture():
    """The discriminating comparison against the alternative design.

    A schema with one ``camera_id`` and two bare arrays has NO input through
    which a mismatched pair can be described -- so its "mismatch is rejected"
    test would be vacuous. This test asserts the real schema materially differs:
    the two halves carry independent identity that can disagree.
    """
    target = _img(camera_id=731, kind=ImageKind.TARGET)
    render = _img(camera_id=122, kind=ImageKind.RENDER)
    # Identity lives on each half, independently...
    assert target.camera_id != render.camera_id
    # ...which is precisely why the event layer can refuse it.
    with pytest.raises(PreviewSchemaError):
        _pair(target=target, render=render)


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------
def test_arrays_are_frozen_in_place():
    arr = np.zeros((4, 6, 3), dtype=np.uint8)
    img = CapturedImage(arr, 1, ImageKind.TARGET)
    with pytest.raises(ValueError):
        img.array[0, 0, 0] = 255
    # freeze-in-place: the caller's own handle is frozen too (ownership transfer)
    with pytest.raises(ValueError):
        arr[0, 0, 0] = 255


def test_copy_of_leaves_the_callers_array_writeable():
    arr = np.zeros((4, 6, 3), dtype=np.uint8)
    img = CapturedImage.copy_of(arr, 1, ImageKind.TARGET)
    arr[0, 0, 0] = 255  # still the caller's
    assert img.array[0, 0, 0] == 0
    with pytest.raises(ValueError):
        img.array[0, 0, 0] = 9


def test_event_attributes_cannot_be_rebound():
    pair = _pair()
    with pytest.raises(Exception):
        pair.iteration = 5


def test_depth_must_match_its_rgb_counterpart():
    with pytest.raises(PreviewSchemaError, match="target_depth shape"):
        _pair(target_depth=np.zeros((9, 9), dtype=np.float32))
    ok = _pair(target_depth=np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(ValueError):
        ok.target_depth[0, 0] = 1.0


# --------------------------------------------------------------------------
# no silent zero
# --------------------------------------------------------------------------
def test_unmeasured_quantities_default_to_none_not_zero():
    pair = _pair()
    assert pair.psnr_db is None
    assert pair.map_version is None
    assert pair.frames_seen is None
    assert pair.frames_ingested is None
    assert pair.keyframes_admitted is None


def test_zero_is_a_legal_measurement_not_a_sentinel():
    """Distinguishability, not a ban on zero: 0 admitted frames is real."""
    pair = _pair(keyframes_admitted=0, frames_seen=0)
    assert pair.keyframes_admitted == 0
    assert pair.frames_seen == 0


def test_negative_counters_are_refused():
    with pytest.raises(PreviewSchemaError, match="frames_seen"):
        _pair(frames_seen=-1)


def test_non_finite_psnr_is_refused():
    with pytest.raises(PreviewSchemaError, match="psnr_db"):
        _pair(psnr_db=float("nan"))


def test_metrics_refuse_non_finite_values():
    assert MetricsPreview(iteration=1, wall_time_s=0.0, metrics={"it_s": 82.4}).as_dict() == {
        "it_s": 82.4
    }
    with pytest.raises(PreviewSchemaError, match="unmeasured scalar"):
        MetricsPreview(iteration=1, wall_time_s=0.0, metrics={"psnr": float("inf")})


# --------------------------------------------------------------------------
# raw input is a DIFFERENT stream
# --------------------------------------------------------------------------
def test_raw_input_has_no_render_and_is_a_distinct_channel():
    raw = RawInputPreview(
        iteration=10, wall_time_s=1.0, source_frame_id=911,
        rgb=np.zeros((4, 6, 3), dtype=np.uint8), timestamp_s=12.5,
    )
    assert not hasattr(raw, "render")
    assert not hasattr(raw, "target")
    assert channel_for_event(raw) is PreviewChannel.RAW_INPUT
    assert channel_for_event(_pair()) is PreviewChannel.TRAINING_PAIR


def test_raw_input_refuses_a_captured_image():
    with pytest.raises(PreviewSchemaError, match="not a CapturedImage"):
        RawInputPreview(
            iteration=1, wall_time_s=0.0, source_frame_id=1,
            rgb=_img(kind=ImageKind.TARGET),
        )


def test_raw_input_carries_its_own_identity_fields():
    raw = RawInputPreview(
        iteration=10, wall_time_s=1.0, source_frame_id=911,
        rgb=np.zeros((4, 6, 3), dtype=np.uint8), timestamp_s=12.5,
        tracking_status="OK",
    )
    assert (raw.source_frame_id, raw.timestamp_s, raw.tracking_status) == (911, 12.5, "OK")


# --------------------------------------------------------------------------
# closed enums / schema version / routing
# --------------------------------------------------------------------------
def test_camera_role_and_phase_are_closed_enums():
    with pytest.raises(PreviewSchemaError, match="camera_role"):
        _pair(camera_role="REPLAY")
    with pytest.raises(PreviewSchemaError, match="phase"):
        _pair(phase="ACTIVE_MAPPING")
    assert {r.value for r in CameraRole} == {"NEW", "LOCAL", "REPLAY", "GLOBAL", "OTHER"}


def test_event_carries_no_preview_seq_field():
    """Sequence identity is minted by the bus, never by a producer."""
    assert "preview_seq" not in TrainingPairPreview.__dataclass_fields__


def test_wrong_schema_version_is_refused():
    with pytest.raises(PreviewSchemaError, match="schema_version"):
        _pair(schema_version=PREVIEW_SCHEMA_VERSION + 1)


def test_channel_routing_uses_exact_type_not_isinstance():
    class Sneaky(TrainingPairPreview):
        pass

    sneaky = Sneaky(
        iteration=1, wall_time_s=0.0,
        target=_img(kind=ImageKind.TARGET), render=_img(kind=ImageKind.RENDER),
    )
    with pytest.raises(PreviewSchemaError, match="not a canonical preview event type"):
        channel_for_event(sneaky)


def test_trajectory_shape_is_enforced():
    ok = TrajectoryPreview(iteration=1, wall_time_s=0.0, poses_est=np.zeros((3, 4, 4)))
    assert ok.poses_est.shape == (3, 4, 4)
    with pytest.raises(PreviewSchemaError, match=r"poses_est must be \(N, 4, 4\)"):
        TrajectoryPreview(iteration=1, wall_time_s=0.0, poses_est=np.zeros((3, 3, 3)))


# --------------------------------------------------------------------------
# map snapshot: schema only, but an honest one
# --------------------------------------------------------------------------
def test_map_snapshot_states_sampled_vs_total_honestly():
    snap = MapPreviewSnapshot(
        iteration=1, wall_time_s=0.0, map_version=17,
        positions=np.zeros((100, 3), dtype=np.float32), n_total_gaussians=486_000,
    )
    assert (snap.n_sampled, snap.n_total_gaussians, snap.is_complete) == (100, 486_000, False)


def test_map_snapshot_refuses_a_total_below_its_own_sample():
    with pytest.raises(PreviewSchemaError, match="n_total_gaussians"):
        MapPreviewSnapshot(
            iteration=1, wall_time_s=0.0, map_version=1,
            positions=np.zeros((100, 3), dtype=np.float32), n_total_gaussians=10,
        )


def test_map_snapshot_refuses_a_non_array_payload():
    """No GaussianMapReadView / live tensor may enter the schema."""

    class FakeReadView:
        pass

    with pytest.raises(PreviewSchemaError, match="must be a numpy.ndarray"):
        MapPreviewSnapshot(
            iteration=1, wall_time_s=0.0, map_version=1,
            positions=FakeReadView(), n_total_gaussians=1,
        )
