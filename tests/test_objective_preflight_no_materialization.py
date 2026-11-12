"""Prepared preflight checks residency, never creates a VGA depth temporary."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eligsir.mapping.integration import depth_helpers
from eligsir.optimization.gaussian.objective import (
    preflight_prepared_objective_auxiliaries,
)


def _args(**overrides):
    return SimpleNamespace(
        **dict(
            streaming_ray_semantics_enabled=True,
            streaming_appearance_only_enabled=False,
            streaming_free_space_loss_weight=1.0,
            surface_confidence_depth_loss=False,
        ) | overrides
    )


def _forbid(*_args, **_kwargs):
    raise AssertionError("residency inspection must not materialize depth")


@pytest.mark.parametrize("shape", [(480, 640), (240, 320), (120, 160), (720, 1280)])
@pytest.mark.parametrize("channel_dimension", [False, True])
def test_preflight_borrows_native_half_quarter_non_vga_depth_without_resize(
    monkeypatch, shape, channel_dimension
):
    depth = torch.ones(((1,) if channel_dimension else ()) + shape)
    cam = SimpleNamespace(_source_identity=object(), _sensor_depth_cache=depth)
    before = vars(cam).copy()
    monkeypatch.setattr(torch.nn.functional, "interpolate", _forbid)
    monkeypatch.setattr(depth_helpers, "_record_sensor_depth_cache_load", _forbid)

    preflight_prepared_objective_auxiliaries(cam, _args(), "cpu", use_depth_loss=True)

    assert vars(cam) == before
    assert cam._sensor_depth_cache is depth
    assert depth._version == 0


@pytest.mark.parametrize("feature", ["ray", "appearance", "free_space"])
def test_preflight_rejects_wrong_device_without_first_resizing(monkeypatch, feature):
    cam = SimpleNamespace(_source_identity=object(), _sensor_depth_cache=torch.ones(1, 12, 18))
    args = _args(
        streaming_ray_semantics_enabled=feature == "ray",
        streaming_appearance_only_enabled=feature == "appearance",
        streaming_free_space_loss_weight=float(feature == "free_space"),
    )
    monkeypatch.setattr(torch.nn.functional, "interpolate", _forbid)
    with pytest.raises(ValueError, match="sensor_depth.*device-resident"):
        preflight_prepared_objective_auxiliaries(cam, args, "cuda:0", use_depth_loss=True)


def test_preflight_rejects_lazy_cache_miss_without_loading_or_latching():
    loads = []

    def load():
        loads.append(True)
        return np.ones((2, 3), dtype=np.uint16)

    cam = SimpleNamespace(
        _source_identity=object(), _sensor_depth_cache=None,
        _sensor_depth_loader=load, _streaming_depth_scale=1000.0,
    )
    before = vars(cam).copy()
    with pytest.raises(ValueError, match="sensor_depth.*resident"):
        preflight_prepared_objective_auxiliaries(cam, _args(), "cpu", use_depth_loss=True)
    assert loads == []
    assert vars(cam) == before


@pytest.mark.parametrize("cached", [None, False])
def test_preflight_absent_depth_preserves_optional_semantics(cached):
    cam = SimpleNamespace(_source_identity=object(), _sensor_depth_cache=cached)
    preflight_prepared_objective_auxiliaries(cam, _args(), "cpu", use_depth_loss=True)
    assert cam._sensor_depth_cache is cached


def test_preflight_disabled_depth_auxiliaries_do_not_load_lazy_depth():
    cam = SimpleNamespace(_source_identity=object(), _sensor_depth_loader=_forbid)
    args = _args(streaming_ray_semantics_enabled=False, streaming_free_space_loss_weight=1.0)
    preflight_prepared_objective_auxiliaries(cam, args, "cuda:0", use_depth_loss=False)
    assert not hasattr(cam, "_sensor_depth_cache")


@pytest.mark.parametrize("cache_state", ["missing", "stale", "resident"])
def test_surface_preflight_never_predicts_or_changes_cached_prior(monkeypatch, cache_state):
    prior = SimpleNamespace(confidence=torch.ones(1, 6, 8))
    cache = ((6, 8, "finite_difference", False), prior)
    cam = SimpleNamespace(
        _source_identity=object(), _streaming_frame=object(),
        image_height=6, image_width=8,
        _surface_prior_cache=None if cache_state == "missing" else cache,
    )
    if cache_state == "stale":
        cam.image_height = 12
    before = vars(cam).copy()
    predictions = []

    def predict(*_args, **_kwargs):
        predictions.append(True)
        return prior

    monkeypatch.setattr(depth_helpers.SensorDepthProvider, "predict", predict)
    args = _args(surface_confidence_depth_loss=True, streaming_depth_filter_enabled=False)
    if cache_state == "resident":
        preflight_prepared_objective_auxiliaries(cam, args, "cpu", use_depth_loss=True)
    else:
        with pytest.raises(ValueError, match="surface_confidence.*resident"):
            preflight_prepared_objective_auxiliaries(cam, args, "cpu", use_depth_loss=True)
    assert predictions == []
    assert vars(cam) == before


def test_preflight_known_absent_depth_does_not_invoke_attached_loader():
    cam = SimpleNamespace(
        _source_identity=object(), _sensor_depth_cache=False, _sensor_depth_loader=_forbid
    )
    preflight_prepared_objective_auxiliaries(cam, _args(), "cpu", use_depth_loss=True)
    assert cam._sensor_depth_cache is False


def test_preflight_explicit_absent_depth_path_preserves_nonlatched_absence():
    cam = SimpleNamespace(_source_identity=object(), _streaming_depth_path=None)
    before = vars(cam).copy()
    stats = depth_helpers._DEPTH_UNAVAILABLE_STATS.copy()
    preflight_prepared_objective_auxiliaries(cam, _args(), "cpu", use_depth_loss=True)
    assert vars(cam) == before
    assert depth_helpers._DEPTH_UNAVAILABLE_STATS == stats


@pytest.mark.parametrize("source", ["path", "array", "bytes"])
def test_preflight_unresident_source_is_rejected_without_io(monkeypatch, source):
    cam = SimpleNamespace(
        _source_identity=object(),
        _streaming_depth_path="missing.png" if source == "path" else None,
        _streaming_frame=SimpleNamespace(
            _depth_np=np.ones((2, 3)) if source == "array" else None,
            _depth_bytes=b"not decoded" if source == "bytes" else None,
        ),
    )
    before = vars(cam).copy()
    monkeypatch.setattr(depth_helpers.os.path, "exists", _forbid)
    monkeypatch.setattr(torch.nn.functional, "interpolate", _forbid)
    with pytest.raises(ValueError, match="sensor_depth.*not resident"):
        preflight_prepared_objective_auxiliaries(cam, _args(), "cpu", use_depth_loss=True)
    assert vars(cam) == before


def test_surface_preflight_checks_real_resident_confidence_device(monkeypatch):
    prior = SimpleNamespace(confidence=torch.ones(1, 6, 8))
    cam = SimpleNamespace(
        _source_identity=object(), _streaming_frame=object(),
        image_height=6, image_width=8,
        _surface_prior_cache=((6, 8, "finite_difference", False), prior),
    )
    monkeypatch.setattr(depth_helpers.SensorDepthProvider, "predict", _forbid)
    args = _args(surface_confidence_depth_loss=True, streaming_depth_filter_enabled=False)
    with pytest.raises(ValueError, match="surface_confidence.*device-resident"):
        preflight_prepared_objective_auxiliaries(cam, args, "cuda:0", use_depth_loss=True)
    assert cam._surface_prior_cache[1] is prior
