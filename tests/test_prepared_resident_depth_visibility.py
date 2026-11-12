"""Prepared READY depth is authoritative without legacy streaming producers."""

from types import SimpleNamespace

import torch

from eligsir.mapping.integration.depth_helpers import _get_sensor_depth


def test_resident_prepared_depth_needs_no_streaming_path_or_loader() -> None:
    depth = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    cam = SimpleNamespace(_sensor_depth_cache=depth)

    resolved = _get_sensor_depth(cam, 2, 3)

    assert resolved.shape == (1, 2, 3)
    assert resolved.device == depth.device
    assert resolved.data_ptr() == depth.data_ptr()
    assert not hasattr(cam, "_streaming_depth_path")
    assert not hasattr(cam, "_sensor_depth_loader")


def test_resident_prepared_depth_resizes_on_its_existing_device() -> None:
    depth = torch.ones((1, 2, 3), dtype=torch.float32)
    cam = SimpleNamespace(_sensor_depth_cache=depth)

    resolved = _get_sensor_depth(cam, 4, 6)

    assert resolved.shape == (1, 4, 6)
    assert resolved.device == depth.device
    assert cam._sensor_depth_cache is depth


def test_depth_failure_sentinel_needs_no_legacy_producer_fields() -> None:
    cam = SimpleNamespace(_sensor_depth_cache=False)

    assert _get_sensor_depth(cam, 2, 3) is None
