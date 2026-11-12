"""Prepared READY cameras must expose their sealed sensor depth to spatial replay."""

from types import SimpleNamespace

import torch


def test_prepared_device_depth_cache_is_usable_without_legacy_reload_markers():
    from eligsir.utils.comparison_report import _get_sensor_depth

    depth = torch.full((1, 3, 4), 1.25, dtype=torch.float32)
    prepared_camera = SimpleNamespace(_sensor_depth_cache=depth)

    assert _get_sensor_depth(prepared_camera, 3, 4) is depth
