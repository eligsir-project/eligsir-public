from __future__ import annotations

import types

import numpy as np

from geometry.sensor_depth import SensorDepthConfig, SensorDepthProvider, stable_depth_mask


def test_stable_depth_mask_rejects_depth_spike():
    depth = np.ones((7, 7), dtype=np.float32)
    depth[3, 3] = 2.0
    mask = stable_depth_mask(
        depth,
        min_depth=0.1,
        max_depth=5.0,
        window=3,
        min_valid_ratio=0.5,
        max_range=0.2,
        median_thresh=0.2,
    )
    assert not bool(mask[3, 3])
    assert bool(mask[1, 1])


def test_sensor_provider_builds_surface_prior(monkeypatch):
    frame = types.SimpleNamespace(
        index=7,
        height=4,
        width=4,
        fx=100.0,
        fy=100.0,
        cx=2.0,
        cy=2.0,
        depth_fx=None,
        depth_fy=None,
        depth_cx=None,
        depth_cy=None,
        depth_scale=1.0,
    )

    import geometry.sensor_depth as sensor_depth

    monkeypatch.setattr(sensor_depth, "load_frame_depth_meters", lambda _frame: np.ones((4, 4), dtype=np.float32))
    prior = SensorDepthProvider(SensorDepthConfig(filter_enabled=False)).predict(frame)
    assert prior.depth.shape == (1, 4, 4)
    assert prior.confidence.shape == (1, 4, 4)
    assert prior.valid.all()
    assert prior.normals.shape == (3, 4, 4)
    assert prior.frame_uid == 7
