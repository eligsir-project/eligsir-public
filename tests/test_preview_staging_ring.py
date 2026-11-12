"""Tests for eligsir/viewers/staging.py (issue #571, scoped).

Torch-free skip guard: these tests exercise the CPU-tensor fallback path
(``tensor.is_cuda is False``), which is the path exercised in CI without a
GPU. The real async-D2H/event-poll path additionally requires a CUDA/HIP
device and is covered separately (see this module's GPU-only test, skipped
without one).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eligsir.viewers.staging import MapSnapshotStaging, PreviewStagingRing, SlotState


def _cpu_tensor(shape, value=1.0):
    return torch.full(shape, float(value), dtype=torch.float32)


def test_submit_and_poll_cpu_fallback_roundtrips():
    ring = PreviewStagingRing(n_slots=2)
    ok = ring.try_submit_pair(_cpu_tensor((4, 4, 3), 1.0), _cpu_tensor((4, 4, 3), 2.0), {"iteration": 5})
    assert ok
    staged = ring.poll_completed()
    assert staged is not None
    assert staged.meta["iteration"] == 5
    assert np.allclose(staged.target, 1.0)
    assert np.allclose(staged.render, 2.0)
    # slot freed -> a second submission must succeed
    assert ring.poll_completed() is None


def test_full_ring_drops_without_blocking():
    ring = PreviewStagingRing(n_slots=1)
    assert ring.try_submit_pair(_cpu_tensor((2, 2, 3)), _cpu_tensor((2, 2, 3)), {"iteration": 1})
    # ring full (slot not yet drained) -> must drop, not raise, not block
    dropped_ok = ring.try_submit_pair(_cpu_tensor((2, 2, 3)), _cpu_tensor((2, 2, 3)), {"iteration": 2})
    assert dropped_ok is False
    assert ring.stats()["staging_dropped"] == 1
    # draining frees the slot for reuse
    staged = ring.poll_completed()
    assert staged.meta["iteration"] == 1
    assert ring.try_submit_pair(_cpu_tensor((2, 2, 3)), _cpu_tensor((2, 2, 3)), {"iteration": 3})


def test_poll_completed_returns_none_when_empty():
    ring = PreviewStagingRing(n_slots=2)
    assert ring.poll_completed() is None


def test_materialized_array_is_owned_not_a_view_into_the_pinned_buffer():
    ring = PreviewStagingRing(n_slots=1)
    ring.try_submit_pair(_cpu_tensor((2, 2, 3), 9.0), _cpu_tensor((2, 2, 3), 9.0), {"iteration": 1})
    staged = ring.poll_completed()
    assert staged.target.flags["OWNDATA"] or staged.target.base is None
    staged.target[0, 0, 0] = 0.0  # must not raise / must not alias internal state
    ring.try_submit_pair(_cpu_tensor((2, 2, 3), 3.0), _cpu_tensor((2, 2, 3), 3.0), {"iteration": 2})
    staged2 = ring.poll_completed()
    assert np.allclose(staged2.target, 3.0)  # unaffected by the mutation above


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA/HIP device")
def test_gpu_path_never_blocks_and_completes_via_event_poll():
    device = "cuda"
    ring = PreviewStagingRing(n_slots=2)
    target = torch.full((8, 8, 3), 1.0, device=device)
    render = torch.full((8, 8, 3), 2.0, device=device)
    assert ring.try_submit_pair(target, render, {"iteration": 1})
    # Poll (non-blocking) until the event fires; must never call .synchronize().
    for _ in range(10000):
        staged = ring.poll_completed()
        if staged is not None:
            assert np.allclose(staged.target, 1.0)
            assert np.allclose(staged.render, 2.0)
            return
    pytest.fail("GPU staging copy did not complete after 10000 non-blocking polls")


# -- MapSnapshotStaging (issue #1042 -- MapPreviewSnapshot transfer) --------


def test_map_staging_submit_and_poll_cpu_fallback_roundtrips():
    staging = MapSnapshotStaging()
    positions = torch.zeros((5, 3), dtype=torch.float32)
    positions[:, 0] = torch.arange(5, dtype=torch.float32)
    colors = torch.full((5, 3), 0.5, dtype=torch.float32)
    ok = staging.try_submit(positions, colors, {"iteration": 10, "map_version": 1, "n_total_gaussians": 5})
    assert ok
    staged = staging.poll_completed()
    assert staged is not None
    assert staged.meta["iteration"] == 10
    assert staged.meta["map_version"] == 1
    assert staged.positions.shape == (5, 3)
    assert np.allclose(staged.colors, 0.5)
    # slot freed -> a second poll returns nothing until the next submission
    assert staging.poll_completed() is None


def test_map_staging_supports_no_color_source():
    staging = MapSnapshotStaging()
    positions = torch.ones((3, 3), dtype=torch.float32)
    assert staging.try_submit(positions, None, {"iteration": 1})
    staged = staging.poll_completed()
    assert staged is not None
    assert staged.colors is None


def test_map_staging_single_slot_drops_when_busy_never_blocks():
    staging = MapSnapshotStaging()
    positions = torch.ones((2, 3), dtype=torch.float32)
    assert staging.try_submit(positions, None, {"iteration": 1})
    # slot still occupied (never polled) -> a second submission must drop,
    # not raise, not block (#571's "drop the snapshot when the staging
    # budget is busy")
    dropped_ok = staging.try_submit(positions, None, {"iteration": 2})
    assert dropped_ok is False
    assert staging.stats()["map_staging_dropped"] == 1
    staged = staging.poll_completed()
    assert staged.meta["iteration"] == 1
    # slot freed -> next submission succeeds
    assert staging.try_submit(positions, None, {"iteration": 3})


def test_map_staging_materialized_array_is_owned_not_a_view():
    staging = MapSnapshotStaging()
    positions = torch.full((2, 3), 9.0, dtype=torch.float32)
    staging.try_submit(positions, None, {"iteration": 1})
    staged = staging.poll_completed()
    assert staged.positions.flags["OWNDATA"] or staged.positions.base is None
    staged.positions[0, 0] = 0.0  # must not raise / must not alias internal state
    staging.try_submit(torch.full((2, 3), 3.0, dtype=torch.float32), None, {"iteration": 2})
    staged2 = staging.poll_completed()
    assert np.allclose(staged2.positions, 3.0)  # unaffected by the mutation above


# -- scales/quats/opacities (issue #1042 follow-up -- oriented ellipsoids) --


def test_map_staging_submit_and_poll_with_scale_quat_opacity_roundtrips():
    staging = MapSnapshotStaging()
    positions = torch.zeros((4, 3), dtype=torch.float32)
    colors = torch.full((4, 3), 0.5, dtype=torch.float32)
    scales = torch.full((4, 3), 0.1, dtype=torch.float32)
    quats = torch.zeros((4, 4), dtype=torch.float32)
    quats[:, 0] = 1.0  # identity, wxyz
    opacities = torch.full((4, 1), 0.8, dtype=torch.float32)
    ok = staging.try_submit(
        positions,
        colors,
        {"iteration": 1, "map_version": 1, "n_total_gaussians": 4},
        scales=scales,
        quats=quats,
        opacities=opacities,
    )
    assert ok
    staged = staging.poll_completed()
    assert staged is not None
    assert staged.scales.shape == (4, 3)
    assert np.allclose(staged.scales, 0.1)
    assert staged.quats.shape == (4, 4)
    assert np.allclose(staged.quats[:, 0], 1.0)
    assert staged.opacities.shape == (4, 1)
    assert np.allclose(staged.opacities, 0.8)
    # slot freed -> a second poll returns nothing until the next submission
    assert staging.poll_completed() is None


def test_map_staging_omitted_scale_quat_opacity_stay_none():
    staging = MapSnapshotStaging()
    positions = torch.ones((3, 3), dtype=torch.float32)
    colors = torch.ones((3, 3), dtype=torch.float32)
    assert staging.try_submit(positions, colors, {"iteration": 1})
    staged = staging.poll_completed()
    assert staged.scales is None
    assert staged.quats is None
    assert staged.opacities is None
    # colors, the pre-existing optional half, is unaffected by the refactor
    assert np.allclose(staged.colors, 1.0)


def test_map_staging_partial_optional_attrs_independent_of_each_other():
    # scales present, quats/opacities absent -- each optional half must be
    # independently gate-able, not an all-or-nothing bundle.
    staging = MapSnapshotStaging()
    positions = torch.ones((2, 3), dtype=torch.float32)
    scales = torch.full((2, 3), 0.05, dtype=torch.float32)
    assert staging.try_submit(positions, None, {"iteration": 1}, scales=scales)
    staged = staging.poll_completed()
    assert staged.colors is None
    assert np.allclose(staged.scales, 0.05)
    assert staged.quats is None
    assert staged.opacities is None
