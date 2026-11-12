"""Real end-to-end functional check for the Map Preview three-way view-mode
toggle (coordinator follow-up to issue #1042's ellipsoid-rendering fix):
"Gaussian Map" (real reconstruction, splats) / "Gaussian Centers" (density-
debugging point cloud) / "Metadata" (future scope, documented no-op stub).

Spins up an ACTUAL ``viser.ViserServer`` (not a mock) and drives the real
``_build_ui``/``_apply`` code from ``viser_server.py`` directly -- no
subprocess needed since these are plain functions, but nothing here is
faked: the scene nodes, the GUI dropdown, and its ``on_update`` callback are
all real viser objects.

Confirms:
  1. Both the Gaussian-splats and point-cloud scene nodes are constructed
     and start in the expected default-visibility state ("Gaussian Map").
  2. A real ``map_snapshot`` command feeds BOTH nodes (via ``_apply``),
     regardless of which is currently visible.
  3. Manually driving the toggle (calling the real registered callback, the
     same one a browser client's dropdown change would invoke) flips
     visibility correctly across all three modes -- i.e. every code path
     executes cleanly under the toggle, including the "Metadata" stub.

Skipped automatically wherever ``viser`` is not installed (sidecar-only
dependency, deliberately not part of the trainer image -- see
``viser_sink.py``'s module docstring).
"""
from __future__ import annotations

import socket

import numpy as np
import pytest

pytest.importorskip("viser")

from eligsir.viewers.sinks.viser_server import (
    _MAP_MODE_CENTERS,
    _MAP_MODE_GAUSSIAN,
    _MAP_MODE_METADATA,
    _apply,
    _build_ui,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server():
    import viser

    port = _free_port()
    s = viser.ViserServer(host="127.0.0.1", port=port)
    yield s
    s.stop()


def test_build_ui_creates_both_map_nodes_with_gaussians_visible_by_default(server):
    handles, texts, dropdown, _cb = _build_ui(server)
    assert "map_gaussians" in handles
    assert "map_points" in handles
    assert handles["map_gaussians"].visible is True
    assert handles["map_points"].visible is False
    assert dropdown.value == _MAP_MODE_GAUSSIAN
    assert set(dropdown.options) == {_MAP_MODE_GAUSSIAN, _MAP_MODE_CENTERS, _MAP_MODE_METADATA}


def test_map_snapshot_feeds_both_nodes_regardless_of_visibility(server):
    handles, texts, dropdown, _cb = _build_ui(server)
    rng = np.random.default_rng(0)
    n = 50
    cmd = {
        "kind": "map_snapshot",
        "positions": rng.uniform(-2, 2, size=(n, 3)).astype(np.float32),
        "colors": rng.uniform(0, 1, size=(n, 3)).astype(np.float32),
        "scales": rng.uniform(0.01, 0.1, size=(n, 3)).astype(np.float32),
        "quats": (lambda q: q / np.linalg.norm(q, axis=1, keepdims=True))(
            rng.normal(size=(n, 4)).astype(np.float32)
        ),
        "opacities": rng.uniform(0.3, 1.0, size=(n, 1)).astype(np.float32),
        "map_version": 1,
        "n_total_gaussians": n,
        "n_sampled": n,
        "iteration": 500,
        "timestamp": 12.5,
    }
    # Must not raise -- both handles get fed in one _apply call.
    _apply(cmd, handles, texts)
    assert handles["map_gaussians"].buffer.shape[0] == n
    assert handles["map_points"].points.shape == (n, 3)


def test_toggling_dropdown_cycles_all_three_modes_correctly(server):
    handles, texts, dropdown, on_change = _build_ui(server)

    # Default: Gaussian Map shown, Gaussian Centers hidden.
    assert handles["map_gaussians"].visible is True
    assert handles["map_points"].visible is False
    assert texts["map_metadata_note"].content == ""

    # -> Gaussian Centers: only the point cloud shown.
    dropdown.value = _MAP_MODE_CENTERS
    on_change(None)  # the real callback ignores its event argument
    assert handles["map_gaussians"].visible is False
    assert handles["map_points"].visible is True
    assert texts["map_metadata_note"].content == ""

    # -> Metadata: both hidden, documented stub note shown. Must not raise
    # (the coordinator's "not yet implemented no-op" requirement).
    dropdown.value = _MAP_MODE_METADATA
    on_change(None)
    assert handles["map_gaussians"].visible is False
    assert handles["map_points"].visible is False
    assert "not yet implemented" in texts["map_metadata_note"].content

    # -> back to Gaussian Map: both flip back, note clears.
    dropdown.value = _MAP_MODE_GAUSSIAN
    on_change(None)
    assert handles["map_gaussians"].visible is True
    assert handles["map_points"].visible is False
    assert texts["map_metadata_note"].content == ""


def test_toggle_after_a_real_snapshot_still_shows_the_freshly_fed_data(server):
    # End-to-end: apply a real snapshot, THEN toggle through all three
    # modes, confirming the data-feed and the visibility-toggle are
    # independent and both correct together (not just each in isolation).
    handles, texts, dropdown, on_change = _build_ui(server)
    n = 10
    cmd = {
        "kind": "map_snapshot",
        "positions": np.ones((n, 3), dtype=np.float32) * 3.0,
        "colors": np.full((n, 3), 0.9, dtype=np.float32),
        "scales": np.full((n, 3), 0.05, dtype=np.float32),
        "quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        "opacities": np.full((n, 1), 1.0, dtype=np.float32),
        "map_version": 2,
        "n_total_gaussians": n,
        "n_sampled": n,
        "iteration": 1000,
        "timestamp": 20.0,
    }
    _apply(cmd, handles, texts)
    assert handles["map_gaussians"].visible is True  # still default-shown

    dropdown.value = _MAP_MODE_CENTERS
    on_change(None)
    assert handles["map_points"].visible is True
    assert np.allclose(handles["map_points"].points, 3.0)

    dropdown.value = _MAP_MODE_METADATA
    on_change(None)
    assert handles["map_gaussians"].visible is False
    assert handles["map_points"].visible is False
