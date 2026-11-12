"""Regression coverage for issue #1402.

``CanonicalViewerConfig.map_max_gaussians`` (default 100_000) caps the live
map preview sample forwarded to the viser sidecar, and
``sample_output_from_model`` selects that many rows via a uniform-stride
index (not a real Gaussian-splatting LOD) -- suspected cause of a "large
blurry discs" artifact reported against a live screenshot after #1400's
device-mismatch fix. This adds a launch-time control
(``--viewer_map_max_gaussians``) for that cap without touching the sampling
mechanism itself, which #1402's own diagnostic experiment must confirm is
the real cause before any bigger LOD rework is attempted.

Deliberately does NOT require ``viser`` to be installed -- mirrors
``test_issue1389_viser_missing_package.py``'s registry-level approach with
the sidecar factory monkeypatched out.
"""
from __future__ import annotations

import pytest

from eligsir.viewers.canonical import (
    VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING,
    CanonicalViewerConfig,
    resolve_viewer_map_max_gaussians,
)
from eligsir.viewers.sinks.viser_sink import ViserUnavailable


def _raising_factory(**kwargs):
    # Mirrors test_issue1389_viser_missing_package.py: raising ViserUnavailable
    # keeps `sinks` empty (no real PreviewSink needed) while still exercising
    # the config-construction path this module cares about.
    raise ViserUnavailable("simulated: no real sidecar needed for this test")


def test_resolve_viewer_map_max_gaussians_zero_means_hard_ceiling():
    assert resolve_viewer_map_max_gaussians(0) == VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING


def test_resolve_viewer_map_max_gaussians_negative_also_means_hard_ceiling():
    # Defensive: argparse int() would already reject most malformed input,
    # but a negative value must not be reinterpreted as "unlimited" via
    # subtraction/underflow anywhere downstream -- treat it the same as 0.
    assert resolve_viewer_map_max_gaussians(-1) == VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING


def test_resolve_viewer_map_max_gaussians_passes_through_small_value():
    assert resolve_viewer_map_max_gaussians(250_000) == 250_000


def test_resolve_viewer_map_max_gaussians_clamps_above_hard_ceiling():
    huge = VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING * 10
    assert resolve_viewer_map_max_gaussians(huge) == VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING


def test_default_config_map_max_gaussians_unchanged():
    """The pre-#1402 default (100_000) must not move -- an unset flag must
    reproduce every run predating this option exactly."""
    assert CanonicalViewerConfig().map_max_gaussians == 100_000


def test_registry_make_viewer_default_cap_unchanged(monkeypatch):
    """No ``map_max_gaussians`` kwarg (the pre-#1402 call shape) must build a
    viewer whose config still carries CanonicalViewerConfig's own default."""
    from eligsir.viewers import registry

    monkeypatch.setitem(registry._SINK_FACTORIES, "viser", _raising_factory)

    viewer = registry.make_viewer("viser", port=0, host="127.0.0.1")

    assert viewer.canonical_viewer.config.map_max_gaussians == 100_000


def test_registry_make_viewer_honors_explicit_map_max_gaussians(monkeypatch):
    from eligsir.viewers import registry

    monkeypatch.setitem(registry._SINK_FACTORIES, "viser", _raising_factory)

    viewer = registry.make_viewer(
        "viser", port=0, host="127.0.0.1", map_max_gaussians=500_000
    )

    assert viewer.canonical_viewer.config.map_max_gaussians == 500_000


def test_registry_make_viewer_zero_map_max_gaussians_resolves_to_hard_ceiling(monkeypatch):
    from eligsir.viewers import registry

    monkeypatch.setitem(registry._SINK_FACTORIES, "viser", _raising_factory)

    viewer = registry.make_viewer("viser", port=0, host="127.0.0.1", map_max_gaussians=0)

    assert (
        viewer.canonical_viewer.config.map_max_gaussians
        == VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING
    )


def test_registry_make_viewer_via_viewer_sinks_also_honors_cap(monkeypatch):
    """``--viewer_sinks viser`` must reach the identical config path as
    ``--viewer viser`` (mirrors the existing port/host contract). Uses a
    non-"none" preset ("rerun") with an explicit sink override to prove this
    is independent of the ``--viewer`` selection itself, per
    ``resolve_preset_sinks``'s "explicit wins over preset, except none"
    contract."""
    from eligsir.viewers import registry

    monkeypatch.setitem(registry._SINK_FACTORIES, "viser", _raising_factory)

    viewer = registry.make_viewer(
        "rerun", viewer_sinks="viser", port=0, host="127.0.0.1", map_max_gaussians=250_000
    )

    assert viewer.canonical_viewer.config.map_max_gaussians == 250_000


def test_arguments_default_matches_config_default():
    """``arguments/__init__.py``'s ``viewer_map_max_gaussians`` default must
    equal ``CanonicalViewerConfig``'s own default, so leaving the flag unset
    is truly a no-op (per this module's docstring claim)."""
    import argparse

    from arguments import StreamingParams

    parser = argparse.ArgumentParser()
    group = StreamingParams(parser)
    args = group.extract(parser.parse_args([]))

    assert args.viewer_map_max_gaussians == CanonicalViewerConfig().map_max_gaussians == 100_000


def test_arguments_accepts_explicit_override():
    import argparse

    from arguments import StreamingParams

    parser = argparse.ArgumentParser()
    group = StreamingParams(parser)
    args = group.extract(parser.parse_args(["--viewer_map_max_gaussians", "0"]))

    assert args.viewer_map_max_gaussians == 0
