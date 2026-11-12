"""Tests for the lazy core viewer registry."""

from __future__ import annotations

import subprocess
import sys

from eligsir.viewers import NullViewer, list_viewers, make_viewer
from eligsir.viewers.sinks.base import PreviewSink


def test_null_viewer_does_not_import_optional_viewer_dependencies():
    viewer = make_viewer("none")
    assert isinstance(viewer, NullViewer)


def test_null_viewer_does_not_import_optional_viewer_dependencies_subprocess():
    script = (
        "import sys\n"
        "from eligsir.viewers import NullViewer, make_viewer\n"
        "viewer = make_viewer('none')\n"
        "assert isinstance(viewer, NullViewer)\n"
        "assert {'rerun', 'rclpy'}.isdisjoint(sys.modules)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=".", capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_viewer_registry_lists_core_viewers_only():
    assert list_viewers() == ["none", "rerun", "viser"]


def test_removed_ros_viewer_fails_closed():
    try:
        make_viewer("ros2_rviz")
    except ValueError as exc:
        msg = str(exc)
    else:
        raise AssertionError("removed ROS viewer must not remain selectable")
    assert "ros2_rviz" in msg
    assert "none" in msg
    assert "rerun" in msg
    assert "viser" in msg


def test_non_ros_legacy_viz_wrapper_still_imports():
    from viz.rvizsplat_export import RVizSplatSnapshot

    assert RVizSplatSnapshot is not None


# --- issue #570/#574/#583: --viewer_sinks unblocking work -------------------
#
# The PreviewBus producer call sites in train_streaming.py used to be
# reachable ONLY via `--viewer viser` (see test_canonical_viewer.py's
# `test_the_legacy_registry_only_adds_the_opt_in_viser_preset` /
# `test_training_entrypoints_only_reach_canonical_viewer_via_the_opt_in_viser_preset`,
# both narrowly scoped to that fact and left unmodified by this change --
# `--viewer viser` with no `--viewer_sinks` still resolves exactly as before).
# `make_viewer`'s new `viewer_sinks` keyword is the second, independent path
# to the same `CanonicalViewerAdapter` construction, and these tests are the
# regression contract for it: (a) leaving it unset changes nothing, (b)
# setting it reaches the canonical adapter regardless of `--viewer`.


class _FakeSink(PreviewSink):
    """Minimal concrete PreviewSink -- no subprocess, no network, no torch."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def test_viewer_sinks_unset_is_behavior_neutral_for_none():
    """Default (`viewer_sinks` omitted) reproduces the pre-existing null path."""
    viewer = make_viewer("none")
    assert isinstance(viewer, NullViewer)
    # Passing an explicit-but-empty value must behave identically to omitting
    # it entirely -- both are "no override", never "empty sink list requested".
    viewer2 = make_viewer("none", viewer_sinks=None)
    assert isinstance(viewer2, NullViewer)
    viewer3 = make_viewer("none", viewer_sinks="")
    assert isinstance(viewer3, NullViewer)


def test_viewer_sinks_explicit_reaches_canonical_adapter_independent_of_viewer(monkeypatch):
    """`--viewer_sinks viser` must build the same adapter `--viewer viser`
    does, without requiring `--viewer` to also say "viser" -- this is the
    literal #574/#583 blocker: a run must be able to select the canonical
    sink without going through the viser preset name.
    """
    from eligsir.viewers import registry as registry_module

    calls = []

    def _fake_viser_factory(**kwargs):
        calls.append(dict(kwargs))
        return _FakeSink(**kwargs)

    monkeypatch.setitem(registry_module._SINK_FACTORIES, "viser", _fake_viser_factory)

    # A non-"none" --viewer (the "none" preset deliberately refuses explicit
    # sinks -- see test_viewer_none_still_refuses_explicit_sinks below) that
    # is NOT "viser" -- --viewer_sinks alone must still select the canonical
    # sink, independent of this preset name.
    viewer = make_viewer("rerun", viewer_sinks="viser", port=9999, host="127.0.0.1")
    assert hasattr(viewer, "canonical_viewer"), (
        "make_viewer(..., viewer_sinks=...) must build a CanonicalViewerAdapter"
    )
    assert len(calls) == 1
    assert calls[0]["port"] == 9999
    assert calls[0]["host"] == "127.0.0.1"
    real_sink = viewer.canonical_viewer.sinks[0]
    assert isinstance(real_sink, _FakeSink)
    viewer.close()


def test_viewer_sinks_overrides_a_different_viewer_preset_without_importing_it(monkeypatch):
    """`--viewer rerun --viewer_sinks viser` must build the canonical adapter
    (explicit sinks win over the preset, per resolve_preset_sinks) and must
    NOT import/construct the legacy RerunViewer along the way.
    """
    from eligsir.viewers import registry as registry_module

    monkeypatch.setitem(
        registry_module._SINK_FACTORIES, "viser", lambda **kwargs: _FakeSink(**kwargs)
    )

    viewer = make_viewer("rerun", viewer_sinks="viser")
    assert hasattr(viewer, "canonical_viewer")
    assert "rerun" not in sys.modules
    viewer.close()


def test_viewer_none_still_refuses_explicit_sinks():
    try:
        make_viewer("none", viewer_sinks="viser")
    except ValueError as exc:
        assert "none" in str(exc)
    else:
        raise AssertionError("--viewer none must refuse to be combined with --viewer_sinks")


def test_unknown_sink_name_gives_an_actionable_core_factory_error():
    """'rviz' is a recognized canonical sink NAME (eligsir.viewers.canonical
    knows about it), but has no core factory (issue #1073's hard cut: the
    RVizSplat/ROS implementation lives in eligsir-ros2, which core must
    never import). This must fail loudly and explain why, not silently no-op
    or raise an opaque KeyError.
    """
    try:
        make_viewer("rerun", viewer_sinks="rviz")
    except ValueError as exc:
        msg = str(exc)
    else:
        raise AssertionError("an unbuildable-in-core sink name must be refused")
    assert "rviz" in msg
    assert "eligsir-ros2" in msg


def test_unknown_sink_name_does_not_import_anything_ros_specific():
    """The refusal above must be pure name-lookup -- it must never attempt an
    import of rclpy or any ROS-specific module, even conditionally, per the
    hard core/ROS boundary (issue #1073).
    """
    script = (
        "import sys\n"
        "from eligsir.viewers import make_viewer\n"
        "try:\n"
        "    make_viewer('rerun', viewer_sinks='rviz')\n"
        "except ValueError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('expected ValueError')\n"
        "assert 'rclpy' not in sys.modules\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=".", capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout


def test_registry_still_never_imports_rclpy_source_level():
    """Static guard alongside the runtime one above: registry.py's own source
    must never mention `rclpy` -- the sink-name lookup that produces the
    actionable error above is pure string matching against
    `_SINK_FACTORIES`, never a conditional ROS import. (`eligsir_ros2`
    may legitimately appear in a comment, as documentation of where the
    'rviz' sink actually lives -- only `rclpy`, an unconditional tell of a
    real ROS import, is disallowed.)
    """
    from pathlib import Path

    import eligsir.viewers.registry as registry_module

    text = Path(registry_module.__file__).read_text()
    assert "rclpy" not in text
