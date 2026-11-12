"""Issue #570: ``CanonicalViewer`` -- cadence, non-blocking submission, and the
explicit assertion that this PR did NOT migrate the training path.

The last section is deliberately an *anti*-regression guard: it asserts the
existing backend-specific path is byte-unchanged, so a future reader cannot
mistake this PR for the migration.
"""
from __future__ import annotations

import numpy as np
import pytest

from eligsir.viewers.canonical import (
    VIEWER_PRESET_SINKS,
    CanonicalViewer,
    CanonicalViewerConfig,
    parse_viewer_sinks,
    resolve_preset_sinks,
)
from eligsir.viewers.events import (
    CapturedImage,
    ImageKind,
    MetricsPreview,
    PreviewChannel,
    RawInputPreview,
    TrainingPairPreview,
)
from eligsir.viewers.sinks.base import PreviewSink


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class CollectingSink(PreviewSink):
    def __init__(self) -> None:
        self.seen = []
        self.closed = 0

    def on_training_pair(self, envelope) -> None:
        self.seen.append(envelope)

    def close(self) -> None:
        self.closed += 1


class BlockingSink(PreviewSink):
    """A sink that would hang forever if the producer ever called it."""

    def on_training_pair(self, envelope) -> None:  # pragma: no cover - must never run
        raise AssertionError("the producer path called a sink")

    def close(self) -> None:
        pass


def _pair(i: int, camera_id: int = 3) -> TrainingPairPreview:
    return TrainingPairPreview(
        iteration=i, wall_time_s=float(i),
        target=CapturedImage(np.zeros((2, 2, 3), np.uint8), camera_id, ImageKind.TARGET),
        render=CapturedImage(np.zeros((2, 2, 3), np.uint8), camera_id, ImageKind.RENDER),
    )


def _viewer(clock=None, sinks=(), **cfg):
    return CanonicalViewer(
        CanonicalViewerConfig(**cfg), sinks=sinks, clock=clock or FakeClock()
    )


# --------------------------------------------------------------------------
# submission never reaches a sink
# --------------------------------------------------------------------------
def test_submit_never_calls_a_sink():
    v = _viewer(sinks=(BlockingSink(),), training_pair_max_fps=0.0)
    for i in range(1, 101):
        assert v.submit(_pair(i)) is not None
    assert v.stats()["preview_training_pair_submitted"] == 100


def test_a_stalled_sink_cannot_block_submission():
    """No thread required: the sink is simply never on the submission path."""
    clock = FakeClock()
    v = _viewer(clock=clock, sinks=(BlockingSink(),), training_pair_max_fps=0.0)
    for i in range(1, 1001):
        clock.advance(0.001)
        v.submit(_pair(i))
    assert v.stats()["pending"] <= v.stats()["max_pending"]


def test_the_viewer_exposes_no_renderer_or_model_handle():
    v = _viewer()
    surface = [n for n in dir(v) if not n.startswith("_")]
    forbidden = ("model", "render", "gaussian", "optimizer", "callback")
    assert not [n for n in surface if any(t in n.lower() for t in forbidden)], surface


# --------------------------------------------------------------------------
# cadence is a MAXIMUM RATE, not a deadline
# --------------------------------------------------------------------------
def test_cadence_limits_output_rate_and_counts_the_skips():
    clock = FakeClock()
    v = _viewer(clock=clock, training_pair_max_fps=10.0)  # min 0.1s spacing
    assert v.submit(_pair(1)) is not None
    clock.advance(0.05)
    assert v.submit(_pair(2)) is None          # too soon -> skipped, not queued
    clock.advance(0.06)
    assert v.submit(_pair(3)) is not None
    stats = v.stats()
    assert stats["preview_training_pair_rate_limited"] == 1
    assert stats["preview_training_pair_submitted"] == 2


def test_rate_limiting_is_per_channel():
    clock = FakeClock()
    v = _viewer(clock=clock, training_pair_max_fps=1.0, metrics_max_hz=0.0)
    v.submit(_pair(1))
    assert v.submit(_pair(2)) is None
    for i in range(5):
        assert (
            v.submit(MetricsPreview(iteration=i, wall_time_s=float(i), metrics={"a": 1.0}))
            is not None
        )


def test_zero_rate_means_unlimited():
    clock = FakeClock()
    v = _viewer(clock=clock, training_pair_max_fps=0.0)
    assert all(v.submit(_pair(i)) is not None for i in range(1, 21))


def test_overload_degrades_the_viewer_not_the_producer():
    """Design doc §6: FPS down, drops up, submission behaviour unchanged."""
    clock = FakeClock()
    v = _viewer(clock=clock, training_pair_max_fps=0.0, sinks=(CollectingSink(),))
    for i in range(1, 501):
        clock.advance(0.001)
        assert v.submit(_pair(i)) is not None   # every call returns, always
    stats = v.stats()
    assert stats["preview_training_pair_submitted"] == 500
    assert stats["preview_training_pair_dropped"] == 499  # nothing drained
    assert stats["pending"] == 1


def test_disabled_viewer_skips_and_counts():
    v = _viewer(enabled=False)
    assert v.submit(_pair(1)) is None
    assert v.stats()["disabled_skips"] == 1
    assert v.stats()["preview_training_pair_submitted"] == 0


def test_negative_rate_is_refused():
    with pytest.raises(ValueError, match="must be >= 0"):
        CanonicalViewerConfig(training_pair_max_fps=-1.0)


# --------------------------------------------------------------------------
# drain is consumer-side, isolated, and does not duplicate capture
# --------------------------------------------------------------------------
def test_drain_delivers_and_isolates():
    class Boom(PreviewSink):
        def on_training_pair(self, envelope) -> None:
            raise ValueError("boom")

        def close(self) -> None:
            pass

    good, bad = CollectingSink(), Boom()
    v = _viewer(sinks=(good, bad), training_pair_max_fps=0.0)
    v.submit(_pair(1))
    stats = v.drain_to_sinks()
    assert len(good.seen) == 1
    assert sum(s.errors for s in stats.values()) == 1
    assert v.stats()["sink_errors"] == 1


def test_two_sinks_receive_the_identical_envelope():
    a, b = CollectingSink(), CollectingSink()
    v = _viewer(sinks=(a, b), training_pair_max_fps=0.0)
    event = _pair(1)
    v.submit(event)
    v.drain_to_sinks()
    assert a.seen[0] is b.seen[0]
    assert a.seen[0].event is event


def test_close_is_idempotent_and_isolates_sink_failures():
    class BadClose(PreviewSink):
        def close(self) -> None:
            raise RuntimeError("nope")

    good = CollectingSink()
    v = _viewer(sinks=(good, BadClose()))
    v.close()
    v.close()
    assert good.closed == 1
    assert v.stats()["sink_errors"] == 1
    assert v.submit(_pair(1)) is None  # closed => disabled


def test_non_sink_is_refused_at_construction():
    with pytest.raises(TypeError, match="PreviewSink"):
        CanonicalViewer(sinks=(object(),))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# preset / multi-sink resolution (pure, unwired)
# --------------------------------------------------------------------------
def test_every_preset_resolves_to_sink_names():
    assert resolve_preset_sinks("none") == ()
    assert resolve_preset_sinks("viser") == ("viser",)
    assert resolve_preset_sinks("rerun") == ("rerun",)
    assert resolve_preset_sinks("ros2_rviz") == ("rviz",)
    assert set(VIEWER_PRESET_SINKS) == {"none", "viser", "rerun", "ros2_rviz"}


def test_multi_sink_spec_is_parsed_in_order():
    assert parse_viewer_sinks("viser,recorder") == ("viser", "recorder")
    assert resolve_preset_sinks("rerun", "viser,rviz") == ("viser", "rviz")


def test_unknown_and_duplicate_sinks_are_refused():
    with pytest.raises(ValueError, match="unknown viewer sink"):
        parse_viewer_sinks("viser,holodeck")
    with pytest.raises(ValueError, match="more than once"):
        parse_viewer_sinks("viser,viser")


def test_viewer_none_cannot_be_silently_re_enabled():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_preset_sinks("none", "viser")


def test_raw_input_and_training_pair_reach_different_channels():
    clock = FakeClock()
    v = _viewer(clock=clock, training_pair_max_fps=0.0, raw_input_max_fps=0.0)
    assert v.submit(_pair(1)).channel is PreviewChannel.TRAINING_PAIR
    raw = RawInputPreview(
        iteration=1, wall_time_s=1.0, source_frame_id=911,
        rgb=np.zeros((2, 2, 3), np.uint8),
    )
    assert v.submit(raw).channel is PreviewChannel.RAW_INPUT


# --------------------------------------------------------------------------
# ANTI-REGRESSION: this PR is the contract, NOT the migration
# --------------------------------------------------------------------------
def test_the_legacy_registry_only_adds_the_opt_in_viser_preset():
    """#570 AC1/AC2 narrowed (not discharged): only 'viser' routes through
    CanonicalViewer; 'none'/'rerun' remain the untouched legacy backends. See
    eligsir/viewers/canonical_adapter.py's module docstring and the #570
    status comment's "pickup order" step 3.

    'ros2_rviz' was removed from the core registry by the #1076 ROS2 hard cut
    (commit 9df507ea); ROS/RViz output now lives in the separate
    eligsir-ros2 package. tests/test_viewer_registry.py owns the
    assertion that the removed name stays absent and fails closed.
    """
    from eligsir.viewers.registry import VIEWERS, list_viewers

    assert list_viewers() == ["none", "rerun", "viser"]
    assert set(VIEWERS) == {"none", "rerun", "viser"}
    src = __import__(
        "inspect"
    ).getsource(__import__("eligsir.viewers.registry", fromlist=["x"]))
    assert "CanonicalViewer(" not in src, (
        "registry.py must not construct CanonicalViewer directly; only "
        "canonical_adapter.py (imported lazily by the 'viser' factory) may"
    )
    assert "canonical_adapter" in src, "the 'viser' preset must route through the adapter"


def test_training_entrypoints_only_reach_canonical_viewer_via_the_opt_in_viser_preset():
    """#570/#572 migration, scoped: train.py/train_streaming.py must not
    import or construct CanonicalViewer directly -- they still only ever call
    ``make_viewer(name, **kwargs)``. The 'viser' preset is opt-in and resolves
    to CanonicalViewer only inside registry.py/canonical_adapter.py, never at
    the training-entrypoint call site itself.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("train.py", "train_streaming.py"):
        text = (root / name).read_text()
        assert "CanonicalViewer" not in text, (
            f"{name} must keep constructing viewers only via make_viewer(); "
            f"routing through CanonicalViewer is registry.py/canonical_adapter.py's job"
        )
        assert "from eligsir.viewers import make_viewer" in text


def test_canonical_viewer_is_not_a_legacy_viewer_subclass():
    from eligsir.viewers.base import Viewer

    assert not issubclass(CanonicalViewer, Viewer)


# --- issue #570/#574/#583: un-gating the producer call sites ---------------
#
# Follow-up to `test_training_entrypoints_only_reach_canonical_viewer_via_the_
# opt_in_viser_preset` above (still true, still unmodified: train_streaming.py
# never imports/constructs CanonicalViewer directly). This section is the
# regression contract for the next step: the two PreviewBus *producer* call
# sites inside train_streaming.py's training loop used to gate on the literal
# string comparison `_viewer_name == "viser"`, which meant they were
# reachable ONLY through the `--viewer viser` preset -- not through
# `--viewer_sinks viser` on its own, and not through any future non-viser
# canonical sink. They must instead gate on whether the constructed viewer IS
# a canonical adapter (`hasattr(_viewer, "canonical_viewer")`), independent
# of which `--viewer`/`--viewer_sinks` spelling produced it.


def test_producer_call_sites_generalized_from_viser_name_to_canonical_flag():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    text = (root / "train_streaming.py").read_text()

    assert '_viewer_is_canonical = hasattr(_viewer, "canonical_viewer")' in text
    assert "elif _viewer_is_canonical and _map_output_lane is not None:" in text
    assert "_map_output_lane.try_submit(" in text
    assert "torch.randperm(_n_total" not in text
    assert "elif not _viewer_is_canonical:" in text
    assert 'if _viewer_is_canonical and hasattr(_viewer, "log_training_pair_tensors"):' in text
    # None of the three PREVIOUS literal-string gates may still be present --
    # if any of these substrings reappears, a producer call site regressed
    # back to being reachable only via the literal "viser" preset name.
    assert '_viewer_name == "viser" and hasattr(_viewer, "log_gaussian_map_tensors")' not in text
    assert '_viewer_name != "viser":' not in text
    assert '_viewer_name == "viser" and hasattr(_viewer, "log_training_pair_tensors")' not in text


def test_viewer_sinks_flag_is_declared_and_defaults_to_no_override():
    import argparse

    from arguments import StreamingParams

    parser = argparse.ArgumentParser()
    sp = StreamingParams(parser)
    # Declaring the flag must not itself change any other default.
    assert sp.viewer == "none"
    assert sp.viewer_sinks == ""

    ns = parser.parse_args([])
    assert ns.viewer_sinks == ""
    ns2 = parser.parse_args(["--viewer_sinks", "viser"])
    assert ns2.viewer_sinks == "viser"
