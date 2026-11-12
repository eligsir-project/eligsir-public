"""Issue #570: the ``PreviewSink`` observation-only contract, and delivery isolation.

These assert the *structural* form of "no sink API exposes live
model/renderer/optimizer callbacks" and "the primary training-pair path
introduces no extra renderer call". The measured forms of the latter need a real
training run and belong to #576.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from eligsir.viewers import canonical as canonical_mod
from eligsir.viewers import events as events_mod
from eligsir.viewers.bus import PreviewBus
from eligsir.viewers.events import (
    CapturedImage,
    ImageKind,
    MetricsPreview,
    PreviewChannel,
    RawInputPreview,
    TrainingPairPreview,
)
from eligsir.viewers.sinks import base as sinks_base
from eligsir.viewers.sinks.base import (
    SINK_HOOKS,
    DeliveryStats,
    NullPreviewSink,
    PreviewSink,
    SinkContractError,
    deliver_pending,
)


def _pair(i: int = 1) -> TrainingPairPreview:
    return TrainingPairPreview(
        iteration=i, wall_time_s=float(i),
        target=CapturedImage(np.zeros((2, 2, 3), np.uint8), 3, ImageKind.TARGET),
        render=CapturedImage(np.zeros((2, 2, 3), np.uint8), 3, ImageKind.RENDER),
    )


class CollectingSink(PreviewSink):
    def __init__(self) -> None:
        self.seen = []

    def on_training_pair(self, envelope) -> None:
        self.seen.append(("pair", envelope))

    def on_raw_input(self, envelope) -> None:
        self.seen.append(("raw", envelope))

    def close(self) -> None:
        self.seen.append(("close", None))


# --------------------------------------------------------------------------
# the contract is enforced at class-definition time
# --------------------------------------------------------------------------
def test_a_hook_that_asks_for_the_model_is_refused_at_class_definition():
    with pytest.raises(SinkContractError, match="live model"):

        class Greedy(PreviewSink):  # noqa: D401
            def on_training_pair(self, envelope, gaussian_model=None) -> None:
                pass


def test_a_hook_that_asks_for_a_render_callback_is_refused():
    with pytest.raises(SinkContractError, match="live model"):

        class Renderer(PreviewSink):
            def on_map_snapshot(self, envelope, render_fn=None) -> None:
                pass


def test_a_widened_hook_is_refused_even_with_an_innocent_name():
    with pytest.raises(SinkContractError, match="exactly \\(self, envelope\\)"):

        class Widened(PreviewSink):
            def on_metrics(self, envelope, deadline_s=0.0) -> None:
                pass


def test_a_var_args_hook_is_refused():
    with pytest.raises(SinkContractError, match="exactly"):

        class Sloppy(PreviewSink):
            def on_raw_input(self, *args) -> None:
                pass


def test_a_hook_returning_a_value_is_refused():
    with pytest.raises(SinkContractError, match="must return None"):

        class Chatty(PreviewSink):
            def on_metrics(self, envelope) -> bool:
                return True


def test_a_conforming_sink_is_accepted():
    sink = CollectingSink()
    assert isinstance(sink, PreviewSink)


def test_every_base_hook_takes_exactly_one_payload_and_returns_none():
    for hook in SINK_HOOKS.values():
        sig = inspect.signature(getattr(PreviewSink, hook))
        assert [p.name for p in sig.parameters.values()] == ["self", "envelope"], hook
        assert sig.return_annotation in (None, "None"), hook


def test_no_sink_api_name_mentions_the_model_renderer_or_optimizer():
    surface = [n for n in dir(PreviewSink) if not n.startswith("__")]
    forbidden = ("model", "render_fn", "renderer", "optimizer", "callback", "gaussian")
    assert not [n for n in surface if any(t in n.lower() for t in forbidden)]


# --------------------------------------------------------------------------
# delivery isolates sink failures
# --------------------------------------------------------------------------
class ExplodingSink(PreviewSink):
    def on_training_pair(self, envelope) -> None:
        raise RuntimeError("sidecar went away")

    def close(self) -> None:
        raise RuntimeError("and stayed away")


def test_a_sink_exception_is_counted_not_raised():
    bus = PreviewBus()
    sub = bus.subscribe("x")
    bus.publish(_pair())
    stats = deliver_pending(bus, sub, ExplodingSink())
    assert isinstance(stats, DeliveryStats)
    assert stats.errors == 1 and stats.delivered == 0
    assert stats.error_types == {"RuntimeError": 1}
    assert "sidecar went away" in stats.last_error


def test_one_exploding_sink_does_not_starve_another():
    bus = PreviewBus()
    bad, good = bus.subscribe("bad"), bus.subscribe("good")
    bus.publish(_pair())
    collector = CollectingSink()
    deliver_pending(bus, bad, ExplodingSink())
    deliver_pending(bus, good, collector)
    assert [k for k, _ in collector.seen] == ["pair"]


def test_delivery_routes_each_channel_to_its_own_hook():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    bus.publish(_pair())
    bus.publish(
        RawInputPreview(
            iteration=1, wall_time_s=1.0, source_frame_id=9,
            rgb=np.zeros((2, 2, 3), np.uint8),
        )
    )
    sink = CollectingSink()
    stats = deliver_pending(bus, sub, sink)
    assert stats.delivered == 2 and stats.errors == 0
    assert [k for k, _ in sink.seen] == ["pair", "raw"]


def test_null_sink_accepts_and_discards_everything():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    bus.publish(_pair())
    bus.publish(MetricsPreview(iteration=1, wall_time_s=1.0, metrics={"a": 1.0}))
    stats = deliver_pending(bus, sub, NullPreviewSink())
    assert (stats.delivered, stats.errors) == (2, 0)


def test_deliver_pending_refuses_a_non_sink():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    with pytest.raises(SinkContractError, match="expects a PreviewSink"):
        deliver_pending(bus, sub, object())  # type: ignore[arg-type]


def test_delivery_can_be_scoped_to_selected_channels():
    bus = PreviewBus()
    sub = bus.subscribe("s")
    bus.publish(_pair())
    bus.publish(MetricsPreview(iteration=1, wall_time_s=1.0, metrics={"a": 1.0}))
    sink = CollectingSink()
    stats = deliver_pending(bus, sub, sink, channels=[PreviewChannel.TRAINING_PAIR])
    assert stats.delivered == 1 and [k for k, _ in sink.seen] == ["pair"]


# --------------------------------------------------------------------------
# structural "no extra renderer call" on the primary path
# --------------------------------------------------------------------------
def test_no_canonical_module_imports_a_renderer_or_a_model():
    """The producer path cannot invoke a rasterization it cannot reach."""
    import eligsir.viewers.bus as bus_mod

    forbidden = (
        "gaussian_renderer",
        "gsplat",
        "scene.gsplat_model",
        "scene.gaussian_model",
        "torch",
    )
    for mod in (events_mod, bus_mod, canonical_mod, sinks_base):
        src = inspect.getsource(mod)
        for token in forbidden:
            assert f"import {token}" not in src, f"{mod.__name__} imports {token}"


def test_no_event_field_is_callable_typed():
    """A callable field is how a render/mutation hook re-enters the schema."""
    for cls in (
        TrainingPairPreview,
        RawInputPreview,
        MetricsPreview,
        events_mod.TrajectoryPreview,
        events_mod.KeyframePreview,
        events_mod.MapPreviewSnapshot,
    ):
        for name, f in cls.__dataclass_fields__.items():
            assert "Callable" not in str(f.type), f"{cls.__name__}.{name}"
            assert "callback" not in name.lower(), f"{cls.__name__}.{name}"
