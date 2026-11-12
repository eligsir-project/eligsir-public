"""``PreviewSink``: the lossy-observer contract every viewer backend implements.

Issue #570, phase A. Canonical design:
``docs/architecture/2026-08-14-canonical-live-viewer-sidecar.md`` §3.3.

    Visualization is a lossy observer of mapping.
    Mapping is never a worker for visualization.

**No concrete sink is implemented here.** Rerun (#573), RVizSplat/ROS (#574),
Viser (#572) and the recorder (#575) each own their own migration. This module
is the contract and the delivery helper only.

Why the "no live model, no renderer work" rule is enforced in TYPES
------------------------------------------------------------------
The design doc (§1.3) states that a viewer must never receive a
``GaussianModel``, an optimizer, a mutation callback, or a render callback that
lets it request work from the training GPU. Stating that in prose is not
enforcement — the old ``Viewer`` ABC's ``log_gaussian_model(view, ...)`` hook is
exactly the shape that erodes under pressure.

Two mechanisms make it structural here:

1. **Every hook takes exactly one parameter, a
   :class:`~eligsir.viewers.bus.PreviewEnvelope`, and returns ``None``.**
   There is no parameter through which a model, a renderer, a callback or a
   request channel could be passed, and no return value through which a sink
   could ask for anything back. A sink is write-only from the trainer's point of
   view.

2. **:meth:`PreviewSink.__init_subclass__` refuses a subclass that widens the
   hook signatures.** A backend that "just needs" a second argument — a render
   callback, a model handle, a deadline — fails at class-definition time, in the
   subclass's own module, not later in review. ``tests/test_canonical_sink_contract.py``
   asserts a widened hook is rejected; without this check that test would be
   unwritable and the rule would be a comment.

The payloads themselves cannot carry a live reference either: every array in
``eligsir.viewers.events`` is a frozen ``numpy.ndarray`` validated at
construction, and a non-ndarray payload is refused there.

Delivery never runs on the training thread
------------------------------------------
:func:`deliver_pending` is the *consumer-side* drain. It polls a bus subscriber
and dispatches to one sink, isolating that sink's exceptions. It is deliberately
not called by ``PreviewBus.publish`` and not called by
``CanonicalViewer.submit``: the producer path holds no sink reference at all, so
a stalled or crashing sink cannot reach it. The real off-thread/off-process
consumer belongs to #571.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from eligsir.viewers.events import PreviewChannel

if TYPE_CHECKING:  # pragma: no cover
    from eligsir.viewers.bus import PreviewBus, PreviewEnvelope, SubscriberHandle

__all__ = [
    "SinkContractError",
    "PreviewSink",
    "NullPreviewSink",
    "DeliveryStats",
    "deliver_pending",
    "SINK_HOOKS",
    "FORBIDDEN_SINK_PARAM_TOKENS",
]

#: Hook name per channel. A sink implements only what it can display.
SINK_HOOKS: Dict[PreviewChannel, str] = {
    PreviewChannel.TRAINING_PAIR: "on_training_pair",
    PreviewChannel.RAW_INPUT: "on_raw_input",
    PreviewChannel.TRAJECTORY: "on_trajectory",
    PreviewChannel.KEYFRAME: "on_keyframe",
    PreviewChannel.METRICS: "on_metrics",
    PreviewChannel.RUNTIME_TELEMETRY: "on_runtime_telemetry",
    PreviewChannel.MAP_SNAPSHOT: "on_map_snapshot",
}

#: Substrings that, appearing in an overridden hook's parameter names, indicate
#: a backend is trying to reopen the model/renderer channel this contract closes.
#: Belt-and-braces on top of the arity check -- the arity check alone already
#: refuses them, this makes the *error message* name the actual rule.
FORBIDDEN_SINK_PARAM_TOKENS = (
    "model",
    "gaussian",
    "render_fn",
    "renderer",
    "optimizer",
    "callback",
    "scene",
    "map_view",
)


class SinkContractError(TypeError):
    """A sink subclass violates the observation-only contract."""


class PreviewSink:
    """Lossy observer of the canonical preview stream.

    Every hook defaults to a no-op, so a sink implements only the channels it
    can render. Hooks receive a :class:`~eligsir.viewers.bus.PreviewEnvelope`
    and nothing else.

    Sinks should normally run in a separate process (design doc §3.3); a
    thread-only sink is acceptable only for trivial work whose GIL/CPU behaviour
    is proven harmless, which for this project is a measured claim, not an
    assumption (see CLAUDE.md's GIL finding).
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _validate_sink_subclass(cls)

    def on_training_pair(self, envelope: "PreviewEnvelope") -> None:
        """The exact optimizer target/render pair for one supervision op."""

    def on_raw_input(self, envelope: "PreviewEnvelope") -> None:
        """The newest source observation. NOT matched to any render."""

    def on_trajectory(self, envelope: "PreviewEnvelope") -> None:
        """Estimated/GT camera trajectory."""

    def on_keyframe(self, envelope: "PreviewEnvelope") -> None:
        """A newly admitted keyframe."""

    def on_metrics(self, envelope: "PreviewEnvelope") -> None:
        """Compact scalar status."""

    def on_runtime_telemetry(self, envelope: "PreviewEnvelope") -> None:
        """Canonical decision, freshness, GPU/transfer, and residency panels."""

    def on_map_snapshot(self, envelope: "PreviewEnvelope") -> None:
        """A bounded, versioned, immutable map preview (schema-only in #570)."""

    def close(self) -> None:
        """Release sink resources. Must not raise; callers isolate it anyway."""


def _validate_sink_subclass(cls: type) -> None:
    for hook in SINK_HOOKS.values():
        func = cls.__dict__.get(hook)
        if func is None:
            continue
        if isinstance(func, (staticmethod, classmethod)):
            raise SinkContractError(
                f"{cls.__name__}.{hook} must be an instance method"
            )
        try:
            sig = inspect.signature(func)
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            continue
        params = list(sig.parameters.values())
        offending = [
            p.name
            for p in params[1:]
            if any(tok in p.name.lower() for tok in FORBIDDEN_SINK_PARAM_TOKENS)
        ]
        if offending:
            raise SinkContractError(
                f"{cls.__name__}.{hook} declares parameter(s) {offending}: a sink "
                f"may never receive the live model, the renderer, the optimizer or "
                f"a callback into training (design doc §1.3). It observes immutable "
                f"preview events only."
            )
        if len(params) != 2 or params[1].kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise SinkContractError(
                f"{cls.__name__}.{hook} must take exactly (self, envelope); got "
                f"{tuple(p.name for p in params)}. Widening a hook is how a sink "
                f"reacquires the ability to request renderer/model work, which this "
                f"contract exists to make impossible."
            )
        ret = sig.return_annotation
        if ret not in (None, "None", inspect.Signature.empty):
            raise SinkContractError(
                f"{cls.__name__}.{hook} must return None; a sink cannot hand a "
                f"value back to the producer (got return annotation {ret!r})"
            )


class NullPreviewSink(PreviewSink):
    """Accept-and-discard sink. The default when a viewer is configured with none."""

    def close(self) -> None:
        """No-op."""


@dataclass
class DeliveryStats:
    """Outcome of one drain pass. Errors are counted, never propagated."""

    delivered: int = 0
    errors: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    last_error: Optional[str] = None


def deliver_pending(
    bus: "PreviewBus",
    handle: "SubscriberHandle",
    sink: PreviewSink,
    *,
    channels: Optional[Sequence[PreviewChannel]] = None,
    stats: Optional[DeliveryStats] = None,
) -> DeliveryStats:
    """Drain ``bus`` into one ``sink``, isolating that sink's exceptions.

    CONSUMER SIDE ONLY. This is never called from ``PreviewBus.publish`` or from
    ``CanonicalViewer.submit`` -- the producer path holds no sink reference, so a
    sink that blocks here cannot block a submission. #571 owns running this off
    the training thread/process.

    A sink raising is a sink bug, not a training failure: it is counted by type
    and the drain continues to the next channel and the next sink.
    """
    stats = stats if stats is not None else DeliveryStats()
    if not isinstance(sink, PreviewSink):
        raise SinkContractError(
            f"deliver_pending expects a PreviewSink, got {type(sink).__name__}"
        )
    wanted = tuple(channels) if channels is not None else tuple(PreviewChannel)
    for channel in wanted:
        envelope = bus.poll(handle, channel)
        if envelope is None:
            continue
        hook = getattr(sink, SINK_HOOKS[channel])
        try:
            hook(envelope)
        except Exception as exc:  # noqa: BLE001 - isolation is the whole point
            stats.errors += 1
            name = type(exc).__name__
            stats.error_types[name] = stats.error_types.get(name, 0) + 1
            stats.last_error = f"{name}: {exc}"
        else:
            stats.delivered += 1
    return stats
