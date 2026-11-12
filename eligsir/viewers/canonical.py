"""``CanonicalViewer``: the single training-facing viewer path (issue #570, phase A).

Canonical design: ``docs/architecture/2026-08-14-canonical-live-viewer-sidecar.md``
§3.1. Parent program: #569.

SCOPE OF THIS MODULE -- READ FIRST
----------------------------------
**This is the contract, not the migration.** ``train.py`` and
``train_streaming.py`` still construct backend viewers through
``eligsir/viewers/registry.py`` exactly as they did before this file
existed; ``registry.py``, ``rerun_viewer.py``, ``ros2_rviz.py`` and ``base.py``
are untouched. No default changes. #570's acceptance criterion 1
("``CanonicalViewer`` is the sole production training-facing viewer class/path")
is therefore **not** discharged by this PR, deliberately:

* migrating the live viewer path changes training behaviour, and
* CLAUDE.md's zero-backpressure observability invariant plus #576's promotion
  gate cannot be validated without a GPU and a live camera.

A partially-delivered contract that says so is honest; a wired-but-unvalidated
one is not. See ``docs/architecture/2026-08-15-issue570-canonical-viewer-core.md``
for the full accounting and the migration/deprecation path for the existing
backend-specific hot-path calls (acceptance criterion 6).

What this class does
--------------------
1. accepts semantic, already-captured preview events from training;
2. enforces a wall-clock *maximum output rate* per channel -- never a deadline
   (design doc §6: if the system is overloaded, viewer FPS drops, the drop
   counter rises, training is unchanged);
3. publishes into the bounded latest-only :class:`~eligsir.viewers.bus.PreviewBus`;
4. counts submitted / rate-limited / dropped;
5. returns immediately, always.

What it deliberately does NOT do
--------------------------------
* render anything -- it has no renderer handle, no model handle, and no
  callable-typed parameter anywhere in its API, so the primary training-pair
  path *structurally* cannot invoke a rasterization. (That is the strongest
  form of "zero extra rasterizations" available without a real training run;
  the measured form belongs to #576.)
* call a sink on the submission path. :meth:`submit` writes one bus slot and
  returns. Sinks are drained by :meth:`drain_to_sinks`, documented and tested
  as consumer-side only.
* encode JPEG/video, do network I/O, write files, or run any backend SDK.

It is intentionally **not** a subclass of the legacy
``eligsir.viewers.base.Viewer``. Inheriting that ABC would re-import the
hook surface (``log_gaussian_model(view, ...)``, ``set_params_callback(cb)``)
whose shape is the problem -- a viewer holding a map view and a callback into
training. The compatibility bridge, if one is wanted at migration time, belongs
in the migration PR where its behavioural cost can be measured.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from eligsir.viewers.bus import PreviewBus, PreviewEnvelope
from eligsir.viewers.events import PreviewChannel, PreviewEvent, channel_for_event
from eligsir.viewers.sinks.base import DeliveryStats, PreviewSink, deliver_pending

__all__ = [
    "CanonicalViewerConfig",
    "CanonicalViewer",
    "VIEWER_PRESET_SINKS",
    "parse_viewer_sinks",
    "resolve_preset_sinks",
    "VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING",
    "resolve_viewer_map_max_gaussians",
]

#: Hard safety ceiling for ``CanonicalViewerConfig.map_max_gaussians`` when a
#: run asks for "no cap" (issue #1402). The live map preview is a dev/demo
#: tool that hands its sample to a `DeviceTransferOwner`'s fixed pinned-memory
#: buffer (`VersionedMapOutputLane`, sized as
#: ``max_gaussians * _FLOAT_FIELDS_PER_GAUSSIAN * 4`` bytes) and then to a
#: browser-side WebGL renderer -- truly unbounded is not a safe option
#: regardless of how large the live map gets. Chosen well above any single
#: fr3/kitchen1/floor2-3 champion run's live Gaussian count observed in this
#: repo's own docs (low hundreds of thousands) while still bounding transfer
#: buffer size and browser payload to a sane multiple of that.
VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING = 2_000_000


def resolve_viewer_map_max_gaussians(
    value: int, *, hard_ceiling: int = VIEWER_MAP_MAX_GAUSSIANS_HARD_CEILING
) -> int:
    """Resolve a ``--viewer_map_max_gaussians`` CLI value to a valid cap.

    ``0`` (or any non-positive value) means "no cap" and resolves to
    ``hard_ceiling`` rather than to a truly unbounded sample -- see the
    ceiling constant's own docstring for why unbounded is never offered. Any
    positive value is clamped to the same ceiling so an accidental typo
    (e.g. an extra zero) cannot silently request an unsafe transfer-buffer
    size.
    """
    resolved = int(value)
    if resolved <= 0:
        return int(hard_ceiling)
    return min(resolved, int(hard_ceiling))

#: Compatibility presets (design doc §10 / issue #570 "Configuration").
#: Pure data: this maps a ``--viewer`` value to a sink-name tuple. **No argparse
#: flag is added by this PR** -- wiring ``--viewer_sinks`` into
#: ``arguments/__init__.py`` and routing ``registry.make_viewer`` through here is
#: migration work, explicitly out of scope. Included as a pure function so the
#: migration PR has a tested resolver to call rather than reinventing one.
VIEWER_PRESET_SINKS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "viser": ("viser",),
    "rerun": ("rerun",),
    "ros2_rviz": ("rviz",),
}

_KNOWN_SINK_NAMES = ("viser", "rerun", "rviz", "recorder")


def parse_viewer_sinks(spec: str) -> Tuple[str, ...]:
    """Parse a ``--viewer_sinks`` style spec (``"viser,recorder"``).

    Order-preserving, duplicate-refusing, unknown-name-refusing. Pure; touches
    no argparse and constructs nothing.
    """
    if spec is None:
        return ()
    names = [part.strip() for part in str(spec).split(",")]
    names = [n for n in names if n]
    out: list = []
    for name in names:
        if name not in _KNOWN_SINK_NAMES:
            raise ValueError(
                f"unknown viewer sink {name!r}; known: {', '.join(_KNOWN_SINK_NAMES)}"
            )
        if name in out:
            raise ValueError(f"viewer sink {name!r} listed more than once")
        out.append(name)
    return tuple(out)


def resolve_preset_sinks(preset: str, sinks_spec: Optional[str] = None) -> Tuple[str, ...]:
    """Resolve ``--viewer <preset>`` (+ optional ``--viewer_sinks``) to sink names.

    An explicit ``sinks_spec`` wins over the preset, except that ``none``
    disables the viewer outright and refuses to be combined with sinks -- an
    explicitly-disabled viewer must not be silently re-enabled by another flag.
    """
    key = str(preset or "none")
    if key not in VIEWER_PRESET_SINKS:
        raise ValueError(
            f"unknown viewer preset {key!r}; known: {', '.join(sorted(VIEWER_PRESET_SINKS))}"
        )
    explicit = parse_viewer_sinks(sinks_spec) if sinks_spec else ()
    if key == "none":
        if explicit:
            raise ValueError("--viewer none cannot be combined with explicit sinks")
        return ()
    return explicit or VIEWER_PRESET_SINKS[key]


@dataclass(frozen=True)
class CanonicalViewerConfig:
    """Capture cadence and budgets. These belong to the viewer, not to a sink.

    Rates are **maximum output rates**, never training deadlines. ``0.0`` means
    "no rate limit for this channel"; a negative rate is refused rather than
    reinterpreted.
    """

    enabled: bool = True
    training_pair_max_fps: float = 8.0
    raw_input_max_fps: float = 15.0
    map_max_hz: float = 0.5
    trajectory_max_hz: float = 2.0
    keyframe_max_hz: float = 0.0
    metrics_max_hz: float = 2.0
    map_max_gaussians: int = 100_000

    def __post_init__(self) -> None:
        for name in (
            "training_pair_max_fps",
            "raw_input_max_fps",
            "map_max_hz",
            "trajectory_max_hz",
            "keyframe_max_hz",
            "metrics_max_hz",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0 (0 = unlimited), got {value}")
            object.__setattr__(self, name, value)
        if int(self.map_max_gaussians) <= 0:
            raise ValueError("map_max_gaussians must be > 0")
        object.__setattr__(self, "map_max_gaussians", int(self.map_max_gaussians))

    def max_rate_for(self, channel: PreviewChannel) -> float:
        return {
            PreviewChannel.TRAINING_PAIR: self.training_pair_max_fps,
            PreviewChannel.RAW_INPUT: self.raw_input_max_fps,
            PreviewChannel.MAP_SNAPSHOT: self.map_max_hz,
            PreviewChannel.TRAJECTORY: self.trajectory_max_hz,
            PreviewChannel.KEYFRAME: self.keyframe_max_hz,
            PreviewChannel.METRICS: self.metrics_max_hz,
        }[channel]


class CanonicalViewer:
    """One class, one path. Training constructs this and nothing else.

    ``clock`` is injectable purely so cadence behaviour is testable without
    sleeping; it defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        config: Optional[CanonicalViewerConfig] = None,
        *,
        sinks: Sequence[PreviewSink] = (),
        bus: Optional[PreviewBus] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config if config is not None else CanonicalViewerConfig()
        self._clock = clock
        self._bus = bus if bus is not None else PreviewBus(name="canonical")
        self._sinks: Tuple[PreviewSink, ...] = tuple(sinks)
        for sink in self._sinks:
            if not isinstance(sink, PreviewSink):
                raise TypeError(
                    f"sinks must implement PreviewSink, got {type(sink).__name__}"
                )
        # One subscriber handle per sink: each sink reads the SAME slot through
        # its own cursor, so a second sink costs a dict lookup, not a second
        # capture/conversion (design doc's "multi-viewer duplicates work" defect).
        self._handles = {
            id(sink): self._bus.subscribe(f"sink[{i}]:{type(sink).__name__}")
            for i, sink in enumerate(self._sinks)
        }
        self._last_emit: Dict[PreviewChannel, float] = {}
        self._rate_limited: Dict[PreviewChannel, int] = {ch: 0 for ch in PreviewChannel}
        self._disabled_skips = 0
        self._sink_errors = 0
        self._closed = False

    # -- producer API (training thread) ----------------------------------
    @property
    def bus(self) -> PreviewBus:
        """The bounded latest-only bus. Consumers poll it; producers do not read it."""
        return self._bus

    @property
    def sinks(self) -> Tuple[PreviewSink, ...]:
        return self._sinks

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled) and not self._closed

    def submit(self, event: PreviewEvent) -> Optional[PreviewEnvelope]:
        """Publish one preview event. Returns immediately; never calls a sink.

        Returns the published envelope, or ``None`` when the event was skipped
        by the cadence limiter or the viewer is disabled/closed. A ``None``
        return is a normal, counted outcome -- not an error, and never something
        the caller should retry.
        """
        if not self.enabled:
            self._disabled_skips += 1
            return None
        channel = channel_for_event(event)
        now = float(self._clock())
        max_rate = self.config.max_rate_for(channel)
        if max_rate > 0.0:
            last = self._last_emit.get(channel)
            if last is not None and (now - last) < (1.0 / max_rate):
                self._rate_limited[channel] += 1
                return None
        self._last_emit[channel] = now
        return self._bus.publish(event, channel=channel, wall_time_s=now)

    # -- consumer API (NOT the training thread) ---------------------------
    def drain_to_sinks(
        self, *, channels: Optional[Sequence[PreviewChannel]] = None
    ) -> Mapping[str, DeliveryStats]:
        """Deliver pending events to every configured sink, isolating failures.

        **Consumer side.** Never called by :meth:`submit`. Running it on the
        training thread would reintroduce exactly the coupling this architecture
        removes; #571 owns the real off-thread/off-process consumer. It exists
        here so the isolation semantics are testable and so a sidecar has one
        canonical drain to call.
        """
        out: Dict[str, DeliveryStats] = {}
        for i, sink in enumerate(self._sinks):
            stats = deliver_pending(
                self._bus, self._handles[id(sink)], sink, channels=channels
            )
            self._sink_errors += stats.errors
            out[f"{i}:{type(sink).__name__}"] = stats
        return out

    def close(self) -> None:
        """Close every sink, isolating failures. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:  # noqa: BLE001 - a sink cannot fail a training run
                self._sink_errors += 1

    # -- telemetry (design doc §12) ---------------------------------------
    def stats(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "viewer_enabled": bool(self.config.enabled),
            "viewer_sinks": [type(s).__name__ for s in self._sinks],
            "sink_errors": self._sink_errors,
            "disabled_skips": self._disabled_skips,
            "pending": self._bus.pending_count(),
            "max_pending": self._bus.max_pending_count,
        }
        out.update(self._bus.stats())
        for channel in PreviewChannel:
            out[f"preview_{channel.value}_rate_limited"] = self._rate_limited[channel]
        return out
