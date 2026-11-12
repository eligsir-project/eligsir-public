"""Live-ingestion pipeline benchmark scenarios.

Orchestration layer over ``eligsir.io.sources.synthetic`` (the
deterministic synthetic RGB-D source + fake SLAM sidecar) that builds and
runs the plan's required benchmark arms (section 6.5, deliverable #4):
no source activity, 30 FPS all-rejected, 30 FPS ~2 accepted/s, 500ms SLAM
delay, SLAM failure, and burst input. Consumed by
``scripts/benchmark_live_pipeline.py``.

Depends on ``eligsir.runtime.profiling.pipeline_trace.PipelineTrace``,
``eligsir.runtime.profiling.queue_metrics.QueueMetrics``, and
``eligsir.runtime.profiling.events.gpu_timed_span`` -- imported lazily
inside ``run_scenario`` so this module and its scenario-construction logic
stay importable/testable in isolation even if those are ever unavailable.
"""

from __future__ import annotations

import queue as stdlib_queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from eligsir.io.sources.synthetic import (
    FakeSlamSidecar,
    RejectionPolicy,
    SlamFailure,
    SyntheticRGBDFrameSource,
    SyntheticSourceConfig,
)

SCENARIO_NAMES = (
    "no_source_activity",
    "all_rejected",
    "two_fps_accepted",
    "slam_delay_500ms",
    "slam_failure",
    "burst",
)


@dataclass
class ScenarioSpec:
    """A fully-constructed benchmark arm: a synthetic source, a rejection
    policy, and a fake SLAM sidecar, ready to be pumped through the
    ingestion pipeline by :func:`run_scenario`."""

    name: str
    source: Optional[SyntheticRGBDFrameSource]
    policy: RejectionPolicy
    slam: FakeSlamSidecar
    description: str = ""
    #: True iff this scenario's whole point is to *deliberately* trigger a
    #: fault (e.g. ``slam_failure``'s injected SLAM crash) -- for such a
    #: scenario, the fault firing is the CORRECT/expected outcome, not a
    #: run-invalidating failure. See ``ScenarioResult.expected_fault`` and
    #: this module's "one manifest wraps six intentionally different
    #: scenarios" validity note (review Fix 7).
    expected_fault: bool = False
    queue_capacity: int = 8


@dataclass(frozen=True)
class BenchmarkWorkload:
    """Production service callbacks used by the benchmark harness.

    The synthetic source is deliberately only a source fixture.  A result is
    production-quotable only when all three callbacks are provided by the
    composition root: the real admission service, map-commit transaction and
    fixed-topology optimization step.  This prevents a CPU policy loop from
    being presented as a live-pipeline performance measurement.
    """

    admit: Callable[[object], bool]
    commit: Callable[[object], None]
    optimize_step: Callable[[], None]


def build_scenario(name: str, *, num_frames: int = 90, realtime: bool = True, seed: int = 0) -> ScenarioSpec:
    """Build one of the required benchmark arms by name.

    Args:
        name: One of :data:`SCENARIO_NAMES`.
        num_frames: Frame budget for finite scenarios (ignored for
            ``no_source_activity``).
        realtime: Whether the source/sidecar actually sleep to match their
            configured pacing (True for real benchmarking; False for fast
            deterministic tests).
        seed: Seed forwarded to the synthetic source for reproducibility.
    """
    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario {name!r}, expected one of {SCENARIO_NAMES}")

    if name == "no_source_activity":
        return ScenarioSpec(
            name=name,
            source=None,
            policy=RejectionPolicy(accept_ratio=0.0),
            slam=FakeSlamSidecar(realtime=realtime),
            description="Pipeline idle -- no frames arrive at all.",
        )

    if name == "all_rejected":
        cfg = SyntheticSourceConfig(seed=seed, fps=30.0, num_frames=num_frames, realtime=realtime)
        return ScenarioSpec(
            name=name,
            source=SyntheticRGBDFrameSource(cfg),
            policy=RejectionPolicy(accept_ratio=0.0),
            slam=FakeSlamSidecar(realtime=realtime),
            description="30 FPS source, admission policy rejects every frame.",
        )

    if name == "two_fps_accepted":
        cfg = SyntheticSourceConfig(seed=seed, fps=30.0, num_frames=num_frames, realtime=realtime)
        return ScenarioSpec(
            name=name,
            source=SyntheticRGBDFrameSource(cfg),
            policy=RejectionPolicy.for_target_rate(source_fps=30.0, target_accepted_fps=2.0),
            slam=FakeSlamSidecar(realtime=realtime),
            description="30 FPS source, ~2 accepted keyframes/s.",
        )

    if name == "slam_delay_500ms":
        cfg = SyntheticSourceConfig(seed=seed, fps=30.0, num_frames=num_frames, realtime=realtime)
        return ScenarioSpec(
            name=name,
            source=SyntheticRGBDFrameSource(cfg),
            policy=RejectionPolicy(accept_ratio=1.0),
            slam=FakeSlamSidecar(delay_seconds=0.5, realtime=realtime),
            description="30 FPS source, every frame admitted, 500ms SLAM delay per frame.",
        )

    if name == "slam_failure":
        cfg = SyntheticSourceConfig(seed=seed, fps=30.0, num_frames=num_frames, realtime=realtime)
        return ScenarioSpec(
            name=name,
            source=SyntheticRGBDFrameSource(cfg),
            policy=RejectionPolicy(accept_ratio=1.0),
            slam=FakeSlamSidecar(fail_after=min(20, num_frames), realtime=realtime),
            description="30 FPS source, SLAM frontend fails after 20 frames.",
            expected_fault=True,
        )

    if name == "burst":
        cfg = SyntheticSourceConfig(
            seed=seed,
            fps=30.0,
            num_frames=num_frames,
            burst_size=30,
            burst_period_frames=None,
            realtime=realtime,
        )
        return ScenarioSpec(
            name=name,
            source=SyntheticRGBDFrameSource(cfg),
            policy=RejectionPolicy(accept_ratio=1.0),
            slam=FakeSlamSidecar(realtime=realtime),
            description="Initial burst of 30 frames emitted back-to-back, then steady 30 FPS.",
        )

    raise AssertionError("unreachable")  # pragma: no cover


@dataclass
class ScenarioResult:
    """Result of running one :class:`ScenarioSpec`.

    Carries three DISTINCT validity dimensions (review Fix 7 -- do not
    conflate these, and see ``scripts/benchmark_live_pipeline.py``'s
    manifest-validity computation for how they combine across a multi-
    scenario report):

    - ``harness_error``: did the *harness itself* crash unexpectedly (a bug
      in this benchmark's own code, not a simulated pipeline fault)? This is
      distinct from ``error``, which records ANY error string including an
      *expected* one (e.g. the injected ``SlamFailure`` message).
    - ``expected_fault`` (copied from the spec): was this scenario's whole
      point to deliberately trigger a fault? If so, the fault firing (and
      ``slam_failures > 0``/``error`` being set) is the CORRECT outcome for
      this scenario, not a run-invalidating one.
    - ``mapping_result_valid``: would this scenario's throughput/latency
      numbers be quotable as a real measurement? False for a scenario that
      hit an *unexpected* fault; True for ``slam_failure`` even though
      ``error`` is set, because the fault there was expected.
    """

    name: str
    description: str
    frames_seen: int = 0
    frames_accepted: int = 0
    slam_failures: int = 0
    wall_seconds: float = 0.0
    trace: dict = field(default_factory=dict)
    queue: dict = field(default_factory=dict)
    gpu_timing: dict = field(default_factory=dict)
    production_services_used: bool = False
    error: Optional[str] = None
    expected_fault: bool = False
    harness_error: bool = False

    @property
    def mapping_result_valid(self) -> bool:
        """True unless this scenario hit a fault that was NOT expected."""
        if self.harness_error:
            return False
        if self.error and not self.expected_fault:
            return False
        return True


def run_scenario(spec: ScenarioSpec, *, workload: BenchmarkWorkload | None = None) -> ScenarioResult:
    """Pump ``spec.source`` through ``spec.policy``/``spec.slam``, recording
    GPU-timed spans and queue metrics via the sibling profiling modules
    (imported lazily -- see module docstring). Falls back to a plain
    ``ScenarioResult`` with empty ``trace``/``queue`` blocks if those
    sibling modules aren't importable (defense-in-depth; they're a normal
    dependency of this package as of PR W3-01's integration).

    ``spec.source`` is produced on its own thread into a real bounded FIFO.
    A full queue records a ``queue_full`` drop, while admission rejection is
    a normal consumed outcome.  This makes burst behaviour measurable rather
    than merely annotating a sequential loop with queue counters.
    """
    result = ScenarioResult(name=spec.name, description=spec.description, expected_fault=spec.expected_fault)
    t_start = time.monotonic()

    try:
        from eligsir.runtime.profiling.pipeline_trace import PipelineTrace
    except Exception:
        PipelineTrace = None  # type: ignore[assignment]
    try:
        from eligsir.runtime.profiling.queue_metrics import QueueMetrics
    except Exception:
        QueueMetrics = None  # type: ignore[assignment]
    try:
        from eligsir.runtime.profiling.events import GpuSpanRegistry, gpu_timed_span
    except Exception:
        gpu_timed_span = None  # type: ignore[assignment]
        GpuSpanRegistry = None  # type: ignore[assignment]

    trace = PipelineTrace() if PipelineTrace is not None else None
    metrics = QueueMetrics(max_depth=spec.queue_capacity) if QueueMetrics is not None else None
    registry = GpuSpanRegistry() if GpuSpanRegistry is not None else None
    result.production_services_used = workload is not None

    if spec.source is not None:
        frames: stdlib_queue.Queue[object | None] = stdlib_queue.Queue(maxsize=spec.queue_capacity)
        producer_done = threading.Event()
        stop_producer = threading.Event()
        initial_burst_emitted = threading.Event()

        burst_size = getattr(getattr(spec.source, "config", None), "burst_size", 0)

        def produce() -> None:
            try:
                for produced, frame in enumerate(spec.source, start=1):
                    if stop_producer.is_set():
                        break
                    if burst_size:
                        # `burst` deliberately emits back-to-back into a
                        # queue smaller than the burst, with the consumer
                        # gated off until the burst finishes (see below) --
                        # a full queue here IS the scenario, and a
                        # non-blocking, single-attempt put is what makes the
                        # resulting `cumulative_drops > 0` deterministic
                        # rather than dependent on how much of the burst a
                        # blocking retry would let through before giving up.
                        try:
                            frames.put_nowait(frame)
                        except stdlib_queue.Full:
                            if metrics is not None:
                                metrics.on_drop("queue_full")
                            if produced >= burst_size:
                                initial_burst_emitted.set()
                            continue
                    else:
                        # issue #507: every other scenario's consumer drains
                        # concurrently from the very first frame (no
                        # burst-style start gate below), so a full queue
                        # here means the consumer is merely momentarily
                        # behind -- CI scheduling/GIL jitter, not real
                        # overload. `put_nowait` + drop-on-Full treated that
                        # as a genuine, permanent frame loss: reproduced
                        # under artificial thread-switch pressure
                        # (sys.setswitchinterval), where the tight,
                        # sleep-free `realtime=False` producer loop can fill
                        # an 8-slot queue and start dropping before the
                        # consumer thread gets scheduled even once. Block
                        # with a bounded retry instead, so the consumer
                        # catching up resolves the pressure instead of
                        # losing the frame -- this is what makes
                        # `frames_seen == num_frames` a real invariant for
                        # these scenarios. Still checks `stop_producer`
                        # between attempts so a mid-run SlamFailure abort
                        # can't leave this thread blocked forever.
                        enqueued = False
                        while not stop_producer.is_set():
                            try:
                                frames.put(frame, timeout=0.05)
                                enqueued = True
                                break
                            except stdlib_queue.Full:
                                continue
                        if not enqueued:
                            break  # stop_producer fired mid-retry -- abandon this frame
                    if metrics is not None:
                        metrics.on_enqueue()
                    if burst_size and produced >= burst_size:
                        initial_burst_emitted.set()
            finally:
                initial_burst_emitted.set()
                producer_done.set()
                # issue #507: a `None` sentinel, not `producer_done.is_set()
                # and frames.empty()`, is what tells the consumer loop below
                # to stop. That combination was two independently-read,
                # non-atomic checks -- `Queue.empty()` is a bare `_qsize()`
                # snapshot with no lock held across a producer's put() and
                # this thread setting `producer_done`, so the consumer could
                # observe "done" and "empty" simultaneously true while the
                # LAST one or two frames were still (or again) in flight,
                # exit early, and silently under-count `frames_seen` with
                # `cumulative_drops` staying at 0 -- exactly the flake this
                # fixes (23 != 25, no drop recorded). Best-effort/bounded:
                # if the consumer already stopped for its own reason (the
                # SlamFailure path below breaks out directly, not via this
                # sentinel), don't hang this daemon thread trying to deliver
                # a sentinel nobody will read.
                try:
                    frames.put(None, timeout=5.0)
                except stdlib_queue.Full:
                    pass

        producer = threading.Thread(target=produce, name=f"benchmark-source-{spec.name}", daemon=True)
        producer.start()
        # A burst is defined as source-side back-to-back arrival.  Let that
        # arrival finish before the consumer starts so capacity pressure is
        # deterministic instead of depending on OS thread scheduling.
        if getattr(getattr(spec.source, "config", None), "burst_size", 0):
            initial_burst_emitted.wait(timeout=1.0)
        while True:
            try:
                frame = frames.get(timeout=0.05)
            except stdlib_queue.Empty:
                continue
            if frame is None:
                break
            result.frames_seen += 1
            if metrics is not None:
                metrics.on_dequeue()
            accepted = workload.admit(frame) if workload is not None else spec.policy.accepts(frame)
            if not accepted:
                continue
            result.frames_accepted += 1
            if trace is not None:
                trace.record_admitted_keyframe()
            try:
                spec.slam.track(frame)
                if workload is not None:
                    # The supplied callbacks own their CUDA spans; this
                    # harness only supplies the scenario-owned registry.
                    span_cm = gpu_timed_span("optimization.step", registry) if gpu_timed_span else None
                    if span_cm is None:
                        workload.optimize_step()
                    else:
                        with span_cm:
                            workload.optimize_step()
                    span_cm = gpu_timed_span("mutation.commit", registry) if gpu_timed_span else None
                    if span_cm is None:
                        workload.commit(frame)
                    else:
                        with span_cm:
                            workload.commit(frame)
            except SlamFailure as exc:
                result.slam_failures += 1
                result.error = str(exc)
                stop_producer.set()
                break
        producer.join(timeout=1.0)

    # GPU spans recorded above via gpu_timed_span are NOT yet resolved -- resolve_pending()
    # performs the (at most one) torch.cuda.synchronize() needed to read their elapsed time,
    # done here at scenario-finalization time so the measured loop above never blocks on the
    # GPU (see eligsir.runtime.profiling.events module docstring).
    if registry is not None:
        registry.resolve_pending()
        result.gpu_timing = registry.snapshot()

    result.wall_seconds = time.monotonic() - t_start
    if trace is not None:
        result.trace = trace.metrics()
    if metrics is not None:
        result.queue = metrics.snapshot()
    return result
