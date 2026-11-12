"""Env-gated ``torch.profiler`` window over a run of mapping ticks.

Why: the per-phase profiler (``SPLAT_PROFILE_PHASES=1``) attributes host wall
to bracketed phases, but cannot say *what* a phase is made of. On the
2026-09-04 prepared-path arms ``m_loss_assembly`` stayed at ~0.8 ms per
supervision view after collapsing its host syncs to one, which means the cost
is the launch/Python overhead of many small ops -- and only an operator-level
profile can rank them. This window records ``count`` consecutive
``application.tick()`` calls starting at tick ``start`` and writes the
operator table (by self CPU time, with call counts) to a text file.

Enable with::

    SPLAT_TORCH_PROFILE_TICKS=<start>:<count>      # e.g. 1000:40
    SPLAT_TORCH_PROFILE_OUT=/output/<run>/tick_torch_profile.txt   # optional

Off (the default, env unset) it is a single attribute check per tick.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any


LAUNCH_EVENT_NAMES = ("cudaLaunchKernel", "hipLaunchKernel", "cudaMemcpyAsync", "cudaMemsetAsync")


def _launches_by_phase(profiler: Any, ticks: int, phase_prefix: str = "m_") -> list[str]:
    """Attribute launch-class events to the innermost enclosing phase range.

    Phase ranges are the ``torch.profiler.record_function`` annotations the
    ingest phase profiler opens for every ``_timed`` bracket while the window
    is armed (same names as ``ingest_phase_profile.json``). Nested ranges are
    resolved with a per-thread stack over start-sorted events, so a launch is
    charged to the innermost open phase; launches outside any phase are
    reported as ``<unattributed>``.
    """
    try:
        events = list(profiler.events())
    except Exception:
        return ["# launches by phase: unavailable"]
    launch_names = set(LAUNCH_EVENT_NAMES)
    per_thread: dict[Any, list[Any]] = {}
    for event in events:
        name = str(getattr(event, "name", ""))
        if name in launch_names or name.startswith(phase_prefix):
            per_thread.setdefault(getattr(event, "thread", 0), []).append(event)
    counts: dict[str, dict[str, int]] = {}
    for thread_events in per_thread.values():
        thread_events.sort(key=lambda item: (item.time_range.start, -item.time_range.end))
        stack: list[Any] = []
        for event in thread_events:
            start = event.time_range.start
            while stack and stack[-1].time_range.end <= start:
                stack.pop()
            name = str(event.name)
            if name in launch_names:
                phase = str(stack[-1].name) if stack else "<unattributed>"
                counts.setdefault(phase, {}).setdefault(name, 0)
                counts[phase][name] += 1
            else:
                stack.append(event)
    if not counts:
        return ["# launches by phase: no phase ranges recorded (set SPLAT_PROFILE_PHASES=1 too)"]
    lines = ["", f"# launch-class events per tick by innermost phase (window of {ticks} ticks)"]
    rows = sorted(counts.items(), key=lambda item: -sum(item[1].values()))
    for phase, by_name in rows:
        total = sum(by_name.values())
        detail = " ".join(f"{k}={v / max(1, ticks):.1f}" for k, v in sorted(by_name.items(), key=lambda kv: -kv[1]))
        lines.append(f"{total / max(1, ticks):8.1f}/tick  {phase:48s} {detail}")
    return lines


class TickTorchProfileWindow:
    ENV_TICKS = "SPLAT_TORCH_PROFILE_TICKS"
    ENV_OUT = "SPLAT_TORCH_PROFILE_OUT"

    def __init__(self, *, env: dict[str, str] | None = None, row_limit: int = 80):
        env = os.environ if env is None else env
        spec = str(env.get(self.ENV_TICKS, "") or "").strip()
        self.enabled = False
        self.start = 0
        self.count = 0
        self.output_path = str(env.get(self.ENV_OUT, "") or "tick_torch_profile.txt")
        self.row_limit = int(row_limit)
        self.tick_index = 0
        self.completed = False
        self._profiler: Any = None
        if spec:
            start_text, _, count_text = spec.partition(":")
            self.start = int(start_text)
            self.count = int(count_text or 1)
            if self.start < 0 or self.count < 1:
                raise ValueError(f"{self.ENV_TICKS} must be '<start>:<count>' with count >= 1")
            self.enabled = True

    def _open(self) -> None:
        import torch

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities, record_shapes=False, profile_memory=False, with_stack=False
        )
        self._profiler.__enter__()

    def _close(self) -> None:
        profiler = self._profiler
        self._profiler = None
        if profiler is None:
            return
        profiler.__exit__(None, None, None)
        averages = profiler.key_averages()
        lines = [
            f"# torch.profiler window: ticks [{self.start}, {self.start + self.count}) "
            f"({self.count} ticks); rows sorted by self CPU time; counts are per window",
            averages.table(sort_by="self_cpu_time_total", row_limit=self.row_limit),
            "",
            "# by call count",
        ]
        by_count = sorted(averages, key=lambda item: -int(item.count))[: self.row_limit]
        total_calls = sum(int(item.count) for item in averages)
        lines.append(f"total_ops={total_calls} ops_per_tick={total_calls / max(1, self.count):.1f}")
        for item in by_count:
            lines.append(
                f"{int(item.count):8d}  self_cpu_us={float(item.self_cpu_time_total):10.0f}  "
                f"cuda_us={float(getattr(item, 'self_device_time_total', getattr(item, 'self_cuda_time_total', 0.0))):10.0f}  {item.key}"
            )
        lines.extend(_launches_by_phase(profiler, self.count))
        text = "\n".join(lines) + "\n"
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(
            f"[torch-profile] window ticks {self.start}..{self.start + self.count - 1} written to "
            f"{self.output_path} (total_ops={total_calls}, ops_per_tick={total_calls / max(1, self.count):.1f})",
            flush=True,
        )
        self.completed = True

    @contextlib.contextmanager
    def tick(self):
        """Wrap one ``application.tick()`` call; opens/closes the window at its bounds."""
        if not self.enabled or self.completed:
            yield
            return
        index = self.tick_index
        self.tick_index += 1
        if index == self.start:
            self._open()
        try:
            yield
        finally:
            if self._profiler is not None and index >= self.start + self.count - 1:
                self._close()

    def close(self) -> None:
        """Flush an open window early (source exhausted before the window ended)."""
        if self._profiler is not None:
            self._close()
