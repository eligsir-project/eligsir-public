"""Default-on per-run system-performance sampler.

PR #1783 (``docs: require per-run system performance and storage evidence``)
made timestamped host/GPU system-performance curves a documented requirement
(``docs/EVALUATION_PROTOCOL.md#system-performance-and-input-storage``) without
shipping an implementation: only ResearchFlow-wrapped runs got a trace, via
``researchflow/execution.py``'s ``TelemetrySampler`` writing into the RF
registry. A plain ``causal_mapping`` (or offline ``train.py``) run had none.

This module is the missing implementation, wired to run for EVERY run by
default (``--system_telemetry``, default ``True``): a background daemon
thread samples host/process CPU, memory, and GPU counters every
``interval_s`` seconds into ``<model_path>/system_performance.jsonl``, plus
phase-marker rows (``{"event": ...}``) recorded by the caller at meaningful
boundaries (mapping endpoint, refinement start/end, source EOF).

Design notes
------------
* **Never raises.** Every reader below is best-effort: a missing sensor, an
  absent binary, or a transient read failure yields ``None`` for that
  channel (the EVALUATION_PROTOCOL's "unavailable channels must carry null,
  never zero or a stale value" rule) rather than raising out of the sampler
  thread or blocking/crashing the run it is observing.
* **No GPU synchronization.** GPU memory is read via the CUDA caching
  allocator's own bookkeeping (``torch.cuda.memory_reserved`` /
  ``memory_allocated``), matching the #480 "observability must never
  backpressure training" invariant already documented for
  ``utils/gpu_memory_telemetry.py``. Utilization/power/temperature come from
  sysfs (ROCm) or a subprocess (``nvidia-smi``), never from an in-process
  CUDA call that could synchronize the device.
* **Module-level "active sampler" registry.** ``run_causal_post_mapping_
  refinement`` and ``CausalEvaluationObserver.source_complete`` live in
  different modules/call frames than the sampler's owner
  (``run_causal_mapping``'s thin wrapper). Rather than threading a sampler
  object through every intervening signature, callers record a phase marker
  via the free function ``mark_system_telemetry(event, **fields)``, which
  looks up whichever sampler ``start_system_telemetry`` most recently
  activated (and is a silent no-op when none is active, e.g. telemetry
  disabled or no run in flight -- matching every other telemetry helper in
  this codebase, e.g. ``record_causal_throughput``).
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from eligsir.plot_style import svg_output_path

TRACE_FILENAME = "system_performance.jsonl"
PLOT_FILENAME = "system_performance.svg"
SUMMARY_FILENAME = "system_performance_summary.json"

# Ordered phase-marker vocabulary the causal trainer emits via
# ``mark_system_telemetry``. Kept here (rather than re-derived from the trace)
# so the summary/plot code has one place to agree on names with callers.
EVENT_MAPPING_ENDPOINT = "mapping_endpoint"
EVENT_REFINEMENT_START = "refinement_start"
EVENT_REFINEMENT_END = "refinement_end"
EVENT_SOURCE_EOF = "source_eof"

_REGISTRY_LOCK = threading.Lock()
_ACTIVE_SAMPLER: "SystemPerformanceSampler | None" = None


def _read_host_mem_gb() -> tuple[float | None, float | None]:
    """Return ``(used_gb, total_gb)`` host RAM, psutil if present else /proc."""

    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        return vm.used / 1e9, vm.total / 1e9
    except Exception:
        pass
    try:
        fields: dict[str, int] = {}
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"):
                    fields[key] = int(rest.strip().split()[0])  # kB
        total_kb = fields.get("MemTotal")
        if total_kb is None:
            return None, None
        if "MemAvailable" in fields:
            used_kb = total_kb - fields["MemAvailable"]
        elif {"MemFree", "Buffers", "Cached"} <= fields.keys():
            used_kb = total_kb - fields["MemFree"] - fields["Buffers"] - fields["Cached"]
        else:
            return None, total_kb / 1e6
        return used_kb / 1e6, total_kb / 1e6
    except (OSError, ValueError, IndexError):
        return None, None


def _read_process_rss_gb(pid: int) -> float | None:
    """Return this process's RSS in GB, psutil if present else /proc/<pid>/status."""

    try:
        import psutil  # type: ignore

        return psutil.Process(pid).memory_info().rss / 1e9
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1e6
    except (OSError, ValueError, IndexError):
        pass
    return None


class _HostCpuProcReader:
    """Host CPU% (0-100 across all logical CPUs) from ``/proc/stat`` deltas.

    Used only when ``psutil`` is unavailable; ``psutil.cpu_percent`` is
    preferred when importable (see ``SystemPerformanceSampler._cpu_pct``).
    """

    def __init__(self, proc_stat_path: str = "/proc/stat") -> None:
        self._path = proc_stat_path
        self._prev_idle: int | None = None
        self._prev_total: int | None = None

    def sample_pct(self) -> float | None:
        """Return the host CPU busy percentage since the previous call."""

        try:
            with open(self._path) as handle:
                line = handle.readline()
            parts = line.split()
            if not parts or parts[0] != "cpu":
                return None
            values = [int(v) for v in parts[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            total = sum(values)
        except (OSError, ValueError, IndexError):
            return None
        prev_idle, prev_total = self._prev_idle, self._prev_total
        self._prev_idle, self._prev_total = idle, total
        if prev_total is None:
            return None
        delta_total = total - prev_total
        delta_idle = idle - prev_idle
        if delta_total <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (delta_total - delta_idle) / delta_total))


class _ProcessCpuProcReader:
    """Process CPU% (0-100 per core, so up to ``100 * n_cpus``) via /proc/<pid>/stat."""

    def __init__(self, pid: int, stat_path: str | None = None) -> None:
        self._pid = pid
        self._path = stat_path or f"/proc/{pid}/stat"
        try:
            self._clk_tck = float(os.sysconf("SC_CLK_TCK"))
        except (ValueError, AttributeError, OSError):
            self._clk_tck = 100.0
        self._prev_ticks: int | None = None
        self._prev_wall: float | None = None

    def sample_pct(self) -> float | None:
        """Return this process's CPU busy percentage since the previous call."""

        try:
            with open(self._path) as handle:
                raw = handle.read()
            after = raw.rsplit(")", 1)[-1].split()
            utime = int(after[11])
            stime = int(after[12])
        except (OSError, ValueError, IndexError):
            return None
        ticks = utime + stime
        now = time.time()
        prev_ticks, prev_wall = self._prev_ticks, self._prev_wall
        self._prev_ticks, self._prev_wall = ticks, now
        if prev_ticks is None:
            return None
        dt = now - prev_wall
        if dt <= 0:
            return None
        cpu_seconds = (ticks - prev_ticks) / self._clk_tck
        return max(0.0, 100.0 * cpu_seconds / dt)


def _default_torch_cuda_mem_gb() -> tuple[float | None, float | None] | None:
    """Process-attributed GPU memory via the torch CUDA caching allocator.

    Uses ``memory_reserved``/``get_device_properties`` bookkeeping only --
    no device synchronization. Returns ``None`` (not a tuple of ``None``)
    when torch/CUDA are unavailable, so callers can fall through to the
    ROCm/nvidia-smi readers for memory too.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        device = torch.cuda.current_device()
        used = float(torch.cuda.memory_reserved(device)) / 1e9
        total = float(torch.cuda.get_device_properties(device).total_memory) / 1e9
        return used, total
    except Exception:
        return None


def _default_rocm_reader(card_glob: str = "/sys/class/drm/card*/device") -> dict[str, Any] | None:
    """Read AMD ROCm/amdgpu sysfs counters for the first matching card.

    Covers the Strix Halo iGPU case called out in the design: GTT (shared
    system memory used as GPU memory) is a distinct counter from VRAM and is
    recorded separately as ``gtt_used_gb`` when present, never summed into
    VRAM usage.
    """

    import glob

    candidates = sorted(glob.glob(card_glob))
    for device_dir in candidates:
        try:
            gpu_busy = _read_int(os.path.join(device_dir, "gpu_busy_percent"))
            vram_used = _read_int(os.path.join(device_dir, "mem_info_vram_used"))
            vram_total = _read_int(os.path.join(device_dir, "mem_info_vram_total"))
            gtt_used = _read_int(os.path.join(device_dir, "mem_info_gtt_used"))
        except OSError:
            continue
        if gpu_busy is None and vram_used is None:
            continue
        power_w = None
        temp_c = None
        hwmon_root = os.path.join(device_dir, "hwmon")
        if os.path.isdir(hwmon_root):
            for entry in sorted(os.listdir(hwmon_root)):
                hwmon_dir = os.path.join(hwmon_root, entry)
                power_raw = _read_int(os.path.join(hwmon_dir, "power1_average"))
                if power_raw is not None:
                    power_w = power_raw / 1e6  # microwatts
                temp_raw = _read_int(os.path.join(hwmon_dir, "temp1_input"))
                if temp_raw is not None:
                    temp_c = temp_raw / 1e3  # millidegrees
                if power_w is not None or temp_c is not None:
                    break
        return {
            "util_pct": float(gpu_busy) if gpu_busy is not None else None,
            "mem_used_gb": vram_used / 1e9 if vram_used is not None else None,
            "mem_total_gb": vram_total / 1e9 if vram_total is not None else None,
            "gtt_used_gb": gtt_used / 1e9 if gtt_used is not None else None,
            "power_w": power_w,
            "temp_c": temp_c,
        }
    return None


def _read_int(path: str) -> int | None:
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _default_nvidia_smi_reader() -> dict[str, Any] | None:
    """Read utilization/memory/power/temperature for GPU 0 via ``nvidia-smi``."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        line = result.stdout.strip().splitlines()[0]
        util, mem_used, mem_total, power, temp = (p.strip() for p in line.split(","))
        return {
            "util_pct": _parse_float(util),
            "mem_used_gb": _parse_float(mem_used, scale=1 / 1000),
            "mem_total_gb": _parse_float(mem_total, scale=1 / 1000),
            "power_w": _parse_float(power),
            "temp_c": _parse_float(temp),
        }
    except (ValueError, IndexError):
        return None


def _parse_float(value: str, *, scale: float = 1.0) -> float | None:
    try:
        return float(value) * scale
    except ValueError:
        return None


class SystemPerformanceSampler:
    """Background daemon-thread sampler writing ``system_performance.jsonl``.

    Construct with the readers left at their defaults for production use;
    tests inject fakes for ``torch_reader``/``rocm_reader``/``nvidia_reader``
    so they never touch a real GPU or shell out to ``nvidia-smi``.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        interval_s: float = 1.0,
        pid: int | None = None,
        torch_reader: Callable[[], tuple[float | None, float | None] | None] = (
            _default_torch_cuda_mem_gb
        ),
        rocm_reader: Callable[[], dict[str, Any] | None] = _default_rocm_reader,
        nvidia_reader: Callable[[], dict[str, Any] | None] = _default_nvidia_smi_reader,
    ) -> None:
        self._path = Path(model_path) / TRACE_FILENAME
        self._interval_s = max(0.05, float(interval_s))
        self._pid = int(pid) if pid is not None else os.getpid()
        self._torch_reader = torch_reader
        self._rocm_reader = rocm_reader
        self._nvidia_reader = nvidia_reader
        self._nvidia_smi_failed = False

        self._host_cpu = _HostCpuProcReader()
        self._process_cpu = _ProcessCpuProcReader(self._pid)
        self._psutil = _try_import_psutil()
        self._psutil_process = None
        if self._psutil is not None:
            try:
                self._psutil_process = self._psutil.Process(self._pid)
                # Prime the non-blocking psutil.cpu_percent() interval timers
                # (first call always returns 0.0 / meaningless).
                self._psutil.cpu_percent(interval=None)
                self._psutil_process.cpu_percent(interval=None)
            except Exception:
                self._psutil_process = None

        self._start_wall = time.time()
        self._start_perf = time.perf_counter()
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "SystemPerformanceSampler":
        """Create the trace file and start the background sampling thread."""

        if self._thread is not None:
            return self
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="system-performance-sampler", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal the sampling thread to stop and join it (best-effort)."""

        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def mark(self, event: str, **fields: Any) -> None:
        """Append a phase-marker row; never raises."""

        row: dict[str, Any] = {
            "event": str(event),
            "t_wall": time.time(),
            "t_elapsed_s": max(0.0, time.perf_counter() - self._start_perf),
        }
        row.update(fields)
        self._append(row)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._append(self._collect_row())
            except Exception:
                pass
            self._stop_event.wait(self._interval_s)

    def _cpu_pct(self) -> tuple[float | None, float | None]:
        """Return ``(process_pct, host_pct)``, psutil if available else /proc."""

        if self._psutil_process is not None:
            try:
                host_pct = self._psutil.cpu_percent(interval=None)
                proc_pct = self._psutil_process.cpu_percent(interval=None)
                return proc_pct, host_pct
            except Exception:
                pass
        return self._process_cpu.sample_pct(), self._host_cpu.sample_pct()

    def _gpu_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "gpu_util_pct": None,
            "gpu_mem_used_gb": None,
            "gpu_mem_total_gb": None,
            "gpu_gtt_used_gb": None,
            "gpu_power_w": None,
            "gpu_temp_c": None,
            "gpu_source": None,
        }
        try:
            torch_mem = self._torch_reader()
        except Exception:
            torch_mem = None
        if torch_mem is not None:
            fields["gpu_mem_used_gb"], fields["gpu_mem_total_gb"] = torch_mem
            fields["gpu_source"] = "torch.cuda"

        rocm = None
        try:
            rocm = self._rocm_reader()
        except Exception:
            rocm = None
        if rocm is not None:
            fields["gpu_util_pct"] = rocm.get("util_pct")
            if fields["gpu_mem_used_gb"] is None:
                fields["gpu_mem_used_gb"] = rocm.get("mem_used_gb")
                fields["gpu_mem_total_gb"] = rocm.get("mem_total_gb")
            fields["gpu_gtt_used_gb"] = rocm.get("gtt_used_gb")
            fields["gpu_power_w"] = rocm.get("power_w")
            fields["gpu_temp_c"] = rocm.get("temp_c")
            fields["gpu_source"] = fields["gpu_source"] or "rocm_sysfs"
            return fields

        if not self._nvidia_smi_failed:
            nv = None
            try:
                nv = self._nvidia_reader()
            except Exception:
                nv = None
            if nv is not None:
                fields["gpu_util_pct"] = nv.get("util_pct")
                if fields["gpu_mem_used_gb"] is None:
                    fields["gpu_mem_used_gb"] = nv.get("mem_used_gb")
                    fields["gpu_mem_total_gb"] = nv.get("mem_total_gb")
                fields["gpu_power_w"] = nv.get("power_w")
                fields["gpu_temp_c"] = nv.get("temp_c")
                fields["gpu_source"] = fields["gpu_source"] or "nvidia_smi"
            else:
                self._nvidia_smi_failed = True
        return fields

    def _collect_row(self) -> dict[str, Any]:
        cpu_process_pct, cpu_host_pct = self._cpu_pct()
        host_used_gb, host_total_gb = _read_host_mem_gb()
        rss_gb = (
            self._psutil_process.memory_info().rss / 1e9
            if self._psutil_process is not None
            else _read_process_rss_gb(self._pid)
        )
        row: dict[str, Any] = {
            "t_wall": time.time(),
            "t_elapsed_s": max(0.0, time.perf_counter() - self._start_perf),
            "cpu_pct_process": cpu_process_pct,
            "cpu_pct_host": cpu_host_pct,
            "rss_gb": rss_gb,
            "host_mem_used_gb": host_used_gb,
            "host_mem_total_gb": host_total_gb,
        }
        row.update(self._gpu_fields())
        return row

    def _append(self, row: dict[str, Any]) -> None:
        try:
            with self._write_lock, self._path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:
            pass


def _try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


def start_system_telemetry(
    model_path: str | Path, *, interval_s: float = 1.0, enabled: bool = True
) -> SystemPerformanceSampler | None:
    """Start (and register as "active") the per-run system-performance sampler.

    Returns ``None`` when ``enabled`` is false so callers can uniformly do
    ``sampler = start_system_telemetry(...); ...; stop_system_telemetry(sampler)``
    without an extra branch -- ``stop_system_telemetry(None)`` is a no-op.
    """

    global _ACTIVE_SAMPLER
    if not enabled:
        return None
    sampler = SystemPerformanceSampler(model_path, interval_s=interval_s).start()
    with _REGISTRY_LOCK:
        _ACTIVE_SAMPLER = sampler
    return sampler


def stop_system_telemetry(sampler: SystemPerformanceSampler | None) -> None:
    """Stop the sampler thread and clear it from the active registry."""

    global _ACTIVE_SAMPLER
    if sampler is None:
        return
    sampler.stop()
    with _REGISTRY_LOCK:
        if _ACTIVE_SAMPLER is sampler:
            _ACTIVE_SAMPLER = None


def mark_system_telemetry(event: str, **fields: Any) -> None:
    """Record a phase-marker row on whichever sampler is currently active.

    Silent no-op when no sampler is active (telemetry disabled, or called
    outside a run) -- matching every other best-effort telemetry helper in
    this codebase (e.g. ``record_causal_throughput``).
    """

    with _REGISTRY_LOCK:
        sampler = _ACTIVE_SAMPLER
    if sampler is not None:
        try:
            sampler.mark(event, **fields)
        except Exception:
            pass


def _load_trace_rows(model_path: str | Path) -> list[dict[str, Any]]:
    trace = Path(model_path) / TRACE_FILENAME
    if not trace.is_file():
        return []
    rows = []
    for line in trace.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def plot_system_performance(model_path: str | Path) -> Path | None:
    """Plot GPU util/mem, host RSS, CPU%, and GPU power over elapsed time.

    Four stacked panels (GPU utilization %, GPU memory GB + host RSS GB, CPU%
    process/host, GPU power W) against ``t_elapsed_s``, with a vertical line
    at every phase-marker row (``{"event": ...}``) written via
    ``mark_system_telemetry``. Matches the sibling report plots' style
    (``eligsir/runtime/causal_throughput.py``'s ``plot_causal_throughput``):
    matplotlib Agg backend, dpi=100, a true no-op (returns ``None``) when the
    trace is absent or empty.
    """

    rows = _load_trace_rows(model_path)
    sample_rows = [row for row in rows if "event" not in row]
    event_rows = [row for row in rows if "event" in row]
    if not sample_rows:
        return None

    import numpy as np

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample_rows = sorted(sample_rows, key=lambda row: float(row["t_elapsed_s"]))
    elapsed_s = np.asarray([float(row["t_elapsed_s"]) for row in sample_rows])

    def _series(key: str) -> np.ndarray:
        return np.asarray(
            [np.nan if row.get(key) is None else float(row[key]) for row in sample_rows]
        )

    gpu_util = _series("gpu_util_pct")
    gpu_mem = _series("gpu_mem_used_gb")
    rss = _series("rss_gb")
    cpu_process = _series("cpu_pct_process")
    cpu_host = _series("cpu_pct_host")
    gpu_power = _series("gpu_power_w")

    fig, axes = plt.subplots(4, 1, figsize=(9, 12), dpi=100, sharex=True)
    util_ax, mem_ax, cpu_ax, power_ax = axes

    util_ax.plot(elapsed_s, gpu_util, color="tab:green", linewidth=1.5)
    util_ax.set_ylabel("GPU util (%)")
    util_ax.set_title("System performance over time")
    util_ax.grid(True, alpha=0.3)

    mem_ax.plot(elapsed_s, gpu_mem, color="tab:purple", linewidth=1.5, label="GPU mem (GB)")
    mem_ax.plot(elapsed_s, rss, color="tab:brown", linewidth=1.2, label="process RSS (GB)")
    mem_ax.set_ylabel("memory (GB)")
    mem_ax.grid(True, alpha=0.3)
    mem_ax.legend(fontsize=8, loc="best")

    cpu_ax.plot(elapsed_s, cpu_process, color="tab:blue", linewidth=1.2, label="process CPU%")
    cpu_ax.plot(elapsed_s, cpu_host, color="tab:blue", alpha=0.4, linestyle="--", linewidth=1.2, label="host CPU%")
    cpu_ax.set_ylabel("CPU (%)")
    cpu_ax.grid(True, alpha=0.3)
    cpu_ax.legend(fontsize=8, loc="best")

    power_ax.plot(elapsed_s, gpu_power, color="tab:red", linewidth=1.5)
    power_ax.set_ylabel("GPU power (W)")
    power_ax.set_xlabel("elapsed wall time (s)")
    power_ax.grid(True, alpha=0.3)

    for row in event_rows:
        t = float(row["t_elapsed_s"])
        for ax in axes:
            ax.axvline(t, color="black", alpha=0.35, linestyle=":", linewidth=1.0)
        util_ax.annotate(
            str(row["event"]),
            xy=(t, 1.0),
            xycoords=("data", "axes fraction"),
            rotation=90,
            fontsize=7,
            ha="right",
            va="top",
            alpha=0.7,
        )

    fig.tight_layout()
    output = svg_output_path(Path(model_path) / PLOT_FILENAME)
    fig.savefig(output, dpi=100)
    plt.close(fig)
    return output


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _segment_bounds(
    event_rows: list[dict[str, Any]], start_event: str | None, end_event: str | None
) -> tuple[float, float] | None:
    """Resolve a ``[start_t, end_t]`` window from named marker events.

    ``start_event=None`` means "from the beginning of the trace". Returns
    ``None`` (no window, not a zero-width one) when a required marker is
    absent -- callers must then report ``null`` stats for that phase rather
    than substituting an incorrect boundary.
    """

    if end_event is None:
        return None
    end_matches = [row for row in event_rows if row.get("event") == end_event]
    if not end_matches:
        return None
    end_t = min(float(row["t_elapsed_s"]) for row in end_matches)
    if start_event is None:
        return 0.0, end_t
    start_matches = [row for row in event_rows if row.get("event") == start_event]
    if not start_matches:
        return None
    start_t = min(float(row["t_elapsed_s"]) for row in start_matches)
    return start_t, end_t


def write_system_performance_summary(model_path: str | Path) -> dict[str, Any] | None:
    """Write ``system_performance_summary.json`` and return its payload.

    Mean/max GPU utilization during the mapping window (start of run through
    ``mapping_endpoint``, or through ``source_eof`` if no mapping_endpoint
    marker was recorded, or the whole trace if neither marker was ever
    recorded at all -- e.g. a non-causal/offline ``train.py`` run, which has
    no such phase boundaries) and during the refinement window
    (``refinement_start`` through ``refinement_end``, only when BOTH are
    present -- unlike the mapping window this has no whole-trace fallback,
    since "no refinement markers" genuinely means no refinement phase ran),
    plus peak GPU memory, peak process RSS, and mean GPU power over the whole
    run. A window with a missing boundary marker reports ``null`` stats
    rather than guessing -- never a fabricated zero.

    Note the gap between ``mapping_endpoint`` and ``refinement_start``:
    ``run_causal_post_mapping_refinement``'s ``mapping_endpoint_reporter``
    runs a full evaluation pass (contact sheets, PSNR-by-distance, report
    writes) in that interval, AFTER the mapping window closes but BEFORE the
    refinement window opens. That GPU work deliberately lands in neither
    window's mean/max utilization -- it still contributes to the whole-run
    ``peak_gpu_mem_used_gb``/``peak_rss_gb``/``mean_gpu_power_w`` figures,
    just not to either phase's utilization stats. This is intentional (the
    mapping-endpoint report is its own, third thing, not mapping and not
    refinement), not a silently dropped span.

    Returns ``None`` (a true no-op) when the trace is absent or has no
    sample rows, matching the plot function's own convention.
    """

    rows = _load_trace_rows(model_path)
    sample_rows = [row for row in rows if "event" not in row]
    event_rows = [row for row in rows if "event" in row]
    if not sample_rows:
        return None

    def _phase_stats(window: tuple[float, float] | None) -> dict[str, Any]:
        if window is None:
            return {"mean_gpu_util_pct": None, "max_gpu_util_pct": None, "n_samples": 0}
        start_t, end_t = window
        values = [
            float(row["gpu_util_pct"])
            for row in sample_rows
            if row.get("gpu_util_pct") is not None
            and start_t <= float(row["t_elapsed_s"]) <= end_t
        ]
        return {
            "mean_gpu_util_pct": _mean(values),
            "max_gpu_util_pct": max(values) if values else None,
            "n_samples": len(values),
        }

    mapping_end_event = (
        EVENT_MAPPING_ENDPOINT
        if any(row.get("event") == EVENT_MAPPING_ENDPOINT for row in event_rows)
        else EVENT_SOURCE_EOF
    )
    mapping_window = _segment_bounds(event_rows, None, mapping_end_event)
    if mapping_window is None:
        # No mapping_endpoint/source_eof marker at all -- e.g. a plain
        # offline (non-causal_mapping) run, which never emits either. There
        # is no distinct "mapping" phase boundary to report, but the whole
        # run IS the only phase that ran, so use it rather than reporting
        # null for a run that plainly has GPU samples.
        mapping_window = (0.0, max(float(row["t_elapsed_s"]) for row in sample_rows))
    refinement_window = _segment_bounds(
        event_rows, EVENT_REFINEMENT_START, EVENT_REFINEMENT_END
    )

    gpu_mem_values = [
        float(row["gpu_mem_used_gb"]) for row in sample_rows if row.get("gpu_mem_used_gb") is not None
    ]
    rss_values = [float(row["rss_gb"]) for row in sample_rows if row.get("rss_gb") is not None]
    power_values = [
        float(row["gpu_power_w"]) for row in sample_rows if row.get("gpu_power_w") is not None
    ]

    payload: dict[str, Any] = {
        "n_samples": len(sample_rows),
        "n_events": len(event_rows),
        "mapping": _phase_stats(mapping_window),
        "refinement": _phase_stats(refinement_window),
        "peak_gpu_mem_used_gb": max(gpu_mem_values) if gpu_mem_values else None,
        "peak_rss_gb": max(rss_values) if rss_values else None,
        "mean_gpu_power_w": _mean(power_values),
    }
    output = Path(model_path) / SUMMARY_FILENAME
    try:
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass
    return payload


def write_system_performance_report(model_path: str | Path) -> dict[str, Any] | None:
    """Convenience wrapper: write both the plot and the summary; never raises."""

    try:
        plot_system_performance(model_path)
    except Exception:
        pass
    try:
        return write_system_performance_summary(model_path)
    except Exception:
        return None
