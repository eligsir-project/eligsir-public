"""Lightweight per-run host resource monitor (env-gated; SPLAT_RESOURCE_MONITOR=1).

Polls the LOCAL host's Glances REST API (v4, ``/api/4/...``) from a background
daemon thread at a low-frequency interval and appends one JSON line per
sample to ``<model_path>/resource_usage.jsonl``. This captures host-level
RAM, CPU, and GPU utilization/VRAM over the course of a training run,
independent of and complementary to the GPU-memory-*allocator* telemetry in
``utils/gpu_memory_telemetry.py`` (which reads PyTorch's own allocator state,
not the host's view of the device).

Follows the same conventions as ``utils/ingest_phase_profiler.py``
(SPLAT_PROFILE_PHASES): an env var gate read once at import time, a class
that is only ever *constructed* when the feature is enabled (so the disabled
cost at every call site is exactly one ``is None`` check), and graceful,
never-crash-training failure handling.

Glances runs on the BARE-METAL HOST, not inside the training container (see
CLAUDE.md's Glances note), so "localhost" from inside the container does not
reach it. Host reachability is resolved once, at monitor construction, by
trying a short candidate list of hostnames (in order):

  1. ``$SPLAT_GLANCES_HOST`` if set (single explicit override, no fallback list)
  2. ``host.docker.internal`` -- resolves when the container is started with
     ``extra_hosts: ["host.docker.internal:host-gateway"]``, which
     ``docker-compose.yml`` now sets, under BOTH real Docker Engine (>=20.10)
     and podman/podman-compose (podman understands the ``host-gateway``
     special value since podman 4.1).
  3. ``host.containers.internal`` -- podman's own equivalent, added
     automatically to rootless podman's default bridge network with no
     compose configuration needed at all (verified against this repo's own
     ``docker-compose.yml`` networking, which sets neither
     ``network_mode: host`` nor any other custom network).

Verified empirically (2026-08-31, this repo's actual host + podman-compose
setup) that both (2) and (3) reach the host's real Glances instance from
inside a plain container on the default bridge network.

If none of the candidates answer within a short connect timeout, the monitor
logs ONE warning and disables itself permanently for the run -- it never
retries in a tight loop and never raises into the training loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

RESOURCE_MONITOR_ENABLED = bool(os.environ.get("SPLAT_RESOURCE_MONITOR"))

DEFAULT_PORT = 61208
DEFAULT_INTERVAL_S = 2.0
_CONNECT_TIMEOUT_S = 2.0
# Candidates tried (in order) when SPLAT_GLANCES_HOST is not set explicitly.
_DEFAULT_HOST_CANDIDATES = ("host.docker.internal", "host.containers.internal", "localhost")

_ARTIFACT_NAME = "resource_usage.jsonl"


def _resolve_base_url() -> Optional[str]:
    """Return the first Glances base URL that answers, or None.

    Tried once at monitor construction, not per-poll -- a host that is
    reachable now is assumed to stay reachable for the run; if it goes away
    mid-run, individual poll failures are handled by the caller (one warning,
    then self-disable), not by re-resolving.
    """
    port = int(os.environ.get("SPLAT_GLANCES_PORT", DEFAULT_PORT))
    explicit = os.environ.get("SPLAT_GLANCES_HOST")
    candidates = [explicit] if explicit else list(_DEFAULT_HOST_CANDIDATES)
    for host in candidates:
        if not host:
            continue
        base = f"http://{host}:{port}"
        try:
            with urllib.request.urlopen(f"{base}/api/4/cpu", timeout=_CONNECT_TIMEOUT_S) as resp:
                if resp.status == 200:
                    return base
        except Exception:
            continue
    return None


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=_CONNECT_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


class ResourceMonitor:
    """Background poller for host CPU/RAM/GPU stats via Glances' REST API.

    Construct only when ``RESOURCE_MONITOR_ENABLED`` (or an explicit opt-in)
    is True -- mirrors ``GPUMemoryResidencyTracker``'s
    "constructed only when enabled" convention in train_streaming.py, so a
    disabled run pays zero cost beyond the one guarding ``if`` at the call
    site.

    Usage::

        monitor = ResourceMonitor(model_path) if RESOURCE_MONITOR_ENABLED else None
        if monitor is not None:
            monitor.start()
        ...
        if monitor is not None:
            monitor.set_iteration(iteration)   # cheap; call every loop tick
        ...
        if monitor is not None:
            monitor.stop()
            summary = monitor.summary()  # -> report.json["resource_monitor"]
    """

    def __init__(self, model_path: str, interval_s: Optional[float] = None):
        self.model_path = model_path
        self.interval_s = float(
            interval_s if interval_s is not None
            else os.environ.get("SPLAT_RESOURCE_MONITOR_INTERVAL_S", DEFAULT_INTERVAL_S)
        )
        self._iteration: Optional[int] = None
        self._iteration_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._base_url: Optional[str] = None
        self._warned = False
        self._disabled = False
        self._samples: list = []  # in-memory mirror for summary(); bounded below
        self._max_samples_in_memory = 100000
        self._path = os.path.join(model_path, _ARTIFACT_NAME)

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._base_url = _resolve_base_url()
        if self._base_url is None:
            self._warn_once(
                "Glances unreachable at any candidate host "
                f"({os.environ.get('SPLAT_GLANCES_HOST') or ', '.join(_DEFAULT_HOST_CANDIDATES)}); "
                "resource monitoring disabled for this run."
            )
            self._disabled = True
            return
        print(
            f"[resource-monitor] enabled: base_url={self._base_url} "
            f"interval={self.interval_s:g}s artifact={self._path}",
            flush=True,
        )
        self._thread = threading.Thread(target=self._run, daemon=True, name="resource-monitor")
        self._thread.start()

    def set_iteration(self, iteration: int) -> None:
        """Record the current training iteration for the next sample.

        Cheap (a lock + int write) -- safe to call every training iteration
        even while the monitor is disabled (no-op guard first).
        """
        if self._disabled:
            return
        with self._iteration_lock:
            self._iteration = int(iteration)

    def stop(self, join_timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)

    # -- internals -------------------------------------------------------
    def _warn_once(self, msg: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"[resource-monitor] WARNING: {msg}", flush=True)

    def _current_iteration(self) -> Optional[int]:
        with self._iteration_lock:
            return self._iteration

    def _poll_once(self) -> Optional[dict]:
        cpu = _get_json(f"{self._base_url}/api/4/cpu")
        mem = _get_json(f"{self._base_url}/api/4/mem")
        try:
            gpu = _get_json(f"{self._base_url}/api/4/gpu")
        except Exception:
            gpu = []  # a host with no GPU plugin is not a monitor failure
        sample = {
            "ts": time.time(),
            "iteration": self._current_iteration(),
            "cpu_percent": cpu.get("total"),
            "mem_percent": mem.get("percent"),
            "mem_used_bytes": mem.get("used"),
            "mem_total_bytes": mem.get("total"),
            "gpu": [
                {
                    "gpu_id": g.get("gpu_id"),
                    "name": g.get("name"),
                    "util_percent": g.get("proc"),
                    "mem_percent": g.get("mem"),
                    "temperature": g.get("temperature"),
                }
                for g in (gpu or [])
            ],
        }
        return sample

    def _run(self) -> None:
        # try/except around the whole loop body per iteration: a single bad
        # poll (transient network blip) must not kill the thread, but a
        # SUSTAINED failure (Glances went down mid-run) should still
        # self-disable rather than spam a warning every interval forever.
        consecutive_failures = 0
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                sample = self._poll_once()
                consecutive_failures = 0
                self._append(sample)
            except Exception as exc:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    self._warn_once(f"poll failed ({exc}); will keep retrying quietly")
                if consecutive_failures >= 10:
                    print(
                        "[resource-monitor] WARNING: 10 consecutive poll failures; "
                        "disabling for the rest of this run.",
                        flush=True,
                    )
                    self._disabled = True
                    return
            elapsed = time.monotonic() - t0
            self._stop_event.wait(max(0.0, self.interval_s - elapsed))

    def _append(self, sample: dict) -> None:
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(sample) + "\n")
        except Exception as exc:  # never let a disk error kill training
            self._warn_once(f"failed to write {self._path}: {exc}")
            return
        if len(self._samples) < self._max_samples_in_memory:
            self._samples.append(sample)

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        """Peak/mean summary of every numeric field, for report.json.

        Reads back the JSONL artifact rather than trusting only the
        in-memory mirror, so a summary computed after ``stop()`` reflects the
        whole run even if the in-memory cap above was hit.
        """
        if self._base_url is None:
            # Never reached a Glances instance at all -- distinct from
            # "reached it, then lost it mid-run" (disabled_mid_run below).
            return {"enabled": False, "reachable": False}
        samples = self._read_all_samples()
        out: dict = {
            "enabled": True,
            "reachable": True,
            "n_samples": len(samples),
            "artifact": _ARTIFACT_NAME,
            "base_url": self._base_url,
            "interval_s": self.interval_s,
            "disabled_mid_run": self._disabled,
        }
        if not samples:
            return out

        def _stats(key):
            vals = [s[key] for s in samples if s.get(key) is not None]
            if not vals:
                return None
            return {"mean": sum(vals) / len(vals), "peak": max(vals), "min": min(vals)}

        out["cpu_percent"] = _stats("cpu_percent")
        out["mem_percent"] = _stats("mem_percent")
        out["mem_used_bytes"] = _stats("mem_used_bytes")

        gpu_util_vals: list = []
        gpu_mem_vals: list = []
        for s in samples:
            for g in s.get("gpu") or []:
                if g.get("util_percent") is not None:
                    gpu_util_vals.append(g["util_percent"])
                if g.get("mem_percent") is not None:
                    gpu_mem_vals.append(g["mem_percent"])
        if gpu_util_vals:
            out["gpu_util_percent"] = {
                "mean": sum(gpu_util_vals) / len(gpu_util_vals),
                "peak": max(gpu_util_vals),
                "min": min(gpu_util_vals),
            }
        if gpu_mem_vals:
            out["gpu_mem_percent"] = {
                "mean": sum(gpu_mem_vals) / len(gpu_mem_vals),
                "peak": max(gpu_mem_vals),
                "min": min(gpu_mem_vals),
            }
        return out

    def _read_all_samples(self) -> list:
        if not os.path.exists(self._path):
            return list(self._samples)
        samples = []
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            return list(self._samples)
        return samples


def disabled_summary() -> dict:
    """The report.json shape for a run where resource monitoring was never
    enabled at all -- distinguished from `enabled: True, disabled_mid_run:
    True` (Glances went away mid-run) and from unreachable-at-start (also
    `enabled: False` here, since the monitor never got as far as sampling)."""
    return {"enabled": False}
