"""Best-effort GPU process co-residency sampling (issue #295 fix item 4).

A training run that shares its GPU with another process (e.g. the
duplicate-launch bug root-caused in ablator's `runner.py` --
`_ensure_container_name`/`reconcile_stale_running`'s startup grace window)
is silently perf- and memory-contaminated: any `it/s`, `timing_summary`, or
throughput-derived claim from that run is untrustworthy. Nothing previously
recorded whether this happened -- confirming it required manually cross-
referencing `queue.jsonl` claim timestamps against a crash log (see
GitHub issue #295's evidence). This module takes a cheap, non-fatal
snapshot of how many processes are resident on the GPU (nvidia-smi or
rocm-smi, whichever is present) at run start and again at report time, so a
contaminated run is self-identifying from its own `report.json` instead of
requiring forensics after the fact.

Both NVIDIA and AMD machines exist in this fleet (see CLAUDE.md); this
degrades gracefully (``tool=None``, ``process_count=None``) when neither
CLI is present, or the probe fails for any reason -- sampling must never be
able to fail or measurably slow down a training run.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
from typing import Optional

_START_SNAPSHOT_FILENAME = "gpu_coresidency_start.json"


def _run(cmd: list, timeout: float = 10.0) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return None
        return r.stdout
    except Exception:
        return None


def _sample_nvidia() -> Optional[dict]:
    out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits"])
    if out is None:
        return None
    processes = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        entry = {"pid": parts[0]}
        if len(parts) > 1:
            try:
                entry["used_memory_mib"] = float(parts[1])
            except ValueError:
                pass
        processes.append(entry)
    return {"tool": "nvidia-smi", "process_count": len(processes), "processes": processes}


def _sample_rocm() -> Optional[dict]:
    out = _run(["rocm-smi", "--showpids"])
    if out is None:
        return None
    processes = []
    for line in out.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        processes.append({"pid": parts[0]})
    if not processes and "PID" not in out and "PROCESS" not in out.upper():
        # Command ran but produced nothing recognizable -- report the tool
        # ran, with an unknown count, rather than silently claiming 0 (which
        # would read as "confirmed no co-residency" when it's really
        # "couldn't tell").
        return {"tool": "rocm-smi", "process_count": None, "processes": [],
                "note": "ran but output not parseable"}
    return {"tool": "rocm-smi", "process_count": len(processes), "processes": processes}


def sample_gpu_coresidency() -> dict:
    """One best-effort snapshot of GPU process residency. Never raises."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    for sampler in (_sample_nvidia, _sample_rocm):
        try:
            result = sampler()
        except Exception:
            result = None
        if result is not None:
            result["timestamp"] = ts
            return result
    return {"tool": None, "process_count": None, "processes": [], "timestamp": ts,
            "note": "no GPU monitoring tool (nvidia-smi/rocm-smi) found"}


def record_start_snapshot(model_path: str) -> Optional[dict]:
    """Sample once at run start and durably persist it under `model_path`,
    so it can be read back at report time regardless of process restarts
    (e.g. a checkpoint-resumed composite run). Best-effort: any failure to
    sample or write is swallowed and logged -- this must never be able to
    break run startup.
    """
    try:
        snapshot = sample_gpu_coresidency()
        os.makedirs(model_path, exist_ok=True)
        path = os.path.join(model_path, _START_SNAPSHOT_FILENAME)
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)
        return snapshot
    except Exception as e:
        print(f"[gpu-coresidency] failed to record start snapshot: {e!r}", flush=True)
        return None


def load_start_snapshot(model_path: str) -> Optional[dict]:
    path = os.path.join(model_path, _START_SNAPSHOT_FILENAME)
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def build_report_block(model_path: str) -> dict:
    """Assemble the ``report.json["gpu_coresidency"]`` block: the start
    snapshot (read back from disk, written by `record_start_snapshot` at
    run startup) plus a fresh end-of-run snapshot, plus a coarse
    ``contaminated`` verdict.

    ``contaminated`` is True only when BOTH samples independently saw more
    than one process resident on the GPU -- a count > 1 at only one end is
    still recorded (nothing is thrown away) but not flagged, since it could
    be this run's own short-lived report/eval subprocess overlapping
    briefly rather than a sustained co-resident trainer. ``None`` when
    either sample lacks a usable process_count (no GPU tool available, or
    the start snapshot was never written -- e.g. a run predating this
    feature, or an entirely non-container/CPU-only smoke test).
    """
    start = load_start_snapshot(model_path)
    end = sample_gpu_coresidency()
    contaminated = None
    if (
        start is not None
        and start.get("process_count") is not None
        and end.get("process_count") is not None
    ):
        contaminated = start["process_count"] > 1 and end["process_count"] > 1
    return {"start": start, "end": end, "contaminated": contaminated}
