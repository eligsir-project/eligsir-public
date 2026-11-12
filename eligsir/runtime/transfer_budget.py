"""Transfer-budget and tail-latency qualification contracts (issue #279).

The packed source-frame and CPU-origin insertion boundaries each permit at
most ONE physical H2D submission -- ground truth verified against
`eligsir/runtime/packed_staging.py`'s `pack_numpy_fields_to_device`, which
lays every field out into a single pinned byte slab and issues exactly one
`transfer.h2d()` call regardless of how many logical fields (float slab, int
slab, ...) are packed into it. `eligsir/mapping/integration/
insertion_committer.py`'s CPU-origin path is this exact function. GPU-native
insertion and resident optimizer batches permit none. Hardware qualification
consumes the same counters plus p50/p95/p99 tail latency, backlog and VRAM
samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class TransferBudget:
    """Maximum transfer operations/bytes allowed at one semantic boundary."""

    max_h2d_ops: int
    max_d2h_ops: int
    max_h2d_bytes: int | None = None
    max_d2h_bytes: int | None = None

    def evaluate(self, delta: Mapping[str, int]) -> tuple[bool, tuple[str, ...]]:
        """Return pass/fail plus stable human-readable budget violations."""

        failures = []
        for key, limit in (
            ("h2d_ops", self.max_h2d_ops),
            ("d2h_ops", self.max_d2h_ops),
            ("h2d_bytes", self.max_h2d_bytes),
            ("d2h_bytes", self.max_d2h_bytes),
        ):
            if limit is not None and int(delta.get(key, 0)) > int(limit):
                failures.append(f"{key}={int(delta.get(key, 0))}>{int(limit)}")
        return not failures, tuple(failures)


LIVE_SOURCE_FRAME_BUDGET = TransferBudget(max_h2d_ops=1, max_d2h_ops=0)
GPU_NATIVE_INSERTION_BUDGET = TransferBudget(max_h2d_ops=0, max_d2h_ops=0)
CPU_ORIGIN_INSERTION_BUDGET = TransferBudget(max_h2d_ops=1, max_d2h_ops=0)
RESIDENT_OPTIMIZER_BATCH_BUDGET = TransferBudget(max_h2d_ops=0, max_d2h_ops=0)


@dataclass
class LatencySamples:
    """Bounded-report-friendly latency sample accumulator."""

    values_ms: list[float] = field(default_factory=list)

    def add(self, value_ms: float) -> None:
        """Add a single latency sample in milliseconds."""
        self.values_ms.append(float(value_ms))

    def extend(self, values_ms: Iterable[float]) -> None:
        """Add multiple latency samples from an iterable."""
        self.values_ms.extend(float(v) for v in values_ms)

    def summary(self) -> dict[str, float | int | None]:
        """Return count and p50/p95/p99/max in milliseconds."""

        if not self.values_ms:
            return {
                "count": 0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_ms": None,
            }
        arr = np.asarray(self.values_ms, dtype=np.float64)
        return {
            "count": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }


@dataclass
class RuntimeQualification:
    """One hardware run's transfer/throughput/tail-latency evidence."""

    backend: str
    device_name: str
    transfer_counts: dict[str, int]
    frame_stall_ms: LatencySamples = field(default_factory=LatencySamples)
    optimizer_step_ms: LatencySamples = field(default_factory=LatencySamples)
    backlog_samples: list[int] = field(default_factory=list)
    vram_bytes_samples: list[int] = field(default_factory=list)
    optimizer_steps: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        """Return JSON-serializable qualification evidence."""

        throughput = (
            float(self.optimizer_steps) / float(self.elapsed_s)
            if self.elapsed_s > 0
            else None
        )
        backlog = np.asarray(self.backlog_samples, dtype=np.float64) if self.backlog_samples else None
        vram = np.asarray(self.vram_bytes_samples, dtype=np.float64) if self.vram_bytes_samples else None
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "transfer_counts": dict(self.transfer_counts),
            "frame_stall": self.frame_stall_ms.summary(),
            "optimizer_step": self.optimizer_step_ms.summary(),
            "optimizer_steps": int(self.optimizer_steps),
            "elapsed_s": float(self.elapsed_s),
            "optimizer_steps_per_s": throughput,
            "backlog": {
                "count": 0 if backlog is None else int(backlog.size),
                "p50": None if backlog is None else float(np.percentile(backlog, 50)),
                "p95": None if backlog is None else float(np.percentile(backlog, 95)),
                "p99": None if backlog is None else float(np.percentile(backlog, 99)),
                "max": None if backlog is None else int(backlog.max()),
            },
            "vram_bytes": {
                "count": 0 if vram is None else int(vram.size),
                "p50": None if vram is None else int(np.percentile(vram, 50)),
                "p95": None if vram is None else int(np.percentile(vram, 95)),
                "p99": None if vram is None else int(np.percentile(vram, 99)),
                "max": None if vram is None else int(vram.max()),
            },
        }


def counter_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    """Subtract two transfer-counter snapshots by key."""

    keys = set(before) | set(after)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


__all__ = [
    "CPU_ORIGIN_INSERTION_BUDGET",
    "GPU_NATIVE_INSERTION_BUDGET",
    "LIVE_SOURCE_FRAME_BUDGET",
    "LatencySamples",
    "RESIDENT_OPTIMIZER_BATCH_BUDGET",
    "RuntimeQualification",
    "TransferBudget",
    "counter_delta",
]
