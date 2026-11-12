"""Per-run system-performance telemetry (default-on, PR #1783 follow-up).

``system_sampler`` samples host/process CPU, memory, and GPU counters on a
background daemon thread into ``<model_path>/system_performance.jsonl``, so
every run -- not just ResearchFlow-wrapped ones -- carries the timestamped
system-performance evidence ``docs/EVALUATION_PROTOCOL.md#system-performance-
and-input-storage`` requires.
"""

from __future__ import annotations
