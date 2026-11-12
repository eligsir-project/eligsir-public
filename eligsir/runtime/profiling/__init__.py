"""Runtime profiling primitives for Wave 3 live-mapping performance work.

See ``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``,
section 6 (PR W3-01), for the full context this package implements.

All public entry points in this package are gated behind ``PROFILING_ENABLED``
(module-level flag in :mod:`eligsir.runtime.profiling.events`, controlled by the
``SPLAT_PROFILE_RUNTIME`` environment variable, default off). When disabled, every entry
point takes a near-zero-cost early-return path: no CUDA event allocation, no dict/list
mutation beyond what a plain counter increment requires. This is a hard acceptance gate
(plan section 6.6): "profiler-off path shows less than 1% overhead".
"""

from eligsir.runtime.profiling.events import (
    PROFILING_ENABLED,
    GpuSpanRegistry,
    gpu_timed_span,
)
from eligsir.runtime.profiling.pipeline_trace import PipelineTrace
from eligsir.runtime.profiling.queue_metrics import QueueMetrics
from eligsir.runtime.profiling.execution_trace import (
    ExecutionTraceRecord,
    ExecutionTraceSink,
)

__all__ = [
    "PROFILING_ENABLED",
    "GpuSpanRegistry",
    "gpu_timed_span",
    "PipelineTrace",
    "QueueMetrics",
    "ExecutionTraceRecord",
    "ExecutionTraceSink",
]
