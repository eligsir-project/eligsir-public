"""Process-wide cv2/BLAS thread-pool pinning for the decode path.

See docs/superpowers/specs/2026-09-08-residency-budget-design.md, "Addendum 2"
for the live measurement and root cause this addresses: on `scn5_decode8`
(ScanNet, `--causal_decode_workers 8`), GPU utilization stayed at 23.4% mean
while the process's CPU usage hit 1012% mean / 1886% peak (of 3200% possible
on 32 cores), with `supervision_views_per_s` completely unchanged (60.3 vs
61.8 at `decode_workers=2`) -- classic thread oversubscription, not more
useful work. `cv2.getNumThreads()` defaults to 32 (all cores) on this host:
OpenCV's own internal thread pool fans a *single* `cv2.resize()` call out
across every core, and each of the N `--causal_decode_workers` Python
threads (real parallelism -- PIL/cv2 release the GIL) independently makes
that call, so N decode threads times cv2's own internal fan-out is N-way
oversubscription on top of itself.

This module pins that down process-wide, once, gated behind
`--causal_thread_pinning_policy pinned` (eligsir/cli/parser.py). It
reuses the exact same five BLAS env var names
`eligsir.runtime.causal_reference_worker._pinned_blas_env` already pins
around its own `mode="process"` backprojection worker -- that helper is a
context manager scoped to one `Process.start()` call and restores the prior
env on exit; this one is a permanent, process-wide, one-time pin for the
in-process decode-worker threads (there is nothing to restore around, since
the whole point is that every decode thread for the rest of this process's
life should see threads-of-1 cv2/BLAS).
"""

from __future__ import annotations

import os

_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_already_pinned = False


def pin_decode_threads() -> dict:
    """Set cv2.setNumThreads(1) and the five BLAS env vars to "1", process-wide.

    Idempotent: safe to call more than once. Every call re-asserts the same
    pinned state (setting an already-1 value to 1 again changes nothing), so
    a second call is functionally a no-op -- it just also reports
    ``already_pinned=True`` so the caller's log line can say so.

    cv2 is imported lazily inside this function: some test/CI environments
    do not have it installed, and that must degrade to a no-op for the cv2
    piece specifically, not raise ImportError out of this call -- the BLAS
    env vars still get pinned either way.

    Returns:
        Metadata dict describing what was changed, for the caller's log line:
        ``cv2_pinned`` (bool), ``blas_vars`` (the five names pinned),
        ``blas_threads`` ("1"), ``already_pinned`` (whether an earlier call
        in this process already did this).
    """
    global _already_pinned

    cv2_pinned = False
    try:
        import cv2  # local import: some test/CI environments lack cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        cv2.setNumThreads(1)
        cv2_pinned = True

    for var in _ENV_VARS:
        os.environ[var] = "1"

    metadata = {
        "cv2_pinned": cv2_pinned,
        "blas_vars": list(_ENV_VARS),
        "blas_threads": "1",
        "already_pinned": _already_pinned,
    }
    _already_pinned = True
    return metadata
