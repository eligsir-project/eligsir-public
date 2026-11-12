"""Regression coverage for issue #1389: 'viser' missing from the training
Docker image used to crash any real run passing ``--viewer viser``, and the
sidecar failure mode itself was a silent 15s hang with no actionable reason.

Deliberately does NOT require ``viser`` to be installed -- these tests only
exercise (a) the sidecar's own missing-import handling (a plain function
call, no subprocess), and (b) the registry-level graceful fallback (with the
real sidecar constructor monkeypatched out), mirroring
``test_viser_server_gaussian_covariance.py``'s "importable/testable in any
environment" approach.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue_mod

import pytest

from eligsir.viewers.sinks.viser_sink import ViserUnavailable


def _dying_server(cmd_queues, port, host, ready_flag, close_flag, error_queue):
    """Module-level (picklable, for the 'spawn' mp context) stand-in for a
    sidecar that fails almost immediately -- e.g. a missing dependency."""
    error_queue.put("'viser' package is not importable in the sidecar process")
    # Simulate immediate process exit without ever setting ready_flag.


def test_run_viser_server_reports_missing_package_fast(monkeypatch):
    """A ModuleNotFoundError for 'viser' inside the sidecar process must be
    reported on ``error_queue`` (not just silently kill the process), and
    ``ready_flag`` must never be set."""
    import builtins

    from eligsir.viewers.sinks import viser_server

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "viser":
            raise ImportError("No module named 'viser'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    close = ctx.Event()
    error_queue = ctx.Queue()

    viser_server.run_viser_server({}, 0, "127.0.0.1", ready, close, error_queue)

    assert not ready.is_set()
    detail = error_queue.get(timeout=1.0)
    assert "viser" in detail.lower()
    assert "not importable" in detail.lower() or "not installed" in detail.lower()


def test_sink_init_fails_fast_with_missing_package_reason(monkeypatch):
    """``ViserPreviewSink.__init__`` must surface the sidecar's own reported
    reason quickly, not the generic 15s-timeout message, when the sidecar
    process dies almost immediately (e.g. missing dependency)."""
    from eligsir.viewers.sinks import viser_sink as viser_sink_mod

    monkeypatch.setattr(
        "eligsir.viewers.sinks.viser_server.run_viser_server", _dying_server
    )

    import time

    start = time.monotonic()
    with pytest.raises(ViserUnavailable) as excinfo:
        viser_sink_mod.ViserPreviewSink(port=0, host="127.0.0.1", start_timeout_s=15.0)
    elapsed = time.monotonic() - start

    assert "not importable" in str(excinfo.value)
    # Must fail fast (well under the 15s timeout), not only after it expires.
    assert elapsed < 5.0


def test_registry_make_viewer_falls_back_when_viser_sidecar_unavailable(monkeypatch):
    """A sidecar that fails to start must degrade to a working (viewer-less)
    ``CanonicalViewerAdapter`` rather than raising out of ``make_viewer`` and
    killing the whole training run (zero-intrusion, issue #1389/#1200)."""
    from eligsir.viewers import registry

    def _raising_factory(**kwargs):
        raise ViserUnavailable("simulated: 'viser' not installed")

    monkeypatch.setitem(registry._SINK_FACTORIES, "viser", _raising_factory)

    viewer = registry.make_viewer("viser", port=0, host="127.0.0.1")

    # No exception propagated; the adapter was still constructed, just with
    # no working sinks.
    assert viewer is not None
    assert tuple(viewer.canonical_viewer._sinks) == ()
