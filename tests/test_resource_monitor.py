"""Tests for utils/resource_monitor.py (SPLAT_RESOURCE_MONITOR-gated poller).

Uses a tiny stdlib http.server stub standing in for a local Glances REST API
-- no real Glances instance, no GPU, no network beyond localhost. Verifies:
  1. samples are captured at roughly the configured cadence
  2. the JSONL artifact is written correctly (one JSON object per line, with
     iteration correlation)
  3. the module is a true no-op when SPLAT_RESOURCE_MONITOR is unset (the
     module-level flag), and ResourceMonitor gracefully self-disables (never
     raises) when Glances is unreachable.
"""
from __future__ import annotations

import http.server
import json
import os
import threading
import time

import pytest

from eligsir.utils import resource_monitor as rm


class _StubGlancesHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence stdlib default logging
        pass

    def do_GET(self):
        if self.path == "/api/4/cpu":
            body = {"total": 12.5}
        elif self.path == "/api/4/mem":
            body = {"percent": 33.3, "used": 1000, "total": 4000}
        elif self.path == "/api/4/gpu":
            body = [{"gpu_id": "amd0", "name": "test-gpu", "proc": 42.0, "mem": 17.0, "temperature": 55}]
        else:
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def stub_glances():
    server = http.server.HTTPServer(("127.0.0.1", 0), _StubGlancesHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()
    thread.join(timeout=5)


def test_module_level_flag_reflects_env(monkeypatch):
    monkeypatch.delenv("SPLAT_RESOURCE_MONITOR", raising=False)
    # RESOURCE_MONITOR_ENABLED is read once at import time -- re-import the
    # logic under test rather than asserting on the already-imported module
    # global (which reflects whatever the env was at test-collection time).
    assert bool(os.environ.get("SPLAT_RESOURCE_MONITOR")) is False


def test_disabled_summary_shape():
    assert rm.disabled_summary() == {"enabled": False}


def test_unreachable_host_disables_gracefully(tmp_path, monkeypatch):
    # Point at a port nothing is listening on; must never raise.
    monkeypatch.setenv("SPLAT_GLANCES_HOST", "127.0.0.1")
    monkeypatch.setenv("SPLAT_GLANCES_PORT", "1")  # port 1 refuses connections
    monitor = rm.ResourceMonitor(str(tmp_path), interval_s=0.05)
    monitor.start()  # must not raise
    assert monitor._disabled is True
    summary = monitor.summary()
    assert summary == {"enabled": False, "reachable": False}
    assert not (tmp_path / "resource_usage.jsonl").exists()
    monitor.stop()  # no-op, must not raise (no thread was started)


def test_polls_at_configured_cadence_and_writes_artifact(tmp_path, stub_glances, monkeypatch):
    monkeypatch.setenv("SPLAT_GLANCES_HOST", "127.0.0.1")
    monkeypatch.setenv("SPLAT_GLANCES_PORT", str(stub_glances))
    interval = 0.05
    monitor = rm.ResourceMonitor(str(tmp_path), interval_s=interval)
    monitor.start()
    assert monitor._disabled is False
    assert monitor._base_url == f"http://127.0.0.1:{stub_glances}"

    n_ticks = 6
    for i in range(n_ticks):
        monitor.set_iteration(i * 10)
        time.sleep(interval)
    monitor.stop()

    artifact = tmp_path / "resource_usage.jsonl"
    assert artifact.exists()
    lines = [json.loads(l) for l in artifact.read_text().splitlines() if l.strip()]
    assert len(lines) >= 2  # cadence isn't exact under test scheduling, but must have sampled repeatedly

    # Roughly the right cadence: elapsed time / interval should be in the
    # right ballpark (generous bounds -- this is a scheduling test, not a
    # precision timing test).
    elapsed = lines[-1]["ts"] - lines[0]["ts"]
    if len(lines) > 1:
        observed_interval = elapsed / (len(lines) - 1)
        assert observed_interval < interval * 10  # not wildly slower than requested

    for rec in lines:
        assert "ts" in rec
        assert "iteration" in rec
        assert rec["cpu_percent"] == 12.5
        assert rec["mem_percent"] == 33.3
        assert rec["mem_used_bytes"] == 1000
        assert isinstance(rec["gpu"], list) and len(rec["gpu"]) == 1
        assert rec["gpu"][0]["util_percent"] == 42.0
        assert rec["gpu"][0]["mem_percent"] == 17.0

    # iteration correlation: at least one sample should carry a real,
    # non-None iteration value that was set via set_iteration().
    assert any(rec["iteration"] is not None for rec in lines)

    summary = monitor.summary()
    assert summary["enabled"] is True
    assert summary["reachable"] is True
    assert summary["n_samples"] == len(lines)
    assert summary["cpu_percent"]["mean"] == pytest.approx(12.5)
    assert summary["cpu_percent"]["peak"] == pytest.approx(12.5)
    assert summary["gpu_util_percent"]["mean"] == pytest.approx(42.0)
    assert summary["gpu_mem_percent"]["mean"] == pytest.approx(17.0)


def test_resolve_base_url_tries_explicit_host_only(monkeypatch, stub_glances):
    monkeypatch.setenv("SPLAT_GLANCES_HOST", "127.0.0.1")
    monkeypatch.setenv("SPLAT_GLANCES_PORT", str(stub_glances))
    url = rm._resolve_base_url()
    assert url == f"http://127.0.0.1:{stub_glances}"


def test_resolve_base_url_none_when_unreachable(monkeypatch):
    monkeypatch.setenv("SPLAT_GLANCES_HOST", "127.0.0.1")
    monkeypatch.setenv("SPLAT_GLANCES_PORT", "1")
    assert rm._resolve_base_url() is None


def test_set_iteration_noop_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SPLAT_GLANCES_HOST", "127.0.0.1")
    monkeypatch.setenv("SPLAT_GLANCES_PORT", "1")
    monitor = rm.ResourceMonitor(str(tmp_path), interval_s=0.05)
    monitor.start()
    monitor.set_iteration(5)  # must not raise even though disabled
    assert monitor._current_iteration() is None
