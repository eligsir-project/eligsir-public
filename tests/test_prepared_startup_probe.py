"""Contract tests for the diagnostic-only prepared-startup probe wrapper."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from eligsir.runtime.application import ComposedMappingApplication

from tests.diagnostics.prepared_startup_probe import observe_startup


class _FakeApplication:
    def __init__(self) -> None:
        self._prepared_retry_pending = False
        self._options = SimpleNamespace(prepared_progress_waiter=lambda **_: True)
        self.seen = []
        self.tick_result = SimpleNamespace(
            state=SimpleNamespace(value="deferred"), retryable=True
        )

    def tick(self, *args, **kwargs):
        self.seen.append((args, kwargs))
        self._prepared_retry_pending = True
        return self.tick_result

    def wait_for_prepared_progress(self, *args, **kwargs):
        self.seen.append((args, kwargs))
        return True


class _FakeWorker:
    def __init__(self) -> None:
        self.seen = []

    def wait_for_progress(self, *args, **kwargs):
        self.seen.append((args, kwargs))
        return False


class _FakeAdapter:
    def __init__(self) -> None:
        self.seen = []

    def packet_from_record(self, *args, **kwargs):
        self.seen.append((args, kwargs))
        return object()


def _install_probe_worker_fake(monkeypatch):
    """Keep launcher tests independent of optional Torch runtime imports."""
    prep = ModuleType("eligsir.runtime.prepared_source_prep_worker")
    prep.__file__ = __file__
    fake_worker = type("PreparedSourcePrepWorker", (_FakeWorker,), {})
    fake_worker.__module__ = prep.__name__
    prep.PreparedSourcePrepWorker = fake_worker
    monkeypatch.setitem(
        sys.modules, "eligsir.runtime.prepared_source_prep_worker", prep
    )


def test_probe_preserves_identity_arguments_and_classifies_calls():
    app = _FakeApplication()
    worker = _FakeWorker()
    adapter = _FakeAdapter()
    emitted = []
    original_tick = _FakeApplication.tick

    with observe_startup(_FakeApplication, _FakeWorker, _FakeAdapter, emitted.append) as snapshot:
        tick_args = ("tick-arg",)
        tick_kwargs = {"sample": 7}
        tick_result = app.tick(*tick_args, **tick_kwargs)
        assert tick_result is app.tick_result
        assert tick_result.retryable is True
        assert app.seen == [(tick_args, tick_kwargs)]

        wait_result = app.wait_for_prepared_progress(timeout_s=0.25)
        worker_result = worker.wait_for_progress(timeout_s=0.5)
        record = object()
        packet = adapter.packet_from_record(record, mode="probe")
        assert wait_result is True
        assert worker_result is False
        assert packet is not None
        assert app.seen[1] == ((), {"timeout_s": 0.25})
        assert worker.seen == [((), {"timeout_s": 0.5})]
        assert adapter.seen == [((record,), {"mode": "probe"})]

        counts = snapshot()["counters"]
        assert counts["tick_calls"] == 1
        assert counts["application_wait_calls"] == 1
        assert counts["worker_wait_calls"] == 1
        assert counts["decode_calls"] == 1
        assert counts["application_wait_progress"] == 1
        assert counts["worker_wait_progress"] == 0
        assert snapshot()["outcomes"]

    assert _FakeApplication.tick is original_tick


def test_probe_restores_every_method_when_wrapped_call_raises():
    class RaisingApplication(_FakeApplication):
        def tick(self, *args, **kwargs):
            raise RuntimeError("synthetic tick failure")

    app = RaisingApplication()
    original_tick = RaisingApplication.tick
    original_wait = RaisingApplication.wait_for_prepared_progress
    original_worker_wait = _FakeWorker.wait_for_progress
    original_decode = _FakeAdapter.packet_from_record

    with pytest.raises(RuntimeError, match="synthetic tick failure"):
        with observe_startup(
            RaisingApplication, _FakeWorker, _FakeAdapter, lambda _row: None
        ):
            app.tick()

    assert RaisingApplication.tick is original_tick
    assert RaisingApplication.wait_for_prepared_progress is original_wait
    assert _FakeWorker.wait_for_progress is original_worker_wait
    assert _FakeAdapter.packet_from_record is original_decode


def test_probe_tick_snapshot_uses_composed_application_private_state(monkeypatch):
    """The probe's private fields are the fields on the real composed type."""
    app = object.__new__(ComposedMappingApplication)
    app._prepared_retry_pending = False
    app._options = SimpleNamespace(prepared_progress_waiter=lambda **_: True)

    def tick(self, *, marker):
        assert self._options.prepared_progress_waiter is not None
        self._prepared_retry_pending = True
        return SimpleNamespace(
            state=SimpleNamespace(value="bootstrap_pending"), retryable=True
        )

    monkeypatch.setattr(ComposedMappingApplication, "tick", tick)
    emitted = []
    with observe_startup(
        ComposedMappingApplication, _FakeWorker, _FakeAdapter, emitted.append
    ) as snapshot:
        result = app.tick(marker="unchanged")
        assert result.retryable is True
        outcome = snapshot()["outcomes"][0]
        assert outcome["armed"] is True
        assert outcome["waiter_bound"] is True
        assert outcome["state"] == "bootstrap_pending"


def test_probe_main_restores_process_globals_on_system_exit(monkeypatch, tmp_path):
    from tests.diagnostics import prepared_startup_probe as probe
    _install_probe_worker_fake(monkeypatch)

    script = tmp_path / "fake_train.py"
    script.write_text("raise SystemExit(17)\n")
    model = tmp_path / "model"
    model.mkdir()
    original_argv = ["pytest", "--sentinel"]
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "argv", original_argv.copy())
    monkeypatch.setattr(probe.runpy, "run_path", lambda *a, **k: (_ for _ in ()).throw(SystemExit(17)))

    with pytest.raises(SystemExit) as error:
        sys.argv = ["probe", str(script), "-m", str(model)]
        probe.main()

    assert error.value.code == 17
    assert sys.argv == ["probe", str(script), "-m", str(model)]
    # The test intentionally sets argv immediately before main; main must
    # restore that invocation state, while the surrounding pytest path stays
    # byte-for-byte unchanged.
    assert sys.path == original_path


def test_probe_main_forwards_script_argv_and_restores_after_success(monkeypatch, tmp_path):
    from tests.diagnostics import prepared_startup_probe as probe
    _install_probe_worker_fake(monkeypatch)

    script = tmp_path / "fake_train.py"
    script.write_text("pass\n")
    model = tmp_path / "model"
    model.mkdir()
    invocation = ["probe", str(script), "--model_path", str(model), "--flag", "value"]
    forwarded = []
    original_path = list(sys.path)
    original_tick = ComposedMappingApplication.tick
    original_application_wait = ComposedMappingApplication.wait_for_prepared_progress

    def fake_run_path(path, *, run_name):
        forwarded.append((path, run_name, list(sys.argv)))
        return {}

    monkeypatch.setattr(probe.runpy, "run_path", fake_run_path)
    monkeypatch.setattr(sys, "argv", invocation.copy())
    probe.main()

    assert forwarded == [(str(script), "__main__", [str(script), "--model_path", str(model), "--flag", "value"])]
    assert sys.argv == invocation
    assert sys.path == original_path
    report = json.loads(
        (model / "qualification_receipts" / "prepared_startup_probe.json").read_text()
    )
    for module_name in (
        "eligsir.runtime.application",
        "eligsir.runtime.causal_replay",
        "eligsir.runtime.prepared_source_prep_worker",
    ):
        module_path = Path(report["modules"][module_name]["path"])
        assert module_path == Path(sys.modules[module_name].__file__)
        assert report["modules"][module_name]["file_sha256"] == hashlib.sha256(
            module_path.read_bytes()
        ).hexdigest()
    assert ComposedMappingApplication.tick is original_tick
    assert (
        ComposedMappingApplication.wait_for_prepared_progress
        is original_application_wait
    )


def test_probe_clean_report_write_error_is_reported(monkeypatch, tmp_path):
    from tests.diagnostics import prepared_startup_probe as probe
    _install_probe_worker_fake(monkeypatch)

    script = tmp_path / "fake_train.py"
    script.write_text("pass\n")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(probe.runpy, "run_path", lambda *a, **k: {})
    monkeypatch.setattr(
        probe.Path,
        "write_text",
        lambda *a, **k: (_ for _ in ()).throw(OSError("report filesystem failure")),
    )
    monkeypatch.setattr(sys, "argv", ["probe", str(script), "-m", str(model)])

    with pytest.raises(OSError, match="report filesystem failure"):
        probe.main()


def test_probe_report_write_error_does_not_mask_training_exception(monkeypatch, tmp_path):
    from tests.diagnostics import prepared_startup_probe as probe
    _install_probe_worker_fake(monkeypatch)

    script = tmp_path / "fake_train.py"
    script.write_text("raise RuntimeError('training failed')\n")
    model = tmp_path / "model"
    model.mkdir()
    monkeypatch.setattr(
        probe.runpy,
        "run_path",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("training failed")),
    )
    monkeypatch.setattr(
        probe.Path,
        "write_text",
        lambda *a, **k: (_ for _ in ()).throw(OSError("report filesystem failure")),
    )
    monkeypatch.setattr(sys, "argv", ["probe", str(script), "-m", str(model)])

    with pytest.raises(RuntimeError, match="training failed"):
        probe.main()


def test_probe_emit_error_does_not_mask_original_tick_return(monkeypatch):
    app = _FakeApplication()
    result = app.tick_result
    class Clock:
        def __init__(self):
            self.values = iter((0.0, 1.0, 6.0))
            self.last = 6.0

        def perf_counter(self):
            self.last = next(self.values, self.last)
            return self.last

    clock = Clock()
    from tests.diagnostics import prepared_startup_probe as probe
    monkeypatch.setattr(probe, "time", SimpleNamespace(perf_counter=clock.perf_counter))

    def failing_emit(_row):
        raise OSError("diagnostic sink unavailable")

    with observe_startup(_FakeApplication, _FakeWorker, _FakeAdapter, failing_emit):
        assert app.tick() is result
