"""Tests for --causal_thread_pinning_policy (cv2/BLAS thread-pool pinning).

See docs/superpowers/specs/2026-09-08-residency-budget-design.md, "Addendum 2"
and .superpowers/sdd/2026-09-08-residency-budget/task-4-brief.md for the live
measurement (scn5_decode8: GPU 23.4%, CPU process mean 1012%/peak 1886% of
3200%, views/s unchanged 60.3 vs 61.8) and root cause (cv2's own internal
thread pool defaults to all 32 cores per resize call, fighting with N
`--causal_decode_workers` Python threads that each independently call it).
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from eligsir.runtime.thread_pinning import _ENV_VARS, pin_decode_threads

_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@pytest.fixture
def saved_environ():
    """Save/restore os.environ exactly, mirroring _pinned_blas_env's own
    save/restore pattern (del if was absent, assign if was set), so this
    test never leaks pinned env vars into other tests."""
    saved = {var: os.environ.get(var) for var in _BLAS_ENV_VARS}
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


class TestPinDecodeThreadsEnvVars:
    def test_reuses_exact_five_pinned_blas_env_var_names(self):
        """The env vars pinned here must be the exact same five names
        `_pinned_blas_env` (causal_reference_worker.py) already pins --
        not a reinvented list."""
        assert tuple(_ENV_VARS) == _BLAS_ENV_VARS

    def test_sets_all_five_blas_env_vars_to_one(self, saved_environ):
        pin_decode_threads()
        for var in _BLAS_ENV_VARS:
            assert os.environ[var] == "1", f"{var} was not pinned to '1'"

    def test_cv2_thread_count_set_to_one(self, saved_environ):
        cv2 = pytest.importorskip("cv2")
        cv2.setNumThreads(32)  # force away from 1 so the assertion is meaningful
        pin_decode_threads()
        assert cv2.getNumThreads() == 1

    def test_metadata_reports_what_was_changed(self, saved_environ):
        metadata = pin_decode_threads()
        assert metadata["blas_threads"] == "1"
        assert set(metadata["blas_vars"]) == set(_BLAS_ENV_VARS)


class TestPinDecodeThreadsIdempotent:
    def test_calling_twice_does_not_raise_and_both_report_success(self, saved_environ, monkeypatch):
        # Module-level guard is process-global; reset it so this test's
        # "first call" is genuinely first regardless of test execution order.
        monkeypatch.setattr(
            "eligsir.runtime.thread_pinning._already_pinned", False
        )
        first = pin_decode_threads()
        second = pin_decode_threads()
        assert first is not None
        assert second is not None
        # Second call sees an already-pinned process, first call does not.
        assert first["already_pinned"] is False
        assert second["already_pinned"] is True
        # Both calls leave the same pinned state behind.
        for var in _BLAS_ENV_VARS:
            assert os.environ[var] == "1"


class TestPinDecodeThreadsWithoutCv2:
    def test_missing_cv2_degrades_to_no_op_for_that_piece(self, saved_environ, monkeypatch):
        """cv2 absence must not raise ImportError out of pin_decode_threads --
        only the cv2-specific piece degrades to a no-op; env vars still get
        pinned."""
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("no cv2 in this test env")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        metadata = pin_decode_threads()
        assert metadata["cv2_pinned"] is False
        for var in _BLAS_ENV_VARS:
            assert os.environ[var] == "1"


class TestCausalEntryThreadPinningWiring:
    """Wiring: run_causal_mapping calls pin_decode_threads() exactly once,
    as early as possible, gated by --causal_thread_pinning_policy, before
    the frame-source/producer construction point
    (resolve_causal_mapping_composition). A sentinel exception raised from a
    monkeypatched resolve_causal_mapping_composition proves ordering: if
    pin_decode_threads() had not already run by the time control reaches
    that call, the spy list below would be empty when the exception
    propagates.
    """

    class _StopAtComposition(Exception):
        pass

    def _run_up_to_composition(self, monkeypatch, *, policy):
        import eligsir.runtime.causal_entry as causal_entry
        import eligsir.utils.execution_semantics as execution_semantics

        calls: list[str] = []

        def _spy_pin_decode_threads():
            calls.append("pin_decode_threads")
            return {"cv2_pinned": False, "blas_vars": [], "blas_threads": "1", "already_pinned": False}

        monkeypatch.setattr(
            "eligsir.runtime.thread_pinning.pin_decode_threads", _spy_pin_decode_threads
        )

        def _stop(**_kwargs):
            calls.append("resolve_causal_mapping_composition")
            raise self._StopAtComposition()

        monkeypatch.setattr(causal_entry, "resolve_causal_mapping_composition", _stop)
        monkeypatch.setattr(
            execution_semantics,
            "resolve_execution_semantics",
            lambda _args: {"release_policy": execution_semantics.RELEASE_DATASET_FPS},
        )

        args = SimpleNamespace(
            streaming_input_fps_cap=30.0,
            system_telemetry=False,
            system_telemetry_interval_s=1.0,
            causal_thread_pinning_policy=policy,
        )
        dataset = SimpleNamespace(source_path="/data/fr3", model_path="/tmp/does-not-matter")

        with pytest.raises(self._StopAtComposition):
            causal_entry.run_causal_mapping(
                dataset=dataset, opt=SimpleNamespace(), pipe=SimpleNamespace(), args=args
            )

        return calls

    def test_pinned_policy_calls_pin_decode_threads_once_before_composition(self, monkeypatch):
        calls = self._run_up_to_composition(monkeypatch, policy="pinned")
        assert calls == ["pin_decode_threads", "resolve_causal_mapping_composition"]

    def test_unbound_policy_never_calls_pin_decode_threads(self, monkeypatch):
        calls = self._run_up_to_composition(monkeypatch, policy="unbound")
        assert calls == ["resolve_causal_mapping_composition"]
