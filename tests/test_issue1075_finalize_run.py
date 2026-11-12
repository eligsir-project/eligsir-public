"""Issue #1075 P1: `eligsir.runtime.finalize.finalize_run` must reproduce,
call-for-call and in order, the hand-copied shutdown tail it replaces at the
four training call sites.

Each test records every observable interaction (signal.signal, save_worker
shutdown, staging finalization) into a single ordered list and asserts the
exact sequence the original inline code produced.
"""

import signal
import sys

import pytest

from _source_inspection import function_source
from eligsir.runtime import finalize as finalize_mod
from eligsir.runtime.finalize import finalize_run


class _Recorder:
    def __init__(self):
        self.events = []

    def signal(self, signum, handler):
        self.events.append(("signal", signum, handler))

    def make_worker(self):
        rec = self

        class _Worker:
            def shutdown(self):
                rec.events.append(("shutdown",))

        return _Worker()


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    monkeypatch.setattr(finalize_mod._signal, "signal", r.signal)
    return r


@pytest.fixture
def fake_finalize(monkeypatch, rec):
    """Install a stub `train._finalize_output_staging` for the lazy import."""
    mod = type(sys)("train")

    def _fin(args, timeout=600.0):
        rec.events.append(("finalize", args))

    mod._finalize_output_staging = _fin
    monkeypatch.setitem(sys.modules, "train", mod)
    return mod


def _h(name):
    """A distinct, identifiable stand-in for an original handler object."""
    return f"<handler:{name}>"


# --------------------------------------------------------------------------
# Call site A: train.py offline path (train.py:1854-1855)
#   _signal.signal(_signal.SIGTERM, _orig_sigterm)
#   save_worker.shutdown()
#   ... (report/run_summary/manifest) ...
#   _finalize_output_staging(args)   <- stays at the call site
# --------------------------------------------------------------------------
def test_offline_path_order(rec):
    finalize_run(
        rec.make_worker(),
        shutdown_first=False,
        finalize_staging=False,
        orig_sigterm=_h("term"),
    )
    assert rec.events == [
        ("signal", signal.SIGTERM, _h("term")),
        ("shutdown",),
    ]


# --------------------------------------------------------------------------
# Call site D: train_streaming.py normal path (train_streaming.py:12461-12463)
#   _signal.signal(_signal.SIGINT, _orig_sigint)
#   _signal.signal(_signal.SIGTERM, _orig_sigterm)
#   save_worker.shutdown()
#   ... (multires stats/run_state/run_summary/manifest) ...
#   _finalize_output_staging(args)   <- stays at the call site
# --------------------------------------------------------------------------
def test_streaming_normal_path_order(rec):
    finalize_run(
        rec.make_worker(),
        shutdown_first=False,
        finalize_staging=False,
        orig_sigint=_h("int"),
        orig_sigterm=_h("term"),
    )
    assert rec.events == [
        ("signal", signal.SIGINT, _h("int")),
        ("signal", signal.SIGTERM, _h("term")),
        ("shutdown",),
    ]


# --------------------------------------------------------------------------
# Call sites B and C: submap_stitch / rolling_seed (train_streaming.py:5074-5077,
# 5088-5091) -- byte-identical to each other:
#   save_worker.shutdown()
#   _signal.signal(_signal.SIGINT, _orig_sigint)
#   _signal.signal(_signal.SIGTERM, _orig_sigterm)
#   _finalize_output_staging(args)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["submap_stitch", "rolling_seed"])
def test_coordinator_mode_order(rec, fake_finalize, mode):
    args = object()
    finalize_run(
        rec.make_worker(),
        shutdown_first=True,
        finalize_staging=True,
        orig_sigint=_h("int"),
        orig_sigterm=_h("term"),
        args=args,
    )
    assert rec.events == [
        ("shutdown",),
        ("signal", signal.SIGINT, _h("int")),
        ("signal", signal.SIGTERM, _h("term")),
        ("finalize", args),
    ]


# --------------------------------------------------------------------------
# Contract details
# --------------------------------------------------------------------------
def test_finalize_called_with_default_timeout(rec, monkeypatch):
    """The normal-completion path must use the 600s default, never the 60s
    emergency timeout."""
    seen = {}
    mod = type(sys)("train")

    def _fin(args, timeout=600.0):
        seen["timeout"] = timeout

    mod._finalize_output_staging = _fin
    monkeypatch.setitem(sys.modules, "train", mod)

    finalize_run(
        rec.make_worker(), shutdown_first=True, finalize_staging=True, args=object()
    )
    assert seen == {"timeout": 600.0}


def test_output_persistence_error_propagates_after_shutdown(rec, monkeypatch):
    """OutputPersistenceError propagating uncaught IS the contract -- and it
    must escape only after the worker is drained and handlers restored, so a
    failed run never leaks the save-worker thread."""
    mod = type(sys)("train")

    class _Boom(RuntimeError):
        pass

    def _fin(args, timeout=600.0):
        rec.events.append(("finalize", args))
        raise _Boom("persist failed")

    mod._finalize_output_staging = _fin
    monkeypatch.setitem(sys.modules, "train", mod)

    with pytest.raises(_Boom):
        finalize_run(
            rec.make_worker(),
            shutdown_first=True,
            finalize_staging=True,
            orig_sigint=_h("int"),
            orig_sigterm=_h("term"),
            args=object(),
        )
    assert [e[0] for e in rec.events] == ["shutdown", "signal", "signal", "finalize"]


def test_shutdown_is_not_swallowed(rec):
    """No try/except around shutdown() at any of the four normal-completion
    sites -- only the emergency handlers guard it."""

    class _Worker:
        def shutdown(self):
            raise RuntimeError("drain failed")

    with pytest.raises(RuntimeError, match="drain failed"):
        finalize_run(_Worker(), shutdown_first=True, finalize_staging=False)


def test_omitted_handlers_are_not_restored(rec):
    """train.py never installs a SIGINT handler; omitting the kwarg must not
    call signal.signal for it (passing None would raise TypeError)."""
    finalize_run(rec.make_worker(), shutdown_first=False, finalize_staging=False)
    assert rec.events == [("shutdown",)]


def test_none_handler_is_passed_through_verbatim(rec):
    """getsignal() can return None for a C-installed handler; today that
    raises TypeError at the call site. Keep it raising, do not guard."""
    finalize_run(
        rec.make_worker(),
        shutdown_first=False,
        finalize_staging=False,
        orig_sigterm=None,
    )
    assert rec.events == [("signal", signal.SIGTERM, None), ("shutdown",)]


def test_finalize_staging_requires_args(rec):
    with pytest.raises(ValueError):
        finalize_run(rec.make_worker(), shutdown_first=True, finalize_staging=True)


def test_required_keywords_are_required(rec):
    with pytest.raises(TypeError):
        finalize_run(rec.make_worker())
    with pytest.raises(TypeError):
        finalize_run(rec.make_worker(), shutdown_first=True)


# --------------------------------------------------------------------------
# Call-site wiring guards
#
# The tests above prove `finalize_run` reproduces each original tail; these
# prove each call site asks for the tail it actually had. Getting a flag
# wrong is silent and severe -- `finalize_staging=True` at the offline /
# normal-streaming sites would mirror-and-delete the staging directory
# BEFORE run_summary.json / artifact_manifest.json are written, losing them
# -- and no executable test in this repo reaches those tails (they need a
# real dataset and GPU). Source inspection is the established idiom here;
# see TestStreamingTrainingSourceWiring in test_output_staging_streaming.py.
# --------------------------------------------------------------------------
def _call_args(source, marker, call_name="finalize_run"):
    """The named completion-call arguments in the branch after ``marker``."""
    _, sep, after = source.partition(marker)
    assert sep, f"marker not found: {marker!r}"
    _, sep, call = after.partition(f"{call_name}(")
    assert sep, f"no {call_name}() call after {marker!r}"
    return call.split(")", 1)[0]


def _alt_trainer_branch(source):
    """The `submap_stitch`/`rolling_seed` early-return branch body (issue #1102)."""
    _, sep, after = source.partition('if _training_mode in ("submap_stitch", "rolling_seed"):')
    assert sep, "alternate-trainer dispatch branch not found"
    # The branch ends at the next statement at the same indentation level.
    return after.split("\n    if _training_mode != ", 1)[0]


def test_offline_call_site_flags():
    # PR #1783 system-telemetry follow-up (2026-09-06) moved the legacy
    # offline reconstruction loop's body out of `training()` and into a new
    # `_train_offline()` -- verbatim, same indentation -- so `training()`
    # could wrap its whole lifetime in a start/stop-system-telemetry
    # try/finally without reindenting ~1400 existing lines. The content this
    # test asserts on (the `begin_offline_completion()` call site and the
    # deferred staging gate) moved there with it.
    src = function_source("train.py", "_train_offline")
    call = _call_args(src, "def _train_offline", "begin_offline_completion")
    assert "final_iteration=opt.iterations" in call
    assert "orig_sigterm=_orig_sigterm" in call
    # The deferred staging gate must still exist further down.
    assert "_finalize_output_staging(args)" in src
    restore = "_signal.signal(_signal.SIGTERM, _orig_sigterm)"
    assert (
        src.index("_finalize_output_staging(args)")
        < src.index("_run_state.raise_if_incomplete()")
        < src.index(restore)
    )


def test_streaming_call_site_flags():
    src = function_source("train_streaming.py", "streaming_training")

    # Issue #1102 moved the two alternate trainers onto the same deferred
    # staging gate the other two sites use: `finalize_staging=False`, then the
    # evidence artifacts (run_completion_state / run_summary /
    # artifact_manifest), then `_finalize_output_staging(args)`, then the
    # fail-closed `raise_if_incomplete()`. `shutdown_first=True` -- the
    # ordering question issue #1102 part 2 raises -- is still preserved
    # verbatim and deliberately undecided.
    for marker in ('"submap_stitch"', '"rolling_seed"'):
        call = _call_args(src, marker)
        assert "shutdown_first=True" in call, marker
        assert "finalize_staging=False" in call, marker
        assert "orig_sigint=_orig_sigint" in call, marker
        assert "orig_sigterm=_orig_sigterm" in call, marker
        # `args=` is only required when finalize_run() itself stages.
        assert "args=args" not in call, marker

    alt_branch = _alt_trainer_branch(src)
    for needed in (
        "_emit_run_evidence(args, _mode_state)",
        "_finalize_output_staging(args)",
        "_mode_state.raise_if_incomplete()",
    ):
        assert needed in alt_branch, needed
    # ...and in that order: evidence before staging, gate last.
    assert (
        alt_branch.index("_emit_run_evidence(args, _mode_state)")
        < alt_branch.index("_finalize_output_staging(args)")
        < alt_branch.index("_mode_state.raise_if_incomplete()")
    )

    # The normal-completion tail sits just before `close_decode_cache()`,
    # ahead of the multires-stats block and the "Mapping runtime complete."
    # print; its staging gate comes several hundred lines later.
    head, sep, rest = src.rpartition("streaming_scene.close_decode_cache()")
    assert sep, "close_decode_cache() anchor not found"
    _, sep, call = head.rpartition("finalize_run(")
    assert sep, "normal-completion tail does not call finalize_run()"
    call = call.split(")", 1)[0]
    assert "shutdown_first=False" in call
    assert "finalize_staging=False" in call
    assert "orig_sigint=_orig_sigint" in call
    assert "orig_sigterm=_orig_sigterm" in call
    # The deferred staging gate must still follow it.
    assert "_finalize_output_staging(args)" in rest
