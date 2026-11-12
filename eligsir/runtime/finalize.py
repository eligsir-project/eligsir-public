"""Shared normal-completion shutdown tail for training runs (issue #1075).

Every training entry point ends with the same three-part tail:

1. restore the signal handlers the run installed for its
   emergency-checkpoint / graceful-shutdown behaviour,
2. drain and stop the ``AsyncSaveWorker``,
3. (some paths) block on output-staging persistence, which mirrors the
   local staging directory onto the canonical ``persist_model_path`` and
   writes the ``.COMPLETE``/``.FAILED`` marker.

That tail was hand-copied at four call sites:

===================================  =========================  ==========
call site                            order                      finalize
===================================  =========================  ==========
``train.py`` (offline path)          shutdown -> late restore   deferred
``train_streaming.py`` (normal)      restore -> shutdown        deferred
``train_streaming.py`` submap_stitch shutdown -> restore        inline
``train_streaming.py`` rolling_seed  shutdown -> restore        inline
===================================  =========================  ==========

Two real differences between the copies are **preserved verbatim** rather
than unified, because either choice is a semantic change inside the
shutdown window that no test in this repo can validate:

* **Ordering.** The offline path drains first and retains its emergency
  ``SIGTERM`` handler until reporting and output finalization complete. The
  streaming call sites retain their existing ordering. Callers declare the
  shared-tail order via the required ``shutdown_first`` keyword.

* **Where staging finalization happens.** ``train.py`` and the normal
  streaming path run several hundred lines of report / ``run_summary`` /
  ``artifact_manifest`` work between the drain and
  ``_finalize_output_staging`` -- that work *must* precede finalization,
  since finalization moves ``args.model_path`` to the canonical path and
  removes the local copy. Those two sites therefore pass
  ``finalize_staging=False`` and keep their own, later
  ``_finalize_output_staging(args)`` call. ``finalize_staging`` is
  required (not defaulted) so no call site can silently skip the gate
  that keeps a staged run from exiting 0 with outputs only in scratch.

Exception semantics are unchanged: neither ``save_worker.shutdown()`` nor
``_finalize_output_staging`` is wrapped in ``try``/``except`` here -- at
all four original sites an ``OutputPersistenceError`` propagating uncaught
*is* the contract (it produces the non-zero exit code). Signal restoration
and the worker drain both happen before finalization, so a raise cannot
leak the worker thread.
"""

from __future__ import annotations

import contextlib
import signal as _signal
from typing import Any, Iterator


class _Unset:
    """Sentinel: this handler was never captured by the caller."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return "<unset>"


UNSET = _Unset()


def _restore(signum: int, handler: Any) -> None:
    if isinstance(handler, _Unset):
        return
    # Deliberately unguarded: `signal.getsignal()` can return None for a
    # handler installed from C, and passing None to `signal.signal()`
    # raises TypeError. That is exactly what every original call site did;
    # swallowing it here would hide a real, currently-loud condition.
    _signal.signal(signum, handler)


def finalize_run(
    save_worker: Any,
    *,
    shutdown_first: bool,
    finalize_staging: bool,
    orig_sigint: Any = UNSET,
    orig_sigterm: Any = UNSET,
    args: Any = None,
) -> None:
    """Run the shared normal-completion shutdown tail.

    :param save_worker: the run's ``AsyncSaveWorker``; ``shutdown()`` is
        called exactly once, unguarded.
    :param shutdown_first: ``True`` drains the worker before restoring
        signal handlers (the ``submap_stitch``/``rolling_seed`` order);
        ``False`` restores handlers first (``train.py`` and the normal
        streaming path). Required -- see the module docstring.
    :param finalize_staging: ``True`` blocks on output-staging
        persistence here; ``False`` leaves that to the caller's own,
        later ``_finalize_output_staging(args)``. Required.
    :param orig_sigint: handler to restore for ``SIGINT``. Omit entirely
        when the caller never installed one (``train.py`` does not).
    :param orig_sigterm: handler to restore for ``SIGTERM``. Omit when
        the caller never installed one.
    :param args: the parsed run args; required when
        ``finalize_staging`` is ``True``.
    """
    if finalize_staging and args is None:
        raise ValueError("finalize_run(finalize_staging=True) requires args")

    def _drain() -> None:
        save_worker.shutdown()

    def _restore_all() -> None:
        _restore(_signal.SIGINT, orig_sigint)
        _restore(_signal.SIGTERM, orig_sigterm)

    if shutdown_first:
        _drain()
        _restore_all()
    else:
        _restore_all()
        _drain()

    if finalize_staging:
        # Lazy import on purpose, mirroring `train_streaming.py`'s existing
        # `from train import _finalize_output_staging` idiom: `train` is a
        # top-level script module that imports heavy training machinery, and
        # importing it at package-module import time would create an
        # import-order cycle. Consolidating `_finalize_output_staging` into
        # `eligsir.runtime.output_staging` is tracked separately.
        from train import _finalize_output_staging

        _finalize_output_staging(args)


def begin_offline_completion(
    save_worker: Any,
    run_state: Any,
    *,
    final_iteration: int,
    orig_sigterm: Any = UNSET,
) -> None:
    """Drain queued snapshots before claiming offline completion.

    ``train.py`` owns restoration of ``orig_sigterm`` after its final report
    and output-staging gate; accepting it here keeps the lifecycle contract
    explicit without restoring the emergency handler prematurely.
    """
    save_worker.shutdown()
    run_state.mark_training_complete(final_iteration=final_iteration)


# ---------------------------------------------------------------------------
# Issue #1102: completion evidence for the alternate trainers
# ---------------------------------------------------------------------------
#
# `submap_stitch` / `rolling_seed` (the two `MappingCoordinator` training
# modes reached from `train_streaming.streaming_training()`'s early returns)
# historically emitted NO completion evidence at all: no
# `run_completion_state.json`, no `run_summary.json`, no
# `artifact_manifest.json`, and no `raise_if_incomplete()` gate. They could
# therefore exit 0 with a failed required evidence phase, and could not be
# ingested by ResearchFlow, whose canonical contract is
# `run_summary.json` + `run_completion_state.json`.
#
# The two helpers below are the shared, torch-free halves of the fix:
# `report_phase()` for the mode-side evidence phase, `emit_run_evidence()`
# for the call-site-side artifact emission. Both are deliberately identical
# in behavior to what `train.py` and the normal streaming path already do
# inline -- this module does not invent a second contract.


class _NoOpPhaseHandle:
    """Stand-in for ``run_completion_state._PhaseHandle`` when no state is
    tracked (a direct/legacy caller that passed ``run_state=None``)."""

    def record_output(self, summary: Any) -> None:
        """Discard the report's output directory (nothing is being recorded)."""
        return None

    def note(self, text: str) -> None:
        """Discard a phase note (nothing is being recorded)."""
        return None


@contextlib.contextmanager
def report_phase(run_state: Any, *, log_prefix: str = "report") -> "Iterator[Any]":
    """Wrap a trainer's final-report block as a recorded evidence phase.

    With a ``run_state`` this is exactly
    ``run_state.phase(FINAL_REPORT_CONTRACT)``: the outcome (SUCCESS /
    FAILED / MISSING_ARTIFACTS) is recorded, the exception is swallowed so
    the remaining shutdown work still runs, and
    ``run_state.raise_if_incomplete()`` re-raises it later.

    With ``run_state=None`` it reproduces the pre-#1102 behavior of the two
    coordinator modes verbatim -- catch, print, continue -- so that a caller
    that tracks no state is not silently given a new failure mode.
    """
    if run_state is None:
        try:
            yield _NoOpPhaseHandle()
        except Exception as exc:  # noqa: BLE001 - preserved legacy behavior
            print(f"[{log_prefix}] Evaluation failed: {exc}", flush=True)
        return

    from eligsir.utils.run_completion_state import FINAL_REPORT_CONTRACT

    with run_state.phase(FINAL_REPORT_CONTRACT) as handle:
        yield handle


def emit_run_evidence(args: Any, run_state: Any) -> None:
    """Write the three root-level evidence artifacts for a finished run.

    Mirrors, call for call, the block that `train.py` (offline) and
    `train_streaming.py` (normal streaming) run between the save-worker
    drain and ``_finalize_output_staging``:

    1. ``run_completion_state.json`` (issue #697 / #998),
    2. ``run_summary.json`` (issue #997 / #1013),
    3. ``artifact_manifest.json`` (issue #1020, best-effort).

    It deliberately does NOT call ``raise_if_incomplete()``: that gate must
    fire *after* output staging, so a failed evidence phase costs the run
    its exit code and never its outputs.
    """
    run_state.write()
    run_state.print_summary()

    from eligsir.utils.run_summary import write_run_summary

    write_run_summary(
        args.model_path,
        completion_state=run_state.to_dict(),
        evidence_class="MANAGED" if getattr(args, "scientific_mode", False) else "UNMANAGED",
        args=args,
    )

    try:
        from eligsir.utils.artifact_manifest import build_artifact_manifest

        build_artifact_manifest(args.model_path)
    except Exception as exc:  # noqa: BLE001 - matches the other two call sites
        print(f"[artifact-manifest] Warning: failed to build artifact manifest: {exc}")
