"""Issue #697 / #998: fail-closed run/report completion state for offline & streaming runs.

Why this module exists
----------------------
The #646/#647 outage: a streaming run trained to completion, entered the
report phase, raised ``TypeError`` inside ``write_post_training_report``, and
STILL printed ``[report-phase] ✓ ... done`` followed by ``Training complete.``
-- because the report call was wrapped in a bare ``except Exception: print(...)``
and the success checkmark was printed unconditionally afterwards. No
``report.json``, no ``comparison/`` directory, no contact sheets; a log that
reads exactly like a healthy finished run to a human and to a log parser.

Issue #998 unifies this completion contract across both offline (`train.py`)
and streaming (`train_streaming.py`) runs so that both execution paths share
the identical schema, state file, lifecycle, and fail-closed exit behavior.

The fix is not "catch that TypeError". It is to make the report phase's
outcome an explicit, machine-readable value that a consumer can read without
parsing console text, and to make a success marker impossible to emit unless
the phase both returned AND produced the artifacts its contract declares
required.

Vocabulary (from the issue)::

    TRAINING_COMPLETE
    REPORT_COMPLETE
    RUN_COMPLETE = TRAINING_COMPLETE && REPORT_COMPLETE && required_artifacts_present

Four outcomes are distinguished, and "not requested" is deliberately NOT a
failure:

==============================  ==========================================
``SUCCESS``                     phase returned and every required artifact
                                declared by its contract exists on disk
``FAILED``                      phase raised
``MISSING_ARTIFACTS``           phase returned, a required artifact is absent
``NOT_REQUESTED``               phase was intentionally disabled
                                (``--no-streaming_report_final``,
                                ``--streaming_benchmark_mode``)
==============================  ==========================================

Design constraints
------------------
**No torch, directly or transitively.** ``tests/test_issue646_report_kwargs_
signature.py`` documents (and #649 tracks) that importing ``torch`` makes a
test unreachable on the only required CI check -- the #646 guard was silently
collected away for exactly that reason. This module is therefore pure stdlib,
and all the logic that the regression tests need to exercise lives HERE, not
in ``train_streaming.streaming_training()`` (a ~6000-line function that is
unreachable without standing up a whole training run). The call site is a
thin shim over this module.

**Artifact paths come from the report writer's own return value.** The
streaming call site passes ``iteration=opt.iterations``, which for an
``--online_mode`` run is the ``2**31`` sentinel and DELIBERATELY names the
output directory (``comparison/iter_2147483647/``) -- see the comment at that
call site. A validator that recomputed the directory from the training loop's
real final iteration would look in a directory that never existed and fire
fail-closed on every online_mode run. So the contract is validated against
``summary["output_dir"]`` as reported by ``write_post_training_report``, never
against a path this module constructs. When no output directory was recorded
at all and the contract requires artifacts, that is itself
``MISSING_ARTIFACTS`` (fail closed), never a pass.

Note on ``--benchmark_dir``: it routes ``timings.jsonl`` only
(``train.py:461-469``); it does NOT move ``report.json``. Reports always live
under ``<model_path>/comparison/<subdir>/``. There is deliberately no
``benchmark_dir`` branch here.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "STATE_FILENAME",
    "SCHEMA_VERSION",
    "PhaseStatus",
    "Outcome",
    "PhaseContract",
    "PhaseResult",
    "RunIncompleteError",
    "RunCompletionState",
    "load_run_completion_state",
    "get_run_outcome",
    "FINAL_REPORT_CONTRACT",
    "BOOTSTRAP_VIEWS_CONTRACT",
    "TRAJECTORY_EVAL_CONTRACT",
    "OPTIMIZATION_CURVE_CONTRACT",
]

STATE_FILENAME = "run_completion_state.json"
SCHEMA_VERSION = 1


class PhaseStatus:
    """Status of one report/evidence phase. Plain strings: this value is
    written to JSON and read by out-of-process consumers (ablator)."""

    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    MISSING_ARTIFACTS = "MISSING_ARTIFACTS"
    NOT_REQUESTED = "NOT_REQUESTED"

    #: statuses that do not block RUN_COMPLETE for a *required* phase
    NON_BLOCKING = (SUCCESS, NOT_REQUESTED)


class Outcome:
    """Top-level, machine-readable run outcome."""

    #: trained, report requested, report produced its required artifacts
    RUN_COMPLETE = "RUN_COMPLETE"
    #: trained, report deliberately not requested (benchmark/profiling run).
    #: NOT a failure -- but also not a scientific result.
    RUN_COMPLETE_NO_REPORT_REQUESTED = "RUN_COMPLETE_NO_REPORT_REQUESTED"
    #: trained, a required report phase raised
    REPORT_FAILED = "REPORT_FAILED"
    #: trained, a required report phase returned but its artifacts are absent
    REPORT_ARTIFACTS_MISSING = "REPORT_ARTIFACTS_MISSING"
    #: the report phase never reached a terminal state (process died inside it)
    REPORT_INCOMPLETE = "REPORT_INCOMPLETE"
    #: training itself never reported completion
    TRAINING_INCOMPLETE = "TRAINING_INCOMPLETE"
    #: historical run without state or unknown state
    UNKNOWN = "UNKNOWN"
    UNKNOWN_LEGACY = "UNKNOWN_LEGACY"
    LEGACY = "LEGACY"


class RunIncompleteError(RuntimeError):
    """Raised at the very end of a run whose evidence phase failed.

    Mirrors ``train.OutputPersistenceError``'s role and shape: this project's
    idiom for "the process must not exit 0". It propagates uncaught through
    ``train.training()`` -> ``__main__``, which means the interpreter dies
    before ``print("\\nTraining complete.")`` is ever reached -- so that string
    can no longer appear in a log whose evidence phase failed.
    """


@dataclass(frozen=True)
class PhaseContract:
    """Declares what one report/evidence phase must produce.

    ``required``
        Whether ``RUN_COMPLETE`` depends on this phase. Optional phases
        (bootstrap-view debug renders, ATE/RPE evaluation, the training-curve
        PNG, the progress video) are recorded honestly when they fail but do
        not downgrade the run result -- issue #697 regression test 5.
    ``required_artifacts``
        Filenames that must exist inside the phase's reported output
        directory. Empty means "returning without raising is the whole
        contract" (used by the optional phases, which have no single
        canonical output file).
    """

    name: str
    label: str
    required: bool = False
    required_artifacts: Tuple[str, ...] = ()


#: The one phase whose success defines "this run produced a scientific result".
#: ``report.json`` is exactly what ablator's ``require_result_artifact`` /
#: ``result_glob = "{model_path}/comparison/*/report.json"`` looks for.
FINAL_REPORT_CONTRACT = PhaseContract(
    name="final_report",
    label="post-run report (renders, contact sheets, videos)",
    required=True,
    required_artifacts=("report.json",),
)

#: Debug-only re-render of the bootstrap views. Written only under
#: ``--streaming_debug_bootstrap_motion``; never a required result.
BOOTSTRAP_VIEWS_CONTRACT = PhaseContract(
    name="bootstrap_views_report",
    label="bootstrap-view comparison renders",
    required=False,
    required_artifacts=(),
)

TRAJECTORY_EVAL_CONTRACT = PhaseContract(
    name="trajectory_eval",
    label="trajectory evaluation (ATE/RPE)",
    required=False,
)

OPTIMIZATION_CURVE_CONTRACT = PhaseContract(
    name="optimization_curve",
    label="optimization-quality curve",
    required=False,
)


@dataclass
class PhaseResult:
    name: str
    label: str
    required: bool
    status: str = PhaseStatus.PENDING
    duration_s: Optional[float] = None
    output_dir: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None
    missing_artifacts: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    note: Optional[str] = None

    @property
    def blocking(self) -> bool:
        """True iff this phase's status prevents RUN_COMPLETE."""
        return self.required and self.status not in PhaseStatus.NON_BLOCKING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "required": self.required,
            "status": self.status,
            "duration_s": (None if self.duration_s is None else round(self.duration_s, 3)),
            "output_dir": self.output_dir,
            "error_type": self.error_type,
            "error": self.error,
            "missing_artifacts": list(self.missing_artifacts),
            "required_artifacts": list(self.required_artifacts),
            "note": self.note,
        }


class _PhaseHandle:
    """Handed to the ``with`` body so it can report where it wrote."""

    def __init__(self, result: PhaseResult):
        self._result = result

    def record_output(self, summary: Any) -> None:
        """Record the phase's output directory.

        Accepts ``write_post_training_report``'s return dict (reads
        ``summary["output_dir"]``) or a bare path string. Deliberately the
        ONLY way an output directory is established -- this module never
        constructs one itself (see the module docstring).
        """
        out_dir: Optional[str]
        if isinstance(summary, dict):
            out_dir = summary.get("output_dir")
        elif isinstance(summary, (str, os.PathLike)):
            out_dir = os.fspath(summary)
        else:
            out_dir = None
        if out_dir:
            self._result.output_dir = os.fspath(out_dir)

    def note(self, text: str) -> None:
        self._result.note = text


class _PhaseContext:
    def __init__(self, state: "RunCompletionState", contract: PhaseContract):
        self._state = state
        self._contract = contract
        self._result: Optional[PhaseResult] = None
        self._t0 = 0.0

    def __enter__(self) -> _PhaseHandle:
        self._result = PhaseResult(
            name=self._contract.name,
            label=self._contract.label,
            required=self._contract.required,
            status=PhaseStatus.PENDING,
            required_artifacts=list(self._contract.required_artifacts),
        )
        self._state._put(self._result)
        self._t0 = time.time()
        self._state._print(f"[report-phase] ▶ {self._contract.label} …")
        # A PENDING record on disk before the body runs: if the process is
        # killed mid-report, the persisted state says PENDING, which is not a
        # success. Never "absent, therefore fine".
        self._state.write()
        return _PhaseHandle(self._result)

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self._result is not None
        res = self._result
        res.duration_s = time.time() - self._t0
        dt = f"{res.duration_s:.1f}s"
        if exc is not None:
            res.status = PhaseStatus.FAILED
            res.error_type = type(exc).__name__
            res.error = str(exc)
            self._state._print(
                f"[report-phase] ✗ {res.label} FAILED ({dt}): "
                f"{res.error_type}: {res.error}"
            )
        else:
            missing = _missing_artifacts(res.output_dir, self._contract.required_artifacts)
            if missing is None:
                res.status = PhaseStatus.MISSING_ARTIFACTS
                res.missing_artifacts = list(self._contract.required_artifacts)
                res.error_type = "OutputDirectoryUnknown"
                res.error = (
                    "phase returned without reporting an output directory, so its "
                    "required artifacts "
                    f"{list(self._contract.required_artifacts)} could not be verified"
                )
                self._state._print(
                    f"[report-phase] ✗ {res.label} INCOMPLETE ({dt}): {res.error}"
                )
            elif missing:
                res.status = PhaseStatus.MISSING_ARTIFACTS
                res.missing_artifacts = missing
                res.error_type = "MissingRequiredArtifacts"
                res.error = (
                    f"phase returned but required artifact(s) {missing} are absent "
                    f"from {res.output_dir}"
                )
                self._state._print(
                    f"[report-phase] ✗ {res.label} INCOMPLETE ({dt}): {res.error}"
                )
            else:
                res.status = PhaseStatus.SUCCESS
                self._state._print(f"[report-phase] ✓ {res.label} done ({dt})")
        self._state.write()
        # Swallow: the remaining shutdown work (output staging, telemetry
        # flush) must still run so a report failure does not additionally
        # destroy the run's outputs. The failure is not lost -- it is in the
        # persisted state and is re-raised by raise_if_incomplete() after
        # persistence has finished.
        return exc is not None


def _missing_artifacts(
    out_dir: Optional[str], required: Sequence[str]
) -> Optional[List[str]]:
    """Return the missing required artifacts.

    ``[]``    -> everything required is present (or nothing was required)
    ``[...]`` -> these are absent
    ``None``  -> artifacts are required but no output directory is known,
                 i.e. presence is unverifiable. Callers must treat this as a
                 failure (fail closed), never as a pass.
    """
    if not required:
        return []
    if not out_dir:
        return None
    return [name for name in required if not os.path.exists(os.path.join(out_dir, name))]


class RunCompletionState:
    """Accumulates phase outcomes and persists them to
    ``<model_path>/run_completion_state.json``.

    Typical use at the end of a streaming run::

        state = RunCompletionState(args.model_path)
        state.mark_training_complete(final_iteration=iteration)
        if report_requested:
            with state.phase(FINAL_REPORT_CONTRACT) as ph:
                ph.record_output(write_post_training_report(...))
        else:
            state.mark_not_requested(FINAL_REPORT_CONTRACT, reason="...")
        ...
        state.write()                 # BEFORE output staging moves the dir
        _finalize_output_staging(args)
        state.raise_if_incomplete()   # AFTER, so a failure cannot lose output
    """

    def __init__(self, model_path: str, printer: Callable[[str], None] = None):
        self.model_path = os.fspath(model_path) if model_path else ""
        self._printer = printer
        self.training_complete: bool = False
        self.final_iteration: Optional[int] = None
        self._phases: "Dict[str, PhaseResult]" = {}
        self._order: List[str] = []
        self.write_error: Optional[str] = None
        self._crash_error_type: Optional[str] = None
        self._crash_error: Optional[str] = None

    # -- plumbing ----------------------------------------------------------
    def _print(self, msg: str) -> None:
        if self._printer is not None:
            self._printer(msg)
        else:
            print(msg, flush=True)

    def _put(self, result: PhaseResult) -> None:
        if result.name not in self._phases:
            self._order.append(result.name)
        self._phases[result.name] = result

    # -- recording ---------------------------------------------------------
    def mark_training_complete(self, final_iteration: Optional[int] = None) -> None:
        self.training_complete = True
        if final_iteration is not None:
            self.final_iteration = int(final_iteration)
        self.write()

    def mark_training_crashed(self, error: BaseException) -> None:
        """Record the real exception that ended training before completion.

        issue #1278/#1284: a crash inside training used to leave no
        ``run_completion_state.json`` at all, and ``train.log``'s tail could
        show a secondary teardown exception rather than the original one
        (e.g. "During handling of the above exception, another exception
        occurred"). Call this from the outermost ``except`` around a training
        entrypoint, with the FIRST exception caught -- before any teardown/
        finalization code that could itself raise -- so the persisted state
        names the true root cause rather than requiring log-tail archaeology.
        Still leaves ``outcome`` as ``TRAINING_INCOMPLETE`` (this run never
        reached ``mark_training_complete()``); this only adds detail to why.
        """
        self._crash_error_type = type(error).__name__
        self._crash_error = str(error)
        self.write()

    def phase(self, contract: PhaseContract) -> _PhaseContext:
        """Context manager wrapping one report/evidence phase.

        The success checkmark is printed by ``__exit__`` and ONLY on the
        success path -- there is no code path on which a raising body can be
        followed by a ``✓``.
        """
        return _PhaseContext(self, contract)

    def mark_not_requested(self, contract: PhaseContract, reason: str = "") -> PhaseResult:
        res = PhaseResult(
            name=contract.name,
            label=contract.label,
            required=contract.required,
            status=PhaseStatus.NOT_REQUESTED,
            required_artifacts=list(contract.required_artifacts),
            note=reason or None,
        )
        self._put(res)
        self.write()
        return res

    def get(self, name: str) -> Optional[PhaseResult]:
        return self._phases.get(name)

    @property
    def phases(self) -> List[PhaseResult]:
        return [self._phases[n] for n in self._order]

    # -- derived verdicts --------------------------------------------------
    @property
    def _required_phases(self) -> List[PhaseResult]:
        return [p for p in self.phases if p.required]

    @property
    def report_requested(self) -> bool:
        req = self._required_phases
        if not req:
            return False
        return any(p.status != PhaseStatus.NOT_REQUESTED for p in req)

    @property
    def report_complete(self) -> bool:
        """Every required phase reached a non-blocking terminal state.

        A required phase that was explicitly NOT_REQUESTED counts as complete
        (nothing was asked for, nothing failed); ``report_requested`` /
        ``scientific_result_complete`` are what distinguish that case.
        """
        return all(not p.blocking for p in self._required_phases)

    @property
    def required_artifacts_present(self) -> bool:
        return not any(
            p.status == PhaseStatus.MISSING_ARTIFACTS for p in self._required_phases
        )

    @property
    def run_complete(self) -> bool:
        return bool(self.training_complete) and self.report_complete and self.required_artifacts_present

    @property
    def scientific_result_complete(self) -> bool:
        """RUN_COMPLETE *and* a report was actually asked for and produced."""
        return self.run_complete and self.report_requested

    @property
    def blocking_failures(self) -> List[PhaseResult]:
        return [p for p in self._required_phases if p.blocking]

    @property
    def outcome(self) -> str:
        if not self.training_complete:
            return Outcome.TRAINING_INCOMPLETE
        blocking = self.blocking_failures
        if blocking:
            statuses = {p.status for p in blocking}
            if PhaseStatus.FAILED in statuses:
                return Outcome.REPORT_FAILED
            if PhaseStatus.MISSING_ARTIFACTS in statuses:
                return Outcome.REPORT_ARTIFACTS_MISSING
            return Outcome.REPORT_INCOMPLETE
        if not self.report_requested:
            return Outcome.RUN_COMPLETE_NO_REPORT_REQUESTED
        return Outcome.RUN_COMPLETE

    @property
    def failure_reason(self) -> Optional[str]:
        parts = []
        if not self.training_complete:
            if self._crash_error is not None:
                parts.append(
                    f"training crashed: {self._crash_error_type}: {self._crash_error}"
                )
            else:
                parts.append("training did not reach completion")
        for p in self.blocking_failures:
            parts.append(f"{p.name}: {p.status}: {p.error or 'no detail'}")
        return "; ".join(parts) if parts else None

    @property
    def failure_type(self) -> Optional[str]:
        if not self.training_complete and self._crash_error_type is not None:
            return self._crash_error_type
        for p in self.blocking_failures:
            if p.error_type:
                return p.error_type
        return None

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "issue": 697,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_path": self.model_path,
            "final_iteration": self.final_iteration,
            "outcome": self.outcome,
            "training_complete": bool(self.training_complete),
            "report_requested": self.report_requested,
            "report_complete": self.report_complete,
            "required_artifacts_present": self.required_artifacts_present,
            "run_complete": self.run_complete,
            "scientific_result_complete": self.scientific_result_complete,
            "failure_type": self.failure_type,
            "failure_reason": self.failure_reason,
            "phases": [p.to_dict() for p in self.phases],
        }

    def path(self) -> str:
        return os.path.join(self.model_path, STATE_FILENAME)

    def write(self) -> Optional[str]:
        """Atomically persist the current state. Never raises.

        Called after every phase transition so that a process killed inside
        the report phase leaves a PENDING (non-success) record behind rather
        than nothing at all.
        """
        if not self.model_path:
            return None
        target = self.path()
        tmp = target + ".tmp"
        try:
            os.makedirs(self.model_path, exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            os.replace(tmp, target)
            self.write_error = None
            return target
        except Exception as e:  # pragma: no cover - telemetry must not kill a run
            self.write_error = f"{type(e).__name__}: {e}"
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return None

    # -- human-facing summary ---------------------------------------------
    def summary_line(self) -> str:
        return (
            f"[run-state] outcome={self.outcome} training_complete={self.training_complete} "
            f"report_requested={self.report_requested} report_complete={self.report_complete} "
            f"required_artifacts_present={self.required_artifacts_present} "
            f"run_complete={self.run_complete}"
        )

    def print_summary(self) -> None:
        self._print(self.summary_line())
        if self.model_path:
            self._print(f"[run-state] machine-readable state -> {self.path()}")
        if self.outcome == Outcome.RUN_COMPLETE_NO_REPORT_REQUESTED:
            self._print(
                "[run-state] NOTE: no final report was requested for this run "
                "(benchmark/profiling mode or --no-streaming_report_final). "
                "Training finished, but this run is NOT a scientific result."
            )
        for p in self.phases:
            if p.status in (PhaseStatus.FAILED, PhaseStatus.MISSING_ARTIFACTS):
                self._print(
                    f"[run-state] {'REQUIRED' if p.required else 'optional'} phase "
                    f"'{p.name}' {p.status}: {p.error}"
                )

    def raise_if_incomplete(self) -> None:
        """Fail closed: raise unless the run is complete as declared.

        Call this AFTER output staging/persistence has run, so that a failed
        evidence phase costs the process its exit code -- never its outputs.
        """
        if self.run_complete:
            return
        self._print(
            "[run-state] ✗✗✗ THIS RUN IS NOT A VALID RESULT ✗✗✗ "
            f"outcome={self.outcome}: {self.failure_reason}"
        )
        raise RunIncompleteError(
            f"{self.outcome}: {self.failure_reason} "
            f"(machine-readable state: {self.path() if self.model_path else '<unwritten>'})"
        )


def load_run_completion_state(
    model_path: str | os.PathLike[str],
) -> Optional[Dict[str, Any]]:
    """Load persisted run completion state from model_path.

    Returns the parsed state dictionary, or None if no state file exists
    (e.g., historical run preceding issue #697/#998).
    """
    if not model_path:
        return None
    path = os.path.join(os.fspath(model_path), STATE_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_run_outcome(model_path: str | os.PathLike[str]) -> str:
    """Return the machine-readable run outcome for model_path.

    Returns Outcome.UNKNOWN_LEGACY if run_completion_state.json is absent.
    """
    state = load_run_completion_state(model_path)
    if state is None:
        return Outcome.UNKNOWN_LEGACY
    return state.get("outcome", Outcome.UNKNOWN)
