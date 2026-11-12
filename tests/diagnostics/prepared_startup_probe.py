"""Temporary managed-run diagnostic; no device synchronization or policy changes.

Usage: python tests/diagnostics/prepared_startup_probe.py train.py <train args>
The original arguments are preserved. This instrumented run is not a matched
performance benchmark; counters include Python-wrapper overhead.
"""

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import runpy
import sys
from threading import Lock
import time


@contextmanager
def observe_startup(application_type, worker_type, adapter_type, emit):
    """Observe real application/worker calls, restoring all methods on exit."""
    counts = Counter()
    outcomes = Counter()
    lock = Lock()
    started = last_report = time.perf_counter()
    originals = []

    def snapshot():
        with lock:
            return {
                "elapsed_s": time.perf_counter() - started,
                "counters": dict(counts),
                "outcomes": [
                    dict(zip(("result_type", "state", "retryable", "armed", "waiter_bound", "calls"),
                             (*key, calls)))
                    for key, calls in outcomes.items()
                ],
            }

    def wrap(owner, name, counter):
        original = getattr(owner, name)
        originals.append((owner, name, original))

        def observed(self, *args, **kwargs):
            nonlocal last_report
            began = time.perf_counter()
            result = original(self, *args, **kwargs)
            ended = time.perf_counter()
            report = False
            with lock:
                counts[counter + "_calls"] += 1
                counts[counter + "_s"] += ended - began
                if name == "tick":
                    state = getattr(result, "state", None)
                    state = getattr(state, "value", state)
                    key = (
                        f"{type(result).__module__}.{type(result).__name__}", state,
                        bool(getattr(result, "retryable", False)),
                        self._prepared_retry_pending,
                        self._options.prepared_progress_waiter is not None,
                    )
                    outcomes[key] += 1
                    if ended - last_report >= 5.0:
                        last_report = ended
                        report = True
                elif name.startswith("wait_for"):
                    counts[counter + "_progress"] += bool(result)
            if report:
                try:
                    emit(snapshot())
                except OSError:
                    pass  # Diagnostic stdout must not turn successful work into failure.
            return result

        setattr(owner, name, observed)

    wrap(application_type, "tick", "tick")
    wrap(application_type, "wait_for_prepared_progress", "application_wait")
    wrap(worker_type, "wait_for_progress", "worker_wait")
    wrap(adapter_type, "packet_from_record", "decode")
    try:
        yield snapshot
    finally:
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)


def main():
    script, *arguments = sys.argv[1:]
    original_argv, original_path = sys.argv, sys.path[:]
    try:
        run_observed(script, arguments)
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def run_observed(script, arguments):
    source_root = Path(script).resolve().parent
    sys.path.insert(0, str(source_root))
    sys.argv = [script, *arguments]
    model_flag = "-m" if "-m" in arguments else "--model_path"
    output = Path(arguments[arguments.index(model_flag) + 1]) / "qualification_receipts"
    output.mkdir(parents=True, exist_ok=True)

    from eligsir.runtime.application import ComposedMappingApplication
    from eligsir.runtime.causal_replay import StreamingFramePacketAdapter
    from eligsir.runtime.prepared_source_prep_worker import PreparedSourcePrepWorker

    types = (ComposedMappingApplication, PreparedSourcePrepWorker, StreamingFramePacketAdapter)
    modules = {}
    for owner in types:
        path = Path(sys.modules[owner.__module__].__file__)
        modules[owner.__module__] = {
            "path": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def emit(row):
        try:
            print("[prepared-startup-probe] " + json.dumps(row, sort_keys=True), flush=True)
        except OSError:
            pass

    emit({"instrumented": True, "modules": modules})
    with observe_startup(*types, emit) as snapshot:
        failed = False
        try:
            runpy.run_path(script, run_name="__main__")
        except BaseException:
            failed = True
            raise
        finally:
            report = {"instrumented": True, "modules": modules, **snapshot()}
            try:
                (output / "prepared_startup_probe.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
            except OSError as error:
                if not failed:
                    raise
                emit({"report_write_error": str(error)})


if __name__ == "__main__":
    main()
