"""Issue #670: torch-free ordering tripwire for the deferred provenance warning.

Complements tests/test_issue670_deferred_git_provenance_warning.py (PR #671),
which already covers the emitter's *behaviour*. Two things it does not cover:

1. **A second entry point.** PR #671 wired the re-emit into `train.py` only,
   while `eligsir/cli/ros_node.py` had the identical ordering --
   `get_git_metadata()` then, ten lines later, `start_output_log(args)` -- so
   its provenance warning was lost from `train.log` exactly the same way.
   That entry point was removed from core by #1076's ROS hard cut (9df507ea)
   and now lives in eligsir-project/eligsir-ros2 (`eligsir_node`), so
   this module guards `train.py` only; the ROS ordering guard belongs to that
   repo. `tests/test_import_boundaries.py` keeps the file itself from
   returning here.

2. **A tripwire that actually runs on the CPU test surface.** #671's
   `test_train_py_calls_the_emitter_after_start_output_log` obtains the source
   via ``import train`` + ``inspect.getsource``; importing `train` pulls in
   torch/gsplat, so on any surface without them that whole test file fails to
   collect and the guard silently does not run -- the same defect class as
   issue #672. These tests read the entry points as TEXT (`pathlib`), import
   nothing but `re`, and therefore execute everywhere.

The invariant, for both entry points: the capture must stay ABOVE
`start_output_log(args)` (`ensure_model_path`/`start_benchmark_log` run in
between and `cfg_args` ordering is load-bearing -- which is why the warning is
deferred rather than the capture moved), and the emitter must be called BELOW
it, since before it there is no `train.log` to write into.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

ENTRY_POINTS = [
    REPO_ROOT / "train.py",
]

# Match the CALL. A plain substring search for "start_output_log" would hit
# `def start_output_log(args):` at train.py:228 and make the assertion pass
# vacuously against the definition instead of the call site.
_TEE = re.compile(r"^[ \t]*start_output_log\(args\)[ \t]*$", re.M)
_CAPTURE = re.compile(r"^[ \t]*for key, value in get_git_metadata\(\)", re.M)
_EMIT = re.compile(r"^[ \t]*emit_deferred_git_provenance_warning\(\)[ \t]*$", re.M)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda p: p.name)
def test_emitter_is_called_after_the_log_tee_is_installed(entry_point):
    source = entry_point.read_text()

    tee = _TEE.search(source)
    emit = _EMIT.search(source)

    assert tee is not None, f"{entry_point.name}: no start_output_log(args) call found"
    assert emit is not None, (
        f"{entry_point.name}: emit_deferred_git_provenance_warning() is never "
        "called -- a degraded run's provenance warning would not reach its own "
        "train.log (issue #670)"
    )
    assert emit.start() > tee.start(), (
        f"{entry_point.name}: emit_deferred_git_provenance_warning() must be "
        "called AFTER start_output_log(args) installs the TeeStream; before "
        "that point there is no train.log to write into"
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda p: p.name)
def test_provenance_capture_still_precedes_the_log_tee(entry_point):
    """The complement: the fix must not have MOVED the capture instead.

    If the capture ever ends up below the tee, the deferral is dead code and
    should be removed deliberately rather than left in place -- and moving it
    there is not free, since `ensure_model_path`/`start_benchmark_log` and
    `cfg_args` ordering sit in between.
    """
    source = entry_point.read_text()

    capture = _CAPTURE.search(source)
    tee = _TEE.search(source)

    assert capture is not None, f"{entry_point.name}: provenance capture call not found"
    assert tee is not None
    assert capture.start() < tee.start(), (
        f"{entry_point.name}: get_git_metadata() now runs AFTER the tee -- the "
        "deferred re-emit is redundant and should be removed (issue #670)"
    )


def test_this_tripwire_imports_nothing_torch_tainted():
    """Guard the guard (issue #672 class).

    The whole point of reading the entry points as text is that this file
    collects on a surface without torch/gsplat. An `import train` creeping in
    here would silently disable both tests above on exactly that surface.
    """
    source = pathlib.Path(__file__).read_text()

    # Real import STATEMENTS only -- the docstrings above legitimately mention
    # `import train` / `inspect.getsource` while describing what not to do.
    assert re.search(r"^[ \t]*import train\b", source, re.M) is None
    assert re.search(r"^[ \t]*from train import\b", source, re.M) is None
    assert re.search(r"^[ \t]*import inspect\b", source, re.M) is None
