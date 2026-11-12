"""Issue #646: every key train_streaming.py sends to the report writer must be
a parameter the report writer actually accepts.

PR #620 (#596) added ``"spf_resolution": _spf_resolution.as_dict()`` to
``train_streaming.py``'s shared ``_streaming_report_kwargs()`` builder without
adding the matching parameter to
``utils.comparison_report.write_post_training_report``. Because all four report
call sites splat that one builder, EVERY streaming run on the merged commit
trained to completion and then raised
``TypeError: write_post_training_report() got an unexpected keyword argument
'spf_resolution'`` in the report phase -- producing no ``report.json``, no
``comparison/`` directory and no contact sheets, while still printing
``[report-phase] ✓ ... done`` and ``Training complete.``. ablator's
``require_result_artifact`` then quarantined the job, idling the whole GPU fleet.

Unit tests of either module in isolation cannot catch this: the mismatch lives
in the SEAM between them, and the calls are ``**kwargs`` splats resolved at
runtime. This test closes the whole CLASS rather than the single key, by
statically extracting the builder's emitted key set and checking it against the
writer's real signature -- so the next additive report field cannot repeat it.

Static (AST) rather than importing and calling the builder. Since #1075 task
P3 the builder is a real module-level function
(``eligsir.runtime.report_kwargs.build_streaming_report_kwargs``) and IS
callable in isolation -- but importing it pulls in torch, which the CPU-CI
image does not have (see the paragraph below), so this guard stays static.
``tests/test_issue1075_report_kwargs_extraction.py`` is the behavioral
counterpart that does call it, behind ``importorskip("torch")``.

BOTH SIDES are read statically, and that is deliberate (added 2026-08-15).
The original version of this file read the *builder* side via AST but the
*writer* side via ``from eligsir.utils.comparison_report import
write_post_training_report``. That import pulls in torch, so this guard --
written for a P0 outage in which EVERY streaming run for two days produced no
``report.json`` -- was unreachable on the only CI surface that is a required
check. Verified by blocking ``torch`` through a ``sys.meta_path`` finder
(#435's own method): the import raises, so in CPU CI the file was collected
away rather than run, and the regression it exists to catch could have merged
again unseen. #649 tracks the general problem (44% of tests unreachable in
CI); this file is one of the cases where the dependency was incidental and
removable, so it is removed rather than waited on.

The runtime signature is still checked -- ``test_runtime_signature_matches_ast``
imports the writer behind ``importorskip("torch")`` and asserts the statically
extracted parameter set equals ``inspect.signature``'s. So the AST extraction
cannot silently drift from reality: in the training image both run and must
agree; in CPU CI the static ones run alone and still close the #646 class.
"""

import ast
import inspect
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"
# Issue #1075 (task P3): the builder body moved out of train_streaming.py's
# streaming_training() closure into its own module, where its ~35 formerly
# implicit closure captures are explicit keyword parameters. Only the *file*
# this guard reads changed -- what it checks (every emitted report kwarg is a
# parameter the writer accepts) is unchanged, and the AST shapes _emitted_keys
# collects (dict literals + ``_extra["name"] = ...``) moved verbatim.
REPORT_KWARGS = REPO_ROOT / "eligsir" / "runtime" / "report_kwargs.py"
COMPARISON_REPORT = REPO_ROOT / "utils" / "comparison_report.py"
BUILDER_NAME = "build_streaming_report_kwargs"
WRITER_NAME = "write_post_training_report"


def _find_builder(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == BUILDER_NAME:
            return node
    raise AssertionError(
        f"{BUILDER_NAME}() not found in {REPORT_KWARGS.name}. If it was renamed "
        "or inlined, retarget this test -- do not delete it; it is the only guard "
        "against the #646 call-seam mismatch."
    )


def _emitted_keys(builder: ast.FunctionDef) -> set:
    """Report-kwarg names the builder can emit.

    Two shapes are used in the builder today and both are collected:
      * string-literal keys of any dict literal it builds/returns, and
      * ``_extra["name"] = ...`` subscript assignments, which are merged into
        the returned dict.

    Dynamic keys (``**other``, computed subscripts) are not resolvable
    statically and are skipped -- this test is a floor, not a ceiling.
    """
    keys = set()
    for node in ast.walk(builder):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    keys.add(target.slice.value)
    return keys


def _find_writer(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == WRITER_NAME
        ):
            return node
    raise AssertionError(
        f"{WRITER_NAME}() not found in {COMPARISON_REPORT.name}. If the report "
        "writer moved, retarget this test -- do not delete it; it is the only "
        "guard against the #646 call-seam mismatch."
    )


def _accepted_params(writer: ast.FunctionDef) -> set:
    """Parameter names the writer accepts, read statically.

    Covers positional-only, positional-or-keyword and keyword-only args. A
    ``**kwargs`` catch-all would make the seam un-checkable (it would accept
    anything at call time and silently drop it), so its presence is reported
    as a failure by ``test_writer_has_no_kwargs_catch_all`` rather than
    quietly widening the accepted set here.
    """
    a = writer.args
    return {
        arg.arg for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)
    }


@pytest.fixture(scope="module")
def writer_ast() -> ast.FunctionDef:
    return _find_writer(ast.parse(COMPARISON_REPORT.read_text()))


@pytest.fixture(scope="module")
def accepted_params(writer_ast) -> set:
    params = _accepted_params(writer_ast)
    # Sanity floor, mirroring builder_keys' -- a broken extraction that
    # returned everything would make the seam test pass vacuously.
    for anchor in ("model_path", "iteration", "gaussians"):
        assert anchor in params, (
            f"writer signature extraction looks broken: expected anchor "
            f"{anchor!r} among {sorted(params)}"
        )
    return params


@pytest.fixture(scope="module")
def builder_keys() -> set:
    tree = ast.parse(REPORT_KWARGS.read_text())
    keys = _emitted_keys(_find_builder(tree))
    # Sanity floor: if the extraction silently stops finding keys, the test
    # would pass vacuously forever. These three are long-standing, from
    # different report generations, and are emitted via both shapes above.
    for anchor in ("pose_provenance", "ingestion", "seed_cohort"):
        assert anchor in keys, (
            f"key extraction looks broken: expected anchor {anchor!r} among "
            f"{sorted(keys)}"
        )
    return keys


def test_every_emitted_report_kwarg_is_accepted_by_the_writer(
    builder_keys, accepted_params
):
    unknown = sorted(builder_keys - accepted_params)
    assert not unknown, (
        f"{BUILDER_NAME}() emits report kwargs that "
        f"write_post_training_report() does not accept: {unknown}. Every "
        "streaming run would train to completion and then die in the report "
        "phase with no report.json (issue #646). Add the parameter to the "
        "writer AND persist it into the emitted report -- a parameter that is "
        "accepted and dropped is a silent half-fix."
    )


def test_spf_resolution_is_accepted_and_persisted(accepted_params, writer_ast):
    """#596's own field, the one that regressed -- pinned explicitly."""
    assert "spf_resolution" in accepted_params
    src = ast.get_source_segment(COMPARISON_REPORT.read_text(), writer_ast) or ""
    assert 'summary["spf_resolution"]' in src, (
        "write_post_training_report accepts spf_resolution but never writes it "
        "to the report -- #596's deliverable is report.json['spf_resolution']."
    )


def test_writer_has_no_kwargs_catch_all(writer_ast):
    """A ``**kwargs`` on the writer would silently defeat this whole guard.

    #646 was survivable-as-a-crash precisely because the writer rejects
    unknown keywords loudly. If someone "fixes" a future mismatch by adding
    ``**_ignored``, every unknown report field would instead be accepted and
    dropped -- report.json would be quietly incomplete rather than absent,
    which is strictly worse and undetectable from the artifact.
    """
    assert writer_ast.args.kwarg is None, (
        f"{WRITER_NAME}() grew a **{writer_ast.args.kwarg.arg} catch-all. That "
        "converts the #646 crash into silent field loss and makes the "
        "call-seam guard above vacuous. Add explicit parameters instead."
    )


def test_runtime_signature_matches_ast(accepted_params):
    """The AST extraction must not drift from the real runtime signature.

    Skipped in CPU CI (importing the writer needs torch) -- which is the whole
    reason the checks above are static. In the training image this runs and
    pins the two views together.
    """
    pytest.importorskip("torch")
    from eligsir.utils.comparison_report import write_post_training_report

    runtime = {
        name
        for name, p in inspect.signature(write_post_training_report).parameters.items()
        if p.kind
        not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    }
    assert runtime == accepted_params, (
        "static parameter extraction disagrees with inspect.signature: "
        f"ast-only={sorted(accepted_params - runtime)}, "
        f"runtime-only={sorted(runtime - accepted_params)}. Fix _accepted_params() "
        "-- the CPU-CI-reachable checks above depend on it being exact."
    )
