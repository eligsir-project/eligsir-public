"""The #595 fields survive the real report call seam -- checked on CPU CI (#646/#647).

WHY THIS FILE EXISTS SEPARATELY FROM ``test_issue646_report_kwargs_signature.py``
--------------------------------------------------------------------------------
#646 was: ``_streaming_report_kwargs()`` gained a key that
``write_post_training_report()`` did not accept, so EVERY streaming run since
2026-08-14 trained to completion and then died in the report phase with no
``report.json`` -- while printing a success checkmark. No test caught it
because tests build the writer's kwargs by hand. #595 adds a new key through
that identical seam, so it must not be able to reintroduce the same outage.

PR #647's guard is the right idea but lands on the wrong surface: it does
``from eligsir.utils.comparison_report import write_post_training_report``, which
imports torch, so ``tests/conftest.py``'s AST scan excludes the whole file from
``cpu-ci`` -- **the only surface any CI job runs** (see
``scripts/run_test_surface.sh``'s header). A regression there would be caught
only by someone running the training image by hand.

This file therefore parses BOTH sides with ``ast`` and imports neither, so it
runs on ``cpu-ci``. It also covers one failure mode #647's key-subset check does
not: a call site passing a keyword EXPLICITLY that the splatted builder also
emits is a duplicate-keyword ``TypeError`` at runtime -- same outage, same
seam, different mechanism.

It deliberately does NOT invoke ``write_post_training_report``: that function
renders images and writes MP4s, which is why no test in this repo calls it.
Static analysis of the actual call sites is strictly more coverage than a
hand-built invocation would give, not less.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"
COMPARISON_REPORT = REPO_ROOT / "utils" / "comparison_report.py"
# Issue #1075 (task P3) split the builder in two: the report-key assembly moved
# to its own module (with the loop's ~35 formerly-implicit closure captures as
# explicit parameters), while train_streaming.py keeps a same-named thin
# wrapper that binds them. Key extraction reads the former; call-site analysis
# reads the latter, which is still what the four report calls splat.
REPORT_KWARGS = REPO_ROOT / "eligsir" / "runtime" / "report_kwargs.py"
BUILDER_NAME = "build_streaming_report_kwargs"
SPLAT_NAME = "_streaming_report_kwargs"
WRITER_NAME = "write_post_training_report"


@pytest.fixture(scope="module")
def train_streaming_tree():
    return ast.parse(TRAIN_STREAMING.read_text())


@pytest.fixture(scope="module")
def writer_def() -> ast.FunctionDef:
    for node in ast.walk(ast.parse(COMPARISON_REPORT.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == WRITER_NAME:
            return node
    raise AssertionError(
        f"{WRITER_NAME}() not found in {COMPARISON_REPORT.name} -- retarget this "
        "test, do not delete it (it guards the #646 outage class)."
    )


def _writer_params(writer: ast.FunctionDef) -> set:
    a = writer.args
    return {
        arg.arg
        for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs)
    }


@pytest.fixture(scope="module")
def writer_params(writer_def) -> set:
    params = _writer_params(writer_def)
    # Sanity floor: a broken extraction must fail loudly, not pass vacuously.
    for anchor in ("model_path", "spf_resolution", "ingestion"):
        assert anchor in params, f"parameter extraction looks broken: {sorted(params)[:20]}"
    return params


def _builder(tree) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == BUILDER_NAME:
            return node
    raise AssertionError(f"{BUILDER_NAME}() not found in {REPORT_KWARGS.name}")


def _emitted_keys(builder: ast.FunctionDef) -> set:
    """String keys the builder can emit (dict literals + ``_extra["k"] = ...``)."""
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


@pytest.fixture(scope="module")
def builder_keys() -> set:
    keys = _emitted_keys(_builder(ast.parse(REPORT_KWARGS.read_text())))
    for anchor in ("pose_provenance", "ingestion", "seed_cohort", "spf_resolution"):
        assert anchor in keys, f"key extraction looks broken: {sorted(keys)}"
    return keys


def _splat_call_sites(tree):
    """Every call that splats the builder -- i.e. the real report call sites."""
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if (
                kw.arg is None
                and isinstance(kw.value, ast.Call)
                and isinstance(kw.value.func, ast.Name)
                and kw.value.func.id == SPLAT_NAME
            ):
                sites.append(node)
    return sites


# --- the guard --------------------------------------------------------------


def test_resolution_kwarg_is_accepted_and_persisted_by_the_writer(builder_keys, writer_params):
    """#595's own field, end to end through the seam that broke in #646."""
    assert "resolution" in builder_keys, (
        f"{BUILDER_NAME}() no longer emits 'resolution' -- #595's deliverable is "
        "report.json['resolution'] and it is now missing from every run."
    )
    assert "resolution" in writer_params, (
        f"{WRITER_NAME}() does not accept 'resolution'. This is the #646 outage "
        "exactly: every streaming run would train to completion and then die in "
        "the report phase with no report.json."
    )
    src = COMPARISON_REPORT.read_text()
    assert 'summary["resolution"]' in src, (
        f"{WRITER_NAME}() accepts 'resolution' but never writes it -- a parameter "
        "accepted and dropped is a silent half-fix (#647's own wording)."
    )


def test_every_emitted_kwarg_is_a_writer_parameter(builder_keys, writer_params):
    """Same assertion as #647's, but on a surface CI actually runs."""
    unknown = sorted(builder_keys - writer_params)
    assert not unknown, (
        f"{BUILDER_NAME}() emits report kwargs {WRITER_NAME}() does not accept: "
        f"{unknown} (issue #646)."
    )


def test_no_call_site_passes_a_keyword_the_builder_also_splats(train_streaming_tree, builder_keys):
    """A duplicate keyword is a runtime TypeError -- the #646 outage by another
    route, and one a key-subset check cannot see."""
    sites = _splat_call_sites(train_streaming_tree)
    assert len(sites) >= 3, (
        f"expected the four splatted report call sites, found {len(sites)} -- "
        "extraction is broken or the call sites were restructured."
    )
    for call in sites:
        explicit = {kw.arg for kw in call.keywords if kw.arg is not None}
        clash = sorted(explicit & builder_keys)
        assert not clash, (
            f"report call site at line {call.lineno} passes {clash} explicitly "
            f"while {BUILDER_NAME}() also emits it: duplicate keyword argument "
            "=> TypeError in the report phase of every streaming run."
        )


def test_writer_keeps_resolution_optional_and_additive(writer_def):
    """Omitting it must reproduce the pre-#595 report byte for byte (the
    ``if resolution:`` gating pattern used by ingestion / spf_resolution)."""
    a = writer_def.args
    kwonly = {arg.arg: d for arg, d in zip(a.kwonlyargs, a.kw_defaults)}
    assert "resolution" in kwonly, "'resolution' must be keyword-only, like its peers"
    default = kwonly["resolution"]
    assert isinstance(default, ast.Constant) and default.value is None


# --- the producing side is really wired into the training loop --------------


def test_train_streaming_builds_and_logs_the_provenance_record(train_streaming_tree):
    """Reachability: the record is constructed from the gate's own locals and
    printed, so train.log and report.json cannot disagree."""
    src = TRAIN_STREAMING.read_text()
    assert "build_resolution_provenance(" in src
    assert "_resolution_provenance.format_log()" in src
    # The gate's outcome is captured, not re-derived later from args.
    assert "_res_gate_fired = True" in src
    assert "hard_gate_fired=_res_gate_fired" in src


def test_tier_log_is_fed_from_decides_return_value(train_streaming_tree):
    """Must be ``_active_tier`` (``decide()``'s return), never
    ``ResolutionController._tier``: once the mapping-done lock engages the two
    diverge permanently, and the TB scalar uses the return value."""
    src = TRAIN_STREAMING.read_text()
    assert "_resolution_tier_log.observe(" in src
    idx = src.index("_resolution_tier_log.observe(")
    call = src[idx: idx + 260]
    assert "_active_tier" in call
    assert "_resolution_controller._tier" not in call
