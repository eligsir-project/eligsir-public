"""Issue #504 -- where the trajectory-coverage block lives, and when it is
emitted at all.

Two independent, separately-measured defects are pinned here:

1. **Wrong-path consumers.** ``write_post_training_report`` writes the block
   at the TOP LEVEL of report.json (``summary["trajectory_coverage"]``) and
   always has (commit ``1140f27``). Several consumers/docstrings read
   ``report["geometry"]["trajectory_coverage"]`` instead -- a path no writer
   in this repo has ever produced (measured 2026-08-14: of 400 real
   ``output/**/comparison/iter_*/report.json`` files, 397 carry the top-level
   key and **0** carry the ``geometry`` one). Those consumers degraded
   silently on EVERY run, not on some runs.

2. **Producer nesting.** The sole call site sat inside
   ``write_post_training_report``'s ``if test_cams:`` body, so a report with
   no test cameras got no block at all -- and with it no
   ``map_reinit_invalidated``, no ``stream_end_reason``, no
   ``frame_drop_detected``. Those fields are exactly the ones CLAUDE.md gives
   the same standing as ``eval_valid == False``, i.e. they matter most on a
   run whose eval is already invalid. 3 of the 400 scanned reports have no
   block for precisely this reason.

Torch-free by construction: ``utils.partition_eval`` imports torch only
lazily inside one unrelated function, and the producer-nesting guard reads
``utils/comparison_report.py`` as TEXT and parses it with ``ast`` rather than
importing it (that module does import torch at module scope). No entry in
``tests/conftest.py``'s ``_NEEDS_TORCH`` is therefore required for this file.
"""
from __future__ import annotations

import ast
import os

from eligsir.utils.partition_eval import (
    compute_trajectory_coverage,
    get_trajectory_coverage,
)


# --- 1. accessor -----------------------------------------------------------

class TestGetTrajectoryCoverage:
    def test_reads_the_real_top_level_location(self):
        block = {"verdict": "FULL", "map_reinit_invalidated": False}
        assert get_trajectory_coverage({"trajectory_coverage": block}) is block

    def test_geometry_fallback_is_defensive_only(self):
        """No real report uses this path (0 of 400 measured); the fallback
        exists so a consumer migrating off the wrong path cannot regress."""
        block = {"verdict": "PARTIAL"}
        assert get_trajectory_coverage({"geometry": {"trajectory_coverage": block}}) is block

    def test_top_level_wins_over_geometry(self):
        top = {"verdict": "FULL"}
        geo = {"verdict": "PARTIAL"}
        got = get_trajectory_coverage(
            {"trajectory_coverage": top, "geometry": {"trajectory_coverage": geo}}
        )
        assert got is top

    def test_absent_block_returns_none_and_is_never_synthesized(self):
        """"Absent" must stay distinguishable from "measured and clean" --
        a fabricated block (e.g. a defaulted stream_end_reason=="bag_complete",
        which CLAUDE.md treats as trustworthy) would be strictly worse than
        None."""
        for report in ({}, {"geometry": {}}, {"trajectory_coverage": None}, None, "nope"):
            assert get_trajectory_coverage(report) is None


# --- 2. not-measured partition semantics -----------------------------------

class TestPartitionDataNotMeasured:
    BASE = dict(scene_source="/data/tum/fr3", n_reg=2500, frames_admitted=2500)

    def test_none_partition_psnr_is_explicitly_unknown_not_partial(self):
        tc = compute_trajectory_coverage(partition_psnr=None, **self.BASE)
        assert tc["partition_data_available"] is False
        assert tc["partitions_with_test_data"] is None
        # NOT False: a run whose partitions were never measured is not
        # evidence of incomplete coverage.
        assert tc["full_coverage"] is None
        assert tc["verdict"] == "UNKNOWN_NO_PARTITION_DATA"

    def test_none_partition_psnr_still_reports_the_safety_fields(self):
        """The whole point of emitting a block for a test-camera-less run."""
        tc = compute_trajectory_coverage(
            partition_psnr=None,
            tracking_reinit_events=[{"frame": 812}],
            stream_end_reason="idle_timeout_no_completion_signal",
            frames_dropped_by_inner_queue=3,
            **self.BASE,
        )
        assert tc["map_reinit_invalidated"] is True
        assert tc["frame_drop_detected"] is True
        assert tc["stream_end_reason"] == "idle_timeout_no_completion_signal"
        # Truncation still outranks the not-measured verdict.
        assert tc["verdict"] == "SUSPECTED_TRUNCATED"
        assert tc["mapping_result_valid"] is False

    def test_never_invents_a_trustworthy_stream_end_reason(self):
        tc = compute_trajectory_coverage(partition_psnr=None, **self.BASE)
        assert tc["stream_end_reason"] is None  # not "bag_complete"

    def test_empty_dict_keeps_its_original_measured_meaning(self):
        """``{}`` == "measured, no partition had test data" -- every
        pre-existing call site must be bit-for-bit unchanged."""
        tc = compute_trajectory_coverage(partition_psnr={}, **self.BASE)
        assert tc["partition_data_available"] is True
        assert tc["partitions_with_test_data"] == 0
        assert tc["full_coverage"] is False
        assert tc["verdict"] == "PARTIAL"

    def test_full_coverage_still_reachable_with_real_partition_data(self):
        tc = compute_trajectory_coverage(
            partition_psnr={f"p{i}": 25.0 for i in range(6)}, **self.BASE
        )
        assert tc["partition_data_available"] is True
        assert tc["full_coverage"] is True
        assert tc["verdict"] == "FULL"


# --- 3. producer nesting guard ---------------------------------------------

def _write_post_training_report_ast() -> ast.FunctionDef:
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "utils", "comparison_report.py",
    )
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "write_post_training_report":
            return node
    raise AssertionError("write_post_training_report not found in utils/comparison_report.py")


def _mentions_trajectory_coverage(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Constant) and sub.value == "trajectory_coverage"
        for sub in ast.walk(node)
    )


def test_coverage_block_is_emitted_unconditionally_not_only_for_test_cams():
    """REGRESSION (issue #504): at least one ``trajectory_coverage`` write
    must sit at the FUNCTION's top level, not nested inside ``if test_cams:``.

    A structural (ast) check rather than an end-to-end one because
    ``write_post_training_report`` needs a real Gaussian model, cameras and a
    renderer -- i.e. a GPU run -- to call. This pins the one property that
    regressed: reachability with zero test cameras.
    """
    fn = _write_post_training_report_ast()
    # `if test_cams:` itself is a top-level statement of the function and
    # contains the partitioned-eval write, so an unconditional write means an
    # UNCONDITIONAL top-level statement -- not an `if`.
    top_level = [
        s for s in fn.body
        if not isinstance(s, ast.If) and _mentions_trajectory_coverage(s)
    ]
    assert top_level, (
        "no top-level trajectory_coverage write in write_post_training_report -- "
        "it is nested again (issue #504), so a report with no test cameras "
        "silently loses map_reinit_invalidated/stream_end_reason/frame_drop_detected"
    )


def test_unconditional_coverage_write_passes_partition_psnr_as_not_measured():
    """The unconditional write must not claim measured partition data. It is
    emitted before any partition PSNR exists, so it passes
    ``partition_psnr=None`` -- the honest not-measured marker -- and the
    partitioned-eval block later overwrites it when a real holdout eval ran.
    """
    fn = _write_post_training_report_ast()
    calls = [
        c
        for stmt in fn.body
        if not isinstance(stmt, ast.If) and _mentions_trajectory_coverage(stmt)
        for c in ast.walk(stmt)
        if isinstance(c, ast.Call)
        and isinstance(c.func, ast.Name)
        and c.func.id == "compute_trajectory_coverage"
    ]
    assert calls, "top-level trajectory_coverage write does not call compute_trajectory_coverage"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "partition_psnr" in kw, "partition_psnr must be passed explicitly"
        assert isinstance(kw["partition_psnr"], ast.Constant) and kw["partition_psnr"].value is None, (
            "the unconditional write must pass partition_psnr=None (not-measured); "
            "passing {} would report a confident PARTIAL/full_coverage=False for a "
            "run whose partitions were never measured"
        )
