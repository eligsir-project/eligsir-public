"""Issue #282's telemetry checker: does a run's seed_fade block deserve trust?

Two kinds of input are exercised, deliberately:

1. **Real, post-fix blocks** produced by replaying the two fr3 anchor runs'
   VERBATIM ``optimization/seed_psnr`` series (``tests/fixtures/seed_fade/
   real_seed_psnr_series.json``, extracted from their own TB event files)
   through the real ``SeedFadeMonitor`` at their real wiring. These must
   certify. If the checker cannot certify the monitor's own honest output,
   the checker is wrong.

2. **The pre-fix blocks those same two runs actually shipped**, transcribed
   from the audit doc. These must be REFUSED -- that is the whole point of
   the provenance check, and it is what lets #283-#290 tell "measured clean"
   apart from "never measured".

Synthetic blocks are used only for the defect signatures that no real run on
disk exhibits any more (the aliased controller_engaged, an observe-only run
that nonetheless applied suppression, broken episode arithmetic).
"""

from __future__ import annotations

import json
import os

import pytest

from eligsir.utils.seed_fade_checker import (
    REQUIRED_282_KEYS,
    check_run_dir,
    check_seed_fade_block,
    classify_schema,
)
from eligsir.utils.seed_fade_monitor import SeedFadeMonitor

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "seed_fade", "real_seed_psnr_series.json"
)
ANCHORS = ("issue278mode_iterbased_spf20", "fr3postmerge_champ_asis")


def _series(run):
    with open(_FIXTURE) as f:
        return json.load(f)["series"][run]


def _replay(run, **kw):
    """Replay at the runs' real wiring: update() every scalar_log_interval
    (=100) iterations, suppression pulled once per iteration in between."""
    mon = SeedFadeMonitor(warmup_iters=2000, sample_interval=100, **kw)
    prev = 0
    for it, val in _series(run):
        for _ in range(max(1, it - prev)):
            mon.effective_suppression_scale()
        prev = it
        mon.update(it, val)
    return mon


# --- the pre-fix blocks these two runs really shipped (audit doc, verbatim) --

SHIPPED_PRE_282 = {
    "issue278mode_iterbased_spf20": {
        "detected": True, "max_dip_db": 15.56, "fade_start_iter": 21500,
        "recovered": True, "controller_engaged": True,
    },
    "fr3postmerge_champ_asis": {
        "detected": True, "max_dip_db": 2.93, "fade_start_iter": 28900,
        "recovered": True, "controller_engaged": True,
    },
}


# =========================== provenance / refusal ===========================


def test_missing_block_is_refused():
    v = check_seed_fade_block(None)
    assert v.schema == "missing"
    assert v.certified is False
    assert v.refusals


@pytest.mark.parametrize("run", ANCHORS)
def test_the_blocks_these_runs_actually_shipped_are_refused(run):
    """The headline provenance check: pre-#282 telemetry is UNMEASURED, not
    clean, and must never be certified for #283-#290."""
    v = check_seed_fade_block(SHIPPED_PRE_282[run])
    assert v.schema == "pre_282"
    assert v.certified is False
    assert any("predates issue #282" in r for r in v.refusals)
    # No internal contradiction is claimed -- there is nothing to contradict.
    assert v.ok is True


def test_pre_282_controller_engaged_is_called_out_as_information_free():
    v = check_seed_fade_block(SHIPPED_PRE_282["fr3postmerge_champ_asis"])
    assert any("alias of `detected`" in w for w in v.warnings)


def test_schema_discriminator_needs_every_282_key():
    real = _replay(ANCHORS[0]).report_block()
    assert classify_schema(real) == "post_282"
    for key in REQUIRED_282_KEYS:
        partial = {k: v for k, v in real.items() if k != key}
        assert classify_schema(partial) == "pre_282", key


# ====================== real post-fix blocks must certify ====================


@pytest.mark.parametrize("run", ANCHORS)
def test_real_replayed_block_certifies(run):
    mon = _replay(run)
    v = check_seed_fade_block(mon.report_block())
    assert v.errors == [], v.errors
    assert v.schema == "post_282"
    assert v.certified is True, v.refusals


@pytest.mark.parametrize("run", ANCHORS)
def test_real_replayed_block_surfaces_the_sticky_latch_shape(run):
    """Both anchors end INSIDE an active fade while `recovered` is True.
    That is not an inconsistency -- it is the misreading that was. The
    checker must name it in warnings and still certify."""
    blk = _replay(run).report_block()
    assert blk["recovered"] is True
    assert blk["fade_active_at_end"] is True
    assert blk["recovered_final"] is False
    v = check_seed_fade_block(blk)
    assert v.certified is True
    assert any("ended INSIDE an active fade" in w for w in v.warnings)
    assert any("never `recovered`" in w for w in v.warnings)


@pytest.mark.parametrize("run", ANCHORS)
def test_multi_episode_runs_warn_that_fade_start_iter_is_the_last_onset(run):
    blk = _replay(run).report_block()
    assert blk["episode_count"] > 1
    v = check_seed_fade_block(blk)
    assert any("is the LAST onset" in w for w in v.warnings)


@pytest.mark.parametrize("run", ANCHORS)
def test_observe_only_replay_certifies_and_is_not_suppression_confounded(run):
    mon = _replay(run, observe_only=True)
    blk = mon.report_block()
    assert blk["suppression_applied"] is False
    assert blk["suppression_confounded"] is False
    v = check_seed_fade_block(blk)
    assert v.errors == [], v.errors
    assert v.certified is True


def test_sample_interval_comparability_is_warned():
    v = check_seed_fade_block(_replay(ANCHORS[0]).report_block())
    assert any("NOT comparable" in w for w in v.warnings)


# ========================= defect-signature detection ========================


def _good():
    """A minimal, self-consistent post-#282 block to mutate."""
    return dict(_replay(ANCHORS[1]).report_block())


def test_dead_producer_zero_samples_is_refused():
    mon = SeedFadeMonitor(warmup_iters=2000, sample_interval=100)
    # Every sample lands inside warmup -> the monitor is never really fed.
    for it in range(100, 2000, 100):
        mon.update(it, 20.0)
    blk = mon.report_block()
    assert blk["n_samples"] == 0
    v = check_seed_fade_block(blk)
    assert v.certified is False
    assert any("NEVER RAN" in r for r in v.refusals)
    assert v.errors == []


def test_aliased_controller_engaged_is_an_error():
    blk = _good()
    blk["controller_engaged"] = blk["detected"]  # the pre-fix alias
    blk["suppression_applied"] = False
    blk["status"] = "FADE_OBSERVED"
    blk["suppression_confounded"] = False
    v = check_seed_fade_block(blk)
    assert v.certified is False
    assert any("aliased-field defect" in e for e in v.errors)


def test_observe_only_that_applied_suppression_is_an_error():
    blk = _good()
    blk["observe_only"] = True
    assert blk["suppression_applied"] is True
    v = check_seed_fade_block(blk)
    assert any("observe-only mode must never" in e for e in v.errors)


def test_observe_only_with_a_throttling_applied_scale_is_an_error():
    blk = _good()
    blk["observe_only"] = True
    blk["suppression_applied"] = False
    blk["last_applied_scale"] = 0.4
    v = check_seed_fade_block(blk)
    assert any("must be exactly 1.0" in e for e in v.errors)


def test_recovered_final_contradicting_fade_active_at_end_is_an_error():
    blk = _good()
    blk["recovered_final"] = False
    blk["fade_active_at_end"] = False
    v = check_seed_fade_block(blk)
    assert any("contradicts" in e for e in v.errors)


def test_episode_count_mismatch_is_an_error():
    blk = _good()
    blk["episode_count"] = blk["episode_count"] + 1
    v = check_seed_fade_block(blk)
    assert any("episode_count" in e for e in v.errors)


def test_open_episode_without_active_fade_is_an_error():
    blk = _good()
    blk["fade_active_at_end"] = False
    blk["recovered_final"] = True
    v = check_seed_fade_block(blk)
    assert any("still open" in e for e in v.errors)


def test_fade_sample_fraction_arithmetic_is_checked():
    blk = _good()
    blk["fade_sample_fraction"] = 0.01
    v = check_seed_fade_block(blk)
    assert any("fade_sample_fraction" in e for e in v.errors)


def test_deepest_dip_smaller_than_max_dip_is_an_error():
    blk = _good()
    blk["deepest_dip_db"] = blk["max_dip_db"] - 1.0
    v = check_seed_fade_block(blk)
    assert any("deepest_dip_db" in e for e in v.errors)


def test_status_must_follow_from_detected_and_suppression():
    blk = _good()
    blk["status"] = "CLEAN"
    v = check_seed_fade_block(blk)
    assert any("status='CLEAN'" in e for e in v.errors)


# ================= the two same-named confounding flags ======================


def test_diverging_confounding_flags_are_explained_not_conflated():
    """severity-confounded and suppression-confounded are different
    predicates. Divergence is legitimate and must be explained, not errored."""
    blk = _good()
    blk["suppression_confounded"] = False
    blk["suppression_applied"] = False
    blk["status"] = "FADE_OBSERVED"
    blk["controller_engaged"] = False
    blk["confounded"] = True  # #346 severity verdict
    v = check_seed_fade_block(blk)
    assert v.errors == [], v.errors
    assert any("must not be conflated" in w for w in v.warnings)


def test_top_level_mirror_must_match_the_nested_severity_verdict():
    blk = _good()
    blk["confounded"] = True
    v = check_seed_fade_block(blk, report={"seed_fade_confounded": False})
    assert any("top-level key is defined as a mirror" in e for e in v.errors)


# ============================== run-dir entry ================================


def test_check_run_dir_uses_numeric_iter_ordering(tmp_path):
    """The online-mode sentinel iter_2147483647 sorts BEFORE iter_30000
    lexically. The final report is the sentinel one, and it is the one whose
    verdict must be returned."""
    run = str(tmp_path)
    comp = os.path.join(run, "comparison")
    for it, block in (
        ("30000", SHIPPED_PRE_282["fr3postmerge_champ_asis"]),   # pre-282 -> refused
        ("2147483647", _replay(ANCHORS[1]).report_block()),       # post-282 -> certified
    ):
        d = os.path.join(comp, f"iter_{it}")
        os.makedirs(d)
        with open(os.path.join(d, "report.json"), "w") as f:
            json.dump({"seed_fade": block}, f)
    # A degenerate debug split that must be ignored entirely.
    d = os.path.join(comp, "iter_2147483647_bootstrap_views")
    os.makedirs(d)
    with open(os.path.join(d, "report.json"), "w") as f:
        json.dump({"seed_fade": None}, f)

    v = check_run_dir(run)
    assert v.schema == "post_282"
    assert v.certified is True, (v.errors, v.refusals)


def test_check_run_dir_on_a_run_with_no_report_is_refused(tmp_path):
    v = check_run_dir(str(tmp_path))
    assert v.certified is False
    assert any("no comparison/iter_" in r for r in v.refusals)


# ================ the COMPOSED block that really lands on disk ===============
#
# `write_post_training_report` does not write `report_block()` -- it writes
# `grade_seed_fade_for_run(report_block(), model_path)`, i.e. the monitor's
# keys PLUS #346's severity/confounded/curve-shape fields. That composed shape
# is what #283-#290 will actually hand this checker, so certifying only the
# raw block would leave the production path unvalidated.


def _composed(run):
    from eligsir.utils.seed_fade_severity import grade_seed_fade

    series = _series(run)
    blk = _replay(run).report_block()
    return grade_seed_fade(blk, [i for i, _ in series], [v for _, v in series])


@pytest.mark.parametrize("run", ANCHORS)
def test_composed_report_block_still_certifies(run):
    composed = _composed(run)
    # the #346 layer really did add its fields on top of the monitor's
    assert "severity" in composed and "confounded" in composed
    assert composed["status"] and composed["n_samples"] > 0
    v = check_seed_fade_block(composed)
    assert v.errors == [], v.errors
    assert v.certified is True


@pytest.mark.parametrize("run", ANCHORS)
def test_top_level_mirror_disagreeing_with_the_composed_block_is_an_error(run):
    composed = _composed(run)
    good = check_seed_fade_block(
        composed, report={"seed_fade_confounded": bool(composed["confounded"])})
    assert good.errors == []
    bad = check_seed_fade_block(
        composed, report={"seed_fade_confounded": not bool(composed["confounded"])})
    assert any("mirror" in e for e in bad.errors)
