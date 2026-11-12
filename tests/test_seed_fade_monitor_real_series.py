"""Issue #282: replay two REAL runs' seed-PSNR series through the monitor.

No GPU needed and no synthetic data: `tests/fixtures/seed_fade/
real_seed_psnr_series.json` is the verbatim `optimization/seed_psnr` scalar
series extracted from two fr3 runs' own TensorBoard event files, and the
ground truth below was independently reconstructed from those same event
files (`streaming/seed_fade_active` for the episode structure, the runs'
`train.log` `[SEED-FADE WARNING]` lines for the onsets, and the runs'
`report.json` for what the OLD report block said).

The point of these tests is not only regression-pinning: they are the
evidence for the audit. Both runs' shipped report blocks read
``{detected: true, max_dip_db: ..., recovered: true, controller_engaged:
true}`` and differed only in a single float, while the underlying
trajectories differ by 5x in depth, by 17 vs 8 episodes, and by 57500
iterations in where the deepest dip actually happened. Both also ended
*inside* an active fade while reporting ``recovered: true``.
"""

import json
import os

import pytest

from eligsir.utils.seed_fade_monitor import SeedFadeMonitor, STATUS_CONFOUNDED

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "seed_fade", "real_seed_psnr_series.json"
)


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


# --------------------------------------------------------------------------
# issue278mode_iterbased_spf20 -- shipped report.json said
#   {detected: true, max_dip_db: 15.56, fade_start_iter: 21500,
#    recovered: true, controller_engaged: true}
# --------------------------------------------------------------------------
def test_issue278_reproduces_the_shipped_legacy_block():
    blk = _replay("issue278mode_iterbased_spf20").report_block()
    assert blk["detected"] is True
    assert blk["max_dip_db"] == pytest.approx(15.56, abs=0.01)
    assert blk["fade_start_iter"] == 21500
    assert blk["recovered"] is True


def test_issue278_new_telemetry_exposes_what_the_legacy_block_hid():
    blk = _replay("issue278mode_iterbased_spf20").report_block()

    # 1. The reported `fade_start_iter` (21500) is the LAST of 17 episodes.
    #    The run's train.log printed its FIRST [SEED-FADE WARNING] at 3800,
    #    and its own iter_7000 report.json recorded fade_start_iter=3800 --
    #    the same field, same run, two different answers.
    assert blk["episode_count"] == 17
    assert blk["first_fade_start_iter"] == 3800
    assert blk["last_fade_start_iter"] == 21500

    # 2. The deepest dip is at iteration 79000, 57500 iterations AFTER the
    #    reported onset -- and inside the post-ingestion recovery ramp, while
    #    the run's global test PSNR was climbing 18.6 -> 19.2 dB.
    assert blk["trough_iter"] == 79000
    assert blk["deepest_dip_db"] == pytest.approx(15.56, abs=0.01)
    assert blk["trough_psnr"] == pytest.approx(11.215, abs=0.01)
    assert blk["trough_reference_psnr"] == pytest.approx(26.779, abs=0.01)

    # 3. `recovered: true` is a sticky latch from the first episode ending.
    #    This run ended INSIDE an active fade that started at 21500 and never
    #    closed -- 81.7% of all post-warmup samples were in-fade.
    assert blk["fade_active_at_end"] is True
    assert blk["recovered_final"] is False
    #    (817 of the 1000 logged samples; 981 of them are post-warmup, which
    #    is the denominator the monitor uses.)
    assert blk["n_samples_in_fade"] == 817
    assert blk["n_samples"] == 981
    assert blk["fade_sample_fraction"] == pytest.approx(817 / 981, abs=0.001)

    # 4. Suppression really was applied, and pinned at the floor.
    assert blk["suppression_applied"] is True
    assert blk["min_applied_scale"] == 0.0
    assert blk["last_applied_scale"] == 0.0
    assert blk["status"] == STATUS_CONFOUNDED
    assert blk["suppression_confounded"] is True

    # 5. Detection sensitivity is expressed in SAMPLES, so its real latency
    #    depends on scalar_log_interval -- 300 iterations here, 1500 for a
    #    run logging every 500. Recorded so runs are comparable.
    assert blk["detection_latency_iters"] == 300


# --------------------------------------------------------------------------
# fr3postmerge_champ_asis (the champion baseline) -- shipped report.json said
#   {detected: true, max_dip_db: 2.93, fade_start_iter: 28900,
#    recovered: true, controller_engaged: true}
# --------------------------------------------------------------------------
def test_champ_asis_reproduces_the_shipped_legacy_block():
    blk = _replay("fr3postmerge_champ_asis").report_block()
    assert blk["detected"] is True
    assert blk["max_dip_db"] == pytest.approx(2.93, abs=0.01)
    assert blk["fade_start_iter"] == 28900
    assert blk["recovered"] is True


def test_champ_asis_also_ended_in_an_unrecovered_fade():
    blk = _replay("fr3postmerge_champ_asis").report_block()
    assert blk["episode_count"] == 8
    assert blk["first_fade_start_iter"] == 9500
    assert blk["last_fade_start_iter"] == 28900
    assert blk["trough_iter"] == 31200
    assert blk["fade_active_at_end"] is True
    assert blk["recovered_final"] is False
    # The champion baseline spent 376 of its 493 post-warmup samples (76%)
    # with its anti-overgrowth losses throttled by the detector, and its
    # report said nothing about it. (512 logged samples, 19 pre-warmup.)
    assert blk["n_samples_in_fade"] == 376
    assert blk["n_samples"] == 493
    assert blk["fade_sample_fraction"] > 0.7
    assert blk["status"] == STATUS_CONFOUNDED


def test_the_two_runs_were_indistinguishable_under_the_legacy_block():
    """The concrete failure of expressiveness that motivated #282."""
    a = _replay("issue278mode_iterbased_spf20").report_block()
    b = _replay("fr3postmerge_champ_asis").report_block()
    legacy_booleans = ("detected", "recovered", "controller_engaged")
    for k in legacy_booleans:
        assert a[k] == b[k], "precondition of the finding"
    # ...while the real trajectories differ enormously.
    assert a["episode_count"] != b["episode_count"]
    assert abs(a["deepest_dip_db"] - b["deepest_dip_db"]) > 12.0
    assert abs(a["trough_iter"] - b["trough_iter"]) > 45000


def test_observe_only_would_not_have_changed_either_measurement():
    """Sanity check for the causal program: replaying the same recorded
    series in observe-only mode yields the identical fade measurement with
    zero suppression. (In a REAL run the scene would evolve differently --
    that is the experiment #283+ has to run; this only pins that the
    detector itself is mode-independent.)"""
    for run in ("issue278mode_iterbased_spf20", "fr3postmerge_champ_asis"):
        ctrl = _replay(run).report_block()
        obs = _replay(run, observe_only=True).report_block()
        for k in ("detected", "max_dip_db", "episode_count", "trough_iter",
                  "deepest_dip_db", "fade_active_at_end", "n_samples_in_fade"):
            assert ctrl[k] == obs[k]
        assert obs["suppression_applied"] is False
        assert obs["min_applied_scale"] == 1.0
        assert obs["suppression_confounded"] is False
