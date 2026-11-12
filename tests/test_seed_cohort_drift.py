"""Tests for utils/seed_cohort_drift.py (issue #287, intrinsic arm).

The load-bearing test here is that a SYNTHETIC CONTAMINATED checkpoint — one
that went through a mid-run ``_reset_running_state()`` — is REFUSED. Issue #358
had to publish a retraction (PR #379) for measuring confidently on a wrong
substrate; the gate exists so that cannot recur here.
"""
import os
import tempfile

import pytest
import torch

from eligsir.utils.gaussian_lineage import (
    BIRTH_METHOD_BOOTSTRAP,
    BIRTH_METHOD_MCMC_GROW,
    BIRTH_METHOD_SOURCE_DEPTH,
    SeedCohortUnavailable,
)
from eligsir.utils.seed_cohort_drift import (
    analyze_run,
    birth_cohort_masks,
    check_uid_join,
    cohort_stats,
    compare_cohort_definitions,
    lineage_trust_report,
    list_checkpoints,
    load_checkpoint,
    paired_cohort_drift,
    seed_cohort_mask,
    seed_psnr_at,
    standin_seed_cohort_mask,
)


# --------------------------------------------------------------------------
# synthetic checkpoint builders
# --------------------------------------------------------------------------

def _params(n, seed=0, mean_shift=0.0, opacity_logit=2.0, log_scale=-3.0):
    g = torch.Generator().manual_seed(seed)
    return {
        "means": torch.rand(n, 3, generator=g) + mean_shift,
        "opacities": torch.full((n,), float(opacity_logit)),
        "quats": torch.zeros(n, 4),
        "scales": torch.full((n, 3), float(log_scale)),
        "sh0": torch.zeros(n, 1, 3),
        "shN": torch.zeros(n, 15, 3),
    }


def make_state(*, n_seed, n_later, iteration, next_uid=None, uid_start=0,
               contaminated=False, drop_lineage=(), bad_length=None,
               mean_shift=0.0, opacity_logit=2.0, seed=0, later_birth_iter=None,
               seed_uid_range="auto"):
    """Build a ``(state, iteration, meta)`` checkpoint tuple.

    ``contaminated=True`` simulates ``_reset_running_state()`` firing mid-run:
    every row is restamped BOOTSTRAP / birth_iter=0 / root_uid=own uid /
    generation=0 and given a fresh contiguous uid block — exactly the issue #283
    signature, including on rows that were genuinely born later.

    ``seed_uid_range`` defaults to the real bootstrap block
    ``(uid_start, uid_start + n_seed)``; pass ``None`` to simulate a pre-#373
    checkpoint, which carries the lineage buffers but not that key.
    """
    n = n_seed + n_later
    later_birth_iter = later_birth_iter if later_birth_iter is not None else max(1, iteration // 2)
    if contaminated:
        birth_iter = torch.zeros(n, dtype=torch.int32)
        birth_method = torch.full((n,), BIRTH_METHOD_BOOTSTRAP, dtype=torch.int8)
        generation = torch.zeros(n, dtype=torch.int32)
        uid = torch.arange(uid_start, uid_start + n, dtype=torch.int64)
        root_uid = uid.clone()
        parent_uid = torch.full((n,), -1, dtype=torch.int64)
    else:
        birth_iter = torch.cat([
            torch.zeros(n_seed, dtype=torch.int32),
            torch.full((n_later,), int(later_birth_iter), dtype=torch.int32),
        ])
        birth_method = torch.cat([
            torch.full((n_seed,), BIRTH_METHOD_BOOTSTRAP, dtype=torch.int8),
            torch.full((n_later,), BIRTH_METHOD_SOURCE_DEPTH, dtype=torch.int8),
        ])
        generation = torch.zeros(n, dtype=torch.int32)
        uid = torch.arange(uid_start, uid_start + n, dtype=torch.int64)
        root_uid = uid.clone()
        parent_uid = torch.full((n,), -1, dtype=torch.int64)

    lineage = {
        "gaussian_uid": uid,
        "parent_uid": parent_uid,
        "root_uid": root_uid,
        "generation": generation,
        "birth_method": birth_method,
        "birth_iter": birth_iter,
        "last_reseed_iter": torch.full((n,), -1, dtype=torch.int32),
    }
    for name in drop_lineage:
        lineage.pop(name, None)
    if bad_length:
        name, new_len = bad_length
        lineage[name] = lineage[name][:new_len]

    if seed_uid_range == "auto":
        seed_uid_range = (int(uid_start), int(uid_start + n_seed))

    state = {
        "layout": "gsplat",
        "active_sh_degree": 3,
        "seed_uid_range": seed_uid_range,
        "next_gaussian_uid": int(next_uid if next_uid is not None else uid_start + n),
        "params": _params(n, seed=seed, mean_shift=mean_shift, opacity_logit=opacity_logit),
        "optimizers": {},
        "spatial_lr_scale": 1.0,
        "visibility_ema": None,
        "streaming_state": lineage,
    }
    return (state, int(iteration), {"n_frames_ingested": 1})


def write_ckpt(d, state_tuple):
    path = os.path.join(d, f"chkpnt{state_tuple[1]}.pth")
    torch.save(state_tuple, path)
    return path


@pytest.fixture()
def tmp_run():
    with tempfile.TemporaryDirectory() as d:
        yield d


# --------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------

def test_clean_checkpoint_is_trustworthy(tmp_run):
    p = write_ckpt(tmp_run, make_state(n_seed=100, n_later=40, iteration=2000))
    rec = load_checkpoint(p)
    report = lineage_trust_report(rec)
    assert report["trustworthy"], report["reasons"]
    assert report["reasons"] == []


def test_contaminated_checkpoint_is_refused(tmp_run):
    """THE load-bearing test: a mid-run _reset_running_state() must be refused."""
    p = write_ckpt(tmp_run, make_state(n_seed=100, n_later=40, iteration=4000,
                                       contaminated=True, uid_start=9000, next_uid=9140))
    rec = load_checkpoint(p)
    report = lineage_trust_report(rec)
    assert report["trustworthy"] is False
    joined = " ".join(report["reasons"])
    assert "birth_iter_degenerate" in joined
    assert "all_bootstrap_signature" in joined


def test_missing_lineage_buffers_are_refused(tmp_run):
    p = write_ckpt(tmp_run, make_state(n_seed=50, n_later=10, iteration=2000,
                                       drop_lineage=("root_uid", "birth_method")))
    report = lineage_trust_report(load_checkpoint(p))
    assert report["trustworthy"] is False
    assert any(r.startswith("lineage_absent") for r in report["reasons"])
    assert "root_uid" in report["reasons"][0] and "birth_method" in report["reasons"][0]


def test_length_mismatch_is_refused(tmp_run):
    """The real w3_champion_fr3_baseline signature: birth_iter shorter than N."""
    p = write_ckpt(tmp_run, make_state(n_seed=50, n_later=10, iteration=2000,
                                       bad_length=("birth_iter", 40)))
    report = lineage_trust_report(load_checkpoint(p))
    assert report["trustworthy"] is False
    assert any(r.startswith("lineage_length_mismatch") for r in report["reasons"])


def test_empty_early_birth_interval_is_refused(tmp_run):
    """Mid-run reset at T: pile at birth_iter==0 with (0, T) empty."""
    p = write_ckpt(tmp_run, make_state(n_seed=200, n_later=20, iteration=4000,
                                       later_birth_iter=3800))
    report = lineage_trust_report(load_checkpoint(p))
    assert report["trustworthy"] is False
    assert any("birth_iter_degenerate" in r for r in report["reasons"])


def test_uid_contiguity_is_advisory_never_a_refusal(tmp_run):
    """Regression guard: a near-contiguous uid block starting well above 0 is
    NORMAL on this streaming path and must not cause a refusal.

    Counter-example on disk: output/w3_champion_fr3_baseline_20260810/chkpnt2000.pth
    has uid 5834..15626 yet predates lineage tracking entirely.
    """
    p = write_ckpt(tmp_run, make_state(n_seed=100, n_later=40, iteration=2000,
                                       uid_start=5834, next_uid=5974))
    report = lineage_trust_report(load_checkpoint(p))
    assert report["trustworthy"] is True
    assert report["advisory"]["uid_min"] == 5834
    assert report["advisory"]["uid_gaps_in_span"] == 0


# --------------------------------------------------------------------------
# uid join
# --------------------------------------------------------------------------

def test_uid_join_ok_across_growth(tmp_run):
    a = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=100, n_later=20,
                                                       iteration=2000, next_uid=120)))
    b = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=100, n_later=60,
                                                       iteration=4000, next_uid=160)))
    join = check_uid_join(a, b)
    assert join["ok"], join["reasons"]
    assert join["n_survived"] == 120
    assert join["n_new"] == 40
    assert join["n_pruned"] == 0


def test_uid_join_detects_reset_between_checkpoints(tmp_run):
    a = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=100, n_later=20,
                                                       iteration=2000, next_uid=120)))
    # reset reissued a disjoint uid block
    b = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=120, n_later=0,
                                                       iteration=4000, uid_start=500,
                                                       next_uid=620)))
    join = check_uid_join(a, b)
    assert join["ok"] is False
    assert any("DISJOINT" in r for r in join["reasons"])


# --------------------------------------------------------------------------
# cohorts + stats
# --------------------------------------------------------------------------

def test_seed_cohort_mask_is_now_the_exact_definition(tmp_run):
    """Issue #416 step 4: this module reports the EXACT cohort, not its old
    stand-in. The two differ in both directions, which is why the swap is a
    real change and not a rename."""
    rec = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=100, n_later=40,
                                                         iteration=2000)))
    assert int(seed_cohort_mask(rec).sum()) == 100
    assert int(standin_seed_cohort_mask(rec).sum()) == 100

    # (a) birth_iter is IRRELEVANT to the exact mask (it keys on the uid block),
    #     but drops the row from the stand-in.
    rec["lineage"]["birth_iter"][0] = 7
    assert int(seed_cohort_mask(rec).sum()) == 100
    assert int(standin_seed_cohort_mask(rec).sum()) == 99
    rec["lineage"]["birth_iter"][0] = 0

    # (b) a non-bootstrap birth_method drops the row from BOTH.
    rec["lineage"]["birth_method"][0] = BIRTH_METHOD_MCMC_GROW
    assert int(seed_cohort_mask(rec).sum()) == 99
    assert int(standin_seed_cohort_mask(rec).sum()) == 99
    rec["lineage"]["birth_method"][0] = BIRTH_METHOD_BOOTSTRAP

    # (c) THE discriminating clause: a slot MCMC relocation revived keeps its
    #     own gaussian_uid but adopts the donor's root_uid. The stand-in still
    #     counts it as a seed Gaussian; the exact mask does not.
    rec["lineage"]["root_uid"][1] = 999999
    assert int(seed_cohort_mask(rec).sum()) == 99
    assert int(standin_seed_cohort_mask(rec).sum()) == 100

    cmp = compare_cohort_definitions(rec)
    assert cmp["exact_alive"] == 99
    assert cmp["standin_alive"] == 100
    assert cmp["in_standin_not_exact"] == 1
    assert cmp["in_exact_not_standin"] == 0
    assert cmp["masks_identical"] is False


def test_pre_373_checkpoint_is_refused_not_approximated(tmp_run):
    """A checkpoint without seed_uid_range must RAISE, not silently fall back
    to the stand-in — that fallback is exactly what #416 step 4 retired."""
    rec = load_checkpoint(write_ckpt(tmp_run, make_state(
        n_seed=100, n_later=40, iteration=2000, seed_uid_range=None)))
    with pytest.raises(SeedCohortUnavailable) as exc:
        seed_cohort_mask(rec)
    assert "predates the SEED uid range" in str(exc.value)
    # the stand-in still evaluates, and the comparison reports the refusal
    # verbatim rather than dropping the row
    cmp = compare_cohort_definitions(rec)
    assert cmp["standin_alive"] == 100
    assert cmp["exact_alive"] is None
    assert "SeedCohortUnavailable" in cmp["exact_error"]


def test_analyze_run_reports_the_refusal_instead_of_crashing(tmp_run):
    write_ckpt(tmp_run, make_state(n_seed=100, n_later=40, iteration=2000,
                                   seed_uid_range=None))
    res = analyze_run(tmp_run, read_tb=False)
    assert res["n_trustworthy_checkpoints"] == 0
    assert res["reportable"] is False
    entry = res["checkpoints"][0]
    assert entry["lineage_untrustworthy"] is True
    assert any("seed_cohort_unavailable" in r for r in entry["reasons"])
    assert "EXACT" in res["cohort_definition"]


def test_seed_uid_block_is_recorded_as_the_bootstrap_count(tmp_run):
    """Pins the arithmetic behind the real substrate's ``(5804, 11608)``:
    ``hi - lo`` is the bootstrap population, and ``lo`` is nonzero because
    ``create_from_pcd``'s reset allocates a first uid block that
    ``training_setup``'s reset then supersedes (see
    ``GsplatGaussianModel._record_seed_uid_range``)."""
    rec = load_checkpoint(write_ckpt(tmp_run, make_state(
        n_seed=5804, n_later=0, iteration=2000, uid_start=5804)))
    lo, hi = rec["seed_uid_range"]
    assert (lo, hi) == (5804, 11608)
    assert hi - lo == 5804
    assert int(seed_cohort_mask(rec).sum()) == 5804


def test_birth_cohort_bins_partition_the_population(tmp_run):
    rec = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=100, n_later=40,
                                                         iteration=8000,
                                                         later_birth_iter=3000)))
    masks = birth_cohort_masks(rec, (2000, 6000))
    assert set(masks) == {"seed", "born_1_to_2000", "born_2001_to_6000", "born_6001_plus"}
    total = sum(int(m.sum()) for m in masks.values())
    assert total == rec["n"]
    assert int(masks["born_2001_to_6000"].sum()) == 40


def test_birth_cohort_bins_partition_is_now_only_coincidental(tmp_run):
    """After #416 step 4 the ``seed`` bin is keyed on the uid block / root_uid /
    birth_method while the ``born_*`` bins are still keyed on ``birth_iter`` —
    two different axes. A seed row with a nonzero birth_iter would land in BOTH
    and be double-counted.

    Not violated on any real substrate measured (0/78 checkpoints of
    issue346q2attrib_S0's dense window), so this pins the current behaviour and
    documents the hazard rather than asserting a structural guarantee that no
    longer holds.
    """
    rec = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=100, n_later=40,
                                                         iteration=8000,
                                                         later_birth_iter=3000)))
    masks = birth_cohort_masks(rec, (2000, 6000))
    assert sum(int(m.sum()) for m in masks.values()) == rec["n"]

    # ...but make ONE seed row look born-later and the sum over-counts by one.
    rec["lineage"]["birth_iter"][0] = 3000
    masks = birth_cohort_masks(rec, (2000, 6000))
    assert int(masks["seed"].sum()) == 100          # exact mask ignores birth_iter
    assert sum(int(m.sum()) for m in masks.values()) == rec["n"] + 1


def test_cohort_stats_activate_opacity_and_scale(tmp_run):
    """Raw checkpoint scales are LOG and opacities are LOGIT; stats must not
    report the raw values (a silent-wrongness trap)."""
    rec = load_checkpoint(write_ckpt(tmp_run, make_state(
        n_seed=64, n_later=0, iteration=2000, opacity_logit=2.0, seed=3)))
    st = cohort_stats(rec, seed_cohort_mask(rec))
    assert st["count"] == 64
    assert st["opacity"]["median"] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-5)
    # log_scale -3.0 -> metric scale exp(-3) ~= 0.0498, isotropic -> ratios ~1
    assert st["scale_max_axis"]["median"] == pytest.approx(0.049787, abs=1e-4)
    assert st["shape"]["needle_s2_s1_median"] == pytest.approx(1.0, abs=1e-5)
    assert st["shape"]["disk_s3_s2_median"] == pytest.approx(1.0, abs=1e-5)
    assert st["birth_method_counts"] == {"bootstrap": 64}


def test_cohort_stats_empty_cohort():
    rec = {"n": 0, "lineage": {}, "opacities": torch.zeros(0), "scales": torch.zeros(0, 3),
           "means": torch.zeros(0, 3), "sh0": torch.zeros(0, 3)}
    assert cohort_stats(rec, torch.zeros(0, dtype=torch.bool)) == {"count": 0}


# --------------------------------------------------------------------------
# paired drift
# --------------------------------------------------------------------------

def test_paired_drift_measures_the_same_individuals(tmp_run):
    a = load_checkpoint(write_ckpt(tmp_run, make_state(
        n_seed=50, n_later=10, iteration=2000, opacity_logit=2.0, seed=1)))
    b = load_checkpoint(write_ckpt(tmp_run, make_state(
        n_seed=50, n_later=30, iteration=4000, opacity_logit=-2.0, seed=1,
        mean_shift=0.25)))
    d = paired_cohort_drift(a, b, "seed")
    assert d["n_matched"] == 50
    assert d["survival_frac"] == 1.0
    # every seed row moved by exactly 0.25 along all three axes
    assert d["position_drift_m"]["median"] == pytest.approx(0.25 * (3 ** 0.5), abs=1e-4)
    # logit 2.0 -> -2.0
    expect = torch.sigmoid(torch.tensor(-2.0)) - torch.sigmoid(torch.tensor(2.0))
    assert d["opacity_delta"]["median"] == pytest.approx(expect.item(), abs=1e-5)


def test_paired_drift_works_for_later_born_cohorts_too(tmp_run):
    """#287 asks for seed-vs-later-born drift; drift is only obtainable from the
    uid join, so every cohort must be joinable, not just ``seed``."""
    a = load_checkpoint(write_ckpt(tmp_run, make_state(
        n_seed=20, n_later=30, iteration=2000, later_birth_iter=1500,
        opacity_logit=2.0, seed=5)))
    b = load_checkpoint(write_ckpt(tmp_run, make_state(
        n_seed=20, n_later=30, iteration=4000, later_birth_iter=1500,
        opacity_logit=-1.0, seed=5, mean_shift=0.5)))
    d = paired_cohort_drift(a, b, "born_1_to_2000", edges=(2000, 6000))
    assert d["n_prev"] == 30 and d["n_matched"] == 30
    assert d["position_drift_m"]["median"] == pytest.approx(0.5 * (3 ** 0.5), abs=1e-4)
    with pytest.raises(ValueError):
        paired_cohort_drift(a, b, "not_a_cohort")


def test_paired_drift_survival_below_one_when_seed_rows_pruned(tmp_run):
    a = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=50, n_later=0,
                                                       iteration=2000, next_uid=50)))
    b = load_checkpoint(write_ckpt(tmp_run, make_state(n_seed=40, n_later=0,
                                                       iteration=4000, next_uid=50)))
    d = paired_cohort_drift(a, b, "seed")
    assert d["n_matched"] == 40
    assert d["survival_frac"] == pytest.approx(0.8)


# --------------------------------------------------------------------------
# misc / orchestration
# --------------------------------------------------------------------------

def test_list_checkpoints_sorts_numerically_not_lexically(tmp_run):
    for it in (2000, 10000, 9000, 400):
        write_ckpt(tmp_run, make_state(n_seed=2, n_later=1, iteration=it))
    got = [os.path.basename(p) for p in list_checkpoints(tmp_run)]
    assert got == ["chkpnt400.pth", "chkpnt2000.pth", "chkpnt9000.pth", "chkpnt10000.pth"]


def test_seed_psnr_at_picks_nearest_step():
    series = {"steps": [100, 200, 300], "values": [10.0, 20.0, 30.0]}
    assert seed_psnr_at(series, 205) == 20.0
    assert seed_psnr_at(series, 299) == 30.0
    assert seed_psnr_at({}, 100) is None


def test_analyze_run_refuses_contaminated_and_reports_clean(tmp_run):
    write_ckpt(tmp_run, make_state(n_seed=100, n_later=20, iteration=2000, next_uid=120))
    write_ckpt(tmp_run, make_state(n_seed=100, n_later=60, iteration=4000, next_uid=160))
    write_ckpt(tmp_run, make_state(n_seed=160, n_later=0, iteration=6000,
                                   contaminated=True, uid_start=900, next_uid=1060))
    res = analyze_run(tmp_run, cohort_edges=(2000,), read_tb=False)
    assert res["discriminator_resolved"] is False
    assert res["n_checkpoints"] == 3
    assert res["n_trustworthy_checkpoints"] == 2
    by_iter = {c["iteration"]: c for c in res["checkpoints"]}
    assert by_iter[6000]["lineage_untrustworthy"] is True
    assert "cohorts" not in by_iter[6000]
    assert by_iter[2000]["cohorts"]["seed"]["count"] == 100
    # only the 2000->4000 pair is analysable
    assert len(res["pairs"]) == 1
    assert res["pairs"][0]["seed_drift"]["n_matched"] == 100
