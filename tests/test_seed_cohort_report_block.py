"""Issue #346: the seed-cohort availability block and the lineage-coverage audit.

Both exist to make "is this run seed-fade-cohort analyzable?" answerable from a
run's own artifacts, instead of from a byte-level scan of its ``chkpnt*.pth``.
"""

import torch

from scripts.lineage_coverage_audit import classify
from eligsir.utils.comparison_report import seed_cohort_report_block
from eligsir.utils.gaussian_lineage import BIRTH_METHOD_BOOTSTRAP, BIRTH_METHOD_MCMC_GROW


class _FakeModel:
    """Minimal stand-in exposing exactly what seed_cohort_summary() reads."""

    def __init__(self, uids, roots, births, seed_range):
        self.gaussian_uid = torch.tensor(uids, dtype=torch.int64)
        self.root_uid = torch.tensor(roots, dtype=torch.int64)
        self.birth_method = torch.tensor(births, dtype=torch.int8)
        self._seed_uid_range = seed_range
        self._seed_uid_range_reason = "" if seed_range else "test: no bootstrap recorded"
        self.get_xyz = torch.zeros((len(uids), 3))


def _healthy():
    # uids 0-2 are the bootstrap block; uid 3 is an MCMC clone of uid 1.
    return _FakeModel(
        uids=[0, 1, 2, 3],
        roots=[0, 1, 2, 1],
        births=[BIRTH_METHOD_BOOTSTRAP] * 3 + [BIRTH_METHOD_MCMC_GROW],
        seed_range=(0, 3),
    )


def test_block_reports_counts_when_cohort_is_determinable():
    block = seed_cohort_report_block(_healthy())
    assert block["available"] is True
    assert block["reason"] == ""
    assert block["n_seed_born"] == 3
    assert block["n_seed_alive"] == 3
    # The clone inherits root_uid=1, so it joins the lineage but not the cohort.
    assert block["n_seed_lineage_alive"] == 4
    assert block["population"] == 4


def test_relocated_seed_slot_leaves_the_cohort():
    m = _healthy()
    # MCMC relocation keeps the slot's uid but rewrites root_uid/birth_method.
    m.root_uid[2] = 3
    m.birth_method[2] = BIRTH_METHOD_MCMC_GROW
    block = seed_cohort_report_block(m)
    assert block["n_seed_alive"] == 2, "a revived capacity slot must not count as a seed"


def test_block_is_emitted_with_a_reason_when_undetermined():
    """The informative case: unavailable is RECORDED, never silently omitted."""
    block = seed_cohort_report_block(_FakeModel([0, 1], [0, 1], [BIRTH_METHOD_BOOTSTRAP] * 2, None))
    assert block["available"] is False
    assert "UNDETERMINED" in block["reason"]


def test_block_survives_a_model_missing_lineage_buffers_entirely():
    m = _healthy()
    del m.root_uid
    block = seed_cohort_report_block(m)
    assert block["available"] is False
    assert "root_uid" in block["reason"]


def test_block_handles_no_model():
    assert seed_cohort_report_block(None)["available"] is False


def test_write_post_training_report_accepts_seed_cohort():
    """train_streaming's ``_streaming_report_kwargs`` puts ``seed_cohort`` in
    EVERY report's kwargs -- mid-run and final, unconditionally, unlike
    ``depth_insertion`` (final-only) and ``lineage_preserving_setup``
    (non-None-only). All four expansion sites feed ``write_post_training_report``
    directly, so an unaccepted keyword would raise on every mid-run report.
    """
    import inspect

    from eligsir.utils.comparison_report import write_post_training_report

    params = inspect.signature(write_post_training_report).parameters
    assert "seed_cohort" in params
    assert params["seed_cohort"].default is None, "must stay opt-in for train.py"


# --- scripts/lineage_coverage_audit.classify -------------------------------


def test_classify_no_lineage_when_buffers_absent():
    status, reason = classify({"birth_method": False, "seed_uid_range": False}, None)
    assert status == "no_lineage"
    assert "#145" in reason


def test_classify_lineage_without_seed_range_is_not_trustworthy():
    """Every checkpoint in output/ as of 2026-08-13 lands in this bucket."""
    status, reason = classify({"birth_method": True, "seed_uid_range": False}, None)
    assert status == "lineage_no_seed_range"
    assert "#283" in reason


def test_classify_trustworthy_requires_the_real_restore_conditions():
    t1 = {"birth_method": True, "seed_uid_range": True}
    full = {
        "loadable": True, "has_streaming_state": True, "seed_uid_range": [0, 100],
        "lineage_buffers_missing": [], "lineage_buffers_wrong_length": [],
    }
    assert classify(t1, full)[0] == "trustworthy"
    assert classify(t1, {**full, "seed_uid_range": None})[0] == "lineage_no_seed_range"
    assert classify(t1, {**full, "has_streaming_state": False})[0] == "lineage_no_seed_range"
    assert classify(t1, {**full, "lineage_buffers_missing": ["root_uid"]})[0] == "lineage_incomplete"
    assert classify(t1, {**full, "lineage_buffers_wrong_length": ["birth_iter(3!=4)"]})[0] == "lineage_incomplete"
