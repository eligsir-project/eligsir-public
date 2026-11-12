"""Regression tests for issue #602 -- admission/cadence invariants and telemetry.

Torch-free and CPU-only by construction, so these run in the lean CI image
(`Dockerfile.ci-cpu`) with no conftest guard. That is deliberate: the whole
point of putting the contract in ``utils/admission_cadence.py`` rather than
inline in ``train_streaming.py`` is that it can be driven by synthetic event
streams instead of a GPU run.

The five scenarios map 1:1 to the issue's numbered regression-test list.
"""

from __future__ import annotations

import json

import pytest

from eligsir.utils.admission_cadence import (
    ADMISSION_SEMANTICS_VERSION,
    AdmissionCadenceRecorder,
    AdmissionInvariantError,
    admission_cadence_report_block,
    check_invariants,
    raise_on_errors,
)


# ------------------------------------------------------------------ #
# Synthetic drivers                                                   #
# ------------------------------------------------------------------ #

def _fake_realized_cadence(*, optimizer_step, source_released,
                           source_ingested=None, keyframe_admitted=None,
                           insertion_commit=0, inserted_gaussians=0,
                           reconcile_verdict=None):
    """The subset of RealizedCadenceTracker.summary() this module reads.

    Built by hand rather than by importing the real tracker so a change in
    that module's internals cannot silently turn these tests green/red for
    the wrong reason -- these assert #602's contract, not #289's.
    """
    block = {
        "schema_version": 2,
        "clocks": {
            "optimizer_step": optimizer_step,
            "source_released": source_released,
            "source_ingested": (source_ingested if source_ingested is not None
                                else source_released),
            "keyframe_admitted": (keyframe_admitted
                                  if keyframe_admitted is not None
                                  else source_released),
            "insertion_commit": insertion_commit,
            "inserted_gaussians": inserted_gaussians,
        },
    }
    if reconcile_verdict is not None:
        block["reconciliation"] = {"verdict": reconcile_verdict}
    return block


def _codes(summary):
    return {v["code"] for v in summary["invariants"]["violations"]}


def _errors(summary):
    return {v["code"] for v in summary["invariants"]["violations"]
            if v["severity"] == "error"}


# ------------------------------------------------------------------ #
# 1. SPF semantics                                                    #
# ------------------------------------------------------------------ #

def test_spf2_release_follows_policy_not_per_frame_update_count():
    """spf=2 over 100 optimizer iterations.

    Asserts the RELEASE POLICY was followed -- roughly one release every two
    optimizer steps -- and deliberately does NOT assert that any individual
    frame received exactly two updates. That conflation is the bug #602
    exists to prevent: ``effective_spf`` is a release cadence, not a
    per-admitted-frame optimizer-exposure guarantee.
    """
    rec = AdmissionCadenceRecorder()
    rec.declare_release_policy(requested=2, resolved=2,
                               nominal_steps_per_frame=2, effective_spf=2)
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=50)
    for i in range(50):
        rec.record_mapping_candidate(frame_id=i)
        rec.record_mapping_admitted(frame_id=i)

    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=100, source_released=50))

    # The release policy held: 100 optimizer steps / 50 releases == 2.
    assert s["realized_exposure"]["optimizer_updates_per_source_release"] == 2.0
    # Per-admitted-frame exposure is REPORTED as a measured quantity...
    assert s["realized_exposure"]["measured"] is True
    assert s["realized_exposure"]["optimizer_updates_per_admitted_frame"] == 2.0
    # ...and because it was measured and agrees with the configured value,
    # no misread warning fires.
    assert "spf_exposure_unmeasured" not in _codes(s)
    assert "spf_exposure_divergence" not in _codes(s)
    assert not s["invariants"]["violated"]


def test_effective_spf_without_measured_exposure_warns():
    """Configured effective_spf but nothing admitted => cannot claim exposure."""
    rec = AdmissionCadenceRecorder()
    rec.declare_release_policy(requested=2, resolved=2, effective_spf=2)
    s = rec.summarize()
    assert "spf_exposure_unmeasured" in _codes(s)
    assert s["verdict"]["optimizer_exposure_verified"] == "unknown"
    # A warning, not an error -- it does not invalidate the run.
    assert not s["invariants"]["violated"]


def test_spf_exposure_divergence_warns_when_realized_differs_materially():
    """effective_spf=50 but the run really delivered ~2 updates/frame."""
    rec = AdmissionCadenceRecorder()
    rec.declare_release_policy(requested=50, resolved=50, effective_spf=50)
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True)
    for i in range(50):
        rec.record_mapping_candidate(frame_id=i)
        rec.record_mapping_admitted(frame_id=i)
    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=100, source_released=50))
    assert "spf_exposure_divergence" in _codes(s)


# ------------------------------------------------------------------ #
# 2. Soft-all under overload                                          #
# ------------------------------------------------------------------ #

def test_soft_all_under_overload_bounds_admission_and_plateaus():
    """Synthetic source faster than the mapper.

    All source observations are counted; mapping admissions are bounded; queue
    depth/age plateau rather than growing without limit; some frames are
    classified skipped_capacity / coalesced; and NO insertion is attributed to
    any skipped frame.
    """
    rec = AdmissionCadenceRecorder()
    rec.declare_release_policy(requested=1, resolved=1, effective_spf=1)
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=300)
    rec.declare_backpressure_owner(owner="mapping_queue",
                                   pending_max_count=8,
                                   pending_max_age_ms=500.0)

    admitted = 0
    for i in range(300):
        rec.record_mapping_candidate(frame_id=i)
        rec.record_source_release(frame_id=i, at_step=i)
        if i % 3 == 0:                       # mapper keeps up with 1 in 3
            rec.record_mapping_admitted(frame_id=i)
            rec.record_first_supervision(frame_id=i, at_step=i + 1)
            rec.record_geometry_insertion(frame_id=i, n_gaussians=100)
            rec.record_insertion_latency(frame_id=i, at_step=i + 2)
            admitted += 1
        elif i % 3 == 1:
            rec.record_mapping_skipped_capacity(frame_id=i)
        else:
            rec.record_mapping_coalesced(frame_id=i, into=i - 1)
        # Pressure plateaus at the bound instead of growing without limit.
        rec.record_pending(depth=min(8, 1 + i % 9), age_ms=100.0 + (i % 7) * 10)

    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=300, source_released=300))
    d = s["dispositions"]

    # Every source observation counted.
    assert s["source_observed"] == 300
    # Admission bounded well below observation.
    assert d["mapping_admitted"] == admitted == 100
    assert d["mapping_admitted"] < s["source_observed"]
    # Real skipping/coalescing happened and is reported as real integers
    # (not null) because the backpressure owner declared itself.
    assert d["mapping_skipped_capacity"] == 100
    assert d["mapping_coalesced"] == 100
    # Insertion happened only for admitted frames.
    assert d["geometry_inserted_frames"] == admitted
    assert s["consistency"]["insertions_after_decline"] == 0
    # Queue plateaued: bounded, not growing.
    assert s["verdict"]["backlog_accumulated"] == "bounded"
    assert s["pending_queue"]["depth"]["max"] <= 8
    # No hard errors.
    assert not s["invariants"]["violated"], s["invariants"]["violations"]


def test_soft_all_overload_queue_percentiles_present():
    rec = AdmissionCadenceRecorder()
    rec.declare_backpressure_owner(owner="q", pending_max_count=10)
    for i in range(1000):
        rec.record_pending(depth=i % 11, age_ms=float(i % 200))
    s = rec.summarize()
    depth = s["pending_queue"]["depth"]
    age = s["pending_queue"]["age_ms"]
    for blk in (depth, age):
        assert blk["p50"] is not None
        assert blk["p95"] is not None
        assert blk["max"] is not None
        assert blk["p50"] <= blk["p95"] <= blk["max"]
    assert depth["samples"] == 1000


def test_monotonically_growing_queue_age_warns():
    rec = AdmissionCadenceRecorder()
    rec.age_window_size = 4
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True)
    # Age climbs in every window -- the mapper never catches up.
    for i in range(40):
        rec.record_pending(depth=i, age_ms=float(i) * 100.0)
    s = rec.summarize()
    assert "pending_age_monotonic_growth" in _codes(s)
    assert s["verdict"]["backlog_accumulated"] == "growing"


# ------------------------------------------------------------------ #
# 3. Soft-all without pressure                                        #
# ------------------------------------------------------------------ #

def test_soft_all_without_pressure_admits_all_useful_frames():
    """Fast mapper: realized admission approaches all frames, no skipping."""
    rec = AdmissionCadenceRecorder()
    rec.declare_release_policy(requested=5, resolved=5, effective_spf=5)
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=120)
    rec.declare_backpressure_owner(owner="mapping_queue",
                                   pending_max_count=32,
                                   pending_max_age_ms=1000.0)
    for i in range(120):
        rec.record_mapping_candidate(frame_id=i)
        rec.record_mapping_admitted(frame_id=i)
        rec.record_geometry_insertion(frame_id=i, n_gaussians=50)
        rec.record_pending(depth=1, age_ms=10.0)

    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=600, source_released=120))
    d = s["dispositions"]

    assert d["mapping_admitted"] == 120
    assert s["source_observed"] == 120
    # Nothing was skipped unnecessarily -- real zeros, because the owner is
    # declared, so these ARE measurements.
    assert d["mapping_skipped_capacity"] == 0
    assert d["mapping_coalesced"] == 0
    assert d["mapping_skipped_redundant"] == 0
    assert s["verdict"]["trajectory_observed"] == "full"
    assert s["verdict"]["admission_mode"] == "soft_all"
    assert s["verdict"]["backlog_accumulated"] == "bounded"
    # Admitting ~100% is fine when there is NO pressure -- the
    # "backpressure ineffective" warning must not fire here.
    assert "soft_all_backpressure_ineffective" not in _codes(s)
    assert not s["invariants"]["violated"]


def test_soft_all_admitting_everything_under_real_pressure_warns():
    """Budget nominally on, but never binds despite persistent pressure."""
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=200)
    rec.declare_backpressure_owner(owner="q", pending_max_count=4,
                                   pending_max_age_ms=100.0)
    for i in range(200):
        rec.record_mapping_candidate(frame_id=i)
        rec.record_mapping_admitted(frame_id=i)
        # Depth sits far above the configured bound for the whole run.
        rec.record_pending(depth=50, age_ms=5000.0)
    rec.record_emergency_transition("test: bound intentionally exceeded")

    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=200, source_released=200))
    assert "soft_all_backpressure_ineffective" in _codes(s)


# ------------------------------------------------------------------ #
# 4. Force-all stress                                                 #
# ------------------------------------------------------------------ #

def test_force_all_stress_marked_and_unbudgeted():
    """Preserves historical `all` behavior and is marked non-production."""
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="force_all_stress",
                                 budgeted=False, source_coverage_target=100,
                                 stress_run=True)
    for i in range(100):
        rec.record_mapping_candidate(frame_id=i)
        rec.record_mapping_admitted(frame_id=i)

    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=100, source_released=100))

    # Historical `all` behavior: everything admitted.
    assert s["dispositions"]["mapping_admitted"] == 100
    assert s["source_observed"] == 100
    # Prominently marked.
    assert s["verdict"]["admission_mode"] == "force_all_stress"
    assert s["resolved_semantics"]["stress_run_declared"] is True
    assert s["resolved_semantics"]["admission_budgeted"] is False
    assert s["verdict"]["admission_budgeted"] is False
    assert "stress" in s["verdict"]["admission_mode_note"]
    assert not s["invariants"]["violated"]


def test_force_all_stress_unmarked_is_a_hard_error():
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="force_all_stress",
                                 budgeted=False, stress_run=False)
    s = rec.summarize()
    assert "stress_run_not_marked" in _errors(s)
    assert s["invariants"]["violated"] is True
    with pytest.raises(AdmissionInvariantError):
        raise_on_errors(s)


def test_force_all_stress_claiming_budgeted_is_a_hard_error():
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="force_all_stress",
                                 budgeted=True, stress_run=True)
    s = rec.summarize()
    assert "stress_run_marked_budgeted" in _errors(s)


# ------------------------------------------------------------------ #
# 5. Deterministic replay                                             #
# ------------------------------------------------------------------ #

def _replay(seed_stream):
    rec = AdmissionCadenceRecorder()
    rec.declare_release_policy(requested=3, resolved=3, effective_spf=3)
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=len(seed_stream))
    rec.declare_backpressure_owner(owner="q", pending_max_count=16,
                                   pending_max_age_ms=800.0)
    for i, kind in enumerate(seed_stream):
        rec.record_mapping_candidate(frame_id=i)
        if kind == "a":
            rec.record_mapping_admitted(frame_id=i)
            rec.record_geometry_insertion(frame_id=i, n_gaussians=10)
        elif kind == "c":
            rec.record_mapping_coalesced(frame_id=i)
        elif kind == "r":
            rec.record_mapping_skipped_redundant(frame_id=i)
        else:
            rec.record_mapping_skipped_capacity(frame_id=i)
        rec.record_pending(depth=i % 17, age_ms=float(i % 300))
    return rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=3 * len(seed_stream), source_released=len(seed_stream)))


def test_deterministic_replay_produces_identical_summaries():
    """Same source/seed/config => byte-identical summary.

    SCOPE NOTE (deliberate reduction): the issue asks for identical
    *admission/coalescing decisions* in file-backed replay mode. Those
    decisions are owned by #600/#601 and do not exist yet, so there is nothing
    to make deterministic on that side. What IS testable today, and what this
    asserts, is that the telemetry layer itself introduces no nondeterminism
    -- notably that the bounded percentile reservoir's sampling uses a
    module-local seeded RNG and never touches the global random stream. A
    reservoir seeded from global entropy would silently make two identical
    runs produce different p50/p95, which is exactly the kind of
    telemetry-induced irreproducibility this test exists to forbid.
    """
    stream = ["a", "c", "s", "a", "r", "a", "s", "c"] * 40
    a = _replay(stream)
    b = _replay(stream)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_reservoir_does_not_perturb_global_random_stream():
    """Enabling telemetry must not change any seeded training draw."""
    import random as _random

    _random.seed(1234)
    before = [_random.random() for _ in range(5)]

    _random.seed(1234)
    rec = AdmissionCadenceRecorder()
    for i in range(50_000):          # far past the reservoir cap
        rec.record_pending(depth=i % 31, age_ms=float(i % 97))
    after = [_random.random() for _ in range(5)]

    assert before == after


# ------------------------------------------------------------------ #
# Hard errors: declined frames must not be inserted or double-classified
# ------------------------------------------------------------------ #

def test_skipped_frame_causing_insertion_is_a_hard_error():
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True)
    rec.declare_backpressure_owner(owner="q", pending_max_count=8)
    rec.record_mapping_candidate(frame_id="f1")
    rec.record_mapping_skipped_capacity(frame_id="f1")
    conflict = rec.record_geometry_insertion(frame_id="f1", n_gaussians=10)

    assert conflict is not None
    s = rec.summarize()
    assert "declined_frame_inserted" in _errors(s)
    assert s["consistency"]["insertions_after_decline"] == 1


def test_coalesced_frame_causing_insertion_is_a_hard_error():
    rec = AdmissionCadenceRecorder()
    rec.record_mapping_candidate(frame_id="f9")
    rec.record_mapping_coalesced(frame_id="f9")
    rec.record_geometry_insertion(frame_id="f9", n_gaussians=1)
    s = rec.summarize()
    assert "declined_frame_inserted" in _errors(s)


def test_frame_admitted_and_skipped_is_a_hard_error():
    rec = AdmissionCadenceRecorder()
    rec.record_mapping_candidate(frame_id="f2")
    rec.record_mapping_admitted(frame_id="f2")
    rec.record_mapping_skipped_capacity(frame_id="f2")
    s = rec.summarize()
    assert "conflicting_disposition" in _errors(s)
    assert s["consistency"]["conflicting_dispositions"] == 1


def test_pending_bound_exceeded_without_emergency_transition_is_an_error():
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True)
    rec.declare_backpressure_owner(owner="q", pending_max_count=4,
                                   pending_max_age_ms=100.0)
    for i in range(20):
        rec.record_pending(depth=99, age_ms=9999.0)
    s = rec.summarize()
    assert "pending_bound_exceeded" in _errors(s)
    assert "pending_age_bound_exceeded" in _errors(s)
    assert s["verdict"]["backlog_accumulated"] == "exceeded_bound"


def test_declared_emergency_transition_clears_the_bound_error():
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True)
    rec.declare_backpressure_owner(owner="q", pending_max_count=4,
                                   pending_max_age_ms=100.0)
    for i in range(20):
        rec.record_pending(depth=99, age_ms=9999.0)
    rec.record_emergency_transition("sensor burst", at_step=10)
    s = rec.summarize()
    assert "pending_bound_exceeded" not in _errors(s)
    assert "pending_age_bound_exceeded" not in _errors(s)


# ------------------------------------------------------------------ #
# The absent != zero rule                                             #
# ------------------------------------------------------------------ #

def test_undeclared_backpressure_reports_null_not_zero():
    """A run predating #601 must not claim 'coalescing ran and never fired'."""
    rec = AdmissionCadenceRecorder()
    rec.record_mapping_candidate(frame_id=1)
    rec.record_mapping_admitted(frame_id=1)
    s = rec.summarize()
    d = s["dispositions"]
    assert d["mapping_coalesced"] is None
    assert d["mapping_skipped_capacity"] is None
    assert d["mapping_skipped_redundant"] is None
    cov = s["field_coverage"]
    for f in ("mapping_coalesced", "mapping_skipped_capacity",
              "mapping_skipped_redundant"):
        assert cov[f]["complete"] is False
        assert "not instrumented" in cov[f]["note"]


def test_undeclared_admission_target_is_unknown_not_permissive():
    rec = AdmissionCadenceRecorder()
    s = rec.summarize()
    assert s["resolved_semantics"]["admission_target_resolved"] is None
    assert s["verdict"]["admission_mode"] == "unknown"
    assert s["verdict"]["trajectory_observed"] == "unknown"
    # An unknown target must not trip the soft-all/stress invariants.
    assert not s["invariants"]["violated"]


def test_declared_backpressure_reports_real_zeros():
    rec = AdmissionCadenceRecorder()
    rec.declare_backpressure_owner(owner="q", pending_max_count=8)
    s = rec.summarize()
    d = s["dispositions"]
    assert d["mapping_coalesced"] == 0
    assert d["mapping_skipped_capacity"] == 0
    assert "mapping_coalesced" not in s["field_coverage"]


# ------------------------------------------------------------------ #
# Requested-vs-resolved divergence                                    #
# ------------------------------------------------------------------ #

def test_requested_all_resolving_to_soft_all_is_a_refinement_not_divergence():
    """Must not fire on every default run once #600 lands."""
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True)
    s = rec.summarize()
    div = s["resolved_semantics"]["policy_divergence"]
    assert div["diverged"] is False
    assert "refines" in div["note"]


def test_real_policy_divergence_is_reported_not_arbitrated():
    """The live in-process override at train_streaming.py:2057-2060."""
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="selective", budgeted=True)
    s = rec.summarize()
    div = s["resolved_semantics"]["policy_divergence"]
    assert div["diverged"] is True
    assert div["requested"] == "all"
    assert div["resolved"] == "selective"
    # Both values survive from their real origin; neither is overwritten.
    assert s["resolved_semantics"]["admission_policy_requested"] == "all"
    assert s["resolved_semantics"]["admission_target_resolved"] == "selective"


def test_unrecognized_target_is_noted_at_declare_time():
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(target_resolved="bogus_mode")
    s = rec.summarize()
    notes = s["resolved_semantics"]["declaration_notes"]
    assert any("unrecognized" in n for n in notes)


# ------------------------------------------------------------------ #
# Derivation fallback (pre-#600/#601 runs)                            #
# ------------------------------------------------------------------ #

def test_dispositions_derive_from_realized_cadence_when_not_recorded():
    """A run with no per-frame disposition call sites still answers the
    acceptance question, and declares that the numbers were derived."""
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=200)
    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=1000, source_released=200, source_ingested=200,
        keyframe_admitted=180, insertion_commit=170,
        inserted_gaussians=17000))
    d = s["dispositions"]
    assert s["disposition_provenance"] == "derived_from_realized_cadence"
    assert d["mapping_candidates"] == 200
    assert d["mapping_admitted"] == 180
    assert d["geometry_inserted_frames"] == 170
    assert d["source_inserted_gaussians"] == 17000
    assert s["source_observed"] == 200
    assert s["verdict"]["trajectory_observed"] == "full"
    assert s["realized_exposure"]["measured"] is True


def test_recorded_dispositions_are_not_overwritten_by_derivation():
    rec = AdmissionCadenceRecorder()
    rec.record_mapping_candidate(frame_id=1)
    rec.record_mapping_admitted(frame_id=1)
    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=100, source_released=999, keyframe_admitted=999))
    assert s["disposition_provenance"] == "recorded"
    assert s["dispositions"]["mapping_admitted"] == 1


# ------------------------------------------------------------------ #
# Artifact shape / plumbing                                           #
# ------------------------------------------------------------------ #

def test_summary_is_json_serializable_and_versioned():
    rec = AdmissionCadenceRecorder()
    rec.declare_release_policy(requested=50, resolved=50, effective_spf=50)
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=10)
    rec.record_mapping_candidate(frame_id=0)
    rec.record_mapping_admitted(frame_id=0)
    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=500, source_released=10))
    text = json.dumps(s, indent=2)
    assert json.loads(text) == s
    assert s["admission_semantics_version"] == ADMISSION_SEMANTICS_VERSION


def test_report_block_returns_none_without_a_recorder():
    """Absent != zero at the top level too: no key means 'predates this'."""
    assert admission_cadence_report_block(None) is None


def test_report_block_embeds_supplied_sub_blocks_verbatim():
    rec = AdmissionCadenceRecorder()
    rc = _fake_realized_cadence(optimizer_step=10, source_released=2)
    clocks = {"map_age_ms": 42.0}
    duty = {"enabled": True, "gpu_busy_frac": 0.61}
    blk = admission_cadence_report_block(
        rec, realized_cadence=rc, mapping_clocks=clocks, duty_cycle=duty)
    assert blk["realized_cadence"] == rc
    assert blk["mapping_clocks"] == clocks
    assert blk["duty_cycle"] == duty


def test_check_invariants_is_pure_and_never_raises():
    for junk in ({}, {"resolved_semantics": {}}, {"dispositions": None},
                 {"pending_queue": {"depth": {}}}):
        assert isinstance(check_invariants(junk), list)


def test_verdict_fields_are_tri_state_not_boolean():
    rec = AdmissionCadenceRecorder()
    v = rec.summarize()["verdict"]
    for key in ("trajectory_observed", "admission_mode",
                "backlog_accumulated", "optimizer_exposure_verified"):
        assert isinstance(v[key], str)
        assert v[key] not in ("True", "False")
    # Every verdict carries a human-readable justification.
    for key in ("trajectory_observed_note", "admission_mode_note",
                "backlog_accumulated_note", "optimizer_exposure_note"):
        assert v[key] and isinstance(v[key], str)


def test_partial_trajectory_is_reported_as_partial():
    rec = AdmissionCadenceRecorder()
    rec.declare_admission_target(policy_requested="all",
                                 target_resolved="soft_all", budgeted=True,
                                 source_coverage_target=2525)
    for i in range(126):                      # the #203 canonical 5% prefix
        rec.record_mapping_candidate(frame_id=i)
        rec.record_mapping_admitted(frame_id=i)
    s = rec.summarize(realized_cadence=_fake_realized_cadence(
        optimizer_step=1000, source_released=126))
    assert s["verdict"]["trajectory_observed"] == "partial"
    assert "126 of 2525" in s["verdict"]["trajectory_observed_note"]


def test_memory_is_bounded_over_a_long_run():
    """O(1) in run length -- the standing observability invariant."""
    rec = AdmissionCadenceRecorder(reservoir_cap=128, ledger_cap=256)
    for i in range(100_000):
        rec.record_mapping_candidate(frame_id=i)
        rec.record_mapping_admitted(frame_id=i)
        rec.record_pending(depth=i % 13, age_ms=float(i % 211))
    assert len(rec._pending_depth._vals) <= 128
    assert len(rec._ledger._order) <= 256
    s = rec.summarize()
    # Exact counts survive even though samples are bounded.
    assert s["dispositions"]["mapping_admitted"] == 100_000
    assert s["pending_queue"]["depth"]["samples"] == 100_000
