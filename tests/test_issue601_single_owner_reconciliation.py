"""Issue #601 <-> #600, after #600 MERGED (PR #1085).

Two things this file pins that the pre-merge stand-in test could not:

1. the adapter really returns #600's **real** ``ObservationDecision``, and
   #601's reason vocabulary really lands inside #600's real ``REASON_*``
   constants -- so future drift in #600 surfaces here rather than in a
   training run;
2. the **single-owner reconciliation** #601 requires of #600's own
   ``SoftAllAdmitter``.

On (2): both mechanisms can refuse a frame for capacity. The backpressure
owner decides *before* materialization -- the whole point of #601 -- while
``SoftAllAdmitter`` runs inside ``admission_fn``, i.e. after the camera has
already been built. Leaving both armed would let a frame the owner already
paid to materialize be discarded by a second, later capacity judgement.
That is two owners, which is exactly what #601 removes. So when the owner
is active, ``SoftAllAdmitter``'s capacity bound is disabled and it degrades
to a pass-through admitter -- a signal provider, the same treatment
``BurstBudget`` gets.
"""

from __future__ import annotations

import pathlib

import pytest

from eligsir.mapping.scheduler.backpressure import (
    BackpressureConfig,
    BackpressureOwner,
    PendingObservation,
    as_observation_decision,
)

MODULE_NAME = "eligsir.mapping.admission.observation_decision"
TRAIN_STREAMING = pathlib.Path("train_streaming.py")


def _decisions():
    owner = BackpressureOwner(
        BackpressureConfig(
            enabled=True,
            pending_max_count=4,
            pending_max_age_control_ticks=10_000,
            max_admissions_per_window=1,
            admission_window_control_ticks=10_000,
        )
    )
    out = []
    for i in range(24):
        for d in owner.submit(
            PendingObservation(
                frame_id=i,
                release_control_tick=i,
                novelty=0.05,
                quality=0.1,
                motion=0.02,
                translation_m=0.001,
                rotation_deg=0.05,
            ),
            control_tick=i,
        ):
            out.append(d)
    return out


class TestAgainstTheRealMergedContract:
    def test_the_real_class_is_what_the_adapter_returns(self):
        real_module = pytest.importorskip(MODULE_NAME)
        owner = BackpressureOwner(BackpressureConfig(enabled=True))
        (decision,) = owner.submit(PendingObservation(frame_id=0), control_tick=0)
        adapted = as_observation_decision(decision)
        assert isinstance(adapted, real_module.ObservationDecision), (
            "the adapter is not using #600's real merged contract"
        )

    def test_reason_codes_match_600s_real_constants(self):
        real_module = pytest.importorskip(MODULE_NAME)
        real_reasons = {
            getattr(real_module, name)
            for name in dir(real_module)
            if name.startswith("REASON_")
        }
        assert real_reasons, "#600 exposes no REASON_* constants"
        for d in _decisions():
            adapted = as_observation_decision(d)
            assert adapted.reason in real_reasons, (
                f"#601 reason {d.reason!r} mapped to {adapted.reason!r}, which is "
                f"not one of #600's real constants {sorted(real_reasons)}"
            )

    def test_the_fine_grained_601_reason_survives_on_the_real_class(self):
        pytest.importorskip(MODULE_NAME)
        for d in _decisions():
            adapted = as_observation_decision(d)
            assert adapted.extra["backpressure_reason"] == d.reason
            assert adapted.extra["owner"] == "BackpressureOwner"

    def test_no_601_decision_violates_the_real_invariants(self):
        pytest.importorskip(MODULE_NAME)
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=3,
                pending_max_age_control_ticks=25,
                max_admissions_per_window=1,
                admission_window_control_ticks=50,
                novelty_protect=0.8,
            )
        )
        for i in range(300):
            for d in owner.submit(
                PendingObservation(
                    frame_id=i,
                    release_control_tick=i,
                    novelty=0.95 if i % 37 == 0 else 0.02,
                    quality=0.1,
                    motion=0.01,
                    translation_m=0.001,
                    rotation_deg=0.02,
                    cheap_keyframe=(i % 11 == 0),
                ),
                control_tick=i,
            ):
                # The REAL __post_init__ must never raise on a #601 decision.
                as_observation_decision(d)


class TestSoftAllAdmitterIsReducedToASignalProvider:
    def test_capacity_bound_is_disabled_when_the_owner_is_active(self):
        src = TRAIN_STREAMING.read_text()
        assert "_soft_all_min_steps = 0" in src, (
            "SoftAllAdmitter's capacity bound is not disabled when the "
            "backpressure owner is active -- two capacity owners remain"
        )
        assert "min_steps_per_admit=_soft_all_min_steps" in src, (
            "SoftAllAdmitter is still constructed from the raw arg, so the "
            "single-owner reduction cannot take effect"
        )

    def test_the_reduction_is_skipped_under_force_all_stress(self):
        src = TRAIN_STREAMING.read_text()
        block = src[src.index("_soft_all_min_steps = int(") :][:1600]
        assert "not _backpressure_cfg.force_all_stress" in block, (
            "force-all must not quietly alter SoftAllAdmitter either -- it "
            "exists to reproduce historical behaviour exactly"
        )

    def test_the_reduction_is_inert_when_the_owner_is_off(self):
        src = TRAIN_STREAMING.read_text()
        block = src[src.index("_soft_all_min_steps = int(") :][:1600]
        assert "_backpressure_owner is not None" in block, (
            "the reduction is not gated on the owner being active -- it would "
            "change behaviour on a default (owner-off) run"
        )

    def test_zero_really_disables_the_real_capacity_bound(self):
        """Guard the assumption the whole reduction rests on."""
        module = pytest.importorskip(MODULE_NAME)
        admitter = module.SoftAllAdmitter(
            min_steps_per_admit=0,
            max_consecutive_skips=4,
            # No optimizer work at all between frames -- the harshest case.
            steps_provider=lambda: 0,
        )
        for _ in range(50):
            decision = admitter.decide()
            assert decision.admit_supervision is True, (
                "min_steps_per_admit=0 did not disable the capacity bound, so "
                "reducing SoftAllAdmitter this way does not actually make the "
                "backpressure owner the single capacity owner"
            )

    def test_the_default_bound_would_otherwise_still_skip(self):
        """Non-vacuity: with the bound left at its default, SoftAllAdmitter
        really does skip -- so disabling it is a real change, not a no-op."""
        module = pytest.importorskip(MODULE_NAME)
        admitter = module.SoftAllAdmitter(
            min_steps_per_admit=1,
            max_consecutive_skips=4,
            steps_provider=lambda: 0,
        )
        outcomes = [admitter.decide().admit_supervision for _ in range(20)]
        assert not all(outcomes), (
            "SoftAllAdmitter never skipped even at its default bound -- this "
            "test can no longer show the reduction matters"
        )


class TestIssue602HooksAreClaimed:
    """#602 (PR #1081) shipped ``declare_backpressure_owner`` plus a
    disposition ledger, with an explicit placeholder noting that "#601 owns
    the real bounded-backpressure controller and has NOT landed". It has now
    landed, so #601 claims both.
    """

    def test_the_owner_declares_itself_not_soft_all(self):
        src = TRAIN_STREAMING.read_text()
        assert 'owner="backpressure_owner"' in src, (
            "the bounded owner does not declare itself to #602's recorder, so "
            "the run would still attribute backpressure to soft_all_admitter"
        )
        assert "pending_max_count=int(_backpressure_cfg.pending_max_count)" in src

    def test_soft_all_remains_the_declaration_when_the_owner_is_off(self):
        """Default runs must be unchanged: #602's original reading stands."""
        src = TRAIN_STREAMING.read_text()
        assert 'owner="soft_all_admitter"' in src, (
            "the soft-all fallback declaration was removed -- default runs "
            "would lose their backpressure-owner attribution entirely"
        )
        assert "elif _admission_semantics[" in src, (
            "the soft-all declaration is no longer the fallback branch"
        )

    def test_ms_age_bound_is_not_fabricated(self):
        """#601's default bound is in ITERATIONS. Reporting a converted ms
        value would fabricate a bound the run does not use."""
        src = TRAIN_STREAMING.read_text()
        block = src[src.index('owner="backpressure_owner"') :][:1400]
        assert '_backpressure_cfg.age_axis == "wallclock"' in block, (
            "pending_max_age_ms is reported without checking the age axis"
        )

    def test_dispositions_are_recorded_for_every_decision(self):
        src = TRAIN_STREAMING.read_text()
        assert "def _record_backpressure_disposition(decision)" in src
        assert "_record_backpressure_disposition(_bp_dec)" in src, (
            "#602's disposition counters are never fed, so coalesce/skip "
            "counts stay absent even with the owner running"
        )

    def test_every_action_maps_to_a_disposition_counter(self):
        """No decision may fall through the bridge unrecorded."""
        import ast

        from eligsir.mapping.scheduler.backpressure import BackpressureAction

        tree = ast.parse(TRAIN_STREAMING.read_text())
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "_record_backpressure_disposition"
        )
        body = ast.unparse(fn)
        for action in BackpressureAction:
            assert action.name in body, (
                f"BackpressureAction.{action.name} has no branch in the "
                "disposition bridge -- those decisions would go unrecorded"
            )
        # ...and each maps to a distinct recorder call.
        for call in (
            "record_mapping_admitted",
            "record_mapping_metadata_only",
            "record_mapping_coalesced",
            "record_mapping_skipped_redundant",
            "record_mapping_skipped_capacity",
        ):
            assert call in body, f"missing recorder call: {call}"
        assert "record_mapping_candidate" in body
