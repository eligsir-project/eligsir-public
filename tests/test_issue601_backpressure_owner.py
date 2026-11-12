"""Issue #601 -- one bounded backpressure owner with representative
coalescing before expensive materialization.

Every test here is synthetic and CPU-only: the owner is a pure decision
module with no torch/GPU/IO dependency, which is exactly what lets the
bounded-queue, determinism and no-materialization guarantees be proven
without a GPU slot.

Test map (numbering follows the issue's own "Tests" section):

1. ``TestSyntheticOverload`` -- source releases faster than mapping for
   10k frames; pending count and age plateau at the configured bounds;
   memory does not scale with source length; the protected frontier
   survives.
2. ``TestRedundantSegment`` -- representatives selected deterministically;
   no expensive materialization for skipped frames.
3. ``TestHighNoveltyUnderOverload`` -- a high-novelty frame embedded in
   overload is retained even while its neighbours are coalesced.
4. ``TestNoPressureControl`` -- soft-all with the owner ENABLED and
   pressure below threshold behaves like historical all (plus the
   separate default-off byte-identity assertion).
5. ``TestForceAllStressMode`` -- #600's force-all bypasses the owner only
   explicitly, and raises a loud stress flag in the report.

Plus ``TestSingleOwnerContract`` for the invariants #601 requires of the
other systems (BurstBudget neutering, terminal decisions, insertion
permission) and ``TestDeterminism`` for the offline-replay requirement.
"""

from __future__ import annotations

import math

import pytest

from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureConfig,
    BackpressureDecision,
    BackpressureOwner,
    PendingObservation,
    PressureState,
    as_observation_decision,
    burst_config_as_signal_provider,
    pose_delta_signals,
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def make_obs(
    frame_id: int,
    *,
    release_control_tick: int = 0,
    novelty: float = 0.1,
    quality: float = 0.5,
    motion: float = 0.1,
    translation_m: float = 0.01,
    rotation_deg: float = 0.5,
    cheap_keyframe: bool = False,
    source_timestamp: float | None = None,
) -> PendingObservation:
    return PendingObservation(
        frame_id=frame_id,
        source_timestamp=(
            float(frame_id) / 30.0 if source_timestamp is None else source_timestamp
        ),
        release_control_tick=release_control_tick,
        cheap_keyframe=cheap_keyframe,
        novelty=novelty,
        quality=quality,
        motion=motion,
        translation_m=translation_m,
        rotation_deg=rotation_deg,
    )


class _MaterializationSpy:
    """Stands in for the expensive half of the drain loop.

    Records every frame that would have had a camera built, RGB/depth
    decoded, a GPU upload done, a multires cache constructed, replay
    residency granted, or depth inserted. #601 requirement 5 is exactly
    "none of this happens for a skipped frame".
    """

    def __init__(self) -> None:
        self.materialized: list[int] = []
        self.inserted: list[int] = []
        self.metadata_only: list[int] = []

    def consume(self, decision: BackpressureDecision) -> None:
        if decision.materializes:
            self.materialized.append(decision.frame_id)
            if decision.allow_geometry_insertion:
                self.inserted.append(decision.frame_id)
        else:
            self.metadata_only.append(decision.frame_id)


def run_stream(
    owner: BackpressureOwner,
    observations,
    *,
    control_ticks_per_frame: int = 1,
    spy: _MaterializationSpy | None = None,
):
    """Drive *owner* with one source release per canonical control epoch."""
    spy = spy or _MaterializationSpy()
    decisions: list[BackpressureDecision] = []
    for n, obs in enumerate(observations):
        control_tick = n * control_ticks_per_frame
        obs = PendingObservation(
            **{**obs.__dict__, "release_control_tick": control_tick}
        )
        for d in owner.submit(
            obs,
            control_tick=control_tick,
            source_time_s=obs.source_timestamp,
        ):
            decisions.append(d)
            spy.consume(d)
    final_source_time = (
        0.0 if not observations else float(observations[-1].source_timestamp)
    )
    for d in owner.flush(
        control_tick=len(observations) * control_ticks_per_frame,
        source_time_s=final_source_time,
    ):
        decisions.append(d)
        spy.consume(d)
    return decisions, spy


# ----------------------------------------------------------------------
# Test 1 -- synthetic overload
# ----------------------------------------------------------------------
class TestSyntheticOverload:
    N_FRAMES = 10_000

    def _overloaded_owner(self, **overrides) -> BackpressureOwner:
        cfg = BackpressureConfig(
            enabled=True,
            pending_max_count=8,
            pending_max_age_control_ticks=40,
            # Deliberately tiny relative to the source rate: 2 expensive
            # admissions per 100 control ticks against one frame per tick.
            max_admissions_per_window=2,
            admission_window_control_ticks=100,
            **overrides,
        )
        return BackpressureOwner(cfg)

    def test_pending_depth_and_age_plateau_at_configured_bounds(self):
        owner = self._overloaded_owner()
        decisions, _ = run_stream(
            owner, [make_obs(i) for i in range(self.N_FRAMES)]
        )
        assert len(decisions) == self.N_FRAMES

        cfg = owner.config
        max_depth = max(d.pending_depth for d in decisions)
        max_age = max(d.pending_age_control_ticks for d in decisions)
        assert max_depth <= cfg.pending_max_count, (
            f"pending depth {max_depth} exceeded the configured bound "
            f"{cfg.pending_max_count} -- the queue is not bounded"
        )
        # Age is bounded by the frontier-protection rule; allow one frame
        # of slack because the bound is checked at observation time.
        assert max_age <= cfg.pending_max_age_control_ticks + 1, (
            f"pending age {max_age} exceeded the configured bound "
            f"{cfg.pending_max_age_control_ticks}"
        )

    def test_memory_does_not_scale_with_source_length(self):
        """EVERY per-frame structure must be bounded, not just the cohort.

        The terminal-decision ledger is the one that used to grow with the
        source: it is now an OrderedDict capped at
        ``decision_ledger_size``.
        """
        owner = self._overloaded_owner(decision_ledger_size=64)
        run_stream(owner, [make_obs(i) for i in range(self.N_FRAMES)])
        # The bounded cohort holds metadata records only, and its size is
        # capped regardless of how long the source ran.
        assert owner.pending_depth <= owner.config.pending_max_count
        assert len(owner.cohort) <= owner.config.pending_max_count
        # The decision ledger is bounded too.
        assert len(owner._decided) <= 64, (
            f"decision ledger grew to {len(owner._decided)} over "
            f"{self.N_FRAMES} source frames -- state scales with source length"
        )
        # Nothing else on the owner grew with the source either.
        # Admission timestamps are pruned to the sliding window, so this is
        # bounded by the window length (forced frontier admits legitimately
        # exceed max_admissions_per_window -- bounding map age wins over the
        # budget, by design).
        assert (
            len(owner._admission_control_ticks) <= owner.config.admission_window_control_ticks + 1
        )
        assert len(owner._keep_reasons) <= owner.config.pending_max_count
        # Telemetry keeps per-decision report samples by design, so exclude it
        # here while stating the expectation explicitly.
        assert owner.telemetry.n_observed == self.N_FRAMES

    def test_evicted_ledger_entries_read_as_unknown_not_admitted(self):
        owner = self._overloaded_owner(decision_ledger_size=16)
        decisions, _ = run_stream(owner, [make_obs(i) for i in range(400)])
        assert owner.decision_for(decisions[0].frame_id) is None, (
            "an aged-out ledger entry must read as unknown"
        )
        recent = decisions[-1]
        assert owner.decision_for(recent.frame_id) is recent.action

    def test_protected_frontier_survives_overload(self):
        owner = self._overloaded_owner()
        decisions, spy = run_stream(
            owner, [make_obs(i) for i in range(self.N_FRAMES)]
        )
        admitted = [d for d in decisions if d.action is BackpressureAction.ADMIT]
        assert admitted, "overload starved mapping completely -- frontier not protected"
        # The newest frontier is repeatedly force-admitted so map age stays
        # bounded, and the last observed frame is never stranded.
        assert owner.telemetry.n_forced_frontier_admits > 0
        gaps = [b.frame_id - a.frame_id for a, b in zip(admitted, admitted[1:])]
        assert max(gaps) <= owner.config.pending_max_count + 1, (
            "an unbounded stretch of source frames received no mapping at all"
        )
        # Mapping is genuinely reduced rather than piled up.
        assert len(spy.materialized) < self.N_FRAMES // 4

    def test_report_block_echoes_the_hard_bounds(self):
        owner = self._overloaded_owner()
        run_stream(owner, [make_obs(i) for i in range(200)])
        block = owner.report_block()
        cfg_block = block["config"]
        for key in (
            "pending_max_count",
            "pending_max_age_control_ticks",
            "pending_max_age_ms",
            "max_admissions_per_window",
            "admission_window_control_ticks",
            "frontier_quota",
            "age_axis",
            "deterministic",
        ):
            assert key in cfg_block, f"resolved bound {key!r} missing from report"
        assert cfg_block["pending_max_count"] == 8
        assert cfg_block["max_admissions_per_window"] == 2
        assert block["n_observed"] == 200
        assert block["n_admitted"] + block["n_skipped"] == block["n_observed"]
        assert 0.0 <= block["skipped_fraction"] <= 1.0
        assert block["pending_depth_p95"] <= 8
        assert block["single_owner_enforced"] is True
        assert block["stress_mode"] is False


# ----------------------------------------------------------------------
# Test 2 -- redundant trajectory segment
# ----------------------------------------------------------------------
class TestRedundantSegment:
    def _redundant_stream(self):
        """F100..F105: a near-stationary redundant segment."""
        return [
            make_obs(
                100 + i,
                novelty=0.05,
                motion=0.01,
                translation_m=0.002,
                rotation_deg=0.05,
            )
            for i in range(6)
        ]

    def _owner(self):
        return BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=4,
                pending_max_age_control_ticks=1000,
                max_admissions_per_window=1,
                admission_window_control_ticks=3,
            )
        )

    def test_representatives_selected_deterministically(self):
        runs = []
        for _ in range(3):
            decisions, _ = run_stream(self._owner(), self._redundant_stream())
            runs.append(
                [
                    (d.frame_id, d.action.value, d.representative_frame_id)
                    for d in decisions
                ]
            )
        assert runs[0] == runs[1] == runs[2], (
            "representative selection is not deterministic across identical runs"
        )

    def test_no_expensive_materialization_for_skipped_frames(self):
        decisions, spy = run_stream(self._owner(), self._redundant_stream())
        skipped = [d for d in decisions if d.skipped]
        assert skipped, "the redundant segment produced no skips at all"
        for d in skipped:
            assert d.frame_id not in spy.materialized
            assert d.frame_id not in spy.inserted
            assert d.admit_supervision is False
            assert d.allow_geometry_insertion is False
            assert d.observe is True, "a skipped frame must still be observed"
        assert set(spy.materialized).isdisjoint({d.frame_id for d in skipped})

    def test_coalesced_frames_name_their_representative(self):
        decisions, _ = run_stream(self._owner(), self._redundant_stream())
        coalesced = [
            d for d in decisions if d.action is BackpressureAction.COALESCE_INTO
        ]
        assert coalesced, "a near-stationary segment produced no coalescing"
        admitted_ids = {
            d.frame_id for d in decisions if d.action is BackpressureAction.ADMIT
        }
        for d in coalesced:
            assert d.representative_frame_id is not None
            assert d.representative_frame_id != d.frame_id
            assert d.representative_frame_id in admitted_ids, (
                "a frame was coalesced into a representative that was itself "
                "never admitted -- its information is lost, not represented"
            )

    def test_every_observation_gets_exactly_one_terminal_decision(self):
        stream = self._redundant_stream()
        decisions, _ = run_stream(self._owner(), stream)
        ids = [d.frame_id for d in decisions]
        assert sorted(ids) == sorted(o.frame_id for o in stream)
        assert len(ids) == len(set(ids)), "a frame received more than one decision"


# ----------------------------------------------------------------------
# Test 3 -- high novelty embedded in overload
# ----------------------------------------------------------------------
class TestHighNoveltyUnderOverload:
    def test_high_novelty_frame_is_retained(self):
        cfg = BackpressureConfig(
            enabled=True,
            pending_max_count=16,
            pending_max_age_control_ticks=10_000,
            max_admissions_per_window=1,
            admission_window_control_ticks=1000,
            novelty_protect=0.85,
        )
        owner = BackpressureOwner(cfg)
        stream = [
            make_obs(i, novelty=0.02, motion=0.005, translation_m=0.001,
                     rotation_deg=0.02)
            for i in range(30)
        ]
        # One genuinely novel observation buried in the redundant run.
        stream[17] = make_obs(
            17, novelty=0.97, motion=0.9, translation_m=0.4, rotation_deg=30.0
        )
        decisions, spy = run_stream(owner, stream)

        by_id = {d.frame_id: d for d in decisions}
        novel = by_id[17]
        assert novel.action is BackpressureAction.ADMIT, (
            f"the high-novelty frame was {novel.action.value} "
            f"({novel.reason}) under overload -- protection did not fire"
        )
        assert novel.reason == "protected_high_novelty"
        assert 17 in spy.materialized

        # ...while its neighbours were compressed away.
        neighbours = [by_id[i] for i in (14, 15, 16, 18, 19, 20)]
        assert any(n.skipped for n in neighbours), (
            "nothing around the novel frame was compressed -- the test did not "
            "actually exercise overload"
        )

    def test_protection_beats_an_exhausted_admission_budget(self):
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=64,
                pending_max_age_control_ticks=100_000,
                max_admissions_per_window=0,  # no budget at all
                admission_window_control_ticks=1000,
                novelty_protect=0.85,
            )
        )
        decisions, _ = run_stream(
            owner,
            [make_obs(0, novelty=0.99, motion=0.9, translation_m=0.5,
                      rotation_deg=45.0)],
        )
        assert decisions[0].action is BackpressureAction.ADMIT
        assert decisions[0].reason == "protected_high_novelty"


# ----------------------------------------------------------------------
# Test 4 -- no-pressure control
# ----------------------------------------------------------------------
class TestNoPressureControl:
    STREAM = [make_obs(i) for i in range(120)]

    def test_soft_all_enabled_but_unpressured_matches_historical_all(self):
        """The owner is ENABLED; capacity simply exceeds the source rate.

        This is the assertion that actually says something about the
        POLICY -- a disabled owner would prove nothing here.
        """
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=32,
                pending_max_age_control_ticks=10_000,
                max_admissions_per_window=1000,  # capacity far above demand
                admission_window_control_ticks=10,
            )
        )
        decisions, spy = run_stream(owner, self.STREAM)
        assert owner.config.enabled is True
        assert all(d.action is BackpressureAction.ADMIT for d in decisions), (
            "soft-all skipped frames while capacity was available"
        )
        assert spy.materialized == [o.frame_id for o in self.STREAM]
        assert all(
            d.pressure_state is PressureState.LOW for d in decisions
        ), "pressure was not LOW in the no-pressure control"
        assert owner.report_block()["skipped_fraction"] == 0.0

    def test_default_config_is_disabled_and_admits_everything(self):
        """Separate, weaker assertion: default-off is byte-identical."""
        owner = BackpressureOwner()  # defaults
        assert owner.config.enabled is False
        decisions, spy = run_stream(owner, self.STREAM)
        assert all(d.action is BackpressureAction.ADMIT for d in decisions)
        assert all(d.reason == "backpressure_disabled" for d in decisions)
        assert spy.materialized == [o.frame_id for o in self.STREAM]


# ----------------------------------------------------------------------
# Test 5 -- force-all stress mode (#600)
# ----------------------------------------------------------------------
class TestForceAllStressMode:
    def test_force_all_bypasses_the_owner_explicitly(self):
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                force_all_stress=True,
                pending_max_count=2,
                max_admissions_per_window=1,
                admission_window_control_ticks=1000,
            )
        )
        decisions, spy = run_stream(owner, [make_obs(i) for i in range(500)])
        assert all(d.action is BackpressureAction.ADMIT for d in decisions)
        assert all(d.reason == "force_all_stress" for d in decisions)
        assert all(d.stress_mode for d in decisions)
        assert len(spy.materialized) == 500, (
            "force-all must reproduce the historical hard-all behaviour exactly"
        )

    def test_stress_mode_raises_a_loud_report_flag(self):
        owner = BackpressureOwner(
            BackpressureConfig(enabled=True, force_all_stress=True)
        )
        run_stream(owner, [make_obs(i) for i in range(10)])
        block = owner.report_block()
        assert block["stress_mode"] is True
        assert block["single_owner_enforced"] is False
        assert "STRESS MODE" in block["stress_mode_warning"]
        assert "Not a production result" in block["stress_mode_warning"]

    def test_stress_mode_is_opt_in_only(self):
        assert BackpressureConfig().force_all_stress is False
        assert BackpressureConfig(enabled=True).force_all_stress is False


# ----------------------------------------------------------------------
# Determinism (issue #601: "deterministic in offline replay mode")
# ----------------------------------------------------------------------
class TestDeterminism:
    def _cfg(self, **kw):
        return BackpressureConfig(
            enabled=True,
            pending_max_count=6,
            pending_max_age_control_ticks=50,
            max_admissions_per_window=2,
            admission_window_control_ticks=40,
            **kw,
        )

    def test_default_age_axis_is_deterministic(self):
        cfg = BackpressureConfig()
        assert cfg.age_axis == "control_tick"
        assert cfg.deterministic is True
        assert BackpressureConfig(age_axis="source_time").deterministic is True
        assert BackpressureConfig(age_axis="wallclock").deterministic is False

    def test_wallclock_jitter_cannot_change_the_retained_set(self):
        """The discriminating check: identical source, different host
        clocks, byte-identical retained set."""
        stream = [
            make_obs(i, novelty=(i % 7) / 10.0, motion=(i % 5) / 10.0)
            for i in range(300)
        ]

        def retained(jitter: float):
            owner = BackpressureOwner(self._cfg())
            out = []
            for n, obs in enumerate(stream):
                jittered = PendingObservation(
                    **{
                        **obs.__dict__,
                        "release_control_tick": n,
                        # Wall clock deliberately garbage/jittered.
                        "wallclock_s": n * jitter + math.sin(n) * jitter,
                    }
                )
                for d in owner.submit(
                    jittered,
                    control_tick=n,
                    source_time_s=obs.source_timestamp,
                    wallclock_now=n * jitter,
                ):
                    out.append((d.frame_id, d.action.value, d.reason))
            return out

        assert retained(0.001) == retained(3.7) == retained(0.0), (
            "host wall-clock jitter changed the retained set -- a decision is "
            "reading a non-deterministic clock"
        )

    def test_source_time_axis_is_also_deterministic(self):
        stream = [make_obs(i) for i in range(200)]
        cfg = self._cfg(age_axis="source_time", pending_max_age_source_s=0.5)
        runs = [
            [
                (d.frame_id, d.action.value)
                for d in run_stream(BackpressureOwner(cfg), stream)[0]
            ]
            for _ in range(2)
        ]
        assert runs[0] == runs[1]


class TestSemanticClockAuthority:
    """Admission is a control-plane decision, never an optimizer-speed effect."""

    @staticmethod
    def _decisions(optimizer_updates_per_source: int):
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=64,
                pending_max_age_control_ticks=1_000,
                max_admissions_per_window=2,
                admission_window_control_ticks=4,
            )
        )
        out = []
        optimizer_updates = 0
        for control_tick in range(12):
            optimizer_updates += optimizer_updates_per_source
            obs = make_obs(control_tick, release_control_tick=control_tick)
            for decision in owner.submit(
                obs,
                control_tick=control_tick,
                source_time_s=obs.source_timestamp,
            ):
                out.append((decision.frame_id, decision.action, decision.reason))
        out.extend(
            (decision.frame_id, decision.action, decision.reason)
            for decision in owner.flush(control_tick=12, source_time_s=12 / 30.0)
        )
        return out, optimizer_updates

    def test_optimizer_speed_cannot_change_semantic_admission(self):
        slow, slow_optimizer_updates = self._decisions(1)
        fast, fast_optimizer_updates = self._decisions(64)

        assert slow_optimizer_updates != fast_optimizer_updates
        assert slow == fast

    def test_control_tick_rollback_is_rejected(self):
        owner = BackpressureOwner(BackpressureConfig(enabled=True))
        owner.submit(make_obs(0), control_tick=7, source_time_s=0.0)

        with pytest.raises(ValueError, match="control_tick.*monotonic"):
            owner.submit(make_obs(1), control_tick=6, source_time_s=1 / 30.0)

    def test_source_time_rollback_is_rejected_when_it_owns_age(self):
        owner = BackpressureOwner(
            BackpressureConfig(enabled=True, age_axis="source_time")
        )
        owner.submit(make_obs(0), control_tick=0, source_time_s=1.0)

        with pytest.raises(ValueError, match="source_time_s.*monotonic"):
            owner.submit(make_obs(1), control_tick=1, source_time_s=0.5)

    def test_source_time_can_own_causal_age_without_optimizer_progress(self):
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                age_axis="source_time",
                pending_max_count=100,
                pending_max_age_source_s=0.5,
                max_admissions_per_window=1,
                admission_window_control_ticks=100,
            )
        )
        first = owner.submit(
            make_obs(0, source_timestamp=0.0),
            control_tick=0,
            source_time_s=0.0,
        )
        second = owner.submit(
            make_obs(1, source_timestamp=1.0),
            control_tick=0,
            source_time_s=1.0,
        )

        assert first[0].action is BackpressureAction.ADMIT
        assert second[0].action is BackpressureAction.ADMIT
        assert second[0].reason == "forced_frontier_admit_age_bound"

    def test_repeated_control_tick_accepts_multiple_source_observations(self):
        owner = BackpressureOwner(BackpressureConfig(enabled=True))

        first = owner.submit(make_obs(0), control_tick=4, source_time_s=0.0)
        second = owner.submit(
            make_obs(1, release_control_tick=4),
            control_tick=4,
            source_time_s=1 / 30.0,
        )

        assert first and second

    def test_stalled_control_tick_does_not_grow_admission_clock_state(self):
        owner = BackpressureOwner(
            BackpressureConfig(enabled=True, force_all_stress=True)
        )

        for frame_id in range(10_000):
            owner.submit(
                make_obs(frame_id, release_control_tick=4),
                control_tick=4,
                source_time_s=frame_id / 30.0,
            )

        assert owner._admission_control_ticks == {4: 10_000}

    def test_future_release_tick_is_rejected_without_poisoning_retry(self):
        owner = BackpressureOwner(BackpressureConfig(enabled=True))

        with pytest.raises(ValueError, match="release_control_tick.*newer"):
            owner.submit(
                make_obs(0, release_control_tick=8),
                control_tick=7,
                source_time_s=0.0,
            )

        retried = owner.submit(make_obs(0), control_tick=7, source_time_s=0.0)
        assert retried[0].action is BackpressureAction.ADMIT

    def test_legacy_iteration_clock_vocabulary_is_rejected(self):
        with pytest.raises(ValueError, match="control_tick"):
            BackpressureConfig(age_axis="iterations")

        owner = BackpressureOwner(BackpressureConfig(enabled=True))
        with pytest.raises(TypeError, match="iteration"):
            owner.submit(make_obs(0), control_tick=0, iteration=999)


# ----------------------------------------------------------------------
# Single-owner contract (#601's "after this lands" requirements)
# ----------------------------------------------------------------------
class TestSingleOwnerContract:
    def test_burst_budget_drop_execution_is_neutered(self):
        from eligsir.mapping.scheduler.burst_budget import BurstConfig

        original = BurstConfig(
            max_per_iter=2, drop_nonkf_backlog=4, drop_nonkf_in_catch_up=True
        )
        signal_only = burst_config_as_signal_provider(original)
        # The cadence cap survives (a genuine, non-overlapping signal)...
        assert signal_only.max_per_iter == 2
        # ...but the overlapping drop execution can never fire again.
        assert signal_only.drop_nonkf_in_catch_up is False
        assert signal_only.drop_nonkf_backlog >= 2**31 - 1
        # The original is untouched (frozen-config discipline).
        assert original.drop_nonkf_in_catch_up is True
        assert burst_config_as_signal_provider(None) is None

    def test_decisions_are_terminal_so_a_defer_path_cannot_resurrect(self):
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=4,
                max_admissions_per_window=1,
                admission_window_control_ticks=500,
            )
        )
        decisions, _ = run_stream(owner, [make_obs(i) for i in range(40)])
        skipped = [d for d in decisions if d.skipped]
        assert skipped
        for d in skipped:
            recorded = owner.decision_for(d.frame_id)
            assert recorded is d.action, (
                "the owner does not remember its own terminal decision, so a "
                "drain/defer path could silently re-admit a skipped frame"
            )
            assert recorded is not BackpressureAction.ADMIT

    def test_geometry_insertion_requires_admit_supervision(self):
        # Invariant is structurally enforced, not merely conventional.
        with pytest.raises(ValueError, match="allow_geometry_insertion"):
            BackpressureDecision(
                frame_id=1,
                action=BackpressureAction.SKIP_REDUNDANT,
                reason="skipped_redundant_intermediate",
                admit_supervision=False,
                allow_geometry_insertion=True,
            )

    def test_admit_supervision_iff_admit(self):
        with pytest.raises(ValueError, match="admit_supervision"):
            BackpressureDecision(
                frame_id=1,
                action=BackpressureAction.SKIP_CAPACITY,
                reason="skipped_capacity_exhausted",
                admit_supervision=True,
            )

    def test_observation_never_waits_for_mapping_admission(self):
        with pytest.raises(ValueError, match="observe must always be True"):
            BackpressureDecision(
                frame_id=1,
                action=BackpressureAction.ADMIT,
                reason="soft_all_capacity_available",
                admit_supervision=True,
                observe=False,
            )

    def test_coalesce_must_name_a_different_representative(self):
        with pytest.raises(ValueError, match="coalesced into itself"):
            BackpressureDecision(
                frame_id=7,
                action=BackpressureAction.COALESCE_INTO,
                reason="coalesced_into_representative",
                representative_frame_id=7,
            )
        with pytest.raises(ValueError, match="representative_frame_id"):
            BackpressureDecision(
                frame_id=7,
                action=BackpressureAction.SKIP_REDUNDANT,
                reason="skipped_redundant_intermediate",
                representative_frame_id=3,
            )

    def test_upstream_hard_gate_resolves_as_metadata_only(self):
        owner = BackpressureOwner(BackpressureConfig(enabled=True))
        (decision,) = owner.submit(
            make_obs(5), control_tick=0, mapping_candidate=False
        )
        assert decision.action is BackpressureAction.METADATA_ONLY
        assert decision.observe is True
        assert decision.mapping_candidate is False
        assert decision.materializes is False


# ----------------------------------------------------------------------
# #600 contract adapter
# ----------------------------------------------------------------------
class TestObservationDecisionAdapter:
    def test_decision_mirrors_the_600_field_names(self):
        owner = BackpressureOwner(BackpressureConfig(enabled=True))
        (decision,) = owner.submit(make_obs(0), control_tick=0)
        view = as_observation_decision(decision)
        for field_name in (
            "observe",
            "mapping_candidate",
            "admit_supervision",
            "allow_geometry_insertion",
            "reason",
        ):
            assert hasattr(view, field_name), (
                f"#600 field {field_name!r} missing from the adapted decision"
            )
        assert view.observe is True
        assert view.admit_supervision is True
        assert view.allow_geometry_insertion is True

    def test_adapter_preserves_skip_semantics(self):
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=2,
                max_admissions_per_window=1,
                admission_window_control_ticks=1000,
            )
        )
        decisions, _ = run_stream(owner, [make_obs(i) for i in range(20)])
        skipped = next(d for d in decisions if d.skipped)
        view = as_observation_decision(skipped)
        assert view.observe is True
        assert view.admit_supervision is False
        assert view.allow_geometry_insertion is False


# ----------------------------------------------------------------------
# Cheap-signal helper
# ----------------------------------------------------------------------
class TestPoseDeltaSignals:
    IDENT = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    def test_identity_pair_is_zero_delta(self):
        t, r = pose_delta_signals(self.IDENT, self.IDENT)
        assert t == pytest.approx(0.0)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_pure_translation(self):
        moved = [row[:] for row in self.IDENT]
        moved[0][3] = 3.0
        moved[1][3] = 4.0
        t, r = pose_delta_signals(self.IDENT, moved)
        assert t == pytest.approx(5.0)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_pure_rotation_90_deg_about_z(self):
        rot = [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        t, r = pose_delta_signals(self.IDENT, rot)
        assert t == pytest.approx(0.0)
        assert r == pytest.approx(90.0, abs=1e-6)

    def test_missing_pose_is_safe(self):
        assert pose_delta_signals(None, self.IDENT) == (0.0, 0.0)
        assert pose_delta_signals(self.IDENT, None) == (0.0, 0.0)
        assert pose_delta_signals([[1, 2]], self.IDENT) == (0.0, 0.0)


# ----------------------------------------------------------------------
# Lookahead mode (the buffered variant of the same policy)
# ----------------------------------------------------------------------
class TestLookaheadMode:
    def _owner(self):
        return BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                resolution_mode="lookahead",
                pending_max_count=4,
                pending_max_age_control_ticks=10_000,
                max_admissions_per_window=1,
                admission_window_control_ticks=1000,
                frontier_quota=1,
                representatives_per_segment=1,
            )
        )

    def test_bounded_cohort_never_exceeds_max_count(self):
        owner = self._owner()
        for i in range(500):
            owner.submit(make_obs(i, release_control_tick=i), control_tick=i)
            assert len(owner.cohort) <= owner.config.pending_max_count

    def test_segment_representative_is_the_highest_scoring_member(self):
        owner = self._owner()
        # Frame 3 is the clear information peak of the segment.
        stream = [
            make_obs(0, novelty=0.1),
            make_obs(1, novelty=0.1),
            make_obs(2, novelty=0.2),
            make_obs(3, novelty=0.6),
            make_obs(4, novelty=0.15),
            make_obs(5, novelty=0.1),
        ]
        decisions = []
        for i, obs in enumerate(stream):
            decisions.extend(
                owner.submit(
                    PendingObservation(**{**obs.__dict__, "release_control_tick": i}),
                    control_tick=i,
                )
            )
        # The compression pass runs while the admission budget is
        # exhausted, so the representative is *retained* (still pending)
        # rather than released. Flush to see its terminal decision.
        decisions.extend(owner.flush(control_tick=len(stream)))

        coalesced = {
            d.frame_id: d.representative_frame_id
            for d in decisions
            if d.action is BackpressureAction.COALESCE_INTO
        }
        assert coalesced, "no coalescing happened"
        assert set(coalesced.values()) == {3}, (
            f"neighbours were coalesced into {set(coalesced.values())}, expected "
            "the highest-scoring member (frame 3)"
        )
        reps = [d for d in decisions if d.reason == "segment_representative"]
        assert reps, "no representative was selected"
        assert reps[0].frame_id == 3, (
            f"representative was frame {reps[0].frame_id}, expected the "
            "highest-scoring member (frame 3)"
        )
        assert reps[0].action is BackpressureAction.ADMIT
        # ...and the newest frontier observation was never compressed away.
        frontier = next(d for d in decisions if d.frame_id == 5)
        assert frontier.action is BackpressureAction.ADMIT

    def test_lookahead_is_deterministic(self):
        stream = [make_obs(i, novelty=(i % 9) / 10.0) for i in range(200)]
        runs = []
        for _ in range(2):
            owner = self._owner()
            out = []
            for i, obs in enumerate(stream):
                for d in owner.submit(
                    PendingObservation(**{**obs.__dict__, "release_control_tick": i}),
                    control_tick=i,
                ):
                    out.append((d.frame_id, d.action.value))
            runs.append(out)
        assert runs[0] == runs[1]

    def test_flush_strands_nothing_at_end_of_stream(self):
        owner = self._owner()
        for i in range(10):
            owner.submit(make_obs(i, release_control_tick=i), control_tick=i)
        pending_before = len(owner.cohort)
        flushed = owner.flush(control_tick=100)
        assert len(flushed) == pending_before
        assert len(owner.cohort) == 0
        assert all(d.action is BackpressureAction.ADMIT for d in flushed)


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------
class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"age_axis": "nonsense"}, "age_axis"),
            ({"resolution_mode": "nonsense"}, "resolution_mode"),
            ({"pending_max_count": 0}, "pending_max_count"),
            ({"admission_window_control_ticks": 0}, "admission_window_control_ticks"),
            ({"max_admissions_per_window": -1}, "max_admissions_per_window"),
            ({"representatives_per_segment": 0}, "representatives_per_segment"),
            ({"frontier_quota": -1}, "frontier_quota"),
        ],
    )
    def test_invalid_config_is_refused(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            BackpressureConfig(**kwargs)

    def test_unknown_reason_code_is_refused(self):
        with pytest.raises(ValueError, match="reason must be one of"):
            BackpressureDecision(
                frame_id=1,
                action=BackpressureAction.ADMIT,
                reason="made_up_reason",
                admit_supervision=True,
            )


# ----------------------------------------------------------------------
# Args / CLI wiring
# ----------------------------------------------------------------------
class TestArgsWiring:
    def test_cli_flags_exist_with_matching_defaults(self):
        from eligsir.cli.parser import build_training_parser
        from eligsir.mapping.scheduler.backpressure import (
            backpressure_config_from_args,
        )

        parser = build_training_parser().parser
        defaults = {
            a.dest: a.default
            for a in parser._actions
            if a.dest and "backpressure" in a.dest
        }
        assert defaults, "no --streaming_backpressure* flags reached the parser"
        assert defaults["streaming_backpressure"] is False, (
            "the backpressure owner must default OFF (byte-identical runs)"
        )
        assert defaults["streaming_backpressure_force_all_stress"] is False
        assert defaults["streaming_backpressure_resolution_mode"] == "causal"
        assert defaults["streaming_backpressure_pending_max_age_control_ticks"] == 600
        assert defaults["streaming_backpressure_admission_window_control_ticks"] == 100
        assert "streaming_backpressure_pending_max_age_iters" not in defaults
        assert "streaming_backpressure_admission_window_iters" not in defaults

        class _Args:
            pass

        args = _Args()
        for dest, value in defaults.items():
            setattr(args, dest, value)
        cfg = backpressure_config_from_args(args)
        # The builder's own literals must agree with the parser defaults --
        # this is the drift the getattr-fallback audit exists to prevent.
        assert cfg.enabled is False
        assert cfg.resolution_mode == "causal"
        assert cfg.pending_max_count == defaults[
            "streaming_backpressure_pending_max_count"
        ]
        assert cfg.max_admissions_per_window == defaults[
            "streaming_backpressure_max_admissions_per_window"
        ]
        assert cfg.deterministic is True
        assert cfg.causal_deltas_are_cumulative is True

    def test_legacy_or_mixed_optimizer_iteration_args_fail_closed(self):
        from types import SimpleNamespace

        from eligsir.mapping.scheduler.backpressure import (
            backpressure_config_from_args,
        )

        for legacy_name in (
            "streaming_backpressure_pending_max_age_iters",
            "streaming_backpressure_admission_window_iters",
        ):
            args = SimpleNamespace(
                streaming_backpressure_pending_max_age_control_ticks=600,
                streaming_backpressure_admission_window_control_ticks=100,
                **{legacy_name: 100},
            )
            with pytest.raises(ValueError, match="legacy optimizer-iteration"):
                backpressure_config_from_args(args)

    def test_report_writer_accepts_the_backpressure_block(self):
        """Statically (AST) -- ``utils.comparison_report`` imports torch,
        which the CPU-only CI image does not have. Same technique
        ``tests/test_issue646_report_kwargs_signature.py`` uses, and for
        the same reason."""
        import ast
        import pathlib

        source = pathlib.Path("utils/comparison_report.py").read_text()
        tree = ast.parse(source)
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "write_post_training_report"
        )
        names = [a.arg for a in fn.args.args + fn.args.kwonlyargs]
        assert "backpressure" in names, (
            "report writer has no `backpressure` parameter -- "
            "tests/test_issue646_report_kwargs_signature.py would fail"
        )

    def test_cumulative_delta_mode_does_not_double_count(self):
        """The training loop reports deltas already accumulated since the
        last ADMIT; adding them again would fire the representative rule
        far too early."""
        cfg = BackpressureConfig(
            enabled=True,
            causal_deltas_are_cumulative=True,
            pending_max_count=1000,
            pending_max_age_control_ticks=1_000_000,
            max_admissions_per_window=1,
            admission_window_control_ticks=1_000_000,
            causal_admit_translation_m=0.10,
            redundant_translation_m=0.01,
        )
        owner = BackpressureOwner(cfg)
        # frame 0 consumes the whole budget; 1..4 report a cumulative
        # translation that never reaches the 0.10 m representative threshold.
        stream = [make_obs(0, translation_m=0.0, rotation_deg=0.0)] + [
            make_obs(i, translation_m=0.02 * i, rotation_deg=0.0, novelty=0.0)
            for i in range(1, 5)
        ]
        decisions, _ = run_stream(owner, stream)
        assert decisions[0].action is BackpressureAction.ADMIT
        assert all(d.skipped for d in decisions[1:]), (
            "cumulative deltas were re-accumulated (double counted)"
        )
