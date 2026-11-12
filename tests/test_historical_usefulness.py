"""CPU-only contracts for #953/#1191 historical usefulness scoring."""

import math
from types import SimpleNamespace

import pytest

from eligsir.mapping.scheduler.historical_candidate_index import (
    HistoricalCandidateConfig,
    HistoricalCandidateIndex,
    HistoricalFrameMetadata,
)
from eligsir.mapping.scheduler.historical_usefulness import (
    EmaUsefulnessTracker,
    MAX_WEIGHT,
    NEUTRAL_WEIGHT,
    UsefulnessObservation,
    usefulness_weight_from_loss,
)


class TestUsefulnessWeightFromLoss:
    def test_zero_loss_is_neutral(self):
        assert usefulness_weight_from_loss(0.0) == NEUTRAL_WEIGHT

    def test_negative_loss_clamped_to_neutral(self):
        assert usefulness_weight_from_loss(-5.0) == NEUTRAL_WEIGHT

    def test_monotonic_in_loss(self):
        low = usefulness_weight_from_loss(0.1)
        mid = usefulness_weight_from_loss(1.0)
        high = usefulness_weight_from_loss(10.0)
        assert NEUTRAL_WEIGHT < low < mid < high < MAX_WEIGHT

    def test_bounded_above(self):
        assert usefulness_weight_from_loss(1e9) <= MAX_WEIGHT
        # A large-but-finite loss should already be very close to the
        # ceiling without needing an astronomical input to get there.
        assert usefulness_weight_from_loss(1e6) > MAX_WEIGHT - 0.01

    def test_loss_scale_rescales(self):
        # The same raw loss reads as "more severe" against a smaller scale.
        assert usefulness_weight_from_loss(
            1.0, loss_scale=0.1
        ) > usefulness_weight_from_loss(1.0, loss_scale=10.0)

    @pytest.mark.parametrize("bad_loss", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_loss(self, bad_loss):
        with pytest.raises(ValueError):
            usefulness_weight_from_loss(bad_loss)

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_loss_scale(self, bad_scale):
        with pytest.raises(ValueError):
            usefulness_weight_from_loss(1.0, loss_scale=bad_scale)


class TestEmaUsefulnessTracker:
    def test_unobserved_frame_returns_default(self):
        tracker = EmaUsefulnessTracker()
        assert tracker.weight(42) == NEUTRAL_WEIGHT
        assert tracker.weight(42, default=2.5) == 2.5

    def test_single_observation_is_not_trusted(self):
        tracker = EmaUsefulnessTracker(min_observations=3)
        obs = tracker.observe(1, 4.0)
        assert isinstance(obs, UsefulnessObservation)
        assert obs.trusted is False
        # Not yet trusted -> weight() falls back to the neutral default even
        # though a single wild observation was folded in. This is the crux
        # of the #1191 "one noisy event must never permanently promote or
        # demote a frame" hysteresis contract.
        assert tracker.weight(1) == NEUTRAL_WEIGHT

    def test_becomes_trusted_after_min_observations(self):
        tracker = EmaUsefulnessTracker(min_observations=3, alpha=0.5)
        for _ in range(2):
            obs = tracker.observe(7, 3.0)
            assert obs.trusted is False
        obs = tracker.observe(7, 3.0)
        assert obs.trusted is True
        assert tracker.weight(7) == pytest.approx(3.0)

    def test_ema_smooths_a_single_spike(self):
        tracker = EmaUsefulnessTracker(min_observations=1, alpha=0.2)
        tracker.observe(9, 1.0)
        tracker.observe(9, 1.0)
        spiked = tracker.observe(9, 100.0)
        # One noisy 100x spike should move the smoothed estimate only a
        # bounded amount, not overwrite it outright.
        assert spiked.smoothed_raw < 25.0
        assert spiked.smoothed_raw > 1.0

    def test_independent_frames_do_not_interfere(self):
        tracker = EmaUsefulnessTracker(min_observations=1)
        tracker.observe(1, 5.0)
        tracker.observe(2, 1.0)
        assert tracker.weight(1) != tracker.weight(2)

    def test_bounded_memory_evicts_lru(self):
        tracker = EmaUsefulnessTracker(min_observations=1, max_tracked_frames=2)
        tracker.observe(1, 1.0)
        tracker.observe(2, 2.0)
        tracker.observe(3, 3.0)
        assert len(tracker) == 2
        # Frame 1 was least-recently touched and should have been evicted.
        assert tracker.weight(1, default=-1.0) == -1.0

    def test_forget_drops_tracked_state(self):
        tracker = EmaUsefulnessTracker(min_observations=1)
        tracker.observe(5, 3.0)
        tracker.forget(5)
        assert tracker.weight(5, default=-1.0) == -1.0

    @pytest.mark.parametrize("bad_alpha", [0.0, -0.1, 1.1, float("nan")])
    def test_rejects_bad_alpha(self, bad_alpha):
        with pytest.raises(ValueError):
            EmaUsefulnessTracker(alpha=bad_alpha)

    def test_rejects_non_finite_observation(self):
        tracker = EmaUsefulnessTracker()
        with pytest.raises(ValueError):
            tracker.observe(1, float("nan"))


class TestHistoricalCandidateIndexUpdateBaseWeight:
    def _index(self):
        return HistoricalCandidateIndex(
            config=HistoricalCandidateConfig(
                shortlist_size=64,
                recent_lane_size=8,
                focus_lane_size=8,
                protected_lane_size=4,
                global_reservoir_size=16,
                global_floor=4,
            )
        )

    def test_update_changes_shortlist_weight(self):
        index = self._index()
        index.register(HistoricalFrameMetadata(frame_id=1, region_id=0, base_weight=1.0))
        assert index.metadata_for(1).base_weight == 1.0
        assert index.update_base_weight(1, 3.0) is True
        assert index.metadata_for(1).base_weight == 3.0

    def test_update_unregistered_frame_returns_false(self):
        index = self._index()
        assert index.update_base_weight(999, 3.0) is False

    def test_update_after_retire_returns_false(self):
        index = self._index()
        index.register(HistoricalFrameMetadata(frame_id=2, region_id=0, base_weight=1.0))
        index.retire(2)
        assert index.update_base_weight(2, 3.0) is False

    @pytest.mark.parametrize("bad_weight", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_weight(self, bad_weight):
        index = self._index()
        index.register(HistoricalFrameMetadata(frame_id=3, region_id=0, base_weight=1.0))
        with pytest.raises(ValueError):
            index.update_base_weight(3, bad_weight)

    def test_update_preserves_other_metadata_fields(self):
        index = self._index()
        index.register(
            HistoricalFrameMetadata(frame_id=4, region_id="p2", base_weight=1.0, protected=True)
        )
        index.update_base_weight(4, 2.0)
        meta = index.metadata_for(4)
        assert meta.region_id == "p2"
        assert meta.protected is True
        assert meta.frame_id == 4


class _HashableSource:
    """Plain identity-hashable stand-in with a ``source_ordinal`` attribute.

    ``SimpleNamespace`` defines ``__eq__`` by value, which makes it
    unhashable -- a real ``SourceIdentity`` is hashable, so this stub must be
    too to serve as a dict key the way production does.
    """

    def __init__(self, source_ordinal):
        self.source_ordinal = source_ordinal


class TestHistoricalUsefulnessBySourceOrdinal:
    """Durable base-weight export for #116 TARGETED_HISTORY ranking."""

    def _make_self(self, *, tracking_enabled, frame_id=1, ordinal=7, base_weight=1.0):
        from eligsir.mapping.scheduler.historical_candidate_index import (
            HistoricalCandidateConfig,
            HistoricalCandidateIndex,
            HistoricalFrameMetadata,
        )

        index = HistoricalCandidateIndex(
            config=HistoricalCandidateConfig(
                shortlist_size=64, recent_lane_size=8, focus_lane_size=8,
                protected_lane_size=4, global_reservoir_size=16, global_floor=4,
            )
        )
        index.register(
            HistoricalFrameMetadata(frame_id=frame_id, region_id=0, base_weight=base_weight)
        )
        source = _HashableSource(ordinal)
        fake_self = SimpleNamespace(
            _optimization_policy=SimpleNamespace(
                streaming_historical_usefulness_tracking=tracking_enabled,
            ),
            _historical_frame_ids_by_source={source: frame_id},
            _historical_candidate_index=index,
        )
        return fake_self

    def _bind(self, fake_self):
        from eligsir.runtime.production_application import CanonicalMappingRuntime

        return CanonicalMappingRuntime.historical_usefulness_by_source_ordinal.__get__(
            fake_self
        )

    def test_off_by_default_returns_empty(self):
        fake_self = self._make_self(tracking_enabled=False, base_weight=3.0)
        export = self._bind(fake_self)
        assert export() == {}

    def test_on_exports_base_weight_by_source_ordinal(self):
        fake_self = self._make_self(tracking_enabled=True, ordinal=7, base_weight=3.0)
        export = self._bind(fake_self)
        assert export() == {7: 3.0}

    def test_source_missing_ordinal_is_omitted(self):
        from eligsir.mapping.scheduler.historical_candidate_index import (
            HistoricalCandidateConfig,
            HistoricalCandidateIndex,
            HistoricalFrameMetadata,
        )

        index = HistoricalCandidateIndex(
            config=HistoricalCandidateConfig(
                shortlist_size=64, recent_lane_size=8, focus_lane_size=8,
                protected_lane_size=4, global_reservoir_size=16, global_floor=4,
            )
        )
        index.register(HistoricalFrameMetadata(frame_id=1, region_id=0, base_weight=2.0))
        fake_self = SimpleNamespace(
            _optimization_policy=SimpleNamespace(
                streaming_historical_usefulness_tracking=True,
            ),
            _historical_frame_ids_by_source={object(): 1},
            _historical_candidate_index=index,
        )
        export = self._bind(fake_self)
        assert export() == {}

    def test_no_candidate_index_returns_empty(self):
        fake_self = SimpleNamespace(
            _optimization_policy=SimpleNamespace(
                streaming_historical_usefulness_tracking=True,
            ),
            _historical_frame_ids_by_source={},
            _historical_candidate_index=None,
        )
        export = self._bind(fake_self)
        assert export() == {}
