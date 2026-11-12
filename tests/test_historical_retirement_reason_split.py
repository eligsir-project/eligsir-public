from __future__ import annotations

from collections import OrderedDict

from eligsir.runtime.historical_retirement import (
    install_modality_aware_historical_retirement,
)


class _CandidateIndex:
    def __init__(self) -> None:
        self.retired = []

    def retire(self, frame_id: int) -> None:
        self.retired.append(int(frame_id))


class _DB:
    def __init__(self, *, materializable: bool, rgb_eligible: bool) -> None:
        self.materializable = materializable
        self.rgb_eligible = rgb_eligible

    def historical_candidate_for_source(self, _source, *, require_rgb=False):
        if not self.materializable:
            return None
        if require_rgb and not self.rgb_eligible:
            return None
        return object()


class _Runtime:
    def __init__(self, db: _DB) -> None:
        self._observation_db = db
        self._historical_sources_by_frame_id = {7: "source-7"}
        self._historical_frame_ids_by_source = {"source-7": 7}
        self._historical_candidate_index = _CandidateIndex()
        self._replay_modalities = {"source-7": object()}
        self._replay_weights = {"source-7": 1.0}
        self._pending_supervision_modalities = {"source-7": object()}
        self._replay_focus_frame_weights = ((7, 1.0),)
        self._replay_focus_sources = ("source-7",)
        self._retired_replay_sources = OrderedDict()
        self._retired_unmaterializable_total = 0
        self.original_retire_calls = 0

    def _retire_historical_identity(self, frame_id: int, *, reason: str = "unmaterializable") -> bool:
        self.original_retire_calls += 1
        self._retired_unmaterializable_total += 1
        return True


def test_materializable_depth_only_record_is_not_counted_as_unmaterializable() -> None:
    runtime = _Runtime(_DB(materializable=True, rgb_eligible=False))
    install_modality_aware_historical_retirement(runtime)

    assert runtime._retire_historical_identity(7) is True

    assert runtime.original_retire_calls == 0
    assert runtime._retired_unmaterializable_total == 0
    assert runtime._retired_rgb_ineligible_total == 1
    assert runtime._historical_sources_by_frame_id == {7: "source-7"}
    assert runtime._historical_frame_ids_by_source == {"source-7": 7}
    assert runtime._historical_candidate_index.retired == [7]
    assert "source-7" not in runtime._replay_modalities
    assert "source-7" not in runtime._replay_weights
    assert "source-7" not in runtime._pending_supervision_modalities
    assert runtime._replay_focus_frame_weights == ()
    assert runtime._replay_focus_sources == ()
    assert "source-7" in runtime._retired_replay_sources


def test_true_unmaterializable_record_keeps_original_eviction_accounting() -> None:
    runtime = _Runtime(_DB(materializable=False, rgb_eligible=False))
    install_modality_aware_historical_retirement(runtime)

    assert runtime._retire_historical_identity(7) is True

    assert runtime.original_retire_calls == 1
    assert runtime._retired_unmaterializable_total == 1
    assert not hasattr(runtime, "_retired_rgb_ineligible_total")


def test_installation_is_idempotent_and_does_not_double_count() -> None:
    runtime = _Runtime(_DB(materializable=True, rgb_eligible=False))
    install_modality_aware_historical_retirement(runtime)
    first_wrapper = runtime._retire_historical_identity
    install_modality_aware_historical_retirement(runtime)

    assert runtime._retire_historical_identity == first_wrapper
    assert runtime._retire_historical_identity(7) is True
    assert runtime._retire_historical_identity(7) is True
    assert runtime._retired_rgb_ineligible_total == 1


def test_wrapper_forwards_reason_keyword_argument() -> None:
    """Test that the wrapper correctly forwards the reason keyword argument.

    This regression test ensures that calls like:
        self._retire_historical_identity(frame_id, reason="rgb_supervision_ineligible")
    work correctly after the wrapper is installed.
    """

    class _RuntimeWithKwargTracking:
        def __init__(self, db: _DB) -> None:
            self._observation_db = db
            self._historical_sources_by_frame_id = {}
            self._historical_candidate_index = _CandidateIndex()
            self.original_calls = []

        def _retire_historical_identity(self, frame_id: int, *, reason: str = "unmaterializable") -> bool:
            self.original_calls.append({"frame_id": frame_id, "reason": reason})
            return True

    # Test with a case where the DB is None, so we hit the first fallthrough
    runtime = _RuntimeWithKwargTracking(_DB(materializable=False, rgb_eligible=False))
    install_modality_aware_historical_retirement(runtime)

    # Call with default reason
    assert runtime._retire_historical_identity(7) is True
    assert runtime.original_calls[-1] == {"frame_id": 7, "reason": "unmaterializable"}

    # Call with rgb_supervision_ineligible reason (the bug case from production)
    assert runtime._retire_historical_identity(8, reason="rgb_supervision_ineligible") is True
    assert runtime.original_calls[-1] == {"frame_id": 8, "reason": "rgb_supervision_ineligible"}
