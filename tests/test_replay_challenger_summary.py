"""Published causal-replay challenger telemetry contract."""

import json
from dataclasses import MISSING, fields
import pytest

from eligsir.runtime.causal_entry import causal_replay_summary_payload
from eligsir.runtime.causal_replay import CausalReplaySummary


_METRICS = {
    "observations_total": 10,
    "trusted_updates_total": 4,
    "replay_challenger_trials_authorized": 3,
    "replay_challenger_trials_completed": 2,
    "replay_challenger_demotions": 1,
    "replay_challenger_turnover_blocked": 4,
    "replay_challenger_no_ready": 5,
    "feedback_busy": 6,
    "ready_replay_evicted_bytes": 7,
}


def _summary() -> CausalReplaySummary:
    values = {}
    for field in fields(CausalReplaySummary):
        if field.default is not MISSING:
            values[field.name] = field.default
        elif field.default_factory is not MISSING:
            values[field.name] = field.default_factory()
    values.update(submitted=0, ticks=0, source_identities=())
    return CausalReplaySummary(**values)


class _Runtime:
    def __init__(self, telemetry):
        self.telemetry = telemetry
        self.calls = 0

    def historical_usefulness_telemetry(self):
        self.calls += 1
        return self.telemetry


def test_summary_publishes_runtime_replay_challenger_telemetry_as_json():
    runtime = _Runtime(_METRICS)

    payload = causal_replay_summary_payload(
        _summary(),
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        supervision_work_block={},
        runtime=runtime,
    )

    assert runtime.calls == 1
    assert json.loads(json.dumps(payload))["replay_challenger"] == _METRICS


def test_summary_publishes_zeroed_off_baseline_from_runtime():
    runtime = _Runtime({key: 0 for key in _METRICS})

    payload = causal_replay_summary_payload(
        _summary(),
        source_hz=None,
        curve_metadata=None,
        ready_bytes_block={},
        supervision_work_block={},
        runtime=runtime,
    )

    assert runtime.calls == 1
    assert payload["replay_challenger"] == {key: 0 for key in _METRICS}


@pytest.mark.parametrize("telemetry", ([], {"feedback_busy": True}, {"feedback_busy": "1"}, {"feedback_busy": -1}))
def test_summary_does_not_coerce_malformed_runtime_counters(telemetry):
    with pytest.raises(TypeError, match="integer counters"):
        causal_replay_summary_payload(
            _summary(), source_hz=None, curve_metadata=None,
            ready_bytes_block={}, supervision_work_block={}, runtime=_Runtime(telemetry),
        )


def test_summary_without_runtime_does_not_fabricate_zero_interventions():
    payload = causal_replay_summary_payload(
        _summary(), source_hz=None, curve_metadata=None,
        ready_bytes_block={}, supervision_work_block={},
    )
    assert payload["replay_challenger"] is None
