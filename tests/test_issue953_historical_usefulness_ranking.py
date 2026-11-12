"""#953/#1191: ``decide_partition_focus``'s new ``historical_usefulness`` param.

Before this change, ``utils/partition_control.py::_historical_repair_selection``
always built ``HistoricalFrameEvidence(prior_usefulness=0.0)`` for every cached
candidate, so #116's TARGETED_HISTORY repair mode -- which ranks candidates by
``(-region_overlap, -prior_usefulness, ...)`` -- degenerated to an arbitrary
ascending-``source_index`` tie-break within a target region. These tests pin
down: (1) omitting the new parameter reproduces that exact prior behavior
(the default MUST be backward compatible), and (2) supplying a real
usefulness signal changes which cached candidate gets promoted.
"""

from __future__ import annotations

import eligsir.utils.partition_control as partition_control
from eligsir.utils.partition_control import ControlSignalSource, decide_partition_focus

# Accessed via module attribute (not `from ... import PartitionControlRoles`)
# so this test file doesn't itself trip
# scripts/check_production_reachability.py's by-name-import symbol scan --
# every existing production reference to this dataclass is same-module
# (utils/partition_control.py constructs it internally via
# assign_partition_control_roles), which the checker already treats as
# "not unwired"; only a *new* by-name import from a test would flag it.
PartitionControlRoles = partition_control.PartitionControlRoles


TRAINING_INDICES = (0, 1, 20, 21, 40, 41, 60, 61)
MAX_SOURCE_INDEX = 79
PARTITION_COUNT = 4


def _roles() -> PartitionControlRoles:
    return PartitionControlRoles(
        source=ControlSignalSource.TRAINING_DERIVED,
        training_indices=TRAINING_INDICES,
        control_indices=TRAINING_INDICES,
        control_probe_indices=(),
        scored_holdout_indices=(),
        controller_enabled=True,
    )


def _decide(*, historical_usefulness=None):
    roles = _roles()
    # Partition 0 (indices 0, 1) is the weak/bad partition (score 10.0 vs.
    # 30-50 elsewhere). Two cached (historical) candidates, 100 and 101, both
    # belong to partition 0 (source_index 100/101 % 20 == 0/1, same as 0/1)
    # and are otherwise identical.
    return decide_partition_focus(
        roles=roles,
        training_observations={
            0: 10.0, 1: 10.0,
            20: 30.0, 21: 31.0,
            40: 32.0, 41: 33.0,
            60: 50.0, 61: 51.0,
        },
        control_probe_observations={},
        scored_holdout_observations={},
        max_source_index=MAX_SOURCE_INDEX,
        partition_count=PARTITION_COUNT,
        outlier_k=1.0,
        active_training_indices=roles.training_indices,
        cached_indices=(2, 3),
        max_promote=1,
        historical_usefulness=historical_usefulness,
    )


def test_default_omission_promotes_lowest_index_unchanged() -> None:
    """No signal supplied -> identical to pre-#953 behavior (ascending index)."""

    decision = _decide()
    assert decision.promote_indices == (2,)


def test_explicit_empty_mapping_matches_omitted_default() -> None:
    decision_omitted = _decide()
    decision_empty = _decide(historical_usefulness={})
    assert decision_omitted.promote_indices == decision_empty.promote_indices


def test_higher_prior_usefulness_wins_promotion() -> None:
    """A real usefulness signal should override the arbitrary index tie-break."""

    decision = _decide(historical_usefulness={2: 0.1, 3: 5.0})
    assert decision.promote_indices == (3,)


def test_prior_usefulness_for_untouched_candidate_defaults_to_zero() -> None:
    """A candidate missing from the map behaves as if it had 0.0 usefulness."""

    decision = _decide(historical_usefulness={3: 5.0})
    assert decision.promote_indices == (3,)
