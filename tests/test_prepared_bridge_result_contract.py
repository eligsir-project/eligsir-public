import pytest

from eligsir.runtime.prepared_runtime_bridge import PreparedRuntimeBridgeResult


def test_bridge_result_requires_exactly_one_outcome_per_submission():
    assert PreparedRuntimeBridgeResult(2, 1, 1, 0, 0).submitted == 2
    with pytest.raises(ValueError, match="exactly one outcome"):
        PreparedRuntimeBridgeResult(1, 1, 1, 0, 0)
    with pytest.raises(ValueError, match="exactly one outcome"):
        PreparedRuntimeBridgeResult(2, 1, 0, 0, 0)
