"""Unit tests for the pure MCMC-suppression-by-insertion-offset decision
function (evidence-driven-mutation-cadence experiment, Task 5 prep / C4 arm).

No torch/GPU dependency -- runs with plain pytest, no Docker required.
"""

from eligsir.mapping.integration.insertion_cadence import (
    _mcmc_suppressed_by_insertion_offset,
)


class TestOffsetDisabledByDefault:
    def test_zero_offset_never_suppresses(self):
        assert _mcmc_suppressed_by_insertion_offset(
            iteration=100, last_flush_iteration=99, offset=0,
        ) is False

    def test_negative_offset_never_suppresses(self):
        assert _mcmc_suppressed_by_insertion_offset(
            iteration=100, last_flush_iteration=100, offset=-5,
        ) is False


class TestOffsetSuppression:
    def test_suppressed_immediately_after_flush(self):
        assert _mcmc_suppressed_by_insertion_offset(
            iteration=100, last_flush_iteration=100, offset=50,
        ) is True

    def test_suppressed_within_window(self):
        assert _mcmc_suppressed_by_insertion_offset(
            iteration=130, last_flush_iteration=100, offset=50,
        ) is True

    def test_not_suppressed_exactly_at_window_edge(self):
        assert _mcmc_suppressed_by_insertion_offset(
            iteration=150, last_flush_iteration=100, offset=50,
        ) is False

    def test_not_suppressed_well_beyond_window(self):
        assert _mcmc_suppressed_by_insertion_offset(
            iteration=500, last_flush_iteration=100, offset=50,
        ) is False
