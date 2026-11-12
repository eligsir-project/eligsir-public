"""Test asynchronous support snapshot consumption and topology-aware staleness detection (#1601 slice 2)."""

from types import SimpleNamespace
import torch
import dataclasses

import pytest
import eligsir.utils.assimilation_pressure as pressure
from eligsir.mapping.optimization_runtime import StreamingOptimizationRuntime


class TestProvenanceAttachmentToSummary:
    """Test that provenance fields are attached to AssimilationSupportSummary at latch time."""

    def test_summary_has_provenance_fields(self):
        """AssimilationSupportSummary must include provenance fields."""
        # Check that the dataclass has the required fields
        summary_fields = {f.name for f in dataclasses.fields(pressure.AssimilationSupportSummary)}
        required_provenance_fields = {
            "optimizer_updates",
            "topology_version",
            "parameter_version",
            "population_size",
        }
        assert required_provenance_fields.issubset(summary_fields), (
            f"Summary missing provenance fields: {required_provenance_fields - summary_fields}"
        )

    def test_provenance_attached_at_latch_time(self):
        """Provenance is attached to the summary when it's latched."""
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

        support_counts = torch.tensor([1.0, 2.0, 3.0, 4.0])
        optimizer_updates = 42
        topology_version = 7
        parameter_version = 3
        population_size = 4

        debt = pressure.publish_support_counts(
            support_counts,
            min_support_views=2,
            optimizer_updates=optimizer_updates,
            topology_version=topology_version,
            parameter_version=parameter_version,
            population_size=population_size,
        )

        summary = pressure.latest_assimilation_support_summary()
        assert summary is not None
        assert summary.optimizer_updates == optimizer_updates
        assert summary.topology_version == topology_version
        assert summary.parameter_version == parameter_version
        assert summary.population_size == population_size

    def test_provenance_defaults_to_zero(self):
        """Provenance fields default to 0 when not provided."""
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

        support_counts = torch.tensor([1.0, 2.0])
        debt = pressure.publish_support_counts(support_counts, min_support_views=1)

        summary = pressure.latest_assimilation_support_summary()
        assert summary is not None
        assert summary.optimizer_updates == 0
        assert summary.topology_version == 0
        assert summary.parameter_version == 0
        assert summary.population_size == 0

    def test_provenance_survives_roundtrip(self):
        """Provenance is preserved in the immutable snapshot."""
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

        support_counts = torch.tensor([2.0, 0.0, 5.0])
        debt1 = pressure.publish_support_counts(
            support_counts,
            min_support_views=2,
            optimizer_updates=10,
            topology_version=5,
            parameter_version=2,
            population_size=3,
        )

        summary1 = pressure.latest_assimilation_support_summary()
        assert summary1.topology_version == 5

        # Publish again with different provenance
        debt2 = pressure.publish_support_counts(
            support_counts,
            min_support_views=2,
            optimizer_updates=11,
            topology_version=6,
            parameter_version=3,
            population_size=3,
        )

        summary2 = pressure.latest_assimilation_support_summary()
        assert summary2.topology_version == 6
        assert summary2 != summary1  # Different provenance makes it a different snapshot


class TestControllerStalenessDetection:
    """Test that controller detects stale snapshots when topology changes."""

    def test_controller_detects_stale_topology(self):
        """Controller returns no-op when snapshot topology differs from current."""
        pressure.configure_assimilation_control(enabled=True)
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None
        pressure._LATEST_SIGNALS = {
            "source_insert_rate": 0.0,
            "support_debt": 0.5,
            "residual_pressure": 0.0,
            "partition_pressure": 0.0,
            "newborn_pressure": 0.0,
            "optimization_rate": 1.0,
        }

        # Publish a snapshot with topology version 5
        support_counts = torch.tensor([2.0, 1.0, 3.0])
        pressure.publish_support_counts(
            support_counts,
            min_support_views=2,
            optimizer_updates=10,
            topology_version=5,
            parameter_version=2,
            population_size=3,
        )

        # Request decision with different topology (6)
        decision = pressure.active_control_decision(
            base_steps=4,
            current_topology_version=6,
        )

        assert decision is not None
        assert "stale" in decision.reason.lower()
        assert decision.optimizer_steps == 4  # Base steps unchanged (no-op)

        pressure.configure_assimilation_control(enabled=False)

    def test_controller_uses_current_snapshot_when_topology_matches(self):
        """Controller uses the snapshot's decision when topology matches."""
        pressure.configure_assimilation_control(enabled=True)
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None
        pressure._LATEST_SIGNALS = {
            "source_insert_rate": 0.0,
            "support_debt": 0.05,  # LOW pressure
            "residual_pressure": 0.0,
            "partition_pressure": 0.0,
            "newborn_pressure": 0.0,
            "optimization_rate": 1.0,
        }

        # Publish a snapshot with topology version 5
        support_counts = torch.tensor([2.0, 1.0, 3.0, 4.0])
        pressure.publish_support_counts(
            support_counts,
            min_support_views=2,
            optimizer_updates=10,
            topology_version=5,
            parameter_version=2,
            population_size=4,
        )

        # Request decision with matching topology
        decision = pressure.active_control_decision(
            base_steps=4,
            current_topology_version=5,
        )

        assert decision is not None
        assert decision.tier == pressure.AssimilationPressureTier.LOW
        assert "stale" not in decision.reason.lower()

        pressure.configure_assimilation_control(enabled=False)

    def test_controller_skip_staleness_check_when_topology_zero(self):
        """When topology_version is 0, staleness check is skipped (backwards compatible)."""
        pressure.configure_assimilation_control(enabled=True)
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None
        pressure._LATEST_SIGNALS = {
            "source_insert_rate": 0.0,
            "support_debt": 0.5,  # MEDIUM pressure
            "residual_pressure": 0.0,
            "partition_pressure": 0.0,
            "newborn_pressure": 0.0,
            "optimization_rate": 1.0,
        }

        # Publish a snapshot with topology version 0 (unknown)
        support_counts = torch.tensor([2.0])
        pressure.publish_support_counts(
            support_counts,
            min_support_views=2,
            optimizer_updates=10,
            topology_version=0,  # Unknown topology
            parameter_version=0,
            population_size=1,
        )

        # Request decision with topology_version=0 (default, no check)
        decision = pressure.active_control_decision(
            base_steps=4,
            current_topology_version=0,  # No check when both are 0
        )

        assert decision is not None
        assert "stale" not in decision.reason.lower()

        pressure.configure_assimilation_control(enabled=False)

    def test_controller_fallback_to_previous_tier_on_stale(self):
        """On stale snapshot, controller preserves previous tier decision."""
        pressure.configure_assimilation_control(enabled=True)
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None
        pressure._LATEST_SIGNALS = {
            "source_insert_rate": 0.0,
            "support_debt": 0.05,  # LOW
            "residual_pressure": 0.0,
            "partition_pressure": 0.0,
            "newborn_pressure": 0.0,
            "optimization_rate": 1.0,
        }

        # First decision: LOW pressure
        support_counts = torch.tensor([2.0, 1.0])
        pressure.publish_support_counts(
            support_counts,
            min_support_views=2,
            topology_version=5,
            population_size=2,
        )
        dec1 = pressure.active_control_decision(4, current_topology_version=5)
        assert dec1.tier == pressure.AssimilationPressureTier.LOW

        # Now topology changes to 6, but snapshot still shows old topology
        dec2 = pressure.active_control_decision(4, current_topology_version=6)
        assert dec2.tier == pressure.AssimilationPressureTier.LOW  # Preserved from previous
        assert "stale" in dec2.reason.lower()

        pressure.configure_assimilation_control(enabled=False)


class TestTopologyMutationInvalidation:
    """Test that topology grow/prune invalidates old population size in next publish."""

    def test_population_size_updated_after_grow(self):
        """After topology grow, next publish uses new population size."""
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

        # Initial publish with 3 Gaussians
        support_counts_1 = torch.tensor([2.0, 1.0, 3.0])
        pressure.publish_support_counts(
            support_counts_1,
            min_support_views=2,
            topology_version=5,
            population_size=3,
        )
        summary_1 = pressure.latest_assimilation_support_summary()
        assert summary_1.population_size == 3

        # After topology grow, publish with 5 Gaussians
        support_counts_2 = torch.tensor([2.0, 1.0, 3.0, 1.0, 2.0])
        pressure.publish_support_counts(
            support_counts_2,
            min_support_views=2,
            topology_version=6,
            population_size=5,  # New population size
        )
        summary_2 = pressure.latest_assimilation_support_summary()
        assert summary_2.population_size == 5
        assert summary_2.population_size != summary_1.population_size

    def test_population_size_updated_after_prune(self):
        """After topology prune, next publish uses reduced population size."""
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

        # Initial publish with 5 Gaussians
        support_counts_1 = torch.tensor([2.0, 1.0, 3.0, 1.0, 2.0])
        pressure.publish_support_counts(
            support_counts_1,
            min_support_views=2,
            topology_version=5,
            population_size=5,
        )
        summary_1 = pressure.latest_assimilation_support_summary()
        assert summary_1.population_size == 5

        # After topology prune, publish with 3 Gaussians
        support_counts_2 = torch.tensor([2.0, 1.0, 3.0])
        pressure.publish_support_counts(
            support_counts_2,
            min_support_views=2,
            topology_version=6,
            population_size=3,  # Pruned
        )
        summary_2 = pressure.latest_assimilation_support_summary()
        assert summary_2.population_size == 3


class TestQueueBoundedness:
    """Test that summary queue remains bounded over many publishes."""

    def test_queue_bounded_over_many_publishes(self):
        """Only one summary is latched; history is not accumulated."""
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

        support_counts = torch.tensor([1.0, 2.0])

        # Publish 10,000 times
        for i in range(10000):
            pressure.publish_support_counts(
                support_counts,
                min_support_views=2,
                optimizer_updates=i,
                topology_version=5,
            )

        # Only the latest summary should be latched
        summary = pressure.latest_assimilation_support_summary()
        assert summary is not None
        assert summary.optimizer_updates == 9999  # Last one

        # The queue is bounded (no memory leak)
        # This is verified by the fact that only one summary is stored
        assert pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY is summary


class TestPR1661GateUnchanged:
    """Test that PR #1661 gate (skip readback when control disabled and telemetry not due) remains intact."""

    def test_pr1661_gate_disabled_control_no_readback_not_due(self):
        """When control disabled and telemetry not due, publish_support_counts is not called."""
        pressure.configure_assimilation_control(enabled=False)

        call_count = [0]
        original_publish = pressure.publish_support_counts

        def counting_publish(*args, **kwargs):
            call_count[0] += 1
            return original_publish(*args, **kwargs)

        pressure.publish_support_counts = counting_publish

        try:
            runtime = StreamingOptimizationRuntime(
                session=SimpleNamespace(model=object()),
                gaussians=SimpleNamespace(
                    get_xyz=torch.zeros((2, 3)),
                    support_count=torch.ones(2),
                ),
                args=SimpleNamespace(streaming_min_support_views=2),
                opt=object(),
                pipe=object(),
                mcmc_strategy=None,
                tb_writer=SimpleNamespace(add_scalar=lambda *_args: None),
                optimizer_type="adam",
                sparse_active_set=False,
                use_energy_mcmc=False,
                fixed_shape_mcmc=False,
                mcmc_local=False,
                loss_assembly_kwargs={"scalar_log_interval": 5},
                iter_since_new_frame=0,
            )

            # Iteration 4: not due (4 % 5 != 0)
            runtime.publish_assimilation_support(4)
            assert call_count[0] == 0, "Should not call publish_support_counts when not due"

            # Iteration 5: due (5 % 5 == 0)
            runtime.publish_assimilation_support(5)
            assert call_count[0] == 1, "Should call publish_support_counts when due"
        finally:
            pressure.publish_support_counts = original_publish
            pressure.configure_assimilation_control(enabled=False)


class TestPublishAssimilationSupportProvenance:
    """Test that publish_assimilation_support passes provenance to publish_support_counts."""

    def test_publish_assimilation_support_passes_provenance(self):
        """publish_assimilation_support passes provenance parameters."""
        pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

        runtime = StreamingOptimizationRuntime(
            session=SimpleNamespace(model=object()),
            gaussians=SimpleNamespace(
                get_xyz=torch.zeros((3, 3)),
                support_count=torch.tensor([2.0, 1.0, 3.0]),
            ),
            args=SimpleNamespace(streaming_min_support_views=2),
            opt=object(),
            pipe=object(),
            mcmc_strategy=None,
            tb_writer=SimpleNamespace(add_scalar=lambda *_args: None),
            optimizer_type="adam",
            sparse_active_set=False,
            use_energy_mcmc=False,
            fixed_shape_mcmc=False,
            mcmc_local=False,
            loss_assembly_kwargs={"scalar_log_interval": 1},
            iter_since_new_frame=0,
        )

        # Call publish_assimilation_support with provenance
        runtime.publish_assimilation_support(
            iteration=10,
            optimizer_updates=42,
            topology_version=7,
            parameter_version=3,
        )

        # Check that provenance was attached to the summary
        summary = pressure.latest_assimilation_support_summary()
        assert summary is not None
        assert summary.optimizer_updates == 42
        assert summary.topology_version == 7
        assert summary.parameter_version == 3
        assert summary.population_size == 3  # Extracted from support.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
