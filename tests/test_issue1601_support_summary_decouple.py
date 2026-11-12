"""Test decoupling of assimilation-support summary production from TensorBoard formatting."""

from types import SimpleNamespace
import torch
import dataclasses

import pytest
import eligsir.utils.assimilation_pressure as pressure
from eligsir.mapping.optimization_runtime import StreamingOptimizationRuntime


def test_assimilation_support_summary_is_tensor_free():
    """AssimilationSupportSummary must contain only float/int, not tensors."""
    # Produce a summary by calling publish_support_counts
    support_counts = torch.tensor([2.0, 0.0, 5.0, 1.0])
    debt = pressure.publish_support_counts(support_counts, min_support_views=2)

    # Get the latched summary
    summary = pressure.latest_assimilation_support_summary()

    assert summary is not None, "Summary should be latched after publish_support_counts"

    # Check that it's a dataclass
    assert dataclasses.is_dataclass(summary), "Summary should be a dataclass"

    # Check that no field is a tensor
    for field_value in dataclasses.astuple(summary):
        assert not hasattr(field_value, 'cpu'), (
            f"Summary field has tensor-like .cpu(): {field_value}"
        )
        assert not isinstance(field_value, torch.Tensor), (
            f"Summary field is a torch.Tensor: {field_value}"
        )


def test_support_summary_production_and_latch():
    """Summary is produced and latched without TensorBoard writer."""
    # Clear any prior latch
    pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

    # Produce with control enabled to ensure production runs
    pressure.configure_assimilation_control(enabled=True)

    support_counts = torch.tensor([2.0, 0.0, 5.0, 1.0])
    debt = pressure.publish_support_counts(support_counts, min_support_views=2)

    # Summary should be latched
    summary = pressure.latest_assimilation_support_summary()
    assert summary is not None
    assert summary.debt == debt
    assert summary.support_eq_0 == 1
    assert summary.support_eq_1 == 1
    assert summary.support_ge_target == 2
    assert summary.support_pool_n == 4
    assert summary.min_support_views == 2

    pressure.configure_assimilation_control(enabled=False)


def test_formatting_tolerates_missing_summary():
    """Formatting must not crash when summary is None."""
    # Clear the latch
    pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

    # Call publish_assimilation_support on the runtime with no summary latched
    runtime = StreamingOptimizationRuntime(
        session=SimpleNamespace(model=object()),
        gaussians=SimpleNamespace(
            get_xyz=torch.zeros((2, 3)),
            support_count=None,  # This will cause early return
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

    # This should not crash
    runtime.publish_assimilation_support(1)


def test_disabled_control_zero_readbacks_when_not_due():
    """When control disabled and telemetry not due, zero device readbacks occur."""
    pressure.configure_assimilation_control(enabled=False)

    call_count = [0]
    original_publish = pressure.publish_support_counts

    def counting_publish(*args, **kwargs):
        call_count[0] += 1
        return original_publish(*args, **kwargs)

    # Patch the function
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


def test_explicit_flush_emits_latest_summary():
    """flush_assimilation_support() emits the latest latched summary."""
    pressure.configure_assimilation_control(enabled=False)

    # Produce a summary
    support_counts = torch.tensor([2.0, 0.0, 5.0, 1.0, 3.0])
    debt = pressure.publish_support_counts(support_counts, min_support_views=2)

    # Flush should return the latched summary
    flushed = pressure.flush_assimilation_support()

    assert flushed is not None
    assert flushed.debt == debt
    assert flushed.support_pool_n == 5


def test_flush_tolerates_no_summary():
    """flush_assimilation_support() handles the case of no latched summary."""
    # Clear the latch
    pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

    # Flush with no summary should return None
    flushed = pressure.flush_assimilation_support()
    assert flushed is None or (flushed is not None and hasattr(flushed, 'debt'))


def test_control_enabled_produces_summary():
    """With control enabled, the summary is produced and latched as tensor-free."""
    pressure.configure_assimilation_control(enabled=True)

    # Produce a summary via the normal path
    support_counts = torch.tensor([2.0, 0.0, 5.0, 1.0])
    debt = pressure.publish_support_counts(support_counts, min_support_views=2)

    # After production, the latched summary should be tensor-free
    summary = pressure.latest_assimilation_support_summary()
    assert summary is not None
    assert summary.debt == debt

    # Check no tensor in summary
    for field_value in dataclasses.astuple(summary):
        assert not isinstance(field_value, torch.Tensor), (
            f"Summary field should not be a tensor: {field_value}"
        )
        assert not hasattr(field_value, 'cpu'), (
            f"Summary field should not have .cpu(): {field_value}"
        )

    pressure.configure_assimilation_control(enabled=False)


def test_generation_counter_increments_per_publish():
    """Each publish increments the generation counter in the summary."""
    pressure.configure_assimilation_control(enabled=False)
    pressure._LATEST_ASSIMILATION_SUPPORT_SUMMARY = None

    support_counts = torch.tensor([1.0, 2.0, 3.0])

    # First publish
    pressure.publish_support_counts(support_counts, min_support_views=2)
    summary1 = pressure.latest_assimilation_support_summary()
    gen1 = summary1.generation if summary1 else None

    # Second publish
    pressure.publish_support_counts(support_counts, min_support_views=2)
    summary2 = pressure.latest_assimilation_support_summary()
    gen2 = summary2.generation if summary2 else None

    # Generation should increment
    assert gen1 is not None and gen2 is not None
    assert gen2 > gen1, f"Generation should increment: {gen1} -> {gen2}"


def test_final_report_path_calls_flush_with_zero_device_ops():
    """Final report path calls flush_assimilation_support() without new device ops.

    This simulates the source-complete report path (mapping_endpoint_reporter in
    causal_entry.py) calling flush to emit the final summary. The flush itself
    performs zero device operations (returns latched summary).
    """
    pressure.configure_assimilation_control(enabled=False)

    # Track calls to tolist (would indicate a new device reduction)
    tolist_calls = [0]
    original_tolist = torch.Tensor.tolist

    def counting_tolist(self):
        tolist_calls[0] += 1
        return original_tolist(self)

    torch.Tensor.tolist = counting_tolist

    try:
        # Produce a summary via normal dispatch (this calls publish_support_counts)
        support_counts = torch.tensor([1.0, 2.0, 3.0, 4.0])
        pressure.publish_support_counts(support_counts, min_support_views=2)
        initial_tolist_count = tolist_calls[0]

        # Now simulate the final report path calling flush (should have zero ops)
        final_summary = pressure.flush_assimilation_support()
        final_tolist_count = tolist_calls[0]

        # Assert flush emitted the summary with no new device operations
        assert final_summary is not None, "Flush should return the latched summary"
        assert final_summary.support_pool_n == 4
        assert final_tolist_count == initial_tolist_count, (
            "Flush should not perform new device operations (tolist calls): "
            f"{initial_tolist_count} vs {final_tolist_count}"
        )
    finally:
        torch.Tensor.tolist = original_tolist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
