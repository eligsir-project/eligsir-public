"""Issue #703: Supervision-work accumulation on the causal path.

A mutable accumulator for execution-shape counters from OptimizationStepResult
objects produced by the mapping runtime's optimizer transaction loop. Mirrors
the now-orphaned train_streaming.py's _per_source_supervision_counts pattern.

Consumed by causal_evaluation.py's report writer and causal_entry.py's summary
JSON assembly.
"""

from __future__ import annotations

from typing import Any

from eligsir.utils.supervision_work import supervision_work_report_block


class SupervisionWorkClocks:
    """Mutable accumulator for supervision work counters across a mapping run.

    Observes OptimizationStepResult objects one per executed mapping tick
    transaction (both B>1 batch and B=1 sequential paths). Aggregates the
    execution-shape counters (loss_objective_call_count, backward_call_count,
    renderer_camera_batch_sizes, loss_views_per_call) into host-int sums,
    counts, and maxima, with optional source exposure distribution tracking.

    Torch-free; all fields are Python int/dict.
    """

    def __init__(self) -> None:
        """Initialize all accumulator fields to zero."""
        self.optimizer_step_count: int = 0
        self.supervision_view_count: int = 0
        self.rasterizer_call_count: int = 0
        self.supervised_pixel_count: int = 0
        self.loss_objective_call_count: int = 0
        self.backward_call_count: int = 0
        self.renderer_camera_batch_size_sum: int = 0
        self.renderer_camera_batch_size_count: int = 0
        self.renderer_camera_batch_size_max: int = 0
        self.loss_views_per_call_sum: int = 0
        self.loss_views_per_call_count: int = 0
        self.loss_views_per_call_max: int = 0
        # Per-source exposure tracking (optional, for exposure distribution).
        # Only meaningful if the caller identifies sources via
        # successful_supervision_source_indices; otherwise remains empty.
        self.per_source_counts: dict[Any, int] = {}

    def observe(self, step_result: Any) -> None:
        """Accumulate execution-shape counters from one OptimizationStepResult.

        Called once per successful execute_fixed_topology_transaction() call on
        the causal mapping path (both B>1 and B=1 branches). Tolerates missing
        or None fields; only accumulates fields that are present and nonzero.

        Args:
            step_result: An OptimizationStepResult object (or similar with
                optional execution-shape fields). No torch tensors are accessed.
        """
        if step_result is None:
            return

        # Core supervision counters.
        self.optimizer_step_count += int(
            getattr(step_result, "optimizer_step_count", 0)
        )
        self.supervision_view_count += int(
            getattr(step_result, "supervision_view_count", 0)
        )
        self.rasterizer_call_count += int(
            getattr(step_result, "rasterizer_call_count", 0)
        )
        self.supervised_pixel_count += int(
            getattr(step_result, "supervised_pixel_count", 0)
        )

        # Issue #703: execution-shape counters.
        self.loss_objective_call_count += int(
            getattr(step_result, "loss_objective_call_count", 0)
        )
        self.backward_call_count += int(
            getattr(step_result, "backward_call_count", 0)
        )

        # Renderer camera batch sizes: tuple of ints, one per rasterizer call.
        renderer_batch_sizes = getattr(
            step_result, "renderer_camera_batch_sizes", ()
        )
        if renderer_batch_sizes:
            sizes = tuple(int(s) for s in renderer_batch_sizes)
            self.renderer_camera_batch_size_sum += sum(sizes)
            self.renderer_camera_batch_size_count += len(sizes)
            if sizes:
                self.renderer_camera_batch_size_max = max(
                    self.renderer_camera_batch_size_max, max(sizes)
                )

        # Loss views per call: tuple of ints, one per assemble_loss() call.
        views_per_call = getattr(step_result, "loss_views_per_call", ())
        if views_per_call:
            views = tuple(int(v) for v in views_per_call)
            self.loss_views_per_call_sum += sum(views)
            self.loss_views_per_call_count += len(views)
            if views:
                self.loss_views_per_call_max = max(
                    self.loss_views_per_call_max, max(views)
                )

        # Optional: source exposure tracking from successful_supervision_source_indices.
        # Each entry in the tuple is one exposure (duplicates are intentional).
        successful_indices = getattr(
            step_result, "successful_supervision_source_indices", ()
        )
        for source_index in successful_indices:
            source_id = int(source_index)
            self.per_source_counts[source_id] = (
                self.per_source_counts.get(source_id, 0) + 1
            )

    def report_block(self, wall_time_s: float) -> dict[str, Any]:
        """Assemble report.json["supervision_work"] from accumulated clocks.

        Returns the output of supervision_work_report_block() with the
        aggregated counts. Per the module docstring in supervision_work.py,
        returns an empty dict if no optimizer step was ever observed, so the
        report writer omits the key entirely rather than serialize meaningless
        zeros.

        Args:
            wall_time_s: Wall-clock elapsed time in seconds for the enclosing
                mapping run. Used to compute supervision_view_rate_hz. Should
                match the observer's _elapsed_s at the report point.

        Returns:
            A dict with keys like "optimization_step_count",
            "supervision_view_count", "rasterizer_calls_per_step",
            "supervision_views_per_step", "renderer_camera_batch_size_mean",
            "loss_views_per_call_mean", etc. Empty dict if optimizer_step_count==0.
        """
        return supervision_work_report_block(
            optimization_step_count=self.optimizer_step_count,
            supervision_view_count=self.supervision_view_count,
            rasterizer_call_count=self.rasterizer_call_count,
            supervised_pixel_count=self.supervised_pixel_count,
            per_camera_supervision_counts=None,  # Per advisor: keep field empty
            wall_time_s=float(wall_time_s),
            loss_objective_call_count=self.loss_objective_call_count,
            backward_call_count=self.backward_call_count,
            renderer_camera_batch_size_sum=self.renderer_camera_batch_size_sum,
            renderer_camera_batch_size_max=self.renderer_camera_batch_size_max,
            loss_views_per_call_sum=self.loss_views_per_call_sum,
            loss_views_per_call_max=self.loss_views_per_call_max,
        )
