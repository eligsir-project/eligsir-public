"""Compatibility surface for the split RGB-D integration implementation.

The coordinator acquires the canonical map backend before planning. When the
run requests the fully resident GPU path and every enabled feature has tensor
parity, dispatch goes to the one-H2D device planner. Legacy planning is an
explicit pre-transfer compatibility fallback, never a mid-pipeline D2H escape.
"""

from eligsir.mapping.integration.frame_preparation import (
    _bootstrap_mask,
    _bootstrap_motion_stats,
    _capture_bootstrap_positions,
    _get_sensor_depth,
    _log_bootstrap_motion,
    _render_streaming_snapshot,
    _timed,
    _update_provisional_support,
    _voxel_downsample_indices,
    _write_bootstrap_motion_ply,
)
from eligsir.mapping.integration.integration_planner import (
    insert_gaussians_from_frame as _planner_insert_gaussians_from_frame,
)
from eligsir.mapping.integration.checkpoint_io import (
    _HOLDOUT_GT_SAVED_IDS,
    _export_ply_snapshot,
    _flush_dense_trajectory,
    _save_cameras_json,
    _save_holdout_gt,
    _save_streaming_checkpoint,
    _save_test_cameras_json,
)
from eligsir.mapping.integration.rolling_seed_application import _run_rolling_seed
from eligsir.mapping.integration.submap_stitch_application import (
    _maybe_create_pose_graph,
    _run_submap_stitch,
)
from eligsir.mapping.integration.loss_diagnostics import (
    _add_guarded,
    _admission_quality_weight_for_stats,
    _nan_term_report,
    _pose_provenance_report_dict,
    _sanitize_gaussian_params,
    _sanitize_gaussian_params_branch_free,
    _sanitize_gaussian_params_diagnostics_only,
    _sanitize_gaussian_params_sampled,
)


def _canonical_backend_for(gaussians):
    if hasattr(gaussians, "params"):
        from eligsir.map.adapters.gsplat import GsplatMapBackend
        return GsplatMapBackend(gaussians)
    from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMapBackend
    return LegacyGraphdecoMapBackend(gaussians)


def insert_gaussians_from_frame(
    gaussians,
    frame,
    args,
    streaming_scene,
    prev_frame=None,
    prev_depth_meters=None,
    render_alpha=None,
    render_depth=None,
    current_frame_idx=0,
    iteration=0,
    _collect_only=False,
    map_backend=None,
    insertion_plan=None,
    insertion_policy=None,
):
    """Run exactly one insertion producer under the canonical map owner."""
    backend = map_backend if map_backend is not None else _canonical_backend_for(gaussians)
    # Historical replay compatibility owns this module.  The canonical live
    # runtime imports the typed device producer directly and never dispatches
    # through this args-driven fallback surface.
    return _planner_insert_gaussians_from_frame(
        gaussians,
        frame,
        args,
        streaming_scene,
        prev_frame=prev_frame,
        prev_depth_meters=prev_depth_meters,
        render_alpha=render_alpha,
        render_depth=render_depth,
        current_frame_idx=current_frame_idx,
        iteration=iteration,
        _collect_only=_collect_only,
        map_backend=backend,
    )


__all__ = [
    "_HOLDOUT_GT_SAVED_IDS",
    "_add_guarded",
    "_admission_quality_weight_for_stats",
    "_bootstrap_mask",
    "_bootstrap_motion_stats",
    "_capture_bootstrap_positions",
    "_export_ply_snapshot",
    "_flush_dense_trajectory",
    "_get_sensor_depth",
    "_log_bootstrap_motion",
    "_maybe_create_pose_graph",
    "_nan_term_report",
    "_pose_provenance_report_dict",
    "_render_streaming_snapshot",
    "_run_rolling_seed",
    "_run_submap_stitch",
    "_sanitize_gaussian_params",
    "_sanitize_gaussian_params_branch_free",
    "_sanitize_gaussian_params_diagnostics_only",
    "_sanitize_gaussian_params_sampled",
    "_save_cameras_json",
    "_save_holdout_gt",
    "_save_streaming_checkpoint",
    "_save_test_cameras_json",
    "_timed",
    "_update_provisional_support",
    "_voxel_downsample_indices",
    "_write_bootstrap_motion_ply",
    "insert_gaussians_from_frame",
]
