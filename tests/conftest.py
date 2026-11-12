"""Environment-aware full-suite discovery and test fixtures.

The required PR gate uses ``scripts/ci_test_suite.py`` and ``tests/pr_core.txt``:
CPU-Torch contracts plus changed/directly related tests. Full CPU-safe discovery
runs nightly/manual; ``pytest tests/`` still discovers the whole tree. See
``tests/README.md`` for selection, qualification, and omission reporting.

Every dependency guard below is conditional. A torch-free diagnostic image
cannot collect some modules; the CPU-Torch/training images must not inherit
those exclusions when their dependencies are installed. Individual optional
hardware/dependency skips retain explicit reasons. Never add an unconditional
ignore to make a reduced gate green.

Historical discovery failures and their fixes remain documented in
``docs/architecture/2026-08-13-issue423-ci-test-discovery.md``. The old CI-wide
counts and instruction to run every test on every PR are superseded by the
current tiered policy; the conditional discovery mechanism is unchanged.
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Whole files that cannot even be COLLECTED in this image: the import chain
# pulls in torch (directly, or transitively via train.py / train_streaming.py
# / scene.gaussian_model.py / utils.comparison_report.py / gaussian_renderer
# -- none of which this deliberately CPU-only image installs, see
# Dockerfile.ci-cpu's "keep this image intentionally light" comment) before
# any individual test function even runs. This is an ImportError/
# ModuleNotFoundError at collection time, not a test failure -- pytest has
# no way to report a per-test reason for these, so `collect_ignore` (a
# conftest.py convention pytest reads automatically) is the right tool, not
# a skip marker.
#
# **Updated 2026-08-14 by #478: this is no longer a hand-maintained list.**
# It used to be exactly that -- a literal ~150-entry list someone had to
# remember to append to whenever a new torch-importing test file was added.
# PR #476 added tests/test_seed_cohort_offline_replay.py without doing so;
# the omission wasn't a per-test skip, it was a COLLECTION-TIME error that
# aborted the entire run (`Interrupted: 1 error during collection`), and the
# CPU suite silently dropped from ~1385 tests to 22 until PR #471 added the
# missing entry as a side effect of unrelated work. The training image has
# torch, so this defect class is completely invisible there -- training-image
# green is not evidence of CPU-image green.
#
# `_torch_collect_scan.discover_torch_needing_test_files()` (issue #478's
# option 2, chosen over option 1's "add a guard test that checks the hand
# list is complete" -- see that PR's description for the tradeoff) walks
# every `tests/test_*.py` file's import graph via AST, recursively, through
# this repo's own modules -- catching not just a direct `import torch` but a
# test file that only imports a repo-local helper which itself needs torch,
# transitively, plus this repo's `importlib.util.spec_from_file_location`
# path-based module-loading idiom. It is guard-aware (a `try/except
# ImportError`, `pytest.importorskip("torch")`, `if TYPE_CHECKING:`, or this
# repo's own `if importlib.util.find_spec("torch") is not None:` idiom does
# NOT get flagged, since none of those can actually break collection) so it
# does not re-introduce #462's opposite bug (silently over-excluding real,
# collectible coverage). See `tests/_torch_collect_scan.py`'s module
# docstring for the full mechanism and its honest, documented gaps, and
# `tests/test_torch_collect_scan.py` for the regression/demonstration suite,
# including a reproduction of this exact historical omission.
#
# Verified 2026-08-14: the derived set is IDENTICAL (0 added, 0 removed) to
# the ~150-entry hand list this replaced, on this tree at HEAD -- i.e. this
# change is behavior-neutral for every test file that exists today, and
# closes the drift class for every one added from now on. The scan itself
# only ever runs when torch is actually absent (see `if not _TORCH_AVAILABLE`
# below) -- zero added cost in the training image, where it's moot anyway
# (nothing in `_NEEDS_TORCH` would be used there regardless of its content).
# Measured wall time of the scan alone in this (torch-having) dev
# environment: ~1.1s for all ~300 test files, cold (no memoization reuse
# across files) -- see that PR's description for the CPU-image number.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _torch_collect_scan  # noqa: E402

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# The scan walks every tests/test_*.py file's import graph -- only worth
# paying for in the image where its result actually matters (torch absent).
# In the training image (torch present) `_NEEDS_TORCH` is never consulted
# below regardless of its content, so skip the scan there entirely: zero
# added conftest-load cost in that image, addressing the "slower at conftest
# load" tradeoff issue #478 named for this option.
_NEEDS_TORCH = (
    _torch_collect_scan.discover_torch_needing_test_files() if not _TORCH_AVAILABLE else []
)

# Needs tensorboard's event-file reader (scripts/seed_fade_summary.py) --
# a moderately heavy pure-CPU dependency this lean image does not install
# for the sake of this single consumer.
_NEEDS_TENSORBOARD = [
    "test_seed_fade_summary.py",
]

# ---------------------------------------------------------------------------
# Accountable, ENVIRONMENT-INDEPENDENT ignores (#462). Everything above is
# ignored only because a dependency is absent; these three are ignored because
# they genuinely fail on `main` in the full training image today, and each one
# is owned by an in-flight fix landing separately. They stay here ONLY until
# that fix merges -- an entry with no live issue behind it is a bug in this
# file, not a fact about the tests. Measured 2026-08-13 on ca1f45e in
# the project training image, GPU attached.
# ---------------------------------------------------------------------------
# All three original entries (#463/#464/#455-owned) were RETIRED 2026-08-13:
# the fixes they deferred to have all landed on main (#453/#460 completed the
# bare-`args` fixtures, #468 fixed the hardcoded resolution-cache device), and
# all three files were re-run directly against this branch -- 39/39 pass, 0
# failures. Keeping them ignored would have silently excluded 39 now-green
# tests, which is the exact class of invisible exclusion #462 exists to end.
#
# Keep this dict as the documented home for "collectible in this image but
# known-red, owned by a specific in-flight fix" entries. Every entry MUST name
# the issue/PR that owns it and be removed when that lands -- an entry with no
# live owner is indistinguishable from the silent exclusion #462 fixed.
_TRAINING_IMAGE_KNOWN_FAILING: dict[str, str] = {}

_TENSORBOARD_AVAILABLE = importlib.util.find_spec("tensorboard") is not None

collect_ignore: list[str] = list(_TRAINING_IMAGE_KNOWN_FAILING)
if not _TORCH_AVAILABLE:
    collect_ignore += [f for f in _NEEDS_TORCH if f not in _TRAINING_IMAGE_KNOWN_FAILING]
if not _TENSORBOARD_AVAILABLE:
    collect_ignore += _NEEDS_TENSORBOARD


# ---------------------------------------------------------------------------
# Individual tests inside otherwise CPU-safe files: the file collects and
# runs real tests today (see each file's own passing tests), only these
# specific functions need a dependency this image doesn't install. Skipping
# (not deselecting) so they stay VISIBLE in CI output with a reason, per
# #423's "make the gap visible" goal -- a deselect list would just make them
# disappear again, repeating the original bug's silence.
# ---------------------------------------------------------------------------
_TORCH_REASON = "requires torch (not installed in the CPU smoke image)"
_TORCH_VIA_BROAD_EXCEPT_REASON = (
    "requires torch, but indirectly: a broad `except Exception:` several "
    "frames down swallows the ModuleNotFoundError and this test's own "
    "assertion fails instead -- verified by isolating this test with "
    "--tb=long inside the ci-cpu image and reading the full exception "
    "chain, not just pytest's short summary"
)
_CIRCULAR_IMPORT_REASON = (
    "requires torch: this test's own import chain (via "
    "eligsir.io.sources.registry / utils.rgbd_sequence_writer -> "
    "scene.readers.* -> scene.gaussian_model) needs torch; the surfaced "
    "'circular import' ImportError is a secondary symptom of Python "
    "unwinding a failed nested import, not the real cause -- verified by "
    "isolating this test with --tb=long and reading the full traceback, "
    "whose deepest frame is `scene/gaussian_model.py:12: import torch`"
)

# RETIRED 2026-08-14 by #470 item B.1. This used to be an unconditional skip
# for tests/test_insertion_hot_path_backend_parity.py::
# test_flush_insertion_batch_backend_parity -- the same bare-`args`-namespace
# class as #455/#463, but in a file #455/#463's own fixes never touched
# (confirmed: PR #465, the "TEST half of #463", was closed unmerged as a
# no-op superset once #460 landed the real fix, and #460's scope was
# test_replay_buffer_reachability.py only -- this file was untouched by
# either). insertion_committer.py:135's flush_insertion_batch reads
# args.streaming_max_new_gaussians_per_frame with NO getattr fallback
# (deliberate, see that function's own "#363 residue" docstring comment --
# the production bare read is correct, same audit conclusion #455/#460
# reached for their own sites), but the test's fixture was a totally empty
# `type("Args", (), {})()`. Fixed by declaring the attribute on the test's
# own _Args fixture at its real parser default (arguments/__init__.py:853,
# 1000) -- no production code changed. This test is GPU-gated
# (`pytestmark = pytest.mark.skipif(not torch.cuda.is_available())`), so the
# fix itself could only be verified by source inspection + a CPU-side
# AttributeError repro (not a real pass) in this session -- a real GPU run
# confirming it passes is still outstanding, see #470's tracking comment.
_SKIP_REASONS: dict[str, str] = {
    "tests/test_causal_evaluation.py::test_causal_holdout_camera_keeps_stable_source_identity_and_lazy_original_rgb": _TORCH_REASON,
    "tests/test_causal_evaluation.py::test_primary_and_frozen_holdout_curves_are_plotted_together": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_evaluation.py::test_spatial_replay_focus_plot_tracks_quantiles_and_physical_cells": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_decision_trace_finalizer_writes_sampling_ledger_and_heatmap": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_uses_exact_mutation_lane_events": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_issue1195_production_gpu_replay.py::test_factory_application_zero_frame_replay_advances_on_available_gpu": _TORCH_REASON,
    "tests/test_partition_replay_focus.py::test_training_spatial_sample_uses_sensor_depth_cells_and_exact_rgb_error": _TORCH_REASON,
    # keyframes.py's render gate does `import torch` inside a try/except
    # Exception, so without torch installed render_fn silently never runs.
    "tests/test_keyframe_admission.py::TestCheapGate::test_motion_pregate_still_renders_moved_frame": _TORCH_VIA_BROAD_EXCEPT_REASON,
    "tests/test_keyframe_admission.py::TestCheapGate::test_motion_pregate_admits_first_frame_with_no_baseline": _TORCH_VIA_BROAD_EXCEPT_REASON,
    "tests/test_keyframe_admission.py::TestCheapGate::test_motion_pregate_accumulates_motion_across_rejections": _TORCH_VIA_BROAD_EXCEPT_REASON,
    "tests/test_keyframe_admission.py::TestAdmissionQualityWeightForStats::test_none_quality_with_quality_enabled_does_not_crash": _TORCH_REASON + " (imports train_streaming)",
    "tests/test_keyframe_admission.py::TestAdmissionQualityWeightForStats::test_real_quality_with_quality_enabled_passes_through": _TORCH_REASON + " (imports train_streaming)",
    "tests/test_keyframe_admission.py::TestAdmissionQualityWeightForStats::test_quality_disabled_returns_none_regardless_of_decision": _TORCH_REASON + " (imports train_streaming)",
    "tests/test_local_regions.py::TestCovisibilitySnapshot::test_snapshot_maps_cameras_to_the_cells_they_saw": _TORCH_REASON,
    "tests/test_local_regions.py::TestCovisibilitySnapshot::test_a_resolution_change_does_not_duplicate_a_camera": _TORCH_REASON,
    "tests/test_local_regions.py::TestCovisibilitySnapshot::test_snapshot_is_read_only_and_idempotent": _TORCH_REASON,
    "tests/test_local_regions.py::TestReportScriptExitPaths::test_missing_level_a_history_exits_2_with_a_not_measured_panel": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_local_regions.py::TestReportScriptExitPaths::test_a_tick_with_no_evidence_exits_3_and_says_so_on_disk": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_loop_closure_wiring.py::test_apply_submap_correction_identity_leaves_means_unchanged": _TORCH_REASON + " (loop_closure.submap_correction, unlike loop_closure.pose_graph, needs it)",
    "tests/test_mapping_phase_boundary.py::test_report_omits_mapping_phase_fields_by_default": _TORCH_REASON,
    "tests/test_mapping_phase_boundary.py::test_report_carries_last_ingest_iter_and_source": _TORCH_REASON,
    "tests/test_mapping_phase_boundary.py::test_true_final_iteration_overrides_sentinel_without_touching_iteration_field": _TORCH_REASON,
    # Container-specific, not a torch gap: this CI image runs as root with no
    # /scratch mount at all, so resolve_staging_root's nearest-existing-
    # ancestor walk for "/scratch/<user>/eligsir" lands on "/" itself
    # (root can always write to "/" in a container) instead of failing the
    # way it would on the real deployment host, where /scratch is always a
    # real, existing mount. Not a product bug: this exact edge case (no
    # /scratch tree anywhere on the machine) does not occur outside this
    # lean CI image.
    "tests/test_output_staging.py::TestResolveStagingRoot::test_falls_through_to_tmpdir_when_scratch_unwritable": "container-specific: runs as root with no /scratch mount, so the ancestor-walk in resolve_staging_root finds '/' (writable by root) before ever reaching a real /scratch check -- does not reproduce outside this lean CI image",
    "tests/test_realityscan_frame_source.py::test_registry_dispatches_to_realityscan_frame_source": _CIRCULAR_IMPORT_REASON,
    "tests/test_realityscan_frame_source.py::test_sfm_without_xmp_does_not_dispatch_to_realityscan": _CIRCULAR_IMPORT_REASON,
    "tests/test_realsense_alignment.py::test_pyrealsense2_importable": "requires pyrealsense2 (RealSense hardware SDK, not installed in the CPU smoke image)",
    "tests/test_rgbd_capture_scripts_pointcloud_flags.py::test_capture_ros2_script_declares_flags_with_current_effective_defaults": _CIRCULAR_IMPORT_REASON,
    "tests/test_rgbd_capture_scripts_pointcloud_flags.py::test_extract_rosbag_script_declares_flags_with_current_effective_defaults": _CIRCULAR_IMPORT_REASON,
    "tests/test_rgbd_capture_scripts_pointcloud_flags.py::test_capture_ros2_script_flag_overrides_reach_args": _CIRCULAR_IMPORT_REASON,
    "tests/test_rgbd_capture_scripts_pointcloud_flags.py::test_extract_rosbag_script_flag_overrides_reach_args": _CIRCULAR_IMPORT_REASON,
    "tests/test_scene_geometry_metrics.py::test_compute_voxel_occupancy_known_bbox": "requires open3d (not installed in the CPU smoke image)",
    "tests/test_scene_geometry_metrics.py::test_compute_voxel_occupancy_empty": "requires open3d (not installed in the CPU smoke image)",
    "tests/test_scene_geometry_metrics.py::test_compare_scene_vs_trajectory_no_plume": "requires scipy (not installed in the CPU smoke image)",
    "tests/test_scene_geometry_metrics.py::test_compare_scene_vs_trajectory_detects_plume": "requires scipy (not installed in the CPU smoke image)",
    "tests/test_scene_geometry_metrics.py::test_compare_scene_vs_trajectory_empty_inputs": "requires scipy (not installed in the CPU smoke image)",
    "tests/test_scene_geometry_metrics.py::test_compare_scene_vs_trajectory_ignores_dead_outlier": "requires scipy (not installed in the CPU smoke image)",
    "tests/test_seed_fade_resume.py::test_state_dict_round_trips_through_torch_save": _TORCH_REASON,
    "tests/test_training_curves.py::test_writes_png_when_events_exist": _TORCH_REASON,
    "tests/test_training_curves.py::test_seed_fade_ever_active_detects_scalar_present": _TORCH_REASON,
    "tests/test_training_curves.py::test_seed_fade_ever_active_false_when_never_set": _TORCH_REASON,
    "tests/test_training_curves.py::test_seed_fade_ever_active_false_when_tag_absent": _TORCH_REASON,
    "tests/test_training_curves.py::test_plot_training_curves_does_not_crash_with_seed_fade_scalar": _TORCH_REASON,
    "tests/test_trajectory_eval_test_cams.py::test_run_trajectory_eval_without_test_cams_is_unchanged": _TORCH_REASON,
    "tests/test_trajectory_eval_test_cams.py::test_run_trajectory_eval_writes_test_trajectory_and_index": _TORCH_REASON,
    "tests/test_trajectory_eval_test_cams.py::test_run_trajectory_eval_test_cams_empty_list_writes_nothing": _TORCH_REASON,
    "tests/test_trajectory_eval_test_cams.py::test_run_trajectory_eval_surfaces_implausible_flags_in_metrics_json": _TORCH_REASON,
    "tests/test_w3_35_mcmc_kernel_audit.py::test_inject_noise_to_position_is_still_pure_pytorch": _TORCH_REASON,
    # RE-CLASSIFIED 2026-08-13, not removed. The #470 entry that used to sit
    # here was an UNCONDITIONAL skip (deliberately worded so `_required_module`
    # would not match it) covering a stale source-inspection assertion. PR #475
    # fixed that assertion for real -- the "third `birth_iter=0` occurrence"
    # was prose in a comment, not an insertion site -- so the #470 reason is
    # dead and the test passes wherever it can run.
    #
    # But it cannot run without torch: it imports
    # eligsir/mapping/integration/rolling_seed_application.py, which does a
    # module-scope `import torch`. Deleting the entry outright therefore turned
    # it red in the CPU smoke image (caught by CI, not by the training image,
    # which cannot see this class of defect at all). It belongs in the same
    # torch-gated bucket its own sibling below already uses.
    "tests/test_always_on_insertion_sites_birth_iter.py::test_rolling_seed_insertions_state_birth_iter_explicitly": _TORCH_REASON,
    "tests/test_always_on_insertion_sites_birth_iter.py::test_submap_stitch_insertion_states_birth_iter_explicitly": _TORCH_REASON,
    # 2026-08-31 CI-baseline audit: same class as the causal_evaluation/
    # causal_replay_driver entries above -- the torch-needing import
    # (`eligsir.mapping.integration.integration_planner`, module-scope
    # `import torch`) happens inside the test FUNCTION BODY via a local
    # `from ... import ...`, not at module scope, so `_torch_collect_scan`'s
    # static per-file AST walk (module-scope imports only) cannot see it and
    # the file collects fine; the test then hard-fails at runtime with a bare
    # ModuleNotFoundError instead of skipping.
    "tests/test_issue1279_depth_sparsity_schedule.py::test_planner_resolves_local_depth_stride_via_schedule_module": _TORCH_REASON,
    "tests/test_issue1279_depth_sparsity_schedule.py::test_planner_never_mutates_args_streaming_depth_stride": _TORCH_REASON,
    # Same class, via eligsir.mapping.integration.device_integration_planner.
    "tests/test_issue1279_device_native_depth_sparsity_schedule.py::test_device_native_planner_resolves_local_stride_via_schedule_module": _TORCH_REASON,
    "tests/test_issue1279_device_native_depth_sparsity_schedule.py::test_device_native_planner_never_mutates_plan_or_policy": _TORCH_REASON,
    "tests/test_issue1279_device_native_depth_sparsity_schedule.py::test_device_native_planner_passes_current_frame_idx_through": _TORCH_REASON,
    # Same class again, via utils.comparison_report's module-scope `import torch`
    # (line 30), reached only once _write_report's local `from
    # utils.comparison_report import write_post_training_report` executes.
    "tests/test_source_ingress_reporting.py::test_write_post_training_report_lands_the_block_and_warns": _TORCH_REASON,
    "tests/test_source_ingress_reporting.py::test_write_post_training_report_is_silent_and_absent_without_drops": _TORCH_REASON,
    "tests/test_coverage_and_termination.py::test_trajectory_coverage_wiring_contract_matches_write_post_training_report_usage": _TORCH_REASON,
    "tests/test_coverage_and_termination.py::test_compute_residual_weights_favors_high_residual_camera": _TORCH_REASON,
    "tests/test_coverage_and_termination.py::test_compute_residual_weights_scores_finite_renderer_overshoot": _TORCH_REASON,
    "tests/test_coverage_and_termination.py::test_compute_residual_weights_fails_open_on_render_error": _TORCH_REASON,
    "tests/test_coverage_and_termination.py::test_compute_residual_weights_empty_pool_returns_none": _TORCH_REASON,
    "tests/test_coverage_and_termination.py::test_sample_training_camera_accepts_render_kwargs_without_error": _TORCH_REASON,
    "tests/test_gsplat_default_strategy_reset_every.py::test_reset_every_defaults_to_opacity_reset_interval": _TORCH_REASON,
    "tests/test_gsplat_default_strategy_reset_every.py::test_reset_every_respects_explicit_opacity_reset_interval_override": _TORCH_REASON,
    "tests/test_gsplat_default_strategy_reset_every.py::test_reset_every_falls_back_to_3000_when_arg_missing": _TORCH_REASON,
    "tests/test_input_quality.py::test_pca_residual_tracks_sensor_noise": _TORCH_REASON,
    "tests/test_issue302_ingestion_mode_provenance.py::test_write_post_training_report_accepts_ingestion_kwarg": _TORCH_REASON,
    # Pre-existing smoke-check breakage found 2026-08-29 (blocking main's
    # required `smoke` check): these tests hard-fail with a bare
    # ModuleNotFoundError instead of skipping, because the missing import
    # happens deep inside a call (`utils/psnr_history.py`'s module-scope
    # `import torch`, or `eligsir/runtime/causal_evaluation.py::
    # mapping_holdout_curve_metadata`'s call into it) that `_torch_collect_scan`
    # cannot see: it walks each test file's own static import graph, not the
    # runtime call graph reached only once the test body executes.
    "tests/test_causal_evaluation.py::test_mapping_holdout_curve_large_jump_endpoint_and_refinement_are_phase_aware": _TORCH_REASON + " (utils.psnr_history)",
    "tests/test_causal_evaluation.py::test_evaluation_tick_joins_post_work_clocks_to_immutable_control_keys": _TORCH_REASON + " (utils.psnr_history)",
    # Same class, but the runtime call chain hits matplotlib instead of torch:
    # `eligsir/runtime/causal_evaluation.py:428`'s plotting helper does a
    # deferred `import matplotlib` inside the function body.
    "tests/test_causal_evaluation.py::test_paired_psnr_work_plot_uses_active_time_and_supervised_views": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_evaluation.py::test_paired_psnr_work_plot_suppresses_invalid_wall_time_panel": "requires matplotlib (not installed in the CPU smoke image)",
    # Same class again: `eligsir/runtime/replay_sampling_telemetry.py:61`'s
    # `_write_heatmap` does a deferred `import matplotlib`, reached only once
    # `CausalReplayDriver.finalize()`/`write_replay_sampling_telemetry()` runs.
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_v3_separates_catalog_shortlist_and_deferred_coverage": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_v3_withholds_eligibility_metrics_without_membership": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_v3_catalog_sequence_contradiction_fails_closed": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_v3_validates_multi_disposition_post_catalog_population": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_v3_contradictory_post_catalog_population_fails_closed": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_v3_ambiguous_legacy_catalog_position_fails_closed": "requires matplotlib (not installed in the CPU smoke image)",
    "tests/test_causal_replay_driver.py::test_sampling_telemetry_v3_zero_executed_ticks_withholds_eligibility_metrics": "requires matplotlib (not installed in the CPU smoke image)",
    # `eligsir/mapping/scheduler/mcmc_cadence.py` is reached via a dynamic
    # `importlib.import_module(...)` inside each test helper (not a static
    # `import`), so `_torch_collect_scan`'s AST walk cannot see that it
    # transitively needs torch (`utils/strategies/mcmc/{energy,proposal,
    # baseline}.py` each do a module-scope `import torch`).
    "tests/test_mcmc_semantic_cadence.py::test_energy_strategy_explicit_due_flag_never_calls_legacy_due": _TORCH_REASON,
    "tests/test_mcmc_semantic_cadence.py::test_temperature_schedule_is_explicitly_optimizer_update_driven": _TORCH_REASON,
    "tests/test_mcmc_semantic_cadence.py::test_legacy_noise_stop_iteration_cannot_actuate_canonical_strategy": _TORCH_REASON,
    "tests/test_mcmc_semantic_cadence.py::test_live_phase_position_noise_gate_overrides_update_horizon": _TORCH_REASON,
    "tests/test_mcmc_semantic_cadence.py::test_live_phase_position_noise_gate_requires_exact_bool": _TORCH_REASON,
    "tests/test_mcmc_semantic_cadence.py::test_factory_threads_semantic_cadence_to_direct_and_live_strategies": _TORCH_REASON,
    "tests/test_mcmc_semantic_cadence.py::test_factory_rejects_non_bool_semantic_cadence": _TORCH_REASON,
    # `utils/trajectory_quality.py:132` does a deferred `import matplotlib`.
    "tests/test_trajectory_quality_profile.py::test_quality_profile_uses_metric_camera_path_and_marks_segments": "requires matplotlib (not installed in the CPU smoke image)",
    # benchmarks/true_camera_scaling.py's module itself is torch-free (that's
    # why it and its test file could be added to Dockerfile.ci-cpu's/
    # Dockerfile.ci-sim's explicit COPY manifests without dragging torch into
    # the lean smoke image), but these three tests specifically exercise
    # paths that DO need it: two subprocess-import the real `render_report.py`
    # entrypoint (module-scope `import torch`) or `import torch` directly
    # inside the test body, and the third needs matplotlib for the analyzer
    # CLI's four-panel plot, same as this file's other matplotlib entries.
    "tests/test_true_camera_scaling_benchmark.py::test_direct_benchmark_entrypoint_can_import_repo_runtime_modules": _TORCH_REASON,
    "tests/test_true_camera_scaling_benchmark.py::test_no_kernel_execute_path_writes_common_row_without_resolving_host_twice": _TORCH_REASON,
    "tests/test_true_camera_scaling_benchmark.py::test_analyzer_cli_writes_summary_and_four_panel_plot": "requires matplotlib (not installed in the CPU smoke image)",
}


# ---------------------------------------------------------------------------
# Skips gated on an external EXECUTABLE rather than an importable module, so
# they evaporate the moment the tool is present (#462's secondary observation).
# ---------------------------------------------------------------------------
_TOOL_SKIPS: dict[str, tuple[str, str]] = {
    # #462, secondary observation: the ROCm training image ships no C++
    # compiler, so Inductor cannot compile and these two assert on
    # `InductorError: InvalidCxxCompiler ... 'g++'` rather than on anything
    # about the probe. Confirmed pre-existing on main by two agents. Gated on
    # the real cause (no g++ on PATH) instead of ignored, so the probe DOES
    # run wherever a compiler exists -- and so the skip line names the gap
    # instead of the suite silently claiming coverage it never had.
    "tests/test_issue254_dynamo_probe.py::test_compiled_matches_eager_for_every_region": (
        "g++",
        "requires a C++ compiler for torch.compile/Inductor; this image ships none (#462)",
    ),
    "tests/test_issue254_dynamo_probe.py::test_probe_region_is_independently_callable": (
        "g++",
        "requires a C++ compiler for torch.compile/Inductor; this image ships none (#462)",
    ),
}


def _required_module(reason: str) -> str | None:
    """Which module a `_SKIP_REASONS` entry is waiting on, or None.

    Same #462 correction as `collect_ignore` above, applied to the per-test
    list: a "requires torch" skip must not fire in an image that HAS torch.
    Every dependency-driven reason in `_SKIP_REASONS` is written as
    "requires <import-name> ..." precisely so this can be read back off the
    reason text instead of maintained as a second, drift-prone parallel list.
    Reasons that are not about a missing dependency (the container-specific
    /scratch one, the #470 bare-`args` failure) match nothing and stay
    unconditional, which is correct -- they are not waiting on an install.
    """
    m = re.match(r"requires ([A-Za-z_][A-Za-z0-9_]*)", reason)
    return m.group(1) if m else None


def _resolve_tree_identity(root):
    """Identify the git tree rooted at `root`, as `(identity, source)`.

    Resolved from, in order:

      1. `TREE_IDENTITY` -- a plain file holding a commit SHA, written by
         `scripts/compare_baseline.sh` into each `git archive` export. Exports
         have no `.git`, so this is the only thing that can identify them.
      2. `git rev-parse` -- for an ordinary checkout/worktree. A dirty tree is
         reported with a `-dirty` suffix.

    Never raises: a missing identity degrades to an explicit `"unknown"`,
    which is itself informative, rather than breaking collection.
    """
    identity = "unknown"
    source = "none"
    stamp = root / "TREE_IDENTITY"
    if stamp.is_file():
        try:
            text = stamp.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            identity = text.splitlines()[0].strip()
            source = "TREE_IDENTITY"
    if source == "none":
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                identity = proc.stdout.strip()
                source = "git"
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if dirty.returncode == 0 and dirty.stdout.strip():
                    identity += "-dirty"
        except (OSError, ValueError, subprocess.SubprocessError):
            pass

    if source == "none":
        # Inside the compose `train` container, `git rev-parse` fails whenever
        # the bound tree is a git *worktree*: there `.git` is a file pointing at
        # a gitdir outside the mount. (From an ordinary checkout `.git` is a
        # real directory inside `./`, so git works and this never runs.)
        # docker-compose.yml forwards `ELIGSIR_GIT_COMMIT` (issues
        # #891/#784), but only passes through whatever the HOST exported -- it
        # is empty for a plain `docker compose run`. So this recovers the
        # identity only when the caller set it, e.g.
        #   ELIGSIR_GIT_COMMIT=$(git rev-parse HEAD) docker compose run ...
        # Otherwise the banner honestly reports "unknown" rather than guessing.
        env_commit = os.environ.get("ELIGSIR_GIT_COMMIT", "").strip()
        if env_commit:
            identity = env_commit
            source = "ELIGSIR_GIT_COMMIT"

    return identity, source


def pytest_configure(config):
    """Announce which git tree this run is actually executing (issue #1106).

    A "baseline vs branch" test comparison is only meaningful if the two runs
    really executed two different trees. Issue #1106 is a case where they did
    not: `CLAUDE.md` documented a `-v "$PWD":/workspace/3dgs-mcmc` flag, but
    `docker-compose.yml` binds `./:/workspace/eligsir` and sets that as
    `working_dir`, so the `-v` target was read by nothing and BOTH arms of the
    comparison silently ran the branch. It reported a byte-identical, perfectly
    green "no regressions" -- and hid a real one.

    A banner naming the tree makes that failure mode visible instead of silent:
    two identical banners side by side is an obvious red flag.
    `scripts/compare_baseline.sh` reads this line back and refuses a run whose
    two arms report the same identity.

    Deliberately a `print` in `pytest_configure` rather than the more idiomatic
    `pytest_report_header`: header lines are suppressed at `-q`, and `-q` is
    this project's standard invocation (`docker compose run --rm train python
    -m pytest -q tests/`, and CI's own command) -- i.e. the banner would be
    invisible in exactly the runs that need it.
    """
    root = Path(str(config.rootpath))
    identity, source = _resolve_tree_identity(root)
    print(f"tree-identity: {identity} (source={source}, rootdir={root})")


def pytest_collection_modifyitems(config, items):
    for item in items:
        tool = _TOOL_SKIPS.get(item.nodeid)
        if tool is not None and shutil.which(tool[0]) is None:
            item.add_marker(pytest.mark.skip(reason=f"[tests/conftest.py] {tool[1]}"))
            continue
        reason = _SKIP_REASONS.get(item.nodeid)
        if not reason:
            continue
        module = _required_module(reason)
        if module is not None and importlib.util.find_spec(module) is not None:
            continue  # the dependency is present here -- let the test run
        item.add_marker(pytest.mark.skip(reason=f"[tests/conftest.py] {reason}"))


# ---------------------------------------------------------------------------
# Comparing two report.json payloads written to two different output dirs
# (issue #544).
#
# Any test that asserts key-level equality between two reports built in two
# `tempfile.TemporaryDirectory()`s is environment-dependent unless the paths
# are handled: several report values are paths INTO the report's own output
# directory (`output_dir` itself, `artifacts`, `train_contact_sheet`, ...),
# so they always differ between two runs. Those keys only *exist* when the
# report can really render/save the artifact, which needs the training
# image's torchvision/cv2 stack -- hence the false green: the CPU CI image
# skips the test, host python omits the key on both sides and passes, and the
# training image (the only place the code under test actually runs) fails.
#
# The fix is structural rather than an allowlist: NORMALIZE each report
# against its OWN `output_dir`, replacing that prefix with a stable
# placeholder, and only then diff. This is self-maintaining (a new
# artifact-path key needs no conftest change) and, unlike excluding the key
# outright, it still catches a real difference in the REST of the path --
# `<OUTPUT_DIR>/comparison/iter_100/train_contact_sheet.png` vs
# `<OUTPUT_DIR>/comparison/iter_200/train_contact_sheet.png` still fails.
#
# Do NOT re-solve this by adding the offending key to a hand-maintained
# volatile set; that is the drift class #518 and #368 removed elsewhere.
# ---------------------------------------------------------------------------

REPORT_OUTPUT_DIR_PLACEHOLDER = "<OUTPUT_DIR>"


def normalize_report_paths(value, output_dir,
                           placeholder: str = REPORT_OUTPUT_DIR_PLACEHOLDER):
    """Return `value` with every occurrence of `output_dir` replaced.

    Applies to string values at ANY depth (dict values, list/tuple items,
    and dict KEYS -- artifact maps are sometimes keyed by path). Both the
    given `output_dir` and its `os.path.realpath()` are substituted, because
    a report may record either form (``/tmp`` is a symlink on some hosts).
    Non-string leaves are returned unchanged.

    Call this per side, each report against its OWN `output_dir`.
    """
    forms = []
    for candidate in (str(output_dir), os.path.realpath(str(output_dir))):
        candidate = candidate.rstrip(os.sep)
        if candidate and candidate not in forms:
            forms.append(candidate)
    # Longest first, so a realpath that CONTAINS the other is replaced first.
    forms.sort(key=len, reverse=True)

    def _sub(node):
        if isinstance(node, str):
            out = node
            for form in forms:
                out = out.replace(form, placeholder)
            return out
        if isinstance(node, dict):
            return {_sub(k): _sub(v) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return type(node)(_sub(v) for v in node)
        return node

    return _sub(value)


def report_values_equal(a, b) -> bool:
    """`==` except that NaN compares equal to itself, at any depth.

    A report legitimately carries NaN (e.g. `partition_psnr_std` when only
    one partition has views); plain `==` reports those as a difference,
    which is a false failure, not a behavior change.
    """
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(report_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(report_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def report_key_diff(report_a, output_dir_a, report_b, output_dir_b,
                    ignore=()) -> list[str]:
    """Sorted list of top-level report keys that really differ.

    Each report is first normalized against its own `output_dir` (see
    `normalize_report_paths`), NaN compares equal to itself, and the result
    is SORTED so a failure is deterministic and names a real key rather than
    whichever key set-iteration order happened to reach first.

    `ignore` is for values that are volatile for a reason paths cannot cover
    (wall-clock timestamps, elapsed seconds) plus the key under test.
    """
    a = normalize_report_paths(dict(report_a), output_dir_a)
    b = normalize_report_paths(dict(report_b), output_dir_b)
    ignore = set(ignore)
    missing = object()
    return sorted(
        k for k in (set(a) | set(b)) - ignore
        if not report_values_equal(a.get(k, missing), b.get(k, missing))
    )


@pytest.fixture
def report_key_diff_fn():
    """Fixture form of `report_key_diff` (see #544).

    Tests may equally `from conftest import report_key_diff`; the fixture
    exists so a test need not depend on `tests/` being importable.
    """
    return report_key_diff
