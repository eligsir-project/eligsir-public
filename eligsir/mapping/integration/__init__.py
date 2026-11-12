"""Frame integration services (admission -> insertion).

Wave 1 consolidation plan step 8 (see
the mapping integration ownership boundary.

The plan calls for six narrow services replacing the ~900-line
``insert_gaussians_from_frame`` + ~560-line ``_flush_insertion_batch`` +
rolling-seed/submap-stitch logic currently in ``train_streaming.py``:

- ``FramePreparationService`` — RGB-D conversion, masks, normals, backprojection
- ``IntegrationPlanner`` — coverage filtering, dedup, candidate selection
- ``RGBDIntegrator`` — normal keyframe geometry integration
- ``BootstrapIntegrator`` — initial/rolling-seed behavior
- ``SubmapIntegrator`` — submap-specific integration policy
- ``InsertionCommitter`` — submits a batch of pending candidates via the
  model's sanctioned insertion entrypoint

**Status (2026-08-05):**

- ``InsertionCommitter`` (:mod:`.insertion_committer`) — real, verified
  extraction. ``_flush_insertion_batch`` was already a standalone,
  dependency-clean function, so it moved verbatim with no math change,
  same as the loss functions in step 7.
- ``geometry_helpers`` (:mod:`.geometry_helpers`) — real, verified
  extraction. The 11-function geometry/masking helper cluster
  (``_sample_stable_depth_mask``, ``_gpu_render_occupancy_masks``,
  ``_kornia_normal_map``, etc.) moved verbatim; ``train_streaming.py``
  re-exports them so every existing call site keeps working.
- ``RGBDIntegrator``/``BootstrapIntegrator``/``SubmapIntegrator`` — real,
  callable implementations, satisfied directly by
  :class:`eligsir.mapping.coordinator.MappingCoordinator`'s
  ``step``/``run_rolling_seed_mode``/``run_submap_stitch_mode`` methods
  rather than by separate wrapper classes in this package (an earlier
  version had one wrapper module per Protocol; each was a one-line
  lazy-import pass-through back into ``train_streaming.py``, which added
  indirection without adding ownership, so they were collapsed into the
  coordinator — see its docstring).
- ``rgbd_integrator_impl`` (:mod:`.rgbd_integrator_impl`) — **Wave 2 Task 2
  (2026-08-05): real extraction, verification mixed by mode — see below.**
  ``insert_gaussians_from_frame``
  (~1360 lines), ``_run_rolling_seed`` (~300 lines), ``_run_submap_stitch``
  (~360 lines), and their full ~20-function helper cluster
  (``_save_streaming_checkpoint``, ``_save_cameras_json``,
  ``_save_test_cameras_json``, ``_save_holdout_gt``,
  ``_flush_dense_trajectory``, ``_export_ply_snapshot``, ``_bootstrap_mask``,
  ``_capture_bootstrap_positions``, ``_bootstrap_motion_stats``,
  ``_log_bootstrap_motion``, ``_write_bootstrap_motion_ply``,
  ``_render_streaming_snapshot``, ``_update_provisional_support``,
  ``_maybe_create_pose_graph``, ``_nan_term_report``, ``_add_guarded``,
  ``_sanitize_gaussian_params``, ``_admission_quality_weight_for_stats``,
  ``_pose_provenance_report_dict``) moved verbatim — the earlier "large
  transitive closure" concern turned out, on tracing every call, to be one
  contiguous span of ``train_streaming.py`` (lines 802-2838 pre-move) that
  moved as a single unit. Three small, widely-shared helpers
  (``_get_sensor_depth``, ``_timed``, ``_voxel_downsample_indices``) stay
  in ``train_streaming.py`` and are imported back lazily, since they are
  each used far more by ``streaming_training()``'s main loop than by this
  cluster (moving them would relocate coupling, not reduce it) —
  ``_timed`` additionally shares a module-level ``_PHASE_OPS`` timing dict
  that a verbatim copy would silently fork. ``train_streaming.py``
  re-exports every moved name via a single import block placed where the
  span used to live, so every existing call site (including tests that
  import these names directly from ``train_streaming``) is unchanged.
  :class:`eligsir.mapping.coordinator.MappingCoordinator` now imports
  and calls these three functions directly (no more lazy import inside
  its methods).

  Verification is real but not uniform across the three functions.
  ``insert_gaussians_from_frame`` (the "normal" mode, the default and most
  heavily used of the three) has a real, passing end-to-end Tier C GPU run
  on ``rgbd_dataset_freiburg1_xyz`` through this exact module, plus a
  passing real-GPU byte-identical parity test
  (``tests/test_integration_services_delegate.py::test_mapping_coordinator_step_matches_direct_call``).
  ``_run_rolling_seed``/``_run_submap_stitch`` do **not** have a
  successful Tier C run: both hit a pre-existing device-mismatch crash in
  ``l1_loss`` on this build, confirmed (via an A/B checkout against the
  pre-Task-2 code) to predate this move rather than be caused by it. That
  A/B is real, valuable evidence that the physical relocation didn't
  change behavior — but it is evidence of "identical to the old broken
  behavior," not evidence that either function runs correctly end-to-end.
  Treat those two as moved-and-behavior-preserved, not moved-and-verified-working.
- ``FramePreparationService``/``IntegrationPlanner`` remain
  :mod:`.contracts` protocols only — ``insert_gaussians_from_frame`` does
  frame-prep, planning, and commit in one function with no existing split
  point to delegate a narrower call to.
"""
