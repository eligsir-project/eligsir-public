"""``_run_rolling_seed`` diagnostic streaming-training mode, split out of
``rgbd_integrator_impl.py`` (WS-E5) -- see that module's docstring for the
history of why this cluster ended up as one file.

Verbatim relocation (plus one bugfix, see below), no other logic change.
``rgbd_integrator_impl.py`` re-exports this function so every existing call
site keeps working unchanged.

Device-mismatch fix (WS-E5, code-review followup): every hardcoded
``device="cuda"`` in this function has been replaced with
``gaussians.get_xyz.device`` -- the device-resolution convention used
throughout the rest of the codebase (e.g. ``utils/energy_mcmc.py``,
``utils/taming_3dgs.py``, ``utils/streaming_depth_lock.py``,
``utils/strategies/mcmc/*.py``: all read ``gaussians.get_xyz.device`` rather
than hardcoding ``"cuda"``). A hardcoded ``"cuda"`` here could previously
mismatch the Gaussian model's actual device (e.g. an explicit
non-default-cuda placement) and crash with a device-mismatch RuntimeError
when combined with tensors read off ``gaussians``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from tqdm import tqdm

from gaussian_renderer import render
from eligsir.optimization.gaussian.losses import _snapshot_energy_grads
from eligsir.mapping.integration.loss_diagnostics import _pose_provenance_report_dict
from eligsir.mapping.integration.frame_windows import collect_frame_windows
from eligsir.runtime.finalize import report_phase as _report_phase


def _run_rolling_seed(
    gaussians,
    streaming_scene,
    opt,
    pipe,
    args,
    background,
    dataset,
    tb_writer,
    save_worker,
    testing_iterations,
    saving_iterations,
    sh_degree_schedule,
    map_backend=None,
    run_state=None,
):
    """
    Diagnostic mode: Aggregates geometry by streaming through windows of frames,
    building depth point clouds, and inserting them into the global map via the
    occupancy grid. No training occurs during the streaming phase.
    Finally, performs global optimization for the specified iterations.
    """
    import torch
    import os
    import numpy as np
    from tqdm import tqdm
    from gaussian_renderer import render

    _device = gaussians.get_xyz.device

    submap_frames = max(2, int(getattr(args, "streaming_submap_frames", 20)))
    refine_iters  = max(0, int(getattr(args, "streaming_global_refine_iters", 5000)))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=_device)

    print(f"[rolling_seed] mode: {submap_frames} frames/window, {refine_iters} global refinement optimization steps", flush=True)

    # Drain the source and chunk (frame, camera) pairs into windows. Shared
    # verbatim with _run_submap_stitch (#1075: same orchestration was inlined
    # in both trainer entrypoints); see frame_windows.py's module docstring.
    all_train_cameras, windows = collect_frame_windows(
        streaming_scene, args, submap_frames
    )

    print(f"[rolling_seed] Aggregating geometry across {len(windows)} windows...", flush=True)

    for sm_idx, sm_pairs in enumerate(windows):
        if not sm_pairs:
            continue

        sm_frames, sm_cams = zip(*sm_pairs)

        # Build PCD from this window
        from scene.streaming_scene import StreamingScene
        _tmp_scene = StreamingScene.__new__(StreamingScene)
        from eligsir.mapping.stores.keyframe_store import KeyframeStore as _KeyframeStore
        _tmp_scene._keyframe_store = _KeyframeStore()
        _tmp_scene.args = args
        _tmp_scene.cameras_extent = streaming_scene.cameras_extent
        # Issue #1118: see the identical note in submap_stitch_application.py.
        # `_build_pcd_from_frames` reads `self._external_seed` unconditionally
        # (#119) but only `__init__` assigns it, and this stub skips `__init__`.
        # `None` rather than `streaming_scene._external_seed`: the seed branch
        # ignores `frames`, so propagating it would give every window the same
        # global cloud instead of per-window geometry.
        _tmp_scene._external_seed = None
        pcd = _tmp_scene._build_pcd_from_frames(list(sm_frames))

        if pcd is None or pcd.points.shape[0] == 0:
            continue

        # Filter points through global occupancy grid
        voxel_size = getattr(args, "streaming_insert_voxel_size", 0.02)
        occupied = streaming_scene.check_occupancy(pcd.points, voxel_size, check_neighbors=True)
        unoccupied = ~occupied

        if not unoccupied.any():
            continue

        # Extract new points
        new_pts = torch.tensor(pcd.points[unoccupied], dtype=torch.float32, device=_device)
        new_cols = torch.tensor(pcd.colors[unoccupied], dtype=torch.float32, device=_device)

        # Compute KNN scale with stricter clamp to prevent blobbing
        if new_pts.shape[0] > 1:
            from eligsir.utils.rocm_knn_fallback import distCUDA2
            dist_sq = distCUDA2(new_pts)
            scales_1d = torch.clamp(dist_sq.sqrt() * 0.5, 1e-4, 0.02)  # max 2cm radius
            import math
            log_scales = torch.log(scales_1d).unsqueeze(-1).expand(-1, 3).contiguous()
        else:
            import math
            log_scales = torch.full((new_pts.shape[0], 3), math.log(0.01), device=_device)

        # Add to global gaussians
        if map_backend is not None:
            from eligsir.map.mutation import GaussianInsertionBatch

            _txn = map_backend.begin_mutation()
            added = _txn.insert(GaussianInsertionBatch(
                points=new_pts, colors=new_cols,
                scales=log_scales,
                init_opacity=0.9,  # High opacity
                is_provisional=False,
                # These window inserts all run BEFORE the global refinement loop
                # below (`opt.iterations = refine_iters` / the `_it` loop) even
                # exists, so there is no training iteration to attribute them to:
                # 0 is the correct, literal birth iteration here, not a placeholder.
                birth_frame=0,
                birth_iter=0,
            ))
            map_backend.commit_mutation(_txn)
        else:
            added = gaussians.add_points_as_gaussians(
                new_pts, new_cols,
                scales=log_scales,
                init_opacity=0.9,  # High opacity
                is_provisional=False,
                birth_frame=0,
                birth_iter=0,  # pre-refinement aggregation; see comment above
            )

        # Update occupancy
        streaming_scene.add_to_occupancy_hash(new_pts)
        print(f"[rolling_seed] Window {sm_idx+1}/{len(windows)}: added {added} points. Total: {gaussians.get_xyz.shape[0]}", flush=True)

    print(f"[rolling_seed] Geometry aggregation complete. Total points: {gaussians.get_xyz.shape[0]}", flush=True)

    if refine_iters <= 0:
        print("[rolling_seed] No global refinement requested. Skipping to evaluation.", flush=True)
    else:
        print(f"[rolling_seed] Running {refine_iters} global refinement optimization steps over {len(all_train_cameras)} optimization views...", flush=True)

        # Scale optimization schedules to fit the shorter refine_iters window
        opt.iterations = refine_iters
        opt.position_lr_max_steps = refine_iters
        opt.densify_until_iter = int(refine_iters * 0.8)  # Stop MCMC at 80% to allow convergence
        opt.mcmc_stop_growth_iter = int(refine_iters * 0.8)

        import numpy as np
        sh_degree_schedule = [int(x) for x in np.linspace(refine_iters // 20, refine_iters // 2, dataset.sh_degree)]

        # Re-setup training to recreate optimizers for all points.
        # preserve_lineage=True (issue #374): this is a MID-RUN training_setup()
        # on an already-populated model (every window's points were inserted
        # above), so the reset inside it would otherwise re-stamp every existing
        # Gaussian as a freshly-born BOOTSTRAP root and reissue uids, destroying
        # the provenance #283/#288 depend on. Legacy GaussianModel has no lineage
        # buffers and no such kwarg, hence the capability check.
        if hasattr(gaussians, "_snapshot_lineage"):
            # iteration=0 (issue #374 item 3): this call runs before the
            # global-refinement loop below has taken a single step -- same
            # literal-not-placeholder convention as this file's birth_iter=0
            # comment above.
            gaussians.training_setup(
                opt, preserve_lineage=True,
                lineage_site="rolling_seed:pre_global_refine", iteration=0,
            )
        else:
            gaussians.training_setup(opt)
        _global_is_selective = getattr(gaussians, "optimizer_type", "adam") == "selective_adam"

        from eligsir.utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
        from eligsir.utils.strategies import make_mcmc_strategy
        from eligsir.utils.compiled_kernels import configure_torch_compile, set_compile_iteration

        densification_strategy = getattr(opt, "densification_strategy", "gsplat_energy_mcmc").lower()
        mcmc_cfg = MCMCScheduleConfig(
            start_iter=opt.densify_from_iter,
            # _mcmc_growth_schedule_horizon (set by apply_online_schedule_overrides,
            # 2026-07-18) is the growth-RATE schedule's normalization horizon,
            # deliberately decoupled from mcmc_stop_growth_iter's hard-stop gate --
            # see that function's docstring. Falls back to mcmc_stop_growth_iter
            # for callers that never went through the online-mode overrides.
            stop_growth_iter=getattr(opt, "_mcmc_growth_schedule_horizon", None)
            or getattr(opt, "mcmc_stop_growth_iter", 12_000),
            stop_reloc_iter=opt.densify_until_iter,
            growth_factor_start=getattr(opt, "mcmc_growth_factor_start", 1.05),
        )
        mcmc_strategy = make_mcmc_strategy(densification_strategy, gaussians=gaussians, args=args)
        mcmc_strategy.initialize_state(gaussians=gaussians, args=args)
        configure_torch_compile(args)

        from eligsir.utils.loss_utils import l1_loss, ssim as _ssim_fn
        import random as _rnd

        # #878: honour --disable_progress_bar / --quiet here too; this bar is
        # reachable from a streaming run and previously ignored both flags.
        _refine_bar_disabled = (
            getattr(args, "disable_progress_bar", False)
            or getattr(args, "quiet", False)
        )
        for _it in tqdm(
            range(1, refine_iters + 1),
            desc="Global refine",
            disable=_refine_bar_disabled,
        ):
            set_compile_iteration(_it)
            cam = _rnd.choice(all_train_cameras)

            xyz_lr = gaussians.update_learning_rate(_it)
            if _it in sh_degree_schedule:
                gaussians.oneupSHdegree()

            bg = torch.rand((3,), device=_device) if opt.random_background else background

            pkg = render(cam, gaussians, pipe, bg)
            img = pkg["render"]
            # Issue #1118: see the identical note in submap_stitch_application.py
            # -- #398 made train-camera GT images CPU-resident outside the hot
            # window, so this read must upload a transient copy like every
            # normal-path consumer already does.
            gt = cam.original_image.to(img.device)
            loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - _ssim_fn(img, gt))

            # Phase 5: differentiable energy loss flag (replaces reg call sites when enabled)
            _energy_loss_enabled_p1 = getattr(args, "energy_loss_enabled", False)
            if _energy_loss_enabled_p1:
                from eligsir.utils.differentiable_energy_loss import compute_differentiable_loss
                _p5_active_mask = pkg["visibility_filter"].detach() if _global_is_selective else None
                _p5_result = compute_differentiable_loss(
                    gaussians=gaussians,
                    iteration=_it,
                    cap_max=args.cap_max,
                    active_mask=_p5_active_mask,
                    include_effective_count=False,
                    w_opacity_reg=args.opacity_reg,
                    w_scale_reg=args.scale_reg,
                    w_opacity_entropy=args.lambda_opacity_entropy,
                    w_anisotropy_reg=float(args.anisotropy_reg),
                    anisotropy_soft_limit=float(getattr(args, "anisotropy_soft_limit", 10.0)),
                    w_needle_reg=float(getattr(args, "needle_reg_weight", 0.0)),
                    needle_threshold=float(getattr(args, "needle_reg_threshold", 0.3)),
                scale_reg_mode=str(getattr(args, "scale_reg_mode", "l1")),
                    max_iterations=opt.iterations,
                )
                loss = loss + _p5_result.total
            else:
                # Regularization to prevent Gaussians from exploding
                from eligsir.utils.compiled_kernels import active_reg_core
                _w_aniso = float(args.anisotropy_reg)
                if _global_is_selective:
                    _active_reg = pkg["visibility_filter"].detach()
                    _reg_loss = active_reg_core(
                        gaussians.get_opacity[_active_reg],
                        gaussians.get_scaling[_active_reg],
                        w_opacity=args.opacity_reg,
                        w_scale=args.scale_reg,
                        w_aniso=_w_aniso,
                    )
                else:
                    _reg_loss = active_reg_core(
                        gaussians.get_opacity,
                        gaussians.get_scaling,
                        w_opacity=args.opacity_reg,
                        w_scale=args.scale_reg,
                        w_aniso=_w_aniso,
                    )
                loss = loss + _reg_loss

            mcmc_strategy.step_pre_backward(gaussians=gaussians, args=args, iteration=_it, render_pkg=pkg, loss=loss)
            loss.backward()

            if _it < refine_iters:
                if _global_is_selective:
                    gaussians.prepare_selective_adam_step()
                    _vis_all = pkg["visibility_filter"].detach()

                    # Zero invisible grads (safety)
                    if getattr(args, "selective_adam_zero_invisible_grads", True):
                        for group in gaussians.optimizer.param_groups:
                            p = group["params"][0]
                            if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                                p.grad[~_vis_all] = 0.0

                    gaussians.optimizer.step(visibility=_vis_all)
                    gaussians.normalize_rotation_params(mask=_vis_all)
                else:
                    gaussians.optimizer.step()
                    if pipe.gsplat_sparse_grad:
                        gaussians.normalize_rotation_params()

                _energy_grad_snapshot = (
                    _snapshot_energy_grads(gaussians)
                    if (args.energy_mcmc and densification_strategy in {"mcmc", "hybrid", "gsplat_energy_mcmc"})
                    else None
                )
                gaussians.optimizer.zero_grad(set_to_none=True)

                # MCMC
                mcmc_strategy.inject_noise(gaussians=gaussians, args=args, xyz_lr=xyz_lr, visible=pkg["visibility_filter"].detach() if _global_is_selective else None, sparse_active_set=_global_is_selective, iteration=_it, tb_writer=tb_writer)

                use_energy_mcmc = args.energy_mcmc and densification_strategy in {"mcmc", "hybrid", "gsplat_energy_mcmc"}
                if use_energy_mcmc:
                    from eligsir.utils.energy_mcmc import compute_gaussian_utility
                    utility = compute_gaussian_utility(
                        gaussians=gaussians, render_pkg=pkg, iteration=_it,
                        grad_overrides=_energy_grad_snapshot,
                        w_alpha=args.energy_w_alpha, w_vis=args.energy_w_vis,
                        w_grad=args.energy_w_grad, w_scale=args.energy_w_scale,
                        w_dead=args.energy_w_dead,
                        w_support=getattr(args, "energy_w_support", 2.0),
                        beta_opacity=getattr(args, "energy_beta_opacity", 1.0),
                        beta_scale=getattr(args, "energy_beta_scale", 0.5),
                        # No getattr fallback: energy_alpha_dead is an
                        # unconditional OptimizationParams default, always
                        # present on the real parsed args (issue #368 -- old
                        # getattr(..., 0.005) fallback drifted from the real
                        # default of 0.05, the same stale value that already
                        # caused a confirmed live bug once before the parser
                        # default was raised; removed rather than re-stated).
                        alpha_dead=args.energy_alpha_dead,
                        w_aniso_utility=float(getattr(args, "utility_aniso_weight", 0.1)),
                        aniso_soft_limit=float(getattr(args, "utility_aniso_soft_limit", 10.0)),
                        args=args,
                    )
                    import math
                    u_temp = min(_it / 30000.0, 1.0)
                    tau_t = max(getattr(args, "energy_temp_tau", 0.4), 1e-6)
                    denom_t = 1.0 - math.exp(-1.0 / tau_t)
                    temperature = args.energy_temp_min + (args.energy_temp_start - args.energy_temp_min) * (
                        1.0 - (1.0 - math.exp(-u_temp / tau_t)) / denom_t
                    )
                    if _global_is_selective:
                        _vis = pkg["visibility_filter"].detach()
                        utility[~_vis] = getattr(args, "energy_w_alpha", 1.0) * gaussians.get_opacity.squeeze(-1)[~_vis]
                else:
                    utility = None
                    temperature = 1.0

                sched = get_mcmc_schedule(_it, gaussians.get_xyz.shape[0], args.cap_max, mcmc_cfg)
                mcmc_strategy.step_post_backward(gaussians=gaussians, args=args, sched=sched, iteration=_it, utility=utility, temperature=temperature, use_energy_mcmc=use_energy_mcmc, tb_writer=tb_writer, should_log_strategy=lambda i: False, render_pkg=pkg, lr=xyz_lr)

    # Save final refined PLY
    refined_ply_dir = os.path.join(args.model_path, "point_cloud", "iteration_final")
    os.makedirs(refined_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(refined_ply_dir, "point_cloud.ply"))

    # Eval
    # Issue #1102: recorded as this run's final-report evidence phase (see the
    # matching comment in submap_stitch_application.py) instead of being
    # swallowed by a bare `except Exception: print(...)`.
    if run_state is not None:
        run_state.mark_training_complete(final_iteration=refine_iters)
    with _report_phase(run_state, log_prefix="rolling_seed") as _ph:
        if streaming_scene.getTestCameras():
            from eligsir.utils.comparison_report import (
                write_post_training_report,
                git_provenance_from_args as _git_provenance_from_args,
            )
            from eligsir.utils.experiment_origin import (
                experiment_origin_from_args as _experiment_origin_from_args,
            )
            from eligsir.utils.execution_semantics import (
                build_execution_semantics_block as _execution_semantics_from_args,
            )
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            _eval_bg = torch.tensor(bg_color, dtype=torch.float32, device=_device)
            _ph.record_output(write_post_training_report(
                args.model_path, refine_iters, gaussians, all_train_cameras, streaming_scene.getTestCameras(),
                render, pipe, _eval_bg, tb_writer=tb_writer, subdir="rolling_seed_final",
                train_metrics_max=int(getattr(args, "streaming_report_train_metrics_max", 128)),
                test_metrics_max=int(getattr(args, "streaming_report_test_metrics_max", 128)),
                trajectory_max_frames=int(getattr(args, "streaming_report_trajectory_max_frames", 120)),
                frame_admission=args.streaming_frame_admission,
                frames_admitted_override=int(getattr(streaming_scene, "_keyframes_admitted", 0)) or None,
                pose_provenance=_pose_provenance_report_dict(args),
                streaming_scene=streaming_scene,
                # Issue #259: this mode writes its own report.json, so it needs
                # the same passive commit record every other report path gets.
                git_provenance=_git_provenance_from_args(args),
                # Issue #511: same task-provenance record every other report
                # path gets (pure accessor, resolved once at startup).
                experiment_origin=_experiment_origin_from_args(args),
                # Issue #168: likewise for the resolved run-mode axes. This
                # mode is one of the separate trainers the contract flags via
                # integration_policy_is_separate_trainer.
                execution_semantics=_execution_semantics_from_args(args),
                # Issue #524 acceptance criterion 3: declared run intent +
                # resolved profile stack + effective config. Pure accessor,
                # same mechanism as experiment_origin/execution_semantics above.
                run_intent=getattr(args, "run_intent_report", None),
                # Issue #705: same startup-validated declaration context as
                # the standard streaming report path.
                experiment_context=getattr(args, "experiment_context", None),
            ))

    print("[rolling_seed] Mapping runtime complete.", flush=True)
