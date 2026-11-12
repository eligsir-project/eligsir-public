"""``_run_submap_stitch`` (H11 submap-stitching training mode) and its
pose-graph-creation helper, split out of ``rgbd_integrator_impl.py``
(WS-E5) -- see that module's docstring for the history of why this cluster
ended up as one file.

Verbatim relocation (plus one bugfix, see below), no other logic change.
``rgbd_integrator_impl.py`` re-exports both functions so every existing call
site keeps working unchanged.

Device-mismatch fix (WS-E5, code-review followup): every hardcoded
``device="cuda"`` / ``.to("cuda")`` in ``_run_submap_stitch`` has been
replaced with the appropriate model's ``.get_xyz.device`` -- the
device-resolution convention used throughout the rest of the codebase (e.g.
``utils/energy_mcmc.py``, ``utils/taming_3dgs.py``,
``utils/streaming_depth_lock.py``, ``utils/strategies/mcmc/*.py``: all read
``gaussians.get_xyz.device`` rather than hardcoding ``"cuda"``). Per-submap
tensors use the fresh ``sub_gaussians`` instance's own device; the top-level
background, the merge-into-global-model tensors, and the global-refinement
tensors use the caller-supplied ``gaussians``' device. A hardcoded
``"cuda"`` here could previously mismatch either model's actual device and
crash with a device-mismatch RuntimeError.
"""

from __future__ import annotations

import os

import torch
from tqdm import tqdm

from gaussian_renderer import render
from scene import GaussianModel
from scene.gsplat_model import GsplatGaussianModel
from scene.streaming_scene import StreamingScene
from eligsir.mapping.integration.loss_diagnostics import _pose_provenance_report_dict
from eligsir.mapping.integration.frame_windows import collect_frame_windows
from eligsir.runtime.finalize import report_phase as _report_phase


def _run_submap_stitch(
    gaussians,
    streaming_scene: "StreamingScene",
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
    run_state=None,
):
    """H11: Submap-stitching training mode.

    Divides the incoming frame stream into fixed-size windows (submaps). Each
    submap is independently bootstrapped from its own depth data, locally
    optimised, then merged into a growing global model. A final global
    refinement pass trains over all cameras.

    This provides a clean per-submap geometry baseline to compare against the
    sliding-window approach.
    """
    from eligsir.utils.graphics_utils import BasicPointCloud
    from eligsir.utils.sh_utils import RGB2SH
    from eligsir.utils.general_utils import inverse_sigmoid

    model_layout = getattr(args, "model_layout", "gsplat").lower()
    model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel

    _device = gaussians.get_xyz.device

    submap_frames = max(2, int(getattr(args, "streaming_submap_frames", 20)))
    submap_iters  = max(100, int(getattr(args, "streaming_submap_iters", 3000)))
    refine_iters  = max(0, int(getattr(args, "streaming_global_refine_iters", 5000)))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=_device)

    print(f"[submap] mode: {submap_frames} frames/submap, "
          f"{submap_iters} iters/submap, {refine_iters} global refine iters", flush=True)

    # Drain the source and chunk (frame, camera) pairs into windows. Shared
    # verbatim with _run_rolling_seed (#1075: same orchestration was inlined
    # in both trainer entrypoints); see frame_windows.py's module docstring.
    all_train_cameras, windows = collect_frame_windows(
        streaming_scene, args, submap_frames
    )

    print(f"[submap] {len(all_train_cameras)} total optimization views collected.", flush=True)
    print(f"[submap] {len(windows)} submaps of up to {submap_frames} frames each.", flush=True)

    # --- Per-submap optimisation -------------------------------------------
    all_gaussian_params = []  # collect param dicts per submap

    for sm_idx, sm_pairs in enumerate(windows):
        if not sm_pairs:
            continue
        sm_frames, sm_cams = zip(*sm_pairs)
        sm_cams = list(sm_cams)
        print(f"[submap {sm_idx}] {len(sm_cams)} optimization views, optimizing for {submap_iters} optimization steps...", flush=True)

        # Build a fresh Gaussian model bootstrapped from this submap's depth
        sub_gaussians = model_cls(dataset.sh_degree)

        # Reuse StreamingScene's _build_pcd_from_frames with the actual frames
        _tmp_scene = StreamingScene.__new__(StreamingScene)
        from eligsir.mapping.stores.keyframe_store import KeyframeStore as _KeyframeStore
        _tmp_scene._keyframe_store = _KeyframeStore()
        _tmp_scene.args = args
        _tmp_scene.cameras_extent = 1.0
        # Issue #1118: `_build_pcd_from_frames` reads `self._external_seed`
        # unconditionally (added by #119's RealityScan depth-less source), but
        # that attribute is only assigned in `__init__`, which this stub
        # deliberately skips -- so every submap raised AttributeError from
        # 2026-08-12 until this line existed. `None` (its `__init__` value) is
        # correct rather than the parent scene's seed: the seed branch returns
        # early and ignores `frames` entirely, so propagating it would give
        # every submap the identical global cloud instead of per-window geometry.
        _tmp_scene._external_seed = None
        pcd = _tmp_scene._build_pcd_from_frames(list(sm_frames))

        if pcd is None or pcd.points.shape[0] == 0:
            # Fallback: small random cloud in unit cube
            import numpy as _np
            from eligsir.utils.graphics_utils import BasicPointCloud
            pcd_pts = _np.random.uniform(-0.5, 0.5, (200, 3)).astype(_np.float32)
            pcd_cols = _np.full((200, 3), 0.5, dtype=_np.float32)
            pcd = BasicPointCloud(points=pcd_pts, colors=pcd_cols, normals=_np.zeros_like(pcd_pts))

        sub_gaussians.create_from_pcd(pcd, spatial_lr_scale=1.0,
                                       init_scale_mode="fixed", init_scale=0.01)
        sub_gaussians.training_setup(opt)
        _sub_is_selective = getattr(sub_gaussians, "optimizer_type", "adam") == "selective_adam"

        # Local optimisation loop
        from eligsir.utils.loss_utils import l1_loss, ssim as _ssim_fn
        import random as _rnd
        for _it in range(1, submap_iters + 1):
            cam = _rnd.choice(sm_cams)
            pkg = render(cam, sub_gaussians, pipe, background)
            img = pkg["render"]
            # Issue #1118: `.to(img.device)` is required, not defensive. #398
            # (`d6f0e257`) made train-camera GT images CPU-resident outside the
            # hot window (`--streaming_offload_camera_images`, default on), so a
            # raw `cam.original_image` read is a CPU tensor and the loss below
            # raised "Expected all tensors to be on the same device". Every
            # normal-path consumer already uploads a transient per-camera copy
            # this same way (e.g. train_streaming.py's `_gt = _c.original_image
            # .to(_im.device)`); these two trainers were simply never updated.
            gt = cam.original_image.to(img.device)
            loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - _ssim_fn(img, gt))
            loss.backward()
            if _it < submap_iters:
                if _sub_is_selective:
                    sub_gaussians.prepare_selective_adam_step()
                    _vis_all = torch.ones(sub_gaussians.get_xyz.shape[0], dtype=torch.bool, device=sub_gaussians.get_xyz.device)
                    sub_gaussians.optimizer.step(visibility=_vis_all)
                    sub_gaussians.normalize_rotation_params()
                else:
                    sub_gaussians.optimizer.step()
                sub_gaussians.optimizer.zero_grad(set_to_none=True)

        print(f"[submap {sm_idx}] done, {sub_gaussians.get_xyz.shape[0]} Gaussians.", flush=True)

        # Save the submap's final parameters (detached) using layout-agnostic snapshot
        all_gaussian_params.append({
            "means":     sub_gaussians.means_param.detach().cpu(),
            "sh0":       sub_gaussians._features_dc.detach().cpu(),
            "shN":       sub_gaussians._features_rest.detach().cpu(),
            "opacities": sub_gaussians.raw_opacities.detach().cpu(),
            "scales":    sub_gaussians.raw_scales.detach().cpu(),
            "quats":     sub_gaussians._rotation.detach().cpu(),
        })

        del sub_gaussians
        torch.cuda.empty_cache()

    if not all_gaussian_params:
        print("[submap] No submap params collected; aborting stitch.", flush=True)
        return

    # --- Merge submaps into global model -----------------------------------
    print(f"[submap] Merging {len(all_gaussian_params)} submaps into global model...", flush=True)

    # Use the pre-built gaussians object; extend it with all submap data.
    # Pass opacities_raw and sh_rest so trained submap values are preserved
    # instead of being overwritten with init_opacity=0.3 / zero shN.
    from eligsir.utils.sh_utils import SH2RGB
    # P3 amortized growth: these inserts are back-to-back with no live-state reads
    # between them (get_xyz is only read after the loop), so they are safe to stage
    # and materialize with a single cat per param/buffer. No-op unless the flag is on.
    # Deliberate exception (Wave 2 backend-ownership pass, 2026-08-06): NOT
    # cut over to map_backend.begin_mutation()/insert()/commit_mutation()
    # like the other insertion sites. Each insert below runs inside this
    # begin_staged_inserts()/flush_staged_inserts() block, which the model
    # itself only fully materializes at the flush -- committing a
    # transaction (and bumping map_version/topology_version) once per
    # staged insert here would report topology changes before they're
    # physically real, which is worse than leaving this diagnostic
    # alternate-mode path on the pre-existing direct-call pattern.
    if getattr(gaussians, "begin_staged_inserts", None) is not None:
        gaussians.begin_staged_inserts()
    for sm_params in all_gaussian_params:
        pts   = sm_params["means"].to(_device)
        cols  = sm_params["sh0"].squeeze(1).to(_device)  # (N,3) SH DC
        rgb_approx = SH2RGB(cols).clamp(0, 1)
        log_scales = sm_params["scales"].to(_device)
        quats = sm_params["quats"].to(_device)
        raw_opacities = sm_params["opacities"].to(_device).reshape(-1)  # (N,) pre-logit
        sh_rest_t = sm_params.get("shN", None)
        if sh_rest_t is not None:
            sh_rest_t = sh_rest_t.to(_device)
        gaussians.add_points_as_gaussians(
            pts, rgb_approx,
            scales=log_scales, rotations=quats,
            opacities_raw=raw_opacities,
            sh_rest=sh_rest_t,
            is_provisional=False, birth_frame=0,
            # This merge runs before the global refinement loop starts, so 0 is
            # the literal birth iteration of every merged row in the GLOBAL run's
            # clock (their per-submap training history is not representable here).
            birth_iter=0,
        )
    if getattr(gaussians, "flush_staged_inserts", None) is not None:
        gaussians.flush_staged_inserts()

    print(f"[submap] Global model: {gaussians.get_xyz.shape[0]} Gaussians total.", flush=True)

    # Save merged PLY
    merged_ply_dir = os.path.join(args.model_path, "point_cloud", "submap_merged")
    os.makedirs(merged_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(merged_ply_dir, "point_cloud.ply"))

    if refine_iters <= 0:
        print("[submap] No global refinement requested. Done.", flush=True)
        return

    # --- Global refinement over all cameras --------------------------------
    print(f"[submap] Running {refine_iters} global refinement optimization steps over {len(all_train_cameras)} optimization views...", flush=True)
    # preserve_lineage=True (issue #374): MID-RUN training_setup() on a model
    # already populated by the submap merge above -- without it the reset inside
    # re-stamps every merged row as a freshly-born BOOTSTRAP root, reissues uids,
    # and flattens per-row submap_id. The sub_gaussians.training_setup() call
    # earlier in this file is a genuine fresh bootstrap and stays unchanged.
    if hasattr(gaussians, "_snapshot_lineage"):
        # iteration=0 (issue #374 item 3): this call runs before the
        # global-refinement loop below has taken a single step -- same
        # literal-not-placeholder convention as this file's birth_iter=0
        # comment above.
        gaussians.training_setup(
            opt, preserve_lineage=True,
            lineage_site="submap_stitch:pre_global_refine", iteration=0,
        )
    else:
        gaussians.training_setup(opt)
    _global_is_selective = getattr(gaussians, "optimizer_type", "adam") == "selective_adam"

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
        cam = _rnd.choice(all_train_cameras)
        pkg = render(cam, gaussians, pipe, background)
        img = pkg["render"]
        # Issue #1118: same #398 CPU-residency upload as the per-submap loop
        # above. This is the global-refinement loop, which PR #1105's missing
        # `import os` fix had kept unreachable until this session -- so it had
        # never executed since #398 landed either.
        gt = cam.original_image.to(img.device)
        loss = (1.0 - opt.lambda_dssim) * l1_loss(img, gt) + opt.lambda_dssim * (1.0 - _ssim_fn(img, gt))
        loss.backward()
        if _it < refine_iters:
            if _global_is_selective:
                gaussians.prepare_selective_adam_step()
                _vis_all = torch.ones(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
                gaussians.optimizer.step(visibility=_vis_all)
                gaussians.normalize_rotation_params()
            else:
                gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    # Save final refined PLY
    refined_ply_dir = os.path.join(args.model_path, "point_cloud", "iteration_final")
    os.makedirs(refined_ply_dir, exist_ok=True)
    gaussians.save_ply(os.path.join(refined_ply_dir, "point_cloud.ply"))

    # Issue #1102: this mode's evidence phase is now recorded in the run's
    # RunCompletionState exactly like the normal streaming path's, instead of
    # being swallowed by a bare `except Exception: print(...)` (the #646/#647
    # pattern). `_ReportPhase` below is the real phase context manager when a
    # `run_state` was supplied and a transparent no-op when it was not, so
    # direct/legacy callers keep their previous behavior.
    if run_state is not None:
        run_state.mark_training_complete(final_iteration=refine_iters)
    with _report_phase(run_state, log_prefix="submap") as _ph:
        # Deliberately INSIDE the phase: a mode that produced no test cameras
        # never calls `record_output`, which the phase machinery scores as
        # MISSING_ARTIFACTS/OutputDirectoryUnknown -- fail-closed, never a
        # silent pass (see utils/run_completion_state.py's `_missing_artifacts`).
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
            _eval_bg = torch.tensor(bg_color, dtype=torch.float32, device=gaussians.get_xyz.device)
            _ph.record_output(write_post_training_report(
                args.model_path,
                refine_iters,
                gaussians,
                all_train_cameras,
                streaming_scene.getTestCameras(),
                render,
                pipe,
                _eval_bg,
                tb_writer=tb_writer,
                subdir="submap_final",
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

    print("[submap] Stitch complete.", flush=True)


def _maybe_create_pose_graph(args):
    """Thin wrapper around loop_closure.factory.maybe_create_pose_graph."""
    from loop_closure.factory import maybe_create_pose_graph
    return maybe_create_pose_graph(args)
