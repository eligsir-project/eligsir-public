"""Regression test locking the ORDER invariant that fixes the "lego brick"
trajectory.mp4 bug (eligsir-goal-20260827 campaign, mapping_endpoint
reports on Replica-like scenes bound to a coarse mapping resolution tier,
e.g. ``quarter``).

Root cause chain, confirmed by bisecting the exact commit that produced the
broken artifact (``f1b00027``, 2026-08-29 -- predates this fix):

1. During the mapping phase, train cameras are bound to a non-native
   resolution tier via ``materialize_resolution_tier`` (e.g. ``quarter``),
   which overwrites ``camera.image_width``/``image_height`` in place.
2. ``utils/comparison_report.py``'s trajectory-video block
   (``write_post_training_report``) draws its camera subset for
   ``trajectory.mp4`` from ``train_cams`` via ``_representative_cams`` and
   renders each camera at whatever ``image_width``/``image_height`` it
   currently has -- there is no independent "render at native" step in the
   video path itself.
3. ``utils.trajectory_video.scale_dims_for_video`` then upscales the
   (possibly tiny, tier-bound) rendered frame to an FHD-class canvas
   (``DEFAULT_MIN_VIDEO_WIDTH``) for presentation. Its own module docstring
   is explicit that this is a "post-render, presentation-only upscale ...
   does not make the Gaussian rasterizer render more detail" -- so a
   coarse-tier render upscaled this way produces exactly the observed
   blocky/lego-brick artifact, even though the *encoded* video correctly
   reports an FHD container resolution.

Issue #1259 (commits 06d54bda/#1354, 4e51752c/#1355) fixed the underlying
mismatch by restoring every camera in ``train_cams``/``test_cams`` to its
NATIVE resolution tier ONCE, up front, before ANY camera is drawn for
metrics/sheets/video in ``write_post_training_report`` -- see
``_restore_native_resolution_for_report`` and
``tests/test_report_train_metrics_resolution_tier_restore.py`` (which
verifies the helper itself restores native ``image_width``/``image_height``
and clears the score-domain mismatch).

That existing test does not verify _where_ the restore call is wired
relative to the trajectory-video camera selection -- and #1354's own second
commit exists precisely because its first pass got that wiring wrong
(restored only the smaller metrics subset, leaving the full ``train_cams``/
``test_cams`` population -- which ``trajectory.mp4``, the depth timeseries,
and every contact/overview sheet independently draw their own subsets from
-- a stale mix of native and tiered cameras). This test locks the ordering
in ``write_post_training_report``'s source so a future edit cannot
silently reintroduce the coarse-render-then-upscale bug by moving the
video/sheet camera selection ahead of the restore call, or by restoring a
subset instead of the full camera lists.

Pure ``ast`` inspection -- no torch, no GPU, no dataset -- mirrors the style
of ``scripts/check_production_reachability.py``.
"""

import ast
import inspect

import eligsir.utils.comparison_report as comparison_report


def _write_post_training_report_source():
    src = inspect.getsource(comparison_report.write_post_training_report)
    # Re-parse just this function body so line numbers below are relative to
    # the function itself in a way that's robust to line-shifts elsewhere in
    # the (very large) module.
    tree = ast.parse(src)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    return tree.body[0]


def _calls_to(func_node, callee_name):
    """Every ast.Call node in func_node whose callee is named callee_name."""
    hits = []
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name == callee_name:
                hits.append(node)
    return hits


def test_restore_native_resolution_helper_exists_and_is_called_on_full_lists():
    """``_restore_native_resolution_for_report`` must still be called directly
    on ``train_cams``/``test_cams`` (the full populations), not on some
    smaller subset -- restoring only a metrics subset is the exact bug
    #1354's second commit fixed (see module docstring)."""
    func_node = _write_post_training_report_source()
    calls = _calls_to(func_node, "_restore_native_resolution_for_report")
    assert calls, (
        "write_post_training_report no longer calls "
        "_restore_native_resolution_for_report -- this is the fix for the "
        "coarse-render-then-upscale trajectory.mp4/contact-sheet bug "
        "(issue #1259/#1354); removing it silently reintroduces it."
    )

    arg_names = set()
    for call in calls:
        for arg in call.args:
            if isinstance(arg, ast.Name):
                arg_names.add(arg.id)

    assert "train_cams" in arg_names and "test_cams" in arg_names, (
        "_restore_native_resolution_for_report must be called directly on "
        f"the full train_cams/test_cams lists, not a derived subset; found "
        f"call arguments: {sorted(arg_names)!r}. Restoring only a "
        "metrics-sized subset leaves trajectory.mp4/contact-sheet/overview "
        "code (which each draw their OWN subset of the same objects) mixed "
        "native/stale-tiered -- exactly #1354's second-commit bug."
    )


def test_restore_native_resolution_runs_before_trajectory_video_camera_selection():
    """The restore call must happen before ``cams_for_video`` is built (the
    camera subset actually rendered into trajectory.mp4), or the video
    renders at whatever tier the cameras still carry -- the "lego brick"
    bug."""
    func_node = _write_post_training_report_source()

    restore_calls = _calls_to(func_node, "_restore_native_resolution_for_report")
    assert restore_calls, "see test_restore_native_resolution_helper_exists_and_is_called_on_full_lists"
    last_restore_lineno = max(c.lineno for c in restore_calls)

    # cams_for_video = _representative_cams(train_cams, ...)
    video_select_linenos = []
    for node in ast.walk(func_node):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "cams_for_video"
            and isinstance(node.value, ast.Call)
        ):
            callee = node.value.func
            callee_name = callee.id if isinstance(callee, ast.Name) else getattr(callee, "attr", None)
            if callee_name == "_representative_cams":
                video_select_linenos.append(node.lineno)

    assert video_select_linenos, (
        "could not find `cams_for_video = _representative_cams(...)` in "
        "write_post_training_report -- update this test if the trajectory-"
        "video camera selection was renamed/restructured."
    )

    first_video_select_lineno = min(video_select_linenos)
    assert last_restore_lineno < first_video_select_lineno, (
        "_restore_native_resolution_for_report must run BEFORE "
        "cams_for_video is selected for trajectory.mp4 (restore line "
        f"{last_restore_lineno}, video-select line {first_video_select_lineno}) "
        "-- otherwise the video renders cameras still bound to a stale, "
        "coarse mapping-phase resolution tier and scale_dims_for_video "
        "upscales that coarse render to the FHD canvas, producing a blocky "
        "'lego brick' trajectory.mp4 (issue #1259)."
    )


def test_restore_native_resolution_for_report_still_forces_native_dims():
    """End-to-end sanity on the helper itself, at real-ish pixel sizes (not
    the 16x8 toy fixture in test_report_train_metrics_resolution_tier_restore.py):
    a camera tiered to QUARTER during mapping must come back at its full
    native width/height, not stay at the downsampled size trajectory.mp4
    would otherwise render (and scale_dims_for_video would then upscale)."""
    from types import SimpleNamespace

    import torch

    from eligsir.contracts.control import ResolutionTier
    from eligsir.mapping.resolution_materialization import materialize_resolution_tier
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    native_h, native_w = 480, 640
    rgb = torch.rand(3, native_h, native_w, dtype=torch.float32)
    rgb, provenance = declare_canonical_score_tensor(rgb, context="test:native_gt")
    camera = SimpleNamespace(
        original_image=rgb,
        score_provenance=provenance,
        _sensor_depth_cache=None,
        image_width=native_w,
        image_height=native_h,
        width=native_w,
        height=native_h,
        fx=500.0,
        fy=500.0,
        cx=native_w / 2.0,
        cy=native_h / 2.0,
        gsplat=object(),
        uid=42,
    )

    materialize_resolution_tier(camera, ResolutionTier.QUARTER)
    # Sanity: the tier bug really would have rendered coarse here.
    assert (camera.image_width, camera.image_height) == (native_w // 4, native_h // 4)

    comparison_report._restore_native_resolution_for_report([camera])

    assert (camera.image_width, camera.image_height) == (native_w, native_h), (
        "camera did not come back to its native resolution -- trajectory.mp4 "
        "would render this camera at the coarse mapping tier and "
        "scale_dims_for_video would upscale that coarse render to the FHD "
        "canvas (the 'lego brick' bug)."
    )
    assert tuple(camera.original_image.shape[-2:]) == (native_h, native_w)
