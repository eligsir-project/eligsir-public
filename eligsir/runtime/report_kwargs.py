"""Issue #1075 (task P3): the shared streaming report-kwargs builder.

``train_streaming.streaming_training()`` writes four ``report.json`` variants
(pre-training, two mid-run ticks, and the final one) and every one of them
splats the *same* kwargs dict into
``utils.comparison_report.write_post_training_report``.  That dict used to be
built by ``_streaming_report_kwargs()``, a ~280-line closure nested inside the
5000-line training loop, which reached ~35 loop locals implicitly through its
closure cell.

Nothing about that dict is loop-shaped: it is a pure projection of a set of
already-finished telemetry objects into report kwargs.  Lifting it here makes
every one of those ~35 dependencies an explicit, named, keyword-only parameter
instead of an invisible free variable -- which is exactly the failure mode
issue #729 hit in production (``_git_provenance_from_args`` resolved as a free
variable of the enclosing scope, was bound only inside a conditional, and every
mid-training report silently died with a ``NameError`` that ``_run_mid_report``
swallowed).  A free variable that is now a required keyword argument cannot
repeat that: an unbound name is a ``TypeError`` at the call site, in every
report tick, not a swallowed ``NameError`` in some of them.

**Behavior is intentionally unchanged.**  The body below is the closure's body
verbatim (comments included); the only edits are mechanical renames of the
captured names to their parameter spellings and the replacement of the
``nonlocal _partition_control_roles`` write with the explicit
``set_partition_control_roles`` callback (issue #951's roles refresh is a real
side effect of building the report and had to stay one).

Two of the collaborators are still passed in as callables rather than imported
here -- ``multires_cache_report_block`` and ``load_bag_source_manifest`` are
defined in ``train_streaming.py`` itself, so importing them would make this
module circular with its only caller.  The rest resolve from ``utils`` /
``eligsir`` packages directly.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from eligsir.utils.sync_pair_ledger import load_sync_pair_ledger

from eligsir.mapping.integration.rgbd_integrator_impl import (
    _pose_provenance_report_dict as pose_provenance_report_dict,
)
from eligsir.mapping.resolution_provenance import resolution_report_block
from eligsir.runtime.deterministic_digest import report_with_global_digest
from eligsir.runtime.transfers import (
    peek_device_transfer_owner,
    transfer_pool_report_block,
)
from eligsir.utils.execution_semantics import (
    build_execution_semantics_block as execution_semantics_from_args,
)
from eligsir.utils.comparison_report import determinism_from_args
from eligsir.utils.source_attestation import source_attestation_from_args
from eligsir.utils.admission_cadence import admission_cadence_report_block
from eligsir.utils.realized_cadence import realized_cadence_report_block
from eligsir.utils.supervision_work import supervision_work_report_block
from eligsir.utils.presentation_ssaa import presentation_ssaa_kwargs_from_args
from eligsir.utils.trajectory_video import presentation_trajectory_interpolate_kwargs_from_args

__all__ = ["build_streaming_report_kwargs"]


def build_streaming_report_kwargs(
    *,
    final: bool = False,
    # --- run configuration / scene state -------------------------------
    args: Any,
    streaming_scene: Any,
    frame_source: Any,
    first_iter: int,
    # --- telemetry objects read at report time -------------------------
    seed_fade_monitor: Any,
    cadence: Any,
    admission_cadence: Any,
    coordinator: Any,
    spf_resolution: Any,
    resolution_provenance: Any,
    resolution_tier_log: Any,
    ingestion_report: Any,
    replica_rgb_report: Any,
    soft_all_admitter: Any,
    soft_all_invariant: dict,
    reinit_tracker: Any,
    partition_control_runtime: Any,
    # --- optional telemetry owners (None => explicit "disabled" block) --
    mem_tracker: Any = None,
    resource_monitor: Any = None,
    duty: Any = None,
    backpressure_owner: Any = None,
    # --- supervision-work clocks (plain counters, read by value) -------
    clk_optimizer_step_id: int = 0,
    clk_supervision_view_count: int = 0,
    clk_rasterizer_call_count: int = 0,
    clk_supervised_pixel_count: int = 0,
    per_source_supervision_counts: Optional[dict] = None,
    wall_time_s: float = 0.0,
    # --- report-shaping caps -------------------------------------------
    report_train_metrics_max: Any = None,
    report_test_metrics_max: Any = None,
    report_trajectory_max_frames: Any = None,
    # --- collaborators owned by the training loop ----------------------
    depth_insertion_report_block: Callable[[], dict],
    lineage_preserving_setup_report_block: Callable[[], Optional[dict]],
    seed_cohort_report_block: Callable[[], dict],
    ingress_ownership_report_block: Callable[[], dict],
    register_supervision_source: Callable[[Any], None],
    git_provenance_from_args: Callable[[Any], Any],
    multires_cache_report_block: Callable[[], dict],
    load_bag_source_manifest: Callable[[Optional[str]], Optional[dict]],
    set_partition_control_roles: Callable[[Any], None],
    cadence_reconcile_kwargs: Callable[[], dict],
    exposure_opacity_survival_block: Optional[Callable[[], Optional[dict]]] = None,
    streaming_phase: Optional[dict] = None,
    coverage_counters: Optional[dict] = None,
) -> dict:
    """Build the kwargs dict every streaming ``report.json`` write splats.

    ``final=True`` only for the end-of-run report -- a pre-run or mid-run
    report legitimately has ``total_inserted == 0``, so the depth-insertion
    gate must not run there (issue #303).

    Side effects (both pre-existing, both deliberate):
      * every train camera is passed to ``register_supervision_source``;
      * ``set_partition_control_roles`` is called with issue #951's refreshed
        roles (this replaces the old ``nonlocal`` write into the training
        loop's ``_partition_control_roles``).
    """
    from eligsir.utils.experiment_origin import (
        experiment_origin_from_args as _experiment_origin_from_args_local,
    )
    # ``final=True`` only for the end-of-run report: a pre-run or mid-run
    # report legitimately has total_inserted == 0, so the depth-insertion
    # gate must not run there (issue #303).
    _extra = {"depth_insertion": depth_insertion_report_block()} if final else {}
    _lineage_block = lineage_preserving_setup_report_block()
    if _lineage_block is not None:
        _extra["lineage_preserving_setup"] = _lineage_block
    _extra["seed_cohort"] = seed_cohort_report_block()
    # Issue #1262 §6.2.1. Emitted ONLY when the resolution-exposure feature is
    # on -- a flag-off report is byte-identical to before this existed.
    if exposure_opacity_survival_block is not None:
        _exposure_block = exposure_opacity_survival_block()
        if _exposure_block is not None:
            _extra["exposure_opacity_survival"] = _exposure_block
    # #602: built ONCE and referenced twice below (as the top-level
    # "realized_cadence" key, and embedded inside "admission_cadence").
    # realized_cadence_report_block is pure -- summary() + reconcile(),
    # never the one-shot maybe_warn() latch -- so calling it twice would
    # be correct today, but it would put two independently-built copies of
    # the same block in one report.json, which is exactly the shape that
    # silently desyncs when a future edit touches one call site.
    _realized_cadence_block = realized_cadence_report_block(
        cadence,
        iterations_elapsed=max(0, cadence.last_iteration - first_iter),
        **cadence_reconcile_kwargs(),
    )
    # Issue #837: always emitted (mid-run and final), same
    # always-on convention as multires_cache/seed_cohort above.
    _extra["ingress_ownership"] = ingress_ownership_report_block()
    if streaming_phase is not None:
        _extra["streaming_phase"] = dict(streaming_phase)
    # Issue #951: always stamp the declared/realized control-vs-scored
    # observation roles. Refresh the final scored pool without changing a
    # frozen CONTROL_PROBE identity set.
    _roles, _unused_control_cams = partition_control_runtime.prepare(
        list(streaming_scene.train_cameras),
        list(streaming_scene.getTestCameras()),
    )
    set_partition_control_roles(_roles)
    _extra["partition_control"] = (
        partition_control_runtime.recorder.report_block(_roles)
    )
    # Issue #601: always emitted when the bounded backpressure owner is
    # active. An ABSENT key means the run predates #601 or never enabled
    # the owner -- deliberately distinguishable from "the owner ran and
    # skipped nothing" (which emits a block with skipped_fraction 0.0).
    if backpressure_owner is not None:
        _extra["backpressure"] = backpressure_owner.report_block()
    # Issue #464: always emitted (mid-run and final) -- an all-failed
    # resolution cache is exactly the case that used to be invisible.
    _extra["multires_cache"] = multires_cache_report_block()
    # Issue #399: always emitted -- an explicit disabled/never-sampled
    # block is the informative case (an ABSENT key means the report
    # predates #399, which must stay distinguishable from "telemetry ran
    # and found nothing").
    if mem_tracker is not None:
        _extra["gpu_memory"] = mem_tracker.report_block()
    else:
        from eligsir.utils.gpu_memory_telemetry import disabled_report_block

        _extra["gpu_memory"] = disabled_report_block(
            "telemetry disabled (--no-streaming_memory_telemetry)"
            if not bool(getattr(args, "streaming_memory_telemetry", True))
            else "telemetry disabled after a snapshot failure (see train.log)"
        )
    # Host resource monitor (SPLAT_RESOURCE_MONITOR=1). Same always-emitted,
    # tri-state convention as gpu_memory above: ABSENT means the report
    # predates this feature, `enabled: false` means it was off (default) or
    # Glances was unreachable at startup (see `reachable`), and a populated
    # block means it actually sampled the host over the run.
    if resource_monitor is not None:
        _extra["resource_monitor"] = resource_monitor.summary()
    else:
        from eligsir.utils.resource_monitor import disabled_summary

        _extra["resource_monitor"] = disabled_summary()
    # Issue #562: same always-emitted, tri-state convention as gpu_memory
    # above -- ABSENT means the report predates #562, `enabled: false` means
    # the timeline was off, `n_windows: 0` means it ran but never closed a
    # window.
    if duty is not None:
        _extra["duty_cycle"] = duty.report_block()
    else:
        from eligsir.utils.duty_cycle import disabled_report_block as _duty_disabled

        _extra["duty_cycle"] = _duty_disabled(
            "telemetry disabled (--streaming_duty_cycle_telemetry not set)"
        )
    # Issue #1259: pinned-transfer-pool high-water-mark, if a
    # DeviceTransferOwner was ever constructed for this run's device. Read-only
    # lookup (never constructs an owner as a side effect of reporting).
    # ABSENT means no owner was ever registered for this device (or this
    # streaming_scene has no live gaussians model to resolve a device from,
    # e.g. a test double) -- distinct from "an owner exists and used 0 bytes".
    _scene_gaussians = getattr(streaming_scene, "gaussians", None)
    _scene_device = getattr(getattr(_scene_gaussians, "get_xyz", None), "device", None)
    _transfer_pool_block = (
        transfer_pool_report_block(peek_device_transfer_owner(_scene_device))
        if _scene_device is not None
        else None
    )
    if _transfer_pool_block is not None:
        _extra["transfer_pool"] = _transfer_pool_block
    for _supervision_camera in streaming_scene.train_cameras:
        register_supervision_source(_supervision_camera)
    return {
        **_extra,
        "train_metrics_max": report_train_metrics_max,
        "test_metrics_max": report_test_metrics_max,
        "trajectory_max_frames": report_trajectory_max_frames,
        # Issue #397: trajectory.mp4 resolution policy -- these are
        # always concrete ints here (StreamingParams' own argparse
        # defaults), so for a streaming run --streaming_report_video_
        # min_width/max_width IS the configuration knob; the
        # SPLAT_REPORT_VIDEO_MIN_WIDTH/MAX_WIDTH env-var fallback inside
        # write_post_training_report only ever fires for a caller (only
        # train.py's legacy path today) that passes None explicitly.
        "report_video_min_width": int(getattr(args, "streaming_report_video_min_width", 1920)),
        "report_video_max_width": int(getattr(args, "streaming_report_video_max_width", 3840)),
        # Issue #397: was previously a function default (30) that no
        # caller could override -- see StreamingParams'
        # streaming_report_video_fps docstring.
        "mp4_fps": int(getattr(args, "streaming_report_video_fps", 30)),
        "seed_fade": seed_fade_monitor.report_block(),
        "frame_admission": args.streaming_frame_admission,
        # Issue #600: the soft-all capacity bound's realized behavior, plus
        # the insertion invariant. `inserted_from_skipped_frame` must be 0;
        # a nonzero value means a mapping-skipped frame reached Phase 2
        # depth insertion and this run's geometry cannot be attributed to
        # its admitted supervision set.
        "soft_all_admission": dict(
            soft_all_admitter.summary(),
            **soft_all_invariant,
            # False for force_all / hybrid_keyframe / fixed_stride, where
            # this admitter is constructed but never wired -- so a reader
            # cannot mistake its all-zero counters for "soft-all ran and
            # never skipped".
            active=(args.streaming_frame_admission
                    not in ("force_all", "preselected", "hybrid_keyframe", "fixed_stride")),
        ),
        # Cumulative, monotonic count of frames ever admitted as keyframes --
        # NOT len(train_cams), which undercounts whenever partition-focus's
        # demote_partitions() has evicted cameras out of the active buffer
        # since admission (observed: reported frames_admitted=22 vs a real
        # admission rate of ~14% shown by the training progress bar's adm%,
        # which is also derived from _keyframes_admitted).
        "frames_admitted_override": int(
            coverage_counters.get("frames_admitted")
            if isinstance(coverage_counters, dict)
            and coverage_counters.get("frames_admitted") is not None
            else getattr(streaming_scene, "_keyframes_admitted", 0)
        ),
        # Coverage-v2 is assembled by the streaming loop at the one point
        # that knows every source disposition.  Forward it verbatim; deriving
        # it from resident cameras would reintroduce the holdout gap.
        "coverage_counters": (
            dict(coverage_counters) if isinstance(coverage_counters, dict) else None
        ),
        "pose_provenance": pose_provenance_report_dict(args),
        # Issue #302: the resolved frame-release pacing and WHY it was
        # chosen. Lands in report.json["ingestion"]; see
        # resolve_ingestion_mode_with_reason's docstring for the
        # online_mode -> dataset_fps regression this exists to make
        # self-identifying rather than forensic.
        "ingestion": dict(ingestion_report),
        # Issue #641: record Replica synthetic RGB source fidelity in the run
        # artifact so pre-fix and post-fix runs self-identify their protocol
        # difference. Omitted (None) for non-Replica runs.
        **({"replica_rgb": dict(replica_rgb_report)} if replica_rgb_report else {}),
        # Issue #596: the REQUESTED-vs-RESOLVED half of the cadence story,
        # and which mechanism owns the resolved value. Lands at
        # report.json["spf_resolution"]; formatted from the same record as
        # train.log's [spf] block, so console and metadata cannot disagree.
        # The REALIZED half stays in report.json["realized_cadence"] (#289).
        "spf_resolution": spf_resolution.as_dict(),
        # Issue #595: how the run's IMAGE resolution was resolved -- native
        # size, requested divisor, the hard --streaming_auto_downscale_max_
        # pixels safety cap, and (nested under "dynamic", never flattened
        # into it) what the throughput-driven ResolutionController did.
        # Lands at report.json["resolution"]. Unrelated to "spf_resolution"
        # above, which is about steps-per-frame; the name collision is
        # historical -- #596 used "resolution" in the sense of "how a value
        # was resolved".
        "resolution": resolution_report_block(
            resolution_provenance, resolution_tier_log
        ),
        # Issue #289: the REALIZED cadence -- what the run's clocks actually
        # did, denominated in optimizer steps, plus the requested-vs-realized
        # reconciliation verdict. Lands at report.json["realized_cadence"].
        # Deliberately top-level, not nested under "geometry" like
        # mapping_clocks: it is a run-level fact about the experiment's
        # validity, and burying it is part of why #278's discrepancy had to
        # be reconstructed from a progress bar.
        "realized_cadence": _realized_cadence_block,
        # Issue #602: admission/cadence invariants, pending-queue
        # telemetry, and the four-field verdict a reviewer reads to answer
        # "did this run observe the full trajectory, admit all / a
        # budgeted subset / an explicit stress-all stream, accumulate
        # backlog, and actually deliver the claimed optimizer exposure?"
        # Embeds the realized_cadence block above verbatim rather than
        # re-deriving (and potentially contradicting) its numbers. Built by
        # a helper function, never an inline dict literal -- a nested
        # literal here is parsed by the #646 kwargs guard as several
        # unknown top-level kwargs.
        "admission_cadence": admission_cadence_report_block(
            admission_cadence,
            realized_cadence=_realized_cadence_block,
        ),
        "reference_map_path": getattr(args, "reference_map", None),
        "map_comparison_align": getattr(args, "map_comparison_align", True),
        "baseline_run": getattr(args, "baseline_run", None),
        # Issue #203 (lane P0-E of #202): the DECLARED experiment
        # validity class. Report-only -- the report additionally derives
        # what this run actually realized and records both. ``final``
        # matters: a mid-run checkpoint report legitimately grades
        # achieved=SMOKE_ONLY even for a run that ends full.
        "experiment_validity_class": getattr(
            args, "experiment_validity_class", None),
        "experiment_validity_is_final": bool(final),
        # Four-clocks telemetry (Wave 2 Task 8) -- a snapshot at report
        # time, using the same rolling-window aggregator that feeds the
        # mapping/clocks/* TensorBoard scalars during training. See
        # write_post_training_report's mapping_clocks docstring and
        # FourClocksSummary (coordinator.py) for field definitions.
        "mapping_clocks": dataclasses.asdict(coordinator.four_clocks.summary()),
        # Issue #703/#755: supervision-work accounting. Canonical metrics for
        # rasterizer invocations and supervised pixels during training, plus
        # derived per-step/per-view ratios. Recorded only if the clocks were
        # incremented (non-zero optimization_step_count); empty otherwise.
        # Used by #755's A/B to adjudicate engine promotion. Assembled by
        # utils/supervision_work.py rather than inline: a nested dict LITERAL
        # here would be read by tests/test_issue646_report_kwargs_signature.py
        # (which walks every dict literal in this builder) as seven unknown
        # top-level report kwargs -- see that helper's docstring.
        "supervision_work": supervision_work_report_block(
            clk_optimizer_step_id,
            clk_supervision_view_count,
            clk_rasterizer_call_count,
            clk_supervised_pixel_count,
            per_source_supervision_counts or None,
            wall_time_s=wall_time_s,
        ),
        # Why the frame source stopped (2026-08-06) -- see
        # eligsir/io/sources/ros2_live.py's stream_end_reason /
        # eligsir/io/sources/datasets/ros2_live.py's property wrapper.
        # None for any source type that doesn't set this (everything
        # except ros2://live bag replay today).
        "stream_end_reason": getattr(frame_source, "stream_end_reason", None),
        # Phase 2c (2026-08-06): non-empty iff ORB-SLAM3's Atlas
        # re-initialized mid-run (a new, independently-originated map --
        # see _check_tracking_reinit). CORRECTED 2026-08-06: the veto is
        # now PERMANENT (see MapReinitTracker's docstring), so every
        # entry here has recovered_at_frame == None -- there is no
        # "settled" state anymore. A run with events here is stronger
        # than "coverage gap": compute_trajectory_coverage() (see
        # utils/partition_eval.py) derives
        # report.json["trajectory_coverage"]["map_reinit_invalidated"]
        # (TOP-LEVEL, not under "geometry" -- issue #504) from
        # this list -- treat that flag exactly like eval_valid=False /
        # stream_end_reason=="idle_timeout_no_completion_signal"
        # (SUSPECTED_TRUNCATED): never quote this run's coverage or
        # PSNR as valid.
        "tracking_reinit_events": list(reinit_tracker.events),
        # W3-15/W3-19: BagReplayPolicy now has a real production
        # ingestion call site -- ROS2LiveStreamingFrameSource's pump
        # loop (eligsir/io/sources/datasets/ros2_live.py, W3-19,
        # 2026-08-10) routes every appended frame through it and
        # exposes `bag_replay_report` as a real, non-None dict for any
        # `ros2://live` (including bag-replay-via-live-source) run.
        # Still reads None for every other source type (TUM, ScanNet,
        # generic RGB-D, etc.) -- those never had a drop-prone bounded
        # producer buffer in front of them to begin with, so there is
        # nothing for this mechanism to guard there.
        "bag_replay_report": (
            getattr(frame_source, "bag_replay_report", None)
            or getattr(getattr(streaming_scene, "_frame_source", None), "bag_replay_report", None)
        ),
        # Issue #865, acceptance item 4: the final accept/reject counter
        # snapshot (accepted_frames, rgb_received, queued_frames, buffer
        # sizes, every reject_* counter) that ros2_live.py's
        # maybe_publish_diagnostics() already computed on every
        # diagnostics tick and previously only ever published to a
        # transient ROS2 DiagnosticArray topic. None for every source
        # type that doesn't set this (everything except ros2://live)
        # -- same None-until-real convention as bag_replay_report above.
        "ingest_counters": (
            getattr(frame_source, "ingest_counters", None)
            or getattr(getattr(streaming_scene, "_frame_source", None), "ingest_counters", None)
        ),
        # Issue #865, acceptance items 1-2: the bag's own real per-topic
        # message counts (read once, up front, by
        # human_launch_scripts/05_run_offline_rosbag.sh, since the bag
        # file is static) -- see load_bag_source_manifest()'s docstring.
        # None for every run that isn't a bag replay launched by that
        # script, or predates this mechanism.
        "bag_source_manifest": load_bag_source_manifest(args.model_path),
        # Issue #1114: the C++ sync node's own per-frame attribution ledger,
        # parsed from <model_path>/rgbd_sync_node.log (which the launch
        # scripts already capture). Lets compute_bag_source_gap tell an
        # explained matcher drop apart from a silently lost frame instead of
        # refusing every clean replay. None whenever no sync node ran or the
        # log is absent/unparseable -- see load_sync_pair_ledger()'s
        # docstring; None means "cannot verify", never "attributed zero".
        "sync_pair_ledger": load_sync_pair_ledger(args.model_path),
        # Issue #259: record which commit/branch/dirty-state this
        # process actually ran from, captured once at __main__ startup
        # in train.py (get_git_metadata()) before dispatch into this
        # streaming loop -- see git_provenance_from_args()'s docstring.
        "git_provenance": git_provenance_from_args(args),
        # Audit §14: which determinism mode (utils.general_utils.safe_state's
        # strict_deterministic) this process actually ran under -- resolved
        # once from --strict_deterministic/--no-strict_deterministic at
        # train.py __main__ startup, before dispatch into this streaming
        # loop. Same passive-record mechanism as git_provenance above.
        "determinism": report_with_global_digest(determinism_from_args(args)),
        # Issue #1302: operator-declared attestation status for this run's
        # RGB-D source (e.g. "legacy_unattested" for the kitchen1/floor2
        # historical bags, which predate eligsir-rgbd-sync's
        # capture-provenance gate). Same passive-record mechanism as
        # git_provenance/determinism above; caps evidence_class at
        # UNMANAGED downstream in utils/run_summary.py -- see
        # utils/source_attestation.py.
        "source_attestation": source_attestation_from_args(args),
        # Issue #511: WHY this run exists / which tracked task owns it.
        # Pure accessor over the object resolved once at train.py
        # startup -- the same one written to the run root as
        # experiment_origin.json, so the two cannot diverge.
        "experiment_origin": _experiment_origin_from_args_local(args),
        # Issue #168 (lane P0-B of #202): the resolved value of every
        # represented run-mode axis, so this run's own report.json answers
        # "what execution semantics actually ran" instead of leaving a
        # reader to re-derive it from raw flags. Same mechanism as
        # git_provenance above -- a record, never a gate.
        "execution_semantics": execution_semantics_from_args(args),
        # Issue #524 acceptance criterion 3: declared run intent +
        # resolved profile stack + effective config, built once at
        # train.py __main__ startup (run_intent_report_from_args()) --
        # pure accessor, same mechanism as experiment_origin/
        # execution_semantics above. Present on every streaming report
        # (pre-training, mid-run, and final).
        "run_intent": getattr(args, "run_intent_report", None),
        # Issue #705: exact declaration context resolved by the real
        # train.py/ROS2 entrypoint. Present on every streaming report.
        "experiment_context": getattr(args, "experiment_context", None),
        # Issue #1323: presentation-only SSAA + sharpening for trajectory.mp4 and
        # train_contact_sheet.png. See presentation_ssaa_kwargs_from_args's docstring --
        # never touches anything feeding a scored metric.
        **presentation_ssaa_kwargs_from_args(args),
        # Issue #397 follow-up: presentation-only motion-smoothness interpolation for
        # trajectory.mp4's real-timestamped pacing path. See
        # presentation_trajectory_interpolate_kwargs_from_args's docstring -- never
        # touches _infer_realtime_fps or any scored metric.
        **presentation_trajectory_interpolate_kwargs_from_args(args),
    }
