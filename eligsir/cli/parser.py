"""Shared command-line parser construction for trainer entry points."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import sys
from typing import Sequence

from arguments import ModelParams, OptimizationParams, OutputStagingParams, PipelineParams, StreamingParams
from eligsir.mapping.hot_tier_cache import MAX_RETAINED_TIERS_PER_FRAME
from eligsir.population_control_deprecation import LEGACY_POPULATION_CONTROL_CLI_GUIDANCE
from eligsir.population_control_deprecation import reject_raw_live_population_conflicts
from eligsir.utils.run_intent import KNOWN_RUN_INTENTS


@dataclass
class TrainingParserBundle:
    """Bundled command-line parser and parameter group extractors.

    Attributes:
        parser: The ArgumentParser for training-related arguments.
        model_params: ModelParams argument group extractor.
        optimization_params: OptimizationParams argument group extractor.
        pipeline_params: PipelineParams argument group extractor.
    """

    parser: ArgumentParser
    model_params: ModelParams
    optimization_params: OptimizationParams
    pipeline_params: PipelineParams


def build_training_parser(description: str = "EliGSiR training") -> TrainingParserBundle:
    """Build the canonical trainer parser and parameter extractors."""

    parser = ArgumentParser(description=description)
    parser.epilog = LEGACY_POPULATION_CONTROL_CLI_GUIDANCE
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    StreamingParams(parser)
    OutputStagingParams(parser)

    # `StreamingParams.streaming_live_admitted_payload_max_frames` defaults to
    # None (its explicit-flag-vs-policy-default sentinel, consumed by
    # eligsir.runtime.config_boundary.resolve_admitted_payload_bounds).
    # ParamGroup.__init__ infers each flag's argparse `type=` from
    # `type(default)`, so a None default registers `type=NoneType`, and
    # actually passing `--streaming_live_admitted_payload_max_frames N` on
    # the CLI crashes with "invalid NoneType value" instead of parsing an
    # int. Patch the already-built action's type in place rather than
    # re-registering the flag (argparse rejects a duplicate option string).
    parser._option_string_actions[
        "--streaming_live_admitted_payload_max_frames"
    ].type = int

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--execution_trainer",
        choices=("offline_reconstruction", "causal_mapping"),
        default="offline_reconstruction",
        help="Canonical execution root. Use causal_mapping for MappingApplication runs.",
    )
    parser.add_argument(
        "--run_profile",
        "--run-profile",
        type=str,
        default=None,
        help=(
            "Named EliGSiR run profile or config path under configs/profiles "
            "(for example live/orbbec_ros2_rviz)."
        ),
    )
    parser.add_argument(
        "--run_intent",
        "--run-intent",
        type=str,
        default=None,
        choices=list(KNOWN_RUN_INTENTS),
        help=(
            "Issue #524 (deferred half): declare which of the three canonical "
            "run intents this run is. Resolves to the same "
            "`--run_profile run_intent/<name>` profile that has been "
            "selectable since PR #550 -- this flag adds no new behavior, only "
            "a first-class name for it, and NO argparse default changes as a "
            "result of setting it (every run_intent/*.yml profile is "
            "test-enforced reporting/telemetry-only, see "
            "tests/test_run_intent_profiles.py). Conflicts loudly with an "
            "explicit --run_profile that names a DIFFERENT profile; omitting "
            "this flag records report.json['run_intent']['declared'] == "
            "'UNDECLARED', not a silent pass. See utils/run_intent.py."
        ),
    )
    parser.add_argument(
        "--print_effective_config",
        "--print-effective-config",
        action="store_true",
        default=False,
        help=(
            "Issue #524 (remaining item 3): print the fully resolved effective "
            "configuration including the run intent, profile stack, and per-option "
            "provenance (where each option came from: default / profile / CLI), then "
            "exit 0 without starting a run. Output is deterministic YAML. Does not "
            "import torch or create a model_path."
        ),
    )
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument(
        "--reference_map",
        type=str,
        default=None,
        help=(
            "Path to an independent, non-neural point-cloud-only reference map "
            "(.ply mesh or point cloud, e.g. an embed_slam TSDF reconstruction) "
            "to compare the final Gaussian checkpoint against in the post-training "
            "report (utils/map_comparison.py). Strictly opt-in -- omit for a normal "
            "run. See CLAUDE.md's 'Gaussian vs. point-cloud-only map comparison' "
            "section for the cross-repo workflow that produces this file, apples-to"
            "-apples only when both maps share the same pseudo-GT trajectory."
        ),
    )
    parser.add_argument(
        "--no_map_comparison_align",
        dest="map_comparison_align",
        action="store_false",
        default=True,
        help=(
            "Skip ICP-aligning --reference_map onto the Gaussian map before "
            "comparing. Only valid when both maps are already known to share the "
            "same pose source (e.g. both built from the same /gt/colmap_pose bag) "
            "-- otherwise the comparison numbers are meaningless."
        ),
    )
    parser.add_argument(
        "--baseline_run",
        type=str,
        default=None,
        help=(
            "Path to a previously-completed run's model_path (its report.json and "
            "psnr_history.jsonl must exist) to automatically compare this run against "
            "in the post-training report (utils.comparison_report.compare_against_baseline). "
            "Strictly opt-in -- omit for a normal run; behavior is unchanged when unset. "
            "Per CLAUDE.md's 'Compare PSNR curves, not just the final number' run contract, "
            "this compares the FULL psnr_history curve (not just the final PSNR), and only "
            "trusts the comparison when both runs' eval_metadata.holdout_hash match."
        ),
    )
    parser.add_argument(
        "--allow_invalid_execution_semantics",
        action="store_true",
        default=False,
        help=(
            "Issue #168 (DAG lane P0-B of #202): downgrade the resolved "
            "execution-semantics validator's hard ERRORS to warnings and run "
            "anyway. The validator only errors on combinations that are "
            "PROVABLY impossible (e.g. the offline trainer with a ros2:// "
            "source, which has no ingestion path at all), so no working "
            "historical spec can trip it -- verified by a test that resolves "
            "every specs/*.json and every configs/profiles/** leaf and asserts "
            "zero errors. This escape hatch exists only so a live ablator "
            "queue cannot be bricked by a validator false positive; it never "
            "suppresses the report.json['execution_semantics'] record."
        ),
    )
    parser.add_argument(
        "--experiment_validity_class",
        type=str,
        default=None,
        choices=[
            "SMOKE_ONLY",
            "DISTRIBUTED_TRAJECTORY_PROXY",
            "FULL_CONTINUAL_VALIDATION",
        ],
        help=(
            "Issue #203 (DAG lane P0-E of #202): DECLARE what kind of evidence this "
            "run is intended to produce. Purely a report-side declaration -- it "
            "changes no training behavior, no defaults, no admission and no loss. "
            "The report ALSO derives the class this run actually realized (from its "
            "own realized source-frame coverage/span/region occupancy) and records "
            "both, plus a mismatch flag, in report.json['experiment_validity'] -- so "
            "a spec claiming FULL_CONTINUAL_VALIDATION that realized a 5%% prefix says "
            "so loudly in its own report. Omitting the flag records declared="
            "'UNDECLARED', which is the LEAST permissive value, never a pass. "
            "SMOKE_ONLY runs get negative_conclusions_permitted=False: they may "
            "conclude only that a code path works/does not or a knob bound/did not, "
            "never 'mechanism X has no effect'."
        ),
    )
    # --- experiment origin / task provenance (issue #511, lane P0-A of #202) --
    # Declares WHY this run exists and which tracked task owns it. Purely
    # report-side: changes no training behavior, no defaults, no loss.
    # Resolved exactly once at train.py startup into args.experiment_origin;
    # written to <model_path>/experiment_origin.json AND embedded as
    # report.json["experiment_origin"] from that same object.
    parser.add_argument(
        "--experiment_origin_issue",
        type=str,
        default=None,
        help=(
            "GitHub issue this run was executed for (494, '#494', or the full "
            "issues URL). Malformed references are REJECTED at startup, never "
            "silently ignored. The output DIRECTORY NAME is never used as task "
            "identity."
        ),
    )
    parser.add_argument(
        "--experiment_origin_pr",
        type=str,
        default=None,
        help=(
            "GitHub pull request this run was executed for (form as per "
            "--experiment_origin_issue). When an issue is also given, the issue "
            "is treated as the owning task."
        ),
    )
    parser.add_argument(
        "--experiment_origin_repo",
        type=str,
        default=None,
        help=(
            "owner/name of the repository owning the issue/PR. Defaults to "
            "eligsir-project/eligsir."
        ),
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help=(
            "Identifier of the experiment/sweep this run belongs to "
            "(e.g. load_capacity_E2). Shared across all arms of one sweep."
        ),
    )
    parser.add_argument(
        "--experiment_arm",
        type=str,
        default=None,
        help="Which arm of --experiment_id this run is (e.g. high_admission).",
    )
    parser.add_argument(
        "--experiment_purpose",
        type=str,
        default=None,
        help=(
            "One sentence: what claim this run was intended to test. Required "
            "for --experiment_origin_ad_hoc."
        ),
    )
    parser.add_argument(
        "--experiment_origin_ad_hoc",
        action="store_true",
        default=False,
        help=(
            "Declare this run as untracked exploratory work (task_kind "
            "'ad_hoc'), requiring --experiment_purpose. Preferred over "
            "fabricating an issue reference. Contradicting it with an "
            "issue/PR reference is an error, not a silent preference."
        ),
    )
    parser.add_argument(
        "--experiment_origin_file",
        type=str,
        default=None,
        help=(
            "Path to a small JSON spec file declaring the origin (fields at the "
            "top level or nested under an 'experiment_origin' key). Lower "
            "precedence than the CLI flags above, field by field; the "
            f"ELIGSIR_EXPERIMENT_ORIGIN env var is the lowest layer."
        ),
    )
    parser.add_argument(
        "--streaming_refinement_eval_interval",
        type=int,
        default=0,
        help=(
            "For online_mode runs: also fire a full mid-training test-PSNR "
            "report every N iterations (0 = off, only the fixed --test_iterations "
            "list fires). --test_iterations is a fixed list unsuited to online_mode's "
            "unbounded rolling-idle-stop termination (the final iteration isn't known "
            "ahead of time) -- this gives a real PSNR-vs-iteration curve spanning "
            "the mapping phase and the post-coverage refinement tail, to find the "
            "refinement-length sweet spot."
        ),
    )
    parser.add_argument(
        "--streaming_psnr_chart_interval",
        type=int,
        default=1000,
        help=(
            "Hard constraint (Run Contracts): always generate a test-holdout "
            "PSNR-over-time chart (psnr_over_time.svg), even for online_mode "
            "runs -- unlike --streaming_refinement_eval_interval's full report "
            "(contact sheets/mp4/geometry), this only renders test cameras and "
            "computes PSNR: cheap enough to run every N iterations synchronously "
            "on the training thread (utils/psnr_history.py). 0 disables it "
            "(smoke tests only -- never for evaluation/ablation runs)."
        ),
    )
    parser.add_argument(
        "--streaming_early_stop_patience_ticks",
        type=int,
        default=0,
        help=(
            "Stop training early once the last N consecutive "
            "--streaming_psnr_chart_interval ticks all fall within "
            "--streaming_early_stop_min_delta_db of each other (flat test-PSNR "
            "curve). 0 (default) disables early stopping entirely -- existing "
            "runs are unaffected unless this is explicitly set. Checked at the "
            "same cadence as the psnr_over_time.svg chart "
            "(utils/psnr_history.check_psnr_plateau); a run that stops early "
            "still falls through to the normal final-save/report path, same as "
            "hitting --iterations or --streaming_max_iterations."
        ),
    )
    parser.add_argument(
        "--streaming_early_stop_min_delta_db",
        type=float,
        default=0.1,
        help=(
            "Max-minus-min test-PSNR range (dB) across the trailing "
            "--streaming_early_stop_patience_ticks window that counts as "
            "'flat' for early stopping. Only consulted when "
            "--streaming_early_stop_patience_ticks > 0."
        ),
    )
    parser.add_argument(
        "--w3_02_evidence_manifest",
        type=str,
        default="",
        help="Opt-in Level-D W3-02 manifest JSON to bind runtime evidence to.",
    )
    parser.add_argument(
        "--w3_02_evidence_arm",
        choices=("baseline", "treatment", "negative_control"),
        default=None,
        help="W3-02 implementation arm associated with --w3_02_evidence_manifest.",
    )
    parser.add_argument(
        "--w3_02_mutation_trace",
        type=str,
        default="",
        help="Frozen mutation-trace artifact to hash into W3-02 runtime evidence.",
    )
    parser.add_argument("--w3_02_git_commit", type=str, default="", help="Host-injected verified checkout SHA for W3-02 evidence.")
    parser.add_argument("--w3_02_clean_checkout", action="store_true", help="Host attests checkout was clean before container launch.")
    for _name, _help in (
        ("config_artifact", "frozen resolved config"),
        ("source_artifact", "frozen source artifact"),
        ("pose_artifact", "frozen pose artifact"),
        ("keyframe_artifact", "frozen keyframe artifact"),
        ("holdout_artifact", "frozen holdout artifact"),
    ):
        parser.add_argument(f"--w3_02_{_name}", type=str, default="", help=f"W3-02 {_help} path to hash at runtime.")
    parser.add_argument(
        "--w3_02_mutation_trace_mode",
        choices=("off", "record", "verify"),
        default="off",
        help=(
            "Level-D mutation-trace freezing for the W3-02 engine A/B "
            "(eligsir/optimization/gaussian/mutation_trace.py). 'record' "
            "writes a per-iteration topology event log (topology_version, N, "
            "op counts, an index hash) at the shared, flag-independent "
            "mutation call sites (depth-insertion flush, MCMC "
            "step_post_backward). 'verify' loads a previously recorded trace "
            "and asserts the treatment run's events match it exactly at every "
            "iteration, raising immediately on the first divergence -- this "
            "is what lets 'mutation_trace_applied' become honestly true "
            "instead of the permanently-false placeholder the salvage "
            "harness shipped with."
        ),
    )
    parser.add_argument(
        "--w3_02_mutation_trace_events",
        type=str,
        default="",
        help="Path to read/write the mutation-trace event JSONL for --w3_02_mutation_trace_mode.",
    )
    parser.add_argument(
        "--streaming_seed_fade_warmup_iters",
        type=int,
        default=2000,
        help=(
            "Ignore seed-view PSNR samples before this iteration when arming "
            "SeedFadeMonitor (utils/seed_fade_monitor.py). An early, anomalously "
            "easy bootstrap-frame render can set an unrealistic running-max "
            "ceiling that the scene's real steady-state quality never again "
            "reaches, reading as a permanent fade and suppressing the "
            "anti-overgrowth losses (lambda_eff_count/lambda_opacity_entropy) "
            "for most of training (observed 2026-07-26 on kitchen1: false "
            "trigger at iter 500, suppressed until iter ~28000 of 60000). 0 "
            "restores the old unconditional behavior."
        ),
    )
    parser.add_argument(
        "--streaming_seed_fade_disabled",
        action="store_true",
        default=False,
        help=(
            "Fully disable SeedFadeMonitor's effect on training: the monitor "
            "still runs and reports detection/fade stats for diagnostics, but "
            "its suppression_scale() output is never applied -- "
            "lambda_eff_count/lambda_opacity_entropy always run at their "
            "configured (unsuppressed) weight. For isolating whether the "
            "controller itself (as opposed to the false-trigger bug fixed by "
            "--streaming_seed_fade_warmup_iters) is contributing to a run's "
            "quality trajectory."
        ),
    )
    parser.add_argument(
        "--streaming_seed_fade_observe_only",
        action="store_true",
        default=False,
        help=(
            "Observation-only seed-fade detection (issue #282): identical in "
            "effect to --streaming_seed_fade_disabled (the two are OR'd), but "
            "named for the causal-experiment use case rather than the "
            "debugging one. The monitor observes, records and reports the "
            "full fade trajectory, but never modulates lambda_eff_count / "
            "lambda_opacity_entropy -- so a run's fade data is not "
            "contaminated by the mitigation the detector itself triggers. "
            "Required for the #281/#346 seed-fade causal program; default "
            "off, so existing behavior is unchanged unless asked for. "
            "report.json['seed_fade']['observe_only'] records which mode ran."
        ),
    )
    parser.add_argument(
        "--streaming_auto_downscale_max_pixels",
        type=int,
        default=1280 * 960,
        help=(
            "Hard, deterministic resolution-scaling gate (2026-07-27): if the "
            "source's native resolution (read from intrinsics.json before any "
            "frame is loaded) exceeds this many pixels, automatically set "
            "--streaming_resolution to the smallest divisor that brings it "
            "under the threshold. Unlike ResolutionController's throughput-based "
            "tiering (which can be disabled via --streaming_multires_it_v_floor/"
            "map_fps_floor 0 -- necessary since it otherwise causes an unbounded "
            "quality-loss bug -- but that removes its memory safety valve), this "
            "check is unconditional and can't be starved by a lucky-fast start. "
            "Only applies when --streaming_resolution is left at its default (1). "
            "0 disables the gate entirely."
        ),
    )
    parser.add_argument(
        "--streaming_auto_downscale_disabled",
        action="store_true",
        default=False,
        help="Fully disable the hard resolution-scaling gate regardless of --streaming_auto_downscale_max_pixels.",
    )
    parser.add_argument(
        "--streaming_full_res_frame_indices",
        type=str,
        default="",
        help=(
            "Comma-separated source frame indices to load at native resolution, "
            "overriding --streaming_resolution/the auto-downscale gate for just "
            "those frames. Only honored by RGBDSequenceFrameSource (frames.jsonl "
            "+ intrinsics.json sources -- TUM/fr3, Replica, ScanNet++). Intended "
            "for a two-pass workflow: run once, use "
            "scripts/find_bad_psnr_views.py --indices-only on the resulting "
            "report.json to find low-PSNR held-out views, then re-run with those "
            "indices here so only the frames that need it pay the full-resolution "
            "memory cost."
        ),
    )
    parser.add_argument(
        "--streaming_vram_brake_fraction",
        type=float,
        default=0.0,
        help=(
            "Suppress MCMC growth whenever torch.cuda.memory_reserved() exceeds "
            "this fraction of total device memory (e.g. 0.9), re-enabling once "
            "usage drops back below it. Reacts to actual measured allocation "
            "rather than a static Gaussian-count ceiling (cap_max), so it still "
            "helps when the real memory bottleneck is per-frame render/backward "
            "cost (high native resolution) or GTT/shared-VRAM pressure rather "
            "than N. 0 (default) disables the brake."
        ),
    )
    parser.add_argument(
        "--streaming_vram_brake_check_interval",
        type=int,
        default=1,
        help="Check the VRAM brake every N iterations (default 1; the check is a cheap driver counter read, not a sync).",
    )
    parser.add_argument(
        "--streaming_vram_brake_stop_fraction",
        type=float,
        default=0.0,
        help=(
            "Second, higher threshold above --streaming_vram_brake_fraction: "
            "gracefully STOP training (break out of the main loop and proceed "
            "to the normal end-of-training save + report, same path a clean "
            "opt.iterations completion takes) once device memory usage "
            "reaches this fraction, instead of only pausing growth. For "
            "cases where memory keeps climbing from something other than new "
            "growth (e.g. deliberately testing native resolution with the "
            "auto-downscale gate disabled) -- pausing growth alone can't help "
            "there, and letting it OOM uncaught loses the run entirely. "
            "0 (default) disables this tier; only meaningful when > "
            "--streaming_vram_brake_fraction."
        ),
    )
    parser.add_argument(
        "--streaming_local_quality_cells",
        action="store_true",
        default=False,
        help=(
            "Issue #358 (Level A only): record a per-microcell test-PSNR "
            "history to <model_path>/local_quality_history.jsonl at every "
            "--streaming_psnr_chart_interval tick. Holdout pixels are mapped "
            "to immutable world-space cells via FIXED sensor/GT depth and the "
            "FIXED holdout pose (never the current map's rendered depth), so "
            "partition membership cannot depend on the reconstruction being "
            "measured. Pure telemetry: reuses the render pass psnr_history "
            "already does (no second render), reduces on-device, and syncs "
            "only a compact per-cell vector once per tick. Off by default; "
            "changes no training behavior when on."
        ),
    )
    parser.add_argument(
        "--streaming_local_quality_cell_size",
        type=float,
        default=0.5,
        help=(
            "Microcell edge length in metres for --streaming_local_quality_"
            "cells. The design doc names 0.5-1.0m as the indoor candidate "
            "range and explicitly leaves the exact scale to be tested; 0.5 "
            "is the conservative (finer) end. Cell identity is deterministic "
            "in the fixed world frame, so this value must not be changed "
            "mid-run -- a different cell size is a different partition."
        ),
    )
    parser.add_argument(
        "--streaming_partition_focus_signal_source",
        choices=("training_derived", "control_probe", "scored_holdout_oracle"),
        default="training_derived",
        help=(
            "Observation role allowed to drive partition-focus actuation. "
            "Canonical causal partition control requires control_probe, a "
            "deterministic trajectory-spanning training-camera cohort refreshed "
            "at every control barrier and disjoint from scored holdout. "
            "training_derived (the shared legacy default) and "
            "scored_holdout_oracle are legacy/research only; the oracle also "
            "stamps the run ORACLE_CONTROL/ORACLE_INTERVENTION."
        ),
    )
    parser.add_argument(
        "--streaming_partition_focus_control_probe_n",
        type=int,
        default=32,
        help=(
            "Maximum deterministic CONTROL_PROBE camera count when "
            "--streaming_partition_focus_signal_source=control_probe. The "
            "evenly distributed identity set is refreshed at each barrier; "
            "canonical causal mapping requires [1, 128]. The historical "
            "0/full-pool semantics are legacy/research only."
        ),
    )
    parser.add_argument(
        "--streaming_partition_focus_max_promote",
        type=int,
        default=20,
        help=(
            "Max number of cached (cold/icecold tier) frames to promote into "
            "train_cameras in a single psnr-tick, across all bad partitions "
            "combined. An uncapped promotion was observed live promoting "
            "100+ frames in one tick, diluting train_cameras' composition "
            "far more aggressively than intended (and contributing to a "
            "live OOM -- see StreamingScene._enforce_hot_window). Demotion "
            "of good-partition cameras is capped to match this tick's "
            "actual promoted count, so the active buffer stays roughly "
            "size-neutral instead of growing every tick. 0 disables "
            "promotion (and therefore balanced demotion) entirely."
        ),
    )
    parser.add_argument(
        "--streaming_partition_focus_outlier_k",
        type=float,
        default=1.5,
        help="Outlier sensitivity for partition-focus detection (median - k*MAD); lower flags more partitions.",
    )
    parser.add_argument(
        "--streaming_inactive_cache_size",
        type=int,
        default=2000,
        help=(
            "Max number of rejected-at-admission frames' lightweight metadata "
            "(StreamingRGBDFrame, not decoded images) to retain in the cold "
            "cache tier for possible later promotion into train_cameras via "
            "partition-focus (StreamingScene.promote_from_cache). Bounded LRU "
            "(oldest evicted first), on top of the --streaming_cold_window "
            "scoping. 0 disables the cold tier entirely -- partition-focus "
            "sampling then only reweights among already-admitted cameras "
            "(and whatever the icecold tier holds)."
        ),
    )
    parser.add_argument(
        "--streaming_cold_window",
        type=int,
        default=2,
        help=(
            "A rejected-at-admission frame only enters the cold cache tier "
            "when its source index is within this many steps of the most "
            "recently admitted keyframe -- i.e. the cold tier holds frames "
            "'surrounding' a selected keyframe, not every rejected frame in "
            "the trajectory (which would be most of it under a gated "
            "admission mode). Negative disables the window check (cache "
            "every rejected frame, subject only to "
            "--streaming_inactive_cache_size's LRU bound)."
        ),
    )
    parser.add_argument(
        "--streaming_icecold_cache_size",
        type=int,
        default=20000,
        help=(
            "Max number of frames' lightweight metadata (StreamingRGBDFrame, "
            "not decoded images -- CPU RAM, not GPU) to retain in the big, "
            "long-term icecold cache tier for possible later re-promotion "
            "into train_cameras. Fed by (a) frames evicted from the smaller "
            "--streaming_inactive_cache_size cold tier's LRU, and (b) frames "
            "demoted out of train_cameras via StreamingScene.demote_"
            "partitions (partitions doing unusually well relative to the "
            "rest of the scene). Bounded LRU (oldest evicted first). 0 "
            "disables the tier entirely -- cold-tier overflow is discarded "
            "outright and demote_partitions becomes a no-op. NOTE: 'not "
            "decoded images' only holds for file-backed sources (RGBDSequence/"
            "TUM/ScanNet-path). Frames from a source with no seekable backing "
            "store (ros2_live, and ScanNet .sens's in-memory path) already "
            "carry decoded pixels on the StreamingRGBDFrame itself (_rgb_np/"
            "_depth_np, or ScanNet's _rgb_bytes/_depth_bytes) -- for those, "
            "this bound is instead --streaming_live_icecold_cache_size."
        ),
    )
    parser.add_argument(
        "--streaming_live_inactive_cache_size",
        type=int,
        default=150,
        help=(
            "Same role as --streaming_inactive_cache_size (cold-tier LRU "
            "bound), applied instead of it when a frame already carries "
            "decoded pixel data (_rgb_np/_depth_np or _rgb_bytes/"
            "_depth_bytes populated) -- ros2_live and ScanNet .sens sources, "
            "which have no seekable file to re-decode from later so the "
            "cached frame IS the decoded image, unlike RGBDSequence/TUM's "
            "near-free path-only metadata. Default sized for a full-res "
            "1280x720 RGB-D pair (~4.5MB/frame raw) -> ~150 frames is "
            "~675MB, versus the 2000-frame file-backed default which would "
            "be ~9GB of retained pixels at the same resolution."
        ),
    )
    parser.add_argument(
        "--streaming_live_icecold_cache_size",
        type=int,
        default=1500,
        help=(
            "Same role as --streaming_icecold_cache_size, applied instead "
            "of it for pixel-carrying frames (see "
            "--streaming_live_inactive_cache_size). Default ~1500 frames "
            "is ~6.75GB of retained pixels at 1280x720 -- keep this well "
            "under available system RAM when tuning for a live run; the "
            "icecold tier is CPU-RAM-resident regardless of GPU."
        ),
    )
    parser.add_argument(
        "--streaming_resolution_cache_max_frames",
        type=int,
        default=400,
        help=(
            "Bounded LRU (by insertion order, oldest evicted first) on how "
            "many admitted train frames' GPU-resident multires resolution_"
            "cache (eligsir.mapping.resolution_cache.build_resolution_"
            "cache) FrameObservationDB retains at once. Fixes a real 2026-"
            "08-04 ScanNet++ VRAM blowup (46.6GB -> 80.8GB in ~13min while "
            "Gaussian count N grew only modestly): the 'high' tier is built "
            "at native source resolution (divisor=1) unconditionally for "
            "EVERY is_train frame, regardless of the ResolutionController's "
            "active tier or --streaming_auto_downscale_max_pixels, and "
            "FrameRecord.resolution_cache was never evicted (FrameRecords "
            "live forever in FrameObservationDB._frames for pose/visibility "
            "history). ~20MB/frame at ScanNet++'s 1920x1440 native res with "
            "zero bound explains the growth directly (~1300 admitted "
            "keyframes by iter 4500). Eviction only clears the cached image "
            "tensors, not the record itself -- a re-sampled evicted frame "
            "falls back to bind_resolution_tier()'s normal per-access load "
            "path, it does not error. 0 = unbounded (legacy/test behavior)."
        ),
    )
    parser.add_argument(
        "--streaming_hot_tier_max_tiers_per_frame",
        type=int,
        default=MAX_RETAINED_TIERS_PER_FRAME,
        help=(
            "How many distinct resolution tiers (of native/half/quarter/"
            "eighth, see eligsir.contracts.control.ResolutionTier) one "
            "HOT frame may retain concurrently in "
            "eligsir.mapping.hot_tier_cache.HotTierCache before its "
            "least-valuable retained tier is evicted (issue #279 follow-up: "
            "this was previously a hardcoded module constant with no CLI "
            "path at all -- see hot_tier_cache.py's own module docstring "
            "for the value-based eviction formula). A tier switch back to a "
            "just-evicted tier for the same still-HOT frame pays the "
            "'CPU-resident tier miss' cost (5-12ms at 640x480, 36-102ms at "
            "1920x1440, PR #1252) even though the frame's native pixels "
            "never left GPU/CPU residency -- raising this toward the total "
            "tier count (4) removes that class of miss entirely for a "
            "resolution scheduler that revisits more than "
            f"{MAX_RETAINED_TIERS_PER_FRAME} distinct tiers per frame, at "
            "the cost of retaining more bytes per HOT frame (see the "
            "companion --streaming_hot_tier_max_bytes budget). Default "
            f"{MAX_RETAINED_TIERS_PER_FRAME} matches the pre-existing "
            "hardcoded behavior -- this flag changes nothing unless set. "
            "NOTE (real fr3 A/B, causal_replay/tum_rgbd, 2026-08-31): raising "
            "this to 4 alone measured ZERO change in camera_materialize_cache_"
            "hits/misses or m_tick_camera_materialize ms/tick on that recipe "
            "-- its admission's misses were dominated by frame-level HOT-set "
            "eviction (see --streaming_keyframe_window/--streaming_resolution_"
            "cache_max_frames, which DID measurably help: +35%% hit rate, "
            "-22%% ms/tick), not per-tier oscillation within an already-HOT "
            "frame. Keep for scenes/schedulers that do revisit >2 tiers per "
            "frame; do not expect it alone to fix a low hit rate."
        ),
    )
    parser.add_argument(
        "--streaming_hot_tier_max_bytes",
        type=int,
        default=None,
        help=(
            "Byte ceiling for multi-tier HOT supervision residency "
            "(eligsir.mapping.hot_tier_cache.HotTierCache), companion "
            "to --streaming_hot_tier_max_tiers_per_frame. When "
            "--streaming_payload_budget_policy is resolution_scaled (default), "
            "None (default) derives the value from device memory (20%% of total); "
            "legacy policy uses 0 = unbounded. Explicit values override the policy. "
            "Frame-count bounding is NOT a memory bound: the same 400-frame cap "
            "holds 1.83 GiB at TUM resolution and 16.5 GiB at ScanNet++ resolution "
            "(PR #1252), so raising --streaming_hot_tier_max_tiers_per_frame on a "
            "large-resolution scene without also setting this can grow VRAM usage unexpectedly."
        ),
    )
    parser.add_argument(
        "--streaming_payload_budget_policy",
        type=str,
        default="resolution_scaled",
        choices=("legacy", "resolution_scaled", "residency"),
        help=(
            "Payload budget policy for admitted frames and hot-tier bytes. "
            "legacy: use profile defaults (512 frames, 0 bytes = unbounded hot tier). "
            "resolution_scaled (default): derive from source native resolution "
            "and device memory (20%% of total VRAM). "
            "residency: dynamic RAM/VRAM/GTT-aware budget resolver. "
            "Explicit flags always override the policy."
        ),
    )
    parser.add_argument(
        "--causal_decode_workers_policy",
        type=str,
        default="fixed",
        choices=("fixed", "cpu_scaled"),
        help=(
            "Sizing policy for the ThreadPoolExecutor that decodes source RGB-D "
            "frames ahead of the pacing clock (see --causal_decode_workers). "
            "fixed (default): use --causal_decode_workers unchanged, "
            "byte-identical to pre-existing behavior. "
            "cpu_scaled: derive the worker count from the source's native "
            "resolution and available CPU cores (see "
            "eligsir.runtime.payload_budget.resolve_decode_workers), "
            "floored at --causal_decode_workers and capped at half the CPU "
            "core count (minimum 2). Addresses the starvation documented at "
            "arguments/__init__.py's causal_decode_workers default: on "
            "high-resolution sources (e.g. ScanNet++), a fixed 2-thread "
            "decode pool can bottleneck the mapper."
        ),
    )
    parser.add_argument(
        "--causal_thread_pinning_policy",
        type=str,
        default="unbound",
        choices=("unbound", "pinned"),
        help=(
            "Whether to pin cv2's internal thread pool and the process's BLAS "
            "thread-pool env vars (OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_NUM_THREADS) "
            "to 1, once, at causal_mapping startup (see "
            "eligsir.runtime.thread_pinning.pin_decode_threads). "
            "unbound (default): byte-identical to pre-existing behavior -- "
            "cv2.getNumThreads() stays at its host default (often all cores). "
            "pinned: each --causal_decode_workers thread does its own "
            "single-threaded decode/resize instead of also fanning out across "
            "every core, addressing the CPU oversubscription measured on "
            "scn5_decode8 (GPU util 23.4%% mean, CPU process 1012%% mean/1886%% "
            "peak of 3200%% possible, views/s unchanged 60.3 vs 61.8 at "
            "decode_workers=2 -- see docs/superpowers/specs/2026-09-08-"
            "residency-budget-design.md, 'Addendum 2'). Default stays "
            "'unbound' until a GPU A/B proves 'pinned' does not regress "
            "views/s on any existing recipe."
        ),
    )
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument(
        "--early_output_iterations",
        nargs="+",
        type=int,
        default=[500, 1000],
        help="Always save point-cloud outputs at these early iterations when they fit in the run.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable tqdm and print concise 500-iteration progress summaries instead.",
    )
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=2000,
        help="Save a checkpoint every N iterations (independent of --checkpoint_iterations).",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=2000,
        help="Save a PLY point cloud every N iterations (independent of --save_iterations).",
    )
    parser.add_argument(
        "--sh_degree_schedule",
        nargs="+",
        type=int,
        default=[1000, 2000, 3000],
        help="Iterations at which to increase SH degree (default: 1000 2000 3000).",
    )
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Path to save PyTorch profiler trace (e.g. /tmp/profile.json). "
            "Captures ~100 iters starting at iter 50 to catch the 100-iter stutter."
        ),
    )
    parser.add_argument(
        "--profile_wait",
        type=int,
        default=1,
        help=(
            "W3-14B: iterations to wait (uninstrumented) before the "
            "--profile capture window opens in the streaming loop -- lets "
            "a profiled window be positioned at a specific steady-state "
            "point (e.g. late in a run) rather than always immediately "
            "after --start_checkpoint. Only used by train_streaming.py; "
            "the legacy train.py loop keeps its own hardcoded wait=1."
        ),
    )
    parser.add_argument(
        "--profile_active",
        type=int,
        default=50,
        help="W3-14B: number of iterations captured in the --profile active window (streaming loop only).",
    )
    parser.add_argument(
        "--no_empty_cache",
        action="store_true",
        default=False,
        help="Skip torch.cuda.empty_cache() calls at test iterations for profiling.",
    )
    parser.add_argument(
        "--log_memory",
        action="store_true",
        default=False,
        help="Log CUDA memory allocator stats every 50 iterations to TensorBoard.",
    )
    return TrainingParserBundle(parser, model_params, optimization_params, pipeline_params)


def parse_training_args(
    argv: Sequence[str] | None = None,
    description: str = "EliGSiR training",
) -> tuple[Namespace, TrainingParserBundle]:
    """Parse trainer args using the shared parser."""

    bundle = build_training_parser(description=description)
    args = bundle.parser.parse_args(argv)
    try:
        reject_raw_live_population_conflicts(
            args, list(sys.argv[1:] if argv is None else argv)
        )
    except ValueError as error:
        bundle.parser.error(str(error))
    return args, bundle
