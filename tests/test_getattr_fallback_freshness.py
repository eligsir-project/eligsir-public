"""Guards against re-introducing stale ``getattr(args, "<flag>", <literal>)``
fallback literals -- both the narrow set issue #363/#367 already fixed and,
as of issue #368, the full repo-wide corpus.

Issue #363 (the CLAUDE.md parameter-default drift audit) found three call
sites whose ``getattr`` fallback literal had drifted from the real argparse
default in ``arguments/__init__.py`` -- a *behavioral* divergence hiding
behind a defensive-looking idiom, not just a docs issue: on any path where
the named attribute is genuinely absent from ``args``, the code would
silently apply the stale literal instead of the parser's real default. PR
#367 fixed those three (``streaming_global_maintenance_interval``,
``streaming_max_new_gaussians_per_frame`` x2) and this test originally
guarded only them.

Issue #368 (this test's current form) is the follow-up repo-wide sweep:
``scripts/audit_getattr_fallbacks.py`` enumerates every
``getattr(args, "<flag>", <literal>)`` call site outside ``tests/`` and
diffs the literal against the *live* full trainer parser
(``eligsir.cli.parser.build_training_parser()`` -- covers every
``ParamGroup`` class AND every flag declared directly in ``parser.py``,
unlike ``configs/profiles/defaults.reference.yml`` which only scans the
four ``ParamGroup`` classes and has a confirmed gap for parser.py-only
flags). Two test functions now share that one script as their single
definition of "a site" and "the real default":

- ``test_no_getattr_fallback_reintroduced_for_guarded_flags``: the growing
  set of specific (file, flag) pairs issue #363/#367/#368 confirmed were
  dead-code hygiene (unconditional ``ParamGroup`` default, reachability-
  audited nil in every real path) and fixed by removing the fallback
  entirely -- these must never carry a ``getattr`` fallback again.
- ``test_repo_wide_getattr_fallback_corpus_does_not_grow``: every remaining
  mismatch in the repo, as of the #368 audit, carried forward as an
  explicit, reasoned allowlist (``_DEFERRED_MISMATCHES`` /
  ``_DEFERRED_NOT_A_FLAG`` below) so a *new* drifted site fails CI
  immediately, without requiring the entire pre-existing corpus to be fixed
  first. This is the "generalize the guard" gating item #368's own body
  asked for -- reusing ``defaults.reference.yml``'s generated-snapshot
  *idea* (per ``tests/test_claude_md_defaults_freshness.py``'s
  ``_EXCLUDED_FLAGS`` pattern) via the live parser instead, since the YAML
  itself has the parser.py gap described above.

Pass 4 (2026-08-14) adds four more functions that close evasions the
repo-wide guard structurally could not see -- a deferred site's fallback
*literal* being changed to a new wrong value, an allowlist entry going stale
and standing as a permit for a future re-drift, a new non-literal fallback
expression sliding in unclassified, and a deliberately-kept getattr being
rewritten to bare access. See the block comment above
``_DEFERRED_MISMATCH_LITERALS`` for the full rationale. Re-audited at that
time: **27 mismatch call sites / 18 unique (file, flag) pairs, 10 not-a-flag
sites / 9 pairs, 8 non-literal sites** -- the issue title's "151 mismatches"
is four passes stale.

Why most of the allowlist is NOT simply fixed here: issue #368 explicitly
asks for a per-site reachability audit before touching a literal, and many
of these fallbacks are not "the same shape" as the #363/#367 sites --
_DEFERRED_MISMATCHES tags each one with why it's deferred rather than
fixed (deliberate safe-off sentinel, ambiguous multi-purpose sentinel,
contested/hot-zone code the project has open issues against, or reads a
flag that plain does not exist -- possible name drift, filed for human
follow-up rather than guessed at). See the PR this test shipped with (and
the triage table posted on issue #368) for the full per-flag reasoning.
"""
from __future__ import annotations

import re

import pytest
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_getattr_fallbacks import diff_against_parser, find_all_sites, real_defaults

# (file, flag) pairs that must never again carry a getattr(args, flag, <literal>)
# fallback -- bare attribute access only. Confirmed by a per-site reachability
# audit: each flag is an unconditional ParamGroup default, always present on
# the real parsed args every reachable call path is ever handed in production.
_GUARDED = [
    # issue #363 / PR #367
    ("train_streaming.py", "streaming_global_maintenance_interval"),
    ("eligsir/mapping/integration/insertion_committer.py", "streaming_max_new_gaussians_per_frame"),
    ("eligsir/mapping/integration/integration_planner.py", "streaming_max_new_gaussians_per_frame"),
    # issue #368 -- energy_alpha_dead (the issue's own named priority: this
    # exact getattr(..., 0.005) fallback literal already caused a confirmed
    # live bug once, before the parser default was raised to 0.05)
    ("train.py", "energy_alpha_dead"),
    ("train_streaming.py", "energy_alpha_dead"),
    ("utils/energy_mcmc.py", "energy_alpha_dead"),
    ("eligsir/mapping/integration/rolling_seed_application.py", "energy_alpha_dead"),
    # issue #368 -- mcmc_dead_opacity_end / mcmc_stop_growth_iter (same
    # drifted-magic-number shape as energy_alpha_dead, same MCMC schedule
    # family arguments/__init__.py documents as historically bug-prone)
    ("eligsir/optimization/gaussian/strategies/gsplat_mcmc_adapter.py", "mcmc_dead_opacity_end"),
    ("utils/strategies/mcmc/energy.py", "mcmc_dead_opacity_end"),
    ("utils/strategies/mcmc/energy.py", "mcmc_stop_growth_iter"),
    # Note: utils/strategies/mcmc/baseline.py's mcmc_dead_opacity_end /
    # mcmc_stop_growth_iter and train_streaming.py's streaming_max_depth are
    # deliberately NOT in this list even though each had mismatched sites
    # fixed here -- each of those files also has a separate, pre-existing (or,
    # for baseline.py's GsplatDefaultStrategy, a deliberately-kept -- see the
    # NOTE at that call site, a real test exercises this exact function with
    # a fully empty args and expects graceful degradation) getattr(...) call
    # for the same flag whose literal now matches the real default, so "zero
    # getattr for this flag in this file" would be a false claim. The
    # repo-wide test below still catches a *new* mismatch reappearing at any
    # of the fixed lines' old (stale) values.
    # issue #368 -- streaming_max_depth
    ("eligsir/optimization/gaussian/objective.py", "streaming_max_depth"),
    # issue #368 -- scalar_log_interval
    ("train.py", "scalar_log_interval"),
    ("train_streaming.py", "scalar_log_interval"),
    # issue #368 -- streaming_keyframe_window / streaming_global_replay_ratio
    ("scene/streaming_scene.py", "streaming_keyframe_window"),
    ("scene/streaming_scene.py", "streaming_global_replay_ratio"),
    ("train_streaming.py", "streaming_global_replay_ratio"),
    # issue #368 follow-up triage (2026-08-13): 76 further (file, flag) pairs
    # fixed by removing the fallback, after a reachability audit established
    # that every one of these files is only ever reached via the full,
    # live-parsed trainer `args` object threaded unchanged from train.py's
    # `__main__` through training()/streaming_training() (confirmed by
    # tracing that thread, plus checking every non-test/ caller of each
    # module for a hand-built or partial-parser Namespace) -- see the #368
    # triage comment for the full per-category breakdown and one file
    # (eligsir/io/sources/registry.py) deliberately EXCLUDED from this
    # sweep because it IS reachable from standalone diagnostic scripts with a
    # restored/partial cfg_args, so a bare-access rewrite there would risk a
    # real AttributeError. utils/rgbd_sequence_writer.py was ALSO excluded
    # from this sweep at the time, on the (incorrect) belief described below
    # -- see issue #395 for the correction, now fixed via
    # ("utils/rgbd_sequence_writer.py", "pointcloud_preprocess") and
    # ("utils/rgbd_sequence_writer.py", "pcd_voxel_size") below.
    ("scene/__init__.py", "init_scale_mode"),
    ("scene/streaming_scene.py", "streaming_eval_hold"),
    ("scene/streaming_scene.py", "streaming_seed_voxel_size"),
    # Re-audited 2026-08-13 for issue #455 (#394 removed
    # `getattr(self.args, "init_scale_mode", "fixed")` here, commit 4e139d2):
    # stays bare, but NOT because nothing else constructs a real StreamingScene.
    # CORRECTION (2026-08-13, review of PR #460): an earlier version of this note
    # claimed "every test uses a _Fake/_Stub/_Recording stand-in". That was
    # FACTUALLY WRONG. tests/test_observation_db_ingestion_wiring.py:643 does
    # `object.__new__(StreamingScene)` with a partial SimpleNamespace and calls
    # initialize_from_frames -- i.e. it reaches this exact bare read. That test
    # was red on main; it is now green because the fixture explicitly declares
    # init_scale_mode="knn". Do not re-derive "unreachable" from this entry.
    # The remaining audit facts that DO hold: the only PRODUCTION construction
    # is train_streaming.py's, with full parsed args, and the one non-test
    # partial-args harness (scripts/bench_accumulate_observed_surface.py's
    # _BenchStreamingScene) copies only accumulate_observed_surface and never
    # reaches initialize_from_frames. Note the removed literal was "fixed", NOT
    # the parser default "knn" -- if this fallback is ever restored, restoring
    # it as "knn" would silently change the bootstrap scale-init path.
    ("scene/streaming_scene.py", "init_scale_mode"),
    ("scene/streaming_scene.py", "pose_source"),
    ("scene/streaming_scene.py", "streaming_keyframe_coverage"),
    # Note: scene/streaming_scene.py's OTHER two streaming_global_reservoir_stride
    # sites (the ones the #368 audit actually flagged as literal/default
    # mismatches) were fixed to bare access, but a third, pre-existing site
    # (ingest_next_frame's reservoir_stride default) is deliberately NOT in
    # _GUARDED: its getattr fallback is `default_stride`, a computed
    # expression (0 or 1 depending on admission mode), not a copy of the real
    # parser default -- a genuinely different, adaptive-default idiom, not the
    # stale-literal-copy shape this list guards against.
    ("scripts/evaluate_streaming_odometry.py", "orbbec_open3d_odom_method"),
    ("eligsir/mapping/integration/integration_planner.py", "streaming_insert_scale_mode"),
    # issue #408 -- the four confirmed-reachable inverted-bool pairs #368 left
    # behind (fallback True vs. real parser default False). Resolved by
    # removing the fallback rather than picking a literal: production
    # reachability is nil (insert_gaussians_from_frame is only ever called from
    # train_streaming.py and eligsir/mapping/coordinator.py, both with the
    # full parsed args), and every partial-args TEST fixture that reaches this
    # function now declares all four explicitly at values that preserve its
    # existing behaviour byte-for-byte. See the #408 PR body for the
    # per-fixture audit.
    ("eligsir/mapping/integration/integration_planner.py", "streaming_depth_filter_enabled"),
    ("eligsir/mapping/integration/integration_planner.py", "streaming_depth_temporal_check"),
    ("eligsir/mapping/integration/integration_planner.py", "streaming_depth_temporal_require_for_insert"),
    ("eligsir/mapping/integration/integration_planner.py", "streaming_depth_temporal_require_for_sensor_closer"),
    # streaming_eval_hold was read identically in rolling_seed_application.py and
    # submap_stitch_application.py; #1075 extracted that shared prologue into
    # frame_windows.py, so the guarded read now lives there (once) instead of in
    # each trainer. Relocated rather than dropped -- the flag is still read as a
    # bare attribute with no getattr fallback, which is what this guard protects.
    ("eligsir/mapping/integration/frame_windows.py", "streaming_eval_hold"),
    ("eligsir/mapping/integration/rolling_seed_application.py", "lambda_opacity_entropy"),
    ("eligsir/mapping/integration/rolling_seed_application.py", "anisotropy_reg"),
    ("eligsir/mapping/integration/rolling_seed_application.py", "streaming_frame_admission"),
    ("eligsir/mapping/integration/submap_stitch_application.py", "streaming_frame_admission"),
    ("eligsir/optimization/gaussian/objective.py", "lambda_opacity_entropy"),
    ("eligsir/optimization/gaussian/objective.py", "lambda_eff_count"),
    ("eligsir/optimization/gaussian/objective.py", "anisotropy_reg"),
    ("eligsir/optimization/gaussian/objective.py", "streaming_young_reg_exempt_iters"),
    ("eligsir/optimization/gaussian/objective.py", "mcmc_target_splat_end"),
    ("train.py", "lambda_opacity_entropy"),
    ("train.py", "lambda_eff_count"),
    ("train.py", "anisotropy_reg"),
    ("train.py", "streaming_young_reg_exempt_iters"),
    ("train.py", "mcmc_target_splat_end"),
    ("train.py", "taming_score_interval"),
    ("train.py", "rgbd_max_depth"),
    ("train.py", "viewer_rviz_transport"),
    ("train.py", "web_viewer_scene_cache_interval"),
    ("train.py", "progress_video_interval"),
    ("train.py", "git_branch"),
    ("train.py", "git_commit"),
    ("train_streaming.py", "streaming_global_reservoir_stride"),
    ("train_streaming.py", "model_path"),
    ("train_streaming.py", "orbbec_pose_source"),
    ("train_streaming.py", "ros2_live_pose_topic"),
    ("train_streaming.py", "ros2_live_pose_source"),
    ("train_streaming.py", "orbbec_pose_topic"),
    ("train_streaming.py", "start_checkpoint"),
    ("train_streaming.py", "streaming_auto_downscale_max_pixels"),
    ("train_streaming.py", "streaming_pause_idle_threshold_s"),
    ("train_streaming.py", "viewer_rviz_transport"),
    ("train_streaming.py", "streaming_depth_loss_weight"),
    ("train_streaming.py", "streaming_depth_loss_type"),
    ("train_streaming.py", "streaming_bootstrap_anchor_weight"),
    ("train_streaming.py", "progress_video_interval"),
    ("train_streaming.py", "streaming_frame_admission"),
    ("train_streaming.py", "streaming_refinement_iterations"),
    ("train_streaming.py", "lambda_opacity_entropy"),
    ("train_streaming.py", "lambda_eff_count"),
    ("train_streaming.py", "anisotropy_reg"),
    ("train_streaming.py", "streaming_young_reg_exempt_iters"),
    ("train_streaming.py", "mcmc_target_splat_end"),
    ("train_streaming.py", "streaming_min_support_views"),
    ("train_streaming.py", "streaming_adcap_hysteresis"),
    ("train_streaming.py", "orbbec_open3d_odom_method"),
    ("utils/strategies/mcmc/base.py", "streaming_bubble_radius"),
    ("utils/strategies/mcmc/base.py", "streaming_young_prot_iters"),
    ("utils/strategies/mcmc/energy.py", "streaming_young_prot_iters"),
    ("utils/strategies/mcmc/legacy.py", "streaming_young_prot_iters"),
    ("utils/streaming_depth_lock.py", "streaming_depth_lock_conflict_thresh"),
    # issue #395 (2026-08-13): the #368 triage's "CONFIRMED ALWAYS-TAKEN, NOT
    # SAFE TO BARE-ACCESS" claim for these two (see the retired comment this
    # replaces, still visible in git history) was itself wrong -- both real
    # callers, scripts/capture_ros2_rgbd_sequence.py and
    # scripts/extract_rosbag_rgbd.py, have declared `--pointcloud-preprocess`
    # and `--pcd-voxel-size` (plus the other six pcd_* flags) in their own
    # argparse parsers since the original baseline import (commit 324e95f,
    # 2026-06-02) -- the earlier triage's "confirmed by grep" check searched
    # for the underscore attribute spelling instead of the argparse flag's
    # dashed spelling and got a false negative. No other caller of
    # write_rgbd_sequence_from_source exists (grep-confirmed), so bare access
    # is safe.
    ("utils/rgbd_sequence_writer.py", "pointcloud_preprocess"),
    ("utils/rgbd_sequence_writer.py", "pcd_voxel_size"),
    ("utils/rgbd_sequence_writer.py", "pcd_outlier_filter"),
    ("utils/rgbd_sequence_writer.py", "pcd_stat_nb_neighbors"),
    ("utils/rgbd_sequence_writer.py", "pcd_stat_std_ratio"),
    ("utils/rgbd_sequence_writer.py", "pcd_radius"),
    ("utils/rgbd_sequence_writer.py", "pcd_min_neighbors"),
    ("utils/rgbd_sequence_writer.py", "pcd_estimate_normals"),
    # issue #368 follow-up triage, pass 3 (2026-08-13): reachability-audited
    # inverted-bool and cap_max-SENTINEL sites the prior pass explicitly
    # deferred as "needs a human read" / "needs per-site semantic review".
    # The semantic-review question (what the fallback VALUE means) turned out
    # to be moot for every one of these: each site is only ever reached
    # through the same full, live-parsed trainer `args` object already
    # established for its file by the _GUARDED entries above it (traced the
    # same way -- every non-test caller confirmed, plus every test that
    # constructs a *partial* args/SimpleNamespace for the enclosing
    # function/class checked for whether it sets this specific attribute).
    # Since the fallback branch is provably dead in every real path, which
    # value it would have produced never mattered -- bare access is
    # behavior-neutral for every real run and only removes the trap for a
    # hypothetical future partial-args caller (which now fails loudly
    # instead of silently taking the wrong branch).
    ("geometry/sensor_depth.py", "streaming_depth_filter_enabled"),
    ("scene/streaming_scene.py", "streaming_depth_filter_enabled"),
    ("scene/streaming_scene.py", "streaming_depth_temporal_require_for_seed"),
    ("scene/streaming_scene.py", "streaming_prefetch"),
    ("scene/streaming_scene.py", "cap_max"),
    ("train.py", "web_viewer_enabled"),
    ("train_streaming.py", "streaming_keyframe_velocity_adaptive"),
    ("train_streaming.py", "streaming_keyframe_coverage_gain_gate"),
    ("train_streaming.py", "streaming_consolidation_burst"),
    ("train_streaming.py", "streaming_lifecycle_enabled"),
    ("train_streaming.py", "streaming_insert_from_depth"),
    ("train_streaming.py", "streaming_maturity_reg_ramp"),
    ("train_streaming.py", "cap_max"),
    ("eligsir/optimization/gaussian/objective.py", "streaming_maturity_reg_ramp"),
    ("eligsir/optimization/gaussian/objective.py", "cap_max"),
    ("eligsir/mapping/integration/rolling_seed_application.py", "cap_max"),
    ("utils/strategies/mcmc/base.py", "cap_max"),
    ("utils/strategies/mcmc/baseline.py", "cap_max"),
    ("utils/strategies/mcmc/energy.py", "streaming_lifecycle_enabled"),
    ("utils/strategies/mcmc/energy.py", "cap_max"),
    ("utils/taming_3dgs.py", "cap_max"),
    # Pre-existing smoke-check breakage found 2026-08-29: the fallback
    # literal (0) had drifted from the real parser default (20, declared as
    # a manual parser.add_argument() in eligsir/cli/parser.py rather
    # than a ParamGroup -- see this flag's ALLOWLIST entry in
    # tests/test_streaming_arg_parity.py). Fixed by removing the fallback in
    # favor of bare attribute access, matching every other entry here.
    ("eligsir/runtime/causal_evaluation.py", "streaming_partition_focus_max_promote"),
]


def test_no_getattr_fallback_reintroduced_for_guarded_flags():
    violations: list[str] = []
    for rel_path, flag in _GUARDED:
        text = (_REPO_ROOT / rel_path).read_text()
        pattern = re.compile(
            r'getattr\(\s*(?:self\.)?args\s*,\s*["\']' + re.escape(flag) + r'["\']\s*,'
        )
        for m in pattern.finditer(text):
            line_no = text[: m.start()].count("\n") + 1
            violations.append(f"  {rel_path}:{line_no}: getattr(args, {flag!r}, ...)")
        assert f"args.{flag}" in text, (
            f"{rel_path} no longer references args.{flag} at all -- if this "
            "flag's read was removed/renamed intentionally, update this test "
            "(and _GUARDED above) rather than deleting the check silently."
        )

    assert not violations, (
        "A getattr(args, <flag>, <literal>) fallback was reintroduced for a "
        "flag issue #363/#368 root-caused as a stale-default trap. These "
        "flags are unconditional ParamGroup defaults, always present on the "
        "real parsed args -- use bare attribute access (args.<flag>), not a "
        "getattr fallback with a hand-copied literal that can drift from "
        "arguments/__init__.py again.\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Repo-wide corpus (issue #368) -- deferred, not fixed, with a reason each.
# Keyed on (file, flag), not line number, so line churn doesn't fail this
# test spuriously; only a *new* mismatch (a different file/flag pair, or a
# flag that's since become resolvable but wasn't fixed) fails CI.
# ---------------------------------------------------------------------------

# Mismatches (flag exists on the real parser, fallback literal differs) that
# are deliberately NOT fixed in the #368 PR. Categories:
#   SAFE-OFF   -- the fallback is a documented/obvious "disabled" sentinel
#                 (0, 0.0, False, "", None) against a nonzero/True/non-empty
#                 real default. Nobody copied the real default here; they
#                 wrote "if unconfigured, do nothing". Issue #368 explicitly
#                 forbids updating a literal toward the real default, so for
#                 this category the only fix would be removing the fallback
#                 outright -- deferred pending a reachability audit per site.
#   SENTINEL   -- cap_max's getattr fallback varies (-1 / 10_000 / 1_000_000)
#                 across call sites and at least one of those values is used
#                 as an "unbounded"/"unset" sentinel downstream, not simply a
#                 stale copy of the real default -- needs individual
#                 per-site semantic review, not a mechanical rewrite.
#   HOT-ZONE   -- code this project has an open, actively-tracked issue
#                 against (parallelism_profile block: issue #228;
#                 streaming_steps_per_frame: W3-38/#302) where CLAUDE.md
#                 instructs readers to trust a specific runtime log line
#                 these getattr calls feed -- editing here risks colliding
#                 with that in-flight work.
#   DRIFT      -- ordinary drifted-magic-number copy, same shape as the
#                 _GUARDED fixes above, not yet audited/fixed in this PR.
_DEFERRED_MISMATCHES = {
    # Issue #397: train.py's getattr fallback is deliberately None, not the
    # real argparse default (1920/3840) -- None is a "let
    # write_post_training_report resolve it" sentinel (env var, then the
    # module default in utils/trajectory_video.py), not a stale copy of the
    # literal. See report_video_min_width/report_video_max_width's
    # docstring on write_post_training_report.
    ("train.py", "streaming_report_video_min_width"): "sentinel None, not a copied literal -- write_post_training_report resolves the real default itself",
    ("train.py", "streaming_report_video_max_width"): "sentinel None, not a copied literal -- write_post_training_report resolves the real default itself",
    # 2026-08-31 CI-baseline audit. Both below are deliberate sentinels
    # documented in their own function's docstring, not stale copies of the
    # real argparse default:
    ("eligsir/runtime/causal_entry.py", "streaming_refinement_checkpoint_interval"): "SAFE-OFF sentinel (0=no periodic mid-refinement checkpoint); matches this call's own downstream function default (checkpoint_interval_steps: int = 0 at causal_entry.py's enter_refinement helper), not a copy of the CLI default 2000",
    ("utils/source_attestation.py", "source_attestation_status"): "sentinel None means 'attribute absent entirely' by design -- source_attestation_from_args's own docstring: returns None (never partial) so a hand-rolled namespace degrades to pre-existing absent-key behavior rather than writing a misleading default; mirrors git_provenance_from_args's contract",
    # SAFE-OFF sentinels (0 / 0.0 / False / "" / None = documented "off"),
    # remaining after the issue #368 follow-up triage (2026-08-13) moved the
    # reachability-safe majority of this bucket into _GUARDED above. What's
    # left here is genuinely reachable from a standalone script / restored
    # cfg_args with a smaller parser (registry.py).
    # eligsir/io/sources/registry.py -- NOT in _GUARDED: confirmed reachable
    # (2026-08-13) from standalone diagnostic scripts (scripts/diagnose_d_vs_ed_depth_bias.py,
    # scripts/diagnose_cohort_removal_psnr.py) that call make_frame_source(cfg_args.source_path,
    # cfg_args) with a restored, potentially-older cfg_args Namespace rather than
    # the live full-trainer args -- bare access here risks a real AttributeError
    # on an old run directory. The realsense_*_topic "" fallback is also a
    # deliberate override-sentinel (empty = no topic override), not a copy error.
    ("eligsir/io/sources/registry.py", "streaming_auto_downscale_max_pixels"): "SAFE-OFF (0=disabled); registry.py is reachable from cfg_args-restoring diagnostic scripts, see note above -- not bare-accessed",
    ("eligsir/io/sources/registry.py", "realsense_color_topic"): "SAFE-OFF (''=unset ROS topic override); registry.py reachability note above",
    ("eligsir/io/sources/registry.py", "realsense_depth_topic"): "SAFE-OFF; registry.py reachability note above",
    ("eligsir/io/sources/registry.py", "realsense_color_info_topic"): "SAFE-OFF; registry.py reachability note above",
    ("eligsir/io/sources/registry.py", "realsense_depth_info_topic"): "SAFE-OFF; registry.py reachability note above",
    # NOTE (2026-08-14, issue #368 pass 4): four entries that used to live
    # here -- scene/streaming_scene.py's streaming_eval_min_baseline_m /
    # streaming_eval_min_angle_deg and
    # eligsir/mapping/optimization_runtime.py's
    # streaming_bootstrap_anchor_weight / streaming_insertion_anchor_decay_steps
    # -- were STALE: their literals had already been corrected to the real
    # parser default, so they are no longer mismatches and the auditor no
    # longer reports them. Leaving them here made this allowlist silently
    # permit a *future* re-drift at those exact sites. They have been pruned
    # from the allowlist and their reachability prose moved (not deleted) to
    # _GETATTR_KEPT_BY_DESIGN below, which asserts the getattr is still
    # there -- see that structure's docstring for why the prose matters.
    ("utils/strategies/mcmc/base.py", "streaming_bubble_guard_m"): "SAFE-OFF (0.0=bubble off); getattr AND literal both left as-is -- correcting the literal to the real default (1.0) changes tests/test_spatial_bubble_mask.py's observed boundary behavior (verified: 2 real test failures), so unlike the anchor/eval_min pairs (now in _GETATTR_KEPT_BY_DESIGN) this one genuinely cannot be touched without a deliberate behavior change",

    # inverted-bool -- genuinely reachable, left untouched. issue #368 follow-up
    # triage pass 3 (2026-08-13) resolved most of this bucket into _GUARDED
    # above (the "needs a human read" framing turned out to be moot once
    # reachability was checked: the fallback branch was dead in every real
    # path, so the VALUE never mattered). What's left here is the opposite
    # finding -- these ARE reachable, confirmed by real partial-args/
    # SimpleNamespace unit tests, so changing them (bare-access or literal)
    # is a genuine behavior decision, not a mechanical cleanup:
    ("eligsir/mapping/integration/depth_helpers.py", "streaming_depth_filter_enabled"): "inverted-bool; depth_helpers.py excluded from this pass (open PR #390)",
    # The four integration_planner.py inverted-bool entries that used to live
    # here were RESOLVED by issue #408 -- moved to _GUARDED above (fallback
    # removed, fixtures pinned). The "confirmed reachable" note they carried
    # was about TEST fixtures, not production: the only production callers of
    # insert_gaussians_from_frame are train_streaming.py and
    # eligsir/mapping/coordinator.py, both with the full parsed args.
    # streaming_lifecycle_enabled was ALSO resolved by issue #408 -- literal
    # changed from False to True (the parser default) per the per-pair table's
    # recommendation; the getattr fallback is kept because the documented
    # graceful-degradation contract (PR #380) has tests reaching it with a
    # fully-empty SimpleNamespace(). Now that the literal matches the real
    # default, it belongs in _GETATTR_KEPT_BY_DESIGN below, not here.

    # SENTINEL -- cap_max. issue #368 follow-up triage pass 3 (2026-08-13)
    # resolved this entire bucket: the "ambiguous meaning" question (is -1 an
    # intentional unbounded sentinel, or a stale copy?) turned out to be moot
    # for every site below once reachability was checked -- each is only ever
    # reached through the file's own already-established full-args-only call
    # chain (train.py's __main__ -> training()/streaming_training(), or a
    # strategy's initialize_state()/step_post_backward() reached only from
    # those two entry points), so the fallback value never actually governs a
    # real run and its semantic meaning never mattered. Moved to _GUARDED.
    # depth_helpers.py cap_max was NOT in this bucket (that file has none) and
    # utils/strategies/mcmc/baseline.py's GsplatDefaultStrategy (the class
    # with the documented empty-args test exception) has no cap_max read at
    # all -- "DefaultStrategy has NO max-Gaussian-count parameter" per its own
    # comment -- so baseline.py's cap_max fix (UpstreamGsplatMCMCStrategy
    # only) needed no exception, unlike that class's mcmc_stop_growth_iter/
    # mcmc_dead_opacity_end pair.
    #
    # insertion_committer.py's cap_max getattr(..., None) was RESOLVED by PR
    # #1142 and is deliberately absent from this allowlist. The deferral it
    # carried (commit 0fe7cb7e, 2026-08-23) asked one open question -- whether
    # the two partial-args fixtures that reach flush_insertion_batch with no
    # cap_max should instead declare it. They now do
    # (tests/test_insertion_hot_path_backend_parity.py's `_Args`,
    # tests/test_streaming_cohort_wiring.py's `_base_args()`, both pinned to
    # the real parser default 500000, far above either fixture's row count so
    # no capacity clamp is introduced), and the site reads `args.cap_max`
    # bare. That removes the last reachable path on which the fallback's
    # ambiguous `None` ("no capacity ceiling" vs. a stale copy of the parser
    # default) could have governed a real GaussianMutationArbiter, which is
    # what made the value a decision rather than a dead literal.

    # HOT-ZONE -- open, actively-tracked issues against this exact code.
    ("train.py", "parallelism_profile"): "HOT-ZONE (issue #228)",
    ("train.py", "optimizer_type"): "HOT-ZONE (issue #228)",
    ("train.py", "sh_update_interval"): "HOT-ZONE (issue #228)",
    ("train.py", "gsplat_sparse_grad"): "HOT-ZONE (issue #228)",
    ("train.py", "compile_after_iter"): "HOT-ZONE (adjacent to issue #228 parallelism block)",
    # train.py fused_ssim was resolved by the shared configure_training_ssim
    # boundary: every production caller supplies the complete parsed training
    # policy, so a false fallback both hid the real default (true) and allowed
    # canonical mapping to skip backend activation entirely.
    ("train_streaming.py", "streaming_steps_per_frame"): "HOT-ZONE (W3-38 / issue #302 admission-density work)",

    # DRIFT -- ordinary stale-magic-number copies. Most of this bucket was
    # fixed (moved to _GUARDED) in the issue #368 follow-up triage
    # (2026-08-13); what's left is either genuinely reachable from a
    # standalone-parser script (rgbd_sequence_writer.py, registry.py -- see
    # below, a real item-4 "silently governing real runs" finding) or not yet
    # re-audited.
    # issue #408 audit (2026-08-13) of the two registry.py DRIFT pairs:
    #   * rosbag_alignment_debug_frames -- FIXED (literal 0 -> real default 3),
    #     entry removed from this allowlist.
    #   * orbbec_open3d_odom_async -- FIXED (literal False -> True per #408's
    #     per-pair table recommendation), entry removed from this allowlist.
    #     Context: BOTH call sites (registry.py :269 RealsenseRosBagFrameSource,
    #     :322 OrbbecRosBagFrameSource) are *bag-file* sources; `ros2://live`
    #     returns at registry.py:137 via ROS2LiveStreamingFrameSource(args) and
    #     never reads this flag. This is bag-replay odometry determinism, NOT
    #     the live-camera path. Reachability in practice: nil (all 337
    #     output/*/cfg_args snapshots declare it, other callers use full parsed
    #     args).
    # issue #997 -- canonical root-level run_summary.json post-hoc builder
    ("utils/run_summary.py", "iterations"): "post-hoc summary inspection; None sentinel if omitted from args/report",
    ("utils/run_summary.py", "model_layout"): "post-hoc summary inspection; None sentinel if omitted from args/report",
    ("utils/run_summary.py", "resolution"): "post-hoc summary inspection; None sentinel if omitted from args/report",
    # utils/rgbd_sequence_writer.py's pointcloud_preprocess/pcd_voxel_size
    # used to be listed here as "CONFIRMED ALWAYS-TAKEN, NOT SAFE TO
    # BARE-ACCESS" (issue #368 follow-up triage, 2026-08-13). That claim was
    # wrong -- issue #395 (2026-08-13) found both real callers
    # (scripts/capture_ros2_rgbd_sequence.py, scripts/extract_rosbag_rgbd.py)
    # already declare `--pointcloud-preprocess`/`--pcd-voxel-size` (the
    # earlier triage's "confirmed by grep" check searched for the underscore
    # attribute spelling instead of the argparse flag's dashed spelling and
    # got a false negative). Fixed to bare access and moved to _GUARDED above
    # (along with the other six pcd_* flags on the same call, which were
    # never mismatches to begin with since their literals already matched
    # the real ModelParams defaults).
}

# Sites whose flag does NOT exist anywhere on the real parsed args at all
# (not a mismatch -- there's no real default to diff against). Reported for
# human follow-up rather than guessed at.
#
# ("utils/strategies/mcmc/base.py", "support_threshold") -- the
# `streaming_shadow_utility_support_threshold` name-drift bug this issue's
# first triage pass flagged as its single most interesting finding -- was
# ALREADY FIXED before this follow-up triage started, by issue #382 / PR #383
# ("fix(#382): bind KNN-isolation support gate to a real, declared flag"), a
# separate PR that landed on main ahead of this one. Confirmed: the call site
# now reads `getattr(args, "streaming_knn_isolation_support_threshold", 0.01)`
# and the auditor no longer reports this pair at all. Entry intentionally
# removed rather than left stale -- see the item-4 callout in the #368 triage
# comment for the full account (this is the SECOND confirmed "silently
# governing real runs" finding in this issue's history, after
# energy_alpha_dead, and it was independently fixed without going through
# this test's allowlist).
_DEFERRED_NOT_A_FLAG = {
    ("eligsir/mapping/integration/loss_diagnostics.py", "pose_provenance_result"):
        "runtime-assigned on args elsewhere (train_streaming.py), not an argparse flag",
    ("utils/comparison_report.py", "git_commit"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), issue #259), not an argparse flag",
    ("utils/comparison_report.py", "git_branch"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), issue #259), not an argparse flag",
    ("utils/comparison_report.py", "git_dirty"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), issue #259), not an argparse flag",
    ("utils/comparison_report.py", "git_dirty_paths"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), tracked-only git_dirty fix, "
        "2026-08-14), not an argparse flag -- optional so older hand-built args namespaces still degrade "
        "cleanly (git_provenance_from_args() omits the key when absent).",
    ("utils/comparison_report.py", "git_provenance_unavailable_reason"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), issue #784), not an argparse "
        "flag -- and deliberately absent from the dict on a SUCCESSFUL capture, so the getattr default is "
        "load-bearing rather than a hand-copied argparse literal that could drift.",
    ("utils/strategies/mcmc/baseline.py", "mcmc_strategy_verbose"): "NO SUCH FLAG anywhere in arguments/__init__.py",
    ("utils/strategies/mcmc/energy.py", "mcmc_strategy_verbose"): "NO SUCH FLAG anywhere in arguments/__init__.py",
    ("utils/strategies/mcmc/baseline.py", "default_grow_grad2d"): "NO SUCH FLAG anywhere in arguments/__init__.py -- reachable via the undocumented, real (but never-recorded-run-used) --densification_strategy gsplat_default",
    ("utils/strategies/mcmc/baseline.py", "default_grow_scale3d"): "NO SUCH FLAG anywhere in arguments/__init__.py -- same gsplat_default note as above",
    # issue #511 / PR #538 -- experiment-origin provenance.
    ("train.py", "persist_model_path"):
        "runtime-assigned on args by train.py's own output-staging setup "
        "(`args.persist_model_path = manager.persist_path`, train.py:214), not an "
        "argparse flag. The None fallback is load-bearing: staging is optional, so "
        "the attribute genuinely is absent on a non-staged run.",
    ("train_streaming.py", "persist_model_path"):
        "same runtime-assigned attribute as the train.py entry above, read from the "
        "streaming path for the same optional-staging reason",
    # NOTE (train.py, experiment_origin) -- entry REMOVED 2026-08-14 by this
    # test's own stale-entry guard, which is the first thing it caught in
    # anger. It was written here against a pre-merge shape of PR #538 that
    # read the attribute back through `getattr(args, "experiment_origin",
    # None)`; the merged #538 assigns it unconditionally in train.py's
    # __main__ (`args.experiment_origin =
    # resolve_experiment_origin_from_args(args)`, train.py:1912) and then
    # bare-accesses it in the same scope, so there is no getattr site left in
    # train.py for the auditor to report. Reachability prose kept per the
    # assertion's own instruction: the attribute IS runtime-assigned and is
    # NOT an argparse flag (the declarable surface is the
    # --experiment_origin_* flags, which the parser DOES define) -- so if a
    # getattr site ever reappears in train.py, it belongs in this dict again
    # rather than being "fixed" into a bare access on the assumption a parser
    # default exists. The two entries below cover the sites that DO exist.
    ("train_streaming.py", "experiment_origin"):
        "runtime-assigned on args by train.py's __main__ "
        "(`args.experiment_origin = resolve_experiment_origin_from_args(args)`, issue "
        "#511), not an argparse flag -- the declarable surface is the "
        "--experiment_origin_* flags, which the parser DOES define. The None fallback "
        "keeps hand-built args namespaces in tests/older callers degrading cleanly.",
    ("utils/experiment_origin.py", "experiment_origin"):
        "same runtime-assigned attribute; experiment_origin_from_args() is a pure "
        "accessor over it and must return None rather than re-resolve when absent -- "
        "re-resolving here is precisely the divergence the module is built to prevent",
    # issue #705 -- immutable Ablator declaration context. Resolved exactly
    # once by both real entrypoints after train.log/output staging exists, then
    # carried on args into offline/streaming report paths. It is transport
    # metadata, not a user-facing training flag; None keeps direct unit callers
    # of the trainer/report closures compatible.
    ("train.py", "experiment_context"):
        "runtime-assigned after start_output_log() by "
        "initialize_experiment_context(args.model_path), not an argparse flag; "
        "the environment variables are the declaration transport surface",
    ("train_streaming.py", "experiment_context"):
        "same runtime-assigned #705 context; the fallback preserves direct "
        "streaming_training() unit callers that do not pass through a real entrypoint",
    ("eligsir/runtime/report_kwargs.py", "experiment_context"):
        "the streaming report-kwargs builder, lifted out of train_streaming.py's "
        "closure by #1075 task P3 -- same runtime-assigned #705 context, same "
        "fallback rationale as the train_streaming.py entry above; only the file "
        "the read lives in changed.",
    ("eligsir/mapping/integration/rolling_seed_application.py", "experiment_context"):
        "same runtime-assigned #705 context on the separate rolling-seed trainer's "
        "own report path",
    ("eligsir/mapping/integration/submap_stitch_application.py", "experiment_context"):
        "same runtime-assigned #705 context on the separate submap-stitch trainer's "
        "own report path",
    # issue #524 acceptance criterion 3 -- run-intent provenance. Same shape
    # as experiment_origin above: train.py's __main__ assigns
    # args.run_intent_report = run_intent_report_from_args(...) unconditionally
    # and bare-accesses it in that same scope (no getattr site left there),
    # but the streaming trainer and the two separate-trainer integration
    # modules read it back defensively for hand-built/older args namespaces.
    # #1075 task P3 moved this read out of train_streaming.py's report-kwargs
    # closure into eligsir/runtime/report_kwargs.py. The reasoning below is
    # unchanged and deliberately kept rather than re-derived -- it is what stops
    # a later pass concluding the fallback is unreachable and removing it.
    ("eligsir/runtime/report_kwargs.py", "run_intent_report"):
        "runtime-assigned on args by train.py's __main__ "
        "(`args.run_intent_report = run_intent_report_from_args(...)`, issue #524), "
        "not an argparse flag -- the declarable surface is --run_intent, which the "
        "parser DOES define. The None fallback keeps hand-built args namespaces in "
        "tests/older callers degrading cleanly.",
    ("eligsir/mapping/integration/rolling_seed_application.py", "run_intent_report"):
        "same runtime-assigned attribute as the train_streaming.py entry above; this "
        "is one of the separate-trainer integration modules "
        "(integration_policy_is_separate_trainer) that reports its own run_intent "
        "block the same way it already reports experiment_origin/execution_semantics.",
    ("eligsir/mapping/integration/submap_stitch_application.py", "run_intent_report"):
        "same runtime-assigned attribute as the train_streaming.py entry above; the "
        "other separate-trainer integration module (see the rolling_seed_application.py "
        "entry immediately above).",
    # issue #639 -- exact holdout-GT persistence. render_report.py is a
    # STANDALONE entry point with its OWN argparse parser; it declares
    # --require_exact_holdout_gt itself (render_report.py's main()), and it is
    # deliberately NOT a trainer flag (nothing in a training run reads it).
    # real_defaults() only introspects build_training_parser(), so a flag owned
    # by another entry point's parser can only ever land here. The False
    # fallback is load-bearing: _process_checkpoint() is also called from
    # hand-built args namespaces in tests, and the safe default for a
    # strictness switch is off.
    ("render_report.py", "require_exact_holdout_gt"):
        "declared by render_report.py's OWN standalone parser, not the training "
        "parser real_defaults() introspects -- see the block comment above",
    # Same shape as render_report.py above: scripts/launch_issue1365_population_
    # qualification.py is a STANDALONE entry point with its OWN argparse parser
    # (main()'s `parser.add_argument("--evaluation-manifest", required=True)`),
    # not the training parser real_defaults() introspects.
    ("scripts/launch_issue1365_population_qualification.py", "evaluation_manifest"):
        "declared by this script's OWN standalone parser, not the training "
        "parser real_defaults() introspects -- see the render_report.py entry above",
    # issue #997 -- canonical root-level run_summary.json post-hoc builder
    ("utils/run_summary.py", "device"):
        "post-hoc summary inspection; optional attribute on args namespace, not an argparse flag in training parser",
    ("utils/run_summary.py", "git_branch"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), issue #259), not an argparse flag",
    ("utils/run_summary.py", "git_commit"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), issue #259), not an argparse flag",
    ("utils/run_summary.py", "git_dirty"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), issue #259), not an argparse flag",
    ("utils/run_summary.py", "git_dirty_paths"):
        "runtime-assigned on args by train.py's __main__ (get_git_metadata(), tracked-only git_dirty fix), not an argparse flag",
}


# ---------------------------------------------------------------------------
# Issue #368 pass 4 (2026-08-14) -- hardening the repo-wide guard against three
# evasions it could not see. The guard below (added in pass 3) already covers
# the FULL corpus, so "generalize the guard" is done; what it could NOT catch:
#
#   1. A deferred site's fallback literal being CHANGED to some new wrong
#      value. The allowlist is keyed on (file, flag) only, so
#      `getattr(args, "streaming_steps_per_frame", 50)` silently becoming
#      `... , 5)` stayed green. _DEFERRED_MISMATCH_LITERALS pins the exact
#      literal set per pair.
#   2. A deferred entry going STALE (its site fixed or removed) and lingering
#      as a standing permit for a future re-drift at that exact location.
#      Four such entries were found live when this test was written.
#   3. A NON-LITERAL fallback expression (`getattr(args, f, some_expr)`), which
#      diff_against_parser() returns in its third bucket and nothing consumed.
#      These are invisible to the mismatch diff by construction -- a new one
#      can be added freely without any guard noticing.
#
# All three pins are set-equality, not subset: a REMOVED site fails too, so a
# fix is noticed and the pin must be updated deliberately rather than rotting.
# ---------------------------------------------------------------------------

def _group_literals(sites):
    """{(path, flag): frozenset(literal_repr)} for an auditor site list."""
    grouped: dict[tuple[str, str], set[str]] = {}
    for s in sites:
        grouped.setdefault((s.path, s.flag), set()).add(s.literal_repr)
    return {k: frozenset(v) for k, v in grouped.items()}


# Exact fallback literals (as the auditor's `literal_repr`) for every deferred
# MISMATCH pair. Multiple call sites of the same (file, flag) legitimately
# carry different literals -- e.g. train.py's sh_update_interval is `None` at
# one site and `1` at three others -- hence a set per pair.
_DEFERRED_MISMATCH_LITERALS = {
    ("eligsir/io/sources/registry.py", "realsense_color_info_topic"): {"''"},
    ("eligsir/io/sources/registry.py", "realsense_color_topic"): {"''"},
    ("eligsir/io/sources/registry.py", "realsense_depth_info_topic"): {"''"},
    ("eligsir/io/sources/registry.py", "realsense_depth_topic"): {"''"},
    ("eligsir/io/sources/registry.py", "streaming_auto_downscale_max_pixels"): {"0"},
    ("eligsir/mapping/integration/depth_helpers.py", "streaming_depth_filter_enabled"): {"True"},
    ("train.py", "compile_after_iter"): {"0"},
    ("train.py", "gsplat_sparse_grad"): {"False"},
    ("train.py", "optimizer_type"): {"'adam'"},
    ("train.py", "parallelism_profile"): {"'off'"},
    ("train.py", "sh_update_interval"): {"1", "None"},
    ("train.py", "streaming_report_video_max_width"): {"None"},
    ("train.py", "streaming_report_video_min_width"): {"None"},
    ("train_streaming.py", "streaming_steps_per_frame"): {"0", "50"},
    ("utils/run_summary.py", "iterations"): {"None"},
    ("utils/run_summary.py", "model_layout"): {"None"},
    ("utils/run_summary.py", "resolution"): {"None"},
    ("utils/strategies/mcmc/base.py", "streaming_bubble_guard_m"): {"0.0"},
}

# Same pin for the NOT-A-FLAG bucket (runtime-assigned attributes and genuine
# name drift). These have no parser default to diff against, so the literal is
# the ONLY thing a guard can hold steady here.
_DEFERRED_NOT_A_FLAG_LITERALS = {
    ("eligsir/mapping/integration/loss_diagnostics.py", "pose_provenance_result"): {"None"},
    ("utils/comparison_report.py", "git_branch"): {"None"},
    ("utils/comparison_report.py", "git_commit"): {"None"},
    ("utils/comparison_report.py", "git_dirty"): {"None"},
    ("utils/comparison_report.py", "git_dirty_paths"): {"None"},
    ("utils/comparison_report.py", "git_provenance_unavailable_reason"): {"None"},
    ("utils/run_summary.py", "device"): {"None"},
    ("utils/run_summary.py", "git_branch"): {"'unknown'"},
    ("utils/run_summary.py", "git_commit"): {"'unknown'"},
    ("utils/run_summary.py", "git_dirty"): {"'unknown'"},
    ("utils/run_summary.py", "git_dirty_paths"): {"[]"},
    ("utils/strategies/mcmc/baseline.py", "default_grow_grad2d"): {"0.0002"},
    ("utils/strategies/mcmc/baseline.py", "default_grow_scale3d"): {"0.01"},
    ("utils/strategies/mcmc/baseline.py", "mcmc_strategy_verbose"): {"False"},
    ("utils/strategies/mcmc/energy.py", "mcmc_strategy_verbose"): {"False"},
}

# getattr(args, flag, <expression>) sites -- fallback is a computed expression,
# not a copied literal, so it cannot drift from a parser default in the
# stale-copy sense this file guards. Pinned purely so a NEW one is noticed and
# consciously classified (an expression can still encode a wrong assumption;
# it just needs a human read, not a literal diff). scene/streaming_scene.py's
# streaming_global_reservoir_stride is the documented example -- its fallback
# is `default_stride`, an adaptive 0-or-1 depending on admission mode.
_KNOWN_NONLITERAL_SITES = {
    # issue #1273: this former finite-source fallback was removed with the
    # retired thinning policy. The paced producer now rejects value admission
    # before replay; do not re-add a computed fallback merely for bare test
    # namespaces without reclassifying it here.
    # issue #1393 follow-up: the fallback is the literal expression
    # `2 * 1024 * 1024 * 1024`, which equals arguments/__init__.py's real
    # argparse default (2_147_483_648) by construction -- it exists only so
    # build_causal_viewer works when called with a bare args-like object
    # that lacks the attribute (e.g. a unit test double), matching the same
    # value get_device_transfer_owner's own default already uses.
    ("eligsir/runtime/causal_entry.py", "streaming_max_transfer_bytes"),
    ("scene/streaming_scene.py", "streaming_global_reservoir_stride"),
    ("eligsir/io/sources/registry.py", "rosbag_sync_threshold_ms"),
    ("train.py", "iterations"),
    ("train.py", "model_layout"),
    ("train_streaming.py", "streaming_replay_buffer"),
    ("utils/strategies/mcmc/baseline.py", "mcmc_noise_stop_iter"),
    ("utils/strategies/mcmc/energy.py", "mcmc_noise_stop_iter"),
    ("utils/strategies/mcmc/legacy.py", "mcmc_noise_stop_iter"),
    # issue #1262: the fallback is the PREDICATE_LATTICE module constant, which
    # IS the parser default -- one named constant shared by the parser default,
    # the config dataclass and the tests, so it cannot drift the way a copied
    # literal can. The getattr itself is kept because exposure_config_from_args
    # is unit-tested against partial SimpleNamespace args.
    ("utils/resolution_exposure.py", "streaming_resolution_exposure_predicate"),
    # Issue #397 follow-up (trajectory-video motion-smoothness interpolation):
    # both fallbacks are the module constants DEFAULT_MAX_INTERPOLATION_RENDERS/
    # DEFAULT_MAX_INTERPOLATION_PER_SLICE, which ARE the parser defaults
    # (arguments/__init__.py sets self.presentation_trajectory_interpolate_max_
    # extra_renders/_max_per_slice to these same literals) -- shared by
    # construction, not copied, so it cannot drift the way a copied literal can.
    # presentation_trajectory_interpolate_kwargs_from_args is also unit-tested
    # against a bare _Args() namespace (tests/test_trajectory_video.py), so the
    # getattr fallback is genuinely reachable, not dead defensiveness.
    ("utils/trajectory_video.py", "presentation_trajectory_interpolate_max_extra_renders"),
    ("utils/trajectory_video.py", "presentation_trajectory_interpolate_max_per_slice"),
    # 2026-08-31 CI-baseline audit: the fallback is MAX_RETAINED_TIERS_PER_FRAME,
    # imported from eligsir.mapping.hot_tier_cache -- the SAME module
    # constant eligsir/cli/parser.py uses as this flag's argparse default
    # (`default=MAX_RETAINED_TIERS_PER_FRAME`). Shared by construction, not
    # copied, so it cannot drift the way a copied literal can. The getattr
    # itself stays because mapping_runtime_config_from_args is exercised in
    # unit tests against partial namespaces missing this attribute.
    ("eligsir/runtime/config_boundary.py", "streaming_hot_tier_max_tiers_per_frame"),
}

# Sites where the fallback literal ALREADY equals the real parser default (so
# the auditor reports nothing) but the getattr must NOT be rewritten to bare
# attribute access, because a real unit test exercises the enclosing function
# with a deliberately partial/empty types.SimpleNamespace -- the documented
# graceful-degradation contract PR #380 first recorded for baseline.py's
# GsplatDefaultStrategy.
#
# This structure exists because these four sites were previously parked in
# _DEFERRED_MISMATCHES, where they had gone stale and were pruned in pass 4.
# Deleting them outright would have destroyed the reachability prose -- and
# this file's own #455 correction note records exactly what happens then: a
# later pass re-derives a wrong "nothing reaches this with partial args"
# conclusion and bare-accesses it. Found empirically in the pass-3 triage: a
# first attempt DID bare-access these and a full suite run surfaced real, new
# AttributeErrors.
_GETATTR_KEPT_BY_DESIGN = {
    ("scene/streaming_scene.py", "streaming_eval_min_baseline_m"):
        "literal already == real default (0.03); getattr kept -- "
        "tests/test_eval_holdouts.py exercises eval_metadata() with a partial SimpleNamespace",
    ("scene/streaming_scene.py", "streaming_eval_min_angle_deg"):
        "literal already == real default (2.0); getattr kept, same test-reachability note",
    ("eligsir/mapping/optimization_runtime.py", "streaming_bootstrap_anchor_weight"):
        "literal already == real default (1.0); getattr kept -- "
        "tests/test_streaming_optimization_runtime.py calls build_step() with args=SimpleNamespace()",
    ("eligsir/mapping/optimization_runtime.py", "streaming_insertion_anchor_decay_steps"):
        "literal already == real default (200, was stale at 2000); getattr kept, same note",
    ("eligsir/mapping/optimization_runtime.py", "streaming_lifecycle_enabled"):
        "literal now == real default (True, changed by issue #408); getattr kept -- "
        "tests/test_streaming_optimization_runtime.py calls "
        "StreamingOptimizationRuntime(..., args=SimpleNamespace()).build_fixed_topology_engine_inputs() "
        "with a FULLY empty SimpleNamespace() at 4 test functions; documented graceful-degradation "
        "contract (PR #380) requires the fallback to degrade cleanly",
}


@pytest.fixture(scope="module")
def fallback_corpus():
    """Classify the immutable source tree once; each guard owns its verdict."""
    return diff_against_parser(find_all_sites(), real_defaults())


def test_deferred_getattr_fallback_literals_are_pinned(fallback_corpus):
    """The repo-wide guard is keyed on (file, flag), so a deferred site's
    fallback literal could be changed to a *new* wrong value without failing
    anything. Pin the exact literal set per deferred pair."""
    mismatches, not_a_flag, _nonliteral = fallback_corpus

    problems: list[str] = []
    for label, observed, expected in (
        ("mismatch", _group_literals([s for s, _ in mismatches]), _DEFERRED_MISMATCH_LITERALS),
        ("not-a-flag", _group_literals(not_a_flag), _DEFERRED_NOT_A_FLAG_LITERALS),
    ):
        for pair in sorted(set(observed) & set(expected)):
            got = observed[pair]
            want = frozenset(expected[pair])
            if got != want:
                problems.append(
                    f"  [{label}] {pair[0]}: getattr(args, {pair[1]!r}, ...) fallback "
                    f"literals are now {sorted(got)}, pinned as {sorted(want)}"
                )

    assert not problems, (
        "A deferred getattr fallback's literal VALUE changed (issue #368 pass 4). "
        "The repo-wide corpus guard only keys on (file, flag) and cannot see "
        "this. If the change is deliberate, update the pin above in the same "
        "commit and say why in the PR body -- several of these sites sit on "
        "paths with open value decisions (#408) or open issues (#228, #302), "
        "where changing a literal IS the behaviour decision.\n" + "\n".join(problems)
    )


def test_deferred_getattr_allowlists_have_no_stale_entries(fallback_corpus):
    """An allowlist entry whose site no longer exists is a standing permit for
    a future re-drift at that exact location. Pass 4 found four such entries
    (sites whose literals had since been corrected to the real default)."""
    mismatches, not_a_flag, nonliteral = fallback_corpus

    live_mismatch = {(s.path, s.flag) for s, _ in mismatches}
    live_not_a_flag = {(s.path, s.flag) for s in not_a_flag}
    live_nonliteral = {(s.path, s.flag) for s in nonliteral}

    stale: list[str] = []
    unverifiable: list[str] = []
    for name, allowed, live in (
        ("_DEFERRED_MISMATCHES", set(_DEFERRED_MISMATCHES), live_mismatch),
        ("_DEFERRED_MISMATCH_LITERALS", set(_DEFERRED_MISMATCH_LITERALS), live_mismatch),
        ("_DEFERRED_NOT_A_FLAG", set(_DEFERRED_NOT_A_FLAG), live_not_a_flag),
        ("_DEFERRED_NOT_A_FLAG_LITERALS", set(_DEFERRED_NOT_A_FLAG_LITERALS), live_not_a_flag),
        ("_KNOWN_NONLITERAL_SITES", set(_KNOWN_NONLITERAL_SITES), live_nonliteral),
    ):
        for pair in sorted(allowed - live):
            # An entry whose FILE does not exist on this test surface is
            # UNVERIFIABLE, not stale. Dockerfile.ci-cpu COPYs a deliberate
            # subset of the tree -- it ships train.py / train_streaming.py /
            # train_ros_node.py but NOT render_report.py (nor compare_maps.py,
            # metrics.py, ...) -- so on the `cpu-ci` surface the auditor
            # cannot see those files at all and every allowlist entry naming
            # one looks like a removed site. That is a property of the image's
            # COPY list, not of the repo. Reporting it as "no such site in the
            # repo any more" is a false verdict, and it made `main` RED on
            # cpu-ci while GREEN on the training image for ~80 minutes on
            # 2026-08-15 (issue #739). The entry that tripped it was added by
            # PR #724, which was itself verified only on the training image --
            # the #649 surface asymmetry, running in the direction #649's own
            # workflow does not cover.
            if not (_REPO_ROOT / pair[0]).exists():
                unverifiable.append(
                    f"  {name}: ({pair[0]}, {pair[1]}) -- file absent on this surface"
                )
                continue
            stale.append(f"  {name}: ({pair[0]}, {pair[1]}) -- no such site in the repo any more")

    if unverifiable:
        # Printed, not silent: a shrinking checkable corpus is worth noticing
        # under `-rs` even though it must never fail this check.
        print(
            "[getattr-allowlist] staleness NOT checked for these entries on "
            "this surface (file not present in this image):\n"
            + "\n".join(unverifiable)
        )

    assert not stale, (
        "Stale getattr allowlist entries (issue #368 pass 4). Each of these "
        "names a (file, flag) that the auditor no longer reports -- the site "
        "was fixed, moved, or removed. Remove the entry, but move its "
        "reachability prose to _GETATTR_KEPT_BY_DESIGN (or a comment) rather "
        "than deleting it: that prose is what stops a later pass from "
        "re-deriving a wrong 'unreachable, safe to bare-access' conclusion.\n"
        + "\n".join(stale)
    )


def test_nonliteral_getattr_fallback_corpus_does_not_grow(fallback_corpus):
    """`getattr(args, flag, <expression>)` sites are invisible to the literal
    diff by construction (diff_against_parser returns them in a third bucket
    nothing consumed before pass 4). Pin the set so a new one gets a human
    read instead of sliding in unclassified."""
    _mismatches, _not_a_flag, nonliteral = fallback_corpus
    observed = {(s.path, s.flag) for s in nonliteral}

    added = sorted(observed - _KNOWN_NONLITERAL_SITES)
    removed = sorted(_KNOWN_NONLITERAL_SITES - observed)
    assert not added and not removed, (
        "The non-literal getattr(args, <flag>, <expression>) corpus changed "
        "(issue #368 pass 4). A computed fallback cannot drift from a parser "
        "default the way a copied literal can, but it can still encode a "
        "wrong assumption -- classify a new one deliberately and add it to "
        "_KNOWN_NONLITERAL_SITES, or remove a fixed one from that set.\n"
        f"  added:   {added}\n  removed: {removed}"
    )


def test_getattr_kept_by_design_sites_still_carry_a_fallback():
    """The inverse of _GUARDED: these sites MUST keep their getattr fallback,
    because a real unit test reaches them with a partial/empty SimpleNamespace.
    Rewriting one to bare attribute access is a real AttributeError, already
    observed once during the pass-3 triage."""
    missing: list[str] = []
    for rel_path, flag in _GETATTR_KEPT_BY_DESIGN:
        text = (_REPO_ROOT / rel_path).read_text()
        pattern = re.compile(
            r'getattr\(\s*(?:self\.)?args\s*,\s*["\']' + re.escape(flag) + r'["\']\s*,'
        )
        if not pattern.search(text):
            missing.append(
                f"  {rel_path}: getattr(args, {flag!r}, ...) is gone -- "
                f"{_GETATTR_KEPT_BY_DESIGN[(rel_path, flag)]}"
            )

    assert not missing, (
        "A getattr fallback that is deliberately KEPT (issue #368 pass 4) was "
        "removed. These are not stale-literal traps -- their literal already "
        "equals the real parser default, and a real unit test exercises the "
        "enclosing function with a deliberately partial args namespace, so "
        "bare attribute access raises AttributeError there. If the fixture "
        "was changed to declare the attribute, update this list in the same "
        "commit.\n" + "\n".join(missing)
    )


def test_repo_wide_getattr_fallback_corpus_does_not_grow(fallback_corpus):
    """A *new* getattr(args, flag, literal) drift site (repo-wide, excluding
    tests/) fails this test. A pre-existing, explicitly allowlisted one does
    not -- see _DEFERRED_MISMATCHES / _DEFERRED_NOT_A_FLAG above for why each
    entry is deferred rather than fixed."""
    mismatches, not_a_flag, _nonliteral = fallback_corpus

    new_mismatches = [
        (s, real) for s, real in mismatches
        if (s.path, s.flag) not in _DEFERRED_MISMATCHES
    ]
    new_not_a_flag = [
        s for s in not_a_flag
        if (s.path, s.flag) not in _DEFERRED_NOT_A_FLAG
    ]

    problems = []
    for s, real in sorted(new_mismatches, key=lambda t: (t[0].path, t[0].lineno)):
        problems.append(
            f"  {s.path}:{s.lineno}: getattr(args, {s.flag!r}, {s.literal_repr})  "
            f"real={real!r} -- NEW mismatch, not in _DEFERRED_MISMATCHES"
        )
    for s in sorted(new_not_a_flag, key=lambda s: (s.path, s.lineno)):
        problems.append(
            f"  {s.path}:{s.lineno}: getattr(args, {s.flag!r}, {s.literal_repr})  "
            f"-- flag does not exist on the real parser, and is NOT in _DEFERRED_NOT_A_FLAG"
        )

    assert not problems, (
        "New getattr(args, <flag>, <literal>) drift found that isn't in this "
        "test's allowlist (issue #368). Either the fallback literal has "
        "drifted from arguments/__init__.py's real default (fix it -- prefer "
        "removing the fallback per _GUARDED's pattern over updating the "
        "literal, which only resets the drift clock), or if this is a "
        "deliberate, reasoned exception (a safe-off sentinel, a runtime-only "
        "attribute, etc.), add it to _DEFERRED_MISMATCHES / "
        "_DEFERRED_NOT_A_FLAG above with a one-line reason -- do not just "
        "widen this test to pass silently.\n" + "\n".join(problems)
    )
