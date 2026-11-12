"""Resolved execution semantics -- the canonical run-mode contract (issue #168).

This module is deliberately **torch-free and import-side-effect-free** so it can
be unit-tested in the lean CPU CI image without ``tests/conftest.py``
registration -- the same scope fence ``utils/experiment_validity.py`` documents
for issues #423/#462/#478. It imports nothing from ``train.py``,
``train_streaming.py`` or ``arguments/`` at module scope.

Why this exists
---------------
"How a run executes" is decided by ~8 *orthogonal* axes that the repository has
historically encoded as overlapping booleans with names implying a single
``offline`` / ``online`` / ``streaming`` mode. The historical demonstration
(issue #168's own opening example, re-verified in code on 2026-08-14 and
hard-cut from current runtime profiles on 2026-08-26):

    configs/profiles/offline/tum_rgbd.yml set ``online_mode: false`` and
    ``streaming_ingestion_mode: iter_based`` but extended a default profile
    which set ``streaming_replay: true``; and ``train.py`` dispatched from
    that legacy flag. A profile named ``offline/*`` therefore ran the
    STREAMING trainer. Current code resolves the root loop exclusively from
    ``execution_trainer``; ``streaming_replay`` is retired and fails closed.

``online_mode: false`` therefore does not mean "offline trainer". It is a
streaming-internal frame-release/stop axis whose name implies a trainer axis.

The eight axes, with the code that actually decides each
--------------------------------------------------------
Every citation below was verified against the tree at the commit this module
landed on. The line numbers are the *decision points*, not merely mentions.

1. ``trainer`` -- which root control loop owns optimization.
   Decided at ``train.py`` from ``execution_trainer`` -- the sole root-trainer
   selector in the repository. Its argparse default is
   ``offline_reconstruction``; causal profiles explicitly select
   ``causal_mapping``.

2. ``source_mode`` -- is input fully materialized, replayed causally, or live.
   No single field decides this; derived here from ``trainer`` +
   ``source_path`` scheme (``ros2://`` => live) + ``ros2_live_bag_replay``.

3. ``release_policy`` -- when frames become visible to training.
   Decided by ``resolve_ingestion_mode_with_reason()`` at
   ``eligsir/mapping/scheduler/frame_scheduler.py:67``; consumed by
   ``FrameScheduler.should_release``. Flag default
   ``arguments/__init__.py:738`` (``"auto"``). Note the ``online_mode`` branch
   resolving to ``dataset_fps`` (issue #302) is a live, warned-about default.

4. ``admission_policy`` -- which released frames are actually trained on.
   ``--streaming_frame_admission``, default ``"all"``
   (``arguments/__init__.py:1240``).

5. ``integration_policy`` -- how geometry enters the map.
   ``--streaming_insert_from_depth`` (default ``True``,
   ``arguments/__init__.py:815``) plus ``--streaming_training_mode``
   (default ``"normal"``, ``arguments/__init__.py:1192``). The latter is
   OVERLOADED and this module records that rather than modelling it away:
   ``placement_only``/``colors_only`` are diagnostic optimization freezes
   inside the normal loop, while ``rolling_seed``/``submap_stitch``
   (``train_streaming.py:3692``/``:3705``, i.e.
   ``MappingCoordinator.run_submap_stitch_mode`` / ``run_rolling_seed_mode``)
   return before the main loop is ever entered -- they are separate trainers
   wearing a mode flag. Exposed as ``integration_policy_is_separate_trainer``.

6. ``optimization_core`` -- trainer-inline compatibility path vs shared engine.
   ``--streaming_optimization_engine`` defaults ``True`` after #755's
   representative-scale + commit-matched promotion decision and is consumed by
   ``train_streaming.py`` via ``_coordinator.optimization_engine_for(gaussians)``.
   The flag is streaming-only: the offline trainer still resolves to
   ``trainer_inline_control`` until #757 migrates it. Explicit
   ``--no-streaming_optimization_engine`` retains the old streaming path for a
   bounded historical-reproduction/control window (issue #758).

7. ``budget_policy`` -- what "run complete" means.
   ``--iterations`` vs ``--streaming_max_iterations`` (default ``0`` = none,
   ``arguments/__init__.py:876``) vs the rolling-idle rule keyed on
   ``--streaming_refinement_iterations`` (default ``50_000``,
   ``arguments/__init__.py:895``), which applies to ``online_mode`` runs only.

8. ``pose_source`` / ``pose_provenance`` -- who supplies the poses and how much
   they are trusted. Flag defaults ``arguments/__init__.py:1850`` / ``:1856``,
   but the RESOLVED value comes from
   ``eligsir/slam/pose_sources.py`` (``normalize_pose_source``),
   which ``train_streaming.py`` calls instead of reading the raw
   flag. The raw request is kept alongside as ``pose_source_requested``, and
   ``external_pose_provider`` records the ``ros2_live_pose_*`` provider the
   resolver deliberately does not encode.

Axis 9 of issue #168 ("evaluation semantics") is deliberately ABSENT: there is
no single field in the codebase that decides it. Holdout policy is spread over
``--streaming_eval_hold``/``--streaming_eval_guard`` and reported through the
separate ``eval_metadata`` block. Inventing an axis the code does not have
would be exactly the "describe intent as if it were implemented" failure this
module exists to prevent.

Scope fence (what this module does NOT do)
------------------------------------------
It does not select a trainer, does not mutate ``args``, and does not change any
run's behavior. It resolves, records, and validates. Validation is
**warn-by-default**, matching every precedent in this repo (``drift_warning``,
``experiment_validity``, the retained ``online_mode`` -> ``dataset_fps``
branch): only *provably impossible* combinations -- ones that could not have
produced a working historical run in the first place -- are hard errors, and
even then this module only reports them; the caller decides.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Bumped when the meaning of an axis changes, so a consumer reading an old
# report.json can tell which vocabulary it is looking at.
# 2 (issue #594): ``pose_source`` now mirrors train_streaming's
# ``open3d_odometry_live`` override, and the resolved pose-provenance tier is
# recorded alongside it as ``pose_provenance_resolved``.
# Not bumped for issue #701's ``supervision_mode``/``requested_supervision_
# batch_size``/``renderer_camera_batching``/``rasterization``/
# ``optimizer_updates_per_supervision_batch`` fields -- purely additive, none
# of the 8 existing axes changed meaning, so a consumer keyed only on
# CONTRACT_VERSION==2 still reads every pre-#701 field identically.
# 3 (issue #753): axis 6's ``optimization_core`` vocabulary changes:
# ``inline_legacy`` (serialized before 2026-08-15) → ``trainer_inline_control``
# (canonical, post-#753). Historical reports carry the old value; see
# LEGACY_OPTIMIZATION_CORE_ALIASES and canonical_optimization_core().
# 4 (issue #600): axis 4's ``admission_policy`` is no longer self-describing.
# The raw CLI token ``all`` used to mean exactly one thing (unconditional
# admission of every released frame). It now resolves to a capacity-aware
# SOFT-ALL, while the historical unconditional behavior is available only under
# the explicitly-named stress token ``force_all``. Consumers must therefore read
# the RESOLVED fields (``admission_target`` / ``admission_budgeted`` /
# ``source_coverage_target``), not the raw token -- a report at
# contract_version<=3 carrying ``admission_policy == "all"`` means force-all
# semantics, one at >=4 carrying the same token means soft-all. This is the
# whole reason the version is bumped: the vocabulary of an existing axis changed
# meaning, exactly like #753's optimization_core rename.
CONTRACT_VERSION = 4

# --- axis 1: trainer -----------------------------------------------------------
TRAINER_OFFLINE = "offline"
TRAINER_STREAMING = "streaming"

# --- axis 2: source temporal semantics ----------------------------------------
SOURCE_MATERIALIZED = "materialized"
SOURCE_CAUSAL_REPLAY = "causal_replay"
SOURCE_LIVE = "live"

# --- axis 3: frame release / ingestion schedule -------------------------------
# These three deliberately reuse ``frame_scheduler``'s OWN vocabulary verbatim
# (``INGESTION_MODE_*``) rather than inventing prettier synonyms -- a second
# name for the same decision is exactly the drift this contract exists to stop.
# ``all_at_start`` is the one value with no production counterpart: the offline
# trainer builds no FrameScheduler at all.
RELEASE_ALL_AT_START = "all_at_start"
RELEASE_ITER_BASED = "iter_based"
RELEASE_DATASET_FPS = "dataset_fps"
RELEASE_WALLCLOCK_STRICT = "wallclock_strict"

# --- axis 4: admission policy (issue #600) ------------------------------------
# The RESOLVED admission target, as opposed to the raw ``--streaming_frame_
# admission`` token. Three values, deliberately not four: ``selective`` covers
# every gated mode (``hybrid_keyframe``, ``fixed_stride``, ...) because for every
# downstream consumer written so far the only question that matters is "is a gate
# allowed to drop frames by design", not which gate.
#
#   soft_all         -- production ``all``. Attempts to retain every useful
#                       observation, but MAY skip mapping supervision for a
#                       frame when the mapping backlog/age bound is exceeded.
#                       Budgeted. A skipped frame still advances pose/source
#                       history and must NOT be depth-inserted.
#   selective         -- an explicit gate decides (hybrid_keyframe et al.).
#                       Unchanged by #600.
#   force_all_stress  -- the historical hard-all. Unconditional admission, no
#                       capacity bound, unbounded mapping backlog permitted.
#                       Opt-in, experimental/stress only -- retained because
#                       assimilation-overload is a scientifically useful
#                       condition (see #603/#290), not because it is safe.
#   preselected       -- canonical MappingApplication handoff (issue #1376:
#                       reworded -- the prior wording could be read as the
#                       producer owning a semantic keyframe/admission
#                       decision). The producer boundary owns PHYSICAL
#                       transport survival/backpressure only -- which packets
#                       arrive at all. It makes no semantic admission choice.
#                       Core still constructs ObservationOwner and the
#                       semantic providers, and still owns the real admission
#                       decision on every packet that survives transport;
#                       "preselected" means there is no SECOND, redundant
#                       transport-level admission gate ahead of core's own.
ADMISSION_TARGET_SOFT_ALL = "soft_all"
ADMISSION_TARGET_SELECTIVE = "selective"
ADMISSION_TARGET_FORCE_ALL_STRESS = "force_all_stress"
ADMISSION_TARGET_PRESELECTED = "preselected"

# Raw ``--streaming_frame_admission`` tokens. ``preselected`` means transport
# already ran its own survival/backpressure at the producer boundary -- core
# still owns the real (semantic) admission decision; see the block above.
ADMISSION_TOKEN_SOFT_ALL = "all"
ADMISSION_TOKEN_FORCE_ALL = "force_all"
ADMISSION_TOKEN_PRESELECTED = "preselected"

# ``source_coverage_target``: what the run promises about SOURCE coverage, which
# #600 makes a separate question from supervision admission. Every offline/replay
# release policy walks the whole source trajectory (frames are observed even when
# their mapping supervision is skipped), so coverage is ``full``; a live source
# has no total to be complete against and is paced by the wall clock instead.
COVERAGE_TARGET_FULL = "full"
COVERAGE_TARGET_REALTIME = "realtime"


def resolve_admission_semantics(
    frame_admission: Any,
    *,
    source_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve a raw ``--streaming_frame_admission`` token to #600 semantics.

    This is the ONE authoritative place that turns the CLI token into meaning.
    Every consumer (``partition_eval``, ``realized_cadence``, the trainer's own
    decision contract, report writers) must call this rather than string-testing
    ``== "all"`` -- that test is exactly what #600's premise correction names as
    the highest-damage hazard, because a soft-all run still carries the token
    ``"all"`` while no longer having hard-all's coverage guarantee.

    Args:
        frame_admission: the raw token, or ``None``/empty (treated as the
            ``"all"`` default, matching argparse).
        source_mode: optional axis-2 value; only used to resolve
            ``source_coverage_target``.

    Returns:
        ``admission_target`` / ``admission_budgeted`` / ``source_coverage_target``
        plus the raw token echoed back as ``admission_policy_requested``.
    """
    token = str(frame_admission or ADMISSION_TOKEN_SOFT_ALL).strip().lower()
    if token == ADMISSION_TOKEN_PRESELECTED:
        target = ADMISSION_TARGET_PRESELECTED
        budgeted = True
    elif token == ADMISSION_TOKEN_FORCE_ALL:
        target = ADMISSION_TARGET_FORCE_ALL_STRESS
        budgeted = False
    elif token == ADMISSION_TOKEN_SOFT_ALL:
        target = ADMISSION_TARGET_SOFT_ALL
        budgeted = True
    else:
        target = ADMISSION_TARGET_SELECTIVE
        budgeted = True
    coverage = (
        COVERAGE_TARGET_REALTIME
        if source_mode == SOURCE_LIVE
        else COVERAGE_TARGET_FULL
    )
    return {
        "admission_policy_requested": token,
        "admission_target": target,
        "admission_budgeted": budgeted,
        "source_coverage_target": coverage,
    }


def _resolve_admission_token(args: Any) -> str:
    """The admission token that will ACTUALLY be in force, mirroring the trainer.

    ``train_streaming.py`` rewrites ``all`` -> ``hybrid_keyframe`` when
    ``--orbbec_pose_source open3d_odometry_live`` is set. That mutation happens
    AFTER the startup banner is printed but BEFORE ``report.json`` is written, so
    the two disagreed: the banner said ``all`` while the report said
    ``hybrid_keyframe`` for the same run.

    Mirroring the override here is exactly the fix #594 applied to the
    ``pose_source`` axis for the same override (and the reason CONTRACT_VERSION
    was bumped to 2), rather than adding a parallel ``admission_policy_requested``
    field that would have to be reconciled against this one forever. The RAW
    request stays visible in ``raw_flags["streaming_frame_admission"]``.

    ``force_all`` is deliberately NOT overridden -- the trainer's own condition
    tests ``== "all"``, so a stress run keeps its declared semantics.
    """
    token = str(
        _get(args, "streaming_frame_admission", ADMISSION_TOKEN_SOFT_ALL)
        or ADMISSION_TOKEN_SOFT_ALL
    )
    if (
        _get(args, "orbbec_pose_source", None) == "open3d_odometry_live"
        and token == ADMISSION_TOKEN_SOFT_ALL
    ):
        return "hybrid_keyframe"
    return token


# --- axis 5: geometry integration policy --------------------------------------
INTEGRATION_BOOTSTRAP_ONLY = "bootstrap_only"
INTEGRATION_INCREMENTAL_RGBD = "incremental_rgbd"
INTEGRATION_ROLLING_SEED = "rolling_seed"
INTEGRATION_SUBMAP_STITCH = "submap_stitch"

# ``streaming_training_mode`` values that return before the main streaming loop
# is entered (train_streaming.py:3692/:3705, run_submap_stitch_mode /
# run_rolling_seed_mode) -- i.e. separate trainers.
SEPARATE_TRAINER_MODES = ("rolling_seed", "submap_stitch")

# --- axis 6: optimization implementation --------------------------------------
# Changed 2026-08-15 (issue #753): CORE_INLINE_LEGACY renamed to CORE_TRAINER_INLINE_CONTROL,
# which better describes what executes (trainer-level inline control loop, trainer-agnostic).
# Historical reports contain the old value; use LEGACY_OPTIMIZATION_CORE_ALIASES and
# canonical_optimization_core() for compatibility.
CORE_TRAINER_INLINE_CONTROL = "trainer_inline_control"
CORE_GAUSSIAN_ENGINE = "gaussian_engine"

# Backward compatibility: map old serialized values to new canonical ones.
# See canonical_optimization_core() for how to use this.
LEGACY_OPTIMIZATION_CORE_ALIASES = {
    "inline_legacy": CORE_TRAINER_INLINE_CONTROL,
}

# --- issue #701: supervision / renderer execution semantics -------------------
# NOT one of the 8 axes above (those decide "how the run is structured"); these
# fields decide "how one optimizer step is actually executed" -- specifically
# whether `--streaming_optimization_batch_size` (B) is real multi-camera
# rasterization or sequential per-view rendering with gradient accumulation.
# TRUE_CAMERA_BATCH is only valid for the canonical causal MappingApplication
# path.  Its work-plan executor uses ``render_true_camera_batch`` and invokes
# gsplat once with C camera matrices.  Other B>1 paths remain sequential
# gradient accumulation, so batch size alone must never imply vectorized
# camera rasterization.  The capability is still deliberately fail-closed:
# all cameras must share a resolution, native gsplat SH and sparse gradients
# are required, and an unsupported transaction fails rather than falling back
# to sequential rendering while retaining the TRUE_CAMERA_BATCH label.
SUPERVISION_SINGLE_VIEW = "SINGLE_VIEW"
SUPERVISION_SEQUENTIAL_GRADIENT_ACCUMULATION = "SEQUENTIAL_GRADIENT_ACCUMULATION"
SUPERVISION_TRUE_CAMERA_BATCH = "TRUE_CAMERA_BATCH"  # reserved, never emitted today

RASTERIZATION_SEQUENTIAL_PER_VIEW = "SEQUENTIAL_PER_VIEW"
RASTERIZATION_VECTORIZED_CAMERA_BATCH = "VECTORIZED_CAMERA_BATCH"  # reserved, never emitted today

# --- axis 7: compute / stop policy --------------------------------------------
BUDGET_FIXED_ITERATIONS = "fixed_iterations"
BUDGET_ROLLING_IDLE = "rolling_idle"
BUDGET_ROLLING_IDLE_CAPPED = "rolling_idle_capped"

AXIS_NAMES = (
    "trainer",
    "source_mode",
    "release_policy",
    "admission_policy",
    "integration_policy",
    "optimization_core",
    "budget_policy",
    "pose_source",
)


def canonical_optimization_core(value: str) -> str:
    """Resolve a (possibly legacy) optimization_core value to its canonical form.

    Maps historical serialized values (from pre-#753 reports) to the current canonical
    vocabulary. Returns the value unchanged if it is not a legacy alias.

    Idempotent: canonical_optimization_core(canonical_optimization_core(x)) == canonical_optimization_core(x).

    Args:
        value: An optimization_core value, possibly from a historical report.

    Returns:
        The canonical (post-#753) value for this core implementation.
    """
    return LEGACY_OPTIMIZATION_CORE_ALIASES.get(str(value or "").strip(), str(value or "").strip())


def _get(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _resolve_release_policy(
    source_path: str, requested: Optional[str], online_mode: bool
) -> Tuple[str, str]:
    """Resolve the release policy, reusing the production resolver.

    Imported lazily and defensively: ``frame_scheduler`` is torch-free today,
    but this module's CPU-testability guarantee must not depend on that staying
    true. The fallback mirrors the same decision table.
    """
    try:
        from eligsir.mapping.scheduler.frame_scheduler import (
            resolve_ingestion_mode_with_reason,
        )

        return resolve_ingestion_mode_with_reason(
            source_path, requested, online_mode=online_mode
        )
    except Exception:  # pragma: no cover - defensive
        mode = (requested or "auto").strip().lower()
        if mode != "auto":
            return mode, "explicit_flag"
        if str(source_path or "").startswith("ros2://"):
            return RELEASE_DATASET_FPS, "ros2_source_default"
        if online_mode:
            return RELEASE_DATASET_FPS, "online_mode_default"
        return RELEASE_ITER_BASED, "offline_default"


def _resolve_pose_source(
    source_path: str, requested: Optional[str], orbbec_pose_source: Optional[str] = None
) -> str:
    """Resolve the pose source the way the trainer actually does.

    Core pose-source names are transport-neutral after #1073. ROS producers
    attach their tracking observation before this contract is evaluated.

    ``train_streaming.py:1798`` applies one further override the pose-source
    resolver itself does not encode: a dataset-pose run with
    ``--orbbec_pose_source open3d_odometry_live`` really runs the in-process
    Open3D odometry estimator, so the resolved source is that estimator, not
    the dataset. Mirrored here (issue #594) so the contradiction check below
    does not fire on that legitimate combination.
    """
    try:
        from eligsir.slam.pose_sources import (
            is_dataset_pose_source,
            normalize_pose_source,
        )
    except ImportError:  # pragma: no cover - torch-free compatibility boundary
        resolved = str(requested or "dataset_pose")
        if orbbec_pose_source == "open3d_odometry_live" and resolved == "dataset_pose":
            return "open3d_odometry_live"
        return resolved

    resolved = str(normalize_pose_source(requested))
    if orbbec_pose_source == "open3d_odometry_live":
        if is_dataset_pose_source(resolved):
            return "open3d_odometry_live"
    return resolved


def _external_pose_provider(args: Any, source_path: str) -> Optional[Dict[str, Any]]:
    """The pose provider a ``ros2://`` run really consumes.

    Deliberately a SEPARATE field from the transport-neutral ``pose_source``.
    A live profile carrying
    ``pose_source: dataset_pose`` alongside ``ros2_live_pose_topic:
    /slam/odom`` is NOT a contradiction; the second field is where the truth
    about the upstream provider lives, and the contract records it.
    """
    if not str(source_path or "").startswith("ros2://"):
        return None
    return {
        "ros2_live_pose_source": str(_get(args, "ros2_live_pose_source", "odom") or "odom"),
        "ros2_live_pose_topic": str(_get(args, "ros2_live_pose_topic", "") or ""),
        "ros2_live_pose_convention": str(
            _get(args, "ros2_live_pose_convention", "ros") or "ros"
        ),
    }


def _resolve_pose_provenance(
    args: Any, source_path: str, resolved_pose_source: str
) -> Dict[str, Any]:
    """Resolve the same provenance tier the trainer banner prints (issue #594).

    ``eligsir.slam.pose_provenance`` is torch-free (stdlib only, and
    ``eligsir/slam/__init__.py`` is empty), so this import is safe in the
    CPU-CI image. On failure the block records an explicit ``error`` string
    rather than a null -- a silently-null provenance key is exactly the
    "silent degradation" failure
    ``tests/test_cpu_smoke_torch_free_modules.py`` documents.
    """
    is_live = str(source_path or "").startswith("ros2://")
    pose_topic = str(
        _get(args, "ros2_live_pose_topic", "") if is_live
        else _get(args, "orbbec_pose_topic", "") or ""
    )
    try:
        from eligsir.slam.pose_provenance import (
            pose_provenance_contradictions,
            resolve_pose_provenance,
        )

        result = resolve_pose_provenance(
            cli_override=_get(args, "pose_provenance", "auto"),
            source_path=source_path,
            pose_topic=pose_topic,
            orbbec_pose_source=_get(args, "orbbec_pose_source", None),
            resolved_pose_source=resolved_pose_source,
        )
        block = dict(result.as_dict())
        block["pose_topic"] = pose_topic
        block["contradictions"] = pose_provenance_contradictions(
            result=result,
            resolved_pose_source=resolved_pose_source,
            source_path=source_path,
        )
        block["error"] = ""
        return block
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "tier": "unknown",
            "method": "unknown",
            "detail": "",
            "provider": "",
            "live": None,
            "topic_evidence_admissible": None,
            "pose_topic": pose_topic,
            "contradictions": [],
            "error": f"pose-provenance resolution failed: {exc!r}",
        }


def _resolve_source_mode(trainer: str, source_path: str, bag_replay: bool) -> str:
    if str(source_path or "").startswith("ros2://"):
        # A recorded bag replayed through the live ROS2 path is causal replay,
        # not genuinely live -- issue #168 section 3 asks for exactly this
        # distinction ("ensure a rosbag can be used as causal replay without
        # being mislabeled as genuinely live").
        return SOURCE_CAUSAL_REPLAY if bag_replay else SOURCE_LIVE
    if trainer == TRAINER_OFFLINE:
        return SOURCE_MATERIALIZED
    return SOURCE_CAUSAL_REPLAY


def _resolve_integration_policy(args: Any, trainer: str) -> str:
    mode = str(_get(args, "streaming_training_mode", "normal") or "normal").lower()
    if mode in SEPARATE_TRAINER_MODES:
        return mode
    if trainer == TRAINER_OFFLINE:
        return INTEGRATION_BOOTSTRAP_ONLY
    if bool(_get(args, "streaming_insert_from_depth", True)):
        return INTEGRATION_INCREMENTAL_RGBD
    return INTEGRATION_BOOTSTRAP_ONLY


def _resolve_budget_policy(args: Any, trainer: str) -> str:
    if trainer == TRAINER_OFFLINE:
        return BUDGET_FIXED_ITERATIONS
    if not bool(_get(args, "online_mode", False)):
        # The rolling-idle rule is scoped to online_mode runs; a causal replay
        # without online_mode has a real, finite --iterations budget.
        return BUDGET_FIXED_ITERATIONS
    if int(_get(args, "streaming_max_iterations", 0) or 0) > 0:
        return BUDGET_ROLLING_IDLE_CAPPED
    return BUDGET_ROLLING_IDLE


def _resolve_optimization_core(args: Any, trainer: str) -> str:
    """Resolve the core that the selected trainer can actually execute.

    ``streaming_optimization_engine`` is deliberately inert on the offline
    trainer until #757 migrates that path, so the promoted True default must
    not make offline reports claim an engine they never enter.
    """
    if trainer == TRAINER_OFFLINE:
        return CORE_TRAINER_INLINE_CONTROL
    return (
        CORE_GAUSSIAN_ENGINE
        if bool(_get(args, "streaming_optimization_engine", True))
        else CORE_TRAINER_INLINE_CONTROL
    )


def _resolve_supervision_mode(args: Any, trainer: str) -> Tuple[str, int]:
    """Resolve issue #701's supervision/renderer execution fields.

    ``requested_supervision_batch_size`` is the REQUESTED B
    (``--streaming_optimization_batch_size``), not the realized one -- a
    skipped/tier-mismatched extra view can make the realized batch smaller
    (see ``train_streaming.py``'s ``batch_tier_skip`` accounting and
    ``engine.py``'s ``skipped_view_count``), but that realized-vs-requested
    gap is exactly the counter work issue #703 owns (see this module's
    docstring / the ``docs/architecture`` note this issue ships alongside).
    This field always reports the requested value, consistent with its own
    name -- it is ``supervision_mode`` that is gated below, not this one.

    The offline trainer has no batching flag at all (``train.py`` never reads
    ``streaming_optimization_batch_size``) -- it always resolves to
    ``SINGLE_VIEW``/1 regardless of what a value that flag might carry over
    from an inherited config profile.

    ``streaming_training_mode == "placement_only"`` (H1, "no backward/step/
    MCMC") disables BOTH real B>1 call sites outright: the inline path's
    accumulation block and the engine path's ``ExtraViewBatchCollector``
    collection block share the identical guard
    ``_opt_batch_size > 1 and not _placement_only and ...``
    (``train_streaming.py``, ~lines 6120 and 6268). A placement_only run with
    B>1 therefore renders and steps the PRIMARY view only -- reporting
    ``SEQUENTIAL_GRADIENT_ACCUMULATION`` there would be exactly the kind of
    over-claim issue #701's hard rule warns against for the renderer fields,
    just for ``supervision_mode`` instead. ``colors_only`` (H2) is NOT gated
    here: it freezes geometry but leaves the accumulation blocks' guard
    unaffected, so B>1 genuinely accumulates under colors_only.
    """
    if trainer == TRAINER_OFFLINE:
        return SUPERVISION_SINGLE_VIEW, 1
    requested = int(_get(args, "streaming_optimization_batch_size", 1) or 1)
    if requested < 1:
        requested = 1
    if requested == 1:
        return SUPERVISION_SINGLE_VIEW, requested
    training_mode = str(
        _get(args, "streaming_training_mode", "normal") or "normal"
    ).lower()
    if training_mode == "placement_only":
        return SUPERVISION_SINGLE_VIEW, requested
    if trainer == TRAINER_STREAMING and str(
        _get(args, "execution_trainer", "")
    ) == "causal_mapping":
        return SUPERVISION_TRUE_CAMERA_BATCH, requested
    return SUPERVISION_SEQUENTIAL_GRADIENT_ACCUMULATION, requested


def resolve_execution_semantics(args: Any) -> Dict[str, Any]:
    """Resolve every represented axis once, from a parsed args namespace.

    Pure: reads ``args`` via ``getattr`` only, mutates nothing. Returns a plain
    JSON-serializable mapping. Never raises on a missing attribute -- an
    incomplete namespace resolves against the documented argparse defaults, so
    this is safe to call from a startup banner before any validation.
    """
    source_path = str(_get(args, "source_path", "") or "")
    online_mode = bool(_get(args, "online_mode", False))
    execution_trainer = str(
        _get(args, "execution_trainer", "offline_reconstruction")
        or "offline_reconstruction"
    )
    trainer = (
        TRAINER_STREAMING
        if execution_trainer == "causal_mapping"
        else TRAINER_OFFLINE
    )
    release_policy, release_reason = _resolve_release_policy(
        source_path, _get(args, "streaming_ingestion_mode", "auto"), online_mode
    )
    # The canonical causal runner uses a real-time source clock whenever an
    # input-rate cap is present.  Keep this decision here, then consume this
    # resolved value in the runner, so provenance and execution cannot drift.
    # An explicit ingestion mode always wins.
    if (
        trainer == TRAINER_STREAMING
        and str(_get(args, "streaming_ingestion_mode", "auto") or "auto") == "auto"
        and release_policy == RELEASE_ITER_BASED
        and float(_get(args, "streaming_input_fps_cap", 0.0) or 0.0) > 0.0
    ):
        release_policy = RELEASE_DATASET_FPS
        release_reason = "causal_fps_cap_default"
    if trainer == TRAINER_OFFLINE:
        # The offline trainer materializes every camera up front (Scene()
        # loads the whole dataset); no scheduler is constructed at all, so the
        # ingestion-mode flag it may have inherited from a profile is inert.
        release_policy, release_reason = RELEASE_ALL_AT_START, "offline_trainer"

    integration_policy = _resolve_integration_policy(args, trainer)
    training_mode = str(
        _get(args, "streaming_training_mode", "normal") or "normal"
    ).lower()
    pose_source = _resolve_pose_source(
        source_path,
        _get(args, "pose_source", "dataset_pose"),
        _get(args, "orbbec_pose_source", None),
    )
    pose_provenance_resolved = _resolve_pose_provenance(args, source_path, pose_source)
    supervision_mode, requested_supervision_batch_size = _resolve_supervision_mode(
        args, trainer
    )

    source_mode = _resolve_source_mode(
        trainer, source_path, bool(_get(args, "ros2_live_bag_replay", False))
    )
    admission = resolve_admission_semantics(
        _resolve_admission_token(args),
        source_mode=source_mode,
    )

    return {
        "contract_version": CONTRACT_VERSION,
        # --- the eight represented axes ---
        "trainer": trainer,
        # Derived from trainer; this records the selected loop but is never a
        # second dispatch selector.
        "root_entrypoint": (
            "eligsir.runtime.causal_entry.run_causal_mapping"
            if trainer == TRAINER_STREAMING
            else "train.training"
        ),
        "source_mode": source_mode,
        "release_policy": release_policy,
        # The RAW token. Kept as axis 4's value for continuity, but as of
        # contract_version 4 it is NO LONGER self-describing -- read
        # ``admission_target`` instead (issue #600).
        "admission_policy": admission["admission_policy_requested"],
        # --- issue #600: resolved admission semantics ---
        # Same requested-vs-resolved split ``pose_source``/``pose_source_
        # requested`` already uses. ``admission_policy_requested`` reconciles
        # the previously-unreconciled cfg_args divergence rather than adding a
        # parallel field.
        "admission_policy_requested": admission["admission_policy_requested"],
        "admission_target": admission["admission_target"],
        "admission_budgeted": admission["admission_budgeted"],
        "source_coverage_target": admission["source_coverage_target"],
        "integration_policy": integration_policy,
        "optimization_core": _resolve_optimization_core(args, trainer),
        "budget_policy": _resolve_budget_policy(args, trainer),
        "pose_source": pose_source,
        # --- axis annotations ---
        "pose_source_requested": str(
            _get(args, "pose_source", "dataset_pose") or "dataset_pose"
        ),
        "external_pose_provider": _external_pose_provider(args, source_path),
        "release_policy_reason": release_reason,
        "pose_provenance": str(_get(args, "pose_provenance", "auto") or "auto"),
        # The RESOLVED provenance (issue #594), as opposed to the raw
        # ``--pose_provenance`` request above -- same requested-vs-resolved
        # split ``pose_source`` / ``pose_source_requested`` already uses.
        "pose_provenance_resolved": pose_provenance_resolved,
        "integration_policy_is_separate_trainer": training_mode
        in SEPARATE_TRAINER_MODES,
        "streaming_training_mode": training_mode,
        # Axis 9 of #168 has no authoritative field in code; say so explicitly
        # rather than inventing a value for it.
        "evaluation_semantics": None,
        "unrepresented_axes": ["evaluation_semantics"],
        # --- issue #701: supervision / renderer execution semantics ---
        # Deliberately separate from the 8 axes above (not one of them) --
        # see the constants' module-level comment for why every field here
        # is a verified constant/derivation, never a guess, and why none of
        # them resolves to "unknown": both real B>1 call sites were read
        # verbatim and confirmed sequential-per-view for every value B can
        # take today.
        "supervision_mode": supervision_mode,
        "requested_supervision_batch_size": requested_supervision_batch_size,
        "renderer_camera_batching": supervision_mode
        == SUPERVISION_TRUE_CAMERA_BATCH,
        "rasterization": (
            RASTERIZATION_VECTORIZED_CAMERA_BATCH
            if supervision_mode == SUPERVISION_TRUE_CAMERA_BATCH
            else RASTERIZATION_SEQUENTIAL_PER_VIEW
        ),
        # Both B=1 and B>1 apply exactly one optimizer.step() per supervision
        # batch (gradient accumulation across B renders, one step) -- verified
        # in train_streaming.py's inline PR3 loop and
        # engine.py's run_batched_step, neither of which calls step() more
        # than once per requests batch.
        "optimizer_updates_per_supervision_batch": 1,
        # --- the raw flags each axis was derived from, for auditability ---
        "raw_flags": {
            "execution_trainer": execution_trainer,
            "streaming_replay": bool(_get(args, "streaming_replay", False)),
            "online_mode": online_mode,
            "streaming_ingestion_mode": str(
                _get(args, "streaming_ingestion_mode", "auto") or "auto"
            ),
            "streaming_frame_admission": str(
                _get(args, "streaming_frame_admission", "all") or "all"
            ),
            "streaming_insert_from_depth": bool(
                _get(args, "streaming_insert_from_depth", True)
            ),
            "streaming_training_mode": training_mode,
            "streaming_optimization_engine": bool(
                _get(args, "streaming_optimization_engine", True)
            ),
            "streaming_optimization_batch_size": int(
                _get(args, "streaming_optimization_batch_size", 1) or 1
            ),
            "iterations": _get(args, "iterations", None),
            "streaming_max_iterations": _get(args, "streaming_max_iterations", 0),
            "streaming_refinement_iterations": _get(
                args, "streaming_refinement_iterations", None
            ),
            "pose_source": str(
                _get(args, "pose_source", "dataset_pose") or "dataset_pose"
            ),
            "source_is_ros2": source_path.startswith("ros2://"),
            "ros2_live_bag_replay": bool(_get(args, "ros2_live_bag_replay", False)),
        },
    }


# Streaming-only flags with NO read on the offline path. Measured 2026-08-14:
# zero occurrences in train.py, and a repo-wide grep for each name outside
# train_streaming.py / eligsir/mapping/ found only argparse definitions,
# docstrings, log strings, and post-hoc report tooling (utils/promote_run.py's
# key list, utils/mapping_phase_backfill.py) -- no read of these VALUES on any
# module the offline trainer reaches. A non-default value for any of them on an
# offline run is therefore inert: the generalized form of the offline/tum_rgbd
# finding. NOTE ``streaming_depth_filter_enabled`` is deliberately EXCLUDED --
# geometry/sensor_depth.py:102 really does read it (via SensorDepthProvider),
# so it cannot be safely classified as inert here.
_STREAMING_ONLY_INERT_ON_OFFLINE = (
    "online_mode",
    "streaming_ingestion_mode",
    "streaming_frame_admission",
    "streaming_insert_from_depth",
    "streaming_training_mode",
    "streaming_optimization_engine",
    "streaming_camera_profile",
    "streaming_max_iterations",
    "streaming_refinement_iterations",
)

_INERT_DEFAULTS = {
    "online_mode": False,
    "streaming_ingestion_mode": "auto",
    "streaming_frame_admission": "all",
    "streaming_insert_from_depth": True,
    "streaming_training_mode": "normal",
    "streaming_optimization_engine": True,
    "streaming_camera_profile": "default",
    "streaming_max_iterations": 0,
    "streaming_refinement_iterations": 50_000,
}


def validate_execution_semantics(
    semantics: Dict[str, Any], args: Any = None
) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for a resolved semantics mapping.

    ``errors`` are combinations that are *provably impossible* -- they could
    not have produced a working historical run, so refusing them cannot break
    an existing spec. ``warnings`` are ambiguities and misleading-but-working
    configurations. This function never raises and never mutates anything; the
    caller decides what to do (``train.py`` prints both and refuses on errors).
    """
    errors: List[str] = []
    warnings: List[str] = []

    trainer = semantics.get("trainer")
    source_mode = semantics.get("source_mode")
    release = semantics.get("release_policy")
    raw = semantics.get("raw_flags", {}) or {}

    # --- hard errors: provably impossible ---
    if trainer == TRAINER_OFFLINE and raw.get("source_is_ros2"):
        errors.append(
            "trainer=offline with a ros2:// source: the offline trainer "
            "materializes every camera up front via Scene() and has no frame "
            "scheduler or ROS2 ingestion path at all, so this cannot run. Select "
            "--execution_trainer causal_mapping."
        )
    if source_mode == SOURCE_LIVE and release == RELEASE_ALL_AT_START:
        errors.append(
            "source_mode=live with release_policy=all_at_start: a live source "
            "has no materialized frame set to release at start."
        )
    if semantics.get("integration_policy_is_separate_trainer") and trainer == (
        TRAINER_OFFLINE
    ):
        errors.append(
            f"streaming_training_mode={semantics.get('streaming_training_mode')!r} "
            "requires the streaming trainer -- it is dispatched inside "
            "train_streaming.py (:3692/:3705, run_submap_stitch_mode / "
            "run_rolling_seed_mode) and is unreachable from the "
            "offline path."
        )

    # --- warnings: ambiguous or misleading, but historically working ---
    if trainer == TRAINER_OFFLINE and args is not None:
        inert = []
        for name in _STREAMING_ONLY_INERT_ON_OFFLINE:
            if name not in _INERT_DEFAULTS:
                continue
            value = _get(args, name, _INERT_DEFAULTS[name])
            if value != _INERT_DEFAULTS[name]:
                inert.append(f"{name}={value!r}")
        if inert:
            warnings.append(
                "trainer=offline but these streaming-only flags are set to "
                "non-default values and have no read on any module the offline "
                "trainer reaches, i.e. they are inert: "
                f"{', '.join(sorted(inert))}. "
                "Either drop them or select --execution_trainer causal_mapping."
            )

    if trainer == TRAINER_STREAMING and not raw.get("online_mode", False):
        # Not an error -- but it IS the axis confusion #168 names. Since #600
        # this must test the RESOLVED target, not the raw token: a soft-all run
        # still carries the token ``all`` while no longer guaranteeing coverage
        # of SUPERVISION (it does still observe every source frame, which is
        # why soft_all is not warned about here -- see source_coverage_target).
        if semantics.get("admission_target") == ADMISSION_TARGET_SELECTIVE:
            warnings.append(
                "trainer=streaming with a GATED admission policy "
                f"({semantics.get('admission_policy')!r}) and online_mode=False: "
                "the rolling-idle coverage-enforcement rule is scoped to "
                "online_mode runs, so admitted-frame coverage is not enforced. "
                "See the Run Contracts 'Full trajectory coverage' section."
            )

    # Issue #600: force-all is a deliberate assimilation-overload STRESS
    # condition, not a production setting. Make every run that uses it say so in
    # its own artifacts, so a reader never has to infer it from the raw token.
    if semantics.get("admission_target") == ADMISSION_TARGET_FORCE_ALL_STRESS:
        warnings.append(
            "admission_target=force_all_stress: this run uses the EXPERIMENTAL "
            "unbounded hard-all admission mode. Every released frame is admitted "
            "for mapping supervision with no capacity bound, permitting an "
            "unbounded mapping backlog. This is a scientifically intentional "
            "assimilation-overload/seed-fade stress condition (issues #600/#603), "
            "NOT a production configuration -- do not compare its quality numbers "
            "against a production soft-all run as though they were the same "
            "policy."
        )

    if release == RELEASE_DATASET_FPS and semantics.get(
        "release_policy_reason"
    ) == "online_mode_default":
        warnings.append(
            "release_policy=dataset_fps was chosen by the --online_mode "
            "default, not requested explicitly. Frame release is paced by a "
            "simulated wall clock and --streaming_steps_per_frame is IGNORED "
            "(issue #302). Pass --streaming_ingestion_mode iter_based to opt out."
        )

    if raw.get("online_mode") and trainer == TRAINER_OFFLINE:
        warnings.append(
            "online_mode=True has NO effect on the offline trainer: train.py "
            "reads it in zero dispatch conditions. It is a streaming-internal "
            "release/stop axis, not a trainer axis (issue #168 section 1)."
        )

    if (
        trainer == TRAINER_STREAMING
        and semantics.get("optimization_core") == CORE_TRAINER_INLINE_CONTROL
    ):
        warnings.append(
            "optimization_core=trainer_inline_control is the temporary "
            "compatibility/control path after #755 promoted gaussian_engine as "
            "the default streaming optimization core. Use it only for historical "
            "reproduction or experiments deliberately pinned to the old "
            "implementation; do not compare it with gaussian_engine unless the "
            "core implementation is the intended variable."
        )

    # --- issue #594: resolved pose source vs. resolved provenance tier -------
    _prov = semantics.get("pose_provenance_resolved") or {}
    for _c in _prov.get("contradictions") or []:
        warnings.append(_c)
    if _prov.get("error"):
        warnings.append(
            "pose provenance could not be resolved for this run: "
            f"{_prov['error']}. Treat every quality claim from it as "
            "provenance-unverified."
        )

    if semantics.get("integration_policy_is_separate_trainer"):
        warnings.append(
            f"integration_policy={semantics.get('integration_policy')!r} returns "
            "before the main streaming loop is entered -- it is a separate "
            "trainer wearing a mode flag, NOT a per-frame integration strategy "
            "comparable to incremental_rgbd (issue #168 section 5)."
        )

    return errors, warnings


def build_execution_semantics_block(args: Any) -> Dict[str, Any]:
    """Build the ``report.json["execution_semantics"]`` block.

    Mirrors ``git_provenance_from_args`` / ``build_experiment_validity_block``:
    a pure record written into the run's own artifacts, never a gate.
    """
    semantics = resolve_execution_semantics(args)
    errors, warnings = validate_execution_semantics(semantics, args)
    semantics["errors"] = errors
    semantics["warnings"] = warnings
    semantics["valid"] = not errors
    return semantics


def compute_progress_postfix(
    iteration: int,
    n_gaussians: int,
    n_frames_ingested: int,
    n_train_cameras: int,
    keyframes_admitted: int,
    elapsed_s: float,
    ema_loss: float,
) -> Dict[str, Any]:
    """
    Compute the tqdm progress-bar postfix dict (issue #701).

    Issue #701 fixes the it/V denominator (was keyframes_admitted,
    should be n_train_cameras), disambiguates labels, and keeps
    legacy parser compatibility.

    Args:
        iteration: current training iteration
        n_gaussians: current Gaussian count
        n_frames_ingested: total frames ingested from source
        n_train_cameras: current number of training cameras
        keyframes_admitted: total unique keyframes admitted
        elapsed_s: wall-clock seconds since training start
        ema_loss: exponential moving average of loss

    Returns:
        Dict with keys: Loss, N, F, V, it/V, mapFPS, adm%
    """
    # Safeguard against zero divisions
    _adm = max(1, keyframes_admitted)
    _views = max(1, n_train_cameras)
    _elapsed = max(1e-6, elapsed_s)

    # it/V: iterations per view (iterations / number of training cameras)
    # This is the training budget each camera has received on average.
    # FIXED 2026-08-15 (issue #701): was dividing by keyframes_admitted,
    # now correctly divides by n_train_cameras to match the label.
    it_per_view = iteration // _views

    return {
        "Loss": f"{ema_loss:.7f}",
        "N": n_gaussians,  # current Gaussian count
        "F": n_frames_ingested,  # frames ingested from source
        "V": _views,  # training cameras (N_views)
        "it/V": it_per_view,  # iterations per view (training budget)
        "mapFPS": f"{_adm / _elapsed:.1f}",  # keyframes admitted per wall-second
        "adm%": f"{100 * _adm / max(1, n_frames_ingested):.0f}",  # admission percentage
    }


def format_execution_semantics(semantics: Dict[str, Any]) -> str:
    """Render the resolved contract as a startup banner.

    Printed once by ``train.py``'s ``__main__`` before trainer dispatch, so
    every run self-identifies its resolved semantics rather than only its raw
    flags (issue #168 section 2, "add a startup summary of resolved execution
    semantics"). Also printed once near the start of ``train_streaming.py``'s
    ``streaming_training()`` (issue #701) so a streaming run's own log carries
    the banner even when the caller inspects `train.log` only from the
    ``[RUN START]`` marker onward -- see that call site's comment for why a
    second, streaming-scoped print is not merely redundant with the
    ``train.py`` ``__main__`` one.
    """
    lines = [
        "==================== RESOLVED EXECUTION SEMANTICS ====================",
        f"[execution] contract_version = {semantics.get('contract_version')}",
    ]
    for axis in AXIS_NAMES:
        lines.append(f"[execution] {axis:<20} = {semantics.get(axis)}")
    lines.append(
        f"[execution] {'root_entrypoint':<20} = "
        f"{semantics.get('root_entrypoint')}"
    )
    lines.append(
        f"[execution] {'release_reason':<20} = "
        f"{semantics.get('release_policy_reason')}"
    )
    # issue #701: supervision / renderer execution semantics -- printed right
    # after the 8 axes, before pose-provenance, so a B>1 run cannot be read
    # without immediately seeing whether the renderer is actually batching.
    lines.append(
        f"[execution] {'supervision_mode':<20} = {semantics.get('supervision_mode')} "
        f"(configured_max_batch_size={semantics.get('requested_supervision_batch_size')}; "
        "realized_B/C_in_tick_telemetry)"
    )
    lines.append(
        f"[execution] {'renderer_batching':<20} = "
        f"{semantics.get('renderer_camera_batching')} "
        f"(rasterization={semantics.get('rasterization')}, "
        f"optimizer_updates_per_batch={semantics.get('optimizer_updates_per_supervision_batch')})"
    )
    lines.append(
        f"[execution] {'pose_provenance':<20} = {semantics.get('pose_provenance')}"
    )
    _prov = semantics.get("pose_provenance_resolved") or {}
    if _prov:
        lines.append(
            f"[execution] {'pose_provenance_tier':<20} = {_prov.get('tier')} "
            f"(method={_prov.get('method')}, provider={_prov.get('provider') or 'n/a'}, "
            f"live={_prov.get('live')})"
        )
    unrep = semantics.get("unrepresented_axes") or []
    if unrep:
        lines.append(
            f"[execution] {'unrepresented':<20} = {', '.join(unrep)} "
            "(no authoritative field in code)"
        )
    for err in semantics.get("errors", []) or []:
        lines.append(f"[execution] ERROR: {err}")
    for warn in semantics.get("warnings", []) or []:
        lines.append(f"[execution] WARNING: {warn}")
    lines.append(
        "======================================================================"
    )
    return "\n".join(lines)
