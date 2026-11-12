"""
Frame release scheduler for streaming replay simulation.

Three ingestion modes (--streaming_ingestion_mode):

  iter_based      (default) Release one frame every streaming_steps_per_frame
                  training iterations — fast, reproducible, no clock.

  dataset_fps     Advance a virtual clock by each iteration's measured wall
                  time; release the next frame when its dataset timestamp is
                  due (capped at streaming_input_fps_cap fps).  Training never
                  drops frames — it runs as many iters as it can between
                  releases.  Results are deterministic regardless of GPU speed.

  wallclock_strict Use perf_counter; late frames are dropped so the map never
                  gets ahead of sensor time.  Most faithful to real SLAM but
                  results vary with hardware.
"""
from __future__ import annotations

import time


INGESTION_MODE_AUTO = "auto"
INGESTION_MODE_ITER_BASED = "iter_based"
INGESTION_MODE_DATASET_FPS = "dataset_fps"
INGESTION_MODE_WALLCLOCK_STRICT = "wallclock_strict"
INGESTION_MODES = {
    INGESTION_MODE_AUTO,
    INGESTION_MODE_ITER_BASED,
    INGESTION_MODE_DATASET_FPS,
    INGESTION_MODE_WALLCLOCK_STRICT,
}



# Why a given ingestion mode was chosen (see resolve_ingestion_mode_with_reason).
# Recorded verbatim in report.json["ingestion"]["reason"] so every run
# self-identifies which pacing it actually ran under (issue #302).
INGESTION_REASON_EXPLICIT = "explicit_flag"
INGESTION_REASON_ROS2_DEFAULT = "ros2_source_default"
INGESTION_REASON_ONLINE_MODE_DEFAULT = "online_mode_default"
INGESTION_REASON_OFFLINE_DEFAULT = "offline_default"
INGESTION_REASON_UNKNOWN_VALUE = "unrecognized_value_passthrough"

# Emitted once at startup whenever the online_mode branch (and only that
# branch) lands on dataset_fps. Kept as a module constant so a future rebase
# cannot silently drop the warning the way 12899c2 dropped 7ffaebc's comment.
ONLINE_MODE_DATASET_FPS_WARNING = (
    "\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
    "[streaming] WARNING: --online_mode selected ingestion_mode=dataset_fps.\n"
    "  Frame release is paced by a SIMULATED WALL CLOCK, not by iterations.\n"
    "  --streaming_steps_per_frame / effective_spf are IGNORED in this mode\n"
    "  (should_release() never reads them). Expect near-immediate source\n"
    "  exhaustion: on a real fr3 run the release rate was exactly\n"
    "  F = 24 + 2 * iteration (bounded only by --streaming_max_drain_per_iter),\n"
    "  dumping all 2525 frames by iteration ~1251 against an advertised 378750.\n"
    "  Iteration wall-time drives the simulated clock, so cadence is NOT\n"
    "  reproducible across machines or across performance-affecting commits.\n"
    "  To opt out: pass --streaming_ingestion_mode iter_based explicitly.\n"
    "  See issue #302 and docs/2026-06-21-scene-explosion-isolation.md.\n"
    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
)


def resolve_ingestion_mode_with_reason(
    source_path: str,
    requested: str | None = None,
    *,
    online_mode: bool = False,
) -> tuple[str, str]:
    """Resolve the frame-release mode AND why it was chosen.

    Returns ``(mode, reason)`` where ``reason`` is one of the
    ``INGESTION_REASON_*`` constants above. ``resolve_ingestion_mode()`` is a
    thin wrapper returning only the mode, so existing callers are unchanged.

    Provenance lives here, next to the decision, deliberately: before issue
    #302 the training loop re-derived "was this explicit or defaulted?" by
    independently re-checking ``args.streaming_ingestion_mode == "auto"``, i.e.
    one fact encoded in two places — exactly the shape of fragility that let
    the regression below survive seven weeks unnoticed.

    !!! WARNING — DO NOT REMOVE THE ``online_mode`` BRANCH BELOW WITHOUT
    !!! READING THIS FIRST, AND DO NOT SILENTLY RESTORE/REVERT IT EITHER.
    !!!
    !!! History (issue #302, and #278 before it):
    !!!   * 7ffaebc (2026-06-21, PR #12) REMOVED ``if online_mode: return
    !!!     DATASET_FPS`` after root-causing live/online scene
    !!!     self-deconstruction to exactly this pacing: dataset_fps advances a
    !!!     simulated clock by each iteration's measured wall time, and slow
    !!!     ingestion iterations release many frames, which slows the next
    !!!     iteration further — a runaway that dumps the whole trajectory in
    !!!     ~1k iterations regardless of steps_per_frame. Full writeup:
    !!!     docs/2026-06-21-scene-explosion-isolation.md.
    !!!   * 12899c2 (2026-06-29, PR #20 — a RVizSplat transport/packaging PR)
    !!!     PUT THE BRANCH BACK VERBATIM and deleted 7ffaebc's warning comment,
    !!!     including its pointer to that doc. No commit message anywhere
    !!!     argues against 7ffaebc's finding; every hallmark of a bad rebase
    !!!     resolution. It went unnoticed for seven weeks, during which every
    !!!     --online_mode run — champion checkpoints included — ran the
    !!!     pre-fix pacing.
    !!!
    !!! The branch is retained TODAY as a deliberate decision (issue #302), not
    !!! as an endorsement: most of this project's streaming corpus was produced
    !!! under it, so a silent default flip would break comparability with every
    !!! recorded --online_mode number at once. Instead the choice is made
    !!! explicit — loudly warned at startup (ONLINE_MODE_DATASET_FPS_WARNING)
    !!! and recorded in report.json["ingestion"] — pending the confirmatory run
    !!! tracked in #302. Interim opt-out with no code change:
    !!! ``--streaming_ingestion_mode iter_based`` (an explicit mode
    !!! short-circuits before this branch).

    ``auto`` preserves offline replay reproducibility while making ROS2
    sources use dataset-timestamp pacing by default. File/dataset sources
    selected with ``online_mode`` also use dataset timestamp pacing so they
    are not tied to the optimizer iteration horizon.

    ``ros2://`` sources default to ``dataset_fps``, not ``wallclock_strict``:
    every current ROS2 launch path (``05_run_offline_rosbag.sh``) replays a
    recorded bag via ``ros2 bag play`` with no ``--rate`` flag, i.e. it is
    already paced to real-time by the bag player itself. ``wallclock_strict``
    on top of that is redundant AND actively drops frames whenever training
    transiently falls behind (e.g. a GPU compute spike, a slow disk write) —
    silently breaking the full-coverage contract. ``dataset_fps`` never
    drops frames and still lets the non-blocking drain loop train at full
    speed on the buffer between releases. Pass
    ``--streaming_ingestion_mode wallclock_strict`` explicitly for a
    genuinely untethered live-camera source that must never process a
    backlog older than real sensor time.
    """
    mode = (requested or INGESTION_MODE_AUTO).strip().lower()
    if mode not in INGESTION_MODES:
        # Preserve FrameScheduler's legacy fallback behavior for unknown values.
        return mode, INGESTION_REASON_UNKNOWN_VALUE
    if mode != INGESTION_MODE_AUTO:
        return mode, INGESTION_REASON_EXPLICIT
    source = str(source_path or "")
    # ros2:// is checked BEFORE online_mode purely for attribution: both land
    # on dataset_fps, so the returned MODE is identical either way, but for a
    # ros2:// source dataset_fps is the intended, documented default (see the
    # paragraph above) and must not trigger the online_mode warning.
    if source.startswith("ros2://"):
        return INGESTION_MODE_DATASET_FPS, INGESTION_REASON_ROS2_DEFAULT
    if online_mode:
        # See the WARNING block in this function's docstring before touching.
        return INGESTION_MODE_DATASET_FPS, INGESTION_REASON_ONLINE_MODE_DEFAULT
    return INGESTION_MODE_ITER_BASED, INGESTION_REASON_OFFLINE_DEFAULT


def resolve_ingestion_mode(
    source_path: str,
    requested: str | None = None,
    *,
    online_mode: bool = False,
) -> str:
    """Resolve the frame-release mode for a source (mode only).

    Thin wrapper over :func:`resolve_ingestion_mode_with_reason` — see that
    function's docstring for the full semantics and, importantly, the
    ``online_mode`` regression history (issue #302).
    """
    mode, _reason = resolve_ingestion_mode_with_reason(
        source_path, requested, online_mode=online_mode
    )
    return mode


# Emitted once at startup whenever both frame_stride and source_stride are > 1.
# Kept as a module constant so a future change cannot silently drop the notice
# the way 12899c2 dropped 7ffaebc's warning comment (issue #805).
COMPOSED_DECIMATION_NOTICE = (
    "[streaming] source-decimation composition: native stride={frame_stride} "
    "× generic stride={source_stride} = effective stride={effective_stride} "
    "(nominal source frame rate ÷ {effective_stride})"
)


class ConflictingStrideError(ValueError):
    """Raised when source sampling ownership cannot be uniquely resolved."""
    pass


# --streaming_source_sampling_mode values (issue #834). "single_owner" is the
# canonical default: exactly one owner (native reader or the generic
# StridedFrameSource wrapper) applies the requested stride, and a conflicting
# both-non-1-and-unequal request refuses to start rather than silently
# composing. "legacy_compose" is the opt-in escape hatch that restores the
# historical frame_stride * source_stride multiplication.
SOURCE_SAMPLING_MODE_SINGLE_OWNER = "single_owner"
SOURCE_SAMPLING_MODE_LEGACY_COMPOSE = "legacy_compose"
SOURCE_SAMPLING_MODES = {SOURCE_SAMPLING_MODE_SINGLE_OWNER, SOURCE_SAMPLING_MODE_LEGACY_COMPOSE}


def resolve_legacy_composition_mode(sampling_mode: str | None) -> bool:
    """Translate --streaming_source_sampling_mode into resolve_source_sampling_policy's
    ``legacy_composition_mode`` boolean.

    Fails LOUDLY and EARLY (a plain ``ValueError``, meant to surface at
    startup/argument-resolution time, before any frame source is built or any
    training happens) on any value other than the two recognized modes — a
    typo (e.g. ``legacy_composed``) must not silently fall back to
    single-owner semantics.
    """
    mode = str(sampling_mode) if sampling_mode is not None else SOURCE_SAMPLING_MODE_SINGLE_OWNER
    if mode not in SOURCE_SAMPLING_MODES:
        raise ValueError(
            f"--streaming_source_sampling_mode={mode!r} is not recognized. "
            f"Valid values: {sorted(SOURCE_SAMPLING_MODES)}."
        )
    return mode == SOURCE_SAMPLING_MODE_LEGACY_COMPOSE


def resolve_source_sampling_policy(
    frame_stride: int,
    source_stride: int,
    legacy_composition_mode: bool = False,
) -> dict:
    """Resolve which owner (native or generic wrapper) should apply source sampling.

    Issue #834: Stop accidental multiplication of independently-active stride knobs.
    Source sampling should be requested once, resolved once, applied once.

    Rules (canonical mode, legacy_composition_mode=False):
    1. both 1 -> no sampling needed
    2. exactly one non-1 -> that one is the owner
    3. both non-1 and equal -> native owner (if available; else generic)
    4. both non-1 and different -> raise ConflictingStrideError (unless legacy mode)

    Args:
        frame_stride:         Requested native stride (applied inside source constructors).
        source_stride:        Requested generic stride (applied via StridedFrameSource wrapper).
        legacy_composition_mode: If True, allow both to be non-1 and apply multiplication
                                (old behavior). If False (default), fail on conflict.

    Returns:
        Dict with keys:
        - effective_stride: The resolved stride to apply (1 if no sampling)
        - owner: "native" or "generic" (only set if effective_stride > 1)
        - requested_frame_stride: Input frame_stride
        - requested_source_stride: Input source_stride
        - legacy_composition_active: True iff legacy mode was used and applied
        - conflict: Conflict details if both non-1 and different, else None

    Raises:
        ConflictingStrideError: If both non-1 and different, and not in legacy mode.
    """
    frame_stride = max(1, int(frame_stride or 1))
    source_stride = max(1, int(source_stride or 1))

    result = {
        "effective_stride": 1,
        "owner": None,
        "requested_frame_stride": frame_stride,
        "requested_source_stride": source_stride,
        "legacy_composition_active": False,
        "conflict": None,
    }

    # Rule 1: both 1 -> no sampling
    if frame_stride == 1 and source_stride == 1:
        return result

    # Rule 2: exactly one non-1
    if frame_stride == 1:
        result["effective_stride"] = source_stride
        result["owner"] = "generic"
        return result

    if source_stride == 1:
        result["effective_stride"] = frame_stride
        result["owner"] = "native"
        return result

    # Rule 3: both non-1 and equal -> native owns it
    if frame_stride == source_stride:
        result["effective_stride"] = frame_stride
        result["owner"] = "native"
        return result

    # Rule 4: both non-1 and conflicting
    conflict_desc = f"frame_stride={frame_stride} vs source_stride={source_stride}"
    if legacy_composition_mode:
        # Legacy: multiply both
        result["effective_stride"] = frame_stride * source_stride
        result["owner"] = "both"
        result["legacy_composition_active"] = True
        result["conflict"] = conflict_desc
        return result
    else:
        # Canonical: raise
        raise ConflictingStrideError(
            f"Both --streaming_frame_stride ({frame_stride}) and "
            f"--streaming_source_stride ({source_stride}) are non-1 and unequal. "
            f"Source sampling must be applied by exactly one owner. "
            f"To use the historical multiplication behavior (stride={frame_stride * source_stride}), "
            f"pass --streaming_source_sampling_mode legacy_compose"
        )


def resolve_source_decimation(
    frame_stride: int,
    source_stride: int,
    frames_before: int | None = None,
    frames_after: int | None = None,
    frames_source: str = "pre_stride",
    legacy_composition_mode: bool = False,
) -> dict:
    """Resolve source-decimation metadata for report.json.

    Under issue #834 (canonical mode): exactly one owner applies sampling at the resolved
    effective stride. Under legacy_composition_mode: both may apply (old behavior, no longer
    default).

    Args:
        frame_stride:           Native stride applied inside dataset reader.
        source_stride:          Generic stride applied post-construction via StridedFrameSource.
        frames_before:          Frame count before source_stride was applied (or None).
        frames_after:           Frame count after source_stride was applied (or None).
        frames_source:          Provenance of frames_before (e.g. "pre_stride", "placeholder").
        legacy_composition_mode: Allow both strides to apply (old behavior).

    Returns:
        Dict with keys: ``frame_stride``, ``source_stride``, ``effective_stride``,
        ``composed`` (bool), ``frames_before_source_stride``, ``frames_after_source_stride``,
        ``frames_before_is_placeholder``, ``sampling_owner``, ``legacy_composition_active``,
        ``conflict``.

    Under canonical mode: ``effective_stride`` is the resolved single-owner stride.
    ``composed`` remains a boolean flag (True iff both inputs > 1) for backward compatibility,
    but should be read as "both were requested, not whether both were applied" — use
    ``legacy_composition_active`` to tell whether multiplication occurred.

    ``sampling_owner`` is "native", "generic", or None (if no sampling).
    ``conflict`` captures details if both inputs > 1 and unequal (for diagnostics).
    """
    frame_stride = max(1, int(frame_stride or 1))
    source_stride = max(1, int(source_stride or 1))

    # Use the new policy resolver to determine which owner applies sampling
    policy = resolve_source_sampling_policy(
        frame_stride,
        source_stride,
        legacy_composition_mode=legacy_composition_mode,
    )

    # Preserve all original 7 keys for backward compatibility
    composed = frame_stride > 1 and source_stride > 1
    effective_stride = policy["effective_stride"]

    return {
        "frame_stride": frame_stride,
        "source_stride": source_stride,
        "effective_stride": effective_stride,
        "composed": composed,  # backward compat: True iff both inputs > 1
        "frames_before_source_stride": frames_before,
        "frames_after_source_stride": frames_after,
        "frames_before_is_placeholder": frames_source == "placeholder",
        # New keys for #834
        "sampling_owner": policy["owner"],
        "legacy_composition_active": policy["legacy_composition_active"],
        "conflict": policy["conflict"],
    }


def resolve_source_sampling_application(
    frame_stride: int,
    source_stride: int,
    sampling_mode: str | None = None,
) -> dict:
    """Resolve a requested (frame_stride, source_stride, sampling_mode) triple
    into the CONCRETE strides each real owner should apply, so exactly one of
    them ever decimates a live run.

    This is the single production entry point on the actual sampling path
    (issue #834 review defect #1: the pure resolver alone was previously only
    reachable from report.json construction, never from the code that builds
    the frame source). The caller (``train_streaming.py``) is expected to:

      1. Set the native reader's stride knob (e.g. override
         ``args.streaming_frame_stride``) to ``result["native_stride_to_apply"]``
         before constructing the frame source.
      2. Apply the generic post-hoc wrapper (``StridedFrameSource``) at
         ``result["generic_stride_to_apply"]`` only if that value is > 1.

    Because both numbers are derived from the SAME resolved policy, at most
    one of them is ever > 1 outside of an explicit ``legacy_compose`` request
    (in which case both apply their own originally-requested stride,
    reproducing the historical multiplication behavior verbatim).

    Args:
        frame_stride:   Requested native stride (--streaming_frame_stride).
        source_stride:  Requested generic stride (--streaming_source_stride).
        sampling_mode:  --streaming_source_sampling_mode string (or None for
                         the default, "single_owner").

    Returns:
        Dict with keys ``policy`` (the raw ``resolve_source_sampling_policy``
        result), ``native_stride_to_apply``, ``generic_stride_to_apply``,
        ``legacy_composition_mode`` (the resolved bool), and
        ``requested_frame_stride`` / ``requested_source_stride`` (normalized,
        >= 1 copies of the inputs, for callers that want to report on exactly
        what was resolved without re-deriving the normalization).

    Raises:
        ValueError: If ``sampling_mode`` is not a recognized value.
        ConflictingStrideError: If both strides are non-1 and unequal, and
            ``sampling_mode`` is not ``"legacy_compose"``.
    """
    frame_stride = max(1, int(frame_stride or 1))
    source_stride = max(1, int(source_stride or 1))
    legacy_composition_mode = resolve_legacy_composition_mode(sampling_mode)

    policy = resolve_source_sampling_policy(
        frame_stride,
        source_stride,
        legacy_composition_mode=legacy_composition_mode,
    )

    owner = policy["owner"]
    if owner == "generic":
        native_stride_to_apply = 1
        generic_stride_to_apply = policy["effective_stride"]
    elif owner == "both":
        # legacy_compose: preserve the historical multiplication behavior
        # verbatim -- both owners apply their own originally-requested stride.
        native_stride_to_apply = frame_stride
        generic_stride_to_apply = source_stride
    elif owner == "native":
        native_stride_to_apply = policy["effective_stride"]
        generic_stride_to_apply = 1
    else:  # owner is None -> no sampling requested
        native_stride_to_apply = 1
        generic_stride_to_apply = 1

    return {
        "policy": policy,
        "native_stride_to_apply": native_stride_to_apply,
        "generic_stride_to_apply": generic_stride_to_apply,
        "legacy_composition_mode": legacy_composition_mode,
        "requested_frame_stride": frame_stride,
        "requested_source_stride": source_stride,
    }


class FrameScheduler:
    RESUME_SCHEMA = "frame_scheduler_v1"

    def __init__(
        self,
        fps: float = 30.0,
        steps_per_frame: int = 50,
        wallclock: bool = False,
        ingestion_mode: str = INGESTION_MODE_ITER_BASED,
        fps_cap: float = 30.0,
    ):
        self.fps = float(fps)
        self.steps_per_frame = max(1, int(steps_per_frame))
        # ingestion_mode overrides legacy wallclock bool
        if ingestion_mode == INGESTION_MODE_ITER_BASED:
            self._mode = INGESTION_MODE_ITER_BASED
        elif ingestion_mode == INGESTION_MODE_DATASET_FPS:
            self._mode = INGESTION_MODE_DATASET_FPS
        elif ingestion_mode == INGESTION_MODE_WALLCLOCK_STRICT:
            self._mode = INGESTION_MODE_WALLCLOCK_STRICT
        else:
            # backwards-compat: honour old wallclock kwarg
            self._mode = INGESTION_MODE_WALLCLOCK_STRICT if wallclock else INGESTION_MODE_ITER_BASED

        self._fps_cap = max(float(fps_cap), 0.1)
        self._next_frame_idx: int = 0
        # Adaptive-cadence override (iter_based mode only, issue #109 E3 arm).
        # None (default) = the plain steps_per_frame formula below governs
        # every release, unchanged from all prior behavior. Once a caller
        # invokes set_next_interval(), should_release() switches to a
        # per-release absolute target iteration instead of the
        # iteration // steps_per_frame formula. This is a deliberate design
        # choice, not an oversight: steps_per_frame // iteration is a GLOBAL
        # formula over the whole elapsed iteration count, so mutating
        # self.steps_per_frame mid-run would retroactively change the
        # expected release count for every past interval too (e.g. lowering
        # it after frame 10 could make should_release() decide several more
        # frames were already "due" and release them all at once). Tracking
        # an explicit absolute next-release iteration avoids that entirely
        # and lets each interval be sized independently.
        self._next_release_iter: int | None = None

        # Simulated clock (dataset_fps mode)
        self._sim_time: float = 0.0          # virtual seconds elapsed
        self._last_loop_time: float | None = None

        # True wall-clock (wallclock_strict mode)
        self._start_time: float | None = None
        self._frames_dropped: int = 0
        # Live counters for backlog / drain-rate diagnostics.
        self._frames_mapped: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def should_release(self, iteration: int, dt: float | None = None) -> bool:
        """Return True if a new frame should be ingested at this iteration.

        Args:
            iteration: Current training iteration (1-based).
            dt:        Wall-clock seconds this iteration took (used only in
                       dataset_fps mode to advance the simulated clock).
        """
        if self._mode == INGESTION_MODE_ITER_BASED:
            if self._next_release_iter is not None:
                return iteration >= self._next_release_iter
            # How many frames should have been released by this iteration?
            # iter=1 always releases the first frame; after that, one per steps_per_frame.
            expected = 1 + max(0, iteration // self.steps_per_frame)
            return self._next_frame_idx < expected

        if self._mode == INGESTION_MODE_DATASET_FPS:
            if dt is not None:
                self._sim_time += dt
            due_time = self._next_frame_idx / self._fps_cap
            return self._sim_time >= due_time

        # wallclock_strict
        if self._start_time is None:
            self._start_time = time.perf_counter()
        now = time.perf_counter()
        target = self._start_time + self._next_frame_idx / max(self._fps_cap, 1e-6)
        return now >= target

    def mark_released(self) -> None:
        """Call once per actually ingested frame."""
        if self._mode == INGESTION_MODE_WALLCLOCK_STRICT:
            # Detect and count dropped frames
            if self._start_time is not None:
                now = time.perf_counter()
                elapsed = now - self._start_time
                expected_idx = int(elapsed * self._fps_cap)
                skipped = max(0, expected_idx - self._next_frame_idx - 1)
                self._frames_dropped += skipped
                if skipped > 0:
                    self._next_frame_idx += skipped  # skip to current
        self._next_frame_idx += 1

    def set_next_interval(self, iteration: int, steps: int) -> None:
        """Force the next release to fire at ``iteration + max(1, steps)``.

        Opt-in override for iter_based mode (issue #109 E3 adaptive-exposure
        arm): call once after each ``mark_released()`` with the caller's own
        per-frame step count (e.g. scaled by that frame's inserted-Gaussian
        count) to make cadence responsive to a live signal instead of the
        fixed ``steps_per_frame`` formula. Never called => behavior is
        byte-for-byte identical to before this method existed. Has no effect
        outside iter_based mode (dataset_fps/wallclock_strict are paced by
        real/simulated time, not iteration count, so an iteration-relative
        override is meaningless there).
        """
        self._next_release_iter = iteration + max(1, int(steps))

    def tick(self, dt: float) -> None:
        """Advance simulated clock by dt seconds (dataset_fps mode only).
        Alternative to passing dt to should_release().
        """
        if self._mode == INGESTION_MODE_DATASET_FPS:
            self._sim_time += dt

    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped

    @property
    def frames_released(self) -> int:
        """Total number of frames released (or attempted) by the scheduler.

        Includes frames dropped in ``wallclock_strict`` mode (those
        ``should_release`` skipped because training fell behind). This is
        the right number to use as a "released but not consumed" gauge.
        """
        return self._next_frame_idx

    def mark_mapped(self) -> None:
        """Call once per frame that the drain loop actually maps.

        Lets the caller compute a live backlog =
        ``frames_released - frames_mapped`` without exposing the
        scheduler's private counters.
        """
        self._frames_mapped += 1

    @property
    def frames_mapped(self) -> int:
        return self._frames_mapped

    @property
    def simulated_time(self) -> float:
        return self._sim_time

    @property
    def wallclock(self) -> bool:
        return self._mode == INGESTION_MODE_WALLCLOCK_STRICT

    @property
    def mode(self) -> str:
        return self._mode

    def resume_state_dict(self) -> dict:
        """Serialize the exact future-affecting iter-based release state."""
        if self._mode != INGESTION_MODE_ITER_BASED:
            raise ValueError("exact resume supports only iter_based frame scheduling")
        return {
            "owner_schema": self.RESUME_SCHEMA,
            "fps": self.fps,
            "steps_per_frame": self.steps_per_frame,
            "mode": self._mode,
            "fps_cap": self._fps_cap,
            "next_frame_idx": self._next_frame_idx,
            "next_release_iter": self._next_release_iter,
            "sim_time": self._sim_time,
            "last_loop_time": self._last_loop_time,
            "start_time": self._start_time,
            "frames_dropped": self._frames_dropped,
            "frames_mapped": self._frames_mapped,
        }

    def load_resume_state_dict(self, payload, resolver=None) -> None:
        del resolver
        keys = {
            "owner_schema", "fps", "steps_per_frame", "mode", "fps_cap",
            "next_frame_idx", "next_release_iter", "sim_time", "last_loop_time",
            "start_time", "frames_dropped", "frames_mapped",
        }
        if not isinstance(payload, dict) or set(payload) != keys:
            raise ValueError("frame scheduler exact-resume state keys mismatch")
        if payload["owner_schema"] != self.RESUME_SCHEMA:
            raise ValueError("unsupported frame scheduler exact-resume state")
        if payload["mode"] != INGESTION_MODE_ITER_BASED:
            raise ValueError("exact resume supports only iter_based frame scheduling")
        expected = (self.fps, self.steps_per_frame, self._mode, self._fps_cap)
        realized = (
            payload["fps"], payload["steps_per_frame"], payload["mode"],
            payload["fps_cap"],
        )
        if realized != expected:
            raise ValueError("frame scheduler configuration mismatch")
        for key in ("next_frame_idx", "frames_dropped", "frames_mapped"):
            value = payload[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"frame scheduler {key} must be a non-negative integer")
        next_release = payload["next_release_iter"]
        if next_release is not None and (
            isinstance(next_release, bool) or not isinstance(next_release, int)
            or next_release < 0
        ):
            raise ValueError("invalid frame scheduler next release iteration")
        if payload["frames_mapped"] > payload["next_frame_idx"]:
            raise ValueError("frame scheduler mapped count exceeds released count")
        if payload["frames_dropped"] != 0:
            raise ValueError("iter_based scheduler cannot contain dropped frames")
        if payload["sim_time"] != 0.0 or payload["last_loop_time"] is not None \
                or payload["start_time"] is not None:
            raise ValueError("iter_based scheduler contains wall-time state")
        self._next_frame_idx = payload["next_frame_idx"]
        self._next_release_iter = next_release
        self._sim_time = 0.0
        self._last_loop_time = None
        self._start_time = None
        self._frames_dropped = 0
        self._frames_mapped = payload["frames_mapped"]
