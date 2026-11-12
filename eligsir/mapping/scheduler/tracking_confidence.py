"""Tracking-confidence keyframe-admission signal (design 2026-07-30).

Mirrors ``frame_quality.py``'s adaptive blur/exposure design exactly: a
causal (no future-peeking), self-disabling soft weight -- never a hard
veto -- so a scene with uniformly confident tracking pays no penalty
anywhere, and only an anomalously low-confidence frame (relative to its
OWN trajectory's typical tracking quality so far) gets downweighted.

Why this exists (2026-07-30 investigation): the floor3/kitchen1 champion
configs both report ``input_quality.tracking_available=False`` -- there is
currently NO tracking-confidence signal reaching keyframe admission at all
for the ros2://live + ORB-SLAM3-external-pose deployment (only the
Open3D-odometry frontend populates ``_odom_stats``/``OdometryStats``, a
different, unused-for-these-scenes code path). advisor's ranking of what
makes a frame good for 3DGS training put pose/tracking confidence ABOVE
blur/exposure ("an uncertain-pose frame poisons the map worse than a
blurry-but-well-posed one") -- this module is the highest-ranked
currently-unexploited signal.

Distinct from the existing motion pregate (see CLAUDE.md's
motion-pregate-silent-admission-corruption memory): the pregate is a HARD
reject for frames on stale/interpolated poses under sparse GT pose -- that
remains correct and is NOT touched here. This module targets a different,
currently-invisible failure mode: ORB-SLAM3 reports state=OK (a fresh,
non-interpolated pose IS available) but with unusually few tracked
inliers (e.g. a thin-texture or motion-blurred frame degrading tracking
quality without losing tracking outright). A frame where ORB-SLAM3 is
RECENTLY_LOST/LOST already tends to fail admission via the existing
pose_delta/rotation_delta/gap terms (a stalled tracker publishes a frozen
or near-frozen pose, which naturally reads as "no motion" to the existing
scorer) or the pregate itself if the pose is flagged stale -- this module
is NOT meant to duplicate that.

UPSTREAM SIGNAL (2026-07-30, IMPLEMENTED + VERIFIED): ORB-SLAM3's tracking
state (``System::GetTrackingState()``) and tracked/inlier point count
(``System::GetTrackedMapPoints()`` size) are now published by
``rgbd_inertial_node_cpp`` in the sibling ``eligsir-orbslam3`` repo
(patches/apply_patch.py, commit b45a57b on branch
feat/slam-pose-convention) on a companion topic ``/slam/tracking_status``
(``std_msgs/Int32MultiArray``, payload ``[tracking_state,
num_tracked_points]``), published atomically -- same ``GrabRGBD``
callback, always immediately after ``/slam/pose``, every tracked frame,
unconditionally. Deliberately an UNSTAMPED second topic paired by FIFO
ARRIVAL ORDER (not ApproximateTime/ExactTime stamp-matching) -- reliable
QoS on a single publisher->subscriber topic pair preserves per-topic FIFO
order, and status is always emitted immediately before the pose it
belongs to, so this is an exact pairing, not an approximation; avoids the
sync-pairing bug class this project already hit once for RGB-D
color/depth. Verified end-to-end in a real (non-mocked) build/run: built
cleanly via colcon against the jazzy-orbbec-v3 image's existing
dependencies; the compiled binary references both topics (confirmed via
``strings``); the real node (loaded ORB vocabulary + settings) advertises
both topics with correct types under ``--network host``; fed synthetic
RGB-D+IMU frames via a throwaway rclpy publisher and confirmed real
``[state, num_tracked]`` data on ``/slam/tracking_status`` via ``ros2
topic echo``; a standalone reproduction of the exact FIFO-pairing
algorithm below paired 58/58 real pose messages with zero unpaired poses
and zero leftover queue entries.

CORRECTED 2026-08-02: the FIFO-pairing scheme described above/below was
verified correct on the PUBLISHER side (58/58 in a synthetic reproduction)
but never verified on the live CONSUMER side end-to-end -- a real
RealSense-bag smoke test showed the pose subscription callback
consistently firing before the tracking-status subscription callback for
the same frame, so the FIFO queue was always empty when popped and
``tracking_weight`` silently stayed ``None`` on every admitted frame
despite ``--streaming_tracking_confidence_enabled``. Two independently
scheduled ROS2 subscription callbacks on two different topics have no
cross-topic ordering guarantee (especially over a Zenoh RMW bridge), even
though the frontend always publishes them back-to-back in the same
callback. Fixed by stamping the tracking-status companion message with
the SAME frame timestamp ``/slam/pose`` uses and joining by exact stamp
key instead -- see ``eligsir/io/sources/ros2_live.py``'s
``tracking_status_by_stamp``/``_stamp_key`` and
``eligsir-orbslam3``'s ``RGBD_NODE_CPP`` (now publishes
``diagnostic_msgs/DiagnosticArray`` on ``/slam/tracking_status_stamped``).

Python consumer: ``eligsir/io/sources/ros2_live.py``'s
``ROS2LiveRGBDSource`` subscribes to the tracking-status companion topic
(derived from ``pose_topic``, default ``/slam/tracking_status_stamped``),
joins each incoming pose with the tracking-status entry sharing its exact
stamp (``on_pose``/``on_tracking_status``,
``state["tracking_status_by_stamp"]``), and attaches the result as
``frame._tracking_status`` on the emitted ``RGBDFrame`` -- ``None``
(byte-identical no-op) whenever the frontend build predates this patch or
the pose source isn't ORB-SLAM3.
train_streaming.py's keyframe-admission scoring reads that attribute (see
the ``streaming_tracking_confidence_enabled`` block there) and, when
present, folds ``tracking_confidence_weight()`` into the existing
blur/exposure ``quality_weight`` multiplicatively.

Still open: this fold-in only fires when
``--streaming_keyframe_quality_enabled`` is also on (a known coupling,
not yet decoupled -- see train_streaming.py's comment at the call site).

REAL BLOCKER (2026-07-30, corrected framing -- was previously misstated
as "waiting on advisor sign-off for a GPU validation run"; that framing
was wrong and is not the actual blocker): a GPU training run on
floor3/kitchen1 would NOT be informative. Both scenes are well-tracked
by construction -- their ~22-23dB plateau comes from something other
than tracking quality (see this session's investigation ruling out
geometry-shape, loop-closure, and iteration-count as the limiter) -- and
this mechanism is DESIGNED to self-disable (causal EMA ratio, weight
stays ~1.0) whenever tracking is uniformly good. A validation run there
is guaranteed to show ~zero delta by construction, not because the
mechanism doesn't work: the exact same dead-end already hit and
correctly declined for the blur/exposure mechanism on ScanNet++ (that
session's cached scenes turned out to not be blurry enough to exercise
the mechanism either). The real blocker is DATA, not review: there is no
genuinely tracking-degraded scene (real relocalization events, real
sustained low-inlier-count stretches, not just brief per-frame
NOT_INITIALIZED/RECENTLY_LOST blips) in this project's current scene
inventory to validate against. Finding or capturing one is the
prerequisite for any informative validation run -- flagged as an open
data-inventory gap, not a code or design gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Mirrors ORB-SLAM3's Tracking::eTrackingState enum values.
TRACKING_STATE_SYSTEM_NOT_READY = -1
TRACKING_STATE_NO_IMAGES_YET = 0
TRACKING_STATE_NOT_INITIALIZED = 1
TRACKING_STATE_OK = 2
TRACKING_STATE_RECENTLY_LOST = 3
TRACKING_STATE_LOST = 4
TRACKING_STATE_OK_KLT = 5

# States in which the pose itself is still considered a fresh, genuine
# tracking result worth scoring on a graduated inlier-count basis (as
# opposed to a frozen/estimated pose that other admission terms, or the
# motion pregate, already handle).
_OK_STATES = frozenset({TRACKING_STATE_OK, TRACKING_STATE_OK_KLT})
# A soft (not zero) floor for RECENTLY_LOST -- graduated per the project's
# "never a hard veto, avoid coverage holes" convention. LOST/NOT_INITIALIZED
# are treated the same way: heavily but not fully discounted, since the
# existing pose-delta/pregate machinery is expected to catch most of the
# actually-bad cases via the frozen/stale pose itself.
_RECENTLY_LOST_FLOOR = 0.3
_LOST_FLOOR = 0.15


@dataclass(frozen=True)
class TrackingStatus:
    """One frame's raw ORB-SLAM3 tracking readout (once the frontend emits it)."""

    state: int
    num_tracked: int


class RunningTrackingStats:
    """Causal EMA tracker for the adaptive tracking-confidence path.

    Same shape as ``frame_quality.RunningBlurExposureStats``: updated once
    per OBSERVED frame (every frame reaching admission scoring, not just
    admitted ones), in ingestion order; ``typical`` stays ``None`` (no
    penalty applied) until ``warmup`` observations, so early frames on a
    live camera aren't judged against a near-meaningless 1-sample average.
    """

    def __init__(self, alpha: float = 0.05, warmup: int = 20):
        self._alpha = float(alpha)
        self._warmup = int(warmup)
        self._n = 0
        self._tracked_ema: Optional[float] = None

    def update(self, num_tracked: int) -> None:
        """Update the EMA with the current number of tracked landmarks.

        Args:
            num_tracked: Number of landmarks successfully tracked this frame.
        """
        self._n += 1
        val = float(num_tracked)
        if self._tracked_ema is None:
            self._tracked_ema = val
        else:
            a = self._alpha
            self._tracked_ema = (1.0 - a) * self._tracked_ema + a * val

    @property
    def typical(self) -> Optional[float]:
        """EMA of tracked landmark count after warmup, None if still warming up."""
        return self._tracked_ema if self._n >= self._warmup else None


class MapReinitTracker:
    """Detects a genuine mid-run ORB-SLAM3 Atlas map re-initialization and
    exposes a PERMANENT hard veto for the rest of the run (Phase 2c,
    2026-08-06; made permanent 2026-08-06 -- see below).

    Distinct from and independent of ``tracking_confidence_weight``'s soft,
    graduated weighting above: a second, post-bootstrap occurrence of
    NOT_INITIALIZED/NO_IMAGES_YET (after tracking has already reached OK at
    least once) means ORB-SLAM3 created a brand-new, independently-
    originated map -- its poses have no defined relationship to the
    existing trajectory's coordinate frame, which is a data-integrity
    issue, not a confidence gradient. See ``train_streaming.py``'s
    ``_check_tracking_reinit`` (now a thin wrapper around this class) for
    the real root-cause writeup: a real floor2 session's raw trajectory
    came out mirrored relative to a clean session, traced to exactly this
    event via ``orbslam_live.log``'s ``"Creation of new map with id: 1"``.

    CORRECTED 2026-08-06: this class used to auto-resume admission (clear
    ``vetoing``) after ``settle_frames`` consecutive OK-tracking frames on
    the NEW map. That was wrong: ``settle_frames`` of clean tracking only
    proves the new Atlas map is internally self-consistent -- it says
    NOTHING about whether the new map shares a coordinate frame with the
    old one (there is no alignment/relocalization step here, by
    construction of the re-init this class detects). Resuming admission on
    that basis silently glued the new map's poses onto the old trajectory
    with zero verified alignment -- exactly the mechanism that produced
    the real mirrored/corrupted floor2 trajectory this class exists to
    catch in the first place. The veto is now permanent for the remainder
    of the run, matching ``OrbslamLogReinitWatcher`` (which was already
    permanent -- there was never a principled reason for the two detectors
    of the SAME event class to disagree on this point; the settle-and-
    resume behavior here was simply the original, less conservative
    design, not an intentional distinction). ``settle_frames`` is kept as
    a constructor argument (now unused, retained only so existing callers
    don't need updating) and consecutive-OK bookkeeping is kept internally
    for the informational ``ok_streak`` value, but it no longer clears
    ``vetoing`` -- once tripped, only a fresh instance (a new run) is
    un-vetoed.

    Call :meth:`observe` once per frame carrying a ``TrackingStatus``
    (skip frames with none -- non-ORB-SLAM3 sources or an unpatched
    frontend build never call this at all). While :attr:`vetoing` is
    True, the caller must reject admission unconditionally, for the rest
    of the run.
    """

    def __init__(self, settle_frames: int = 30):
        self.settle_frames = int(settle_frames)
        self.ever_ok = False
        self.vetoing = False
        self.ok_streak = 0
        self.events: list = []  # [{"frame": int, "recovered_at_frame": None}]

    def observe(self, status: TrackingStatus, frame_index: int) -> Optional[str]:
        """Update state from one frame's tracking status.

        Returns ``"reinit_started"`` the call a genuine re-init is first
        detected, else None. Once returned, ``self.vetoing`` stays True for
        every subsequent call on this instance -- there is no
        ``"reinit_settled"`` event anymore (see class docstring). The
        caller should always check ``self.vetoing`` directly for the
        actual admission decision.
        """
        if status.state in _OK_STATES:
            self.ever_ok = True
            if not self.vetoing:
                self.ok_streak = 0
            # Once vetoing, tracking quality on the new map is irrelevant
            # to the admission decision (permanent veto) -- streak is no
            # longer load-bearing, but keep it monotonically informative
            # rather than silently freezing it.
            elif self.ok_streak < self.settle_frames:
                self.ok_streak += 1
            return None
        if status.state in (TRACKING_STATE_NOT_INITIALIZED, TRACKING_STATE_NO_IMAGES_YET):
            if self.vetoing:
                self.ok_streak = 0
                return None
            if self.ever_ok:
                self.vetoing = True
                self.ok_streak = 0
                self.events.append({"frame": int(frame_index), "recovered_at_frame": None})
                return "reinit_started"
            # else: normal one-time startup lifecycle (ever_ok still False) -- no-op.
            return None
        # RECENTLY_LOST/LOST/SYSTEM_NOT_READY: not itself a reinit signal.
        if self.vetoing:
            self.ok_streak = 0
        return None


class OrbslamLogReinitWatcher:
    """Fallback map-reinit detector that tails ``orbslam_live.log`` directly,
    independent of ``MapReinitTracker``/``/slam/tracking_status_stamped``.

    Why this exists (2026-08-06): ``MapReinitTracker`` only ever sees a
    re-init if ``frame._tracking_status`` is populated, which requires the
    ORB-SLAM3 wrapper to publish ``/slam/tracking_status_stamped`` -- CLAUDE.md
    already documents this topic as confirmed silently absent on at least
    one deployed ``eligsir-orbslam3`` image build. On such a build,
    ``MapReinitTracker.observe()`` is called with ``tstatus=None`` every
    single frame and never fires -- a real, log-confirmed re-init produces
    ZERO warning anywhere during the run; the only thing that ever catches
    it is ``human_launch_scripts/05_run_offline_rosbag.sh``'s post-hoc
    ``grep -c "Creation of new map with id:"`` check, which only runs AFTER
    the full run completes. That means a long/live run gives its operator
    no indication of trajectory corruption until it's already over (or
    never, if the post-hoc check isn't inspected). This class closes that
    gap by tailing the same log file the shell script greps, from inside
    the training loop itself, so the SAME warning fires live and the SAME
    ``MapReinitTracker.vetoing`` hard-veto engages immediately -- not
    contingent on the ROS2 topic ever being fixed.

    Deliberately does NOT attempt automatic recovery/settling the way the
    topic-based path does (``settle_frames`` consecutive OK frames): there
    is no reliable single-line log signal for "tracking has been cleanly
    OK for N consecutive frames." Once detected, the veto this class drives
    is permanent for the rest of the run -- the conservative, correct
    choice for a genuine data-integrity issue (see ``MapReinitTracker``'s
    own docstring: a second map's poses have no defined relationship to the
    first map's coordinate frame). If the ROS2-topic path is later fixed on
    a given deployment, both detectors run concurrently and harmlessly --
    they drive the same boolean.
    """

    _CREATION_MARKER = "Creation of new map with id:"

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._offset = 0
        self.map_creation_count = 0
        self.detected = False

    def check(self) -> bool:
        """Read any new bytes appended to the log since the last call and
        update ``map_creation_count``. Returns True exactly once -- on the
        call where the count first exceeds 1 -- and False every other time
        (including all calls before that transition and all calls after).
        Safe to call every frame: cheap (seek + read new bytes only), and
        silently no-ops if the log doesn't exist yet (written by a sibling
        container this process doesn't control the startup order of)."""
        if self.detected:
            return False
        try:
            with open(self.log_path, "r", errors="replace") as f:
                f.seek(self._offset)
                new_text = f.read()
                self._offset = f.tell()
        except OSError:
            return False
        if new_text:
            self.map_creation_count += new_text.count(self._CREATION_MARKER)
        if self.map_creation_count > 1 and not self.detected:
            self.detected = True
            return True
        return False


def tracking_confidence_weight(
    status: Optional[TrackingStatus],
    *,
    running_typical: Optional[float] = None,
    floor_weight: float = 0.4,
) -> float:
    """Soft [0, 1] multiplicative weight -- never a hard veto.

    ``status is None`` (no tracking signal available, e.g. current default
    with the frontend not yet emitting it, or a non-ORB-SLAM3 pose source)
    returns 1.0 unconditionally: byte-identical to today's behavior.

    State handling: OK/OK_KLT are scored on a graduated, causal,
    scene-relative inlier-count ramp (mirrors
    ``evaluate_frame_quality``'s adaptive blur ramp exactly -- no penalty
    until ``running_typical`` warms up, and even then the floor is
    ``floor_weight``, never 0). RECENTLY_LOST/LOST/NOT_INITIALIZED get a
    fixed, still-nonzero discount rather than a ramp (there is no
    meaningful "typical inlier count" to compare against when tracking
    itself is degraded/absent).
    """

    if status is None:
        return 1.0
    if status.state in _OK_STATES:
        if running_typical is None or running_typical <= 0:
            return 1.0  # cold start / no baseline yet: no penalty
        ratio = float(status.num_tracked) / float(running_typical)
        # Only frames BELOW their own trajectory's typical inlier count are
        # penalized; at or above typical, weight is 1.0 (mirrors the blur
        # ramp's floor->typical mapping, generalized to a ratio so it's
        # scale-free across scenes with very different absolute inlier
        # counts -- avoids the exact per-scene-absolute-threshold tuning
        # trap CLAUDE.md's self-disable constraint warns against).
        return float(np.clip(ratio, floor_weight, 1.0))
    if status.state == TRACKING_STATE_RECENTLY_LOST:
        return _RECENTLY_LOST_FLOOR
    if status.state in (TRACKING_STATE_NOT_INITIALIZED, TRACKING_STATE_NO_IMAGES_YET):
        # Fires at the start of every sequence/re-localization (map not yet
        # bootstrapped) -- verified via a real end-to-end run (2026-07-30,
        # real kitchen1 RGB-D frames + IMU excitation fed through the
        # patched frontend): state cycles 1 (NOT_INITIALIZED, ~6 frames) ->
        # 2 (OK, the vast majority once the map initializes) -> occasional
        # brief 3 (RECENTLY_LOST) dips, exactly the expected lifecycle.
        # Explicit (not the LOST/SYSTEM_NOT_READY catch-all) because these
        # states are EXPECTED/transient at sequence start, not necessarily a
        # sign of degraded input -- same fixed nonzero floor as LOST for now
        # (no signal to ramp against pre-init), but named separately so a
        # future revision can treat "still bootstrapping" differently from
        # "lost after having tracked" without re-deriving this from scratch.
        return _LOST_FLOOR
    # LOST, SYSTEM_NOT_READY.
    return _LOST_FLOOR
