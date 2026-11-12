from __future__ import annotations

from eligsir.mapping.scheduler.tracking_confidence import (
    MapReinitTracker,
    OrbslamLogReinitWatcher,
    RunningTrackingStats,
    TrackingStatus,
    TRACKING_STATE_LOST,
    TRACKING_STATE_NO_IMAGES_YET,
    TRACKING_STATE_NOT_INITIALIZED,
    TRACKING_STATE_OK,
    TRACKING_STATE_OK_KLT,
    TRACKING_STATE_RECENTLY_LOST,
    tracking_confidence_weight,
)


def test_enum_values_match_real_orb_slam3_header():
    # Verified 2026-07-30 against the actual Tracking.h in the deployed
    # ORB-SLAM3 fork image (the image reference is intentionally external to
    # this publication export):
    # enum eTrackingState { SYSTEM_NOT_READY=-1, NO_IMAGES_YET=0,
    # NOT_INITIALIZED=1, OK=2, RECENTLY_LOST=3, LOST=4, OK_KLT=5 }.
    # This is the standard upstream ORB-SLAM3 ordering, confirmed to hold
    # unmodified in this fork -- NOT assumed. OK=2, not 1: a wrong guess
    # here (e.g. OK=1) would invert the mechanism (reward bad tracking).
    assert TRACKING_STATE_NO_IMAGES_YET == 0
    assert TRACKING_STATE_NOT_INITIALIZED == 1
    assert TRACKING_STATE_OK == 2
    assert TRACKING_STATE_RECENTLY_LOST == 3
    assert TRACKING_STATE_LOST == 4
    assert TRACKING_STATE_OK_KLT == 5


def test_not_initialized_and_no_images_yet_get_explicit_soft_floor():
    # These fire at the start of every sequence -- verified via a real
    # end-to-end run (real kitchen1 RGB-D frames + IMU excitation through
    # the patched frontend): state cycled 1 -> 2 -> occasional 3, exactly
    # this lifecycle. Must not be treated as "no signal" (1.0) nor crash.
    assert tracking_confidence_weight(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0)) == 0.15
    assert tracking_confidence_weight(TrackingStatus(TRACKING_STATE_NO_IMAGES_YET, 0)) == 0.15


def test_no_signal_is_byte_identical_noop():
    assert tracking_confidence_weight(None) == 1.0


def test_cold_start_no_penalty():
    status = TrackingStatus(state=TRACKING_STATE_OK, num_tracked=5)
    assert tracking_confidence_weight(status, running_typical=None) == 1.0


def test_ok_state_at_or_above_typical_is_unweighted():
    status = TrackingStatus(state=TRACKING_STATE_OK, num_tracked=100)
    assert tracking_confidence_weight(status, running_typical=80.0) == 1.0


def test_ok_state_below_typical_ramps_down_but_never_to_zero():
    status = TrackingStatus(state=TRACKING_STATE_OK, num_tracked=10)
    weight = tracking_confidence_weight(status, running_typical=100.0, floor_weight=0.4)
    assert abs(weight - 0.4) < 1e-9


def test_recently_lost_and_lost_are_soft_not_hard_veto():
    assert tracking_confidence_weight(TrackingStatus(TRACKING_STATE_RECENTLY_LOST, 0)) == 0.3
    assert tracking_confidence_weight(TrackingStatus(TRACKING_STATE_LOST, 0)) == 0.15


def test_running_stats_warmup_gating():
    stats = RunningTrackingStats(warmup=5)
    assert stats.typical is None
    for _ in range(4):
        stats.update(50)
    assert stats.typical is None
    stats.update(50)
    assert stats.typical is not None


def test_running_stats_is_causal_ema():
    stats = RunningTrackingStats(alpha=0.5, warmup=1)
    stats.update(100)
    assert stats.typical == 100.0
    stats.update(0)
    assert stats.typical == 50.0


# ---------------------------------------------------------------------------
# MapReinitTracker (Phase 2c, 2026-08-06)
# ---------------------------------------------------------------------------

def test_startup_lifecycle_never_vetoes():
    """The expected one-time startup lifecycle (NOT_INITIALIZED for a few
    frames, then OK) must never trigger a veto -- MapReinitTracker only
    reacts to NOT_INITIALIZED/NO_IMAGES_YET AFTER tracking has already
    reached OK once. ever_ok=False guards exactly this."""
    tracker = MapReinitTracker(settle_frames=5)
    for _ in range(6):
        assert tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 0) is None
        assert not tracker.vetoing
    assert tracker.observe(TrackingStatus(TRACKING_STATE_OK, 100), 6) is None
    assert not tracker.vetoing
    assert tracker.events == []


def test_real_floor2_reinit_sequence_vetoes_permanently():
    """Replays the real event sequence from a floor2 session that produced
    a mirrored trajectory (output/phase2b_repro_floor2_v5/orbslam_live.log):
    normal startup -> well-tracked for a while -> 'TRACK_REF_KF: Less than
    15 matches!!' -> 'Creation of new map with id: 1' (a genuine
    post-bootstrap re-init). Exact per-frame num_tracked counts from that
    session were not captured (input_quality read back None that run --
    the topic wasn't reaching the consumer at all, root-caused
    separately), so this uses representative values for the same real
    state-transition sequence, which is what the tracker actually keys on.

    CORRECTED 2026-08-06: MapReinitTracker no longer auto-resumes admission
    after a settle window of clean OK-tracking frames on the new map --
    settle_frames of clean tracking only proves the NEW map tracks
    cleanly, it says nothing about whether it shares a coordinate frame
    with the OLD map, so resuming on that basis re-admitted unaligned
    poses (the exact failure this detector exists to prevent). The veto is
    now permanent for the rest of the run, no matter how much subsequent
    OK tracking is observed.
    """
    tracker = MapReinitTracker(settle_frames=30)

    # Normal startup lifecycle.
    for _ in range(6):
        tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 0)
    # Well-tracked for a long stretch (this is what "ever_ok" latches on).
    for i in range(200):
        event = tracker.observe(TrackingStatus(TRACKING_STATE_OK, 300), i)
        assert event is None
        assert not tracker.vetoing

    # 'TRACK_REF_KF: Less than 15 matches!!' -> tracking lost -> Atlas
    # creates a brand-new map ('Creation of new map with id: 1').
    event = tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 200)
    assert event == "reinit_started"
    assert tracker.vetoing
    assert tracker.events == [{"frame": 200, "recovered_at_frame": None}]

    # While vetoing, admission must stay rejected regardless of tracking
    # quality on individual frames (a caller checks tracker.vetoing, not
    # the return value, for the actual admission decision).
    for i in range(201, 210):
        tracker.observe(TrackingStatus(TRACKING_STATE_RECENTLY_LOST, 5), i)
        assert tracker.vetoing

    # Tracking "recovers" into the new map -- even a very long run of clean
    # OK frames on the new map (well past the old settle_frames=30 window)
    # must NOT clear the veto. No "reinit_settled" event exists anymore.
    for i in range(210, 210 + 500):
        event = tracker.observe(TrackingStatus(TRACKING_STATE_OK, 280), i)
        assert event is None
        assert tracker.vetoing
    assert tracker.events == [{"frame": 200, "recovered_at_frame": None}]


def test_veto_stays_permanent_across_full_remaining_run():
    """Extends the above with an explicit 'simulated run continues for many
    more frames, including further tracking-quality noise' check -- no
    number or combination of subsequent frames, of any tracking state,
    ever clears self.vetoing once a genuine reinit has been observed. This
    directly covers the admission-gate contract: after a confirmed reinit,
    no further admission occurs for the remainder of the run."""
    tracker = MapReinitTracker(settle_frames=5)
    tracker.observe(TrackingStatus(TRACKING_STATE_OK, 100), 0)  # latch ever_ok
    event = tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 1)
    assert event == "reinit_started"
    assert tracker.vetoing

    # A long, varied remainder-of-run sequence: clean OK streaks well past
    # settle_frames, brief RECENTLY_LOST dips, more NOT_INITIALIZED blips.
    states = (
        [TRACKING_STATE_OK] * 50
        + [TRACKING_STATE_RECENTLY_LOST] * 3
        + [TRACKING_STATE_OK] * 200
        + [TRACKING_STATE_NOT_INITIALIZED] * 2
        + [TRACKING_STATE_OK] * 1000
    )
    for i, state in enumerate(states, start=2):
        tracker.observe(TrackingStatus(state, 200), i)
        assert tracker.vetoing, f"veto cleared unexpectedly at frame {i} (state={state})"

    # Exactly one event was ever recorded -- the second NOT_INITIALIZED
    # blip mid-veto is not a new event (still the same unresolved episode).
    assert len(tracker.events) == 1
    assert tracker.events[0]["recovered_at_frame"] is None


def test_second_reinit_while_already_vetoed_is_not_a_new_event():
    """Once vetoing, a further NOT_INITIALIZED/NO_IMAGES_YET blip is still
    the same unresolved episode -- must not append a second event (the
    veto never having cleared, there's nothing to be a "second" reinit
    relative to)."""
    tracker = MapReinitTracker(settle_frames=2)
    tracker.observe(TrackingStatus(TRACKING_STATE_OK, 100), 0)
    event = tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 1)
    assert event == "reinit_started"
    assert tracker.vetoing
    for i in range(2, 10):
        tracker.observe(TrackingStatus(TRACKING_STATE_OK, 100), i)
        assert tracker.vetoing
    event = tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 10)
    assert event is None
    assert tracker.vetoing
    assert len(tracker.events) == 1


# OrbslamLogReinitWatcher (2026-08-06) -- log-tail fallback, independent of
# /slam/tracking_status_stamped, for when that topic is silently absent.

def test_missing_log_file_is_a_harmless_noop(tmp_path):
    watcher = OrbslamLogReinitWatcher(str(tmp_path / "does_not_exist.log"))
    assert watcher.check() is False
    assert watcher.check() is False
    assert not watcher.detected


def test_single_map_creation_never_fires(tmp_path):
    """A normal, healthy run has exactly one 'Creation of new map' line at
    genuine startup -- must never be treated as a re-init."""
    log = tmp_path / "orbslam_live.log"
    log.write_text("some preamble\nCreation of new map with id: 0\nmore lines\n")
    watcher = OrbslamLogReinitWatcher(str(log))
    assert watcher.check() is False
    assert watcher.map_creation_count == 1
    assert not watcher.detected


def test_second_map_creation_fires_exactly_once(tmp_path):
    """Mirrors the real floor2 log sequence this session found: a burst of
    'Fail to track local map!' followed by a second 'Creation of new map'
    line. Must fire True on the call where the count crosses 1, and False
    on every call after (idempotent)."""
    log = tmp_path / "orbslam_live.log"
    log.write_text("Creation of new map with id: 0\n")
    watcher = OrbslamLogReinitWatcher(str(log))
    assert watcher.check() is False

    with open(log, "a") as f:
        f.write("Fail to track local map!\n" * 9)
        f.write("Creation of new map with id: 1\n")
        f.write("Stored map with ID: 0\n")
    assert watcher.check() is True
    assert watcher.map_creation_count == 2
    assert watcher.detected

    # Further appends (e.g. a THIRD creation) must not re-fire -- permanent
    # veto is a one-way transition by design (see class docstring).
    with open(log, "a") as f:
        f.write("Creation of new map with id: 2\n")
    assert watcher.check() is False


# ---------------------------------------------------------------------------
# report.json invalidation field (2026-08-06) -- distinct from eval_valid,
# see utils/partition_eval.py's compute_trajectory_coverage docstring.
# ---------------------------------------------------------------------------

def test_compute_trajectory_coverage_flags_map_reinit_invalidated():
    from eligsir.utils.partition_eval import compute_trajectory_coverage

    tracker = MapReinitTracker(settle_frames=5)
    tracker.observe(TrackingStatus(TRACKING_STATE_OK, 100), 0)
    tracker.observe(TrackingStatus(TRACKING_STATE_NOT_INITIALIZED, 0), 1)
    assert tracker.vetoing

    result = compute_trajectory_coverage(
        scene_source="/data/floor2",
        n_reg=1000,
        frames_admitted=1,
        partition_psnr={},
        frame_admission="all",
        stream_end_reason=None,
        tracking_reinit_events=list(tracker.events),
    )
    assert result["map_reinit_invalidated"] is True
    assert result["map_reinit_reason"] is not None
    assert "1" in result["map_reinit_reason"]


def test_compute_trajectory_coverage_map_reinit_invalidated_false_by_default():
    from eligsir.utils.partition_eval import compute_trajectory_coverage

    result = compute_trajectory_coverage(
        scene_source="/data/floor2",
        n_reg=1000,
        frames_admitted=999,
        partition_psnr={},
        frame_admission="all",
        stream_end_reason=None,
        tracking_reinit_events=[],
    )
    assert result["map_reinit_invalidated"] is False
    assert result["map_reinit_reason"] is None
    # Sanity: this field is independent of eval_valid (not present on this
    # dict at all -- eval_valid lives in comparison_report.py's summary,
    # computed from holdout-split validity, unrelated to reinit detection).
    assert "eval_valid" not in result


def test_tails_incrementally_not_rereading_whole_file(tmp_path):
    """Growing the log across multiple check() calls before the second
    creation line must still correctly accumulate the count exactly once
    per real occurrence, not double-count from re-scanning already-read
    bytes."""
    log = tmp_path / "orbslam_live.log"
    log.write_text("Creation of new map with id: 0\n")
    watcher = OrbslamLogReinitWatcher(str(log))
    watcher.check()
    for _ in range(5):
        with open(log, "a") as f:
            f.write("Fail to track local map!\n")
        assert watcher.check() is False
    assert watcher.map_creation_count == 1
    with open(log, "a") as f:
        f.write("Creation of new map with id: 1\n")
    assert watcher.check() is True
    assert watcher.map_creation_count == 2
