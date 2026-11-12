"""Determinism/invariant tests for the canonical source-identity contract (#1074).

These are the issue's "Determinism tests" acceptance bullets in executable
form:

* source ordinal assigned before drops and never renumbered;
* one dropped observation leaves all later identities unchanged;
* timestamp ns round-trip is exact;
* bag/source holdout identities remain stable under deterministic drop simulation;
* identical packets produce identical canonical metadata/fingerprints.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from eligsir.contracts import (
    BufferOwnership,
    BufferRef,
    CameraIntrinsics,
    IngressAction,
    IngressEvent,
    MemoryDomain,
    Pose,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def _packet(identity: SourceIdentity, *, pixels: int = 0) -> SensorPacket:
    rgb = BufferRef.from_numpy(
        np.full((4, 4, 3), pixels, dtype=np.uint8), encoding="rgb8"
    )
    depth = BufferRef.from_numpy(
        np.full((4, 4), pixels, dtype=np.uint16), encoding="16UC1", units="raw_counts"
    )
    return SensorPacket(identity, rgb, depth, _intrinsics())


# --------------------------------------------------------------------------
# Ordinals are assigned before drops and are never renumbered
# --------------------------------------------------------------------------


class _Source:
    """Minimal model of a source that stamps identity BEFORE its drop policy.

    This mirrors the real rule #1074 mandates: the ordinal is assigned at
    acceptance from the transport, so a later drop consumes its ordinal and
    produces no observation, rather than shifting its successors down.
    """

    def __init__(self, source_id: str = "camera", stream_epoch: int = 0) -> None:
        self.source_id = source_id
        self.stream_epoch = stream_epoch
        self._next_ordinal = 0
        self.events: list[IngressEvent] = []

    def receive(self, timestamp_ns: int, *, drop: bool = False) -> SourceIdentity | None:
        identity = SourceIdentity(
            self.source_id, self.stream_epoch, self._next_ordinal, timestamp_ns
        )
        self._next_ordinal += 1  # consumed whether or not the frame survives
        if drop:
            self.events.append(
                IngressEvent(
                    source=identity,
                    stage="source_queue",
                    owner="test_policy",
                    action=IngressAction.DROP,
                    reason="simulated_overflow",
                )
            )
            return None
        return identity


def test_source_ordinal_is_assigned_before_drops():
    source = _Source()
    delivered = [source.receive(1000 + i) for i in range(3)]
    assert [d.source_ordinal for d in delivered] == [0, 1, 2]


def test_one_dropped_observation_leaves_all_later_identities_unchanged():
    stamps = [1_000_000_000 + i * 33_000_000 for i in range(6)]

    clean = _Source()
    clean_ids = [clean.receive(ts) for ts in stamps]

    lossy = _Source()
    lossy_ids = [lossy.receive(ts, drop=(i == 2)) for i, ts in enumerate(stamps)]

    # The dropped observation produces nothing...
    assert lossy_ids[2] is None
    # ...and every surviving identity is byte-identical to the no-drop run.
    for i, identity in enumerate(lossy_ids):
        if i == 2:
            continue
        assert identity == clean_ids[i]
        assert identity.source_ordinal == i

    # The loss is attributed, not silent, and it names the consumed ordinal.
    assert len(lossy.events) == 1
    assert lossy.events[0].source.source_ordinal == 2
    assert lossy.events[0].action is IngressAction.DROP


def test_holdout_identities_are_stable_under_deterministic_drop_simulation():
    """An every-Nth holdout keyed on the canonical ordinal survives drops.

    Keyed on a post-drop position instead, dropping one frame reshuffles the
    whole split -- the exact defect #1074's "no post-drop index may define
    physical source identity" bullet forbids.
    """
    hold = 3
    stamps = [1_000_000_000 + i * 33_000_000 for i in range(12)]

    def holdouts(drop_at: set[int]) -> set[int]:
        source = _Source()
        kept = [
            source.receive(ts, drop=(i in drop_at)) for i, ts in enumerate(stamps)
        ]
        return {
            identity.timestamp_ns
            for identity in kept
            if identity is not None and identity.source_ordinal % hold == 0
        }

    baseline = holdouts(set())
    for drop_at in ({1}, {4}, {1, 7}, {2, 5, 9}):
        realized = holdouts(drop_at)
        dropped_stamps = {stamps[i] for i in drop_at}
        # Every surviving holdout is the SAME physical frame as in the clean run.
        assert realized == baseline - dropped_stamps

    # Contrast: a post-drop positional rule is NOT stable.
    def positional_holdouts(drop_at: set[int]) -> set[int]:
        kept = [ts for i, ts in enumerate(stamps) if i not in drop_at]
        return {ts for pos, ts in enumerate(kept) if pos % hold == 0}

    assert positional_holdouts({1}) != positional_holdouts(set()) - {stamps[1]}


# --------------------------------------------------------------------------
# Integer nanoseconds are canonical
# --------------------------------------------------------------------------


def test_timestamp_ns_round_trip_is_exact():
    stamps = [0, 1, 1_699_999_999_123_456_789, 2**53 + 7]
    for ts in stamps:
        assert SourceIdentity("camera", 0, 0, ts).timestamp_ns == ts


def test_timestamp_seconds_is_documented_lossy_and_never_canonical():
    ts = 1_699_999_999_123_456_789
    identity = SourceIdentity("camera", 0, 0, ts)
    # float64 at a ROS-epoch magnitude cannot represent nanoseconds; this is
    # exactly why the integer field is the canonical one.
    assert int(identity.timestamp_seconds * 1e9) != ts


def test_numpy_integer_stamps_are_accepted_and_canonicalized():
    identity = SourceIdentity("camera", np.int32(0), np.int64(5), np.int64(1234))
    assert identity.timestamp_ns == 1234
    assert type(identity.timestamp_ns) is int
    assert type(identity.source_ordinal) is int


@pytest.mark.parametrize("bad", [True, False])
def test_bool_is_rejected_as_an_ordinal_or_stamp(bad):
    with pytest.raises(TypeError, match="bool"):
        SourceIdentity("camera", 0, bad, 0)
    with pytest.raises(TypeError, match="bool"):
        SourceIdentity("camera", 0, 0, bad)


def test_float_stamps_are_rejected():
    with pytest.raises(TypeError):
        SourceIdentity("camera", 0, 0, 1.5)


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def test_identical_packets_produce_identical_metadata_fingerprints():
    identity = SourceIdentity("camera", 0, 9, 1_699_999_999_123_456_789)
    assert _packet(identity).metadata_fingerprint() == _packet(identity).metadata_fingerprint()


def test_fingerprint_covers_metadata_not_pixels():
    identity = SourceIdentity("camera", 0, 9, 1234)
    # Same metadata, different pixel content -> same fingerprint, matching
    # BufferRef's own compare=False payload boundary.
    assert _packet(identity, pixels=0).metadata_fingerprint() == _packet(
        identity, pixels=200
    ).metadata_fingerprint()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_id": "other"},
        {"stream_epoch": 1},
        {"source_ordinal": 10},
        {"timestamp_ns": 1235},
        {"pair_id": "p"},
    ],
)
def test_fingerprint_changes_with_every_identity_field(kwargs):
    base = dict(source_id="camera", stream_epoch=0, source_ordinal=9, timestamp_ns=1234)
    assert (
        SourceIdentity(**base).fingerprint()
        != SourceIdentity(**{**base, **kwargs}).fingerprint()
    )


def test_fingerprint_is_stable_across_processes():
    """Not Python's salted hash(): must survive a different PYTHONHASHSEED."""
    program = (
        "from eligsir.contracts import SourceIdentity;"
        "print(SourceIdentity('camera', 0, 9, 1234).fingerprint())"
    )
    digests = set()
    for seed in ("0", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1


# --------------------------------------------------------------------------
# Tracking provenance
# --------------------------------------------------------------------------


def test_lost_tracking_may_omit_a_pose():
    identity = SourceIdentity("camera", 0, 0, 0)
    observation = TrackingObservation(
        source=identity, pose=None, status=TrackingStatus.LOST, provider="orbslam3"
    )
    assert observation.pose is None
    assert observation.usable is False


@pytest.mark.parametrize("status", [TrackingStatus.VALID, TrackingStatus.DEGRADED])
def test_usable_tracking_without_a_pose_is_refused(status):
    identity = SourceIdentity("camera", 0, 0, 0)
    with pytest.raises(ValueError, match="requires a pose"):
        TrackingObservation(
            source=identity, pose=None, status=status, provider="orbslam3"
        )


def test_packet_may_carry_a_pose_free_lost_observation():
    identity = SourceIdentity("camera", 0, 3, 300)
    packet = SensorPacket(
        identity,
        BufferRef.from_numpy(np.zeros((2, 2, 3), np.uint8), encoding="rgb8"),
        None,
        _intrinsics(),
        TrackingObservation(
            source=identity, pose=None, status=TrackingStatus.LOST, provider="p"
        ),
    )
    assert packet.depth is None
    assert packet.tracking.pose is None


def test_buffer_ref_from_numpy_records_real_layout_without_copying():
    dense = np.zeros((8, 8, 3), dtype=np.uint8)
    ref = BufferRef.from_numpy(dense, encoding="rgb8")
    assert ref.data is dense  # zero copy
    assert ref.spec.contiguous is True
    assert ref.spec.shape == (8, 8, 3)
    assert ref.spec.dtype == "uint8"
    assert ref.spec.memory_domain is MemoryDomain.HOST
    assert ref.ownership is BufferOwnership.OWNED

    # A strided view is described honestly, not silently densified.
    view = dense[::2, ::2]
    view_ref = BufferRef.from_numpy(view, encoding="rgb8")
    assert view_ref.data is view
    assert view_ref.spec.contiguous is False
    assert view_ref.spec.strides_bytes == tuple(view.strides)


# --------------------------------------------------------------------------
# Canonical calibration record
# --------------------------------------------------------------------------


def test_camera_intrinsics_is_keyword_only_and_frozen():
    """The two merged definitions had different positional orders, so a
    surviving positional call would bind width->fx silently. Refuse it."""
    with pytest.raises(TypeError):
        CameraIntrinsics(500.0, 500.0, 320.0, 240.0, 640, 480)  # type: ignore[misc]
    intr = _intrinsics()
    with pytest.raises(Exception):
        intr.fx = 1.0  # type: ignore[misc]


def test_camera_intrinsics_defaults_depth_to_colour():
    intr = CameraIntrinsics(fx=1.0, fy=2.0, cx=3.0, cy=4.0, width=5, height=6)
    assert (intr.depth_fx, intr.depth_fy) == (1.0, 2.0)
    assert (intr.depth_width, intr.depth_height) == (5, 6)


def test_camera_intrinsics_distortion_is_normalized_to_a_tuple():
    """A list vs tuple must not change the record's identity or fingerprint."""
    a = CameraIntrinsics(fx=1, fy=1, cx=1, cy=1, width=2, height=2, D=[0.1, 0.2])
    b = CameraIntrinsics(fx=1, fy=1, cx=1, cy=1, width=2, height=2, D=(0.1, 0.2))
    assert a.D == (0.1, 0.2) and a == b


def test_intrinsics_json_round_trip_preserves_distinct_depth_intrinsics(tmp_path):
    """ScanNet-style 640x480 depth vs 1296x968 colour must survive the format."""
    from eligsir.utils.rgbd_frames import read_intrinsics, write_intrinsics

    original = CameraIntrinsics(
        fx=1170.0, fy=1170.0, cx=648.0, cy=484.0, width=1296, height=968,
        depth_fx=577.0, depth_fy=577.0, depth_cx=320.0, depth_cy=240.0,
        depth_width=640, depth_height=480, depth_scale=1000.0,
        camera_frame="cam", world_frame="map", distortion_model="plumb_bob",
        D=(0.1, 0.2, 0.0, 0.0, 0.0), intrinsics_source="scannet",
    )
    path = tmp_path / "intrinsics.json"
    write_intrinsics(path, original)
    assert read_intrinsics(path) == original


def test_intrinsics_json_without_depth_keys_still_reads(tmp_path):
    """Files written before the merge must read back unchanged."""
    import json

    from eligsir.utils.rgbd_frames import read_intrinsics

    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps({
        "width": 640, "height": 480, "fx": 500.0, "fy": 500.0,
        "cx": 320.0, "cy": 240.0, "depth_scale": 1000.0,
    }))
    intr = read_intrinsics(path)
    assert (intr.depth_width, intr.depth_height) == (640, 480)
    assert intr.depth_fx == 500.0


def test_pose_free_identity_equality_is_value_based():
    a = SourceIdentity("camera", 0, 4, 400)
    b = SourceIdentity("camera", 0, 4, 400)
    assert a == b and a.fingerprint() == b.fingerprint()
    assert isinstance(Pose.identity().c2w, np.ndarray)
