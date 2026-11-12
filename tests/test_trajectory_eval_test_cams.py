import json
import numpy as np
import pytest

from eligsir.utils.trajectory_eval import _cams_to_c2w, _trajectory_plausibility, run_trajectory_eval


class _FakeCam:
    def __init__(self, name, source_index, timestamp, xyz, quat_wxyz=(1.0, 0.0, 0.0, 0.0)):
        self.image_name = name
        self._streaming_timestamp = timestamp
        # R stores R_c2w (camera-to-world rotation), T stores t_w2c, per the
        # convention documented at utils/trajectory_eval.py:502-505.
        self.R = np.eye(3, dtype=np.float64)
        self.T = -np.asarray(xyz, dtype=np.float64)  # t_w2c = -R_c2w @ t_c2w for R=I
        self.fid = timestamp


class _FakeOfflineCam:
    """Mimics a REAL legacy (non-streaming) train.py ``scene.cameras.Camera``:
    no ``_streaming_timestamp`` attribute, ``colmap_id`` holding the real,
    pre-shuffle ``CameraInfo.uid`` (dataset-order index every scene reader
    assigns, see scene/readers/*.py, carried through unchanged by
    ``loadCam``), and ``uid`` holding the camera's position in whatever
    (already-shuffled) list ``cameraList_from_camInfos`` was called with --
    i.e. ``uid`` is deliberately set to something UNRELATED to real frame
    order here, exactly like the real Camera class, to prove the export
    logic reads ``colmap_id`` and not ``uid``.

    Used to reproduce the real Replica room0 shuffle/holdout-compaction bug
    (2026-08-22): the first fix attempt used ``cam.uid`` and still
    reproduced the original bug end-to-end for exactly this reason -- see
    ``utils/trajectory_eval.py``'s ``_camera_frame_id`` docstring."""

    def __init__(self, colmap_id, image_name, xyz, shuffled_uid=None):
        self.colmap_id = colmap_id
        # Simulate cameraList_from_camInfos's `uid=id` (position in the
        # shuffled list) with a value that is never equal to colmap_id,
        # so a test that accidentally reads `uid` instead of `colmap_id`
        # fails loudly instead of silently passing by coincidence.
        self.uid = shuffled_uid if shuffled_uid is not None else (colmap_id + 9000)
        self.image_name = image_name
        self.R = np.eye(3, dtype=np.float64)
        self.T = -np.asarray(xyz, dtype=np.float64)


def test_run_trajectory_eval_without_test_cams_is_unchanged(tmp_path):
    train_cams = [_FakeCam("000008", 8, 1000.0, [0.0, 0.0, 0.0])]
    run_trajectory_eval(output_dir=str(tmp_path), train_cams=train_cams)

    assert (tmp_path / "trajectory_tum.txt").exists()
    assert not (tmp_path / "trajectory_test_tum.txt").exists()
    assert not (tmp_path / "trajectory_test_index.json").exists()


def test_run_trajectory_eval_writes_test_trajectory_and_index(tmp_path):
    train_cams = [_FakeCam("000008", 8, 1000.0, [0.0, 0.0, 0.0])]
    test_cams = [
        _FakeCam("001504", 1504, 1001.5, [1.0, 2.0, 3.0]),
        _FakeCam("003008", 3008, 1003.0, [4.0, 5.0, 6.0]),
    ]

    run_trajectory_eval(output_dir=str(tmp_path), train_cams=train_cams, test_cams=test_cams)

    test_tum_path = tmp_path / "trajectory_test_tum.txt"
    assert test_tum_path.exists()
    lines = [l for l in test_tum_path.read_text().splitlines() if not l.startswith("#")]
    assert len(lines) == 2
    first_fields = [float(x) for x in lines[0].split()]
    assert first_fields[0] == pytest.approx(1001.5)
    assert first_fields[1:4] == pytest.approx([1.0, 2.0, 3.0])

    index_path = tmp_path / "trajectory_test_index.json"
    assert index_path.exists()
    index = json.loads(index_path.read_text())
    assert index == [
        {"name": "001504", "source_index": 1504},
        {"name": "003008", "source_index": 3008},
    ]


def test_run_trajectory_eval_test_cams_empty_list_writes_nothing(tmp_path):
    train_cams = [_FakeCam("000008", 8, 1000.0, [0.0, 0.0, 0.0])]
    run_trajectory_eval(output_dir=str(tmp_path), train_cams=train_cams, test_cams=[])

    assert not (tmp_path / "trajectory_test_tum.txt").exists()
    assert not (tmp_path / "trajectory_test_index.json").exists()


# ---------------------------------------------------------------------------
# _trajectory_plausibility: implausible-distance / degenerate-pose detection
#
# Regression coverage for the real 2026-08-12 bug: a RealityScan Mobile
# hallway capture reported "distance: 2213 m" for a physically ~30-60m
# out-and-back walk. Root cause was two independent defects that BOTH
# produced the same symptom: (1) the frame order fed to _trajectory_distance
# was uncorrelated with true capture order (so summing consecutive-array
# deltas summed essentially random jumps), and (2) the underlying poses
# themselves were 99.97%-collinear (rank-1 via SVD) with large near-random
# magnitude along that one axis -- the classic monocular-SfM "forward
# motion"/bas-relief degeneracy. These tests reproduce each signature
# directly (not via the real RealityScan importer, which isn't installed in
# this environment) and confirm the detector added in
# utils/trajectory_eval.py catches both, without false-positiving on a
# healthy, correctly-ordered out-and-back path.
# ---------------------------------------------------------------------------

def _c2w_from_positions(positions: np.ndarray) -> np.ndarray:
    n = len(positions)
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, :3, 3] = positions
    return c2w


def _healthy_out_and_back(n_per_leg: int = 60, length: float = 18.0, seed: int = 0) -> np.ndarray:
    """A real out-and-back hallway walk: X goes 0->length->0, small vertical
    (Y) jitter (phone held roughly level), and lateral (Z) sway from natural
    human walking gait -- both noise and a slow side-to-side drift, since a
    person (unlike a rail-guided cart) doesn't walk a perfectly straight
    line. This lateral extent is what keeps a genuinely healthy corridor
    walk's collinearity_frac measurably below a degenerate SfM collapse's
    (~0.991 here vs. ~0.9997 measured on the real bug's capture, see
    _trajectory_distance's docstring) -- a pure noise-only jitter with too
    little amplitude is not actually representative of real, unaided
    handheld walking and would give a false sense of the detector's margin."""
    rng = np.random.default_rng(seed)
    x_out = np.linspace(0.0, length, n_per_leg)
    x_back = np.linspace(length, 0.0, n_per_leg)
    x = np.concatenate([x_out, x_back])
    n = len(x)
    y = rng.normal(0.0, 0.03, size=n)  # near-zero vertical jitter
    z = rng.normal(0.0, 0.4, size=n) + 0.3 * np.sin(np.linspace(0.0, 6 * np.pi, n))
    return np.stack([x, y, z], axis=1)


def test_healthy_out_and_back_not_flagged():
    positions = _healthy_out_and_back()
    c2w = _c2w_from_positions(positions)
    result = _trajectory_plausibility(c2w)

    assert result["implausible_distance"] is False
    assert result["degenerate_poses"] is False
    # A real there-and-back walk's path length should be a small multiple of
    # its bounding-box diagonal (~2x), nowhere near the 82x seen in the real
    # bug (2444m path / 29.7m bbox diagonal on the actual pixel10a capture).
    assert result["distance_over_bbox_diagonal"] < 4.0


def test_scrambled_order_of_healthy_positions_flags_implausible_but_not_degenerate():
    """Same physical positions as the healthy case, but fed in a scrambled
    (non-chronological) order -- reproduces defect (1) from the real bug in
    isolation. The positions themselves are NOT collinear/degenerate, so
    only implausible_distance should fire, confirming the two checks are
    independent as designed."""
    positions = _healthy_out_and_back()
    rng = np.random.default_rng(1)
    scrambled = positions[rng.permutation(len(positions))]
    c2w = _c2w_from_positions(scrambled)
    result = _trajectory_plausibility(c2w)

    assert result["implausible_distance"] is True
    assert result["degenerate_poses"] is False


def test_degenerate_collinear_poses_flag_both():
    """Reproduces defect (2) from the real bug directly: positions collapsed
    onto a single line through near the origin (rank-1 via SVD) with large,
    effectively random magnitude along that line per frame -- the monocular
    SfM forward-motion/bas-relief degeneracy. Both checks should fire: the
    poses are geometrically degenerate, and the resulting distance is not a
    real path length."""
    rng = np.random.default_rng(2)
    direction = np.array([2.0, -2.7, 0.0])  # matches the real bug's dominant axis shape
    magnitudes = rng.uniform(0.0, 30.0, size=200)
    positions = magnitudes[:, None] * direction[None, :]
    # tiny perpendicular residual, matching the real ~0.27m mean residual
    # measured on the actual pixel10a capture (over a ~30m span).
    positions += rng.normal(0.0, 0.2, size=positions.shape)
    c2w = _c2w_from_positions(positions)
    result = _trajectory_plausibility(c2w)

    assert result["degenerate_poses"] is True
    assert result["implausible_distance"] is True
    assert result["collinearity_frac"] > 0.999


def test_cams_to_c2w_uses_colmap_id_not_uid_or_list_position_when_shuffled():
    """Root-cause regression for the 2026-08-22 Replica room0 bug: a camera
    list in SHUFFLED order (as scene.getTrainCameras() is, post
    Scene.__init__'s random.shuffle) must still recover each camera's real
    dataset-order position (its poses form a smooth, monotonically-moving
    path along x when read back in real order) -- not the list's shuffled
    position, and NOT ``cam.uid`` either (a real Camera's ``.uid`` IS that
    shuffled list position, per ``cameraList_from_camInfos``'s ``uid=id``
    rebind -- only ``.colmap_id`` survives the shuffle unchanged). Getting
    this wrong would scramble the exported path and (against a real GT
    trajectory) collapse a Sim3 alignment's scale to nonsense -- exactly
    what happened on the real run, twice (once via list position, then
    again via a first fix attempt that read ``.uid``)."""
    pytest.importorskip("torch")  # _cams_to_c2w lazily imports torch
    # Real per-frame positions: a smooth path, colmap_id 0..9 in physical order.
    real_positions = {i: [float(i), 0.0, 0.0] for i in range(10)}
    ordered_cams = [
        _FakeOfflineCam(colmap_id=i, image_name=f"{i:06d}", xyz=real_positions[i])
        for i in range(10)
    ]

    rng = np.random.default_rng(7)
    shuffled_cams = [ordered_cams[i] for i in rng.permutation(len(ordered_cams))]

    ts, c2w, order = _cams_to_c2w(shuffled_cams)

    # ts must equal the REAL per-camera colmap_id, not 0..9 by list position,
    # and not the (deliberately unrelated) `.uid` value either.
    assert sorted(ts.tolist()) == list(range(10))
    # Output must come back in temporal (colmap_id) order regardless of input order.
    assert ts.tolist() == sorted(ts.tolist())
    # Positions must be monotonically increasing along x once reordered --
    # i.e. this is a real, non-scrambled path, not a random permutation.
    xs = c2w[:, 0, 3]
    assert np.all(np.diff(xs) > 0)
    # `order` must let a caller reproduce the same reordering on a parallel
    # per-camera list (e.g. trajectory_test_index.json).
    reordered_cams = [shuffled_cams[i] for i in order]
    assert [c.colmap_id for c in reordered_cams] == list(range(10))


def test_run_trajectory_eval_with_shuffled_and_holdout_compacted_cams_recovers_real_order(tmp_path):
    """End-to-end version of the above, through run_trajectory_eval: a
    holdout-compacted (every-3rd frame removed, remaining re-indexed 0..N-1)
    AND shuffled train camera list must still export a temporally-ordered,
    correctly-timestamped trajectory_tum.txt keyed by each camera's real
    uid -- exactly the RGBD-sequence/Replica --eval scenario that produced
    scale=0.0203 / ate_mean=1.44m against real GT before this fix."""
    pytest.importorskip("torch")  # run_trajectory_eval -> _cams_to_c2w lazily imports torch
    real_uids = [i for i in range(15) if i % 3 != 0]  # holdout-compacted subset
    real_positions = {i: [float(i), 0.0, 0.0] for i in real_uids}
    cams = [
        _FakeOfflineCam(colmap_id=i, image_name=f"{i:06d}", xyz=real_positions[i])
        for i in real_uids
    ]
    rng = np.random.default_rng(11)
    shuffled = [cams[i] for i in rng.permutation(len(cams))]

    run_trajectory_eval(output_dir=str(tmp_path), train_cams=shuffled)

    lines = [
        l for l in (tmp_path / "trajectory_tum.txt").read_text().splitlines()
        if not l.startswith("#")
    ]
    assert len(lines) == len(real_uids)
    rows = [[float(x) for x in l.split()] for l in lines]
    ts_out = [r[0] for r in rows]
    x_out = [r[1] for r in rows]
    assert ts_out == sorted(real_uids, key=float)
    # Each row's x-position must equal its own real uid (not its position in
    # the shuffled/compacted input list).
    for t, x in zip(ts_out, x_out):
        assert x == pytest.approx(t)


def test_run_trajectory_eval_surfaces_implausible_flags_in_metrics_json(tmp_path):
    """End-to-end: run_trajectory_eval must propagate the plausibility flags
    into metrics.json so a human (or automation) reading the standard output
    artifact sees the caveat -- not just a raw, unqualified distance_m."""
    rng = np.random.default_rng(3)
    direction = np.array([1.0, -1.35, 0.0])
    magnitudes = rng.uniform(0.0, 30.0, size=50)
    positions = magnitudes[:, None] * direction[None, :]
    positions += rng.normal(0.0, 0.2, size=positions.shape)

    train_cams = [
        _FakeCam(f"{i:06d}", i, float(i), positions[i].tolist())
        for i in range(len(positions))
    ]
    run_trajectory_eval(output_dir=str(tmp_path), train_cams=train_cams)

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["degenerate_poses"] is True
    assert metrics["implausible_distance"] is True
    assert metrics["distance_m"] > 0
