"""Issue #398: holdout/test camera GT images must not stay GPU-resident.

`StreamingScene.ingest_next_frame` builds a holdout camera EAGERLY
(`_frame_to_camera(frame, load_image=True)`) and then falls through the common
tail's `prepare_camera_for_render(cam, device="cuda")`. Unlike an admitted train
camera -- which the hot-window eviction (`offload_camera_image`) pages back to
CPU a few frames later -- nothing ever evicted a holdout image again, so every
future evaluation image sat in VRAM for the whole mapping phase. Residency grew
as O(n_holdouts x W x H): 993 holdouts at 960x720 float32 is ~7.7 GiB of
future-eval pixels held during training.

The fix keeps the eager DECODE (a live/bag frame's pixels cannot be re-decoded
later -- the source releases them as the cursor advances) but immediately pages
the tensor to CPU, and returns residency to that bound after any mid-training
evaluation.

WHAT THIS FILE CAN AND CANNOT PROVE. There is no GPU in the environment these
tests run in, so the VRAM saving itself is NOT measured here. What is measured
is the mechanism: the residency COUNT is O(1) rather than O(n_holdouts), the
pixels survive the move unchanged, poses/intrinsics never needed them, holdout
identity/hashes are bit-identical with the mechanism on and off, and the
evaluation GT fetch uploads a transient copy instead of rebinding camera state.
The fake-device tensor below exists precisely so residency is counted against a
non-CPU device on a CPU-only host -- forcing `data_device="cpu"` would make the
offload a no-op and prove nothing.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scene.cameras import (
    gpu_resident_image_bytes,
    gpu_resident_image_count,
    offload_camera_image,
    offload_camera_images,
)
from scene.streaming_scene import StreamingScene
from eligsir.utils.baseline_comparison import _camera_set_hash


# ---------------------------------------------------------------------------
# Fake "GPU" tensor: enough surface for offload_camera_image / the residency
# accounting helpers, with a device this host does not actually have.
# ---------------------------------------------------------------------------
class _FakeTensor:
    def __init__(self, values: torch.Tensor, device_type: str = "cuda"):
        self.values = values
        self.device = SimpleNamespace(type=device_type, index=0)

    @property
    def _version(self):
        return self.values._version

    def detach(self):
        return self

    def contiguous(self):
        return self

    def numel(self):
        return int(self.values.numel())

    def element_size(self):
        return int(self.values.element_size())

    def to(self, device=None, **kwargs):
        target = device if isinstance(device, str) else getattr(device, "type", device)
        return _FakeTensor(self.values.clone(), device_type=str(target))

    def __getitem__(self, item):
        return _FakeTensor(self.values[item], device_type=self.device.type)

    def clamp(self, lo, hi):
        return _FakeTensor(self.values.clamp(lo, hi), device_type=self.device.type)


class _FakeFrame:
    def __init__(self, index: int) -> None:
        self.index = index
        self._odom_valid = True
        self.timestamp = float(index)


class _FakeCam:
    """Camera stand-in with a fake-GPU-resident image, as ingest builds it."""

    def __init__(self, index: int, load_image: bool, h: int = 8, w: int = 8) -> None:
        import numpy as np

        self.R = np.eye(3, dtype=np.float64)
        self.T = np.zeros(3, dtype=np.float64)
        self.image_name = f"{index:06d}"
        self.uid = index
        self.width = self.image_width = w
        self.height = self.image_height = h
        self.fx = self.fy = 50.0
        self.cx = w / 2.0
        self.cy = h / 2.0
        self.FoVx = self.FoVy = 1.0
        self.c2w = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        self._materialize_calls = 0
        if load_image:
            # Deterministic per-camera pixel content so a later comparison can
            # detect a swapped/corrupted image, not just a shape match.
            base = torch.arange(3 * h * w, dtype=torch.float32).reshape(3, h, w)
            self.original_image = _FakeTensor((base + index) / 1000.0)

    def materialize_pending_image(self):
        self._materialize_calls += 1


class _FakeIngestSelf:
    """Mirrors tests/test_streaming_scene_lazy_camera_image.py's harness."""

    def __init__(self, frames, holdout_indices, offload: bool, img_hw=(8, 8)):
        self.args = SimpleNamespace(
            streaming_eval_hold=4,
            streaming_eval_min_baseline_m=0.03,
            streaming_eval_min_angle_deg=2.0,
            streaming_eval_guard=0,
            streaming_offload_camera_images=offload,
            streaming_replay_buffer=512,
            streaming_frame_admission="all",
            streaming_global_reservoir_stride=0,
            streaming_keyframe_window=8,
            streaming_track_scene_complexity=False,
            streaming_inactive_cache_size=0,
        )
        self._img_hw = img_hw
        self._active_source_indices = set()
        self._active_cam_by_idx = {}
        self._last_admitted_source_idx = None
        self._frames = frames
        self._source_idx = 0
        self._keyframes_rejected = 0
        self._keyframes_admitted = 0
        self._holdout_indices = set(holdout_indices)
        self._guard_indices = set()
        self._test_cameras = []
        self.train_cameras = []
        self._replay_buffer = []
        self._n_train_frames_ingested = 0
        self._global_reservoir = []
        self._steps_since_new_frame = 0
        self._pending_extent_update = False
        self.current_camera = None
        self._dense_trajectory = []
        self._dense_trajectory_flushed = 0

    # --- source plumbing -------------------------------------------------
    def has_next_frame(self):
        return self._source_idx < len(self._frames)

    def _get_source_frame(self, index):
        if index >= len(self._frames):
            return None
        return self._frames[index]

    def _frame_to_camera(self, frame, load_image=True):
        h, w = self._img_hw
        cam = _FakeCam(frame.index, load_image, h=h, w=w)
        cam._streaming_frame = frame
        return cam

    # --- bookkeeping stubs ----------------------------------------------
    def accumulate_observed_surface(self, frame):
        pass

    def _cache_inactive_frame(self, frame):
        pass

    def _release_source_frames_upto_cursor(self):
        pass

    def _append_dense_trajectory(self, cam, frame, is_train, is_holdout):
        self._dense_trajectory.append(
            {"index": frame.index, "is_holdout": bool(is_holdout),
             "admitted": bool(is_train and not is_holdout)}
        )

    def flush_deferred_extent_update(self):
        pass

    def update_cameras_extent(self):
        pass

    # --- real methods under test ----------------------------------------
    # Bound from the real StreamingScene, never stubbed -- same pattern as the
    # sibling residency harnesses (tests/test_replay_camera_residency.py,
    # tests/test_streaming_scene_lazy_camera_image.py). `ingest_next_frame`'s
    # admitted-train branch calls `_offload_evicted_camera()` (the hot-window
    # eviction cursor), which in turn calls `_offload_camera_hot_window()` --
    # the latter reads `args.streaming_keyframe_window` with NO getattr
    # fallback (issue #368), so binding the real one keeps that contract
    # exercised here too, and avoids an AttributeError crash on that branch
    # (the #548 fake-drift defect this file was patched for). Binding rather
    # than stubbing does make a *signature* change on either bound method
    # surface as a diagnosable failure here. It does NOT give this file
    # behavioral coverage of the hot-window eviction cursor itself -- no
    # assertion in this file reads `train_cameras` residency/eviction state
    # (verified 2026-08-15, issue #548 mutation check: neutering
    # `_offload_evicted_camera` to an unconditional no-op left all 13 tests
    # in this file green). That coverage lives in
    # tests/test_replay_camera_residency.py instead.
    offload_test_camera_images = StreamingScene.offload_test_camera_images
    gpu_resident_test_image_count = StreamingScene.gpu_resident_test_image_count
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera
    # Issue #1115: #1084 extracted the inline is_holdout/is_guard computation
    # out of `ingest_next_frame` into the public `holdout_status()` predicate
    # (verbatim -- the extraction changed no rule), so bind the REAL one here
    # rather than stubbing a constant: the holdout-reservation rule this file
    # exercises must stay the production rule, read off this fake's own
    # `_holdout_indices`/`_guard_indices`/`args.streaming_eval_hold`.
    holdout_status = StreamingScene.holdout_status


def _ingest_all(n_frames: int, holdout_every: int, offload: bool, img_hw=(8, 8),
                monkeypatch=None):
    frames = [_FakeFrame(i) for i in range(n_frames)]
    holdouts = {i for i in range(n_frames) if i % holdout_every == 0}
    fake_self = _FakeIngestSelf(frames, holdouts, offload=offload, img_hw=img_hw)
    results = [StreamingScene.ingest_next_frame(fake_self) for _ in range(n_frames)]
    return fake_self, results


@pytest.fixture(autouse=True)
def _stub_prepare(monkeypatch):
    """`ingest_next_frame`'s tail calls prepare_camera_for_render(cam, "cuda").

    Stubbed to emulate exactly the part that matters here: it pages the image
    ONTO the GPU (that is what left holdouts resident), so the offload under
    test has something real to undo.
    """
    def _fake_prepare(cam, device="cuda"):
        img = getattr(cam, "original_image", None)
        if img is not None and img.device.type != "cuda":
            cam.original_image = img.to("cuda")

    monkeypatch.setattr("scene.cameras.prepare_camera_for_render", _fake_prepare)


# ---------------------------------------------------------------------------
# 1. Residency is O(1), not O(test-frame-count)
# ---------------------------------------------------------------------------
def test_holdout_images_are_not_gpu_resident_after_ingest():
    scene, _ = _ingest_all(n_frames=40, holdout_every=4, offload=True)
    assert len(scene._test_cameras) == 10
    assert scene.gpu_resident_test_image_count() == 0
    assert gpu_resident_image_bytes(scene._test_cameras) == 0


def test_residency_does_not_scale_with_holdout_count():
    """1000 synthetic holdouts must not produce O(n) resident images.

    Run at two very different holdout counts and assert the resident count is
    identical (and zero) at both -- a count that tracks n_holdouts is exactly
    the pre-#398 behavior. The `offload=False` arm is the control: it must
    still show the O(n) growth, proving this test measures the mechanism and
    that `--no-streaming_offload_camera_images` reproduces the old behavior.
    """
    small, _ = _ingest_all(n_frames=100, holdout_every=1, offload=True)
    large, _ = _ingest_all(n_frames=1000, holdout_every=1, offload=True)
    assert len(small._test_cameras) == 100
    assert len(large._test_cameras) == 1000
    assert small.gpu_resident_test_image_count() == 0
    assert large.gpu_resident_test_image_count() == 0

    ctrl_small, _ = _ingest_all(n_frames=100, holdout_every=1, offload=False)
    ctrl_large, _ = _ingest_all(n_frames=1000, holdout_every=1, offload=False)
    assert ctrl_small.gpu_resident_test_image_count() == 100
    assert ctrl_large.gpu_resident_test_image_count() == 1000


def test_resident_bytes_at_scannetpp_shape_are_zero():
    """The #398 headline case, in the shape it was reported in: 960x720 RGB
    float32 holdouts. Byte accounting is over a fake device, so this asserts
    the residency BOOKKEEPING, not a measured VRAM figure."""
    scene, _ = _ingest_all(n_frames=64, holdout_every=1, offload=True, img_hw=(720, 960))
    ctrl, _ = _ingest_all(n_frames=64, holdout_every=1, offload=False, img_hw=(720, 960))
    assert gpu_resident_image_bytes(scene._test_cameras) == 0
    # ~7.9 MiB per image before the fix.
    assert gpu_resident_image_bytes(ctrl._test_cameras) == 64 * 3 * 720 * 960 * 4


def test_resume_rematerialized_holdouts_are_also_offloaded():
    """materialize_missed_holdouts() rebuilds EVERY already-passed holdout in
    one loop on a checkpoint resume -- the same defect, in burst form."""
    frames = [_FakeFrame(i) for i in range(40)]
    holdouts = {i for i in range(40) if i % 4 == 0}
    scene = _FakeIngestSelf(frames, holdouts, offload=True)
    scene._source_idx = 40  # cursor restored past every holdout
    n = StreamingScene.materialize_missed_holdouts(scene)
    assert n == len(holdouts)
    assert scene.gpu_resident_test_image_count() == 0


# ---------------------------------------------------------------------------
# 2. Pixels survive; poses/intrinsics never needed them
# ---------------------------------------------------------------------------
def test_holdout_pixels_bit_identical_after_offload():
    scene, _ = _ingest_all(n_frames=12, holdout_every=4, offload=True)
    for cam in scene._test_cameras:
        idx = cam._streaming_frame.index
        expected = (torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)
                    + idx) / 1000.0
        assert torch.equal(cam.original_image.values, expected)


def test_poses_and_intrinsics_usable_without_touching_pixels():
    """camera_to_JSON is what test_cameras.json / the dense trajectory are
    built from; it must work off geometry alone."""
    from eligsir.utils.camera_utils import camera_to_JSON

    scene, _ = _ingest_all(n_frames=12, holdout_every=4, offload=True)
    for i, cam in enumerate(scene._test_cameras):
        entry = camera_to_JSON(i, cam)
        assert entry["width"] == 8 and entry["height"] == 8
        assert entry["img_name"] == cam.image_name
    # And the dense trajectory recorded every holdout by pose, unchanged.
    assert sum(1 for e in scene._dense_trajectory if e["is_holdout"]) == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real GPU")
def test_real_cuda_round_trip_is_bit_identical():
    """GPU-gated: the real `offload_camera_image` move preserves every bit.

    NOT exercised in the CPU-only environment this suite normally runs in --
    it is the on-GPU counterpart of the fake-device tests above.
    """
    img = torch.rand(3, 32, 32, device="cuda")
    ref = img.clone()
    cam = SimpleNamespace(original_image=img)
    from eligsir.utils.score_contract import declare_camera_score_provenance

    declare_camera_score_provenance(cam, context="test_holdout_round_trip")
    assert offload_camera_image(cam) is True
    assert cam.original_image.device.type == "cpu"
    assert torch.equal(cam.original_image, ref.cpu())
    # ...and what an eval pass fetches is identical to what was captured.
    from eligsir.utils.comparison_report import _gt_image
    assert torch.equal(_gt_image(cam), ref[:3].clamp(0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Holdout identity / hashes are bit-identical with the mechanism on and off
# ---------------------------------------------------------------------------
def test_holdout_hash_unchanged_on_the_online_every_nth_branch():
    """`_holdout_indices is None` -- the live/lazy-source branch, which is this
    project's stated design target.

    Deliberately separate from the precomputed-branch test below: with
    `_holdout_indices` set, `eval_metadata()` hashes the index SET and never
    reads `_test_cameras` at all, so that test's holdout_hash assertion does
    not exercise the branch that derives the hash FROM the realized test
    cameras. This one does -- and it must be equally pixel-independent.
    """
    def _run(offload: bool):
        frames = [_FakeFrame(i) for i in range(40)]
        scene = _FakeIngestSelf(frames, holdout_indices=set(), offload=offload)
        scene._holdout_indices = None  # -> online every-Nth (streaming_eval_hold=4)
        for _ in range(40):
            StreamingScene.ingest_next_frame(scene)
        return scene

    on, off = _run(True), _run(False)
    assert [c._streaming_frame.index for c in on._test_cameras] == \
           [c._streaming_frame.index for c in off._test_cameras] == list(range(0, 40, 4))
    meta_on = StreamingScene.eval_metadata(on)
    meta_off = StreamingScene.eval_metadata(off)
    assert meta_on == meta_off
    # issue #745 (2026-08-15): the live/lazy branch now emits a RULE hash
    # (see StreamingScene.eval_metadata's docstring), not a hash of the
    # realized _test_cameras set -- "online_every_nth_hashed" was the old
    # realized-set selection string and is retired for new runs (still
    # readable on pre-fix report.json files, which is exactly why the
    # string changed rather than being reinterpreted in place).
    assert meta_on["holdout_selection"] == "online_every_nth_ordinal_rule"
    assert meta_on["holdout_hash"] is not None
    assert _camera_set_hash(on._test_cameras) == _camera_set_hash(off._test_cameras)
    # ...and the residency fix still applies on this branch.
    assert on.gpu_resident_test_image_count() == 0
    assert off.gpu_resident_test_image_count() == len(off._test_cameras)


def test_holdout_identity_and_hashes_unchanged():
    """The comparability foundation: nothing about WHICH frames are holdouts,
    or the hashes derived from them, may move. All three hashes in the
    Run Contracts' ladder derive from frame indices / camera identity and
    never from pixels -- this pins that."""
    on, res_on = _ingest_all(n_frames=40, holdout_every=4, offload=True)
    off, res_off = _ingest_all(n_frames=40, holdout_every=4, offload=False)

    assert on._holdout_indices == off._holdout_indices
    idx_on = [c._streaming_frame.index for c in on._test_cameras]
    idx_off = [c._streaming_frame.index for c in off._test_cameras]
    assert idx_on == idx_off

    # holdout_hash (selection rule) -- StreamingScene.eval_metadata
    meta_on = StreamingScene.eval_metadata(on)
    meta_off = StreamingScene.eval_metadata(off)
    assert meta_on == meta_off
    assert meta_on["holdout_hash"] is not None

    # realized_holdout_hash (the pool) and scored_holdout_hash (the scored
    # subset) both go through _camera_set_hash.
    assert _camera_set_hash(on._test_cameras) == _camera_set_hash(off._test_cameras)
    assert _camera_set_hash(on._test_cameras[:5]) == _camera_set_hash(off._test_cameras[:5])
    assert _camera_set_hash(on._test_cameras) is not None

    # Train-side bookkeeping is equally untouched.
    assert on._keyframes_admitted == off._keyframes_admitted
    assert on._keyframes_rejected == off._keyframes_rejected
    assert [c._streaming_frame.index for c in on.train_cameras] == \
           [c._streaming_frame.index for c in off.train_cameras]
    assert [r[2] for r in res_on] == [r[2] for r in res_off]  # is_train per frame


def test_holdouts_excluded_from_training_replay_and_insertion():
    scene, results = _ingest_all(n_frames=40, holdout_every=4, offload=True)
    holdout_idx = scene._holdout_indices
    train_idx = {c._streaming_frame.index for c in scene.train_cameras}
    replay_idx = {c._streaming_frame.index for c in scene._replay_buffer}
    reservoir_idx = {c._streaming_frame.index for c in scene._global_reservoir}

    assert holdout_idx & train_idx == set()
    assert holdout_idx & replay_idx == set()
    assert holdout_idx & reservoir_idx == set()
    # ingest_next_frame reports is_train=False for a holdout, so the caller's
    # depth-insertion branch (gated on is_train) never sees it.
    for cam, frame, is_train in results:
        if frame.index in holdout_idx:
            assert is_train is False
            assert cam is not None  # ...but it IS a real eval camera
    # A holdout is never handed to the admission/materialize path either.
    for cam in scene._test_cameras:
        assert cam._materialize_calls == 0


# ---------------------------------------------------------------------------
# 4. Evaluation returns residency to the pre-eval bound
# ---------------------------------------------------------------------------
def test_gt_fetch_uploads_a_transient_copy_and_leaves_the_camera_cpu_resident():
    """`utils.comparison_report._gt_image` is the single GT-fetch helper every
    report/PSNR path funnels through. It must copy to the render device WITHOUT
    rebinding `cam.original_image` -- that is what makes a full metrics
    traversal O(1) in residency instead of re-accumulating the whole test set.
    """
    from eligsir.utils.comparison_report import _gt_image

    cams = []
    for i in range(50):
        cam = SimpleNamespace()
        cam.original_image = _FakeTensor(torch.rand(3, 8, 8), device_type="cpu")
        from eligsir.utils.image_provenance import provenance_for_canonical_float

        cam.score_provenance = provenance_for_canonical_float(
            "float32", (3, 8, 8), (8, 8)
        )
        cams.append(cam)

    for cam in cams:
        gt = _gt_image(cam)
        assert gt.device.type == "cuda"          # transient copy on the render device
        assert cam.original_image.device.type == "cpu"  # camera state untouched
    assert gpu_resident_image_count(cams) == 0


def test_offload_test_camera_images_restores_the_bound_after_an_eval():
    """The enforcement point wired in after every mid-training PSNR eval: even
    if some future eval path DOES page holdout images in, residency returns to
    the pre-eval bound instead of ratcheting upward over the run."""
    scene, _ = _ingest_all(n_frames=40, holdout_every=4, offload=True)
    assert scene.gpu_resident_test_image_count() == 0

    # Simulate a regression: an eval path pages every holdout onto the GPU.
    for cam in scene._test_cameras:
        cam.original_image = cam.original_image.to("cuda")
    assert scene.gpu_resident_test_image_count() == len(scene._test_cameras)

    moved = scene.offload_test_camera_images()
    assert moved == len(scene._test_cameras)
    assert scene.gpu_resident_test_image_count() == 0
    # Idempotent: a second call moves nothing and changes nothing.
    assert scene.offload_test_camera_images() == 0
    assert scene.gpu_resident_test_image_count() == 0


def test_offload_helpers_ignore_cameras_without_images():
    cams = [SimpleNamespace(), SimpleNamespace(original_image=None)]
    assert offload_camera_images(cams) == 0
    assert gpu_resident_image_count(cams) == 0
    assert gpu_resident_image_bytes(cams) == 0


def test_offload_disabled_flag_is_respected():
    """`--no-streaming_offload_camera_images` must reproduce the exact
    pre-#398 residency behavior, including for the post-eval restore."""
    scene, _ = _ingest_all(n_frames=40, holdout_every=4, offload=False)
    assert scene.gpu_resident_test_image_count() == len(scene._test_cameras)
    assert scene.offload_test_camera_images() == 0
    assert scene.gpu_resident_test_image_count() == len(scene._test_cameras)
