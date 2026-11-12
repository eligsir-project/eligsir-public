"""Issue #396: replay-camera GPU re-residency lifecycle.

The hypothesis under test is that ``StreamingScene``'s camera-image offload is
**one-way**: the admission path pages a camera out once, with a monotonic
single-index cursor, and a later replay page-in
(``prepare_camera_for_render``) is never undone -- so the resident set becomes
``hot window + every historical camera ever replayed`` instead of the hot window.

Everything here runs CPU-only (no CUDA, per #478's collection gating). The
device-residency plumbing is exercised through the **real**
``scene.cameras.prepare_camera_for_render`` / ``offload_camera_image`` and the
**real** ``StreamingScene`` residency methods: only the image tensor itself is a
duck-typed stand-in (``_FakeImage``) carrying a mutable ``.device``. Both
production functions touch an image only via ``.device``, ``.detach()``,
``.to(device=...)`` and ``.contiguous()``, so substituting the tensor exercises
the real control flow rather than re-implementing it -- which is what lets a
host without a GPU observe a *GPU* residency policy.

What this file does NOT do: it does not measure bytes, and it does not prove any
claim about a real ScanNet++ run's VRAM. Byte-level per-subsystem residency
attribution is issue #399 / PR #513's ``GPUMemoryResidencyTracker``; the resident
*count* asserted here is the same quantity that tracker reports as
``cold_resident_count``, and once #513 lands these assertions should be
re-expressed against it rather than against the local counting helper.
"""

import pytest

from scene.cameras import offload_camera_image, prepare_camera_for_render
from scene.streaming_scene import StreamingScene
from eligsir.utils.camera_residency import CameraResidencyLedger


# ------------------------------------------------------------------ fakes ----


class _Device:
    """Minimal stand-in for ``torch.device`` with the one field the code reads."""

    def __init__(self, type_):
        self.type = type_

    def __eq__(self, other):
        return getattr(other, "type", None) == self.type

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"_Device({self.type!r})"


class _FakeImage:
    """Duck-typed image tensor: only ``.device``/``.detach()``/``.to()``/
    ``.contiguous()`` are ever touched by the residency functions under test."""

    def __init__(self, device="cpu"):
        self.device = _Device(device)

    def detach(self):
        return self

    def contiguous(self):
        return self

    def to(self, device=None, **_kwargs):
        out = _FakeImage()
        # `device` arrives as either a str or a torch.device; both expose enough
        # to recover the device *type*, which is all this policy branches on.
        out.device = _Device(getattr(device, "type", None) or str(device).split(":")[0])
        return out


class _AnyDevice:
    """Compares equal to any device, so ``prepare_camera_for_render`` skips the
    gsplat-tensor rebuild and only exercises the image-paging branch."""

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


class _FakeGsplat:
    def __init__(self):
        self.viewmat = type("V", (), {"device": _AnyDevice()})()


class _FakeCam:
    def __init__(self, idx, device="cuda"):
        self.uid = idx
        self.original_image = _FakeImage(device)
        # Pre-populated so prepare_camera_for_render does not try to build real
        # gsplat tensors (which would need a real torch camera).
        self.gsplat = _FakeGsplat()

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"_FakeCam({self.uid}, {self.original_image.device.type})"


class _FakeArgs:
    def __init__(self, hot_window):
        self.streaming_keyframe_window = hot_window
        self.streaming_offload_camera_images = True


class _FakeScene:
    """Only the state the real residency methods read, with those methods bound
    from the real ``StreamingScene`` (same pattern as
    ``tests/test_partition_focus_sampling.py``)."""

    def __init__(self, hot_window=4, sampling_window=None):
        self.args = _FakeArgs(hot_window)
        self.train_cameras = []
        self.current_camera = None
        self._sampling_window = sampling_window

    # --- real production code under test ---------------------------------
    _offload_camera_hot_window = StreamingScene._offload_camera_hot_window
    _offload_evicted_camera = StreamingScene._offload_evicted_camera
    hot_camera_ids = StreamingScene.hot_camera_ids
    _enforce_hot_window = StreamingScene._enforce_hot_window

    def _effective_keyframe_window(self):
        """The coverage-ADAPTIVE sampling window. Nothing in the residency path
        may consult this -- see the huge-sampling-window test below."""
        if self._sampling_window is None:
            return self.args.streaming_keyframe_window
        return self._sampling_window

    # --- harness ----------------------------------------------------------
    def admit(self, cam):
        """The residency-relevant half of ``ingest_next_frame``: append, then run
        the single-index eviction cursor, then page the new camera in."""
        self.train_cameras.append(cam)
        self._offload_evicted_camera()
        self.current_camera = cam
        prepare_camera_for_render(cam, device="cuda")


def pre_fix_inline_cursor(scene):
    """Verbatim transcription of the eviction block that was INLINE in
    ``ingest_next_frame`` before this fix (``scene/streaming_scene.py:2284-2294``
    at commit ``f00d945``), kept so the leak can be reproduced against an
    unmodified checkout of ``main`` -- which has no
    ``_offload_evicted_camera`` to bind -- and so the extraction can be proven
    behaviour-identical (see the equivalence test below)."""
    if bool(getattr(scene.args, "streaming_offload_camera_images", True)):
        hot_window = max(1, int(scene.args.streaming_keyframe_window))
        evict_at = len(scene.train_cameras) - hot_window - 1
        if evict_at >= 0:
            offload_camera_image(scene.train_cameras[evict_at])


def resident_count(cams):
    """Number of cameras holding a device-resident image.

    Same quantity #399/PR #513's ``GPUMemoryResidencyTracker`` reports as
    ``hot_resident_count + cold_resident_count``; counted locally here only
    because that module is not on ``main`` yet.
    """
    return sum(1 for c in cams if c.original_image.device.type != "cpu")


def cold_resident_count(scene):
    """Cameras resident *past the boundary the offload policy evicts at* -- the
    #396 leak counter. Keyed on the RAW keyframe window, never on
    ``_effective_keyframe_window()``."""
    window = scene.args.streaming_keyframe_window
    cold = scene.train_cameras[:-window] if window else scene.train_cameras
    return resident_count(cold)


# ------------------------------------------------- fake-plumbing sanity ------


def test_fake_image_round_trips_through_the_real_residency_functions():
    """If this fails, every other assertion in this file is meaningless."""
    cam = _FakeCam(0, device="cuda")
    assert cam.original_image.device.type == "cuda"
    assert offload_camera_image(cam) is True
    assert cam.original_image.device.type == "cpu"
    prepare_camera_for_render(cam, device="cuda")
    assert cam.original_image.device.type == "cuda"


# --------------------------------------------------- the eviction cursor -----


def test_extracted_cursor_matches_pre_fix_inline_transcription():
    """The extraction is behaviour-neutral: same page-out decision, same
    iteration, over a full admission sequence at several window sizes."""
    for window in (1, 3, 4, 7, 120):
        a = _FakeScene(hot_window=window)
        b = _FakeScene(hot_window=window)
        for i in range(300):
            cam_a, cam_b = _FakeCam(i), _FakeCam(i)
            a.train_cameras.append(cam_a)
            b.train_cameras.append(cam_b)
            a._offload_evicted_camera()
            pre_fix_inline_cursor(b)
            devs_a = [c.original_image.device.type for c in a.train_cameras]
            devs_b = [c.original_image.device.type for c in b.train_cameras]
            assert devs_a == devs_b, (window, i)


def test_admission_alone_bounds_residency_to_the_hot_window():
    scene = _FakeScene(hot_window=4)
    for i in range(20):
        scene.admit(_FakeCam(i))
    assert resident_count(scene.train_cameras) == 4
    assert cold_resident_count(scene) == 0


def test_eviction_cursor_never_revisits_an_index():
    """The mechanism behind the leak, pinned directly: the cursor offloads index
    ``len-hot_window-1`` exactly once. Re-paging that camera in leaves it
    resident forever as far as the admission path is concerned."""
    scene = _FakeScene(hot_window=4)
    for i in range(20):
        scene.admit(_FakeCam(i))
    leaked = scene.train_cameras[3]
    prepare_camera_for_render(leaked, device="cuda")  # replay page-in
    for i in range(20, 40):  # 20 more admissions, cursor is far past index 3
        scene.admit(_FakeCam(i))
    assert leaked.original_image.device.type == "cuda", (
        "the admission cursor is expected NOT to be able to re-evict this; if it "
        "can, the leak premise has changed and the ledger may be redundant"
    )
    assert cold_resident_count(scene) == 1


# ---------------------------------------------------------- the leak/fix -----


def _replay_without_release(scene, cams):
    """Reproduces current-main behaviour: page in, use, never page out."""
    for cam in cams:
        prepare_camera_for_render(cam, device="cuda")


def test_replay_without_release_is_unbounded_this_is_the_bug():
    """Baseline that FAILS the intended bound -- the leak, demonstrated."""
    scene = _FakeScene(hot_window=4)
    for i in range(50):
        scene.admit(_FakeCam(i))
    cold = scene.train_cameras[:-4]
    _replay_without_release(scene, cold)
    assert resident_count(scene.train_cameras) == 50
    assert cold_resident_count(scene) == len(cold) == 46


def test_ledger_release_bounds_residency_across_replay_of_every_cold_camera():
    """The fix: 46 distinct cold cameras replayed once each, one per iteration,
    with the ledger released at the top of the next iteration."""
    scene = _FakeScene(hot_window=4)
    for i in range(50):
        scene.admit(_FakeCam(i))
    ledger = CameraResidencyLedger()
    for cam in scene.train_cameras[:-4]:
        ledger.release(scene.hot_camera_ids())  # top of iteration
        prepare_camera_for_render(cam, device="cuda")
        ledger.note_paged_in(cam)
    ledger.release(scene.hot_camera_ids())  # top of the following iteration
    assert cold_resident_count(scene) == 0
    assert resident_count(scene.train_cameras) == 4
    assert ledger.stats()["paged_out"] == 46


def test_residency_does_not_grow_with_history_length():
    """The acceptance criterion: replaying thousands of old views does not make
    the bound a function of history length."""
    bounds = {}
    for n in (200, 400, 800):
        scene = _FakeScene(hot_window=4)
        for i in range(n):
            scene.admit(_FakeCam(i))
        ledger = CameraResidencyLedger()
        for cam in scene.train_cameras[:-4]:
            ledger.release(scene.hot_camera_ids())
            prepare_camera_for_render(cam, device="cuda")
            ledger.note_paged_in(cam)
        ledger.release(scene.hot_camera_ids())
        bounds[n] = resident_count(scene.train_cameras)
    assert set(bounds.values()) == {4}, bounds


def test_repeated_replay_of_the_same_cold_camera_stays_bounded():
    scene = _FakeScene(hot_window=4)
    for i in range(30):
        scene.admit(_FakeCam(i))
    ledger = CameraResidencyLedger()
    victim = scene.train_cameras[0]
    for _ in range(500):
        ledger.release(scene.hot_camera_ids())
        prepare_camera_for_render(victim, device="cuda")
        ledger.note_paged_in(victim)
    ledger.release(scene.hot_camera_ids())
    assert cold_resident_count(scene) == 0
    # One page-out per iteration, not an accumulating set.
    assert ledger.stats()["paged_out"] == 500
    assert ledger.max_concurrent_paged_in == 1


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_batched_supervision_transient_rise_returns_to_the_bound(batch):
    """B>1: residency may transiently rise by the batch size, and every view of
    the batch must survive until the whole batch is done -- no mid-batch
    offload could page a tensor out from under another view's backward."""
    scene = _FakeScene(hot_window=4)
    for i in range(60):
        scene.admit(_FakeCam(i))
    ledger = CameraResidencyLedger()
    cold = scene.train_cameras[:-4]
    for start in range(0, len(cold) - batch, batch):
        ledger.release(scene.hot_camera_ids())
        views = cold[start:start + batch]
        for cam in views:
            prepare_camera_for_render(cam, device="cuda")
            ledger.note_paged_in(cam)
        # Every view of the batch is simultaneously resident, mid-step.
        assert all(c.original_image.device.type == "cuda" for c in views)
        assert cold_resident_count(scene) <= 4 + batch
    ledger.release(scene.hot_camera_ids())
    assert cold_resident_count(scene) == 0
    assert ledger.max_concurrent_paged_in == batch


def test_hot_camera_sampled_through_the_replay_path_stays_resident():
    """A HOT camera reached via the replay API must NOT be paged out -- that
    would cost an H2D transfer every iteration for no memory benefit."""
    scene = _FakeScene(hot_window=4)
    for i in range(20):
        scene.admit(_FakeCam(i))
    hot = scene.train_cameras[-1]
    ledger = CameraResidencyLedger()
    prepare_camera_for_render(hot, device="cuda")
    ledger.note_paged_in(hot)
    ledger.release(scene.hot_camera_ids())
    assert hot.original_image.device.type == "cuda"
    assert ledger.stats()["paged_out"] == 0
    assert ledger.stats()["retained_hot"] == 1


def test_current_camera_outside_train_cameras_is_treated_as_hot():
    """H8 warmup returns ``current_camera`` directly for N consecutive
    iterations, and on a holdout frame it is not in ``train_cameras`` at all."""
    scene = _FakeScene(hot_window=2)
    for i in range(10):
        scene.admit(_FakeCam(i))
    holdout = _FakeCam(999)
    scene.current_camera = holdout
    prepare_camera_for_render(holdout, device="cuda")
    ledger = CameraResidencyLedger()
    ledger.note_paged_in(holdout)
    ledger.release(scene.hot_camera_ids())
    assert holdout.original_image.device.type == "cuda"


# --------------------------------------------- the effective-window trap -----


def test_release_offloads_leaked_camera_even_when_sampling_window_is_huge():
    """THE high-risk regression guard.

    At the real default ``--streaming_keyframe_coverage 0.6`` the coverage-
    adaptive *sampling* window (``_effective_keyframe_window()``) is ~4800 on an
    8000-frame ScanNet++ run, while the boundary the offload policy actually
    evicts at is the raw ``--streaming_keyframe_window`` (120). A hotness
    predicate keyed on the sampling window would classify a leaked camera at
    position -3000 as HOT and the whole mechanism would silently become a no-op
    on exactly the runs it was written for. Mirrors PR #513's counter-side test.
    """
    scene = _FakeScene(hot_window=120, sampling_window=4800)
    for i in range(8000):
        scene.admit(_FakeCam(i))
    assert scene._effective_keyframe_window() == 4800  # the trap is armed
    assert scene._offload_camera_hot_window() == 120  # policy ignores it

    leaked = scene.train_cameras[-3000]
    prepare_camera_for_render(leaked, device="cuda")
    ledger = CameraResidencyLedger()
    ledger.note_paged_in(leaked)
    ledger.release(scene.hot_camera_ids())

    assert leaked.original_image.device.type == "cpu"
    assert resident_count(scene.train_cameras) == 120
    assert cold_resident_count(scene) == 0


def test_hot_camera_ids_size_tracks_the_raw_window_not_the_sampling_window():
    scene = _FakeScene(hot_window=120, sampling_window=4800)
    for i in range(8000):
        scene.admit(_FakeCam(i))
    # current_camera is train_cameras[-1], already in the tail -> exactly 120.
    assert len(scene.hot_camera_ids()) == 120


# --------------------------------------------------------- escape hatch ------


def test_disabled_flag_makes_the_ledger_inert():
    """``--no-streaming_offload_camera_images`` must disable the whole paging
    policy, not only the eviction half of it."""
    scene = _FakeScene(hot_window=4)
    scene.args.streaming_offload_camera_images = False
    for i in range(20):
        scene.admit(_FakeCam(i))
    assert resident_count(scene.train_cameras) == 20  # nothing evicted
    ledger = CameraResidencyLedger(enabled=False)
    for cam in scene.train_cameras:
        ledger.note_paged_in(cam)
    assert ledger.pending_count() == 0
    assert ledger.release(scene.hot_camera_ids()) == 0
    assert resident_count(scene.train_cameras) == 20


def test_release_with_no_page_ins_is_a_noop():
    ledger = CameraResidencyLedger()
    assert ledger.release(set()) == 0
    assert ledger.stats()["paged_in"] == 0


def test_release_routes_batchable_page_out_through_injected_transfer_owner():
    class _Owner:
        def __init__(self):
            self.calls = []

        def offload_cameras(self, cameras):
            cameras = tuple(cameras)
            self.calls.append(cameras)
            for cam in cameras:
                cam.original_image = _FakeImage("cpu")
            return len(cameras)

    owner = _Owner()
    ledger = CameraResidencyLedger(transfer_owner=owner)
    cam = _FakeCam(7, device="cuda")
    cam.original_image.shape = (3, 4, 5)
    cam.original_image.dtype = "float32"
    ledger.note_paged_in(cam)

    assert ledger.release(set()) == 1
    assert owner.calls == [(cam,)]
    assert cam.original_image.device.type == "cpu"


def test_note_paged_in_dedups_repeat_sampling_within_one_iteration():
    """A camera can be handed back more than once in one iteration (H8 warmup,
    the consolidation-burst queue). It must be released once, not N times."""
    scene = _FakeScene(hot_window=2)
    for i in range(10):
        scene.admit(_FakeCam(i))
    cam = scene.train_cameras[0]
    ledger = CameraResidencyLedger()
    prepare_camera_for_render(cam, device="cuda")
    for _ in range(5):
        ledger.note_paged_in(cam)
    assert ledger.pending_count() == 1
    assert ledger.release(scene.hot_camera_ids()) == 1
    assert ledger.stats()["paged_in"] == 1


# ---------------------------------------- the un-fixed backstop, pinned ------


def test_enforce_hot_window_still_uses_the_same_boundary():
    """``_enforce_hot_window`` remains the periodic backstop for page-in paths
    that do not register with the ledger. Its boundary must not drift from the
    ledger's -- both now route through ``_offload_camera_hot_window``."""
    scene = _FakeScene(hot_window=3, sampling_window=900)
    for i in range(40):
        scene.admit(_FakeCam(i))
    for cam in scene.train_cameras:
        prepare_camera_for_render(cam, device="cuda")
    assert resident_count(scene.train_cameras) == 40
    scene._enforce_hot_window()
    assert resident_count(scene.train_cameras) == 3
