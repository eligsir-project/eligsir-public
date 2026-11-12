"""CPU-verifiable regression coverage for issue #399's residency telemetry.

Everything here runs on a CPU-only host (CI collects tests without CUDA, #478)
by constructing the tracker with ``device="cpu"`` and injecting the allocator
readers -- so the FULL attribution walk (including the ``unattributed``
residual) is exercised, not skipped.

The load-bearing test is ``test_attribution_stays_bounded_as_history_grows``:
it fakes frame-history growth (the real long GPU run is explicitly out of scope
for #399) and asserts that

  * the residency buckets a bounded offload policy is supposed to bound really
    do stay bounded, while the deliberately-leaking bucket is *detected* as
    growing (the telemetry must be able to see a leak, not just be quiet);
  * the telemetry's OWN retained series stays bounded, i.e. it does not
    reproduce the unbounded-growth bug class it exists to measure.

**HONESTY FENCE, read before citing this file anywhere.** Every test in this
module runs entirely against synthetic/fake tensors and fake scene/camera/
gaussian objects on CPU. They pin the telemetry's *bookkeeping arithmetic and
invariants* -- dedup by untyped storage, first-claim-wins bucket priority,
bucket-sum-vs-attributed-total reconciliation, the hot/cold/holdout split, the
``cold_resident_count`` discriminator keying on the offload bound rather than
the sampling window, and graceful (non-crashing, zero-valued) behavior when a
bucket is empty. **They do NOT and cannot prove issue #399's actual
acceptance criterion -- that persistent GPU memory plateaus instead of
growing with total history length on a real long streaming run.** That needs
a real GPU run (no GPU was available when these tests were added: the single
iGPU was held by a live ablator job, and satellite machines are unusable per
issue #566). Issue #399 must stay OPEN until that run happens; this file is
necessary supporting evidence, not sufficient evidence, for closing it.
"""

import time

import pytest
import torch

from eligsir.utils.gpu_memory_telemetry import (
    BUCKET_ORDER,
    GPUMemoryResidencyTracker,
    disabled_report_block,
    residency_owner_contract,
)


# ---------------------------------------------------------------- fakes ------


class _FakeCam:
    def __init__(self, h=8, w=8, resident=True, frame_index=None):
        self.original_image = torch.zeros(3, h, w) if resident else None
        self._sensor_depth_cache = torch.zeros(1, h, w)
        self._streaming_frame = type("F", (), {"index": frame_index})()

    def offload(self):
        self.original_image = None


class _FakeArgs:
    streaming_keyframe_window = 4
    streaming_keyframe_coverage = 0.0
    streaming_bootstrap_in_window = False


class _FakeScene:
    """Only the attributes the tracker's camera walk reads.

    ``window`` is the RAW ``--streaming_keyframe_window`` (the offload bound).
    ``sampling_window`` defaults to the same value, but can be set larger to
    reproduce the real default configuration, where
    ``streaming_keyframe_coverage=0.6`` makes ``_effective_keyframe_window()``
    an order of magnitude bigger than the boundary eviction uses.
    """

    def __init__(self, window=4, sampling_window=None):
        self.args = _FakeArgs()
        self.args.streaming_keyframe_window = window
        self.train_cameras = []
        self._replay_buffer = []
        self._global_reservoir = []
        self._test_cameras = []
        self.observation_db = None
        self._window = window
        self._sampling_window = window if sampling_window is None else sampling_window

    def _effective_keyframe_window(self):
        return self._sampling_window


class _FakeHandle:
    def __init__(self, h, w):
        self.rgb = torch.zeros(3, h, w)
        self.depth = torch.zeros(1, h, w)


class _FakeRecord:
    def __init__(self, h, w, tiers=("high",)):
        self.resolution_cache = {t: _FakeHandle(h, w) for t in tiers}


class _FakeDB:
    def __init__(self):
        self._frames = {}


class _FakeGaussians:
    def __init__(self, n=16):
        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.zeros(n, 3)),
                "opacities": torch.nn.Parameter(torch.zeros(n)),
            }
        )
        # A real Adam-style state dict keyed by the parameter objects.
        self.optimizer = type(
            "Opt",
            (),
            {
                "state": {
                    p: {"exp_avg": torch.zeros_like(p), "exp_avg_sq": torch.zeros_like(p)}
                    for p in self.params.values()
                }
            },
        )()
        self.birth_iter = torch.zeros(n, dtype=torch.int32)
        self.gaussian_uid = torch.zeros(n, dtype=torch.int32)


def _tracker(allocated_fn, **kw):
    return GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=allocated_fn,
        reserved_bytes_fn=lambda: allocated_fn() * 2,
        peak_bytes_fn=allocated_fn,
        **kw,
    )


# ------------------------------------------------------- attribution ---------


def test_buckets_are_real_and_deduped_by_storage():
    g = _FakeGaussians(n=32)
    scene = _FakeScene(window=2)
    scene.train_cameras = [_FakeCam() for _ in range(5)]

    trk = _tracker(lambda: 10**9)
    s = trk.snapshot(gaussians=g, streaming_scene=scene, frames_ingested=5, iteration=100)

    b = s["buckets"]
    assert b["gaussian_params"] == 32 * 3 * 4 + 32 * 4
    # Optimizer state is billed separately and never double-bills a parameter.
    assert b["gaussian_optimizer"] == 2 * b["gaussian_params"]
    assert b["gaussian_buffers"] == 32 * 4 * 2  # birth_iter + gaussian_uid
    # window=2 => 2 hot, 3 cold.
    assert s["n_hot_cameras"] == 2 and s["n_cold_cameras"] == 3
    assert s["hot_resident_count"] == 2 and s["cold_resident_count"] == 3
    assert b["camera_hot_images"] == 2 * 3 * 8 * 8 * 4
    assert b["camera_cold_images"] == 3 * 3 * 8 * 8 * 4
    assert b["camera_depth_cache"] == 5 * 1 * 8 * 8 * 4
    # Everything the walk did not reach is explicit, never folded in.
    assert b["unattributed"] == 10**9 - s["attributed_bytes"]
    assert set(b) == set(BUCKET_ORDER) | {"unattributed"}


def test_views_of_one_storage_are_billed_once():
    """A view/slice shares its base storage; keying on data_ptr would both
    double-count and under-size it. Dedup must be by untyped storage."""
    scene = _FakeScene(window=1)
    base = torch.zeros(3, 8, 8)
    cam_a = _FakeCam()
    cam_a.original_image = base
    cam_b = _FakeCam()
    cam_b.original_image = base[:, 2:, :]  # a view: different data_ptr, same storage
    scene.train_cameras = [cam_a, cam_b]

    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene)
    total = s["buckets"]["camera_hot_images"] + s["buckets"]["camera_cold_images"]
    assert total == base.untyped_storage().nbytes()


def test_offloaded_cold_camera_is_not_counted_resident():
    scene = _FakeScene(window=2)
    scene.train_cameras = [_FakeCam() for _ in range(6)]
    for cam in scene.train_cameras[:-2]:
        cam.offload()

    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene)
    assert s["cold_resident_count"] == 0
    assert s["buckets"]["camera_cold_images"] == 0
    assert s["hot_resident_count"] == 2


def test_phase_snapshot_reports_current_bytes_without_claiming_a_torch_global_peak():
    """A report boundary is one current allocator sample, not a peak reset.

    Removing the phase label, substituting the tracker's sampled peak for the
    current values, or reading ``torch.cuda.max_memory_allocated`` here must
    break this contract.
    """
    reads = {"allocated": 0, "reserved": 0}

    def allocated():
        reads["allocated"] += 1
        return 100

    def reserved():
        reads["reserved"] += 1
        return 160

    tracker = GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=allocated,
        reserved_bytes_fn=reserved,
        peak_bytes_fn=lambda: (_ for _ in ()).throw(AssertionError("no torch peak read")),
    )

    sample = tracker.phase_snapshot("report_entry")

    assert sample == {
        "phase": "report_entry",
        "status": "measured",
        "current_allocated_bytes": 100,
        "current_reserved_bytes": 160,
        "current_reserved_overhead_bytes": 60,
        "sampled_peak_allocated_bytes": 100,
        "sampled_peak_reserved_bytes": 160,
        "torch_global_peak_allocated_bytes": None,
        "torch_global_peak_reason": "not_read_by_phase_sample",
    }
    assert reads == {"allocated": 1, "reserved": 1}


def test_leak_past_offload_bound_is_counted_even_when_sampling_window_is_huge():
    """The #396 discriminator must key on the OFFLOAD bound, not the sampling
    window.

    Real defaults: ``--streaming_keyframe_window 120`` is what
    ``ingest_next_frame`` evicts against, but
    ``_effective_keyframe_window()`` returns
    ``max(120, 0.6 * n_total)`` -- ~4800 on an 8000-frame trajectory. A camera
    at position -3000 that a global replay paged back in and never paged out is
    the leak; keying on the sampling window would classify it "hot" and never
    count it. This test pins that: it fails if the classification regresses to
    ``_effective_keyframe_window()``.
    """
    offload_bound = 120
    n = 8000
    scene = _FakeScene(window=offload_bound, sampling_window=int(0.6 * n))
    scene.train_cameras = [
        _FakeCam(resident=(i >= n - offload_bound), frame_index=i) for i in range(n)
    ]
    # The leak: one old camera replayed back onto the device, never paged out.
    leaked_idx = n - 3000
    scene.train_cameras[leaked_idx].original_image = torch.zeros(3, 8, 8)

    trk = _tracker(lambda: 10**9)
    s = trk.snapshot(streaming_scene=scene, frames_ingested=n)

    assert s["offload_bound"] == offload_bound
    assert s["sampling_window"] == 4800
    assert s["n_hot_cameras"] == offload_bound
    assert s["hot_resident_count"] == offload_bound
    assert s["cold_resident_count"] == 1
    assert s["resident_beyond_offload_bound_count"] == 1
    assert s["resident_beyond_offload_bound_bytes"] == 3 * 8 * 8 * 4


def test_signed_residual_exposes_over_attribution():
    """`unattributed` is clamped at 0 so it reads as a byte count; the signed
    residual must still show over-attribution, which is the case where the
    diagnostic itself is wrong."""
    scene = _FakeScene(window=1)
    scene.train_cameras = [_FakeCam()]
    # Allocator reports LESS than the walk attributes.
    trk = _tracker(lambda: 1)
    s = trk.snapshot(streaming_scene=scene)
    assert s["buckets"]["unattributed"] == 0
    assert s["residual_bytes_signed"] < 0
    assert s["residual_bytes_signed"] == 1 - s["attributed_bytes"]


def test_holdout_residency_is_its_own_discriminator():
    """#398: holdout pixels must be reportable independently of train cameras."""
    scene = _FakeScene(window=2)
    scene.train_cameras = [_FakeCam() for _ in range(2)]
    scene._test_cameras = [_FakeCam() for _ in range(7)]

    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene)
    assert s["n_holdout_cameras"] == 7
    assert s["holdout_resident_count"] == 7
    assert s["holdout_resident_bytes"] == 7 * 3 * 8 * 8 * 4
    assert s["buckets"]["camera_hot_images"] == 2 * 3 * 8 * 8 * 4


def test_multires_double_representation_is_detected():
    """#333: a camera holding BOTH a resident original_image and a high-tier
    cache handle is the 'two copies of native-resolution RGB' case."""
    scene = _FakeScene(window=3)
    scene.train_cameras = [_FakeCam(frame_index=i) for i in range(3)]
    db = _FakeDB()
    db._frames = {0: _FakeRecord(8, 8), 1: _FakeRecord(8, 8, tiers=("low",))}
    scene.observation_db = db

    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene)
    assert s["multires_cached_frames"] == 2
    # frame 0 has a high tier AND a resident image; frame 1 only has "low".
    assert s["multires_double_represented"] == 1
    assert s["multires_bytes"] == 2 * (3 * 8 * 8 * 4 + 1 * 8 * 8 * 4)


# --------------------------------------------------- bookkeeping invariants --
#
# Added under issue #399's follow-up (agent/issue399-memory-bookkeeping-
# invariants): these pin arithmetic invariants the module's docstring asserts
# but the original 12-test suite did not directly exercise -- bucket-sum
# reconciliation, first-claim-wins dedup across a param/buffer alias, graceful
# behavior on empty inputs, and the offload-bound-vs-sampling-window
# independence that PR #513's own follow-up comment corrected. See this
# module's HONESTY FENCE above: bookkeeping arithmetic only, no real-hardware
# claim.


def test_bucket_sum_reconciles_with_allocated_bytes_when_headroom_exists():
    """sum(named buckets) must equal attributed_bytes exactly, and
    attributed_bytes + unattributed must reconcile with allocated_bytes when
    the allocator reports enough headroom that the unattributed clamp never
    engages (the common, non-pathological case)."""
    g = _FakeGaussians(n=10)
    scene = _FakeScene(window=3)
    scene.train_cameras = [_FakeCam() for _ in range(9)]
    scene._test_cameras = [_FakeCam() for _ in range(4)]
    db = _FakeDB()
    db._frames = {0: _FakeRecord(8, 8)}
    scene.observation_db = db

    allocated = 5 * 10**7
    trk = _tracker(lambda: allocated)
    s = trk.snapshot(gaussians=g, streaming_scene=scene)

    named_bucket_sum = sum(s["buckets"][b] for b in BUCKET_ORDER)
    assert named_bucket_sum == s["attributed_bytes"]
    assert s["buckets"]["unattributed"] == allocated - s["attributed_bytes"]
    assert named_bucket_sum + s["buckets"]["unattributed"] == allocated
    assert s["residual_bytes_signed"] == allocated - s["attributed_bytes"]


def test_first_claim_wins_when_a_buffer_aliases_a_param_storage():
    """A tensor already billed to an earlier-priority bucket (gaussian_params)
    must not be re-billed when a later-priority bucket (gaussian_buffers)
    also references the identical storage -- exactly the scenario the
    module's docstring calls out ("a buffer that aliases a parameter")."""
    g = _FakeGaussians(n=8)
    baseline = _tracker(lambda: 10**6).snapshot(gaussians=g)

    # Alias a param's storage into a plain attribute -- vars(gaussians) walks
    # this under gaussian_buffers, the lowest-priority model bucket.
    g.alias_of_means = g.params["means"]
    aliased = _tracker(lambda: 10**6).snapshot(gaussians=g)

    assert aliased["attributed_bytes"] == baseline["attributed_bytes"], (
        "aliasing a param's storage into a buffer attribute must not "
        "increase the attributed total -- gaussian_params claims it first"
    )
    assert aliased["buckets"]["gaussian_params"] == baseline["buckets"]["gaussian_params"]
    assert aliased["buckets"]["gaussian_buffers"] == baseline["buckets"]["gaussian_buffers"]


def test_snapshot_with_nothing_provided_is_graceful():
    """No gaussians, no streaming_scene, no observation_db -- every bucket
    and discriminator must read a clean zero, not raise."""
    trk = _tracker(lambda: 0)
    s = trk.snapshot()

    assert s["allocated_bytes"] == 0
    assert s["attributed_bytes"] == 0
    for b in BUCKET_ORDER:
        assert s["buckets"][b] == 0
    assert s["buckets"]["unattributed"] == 0
    assert s["residual_bytes_signed"] == 0
    assert s["n_hot_cameras"] == 0
    assert s["n_cold_cameras"] == 0
    assert s["n_holdout_cameras"] == 0
    assert s["cold_resident_count"] == 0
    assert s["holdout_resident_count"] == 0
    assert s["multires_cached_frames"] == 0
    assert s["multires_double_represented"] == 0
    assert s["offload_bound"] == 0
    assert s["sampling_window"] == 0

    # report_block() over a run that only ever took this empty snapshot must
    # also stay graceful.
    block = trk.report_block()
    assert block["n_snapshots"] == 1
    assert block["final"] is not None
    assert block["growth"]["available"] is False


def test_scene_present_with_zero_cameras_does_not_crash():
    """A streaming_scene with empty train/test camera lists (e.g. before the
    first frame is ingested) must not crash the window slicing or the
    multires walk over an empty camera list."""
    scene = _FakeScene(window=50, sampling_window=200)
    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene)

    assert s["n_hot_cameras"] == 0
    assert s["n_cold_cameras"] == 0
    assert s["n_holdout_cameras"] == 0
    assert s["offload_bound"] == 50
    assert s["sampling_window"] == 200
    assert s["buckets"]["camera_hot_images"] == 0
    assert s["buckets"]["camera_cold_images"] == 0
    assert s["buckets"]["camera_holdout_images"] == 0
    assert s["multires_cached_frames"] == 0


def test_hot_cold_classification_is_independent_of_sampling_window_value():
    """Regression pin for PR #513's own correction comment: hot/cold
    residency classification must key on the OFFLOAD bound
    (--streaming_keyframe_window) and must NOT vary with the sampling window
    (_effective_keyframe_window()), which is reported as context only."""
    n = 500
    offload_bound = 30
    results = []
    for sampling_window in (30, 300, 3000):
        scene = _FakeScene(window=offload_bound, sampling_window=sampling_window)
        scene.train_cameras = [_FakeCam(frame_index=i) for i in range(n)]
        trk = _tracker(lambda: 10**9)
        s = trk.snapshot(streaming_scene=scene)
        results.append(
            (s["n_hot_cameras"], s["hot_resident_count"], s["cold_resident_count"])
        )
        assert s["sampling_window"] == sampling_window
        assert s["offload_bound"] == offload_bound

    assert len(set(results)) == 1, (
        f"hot/cold classification must not vary with sampling_window: {results}"
    )


def test_multires_bucket_is_empty_and_graceful_with_no_observation_db():
    scene = _FakeScene(window=2)
    scene.train_cameras = [_FakeCam() for _ in range(2)]
    # observation_db left at its default None.
    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene)
    assert s["multires_cached_frames"] == 0
    assert s["multires_double_represented"] == 0
    assert s["multires_bytes"] == 0


def test_multires_bucket_is_empty_when_db_has_no_frames():
    scene = _FakeScene(window=2)
    scene.train_cameras = [_FakeCam() for _ in range(2)]
    scene.observation_db = _FakeDB()  # _frames == {}
    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene)
    assert s["multires_cached_frames"] == 0
    assert s["multires_bytes"] == 0


def test_reserved_overhead_and_residual_invariants():
    """reserved_overhead_bytes = max(0, reserved - allocated); it must clamp
    at 0 rather than go negative when the (injected) allocator readers report
    reserved < allocated, and residual_bytes_signed must always equal
    allocated - attributed exactly, unclamped."""
    scene = _FakeScene(window=2)
    scene.train_cameras = [_FakeCam() for _ in range(2)]
    g = _FakeGaussians(n=4)

    trk = GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=lambda: 1000,
        reserved_bytes_fn=lambda: 4000,
        peak_bytes_fn=lambda: 1000,
    )
    s = trk.snapshot(gaussians=g, streaming_scene=scene)
    assert s["reserved_bytes"] == 4000
    assert s["reserved_overhead_bytes"] == 3000
    assert s["residual_bytes_signed"] == s["allocated_bytes"] - s["attributed_bytes"]

    trk2 = GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=lambda: 1000,
        reserved_bytes_fn=lambda: 500,  # pathological: reserved < allocated
        peak_bytes_fn=lambda: 1000,
    )
    s2 = trk2.snapshot(gaussians=g, streaming_scene=scene)
    assert s2["reserved_overhead_bytes"] == 0


def test_growth_stats_requires_at_least_two_varying_points():
    scene = _FakeScene(window=1)
    scene.train_cameras = [_FakeCam()]

    trk = _tracker(lambda: 10**6)
    assert trk.growth_stats()["available"] is False  # zero snapshots

    trk.snapshot(streaming_scene=scene, frames_ingested=5)
    g1 = trk.growth_stats()
    assert g1["available"] is False
    assert "fewer than 2" in g1["reason"]

    trk2 = _tracker(lambda: 10**6)
    trk2.snapshot(streaming_scene=scene, frames_ingested=5)
    trk2.snapshot(streaming_scene=scene, frames_ingested=5)  # no variation
    g2 = trk2.growth_stats()
    assert g2["available"] is False
    assert "no variation" in g2["reason"]

    trk3 = _tracker(lambda: 10**6)
    trk3.snapshot(streaming_scene=scene, frames_ingested=5)
    trk3.snapshot(streaming_scene=scene, frames_ingested=10)
    g3 = trk3.growth_stats()
    assert g3["available"] is True
    assert g3["n_points"] == 2


# ------------------------------------------------- boundedness (headline) ----


def test_attribution_stays_bounded_as_history_grows():
    """Fake 4000 frames of history growth against a bounded hot window.

    Asserts the three things #399 has to be able to say:
      1. bounded buckets stay bounded (no growth with frame count);
      2. an unbounded bucket is DETECTED as growing (a quiet telemetry that
         cannot see a leak is worthless);
      3. the telemetry's own retained series is bounded.
    """
    window = 120
    # Realistic default config: the sampling window is coverage-adaptive and
    # much larger than the offload bound; classification must still use the
    # offload bound.
    scene = _FakeScene(window=window, sampling_window=2400)
    db = _FakeDB()
    scene.observation_db = db
    g = _FakeGaussians(n=64)

    # Fixed fake allocator total, so bucket growth (not allocator noise) is
    # what the assertions measure.
    trk = _tracker(lambda: 8 * 10**9, max_history=64)

    resolution_cache_bound = 50  # stands in for --streaming_resolution_cache_max_frames
    n_frames = 4000
    for f in range(n_frames):
        cam = _FakeCam(frame_index=f)
        scene.train_cameras.append(cam)
        # Bounded hot-window offload, as StreamingScene.ingest_next_frame does.
        evict_at = len(scene.train_cameras) - window - 1
        if evict_at >= 0:
            scene.train_cameras[evict_at].offload()
        # Bounded LRU resolution cache, as FrameObservationDB does.
        db._frames[f] = _FakeRecord(8, 8)
        for old in list(db._frames)[:-resolution_cache_bound]:
            db._frames[old].resolution_cache = {}
        # Holdouts DELIBERATELY leak here (every 10th frame stays resident) so
        # the growth detector has something real to catch -- this mirrors
        # #398's suspected behavior, it is not a claim about current code.
        if f % 10 == 0:
            scene._test_cameras.append(_FakeCam(frame_index=f))

        if f % 25 == 0:
            trk.snapshot(
                iteration=f * 10,
                frames_ingested=f,
                gaussians=g,
                streaming_scene=scene,
            )

    block = trk.report_block()
    growth = block["growth"]
    assert growth["available"] is True
    per_bucket = growth["per_bucket_bytes_per_frame"]

    one_image = 3 * 8 * 8 * 4

    # (1) bounded buckets: essentially zero bytes per ingested frame.
    for bucket in (
        "gaussian_params",
        "gaussian_optimizer",
        "camera_hot_images",
        "camera_cold_images",
        "multires_cache",
    ):
        assert abs(per_bucket[bucket]) < 0.01 * one_image, (
            f"{bucket} grew {per_bucket[bucket]:.1f} B/frame; expected bounded"
        )
    final = block["final"]
    assert final["cold_resident_count"] == 0
    assert final["hot_resident_count"] == window
    assert final["multires_cached_frames"] <= resolution_cache_bound
    assert abs(growth["cold_resident_count_per_frame"]) < 1e-6

    # (2) the injected leak IS detected, at the right order of magnitude
    # (one resident holdout image per 10 ingested frames).
    assert per_bucket["camera_holdout_images"] > 0.05 * one_image
    assert growth["holdout_resident_count_per_frame"] == pytest.approx(0.1, rel=0.05)

    # (3) the telemetry's own history is bounded.
    assert block["history_len"] <= trk.max_history
    assert len(block["series"]) <= trk.max_history
    assert block["n_snapshots"] == 160  # 4000/25
    assert block["history_stride"] >= 4
    # Decimation must preserve monotonic iteration order and span the run.
    iters = [s["iteration"] for s in block["series"]]
    assert iters == sorted(iters)
    assert iters[0] == 0 and iters[-1] >= 0.5 * (n_frames - 25) * 10


# --------------------------------------------------------- plumbing ----------


def test_tb_scalars_are_flat_floats():
    scene = _FakeScene(window=1)
    scene.train_cameras = [_FakeCam()]
    trk = _tracker(lambda: 10**6)
    s = trk.snapshot(streaming_scene=scene, gaussians=_FakeGaussians())
    tb = trk.tb_scalars(s)
    assert tb, "expected non-empty TB scalar dict"
    assert all(isinstance(v, float) for v in tb.values())
    assert "memory/bucket_mib/unattributed" in tb
    assert "memory/residency/cold_resident_count" in tb
    assert 0.0 <= tb["memory/attributed_frac"] <= 1.0


def test_report_block_distinguishes_off_from_never_sampled():
    off = disabled_report_block("telemetry disabled (--no-streaming_memory_telemetry)")
    assert off["enabled"] is False and off["n_snapshots"] == 0

    trk = _tracker(lambda: 0)
    never = trk.report_block()
    assert never["enabled"] is True
    assert never["n_snapshots"] == 0
    assert never["final"] is None
    assert never["growth"]["available"] is False


def test_disabled_report_block_has_all_structural_keys():
    """Disabled and enabled arms must have the same structural keys so a
    consumer can read the same paths on both arms and branch on value, not
    key presence (mirroring duty_cycle and the #282 NO_SAMPLES lesson)."""
    disabled = disabled_report_block()

    # Create an enabled arm with samples so it has all structural keys.
    scene = _FakeScene(window=1)
    scene.train_cameras = [_FakeCam()]
    trk = _tracker(lambda: 10**6)
    trk.snapshot(streaming_scene=scene, frames_ingested=1)
    trk.snapshot(streaming_scene=scene, frames_ingested=2)  # Need 2+ for growth stats
    enabled = trk.report_block()

    # Verify enabled arm has samples and the expected structural keys.
    assert enabled["enabled"] is True
    assert enabled["n_snapshots"] == 2
    assert enabled["growth"]["available"] is True

    # Derive the required structural keys from the enabled arm's keys,
    # excluding optional/conditional keys that are not guaranteed to be present
    # in every enabled report block (CUDA-only or conditional on state).
    # This prevents the hardcoded key set from silently diverging from the
    # actual report_block() output.
    enabled_keys = set(enabled.keys())

    # Keys that are optional and should not be required on disabled arm:
    optional_keys = {
        # CUDA-only key: only present when device is cuda and we can call
        # torch.cuda.max_memory_allocated(). Disabled arm has no device, so
        # torch_peak_allocated_bytes is not present there.
        "torch_peak_allocated_bytes",
        # Conditional on n_snapshots: only when enabled arm has n_snapshots == 0
        # does the "reason" key appear (see report_block() line ~636-641).
        # The disabled arm ALWAYS has "reason" (it's truthful when disabled),
        # so "reason" is omitted from the optional set and is therefore required.
    }

    required_structural_keys = enabled_keys - optional_keys

    # Check disabled arm has all required structural keys.
    disabled_keys = set(disabled.keys())
    missing_keys = required_structural_keys - disabled_keys
    assert (
        not missing_keys
    ), f"disabled_report_block missing keys: {missing_keys}"

    # Check that disabled arm is a superset of required (the disabled arm may
    # have keys the enabled arm does not, e.g., "reason" when enabled has
    # n_snapshots > 0 and thus no "reason").
    extra_in_disabled = disabled_keys - required_structural_keys
    # Extra keys are permitted and expected (e.g., "reason" on disabled);
    # just document what was found.
    assert extra_in_disabled <= {"reason"}, (
        f"disabled_report_block has unexpected extra keys: {extra_in_disabled}"
    )

    # Every required structural key should resolve on disabled without KeyError.
    for key in required_structural_keys:
        _ = disabled[key]

    # growth is tri-state with "available" key.
    assert "available" in disabled["growth"]
    assert disabled["growth"]["available"] is False
    assert isinstance(disabled["series"], list)
    assert isinstance(disabled["bucket_order"], list)

    # Measurement keys must be None (not 0), because zero is a valid
    # measurement and would silently misrepresent an unmeasured value as a
    # real zero peak. The anti-pattern we prevent: a downstream consumer
    # reading a zero peak as real data without checking enabled/reason first.
    assert disabled["peak_allocated_bytes"] is None, (
        "peak_allocated_bytes must be None, not 0, to distinguish "
        "unmeasured from a real zero peak"
    )
    assert disabled["peak_reserved_bytes"] is None, (
        "peak_reserved_bytes must be None, not 0, to distinguish "
        "unmeasured from a real zero peak"
    )
    assert disabled["history_stride"] is None, (
        "history_stride must be None, not 0, to distinguish "
        "unmeasured from a real zero stride"
    )
    assert disabled["history_max"] is None, (
        "history_max must be None, not 0, to distinguish "
        "unmeasured from a real zero max"
    )


def test_report_block_is_json_serializable():
    import json

    scene = _FakeScene(window=1)
    scene.train_cameras = [_FakeCam()]
    trk = _tracker(lambda: 10**6)
    for i in range(5):
        trk.snapshot(iteration=i, frames_ingested=i + 1, streaming_scene=scene)
    json.dumps(trk.report_block())


def test_gtt_reader_is_explicit_shared_device_memory_not_vram():
    """GTT is a device-wide DRM reading, never a per-process VRAM claim."""
    trk = GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=lambda: 10,
        reserved_bytes_fn=lambda: 20,
        peak_bytes_fn=lambda: 10,
        gtt_bytes_fn=lambda: (123, 456),
    )

    sample = trk.snapshot()
    assert sample["gtt_used_bytes"] == 123
    assert sample["gtt_total_bytes"] == 456
    assert sample["gtt_available"] is True
    assert "shared_system_memory" in sample["gtt_memory_semantics"]
    assert "per_process" in sample["gtt_memory_semantics"]
    assert "vram" not in sample["gtt_memory_semantics"].lower()

    block = trk.report_block()
    assert block["peak_gtt_used_bytes"] == 123
    assert block["gtt_memory_semantics"] == sample["gtt_memory_semantics"]
    assert block["series"][-1]["gtt_total_bytes"] == 456


def test_unavailable_gtt_is_explicit_null_not_zero():
    trk = GPUMemoryResidencyTracker(
        device="cpu",
        total_bytes_fn=lambda: 10,
        reserved_bytes_fn=lambda: 20,
        peak_bytes_fn=lambda: 10,
        gtt_bytes_fn=lambda: None,
    )

    sample = trk.snapshot()
    assert sample["gtt_available"] is False
    assert sample["gtt_used_bytes"] is None
    assert sample["gtt_total_bytes"] is None
    assert trk.report_block()["peak_gtt_used_bytes"] is None


def test_gtt_fields_serialize_to_report_artifact_with_shared_memory_semantics(
    tmp_path, monkeypatch, capsys
):
    """The artifact and human summary expose global shared GTT separately."""
    import json
    import sys
    from unittest.mock import MagicMock

    from eligsir.utils.comparison_report import write_post_training_report
    from eligsir.utils.gpu_memory_telemetry import GTT_MEMORY_SEMANTICS

    class _Gaussians:
        @property
        def get_xyz(self):
            return torch.zeros(0, 3)

    tracker = GPUMemoryResidencyTracker(
        device="cpu",
        gtt_bytes_fn=lambda: (128 * 1024 * 1024, 512 * 1024 * 1024),
    )
    tracker.snapshot(iteration=7, gaussians=_Gaussians())
    previous = sys.modules.get("lpipsPyTorch")
    monkeypatch.setitem(sys.modules, "lpipsPyTorch", MagicMock())
    try:
        summary = write_post_training_report(
            model_path=str(tmp_path),
            iteration=7,
            gaussians=_Gaussians(),
            train_cams=[],
            test_cams=[],
            render_fn=lambda *_args: {"render": torch.zeros(3, 2, 2)},
            pipe=object(),
            background=None,
            gpu_memory=tracker.report_block(),
        )
    finally:
        if previous is None:
            sys.modules.pop("lpipsPyTorch", None)
        else:
            sys.modules["lpipsPyTorch"] = previous

    stdout = capsys.readouterr().out
    assert "torch_allocated=" in stdout
    assert "gtt_global=128.0MiB/512.0MiB" in stdout
    assert "gtt_scope=device_global_shared_not_per_process" in stdout

    on_disk = json.loads(
        (tmp_path / "comparison" / "iter_7" / "report.json").read_text()
    )
    for report in (summary, on_disk):
        gtt = report["gpu_memory"]
        assert gtt["gtt_memory_semantics"] == GTT_MEMORY_SEMANTICS
        assert "vram" not in gtt["gtt_memory_semantics"].lower()
        assert gtt["final"]["gtt_used_bytes"] == 128 * 1024 * 1024
        assert gtt["final"]["gtt_total_bytes"] == 512 * 1024 * 1024
        assert gtt["final"]["gtt_available"] is True


def test_no_cuda_synchronize_or_empty_cache_in_module_source():
    """#424/#480 guard: the walk must never force a host/device sync, and must
    never perturb the allocator. Checked against the module source so the
    invariant cannot silently regress on a CPU-only CI host."""
    import inspect

    import eligsir.utils.gpu_memory_telemetry as mod

    src = inspect.getsource(mod)
    for forbidden in ("cuda.synchronize", "empty_cache", ".item()", "torch.cuda.reset"):
        # Allowed inside comments/docstrings (the module documents them), so
        # strip comment lines and prose lines before checking.
        code_lines = [
            ln
            for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith(("#", '"', "*", "`"))
        ]
        offenders = [ln for ln in code_lines if forbidden in ln and "never" not in ln]
        assert not offenders, f"{forbidden} appears in executable-looking lines: {offenders}"


def test_flags_and_report_kwarg_are_wired():
    """Wiring check: the CLI knobs exist with the documented defaults and the
    report writer really accepts the block."""
    import inspect
    from argparse import ArgumentParser

    from arguments import ModelParams  # noqa: F401  (import parity with parser use)
    from eligsir.utils.comparison_report import write_post_training_report

    assert "gpu_memory" in inspect.signature(write_post_training_report).parameters

    from arguments import StreamingParams

    parser = ArgumentParser()
    sp = StreamingParams(parser)
    args = parser.parse_args([])
    filled = sp.extract(args)
    assert filled.streaming_memory_telemetry is True
    assert filled.streaming_memory_telemetry_interval == 0
    assert filled.streaming_memory_telemetry_max_samples == 512

    off = sp.extract(parser.parse_args(["--no-streaming_memory_telemetry"]))
    assert off.streaming_memory_telemetry is False


def test_snapshot_cost_is_small_at_realistic_scale(capsys):
    """Measured overhead, not asserted-by-assumption.

    Builds ~8000 cameras (ScanNet++ scale, the #290 run that motivated #399)
    plus a 400-frame resolution cache, then times repeated snapshots. Prints
    the per-snapshot cost and the amortized per-iteration cost at
    scalar_log_interval=100 so the PR can quote a real number.
    """
    window = 120
    scene = _FakeScene(window=window)
    scene.train_cameras = [
        _FakeCam(h=4, w=4, resident=(i >= 8000 - window), frame_index=i) for i in range(8000)
    ]
    scene._test_cameras = [_FakeCam(h=4, w=4, frame_index=10**6 + i) for i in range(993)]
    db = _FakeDB()
    db._frames = {i: _FakeRecord(4, 4) for i in range(400)}
    scene.observation_db = db
    g = _FakeGaussians(n=4096)

    trk = _tracker(lambda: 8 * 10**9, max_history=64)
    reps = 20
    t0 = time.perf_counter()
    for i in range(reps):
        trk.snapshot(iteration=i, frames_ingested=8000, gaussians=g, streaming_scene=scene)
    per_snapshot_ms = (time.perf_counter() - t0) / reps * 1e3

    with capsys.disabled():
        print(
            f"\n[#399 overhead] {per_snapshot_ms:.2f} ms/snapshot at "
            f"8000 train + 993 holdout cameras + 400 cached frames; "
            f"amortized {per_snapshot_ms / 100 * 1e3:.1f} us/iteration at "
            f"scalar_log_interval=100"
        )
    # Generous ceiling: the point is that this is milliseconds, not seconds,
    # at a cadence of once per 100 iterations. A regression into per-tensor
    # tensor ops (or a sync) would blow straight through it.
    assert per_snapshot_ms < 250.0, f"snapshot cost regressed to {per_snapshot_ms:.1f} ms"


def test_residency_owner_contract_names_every_bucket():
    contract = residency_owner_contract()

    assert contract["version"] >= 1
    assert set(contract["buckets"]) == set(BUCKET_ORDER)
    for name, owner in contract["buckets"].items():
        assert owner["owner"]
        assert owner["source"]
        assert owner["measurement"]
