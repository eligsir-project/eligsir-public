import inspect
from types import SimpleNamespace

from eligsir.contracts.work_clocks import RuntimeWorkClockOwner
from eligsir.contracts.runtime import SourceIdentity
from eligsir.runtime import production_application as production
from eligsir.runtime.production_application import CanonicalMappingRuntime


class _FakeGaussians:
    def __init__(self):
        self.sh_increments = 0

    def oneupSHdegree(self):
        self.sh_increments += 1


class _Candidate:
    pts = SimpleNamespace(device="cpu")

    def __len__(self):
        return 1


def test_sh_degree_consumes_every_crossed_pixel_exposure_threshold():
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._runtime_policy = SimpleNamespace(
        sh_degree_schedule_rgb_pixels=(100, 200, 300)
    )
    runtime.gaussians = _FakeGaussians()

    runtime._maybe_raise_sh_degree(supervised_rgb_pixels=250)
    assert runtime.gaussians.sh_increments == 2

    runtime._maybe_raise_sh_degree(supervised_rgb_pixels=250)
    assert runtime.gaussians.sh_increments == 2

    runtime._maybe_raise_sh_degree(supervised_rgb_pixels=1_000)
    assert runtime.gaussians.sh_increments == 3


def test_canonical_insertion_birth_clock_is_completed_optimizer_updates(monkeypatch):
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._work_clock_owner = RuntimeWorkClockOwner()
    runtime._work_clock_owner.record_optimization(
        optimizer_updates=2,
        supervised_views=2,
        supervised_pixels=200,
        mapping_active_ns=0,
    )
    runtime.iteration = 9_999
    runtime.scene = object()
    runtime.gaussians = object()
    runtime.pipe = object()
    runtime._background = object()
    runtime.frames_seen = 3
    runtime._depth_insertion_policy = SimpleNamespace(initial_opacity=0.1)
    runtime._transfer_owner = None
    source = SourceIdentity("test", 0, 1, 1)
    candidate = _Candidate()
    captured = {}

    monkeypatch.setattr(production, "render", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        production,
        "insert_gaussians_from_frame_device_native",
        lambda *_args, **_kwargs: (candidate, object()),
    )

    monkeypatch.setattr(
        production,
        "normalize_pending_candidates",
        lambda candidates, **kwargs: candidates[0],
    )

    def _proposal(_plan, _candidates, *, key, iteration):
        captured["birth_clock"] = iteration
        return object()

    monkeypatch.setattr(production, "insertion_proposal_from_plan", _proposal)
    plan = SimpleNamespace(
        insertion=SimpleNamespace(enabled=True),
        supervision_work=(SimpleNamespace(source=source, modality_eligibility=object()),),
    )

    assert runtime._prepare_depth_insertion_proposal(
        object(), object(), source,
        insertion_plan=plan.insertion,
        modality_eligibility=object(),
        prev_frame=None,
        prev_depth_meters=None,
    ) is not None
    assert captured["birth_clock"] == 2


def test_canonical_insertion_prep_retains_measured_zero_filters_for_decision_trace(
    monkeypatch,
):
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._population_selection_trace_enabled = True
    runtime.scene = object()
    runtime.gaussians = object()
    runtime.pipe = object()
    runtime._background = object()
    runtime.frames_seen = 3
    runtime._depth_insertion_policy = SimpleNamespace(
        initial_opacity=0.1, debug_statistics=True
    )
    runtime._transfer_owner = object()
    source = SourceIdentity("test", 0, 1, 1)

    monkeypatch.setattr(production, "render", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        production,
        "insert_gaussians_from_frame_device_native",
        lambda *_args, **_kwargs: (
            None,
            {
                "producer": "device_resident_v1",
                "raw_candidates": 0,
                "after_alpha_depth_mask": 0,
                "after_grazing_filter": 0,
                "insertion_render_reused": False,
                "source_lease_hit": False,
                "source_h2d_ops": 1,
                "source_d2h_ops": 0,
            },
        ),
    )

    assert runtime._prepare_depth_insertion_proposal(
        object(), object(), source,
        insertion_plan=object(),
        modality_eligibility=object(),
        prev_frame=None,
        prev_depth_meters=None,
    ) is None

    assert runtime._decision_trace_insertion_filter == {
        "source": source.fingerprint(),
        "stats": {
            "producer": "device_resident_v1",
            "raw_candidates": 0,
            "after_alpha_depth_mask": 0,
            "after_grazing_filter": 0,
        },
    }


def test_canonical_tick_clears_prior_insertion_filter_before_no_work_result():
    runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
    runtime._closed = False
    runtime._population_selection_trace_enabled = True
    runtime._decision_trace_insertion_filter = {"source": "stale", "stats": {"raw_candidates": 8}}
    runtime._control_tick_epoch = 0
    runtime._work_clock_owner = SimpleNamespace(record_control_tick=lambda: None)
    runtime._latest_tick_work = runtime._latest_report_work = None
    runtime.initialized = False
    runtime._bootstrap = ()
    runtime._observation_db = SimpleNamespace(
        pending_cold_payload_spill_count=0,
        take_cold_payload_spill_failures=lambda: (),
    )
    runtime.bootstrap_frames = 1
    runtime._tick_result = lambda *_args, **_kwargs: runtime._decision_trace_insertion_filter

    assert runtime.tick() is None


def test_sh_degree_threshold_is_invariant_to_transaction_batching():
    def _crossings(chunks):
        runtime = CanonicalMappingRuntime.__new__(CanonicalMappingRuntime)
        runtime._runtime_policy = SimpleNamespace(
            sh_degree_schedule_rgb_pixels=(400, 800)
        )
        runtime.gaussians = _FakeGaussians()
        owner = RuntimeWorkClockOwner()
        crossings = []
        for views in chunks:
            owner.record_optimization(
                optimizer_updates=1,
                supervised_views=views,
                supervised_pixels=views * 100,
                mapping_active_ns=0,
            )
            pixels = owner.snapshot().supervised_rgb_pixels
            before = runtime.gaussians.sh_increments
            runtime._maybe_raise_sh_degree(supervised_rgb_pixels=pixels)
            crossings.extend(
                [pixels] * (runtime.gaussians.sh_increments - before)
            )
        return crossings

    assert _crossings([1] * 8) == _crossings([4, 4]) == [400, 800]


def test_iteration_commit_never_precedes_transaction_dispatch():
    """Regression guard for the shutdown clock-mismatch bug (issue #1259).

    A real production crash traced a long fr3 diagnostic run's CLEAN training
    (34,379 iterations, converged PSNR) hitting a CUDA OOM during the backward
    pass of `execute_fixed_topology_transaction` inside `tick()`'s
    post-mapping-refinement loop. That OOM was then MASKED by a second,
    unrelated-looking crash at shutdown:
    ``RuntimeError: point-cloud snapshot optimizer clock disagrees with
    runtime iteration`` from `flush()`.

    Root cause: `tick()` used to increment `self.iteration` *before*
    dispatching the GPU transaction that corresponds to it, and only recorded
    the matching `_work_clock_owner.optimizer_updates` advance *after* that
    transaction returned. When the transaction raised (OOM or otherwise)
    partway through, `self.iteration` had already been bumped but the work
    clock had not -- permanently desynchronizing the two counters for the
    rest of the runtime's lifetime, so every later `flush()`/`close()` call
    raised a confusing, unrelated error that hid the real failure.

    The fix makes the commit atomic with transaction success: compute the
    attempted iteration id, pass THAT into the dispatched transaction, and
    only assign it to `self.iteration` after
    `execute_fixed_topology_transaction(...)` has returned without raising.
    This test statically enforces that ordering can't silently regress
    (a real GPU OOM is not practical to reproduce in a CPU-only unit test).
    """

    source = inspect.getsource(CanonicalMappingRuntime.tick)
    lines = source.splitlines()

    dispatch_indices = [
        index
        for index, line in enumerate(lines)
        if "execute_fixed_topology_transaction(" in line
    ]
    commit_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip() == "self.iteration = attempted_iteration"
    ]

    # One true-camera-batch dispatch site, one B=1 sequential dispatch site --
    # each needs its own commit immediately after it.
    assert len(dispatch_indices) == 2, (
        "expected exactly two execute_fixed_topology_transaction(...) call "
        "sites in CanonicalMappingRuntime.tick"
    )
    assert len(commit_indices) == 2, (
        "expected exactly two `self.iteration = attempted_iteration` commits "
        "in CanonicalMappingRuntime.tick"
    )
    for dispatch_index, commit_index in zip(dispatch_indices, commit_indices):
        assert commit_index > dispatch_index, (
            "self.iteration must be committed only after "
            "execute_fixed_topology_transaction(...) returns -- committing "
            "it any earlier reintroduces the shutdown clock-mismatch bug "
            "(issue #1259) whenever that transaction raises"
        )

    # And there must be no unconditional, pre-dispatch `self.iteration`
    # mutation left over from the old eager-increment code.
    unsafe_writes = [
        line.strip()
        for line in lines
        if line.strip() in ("self.iteration += 1", "self.iteration = self.iteration + 1")
    ]
    assert not unsafe_writes, (
        "found an eager `self.iteration` mutation independent of transaction "
        "success in CanonicalMappingRuntime.tick"
    )
