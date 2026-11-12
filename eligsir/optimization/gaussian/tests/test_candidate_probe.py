"""CPU/fake-render contract tests for the candidate probe (issue #803).

This codebase has NO CPU rasterization path (see ``test_step_parity.py``'s
docstring), so everything here substitutes a deterministic fake renderer for
``candidate_probe.invoke_render``. That is a real limitation and it is the
reason the GPU companion file exists: what is proven here is the *contract*
(ordering, identity association, versioning, accounting, skip semantics,
determinism, and -- most importantly -- non-mutation of every piece of state
these fakes can model), not the numerical agreement of a probe render with a
supervision render. That agreement is `test_candidate_probe_gpu.py`'s job.

The non-mutation tests deliberately use a fake that CAN mutate: several of
them run a renderer that deliberately steps the optimizer, dirties ``.grad``,
or bumps a version counter, and assert the probe *raises*. A guard that has
only ever been exercised against well-behaved inputs is not a guard.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from eligsir.contracts.control import ResolutionTier  # noqa: E402
from eligsir.mapping.evidence.canonical import (  # noqa: E402
    SCHEMA_VERSION,
    EvidenceAvailability,
)
from eligsir.optimization.gaussian import candidate_probe  # noqa: E402
from eligsir.optimization.gaussian.candidate_probe import (  # noqa: E402
    CANDIDATE_PROBE_EVIDENCE_SCHEMA_VERSION,
    CANDIDATE_PROBE_OBJECTIVE_VERSION,
    CandidateProbeAccounting,
    CandidateProbeRequest,
    CandidateProbeView,
    ProbeEvidenceChannel,
    ProbeStateMutationError,
    ProbeUnsafeContextError,
    build_pairwise_evidence,
    capture_state_fingerprint,
    diff_state_fingerprints,
    probe_resolution_binding,
    residual_sketch,
)
from eligsir.optimization.gaussian.contracts import (  # noqa: E402
    OptimizationRuntimeContext,
    RenderOutputs,
)
from eligsir.optimization.gaussian.engine import (  # noqa: E402
    GaussianOptimizationEngine,
)

H, W = 16, 24
N = 12


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCamera:
    """Minimal camera carrying everything the probe reads."""

    def __init__(self, seed: int, *, height: int = H, width: int = W):
        generator = torch.Generator().manual_seed(seed)
        self.original_image = torch.rand(3, height, width, generator=generator)
        self.image_height = self.height = height
        self.image_width = self.width = width
        self.fx = self.fy = 100.0
        self.cx, self.cy = width / 2.0, height / 2.0
        self.camera_center = torch.tensor([float(seed), 0.0, 0.0])
        world_view = torch.eye(4)
        self.world_view_transform = world_view
        self.gsplat = None
        self.uid = seed


class FakeGaussians:
    """A fixed-topology parameter bag in the gsplat ``ParameterDict`` layout."""

    def __init__(self, n: int = N):
        generator = torch.Generator().manual_seed(7)
        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(torch.rand(n, 3, generator=generator)),
                "opacities": torch.nn.Parameter(torch.rand(n, generator=generator)),
                "scales": torch.nn.Parameter(torch.rand(n, 3, generator=generator)),
                "quats": torch.nn.Parameter(torch.rand(n, 4, generator=generator)),
                "sh0": torch.nn.Parameter(torch.rand(n, 1, 3, generator=generator)),
            }
        )
        self.optimizer = torch.optim.Adam(self.params.parameters(), lr=1e-3)

    @property
    def get_xyz(self):
        return self.params["means"]


class FakeSession:
    def __init__(self, model):
        self.model = model
        self.parameter_version = 3
        self.optimizer = model.optimizer
        self.lr_calls = 0

    def update_learning_rate(self, iteration):
        self.lr_calls += 1
        return 1e-4


class FakeBackend:
    def __init__(self, model):
        self.map_content_version = 11
        self.topology_version = 5
        self.coordinate_frame_version = 2
        self._session = FakeSession(model)

    def begin_optimization(self):
        return self._session


def _fake_render_factory(gaussians, *, visible_from=0, visible_to=N, bias=0.0):
    """Deterministic renderer: constant-ish image plus a per-camera bias."""

    def _render(model, camera, pipe, background, *, update_sh_rest, render_depth):
        height, width = camera.original_image.shape[-2:]
        base = camera.original_image.clone()
        # A deterministic, camera-dependent perturbation so different views
        # produce genuinely different residuals.
        base[:, : height // 2, :] = (base[:, : height // 2, :] + 0.25 + bias).clamp(
            0.0, 1.0
        )
        visible = torch.zeros(N, dtype=torch.bool)
        visible[visible_from:visible_to] = True
        payload = {
            "render": base,
            "alpha": torch.full((1, height, width), 0.75),
            "visibility_filter": visible,
        }
        if render_depth:
            payload["rendered_depth"] = torch.full((1, height, width), 2.0)
        return RenderOutputs(payload=payload)

    return _render


@pytest.fixture()
def rig(monkeypatch):
    model = FakeGaussians()
    backend = FakeBackend(model)
    engine = GaussianOptimizationEngine(backend)
    runtime = OptimizationRuntimeContext(
        session=backend.begin_optimization(), gaussians=model, pipe=object()
    )
    monkeypatch.setattr(
        candidate_probe, "invoke_render", _fake_render_factory(model)
    )
    return engine, backend, model, runtime


def _request(n_views=2, **kwargs):
    views = tuple(
        CandidateProbeView(
            camera=FakeCamera(seed=i + 1),
            background=None,
            source_id=f"frame-{i:04d}",
            source_ordinal=100 + i,
            timestamp_ns=1_000 + i,
            render_depth=False,
        )
        for i in range(n_views)
    )
    return CandidateProbeRequest(views=views, probe_batch_id="batch-a", **kwargs)


# ---------------------------------------------------------------------------
# B = 1 / B > 1 ordering and per-view identity
# ---------------------------------------------------------------------------


def test_single_view_probe_returns_one_result(rig):
    engine, _, _, runtime = rig
    batch = engine.probe_candidates(_request(n_views=1), runtime)
    assert len(batch.per_view) == 1
    assert batch.per_view[0].view_index == 0
    assert batch.per_view[0].source_id == "frame-0000"


def test_batch_results_are_index_aligned_and_carry_exact_source_identity(rig):
    engine, _, _, runtime = rig
    request = _request(n_views=4)
    batch = engine.probe_candidates(request, runtime)
    assert len(batch.per_view) == 4
    for index, (view, result) in enumerate(zip(request.views, batch.per_view)):
        assert result.view_index == index
        assert result.source_id == view.source_id
        assert result.source_ordinal == view.source_ordinal
        assert result.timestamp_ns == view.timestamp_ns
        # The identity a scheduler will actually key off must agree too.
        assert result.evidence.identity.source_id == view.source_id
        assert result.evidence.identity.source_ordinal == view.source_ordinal
        assert result.evidence.identity.timestamp_ns == view.timestamp_ns
    assert batch.result_for("frame-0002").view_index == 2


def test_per_view_evidence_is_unreduced_not_batch_averaged(rig):
    """B>1 must not collapse to one number: each view keeps its own residual."""

    engine, _, _, runtime = rig
    batch = engine.probe_candidates(_request(n_views=3), runtime)
    values = [
        result.evidence.probe.rgb_l1_mean.value for result in batch.per_view
    ]
    assert all(value is not None for value in values)
    assert len(set(round(value, 9) for value in values)) > 1, (
        "three different cameras produced identical residuals -- the probe is "
        "reporting a reduced batch statistic, not per-view evidence"
    )


def test_batch_of_one_matches_that_view_inside_a_larger_batch(rig):
    """A view's evidence must not depend on who it was probed alongside."""

    engine, _, _, runtime = rig
    solo = _request(n_views=1)
    grouped = _request(n_views=3)
    solo_value = engine.probe_candidates(solo, runtime).per_view[0]
    grouped_value = engine.probe_candidates(grouped, runtime).per_view[0]
    assert solo_value.source_id == grouped_value.source_id
    assert math.isclose(
        solo_value.evidence.probe.rgb_l1_mean.value,
        grouped_value.evidence.probe.rgb_l1_mean.value,
        rel_tol=0.0,
        abs_tol=0.0,
    )


# ---------------------------------------------------------------------------
# Objective components / normalization
# ---------------------------------------------------------------------------


def test_rgb_objective_components_are_normalized_and_bounded(rig):
    engine, _, _, runtime = rig
    batch = engine.probe_candidates(_request(n_views=2, lambda_dssim=0.2), runtime)
    for result in batch.per_view:
        probe = result.evidence.probe
        assert 0.0 <= probe.rgb_l1_mean.value <= 1.0
        assert 0.0 <= probe.rgb_mse_mean.value <= 1.0
        assert 0.0 <= probe.well_explained_pixel_fraction.value <= 1.0
        assert 0.0 <= probe.residual_entropy.value <= 1.0
        assert 0.0 <= probe.residual_top_decile_mass_fraction.value <= 1.0
        assert 0.0 <= probe.alpha_mean.value <= 1.0
        assert 0.0 <= probe.visible_gaussian_fraction.value <= 1.0
        # The photometric objective must be the documented combination.
        expected = 0.8 * probe.rgb_l1_mean.value + 0.2 * (1.0 - probe.rgb_ssim.value)
        assert math.isclose(probe.rgb_objective.value, expected, rel_tol=1e-6)
        assert probe.probe_pixel_count.value == float(H * W)


def test_residual_sketch_is_mean_preserving_per_cell():
    """Average pooling: each cell is its footprint's MEAN residual.

    Mean- rather than sum-preserving on purpose: two probes of the same view
    at different resolution tiers must produce sketches on the same scale, or
    a pairwise residual-similarity comparison across tiers is meaningless.
    """

    uniform = torch.ones(4, 4)
    spike = torch.zeros(4, 4)
    spike[0, 0] = 1.0
    assert torch.equal(residual_sketch(uniform, 2), torch.ones(2, 2))
    assert residual_sketch(spike, 2)[0, 0].item() == pytest.approx(0.25)
    assert float(residual_sketch(spike, 2).sum()) == pytest.approx(0.25)
    # And it is genuinely resolution-independent for a constant field.
    assert torch.equal(residual_sketch(torch.ones(8, 8), 2), torch.ones(2, 2))


def test_unrequested_channels_are_reason_coded_not_silently_zero(rig):
    engine, _, _, runtime = rig
    request = _request(
        n_views=1, evidence_channels=frozenset({ProbeEvidenceChannel.RGB})
    )
    probe = engine.probe_candidates(request, runtime).per_view[0].evidence.probe
    assert probe.rgb_l1_mean.value is not None
    assert probe.alpha_mean.value is None
    assert probe.alpha_mean.availability is EvidenceAvailability.NOT_REQUESTED
    assert probe.gradient_norm.availability is EvidenceAvailability.NOT_REQUESTED


def test_depth_absence_is_reason_coded(rig):
    engine, _, _, runtime = rig
    request = _request(n_views=1)
    request = CandidateProbeRequest(
        views=(
            CandidateProbeView(
                camera=FakeCamera(seed=1),
                source_id="f",
                render_depth=True,
            ),
        ),
        probe_batch_id="depthless",
    )
    probe = engine.probe_candidates(request, runtime).per_view[0].evidence.probe
    assert probe.depth_l1_mean_m.value is None
    assert (
        probe.depth_l1_mean_m.availability
        is EvidenceAvailability.UNEVALUABLE_NO_MAP_SUPPORT
    )
    assert probe.depth_l1_mean_m.reason


# ---------------------------------------------------------------------------
# Version propagation
# ---------------------------------------------------------------------------


def test_probe_never_populates_an_empty_sensor_depth_cache(rig, monkeypatch):
    """A probe must not decode depth into the camera cache at probe resolution.

    ``_get_sensor_depth``'s first-access branch WRITES the cache at whatever
    size it is asked for. Inside a reduced-resolution probe that would leave a
    quarter-resolution depth cache behind that survives the tier restore (the
    restore replays a native state whose ``depth`` was captured as ``None``),
    so a later native supervision render reads the wrong grid. That is a silent
    training-state corruption in an attribute no Gaussian/optimizer hash covers.
    """

    engine, _, _, runtime = rig
    camera = FakeCamera(seed=9)
    camera._sensor_depth_cache = None
    camera._streaming_depth_path = "/nonexistent/depth.png"
    camera._streaming_depth_scale = 1000.0

    def _depth_render(model, cam, pipe, background, *, update_sh_rest, render_depth):
        height, width = cam.original_image.shape[-2:]
        return RenderOutputs(
            payload={
                "render": cam.original_image.clone(),
                "alpha": torch.full((1, height, width), 0.6),
                "visibility_filter": torch.ones(N, dtype=torch.bool),
                "rendered_depth": torch.full((1, height, width), 1.5),
            }
        )

    monkeypatch.setattr(candidate_probe, "invoke_render", _depth_render)
    request = CandidateProbeRequest(
        views=(CandidateProbeView(camera=camera, source_id="d", render_depth=True),),
        probe_batch_id="depthcache",
        resolution_tier=ResolutionTier.QUARTER,
    )
    probe = engine.probe_candidates(request, runtime).per_view[0].evidence.probe
    assert probe.depth_l1_mean_m.value is None
    assert "must not populate" in probe.depth_l1_mean_m.reason
    assert camera._sensor_depth_cache is None, (
        "the probe populated the sensor-depth cache; at a reduced tier that "
        "leaves a wrong-resolution cache behind for the next real render"
    )


def test_full_channel_probe_constructs_every_evidence_field(rig, monkeypatch):
    """Guards against a renamed evidence field failing only at runtime.

    ``GaussianProbeEvidence(**blocks)`` raises ``TypeError`` for an unknown
    key, so this test is what turns a field rename into a CI failure instead of
    a first-live-probe failure.
    """

    engine, _, _, runtime = rig
    monkeypatch.setattr(candidate_probe, "invoke_render", DifferentiableFakeRender())
    batch = engine.probe_candidates(
        _request(
            n_views=1,
            lambda_dssim=0.0,
            evidence_channels=frozenset(ProbeEvidenceChannel),
        ),
        runtime,
    )
    probe = batch.per_view[0].evidence.probe
    populated = [
        name
        for name in type(probe).__dataclass_fields__
        if getattr(probe, name).value is not None
    ]
    # Everything except the depth block (this fixture has no sensor depth).
    assert len(populated) >= 18, sorted(populated)
    assert not batch.per_view[0].skipped


def test_pose_angle_uses_the_documented_forward_axis_convention(rig, monkeypatch):
    """A camera rotated 90 degrees about Y must report a 90 degree separation.

    An identity-pose fixture cannot test this: every column choice gives 0.
    ``world_view_transform`` here is the TRANSPOSED world-to-view matrix, so
    its top-left 3x3 is the camera-to-world rotation and its third column is
    the camera's world-space forward axis.
    """

    engine, _, _, runtime = rig
    rotated = FakeCamera(seed=2)
    turned = torch.eye(4)
    turned[:3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    rotated.world_view_transform = turned
    request = CandidateProbeRequest(
        views=(
            CandidateProbeView(camera=FakeCamera(seed=1), source_id="a"),
            CandidateProbeView(camera=rotated, source_id="b"),
        ),
        probe_batch_id="pose",
    )
    pair = engine.probe_candidates(request, runtime).pairwise[0]
    assert pair.pose_angle_deg == pytest.approx(90.0, abs=1e-3)


def test_versions_propagate_from_the_live_backend_to_every_result(rig):
    engine, backend, _, runtime = rig
    backend.map_content_version = 41
    backend.topology_version = 9
    backend.coordinate_frame_version = 3
    batch = engine.probe_candidates(
        _request(n_views=2, optimization_step=1234), runtime
    )
    assert (batch.map_version, batch.topology_version) == (41, 9)
    assert batch.coordinate_frame_version == 3
    assert batch.optimization_step == 1234
    assert batch.objective_version == CANDIDATE_PROBE_OBJECTIVE_VERSION
    assert batch.evidence_schema_version == CANDIDATE_PROBE_EVIDENCE_SCHEMA_VERSION
    assert batch.probe_batch_id == "batch-a"
    for result in batch.per_view:
        identity = result.evidence.identity
        assert identity.map_version == 41
        assert identity.topology_version == 9
        assert identity.coordinate_frame_version == 3
        assert identity.optimization_step == 1234
        assert identity.schema_version == SCHEMA_VERSION
        assert identity.resolution_bucket == ResolutionTier.NATIVE.value
        # Evidence is stale the moment the map moves -- the canonical envelope
        # already knows how to say so, and the probe must have filled it in.
        assert not result.evidence.stale_against(
            map_version=41, topology_version=9, coordinate_frame_version=3
        )
        assert result.evidence.stale_against(map_version=42, topology_version=9)


def test_a_topology_change_between_batches_is_recorded_not_hidden(rig):
    engine, backend, _, runtime = rig
    first = engine.probe_candidates(_request(n_views=1), runtime)
    backend.topology_version += 1
    backend.map_content_version += 1
    second = engine.probe_candidates(_request(n_views=1), runtime)
    assert second.topology_version == first.topology_version + 1
    assert second.map_version == first.map_version + 1


# ---------------------------------------------------------------------------
# State immutability -- the crux of #803
# ---------------------------------------------------------------------------


def _full_state_snapshot(model, session):
    """A VALUE-level snapshot, complementing the probe's own cheap fingerprint."""

    return {
        "params": {
            name: tensor.detach().clone()
            for name, tensor in model.params.items()
        },
        "grads": {
            name: (None if tensor.grad is None else tensor.grad.detach().clone())
            for name, tensor in model.params.items()
        },
        "opt_state": {
            id(key): {
                k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
                for k, v in value.items()
            }
            for key, value in model.optimizer.state.items()
        },
        "lrs": [group["lr"] for group in model.optimizer.param_groups],
        "lr_calls": session.lr_calls,
        "parameter_version": session.parameter_version,
    }


def test_probe_changes_no_parameter_optimizer_or_scheduler_state(rig):
    engine, backend, model, runtime = rig
    session = backend.begin_optimization()
    # Give the optimizer real state and real pre-existing gradients first, so
    # this test covers the "probing between steps" case rather than only the
    # trivially-empty one.
    loss = sum(tensor.square().sum() for tensor in model.params.values())
    loss.backward()
    model.optimizer.step()

    before = _full_state_snapshot(model, session)
    engine.probe_candidates(_request(n_views=3), runtime)
    after = _full_state_snapshot(model, session)

    for name in before["params"]:
        assert torch.equal(before["params"][name], after["params"][name])
        left, right = before["grads"][name], after["grads"][name]
        assert (left is None) == (right is None)
        if left is not None:
            assert torch.equal(left, right), f"probe changed .grad for {name}"
    assert before["lrs"] == after["lrs"]
    assert before["lr_calls"] == after["lr_calls"] == 0
    assert before["parameter_version"] == after["parameter_version"]
    assert set(before["opt_state"]) == set(after["opt_state"])
    for key, entry in before["opt_state"].items():
        for field_name, value in entry.items():
            other = after["opt_state"][key][field_name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other)
            else:
                assert value == other


def test_probe_leaves_grad_absent_when_it_was_absent(rig):
    engine, _, model, runtime = rig
    assert all(tensor.grad is None for tensor in model.params.values())
    engine.probe_candidates(_request(n_views=2), runtime)
    assert all(tensor.grad is None for tensor in model.params.values()), (
        "the probe accumulated gradients into .grad -- forward-only mode must "
        "run under no_grad and gradient mode must use torch.autograd.grad"
    )


def test_probe_does_not_consume_the_rng_stream(rig):
    engine, _, _, runtime = rig
    torch.manual_seed(1234)
    expected = torch.rand(8)
    torch.manual_seed(1234)
    engine.probe_candidates(_request(n_views=3), runtime)
    assert torch.equal(torch.rand(8), expected), (
        "the probe advanced the global RNG stream; a run WITH probing would "
        "diverge from an otherwise identical run without it"
    )


def test_probe_raises_when_the_render_path_mutates_state(rig, monkeypatch):
    engine, _, model, runtime = rig
    honest = candidate_probe.invoke_render

    def _sneaky(*args, **kwargs):
        outputs = honest(*args, **kwargs)
        with torch.no_grad():
            model.params["means"].add_(1.0)  # exactly what must never happen
        return outputs

    monkeypatch.setattr(candidate_probe, "invoke_render", _sneaky)
    with pytest.raises(ProbeStateMutationError, match="param.means"):
        engine.probe_candidates(_request(n_views=1), runtime)


def test_probe_raises_when_the_render_path_dirties_grad(rig, monkeypatch):
    engine, _, model, runtime = rig
    honest = candidate_probe.invoke_render

    def _sneaky(*args, **kwargs):
        outputs = honest(*args, **kwargs)
        model.params["means"].grad = torch.zeros_like(model.params["means"])
        return outputs

    monkeypatch.setattr(candidate_probe, "invoke_render", _sneaky)
    with pytest.raises(ProbeStateMutationError, match="grad.means"):
        engine.probe_candidates(_request(n_views=1), runtime)


def test_probe_raises_when_topology_moves_under_it(rig, monkeypatch):
    engine, backend, _, runtime = rig
    honest = candidate_probe.invoke_render

    def _sneaky(*args, **kwargs):
        backend.topology_version += 1
        return honest(*args, **kwargs)

    monkeypatch.setattr(candidate_probe, "invoke_render", _sneaky)
    with pytest.raises(ProbeStateMutationError, match="topology_version"):
        engine.probe_candidates(_request(n_views=1), runtime)


def test_probe_from_inside_an_optimizer_transaction_fails_loudly(rig):
    engine, _, _, runtime = rig
    with engine._optimizer_transaction():
        with pytest.raises(ProbeUnsafeContextError):
            engine.probe_candidates(_request(n_views=1), runtime)
    # ... and the marker is released again afterwards.
    engine.probe_candidates(_request(n_views=1), runtime)


def test_fingerprint_diff_reports_every_kind_of_change():
    assert diff_state_fingerprints({"a": 1}, {"a": 1}) == []
    assert "a: 1 -> 2" in diff_state_fingerprints({"a": 1}, {"a": 2})
    assert any("disappeared" in p for p in diff_state_fingerprints({"a": 1}, {}))
    assert any("appeared" in p for p in diff_state_fingerprints({}, {"a": 1}))


def test_fingerprint_notices_a_value_preserving_inplace_write():
    """``_version`` catches a write even when the values come back identical."""

    model = FakeGaussians()
    before = capture_state_fingerprint(gaussians=model)
    with torch.no_grad():
        model.params["means"].mul_(1.0)  # numerically a no-op, still a write
    assert diff_state_fingerprints(before, capture_state_fingerprint(gaussians=model))


# ---------------------------------------------------------------------------
# Resolution binding
# ---------------------------------------------------------------------------


def test_probe_restores_an_unbound_cameras_native_resolution_exactly():
    camera = FakeCamera(seed=3)
    original_image = camera.original_image
    original_geometry = (camera.image_width, camera.image_height, camera.fx, camera.cx)
    with probe_resolution_binding(camera, ResolutionTier.QUARTER) as rebound:
        assert rebound
        assert camera.image_width == W // 4
        assert camera.original_image.shape[-1] == W // 4
    assert camera.original_image is original_image
    assert (camera.image_width, camera.image_height, camera.fx, camera.cx) == (
        original_geometry
    )
    # An unbound camera must come back UNBOUND: leaving the marker attributes
    # behind would silently change how a later supervision bind behaves.
    assert not hasattr(camera, "_canonical_resolution_tier")


def test_probe_restores_a_previously_bound_camera_to_its_own_tier():
    from eligsir.mapping.resolution_materialization import (
        materialize_resolution_tier,
    )

    camera = FakeCamera(seed=4)
    materialize_resolution_tier(camera, ResolutionTier.HALF)
    half_width = camera.image_width
    with probe_resolution_binding(camera, ResolutionTier.QUARTER):
        assert camera.image_width == W // 4
    assert camera._canonical_resolution_tier is ResolutionTier.HALF
    assert camera.image_width == half_width


def test_probing_at_the_cameras_existing_tier_touches_nothing():
    camera = FakeCamera(seed=5)
    with probe_resolution_binding(camera, ResolutionTier.NATIVE) as rebound:
        assert rebound is False
    assert not hasattr(camera, "_canonical_native_resolution_state")


def test_probe_at_reduced_resolution_reports_the_reduced_pixel_count(rig):
    engine, _, _, runtime = rig
    request = _request(n_views=2, resolution_tier=ResolutionTier.HALF)
    batch = engine.probe_candidates(request, runtime)
    assert batch.resolution_bucket == "half"
    for result in batch.per_view:
        assert (result.probe_height, result.probe_width) == (H // 2, W // 2)
        assert result.probe_pixel_count == (H // 2) * (W // 2)
        assert result.evidence.identity.resolution_bucket == "half"
    assert batch.accounting.candidate_probe_pixel_count == 2 * (H // 2) * (W // 2)


# ---------------------------------------------------------------------------
# Skipped / invalid candidates
# ---------------------------------------------------------------------------


def test_one_broken_candidate_does_not_destroy_the_cohort(rig, monkeypatch):
    engine, _, _, runtime = rig
    honest = candidate_probe.invoke_render

    def _selective(model, camera, pipe, background, **kwargs):
        if camera.uid == 2:
            raise ValueError("synthetic renderer failure")
        return honest(model, camera, pipe, background, **kwargs)

    monkeypatch.setattr(candidate_probe, "invoke_render", _selective)
    batch = engine.probe_candidates(_request(n_views=3), runtime)

    assert len(batch.per_view) == 3, "a skipped candidate must keep its slot"
    broken = batch.per_view[1]
    assert broken.skipped and "synthetic renderer failure" in broken.skip_reason
    assert broken.source_id == "frame-0001"
    assert broken.evidence.rgb_eligible is False
    assert broken.evidence.hard_reject_reasons == ("candidate_probe_failed",)
    assert not batch.per_view[0].skipped and not batch.per_view[2].skipped
    assert batch.accounting.candidate_probe_skipped_view_count == 1
    assert batch.accounting.candidate_probe_view_count == 2
    assert batch.accounting.candidate_probe_rasterizer_call_count == 2


def test_skipped_candidates_are_excluded_from_pairwise_evidence(rig, monkeypatch):
    engine, _, _, runtime = rig
    honest = candidate_probe.invoke_render

    def _selective(model, camera, pipe, background, **kwargs):
        if camera.uid == 2:
            raise ValueError("boom")
        return honest(model, camera, pipe, background, **kwargs)

    monkeypatch.setattr(candidate_probe, "invoke_render", _selective)
    batch = engine.probe_candidates(_request(n_views=3), runtime)
    assert len(batch.pairwise) == 1
    assert {batch.pairwise[0].view_index_a, batch.pairwise[0].view_index_b} == {0, 2}


def test_a_request_with_no_views_is_refused():
    with pytest.raises(ValueError):
        CandidateProbeRequest(views=())


# ---------------------------------------------------------------------------
# Pairwise (Phase 2) evidence
# ---------------------------------------------------------------------------


def test_pairwise_evidence_is_deterministic_and_complete(rig):
    engine, _, _, runtime = rig
    request = _request(n_views=4)
    first = engine.probe_candidates(request, runtime)
    second = engine.probe_candidates(request, runtime)
    assert len(first.pairwise) == 4 * 3 // 2
    assert [(p.view_index_a, p.view_index_b) for p in first.pairwise] == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    ]
    assert first.pairwise == second.pairwise


def test_pairwise_visible_set_overlap_matches_the_masks(rig, monkeypatch):
    engine, model, _, runtime = rig[0], rig[2], None, rig[3]
    renderers = {
        1: _fake_render_factory(model, visible_from=0, visible_to=8),
        2: _fake_render_factory(model, visible_from=4, visible_to=12),
    }

    def _dispatch(gm, camera, pipe, background, **kwargs):
        return renderers[camera.uid](gm, camera, pipe, background, **kwargs)

    monkeypatch.setattr(candidate_probe, "invoke_render", _dispatch)
    batch = engine.probe_candidates(_request(n_views=2), runtime)
    pair = batch.pairwise[0]
    assert pair.visible_gaussian_intersection == 4
    assert pair.visible_gaussian_jaccard == pytest.approx(4 / 12)
    assert pair.visible_gaussian_containment == pytest.approx(4 / 8)
    assert batch.per_view[0].visible_gaussian_ids.tolist() == list(range(8))


def test_pairwise_pose_evidence_uses_the_admission_side_camera_geometry(rig):
    engine, _, _, runtime = rig
    batch = engine.probe_candidates(_request(n_views=2), runtime)
    pair = batch.pairwise[0]
    # FakeCamera places centres at x = seed, i.e. 1.0 and 2.0.
    assert pair.pose_baseline_m == pytest.approx(1.0)
    assert pair.pose_angle_deg == pytest.approx(0.0, abs=1e-4)
    assert -1.0 <= pair.residual_sketch_cosine <= 1.0


def test_pairwise_is_skippable(rig):
    engine, _, _, runtime = rig
    batch = engine.probe_candidates(
        _request(n_views=3, compute_pairwise=False), runtime
    )
    assert batch.pairwise == ()


def test_pairwise_helper_is_pure_and_order_independent_of_call_count():
    assert build_pairwise_evidence([]) == ()


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_probe_accounting_is_separate_from_supervision_accounting(rig):
    engine, _, _, runtime = rig
    assert engine.probe_accounting == CandidateProbeAccounting()
    batch = engine.probe_candidates(_request(n_views=3), runtime)
    accounting = batch.accounting
    assert accounting.candidate_probe_batch_count == 1
    assert accounting.candidate_probe_view_count == 3
    assert accounting.candidate_probe_rasterizer_call_count == 3
    assert accounting.candidate_probe_pixel_count == 3 * H * W
    assert accounting.candidate_probe_wall_ms > 0.0
    assert accounting.candidate_probe_gradient_view_count == 0
    # Cumulative accounting accrues across batches.
    engine.probe_candidates(_request(n_views=2), runtime)
    assert engine.probe_accounting.candidate_probe_batch_count == 2
    assert engine.probe_accounting.candidate_probe_view_count == 5
    # And it must NOT be reachable through, or added into, #703's names.
    assert not hasattr(accounting, "supervision_view_count")
    assert not hasattr(accounting, "supervised_pixel_count")


def test_rasterizer_call_count_makes_a_true_batch_claim_verifiable(rig):
    """The sequential reference implementation charges one call per view."""

    engine, _, _, runtime = rig
    batch = engine.probe_candidates(_request(n_views=5), runtime)
    assert batch.accounting.candidate_probe_rasterizer_call_count == 5
    assert batch.accounting.candidate_probe_view_count == 5


# ---------------------------------------------------------------------------
# Gradient probe (Phase 3) -- correctness of the definition
# ---------------------------------------------------------------------------


class DifferentiableFakeRender:
    """A renderer whose output really depends on the Gaussian parameters.

    Needed to test the gradient channel at all: an image that does not carry
    the parameters' autograd graph would make every per-view gradient zero and
    the test would pass without proving anything.
    """

    def __call__(self, model, camera, pipe, background, **kwargs):
        height, width = camera.original_image.shape[-2:]
        scalar = model.params["means"].mean() + model.params["opacities"].mean()
        # Camera-dependent coupling, so the two views genuinely have DIFFERENT
        # gradients -- without it a batch-averaged gradient would be
        # indistinguishable from a correct per-view one.
        image = camera.original_image * 0.5 + scalar * float(camera.uid)
        payload = {
            "render": image,
            "alpha": torch.full((1, height, width), 0.5),
            "visibility_filter": torch.ones(N, dtype=torch.bool),
        }
        return RenderOutputs(payload=payload)


def test_gradient_probe_is_per_view_and_leaves_grad_untouched(rig, monkeypatch):
    engine, _, model, runtime = rig
    monkeypatch.setattr(candidate_probe, "invoke_render", DifferentiableFakeRender())
    request = _request(
        n_views=2,
        evidence_channels=frozenset(
            {ProbeEvidenceChannel.RGB, ProbeEvidenceChannel.GRADIENT}
        ),
        lambda_dssim=0.0,
    )
    batch = engine.probe_candidates(request, runtime)

    assert all(tensor.grad is None for tensor in model.params.values()), (
        "gradient-probe mode accumulated into .grad; it must use "
        "torch.autograd.grad, never loss.backward()"
    )
    assert batch.accounting.candidate_probe_gradient_view_count == 2
    for result in batch.per_view:
        gradient = result.gradient
        assert gradient is not None
        assert gradient.total_norm > 0.0
        assert math.isclose(
            gradient.total_norm, math.sqrt(gradient.total_energy), rel_tol=1e-6
        )
        # Per-family decomposition uses canonical names, not layout names.
        assert set(gradient.energy_by_family) <= {
            "means", "opacity", "scale", "rotation", "color"
        }
        assert gradient.energy_by_family["means"] > 0.0
        assert result.evidence.probe.gradient_norm.value == pytest.approx(
            gradient.total_norm
        )
        assert result.evidence.probe.gradient_energy_means.value > 0.0


def test_gradient_probe_matches_an_independently_computed_per_view_gradient(
    rig, monkeypatch
):
    """The reference: what ``autograd.grad`` on THIS view's own loss returns.

    This is the property issue #803 singles out -- a per-view gradient must be
    the gradient of that view's own objective, never a share of a reduced
    batch gradient. So the check is against a hand-computed single-view
    gradient, and (below) explicitly against the batch-mean, which must differ.
    """

    engine, _, model, runtime = rig
    render = DifferentiableFakeRender()
    monkeypatch.setattr(candidate_probe, "invoke_render", render)
    views = _request(n_views=2, lambda_dssim=0.0).views

    reference = []
    for view in views:
        image = render(model, view.camera, None, None).payload["render"]
        loss = (image - view.camera.original_image[:3]).abs().mean()
        grads = torch.autograd.grad(loss, [model.params["means"]])
        reference.append(float(grads[0].square().sum()))
    assert not math.isclose(reference[0], reference[1], rel_tol=1e-9), (
        "fixture too weak: the two views have the same gradient, so this test "
        "could not tell a per-view gradient from a batch-averaged one"
    )

    batch = engine.probe_candidates(
        CandidateProbeRequest(
            views=views,
            probe_batch_id="grad",
            lambda_dssim=0.0,
            evidence_channels=frozenset(
                {ProbeEvidenceChannel.RGB, ProbeEvidenceChannel.GRADIENT}
            ),
        ),
        runtime,
    )
    for expected, result in zip(reference, batch.per_view):
        assert result.gradient.energy_by_family["means"] == pytest.approx(
            expected, rel=1e-5
        )
    batch_mean = sum(reference) / 2.0
    assert not math.isclose(
        batch.per_view[0].gradient.energy_by_family["means"], batch_mean, rel_tol=1e-6
    )


def test_gradient_sketch_supports_pairwise_alignment(rig, monkeypatch):
    engine, _, _, runtime = rig
    monkeypatch.setattr(candidate_probe, "invoke_render", DifferentiableFakeRender())
    batch = engine.probe_candidates(
        _request(
            n_views=2,
            lambda_dssim=0.0,
            evidence_channels=frozenset(
                {ProbeEvidenceChannel.RGB, ProbeEvidenceChannel.GRADIENT}
            ),
        ),
        runtime,
    )
    cosine = batch.pairwise[0].gradient_sketch_cosine
    assert cosine is not None and -1.0 <= cosine <= 1.0


# ---------------------------------------------------------------------------
# No admission policy inside the engine
# ---------------------------------------------------------------------------


def test_probe_ships_no_combined_quality_scalar(rig):
    """Issue #803 forbids a fused score here; #500 owns that decision."""

    engine, _, _, runtime = rig
    batch = engine.probe_candidates(_request(n_views=2), runtime)
    forbidden = {"candidate_quality", "score", "utility", "rank", "admit", "priority"}
    for result in batch.per_view:
        names = set(type(result.evidence.probe).__dataclass_fields__)
        assert not (names & forbidden)
        assert not (set(type(result).__dataclass_fields__) & forbidden)
    assert not (set(type(batch).__dataclass_fields__) & forbidden)
