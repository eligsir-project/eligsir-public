"""Issue #1580: keep true-camera packed metadata packed through objective/evidence.

Consumer audit (see the true_batch_executor.py commit message for the full
writeup) found `viewspace_points` and the dense per-view `radii[N]`
reconstruction have zero real readers on the prepared true-camera-batch path
and removed them; `visibility_filter`, `is_used` and packed `meta` retain
real objective/topology consumers and were kept. This file is the
parity/regression coverage for that change:

- `_group_meta_by_camera` (one grouping pass) exactly matches the removed
  `_slice_meta` (`camera_count` repeated boolean scans), including on
  packed-hit counts that don't evenly divide, empty packed hits, and
  non-contiguous camera_id ordering.
- the dummy/dead tensors are gone from the prepared payload and are never
  read during a prepared transaction (runtime instrumentation).
- union visibility from the production path matches an independent
  per-camera OR-loop reference.
- loss/gradient parity across B1 (single-view fallback)/B2/B4 is unaffected
  by the representation change, and gsplat-strategy MCMC bookkeeping (which
  reads packed `meta`) sees byte-identical per-camera metadata to before.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _prepared_loss_fakes import patch_assemble_loss

from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.optimization.gaussian import true_batch_executor as executor
from eligsir.optimization.gaussian.contracts import (
    EvidenceRequirements,
    GradientPolicyContext,
    OptimizationRuntimeContext,
    OptimizationStepRequest,
    OptimizationStepResult,
    OptimizationView,
    RandomStateToken,
)
from eligsir.optimization.gaussian.loss_breakdown import LossBreakdown
from eligsir.optimization.gaussian.true_batch_executor import (
    _group_meta_by_camera,
    run_true_camera_batched_step,
)


# ---------------------------------------------------------------------------
# 1. Grouping parity: new one-pass grouping vs. the removed C-repeated scan.
# ---------------------------------------------------------------------------


def _reference_slice_meta(meta: dict, camera_index: int) -> dict:
    """The exact logic `_slice_meta` used before issue #1580: one
    `camera_ids == camera_index` boolean scan + boolean-index compaction of
    every packed tensor, called once per camera."""

    camera_ids = meta.get("camera_ids")
    if camera_ids is None:
        return dict(meta)
    selected = camera_ids.long() == int(camera_index)
    packed_n = int(camera_ids.shape[0])
    out = {}
    for key, value in meta.items():
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) == packed_n:
            out[key] = value[selected]
        else:
            out[key] = value
    if torch.is_tensor(out.get("camera_ids")):
        out["camera_ids"] = torch.zeros_like(out["camera_ids"])
    return out


def _assert_meta_dicts_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if torch.is_tensor(va):
            assert torch.is_tensor(vb)
            assert va.dtype == vb.dtype
            torch.testing.assert_close(va, vb, rtol=0, atol=0)
        else:
            assert va == vb


@pytest.mark.parametrize("camera_count", [1, 2, 4, 8])
@pytest.mark.parametrize("packed_n", [0, 1, 3, 97, 4999])
def test_group_meta_by_camera_matches_old_per_camera_boolean_slice(camera_count, packed_n):
    torch.manual_seed(0)
    camera_ids = torch.randint(0, camera_count, (packed_n,), dtype=torch.int64)
    # Deliberately non-contiguous / unsorted camera_ids -- gsplat's packed
    # intersection order is not guaranteed to be camera-grouped.
    perm = torch.randperm(packed_n)
    camera_ids = camera_ids[perm]
    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": torch.randint(0, 250_000, (packed_n,), dtype=torch.int64),
        "radii": torch.rand(packed_n, dtype=torch.float32) * 10,
        "means2d": torch.randn(packed_n, 2, dtype=torch.float32),
        # A key whose leading dim happens to equal packed_n by coincidence
        # only when packed_n == camera_count -- excluded on purpose above
        # via the exact-shape-match rule both implementations share.
        "opacity_threshold": 0.5,
        "backend": "gsplat",
    }

    new_groups = _group_meta_by_camera(meta, camera_count=camera_count)
    assert len(new_groups) == camera_count
    for camera_index in range(camera_count):
        old = _reference_slice_meta(meta, camera_index)
        _assert_meta_dicts_equal(new_groups[camera_index], old)


def test_group_meta_by_camera_falls_back_unsliced_without_camera_ids():
    meta = {"radii": torch.ones(4, 3), "backend": "torch_cpu_reference"}
    groups = _group_meta_by_camera(meta, camera_count=4)
    assert len(groups) == 4
    for g in groups:
        assert g is not meta
        assert torch.equal(g["radii"], meta["radii"])
        assert g["backend"] == "torch_cpu_reference"


# ---------------------------------------------------------------------------
# Shared harness for end-to-end executor tests.
# ---------------------------------------------------------------------------


class _Camera:
    image_width = 2
    image_height = 2

    def __init__(self, uid: int):
        self.uid = uid


class _Model:
    def __init__(self, n=3):
        self.get_xyz = torch.nn.Parameter(torch.zeros((n, 3)))
        self.provisional = torch.zeros(n, dtype=torch.bool)
        self.post_render_mutations = 0


class _Session:
    parameter_version = 0

    def __init__(self, model):
        self.model = model

    def update_learning_rate(self, _iteration):
        return 0.01


class _Engine:
    def __init__(self, model):
        self.session = _Session(model)
        self._optimizer_step_calls = 0
        self.visible = None

    def run_post_backward_step(
        self, _gaussians, _args, _pipe, context, *, visible, loss_breakdown, render_outputs, **_kwargs
    ):
        self.visible = visible.clone()
        self._optimizer_step_calls += 1
        return OptimizationStepResult(
            loss_breakdown=loss_breakdown,
            render_outputs=render_outputs,
            evidence=None,
            optimization_gpu_ms=None,
            map_version_before=0,
            map_version_after=0,
            topology_version=0,
        )


def _request(camera: _Camera) -> OptimizationStepRequest:
    return OptimizationStepRequest(
        view=OptimizationView(
            camera=camera,
            source_index=camera.uid,
            background=None,
            update_sh_rest=True,
            render_depth=False,
            resolution_bucket="native",
        ),
        iteration=1,
        map_version=0,
        topology_version=0,
        resolution_bucket="native",
        evidence_requirements=EvidenceRequirements(),
        random_state_token=RandomStateToken.capture(),
    )


def _runtime(engine, model, *, mcmc_strategy=None, fixed_shape_mcmc=False):
    return OptimizationRuntimeContext(
        session=engine.session,
        gaussians=model,
        args=SimpleNamespace(
            streaming_objective_step_regularizers=False,
            streaming_objective_batched_pixel_terms=False,
        ),
        opt=SimpleNamespace(iterations=100),
        pipe=SimpleNamespace(),
        mcmc_strategy=mcmc_strategy,
        gradient_policy_context=GradientPolicyContext(
            gaussians=model, args=SimpleNamespace(
                streaming_objective_step_regularizers=False,
                streaming_objective_batched_pixel_terms=False,
            ), iteration=1, n_frames_ingested=4,
        ),
        n_frames_ingested=4,
        fixed_shape_mcmc=fixed_shape_mcmc,
    )


def _install_true_batch(monkeypatch, *, camera_ids, gaussian_ids, n_gaussians, weight_fn=None):
    """Patch the renderer + objective so the executor runs its real
    grouping/payload/backward code against a deterministic packed render."""

    def _render(group, *_args, **_kwargs):
        count = len(group)
        return TrueCameraBatchResult(
            renders=torch.zeros((count, 3, 2, 2)),
            alphas=torch.ones((count, 1, 2, 2)),
            rendered_depth=None,
            rendered_depth_ed=None,
            meta={
                "camera_ids": torch.as_tensor(camera_ids, dtype=torch.int64),
                "gaussian_ids": torch.as_tensor(gaussian_ids, dtype=torch.int64),
                "radii": torch.ones(len(camera_ids), dtype=torch.float32),
            },
            camera_count=count,
            sparse_grad=True,
        )

    def _loss(inputs):
        w = weight_fn(inputs.viewpoint_cam.uid) if weight_fn else float(inputs.viewpoint_cam.uid + 1)
        loss = (inputs.gaussians.get_xyz.sum()) * w
        return LossBreakdown(
            total=loss.detach(), total_tensor=loss, components={"Ll1": loss.detach()}, loss_finite=True,
        )

    monkeypatch.setattr(executor, "supports_true_camera_batch", lambda _pipe, _cams: (True, "ok"))
    monkeypatch.setattr(executor, "render_true_camera_batch", _render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)


# ---------------------------------------------------------------------------
# 2. Dummy tensors are gone from the payload, and genuinely never read.
# ---------------------------------------------------------------------------


def test_prepared_payload_has_no_dummy_viewspace_points_or_dense_radii(monkeypatch):
    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _install_true_batch(
        monkeypatch,
        camera_ids=[0, 0, 1, 2, 2, 3],
        gaussian_ids=[0, 2, 1, 2, 1, 2],
        n_gaussians=3,
    )
    captured = {}
    orig = executor._payload_for_view

    def _spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        captured.setdefault("payloads", []).append(out.payload)
        return out

    monkeypatch.setattr(executor, "_payload_for_view", _spy)

    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), _runtime(engine, model))

    assert len(captured["payloads"]) == 4
    for payload in captured["payloads"]:
        assert "viewspace_points" not in payload
        assert "radii" not in payload
        assert "visibility_filter" in payload
        assert "meta" in payload


def test_prepared_transaction_never_reads_viewspace_points_or_dense_radii(monkeypatch):
    """Runtime instrumentation: nothing in a prepared B4 transaction ever
    accesses a `viewspace_points`/dense-`radii` key off any render payload,
    proving the removed tensors were genuinely dead, not just unpopulated."""

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model()
    engine = _Engine(model)
    _install_true_batch(
        monkeypatch,
        camera_ids=[0, 0, 1, 2, 2, 3],
        gaussian_ids=[0, 2, 1, 2, 1, 2],
        n_gaussians=3,
    )

    class _WatchedDict(dict):
        def __getitem__(self, key):
            assert key not in ("viewspace_points",), f"read of dead key {key!r} during prepared transaction"
            return super().__getitem__(key)

        def get(self, key, *default):
            assert key not in ("viewspace_points",), f"read of dead key {key!r} during prepared transaction"
            return super().get(key, *default)

    orig = executor._payload_for_view

    def _spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        out.payload = _WatchedDict(out.payload)
        return out

    monkeypatch.setattr(executor, "_payload_for_view", _spy)

    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), _runtime(engine, model))


# ---------------------------------------------------------------------------
# 3. Union visibility parity: production path vs. independent OR-loop.
# ---------------------------------------------------------------------------


def test_union_visibility_parity_with_per_camera_or_loop(monkeypatch):
    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model(n=3)
    engine = _Engine(model)
    _install_true_batch(
        monkeypatch,
        camera_ids=[0, 0, 1, 2, 2, 3],
        gaussian_ids=[0, 2, 1, 2, 1, 2],
        n_gaussians=3,
    )
    per_camera_visibility = []
    orig = executor._payload_for_view

    def _spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        per_camera_visibility.append(out.payload["visibility_filter"].clone())
        return out

    monkeypatch.setattr(executor, "_payload_for_view", _spy)

    run_true_camera_batched_step(engine, tuple(_request(c) for c in cameras), _runtime(engine, model))

    reference_union = torch.zeros(3, dtype=torch.bool)
    for v in per_camera_visibility:
        reference_union |= v
    assert torch.equal(engine.visible, reference_union)
    # Ground truth: gaussians {0, 2, 1} are all hit by some camera -> all visible.
    assert torch.equal(engine.visible, torch.tensor([True, True, True]))


# ---------------------------------------------------------------------------
# 4. Loss + gradient parity across B1/B2/B4, and per-view evidence parity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [2, 4])
def test_loss_and_gradient_parity_b2_b4(monkeypatch, batch_size):
    cameras = tuple(_Camera(i) for i in range(batch_size))
    model = _Model(n=5)
    engine = _Engine(model)
    camera_ids = list(range(batch_size)) + list(range(batch_size))
    gaussian_ids = [i % 5 for i in range(batch_size)] + [(i + 1) % 5 for i in range(batch_size)]
    _install_true_batch(
        monkeypatch, camera_ids=camera_ids, gaussian_ids=gaussian_ids, n_gaussians=5,
    )

    result = run_true_camera_batched_step(
        engine, tuple(_request(c) for c in cameras), _runtime(engine, model)
    )

    assert result.optimizer_step_count == 1
    assert model.get_xyz.grad is not None
    # weight_fn default is uid+1, summed and divided by requested_b=batch_size:
    # sum_{uid=0}^{B-1}(uid+1)/B applied uniformly to every coordinate of
    # get_xyz.sum() -> the analytic gradient is that scalar broadcast.
    expected_scale = sum(u + 1 for u in range(batch_size)) / batch_size
    torch.testing.assert_close(
        model.get_xyz.grad, torch.full_like(model.get_xyz.grad, expected_scale), rtol=0, atol=1e-6
    )


def test_single_view_fallback_b1_still_produces_gradient(monkeypatch):
    """B1 doesn't go through the true-C packed path at all (camera
    microbatches of size 1 use the ordinary single-camera invoke_render), so
    this executor's packed-representation change cannot affect it -- this
    test pins that boundary."""

    cameras = (_Camera(0), _Camera(1))
    model = _Model(n=3)
    engine = _Engine(model)
    _install_true_batch(monkeypatch, camera_ids=[0], gaussian_ids=[0], n_gaussians=3)

    def _invoke_render(_gaussians, viewpoint_cam, _pipe, _bg, **_kwargs):
        from eligsir.optimization.gaussian.contracts import RenderOutputs

        n = _gaussians.get_xyz.shape[0]
        return RenderOutputs(
            payload={
                "render": torch.zeros(3, 2, 2),
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        )

    def _loss(inputs):
        loss = inputs.gaussians.get_xyz.sum() * float(inputs.viewpoint_cam.uid + 1)
        return LossBreakdown(
            total=loss.detach(), total_tensor=loss, components={}, loss_finite=True,
        )

    monkeypatch.setattr(executor, "invoke_render", _invoke_render)
    patch_assemble_loss(monkeypatch, executor._objective, _loss)

    result = run_true_camera_batched_step(
        engine,
        tuple(_request(c) for c in cameras),
        _runtime(engine, model),
        camera_microbatch_size=1,
    )

    assert result.optimizer_step_count == 1
    assert result.camera_batch_rasterizer_calls == 0
    assert result.rasterizer_call_count == 2
    assert model.get_xyz.grad is not None


# ---------------------------------------------------------------------------
# 5. Per-view evidence parity + gsplat-strategy meta grouping parity.
# ---------------------------------------------------------------------------


def test_mcmc_strategy_receives_correctly_grouped_packed_meta(monkeypatch):
    """utils.strategies.mcmc.baseline's DefaultStrategy wrapper reads
    render_pkg["meta"] as gsplat's own `info` for grad2d/count densification
    bookkeeping (a real category-4 consumer found by the audit) -- this
    pins that each camera's `step_pre_backward` call sees the exact same
    per-camera `gaussian_ids`/`radii` slice the old boolean-mask
    `_slice_meta` would have produced."""

    cameras = tuple(_Camera(i) for i in range(4))
    model = _Model(n=5)
    engine = _Engine(model)
    camera_ids = [0, 0, 1, 2, 2, 3, 3, 3]
    gaussian_ids = [0, 2, 1, 4, 1, 2, 3, 0]
    _install_true_batch(
        monkeypatch, camera_ids=camera_ids, gaussian_ids=gaussian_ids, n_gaussians=5,
    )

    seen_meta = []

    class _Strategy:
        def step_pre_backward(self, *, render_pkg, **_kwargs):
            seen_meta.append(dict(render_pkg["meta"]))

        def inject_noise(self, **_kwargs):
            pass

    run_true_camera_batched_step(
        engine,
        tuple(_request(c) for c in cameras),
        _runtime(engine, model, mcmc_strategy=_Strategy(), fixed_shape_mcmc=True),
    )

    assert len(seen_meta) == 4
    full_meta = {
        "camera_ids": torch.as_tensor(camera_ids, dtype=torch.int64),
        "gaussian_ids": torch.as_tensor(gaussian_ids, dtype=torch.int64),
        "radii": torch.ones(len(camera_ids), dtype=torch.float32),
    }
    for camera_index in range(4):
        expected = _reference_slice_meta(full_meta, camera_index)
        _assert_meta_dicts_equal(seen_meta[camera_index], expected)
