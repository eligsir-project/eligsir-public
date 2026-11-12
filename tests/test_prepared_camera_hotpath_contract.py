from __future__ import annotations

from types import SimpleNamespace

import torch

import scene.cameras as cameras


def _prepared_cam(offset: float):
    viewmat = torch.eye(4, dtype=torch.float32)
    viewmat[0, 3] = offset
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    center = torch.tensor([[offset, 0.0, 0.0]], dtype=torch.float32)
    return SimpleNamespace(
        data_device=torch.device("cpu"),
        image_width=640,
        image_height=480,
        world_view_transform=viewmat.transpose(0, 1),
        camera_center=center[0],
        gsplat=cameras.GsplatCameraTensors(
            viewmat=viewmat,
            K=K,
            camera_center=center,
            width=640,
            height=480,
        ),
    )


def test_true_camera_batch_reuses_sealed_device_intrinsics(monkeypatch):
    cams = (_prepared_cam(0.0), _prepared_cam(1.0))

    def fail_legacy_intrinsics(_cam):
        raise AssertionError("prepared batching must not rebuild host intrinsics")

    monkeypatch.setattr(cameras, "_camera_intrinsics", fail_legacy_intrinsics)
    batch = cameras.make_gsplat_camera_batch(cams, device="cpu")

    assert torch.equal(batch.Ks[0], cams[0].gsplat.K)
    assert torch.equal(batch.Ks[1], cams[1].gsplat.K)
    assert batch.camera_centers.shape == (2, 3)
    assert batch.widths == (640, 640)
    assert batch.heights == (480, 480)


def test_prepared_score_provenance_does_not_call_full_tensor_validator(monkeypatch):
    cam = object.__new__(cameras.Camera)
    torch.nn.Module.__init__(cam)
    cam.original_image = torch.zeros((3, 2, 3), dtype=torch.float32)

    import eligsir.utils.score_contract as score_contract

    def fail_validation(*_args, **_kwargs):
        raise AssertionError("READY binding must not run score-domain reductions")

    monkeypatch.setattr(score_contract, "declare_camera_score_provenance", fail_validation)
    cam._declare_prepared_score_provenance()

    assert cam.score_provenance is not None
    assert cam._gt_finite_validated is True
    assert cam._gt_finite_validated_id == id(cam.original_image)
    assert cam._gt_finite_validated_version == int(cam.original_image._version)


def test_prepared_attestation_fails_closed_on_in_place_mutation():
    """issue #1671: object id survives an in-place tensor mutation, so the
    latch must also fence on `Tensor._version` or a mutated prepared RGB
    payload would be trusted without re-validation."""
    cam = object.__new__(cameras.Camera)
    torch.nn.Module.__init__(cam)
    cam.original_image = torch.zeros((3, 2, 2), dtype=torch.float32)
    cam._declare_prepared_score_provenance()
    sealed_version = cam._gt_finite_validated_version

    cam.original_image.add_(1.0)  # in-place: id unchanged, `_version` bumped

    assert cam._gt_finite_validated_id == id(cam.original_image)
    assert int(cam.original_image._version) != sealed_version


def test_objective_resanitizes_gt_after_in_place_mutation_despite_same_id():
    from eligsir.cli.parser import parse_training_args
    from eligsir.optimization.gaussian.contracts import RenderOutputs
    from eligsir.optimization.gaussian.objective import LossAssemblyInputs, assemble_loss

    args, bundle = parse_training_args(["-s", "/dev/null"])
    args.opacity_reg = 0.0
    args.scale_reg = 0.0
    args.anisotropy_reg = 0.0
    args.streaming_young_reg_exempt_iters = 0
    args.streaming_maturity_reg_ramp = False
    args.streaming_recency_reg_horizon = 0

    cam = object.__new__(cameras.Camera)
    torch.nn.Module.__init__(cam)
    cam.original_image = torch.zeros((3, 4, 4), dtype=torch.float32)
    cam._declare_prepared_score_provenance()

    # Mutate the sealed tensor in place with a non-finite pixel. Object id is
    # unchanged, so a latch keyed on id alone would wrongly trust this tensor.
    cam.original_image[0, 0, 0] = float("nan")

    n = 3
    gaussians = SimpleNamespace(
        _opacity=torch.zeros((n, 1), requires_grad=True),
        get_scaling=torch.ones((n, 3), requires_grad=True),
        get_xyz=torch.zeros((n, 3)),
        provisional=torch.zeros(n, dtype=torch.bool),
    )
    gaussians.get_opacity = torch.sigmoid(gaussians._opacity)

    inputs = LossAssemblyInputs(
        gaussians=gaussians,
        viewpoint_cam=cam,
        args=args,
        opt=bundle.optimization_params.extract(args),
        pipe=bundle.pipeline_params.extract(args),
        render_outputs=RenderOutputs(
            payload={
                "render": torch.zeros((3, 4, 4), requires_grad=True),
                "visibility_filter": torch.ones(n, dtype=torch.bool),
            }
        ),
        image=torch.zeros((3, 4, 4), requires_grad=True),
        iteration=1,
        optimizer_update_id=1,
        n_frames_ingested=1,
    )

    breakdown = assemble_loss(inputs)

    assert bool(torch.isfinite(breakdown.total_tensor))
    assert bool(torch.isfinite(cam.original_image).all())
    assert cam._gt_finite_validated_version == int(cam.original_image._version)
