"""CPU contracts for the cache-independent post-mapping fitter baseline."""


def test_generic_post_mapping_fitter_imports_without_source_builder():
    from eligsir.mapping import post_mapping_pose_align as align
    from eligsir.utils import eval_pose_align

    assert callable(eval_pose_align._fit_pose_delta)
    assert callable(eval_pose_align._frozen_gaussians)
    assert callable(align.align_training_frames)
    assert not hasattr(align, "build_frame_handles_from_source")


def test_shared_tier_helpers_keep_intrinsics_and_gt_decimation_together():
    import torch
    from eligsir.mapping.post_mapping_pose_align import PostMappingPoseAlignConfig
    from eligsir.pose_opt import _TieredCamera, _tiered_gt

    class _Camera:
        image_width, image_height = 16, 8
        fx, fy, cx, cy = 80.0, 40.0, 8.0, 4.0
        gsplat = None

    image = torch.arange(3 * 8 * 16, dtype=torch.float32).reshape(3, 8, 16)
    half = _TieredCamera(_Camera(), "half")
    eighth = _TieredCamera(_Camera(), "eighth")

    assert (half.image_width, half.image_height, half.fx, half.fy, half.cx, half.cy) == (8, 4, 40.0, 20.0, 4.0, 2.0)
    assert (eighth.image_width, eighth.image_height) == (2, 1)
    assert torch.equal(_tiered_gt(image, "half"), image[:, ::2, ::2])
    assert torch.equal(_tiered_gt(image, "eighth"), image[:, ::8, ::8])
    assert PostMappingPoseAlignConfig(tier="eighth").tier == "eighth"


def test_fitter_runs_fifty_fresh_adam_steps_with_warm_start_and_restores_gaussians():
    import torch
    from eligsir.utils.eval_pose_align import _fit_pose_delta, _frozen_gaussians

    class _Camera:
        def __init__(self):
            self.world_view_transform = torch.eye(4)
            self.projection_matrix = torch.eye(4)
            self.camera_center = torch.zeros(3)

    class _Gaussians:
        def __init__(self):
            parameter = torch.nn.Parameter(torch.ones(1))
            self.optimizer = torch.optim.Adam([parameter], lr=1e-3)

    gaussians = _Gaussians()
    camera = _Camera()
    losses = {}

    def render_fn(cam, *_args):
        return {"render": cam.camera_center.view(3, 1, 1).expand(3, 2, 2)}

    with _frozen_gaussians(gaussians):
        assert all(not parameter.requires_grad for group in gaussians.optimizer.param_groups for parameter in group["params"])
        delta = _fit_pose_delta(
            camera, gaussians, render_fn, object(), None,
            torch.ones(3, 2, 2), lambda *_args: torch.zeros(()),
            n_steps=50, lr=5e-2,
            init_delta6=torch.tensor([0., 0., 0., .2, .2, .2]), loss_out=losses,
        )

    assert all(parameter.requires_grad for group in gaussians.optimizer.param_groups for parameter in group["params"])
    assert delta.shape == (4, 4)
    assert torch.isfinite(delta).all()
    assert losses["initial"] > losses["final"]


def test_pose_proxy_and_bake_refresh_canonical_gsplat_geometry_and_legacy_rt():
    import numpy as np
    import torch
    from scene.cameras import GsplatCameraTensors
    from eligsir.mapping.renderer.gsplat_backend import _camera_tensors
    from eligsir.pose_opt import apply_pose_delta, bake_pose_delta_into_camera, se3_delta_matrix
    from eligsir.utils.rgbd_frames import c2w_to_camera_rt

    class _Camera:
        image_width = image_height = 2
        fx = fy = cx = cy = 1.0

        def __init__(self):
            c2w = torch.eye(4)
            c2w[:3, 3] = torch.tensor([1.0, -2.0, 0.5])
            self.world_view_transform = torch.inverse(c2w).transpose(0, 1).contiguous()
            self.projection_matrix = torch.eye(4)
            self.full_proj_transform = self.world_view_transform.clone()
            self.camera_center = c2w[:3, 3].clone()
            self.c2w = c2w.clone()
            self.R, self.T = c2w_to_camera_rt(c2w.numpy())
            self.gsplat = GsplatCameraTensors(
                viewmat=torch.inverse(c2w).contiguous(),
                K=torch.tensor([[3.0, 0.0, 1.0], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]),
                camera_center=c2w[:3, 3].view(1, 3).contiguous(),
                width=2,
                height=2,
            )

    camera = _Camera()
    delta6 = torch.tensor([0.0, 0.0, 0.25, 0.1, 0.2, -0.3], requires_grad=True)
    delta = se3_delta_matrix(delta6)
    expected_c2w = camera.c2w @ delta
    expected_w2c = torch.inverse(expected_c2w)

    proxy = apply_pose_delta(camera, delta)
    viewmat, K, center = _camera_tensors(proxy, "cpu")
    torch.testing.assert_close(viewmat, expected_w2c)
    torch.testing.assert_close(center, expected_c2w[:3, 3].view(1, 3))
    torch.testing.assert_close(K, camera.gsplat.K)
    assert (proxy.gsplat.width, proxy.gsplat.height) == (2, 2)
    (viewmat.sum() + center.sum()).backward()
    assert delta6.grad is not None and torch.isfinite(delta6.grad).all()

    bake_pose_delta_into_camera(camera, delta.detach())
    baked_viewmat, baked_K, baked_center = _camera_tensors(camera, "cpu")
    torch.testing.assert_close(baked_viewmat, expected_w2c)
    torch.testing.assert_close(baked_center, expected_c2w[:3, 3].view(1, 3))
    torch.testing.assert_close(baked_K, K)
    expected_R, expected_T = c2w_to_camera_rt(expected_c2w.detach().numpy())
    np.testing.assert_allclose(camera.R, expected_R)
    np.testing.assert_allclose(camera.T, expected_T)


def test_generic_alignment_preserves_order_warm_start_clamp_and_exposure(monkeypatch):
    import numpy as np
    import torch
    from scene import cameras
    from eligsir.mapping.post_mapping_pose_align import (
        PoseAlignFrameHandle,
        PostMappingPoseAlignConfig,
        PostMappingPoseAlignStats,
        align_training_frames,
    )
    from eligsir.pose_opt import se3_delta_matrix
    import eligsir.utils.eval_pose_align as eval_pose_align

    class _Camera:
        image_width = image_height = 2
        fx = fy = cx = cy = 1.0

        def __init__(self):
            self.world_view_transform = torch.eye(4)
            self.projection_matrix = torch.eye(4)
            self.camera_center = torch.zeros(3)
            self.original_image = torch.ones(3, 2, 2)

    parameter = torch.nn.Parameter(torch.ones(1))
    gaussians = type("Gaussians", (), {
        "get_xyz": torch.zeros(1, 3),
        "optimizer": torch.optim.Adam([parameter], lr=1e-3),
    })()
    starts, persisted = [], []

    def fake_fit(camera, *_args, init_delta6=None, loss_out=None, **_kwargs):
        starts.append(None if init_delta6 is None else init_delta6.clone())
        loss_out.update(initial=2.0, final=1.0)
        return se3_delta_matrix(torch.tensor([0., 0., .5, .1, 0., 0.]))

    monkeypatch.setattr(cameras, "prepare_camera_for_render", lambda camera, **_kwargs: camera)
    monkeypatch.setattr(eval_pose_align, "_fit_pose_delta", fake_fit)
    handles = [
        PoseAlignFrameHandle("late", True, 2., _Camera, lambda *result: persisted.append(("late", result)), 2),
        PoseAlignFrameHandle("early", True, 1., _Camera, lambda *result: persisted.append(("early", result)), 1),
        PoseAlignFrameHandle("holdout", False, 3., _Camera, lambda *result: persisted.append(("holdout", result)), 3),
    ]
    stats = PostMappingPoseAlignStats()
    align_training_frames(
        handles, gaussians, lambda *_args: {"render": torch.zeros(3, 2, 2)}, object(), None,
        config=PostMappingPoseAlignConfig(steps=1, tier="native", exposure_align=True, max_deg=1., max_mm=20.),
        stats=stats, ssim_fn=lambda *_args: torch.zeros(()),
    )

    assert len(starts) == 2 and starts[0] is None and starts[1] is not None
    assert [name for name, _result in persisted] == ["early", "late"]
    assert stats.aligned == 2 and stats.skipped_not_train == 1
    assert stats.clamped_rotation == 2 and stats.clamped_translation == 2
    assert all(result[1] is not None and result[2] is not None for _name, result in persisted)
    assert all(not np.allclose(result[0], np.eye(4)) for _name, result in persisted)
