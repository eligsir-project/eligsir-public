from arguments import get_default_pipeline_params


def test_default_pipeline_params_matches_real_defaults():
    pipe = get_default_pipeline_params()
    assert pipe.sh_impl == "gsplat"
    assert pipe.sh_color_clamp_mode == "clamped"
    assert pipe.tile_size == 16
    assert pipe.near_plane == 0.01
    assert pipe.far_plane == 1e10
    assert pipe.radius_clip == 0.0
    assert pipe.eps2d == 0.3
    assert pipe.render_mode == "RGB"
    assert pipe.absgrad is False
    assert pipe.rasterize_mode == "classic"
    assert pipe.convert_SHs_python is False
    assert pipe.compute_cov3D_python is False
    assert pipe.debug is False


def test_default_pipeline_params_does_not_include_optimization_fields():
    # gsplat_sparse_grad and sparse_mode_detach_sh_dir are OptimizationParams
    # fields, not PipelineParams -- get_default_pipeline_params() must not
    # set them. gsplat_backend.py's own getattr(pipe, "gsplat_sparse_grad",
    # False) fallback is what supplies the correct (False) effective value
    # for headless eval call sites with no training loop to keep raw
    # rotations normalized -- see this plan's Global Constraints for why.
    pipe = get_default_pipeline_params()
    assert not hasattr(pipe, "gsplat_sparse_grad")
    assert not hasattr(pipe, "sparse_mode_detach_sh_dir")
