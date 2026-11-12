import torch
from eligsir.utils.compiled_kernels import _sh_to_rgb_deg1, sh_to_rgb_eager


def _fixture():
    torch.manual_seed(0)
    N = 8
    features_dc = torch.randn(N, 1, 3) * 3.0  # scaled up so some values overshoot [0,1] post +0.5
    features_rest = torch.zeros(N, 15, 3)
    dir_pp_normalized = torch.nn.functional.normalize(torch.randn(N, 3), dim=1)
    return features_dc, features_rest, dir_pp_normalized


def test_deg1_clamped_default_matches_existing_behavior():
    features_dc, features_rest, dir_pp_normalized = _fixture()
    out_default = _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized)
    out_explicit = _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized, clamp_mode="clamped")
    assert torch.equal(out_default, out_explicit)
    assert out_default.min() >= 0.0 and out_default.max() <= 1.0


def test_deg1_lower_only_removes_upper_clamp():
    features_dc, features_rest, dir_pp_normalized = _fixture()
    out_lower = _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized, clamp_mode="lower_only")
    assert out_lower.min() >= 0.0
    assert out_lower.max() > 1.0  # fixture is scaled specifically to overshoot


def test_deg1_invalid_clamp_mode_raises():
    features_dc, features_rest, dir_pp_normalized = _fixture()
    try:
        _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized, clamp_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_sh_to_rgb_eager_lower_only_bypasses_registry_cache():
    features_dc, features_rest, dir_pp_normalized = _fixture()
    out_default = sh_to_rgb_eager(1, features_dc, features_rest, dir_pp_normalized)
    out_lower = sh_to_rgb_eager(1, features_dc, features_rest, dir_pp_normalized, clamp_mode="lower_only")
    assert torch.equal(out_default, _sh_to_rgb_deg1(features_dc, features_rest, dir_pp_normalized))
    assert out_lower.max() > 1.0


def test_python_sh_colors_default_matches_pre_change_expression():
    import torch as _torch
    from eligsir.mapping.renderer.gsplat_backend import _python_sh_colors

    class _FakePipe:
        pass

    class _FakePC:
        active_sh_degree = 0
        _features_dc = _torch.randn(5, 1, 3) * 3.0

    pc = _FakePC()
    pipe = _FakePipe()
    out = _python_sh_colors(pc, means=None, cam_center=None, sparse_grad=False, pipe=pipe, update_sh_rest=True, compiled=False)
    expected = _torch.clamp(pc._features_dc[:, 0, :] + 0.5, 0.0, 1.0).contiguous()
    assert _torch.equal(out, expected)
