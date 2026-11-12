import torch
from scripts.sh_clamp_diagnostic.phase3_export_parity import eval_sh_playcanvas_style
from eligsir.utils.sh_utils import eval_sh


def test_eval_sh_playcanvas_style_matches_lower_only_clamp():
    torch.manual_seed(1)
    N = 6
    deg = 1
    shs = torch.randn(N, (deg + 1) ** 2, 3) * 2.0
    dirs = torch.nn.functional.normalize(torch.randn(N, 3), dim=1)

    out = eval_sh_playcanvas_style(shs, dirs, deg)
    raw = eval_sh(deg, shs.transpose(1, 2), dirs)
    expected = torch.clamp_min(raw + 0.5, 0.0)

    assert torch.allclose(out, expected, atol=1e-5)
    assert out.min() >= 0.0
