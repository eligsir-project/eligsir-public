import pytest
import torch
from scripts.sh_clamp_diagnostic.native_gsplat_parity import compare_sh_implementations

# Issue #555: this test allocates its inputs with `device="cuda"` outright,
# so without device passthrough it fails with `No HIP GPUs are available` --
# a false RED, not a defect. Capability probe, never an image tag. See
# tests/test_gaussian_model_ply_degree.py for the full rationale.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs GPU (see issue #555)"
)


def test_forward_parity_within_tight_tolerance_for_all_degrees():
    torch.manual_seed(2)
    for deg in [0, 1, 2, 3]:
        N = 50
        features_dc = torch.randn(N, 1, 3, device="cuda") * 1.5
        features_rest = torch.randn(N, 15, 3, device="cuda") * 0.5 if deg > 0 else torch.zeros(N, 15, 3, device="cuda")
        dirs = torch.nn.functional.normalize(torch.randn(N, 3, device="cuda"), dim=1)

        result = compare_sh_implementations(deg, features_dc, features_rest, dirs)
        assert result["max_abs_error"] < 1e-3, f"degree {deg} exceeded tolerance: {result}"
