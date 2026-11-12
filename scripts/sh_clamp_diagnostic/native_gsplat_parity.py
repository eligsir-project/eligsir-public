import torch

from eligsir.utils.sh_utils import eval_sh
from eligsir.utils.compiled_kernels import _apply_sh_clamp

# Confirmed by reading eligsir/mapping/renderer/gsplat_backend.py:
#   - `_gsplat_sh_coeffs` (line ~100) does NOT call any SH evaluation function
#     itself -- it only concatenates raw features_dc/features_rest into a
#     [N, K, 3] coefficient tensor.
#   - SH evaluation for the "gsplat" backend happens INSIDE the `rasterization()`
#     call (imported at the top of that file as
#     `from gsplat.rendering import rasterization`), which is passed the raw
#     coefficients plus `sh_degree=pc.active_sh_degree` and evaluates SH
#     internally per visible Gaussian -- not reachable as a standalone call
#     without a full camera/projection setup.
#   - However, gsplat's installed package (this repo's custom ROCm/gsplat fork)
#     DOES expose a standalone callable SH op independent of `rasterization()`:
#     `gsplat.cuda._wrapper.spherical_harmonics(degrees_to_use, dirs, coeffs,
#     masks=None) -> Tensor`, confirmed via
#     `python -c "import gsplat.cuda._wrapper as w; print(dir(w))"` inside the
#     `train` container, which lists `spherical_harmonics` (and the underlying
#     autograd Function `_SphericalHarmonics`). This is the same op the
#     "gsplat" render backend relies on internally, just called directly here
#     on synthetic directions/coefficients for a forward-only comparison.
from gsplat.cuda._wrapper import spherical_harmonics as _native_spherical_harmonics


def compare_sh_implementations(deg: int, features_dc: torch.Tensor, features_rest: torch.Tensor, dirs: torch.Tensor) -> dict:
    """Forward-only numerical comparison of EliGSiR's custom eval_sh
    against native gsplat's SH op, with the SAME lower-only activation
    applied to both sides -- isolates implementation/basis differences
    from activation differences.
    """
    shs = torch.cat([features_dc, features_rest[:, : (deg + 1) ** 2 - 1, :]], dim=1)

    raw_custom = eval_sh(deg, shs.transpose(1, 2), dirs)
    custom_out = _apply_sh_clamp(raw_custom, "lower_only")

    raw_native = _native_spherical_harmonics(deg, dirs, shs)
    native_out = _apply_sh_clamp(raw_native, "lower_only")

    diff = (custom_out - native_out).abs()
    return {
        "max_abs_error": diff.max().item(),
        "mean_abs_error": diff.mean().item(),
        "rms_error": diff.pow(2).mean().sqrt().item(),
    }
