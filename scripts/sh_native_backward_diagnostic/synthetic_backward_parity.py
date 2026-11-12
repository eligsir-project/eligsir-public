import torch

from eligsir.utils.sh_utils import eval_sh
from gsplat.cuda._wrapper import spherical_harmonics as _native_spherical_harmonics


def compare_backward_gradients(deg: int, features_dc: torch.Tensor, features_rest: torch.Tensor, dirs: torch.Tensor) -> dict:
    """Backward-gradient comparison between EliGSiR's custom eval_sh
    and native gsplat's SH op, on the same synthetic inputs. Both sides
    get a fresh, independent leaf-tensor copy (requires_grad=True) of the
    same underlying values so gradients aren't accidentally shared or
    accumulated across the two computation graphs.

    Uses a plain `.sum()` loss on each side (no clamp/activation applied --
    this isolates the raw SH polynomial + its backward implementation,
    since Task 3's real-checkpoint audit already covers the full
    activation-inclusive pipeline).
    """
    # Deviation from the task-1 brief's verbatim code: native gsplat's
    # `spherical_harmonics` op requires CUDA tensors (confirmed empirically --
    # raises `RuntimeError: dirs must be a CUDA tensor` on CPU inputs), while
    # this module's test file (also verbatim per the brief) constructs inputs
    # with `torch.randn(...)` with no explicit device, i.e. CPU tensors. The
    # sibling `scripts/sh_clamp_diagnostic/native_gsplat_parity.py` forward-only
    # comparison relies on its *caller* constructing CUDA tensors directly
    # instead. To keep this task's test file byte-for-byte as specified in the
    # brief, the device move is done here instead, inside the comparison
    # function, so the function is robust regardless of input tensor device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shs_full = torch.cat([features_dc, features_rest[:, : (deg + 1) ** 2 - 1, :]], dim=1).to(device)
    dirs = dirs.to(device)

    shs_custom = shs_full.detach().clone().requires_grad_(True)
    dirs_custom = dirs.detach().clone().requires_grad_(True)
    raw_custom = eval_sh(deg, shs_custom.transpose(1, 2), dirs_custom)
    raw_custom.sum().backward()

    shs_native = shs_full.detach().clone().requires_grad_(True)
    dirs_native = dirs.detach().clone().requires_grad_(True)
    raw_native = _native_spherical_harmonics(deg, dirs_native, shs_native)
    raw_native.sum().backward()

    n_dc = features_dc.shape[1]
    grad_dc_custom, grad_rest_custom = shs_custom.grad[:, :n_dc, :], shs_custom.grad[:, n_dc:, :]
    grad_dc_native, grad_rest_native = shs_native.grad[:, :n_dc, :], shs_native.grad[:, n_dc:, :]

    # At deg == 0, dirs do not participate in either implementation's SH
    # polynomial (result depends only on the DC term), so .grad can be None
    # on one or both sides -- treat a None gradient as all-zero rather than
    # erroring, since "no dependency" is the correct/expected outcome here.
    dirs_grad_custom = dirs_custom.grad if dirs_custom.grad is not None else torch.zeros_like(dirs_custom)
    dirs_grad_native = dirs_native.grad if dirs_native.grad is not None else torch.zeros_like(dirs_native)

    diff_dc = (grad_dc_custom - grad_dc_native).abs()
    diff_rest = (grad_rest_custom - grad_rest_native).abs()
    diff_dir = (dirs_grad_custom - dirs_grad_native).abs()

    def _safe_max(t: torch.Tensor) -> float:
        # At deg == 0, features_rest is sliced to zero width ((deg+1)**2-1 == 0),
        # so diff_rest is an empty tensor -- .max()/.mean() error on empty
        # tensors with no dim specified. There is no "rest" gradient to compare
        # in that case, so report 0.0 (no disagreement possible on an empty set).
        return t.max().item() if t.numel() > 0 else 0.0

    def _safe_mean(t: torch.Tensor) -> float:
        return t.mean().item() if t.numel() > 0 else 0.0

    return {
        "max_abs_grad_error_dc": _safe_max(diff_dc),
        "mean_abs_grad_error_dc": _safe_mean(diff_dc),
        "max_abs_grad_error_rest": _safe_max(diff_rest),
        "mean_abs_grad_error_rest": _safe_mean(diff_rest),
        "max_abs_grad_error_dir": _safe_max(diff_dir),
        "mean_abs_grad_error_dir": _safe_mean(diff_dir),
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Synthetic backward-gradient parity across SH degrees 0-3")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    n = 64
    results = {}
    for deg in range(4):
        features_dc = torch.randn(n, 1, 3)
        features_rest = torch.randn(n, 15, 3)
        dirs = torch.nn.functional.normalize(torch.randn(n, 3), dim=1)
        results[f"degree_{deg}"] = compare_backward_gradients(deg, features_dc, features_rest, dirs)

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
