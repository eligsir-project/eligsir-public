import os

import pytest

from scripts.sh_native_backward_diagnostic.gradient_graph_audit import compute_gradient_audit

_CKPT_FR3 = "/data/ckpt_fr3"


@pytest.mark.skipif(
    not os.path.isdir(_CKPT_FR3),
    reason=f"needs a real, in-progress training checkpoint mounted at {_CKPT_FR3}",
)
def test_reports_expected_keys_and_finite_values():
    # Uses a real frozen checkpoint fixture path (same convention as
    # tests/test_sh_clamp_diagnostic_ply_degree.py in this repo -- only
    # runs meaningfully when the real fr3 checkpoint is mounted at
    # /data/ckpt_fr3; this test still asserts the function's return
    # *shape* is correct, not specific numeric values, since those depend
    # on the real (mutable, potentially still-training) checkpoint state).
    result = compute_gradient_audit("/data/ckpt_fr3", "gsplat")
    expected_keys = {
        "means_grad_norm_python", "means_grad_norm_gsplat",
        "sh0_grad_norm_python", "sh0_grad_norm_gsplat",
        "shN_grad_norm_python", "shN_grad_norm_gsplat",
        "means_grad_max_abs_diff",
    }
    assert set(result.keys()) == expected_keys
    for v in result.values():
        assert v == v, f"NaN found in gradient audit result: {result}"  # v==v is False for NaN
        assert v >= 0.0 or v != v  # norms/diffs are non-negative (redundant with NaN check, defensive)
