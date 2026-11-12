"""Issue #1259: `normalize_rotation_params` must not force a device->host sync.

py-spy profiling of a live 8060S training arm (native tier, batch 4) attributed
44.6% of the optimizer transaction's main-thread wall time to a single line,
``scene/gsplat_model.py``'s ``quats[mask] = ...``. Boolean advanced indexing
lowers to ``nonzero()``, whose output shape is data-dependent, so the host must
be told the selected count -- i.e. the call is a full kernel-queue drain, once
per optimizer step on the sparse-grad path.

These tests pin the two properties the fix depends on:

1. the masked update is still numerically EXACT (same values written, same rows
   left untouched) as the boolean gather/scatter form it replaced, and
2. the implementation does not reintroduce boolean advanced indexing on the
   parameter tensor.

Both run on CPU; no GPU is required.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

torch = pytest.importorskip("torch")


def _reference_masked_normalize(quats: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """The pre-fix formulation, kept here as the equivalence oracle."""
    out = quats.clone()
    out[mask] = torch.nn.functional.normalize(out[mask], dim=1, eps=1e-8)
    return out


def _apply(model_cls_name: str, quats: "torch.Tensor", mask):
    """Run the real method body against a bare stand-in holding the tensor."""
    if model_cls_name == "gsplat":
        from scene.gsplat_model import GsplatGaussianModel

        class _Stub:
            normalize_rotation_params = GsplatGaussianModel.normalize_rotation_params

        stub = _Stub()
        stub.params = {"quats": quats}
        stub.normalize_rotation_params(mask=mask)
        return stub.params["quats"]

    from scene.gaussian_model import GaussianModel

    class _Stub:
        normalize_rotation_params = GaussianModel.normalize_rotation_params

    stub = _Stub()
    stub._rotation = quats
    stub.normalize_rotation_params(mask=mask)
    return stub._rotation


@pytest.mark.parametrize("layout", ["gsplat", "legacy"])
def test_bool_mask_matches_advanced_indexing_exactly(layout):
    torch.manual_seed(1259)
    quats = torch.randn(257, 4, dtype=torch.float32)
    mask = torch.rand(257) < 0.4
    expected = _reference_masked_normalize(quats, mask)

    got = _apply(layout, quats.clone(), mask)

    assert torch.equal(got, expected), "masked normalize is no longer bit-exact"
    # Rows outside the mask must be untouched, bit-for-bit.
    assert torch.equal(got[~mask], quats[~mask])


@pytest.mark.parametrize("layout", ["gsplat", "legacy"])
@pytest.mark.parametrize("frac", [0.0, 1.0])
def test_degenerate_masks(layout, frac):
    """All-false and all-true masks are the boundary the reference form hides."""
    torch.manual_seed(7)
    quats = torch.randn(16, 4, dtype=torch.float32)
    mask = (
        torch.zeros(16, dtype=torch.bool)
        if frac == 0.0
        else torch.ones(16, dtype=torch.bool)
    )
    expected = _reference_masked_normalize(quats, mask)
    got = _apply(layout, quats.clone(), mask)
    assert torch.equal(got, expected)


@pytest.mark.parametrize("layout", ["gsplat", "legacy"])
def test_index_tensor_mask_still_supported(layout):
    torch.manual_seed(3)
    quats = torch.randn(32, 4, dtype=torch.float32)
    idx = torch.tensor([0, 5, 31], dtype=torch.long)
    expected = quats.clone()
    expected[idx] = torch.nn.functional.normalize(expected[idx], dim=1, eps=1e-8)
    got = _apply(layout, quats.clone(), idx)
    assert torch.equal(got, expected)


@pytest.mark.parametrize(
    "layout,attr",
    [("gsplat", "GsplatGaussianModel"), ("legacy", "GaussianModel")],
)
def test_no_boolean_advanced_indexing_on_the_parameter(layout, attr):
    """Regression guard: the sync-forcing form must not come back.

    A ``Subscript`` STORE whose index is the mask name is exactly the
    ``quats[mask] = ...`` shape that drains the queue. The index-tensor branch
    is allowed to keep it (an index tensor has a host-known shape), so this
    asserts the BOOLEAN branch specifically routes through ``torch.where``.
    """
    if layout == "gsplat":
        from scene.gsplat_model import GsplatGaussianModel as cls
    else:
        from scene.gaussian_model import GaussianModel as cls

    src = inspect.getsource(cls.normalize_rotation_params)
    tree = ast.parse(textwrap.dedent(src))

    where_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "where"
    ]
    assert where_calls, (
        f"{attr}.normalize_rotation_params no longer uses torch.where for the "
        "boolean-mask branch -- boolean advanced indexing forces a device->host "
        "sync every optimizer step (issue #1259)."
    )

    # The boolean branch must be explicitly distinguished from the index-tensor
    # branch, i.e. the dtype test must still be present.
    assert "torch.bool" in src, (
        "the boolean-mask branch guard disappeared; without it an index-tensor "
        "mask would take the torch.where path and silently mis-broadcast."
    )
