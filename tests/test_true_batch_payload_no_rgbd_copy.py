"""Per-view training payloads borrow RGB/depth; no unused concatenated image."""

import pytest
import torch

from eligsir.mapping.renderer.true_camera_batch import TrueCameraBatchResult
from eligsir.optimization.gaussian.true_batch_executor import _payload_for_view


@pytest.mark.parametrize("camera_count", [2, 4, 8])
@pytest.mark.parametrize("with_depth", [False, True])
def test_payload_borrows_requested_outputs_without_cat_and_preserves_gradients(
    monkeypatch, camera_count, with_depth
):
    rgb = torch.randn(camera_count, 3, 3, 5, requires_grad=True)
    alpha = torch.ones(camera_count, 1, 3, 5, requires_grad=True)
    depth = torch.full((camera_count, 1, 3, 5), 2.0, requires_grad=True)
    depth_ed = torch.full((camera_count, 1, 3, 5), 4.0, requires_grad=True)
    batch = TrueCameraBatchResult(
        renders=rgb, alphas=alpha,
        rendered_depth=depth if with_depth else None,
        rendered_depth_ed=depth_ed if with_depth else None,
        meta={}, camera_count=camera_count, sparse_grad=True,
    )

    def forbid_cat(*_args, **_kwargs):
        raise AssertionError("training payload must not concatenate unused RGB-D")

    monkeypatch.setattr(torch, "cat", forbid_cat)
    loss = rgb.new_zeros(())
    for index in range(camera_count):
        visible = torch.tensor([True, False, True])
        meta = {"gaussian_ids": torch.tensor([0, 2])}
        payload = _payload_for_view(
            batch, camera_index=index, visibility=visible, meta_i=meta
        ).payload
        assert payload["render"].data_ptr() == rgb[index].data_ptr()
        torch.testing.assert_close(payload["render"], rgb[index], rtol=0, atol=0)
        assert payload["visibility_filter"] is payload["is_used"] is visible
        assert payload["meta"] is meta
        assert "render_full" not in payload
        assert payload["alpha"].shape == (3, 5, 1)
        loss = loss + 2 * payload["render"].sum() + 7 * payload["alpha"].sum()
        if with_depth:
            assert payload["rendered_depth"].data_ptr() == depth[index].data_ptr()
            assert payload["rendered_depth_ed"].data_ptr() == depth_ed[index].data_ptr()
            loss = loss + 3 * payload["rendered_depth"].sum() + 5 * payload["rendered_depth_ed"].sum()
        else:
            assert "rendered_depth" not in payload and "rendered_depth_ed" not in payload
    loss.backward()
    torch.testing.assert_close(rgb.grad, torch.full_like(rgb, 2), rtol=0, atol=0)
    torch.testing.assert_close(alpha.grad, torch.full_like(alpha, 7), rtol=0, atol=0)
    if with_depth:
        torch.testing.assert_close(depth.grad, torch.full_like(depth, 3), rtol=0, atol=0)
        torch.testing.assert_close(depth_ed.grad, torch.full_like(depth_ed, 5), rtol=0, atol=0)
    else:
        assert depth.grad is depth_ed.grad is None
