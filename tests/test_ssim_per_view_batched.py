"""Test batched SSIM with up-front non-finite view exclusion.

CPU-only tests in reference mode via set_ssim_mode("ref").
"""

import pytest

torch = pytest.importorskip("torch")

from eligsir.utils.loss_utils import set_ssim_mode, ssim_per_view, ssim


class TestSSIMPerViewBatchedParity:
    """Parity between batched and per-view paths."""

    def test_all_finite_batch_parity(self):
        """B=4 random [4,3,32,40] images: ssim_per_view allclose to per-image ssim()."""
        set_ssim_mode("ref")
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40, requires_grad=True)
        img2 = torch.randn(4, 3, 32, 40, requires_grad=True)

        # Batched path
        batched_results = ssim_per_view(img1, img2, window_size=11)

        # Per-view path (reference)
        per_view_results = tuple(
            ssim(img1[i : i + 1], img2[i : i + 1], window_size=11)
            for i in range(4)
        )

        # All results should be scalars
        assert len(batched_results) == 4
        assert all(r.dim() == 0 for r in batched_results)

        # Compare values
        for i, (batched, per_view) in enumerate(zip(batched_results, per_view_results)):
            torch.testing.assert_close(batched, per_view, atol=1e-6, rtol=1e-5)

    def test_gradient_parity(self):
        """Loss = sum of 4 scalars; img1.grad allclose to per-view path."""
        set_ssim_mode("ref")
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40, requires_grad=True)
        img2 = torch.randn(4, 3, 32, 40, requires_grad=True)

        # Batched path
        batched_results = ssim_per_view(img1, img2, window_size=11)
        batched_loss = sum(batched_results)
        batched_loss.backward()
        batched_grad = img1.grad.clone()

        # Per-view path
        img1.grad = None
        per_view_results = tuple(
            ssim(img1[i : i + 1], img2[i : i + 1], window_size=11)
            for i in range(4)
        )
        per_view_loss = sum(per_view_results)
        per_view_loss.backward()
        per_view_grad = img1.grad.clone()

        torch.testing.assert_close(batched_grad, per_view_grad, atol=1e-6, rtol=1e-5)


class TestSSIMPerViewNaNIsolation:
    """NaN views are excluded and do not propagate to finite views."""

    def test_nan_view_isolated(self):
        """Set view 1 to NaN; returned scalar 1 is NaN; others match per-view."""
        set_ssim_mode("ref")
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40, requires_grad=True)
        img2 = torch.randn(4, 3, 32, 40, requires_grad=True)

        # Set view 1 to NaN
        with torch.no_grad():
            img1[1].fill_(float("nan"))

        # Get per-view reference for finite views only
        per_view_refs = {}
        for i in [0, 2, 3]:
            per_view_refs[i] = ssim(
                img1[i : i + 1], img2[i : i + 1], window_size=11
            ).detach()

        # Batched path
        batched_results = ssim_per_view(img1, img2, window_size=11)

        # Check returned scalars
        assert torch.isnan(batched_results[1])  # NaN view is NaN
        for i in [0, 2, 3]:
            # Finite views should match per-view values
            torch.testing.assert_close(
                batched_results[i], per_view_refs[i], atol=1e-6, rtol=1e-5
            )

    def test_nan_isolation_gradient_clean(self):
        """loss = s0 + s2 + s3; finite-view grads finite, row 1 excluded."""
        set_ssim_mode("ref")
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40, requires_grad=True)
        img2 = torch.randn(4, 3, 32, 40, requires_grad=True)

        # Set view 1 to NaN
        with torch.no_grad():
            img1[1].fill_(float("nan"))

        batched_results = ssim_per_view(img1, img2, window_size=11)

        # Sum only finite views
        finite_loss = batched_results[0] + batched_results[2] + batched_results[3]
        finite_loss.backward()

        # Check grads: finite views are finite, NaN view may be anything
        finite_grad_mask = torch.isfinite(img1.grad[[0, 2, 3]])
        assert finite_grad_mask.all(), "Finite views must have finite gradients"


class TestSSIMPerViewCallCounting:
    """Verify batched vs per-view calls match the expected optimization."""

    def test_all_finite_single_batched_call(self):
        """All-finite B=4 -> exactly one batched call with batch 4."""
        set_ssim_mode("ref")
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40, requires_grad=True)
        img2 = torch.randn(4, 3, 32, 40, requires_grad=True)

        call_log = []

        # Monkeypatch _batched_ssim_map
        import eligsir.utils.loss_utils as loss_utils
        original_batched_map = loss_utils._batched_ssim_map if hasattr(loss_utils, '_batched_ssim_map') else None

        def tracked_batched_map(a, b, window_size):
            call_log.append(("batched_map", a.shape[0]))
            # Call the original if it exists, otherwise use _ssim path
            if original_batched_map is not None:
                return original_batched_map(a, b, window_size)
            else:
                # Reference fallback
                channel = a.size(1)
                window = loss_utils._reference_ssim_window(a, window_size, channel)
                return loss_utils._ssim(a, b, window, window_size, channel, size_average=False)

        loss_utils._batched_ssim_map = tracked_batched_map

        try:
            results = ssim_per_view(img1, img2, window_size=11)
            assert len(results) == 4
            assert len(call_log) == 1
            assert call_log[0] == ("batched_map", 4)
        finally:
            if original_batched_map is not None:
                loss_utils._batched_ssim_map = original_batched_map
            elif hasattr(loss_utils, '_batched_ssim_map'):
                delattr(loss_utils, '_batched_ssim_map')

    def test_mixed_nan_views_call_pattern(self):
        """One NaN view -> one call with batch 3 and one ssim call for NaN view."""
        set_ssim_mode("ref")
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40, requires_grad=True)
        img2 = torch.randn(4, 3, 32, 40, requires_grad=True)

        # Set view 1 to NaN
        with torch.no_grad():
            img1[1].fill_(float("nan"))

        call_log = []

        import eligsir.utils.loss_utils as loss_utils
        original_batched_map = loss_utils._batched_ssim_map if hasattr(loss_utils, '_batched_ssim_map') else None
        original_ssim = loss_utils.ssim

        def tracked_batched_map(a, b, window_size):
            call_log.append(("batched_map", a.shape[0]))
            if original_batched_map is not None:
                return original_batched_map(a, b, window_size)
            else:
                channel = a.size(1)
                window = loss_utils._reference_ssim_window(a, window_size, channel)
                return loss_utils._ssim(a, b, window, window_size, channel, size_average=False)

        def tracked_ssim(a, b, window_size=11, size_average=True):
            call_log.append(("ssim", a.shape[0]))
            return original_ssim(a, b, window_size, size_average)

        loss_utils._batched_ssim_map = tracked_batched_map
        loss_utils.ssim = tracked_ssim

        try:
            results = ssim_per_view(img1, img2, window_size=11)
            assert len(results) == 4
            # Should have 1 batched call with batch 3 + 1 ssim call with batch 1 (NaN)
            assert ("batched_map", 3) in call_log
            assert ("ssim", 1) in call_log
        finally:
            loss_utils.ssim = original_ssim
            if original_batched_map is not None:
                loss_utils._batched_ssim_map = original_batched_map
            elif hasattr(loss_utils, '_batched_ssim_map'):
                delattr(loss_utils, '_batched_ssim_map')


class TestSSIMPerViewFusedWiring:
    """Verify fused mode wiring when _FUSED_SSIM_MAP_FN is available."""

    def test_fused_batched_call_shape_and_contiguity(self):
        """Fused mode with fake _FUSED_SSIM_MAP_FN sees B, contiguous, padding 'same'."""
        set_ssim_mode("ref")  # Start in ref mode
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40, requires_grad=True)
        img2 = torch.randn(4, 3, 32, 40, requires_grad=True)

        import eligsir.utils.loss_utils as loss_utils

        # Save original state
        original_mode = loss_utils._SSIM_MODE
        original_fused_fn = loss_utils._FUSED_SSIM_MAP_FN

        fused_calls = []

        def fake_fused_map_fn(C1, C2, a, b, padding, train):
            # Record call details
            fused_calls.append({
                "batch": a.shape[0],
                "channel": a.shape[1],
                "height": a.shape[2],
                "width": a.shape[3],
                "contiguous_a": a.is_contiguous(),
                "contiguous_b": b.is_contiguous(),
                "padding": padding,
                "train": train,
            })
            # Return ones-shaped map
            return torch.ones_like(a)

        try:
            # Manually set fused mode with fake function
            loss_utils._SSIM_MODE = "fused"
            loss_utils._FUSED_SSIM_MAP_FN = fake_fused_map_fn

            results = ssim_per_view(img1, img2, window_size=11)

            assert len(results) == 4
            assert len(fused_calls) == 1
            call = fused_calls[0]
            assert call["batch"] == 4
            assert call["channel"] == 3
            assert call["contiguous_a"]
            assert call["contiguous_b"]
            assert call["padding"] == "same"
            assert call["train"] is True  # torch.is_grad_enabled()
        finally:
            loss_utils._SSIM_MODE = original_mode
            loss_utils._FUSED_SSIM_MAP_FN = original_fused_fn


class TestSSIMPerViewReturnType:
    """Verify return type contract: tuple of B 0-d tensors."""

    def test_returns_tuple_of_scalars(self):
        """ssim_per_view returns exactly B 0-d tensors."""
        set_ssim_mode("ref")
        torch.manual_seed(42)
        img1 = torch.randn(4, 3, 32, 40)
        img2 = torch.randn(4, 3, 32, 40)

        result = ssim_per_view(img1, img2, window_size=11)

        assert isinstance(result, tuple)
        assert len(result) == 4
        for item in result:
            assert isinstance(item, torch.Tensor)
            assert item.dim() == 0  # Scalar
            assert item.dtype == torch.float32
