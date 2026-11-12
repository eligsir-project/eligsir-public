"""CPU parity tests for issue #1670 device-side gating of young-isotropy ramp.

Verifies that the refactored form (unconditional with device-side gate) is numerically
identical to the old form (conditional on host bool) for both all-zero and non-zero ramp cases.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.testing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.optimization.gaussian.losses import _streaming_young_isotropy_reg


# ============================================================================
# Site: losses._streaming_young_isotropy_reg
# ============================================================================

class _GaussiansWithYoungState:
    """Gaussians mock with young/immature state."""
    def __init__(self, n_gaussian=10, n_immature=5):
        self.get_xyz = torch.zeros(n_gaussian, 3, dtype=torch.float32)
        self.get_scaling = torch.ones(n_gaussian, 3, dtype=torch.float32)
        self.get_opacity = torch.ones(n_gaussian, 1, dtype=torch.float32)
        # Mark first n_immature as lifecycle_state=1 (provisional/young)
        self.lifecycle_state = torch.full((n_gaussian,), 2, dtype=torch.int32)
        self.lifecycle_state[:n_immature] = 1
        self.birth_frame = torch.zeros(n_gaussian, dtype=torch.int32)
        self.birth_iter = torch.zeros(n_gaussian, dtype=torch.int32)


def _streaming_young_isotropy_reg_old_form(gaussians, args, n_frames_ingested):
    """Old form: conditional on host bool(ramp > 0).any()."""
    from eligsir.utils.compiled_kernels import young_isotropy_ramp

    # Immature check (already device-side, not a host sync)
    immature_mask = gaussians.lifecycle_state <= 1
    if not immature_mask.any():
        return torch.zeros((), dtype=gaussians.get_xyz.dtype, device=gaussians.get_xyz.device)

    weight = float(getattr(args, "streaming_young_isotropy_weight", 0.0))
    birth_frame = gaussians.birth_frame
    ramp_frames = float(getattr(args, "streaming_young_isotropy_ramp_frames", 15))
    age = (n_frames_ingested - birth_frame[immature_mask].long()).clamp(min=0)
    ramp = young_isotropy_ramp(age, ramp_frames)

    # Host sync: conditional on bool(ramp > 0).any()
    if not bool((ramp > 0).any()):
        return None

    s = gaussians.get_scaling[immature_mask]
    if s.shape[-1] < 2:
        return None
    s_max = s.max(dim=-1).values
    s_min = s.min(dim=-1).values.clamp(min=1e-6)
    aniso_term = (s_max / s_min - 1.0).pow(2)
    return weight * (ramp * aniso_term).mean()


def test_young_isotropy_early_age_zero_ramp_exact_zero():
    """Early age (ramp all-zero): new form returns exactly 0.0 (no host sync)."""
    g = _GaussiansWithYoungState(n_gaussian=5, n_immature=3)
    # Scale up min dimension to avoid early return
    g.get_scaling = torch.ones(5, 3, dtype=torch.float32)

    args = SimpleNamespace(
        streaming_young_isotropy_weight=0.05,
        streaming_young_isotropy_ramp_frames=15,
    )
    # n_frames_ingested = 50: age = 50 - 0 = 50 >> ramp_frames=15 → ramp all ~0
    n_frames_ingested = 50

    result = _streaming_young_isotropy_reg(g, args, n_frames_ingested)
    # New form: ramp all-zero → (0 * aniso_term).mean() = 0.0
    assert torch.isfinite(result)
    assert result == 0.0


def test_young_isotropy_active_ramp_parity():
    """Active ramp (ramp has positive values): new form matches old form."""
    g = _GaussiansWithYoungState(n_gaussian=5, n_immature=3)
    g.get_scaling = torch.ones(5, 3, dtype=torch.float32)
    g.birth_frame = torch.zeros(5, dtype=torch.int32)  # All born at frame 0

    args = SimpleNamespace(
        streaming_young_isotropy_weight=0.1,
        streaming_young_isotropy_ramp_frames=15,
    )
    n_frames_ingested = 5  # age = 5, ramp(5, 15) = clamp(1 - 5/15) = 0.667

    old_result = _streaming_young_isotropy_reg_old_form(g, args, n_frames_ingested)
    new_result = _streaming_young_isotropy_reg(g, args, n_frames_ingested)

    # Old form might return None if ramp is all-zero (but it's not here)
    assert old_result is not None
    assert torch.isfinite(new_result)
    torch.testing.assert_close(old_result, new_result, atol=1e-6, rtol=1e-5)


def test_young_isotropy_no_immature_parity():
    """No immature Gaussians: both forms return exact 0.0."""
    g = _GaussiansWithYoungState(n_gaussian=5, n_immature=0)  # All mature
    g.get_scaling = torch.ones(5, 3, dtype=torch.float32)

    args = SimpleNamespace(
        streaming_young_isotropy_weight=0.05,
        streaming_young_isotropy_ramp_frames=15,
    )
    n_frames_ingested = 10

    old_result = _streaming_young_isotropy_reg_old_form(g, args, n_frames_ingested)
    new_result = _streaming_young_isotropy_reg(g, args, n_frames_ingested)

    # Both return exact 0.0
    assert torch.isfinite(new_result)
    assert new_result == 0.0
    # Old form also returns 0.0 (from the immature_mask.any() check)
    if old_result is not None:
        torch.testing.assert_close(old_result, new_result, atol=1e-6, rtol=1e-5)


def test_young_isotropy_mixed_ramp_values_parity():
    """Mixed ramp values (some active, some not): new form matches old form."""
    g = _GaussiansWithYoungState(n_gaussian=10, n_immature=5)
    g.get_scaling = torch.ones(10, 3, dtype=torch.float32)
    # Vary birth frames to get mixed ramp values
    g.birth_frame[:5] = torch.tensor([0, 5, 10, 0, 5], dtype=torch.int32)

    args = SimpleNamespace(
        streaming_young_isotropy_weight=0.1,
        streaming_young_isotropy_ramp_frames=10,
    )
    n_frames_ingested = 12
    # age = 12 - birth_frame
    # ramp(12-0, 10) = 0, ramp(12-5, 10) = 0.3, ramp(12-10, 10) = 0.8, etc.
    # So ramp has some positive, some zero values

    old_result = _streaming_young_isotropy_reg_old_form(g, args, n_frames_ingested)
    new_result = _streaming_young_isotropy_reg(g, args, n_frames_ingested)

    # Both should be present
    assert old_result is not None
    assert torch.isfinite(new_result)
    torch.testing.assert_close(old_result, new_result, atol=1e-6, rtol=1e-5)


# ============================================================================
# Mutation Control: verify the guard detects reintroduced host sync
# ============================================================================

def test_mutation_control_reintroduce_bool_ramp_check():
    """Mutation: reintroduce `if not bool((ramp > 0).any()): return None` → static guard must fail.

    This test verifies that the static guard (test_issue1670_hotpath_host_sync_guard.py)
    actually catches the host sync if we were to reintroduce it.
    """
    import ast
    import inspect
    import textwrap

    # Get the current implementation (without the host sync)
    source, original_lineno = (
        inspect.getsource(_streaming_young_isotropy_reg),
        inspect.getsourcelines(_streaming_young_isotropy_reg)[1],
    )
    dedented_source = textwrap.dedent(source)

    # Verify it parses and has no bool(*(ramp > 0).any()) pattern currently
    tree = ast.parse(dedented_source)

    # Now test the mutation: inject a bool((ramp > 0).any()) check
    mutated_source = dedented_source.replace(
        "# Device-side gating: compute unconditionally; ramp-all-zero gives exactly 0.0 (no host sync).",
        "if not bool((ramp > 0).any()):\n        return None\n    # Device-side gating would go here:",
    )

    # Parse the mutated version
    mutated_tree = ast.parse(mutated_source)

    # Use the same detector as in test_issue1670_hotpath_host_sync_guard.py
    from test_issue1670_hotpath_host_sync_guard import _HostSyncDetector

    detector = _HostSyncDetector(
        "eligsir.optimization.gaussian.losses", "_streaming_young_isotropy_reg"
    )
    detector.visit(mutated_tree)

    # The mutated version should have violations
    assert len(detector.violations) > 0, (
        "Mutation control failed: bool((ramp > 0).any()) not detected. "
        "This means the static guard is not properly checking the function."
    )
    # Specifically, it should detect a bool_tensor_reduction
    assert any(
        v.get("type") == "bool_tensor_reduction" for v in detector.violations
    ), (
        f"Expected bool_tensor_reduction violation, got {detector.violations}"
    )
