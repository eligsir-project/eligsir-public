"""W3-35: MCMC noise-injection / relocation native-kernel audit — drift guard.

Task #35 asked whether upstream gsplat ships a fused native CUDA/HIP kernel
for MCMC noise-injection/relocation that this project does not yet call, in
the same spirit as the HIPified ``simple_knn`` extension documented in
CLAUDE.md's "ROCm Fallbacks" section (a real, ~30-290x win at large N).

Verdict (full writeup: ``docs/architecture/2026-08-10-w3-35-mcmc-kernel-audit.md``):

- ``gsplat.strategy.ops.compute_relocation`` (called from
  ``utils/strategies/mcmc/proposal.py``'s ``relocate_dead``/``add_energy_guided``,
  the default ``gsplat`` model-layout path) already dispatches to a compiled
  native kernel (``_C.relocation``) that this project's ROCm-forked gsplat
  build (``ROCm/gsplat`` @ ``release/1.5.3b2``, see Dockerfile) already
  successfully compiles and runs on this exact AMD hardware. No porting gap.
- ``gsplat.strategy.ops.inject_noise_to_position`` (called from
  ``utils/strategies/mcmc/energy.py``'s ``inject_noise`` default branch) is
  pure PyTorch in the pinned version (1.5.3) — no native kernel to dispatch
  to at all.
- Upstream ``nerfstudio-project/gsplat`` *main* branch (unreleased, no tag
  newer than v1.5.3 exists as of this audit) added a genuinely fused native
  CUDA kernel for exactly this operation, ``torch.ops.gsplat.mcmc_perturb_
  positions`` (``gsplat/cuda/csrc/MCMCPerturbCUDA.cu``, commit ``b5392feb``,
  2026-04-22) — this project's premise was correct, this kernel exists and
  we do not call it.
- However, a real isolated GPU microbenchmark of the CURRENT pure-PyTorch
  ``inject_noise_to_position`` (this repo's exact call pattern, N=7900 and
  N=28400 — the two W3-56 operating points) measured ~0.16ms/call,
  independent of N (launch-overhead dominated, consistent with W3-56's
  sublinear-per-call-cost finding) — roughly 1.0-1.3% of a real training
  step's wall-clock at both operating points. Even eliminating ALL of that
  cost via fusion would land far below this project's established ≥5%
  end-to-end-throughput bar for promoting an optimization. Verdict: DEFER,
  no action taken, no production code changed by this task.

This test is a drift guard, not a parity test (no production code changed):
if gsplat is ever upgraded past 1.5.3, or if the ROCm fork ever backports
the fused kernel, these assertions will start failing — which is the
correct trigger to redo this audit's benchmark against the new code path,
since the conclusion above is only valid for the exact gsplat build in use
at the time of this audit.
"""
import inspect

import pytest


def test_gsplat_version_matches_audited_build():
    """Documents the exact upstream state this audit's conclusion is valid for.

    A version bump does not necessarily mean the fused kernel arrived (it
    may not have been backported by ROCm/gsplat yet either), but it is the
    cheapest possible signal that something changed and the conclusion in
    docs/architecture/2026-08-10-w3-35-mcmc-kernel-audit.md should be
    re-checked rather than assumed to still hold.
    """
    gsplat = pytest.importorskip("gsplat")
    version = getattr(gsplat, "__version__", None)
    assert version == "1.5.3", (
        f"gsplat version changed ({version!r} != '1.5.3') since the W3-35 "
        "MCMC noise-injection/relocation kernel audit — re-check "
        "docs/architecture/2026-08-10-w3-35-mcmc-kernel-audit.md's "
        "conclusion (upstream may have shipped the fused "
        "mcmc_perturb_positions kernel by now)."
    )


def test_inject_noise_to_position_is_still_pure_pytorch():
    """Confirms the pinned gsplat's inject_noise_to_position has no native
    fused-kernel dispatch (torch.ops.gsplat.mcmc_perturb_positions or
    equivalent). If this starts failing, the fused kernel has arrived in
    our build and the W3-35 "defer, cost is <1.3% of wall-clock" verdict
    needs to be re-measured against the new code path before being trusted.

    The load-bearing check is on the torch op *registry*, not source text:
    a backport could rename the dispatch helper (e.g. `_fused_perturb`
    instead of `_cuda_fused_mcmc_perturb`) and still pass a text-only
    check while the kernel is live and callable. Checking
    ``torch.ops.gsplat`` directly can't be defeated by a rename. The
    source-text check is kept as a secondary, human-readable signal only.
    """
    import torch

    ops = pytest.importorskip("gsplat.strategy.ops")
    gsplat_ops_namespace = getattr(torch.ops, "gsplat", None)
    assert not (
        gsplat_ops_namespace is not None
        and hasattr(gsplat_ops_namespace, "mcmc_perturb_positions")
    ), (
        "torch.ops.gsplat.mcmc_perturb_positions is now registered — the "
        "upstream native kernel this W3-35 audit found on gsplat's main "
        "branch (unreleased at audit time) has apparently landed in our "
        "pinned build. Re-run the W3-35 microbenchmark against this new "
        "path before drawing any conclusion from it."
    )
    # Secondary, human-readable signal: the pinned version's own source
    # calls no torch.ops.gsplat.* native op at all in this function body.
    src = inspect.getsource(ops.inject_noise_to_position)
    assert "torch.ops.gsplat" not in src


def test_compute_relocation_dispatches_to_native_kernel():
    """compute_relocation (used by the default gsplat model-layout MCMC
    proposal path, utils/strategies/mcmc/proposal.py) must keep dispatching
    to gsplat's compiled native `relocation` kernel -- confirms this
    project's ROCm-forked gsplat build still successfully exposes it, which
    is the load-bearing fact behind this audit's "relocation already uses
    the best available native kernel, no further porting needed" finding.
    """
    try:
        from gsplat.cuda._backend import _C as C
    except Exception as exc:  # pragma: no cover - environment-dependent
        # importorskip only catches ImportError; on a CPU-only/no-native-extension
        # runner this can raise other exceptions (e.g. from a failed .so load).
        # This test is only meaningful where the native backend is actually
        # loadable (a GPU training container), so skip rather than fail there.
        pytest.skip(f"gsplat native backend not loadable in this environment: {exc!r}")
    assert hasattr(C, "relocation"), (
        "gsplat's native _C.relocation kernel is no longer exposed by this "
        "build -- utils/strategies/mcmc/proposal.py's compute_relocation "
        "call would silently be missing its compiled backend. This "
        "contradicts the W3-35 audit's finding; re-check the ROCm/gsplat "
        "build (Dockerfile gsplat-builder stage)."
    )
