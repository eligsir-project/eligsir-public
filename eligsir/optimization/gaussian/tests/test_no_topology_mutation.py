"""Static + runtime check: the optimization engine never mutates topology.

Wave 3, PR W3-02 (plan section 7.5's proposed file list; deliverable #2 of
section 7.7: "No structural mutation is reachable from the engine"). Two
independent checks:

1. Static: ``engine.py``'s source never references any of the known
   topology-mutating entry points (``mcmc_strategy.step_post_backward``,
   ``begin_mutation``, ``GaussianMutationTransaction``, insert/prune/grow/
   relocate/reseed APIs). This catches an accidental import/call creeping
   in without needing a GPU.
2. Runtime (GPU-gated): the toy fixture's Gaussian count
   (``gaussians.get_xyz.shape[0]``) is identical before and after a full
   ``GaussianOptimizationEngine.run_step()`` call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ENGINE_PATH = Path(__file__).resolve().parents[1] / "engine.py"

_FORBIDDEN_CALL_PATTERNS = (
    ".step_post_backward(",
    ".begin_mutation(",
    "GaussianMutationTransaction(",
    ".insert_gaussians",
    ".prune_gaussians",
    ".grow_gaussians",
    ".relocate_gaussians",
    ".reseed(",
)


def _non_docstring_lines(source: str) -> str:
    """Strip triple-quoted docstrings/comment-only prose before scanning
    for forbidden call patterns, so this test only fires on real code (an
    accidental call), not on prose that merely *names* the excluded APIs
    (e.g. this module's own scope docstrings explaining what is excluded)."""
    import re

    without_docstrings = re.sub(r'"""(?:.|\n)*?"""', "", source)
    lines = [
        line for line in without_docstrings.splitlines()
        if not line.strip().startswith("#")
    ]
    return "\n".join(lines)


def test_engine_source_never_references_topology_mutation_entry_points():
    source = _non_docstring_lines(_ENGINE_PATH.read_text())
    hits = [name for name in _FORBIDDEN_CALL_PATTERNS if name in source]
    assert not hits, (
        f"engine.py references topology-mutation entry point(s) {hits!r} -- "
        "GaussianOptimizationEngine must own a fixed-topology numerical "
        "update only (plan section 7.4, 'Explicitly exclude topology "
        "mutation'); insert/prune/grow/relocate/reseed are "
        "GaussianMutationTransaction operations, reached through "
        "GaussianMapBackend.begin_mutation, never through this class."
    )


def test_engine_module_has_no_import_of_mutation_module():
    from eligsir.optimization.gaussian import engine as engine_module

    assert "eligsir.map.mutation" not in engine_module.__dict__
    assert not any(
        getattr(v, "__module__", "") == "eligsir.map.mutation"
        for v in vars(engine_module).values()
    )


torch = pytest.importorskip("torch")

CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import gsplat  # noqa: F401
    GSPLAT_AVAILABLE = True
except Exception:
    GSPLAT_AVAILABLE = False


@pytest.mark.skipif(
    not (CUDA_AVAILABLE and GSPLAT_AVAILABLE),
    reason=(
        "requires CUDA/ROCm + gsplat; run inside the training container: "
        "docker compose run --rm train python -m pytest "
        "eligsir/optimization/gaussian/tests/ -q"
    ),
)
def test_run_step_does_not_change_gaussian_count():
    from eligsir.map.adapters.gsplat import GsplatMapBackend
    from eligsir.optimization.gaussian.contracts import (
        EvidenceRequirements, GradientPolicyContext, OptimizationRuntimeContext,
        OptimizationStepRequest, OptimizationView, RandomStateToken,
    )
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine
    from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (
        _bg,
        _build_args,
        _build_camera,
        _build_gaussians,
    )
    args, opt, pipe, _ns = _build_args()
    torch.manual_seed(1234)
    gaussians = _build_gaussians(seed=0)
    gaussians.training_setup(opt)
    cam = _build_camera(seed=1)

    backend = GsplatMapBackend(gaussians)
    engine = GaussianOptimizationEngine(backend)

    n_before = int(gaussians.get_xyz.shape[0])

    context = GradientPolicyContext(
        gaussians=gaussians, args=args, iteration=0, n_frames_ingested=0,
    )

    class _NoopMCMCStrategy:
        """Stub strategy exercising only ``step_pre_backward`` (the sole
        hook ``render_and_backward`` calls) as a no-op -- this test is
        about topology-count invariance across the engine's own
        orchestration, not about the real MCMC strategy's pre-backward
        behavior."""

        def step_pre_backward(self, **_kwargs):
            return None

    strategy = _NoopMCMCStrategy()
    result = engine.run_step(
        OptimizationStepRequest(
            view=OptimizationView(camera=cam, background=_bg()), iteration=0,
            map_version=0, topology_version=0, resolution_bucket="default",
            evidence_requirements=EvidenceRequirements(),
            random_state_token=RandomStateToken.capture(),
        ),
        OptimizationRuntimeContext(
            gaussians=gaussians, args=args, opt=opt, pipe=pipe,
            mcmc_strategy=strategy, gradient_policy_context=context,
        ),
    )

    n_after = int(gaussians.get_xyz.shape[0])
    assert n_after == n_before, (
        "GaussianOptimizationEngine.run_step() changed the Gaussian count "
        f"({n_before} -> {n_after}) -- this engine must never mutate "
        "topology (insert/prune/grow/relocate/reseed)."
    )
    assert result is not None
