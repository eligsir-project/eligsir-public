"""Issue #254, Phase 1 (feasibility probe): CPU-only ``torch.compile``/dynamo
graph-break inventory over the *compile-addressable* sub-regions of the
canonical multi-view optimization transaction.

What this is
------------
Issue #254 asks whether the canonical optimization transaction

    OptimizationBatch -> batched render -> objective assembly -> backward
                      -> gradient policies -> optimizer update

can be compiled/fused. Its Phase 1 explicitly says to start with
``torch.compile`` on "loss + backward-adjacent tensor logic around the
existing gsplat op", *not* with a bespoke renderer. gsplat therefore stays
an opaque custom-op boundary, and the addressable region is everything
around it.

This script enumerates the real, production functions that make up that
surrounding region, runs ``torch._dynamo.explain`` on each with CPU-safe
synthetic inputs, and classifies every resulting graph break by cause. It
also checks that ``torch.compile``'d output matches eager output for the
regions that compile at all.

What this is NOT
----------------
* It is **not** a performance measurement. Graph-break counts say what a
  compiler *can* fuse; they say nothing about what fusing would save. The
  Phase 0 decision gate (is the addressable fraction big enough to clear
  #254's ~5% end-to-end bar?) is answered from real GPU phase timings, not
  from here — see
  ``docs/architecture/2026-08-13-issue254-phase0-addressable-fraction-and-compiler-probe.md``.
* It is **not** an ROCm result. This runs on CPU. Python-level break causes
  (host syncs, data-dependent branches, ``.item()``) transfer across
  backends unchanged, because they are decided by dynamo before any device
  code is involved. Inductor codegen quality, kernel fusion payoff, and
  ROCm-specific fallbacks do **not** transfer and are not probed here.
* Nothing here is wired into training. No production call site changes.

Run
---
    python scripts/issue254_dynamo_probe.py            # table to stdout
    python scripts/issue254_dynamo_probe.py --json out.json

Inside the container (the repo's own convention for anything that must run
against the project's real pinned torch):

    docker compose run --rm train python scripts/issue254_dynamo_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import torch

# Repo root on sys.path[0]: this script lives in ``scripts/`` and imports
# top-level project packages (``utils``, ``eligsir``). Without this an
# unrelated site-packages ``utils`` shadows the project one.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------
# Break-cause classification
# --------------------------------------------------------------------------
#
# The distinction that matters for #254 is NOT "how many breaks" but "what
# kind":
#
#   * CONFIG_GUARD  — dynamo specialized on a Python/config value (an
#     ``args.*`` flag, a dtype, a shape). These are *not* breaks in the
#     costly sense: they become guards, and a differently-configured run
#     just recompiles once. A flag-heavy objective produces many of these
#     and is still perfectly compilable.
#   * TENSOR_DEPENDENT — control flow that reads a tensor's *value* on the
#     host (``.item()``, ``float(t)``, ``int(t.sum())``, ``if t.any()``,
#     non-finite guards). These are real breaks: they force a device->host
#     sync, split the graph every single iteration, and cannot be guarded
#     away. This is the category that has to shrink before a large fused
#     region exists.
#   * UNSUPPORTED — an op/construct dynamo cannot trace (custom op without
#     a meta kernel, unsupported builtin, etc.).
#   * OTHER — unclassified; inspect the raw reason text.
#
# W3-34A (docs/architecture/2026-08-10-w3-34a-graph-capture-feasibility.md)
# found exactly the TENSOR_DEPENDENT category as the blocker set for
# CUDA/HIP *graph capture* — different mechanism (capture forbids host syncs
# outright), same causes. This probe re-derives that list independently for
# the dynamo/compile path rather than assuming it carries over.

_TENSOR_DEPENDENT_MARKERS = (
    "item()",
    "capture_scalar_outputs",
    "data-dependent",
    "data dependent",
    "Data-dependent",
    "float() argument",
    "bool(Tensor",
    "Dynamic control flow",
    "generic_jump",
)
_CONFIG_GUARD_MARKERS = (
    "guard",
    "specialize",
    "constant",
)
_UNSUPPORTED_MARKERS = (
    "Unsupported",
    "not supported",
    "unsupported",
)


def classify_break(reason: str) -> str:
    """Map a dynamo break-reason string to one of the four buckets above.

    Order matters: a reason can match several markers, and
    ``TENSOR_DEPENDENT`` is the strictest/most consequential reading, so it
    wins. ``UNSUPPORTED`` is checked before ``CONFIG_GUARD`` because nearly
    every dynamo reason string starts with the word "Unsupported".
    """
    text = str(reason)
    if any(m in text for m in _TENSOR_DEPENDENT_MARKERS):
        return "TENSOR_DEPENDENT"
    if any(m in text for m in _UNSUPPORTED_MARKERS):
        return "UNSUPPORTED"
    if any(m in text for m in _CONFIG_GUARD_MARKERS):
        return "CONFIG_GUARD"
    return "OTHER"


# --------------------------------------------------------------------------
# Probe regions
# --------------------------------------------------------------------------


@dataclass
class ProbeRegion:
    """One named candidate region of the optimization transaction.

    ``build`` returns ``(callable, args_factory)``. ``args_factory`` is
    called fresh for every invocation so in-place regions (sanitize, the
    grad-masking policy, the optimizer step) never observe state left
    behind by a previous call — without that, an eager-vs-compiled parity
    check on an in-place function silently compares against already-mutated
    inputs.
    """

    name: str
    transaction_phase: str
    description: str
    build: Callable[[], tuple[Callable[..., Any], Callable[[], tuple]]]
    # Set when the region is genuinely in-place: parity is then checked on
    # the mutated inputs rather than on the return value.
    inplace_state: Callable[[tuple], Any] | None = None


@dataclass
class ProbeResult:
    name: str
    transaction_phase: str
    description: str
    graph_count: int | None = None
    graph_break_count: int | None = None
    op_count: int | None = None
    break_classes: dict = field(default_factory=dict)
    break_reasons: list = field(default_factory=list)
    parity: str = "not_run"
    max_abs_diff: float | None = None
    error: str | None = None


# ---- region builders ------------------------------------------------------


class _ToyGaussians:
    """Minimal legacy-layout Gaussian state with the raw tensors the
    production sanitize/grad-policy code actually reads.

    Deliberately the *legacy* (``_xyz``/``_opacity``/...) attribute names:
    that is what ``_sanitize_gaussian_params`` looks for by name, so a
    gsplat-layout stub would make the probe trivially break-free by simply
    never entering any of the code under test.
    """

    def __init__(self, n: int = 64) -> None:
        g = torch.Generator().manual_seed(254)
        self._xyz = torch.nn.Parameter(torch.randn(n, 3, generator=g))
        self._opacity = torch.nn.Parameter(torch.randn(n, 1, generator=g))
        self._scaling = torch.nn.Parameter(torch.randn(n, 3, generator=g))
        self._rotation = torch.nn.Parameter(torch.randn(n, 4, generator=g))
        self._features_dc = torch.nn.Parameter(torch.randn(n, 1, 3, generator=g))
        self._features_rest = torch.nn.Parameter(torch.randn(n, 15, 3, generator=g))
        self.provisional = torch.zeros(n, dtype=torch.bool)
        params = [
            self._xyz, self._opacity, self._scaling,
            self._rotation, self._features_dc, self._features_rest,
        ]
        # Seeded from the same generator, NOT ``torch.randn_like`` (which
        # draws from global RNG): the eager and compiled arms each call the
        # factory separately, so an unseeded draw would hand them different
        # gradients and make the parity check meaningless.
        for p in params:
            p.grad = torch.randn(p.shape, generator=g)
        # One param group per tensor: the active-mask policy indexes
        # ``group["params"][0]``, matching the real optimizer construction.
        self.optimizer = torch.optim.Adam([{"params": [p]} for p in params], lr=1e-3)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    def clone_state(self) -> dict:
        return {
            k: getattr(self, k).detach().clone()
            for k in ("_xyz", "_opacity", "_scaling", "_rotation",
                      "_features_dc", "_features_rest")
        }

    def clone_grads(self) -> dict:
        return {
            k: getattr(self, k).grad.detach().clone()
            for k in ("_xyz", "_opacity", "_scaling", "_rotation",
                      "_features_dc", "_features_rest")
        }


def _build_photometric_core():
    """The photometric term exactly as ``objective.assemble_loss`` writes it
    (``objective.py`` ~line 266-268), using the same production helpers.

    NOTE (documented limitation, see the module docstring): on CPU,
    ``utils.loss_utils.ssim`` falls back to the reference torch SSIM because
    the ``fused_ssim`` HIP kernel is unavailable. On the real ROCm path the
    fused kernel is used instead and would appear to dynamo as an opaque
    custom-op boundary — i.e. the *production* photometric region is at
    least as fragmented as what this probe measures, never less.
    """
    from eligsir.utils.loss_utils import l1_loss, ssim

    lambda_dssim = 0.2

    def region(image, gt_image):
        ll1 = l1_loss(image, gt_image)
        s = ssim(image, gt_image)
        return (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - s)

    def factory():
        g = torch.Generator().manual_seed(11)
        return (
            torch.rand(3, 32, 32, generator=g),
            torch.rand(3, 32, 32, generator=g),
        )

    return region, factory


def _build_sanitize_production():
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params,
    )

    def region(gaussians):
        return _sanitize_gaussian_params(gaussians, "probe", 0)

    def factory():
        return (_ToyGaussians(),)

    return region, factory


def _build_sanitize_branch_free():
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_branch_free,
    )

    def region(gaussians):
        _sanitize_gaussian_params_branch_free(gaussians)
        return None

    def factory():
        return (_ToyGaussians(),)

    return region, factory


def _build_active_mask_policy():
    from eligsir.optimization.gaussian.contracts import GradientPolicyContext
    from eligsir.optimization.gaussian.gradient_policies.active_mask import (
        ActiveMaskGradientPolicy,
    )

    policy = ActiveMaskGradientPolicy()

    def region(context):
        return policy.apply(context)

    def factory():
        gaussians = _ToyGaussians()
        ctx = GradientPolicyContext(
            gaussians=gaussians, args=object(), iteration=1, n_frames_ingested=1,
        )
        ctx.active_set.sparse_active_set = True
        ctx.active_set.zero_invisible_grads = True
        mask = torch.zeros(gaussians.provisional.numel(), dtype=torch.bool)
        mask[::2] = True
        ctx.active_set.visible_mask = mask
        return (ctx,)

    return region, factory


def _build_optimizer_step():
    """The optimizer tail of the transaction: a dense Adam step over the same
    parameter set. (``selective_adam`` is a custom CUDA op with no CPU
    implementation, so the dense-Adam path is what is probe-able here; per
    issue #228 dense Adam is also a real, supported production
    configuration, not a stand-in.)"""

    def region(gaussians):
        gaussians.optimizer.step()
        return None

    def factory():
        return (_ToyGaussians(),)

    return region, factory


REGIONS: list[ProbeRegion] = [
    ProbeRegion(
        name="photometric_core",
        transaction_phase="objective assembly",
        description="(1-lambda)*L1 + lambda*(1-SSIM), verbatim from objective.assemble_loss",
        build=_build_photometric_core,
    ),
    ProbeRegion(
        name="sanitize_production",
        transaction_phase="pre/post-step safety net",
        description="_sanitize_gaussian_params (pre-W3-61 production, 3 call sites per iteration)",
        build=_build_sanitize_production,
        inplace_state=lambda a: a[0].clone_state(),
    ),
    ProbeRegion(
        name="sanitize_branch_free",
        transaction_phase="pre/post-step safety net",
        description="_sanitize_gaussian_params_branch_free (wired into production by W3-61)",
        build=_build_sanitize_branch_free,
        inplace_state=lambda a: a[0].clone_state(),
    ),
    ProbeRegion(
        name="active_mask_policy",
        transaction_phase="gradient policies",
        description="ActiveMaskGradientPolicy.apply (invisible-row grad zeroing)",
        build=_build_active_mask_policy,
        inplace_state=lambda a: a[0].gaussians.clone_grads(),
    ),
    ProbeRegion(
        name="optimizer_step_dense_adam",
        transaction_phase="optimizer update",
        description="torch.optim.Adam.step over the Gaussian parameter groups",
        build=_build_optimizer_step,
        inplace_state=lambda a: a[0].clone_state(),
    ),
]


# --------------------------------------------------------------------------
# Probe driver
# --------------------------------------------------------------------------


def _max_abs_diff(a: Any, b: Any) -> float:
    """Largest elementwise |a-b| over two arbitrarily-nested comparable
    results. Returns 0.0 for structures with nothing numeric to compare
    (e.g. two ``None`` returns from an in-place region whose real output is
    checked via ``inplace_state`` instead)."""
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float("inf")
        return float((a.float() - b.float()).abs().max()) if a.numel() else 0.0
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max([_max_abs_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max([_max_abs_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, bool) and isinstance(b, bool):
        return 0.0 if a == b else float("inf")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0


def probe_region(region: ProbeRegion, *, atol: float = 1e-6) -> ProbeResult:
    result = ProbeResult(
        name=region.name,
        transaction_phase=region.transaction_phase,
        description=region.description,
    )
    try:
        fn, factory = region.build()
    except Exception as exc:  # pragma: no cover - import/environment guard
        result.error = f"build failed: {type(exc).__name__}: {exc}"
        return result

    import torch._dynamo as dynamo

    # --- 1. graph-break inventory -----------------------------------------
    try:
        dynamo.reset()
        explanation = dynamo.explain(fn)(*factory())
        result.graph_count = int(getattr(explanation, "graph_count", 0) or 0)
        result.graph_break_count = int(
            getattr(explanation, "graph_break_count", 0) or 0
        )
        result.op_count = int(getattr(explanation, "op_count", 0) or 0)
        classes: dict[str, int] = {}
        for raw in getattr(explanation, "break_reasons", []) or []:
            text = str(getattr(raw, "reason", raw))
            bucket = classify_break(text)
            classes[bucket] = classes.get(bucket, 0) + 1
            # First line only: dynamo reasons carry multi-line hints/links
            # that would bury the actual cause in any summary table. The
            # user-stack frames are what make a break actionable (which
            # project file:line caused it), so keep the last few.
            frames = []
            for frame in (getattr(raw, "user_stack", None) or [])[-3:]:
                try:
                    frames.append(f"{frame.filename}:{frame.lineno} in {frame.name}")
                except Exception:
                    frames.append(str(frame))
            result.break_reasons.append(
                {
                    "class": bucket,
                    "reason": text.strip().splitlines()[0],
                    "user_stack": frames,
                }
            )
        result.break_classes = classes
    except Exception as exc:
        result.error = f"explain failed: {type(exc).__name__}: {exc}"
        return result

    # --- 2. compiled-vs-eager parity --------------------------------------
    try:
        dynamo.reset()
        eager_args = factory()
        eager_out = fn(*eager_args)
        eager_state = (
            region.inplace_state(eager_args) if region.inplace_state else None
        )

        dynamo.reset()
        compiled = torch.compile(fn)
        compiled_args = factory()
        compiled_out = compiled(*compiled_args)
        compiled_state = (
            region.inplace_state(compiled_args) if region.inplace_state else None
        )

        diff = _max_abs_diff(eager_out, compiled_out)
        if eager_state is not None:
            diff = max(diff, _max_abs_diff(eager_state, compiled_state))
        result.max_abs_diff = diff
        result.parity = "PASS" if diff <= atol else "FAIL"
    except Exception as exc:
        result.parity = "ERROR"
        result.error = f"compile/parity failed: {type(exc).__name__}: {exc}"
    finally:
        dynamo.reset()
    return result


def run_all(atol: float = 1e-6) -> list[ProbeResult]:
    return [probe_region(r, atol=atol) for r in REGIONS]


def format_table(results: list[ProbeResult]) -> str:
    lines = [
        "| region | transaction phase | graphs | breaks | tensor-dep | other | parity | max|diff| |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in results:
        td = r.break_classes.get("TENSOR_DEPENDENT", 0)
        other = sum(v for k, v in r.break_classes.items() if k != "TENSOR_DEPENDENT")
        diff = "-" if r.max_abs_diff is None else f"{r.max_abs_diff:.2e}"
        lines.append(
            f"| `{r.name}` | {r.transaction_phase} | "
            f"{'-' if r.graph_count is None else r.graph_count} | "
            f"{'-' if r.graph_break_count is None else r.graph_break_count} | "
            f"{td} | {other} | {r.parity} | {diff} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write the full result record (incl. every break reason) here")
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    results = run_all(atol=args.atol)
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print()
    print(format_table(results))
    print()
    for r in results:
        if r.error:
            print(f"[{r.name}] ERROR: {r.error}")
        for br in r.break_reasons:
            print(f"[{r.name}] {br['class']}: {br['reason']}")
            for frame in br.get("user_stack", []):
                print(f"[{r.name}]     at {frame}")

    if args.json_path:
        payload = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "results": [asdict(r) for r in results],
        }
        with open(args.json_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
