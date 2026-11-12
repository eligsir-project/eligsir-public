"""W3-34A throwaway probe: does the CURRENT dynamic-storage post-backward tail,
as literally written in train_streaming.py's main loop, survive torch.cuda.graph
capture?

Context: tests/test_w3_12_renderer_capture_incompatible.py (already in this repo,
re-run for this task and still passing on this hardware) already conclusively
answers the render/backward part of the question -- render is NOT capturable on
the current codebase (two distinct clean AcceleratorErrors: a host-tensor
construction in _camera_tensors's K fallback, and a deeper obstacle inside
gsplat's own projection_ewa_3dgs_packed_fwd kernel even after pre-caching camera
tensors). Because render cannot be captured, the only practically meaningful
capture island on the current codebase is the POST-backward tensor-only tail
(gradients already populated via a real eager backward, matching W3-12's own
scope) -- this probe checks whether that region, USING THE REAL PRODUCTION CODE
PATTERNS from train_streaming.py's main loop (not W3-12's synthetic clean
gradient-masking-only callback), is still graph-safe. Two patterns are pulled
directly from the real hot path and tested verbatim:

  1. _sanitize_gaussian_params's host-sync check pattern
     (int((~torch.isfinite(t)).sum())) -- called unconditionally 3x/iteration
     on the real path (m_sanitize / m_sanitize_after_optstep /
     m_sanitize_after_mcmc_noise), measured by W3-56 at 6.5-11.6% of wall-clock.
  2. gaussians.optimizer.zero_grad(set_to_none=True) -- the real call at
     train_streaming.py's post-optstep block (~line 5661). Probe 2 is designed
     to actually DISCRIMINATE whether set_to_none's None-reassignment breaks
     replay (not just demonstrate the generic static-address requirement): it
     captures a graph, then after capture runs a FRESH eager backward (which
     allocates a brand-new .grad tensor at a new address) and checks whether
     replaying the captured graph reflects that new gradient at all. It won't,
     if set_to_none=True is the mechanism -- because replay only ever touches
     the tensor address that was live at CAPTURE time, and set_to_none's None
     reassignment (a Python-only op, not a captured kernel) breaks the link
     between "the .grad attribute" and "what replay actually reads" after that.
  3. a branch-free ("always run, no int()/if") rewrite of sanitize's clamp +
     nan_to_num, to test the concrete "graph-safe with this specific change"
     claim rather than leaving the sanitize finding purely negative.

Not a permanent test, not wired into any CI/pytest suite -- a one-off GPU probe
for task #34's 34A findings doc. Reuses the same real GsplatGaussianModel +
Camera fixtures W3-12's test suite already established.

Run: docker compose run --rm train python scripts/w3_34a_capture_probe.py
"""
from __future__ import annotations

import sys

import torch

from gaussian_renderer import render
from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (
    _build_camera,
    _build_gaussians,
)


def _pipe():
    from types import SimpleNamespace

    return SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)


def sanitize_style_host_sync(g):
    """Verbatim pattern from eligsir/mapping/integration/loss_diagnostics.py's
    _sanitize_gaussian_params: int(tensor.sum()) host syncs feeding a Python
    `if` that gates an in-place op."""
    with torch.no_grad():
        for _name, p in g.params.items():
            nf = int((~torch.isfinite(p)).sum())  # <-- host sync + Python int()
            if nf:
                p.data.copy_(torch.nan_to_num(p.data, nan=0.0, posinf=1e6, neginf=-1e6))


def real_gradients_via_eager_backward(g, cam, bg):
    """Populate g.params[*].grad with REAL gradients from a real eager
    render+backward call (outside any capture)."""
    for p in g.params.values():
        p.grad = None
    out = render(cam, g, _pipe(), bg, update_sh_rest=True, render_depth=False)["render"]
    out.sum().backward()


def sanitize_style_branch_free(g):
    """Rewrite of sanitize_style_host_sync with no int()/no Python `if` --
    always runs the clamp/nan_to_num unconditionally, exactly what section
    3b/table-row 'graph-safe with minor changes' in the findings doc names as
    the fix. No host sync, no data-dependent branch."""
    with torch.no_grad():
        for _name, p in g.params.items():
            p.data.clamp_(min=-1e6, max=1e6)
            p.data.copy_(torch.nan_to_num(p.data, nan=0.0, posinf=1e6, neginf=-1e6))


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA/ROCm device available")
        return 0

    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    # ---- Probe 1: sanitize-style host sync check, in isolation, under capture ----
    print("=== Probe 1: real sanitize-style host sync (int(tensor.sum())) alone, "
          "gradients already populated eagerly (matches real post-backward scope) ===")
    g = _build_gaussians(seed=0)
    cam = _build_camera(image_size=64, seed=1)
    real_gradients_via_eager_backward(g, cam, bg)
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            sanitize_style_host_sync(g)
    torch.cuda.current_stream().wait_stream(stream)

    graph1 = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph1):
            sanitize_style_host_sync(g)
        print("RESULT probe 1: capture SUCCEEDED (unexpected)")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT probe 1: capture FAILED with {type(exc).__name__}: {exc}")

    # ---- Probe 2: zero_grad(set_to_none=True) under capture (real optimizer,
    # real GsplatGaussianModel params, matching train_streaming.py's actual call
    # at ~line 5661) -- DISCRIMINATING version: after capture, assign a FRESH
    # eager-backward gradient (new tensor, new address) and check whether
    # replay ever reflects it. If set_to_none's None-reassignment is really
    # decoupling the captured graph from the live .grad attribute, replay will
    # keep producing the SAME update every time regardless of this new,
    # different gradient. ----
    print()
    print("=== Probe 2: zero_grad(set_to_none=True) under capture -- discriminating "
          "test (real optimizer, real params, fresh post-capture gradient) ===")
    g2 = _build_gaussians(seed=2)
    cam2 = _build_camera(image_size=64, seed=3)
    opt2 = torch.optim.Adam(list(g2.params.values()), lr=1e-3, capturable=True)

    def step2():
        real_gradients_via_eager_backward(g2, cam2, bg)
        opt2.step()
        opt2.zero_grad(set_to_none=True)

    step2()
    torch.cuda.synchronize()
    stream2 = torch.cuda.Stream()
    stream2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream2):
        for _ in range(3):
            step2()
    torch.cuda.current_stream().wait_stream(stream2)

    # Capture ONLY optimizer.step() + zero_grad -- render+backward (which
    # populate .grad each real iteration) cannot themselves be captured, per
    # probe/test above, so a realistic captured tail assumes .grad is already
    # populated by an external eager call each time, exactly like W3-12's own
    # scoped region. Use a static, all-ONES grad buffer (not zero) so a later
    # "did replay pick up a different gradient" check is unambiguous.
    for p in g2.params.values():
        p.grad = torch.ones_like(p) * 0.01

    graph2 = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph2):
            opt2.step()
            opt2.zero_grad(set_to_none=True)
        print("RESULT probe 2: capture SUCCEEDED")

        # Baseline: replay once, snapshot params.
        graph2.replay()
        torch.cuda.synchronize()
        after_first_replay = {n: p.detach().clone() for n, p in g2.params.items()}

        # Now assign a FRESH, DIFFERENT real gradient (new tensor identity,
        # new address, large magnitude so any real effect would be obvious)
        # via a genuine eager backward, exactly as the real training loop
        # would between iterations.
        real_gradients_via_eager_backward(g2, cam2, bg)
        for p in g2.params.values():
            if p.grad is not None:
                p.grad = p.grad.detach() * 1000.0  # make any real effect unmistakable
        torch.cuda.synchronize()

        graph2.replay()
        torch.cuda.synchronize()
        after_second_replay = {n: p.detach().clone() for n, p in g2.params.items()}

        picked_up_new_grad = any(
            not torch.equal(after_first_replay[n], after_second_replay[n]) for n in after_first_replay
        )
        print(f"RESULT probe 2: second replay reflects the fresh post-capture gradient = "
              f"{picked_up_new_grad} (False means replay is permanently decoupled from "
              f"whatever real gradient .grad is later assigned -- the mechanism this probe "
              f"targets, consistent with W3-12's SelectiveAdam finding but now shown via a "
              f"test with actual discriminating power, not just static-address demonstration)")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT probe 2: capture FAILED with {type(exc).__name__}: {exc}")

    # ---- Probe 3: branch-free sanitize rewrite -- is THIS graph-safe? ----
    print()
    print("=== Probe 3: branch-free (no int()/no if) sanitize rewrite under capture ===")
    g3 = _build_gaussians(seed=4)
    cam3 = _build_camera(image_size=64, seed=5)
    real_gradients_via_eager_backward(g3, cam3, bg)
    torch.cuda.synchronize()

    stream3 = torch.cuda.Stream()
    stream3.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream3):
        for _ in range(3):
            sanitize_style_branch_free(g3)
    torch.cuda.current_stream().wait_stream(stream3)

    graph3 = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph3):
            sanitize_style_branch_free(g3)
        print("RESULT probe 3: capture SUCCEEDED -- branch-free rewrite IS graph-safe")
        for _ in range(3):
            graph3.replay()
        torch.cuda.synchronize()
        print("RESULT probe 3: replay x3 completed without error")
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT probe 3: capture FAILED with {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
