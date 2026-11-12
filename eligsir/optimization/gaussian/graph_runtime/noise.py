"""Graph-safe explicit randomness (plan section 17.7, deliverable 3).

Confirmed empirically on this ROCm/gfx1151 PyTorch build (see
``capability.py``'s ``_probe_in_graph_generator_rng`` and
``tests/test_w3_12_graph_runtime_rng.py``): calling
``torch.randn(..., generator=gen)`` *inside* a ``torch.cuda.graph(...)``
capture region raises ``RuntimeError: Attempt to increase offset for a
CUDA generator not in capture mode`` -- i.e. PyTorch's CUDA-generator RNG
op is not graph-safe on this build. This is at least caught cleanly (a
real Python exception, not the renderer's page-fault crash), but it still
means any stochastic term that needs to run *inside* a captured region
(e.g. a future PR that folds MCMC noise injection into the same graph as
the optimizer step) cannot rely on the normal ``torch.randn(generator=...)``
call working there.

**Note on this PR's actual capture scope**: the region
``TorchAcceleratorGraphRuntime`` captures today (see ``runtime.py``'s
module docstring) is gradient-policy masking + optimizer step + zeroing,
called *after* an eager ``backward()`` -- MCMC noise injection
(``mcmc_strategy.inject_noise``) runs later still, outside this capture
region entirely (see ``engine.py``'s ``run_step``). So today's captured
region contains **no RNG calls at all**, and this module is infrastructure
for whenever a future capture-scope expansion needs one, not something
this PR's own capture path currently exercises in production. It is
still a real, tested, working mechanism (deliverable 3 is about the
mechanism being available, explicit, and reproducible -- not about it
being wired into the current capture region, which has no randomness to
make safe).

## The pattern

Pre-generate the random tensor **outside** the graph (host-issued RNG call,
same ``torch.Generator`` every time, explicit seed), copy it into a static
buffer, then let the graph read that static buffer. Reseeding + refreshing
the buffer, then replaying, is fully deterministic and reproducible given
the same seed sequence -- proven in
``tests/test_w3_12_graph_runtime_rng.py``.
"""

from __future__ import annotations


class GraphSafeNoise:
    """Owns one pre-allocated static noise buffer plus the
    ``torch.Generator`` that refreshes it. ``refresh(seed)`` must be
    called (outside any graph capture/replay) before each replay that
    should consume new noise; the buffer is then a valid graph-capturable
    input for that replay.
    """

    def __init__(self, shape, device: str, dtype=None):
        import torch

        self._generator = torch.Generator(device=device)
        self._dtype = dtype or torch.float32
        self.buffer = torch.zeros(shape, device=device, dtype=self._dtype)

    def refresh(self, seed: int) -> None:
        """Reseed and redraw into ``self.buffer`` in place. Must be called
        outside graph capture/replay (it allocates a fresh sample tensor
        internally via ``torch.randn`` before copying into the static
        buffer) -- this is exactly why it cannot be moved inside the
        capture region on this build; see the module docstring.
        """
        import torch

        self._generator.manual_seed(seed)
        sample = torch.randn(
            self.buffer.shape,
            device=self.buffer.device,
            dtype=self._dtype,
            generator=self._generator,
        )
        self.buffer.copy_(sample)
