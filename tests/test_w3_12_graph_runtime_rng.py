"""Wave 3, PR W3-12: deliverable 3 -- "randomness is explicit and
reproducible" under graph capture.

Three things are established here, both on real GPU tensors:

1. In-graph ``torch.randn(..., generator=<explicit torch.Generator>)`` is
   confirmed NOT graph-safe on this ROCm build (raises inside capture) --
   already asserted at the capability-probe level
   (``test_w3_12_graph_capability_probe.py``); this file re-derives it
   directly against ``torch.cuda.graph`` (not through the capability
   module) so the finding is not solely dependent on that module's own
   correctness.
2. **Scope check added after review** (an earlier draft's claim was too
   broad): the DEFAULT CUDA generator (no explicit ``generator=`` kwarg)
   IS graph-safe -- it captures, replays, and its state genuinely
   advances across separate ``.replay()`` calls (different random values
   each time). So "in-graph RNG is unsupported" was never the correct
   claim; only "in-graph RNG via an explicit, separately-constructed
   generator" is unsupported here.
3. ``GraphSafeNoise``'s pre-generated-static-buffer pattern IS graph-safe
   and reproducible: same seed sequence -> bit-identical replay outputs,
   different seed -> different outputs. Still the right choice for this
   PR's purposes (an explicitly seeded, independently reproducible noise
   stream, decoupled from the default generator's shared global state),
   despite point 2 -- not because the default generator doesn't work.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian.graph_runtime.noise import GraphSafeNoise  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")

_DEVICE = "cuda"


def test_in_graph_generator_rng_is_not_capturable():
    gen = torch.Generator(device=_DEVICE)
    gen.manual_seed(123)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.randn(64, device=_DEVICE, generator=gen)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError):
        with torch.cuda.graph(graph):
            torch.randn(64, device=_DEVICE, generator=gen)


def test_default_generator_in_graph_rng_is_capturable_and_advances():
    """The scope-narrowing check: without an explicit ``generator=``
    kwarg (i.e. using the default CUDA generator), in-graph RNG DOES
    work on this build -- capture succeeds, and consecutive replays
    produce genuinely different values (proving the generator's state is
    really advancing inside the graph, not just that capture didn't
    raise). This is why the "in-graph RNG is unsupported" claim in an
    earlier draft was too broad.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.randn(64, device=_DEVICE)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = torch.randn(64, device=_DEVICE)

    graph.replay()
    torch.cuda.synchronize()
    first = static_out.clone()

    graph.replay()
    torch.cuda.synchronize()
    second = static_out.clone()

    assert not torch.equal(first, second), (
        "expected the default generator's state to advance across "
        "replays -- if this now fails, default-generator in-graph RNG "
        "may have changed behavior on this build"
    )


def test_graph_safe_noise_reproducible_across_capture_and_replay():
    noise = GraphSafeNoise(shape=(2048,), device=_DEVICE)

    # Warm up + capture a trivial graph that reads the static buffer.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            noise.refresh(seed=1000 + i)
            _ = noise.buffer * 2.0
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = noise.buffer * 2.0

    def replay_with_seed(seed: int) -> torch.Tensor:
        noise.refresh(seed)
        graph.replay()
        torch.cuda.synchronize()
        return static_out.clone()

    a = replay_with_seed(999)
    b = replay_with_seed(999)
    c = replay_with_seed(1000)

    assert torch.equal(a, b), "same seed must reproduce bit-identical output"
    assert not torch.equal(a, c), "different seed must produce different output"


def test_graph_safe_noise_matches_eager_generator_output():
    """The static-buffer value itself (not just downstream graph output)
    must match what a plain eager torch.randn(generator=...) call with
    the same seed produces -- proving the wrapper doesn't silently alter
    the distribution/sequence.
    """
    noise = GraphSafeNoise(shape=(512,), device=_DEVICE)
    noise.refresh(seed=42)

    reference_gen = torch.Generator(device=_DEVICE)
    reference_gen.manual_seed(42)
    reference = torch.randn(512, device=_DEVICE, generator=reference_gen)

    assert torch.equal(noise.buffer, reference)
