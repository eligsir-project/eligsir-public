"""Live, non-destructive graph-capability probe (plan section 17.4).

The plan's ROCm warning (section 2.4, quoted in this workstream's brief) is
explicit: "do not assume ROCm graph support mirrors CUDA's." This module
answers that empirically rather than by assumption for the mechanisms
below. It deliberately does **not** attempt to capture the real
production render path here (that investigation needs a full
``GsplatGaussianModel`` + ``Camera`` fixture heavier than a lightweight
probe function should construct on every call) -- that finding lives in
``tests/test_w3_12_renderer_capture_incompatible.py`` instead. Summary
(see that test and
``docs/architecture/2026-08-09-w3-12-graph-runtime.md`` for the full
writeup, including a correction of an earlier, wrong "hard crash" claim
that was never actually isolated to graph capture): capturing the real
render path raises a clean, catchable ``torch.AcceleratorError: HIP error:
operation not permitted when stream is capturing`` -- root-caused to a
host-tensor-construction call in camera preprocessing, not a ROCm-specific
inability to raise exceptions cleanly. No process crash was found for the
real render path.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphCapabilityReport:
    torch_version: str
    backend: str  # "rocm", "cuda", or "cpu"
    device_name: str | None
    hip_version: str | None
    cuda_version: str | None
    cuda_graph_api_present: bool
    plain_adam_capturable: bool
    selective_adam_capturable: bool | None  # None: gsplat unavailable, not tested
    in_graph_generator_rng_supported: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["notes"] = list(self.notes)
        return d


def _probe_plain_adam_capturable(torch, device: str) -> tuple[bool, str | None]:
    """torch.optim.Adam needs ``capturable=True`` to be graph-safe at all
    (confirmed empirically on this build: without it, capture raises
    ``RuntimeError: Attempting CUDA graph capture of step() for an
    instance of Adam but param_groups' capturable is False`` -- a real,
    catchable, well-behaved error).
    """
    try:
        x = torch.randn(64, 8, device=device)
        lin = torch.nn.Linear(8, 4).to(device)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3, capturable=True)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                lin(x).pow(2).mean().backward()
                opt.step()
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            lin(x).pow(2).mean().backward()
            opt.step()
        g.replay()
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        return False, f"{type(exc).__name__}: {exc}"


def _probe_selective_adam_capturable(torch, device: str) -> tuple[bool | None, str | None]:
    try:
        from gsplat.optimizers import SelectiveAdam
    except Exception:  # noqa: BLE001
        return None, "gsplat.optimizers.SelectiveAdam not importable"

    try:
        n = 256
        param = torch.nn.Parameter(torch.randn(n, 8, device=device))
        opt = SelectiveAdam([param], eps=1e-15, betas=(0.9, 0.999))
        visibility = torch.ones(n, dtype=torch.bool, device=device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                (param**2).sum().backward()
                opt.step(visibility)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            (param**2).sum().backward()
            opt.step(visibility)
        g.replay()
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _probe_in_graph_generator_rng(torch, device: str) -> tuple[bool, str | None]:
    """In-graph ``torch.randn(..., generator=<explicit torch.Generator>)``
    -- the mechanism you would reach for first for graph-safe randomness
    with an independently seeded/reproducible stream. Confirmed on this
    build to fail cleanly (a catchable ``RuntimeError``, not a crash) with
    "Attempt to increase offset for a CUDA generator not in capture mode."

    **Scope, corrected after review**: this is specifically about an
    EXPLICIT, unregistered ``torch.Generator`` instance -- it is NOT a
    claim that in-graph RNG is unsupported in general. The DEFAULT CUDA
    generator (``torch.randn(64, device=device)``, no ``generator=``
    kwarg) DOES capture and replay correctly on this build, and its state
    genuinely advances across replays (two consecutive `.replay()` calls
    produce different values) -- verified directly, not assumed; see
    ``tests/test_w3_12_graph_runtime_rng.py::test_default_generator_in_graph_rng_is_capturable_and_advances``.
    This matches PyTorch's own documented mechanism for captured dropout
    etc., which relies on the default generator being graph-capture-aware
    in a way a bare, separately-constructed ``torch.Generator`` is not.
    :mod:`.noise`'s pre-generated-static-buffer pattern is still the right
    choice for THIS PR's purposes (an explicitly seeded, independently
    reproducible noise stream decoupled from the default generator's
    global, shared state), but "in-graph RNG is unsupported here" would
    overstate the finding -- the narrower, correct claim is "in-graph RNG
    via an explicit, separately-seeded generator is unsupported here."
    """
    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(123)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                torch.randn(64, device=device, generator=gen)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.randn(64, device=device, generator=gen)
        g.replay()
        torch.cuda.synchronize()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def probe_graph_capability(device: str = "cuda") -> GraphCapabilityReport:
    """Run the live, safe (non-renderer) sub-probes and return a report.

    Requires a GPU; callers must guard with
    ``torch.cuda.is_available()`` themselves (matching this repo's
    ``pytest.mark.skipif(not torch.cuda.is_available(), ...)`` convention)
    -- this function does not skip internally, it raises if no device is
    present, since a capability report with no device to probe is not a
    meaningful result.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("probe_graph_capability() requires a CUDA/HIP device")

    is_rocm = getattr(torch.version, "hip", None) is not None
    backend = "rocm" if is_rocm else "cuda"
    device_name = torch.cuda.get_device_name(0)
    hip_version = getattr(torch.version, "hip", None)
    cuda_version = getattr(torch.version, "cuda", None)
    cuda_graph_api_present = hasattr(torch.cuda, "CUDAGraph")

    notes = [
        f"torch {torch.__version__}, backend={backend}, device={device_name}",
    ]

    adam_ok, adam_err = _probe_plain_adam_capturable(torch, device)
    if adam_err:
        notes.append(f"plain_adam_capturable probe error: {adam_err}")

    sel_ok, sel_err = _probe_selective_adam_capturable(torch, device)
    if sel_err:
        notes.append(f"selective_adam_capturable probe note: {sel_err}")

    rng_ok, rng_err = _probe_in_graph_generator_rng(torch, device)
    if rng_err:
        notes.append(f"in_graph_generator_rng probe error: {rng_err}")

    notes.append(
        "renderer (real gaussian_renderer.render path) capture is NOT "
        "probed here (needs a heavier model/camera fixture than this "
        "lightweight probe builds) -- see "
        "tests/test_w3_12_renderer_capture_incompatible.py: capture raises "
        "a clean, catchable torch.AcceleratorError (HIP: operation not "
        "permitted when stream is capturing), root-caused to a host-tensor "
        "construction in camera preprocessing -- not a process crash, and "
        "not a ROCm-specific inability to raise cleanly. See "
        "docs/architecture/2026-08-09-w3-12-graph-runtime.md"
    )

    return GraphCapabilityReport(
        torch_version=torch.__version__,
        backend=backend,
        device_name=device_name,
        hip_version=hip_version,
        cuda_version=cuda_version,
        cuda_graph_api_present=cuda_graph_api_present,
        plain_adam_capturable=adam_ok,
        selective_adam_capturable=sel_ok,
        in_graph_generator_rng_supported=rng_ok,
        notes=tuple(notes),
    )
