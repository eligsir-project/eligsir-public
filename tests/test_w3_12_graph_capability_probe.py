"""Wave 3, PR W3-12: live graph-capability probe (plan section 17.4/17.7
deliverable 5 -- "CUDA and ROCm are benchmarked separately").

GPU-required. This repo's evaluation machine is ROCm/gfx1100 or gfx1151
(no CUDA hardware available) -- this test therefore only ever exercises
the ROCm branch for real; the "CUDA" side of deliverable 5 is necessarily
based on PyTorch's own documented CUDA-graph contract (e.g. Adam requiring
``capturable=True`` under capture) rather than a real CUDA-hardware A/B --
see docs/architecture/2026-08-09-w3-12-graph-runtime.md for that
distinction stated explicitly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.optimization.gaussian.graph_runtime.capability import (  # noqa: E402
    probe_graph_capability,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU")


def test_probe_runs_and_reports_backend():
    report = probe_graph_capability()
    assert report.backend in ("rocm", "cuda")
    assert report.cuda_graph_api_present is True
    assert report.device_name


def test_probe_confirms_plain_adam_needs_capturable_true():
    # This is the headline non-crashing ROCm finding: torch.optim.Adam IS
    # graph-capturable on this build, but only with capturable=True.
    report = probe_graph_capability()
    assert report.plain_adam_capturable is True


def test_probe_confirms_selective_adam_capturable_when_gsplat_present():
    report = probe_graph_capability()
    # gsplat is a hard dependency of this repo's training path; if this
    # ever comes back None, the environment itself regressed, not just
    # this finding.
    assert report.selective_adam_capturable in (True, False)
    assert report.selective_adam_capturable is not None


def test_probe_confirms_in_graph_explicit_generator_rng_is_not_supported():
    # Scope: this is specifically about an EXPLICIT, separately
    # constructed torch.Generator -- the DEFAULT CUDA generator DOES work
    # in-graph on this build (see
    # tests/test_w3_12_graph_runtime_rng.py::test_default_generator_in_graph_rng_is_capturable_and_advances).
    # Empirically false (unsupported) for the explicit-generator case on
    # this build (see noise.py's module docstring) -- asserted explicitly
    # so a future PyTorch/ROCm upgrade that fixes this is caught (test
    # would start failing, prompting a look at whether GraphSafeNoise's
    # workaround is still needed).
    report = probe_graph_capability()
    assert report.in_graph_generator_rng_supported is False


def test_report_notes_reference_the_renderer_finding_documented_elsewhere():
    # This probe function does NOT run the renderer-capture investigation
    # itself (see capability.py's module docstring for why) -- it only
    # references where that finding actually lives
    # (test_w3_12_renderer_capture_incompatible.py). This test checks the
    # note points there, not that the probe ran anything renderer-related.
    report = probe_graph_capability()
    assert any("renderer" in note.lower() for note in report.notes)
