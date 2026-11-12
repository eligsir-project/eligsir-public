"""Wave 3, PR W3-12: Subagent D scope -- renderer capture investigation.

**Correction (this file was rewritten after review caught a confound in
its first version):** an earlier draft of this test called
``gsplat.rendering.rasterization`` directly with hand-built synthetic
tensors and observed the whole process crash with a raw GPU page fault.
That crash was investigated further and found to be reproducible with
**zero graph-capture involvement at all** (a single plain, eager
``rasterization(...).sum().backward()`` call with no
``torch.cuda.graph`` in sight also crashed, at some but not all image
resolutions) -- i.e. it was a bug/confound in that standalone
reproduction's hand-built inputs or calling convention, never actually
isolated to graph capture, and NOT the finding it was believed to be. A
real, passing GPU test in this repo
(``eligsir/optimization/gaussian/tests/test_gpu_parity_harness.py``)
and a real training smoke run (``train.py`` on the Mip-NeRF360 "bicycle"
scene, 54k Gaussians, backward every iteration) both confirm the actual
production render path's forward+backward is completely healthy on this
hardware -- so "rasterization crashes the process" was never a safe
general claim.

**This test replaces that draft with a confound-free probe against the
REAL production render path** (``gaussian_renderer.render`` ->
``eligsir.mapping.renderer.gsplat_backend.render``, using a real
``GsplatGaussianModel`` + ``scene.cameras.Camera`` built the same way
``test_gpu_parity_harness.py`` does), and finds a real, DIFFERENT, and
better-behaved result:

Capturing the real render path inside ``torch.cuda.graph(...)`` raises a
clean, catchable ``torch.AcceleratorError: HIP error: operation not
permitted when stream is capturing`` -- not a crash. Root cause
identified precisely: ``_camera_tensors()``
(``eligsir/mapping/renderer/gsplat_backend.py``) falls back to
``K = torch.tensor([[fx, 0, cx], ...], device=device, ...)`` -- a fresh
host-to-device tensor CONSTRUCTION from Python floats on every call, when
the camera object doesn't already carry pre-cached ``_gsplat_viewmat``/
``_gsplat_K``/``_gsplat_camera_center`` device tensors (a dormant fast
path the function already checks for first, but nothing in the codebase
currently populates). This class of host-sync/allocation op is
universally disallowed during CUDA/HIP stream capture -- expected,
correctly-diagnosed behavior, the same class of restriction CUDA itself
documents, not a ROCm-specific defect.

**This retracts the earlier "ROCm crashes uncatchably where CUDA would
raise cleanly" claim.** On this actual, confound-free probe, ROCm/HIP
behaved exactly as expected: a real illegal-during-capture operation
raised a normal, catchable exception. See
``docs/architecture/2026-08-09-w3-12-graph-runtime.md`` for the full,
corrected writeup, including a second probe (pre-caching the camera
tensors to route around ``_camera_tensors``'s fallback) that still hits
the SAME class of clean exception -- but now, per a manually-captured
traceback, at a genuinely DIFFERENT, deeper site: inside gsplat's own
native kernel call (``gsplat.cuda._wrapper.fully_fused_projection`` ->
``_FullyFusedProjectionPacked.forward`` -> the
``projection_ewa_3dgs_packed_fwd`` HIP kernel), not still in
``_camera_tensors``. So pre-caching the camera tensors genuinely fixed
the first obstacle; render capture is blocked by a second, distinct
obstacle inside gsplat's packed-mode projection kernel itself. Every
obstacle found so far has been a normal, catchable exception, never a
crash.

GPU-required (skipped without one).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

try:
    import gsplat  # noqa: F401

    GSPLAT_AVAILABLE = True
except Exception:  # noqa: BLE001
    GSPLAT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and GSPLAT_AVAILABLE),
    reason="needs CUDA/ROCm + gsplat",
)


def test_real_render_path_capture_raises_clean_catchable_exception():
    from gaussian_renderer import render

    from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (
        _build_camera,
        _build_gaussians,
    )

    g = _build_gaussians(seed=0)
    cam = _build_camera(image_size=64, seed=1)
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    def fwd():
        return render(cam, g, _pipe(), bg, update_sh_rest=True, render_depth=False)["render"]

    # Sanity: the real render path works completely fine eagerly (proves
    # any failure below is capture-specific, not a fixture problem).
    out = fwd()
    out.sum().backward()
    for p in g.params.values():
        p.grad = None
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out = fwd()
            out.sum().backward()
            for p in g.params.values():
                p.grad = None
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - exact type asserted below
        with torch.cuda.graph(graph):
            static_out = fwd()
            static_out.sum().backward()

    # The exact, real, catchable failure -- not a crash, not a bare
    # `Exception`. If this ever changes (e.g. a PyTorch/gsplat upgrade
    # fixes camera-tensor caching or the deeper obstacle), this assertion
    # should fail loudly rather than silently pass on a different error.
    assert "stream is capturing" in str(excinfo.value), (
        f"expected the known stream-capture violation, got: {excinfo.value!r}"
    )


def test_precaching_camera_tensors_still_hits_a_capture_violation_one_level_deeper():
    """Routing around ``_camera_tensors``'s host torch.tensor(...)
    fallback (by pre-populating the dormant ``_gsplat_viewmat``/
    ``_gsplat_K``/``_gsplat_camera_center`` cache attributes) does NOT
    make render capturable -- there is at least one more capture
    violation somewhere deeper in the render call. Still a clean,
    catchable exception, not a crash -- consistent with the previous
    test, and evidence that render capture needs more than just camera-
    tensor caching to become viable, without pretending to know exactly
    how much more.
    """
    from gaussian_renderer import render
    from eligsir.mapping.renderer.gsplat_backend import _camera_tensors
    from eligsir.optimization.gaussian.tests.test_gpu_parity_harness import (
        _build_camera,
        _build_gaussians,
    )

    g = _build_gaussians(seed=0)
    cam = _build_camera(image_size=64, seed=1)
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

    viewmat, K, cam_center = _camera_tensors(cam, "cuda")
    cam._gsplat_viewmat = viewmat.clone()
    cam._gsplat_K = K.clone()
    cam._gsplat_camera_center = cam_center.clone()

    def fwd():
        return render(cam, g, _pipe(), bg, update_sh_rest=True, render_depth=False)["render"]

    out = fwd()
    out.sum().backward()
    for p in g.params.values():
        p.grad = None
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            out = fwd()
            out.sum().backward()
            for p in g.params.values():
                p.grad = None
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with pytest.raises(Exception) as excinfo:  # noqa: PT011
        with torch.cuda.graph(graph):
            static_out = fwd()
            static_out.sum().backward()

    assert "stream is capturing" in str(excinfo.value), (
        f"expected a stream-capture violation even with camera tensors "
        f"pre-cached, got: {excinfo.value!r}"
    )
    # The traceback (captured manually during this investigation, not
    # asserted on here since exact frame text is fragile across gsplat
    # versions) pins the site precisely: it is NOT still in
    # _camera_tensors -- it is now inside gsplat's own native kernel call,
    # gsplat.cuda._wrapper.fully_fused_projection ->
    # _FullyFusedProjectionPacked.forward -> the
    # "projection_ewa_3dgs_packed_fwd" HIP kernel. Pre-caching the camera
    # tensors genuinely fixed the FIRST obstacle; this is a real, distinct
    # SECOND obstacle inside gsplat's packed-mode projection kernel
    # itself (consistent with a data-dependent tile-intersection-sizing
    # host sync, though that mechanism is not independently confirmed
    # here -- only the call site is).


def _pipe():
    from types import SimpleNamespace

    return SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
