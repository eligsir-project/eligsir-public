"""Issue #597: compile-mode status must be machine-readable, and must
distinguish intentional `off` from unsupported / failed-and-fell-back.

Before this, `--compile_mode=off` left only a console line, so a finished run
could not be asked which of three states it was in. These tests drive
`configure_torch_compile` with real namespaces and read the resolved block.

Test surface note: this file imports torch (via utils.compiled_kernels) and is
therefore NOT collectable in the CPU-CI image -- see tests/conftest.py's
collect_ignore. It runs on `host` and on the training image.
"""
from __future__ import annotations

import argparse

import pytest

torch = pytest.importorskip("torch")

from eligsir.utils import compiled_kernels as ck  # noqa: E402


def _args(**kw):
    base = dict(
        compile_mode="off",
        compile_dynamic=True,
        compile_after_iter=12000,
        compile_sh=False,
        compile_energy=False,
        compile_utility=False,
        compile_loss=False,
        fused_ssim=True,
        optimizer_type="selective_adam",
        gsplat_sparse_grad=True,
        sh_update_interval=16,
        parallelism_profile="safe",
        sh_impl="gsplat",
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _reset_registry():
    """configure_torch_compile mutates module-global process state."""
    saved_config = ck._COMPILE_CONFIG
    saved_registry = ck.REGISTRY
    ck.REGISTRY = ck._CompiledKernelRegistry()
    ck._COMPILE_CONFIG = None
    yield
    ck.REGISTRY = saved_registry
    ck._COMPILE_CONFIG = saved_config


def test_never_configured_is_its_own_state_not_off():
    block = ck.build_compute_backend_block()
    assert block["configured"] is False
    assert block["compile_status"] == ck.COMPILE_STATUS_NOT_CONFIGURED
    assert block["requested_compile_mode"] is None


def test_explicit_off_is_disabled_by_config_and_records_fused_kernels():
    ck.configure_torch_compile(_args(compile_mode="off"))
    block = ck.build_compute_backend_block()
    assert block["compile_status"] == ck.COMPILE_STATUS_DISABLED_BY_CONFIG
    assert block["requested_compile_mode"] == "off"
    assert block["torch_compilation_active"] is False
    assert block["compile_requested_but_not_active"] is False
    # AC 2: fused/custom kernels are recorded and clearly independent.
    fused = block["fused_kernels"]
    assert fused["ssim_backend_requested"] == "fused"
    assert fused["optimizer_type"] == "selective_adam"
    assert fused["gsplat_sparse_grad"] is True
    assert fused["sh_update_interval"] == 16
    assert fused["parallelism_profile"] == "safe"


def test_off_startup_summary_separates_compilation_from_fused_kernels(capsys):
    ck.configure_torch_compile(_args(compile_mode="off"))
    out = capsys.readouterr().out
    assert "[backend] torch.compile: status=disabled_by_config" in out
    assert "fused/custom kernels (INDEPENDENT of torch.compile)" in out
    assert "UNAFFECTED" in out


def test_requested_but_no_kernel_flags_is_not_the_same_as_off():
    ck.configure_torch_compile(_args(compile_mode="reduce-overhead"))
    block = ck.build_compute_backend_block()
    assert block["compile_status"] == ck.COMPILE_STATUS_REQUESTED_NO_KERNELS
    assert block["compile_requested_but_not_active"] is True
    assert block["resolved_compile_mode"] == "reduce-overhead"


def test_requested_but_compile_after_iter_not_reached():
    ck.configure_torch_compile(
        _args(compile_mode="reduce-overhead", compile_sh=True, compile_after_iter=12000)
    )
    block = ck.build_compute_backend_block()
    assert block["compile_status"] == ck.COMPILE_STATUS_REQUESTED_NOT_YET_COMPILED
    assert block["kernels"]["sh_to_rgb_deg1"]["requested"] is True
    assert block["kernels"]["sh_to_rgb_deg1"]["compiled"] is None


def test_compile_failure_is_loud_and_recorded_not_silent_off(monkeypatch, capsys):
    """AC 4: a requested compile that falls back cannot masquerade as `off`."""

    def _boom(fn, **kw):
        raise RuntimeError("no compiler in this image")

    monkeypatch.setattr(ck.torch, "compile", _boom)
    ck.configure_torch_compile(
        _args(compile_mode="reduce-overhead", compile_sh=True, compile_after_iter=0)
    )
    ck.set_compile_iteration(10)
    ck.REGISTRY._get_compiled("sh_to_rgb_deg1")

    out = capsys.readouterr().out
    assert "WARNING: torch.compile FAILED" in out
    assert "NOT an intentionally-uncompiled run" in out

    block = ck.build_compute_backend_block()
    assert block["compile_status"] == ck.COMPILE_STATUS_COMPILE_FAILED_FALLBACK
    assert block["compile_status"] != ck.COMPILE_STATUS_DISABLED_BY_CONFIG
    assert "no compiler in this image" in (
        block["compile_failures"]["sh_to_rgb_deg1"]["compile_error"]
    )


def test_partial_compile_is_reported_as_partially_compiled(monkeypatch):
    real_names = []

    def _selective(fn, **kw):
        real_names.append(fn.__name__)
        if fn.__name__ == "_sh_to_rgb_deg2":
            raise RuntimeError("graph break")
        return fn.__call__  # a distinct object => "compiled"

    monkeypatch.setattr(ck.torch, "compile", _selective)
    ck.configure_torch_compile(
        _args(compile_mode="default", compile_sh=True, compile_after_iter=0)
    )
    ck.set_compile_iteration(1)
    for name in ("sh_to_rgb_deg1", "sh_to_rgb_deg2", "sh_to_rgb_deg3"):
        ck.REGISTRY._get_compiled(name)

    block = ck.build_compute_backend_block()
    assert block["compile_status"] == ck.COMPILE_STATUS_PARTIALLY_COMPILED
    assert block["torch_compilation_active"] is True
    assert block["kernels"]["sh_to_rgb_deg2"]["compiled"] is False


def test_all_requested_kernels_compiled_is_enabled(monkeypatch):
    monkeypatch.setattr(ck.torch, "compile", lambda fn, **kw: fn.__call__)
    ck.configure_torch_compile(
        _args(compile_mode="default", compile_sh=True, compile_after_iter=0)
    )
    ck.set_compile_iteration(1)
    for name in ("sh_to_rgb_deg1", "sh_to_rgb_deg2", "sh_to_rgb_deg3"):
        ck.REGISTRY._get_compiled(name)
    assert ck.build_compute_backend_block()["compile_status"] == (
        ck.COMPILE_STATUS_ENABLED
    )


def test_runtime_failure_is_recorded_as_a_fallback(monkeypatch):
    def _blows_up_at_call(fn, **kw):
        def _raiser(*a, **k):
            raise RuntimeError("kernel exploded")
        return _raiser

    monkeypatch.setattr(ck.torch, "compile", _blows_up_at_call)
    ck.configure_torch_compile(
        _args(compile_mode="default", compile_energy=True, compile_after_iter=0)
    )
    ck.set_compile_iteration(1)
    opac = torch.rand(8)
    ck.REGISTRY("opacity_entropy_core", opac)  # falls back to eager, succeeds
    block = ck.build_compute_backend_block()
    assert "kernel exploded" in (
        block["compile_failures"]["opacity_entropy_core"]["runtime_error"]
    )
    assert block["compile_status"] == ck.COMPILE_STATUS_COMPILE_FAILED_FALLBACK


def test_unsupported_status_is_defined_but_unreachable_today():
    """#597 asks for `unsupported`; no current path produces it (documented)."""
    cfg = {"requested_compile_mode": "default", "torch_compile_available": False}
    assert ck.resolve_compile_status(cfg, {}) == ck.COMPILE_STATUS_UNSUPPORTED
    assert ck._COMPILE_AVAILABLE is True  # ...hence unreachable in practice


# --------------------------------------------------------------------------
# AC 3: comparison tooling surfaces backend differences
# --------------------------------------------------------------------------

def test_backend_difference_between_two_runs_is_surfaced():
    ck.configure_torch_compile(_args(compile_mode="off"))
    a = ck.build_compute_backend_block()
    ck._COMPILE_CONFIG = None
    ck.REGISTRY = ck._CompiledKernelRegistry()
    ck.configure_torch_compile(
        _args(compile_mode="off", fused_ssim=False, optimizer_type="adam")
    )
    b = ck.build_compute_backend_block()
    diff = ck.compute_backend_differences(a, b)
    assert diff["comparable"] is True
    assert diff["throughput_comparable"] is False
    assert diff["differences"]["fused_kernels.ssim_backend_requested"] == {
        "current": "fused", "baseline": "ref",
    }
    assert "fused_kernels.optimizer_type" in diff["differences"]


def test_identical_backends_compare_clean():
    ck.configure_torch_compile(_args(compile_mode="off"))
    a = ck.build_compute_backend_block()
    diff = ck.compute_backend_differences(a, dict(a))
    assert diff["differences"] == {}
    assert diff["throughput_comparable"] is True


def test_missing_baseline_block_is_reported_not_guessed():
    ck.configure_torch_compile(_args(compile_mode="off"))
    diff = ck.compute_backend_differences(ck.build_compute_backend_block(), None)
    assert diff["comparable"] is False
    assert "compute_backend" in diff["reason"]


# --------------------------------------------------------------------------
# report.json integration (AC 1): the block really lands in the manifest
# --------------------------------------------------------------------------

def test_report_json_carries_the_compute_backend_block(tmp_path, monkeypatch):
    import json
    import sys
    from unittest.mock import MagicMock

    monkeypatch.setitem(sys.modules, "lpipsPyTorch", MagicMock())
    from eligsir.utils.comparison_report import write_post_training_report

    class _FakeGaussian:
        def get_xyz(self):
            return torch.zeros(0, 3)

    ck.configure_torch_compile(_args(compile_mode="off"))
    out = tmp_path / "run"
    out.mkdir()
    summary = write_post_training_report(
        model_path=str(out),
        iteration=100,
        gaussians=_FakeGaussian(),
        train_cams=[],
        test_cams=[],
        render_fn=lambda cam, g, p, b: {"render": torch.zeros(3, 8, 8)},
        pipe=object(),
        background=None,
    )
    assert summary["compute_backend"]["compile_status"] == (
        ck.COMPILE_STATUS_DISABLED_BY_CONFIG
    )
    written = json.loads(
        (out / "comparison" / "iter_100" / "report.json").read_text()
    )
    assert written["compute_backend"]["requested_compile_mode"] == "off"
    assert written["compute_backend"]["fused_kernels"]["optimizer_type"] == (
        "selective_adam"
    )
