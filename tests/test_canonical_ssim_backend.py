from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from eligsir.runtime.production_application import CanonicalMappingRuntime
from eligsir.utils import loss_utils


@pytest.mark.parametrize(
    ("fused_ssim", "compile_loss", "expected"),
    (
        (True, False, "fused"),
        (False, True, "compiled"),
        (False, False, "ref"),
    ),
)
def test_training_ssim_selector_activates_requested_backend(
    monkeypatch, fused_ssim: bool, compile_loss: bool, expected: str
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_set_ssim_mode(mode: str, dynamic: bool = True) -> str:
        calls.append((mode, bool(dynamic)))
        return mode

    monkeypatch.setattr(loss_utils, "set_ssim_mode", fake_set_ssim_mode)

    resolved = loss_utils.configure_training_ssim(
        SimpleNamespace(
            fused_ssim=fused_ssim,
            compile_loss=compile_loss,
            compile_dynamic=False,
        )
    )

    assert resolved == expected
    assert calls == [(expected, False if expected == "compiled" else True)]


def test_canonical_initializer_activates_ssim_after_compile_configuration() -> None:
    source = inspect.getsource(CanonicalMappingRuntime._initialize)
    compile_at = source.index("configure_torch_compile(self._optimization_policy)")
    ssim_at = source.index("configure_training_ssim(self._optimization_policy)")

    assert compile_at < ssim_at
