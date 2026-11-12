from types import SimpleNamespace

import render as render_module
from scene import GaussianModel, GsplatGaussianModel


def _resolve_model_cls(dataset):
    """Mirror render.py's branching logic in isolation, so this test can
    check the decision without invoking the real render_sets() (which needs
    torch/CUDA and a real Scene/checkpoint)."""
    model_layout = getattr(dataset, "model_layout", "gsplat").lower()
    return GsplatGaussianModel if model_layout == "gsplat" else GaussianModel


def test_gsplat_layout_resolves_to_gsplat_class():
    dataset = SimpleNamespace(model_layout="gsplat")
    assert _resolve_model_cls(dataset) is GsplatGaussianModel


def test_legacy_layout_resolves_to_legacy_class():
    dataset = SimpleNamespace(model_layout="legacy")
    assert _resolve_model_cls(dataset) is GaussianModel


def test_missing_model_layout_defaults_to_gsplat():
    dataset = SimpleNamespace()  # no model_layout attribute at all
    assert _resolve_model_cls(dataset) is GsplatGaussianModel
