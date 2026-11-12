import argparse

import scripts.render_view_check as render_view_check
from scene.gaussian_model import GaussianModel as LegacyGaussianModel
from scene.gsplat_model import GsplatGaussianModel


def _build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--index", type=int, default=-1)
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model_layout", default="gsplat", choices=["gsplat", "legacy"])
    return ap


def test_model_layout_defaults_to_gsplat():
    args = _build_parser().parse_args(["--ply", "x.ply", "--cameras", "c.json", "--tag", "t"])
    assert args.model_layout == "gsplat"


def test_model_layout_can_be_set_to_legacy():
    args = _build_parser().parse_args(
        ["--ply", "x.ply", "--cameras", "c.json", "--tag", "t", "--model_layout", "legacy"]
    )
    assert args.model_layout == "legacy"


def test_resolve_gaussian_model_cls_matches_flag():
    assert render_view_check.resolve_gaussian_model_cls("gsplat") is GsplatGaussianModel
    assert render_view_check.resolve_gaussian_model_cls("legacy") is LegacyGaussianModel
