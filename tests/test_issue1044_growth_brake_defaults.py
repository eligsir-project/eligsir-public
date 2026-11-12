"""Regression guard for the #1044 growth-brake defaults.

Both brakes added for issue #1044 are one edited literal away from silently
changing every streaming run's behavior, and one of them was caught mid-review
shipping a default that would have re-introduced a previously root-caused,
quantified quality regression. These tests pin the two defaults to the values
their evidence supports so a future change has to be deliberate.

Torch-free by construction (argparse + ``arguments/`` only, which imports no
torch), so this runs in the lean CPU CI image rather than being silently
excluded by ``tests/conftest.py`` -- see issue #536.
"""

from __future__ import annotations

import argparse

from arguments import StreamingParams


def _streaming_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = StreamingParams(parser)
    args = parser.parse_args([])
    return group.extract(args)


def test_growth_staleness_brake_defaults_off():
    """``--streaming_growth_staleness_max_iters`` must default to 0 (disabled).

    Default-on suppresses MCMC growth across the ENTIRE refinement tail of
    every streaming run: ``streaming_adaptive_mapping_end_threshold`` (1000)
    declares mapping over after 1000 idle iterations, and
    ``streaming_refinement_iterations`` (50,000) then runs with, by
    construction, zero further depth insertions -- so the brake latches shortly
    after mapping-end and never releases.

    That is precisely the growth-stop-coupled-to-mapping-end behavior the
    2026-07-18 ``[ADAPTIVE-HORIZON]`` correction in ``train_streaming.py``
    removed after real-GPU testing measured it starving floor3 of capacity:
    N frozen at ~13k vs a healthy ~230k, test PSNR 19dB vs 22dB. MCMC growth is
    an ongoing densification mechanism independent of new-frame arrival;
    "mapping ended" is not "growth has saturated".

    Do not flip this default without new real-GPU evidence that growth
    suppression during the refinement tail is harmless on a full-length run.
    """
    args = _streaming_defaults()
    assert args.streaming_growth_staleness_max_iters == 0


def test_growth_density_brake_default_is_the_measured_ceiling():
    """``--streaming_growth_max_gaussians_per_view`` must default to 8000.

    Justified by a measurement over this repo's own run corpus (2026-08-23):
    472 real-holdout ``report.json``s under ``output/``, ratio
    ``geometry.shape.n_total / n_train`` (``n_total``, not ``n_live`` -- the
    brake compares against the full ``gaussians.get_xyz.shape[0]`` population).
    p50 196, p90 1,424, p99 24,806. Only 23 of 472 reports exceed 8000/view and
    all 23 belong to a single known-degenerate first-5%-prefix run.

    Lowering this default would start binding on healthy runs and silently
    break comparability against recorded champion numbers; raising it weakens
    the ceiling. Re-run the corpus measurement before changing it.
    """
    args = _streaming_defaults()
    assert args.streaming_growth_max_gaussians_per_view == 8000


def test_growth_brakes_accept_zero_to_disable():
    """Both brakes accept an explicit 0 and round-trip it through ``extract``.

    SCOPE, deliberately narrow: this checks only that ``--<flag> 0`` parses and
    survives ``extract()`` as integer ``0``. It does NOT verify that
    ``train_streaming.py`` then treats 0 as "disabled" -- that behavior lives in
    the ``if _growth_per_view_cap > 0:`` / ``if _growth_staleness_max_iters >
    0:`` guards inside ``streaming_training``, which this torch-free test
    cannot reach. Documented rather than implied, because a docstring claiming
    coverage it does not have is worse than no test.
    """
    parser = argparse.ArgumentParser()
    group = StreamingParams(parser)
    args = group.extract(
        parser.parse_args(
            [
                "--streaming_growth_max_gaussians_per_view", "0",
                "--streaming_growth_staleness_max_iters", "0",
            ]
        )
    )
    assert args.streaming_growth_max_gaussians_per_view == 0
    assert args.streaming_growth_staleness_max_iters == 0
