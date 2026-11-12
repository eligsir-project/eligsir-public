"""CPU tests for the issue #254 Phase-1 compiler-feasibility probe.

Scope, stated plainly so this file is not over-read: nothing in issue #254
is wired into training. These tests do **not** assert behavior-neutrality
of a compiled production path, because no compiled production path exists.
What they assert is:

1. the probe harness itself runs and reports every region (a broken probe
   would otherwise silently report zero breaks everywhere);
2. ``torch.compile``'d output equals eager output, on CPU, for each region
   the probe compiles — i.e. the probe's own parity claim is checked, not
   asserted;
3. the break classifier separates host-sync/data-dependent causes from
   config-guard/unsupported ones, since that split is the entire analytical
   content of the probe;
4. the two specific host syncs the probe attributes to
   ``_sanitize_gaussian_params`` are still there (a regression tripwire: if
   someone removes them, this test fails loudly and the #254 doc's
   conclusion has to be revisited rather than silently going stale).

Run: ``python -m pytest tests/test_issue254_dynamo_probe.py -q``
(CPU-only; no GPU, no gsplat, no training run needed).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.issue254_dynamo_probe import (  # noqa: E402
    REGIONS,
    classify_break,
    probe_region,
    run_all,
)


@pytest.fixture(scope="module")
def results():
    return run_all()


def test_every_region_is_probed(results):
    assert len(results) == len(REGIONS)
    assert {r.name for r in results} == {r.name for r in REGIONS}


def test_no_region_failed_to_build_or_explain(results):
    """A build/import failure would make the whole probe report vacuous
    zeros, which is exactly how a feasibility study silently becomes
    wrong. Fail loudly instead."""
    broken = {r.name: r.error for r in results if r.graph_count is None}
    assert not broken, f"probe regions failed before measuring: {broken}"


def test_compiled_matches_eager_for_every_region(results):
    """Requirement 2. Any region that compiles at all must produce
    numerically identical results to the eager path on CPU."""
    bad = {
        r.name: (r.parity, r.max_abs_diff, r.error)
        for r in results
        if r.parity != "PASS"
    }
    assert not bad, f"compiled/eager parity not established: {bad}"


def test_break_classifier_separates_host_syncs_from_config_guards():
    assert classify_break(
        "Unsupported Tensor.item() call with capture_scalar_outputs=False"
    ) == "TENSOR_DEPENDENT"
    assert classify_break("Dynamic control flow on a data-dependent value") == (
        "TENSOR_DEPENDENT"
    )
    assert classify_break("Unsupported: some C extension") == "UNSUPPORTED"
    assert classify_break("specialize on constant int") == "CONFIG_GUARD"


def test_production_sanitize_still_has_its_host_syncs(results):
    """Regression tripwire for the #254 doc's headline blocker.

    ``_sanitize_gaussian_params`` runs three times per training iteration
    and costs 6.5-11.6% of wall-clock (W3-56). It is not compilable as
    written because of two host syncs — ``int((_sc > 15.0).sum())`` and
    ``float(gaussians.get_scaling.max())``. If a future change removes
    them, this assertion fails and
    ``docs/architecture/2026-08-13-issue254-phase0-addressable-fraction-and-compiler-probe.md``
    must be re-derived rather than left stale.
    """
    prod = next(r for r in results if r.name == "sanitize_production")
    assert prod.break_classes.get("TENSOR_DEPENDENT", 0) >= 1, prod.break_reasons
    stacks = " ".join(
        frame for br in prod.break_reasons for frame in br.get("user_stack", [])
    )
    assert "loss_diagnostics.py" in stacks, stacks


def test_branch_free_sanitize_is_a_single_graph(results):
    """The W3-34B branch-free rewrite (unwired, parity-tested separately in
    ``tests/test_w3_34b_sanitize_parity.py``) is the direct comparison that
    shows the blocker is the diagnostics, not the numerics: same math, zero
    graph breaks."""
    bf = next(r for r in results if r.name == "sanitize_branch_free")
    assert bf.graph_break_count == 0, bf.break_reasons
    assert bf.graph_count == 1


def test_gradient_policy_and_optimizer_tail_are_compilable(results):
    """The post-backward tail (grad masking + optimizer step) traces into a
    single graph each — i.e. the tail is *not* where the compile blockers
    are, which is the reason the doc's recommendation is DEFER-on-payoff
    rather than "unblock the tail first"."""
    for name in ("active_mask_policy", "optimizer_step_dense_adam"):
        r = next(x for x in results if x.name == name)
        assert r.graph_break_count == 0, (name, r.break_reasons)


def test_probe_region_is_independently_callable():
    """The probe must work one region at a time, so a future session can
    add a region and iterate without re-running the whole matrix."""
    region = next(r for r in REGIONS if r.name == "sanitize_branch_free")
    result = probe_region(region)
    assert result.error is None, result.error
    assert result.parity == "PASS"
