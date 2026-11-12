"""Parity/gradient/static-regression tests for ``GuardedLossAccumulator``
(issue #1670, Part B: "one B4 numerical-status/accounting record").

Part A of #1670 (the sync-free dense-masked ``active_reg_core_masked``
baseline) is already merged and covered by
``tests/test_active_reg_core_masked_parity.py``; this file is unrelated to
that and covers only the NEW consolidation mechanism.

Before this change, ``objective.assemble_loss`` resolved each guarded
term's finite predicate immediately via ``bool(torch.isfinite(term))`` --
one independent GPU->CPU host sync per term, up to ~6-10+ times per view.
``GuardedLossAccumulator`` defers every one of those decisions (queued via
``add()``, which never syncs) to a single ``resolve()`` call that performs
at most one host materialization (one ``.tolist()``) for the whole batch of
queued terms, while preserving:

- exact drop-and-continue semantics (a non-finite term contributes nothing,
  not even a zero-valued graph node wired to its own producing ops);
- exact reporting (``_nan_term_report`` called once per dropped *named*
  term, and not at all for silently-dropped terms, matching the pre-change
  per-call-site behavior);
- exact gradient-zero-on-drop (verified via ``.grad``, not just value);
- byte-identical final loss value relative to sequential per-term
  resolution (the old ``_add_guarded`` chaining), for a fixed queue order.
"""
from __future__ import annotations

import ast

import torch

from eligsir.mapping.integration import loss_diagnostics
from eligsir.mapping.integration.loss_diagnostics import (
    GuardedLossAccumulator,
    _add_guarded,
)

_OBJECTIVE_REL_PATH = "eligsir/optimization/gaussian/objective.py"
_DIAGNOSTICS_REL_PATH = "eligsir/mapping/integration/loss_diagnostics.py"


def _old_sequential_resolution(loss, terms, iteration, reports):
    """Reference behavior: sequential ``_add_guarded`` chaining, exactly as
    ``objective.assemble_loss`` did before #1670 Part B. ``terms`` is a list
    of ``(value_tensor, name)``. ``reports`` collects the names
    ``_nan_term_report`` would have been called with, via monkeypatch."""
    for value, name in terms:
        loss = _add_guarded(loss, value, name, iteration)
    return loss


def test_parity_all_finite_matches_old_sequential_path(monkeypatch):
    reports_old: list[str] = []
    reports_new: list[str] = []
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: reports_old.append(name))

    base = torch.tensor(1.5)
    terms = [
        (torch.tensor(0.25), "a"),
        (torch.tensor(0.5), "b"),
        (torch.tensor(-0.75), "c"),
    ]
    old_loss = _old_sequential_resolution(base.clone(), terms, iteration=0, reports=reports_old)

    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: reports_new.append(name))
    acc = GuardedLossAccumulator()
    for value, name in terms:
        acc.add(value, name)
    new_loss, new_finite = acc.resolve(base.clone(), iteration=0)

    assert torch.equal(old_loss, new_loss)
    assert new_finite is True
    assert reports_old == [] and reports_new == []


def test_parity_some_nonfinite_matches_old_sequential_path_and_reports(monkeypatch):
    reports_old: list[str] = []
    base = torch.tensor(2.0)
    terms = [
        (torch.tensor(0.25), "finite_a"),
        (torch.tensor(float("nan")), "nan_term"),
        (torch.tensor(0.5), "finite_b"),
        (torch.tensor(float("inf")), "inf_term"),
        (torch.tensor(-0.25), "finite_c"),
    ]
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: reports_old.append(name))
    old_loss = _old_sequential_resolution(base.clone(), terms, iteration=7, reports=reports_old)

    reports_new: list[str] = []
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: reports_new.append(name))
    acc = GuardedLossAccumulator()
    for value, name in terms:
        acc.add(value, name)
    new_loss, new_finite = acc.resolve(base.clone(), iteration=7)

    assert torch.equal(old_loss, new_loss)
    assert new_finite is True  # base + finite terms only is still finite
    assert reports_old == ["nan_term", "inf_term"]
    assert reports_new == ["nan_term", "inf_term"]


def test_final_total_nonfinite_is_detected_without_any_finite_named_term_dropped():
    """A term individually finite can still make the OVERALL total
    non-finite (e.g. inf - inf upstream isn't modeled here, but a huge
    finite value plus base can overflow to inf). The single boundary must
    catch that too, exactly like the old final `bool(torch.isfinite(loss))`
    gate did."""
    base = torch.tensor(float("inf"))
    acc = GuardedLossAccumulator()
    acc.add(torch.tensor(1.0), "finite_term")
    loss, finite = acc.resolve(base, iteration=0)
    assert finite is False
    assert not torch.isfinite(loss)


def test_silent_drop_when_name_is_none(monkeypatch):
    """Matches the appearance-only/soft-anchor call sites, which never
    called `_nan_term_report` on a non-finite drop."""
    reports: list[str] = []
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: reports.append(name))
    acc = GuardedLossAccumulator()
    acc.add(torch.tensor(float("nan")), name=None)
    loss, finite = acc.resolve(torch.tensor(1.0), iteration=0)
    assert torch.equal(loss, torch.tensor(1.0))  # dropped, base unchanged
    assert finite is True
    assert reports == []


def test_check_term_split_guards_on_raw_term_but_adds_weighted_value(monkeypatch):
    """Mirrors the depth/free-space call sites: the finite check runs on the
    unweighted term, but the weighted term is what gets added."""
    reports: list[str] = []
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: reports.append(name))
    acc = GuardedLossAccumulator()
    raw = torch.tensor(3.0)
    acc.add(2.0 * raw, "depth_loss", check_term=raw)
    loss, finite = acc.resolve(torch.tensor(0.0), iteration=0)
    assert finite is True
    assert torch.equal(loss, torch.tensor(6.0))
    assert reports == []

    reports.clear()
    acc2 = GuardedLossAccumulator()
    bad_raw = torch.tensor(float("nan"))
    acc2.add(2.0 * bad_raw, "depth_loss", check_term=bad_raw)
    loss2, finite2 = acc2.resolve(torch.tensor(5.0), iteration=0)
    assert finite2 is True
    assert torch.equal(loss2, torch.tensor(5.0))  # dropped, base unchanged
    assert reports == ["depth_loss"]


def test_on_resolved_callback_fires_with_correct_flag_and_in_queue_order():
    calls: list[tuple[str, bool]] = []
    acc = GuardedLossAccumulator()
    acc.add(torch.tensor(1.0), "kept", on_resolved=lambda ok: calls.append(("kept", ok)))
    acc.add(torch.tensor(float("nan")), "dropped", on_resolved=lambda ok: calls.append(("dropped", ok)))
    acc.add(torch.tensor(2.0), "kept2", on_resolved=lambda ok: calls.append(("kept2", ok)))
    acc.resolve(torch.tensor(0.0), iteration=0)
    assert calls == [("kept", True), ("dropped", False), ("kept2", True)]


def test_none_term_is_a_no_op():
    acc = GuardedLossAccumulator()
    acc.add(None, "should_not_appear")
    loss, finite = acc.resolve(torch.tensor(3.0), iteration=0)
    assert torch.equal(loss, torch.tensor(3.0))
    assert finite is True


def test_empty_queue_still_gates_on_base_loss_finiteness():
    acc = GuardedLossAccumulator()
    loss, finite = acc.resolve(torch.tensor(float("inf")), iteration=0)
    assert finite is False
    loss2, finite2 = acc.resolve(torch.tensor(1.0), iteration=0)
    assert finite2 is True


# --------------------------------------------------------------------------
# Issue #1591: the exact accepted depth-valid-sample count folded into
# `resolve()`'s single materialization instead of its own separate D2H.
# --------------------------------------------------------------------------


def test_depth_valid_sample_count_is_none_when_never_queued():
    """A view with no depth loss (or that never sets
    `depth_valid_sample_count_device`) must not fabricate a count -- the
    caller (objective.assemble_loss) relies on `None` to mean "leave the
    existing default (0) alone", exactly like the pre-#1591 `is not None`
    guard did for the bare device-tensor local it replaces."""
    acc = GuardedLossAccumulator()
    acc.add(torch.tensor(1.0), "some_term")
    loss, finite = acc.resolve(torch.tensor(0.0), iteration=0)
    assert finite is True
    assert acc.resolved_depth_valid_sample_count is None


def test_depth_valid_sample_count_exact_when_gating_term_finite():
    """A finite depth term (B1: one view, no rejection) contributes its
    exact accepted count."""
    acc = GuardedLossAccumulator()
    acc.depth_valid_sample_count_device = torch.tensor(4321, dtype=torch.int64)
    acc.add(
        torch.tensor(0.75),
        "depth_loss",
        gates_depth_valid_sample_count=True,
    )
    loss, finite = acc.resolve(torch.tensor(1.0), iteration=0)
    assert finite is True
    assert torch.equal(loss, torch.tensor(1.75))
    assert acc.resolved_depth_valid_sample_count == 4321


def test_depth_valid_sample_count_is_exact_zero_for_empty_accepted_mask():
    """An empty accepted mask (`valid.count_nonzero()` == 0, e.g. a frame
    with no valid sensor depth pixels at all) is a completely different
    case from rejection: the depth term itself can still be finite (a
    zero-safe masked loss, per #1672), and the exact count must come back
    as precisely 0, not None and not skipped."""
    acc = GuardedLossAccumulator()
    acc.depth_valid_sample_count_device = torch.tensor(0, dtype=torch.int64)
    acc.add(
        torch.tensor(0.0),
        "depth_loss",
        gates_depth_valid_sample_count=True,
    )
    loss, finite = acc.resolve(torch.tensor(2.0), iteration=0)
    assert finite is True
    assert acc.resolved_depth_valid_sample_count == 0


def test_depth_valid_sample_count_is_zeroed_when_gating_term_is_rejected(monkeypatch):
    """Issue #1591's own explicit requirement: 'a rejected/non-finite depth
    term contributes exactly zero'. The gating decision must be the SAME
    finite decision the depth term's own add-or-drop uses -- not a second,
    independently-timed check -- so construct a case where the queued
    count is large and positive but the gating term is non-finite, and
    confirm the returned count is exactly 0 (not the large positive
    count, and not None)."""
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: None)
    acc = GuardedLossAccumulator()
    acc.depth_valid_sample_count_device = torch.tensor(999_999, dtype=torch.int64)
    acc.add(
        torch.tensor(float("nan")),
        "depth_loss",
        gates_depth_valid_sample_count=True,
    )
    loss, finite = acc.resolve(torch.tensor(3.0), iteration=0)
    assert finite is True  # the drop-and-continue base loss stays finite
    assert torch.equal(loss, torch.tensor(3.0))  # dropped, base unchanged
    assert acc.resolved_depth_valid_sample_count == 0


def test_depth_valid_sample_count_only_the_gating_term_counts_other_terms_dont():
    """Only the term flagged `gates_depth_valid_sample_count=True` decides
    the count -- an unrelated non-finite term queued alongside it must not
    zero the count, and an unrelated finite term must not be mistaken for
    the gate either."""
    acc = GuardedLossAccumulator()
    acc.add(torch.tensor(float("nan")), "unrelated_bad")  # dropped, unrelated
    acc.depth_valid_sample_count_device = torch.tensor(42, dtype=torch.int64)
    acc.add(
        torch.tensor(1.0),
        "depth_loss",
        gates_depth_valid_sample_count=True,
    )
    acc.add(torch.tensor(2.0), "unrelated_good")
    loss, finite = acc.resolve(torch.tensor(0.0), iteration=0)
    assert finite is True
    assert acc.resolved_depth_valid_sample_count == 42


def test_depth_valid_sample_count_shares_the_single_tolist_materialization():
    """The whole point of #1591: the depth count must ride in the SAME
    host round trip as the finite predicates, not cause a second one.
    Count real `Tensor.tolist()` invocations via monkeypatch (not AST --
    this is the runtime-behavior counterpart to the static AST guards
    below) across a `resolve()` call that queues a depth count alongside
    several other guarded terms."""
    calls = {"tolist": 0}
    real_tolist = torch.Tensor.tolist

    def _counting_tolist(self):
        calls["tolist"] += 1
        return real_tolist(self)

    import unittest.mock as mock

    acc = GuardedLossAccumulator()
    acc.add(torch.tensor(1.0), "a")
    acc.add(torch.tensor(2.0), "b")
    acc.depth_valid_sample_count_device = torch.tensor(7, dtype=torch.int64)
    acc.add(torch.tensor(3.0), "depth_loss", gates_depth_valid_sample_count=True)
    acc.add(torch.tensor(4.0), "c")

    with mock.patch.object(torch.Tensor, "tolist", _counting_tolist):
        loss, finite = acc.resolve(torch.tensor(0.0), iteration=0)

    assert finite is True
    assert acc.resolved_depth_valid_sample_count == 7
    assert calls["tolist"] == 1, (
        f"expected exactly one .tolist() host materialization for the whole "
        f"accounting boundary (finite predicates + depth count together), "
        f"got {calls['tolist']}"
    )


def test_b4_style_repeated_resolve_calls_each_produce_exact_per_view_counts_to_sum():
    """True-C B4 aggregation (confirmed by reading
    `true_batch_executor._aggregate_loss`) happens ONE LEVEL UP from
    `assemble_loss`/`GuardedLossAccumulator`: each view still gets its own
    fresh `GuardedLossAccumulator` and its own `resolve()` call (one
    accounting boundary per view, each already a single `.tolist()`), and
    the per-view exact host ints returned from each are summed in plain
    Python by the caller (`sum(b.depth_valid_sample_count for b in
    breakdowns)`). This test pins that each of 4 simulated per-view
    accumulators (a stand-in for one B4) returns its own exact count
    (including a rejected view contributing exactly 0), and that summing
    those already-exact per-view results reproduces the expected B4
    total -- the "aggregate exact sum once" the issue's checklist asks
    for, at the granularity this architecture actually supports without
    the larger true_batch_executor/engine restructuring the issue's own
    session-tracking comments (see issue #1591) found to be out of scope
    for a session-sized change.
    """
    per_view_counts_device = [5, 101, 0, 37]
    per_view_finite = [True, True, True, False]  # view index 3 is rejected

    resolved = []
    for count, is_finite in zip(per_view_counts_device, per_view_finite):
        acc = GuardedLossAccumulator()
        acc.depth_valid_sample_count_device = torch.tensor(count, dtype=torch.int64)
        term = torch.tensor(1.0) if is_finite else torch.tensor(float("nan"))
        acc.add(term, "depth_loss", gates_depth_valid_sample_count=True)
        _loss, _finite = acc.resolve(torch.tensor(0.0), iteration=0)
        resolved.append(acc.resolved_depth_valid_sample_count)

    assert resolved == [5, 101, 0, 0]
    assert sum(resolved) == 106


# --------------------------------------------------------------------------
# `add_unconditional`: interleaved unconditional additions (H9 anchor,
# mature-anchor in objective.assemble_loss) must reproduce the ORIGINAL
# bit-exact summation order, not just the same set of terms summed in some
# other (mathematically equivalent but not bit-identical) order.
# --------------------------------------------------------------------------


def test_unconditional_term_is_always_added_never_reported(monkeypatch):
    reports: list[str] = []
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: reports.append(name))
    acc = GuardedLossAccumulator()
    acc.add_unconditional(torch.tensor(2.0))
    loss, finite = acc.resolve(torch.tensor(1.0), iteration=0)
    assert torch.equal(loss, torch.tensor(3.0))
    assert finite is True
    assert reports == []


def test_unconditional_term_gradient_flows_through_regardless_of_other_drops():
    p = torch.tensor(4.0, requires_grad=True)
    unconditional = p * 2.0  # d/dp = 2

    acc = GuardedLossAccumulator()
    acc.add(torch.tensor(float("nan")), "bad")  # dropped
    acc.add_unconditional(unconditional)
    loss, finite = acc.resolve(torch.tensor(0.0), iteration=0)
    assert finite is True
    loss.backward()
    assert torch.allclose(p.grad, torch.tensor(2.0))


def test_unconditional_replay_matches_original_interleaved_summation_order_bit_exactly():
    """Reproduces objective.assemble_loss's actual shape: two batches of
    guarded terms with an unconditional addition interleaved between them
    (H9 anchor / mature-anchor sit between the active_reg/depth guards and
    the soft_anchor/free_space/surface/knn_reg guards). The accumulator's
    replay -- guarded-batch-1, then unconditional, then guarded-batch-2,
    all queued and replayed in that exact order -- must equal a literal
    Python transcription of the original interleaved code, not just an
    equal-valued reordering: assert `torch.equal` (bit-exact), not
    `allclose`, using values deliberately chosen so a reordered summation
    would be detectably different (mixing very different magnitudes)."""
    base = torch.tensor(1.0)
    g1 = torch.tensor(1e-8)
    g2 = torch.tensor(1e8)
    unconditional = torch.tensor(0.5)
    g3 = torch.tensor(3.25)
    g4 = torch.tensor(-1e8)

    # Literal transcription of the original interleaved code shape.
    reference = base
    reference = reference + g1
    reference = reference + g2
    reference = reference + unconditional
    reference = reference + g3
    reference = reference + g4

    acc = GuardedLossAccumulator()
    acc.add(g1, "g1")
    acc.add(g2, "g2")
    acc.add_unconditional(unconditional)
    acc.add(g3, "g3")
    acc.add(g4, "g4")
    result, finite = acc.resolve(base, iteration=0)

    assert finite is True
    assert torch.equal(reference, result), (
        f"expected bit-exact match with the original interleaved summation "
        f"order, got reference={reference.item()!r} result={result.item()!r}"
    )


# --------------------------------------------------------------------------
# Gradient tests: a dropped term must contribute EXACTLY zero gradient (not
# merely a masked-to-zero value that still backprops NaN through its own
# producing ops -- see GuardedLossAccumulator's docstring on why
# `torch.where(finite, term, zero)` is unsafe for the real, grad-tracking
# accumulation and is only used, under `torch.no_grad()`, to fold the
# finite check into the single host materialization).
# --------------------------------------------------------------------------


def test_dropped_term_gradient_is_exactly_zero_even_with_blowing_up_local_jacobian():
    """Construct a term whose FORWARD value is inf/nan AND whose local
    backward Jacobian is also inf/nan at that point (division by exactly
    zero) -- the classic case where `torch.where`-based masking leaks NaN
    into the kept parameters via `0 * inf`. The accumulator must not do
    that: `x`'s gradient must come out exactly 0.0, not nan."""
    x = torch.tensor(1.0, requires_grad=True)
    denom = torch.tensor(0.0)  # will NOT be a leaf needing grad; only x is
    bad_term = x / denom  # inf; d(bad_term)/dx = 1/denom = inf too

    y = torch.tensor(2.0, requires_grad=True)
    good_term = y * 3.0

    acc = GuardedLossAccumulator()
    acc.add(bad_term, "bad")
    acc.add(good_term, "good")
    loss, finite = acc.resolve(torch.tensor(0.0), iteration=0)
    assert finite is True
    loss.backward()

    # The dropped term never enters the loss graph at all (drop-and-continue,
    # not masked-to-zero), so `x` -- which is reachable from `loss` only
    # through the dropped `bad_term` -- gets no gradient contribution
    # whatsoever: `x.grad` stays `None`, not a NaN-poisoned or even a
    # zero-valued tensor. `None` and all-zero are the same "no contribution"
    # fact (matching `tests/test_active_reg_core_masked_parity.py`'s own
    # convention); what matters here is specifically that it is NOT `nan`.
    assert x.grad is None
    assert y.grad is not None
    assert torch.allclose(y.grad, torch.tensor(3.0))


def test_kept_term_gradient_matches_hand_computed_reference():
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(5.0, requires_grad=True)
    term_a = a.pow(2)  # d/da = 2a = 4
    term_b = (b * 3.0).sin()  # d/db = 3*cos(3b)

    acc = GuardedLossAccumulator()
    acc.add(term_a, "a_term")
    acc.add(term_b, "b_term")
    loss, finite = acc.resolve(torch.tensor(0.0), iteration=0)
    assert finite is True
    loss.backward()

    assert torch.allclose(a.grad, torch.tensor(4.0))
    assert torch.allclose(b.grad, 3.0 * torch.cos(torch.tensor(15.0)))


def test_gradient_matches_old_sequential_add_guarded_path(monkeypatch):
    """End-to-end: build the SAME set of terms (mix of finite/non-finite)
    on two independent leaf-parameter copies, resolve one via the old
    sequential `_add_guarded` chain and one via the accumulator, and assert
    both the loss value and every surviving leaf's gradient match."""
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", lambda name, it: None)

    def _build(seed: int):
        g = torch.Generator().manual_seed(seed)
        p = torch.rand(4, generator=g) * 2.0 - 1.0
        p.requires_grad_(True)
        return p

    p_old = _build(0)
    p_new = torch.tensor(p_old.detach().tolist(), requires_grad=True)

    def _terms(p):
        finite1 = (p[0] * p[1]).sin()
        nan_term = p[2] / (p[2] - p[2])  # divide by an exact 0 -> inf (or nan if p[2] == 0)
        finite2 = p[3].pow(2) * 0.5
        return [(finite1, "f1"), (nan_term, "bad"), (finite2, "f2")]

    old_loss = _old_sequential_resolution(
        torch.tensor(1.0), _terms(p_old), iteration=0, reports=[]
    )
    old_loss.backward()

    acc = GuardedLossAccumulator()
    for value, name in _terms(p_new):
        acc.add(value, name)
    new_loss, new_finite = acc.resolve(torch.tensor(1.0), iteration=0)
    new_loss.backward()

    assert new_finite is True
    assert torch.allclose(old_loss, new_loss)
    assert torch.allclose(p_old.grad, p_new.grad, equal_nan=True)
    # The dropped term's own parameter (p[2]) must have exactly zero
    # gradient contribution from it in BOTH paths.
    assert p_old.grad[2] == 0.0
    assert p_new.grad[2] == 0.0


# --------------------------------------------------------------------------
# Static hot-path regression: no `.item()`/`.cpu()`/`.tolist()`/bare
# `bool(<tensor>)` inside the per-term accumulation call sites in
# `objective.assemble_loss`, and at most one such construct at the single
# `resolve()` boundary. Uses AST inspection (not import + reflection, and
# not naive textual substring search) so the guard runs without importing
# torch-dependent production modules and cannot be skewed by a docstring or
# comment that happens to mention `.tolist()` (as this very module's
# docstrings do, to explain the mechanism). Mirrors this repo's established
# import-free source-inspection idiom (see `tests/_source_inspection.py`),
# implemented directly here via `ast` since that helper only slices out
# whole function/module sources, not the finer call-site-level detection
# this guard needs.
# --------------------------------------------------------------------------


def _iter_calls(node, *, skip_nested_defs: bool = False):
    """Yield every ``ast.Call`` under ``node``.

    ``skip_nested_defs=True`` does not descend into nested
    ``def``/``lambda`` bodies (e.g. the ``on_resolved`` callbacks
    ``assemble_loss`` defines and hands to ``_guard_acc.add(...)``): those
    bodies are defined lexically before the accounting boundary but only
    ever EXECUTE later, when `GuardedLossAccumulator.resolve()` invokes
    them one at a time after the finite decision for that term is already
    known -- so a host sync inside one of them is not a per-term-loop sync
    at all, and treating it as one would make this guard flag exactly the
    deferred-side-effect pattern issue #1670 Part B asks for."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current is not node and skip_nested_defs and isinstance(
            current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            continue
        if isinstance(current, ast.Call):
            yield current
        stack.extend(ast.iter_child_nodes(current))


def _is_sync_attr_call(call: ast.Call) -> bool:
    """``<expr>.item()`` / ``.cpu()`` / ``.tolist()`` -- an ``ast.Call``
    whose function is an attribute access with one of those names."""
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr in ("item", "cpu", "tolist")
    )


def _is_bare_tensor_bool_call(call: ast.Call) -> bool:
    """``bool(torch.isfinite(...))``-shaped guards specifically -- the exact
    pattern #1670 Part B removes from the per-term loop. Deliberately
    narrow (matches this codebase's actual idiom: gating a value-add
    decision on a freshly host-synced tensor bool) rather than flagging
    every ``bool(...)`` call in the function, which would false-positive on
    unrelated uses (e.g. ``bool(_risk_mask.any())`` gating a cardinality
    branch, not a finite-add decision)."""
    if not (isinstance(call.func, ast.Name) and call.func.id == "bool"):
        return False
    if len(call.args) != 1:
        return False
    inner = call.args[0]
    return (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "isfinite"
    )


def _sync_findings(node, *, before_line: int | None = None, skip_nested_defs: bool = False):
    """Every sync-shaped call (attr sync or bare tensor bool) inside
    ``node``, optionally restricted to before ``before_line`` (exclusive)."""
    findings = []
    for call in _iter_calls(node, skip_nested_defs=skip_nested_defs):
        if before_line is not None and call.lineno >= before_line:
            continue
        if _is_sync_attr_call(call) or _is_bare_tensor_bool_call(call):
            findings.append(call)
    return findings


def _parse_function_node(rel_path: str, func_name: str) -> ast.FunctionDef:
    import pathlib

    path = pathlib.Path(rel_path)
    if not path.is_absolute():
        path = pathlib.Path(__file__).resolve().parent.parent / rel_path
    tree = ast.parse(path.read_text(), filename=str(path))
    matches = [
        n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name
    ]
    assert len(matches) == 1, f"expected exactly one module-level {func_name!r} in {rel_path}"
    return matches[0]


def _parse_class_method_node(rel_path: str, class_name: str, method_name: str) -> ast.FunctionDef:
    import pathlib

    path = pathlib.Path(rel_path)
    if not path.is_absolute():
        path = pathlib.Path(__file__).resolve().parent.parent / rel_path
    tree = ast.parse(path.read_text(), filename=str(path))
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in class_node.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def test_assemble_loss_has_no_per_term_host_sync_before_the_single_boundary():
    func_node = _parse_function_node(_OBJECTIVE_REL_PATH, "prepare_loss")
    boundary_lineno = next(
        node.lineno for node in func_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "finalize"
    )

    # Every guarded-term call site routes through `_guard_acc.add(...)`,
    # which never syncs -- confirm there are several of them (a regression
    # that inlined a guard back as a bare `bool(torch.isfinite(...))` call
    # would still leave `_guard_acc.add` calls for the OTHER terms, so this
    # count check and the zero-findings check below are complementary, not
    # redundant).
    add_call_sites = [
        call for call in _iter_calls(func_node)
        if isinstance(call.func, ast.Attribute)
        and call.func.attr == "add"
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "_guard_acc"
    ]
    assert len(add_call_sites) >= 8, (
        "expected assemble_loss to route every guarded term through "
        "_guard_acc.add(...); found fewer call sites than expected -- did "
        "a guarded term regress to an inline bool(torch.isfinite(...)) "
        "check?"
    )

    # The actual regression guard: no `.item()`/`.cpu()`/`.tolist()` call
    # and no bare `bool(torch.isfinite(...))` call ANYWHERE in the function
    # before the single accounting-boundary statement. (After the boundary,
    # the function's non-finite-loss diagnostic branch legitimately does a
    # couple of its own host syncs -- e.g. `bool(torch.isfinite(image).all())`
    # in the rate-limited `[nan-guard]` print -- but that only ever runs on
    # the rare already-non-finite path, once per skipped step, and is
    # unrelated to any per-term decision; it is deliberately out of scope
    # for this guard, which is why the check is scoped to "before the
    # boundary" rather than "anywhere in the function".)
    findings = _sync_findings(func_node, before_line=boundary_lineno, skip_nested_defs=True)
    assert not findings, (
        "found a host-sync-shaped call "
        f"(.item()/.cpu()/.tolist()/bool(torch.isfinite(...))) at line(s) "
        f"{[f.lineno for f in findings]} in assemble_loss BEFORE the single "
        "_guard_acc.resolve(...) accounting boundary -- issue #1670 Part B "
        "requires every per-term finite decision to go through "
        "GuardedLossAccumulator.add()/resolve() instead."
    )


def test_guarded_loss_accumulator_resolve_is_the_only_sync_boundary():
    for method in ("add", "add_unconditional", "status_tensor"):
        node = _parse_class_method_node(_DIAGNOSTICS_REL_PATH, "GuardedLossAccumulator", method)
        assert not _sync_findings(node), method
    node = _parse_class_method_node(_DIAGNOSTICS_REL_PATH, "GuardedLossAccumulator", "resolve")
    findings = _sync_findings(node)
    assert len(findings) == 1 and findings[0].func.attr == "tolist"


def test_static_guard_actually_fails_against_the_pre_change_add_guarded_function():
    """Sanity-check the discriminating power of the two guards above: run
    the SAME bare-`bool(torch.isfinite(...))` check against `_add_guarded`
    (the pre-#1670-Part-B function, left untouched on purpose -- see
    loss_diagnostics.py's module docstring reasoning and objective.py's
    dead train_streaming.py callers) and confirm it DOES flag the pattern
    the new call sites no longer contain. If this test ever started
    passing (i.e. `_add_guarded`'s own source stopped matching), the guards
    above would not be proving anything."""
    add_guarded_node = _parse_function_node(_DIAGNOSTICS_REL_PATH, "_add_guarded")
    assert any(_is_bare_tensor_bool_call(c) for c in _iter_calls(add_guarded_node)), (
        "expected the untouched, pre-existing _add_guarded() to still "
        "contain its original bool(torch.isfinite(term)) host sync -- "
        "this is the exact pattern issue #1670 Part B removes from "
        "objective.assemble_loss's guarded call sites; if _add_guarded no "
        "longer matches, the two guards above are not exercising a real "
        "regression detector any more."
    )


# --------------------------------------------------------------------------
# Issue #1591 static regression: no per-view `.item()`/`.cpu()`/`.tolist()`
# on a depth-sample-count-shaped expression anywhere in `assemble_loss`,
# outside `GuardedLossAccumulator.resolve()`'s own single materialization
# (which lives in a different module, loss_diagnostics.py, and is not
# scanned by this guard at all). Deliberately a SEPARATE, narrower check
# from `test_assemble_loss_has_no_per_term_host_sync_before_the_single_boundary`
# above: that guard is scoped to "before the boundary" (because a
# legitimate already-non-finite diagnostic sync is allowed after it, e.g.
# the `[nan-guard]` print's `bool(torch.isfinite(image).all())`); this one
# instead targets the exact depth-cardinality D2H pattern #1591 removes
# (`int(_depth_valid_sample_count_device.item())`) by matching on the
# receiver expression's text, anywhere in the function -- so a regression
# that reintroduces that specific call AFTER the boundary (which the
# pre-boundary-only guard would not catch) is still caught here.
# --------------------------------------------------------------------------


def _depth_cardinality_sync_findings(node):
    findings = []
    for call in _iter_calls(node, skip_nested_defs=True):
        if not _is_sync_attr_call(call):
            continue
        receiver_src = ast.unparse(call.func.value).lower()
        if "depth" in receiver_src and ("count" in receiver_src or "sample" in receiver_src):
            findings.append(call)
    return findings


def test_assemble_loss_has_no_depth_cardinality_sync_outside_guard_resolve():
    func_node = _parse_function_node(_OBJECTIVE_REL_PATH, "prepare_loss")
    findings = _depth_cardinality_sync_findings(func_node)
    assert not findings, (
        "found a host-sync call (.item()/.cpu()/.tolist()) on a "
        "depth-sample-count-shaped expression at line(s) "
        f"{[f.lineno for f in findings]} in assemble_loss -- issue #1591 "
        "requires the exact accepted depth-valid-sample count to be "
        "drained solely via GuardedLossAccumulator.resolve()'s single "
        "materialization (in loss_diagnostics.py), not a separate "
        "objective.py-local .item()/.cpu()/.tolist() call."
    )


def test_depth_cardinality_static_guard_flags_the_pre_1591_pattern():
    """Sanity-check the guard's discriminating power directly against the
    exact pre-#1591 source line it must catch
    (`int(_depth_valid_sample_count_device.item())`), independent of
    whether objective.py still contains it -- mirrors
    `test_static_guard_actually_fails_against_the_pre_change_add_guarded_function`'s
    role for the #1670 guards above."""
    src = "def f():\n    x = int(_depth_valid_sample_count_device.item())\n"
    node = ast.parse(src).body[0]
    findings = _depth_cardinality_sync_findings(node)
    assert findings, (
        "expected the depth-cardinality static guard to flag the exact "
        "pre-#1591 `.item()` pattern on `_depth_valid_sample_count_device` "
        "-- if it doesn't, the guard above is not exercising a real "
        "regression detector."
    )


# --------------------------------------------------------------------------
# batched_status_packet: render-group-level aggregation
# --------------------------------------------------------------------------


def test_batched_status_packet_parity_four_accumulators_mixed_terms():
    """Parity test: 4 accumulators with mixed term counts (0, 1, 3, 5 guarded),
    one with unconditional term, one with depth gate whose count is finite,
    one with depth gate whose gating term is NaN, one with NaN check term but
    finite value, one whose base loss is inf.

    Assert batched_status_packet output == torch.cat([acc.status_tensor(l) for acc, l in entries])
    """
    # Accumulator 0: 0 guarded terms (empty)
    acc0 = GuardedLossAccumulator()
    loss0 = torch.tensor(1.0)

    # Accumulator 1: 1 guarded term + unconditional
    acc1 = GuardedLossAccumulator()
    acc1.add(torch.tensor(0.5), "term1a")
    acc1.add_unconditional(torch.tensor(0.25))
    loss1 = torch.tensor(0.5)

    # Accumulator 2: 3 guarded terms, one with depth gate (finite)
    acc2 = GuardedLossAccumulator()
    acc2.add(torch.tensor(1.0), "term2a")
    acc2.depth_valid_sample_count_device = torch.tensor(42, dtype=torch.int64)
    acc2.add(torch.tensor(2.0), "term2b", gates_depth_valid_sample_count=True)
    acc2.add(torch.tensor(0.75), "term2c")
    loss2 = torch.tensor(1.5)

    # Accumulator 3: 5 guarded terms, depth gate with NaN check term (rejected)
    acc3 = GuardedLossAccumulator()
    acc3.add(torch.tensor(0.1), "term3a")
    acc3.add(torch.tensor(0.2), "term3b")
    acc3.depth_valid_sample_count_device = torch.tensor(999, dtype=torch.int64)
    acc3.add(torch.tensor(float("nan")), "term3c", gates_depth_valid_sample_count=True)
    acc3.add(torch.tensor(0.3), "term3d")
    acc3.add(torch.tensor(float("nan")), "term3e", check_term=torch.tensor(float("nan")))
    loss3 = torch.tensor(0.8)

    entries = [
        (acc0, loss0),
        (acc1, loss1),
        (acc2, loss2),
        (acc3, loss3),
    ]

    # Build reference (per-view path)
    reference_packets = [acc.status_tensor(loss) for acc, loss in entries]
    reference = torch.cat(reference_packets)

    # Build fresh accumulators for batched path (status_tensor doesn't mutate, but keeping clean)
    acc0_b = GuardedLossAccumulator()

    acc1_b = GuardedLossAccumulator()
    acc1_b.add(torch.tensor(0.5), "term1a")
    acc1_b.add_unconditional(torch.tensor(0.25))

    acc2_b = GuardedLossAccumulator()
    acc2_b.add(torch.tensor(1.0), "term2a")
    acc2_b.depth_valid_sample_count_device = torch.tensor(42, dtype=torch.int64)
    acc2_b.add(torch.tensor(2.0), "term2b", gates_depth_valid_sample_count=True)
    acc2_b.add(torch.tensor(0.75), "term2c")

    acc3_b = GuardedLossAccumulator()
    acc3_b.add(torch.tensor(0.1), "term3a")
    acc3_b.add(torch.tensor(0.2), "term3b")
    acc3_b.depth_valid_sample_count_device = torch.tensor(999, dtype=torch.int64)
    acc3_b.add(torch.tensor(float("nan")), "term3c", gates_depth_valid_sample_count=True)
    acc3_b.add(torch.tensor(0.3), "term3d")
    acc3_b.add(torch.tensor(float("nan")), "term3e", check_term=torch.tensor(float("nan")))

    entries_b = [
        (acc0_b, loss0),
        (acc1_b, loss1),
        (acc2_b, loss2),
        (acc3_b, loss3),
    ]

    # Test batched path
    batched = loss_diagnostics.batched_status_packet(entries_b)

    # Assert byte-identical
    assert torch.equal(batched, reference), (
        f"batched_status_packet output differs from per-view concatenation:\n"
        f"reference: {reference}\n"
        f"batched:   {batched}"
    )


def test_batched_status_packet_resolve_end_to_end(monkeypatch):
    """Feed iter(batched_status_packet(entries).tolist()) to each accumulator's
    resolve and assert the resolved losses and resolved_depth_valid_sample_count
    equal those from per-view path, and rejected names are reported.
    """
    reports_batched: list[str] = []
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report",
                       lambda name, it: reports_batched.append(name))

    # Build accumulators
    acc0 = GuardedLossAccumulator()
    acc0.add(torch.tensor(1.0), "a0")
    loss0 = torch.tensor(0.5)

    acc1 = GuardedLossAccumulator()
    acc1.add(torch.tensor(float("nan")), "bad1")
    acc1.add(torch.tensor(2.0), "good1")
    loss1 = torch.tensor(1.5)

    acc2 = GuardedLossAccumulator()
    acc2.depth_valid_sample_count_device = torch.tensor(100, dtype=torch.int64)
    acc2.add(torch.tensor(0.5), "depth_term", gates_depth_valid_sample_count=True)
    loss2 = torch.tensor(2.0)

    entries = [
        (acc0, loss0),
        (acc1, loss1),
        (acc2, loss2),
    ]

    # Build reference with per-view resolve
    ref_losses = []
    ref_finites = []
    ref_depths = []
    reports_ref: list[str] = []

    def capture_ref_reports(name, it):
        reports_ref.append(name)

    monkeypatch.setattr(loss_diagnostics, "_nan_term_report", capture_ref_reports)
    for acc, loss in entries:
        resolved_loss, is_finite = acc.resolve(loss, iteration=0)
        ref_losses.append(resolved_loss)
        ref_finites.append(is_finite)
        ref_depths.append(acc.resolved_depth_valid_sample_count)

    # Build fresh accumulators for batched path
    acc0_b = GuardedLossAccumulator()
    acc0_b.add(torch.tensor(1.0), "a0")

    acc1_b = GuardedLossAccumulator()
    acc1_b.add(torch.tensor(float("nan")), "bad1")
    acc1_b.add(torch.tensor(2.0), "good1")

    acc2_b = GuardedLossAccumulator()
    acc2_b.depth_valid_sample_count_device = torch.tensor(100, dtype=torch.int64)
    acc2_b.add(torch.tensor(0.5), "depth_term", gates_depth_valid_sample_count=True)

    entries_b = [
        (acc0_b, loss0),
        (acc1_b, loss1),
        (acc2_b, loss2),
    ]

    # Run batched path
    reports_batched.clear()
    monkeypatch.setattr(loss_diagnostics, "_nan_term_report",
                       lambda name, it: reports_batched.append(name))

    batched_packet = loss_diagnostics.batched_status_packet(entries_b)
    status = iter(batched_packet.tolist())

    batched_losses = []
    batched_finites = []
    batched_depths = []
    for acc, loss in entries_b:
        resolved_loss, is_finite = acc.resolve(loss, iteration=0, status=status)
        batched_losses.append(resolved_loss)
        batched_finites.append(is_finite)
        batched_depths.append(acc.resolved_depth_valid_sample_count)

    # Assert all match
    for i, (ref_loss, batch_loss) in enumerate(zip(ref_losses, batched_losses)):
        assert torch.allclose(ref_loss, batch_loss), f"loss mismatch at index {i}"

    assert ref_finites == batched_finites, f"finites mismatch: {ref_finites} vs {batched_finites}"
    assert ref_depths == batched_depths, f"depths mismatch: {ref_depths} vs {batched_depths}"
    assert set(reports_ref) == set(reports_batched), f"reports mismatch: {reports_ref} vs {reports_batched}"


def test_batched_status_packet_launch_bound():
    """Assert batched path calls torch.isfinite at most 3 times (regardless of
    8 views x 5 terms) using monkeypatch-count or torch.profiler.
    """
    import unittest.mock as mock

    isfinite_calls = 0
    real_isfinite = torch.isfinite

    def counting_isfinite(x):
        nonlocal isfinite_calls
        isfinite_calls += 1
        return real_isfinite(x)

    # Build 8 accumulators with 5 terms each
    entries = []
    for i in range(8):
        acc = GuardedLossAccumulator()
        for j in range(5):
            acc.add(torch.tensor(float(j + 1)), f"term_{i}_{j}")
        loss = torch.tensor(float(i + 1) * 0.5)
        entries.append((acc, loss))

    # Monkeypatch and call
    with mock.patch("torch.isfinite", side_effect=counting_isfinite):
        batched_packet = loss_diagnostics.batched_status_packet(entries)

    # The batched implementation should call isfinite at most 3 times:
    # 1. on checks tensor stack
    # 2. on totals tensor
    # 3. (possibly one more for safety)
    assert isfinite_calls <= 3, (
        f"batched_status_packet called torch.isfinite {isfinite_calls} times, "
        f"expected at most 3 regardless of 8 views x 5 terms"
    )


def test_batched_status_packet_empty_group():
    """Empty group (entries=[]) returns an empty int64 tensor."""
    result = loss_diagnostics.batched_status_packet([])
    assert result.dtype == torch.int64
    assert result.shape == (0,)
    assert torch.equal(result, torch.tensor([], dtype=torch.int64))
