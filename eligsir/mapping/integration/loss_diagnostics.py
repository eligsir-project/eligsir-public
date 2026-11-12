"""Small cross-cutting loss/diagnostic helpers, split out of
``rgbd_integrator_impl.py`` (WS-E5) -- see that module's docstring for the
history of why this cluster ended up as one file.

Verbatim relocation, no math/logic change. ``rgbd_integrator_impl.py``
re-exports everything from here so every existing call site keeps working
unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch


def _nan_term_report(name, iteration, _limit=12):
    """Rate-limited notice that a loss term went non-finite and was dropped.
    Keeps the root cause visible (which term / when) without spamming the log."""
    _c = getattr(_nan_term_report, "_counts", None)
    if _c is None:
        _c = {}
        _nan_term_report._counts = _c
    n = _c.get(name, 0) + 1
    _c[name] = n
    if n <= _limit:
        print(f"[loss-guard] step={iteration} dropped non-finite term '{name}' "
              f"(occurrence {n})", flush=True)


def _add_guarded(loss, term, name, iteration):
    """loss + term, but if term is non-finite drop it (detached zero, so no NaN
    gradient) and name it via _nan_term_report. Used to bisect which regularizer
    term poisons the loss when params render finite yet the loss goes NaN."""
    if term is None:
        return loss
    if isinstance(term, float):
        if term != term or term in (float("inf"), float("-inf")):
            _nan_term_report(name, iteration)
            return loss
        return loss + term
    if not bool(torch.isfinite(term)):
        _nan_term_report(name, iteration)
        return loss
    return loss + term


class _PendingGuardedTerm:
    """One queued-but-not-yet-decided entry for :class:`GuardedLossAccumulator`.

    ``check_term``/``value_term`` are split (rather than reusing one tensor
    for both) because a couple of real call sites in
    ``objective.assemble_loss`` guard on the *unweighted* term (e.g.
    ``L_depth``) but add the *weighted* one (``depth_loss_weight * L_depth``)
    -- exactly mirroring what the old inline code did (check ``L_depth``,
    add ``depth_loss_weight * L_depth``).
    """

    __slots__ = ("value_term", "check_term", "name", "on_resolved", "guarded")

    def __init__(self, value_term, check_term, name, on_resolved, guarded=True):
        self.value_term = value_term
        self.check_term = check_term
        self.name = name
        self.on_resolved = on_resolved
        self.guarded = guarded


class GuardedLossAccumulator:
    """Queue guarded/unconditional terms and replay them in exact addition order.

    status_tensor is detached: its forward-only selection cannot connect a
    rejected term's potentially non-finite Jacobian to the live loss graph.
    resolve alone preserves the B1 API; a render group may concatenate status
    tensors and pass one already-read iterator to each resolve in view order.
    The int64 packet also carries the exact accepted depth sample count (#1591).
    """

    __slots__ = (
        "_pending",
        "depth_valid_sample_count_device",
        "_depth_count_gate_index",
        "resolved_depth_valid_sample_count",
    )

    def __init__(self):
        self._pending: list[_PendingGuardedTerm] = []
        self.depth_valid_sample_count_device = None
        self._depth_count_gate_index = None
        self.resolved_depth_valid_sample_count = None

    def add(
        self, value_term, name=None, *, check_term=None, on_resolved=None,
        gates_depth_valid_sample_count=False,
    ):
        """Queue a guarded value; check_term may be its unweighted scalar.

        Callbacks run once, after add/drop, in queue order. Only the designated
        depth term gates depth_valid_sample_count_device; None is a no-op.
        """
        if value_term is None:
            return
        if gates_depth_valid_sample_count:
            self._depth_count_gate_index = len(self._pending)
        self._pending.append(_PendingGuardedTerm(
            value_term, check_term if check_term is not None else value_term,
            name, on_resolved,
        ))

    def add_unconditional(self, value_term):
        """Preserve an unguarded addition at its original position."""
        if value_term is not None:
            self._pending.append(_PendingGuardedTerm(value_term, None, None, None, guarded=False))

    def status_tensor(self, loss):
        """Return guarded flags, final finite flag, and accepted count (-1 if absent)."""
        with torch.no_grad():
            tentative = loss.detach()
            flags = []
            depth_count = torch.full((), -1, dtype=torch.int64, device=loss.device)
            for index, pending in enumerate(self._pending):
                if not pending.guarded:
                    tentative = tentative + pending.value_term.detach()
                    continue
                finite = torch.isfinite(pending.check_term)
                flags.append(finite.reshape(()).to(torch.int64))
                if index == self._depth_count_gate_index and self.depth_valid_sample_count_device is not None:
                    depth_count = torch.where(
                        finite, self.depth_valid_sample_count_device,
                        torch.zeros_like(self.depth_valid_sample_count_device),
                    ).reshape(()).to(torch.int64)
                tentative = tentative + torch.where(
                    finite, pending.value_term.detach(), torch.zeros_like(tentative)
                )
            return torch.stack([*flags, torch.isfinite(tentative).to(torch.int64), depth_count])

    def resolve(self, loss, iteration, *, status=None):
        """Replay accepted additions and callbacks using one local or shared readback.

        Rejected terms never enter the gradient graph (not even via a masked
        zero). An empty term queue uses the same packet, not a separate item().
        """
        if status is None:
            status = iter(self.status_tensor(loss).tolist())
        flags = iter([bool(next(status)) for pending in self._pending if pending.guarded])
        loss_finite = bool(next(status))
        depth_count = int(next(status))
        self.resolved_depth_valid_sample_count = None if depth_count < 0 else depth_count
        for pending in self._pending:
            if not pending.guarded:
                loss = loss + pending.value_term
                continue
            finite = next(flags)
            if finite:
                loss = loss + pending.value_term
            elif pending.name is not None:
                _nan_term_report(pending.name, iteration)
            if pending.on_resolved is not None:
                pending.on_resolved(finite)
        self._pending.clear()
        self.depth_valid_sample_count_device = None
        self._depth_count_gate_index = None
        return loss, loss_finite


def _sanitize_gaussian_params(gaussians, where, iteration, _limit=12):
    """Detect + neutralize non-finite Gaussian state at a PRODUCER site (after an
    optimizer step or an MCMC op). Distinguishes the two failure modes that both
    surface as a NaN in the GLOBAL opacity/scale regularizers while the color
    render stays finite (the offending Gaussian is invisible in the current view):

      (1) exp() overflow: raw ``_scaling`` is finite-but-large (e.g. ~90) so
          ``get_scaling = exp(_scaling)`` -> Inf, and the anisotropy term
          ``s_max/s_min`` -> Inf/Inf = NaN. Fix: clamp raw ``_scaling`` (preventive,
          every call — real Gaussian scales sit far below exp(15)).
      (2) genuine NaN/Inf in a raw parameter (exploding grad / bad MCMC write).
          Fix: nan_to_num the raw tensors in-place.

    Rate-limited report names WHICH raw tensor and WHERE it went bad, so the
    producer is identified instead of guessing at consumers downstream.

    SUPERSEDED IN PRODUCTION as of W3-61 (commit 75284bd, 2026-08-10): no
    production call site calls this any more. ``train_streaming.py`` calls
    ``_sanitize_gaussian_params_sampled`` at all four sites instead, which
    splits this function into an unconditional branch-free numeric core
    plus a sampled diagnostics half. This original is retained as the
    reference implementation that both parity test suites
    (``tests/test_w3_34b_sanitize_parity.py``,
    ``tests/test_issue424_sampled_sanitize_sequence_parity.py``) assert the
    replacement against, and is still used by the W3-34 benchmark and
    issue-#254 dynamo probe scripts as the "before" arm. Do not delete it
    without rehoming those; do not reintroduce it into the training loop
    (it host-syncs ~8x per call -- one ``int()``/``float()`` per checked
    tensor plus the scale-max read -- which is exactly what W3-61 removed
    from the hot path).
    """
    with torch.no_grad():
        # Preventive: keep exp(_scaling) finite. exp(15) ~ 3.3e6; any real scale is
        # orders below this, so a hard clamp here is behaviour-neutral in practice
        # while removing the Inf->NaN anisotropy path entirely.
        if hasattr(gaussians, "_scaling") and gaussians._scaling is not None:
            _sc = gaussians._scaling
            _over = int((_sc > 15.0).sum()) if _sc.numel() else 0
            if _over:
                _sc.data.clamp_(max=15.0)
            _get_sc_max = float(gaussians.get_scaling.max()) if _sc.numel() else 0.0
        else:
            _over, _get_sc_max = 0, 0.0

        _bad = {}
        for _nm in ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest"):
            _t = getattr(gaussians, _nm, None)
            if _t is None:
                continue
            _nf = int((~torch.isfinite(_t)).sum())
            if _nf:
                _bad[_nm] = _nf
                _t.data.copy_(torch.nan_to_num(_t.data, nan=0.0, posinf=1e6, neginf=-1e6))

        if _bad or _over:
            _c = getattr(_sanitize_gaussian_params, "_counts", None)
            if _c is None:
                _c = {}
                _sanitize_gaussian_params._counts = _c
            _k = where
            _n = _c.get(_k, 0) + 1
            _c[_k] = _n
            if _n <= _limit:
                print(f"[param-guard] step={iteration} at '{where}': "
                      f"raw_nonfinite={_bad if _bad else 'none'} "
                      f"scaling_over15={_over} get_scaling_max={_get_sc_max:.3g} "
                      f"(occurrence {_n})", flush=True)
        return bool(_bad or _over)


# Raw Gaussian tensor names sanitize touches, in the exact order the
# production ``_sanitize_gaussian_params`` above visits them. Shared by the
# branch-free rewrite and its diagnostics counterpart below so the two never
# drift apart on which tensors are covered.
_SANITIZE_TENSOR_NAMES = ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest")


def _assert_no_protected_sanitize_proposal(gaussians, protected_mask) -> None:
    """Asynchronously reject and record protected-row correction proposals."""
    if protected_mask is None:
        return
    n = int(gaussians.get_xyz.shape[0])
    if protected_mask.ndim != 1 or protected_mask.shape[0] != n:
        raise RuntimeError(
            f"frozen seed sanitize mask shape mismatch: {tuple(protected_mask.shape)} vs N={n}"
        )
    protected_mask = protected_mask.to(dtype=torch.bool)
    violation = torch.zeros((), dtype=torch.bool, device=protected_mask.device)
    scaling = getattr(gaussians, "_scaling", None)
    if scaling is not None:
        violation = violation | (scaling.detach()[protected_mask] > 15.0).any()
    for name in _SANITIZE_TENSOR_NAMES:
        tensor = getattr(gaussians, name, None)
        if tensor is not None:
            violation = violation | (~torch.isfinite(tensor.detach()[protected_mask])).any()
    previous = getattr(gaussians, "_seed_lineage_sanitize_violations", 0)
    gaussians._seed_lineage_sanitize_violations = (
        previous + violation.to(dtype=torch.int64)
    )
    torch._assert_async(
        ~violation,
        "frozen seed row requires sanitize correction",
    )


def _sanitize_gaussian_params_branch_free(gaussians, protected_mask=None) -> None:
    """W3-34B: capture-safe numeric core of ``_sanitize_gaussian_params``,
    extracted for the three-arm (current-eager / branch-free-eager /
    branch-free-graph) ROI experiment in
    ``docs/architecture/2026-08-10-w3-34b-sanitize-graph-roi.md``.

    WIRED INTO PRODUCTION since W3-61 (commit 75284bd, 2026-08-10) --
    reached from all four ``train_streaming.py`` call sites via
    ``_sanitize_gaussian_params_sampled`` below, which calls this
    unconditionally on every iteration. (This paragraph previously read
    "NOT wired into any production call site", which was true only before
    W3-61 landed; the stale wording outlived the change and directly
    caused issue #424 to be filed against a gap that no longer existed.
    Keep it accurate.)

    Parity-tested by ``tests/test_w3_34b_sanitize_parity.py`` (this
    function vs. ``_sanitize_gaussian_params``, 32 input classes) and, at
    the level production actually calls, by
    ``tests/test_issue424_sampled_sanitize_sequence_parity.py``
    (``_sanitize_gaussian_params_sampled`` vs. ``_sanitize_gaussian_params``
    over randomized iteration sequences). See
    ``docs/architecture/2026-08-10-w3-34b-sanitize-graph-roi.md`` for the
    ROI evidence and ``...-w3-61-sanitize-production-wiring.md`` for the
    wiring + its GPU verification.

    Same numerical intent as the production function above (clamp raw
    ``_scaling`` to keep ``exp(_scaling)`` finite; nan_to_num every raw
    tensor), but with every gate removed: no ``int(tensor.sum())`` host
    sync, no Python ``if`` deciding whether to run the op. Every op runs
    unconditionally on every call.

    Why this is bit-identical to the gated production version on any input
    (proven in the parity test, not just asserted here): both the clamp
    and ``nan_to_num`` are idempotent on values that don't need fixing --
    clamping a value already `<=` the max leaves it unchanged, and
    ``nan_to_num`` passes an already-finite value through unchanged. The
    production ``if`` only ever skips a call whose output would have been
    a no-op anyway, so removing the gate cannot change any resulting
    tensor value -- it can only remove the (rare-branch-taken) skip and
    the diagnostic accounting that decided whether to skip.

    NaN is the one case that argument does NOT cover, and it is worth
    stating explicitly because it is the only input class where the two
    implementations genuinely take different code paths (issue #424
    review, 2026-08-13). ``NaN <= 15.0`` is false, so "already `<=` the
    max" does not apply to it; and production's gate
    ``int((_sc > 15.0).sum())`` also excludes NaN, since every NaN
    comparison is false. So on a ``_scaling`` holding a NaN but no
    value above 15.0, production computes ``_over == 0`` and **skips the
    clamp entirely**, while this function **applies** it. The results are
    still bit-identical -- but for a reason the paragraph above does not
    give: ``torch.clamp`` PROPAGATES NaN rather than clamping it
    (``torch.clamp(nan, max=15.0) -> nan``, verified empirically, not
    inferred from the docs), so the clamp this function applies is a
    value-level no-op there too, and the ``nan_to_num`` that follows then
    maps the NaN to 0.0 identically in both. ``-inf`` reaches the same
    skip-vs-apply split via the same false comparison, but IS covered by
    the original argument (``-inf <= 15.0`` holds), so it needs no
    separate justification.

    Both skip-vs-apply cases are already pinned by
    ``tests/test_w3_34b_sanitize_parity.py::
    test_nonfinite_value_replaced_identically_in_every_tensor``
    (``tensor_name="_scaling"`` with ``bad_value`` NaN / ``-inf``: one
    element is corrupted while the rest stay at the healthy ``-3.0``, so
    nothing exceeds 15.0 and production's gate is genuinely not taken).
    If that clamp ever changes to a form that does not propagate NaN
    (e.g. a ``torch.minimum``/``where``-based rewrite), this equivalence
    breaks and those cases are what will catch it.

    Diagnostic counting/logging is deliberately NOT done here -- that is
    exactly the part of production sanitize this task's benchmark (arm
    A') found is the dominant cost (an unconditional
    ``float(gaussians.get_scaling.max())`` host sync every call, feeding a
    print that fires at most 12 times total). See
    ``_sanitize_gaussian_params_diagnostics_only`` below for a version of
    that reporting meant to run at a sampled interval, decoupled from this
    hot-path numeric core, per 34A's own suggested fix shape.
    """
    with torch.no_grad():
        protected = None
        if protected_mask is not None:
            protected = protected_mask.to(dtype=torch.bool)

        def _write_corrected(tensor, corrected):
            if protected is None:
                tensor.data.copy_(corrected)
                return
            row_mask = protected.reshape(
                protected.shape[0], *([1] * (tensor.ndim - 1))
            )
            tensor.data.copy_(torch.where(row_mask, tensor.data, corrected))

        _sc = getattr(gaussians, "_scaling", None)
        if _sc is not None:
            if protected is None:
                _sc.data.clamp_(max=15.0)
            else:
                _write_corrected(_sc, torch.clamp(_sc.data, max=15.0))
        for _nm in _SANITIZE_TENSOR_NAMES:
            _t = getattr(gaussians, _nm, None)
            if _t is None:
                continue
            _write_corrected(
                _t,
                torch.nan_to_num(_t.data, nan=0.0, posinf=1e6, neginf=-1e6),
            )


def _sanitize_gaussian_params_diagnostics_only(gaussians, where, iteration, _limit=12):
    """W3-34B companion to ``_sanitize_gaussian_params_branch_free``: the
    counting/logging half of production sanitize, split out so it can be
    called at a SAMPLED interval (e.g. every N iterations) rather than
    every call, decoupled from the branch-free numeric core's hot path.

    WIRED INTO PRODUCTION since W3-61 (commit 75284bd, 2026-08-10), but
    only on sampled ticks -- see ``_sanitize_gaussian_params_sampled``
    below for the cadence. (Previously read "Not wired into production";
    stale since W3-61. See issue #424 for what that staleness cost.)

    Not itself graph-capturable (contains the same host syncs as the
    original diagnostic path, by design -- it is meant to run outside any
    captured region).

    MUST REMAIN READ-ONLY. It only counts and prints; it never writes a
    tensor. That is load-bearing, not incidental: the sampled wrapper's
    tensor-state parity with the original ``_sanitize_gaussian_params``
    holds precisely because whether this function ran on a given
    iteration cannot affect any resulting value. Pinned by
    ``tests/test_issue424_sampled_sanitize_sequence_parity.py``'s
    ``test_diagnostics_half_never_writes``.

    Intentionally reproduces the exact accounting production does (same
    tensor names, same >15.0 / non-finite checks, same rate-limited print
    format) so swapping in ``(branch_free_core, this)`` in place of the
    single gated production call preserves the diagnostic's observable
    behavior at whatever sampling cadence the caller chooses -- at cadence
    'every iteration' it reproduces production's print output verbatim
    (only the underlying numeric write already happened unconditionally
    via the branch-free core, not gated on what this function finds).
    """
    with torch.no_grad():
        _sc = getattr(gaussians, "_scaling", None)
        if _sc is not None and _sc.numel():
            _over = int((_sc > 15.0).sum())
            _get_sc_max = float(gaussians.get_scaling.max())
        else:
            _over, _get_sc_max = 0, 0.0

        _bad = {}
        for _nm in _SANITIZE_TENSOR_NAMES:
            _t = getattr(gaussians, _nm, None)
            if _t is None:
                continue
            _nf = int((~torch.isfinite(_t)).sum())
            if _nf:
                _bad[_nm] = _nf

        if _bad or _over:
            _c = getattr(_sanitize_gaussian_params_diagnostics_only, "_counts", None)
            if _c is None:
                _c = {}
                _sanitize_gaussian_params_diagnostics_only._counts = _c
            _k = where
            _n = _c.get(_k, 0) + 1
            _c[_k] = _n
            if _n <= _limit:
                print(f"[param-guard] step={iteration} at '{where}': "
                      f"raw_nonfinite={_bad if _bad else 'none'} "
                      f"scaling_over15={_over} get_scaling_max={_get_sc_max:.3g} "
                      f"(occurrence {_n})", flush=True)
        return bool(_bad or _over)


# W3-61: default cadence for ``_sanitize_gaussian_params_sampled``'s
# diagnostics-only half. Chosen deliberately, not arbitrarily -- see
# docs/architecture/<date>-w3-61-sanitize-production-wiring.md for the full
# reasoning. Two components:
#   - ``_SANITIZE_DIAG_BURN_IN_ITERS``: run the diagnostic EVERY iteration for
#     this many iterations from the start of a run. Real tracked corruption
#     incidents in this codebase (floor2 init-seed scale explosion: bbox
#     jumps 5m->40-48m within the first ~2000 iterations; see CLAUDE.md /
#     the floor2-init-seed-scale-explosion memory) develop early and are
#     sustained, not single-iteration blips -- a burn-in window at full
#     cadence directly covers the one incident class this project has
#     concretely observed, without paying every-iteration diagnostic cost
#     for the entire run.
#   - ``_SANITIZE_DIAG_DEFAULT_INTERVAL``: after burn-in, sample at this
#     interval. Reuses ``args.scalar_log_interval``'s default (100) -- an
#     existing TB-logging cadence convention already used throughout
#     train_streaming.py -- rather than inventing a new magic number.
_SANITIZE_DIAG_BURN_IN_ITERS = 2000
_SANITIZE_DIAG_DEFAULT_INTERVAL = 100


def _sanitize_gaussian_params_sampled(
    gaussians,
    where,
    iteration,
    *,
    diag_interval: int = _SANITIZE_DIAG_DEFAULT_INTERVAL,
    burn_in_iters: int = _SANITIZE_DIAG_BURN_IN_ITERS,
    _limit: int = 12,
    protected_mask=None,
) -> None:
    """W3-61 production call-site wrapper: always runs the branch-free
    numeric core (every call, every iteration -- this is the hot-path fix,
    and it alone means non-finite/exp-overflowing Gaussian state is
    neutralized on the SAME cadence production always had, regardless of
    diagnostic sampling), and calls the diagnostics-only counterpart
    (counting + rate-limited print) either every iteration during an early
    burn-in window or at a sampled interval afterward.

    This is a genuine detection-vs-speed trade, not a free optimization --
    see ``_sanitize_gaussian_params_diagnostics_only``'s docstring and
    docs/architecture/2026-08-10-w3-34b-sanitize-graph-roi.md's "Proposed
    change" section for why sampling loosens visibility into HOW OFTEN
    correction fired, even though the correction itself never becomes
    sampled (that part of production's behavior is fully preserved by the
    unconditional branch-free call below).

    Ordering matters and is deliberate: on a sampled tick, the diagnostics
    scan runs BEFORE the branch-free correction, so it inspects the true
    pre-correction state (what actually needed fixing) rather than the
    already-clean state the correction below would otherwise leave behind
    -- running them in the opposite order would make the diagnostic never
    find anything to report, silently defeating its whole purpose.
    """
    _assert_no_protected_sanitize_proposal(gaussians, protected_mask)
    _interval = max(1, int(diag_interval))
    if iteration <= max(0, int(burn_in_iters)) or iteration % _interval == 0:
        _sanitize_gaussian_params_diagnostics_only(gaussians, where, iteration, _limit=_limit)
    _sanitize_gaussian_params_branch_free(gaussians, protected_mask=protected_mask)


def _admission_quality_weight_for_stats(decision, quality_enabled: bool):
    """Convert a KeyframeAdmitter KeyframeDecision's ``quality`` field into
    the value stored under ``stats["quality_weight"]`` (and, from there,
    persisted into keyframe_admission_log.jsonl).

    ``decision.quality`` is ``None`` whenever quality scoring did not run
    for this frame -- notably the bootstrap-skip case in keyframes.py's
    ``admit()`` (no baseline keyframe yet), and also every cheap-gate
    branch (forced/no_motion/etc.) that returns before reaching the
    quality section at all -- and a real float (possibly 0.0) only when
    scoring genuinely ran. Preserve that distinction (``None`` vs a real
    0.0) rather than coercing one into the other.

    FIXED 2026-07-31: an earlier version of this conversion unconditionally
    called ``float(decision.quality)``, which crashed with
    ``TypeError: float() argument must be a string or a real number, not
    'NoneType'`` on every admit() call that hit the bootstrap-skip path --
    found live on a floor2 verification run (crashed before writing any
    admission-log content at all). Extracted to a standalone function, not
    left inline in the ``_keyframe_admission_closure`` nested closure, so
    it can be exercised directly by a unit test without needing to
    construct the whole streaming_training() closure scope -- see
    tests/test_keyframe_admission.py's
    test_admission_quality_weight_for_stats_handles_none.
    """
    if not quality_enabled:
        return None
    return float(decision.quality) if decision.quality is not None else None


def batched_status_packet(entries: list[tuple[GuardedLossAccumulator, torch.Tensor]]) -> torch.Tensor:
    """One detached int64 packet for a render group, byte-identical in layout and values to
    torch.cat([acc.status_tensor(loss) for acc, loss in entries]), built with O(1) kernel launches
    in the number of views/terms (stack + isfinite + where + index_add, not per-term loops).

    Rules (must hold exactly; the layout is consumed by `resolve` via `next(status)`):
    - Per accumulator i, the packet segment is `[flag_0..flag_{g_i-1}, finite_total_i, depth_count_i]` where
      g_i = number of guarded pending terms, `flag_k = isfinite(check_term_k)`, `finite_total_i =
      isfinite(loss_i.detach() + sum(unconditional value terms) + sum(value_term_k for finite flag_k))`,
      `depth_count_i = -1` if the accumulator has no gating index or `depth_valid_sample_count_device is None`,
      else `depth_valid_sample_count_device` if the gating term is finite, else 0.
    """
    if not entries:
        return torch.tensor([], dtype=torch.int64)

    with torch.no_grad():
        # Phase 1: Gather Python lists across all accumulators
        checks = []  # guarded check_terms
        values = []  # guarded value_terms detached
        seg = []  # accumulator index per guarded term
        base_parts = []  # loss.detach() + unconditional terms, per accumulator

        gate_info = []  # (gate_index, depth_count_device) pairs for accumulators with gates
        gate_positions = []  # indices into checks/values for gating terms
        count_tensors = []  # depth_valid_sample_count_device values

        for acc_idx, (acc, loss) in enumerate(entries):
            # Compute base = loss.detach() + sum(unconditional terms)
            base = loss.detach()
            for pending in acc._pending:
                if not pending.guarded:
                    base = base + pending.value_term.detach()
            base_parts.append(base)

            # Collect guarded terms
            for term_idx, pending in enumerate(acc._pending):
                if not pending.guarded:
                    continue
                checks.append(pending.check_term)
                values.append(pending.value_term.detach())
                seg.append(acc_idx)

                # Track gating term info
                if term_idx == acc._depth_count_gate_index and acc.depth_valid_sample_count_device is not None:
                    gate_positions.append(len(checks) - 1)
                    count_tensors.append(acc.depth_valid_sample_count_device)
                    gate_info.append(acc_idx)

        # Phase 2: Stack checks and compute finite flags, values, and masked values
        if checks:
            C = torch.stack([c.reshape(()) for c in checks])
            F = torch.isfinite(C)
            V = torch.stack(values)
            masked = torch.where(F, V, torch.zeros_like(V))
        else:
            C = None
            F = torch.tensor([], dtype=torch.bool)
            V = None
            masked = None

        # Phase 3: Compute totals and depth counts
        n_acc = len(entries)
        if masked is not None:
            seg_tensor = torch.tensor(seg, dtype=torch.int64, device=masked.device)
            totals = torch.stack(base_parts).index_add(0, seg_tensor, masked)
        else:
            totals = torch.stack(base_parts)

        finite_total = torch.isfinite(totals)

        # Depth counts vector
        depth_vec = torch.full((n_acc,), -1, dtype=torch.int64, device=totals.device)
        if gate_positions:
            gate_pos_tensor = torch.tensor(gate_positions, dtype=torch.int64, device=F.device)
            gate_f = F[gate_pos_tensor]
            count_stack = torch.stack(count_tensors)
            depth_values = torch.where(gate_f, count_stack, torch.zeros_like(count_stack))
            gate_acc_indices = torch.tensor(gate_info, dtype=torch.int64, device=depth_vec.device)
            depth_vec.scatter_(0, gate_acc_indices, depth_values.to(torch.int64))

        # Phase 4: Assemble interleaved packet
        # Build a flat sequence: for each accumulator, append [flags for that acc, finite_total, depth_count]
        packet_parts = []

        if F.numel() > 0:
            F_int64 = F.to(torch.int64)
        finite_total_int64 = finite_total.to(torch.int64)

        current_idx = 0
        for acc_idx, (acc, loss) in enumerate(entries):
            # Count guarded terms for this accumulator
            n_guarded = sum(1 for p in acc._pending if p.guarded)

            # Append flags for this accumulator
            if n_guarded > 0:
                acc_flags = F_int64[current_idx:current_idx + n_guarded]
                packet_parts.append(acc_flags)
                current_idx += n_guarded

            # Append finite_total for this accumulator
            packet_parts.append(finite_total_int64[acc_idx:acc_idx + 1])

            # Append depth_count for this accumulator
            packet_parts.append(depth_vec[acc_idx:acc_idx + 1])

        if packet_parts:
            result = torch.cat(packet_parts)
        else:
            result = torch.tensor([], dtype=torch.int64)

        return result


def _pose_provenance_report_dict(args) -> Optional[dict]:
    result = getattr(args, "pose_provenance_result", None)
    return result.as_dict() if result is not None else None
