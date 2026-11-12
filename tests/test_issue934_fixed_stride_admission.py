"""Tests for issue #934: `fixed_stride` frame admission, the principal
experimental CONTROL arm for comparing `--streaming_frame_admission
hybrid_keyframe`'s smart selector against an explicit fixed-stride policy at
a matched realized admission/work budget.

Two layers, mirroring this repo's existing convention (e.g.
tests/test_active_admission.py for a pure-Python unit-testable object,
tests/test_issue937_admission_log_score_field.py for the source-level AST
guards needed because train_streaming.py's admission dispatch/telemetry
wiring has no injectable seam):

  * `TestFixedStrideAdmitter*` -- real behavioural tests against
    `eligsir.mapping.admission.fixed_stride.FixedStrideAdmitter`, the
    pure, stateless-beyond-a-counter object train_streaming.py's
    `_fixed_stride_admission_closure` bridges to (mirroring
    `KeyframeAdmitter`/`_keyframe_admission_closure`'s split).
  * `Test*Wiring` -- AST-level guards over train_streaming.py and
    arguments/__init__.py, verifying the mode is actually wired into the
    admission dispatch site, the realized-policy fingerprint, and the
    keyframe_admission_log.jsonl sidecar -- and that the existing `all` /
    `hybrid_keyframe` dispatch and fingerprint-recording code is
    byte-for-byte unchanged (the hard "behavior-neutral for existing modes"
    requirement from #934).
"""
from __future__ import annotations

import ast
import os

import pytest

from eligsir.mapping.admission.fixed_stride import (
    FixedStrideAdmitter,
    FixedStrideDecision,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_STREAMING = os.path.join(_REPO_ROOT, "train_streaming.py")
_ARGUMENTS = os.path.join(_REPO_ROOT, "arguments", "__init__.py")


# ---------------------------------------------------------------------------
# Behavioural tests: FixedStrideAdmitter itself
# ---------------------------------------------------------------------------


class TestFixedStrideAdmitterStride:
    def test_admits_every_nth_candidate(self) -> None:
        """Stride N admits exactly candidates 0, N, 2N, ... over a synthetic
        candidate stream -- the core contract of the control arm."""
        n = 4
        admitter = FixedStrideAdmitter(stride_n=n)
        n_candidates = 21
        admitted_ordinals = []
        for _ in range(n_candidates):
            decision = admitter.admit()
            if decision.admitted:
                admitted_ordinals.append(decision.ordinal)
        assert admitted_ordinals == [0, 4, 8, 12, 16, 20]

    def test_first_candidate_always_admitted(self) -> None:
        """Bootstrap needs an initial frame regardless of N."""
        for n in (1, 2, 5, 10, 100):
            admitter = FixedStrideAdmitter(stride_n=n)
            decision = admitter.admit()
            assert decision.admitted is True
            assert decision.ordinal == 0

    def test_stride_one_admits_every_candidate(self) -> None:
        admitter = FixedStrideAdmitter(stride_n=1)
        for _ in range(10):
            assert admitter.admit().admitted is True

    def test_non_admitted_candidates_use_skip_reason(self) -> None:
        admitter = FixedStrideAdmitter(stride_n=3)
        decisions = [admitter.admit() for _ in range(6)]
        for d in decisions:
            if d.admitted:
                assert d.reason == "fixed_stride"
            else:
                assert d.reason == "fixed_stride_skip"
        # Reasons must never collide with hybrid_keyframe's own owner labels
        # (#934's evidence requirement: admission-owner shares must not be
        # misreadable as an unintended hybrid_keyframe mechanism firing).
        reasons = {d.reason for d in decisions}
        assert "max_gap" not in reasons
        assert "forced" not in reasons

    def test_stride_n_clamped_to_at_least_one(self) -> None:
        """A misconfigured N<=0 must not divide by zero or admit nothing."""
        admitter = FixedStrideAdmitter(stride_n=0)
        assert admitter.stride_n == 1
        assert admitter.admit().admitted is True

    def test_n_evaluated_tracks_candidate_count(self) -> None:
        admitter = FixedStrideAdmitter(stride_n=5)
        for i in range(7):
            admitter.admit()
            assert admitter.n_evaluated == i + 1


class TestFixedStrideAdmitterDeterminism:
    def test_deterministic_across_two_identical_runs(self) -> None:
        """Same stride, same candidate count -> identical admit sequence,
        every run -- this is a control arm and must have zero randomness."""

        def run(n_candidates: int, stride: int):
            admitter = FixedStrideAdmitter(stride_n=stride)
            return [admitter.admit().admitted for _ in range(n_candidates)]

        first = run(37, stride=6)
        second = run(37, stride=6)
        assert first == second

    def test_deterministic_regardless_of_frame_content(self) -> None:
        """The admitter takes no frame/cam argument at all -- admit() has
        no channel for frame content to influence the decision, which is
        the structural guarantee behind "information-free"."""
        admitter = FixedStrideAdmitter(stride_n=3)
        # Nothing about "frame content" can be passed in; the signature
        # itself is the proof. Exercise it purely to show no state beyond
        # the ordinal counter affects the outcome.
        results = [admitter.admit().admitted for _ in range(9)]
        assert results == [True, False, False, True, False, False, True, False, False]


class TestFixedStrideDecisionShape:
    def test_decision_is_plain_dataclass(self) -> None:
        d = FixedStrideAdmitter(stride_n=2).admit()
        assert isinstance(d, FixedStrideDecision)
        assert d.stride_n == 2
        assert d.ordinal == 0


# ---------------------------------------------------------------------------
# Source-level AST guards over train_streaming.py / arguments/__init__.py
# ---------------------------------------------------------------------------


def _module_ast(path: str) -> ast.Module:
    with open(path, "r", encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class TestArgumentDeclared:
    def test_streaming_fixed_stride_n_declared_with_default_10(self) -> None:
        src = _source(_ARGUMENTS)
        assert "self.streaming_fixed_stride_n = 10" in src

    def test_streaming_frame_admission_default_still_all(self) -> None:
        """Behavior-neutral guard: adding fixed_stride must not change the
        existing default admission mode."""
        src = _source(_ARGUMENTS)
        assert 'self.streaming_frame_admission = "all"' in src


class TestAdmissionRenderGateNeverInvokedForFixedStride:
    def test_fixed_stride_closure_never_calls_render(self) -> None:
        """The control arm must never invoke the expensive admission
        render/overlap gate -- it is deliberately information-free. Verified
        at the source level: the FixedStrideAdmitter class itself (the sole
        decision engine `_fixed_stride_admission_closure` bridges to) does
        not reference render/pipe/background/renderer identifiers anywhere,
        i.e. it structurally cannot call the renderer -- unlike
        KeyframeAdmitter, which is constructed with a bound `render_fn`."""
        from eligsir.mapping.admission import fixed_stride as fixed_stride_mod

        src = _source(fixed_stride_mod.__file__)
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        forbidden = {"render", "render_fn", "pipe", "background", "renderer"}
        assert not (names & forbidden), f"FixedStrideAdmitter references: {names & forbidden}"
        assert not (attrs & forbidden), f"FixedStrideAdmitter references: {attrs & forbidden}"

    def test_dispatch_site_wires_fixed_stride_with_no_pregate(self) -> None:
        """`admission_fn` gets the fixed_stride closure and `pregate_fn`
        stays None for this mode, matching the docstring's "same candidate
        population" contract (a pregate would filter candidates before they
        reach admission, changing the population)."""
        src = _source(_TRAIN_STREAMING)
        idx = src.index('elif args.streaming_frame_admission == "fixed_stride":')
        block = src[idx: idx + 700]
        assert "admission_fn = _fixed_stride_admission_closure" in block
        # No ASSIGNMENT to pregate_fn in this branch (comments may still
        # mention the name to explain the omission).
        assert "pregate_fn =" not in block


class TestFingerprintAndSidecarParity:
    """Issue #934 telemetry-parity checklist items 2-4."""

    def test_fixed_stride_closure_records_candidate_and_admit_reject(self) -> None:
        src = _source(_TRAIN_STREAMING)
        idx = src.index("def _fixed_stride_admission_closure(cam, frame):")
        end = src.index("\n    def _integrity_observer_closure", idx)
        block = src[idx:end]
        assert "_cadence.record_candidate_evaluated()" in block
        assert "_admission_fingerprint.record_candidate()" in block
        assert "_admission_fingerprint.record_admit(" in block
        assert "_admission_fingerprint.record_reject(" in block

    def test_reason_strings_are_unambiguous_not_reused_from_hybrid_keyframe(self) -> None:
        """#934 explicitly forbids reusing max_gap/forced owner labels. The
        actual reason strings are produced by FixedStrideAdmitter (see
        eligsir/mapping/admission/fixed_stride.py); this checks both
        that module and confirms the closure never overrides them with a
        hybrid_keyframe-style label."""
        from eligsir.mapping.admission import fixed_stride as fixed_stride_mod

        fs_src = _source(fixed_stride_mod.__file__)
        assert '"fixed_stride" if admitted else "fixed_stride_skip"' in fs_src

        train_src = _source(_TRAIN_STREAMING)
        idx = train_src.index("def _fixed_stride_admission_closure(cam, frame):")
        end = train_src.index("\n    def _integrity_observer_closure", idx)
        block = train_src[idx:end]
        # Strip comments/docstrings so only real code is checked -- the
        # docstring legitimately explains the constraint using these words.
        code_only = "\n".join(
            line for line in block.splitlines() if not line.strip().startswith("#")
        )
        assert 'reason="max_gap"' not in code_only
        assert 'reason == "max_gap"' not in code_only
        assert "decision.reason" in block  # reason is threaded through, not hardcoded

    def test_sidecar_whitelist_carries_fixed_stride_only_fields(self) -> None:
        """`candidate_ordinal`/`fixed_stride_n` must reach
        keyframe_admission_log.jsonl, or the mode's own admission-owner
        evidence (which candidate ordinal produced which decision) is lost
        the same way #937/#928 lost `score`/`precedence`."""
        tree = _module_ast(_TRAIN_STREAMING)
        whitelist: set = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            target = node.target
            if not (isinstance(target, ast.Name) and target.id == "_k"):
                continue
            if not isinstance(node.iter, ast.Tuple):
                continue
            keys = {
                e.value
                for e in node.iter.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
            if "quality_weight" in keys:
                whitelist |= keys
        assert {"candidate_ordinal", "fixed_stride_n"} <= whitelist

    def test_sidecar_write_not_gated_on_hybrid_keyframe_mode(self) -> None:
        """The sidecar append block must fire for ANY mode that populates
        `_streaming_admission_stats`, not just hybrid_keyframe -- otherwise
        fixed_stride runs would silently produce no
        keyframe_admission_log.jsonl at all. Guards against the aggregate
        `_admit_cnt` block's `== "hybrid_keyframe"` gate being copy-pasted
        onto the sidecar-write gate."""
        src = _source(_TRAIN_STREAMING)
        gate_idx = src.index('if _admission_stats and bool(getattr(args, "streaming_admission_log_enabled"')
        # The condition itself must not additionally require hybrid_keyframe.
        gate_line_end = src.index("\n", gate_idx)
        gate_line = src[gate_idx:gate_line_end]
        assert "hybrid_keyframe" not in gate_line

    def test_nominal_policy_is_the_raw_arg_not_a_hardcoded_literal(self) -> None:
        """`_admission_fingerprint.nominal_policy` must read straight from
        `args.streaming_frame_admission`, so it reports "fixed_stride"
        automatically without any per-mode special-casing."""
        src = _source(_TRAIN_STREAMING)
        assert (
            "_admission_fingerprint.nominal_policy = str(args.streaming_frame_admission)"
            in src
        )

    def test_report_admission_mode_fields_read_raw_arg(self) -> None:
        """report.json's frame_admission fields must likewise read straight
        through, not enumerate known modes."""
        # #1075 task P3 split these across two files: one read stayed in
        # train_streaming.py, the other moved with the report-kwargs builder
        # into eligsir/runtime/report_kwargs.py. Both must still read the
        # raw arg, so the count is taken over the pair.
        needle = '"frame_admission": args.streaming_frame_admission,'
        total = _source(_TRAIN_STREAMING).count(needle)
        total += _source(
            os.path.join(_REPO_ROOT, "eligsir", "runtime", "report_kwargs.py")
        ).count(needle)
        assert total >= 2


class TestExistingModesUnchanged:
    """Hard requirement: this commit must be provably behavior-neutral for
    `all` and `hybrid_keyframe`."""

    def test_hybrid_keyframe_dispatch_block_unchanged(self) -> None:
        src = _source(_TRAIN_STREAMING)
        expected = (
            'if args.streaming_frame_admission == "hybrid_keyframe":\n'
            "                # Milestone 4: use the centralised KeyframeAdmitter\n"
            "                # (the inline _hybrid_keyframe_admission closure is\n"
            "                # retained for back-compat with non-hybrid modes but\n"
            "                # the hybrid_keyframe mode goes through the new\n"
            "                # wrapper).\n"
            "                admission_fn = _keyframe_admission_closure\n"
            "                # Only wired alongside hybrid_keyframe: admission=all must\n"
            "                # admit every frame unconditionally (see _keyframe_pregate_closure\n"
            "                # docstring) so pregate_fn stays None for every other mode.\n"
            "                pregate_fn = _keyframe_pregate_closure\n"
        )
        assert expected in src

    def test_fixed_stride_branch_is_a_separate_elif_after_hybrid_keyframe(self) -> None:
        """The new mode must be an additive `elif`, never a change to the
        `if` clause that hybrid_keyframe/all already rely on."""
        src = _source(_TRAIN_STREAMING)
        hk_idx = src.index('if args.streaming_frame_admission == "hybrid_keyframe":')
        fs_idx = src.index('elif args.streaming_frame_admission == "fixed_stride":')
        assert fs_idx > hk_idx
        between = src[hk_idx:fs_idx]
        # Nothing but the unchanged hybrid_keyframe block should sit between.
        assert "pregate_fn = _keyframe_pregate_closure" in between

    def test_all_mode_still_gets_no_admission_fn_no_pregate_fn(self) -> None:
        """`admission_fn`/`pregate_fn` are initialised to None right before
        the if/elif chain, and neither `all` (falls through both branches)
        nor `fixed_stride` sets `pregate_fn` -- only `hybrid_keyframe` does."""
        src = _source(_TRAIN_STREAMING)
        idx = src.index("admission_fn = None\n            pregate_fn = None\n")
        assert idx != -1

    def test_online_pose_source_all_to_hybrid_override_untouched(self) -> None:
        """The pre-existing `open3d_odometry_live` + `admission=="all"` ->
        forced hybrid_keyframe override must be untouched by this change
        (it only ever fires when the mode is exactly "all")."""
        src = _source(_TRAIN_STREAMING)
        assert (
            'args.orbbec_pose_source == "open3d_odometry_live"\n'
            '        and args.streaming_frame_admission == "all"\n'
            "    ):\n"
            '        args.streaming_frame_admission = "hybrid_keyframe"'
            in src
        )
