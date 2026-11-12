"""Issue #837 criterion 8, `overflow_drop_policy` half.

Covers ``eligsir/io/policies/ingress_overflow_ownership.py``, the new
consumer module that feeds issue #836's REALIZED
``(inner_policy, outer_policy)`` overflow-policy pair into the
``eligsir.io.ingress_ownership`` contract as a single owner
declaration, and is wired into ``train_streaming.py``'s
``_ingress_ownership_report_block`` (the exact template this task followed
-- see that function's docstring).

``train_streaming.py`` itself cannot be imported in this environment (its
own module-level imports pull in a torchvision/torch combination that
fails to initialize here, independent of anything in this change -- see
CLAUDE.md: all real training runs happen inside the Docker container, not
host Python), so these tests exercise the new module directly plus the
same production ``resolve_ingress_ownership``/``resolve_ingress_policies``
entry points ``_ingress_ownership_report_block`` calls, matching the style
``tests/test_issue837_ingress_ownership.py`` already uses (pure resolver
calls, no training loop, no GPU, no ROS).

Two traps this suite exists to pin, named in the task spec:

- Trap 1: declaring BOTH the inner and outer overflow-policy boundaries as
  separate mutating owners against ``Responsibility.OVERFLOW_DROP_POLICY``
  trips ``DUPLICATE_INGRESS_AUTHORITY`` on every single live run. Test
  ``test_realistic_live_run_resolves_no_duplicate_authority`` (and its bag
  -replay counterpart) pin that a real run's resolved block is ``valid``
  with NO such failure code.
- Trap 2: a non-``ros2://live`` source must leave
  ``Responsibility.OVERFLOW_DROP_POLICY`` absent/unresolved, never
  fabricate an owner. ``test_unwrap_returns_none_for_non_ros2_live_source``
  and ``test_absent_overflow_policy_leaves_responsibility_unresolved`` pin
  that.
"""

from __future__ import annotations

from types import SimpleNamespace

from eligsir.io.ingress_ownership import (
    DUPLICATE_INGRESS_AUTHORITY,
    OwnerDeclaration,
    OwnershipRole,
    Responsibility,
    clear_registry,
    resolve_ingress_ownership,
)
from eligsir.io.policies.ingress_overflow import (
    FrameDropMode,
    LossSemantics,
    resolve_ingress_policies,
)
from eligsir.io.policies.ingress_overflow_ownership import (
    INNER_BOUNDARY_LABEL,
    OUTER_BOUNDARY_LABEL,
    OVERFLOW_DROP_POLICY_OWNER,
    resolve_overflow_drop_policy_owner_declaration,
    unwrap_frame_source_overflow_policies,
)


def setup_function(_fn):
    clear_registry()


def teardown_function(_fn):
    clear_registry()


# --------------------------------------------------------------------------
# resolve_overflow_drop_policy_owner_declaration: shape correctness
# --------------------------------------------------------------------------


def test_declaration_is_a_single_decision_owner_carrying_both_boundaries():
    inner, outer = resolve_ingress_policies(SimpleNamespace(ros2_live_bag_replay=False))

    decl = resolve_overflow_drop_policy_owner_declaration(inner, outer)

    assert decl.responsibility == Responsibility.OVERFLOW_DROP_POLICY
    assert decl.owner == OVERFLOW_DROP_POLICY_OWNER
    assert decl.role == OwnershipRole.DECISION_OWNER
    # mode surfaces the dimension that actually varies with source kind
    # (the inner boundary's loss semantics), not the outer boundary's
    # unconditional NO_SILENT_LOSS.
    assert decl.mode == LossSemantics.REAL_TIME_DROP.value
    assert decl.attributes["inner_boundary"] == INNER_BOUNDARY_LABEL
    assert decl.attributes["outer_boundary"] == OUTER_BOUNDARY_LABEL
    assert decl.attributes["inner_loss_semantics"] == LossSemantics.REAL_TIME_DROP.value
    assert decl.attributes["outer_loss_semantics"] == LossSemantics.NO_SILENT_LOSS.value
    assert decl.attributes["inner_drop_mode"] == FrameDropMode.OLDEST.value


def test_declaration_reflects_bag_replay_source_kind():
    inner, outer = resolve_ingress_policies(SimpleNamespace(ros2_live_bag_replay=True))

    decl = resolve_overflow_drop_policy_owner_declaration(inner, outer)

    assert decl.mode == LossSemantics.NO_SILENT_LOSS.value
    assert decl.attributes["inner_source_kind"] == "deterministic_bag_replay"
    assert decl.attributes["outer_source_kind"] == "deterministic_bag_replay"


# --------------------------------------------------------------------------
# Trap 1: a realistic live run resolves with NO duplicate authority
# --------------------------------------------------------------------------


def test_realistic_live_run_resolves_no_duplicate_authority():
    """Pins Trap 1: the two #852-resolved policies must become ONE
    declaration, not two, or every real ros2://live run would fail
    structurally with DUPLICATE_INGRESS_AUTHORITY."""
    inner, outer = resolve_ingress_policies(SimpleNamespace(ros2_live_bag_replay=False))
    declarations = [resolve_overflow_drop_policy_owner_declaration(inner, outer)]

    result = resolve_ingress_ownership(declarations)

    assert result.valid is True
    assert DUPLICATE_INGRESS_AUTHORITY not in result.failure_codes
    overflow = result.responsibilities[Responsibility.OVERFLOW_DROP_POLICY]
    assert overflow.duplicate is False
    assert overflow.resolved_owner is not None
    assert overflow.resolved_owner.owner == OVERFLOW_DROP_POLICY_OWNER
    assert len(overflow.mutating_declarations) == 1


def test_realistic_bag_replay_run_resolves_no_duplicate_authority():
    inner, outer = resolve_ingress_policies(SimpleNamespace(ros2_live_bag_replay=True))
    declarations = [resolve_overflow_drop_policy_owner_declaration(inner, outer)]

    result = resolve_ingress_ownership(declarations)

    assert result.valid is True
    assert result.failure_codes == []
    overflow = result.responsibilities[Responsibility.OVERFLOW_DROP_POLICY]
    assert overflow.duplicate is False


def test_declaring_both_boundaries_separately_WOULD_have_tripped_the_guard():
    """Negative control proving the guard this task exists to avoid is
    real: two independent MUTATOR/DECISION_OWNER declarations for the two
    boundaries (the wrong design this task's Trap 1 warns against) DOES
    trigger DUPLICATE_INGRESS_AUTHORITY. Confirms the single-owner
    representation is load-bearing, not merely stylistic."""
    inner, outer = resolve_ingress_policies(SimpleNamespace(ros2_live_bag_replay=False))
    wrong_declarations = [
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="inner_emit_queue_policy",
            role=OwnershipRole.DECISION_OWNER,
            mode=inner.loss_semantics.value,
        ),
        OwnerDeclaration(
            Responsibility.OVERFLOW_DROP_POLICY,
            owner="outer_resident_buffer_policy",
            role=OwnershipRole.DECISION_OWNER,
            mode=outer.loss_semantics.value,
        ),
    ]

    result = resolve_ingress_ownership(wrong_declarations)

    assert result.valid is False
    assert DUPLICATE_INGRESS_AUTHORITY in result.failure_codes


# --------------------------------------------------------------------------
# Trap 2: non-ros2://live sources get no fabricated owner
# --------------------------------------------------------------------------


def test_unwrap_returns_none_for_non_ros2_live_source():
    """A TUM/ScanNet/generic frame source has neither _inner_policy nor
    _outer_policy -- must resolve to (None, None), never a guess."""
    plain_source = SimpleNamespace(some_other_attr=123)

    inner, outer = unwrap_frame_source_overflow_policies(plain_source)

    assert inner is None
    assert outer is None


def test_unwrap_finds_policies_through_one_level_of_wrapping():
    """StridedFrameSource (issue #834) wraps the real ros2://live source
    at self._src -- the unwrap must see through that indirection."""
    inner_policy, outer_policy = resolve_ingress_policies(
        SimpleNamespace(ros2_live_bag_replay=False)
    )
    real_ros2_live_source = SimpleNamespace(
        _inner_policy=inner_policy, _outer_policy=outer_policy
    )
    wrapped = SimpleNamespace(_src=real_ros2_live_source)

    found_inner, found_outer = unwrap_frame_source_overflow_policies(wrapped)

    assert found_inner is inner_policy
    assert found_outer is outer_policy


def test_unwrap_does_not_loop_forever_on_self_referencing_wrapper():
    """Defensive: a pathological object that points _src at itself must
    not hang the unwrap walk."""
    cyclic = SimpleNamespace()
    cyclic._src = cyclic

    inner, outer = unwrap_frame_source_overflow_policies(cyclic)

    assert inner is None
    assert outer is None


def test_absent_overflow_policy_leaves_responsibility_unresolved():
    """Mirrors the existing source_sampling behavior when sampling_owner is
    None: no declaration is appended at all, so the responsibility resolves
    with no owner and is not reported as a spurious failure."""
    declarations: list = []  # what _ingress_ownership_report_block builds
    # when unwrap_frame_source_overflow_policies returned (None, None).

    result = resolve_ingress_ownership(declarations)

    overflow = result.responsibilities[Responsibility.OVERFLOW_DROP_POLICY]
    assert overflow.resolved_owner is None
    assert overflow.mutating_declarations == []
    assert overflow.duplicate is False
    # Absence of a declaration is not itself a structural failure.
    assert result.valid is True
    assert DUPLICATE_INGRESS_AUTHORITY not in result.failure_codes
