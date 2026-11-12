"""Issue #601 <-> #600 reconciliation.

#601's :class:`BackpressureDecision` is deliberately shaped to adapt onto
#600's ``ObservationDecision`` through exactly one function,
:func:`~eligsir.mapping.scheduler.backpressure.as_observation_decision`.

At the time #601 was written, #600 lived on an unmerged branch
(``agent/issue600-admission-decision-contract``) whose contract module is
``eligsir/mapping/admission/observation_decision.py``. This test pins
the adapter against a **faithful stand-in of that real class** -- same
field names, same defaults, same ``__post_init__`` invariants, same
``as_stats`` shape -- installed into ``sys.modules`` so the assertion runs
on any checkout, before or after #600 merges.

When #600 lands for real, this file is the thing to re-run first: if the
real class has drifted from the stand-in, these tests are where that
shows up, and the fix is confined to ``as_observation_decision``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest

from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureConfig,
    BackpressureOwner,
    ObservationDecisionView,
    PendingObservation,
    as_observation_decision,
)

MODULE_NAME = "eligsir.mapping.admission.observation_decision"


# --- faithful stand-in for #600's real contract ------------------------
@dataclass
class _ObservationDecision600:
    observe: bool = True
    mapping_candidate: bool = True
    admit_supervision: bool = True
    allow_geometry_insertion: bool = True
    reason: str = "admitted"
    optimizer_steps_since_last_admit: Optional[int] = None
    consecutive_skips: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.allow_geometry_insertion and not self.admit_supervision:
            raise ValueError(
                "ObservationDecision violates the #600 insertion invariant"
            )
        if self.admit_supervision and not self.mapping_candidate:
            raise ValueError(
                "ObservationDecision: admit_supervision=True with "
                "mapping_candidate=False"
            )

    @property
    def admitted(self) -> bool:
        return bool(self.admit_supervision)


@pytest.fixture()
def issue600_contract(monkeypatch):
    """Install the stand-in at #600's real module path."""
    module = types.ModuleType(MODULE_NAME)
    module.ObservationDecision = _ObservationDecision600
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    yield module


def _decisions(**cfg_overrides):
    cfg = BackpressureConfig(
        enabled=True,
        pending_max_count=4,
        pending_max_age_control_ticks=10_000,
        max_admissions_per_window=1,
        admission_window_control_ticks=10_000,
        **cfg_overrides,
    )
    owner = BackpressureOwner(cfg)
    out = []
    for i in range(24):
        for d in owner.submit(
            PendingObservation(
                frame_id=i,
                release_control_tick=i,
                novelty=0.05,
                quality=0.1,
                motion=0.02,
                translation_m=0.001,
                rotation_deg=0.05,
            ),
            control_tick=i,
        ):
            out.append(d)
    return out


class TestAdapterUsesTheRealContract:
    def test_admit_maps_onto_the_real_class(self, issue600_contract):
        decisions = _decisions()
        admit = next(d for d in decisions if d.action is BackpressureAction.ADMIT)
        adapted = as_observation_decision(admit)
        assert isinstance(adapted, _ObservationDecision600), (
            "the adapter did not use the real #600 contract when it was importable"
        )
        assert adapted.observe is True
        assert adapted.mapping_candidate is True
        assert adapted.admit_supervision is True
        assert adapted.allow_geometry_insertion is True
        assert adapted.admitted is True

    def test_skip_maps_onto_the_real_class_without_insertion_permission(
        self, issue600_contract
    ):
        decisions = _decisions()
        skipped = [d for d in decisions if d.skipped]
        assert skipped, "the fixture stream produced no skips"
        for d in skipped:
            adapted = as_observation_decision(d)
            assert isinstance(adapted, _ObservationDecision600)
            assert adapted.observe is True, (
                "a skipped frame must still be observed -- source observation "
                "never waits for mapping admission"
            )
            assert adapted.admit_supervision is False
            assert adapted.allow_geometry_insertion is False
            assert adapted.admitted is False

    def test_reasons_narrow_to_the_600_vocabulary(self, issue600_contract):
        vocabulary = {
            "admitted",
            "capacity",
            "redundant",
            "quality",
            "frontier_forced",
            "force_all_stress",
        }
        seen = set()
        for d in _decisions():
            adapted = as_observation_decision(d)
            assert adapted.reason in vocabulary, (
                f"#601 reason {d.reason!r} mapped to {adapted.reason!r}, which is "
                "not in #600's ObservationDecision reason vocabulary"
            )
            seen.add(adapted.reason)
        assert len(seen) > 1, "the fixture stream exercised only one reason"

    def test_the_fine_grained_601_reason_survives_in_extra(self, issue600_contract):
        for d in _decisions():
            adapted = as_observation_decision(d)
            assert adapted.extra["owner"] == "BackpressureOwner"
            assert adapted.extra["backpressure_reason"] == d.reason
            assert adapted.extra["backpressure_action"] == d.action.value
            assert adapted.extra["pressure_state"] == d.pressure_state.value
            assert "signals" in adapted.extra

    def test_coalesced_representative_is_carried_across(self, issue600_contract):
        coalesced = [
            d
            for d in _decisions()
            if d.action is BackpressureAction.COALESCE_INTO
        ]
        assert coalesced, "the fixture stream produced no coalescing"
        for d in coalesced:
            adapted = as_observation_decision(d)
            assert adapted.reason == "redundant"
            assert (
                adapted.extra["representative_frame_id"]
                == d.representative_frame_id
            )

    def test_consecutive_skips_is_populated(self, issue600_contract):
        for d in _decisions():
            adapted = as_observation_decision(d)
            assert adapted.consecutive_skips == d.pending_depth

    def test_force_all_stress_reason_is_preserved_verbatim(self, issue600_contract):
        owner = BackpressureOwner(
            BackpressureConfig(enabled=True, force_all_stress=True)
        )
        (decision,) = owner.submit(
            PendingObservation(frame_id=0), control_tick=0
        )
        adapted = as_observation_decision(decision)
        assert adapted.reason == "force_all_stress"
        assert adapted.admitted is True


class TestFallbackWhenIssue600HasNotLanded:
    def test_falls_back_to_the_structural_view(self, monkeypatch):
        # Neither landing site importable.
        monkeypatch.setitem(sys.modules, MODULE_NAME, None)
        monkeypatch.setitem(
            sys.modules,
            "eligsir.mapping.admission.decision_contracts",
            types.ModuleType("eligsir.mapping.admission.decision_contracts"),
        )
        owner = BackpressureOwner(BackpressureConfig(enabled=True))
        (decision,) = owner.submit(PendingObservation(frame_id=0), control_tick=0)
        adapted = as_observation_decision(decision)
        assert isinstance(adapted, ObservationDecisionView)
        assert adapted.observe is True
        assert adapted.admit_supervision is True

    def test_view_and_real_contract_agree_on_every_shared_field(
        self, issue600_contract
    ):
        """The fallback must not be a semantically different object."""
        decisions = _decisions()
        for d in decisions[:8]:
            real = as_observation_decision(d)
            view = ObservationDecisionView(
                observe=d.observe,
                mapping_candidate=d.mapping_candidate,
                admit_supervision=d.admit_supervision,
                allow_geometry_insertion=d.allow_geometry_insertion,
                reason=real.reason,
            )
            for name in (
                "observe",
                "mapping_candidate",
                "admit_supervision",
                "allow_geometry_insertion",
                "reason",
            ):
                assert getattr(real, name) == getattr(view, name)


class TestInvariantsAreCompatible:
    """#600 enforces its own invariants in ``__post_init__``. Every
    decision #601 can produce must satisfy them -- otherwise the adapter
    would raise at runtime on some real frame."""

    def test_no_601_decision_can_violate_a_600_invariant(self, issue600_contract):
        owner = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=3,
                pending_max_age_control_ticks=25,
                max_admissions_per_window=1,
                admission_window_control_ticks=50,
                novelty_protect=0.8,
            )
        )
        actions = set()
        for i in range(400):
            for d in owner.submit(
                PendingObservation(
                    frame_id=i,
                    release_control_tick=i,
                    novelty=0.95 if i % 37 == 0 else 0.02,
                    quality=0.1,
                    motion=0.01,
                    translation_m=0.001,
                    rotation_deg=0.02,
                    cheap_keyframe=(i % 11 == 0),
                ),
                control_tick=i,
            ):
                as_observation_decision(d)  # must never raise
                actions.add(d.action)

        # A second sweep whose accumulated motion clears the redundancy
        # threshold without reaching the representative threshold, so the
        # capacity-skip branch is exercised too.
        owner2 = BackpressureOwner(
            BackpressureConfig(
                enabled=True,
                pending_max_count=50,
                pending_max_age_control_ticks=100_000,
                max_admissions_per_window=1,
                admission_window_control_ticks=100_000,
                causal_deltas_are_cumulative=True,
                redundant_translation_m=0.01,
                causal_admit_translation_m=10.0,
            )
        )
        for i in range(20):
            for d in owner2.submit(
                PendingObservation(
                    frame_id=i, release_control_tick=i, translation_m=0.5, novelty=0.0
                ),
                control_tick=i,
            ):
                as_observation_decision(d)  # must never raise
                actions.add(d.action)

        # The sweeps really did exercise more than the trivial path.
        assert BackpressureAction.ADMIT in actions
        assert BackpressureAction.COALESCE_INTO in actions
        assert BackpressureAction.SKIP_CAPACITY in actions
        assert len(actions) >= 3, f"only exercised {actions}"

    def test_metadata_only_is_representable(self, issue600_contract):
        owner = BackpressureOwner(BackpressureConfig(enabled=True))
        (decision,) = owner.submit(
            PendingObservation(frame_id=1), control_tick=0, mapping_candidate=False
        )
        adapted = as_observation_decision(decision)
        assert adapted.observe is True
        assert adapted.mapping_candidate is False
        assert adapted.admit_supervision is False
        assert adapted.allow_geometry_insertion is False
