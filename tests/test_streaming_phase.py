import pytest

from eligsir.runtime.streaming_phase import Phase, StreamingPhaseController


def test_mapping_boundary_requires_source_and_dispositions_complete():
    controller = StreamingPhaseController(refinement_steps=3)

    assert controller.observe_source(
        source_exhausted=False,
        dispositions_complete=True,
        successful_optimization_steps=10,
        iteration=20,
    ) is None
    assert controller.observe_source(
        source_exhausted=True,
        dispositions_complete=False,
        successful_optimization_steps=11,
        iteration=21,
    ) is None
    assert controller.phase is Phase.MAPPING


def test_mapping_boundary_is_emitted_once_and_counts_only_successful_steps():
    controller = StreamingPhaseController(refinement_steps=3)

    transition = controller.observe_source(
        source_exhausted=True,
        dispositions_complete=True,
        successful_optimization_steps=11,
        iteration=21,
    )
    assert transition.from_phase is Phase.MAPPING
    assert transition.to_phase is Phase.REFINEMENT
    assert controller.mapping_end_iteration == 21
    assert controller.mapping_optimizer_steps == 11

    # Loop iterations without a real optimizer step do not spend refinement work.
    assert controller.observe_source(
        source_exhausted=True,
        dispositions_complete=True,
        successful_optimization_steps=11,
        iteration=22,
    ) is None
    assert controller.refinement_steps_completed == 0

    assert controller.observe_source(True, True, 12, 23) is None
    assert controller.observe_source(True, True, 13, 24) is None
    complete = controller.observe_source(True, True, 14, 25)
    assert complete.from_phase is Phase.REFINEMENT
    assert complete.to_phase is Phase.COMPLETE
    assert controller.refinement_steps_completed == 3

    assert controller.observe_source(True, True, 15, 26) is None


def test_successful_optimizer_counter_must_be_monotonic():
    controller = StreamingPhaseController(refinement_steps=2)
    controller.observe_source(True, True, 5, 10)
    with pytest.raises(ValueError, match="monotonic"):
        controller.observe_source(True, True, 4, 11)


def test_phase_state_round_trip_preserves_exact_boundary():
    controller = StreamingPhaseController(refinement_steps=3)
    controller.observe_source(True, True, 11, 21)
    controller.observe_source(True, True, 12, 22)

    restored = StreamingPhaseController(refinement_steps=3)
    restored.load_resume_state_dict(controller.resume_state_dict())

    assert restored.phase is Phase.REFINEMENT
    assert restored.mapping_end_iteration == 21
    assert restored.mapping_optimizer_steps == 11
    assert restored.refinement_steps_completed == 1
    assert restored.observe_source(True, True, 13, 23) is None
    assert restored.observe_source(True, True, 14, 24).to_phase is Phase.COMPLETE


def test_phase_state_rejects_changed_refinement_budget():
    controller = StreamingPhaseController(refinement_steps=3)
    payload = controller.resume_state_dict()

    with pytest.raises(ValueError, match="refinement_steps"):
        StreamingPhaseController(refinement_steps=4).load_resume_state_dict(payload)


def test_phase_state_late_validation_failure_is_atomic():
    target = StreamingPhaseController(refinement_steps=3)
    target.observe_source(True, True, 11, 21)
    before = target.resume_state_dict()
    broken = dict(before, last_successful_optimization_steps=-1)

    with pytest.raises(ValueError, match="successful"):
        target.load_resume_state_dict(broken)

    assert target.resume_state_dict() == before
