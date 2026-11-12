import torch

from eligsir.contracts.control import ResolutionTier
from eligsir.contracts.runtime import SourceIdentity
from eligsir.mapping.supervision.frame_belief import (
    FrameModalityState,
    FrameSupervisionBelief,
    FrameSupervisionFact,
)
from eligsir.optimization.gaussian.supervision_evidence import SupervisionModality


def _source(ordinal: int) -> SourceIdentity:
    return SourceIdentity("camera", 0, ordinal, ordinal * 1_000)


def _fact(ordinal: int, *, modality="rgb", effectiveness=0.8):
    return FrameSupervisionFact(
        source=_source(ordinal),
        modality=modality,
        tier=ResolutionTier.QUARTER,
        intrinsic_reliability=0.9,
        rgb_debt=0.7,
        depth_debt=0.2,
        realized_effectiveness=effectiveness,
        complementarity=0.4,
        materialization_pixel_cost=120.0,
    )


def test_replay_updates_exposure_but_not_independent_physical_view_count():
    belief = FrameSupervisionBelief(max_frames=4)
    belief.update(_fact(1))
    belief.update(_fact(1, effectiveness=1.0))

    state = belief.state(_source(1), "rgb", ResolutionTier.QUARTER)
    assert state is not None
    assert state.optimization_exposure_count == 2
    assert state.independent_physical_view_count == 1
    assert belief.independent_physical_view_count == 1


def test_independent_sources_and_modalities_are_kept_separate():
    belief = FrameSupervisionBelief(max_frames=4)
    belief.update(_fact(1, modality="rgb"))
    belief.update(_fact(1, modality="depth"))
    belief.update(_fact(2, modality="rgb"))

    assert belief.independent_physical_view_count == 2
    assert belief.state(_source(1), "rgb", ResolutionTier.QUARTER).rgb_debt == 0.7
    assert belief.state(_source(1), "depth", ResolutionTier.QUARTER).depth_debt == 0.2
    assert belief.state_count == 3


def test_updates_are_bounded_and_store_does_not_schedule():
    belief = FrameSupervisionBelief(max_frames=1, max_exposures_per_frame=3)
    fact = _fact(1, effectiveness=9.0)
    for _ in range(20):
        belief.update(fact)
    state = belief.state(_source(1), "rgb", ResolutionTier.QUARTER)
    assert state.optimization_exposure_count == 3
    assert 0.0 <= state.realized_effectiveness <= 1.0
    assert belief.scheduling_decision is None


def test_from_batch_preserves_source_modality_and_tier_identity():
    from eligsir.optimization.gaussian.supervision_evidence import (
        EvidenceSnapshotToken,
        SupervisionEvidenceBatch,
    )

    token = EvidenceSnapshotToken(*(0 for _ in range(8)))
    batch = SupervisionEvidenceBatch(
        snapshot_token=token,
        transaction_id=1,
        modalities=(SupervisionModality(), SupervisionModality(depth_loss_enabled=True)),
        intrinsic_reliability=torch.tensor([0.8, 0.6]),
        rgb_residual_summary=torch.tensor([0.2, 0.3]),
        depth_residual_summary=torch.tensor([0.0, 0.4]),
        alpha_support_summary=torch.tensor([0.9, 0.5]),
    )
    belief = FrameSupervisionBelief(max_frames=4)
    belief.update_batch(
        batch,
        sources=(_source(1), _source(2)),
        tiers=(ResolutionTier.NATIVE, ResolutionTier.HALF),
    )
    assert belief.state(_source(1), "rgb", ResolutionTier.NATIVE) is not None
    assert belief.state(_source(2), "depth", ResolutionTier.HALF) is not None
