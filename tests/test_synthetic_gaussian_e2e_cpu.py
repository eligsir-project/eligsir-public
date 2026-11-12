import pytest

pytest.importorskip("torch")

from eligsir.runtime.session import MappingSession
from synthetic_gaussian_ci import (
    ExactPacketTrackingProvider,
    TinyCpuGaussianReconstructor,
    packet_from_view,
    render_analytic_view,
)


def test_synthetic_scene_reconstructs_through_canonical_mapping_session_on_cpu():
    train_x = (-0.24, -0.08, 0.08, 0.24)
    provider = ExactPacketTrackingProvider()
    mapper = TinyCpuGaussianReconstructor()

    with MappingSession(provider, mapper=mapper, require_tracking=True) as session:
        for index, camera_x in enumerate(train_x):
            outcome = session.submit(packet_from_view(render_analytic_view(index, camera_x)))
            assert outcome.accepted
            assert outcome.frame is not None
            assert outcome.frame.device == "cpu"

        assert session.submitted_count == len(train_x)
        assert session.mapped_count == len(train_x)
        assert session.rejected_count == 0

    assert provider.closed

    stats = mapper.finalize(optimizer_steps=10)
    holdout = render_analytic_view(100, 0.0)
    metrics = mapper.evaluate(holdout)

    assert 40 <= stats.gaussian_count <= 200
    assert stats.optimizer_steps == 10
    assert stats.final_loss < stats.initial_loss * 0.8
    assert metrics.mse < 0.1
    assert metrics.psnr_db > 10.0
