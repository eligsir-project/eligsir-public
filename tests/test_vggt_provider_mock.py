from __future__ import annotations

import torch

from geometry.priors import SurfacePrior
from geometry.vggt_provider import OptionalVGGTPriorProvider, VGGTProviderConfig


def test_vggt_provider_is_optional():
    provider = OptionalVGGTPriorProvider()
    assert not provider.available()


def test_vggt_provider_mock_predicts_prior():
    def predict(_packet, _cam):
        depth = torch.ones(1, 2, 2)
        return SurfacePrior(depth, depth, depth.bool(), None, None, None, frame_uid=1)

    provider = OptionalVGGTPriorProvider(predict, VGGTProviderConfig(mode="sensor_vggt_offline"))
    assert provider.available()
    prior = provider.predict(object(), None)
    assert prior.depth.shape == (1, 2, 2)
