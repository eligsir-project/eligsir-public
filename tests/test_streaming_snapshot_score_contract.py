import json
from types import SimpleNamespace

import pytest
import torch

from eligsir.mapping.integration.frame_preparation import (
    _render_streaming_snapshot,
)
from eligsir.utils.score_contract import declare_camera_score_provenance


def test_streaming_snapshot_scores_finite_renderer_overshoot(tmp_path):
    cam = SimpleNamespace(
        image_name="view",
        original_image=torch.zeros(3, 2, 2),
    )
    declare_camera_score_provenance(cam, context="test_streaming_snapshot_gt")
    scene = SimpleNamespace(
        get_local_cameras=lambda: [cam],
        getTestCameras=lambda: [],
    )
    gaussians = SimpleNamespace(get_xyz=torch.zeros(1, 3))
    scalars = []
    writer = SimpleNamespace(
        add_scalar=lambda name, value, step: scalars.append((name, value, step))
    )

    _render_streaming_snapshot(
        gaussians,
        scene,
        lambda *_args: {"render": torch.full((3, 2, 2), 1.1)},
        None,
        None,
        SimpleNamespace(model_path=str(tmp_path)),
        7,
        writer,
    )

    assert scalars[0][0] == "streaming_eval/train_psnr"
    assert scalars[0][2] == 7
    assert ("streaming_eval/train_score_clamp_fraction", 1.0, 7) in scalars
    with open(
        tmp_path / "streaming_renders" / "frame_7" / "train" / "score_domain_clamp.json"
    ) as f:
        clamp = json.load(f)
    assert clamp["samples_changed"] == 12
    assert clamp["value_max"] == pytest.approx(1.1)
