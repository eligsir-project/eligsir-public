from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest


pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _mock_lpips_import():
    previous = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("lpipsPyTorch", None)
        else:
            sys.modules["lpipsPyTorch"] = previous


class _FakeGaussian:
    def get_xyz(self):
        import torch

        return torch.zeros(0, 3)


def _render(_cam, _gaussians, _pipe, _background):
    import torch

    return {"render": torch.zeros(3, 2, 2)}


def test_oracle_partition_control_is_persisted_and_refuses_unbiased_verdict(tmp_path) -> None:
    from eligsir.utils.comparison_report import write_post_training_report

    block = {
        "control_signal_source": "scored_holdout_oracle",
        "control_probe_hash": None,
        "control_probe_n": 0,
        "scored_holdout_hash": "abc123",
        "scored_holdout_n": 8,
        "control_scored_overlap_n": 8,
        "oracle_control_enabled": True,
        "controller_actuation_used": True,
        "scientific_eval_status": "ORACLE_INTERVENTION",
    }
    summary = write_post_training_report(
        model_path=str(tmp_path),
        iteration=1,
        gaussians=_FakeGaussian(),
        train_cams=[],
        test_cams=[],
        render_fn=_render,
        pipe=object(),
        background=None,
        partition_control=block,
    )

    on_disk = json.loads(
        (tmp_path / "comparison" / "iter_1" / "report.json").read_text()
    )
    assert summary["partition_control"] == block
    assert on_disk["scientific_eval_status"] == "ORACLE_INTERVENTION"
    assert on_disk["unbiased_test_verdict_permitted"] is False


def test_non_oracle_partition_control_preserves_unbiased_verdict_eligibility(tmp_path) -> None:
    from eligsir.utils.comparison_report import write_post_training_report

    block = {
        "control_signal_source": "training_derived",
        "control_probe_hash": None,
        "control_probe_n": 0,
        "scored_holdout_hash": "abc123",
        "scored_holdout_n": 8,
        "control_scored_overlap_n": 0,
        "oracle_control_enabled": False,
        "controller_actuation_used": True,
        "scientific_eval_status": "VALID_NON_ORACLE_CONTROL",
    }
    summary = write_post_training_report(
        model_path=str(tmp_path),
        iteration=1,
        gaussians=_FakeGaussian(),
        train_cams=[],
        test_cams=[],
        render_fn=_render,
        pipe=object(),
        background=None,
        partition_control=block,
    )

    assert summary["scientific_eval_status"] == "VALID_NON_ORACLE_CONTROL"
    assert summary["unbiased_test_verdict_permitted"] is True


def test_disabled_partition_control_only_adds_inert_nested_metadata(tmp_path) -> None:
    from eligsir.utils.comparison_report import write_post_training_report

    block = {
        "control_signal_source": "training_derived",
        "controller_actuation_used": False,
        "scientific_eval_status": "CONTROL_DISABLED",
    }
    summary = write_post_training_report(
        model_path=str(tmp_path),
        iteration=1,
        gaussians=_FakeGaussian(),
        train_cams=[],
        test_cams=[],
        render_fn=_render,
        pipe=object(),
        background=None,
        partition_control=block,
    )

    assert summary["partition_control"] == block
    assert "scientific_eval_status" not in summary
    assert "unbiased_test_verdict_permitted" not in summary
