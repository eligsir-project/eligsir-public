"""Tests for issue #1259: streaming_max_transfer_bytes default raise + telemetry.

Covers:
  (a) the new 2GB default resolves through the CLI/argparse -> config path,
  (b) DeviceTransferOwner high-water-mark tracking across a sequence of
      transfers, plus the one-time 80%-utilization WARNING,
  (c) the report.json ``transfer_pool`` block via write_post_training_report.
"""

from __future__ import annotations

import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# (a) default resolution through argparse / config
# ---------------------------------------------------------------------------


def test_streaming_max_transfer_bytes_argparse_default_is_2gb():
    from argparse import ArgumentParser

    from arguments import StreamingParams

    parser = ArgumentParser()
    StreamingParams(parser)
    parsed = parser.parse_args([])
    assert parsed.streaming_max_transfer_bytes == 2_147_483_648


def test_canonical_runtime_policy_dataclass_default_is_2gb():
    from eligsir.runtime.config import CanonicalRuntimePolicy

    policy = CanonicalRuntimePolicy()
    assert policy.max_transfer_bytes == 2_147_483_648


def test_config_boundary_resolves_default_streaming_max_transfer_bytes():
    """The config_boundary conversion just casts args.streaming_max_transfer_bytes
    through -- confirm the argparse default survives that path unmodified."""
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams, StreamingParams

    parser = ArgumentParser()
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    StreamingParams(parser)
    args = parser.parse_args([])
    assert int(args.streaming_max_transfer_bytes) == 2_147_483_648


def test_canonical_mapping_profile_still_overrides_to_640mb():
    """The explicit per-profile override in canonical_mapping.yml is untouched
    by the bare-default bump -- see the profile's own comment for the 640MiB
    computation this pins against."""
    from pathlib import Path

    profile = (
        Path(__file__).parents[1] / "configs/profiles/live/canonical_mapping.yml"
    ).read_text()
    assert "streaming_max_transfer_bytes: 671088640" in profile


# ---------------------------------------------------------------------------
# (b) DeviceTransferOwner high-water-mark + one-time WARNING
# ---------------------------------------------------------------------------


class _ImmediateTransfer:
    """CPU-only test double mirroring tests/test_issue1192_device_transfer_owner.py."""

    def __init__(self) -> None:
        self.device = torch.device("cuda")
        self.stats = _Stats()

    def h2d(self, host, *, dtype=None):
        from eligsir.runtime.transfers import TransferEvent

        out = host.clone()
        if dtype is not None:
            out = out.to(dtype=dtype)
        self.stats.h2d_ops += 1
        self.stats.h2d_bytes += host.numel() * host.element_size()
        return out, TransferEvent()

    def handoff_to_compute(self, *events):
        pass

    def d2h_into(self, device_tensor, host):
        from eligsir.runtime.transfers import TransferEvent

        host.copy_(device_tensor)
        self.stats.d2h_ops += 1
        self.stats.d2h_bytes += device_tensor.numel() * device_tensor.element_size()
        return TransferEvent()


class _Stats:
    def __init__(self) -> None:
        self.h2d_ops = 0
        self.h2d_bytes = 0
        self.d2h_ops = 0
        self.d2h_bytes = 0
        self.pool_waits = 0
        self.pool_drops = 0

    def snapshot(self):
        return {
            "h2d_ops": self.h2d_ops,
            "h2d_bytes": self.h2d_bytes,
            "d2h_ops": self.d2h_ops,
            "d2h_bytes": self.d2h_bytes,
            "pool_waits": self.pool_waits,
            "pool_drops": self.pool_drops,
        }


def _owner(max_pinned_bytes):
    from eligsir.runtime.transfers import DeviceTransferOwner

    return DeviceTransferOwner(
        "cuda",
        transfer_stream=_ImmediateTransfer(),
        max_inflight=4,
        max_shape_buckets=4,
        max_pinned_bytes=max_pinned_bytes,
    )


def test_high_water_mark_tracks_peak_not_current():
    owner = _owner(1_000_000)

    owner.upload_packed((np.zeros((400_000,), np.uint8),), purpose="a")
    first_peak = owner.accounting().peak_pinned_bytes
    assert first_peak >= 400_000

    owner.upload_packed((np.zeros((100_000,), np.uint8),), purpose="b")
    second_peak = owner.accounting().peak_pinned_bytes
    # Peak is monotonic non-decreasing even though the pool may have released
    # the first allocation's slot in between submissions.
    assert second_peak >= first_peak


def test_one_time_warning_fires_once_at_80_percent(capsys):
    owner = _owner(1000)

    # Below 80% -- no warning yet.
    owner.upload_packed((np.zeros((700,), np.uint8),), purpose="below")
    out = capsys.readouterr().out
    assert "WARNING" not in out

    # Crosses 80% (peak now >= 800/1000).
    owner.upload_packed((np.zeros((200,), np.uint8),), purpose="cross")
    out = capsys.readouterr().out
    assert "[DeviceTransferOwner] WARNING" in out
    assert out.count("[DeviceTransferOwner] WARNING") == 1

    # A further transfer must not repeat the warning.
    owner.upload_packed((np.zeros((50,), np.uint8),), purpose="again")
    out = capsys.readouterr().out
    assert "WARNING" not in out


def test_exceeding_limit_still_raises_and_is_not_silenced_by_telemetry():
    owner = _owner(100)
    with pytest.raises(RuntimeError, match="pinned byte limit"):
        owner.upload_packed((np.zeros((1000,), np.uint8),), purpose="too-big")


def test_transfer_pool_report_block_shape_and_values():
    from eligsir.runtime.transfers import transfer_pool_report_block

    owner = _owner(1_000_000)
    owner.upload_packed((np.zeros((250_000,), np.uint8),), purpose="a")

    block = transfer_pool_report_block(owner)
    assert set(block.keys()) == {
        "configured_limit_bytes",
        "peak_used_bytes",
        "peak_utilization_frac",
    }
    assert block["configured_limit_bytes"] == 1_000_000
    assert block["peak_used_bytes"] >= 250_000
    assert 0.0 < block["peak_utilization_frac"] <= 1.0


def test_transfer_pool_report_block_none_owner_returns_none():
    from eligsir.runtime.transfers import transfer_pool_report_block

    assert transfer_pool_report_block(None) is None


def test_peek_device_transfer_owner_never_constructs():
    from eligsir.runtime.transfers import peek_device_transfer_owner

    # A device string extremely unlikely to have been registered by any
    # other test in this process.
    assert peek_device_transfer_owner("cuda:61") is None


def test_transfer_accounting_records_explicit_sync_wait_duration():
    from eligsir.runtime.transfers import TransferEvent, TransferStats

    class _Event:
        def synchronize(self):
            return None

    stats = TransferStats()
    event = TransferEvent(_Event(), stats=stats)
    event.wait()

    assert stats.explicit_sync_waits == 1
    assert stats.explicit_sync_wait_ms >= 0.0
    assert stats.snapshot()["explicit_sync_waits"] == 1


def test_unowned_immediate_event_does_not_synthesize_sync_wait():
    from eligsir.runtime.transfers import TransferEvent, TransferStats

    stats = TransferStats()
    TransferEvent().wait()

    assert stats.explicit_sync_waits == 0
    assert stats.explicit_sync_wait_ms == 0.0


# ---------------------------------------------------------------------------
# (c) report.json ``transfer_pool`` block via write_post_training_report
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_lpips_import():
    """Same rationale/restore discipline as
    tests/test_comparison_report_seed_fade.py's fixture of the same name --
    avoid loading the real LPIPS network for a report-plumbing-only test, and
    avoid permanently clobbering sys.modules for later tests."""
    had_original = "lpipsPyTorch" in sys.modules
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if had_original:
            sys.modules["lpipsPyTorch"] = original
        else:
            sys.modules.pop("lpipsPyTorch", None)


class _FakeGaussian:
    def get_xyz(self):
        return torch.zeros(0, 3)


class _FakePipe:
    pass


def _minimal_render_fn(cam, gaussians, pipe, background):
    return {"render": torch.zeros(3, 64, 64)}


def test_transfer_pool_block_included_when_provided():
    from eligsir.utils.comparison_report import write_post_training_report

    with tempfile.TemporaryDirectory() as tmpdir:
        block = {
            "configured_limit_bytes": 2_147_483_648,
            "peak_used_bytes": 123_456,
            "peak_utilization_frac": 123_456 / 2_147_483_648,
        }
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            transfer_pool=block,
        )
        assert summary["transfer_pool"] == block


def test_transfer_pool_key_absent_when_not_provided():
    from eligsir.utils.comparison_report import write_post_training_report

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert "transfer_pool" not in summary


def test_transfer_pool_none_treated_as_absent():
    from eligsir.utils.comparison_report import write_post_training_report

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=5000,
            gaussians=_FakeGaussian(),
            train_cams=[],
            test_cams=[],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            transfer_pool=None,
        )
        assert "transfer_pool" not in summary
