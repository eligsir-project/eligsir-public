from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from eligsir.utils.tick_torch_profile import TickTorchProfileWindow


def test_window_is_inert_when_env_unset():
    window = TickTorchProfileWindow(env={})
    assert window.enabled is False
    for _ in range(3):
        with window.tick():
            torch.ones(4).sum()
    assert window.completed is False
    assert window.tick_index == 0


def test_window_records_only_the_requested_ticks_and_writes_the_table(tmp_path):
    out = tmp_path / "prof" / "tick_torch_profile.txt"
    window = TickTorchProfileWindow(env={"SPLAT_TORCH_PROFILE_TICKS": "2:3", "SPLAT_TORCH_PROFILE_OUT": str(out)})
    assert (window.start, window.count) == (2, 3)
    for index in range(8):
        with window.tick():
            torch.ones(4).mul(2.0).sum()
        if index < 4:
            assert window.completed is False
    assert window.completed is True
    text = out.read_text()
    assert "torch.profiler window: ticks [2, 5)" in text
    assert "ops_per_tick=" in text
    assert "aten::" in text


def test_close_flushes_an_open_window(tmp_path):
    out = tmp_path / "t.txt"
    window = TickTorchProfileWindow(env={"SPLAT_TORCH_PROFILE_TICKS": "0:10", "SPLAT_TORCH_PROFILE_OUT": str(out)})
    with window.tick():
        torch.ones(2).sum()
    assert window.completed is False
    window.close()
    assert window.completed is True and out.exists()


def test_bad_spec_is_rejected():
    with pytest.raises(ValueError):
        TickTorchProfileWindow(env={"SPLAT_TORCH_PROFILE_TICKS": "5:0"})


def test_launches_are_attributed_to_the_innermost_phase_range(tmp_path, monkeypatch):
    """CPU-only check of the attribution walk: nested record_function ranges
    named like phases, with a fake launch-class event name substituted for
    a CPU op via LAUNCH_EVENT_NAMES."""
    import eligsir.utils.tick_torch_profile as mod

    monkeypatch.setattr(mod, "LAUNCH_EVENT_NAMES", ("aten::mul",))
    out = tmp_path / "t.txt"
    window = TickTorchProfileWindow(env={"SPLAT_TORCH_PROFILE_TICKS": "0:2", "SPLAT_TORCH_PROFILE_OUT": str(out)})
    for _ in range(2):
        with window.tick():
            with torch.profiler.record_function("m_outer"):
                torch.ones(3).mul(2.0)
                with torch.profiler.record_function("m_inner"):
                    torch.ones(3).mul(2.0)
                    torch.ones(3).mul(2.0)
    text = out.read_text()
    assert "launch-class events per tick by innermost phase" in text
    inner = [line for line in text.splitlines() if "m_inner" in line and "/tick" in line]
    outer = [line for line in text.splitlines() if "m_outer" in line and "/tick" in line]
    assert inner and inner[0].strip().startswith("2.0/tick")
    assert outer and outer[0].strip().startswith("1.0/tick")
