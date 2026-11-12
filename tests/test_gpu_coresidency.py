"""Tests for utils/gpu_coresidency.py (issue #295 fix item 4): a run that
shared its GPU with another process should be self-identifying from its
own report.json instead of requiring after-the-fact forensics.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from eligsir.utils import gpu_coresidency as gc


def test_sample_gpu_coresidency_never_raises_when_no_tool_present(monkeypatch):
    monkeypatch.setattr(gc, "_run", lambda cmd, timeout=10.0: None)
    result = gc.sample_gpu_coresidency()
    assert result["tool"] is None
    assert result["process_count"] is None
    assert "timestamp" in result


def test_sample_nvidia_parses_csv(monkeypatch):
    csv_out = "1234, 512\n5678, 1024\n"

    def fake_run(cmd, timeout=10.0):
        if cmd[0] == "nvidia-smi":
            return csv_out
        return None

    monkeypatch.setattr(gc, "_run", fake_run)
    result = gc.sample_gpu_coresidency()
    assert result["tool"] == "nvidia-smi"
    assert result["process_count"] == 2
    assert result["processes"][0]["pid"] == "1234"
    assert result["processes"][0]["used_memory_mib"] == 512.0


def test_sample_nvidia_empty_output_is_zero_processes(monkeypatch):
    monkeypatch.setattr(gc, "_run", lambda cmd, timeout=10.0: "")
    result = gc.sample_gpu_coresidency()
    assert result["tool"] == "nvidia-smi"
    assert result["process_count"] == 0


def test_sample_falls_back_to_rocm_when_nvidia_absent(monkeypatch):
    def fake_run(cmd, timeout=10.0):
        if cmd[0] == "nvidia-smi":
            return None
        if cmd[0] == "rocm-smi":
            return "PID  DRM PROCESS\n1111  card0  python\n"
        return None

    monkeypatch.setattr(gc, "_run", fake_run)
    result = gc.sample_gpu_coresidency()
    assert result["tool"] == "rocm-smi"
    assert result["process_count"] == 1
    assert result["processes"][0]["pid"] == "1111"


def test_sample_never_raises_on_sampler_exception(monkeypatch):
    def boom(cmd, timeout=10.0):
        raise RuntimeError("subprocess exploded")

    monkeypatch.setattr(gc, "_run", boom)
    result = gc.sample_gpu_coresidency()
    assert result["tool"] is None
    assert result["process_count"] is None


def test_record_and_load_start_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(gc, "_run", lambda cmd, timeout=10.0: None)
    model_path = str(tmp_path / "run1")
    snap = gc.record_start_snapshot(model_path)
    assert snap is not None
    loaded = gc.load_start_snapshot(model_path)
    assert loaded == snap
    assert os.path.isfile(os.path.join(model_path, "gpu_coresidency_start.json"))


def test_load_start_snapshot_missing_returns_none(tmp_path):
    assert gc.load_start_snapshot(str(tmp_path / "never_started")) is None


def test_record_start_snapshot_never_raises_on_write_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(gc, "_run", lambda cmd, timeout=10.0: None)

    def boom_makedirs(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "makedirs", boom_makedirs)
    result = gc.record_start_snapshot(str(tmp_path / "run1"))
    assert result is None
    captured = capsys.readouterr()
    assert "gpu-coresidency" in captured.out


def test_build_report_block_flags_contamination_when_both_ends_agree(tmp_path, monkeypatch):
    model_path = str(tmp_path / "run1")
    os.makedirs(model_path)
    with open(os.path.join(model_path, "gpu_coresidency_start.json"), "w") as f:
        json.dump({"tool": "nvidia-smi", "process_count": 2, "processes": []}, f)

    monkeypatch.setattr(
        gc, "sample_gpu_coresidency",
        lambda: {"tool": "nvidia-smi", "process_count": 2, "processes": [], "timestamp": "t"},
    )
    block = gc.build_report_block(model_path)
    assert block["contaminated"] is True
    assert block["start"]["process_count"] == 2
    assert block["end"]["process_count"] == 2


def test_build_report_block_not_contaminated_when_only_one_end_busy(tmp_path, monkeypatch):
    model_path = str(tmp_path / "run1")
    os.makedirs(model_path)
    with open(os.path.join(model_path, "gpu_coresidency_start.json"), "w") as f:
        json.dump({"tool": "nvidia-smi", "process_count": 1, "processes": []}, f)

    monkeypatch.setattr(
        gc, "sample_gpu_coresidency",
        lambda: {"tool": "nvidia-smi", "process_count": 2, "processes": [], "timestamp": "t"},
    )
    block = gc.build_report_block(model_path)
    assert block["contaminated"] is False


def test_build_report_block_unknown_when_no_start_snapshot(tmp_path, monkeypatch):
    model_path = str(tmp_path / "never_started")
    monkeypatch.setattr(
        gc, "sample_gpu_coresidency",
        lambda: {"tool": "nvidia-smi", "process_count": 1, "processes": [], "timestamp": "t"},
    )
    block = gc.build_report_block(model_path)
    assert block["contaminated"] is None
    assert block["start"] is None
