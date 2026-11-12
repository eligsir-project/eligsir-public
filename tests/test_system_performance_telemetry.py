"""CPU-only unit tests for ``eligsir/telemetry/system_sampler.py``.

Covers: the sampler thread writing rows and phase markers, graceful handling
of a missing GPU (all three backends returning ``None``), clean stop, the
plot function on synthetic data, and the summary file. No GPU/nvidia-smi/
sysfs access -- every GPU reader is injected as a fake so these tests run
identically in CI and on a GPU-less box.
"""

from __future__ import annotations

import json
import time

import pytest


def _sampler(tmp_path, **kwargs):
    from eligsir.telemetry.system_sampler import SystemPerformanceSampler

    defaults = dict(
        interval_s=0.02,
        torch_reader=lambda: None,
        rocm_reader=lambda: None,
        nvidia_reader=lambda: None,
    )
    defaults.update(kwargs)
    return SystemPerformanceSampler(tmp_path, **defaults)


def _rows(tmp_path):
    path = tmp_path / "system_performance.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestSampler:
    def test_writes_sample_rows_with_all_channels_present(self, tmp_path):
        sampler = _sampler(tmp_path)
        sampler.start()
        time.sleep(0.15)
        sampler.stop()

        rows = _rows(tmp_path)
        assert len(rows) >= 2
        for row in rows:
            for key in (
                "t_wall",
                "t_elapsed_s",
                "cpu_pct_process",
                "cpu_pct_host",
                "rss_gb",
                "host_mem_used_gb",
                "host_mem_total_gb",
                "gpu_util_pct",
                "gpu_mem_used_gb",
                "gpu_mem_total_gb",
                "gpu_power_w",
                "gpu_temp_c",
            ):
                assert key in row

    def test_missing_gpu_channels_are_null_not_zero(self, tmp_path):
        sampler = _sampler(tmp_path)
        sampler.start()
        time.sleep(0.08)
        sampler.stop()

        rows = _rows(tmp_path)
        assert rows
        for row in rows:
            assert row["gpu_util_pct"] is None
            assert row["gpu_mem_used_gb"] is None
            assert row["gpu_power_w"] is None
            assert row["gpu_temp_c"] is None

    def test_torch_reader_supplies_gpu_memory_without_util_or_power(self, tmp_path):
        sampler = _sampler(tmp_path, torch_reader=lambda: (3.5, 16.0))
        sampler.start()
        time.sleep(0.08)
        sampler.stop()

        rows = _rows(tmp_path)
        assert rows
        assert any(r["gpu_mem_used_gb"] == 3.5 for r in rows)
        assert all(r["gpu_util_pct"] is None for r in rows)

    def test_rocm_reader_supplies_util_power_temp_and_gtt(self, tmp_path):
        sampler = _sampler(
            tmp_path,
            rocm_reader=lambda: {
                "util_pct": 42.0,
                "mem_used_gb": 2.0,
                "mem_total_gb": 8.0,
                "gtt_used_gb": 1.0,
                "power_w": 15.0,
                "temp_c": 55.0,
            },
        )
        sampler.start()
        time.sleep(0.08)
        sampler.stop()

        rows = _rows(tmp_path)
        assert rows
        row = rows[0]
        assert row["gpu_util_pct"] == 42.0
        assert row["gpu_gtt_used_gb"] == 1.0
        assert row["gpu_power_w"] == 15.0
        assert row["gpu_source"] == "rocm_sysfs"

    def test_nvidia_reader_used_only_when_rocm_absent_and_caches_failure(self, tmp_path):
        calls = {"n": 0}

        def nvidia_reader():
            calls["n"] += 1
            return None

        sampler = _sampler(tmp_path, nvidia_reader=nvidia_reader, interval_s=0.01)
        sampler.start()
        time.sleep(0.1)
        sampler.stop()

        # First failure should be cached: nvidia-smi must not be re-invoked
        # every sample once it has already failed once.
        assert calls["n"] == 1

    def test_a_gpu_reader_that_raises_does_not_crash_the_sampler(self, tmp_path):
        def boom():
            raise RuntimeError("nvidia-smi exploded")

        sampler = _sampler(tmp_path, nvidia_reader=boom, interval_s=0.01)
        sampler.start()
        time.sleep(0.08)
        sampler.stop()

        rows = _rows(tmp_path)
        assert rows  # sampler kept running despite the reader raising

    def test_mark_appends_an_event_row_with_elapsed_time(self, tmp_path):
        sampler = _sampler(tmp_path, interval_s=10.0)
        sampler.start()
        time.sleep(0.05)
        sampler.mark("mapping_endpoint", iteration=1234)
        sampler.stop()

        rows = _rows(tmp_path)
        events = [r for r in rows if "event" in r]
        assert len(events) == 1
        assert events[0]["event"] == "mapping_endpoint"
        assert events[0]["iteration"] == 1234
        assert events[0]["t_elapsed_s"] >= 0.0

    def test_stop_is_idempotent_and_clean(self, tmp_path):
        sampler = _sampler(tmp_path, interval_s=0.01)
        sampler.start()
        time.sleep(0.05)
        sampler.stop()
        # A second stop() must not raise or hang.
        sampler.stop()
        assert sampler._thread is None

    def test_start_is_idempotent(self, tmp_path):
        sampler = _sampler(tmp_path, interval_s=0.01)
        sampler.start()
        first_thread = sampler._thread
        sampler.start()
        assert sampler._thread is first_thread
        sampler.stop()


class TestActiveSamplerRegistry:
    def test_mark_system_telemetry_is_a_noop_when_nothing_active(self, tmp_path):
        from eligsir.telemetry.system_sampler import mark_system_telemetry

        # Must never raise even with no sampler ever started in this process.
        mark_system_telemetry("mapping_endpoint")

    def test_start_stop_system_telemetry_round_trip(self, tmp_path):
        from eligsir.telemetry.system_sampler import (
            mark_system_telemetry,
            start_system_telemetry,
            stop_system_telemetry,
        )

        sampler = start_system_telemetry(tmp_path, interval_s=10.0)
        try:
            mark_system_telemetry("source_eof")
        finally:
            stop_system_telemetry(sampler)

        rows = _rows(tmp_path)
        assert any(r.get("event") == "source_eof" for r in rows)

    def test_start_system_telemetry_disabled_returns_none_and_writes_nothing(self, tmp_path):
        from eligsir.telemetry.system_sampler import (
            start_system_telemetry,
            stop_system_telemetry,
        )

        sampler = start_system_telemetry(tmp_path, enabled=False)
        assert sampler is None
        stop_system_telemetry(sampler)  # must be a no-op, not raise
        assert not (tmp_path / "system_performance.jsonl").exists()

    def test_stop_system_telemetry_clears_the_active_registry(self, tmp_path):
        from eligsir.telemetry import system_sampler as mod

        sampler = mod.start_system_telemetry(tmp_path, interval_s=10.0)
        mod.stop_system_telemetry(sampler)
        with mod._REGISTRY_LOCK:
            assert mod._ACTIVE_SAMPLER is None


def _write_synthetic_trace(tmp_path, rows):
    path = tmp_path / "system_performance.jsonl"
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class TestPlot:
    def test_plot_is_a_true_noop_without_a_trace(self, tmp_path):
        from eligsir.telemetry.system_sampler import plot_system_performance

        assert plot_system_performance(tmp_path) is None
        assert not (tmp_path / "system_performance.svg").exists()

    def test_plot_is_a_true_noop_for_an_empty_trace(self, tmp_path):
        from eligsir.telemetry.system_sampler import plot_system_performance

        (tmp_path / "system_performance.jsonl").write_text("")
        assert plot_system_performance(tmp_path) is None

    def test_plot_renders_a_vector_svg_with_synthetic_samples_and_markers(self, tmp_path):
        from eligsir.telemetry.system_sampler import plot_system_performance

        rows = [
            {
                "t_wall": 1000.0 + i,
                "t_elapsed_s": float(i),
                "cpu_pct_process": 10.0 + i,
                "cpu_pct_host": 20.0 + i,
                "rss_gb": 1.0 + 0.1 * i,
                "host_mem_used_gb": 8.0,
                "host_mem_total_gb": 32.0,
                "gpu_util_pct": None if i < 2 else 50.0 + i,
                "gpu_mem_used_gb": 2.0 + 0.05 * i,
                "gpu_mem_total_gb": 16.0,
                "gpu_gtt_used_gb": None,
                "gpu_power_w": 30.0 + i,
                "gpu_temp_c": 60.0,
                "gpu_source": "rocm_sysfs",
            }
            for i in range(10)
        ]
        rows.append({"event": "mapping_endpoint", "t_wall": 1005.0, "t_elapsed_s": 5.0})
        _write_synthetic_trace(tmp_path, rows)

        output = plot_system_performance(tmp_path)
        assert output is not None
        assert output.is_file()
        assert output.name == "system_performance.svg"
        assert output.stat().st_size > 0
        assert "<svg" in output.read_text()
        assert not (tmp_path / "system_performance.png").exists()


class TestSummary:
    def test_summary_is_a_true_noop_without_a_trace(self, tmp_path):
        from eligsir.telemetry.system_sampler import write_system_performance_summary

        assert write_system_performance_summary(tmp_path) is None
        assert not (tmp_path / "system_performance_summary.json").exists()

    def test_summary_splits_mapping_and_refinement_windows(self, tmp_path):
        from eligsir.telemetry.system_sampler import write_system_performance_summary

        rows = []
        # Mapping window: t=0..5, gpu_util 10-50, peak mem/rss here.
        for i in range(6):
            rows.append(
                {
                    "t_elapsed_s": float(i),
                    "gpu_util_pct": 10.0 * (i + 1),
                    "gpu_mem_used_gb": 5.0,
                    "rss_gb": 3.0,
                    "gpu_power_w": 40.0,
                }
            )
        rows.append({"event": "mapping_endpoint", "t_elapsed_s": 5.5})
        rows.append({"event": "refinement_start", "t_elapsed_s": 6.0})
        # Refinement window: t=6..8, gpu_util 90-100.
        for i, t in enumerate((6.5, 7.0, 7.5)):
            rows.append(
                {
                    "t_elapsed_s": t,
                    "gpu_util_pct": 90.0 + i * 5.0,
                    "gpu_mem_used_gb": 6.0,
                    "rss_gb": 3.5,
                    "gpu_power_w": 80.0,
                }
            )
        rows.append({"event": "refinement_end", "t_elapsed_s": 8.0})
        _write_synthetic_trace(tmp_path, rows)

        summary = write_system_performance_summary(tmp_path)
        assert summary is not None
        assert summary["mapping"]["n_samples"] == 6
        assert summary["mapping"]["mean_gpu_util_pct"] == pytest.approx(35.0)
        assert summary["mapping"]["max_gpu_util_pct"] == pytest.approx(60.0)
        assert summary["refinement"]["n_samples"] == 3
        assert summary["refinement"]["mean_gpu_util_pct"] == pytest.approx(95.0)
        assert summary["refinement"]["max_gpu_util_pct"] == pytest.approx(100.0)
        assert summary["peak_gpu_mem_used_gb"] == pytest.approx(6.0)
        assert summary["peak_rss_gb"] == pytest.approx(3.5)
        assert summary["mean_gpu_power_w"] == pytest.approx((40.0 * 6 + 80.0 * 3) / 9)

        on_disk = json.loads((tmp_path / "system_performance_summary.json").read_text())
        assert on_disk == summary

    def test_missing_refinement_markers_report_null_not_zero(self, tmp_path):
        from eligsir.telemetry.system_sampler import write_system_performance_summary

        rows = [
            {"t_elapsed_s": 0.0, "gpu_util_pct": 20.0, "gpu_mem_used_gb": 1.0, "rss_gb": 1.0, "gpu_power_w": 10.0},
            {"t_elapsed_s": 1.0, "gpu_util_pct": 30.0, "gpu_mem_used_gb": 1.5, "rss_gb": 1.2, "gpu_power_w": 12.0},
        ]
        _write_synthetic_trace(tmp_path, rows)

        summary = write_system_performance_summary(tmp_path)
        assert summary is not None
        # No mapping_endpoint/source_eof marker at all -> fall back to whole
        # trace for "mapping", but "refinement" has no start/end markers at
        # all and must report null, never a fabricated zero.
        assert summary["refinement"] == {
            "mean_gpu_util_pct": None,
            "max_gpu_util_pct": None,
            "n_samples": 0,
        }
        assert summary["mapping"]["n_samples"] == 2

    def test_zero_width_refinement_window_reports_null_not_zero(self, tmp_path):
        """requested_steps == 0: refinement_start/end fire back-to-back."""

        from eligsir.telemetry.system_sampler import write_system_performance_summary

        rows = [
            {"t_elapsed_s": 0.0, "gpu_util_pct": 20.0, "gpu_mem_used_gb": 1.0, "rss_gb": 1.0, "gpu_power_w": 10.0},
            {"event": "mapping_endpoint", "t_elapsed_s": 1.0},
            {"event": "refinement_start", "t_elapsed_s": 1.001},
            {"event": "refinement_end", "t_elapsed_s": 1.002},
        ]
        _write_synthetic_trace(tmp_path, rows)

        summary = write_system_performance_summary(tmp_path)
        assert summary is not None
        assert summary["refinement"]["n_samples"] == 0
        assert summary["refinement"]["mean_gpu_util_pct"] is None
        assert summary["refinement"]["max_gpu_util_pct"] is None


class TestWriteSystemPerformanceReport:
    def test_report_writes_both_plot_and_summary_and_never_raises_on_bad_trace(self, tmp_path):
        from eligsir.telemetry.system_sampler import write_system_performance_report

        (tmp_path / "system_performance.jsonl").write_text("not json\n{{{\n")
        # A corrupt trace must not raise -- the run this observes must never
        # be broken by its own telemetry.
        result = write_system_performance_report(tmp_path)
        assert result is None

    def test_report_produces_artifacts_for_a_real_sampler_run(self, tmp_path):
        from eligsir.telemetry.system_sampler import write_system_performance_report

        sampler = _sampler(tmp_path, interval_s=0.02)
        sampler.start()
        time.sleep(0.1)
        sampler.mark("mapping_endpoint")
        sampler.stop()

        result = write_system_performance_report(tmp_path)
        assert result is not None
        assert (tmp_path / "system_performance.svg").is_file()
        assert (tmp_path / "system_performance_summary.json").is_file()
