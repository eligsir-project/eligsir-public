"""Regression coverage for retained source-ingress report reading."""

from eligsir.runtime.source_ingress import source_ingress_warning


def test_historical_lossy_report_block_remains_loud():
    warning = source_ingress_warning(
        {
            "ingress_frames_dropped": 3,
            "ingress_dropped_at_arrival": 2,
            "ingress_dropped_at_enqueue": 1,
            "ingestion_policy": "fifo_drop_newest",
            "ingress_information_coverage": 0.75,
            "ingress_max_observed_buffer_lag_s": 0.2,
        }
    )
    assert warning is not None
    assert "dropped 3 frame(s)" in warning
    assert 'report.json["source_ingress"]' in warning


def test_preserved_or_absent_report_block_is_silent():
    assert source_ingress_warning(None) is None
    assert source_ingress_warning({"ingress_frames_dropped": 0}) is None
