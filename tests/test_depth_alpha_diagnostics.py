import pytest

torch = pytest.importorskip("torch")

from eligsir.utils.depth_alpha_diagnostics import compute_alpha_bin_residuals, log_alpha_bin_residuals_to_tb


def test_bins_pixels_by_alpha_and_computes_mae():
    # 4 pixels: alpha 0.05 (bin0), 0.05 (bin0), 0.5 (bin3), 0.95 (bin5)
    alpha = torch.tensor([0.05, 0.05, 0.5, 0.95])
    sensor_d = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rend_d = torch.tensor([1.1, 2.2, 3.0, 4.0])  # errors: 0.1, 0.2, 0.0, 0.0
    rend_ed = torch.tensor([1.0, 2.0, 3.5, 4.5])  # errors: 0.0, 0.0, 0.5, 0.5

    stats = compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha)

    assert stats["n_valid_sensor"] == 4
    bin0 = stats["bins"][0]
    assert bin0["range"] == (0.0, 0.1)
    assert bin0["n"] == 2
    assert bin0["frac"] == pytest.approx(0.5)
    assert bin0["d_mae"] == pytest.approx((0.1 + 0.2) / 2)
    assert bin0["ed_mae"] == pytest.approx(0.0)

    bin3 = stats["bins"][3]  # [0.5, 0.7)
    assert bin3["n"] == 1
    assert bin3["d_mae"] == pytest.approx(0.0)
    assert bin3["ed_mae"] == pytest.approx(0.5)

    bin5 = stats["bins"][5]  # [0.9, 1.0+eps)
    assert bin5["n"] == 1
    assert bin5["d_mae"] == pytest.approx(0.0)
    assert bin5["ed_mae"] == pytest.approx(0.5)


def test_empty_bins_report_none_not_zero():
    alpha = torch.tensor([0.05])
    sensor_d = torch.tensor([1.0])
    rend_d = torch.tensor([1.0])
    rend_ed = torch.tensor([1.0])

    stats = compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha)

    for b in stats["bins"][1:]:
        assert b["n"] == 0
        assert b["frac"] == 0.0
        assert b["d_mae"] is None
        assert b["ed_mae"] is None


def test_invalid_sensor_pixels_excluded():
    # sensor_d <= 0 means no real sensor reading -- must not count toward n_valid_sensor
    alpha = torch.tensor([0.05, 0.05, 0.95])
    sensor_d = torch.tensor([0.0, -1.0, 3.0])
    rend_d = torch.tensor([1.0, 1.0, 3.0])
    rend_ed = torch.tensor([1.0, 1.0, 3.0])

    stats = compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha)

    assert stats["n_valid_sensor"] == 1
    assert stats["bins"][0]["n"] == 0
    assert stats["bins"][5]["n"] == 1


def test_zero_valid_sensor_pixels_gives_zero_fractions_no_crash():
    alpha = torch.tensor([0.05, 0.5])
    sensor_d = torch.tensor([0.0, 0.0])
    rend_d = torch.tensor([1.0, 1.0])
    rend_ed = torch.tensor([1.0, 1.0])

    stats = compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha)

    assert stats["n_valid_sensor"] == 0
    for b in stats["bins"]:
        assert b["frac"] == 0.0
        assert b["n"] == 0


def test_shapes_hw1_vs_1hw_both_work():
    # [1, H, W]-style and [H, W, 1]-style inputs should both flatten cleanly
    alpha_1hw = torch.tensor([[[0.05, 0.95]]])  # [1, 1, 2]
    sensor_d = torch.tensor([[[1.0, 2.0]]])
    rend_d = torch.tensor([[[1.0, 2.0]]])
    rend_ed = torch.tensor([[[1.5, 2.5]]])

    stats = compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha_1hw)
    assert stats["n_valid_sensor"] == 2
    assert stats["bins"][0]["n"] == 1
    assert stats["bins"][5]["n"] == 1


def test_does_not_require_grad_tracking_input():
    alpha = torch.tensor([0.5], requires_grad=True)
    sensor_d = torch.tensor([1.0], requires_grad=True)
    rend_d = torch.tensor([1.0], requires_grad=True)
    rend_ed = torch.tensor([1.0], requires_grad=True)

    # Must not raise even though inputs carry grad -- everything is detached
    # internally before any .item()/indexing.
    stats = compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha)
    assert stats["n_valid_sensor"] == 1


class _FakeTBWriter:
    def __init__(self):
        self.scalars = {}

    def add_scalar(self, tag, value, step):
        self.scalars[(tag, step)] = value


def test_log_alpha_bin_residuals_to_tb_skips_empty_bins():
    alpha = torch.tensor([0.05])
    sensor_d = torch.tensor([1.0])
    rend_d = torch.tensor([1.0])
    rend_ed = torch.tensor([1.0])
    stats = compute_alpha_bin_residuals(rend_d, rend_ed, sensor_d, alpha)

    tb = _FakeTBWriter()
    log_alpha_bin_residuals_to_tb(tb, stats, iteration=42)

    assert tb.scalars[("streaming/depth_alpha_bin/n_valid_sensor", 42)] == 1
    assert tb.scalars[("streaming/depth_alpha_bin/frac_0.00", 42)] == 1.0
    assert ("streaming/depth_alpha_bin/d_mae_0.00", 42) in tb.scalars
    # Bin [0.1, 0.3) is empty -- must not log a misleading d_mae/ed_mae of 0.0
    assert ("streaming/depth_alpha_bin/d_mae_0.10", 42) not in tb.scalars
    assert ("streaming/depth_alpha_bin/ed_mae_0.10", 42) not in tb.scalars
    # frac is still logged for empty bins (0.0 is the correct value there)
    assert tb.scalars[("streaming/depth_alpha_bin/frac_0.10", 42)] == 0.0
