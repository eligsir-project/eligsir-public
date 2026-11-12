"""WS6: process-pool CPU eval workers -- serial vs parallel must agree."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.batch_shape_stats import compute_batch

plyfile = pytest.importorskip("plyfile")


def _write_fixture_ply(path, seed: int, n: int = 20) -> None:
    rng = np.random.RandomState(seed)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("opacity", "f4"),
    ]
    data = np.zeros(n, dtype=dtype)
    data["x"] = rng.randn(n)
    data["y"] = rng.randn(n)
    data["z"] = rng.randn(n)
    data["scale_0"] = rng.randn(n) * 0.1 - 2.0
    data["scale_1"] = rng.randn(n) * 0.1 - 2.0
    data["scale_2"] = rng.randn(n) * 0.1 - 3.0
    data["opacity"] = rng.randn(n) * 0.5
    el = plyfile.PlyElement.describe(data, "vertex")
    plyfile.PlyData([el]).write(str(path))


@pytest.fixture
def fixture_plys(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"fixture_{i}.ply"
        _write_fixture_ply(p, seed=i)
        paths.append(str(p))
    return paths


def test_serial_default_matches_no_workers_arg(fixture_plys):
    a = compute_batch(fixture_plys)
    b = compute_batch(fixture_plys, workers=0)
    assert a == b


def test_serial_and_parallel_produce_identical_results(fixture_plys):
    serial = compute_batch(fixture_plys, workers=1)
    parallel = compute_batch(fixture_plys, workers=2)
    assert [r["path"] for r in serial] == [r["path"] for r in parallel]
    for s, p in zip(serial, parallel):
        assert s["ok"] and p["ok"]
        assert s["stats"] == p["stats"]


def test_failure_is_reported_per_path_not_fatal(tmp_path, fixture_plys):
    bad_path = str(tmp_path / "does_not_exist.ply")
    paths = fixture_plys + [bad_path]
    results = compute_batch(paths, workers=2)
    assert len(results) == len(paths)
    ok_flags = {r["path"]: r["ok"] for r in results}
    assert ok_flags[bad_path] is False
    assert all(ok_flags[p] for p in fixture_plys)
