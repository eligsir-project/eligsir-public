"""Regression coverage for the external SuperSplat-coordinate PLY copy."""

from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement
import subprocess
import sys


def _write_native_gaussian_ply(path):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        *[(f"f_rest_{index}", "f4") for index in range(9)],
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    vertices = np.zeros(1, dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = 1.0, 2.0, 3.0
    vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = 1.0, 2.0, 3.0
    for index in range(9):
        vertices[f"f_rest_{index}"] = float(index + 1)
    vertices["opacity"] = -0.2
    vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = 0.1, 0.2, 0.3
    vertices["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertices, "vertex")], comments=["active_sh_degree 1"]).write(path)


def test_supersplat_export_bakes_default_orientation_cancellation(tmp_path):
    from scripts.export_supersplat_ply import export_supersplat_ply
    from eligsir.viewers.rvizsplat_export import _quat_wxyz_to_rotmat

    source = tmp_path / "native.ply"
    output = tmp_path / "supersplat.ply"
    _write_native_gaussian_ply(source)

    metadata = export_supersplat_ply(source, output)
    result = PlyData.read(output).elements[0].data

    np.testing.assert_allclose(
        [result["x"][0], result["y"][0], result["z"][0]], [-1.0, -3.0, -2.0]
    )
    np.testing.assert_allclose(
        _quat_wxyz_to_rotmat(np.stack([result[f"rot_{index}"] for index in range(4)], axis=1)),
        np.asarray([[[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]]]),
        atol=1e-6,
    )
    assert not np.array_equal(
        [result[f"f_rest_{index}"][0] for index in range(9)], np.arange(1.0, 10.0)
    )
    assert metadata["basis"] == "supersplat_default_z180_to_xminus90_z0"


def test_supersplat_export_cli_runs_from_repository_root(tmp_path):
    source = tmp_path / "native.ply"
    output = tmp_path / "supersplat.ply"
    _write_native_gaussian_ply(source)

    completed = subprocess.run(
        [sys.executable, "scripts/export_supersplat_ply.py", str(source), str(output)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
