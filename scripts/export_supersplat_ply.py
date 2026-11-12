"""Write a native Gaussian PLY copy aligned to the SuperSplat viewer basis.

EliGSiR retains canonical map coordinates in its normal ``point_cloud.ply``.
This exporter writes a separate copy whose rigid basis makes SuperSplat's
default Z=180 degree orientation display as X=-90 degrees, Z=0 degrees.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

# ``python scripts/export_supersplat_ply.py`` makes ``scripts/`` the import
# root; expose the repository package just as ``python -m`` would.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.viewers.rvizsplat_export import (
    RVizSplatSnapshot,
    transform_snapshot_basis,
)


# Rz(180) @ Rx(-90): default SuperSplat Z=180 then yields Rx(-90).
SUPERSPLAT_DEFAULT_BASIS = np.asarray(
    ((-1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0)),
    dtype=np.float64,
)
SUPERSPLAT_BASIS_NAME = "supersplat_default_z180_to_xminus90_z0"


def _indexed_fields(names: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    fields = [name for name in names if name.startswith(prefix)]
    return tuple(sorted(fields, key=lambda name: int(name.removeprefix(prefix))))


def _snapshot_from_native_vertex(vertex: Any) -> tuple[RVizSplatSnapshot, tuple[str, ...]]:
    names = tuple(vertex.dtype.names or ())
    required = {"x", "y", "z", "opacity", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"native Gaussian PLY is missing fields: {', '.join(missing)}")
    dc_fields = _indexed_fields(names, "f_dc_")
    rest_fields = _indexed_fields(names, "f_rest_")
    if len(dc_fields) != 3 or len(rest_fields) % 3:
        raise ValueError("native Gaussian PLY has an invalid SH field layout")
    coefficient_count = 1 + len(rest_fields) // 3
    degree = int(round(np.sqrt(coefficient_count))) - 1
    if degree < 0 or (degree + 1) ** 2 != coefficient_count:
        raise ValueError("native Gaussian PLY has a non-square SH coefficient count")
    positions = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).astype(np.float32)
    quaternions = np.stack([vertex[f"rot_{index}"] for index in range(4)], axis=1).astype(np.float32)
    scale_fields = _indexed_fields(names, "scale_")
    scales = (
        np.stack([vertex[field] for field in scale_fields], axis=1).astype(np.float32)
        if len(scale_fields) == 3
        else np.zeros_like(positions)
    )
    dc = np.stack([vertex[field] for field in dc_fields], axis=1).astype(np.float32)
    rest = np.stack([vertex[field] for field in rest_fields], axis=1).astype(np.float32)
    # Native PLY is channel-major; RVizSplatSnapshot is coefficient-major RGB.
    features = np.concatenate(
        (dc[:, None, :], rest.reshape(len(vertex), 3, -1).transpose(0, 2, 1)), axis=1
    )
    return (
        RVizSplatSnapshot(
            positions=positions,
            quaternions=quaternions,
            scales=scales,
            opacities=np.asarray(vertex["opacity"], dtype=np.float32),
            spherical_harmonics=features.reshape(len(vertex), -1),
            sh_degree=degree,
        ),
        rest_fields,
    )


def export_supersplat_ply(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Copy native Gaussian PLY ``source`` into the SuperSplat display basis."""

    source, output = Path(source), Path(output)
    ply = PlyData.read(source)
    if "vertex" not in ply:
        raise ValueError("native Gaussian PLY has no vertex element")
    vertex = ply["vertex"].data
    snapshot, rest_fields = _snapshot_from_native_vertex(vertex)
    transformed = transform_snapshot_basis(snapshot, SUPERSPLAT_DEFAULT_BASIS)
    result = vertex.copy()
    result["x"], result["y"], result["z"] = transformed.positions.T
    for index in range(4):
        result[f"rot_{index}"] = transformed.quaternions[:, index]
    result["f_dc_0"], result["f_dc_1"], result["f_dc_2"] = transformed.spherical_harmonics[:, :3].T
    if rest_fields:
        rest = transformed.spherical_harmonics[:, 3:].reshape(len(result), -1, 3)
        native_rest = rest.transpose(0, 2, 1).reshape(len(result), -1)
        for index, field in enumerate(rest_fields):
            result[field] = native_rest[:, index]
    comments = [*ply.comments, f"coordinate_basis {SUPERSPLAT_BASIS_NAME}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(result, "vertex")], comments=comments).write(output)
    return {"basis": SUPERSPLAT_BASIS_NAME, "gaussians": int(len(result)), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="native EliGSiR Gaussian point_cloud.ply")
    parser.add_argument("output", type=Path, help="separate SuperSplat-coordinate PLY copy")
    args = parser.parse_args()
    metadata = export_supersplat_ply(args.source, args.output)
    print(f"[supersplat-export] {metadata['gaussians']} Gaussians -> {metadata['output']}")


if __name__ == "__main__":
    main()
