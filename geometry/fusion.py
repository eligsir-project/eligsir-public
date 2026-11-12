"""Fusion rules for sensor and optional learned geometry priors."""

from __future__ import annotations

import torch

from geometry.priors import GeometrySource, SurfacePrior


def fuse_surface_priors(
    sensor: SurfacePrior | None,
    learned: SurfacePrior | None,
    *,
    sensor_trust_threshold: float = 0.5,
    learned_trust_threshold: float = 0.7,
) -> SurfacePrior:
    """Fuse priors with conservative sensor-first semantics."""

    if sensor is None and learned is None:
        raise ValueError("At least one prior is required")
    if sensor is None:
        return learned
    if learned is None:
        return sensor
    if sensor.depth.shape != learned.depth.shape:
        raise ValueError("SurfacePrior depth shapes must match for fusion")

    sensor_trusted = sensor.valid & (sensor.confidence >= sensor_trust_threshold)
    learned_ok = learned.valid & (learned.confidence >= learned_trust_threshold)
    use_learned = (~sensor_trusted) & learned_ok

    depth = torch.where(use_learned, learned.depth, sensor.depth)
    confidence = torch.where(use_learned, learned.confidence * 0.75, sensor.confidence)
    valid = sensor_trusted | use_learned
    normals = None
    if sensor.normals is not None and learned.normals is not None:
        normals = torch.where(use_learned.expand_as(sensor.normals), learned.normals, sensor.normals)
    elif sensor.normals is not None:
        normals = sensor.normals
    elif learned.normals is not None:
        normals = learned.normals
    uncertainty = 1.0 - confidence
    source = torch.full_like(sensor.source if sensor.source is not None else sensor.depth.to(torch.int16), int(GeometrySource.SENSOR))
    source = torch.where(use_learned, torch.full_like(source, int(GeometrySource.VGGT)), source)
    source = torch.where(sensor_trusted & learned_ok, torch.full_like(source, int(GeometrySource.FUSED)), source)
    return SurfacePrior(
        depth=depth,
        confidence=confidence,
        valid=valid,
        normals=normals,
        uncertainty=uncertainty,
        source=source,
        frame_uid=sensor.frame_uid,
        normal_frame=sensor.normal_frame,
    )
