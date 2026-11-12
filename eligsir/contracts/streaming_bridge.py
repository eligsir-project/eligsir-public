"""Bridge legacy streaming records into the canonical runtime ownership contract.

The bridge is intentionally CPU-only. Admission stays lazy: rejected candidate
frames never decode pixels, while an admitted frame is represented as a
``SensorPacket`` and then a ``MappingFrame`` before camera supervision consumes
it. ``DeviceTransferOwner`` remains the owner of the later grouped image H2D.
"""

from __future__ import annotations

import numpy as np

from eligsir.contracts.materialize import (
    ColorPolicy,
    DepthPolicy,
    MaterializationPolicy,
    materialize_mapping_frame,
    torch_backend,
)
from eligsir.contracts.runtime import (
    BufferOwnership,
    BufferRef,
    BufferSpec,
    MemoryDomain,
    SensorPacket,
    SourceIdentity,
    TrackingObservation,
    TrackingStatus,
)
from eligsir.contracts.types import CameraIntrinsics, Pose


def frame_source_identity(frame) -> SourceIdentity:
    ordinal = getattr(frame, "source_ordinal", None)
    if ordinal is None:
        ordinal = int(frame.index)
    source_id = str(
        getattr(frame, "source_id", "")
        or getattr(frame, "rgb_path", "")
        or "streaming-rgbd"
    )
    timestamp_ns = int(round(float(frame.timestamp) * 1_000_000_000.0))
    return SourceIdentity(
        source_id=source_id,
        stream_epoch=int(getattr(frame, "stream_epoch", 0) or 0),
        source_ordinal=int(ordinal),
        timestamp_ns=timestamp_ns,
    )


def _intrinsics(frame, *, width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=float(frame.fx),
        fy=float(frame.fy),
        cx=float(frame.cx),
        cy=float(frame.cy),
        width=int(width),
        height=int(height),
        depth_fx=(float(frame.depth_fx) if frame.depth_fx is not None else None),
        depth_fy=(float(frame.depth_fy) if frame.depth_fy is not None else None),
        depth_cx=(float(frame.depth_cx) if frame.depth_cx is not None else None),
        depth_cy=(float(frame.depth_cy) if frame.depth_cy is not None else None),
        depth_width=(int(frame.depth_width) if frame.depth_width is not None else None),
        depth_height=(int(frame.depth_height) if frame.depth_height is not None else None),
        depth_scale=float(frame.depth_scale),
        intrinsics_source="streaming_frame",
    )


def mapping_frame_from_streaming_frame(frame, rgb_chw, depth_raw=None):
    """Build the CPU ``SensorPacket -> MappingFrame`` representation.

    ``rgb_chw`` is the already-decoded CHW tensor produced by the lazy camera
    loader. ``depth_raw`` may be a NumPy raw-depth image; when present its unit
    conversion is performed by the canonical materializer. No device upload is
    performed here.
    """

    import torch

    if not isinstance(rgb_chw, torch.Tensor) or rgb_chw.device.type != "cpu":
        raise ValueError("streaming bridge requires CPU RGB so batch prefetch owns H2D")
    if rgb_chw.ndim != 3 or rgb_chw.shape[0] < 3:
        raise ValueError(f"expected CHW RGB tensor, got {tuple(rgb_chw.shape)}")

    rgb_chw = rgb_chw[:3].contiguous()
    source = frame_source_identity(frame)
    h, w = int(rgb_chw.shape[1]), int(rgb_chw.shape[2])
    intr = _intrinsics(frame, width=w, height=h)
    rgb_ref = BufferRef(
        data=rgb_chw,
        spec=BufferSpec(
            shape=tuple(int(v) for v in rgb_chw.shape),
            dtype=str(rgb_chw.dtype).replace("torch.", ""),
            memory_domain=MemoryDomain.HOST,
            encoding="rgb32f_chw",
            contiguous=True,
        ),
        ownership=BufferOwnership.OWNED,
    )
    depth_ref = None
    if depth_raw is not None:
        depth_np = np.asarray(depth_raw)
        # The units tag decides whether the materializer divides by
        # intrinsics.depth_scale. Tagging unconditionally "raw" double-scales
        # any source that already hands back metric depth: a 2.0 m float32
        # frame would materialize as 0.002 m at depth_scale=1000, silently
        # placing every backprojected point at the wrong range. This mirrors
        # utils.rgbd_frames.depth_to_meters, which likewise divides integer
        # counts and passes float input through untouched.
        is_metric = depth_np.dtype in (np.float32, np.float64)
        depth_ref = BufferRef.from_numpy(
            depth_np,
            encoding="depth_metric" if is_metric else "depth_raw",
            units="m" if is_metric else "raw",
        )

    tracking = None
    if getattr(frame, "c2w", None) is not None:
        tracking = TrackingObservation(
            source=source,
            pose=Pose(np.asarray(frame.c2w, dtype=np.float32)),
            status=TrackingStatus.VALID,
            provider="streaming_source_pose",
            confidence=1.0,
        )

    packet = SensorPacket(
        source=source,
        rgb=rgb_ref,
        depth=depth_ref,
        intrinsics=intr,
        tracking=tracking,
    )
    return materialize_mapping_frame(
        packet,
        MaterializationPolicy(
            device="cpu",
            color=ColorPolicy.NORMALIZED_FLOAT32,
            depth=DepthPolicy.METERS_FLOAT32,
        ),
        backend=torch_backend(torch),
    )


def _cpu_numpy(payload, *, name: str) -> np.ndarray:
    """Return a CPU array view and reject an implicit device readback."""

    if isinstance(payload, np.ndarray):
        return payload
    device = getattr(payload, "device", None)
    if device is not None and getattr(device, "type", str(device).split(":", 1)[0]) != "cpu":
        raise ValueError(f"MappingFrame {name} must remain CPU-owned until transfer")
    detach = getattr(payload, "detach", None)
    if callable(detach):
        payload = detach()
    numpy = getattr(payload, "numpy", None)
    return numpy() if callable(numpy) else np.asarray(payload)


def mapping_frame_source_arrays(
    frame,
    *,
    required: bool = True,
    require_pose: bool = True,
):
    """Return canonical insertion ``(metric_depth, HWC-float RGB)`` arrays.

    Production insertion is a hard boundary: RGB, registered metric depth,
    calibration and pose must all come from the same ``MappingFrame``.  Missing
    or mismatched data raises instead of selecting a decode/reload path.  The
    explicit ``required=False`` compatibility mode exists only for the retired
    host planner while its parity fixtures remain in the repository.
    """

    mapping = getattr(frame, "mapping_frame", None)
    if mapping is None:
        mapping = getattr(frame, "_mapping_frame", None)
    if mapping is None:
        if required:
            raise RuntimeError("canonical insertion requires a canonical MappingFrame")
        return None
    if str(mapping.device) != "cpu":
        raise ValueError("canonical insertion MappingFrame must be CPU-owned")
    if require_pose and mapping.pose is None:
        raise ValueError("canonical insertion MappingFrame requires a metric pose")

    rgb = _cpu_numpy(mapping.rgb.data, name="RGB")
    if rgb.ndim != 3:
        raise ValueError(f"MappingFrame RGB must be rank 3, got {rgb.shape}")
    if rgb.shape[-1] in (3, 4):
        rgb_hwc = rgb[..., :3]
    elif rgb.shape[0] in (3, 4):
        rgb_hwc = np.moveaxis(rgb[:3], 0, -1)
    else:
        raise ValueError(f"MappingFrame RGB has no three-channel axis: {rgb.shape}")
    if np.issubdtype(rgb_hwc.dtype, np.integer):
        scale = float(np.iinfo(rgb_hwc.dtype).max)
        rgb_hwc = rgb_hwc.astype(np.float32) / scale
    else:
        rgb_hwc = rgb_hwc.astype(np.float32, copy=False)

    intrinsics = mapping.intrinsics
    target_hw = (int(intrinsics.height), int(intrinsics.width))
    if tuple(rgb_hwc.shape[:2]) != target_hw:
        raise ValueError(
            "MappingFrame RGB resolution disagrees with canonical intrinsics: "
            f"rgb={tuple(rgb_hwc.shape[:2])} intrinsics={target_hw}"
        )
    frame_hw = (int(frame.height), int(frame.width))
    if frame_hw != target_hw:
        raise ValueError(
            "mapper-local frame resolution disagrees with canonical intrinsics: "
            f"frame={frame_hw} intrinsics={target_hw}"
        )
    if mapping.depth is None:
        raise ValueError("canonical insertion MappingFrame requires metric depth")
    units = (mapping.depth.spec.units or "").lower()
    if units not in {"m", "meter", "meters", "metre", "metres"}:
        raise ValueError("MappingFrame insertion depth must already be metric")
    depth = _cpu_numpy(mapping.depth.data, name="depth")
    if depth.ndim != 2:
        raise ValueError(f"MappingFrame depth must be rank 2, got {depth.shape}")
    if tuple(depth.shape) != target_hw:
        raise ValueError(
            "MappingFrame RGB/depth resolution mismatch: "
            f"rgb={target_hw} depth={tuple(depth.shape)}"
        )
    depth_hw = (int(intrinsics.depth_height), int(intrinsics.depth_width))
    if depth_hw != tuple(depth.shape):
        raise ValueError(
            "MappingFrame depth resolution disagrees with canonical intrinsics: "
            f"depth={tuple(depth.shape)} intrinsics={depth_hw}"
        )
    calibration = (
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        intrinsics.depth_fx,
        intrinsics.depth_fy,
        intrinsics.depth_cx,
        intrinsics.depth_cy,
    )
    if not all(np.isfinite(float(value)) for value in calibration):
        raise ValueError("canonical insertion intrinsics must be finite")
    if min(float(intrinsics.fx), float(intrinsics.fy),
           float(intrinsics.depth_fx), float(intrinsics.depth_fy)) <= 0.0:
        raise ValueError("canonical insertion focal lengths must be positive")
    # Equal raster dimensions do not establish registration.  The canonical
    # contract currently carries no depth-to-colour transform or registered
    # pixel-map provenance, so accepting distinct calibration here would pair
    # a depth sample with the wrong RGB ray.  A future contract may relax this
    # only by carrying and validating that explicit registration mapping.
    color_calibration = np.asarray(
        (intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy),
        dtype=np.float64,
    )
    depth_calibration = np.asarray(
        (
            intrinsics.depth_fx,
            intrinsics.depth_fy,
            intrinsics.depth_cx,
            intrinsics.depth_cy,
        ),
        dtype=np.float64,
    )
    if not np.allclose(
        color_calibration, depth_calibration, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            "canonical insertion requires registered RGB/depth calibration "
            "with shared pixel geometry"
        )
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32, copy=False)
    return depth, np.ascontiguousarray(rgb_hwc)


__all__ = ["mapping_frame_from_streaming_frame", "mapping_frame_source_arrays"]
