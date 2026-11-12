"""Data contracts for the prepared-candidate pipeline (plan section 8.3).

Three stages, three frozen dataclasses, one direction of travel:

``RawFramePacket`` (arrived at the source, nothing decoded yet)
  -> ``TrackedFramePacket`` (a pose/tracking-state has been attached by the
     SLAM frontend, still nothing decoded)
  -> ``PreparedFrameCandidate`` (CPU-side preparation has run: decode,
     resize, blur/exposure/valid-depth-fraction, cheap pose novelty --
     see ``eligsir.mapping.candidates.preparation``).

None of these types touch the GPU or torch. ``PreparedFrameCandidate``
deliberately carries CPU-resident numpy buffers, not GPU tensors and not a
``scene.cameras.Camera`` -- full-resolution GPU supervision-image
materialization is a separate, later step that only ever runs for an
ADMITTED candidate (see ``eligsir.mapping.candidates.prefilter`` and
the plan's non-negotiable deliverable #3: a rejected candidate must never
allocate a full-resolution GPU training image).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from eligsir.utils.rgbd_frames import CameraIntrinsics
from eligsir.contracts.types import Pose


class TrackingState(str, Enum):
    """Coarse SLAM-frontend tracking state attached to a raw frame.

    Deliberately smaller than ``eligsir.mapping.scheduler.
    tracking_confidence.TrackingStatus`` (which mirrors ORB-SLAM3's own
    numeric state machine byte-for-byte for that adapter's veto logic) --
    this is a frontend-agnostic summary any tracker can populate.
    """

    OK = "ok"
    LOST = "lost"
    INITIALIZING = "initializing"
    REINITIALIZED = "reinitialized"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UndecodedBufferRef:
    """A reference to an undecoded RGB or depth buffer -- never decoded
    pixels. Exactly one of ``path``/``data``/``array`` should be set;
    ``array`` covers sources that already hold decoded data in memory
    (e.g. ``StreamingRGBDFrame._rgb_np``/``_depth_np``, a live ROS2 frame)
    without forcing a round-trip through bytes or disk.

    DELIBERATELY NOT :class:`eligsir.contracts.runtime.BufferRef`
    (issue #1074). This type was named ``BufferRef`` until 2026-08-23, which
    made it read like a duplicate of the canonical record and put it on
    #1074's "migrate every consumer to the canonical family" list. It is not
    a duplicate -- the two express opposite halves of the decode boundary,
    and unifying them is a real design change, not a rename:

    * **This type is a promise; the canonical one is a payload.** It exists
      precisely to name a buffer whose bytes have NOT been read yet
      (``path=``) or that is absent entirely (``UndecodedBufferRef()`` for a
      frame with no depth). The canonical ``BufferRef`` requires a concrete
      :class:`~eligsir.contracts.runtime.BufferSpec`, whose
      ``__post_init__`` rejects an empty shape and an empty
      ``dtype``/``encoding`` -- so neither the path-only nor the empty case
      is constructible as a canonical ``BufferRef`` at all.
    * **Unifying would force eager decode on the ingest thread.** Making
      these path-only refs canonical means decoding at packet-construction
      time, which both violates #1074's own performance requirement (no
      eager conversion when a native view can survive to materialization)
      and moves image decode off the preparation worker onto the live
      ingest path -- the exact cost this package's three-stage pipeline was
      built to avoid (see this module's docstring).

    The canonical ``BufferRef`` is the right type for a buffer that has
    already been decoded/handed over with known layout and ownership. Should
    this pipeline ever want that, the conversion belongs at the
    ``decode_rgb``/``decode_depth_m`` boundary in ``preparation.py``, where
    real shape/dtype/encoding first exist -- not here.

    Note that boundary is POST-scaling: ``decode_depth_m`` already divides by
    ``depth_scale`` and returns float32 metres, so a canonical ref built there
    carries ``encoding="32FC1", units="meters"`` -- not raw sensor counts.
    That matters because ``materialize._depth_scale`` treats ``units="meters"``
    as "already metric" and correctly no-ops rather than dividing a second
    time; handing it a counts-labelled buffer instead would double-scale.
    """

    path: Optional[str] = None
    data: Optional[bytes] = None
    array: Optional[np.ndarray] = None

    def is_empty(self) -> bool:
        return self.path is None and self.data is None and self.array is None


@dataclass(frozen=True)
class RawFramePacket:
    """A frame as it arrives at the source: identity, timing, undecoded
    buffers, and intrinsics. No pose yet -- that is attached by the SLAM
    frontend to produce a :class:`TrackedFramePacket`."""

    source_index: int
    timestamp_ns: int
    rgb_ref: UndecodedBufferRef
    depth_ref: UndecodedBufferRef
    intrinsics: CameraIntrinsics


@dataclass(frozen=True)
class TrackedFramePacket:
    """A :class:`RawFramePacket` with a SLAM pose attached. Still nothing
    decoded -- ``world_T_camera`` is the only new information, cheap
    enough to gate a coarse pose-novelty prefilter reject (see
    ``prefilter.should_skip_decode``) before ever touching ``frame``'s
    buffers.
    """

    frame: RawFramePacket
    world_T_camera: np.ndarray
    tracking_state: TrackingState
    tracking_confidence: float
    slam_map_id: int


@dataclass(frozen=True)
class PreparedImageRef:
    """CPU-resident RGB buffer produced by the preparation worker.

    ``array`` is intentionally not a GPU tensor and not wrapped in a
    ``scene.cameras.Camera`` -- see this module's docstring. It may be a
    resolution-normalized thumbnail (``downsampled=True``) when the
    preparation worker only needed it for quality scoring, not for actual
    training supervision.
    """

    array: Optional[np.ndarray]
    width: int
    height: int
    downsampled: bool = False


@dataclass(frozen=True)
class PreparedDepthRef:
    """CPU-resident depth buffer (meters, float32) produced by the
    preparation worker, plus the valid-fraction summary admission needs
    without re-scanning the array."""

    array: Optional[np.ndarray]
    width: int
    height: int
    valid_fraction: float


@dataclass(frozen=True)
class InputEvidence:
    """Cheap, deterministic per-frame quality signal computed off the
    optimizer thread. Mirrors ``eligsir.mapping.scheduler.
    frame_quality.FrameQuality`` (reused, not reimplemented -- see
    ``preparation.py``) but only carries the fields the prefilter and
    admission scoring actually need downstream.
    """

    blur_laplacian: float
    exposure_luma: float
    white_clip_ratio: float
    black_clip_ratio: float
    valid_depth_fraction: float
    quality_weight: float
    reject_reason: Optional[str] = None


@dataclass(frozen=True)
class PreparedFrameCandidate:
    """Output of the preparation worker: everything admission scoring
    needs, with no full-resolution GPU image materialized yet."""

    source_index: int
    timestamp_ns: int
    pose: Pose
    rgb_ref: PreparedImageRef
    depth_ref: PreparedDepthRef
    input_evidence: InputEvidence
    coarse_novelty: float
