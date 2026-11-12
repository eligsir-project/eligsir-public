"""Immutable, versioned canonical preview event schemas (issue #570, phase A).

Canonical design: ``docs/architecture/2026-08-14-canonical-live-viewer-sidecar.md``
(§4 "Canonical preview event model"). Parent program: #569.

**Nothing in this module is wired into ``train.py`` / ``train_streaming.py`` /
``eligsir/viewers/registry.py``.** No default anywhere in the repo changes
because this file exists. See the PR body for exactly which of #570's
acceptance criteria that leaves undischarged.

Why these types exist
---------------------
The defect this schema is built to make *unconstructible* is §1.2 of the design
doc: today the newest source RGB/depth and the current optimizer render reach
the viewer through **separate** calls, so a UI can place ``source frame 731``
next to ``render of replay frame 122`` and imply they are a matched comparison.

The fix is not a docstring and not a runtime assertion somewhere downstream. It
is that :class:`TrainingPairPreview` takes **two independently stamped images**
(:class:`CapturedImage`, each carrying its own ``camera_id`` /
``source_frame_id`` / shape / :class:`ImageKind`) and refuses, in
``__post_init__``, any pair whose two halves disagree. A schema with a single
``camera_id: int`` field and two bare arrays cannot express the mismatch at all,
which means it also cannot reject it — the check would be vacuous theatre. The
mismatch must be *expressible at the input layer* for the refusal at the event
layer to mean anything, and ``tests/test_canonical_preview_events.py`` asserts
exactly that.

The same principle separates :class:`RawInputPreview` from
:class:`TrainingPairPreview` structurally rather than by convention: the raw
event has no ``render`` field and does not accept a :class:`CapturedImage` at
all, so "the newest sensor frame" and "the matched optimizer pair" are different
types with no overlapping constructor. A sink cannot confuse them by accident.

Array ownership rule: FREEZE IN PLACE, submission transfers ownership
---------------------------------------------------------------------
``@dataclass(frozen=True)`` blocks attribute *rebinding*; it does nothing about
``event.target.array[0, 0] = 255``. Every array accepted here therefore has
``flags.writeable`` cleared in ``__post_init__``.

That is a **freeze in place**, not a copy: constructing an event transfers
ownership of the array to the event. The caller must not retain a mutable
reference. This is deliberate — the canonical path is advertised as
zero-extra-copy (design doc §5, §13), and the arrays it is designed to carry are
the viewer's *own* staged host buffers (§5.1's fixed preview staging ring, owned
by #571), not tensors the trainer is still writing into. A caller that cannot
transfer ownership must use :meth:`CapturedImage.copy_of`, which copies first and
then freezes, and pays for the copy explicitly rather than silently.

Honest limit: clearing ``writeable`` on a *view* does not freeze the view's base
array. Pass an array that owns its data (or a ``.copy()``) when the source buffer
is reused.

No silent zero
--------------
Mirrors ``utils/causal_quality.MetricValue``: a quantity that was not measured is
``None``, never ``0``. ``psnr_db``, ``map_version``, ``source_frame_id`` and the
``frames_*`` counters are all ``Optional`` and default to ``None``.

This is a distinguishability rule, **not** a ban on zero: ``frames_admitted == 0``
is a real, correct measurement early in a run and is accepted. What is refused is
``0`` standing in for "unknown", and negative values standing in for "absent".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PREVIEW_SCHEMA_VERSION",
    "PreviewSchemaError",
    "PreviewChannel",
    "ImageKind",
    "CameraRole",
    "MappingPhase",
    "CapturedImage",
    "PreviewEvent",
    "TrainingPairPreview",
    "RawInputPreview",
    "TrajectoryPreview",
    "KeyframePreview",
    "MetricsPreview",
    "RuntimeTelemetryPreview",
    "MapPreviewSnapshot",
    "channel_for_event",
    "EVENT_TYPE_BY_CHANNEL",
]

#: Bumped whenever any event's field set or invariants change. Every event
#: instance carries it (``schema_version``) so a sidecar/recorder that outlives
#: a trainer upgrade can refuse a payload it does not understand rather than
#: mis-parse it.
PREVIEW_SCHEMA_VERSION = 1


class PreviewSchemaError(ValueError):
    """A preview event was constructed with a self-contradictory payload."""


class PreviewChannel(Enum):
    """Independent latest-only bus slots (design doc §3.2).

    Separate slots per *semantic* channel, so frequent camera frames cannot
    evict a rare map snapshot.
    """

    TRAINING_PAIR = "training_pair"
    RAW_INPUT = "raw_input"
    TRAJECTORY = "trajectory"
    KEYFRAME = "keyframe"
    METRICS = "metrics"
    RUNTIME_TELEMETRY = "runtime_telemetry"
    MAP_SNAPSHOT = "map_snapshot"


class ImageKind(Enum):
    """Which half of a supervision operation an image is.

    Load-bearing: :class:`TrainingPairPreview` refuses a pair whose ``target``
    is not ``TARGET`` or whose ``render`` is not ``RENDER``, so the two halves
    cannot be swapped even when their camera ids happen to agree.
    """

    TARGET = "target"
    RENDER = "render"


class CameraRole(Enum):
    """Why the optimizer selected this camera (issue #570's closed enum).

    A UI that shows a ``REPLAY``/``GLOBAL`` pair without labelling it as such
    reads as a jump in the source stream; the role is what lets a sink say
    otherwise (design doc §15 "UI/semantic tests").
    """

    NEW = "NEW"
    LOCAL = "LOCAL"
    REPLAY = "REPLAY"
    GLOBAL = "GLOBAL"
    OTHER = "OTHER"


class MappingPhase(Enum):
    """Coarse training phase at capture time."""

    BOOTSTRAP = "BOOTSTRAP"
    ACTIVE_MAPPING = "ACTIVE_MAPPING"
    REFINEMENT = "REFINEMENT"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _freeze_array(
    array: object,
    name: str,
    *,
    allowed_ndim: Tuple[int, ...],
) -> np.ndarray:
    """Validate shape/finiteness-of-dtype and clear ``writeable`` in place."""
    if not isinstance(array, np.ndarray):
        raise PreviewSchemaError(
            f"{name} must be a numpy.ndarray (preview payloads are plain CPU "
            f"arrays; a live tensor/model reference must never reach a sink), "
            f"got {type(array).__name__}"
        )
    if array.ndim not in allowed_ndim:
        raise PreviewSchemaError(
            f"{name} must have ndim in {allowed_ndim}, got shape {array.shape}"
        )
    if array.size == 0:
        raise PreviewSchemaError(
            f"{name} is empty; an absent image must be None, not a zero-sized array"
        )
    try:
        array.flags.writeable = False
    except ValueError as exc:  # pragma: no cover - non-owning read-only base
        raise PreviewSchemaError(
            f"{name} could not be frozen (submission transfers ownership; see "
            f"this module's array-ownership rule): {exc}"
        ) from exc
    return array


def _check_optional_count(value: Optional[int], name: str) -> Optional[int]:
    if value is None:
        return None
    value = int(value)
    if value < 0:
        raise PreviewSchemaError(
            f"{name} must be >= 0 or None (absence is None, never a negative "
            f"sentinel), got {value}"
        )
    return value


def _check_count(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise PreviewSchemaError(f"{name} must be >= 0, got {value}")
    return value


def _check_enum(value: object, enum_cls: type, name: str):
    if not isinstance(value, enum_cls):
        raise PreviewSchemaError(
            f"{name} must be a {enum_cls.__name__} member (closed enum), got "
            f"{value!r}"
        )
    return value


def _check_optional_finite(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise PreviewSchemaError(f"{name} must be finite or None, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# CapturedImage -- the stamped half of a pair
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CapturedImage:
    """One image plus the identity of what it depicts.

    Carrying identity *on the image* rather than once on the containing event
    is the whole point: it is what makes a mismatched pair expressible as an
    input and therefore refusable as an event.
    """

    array: np.ndarray
    camera_id: int
    kind: ImageKind
    source_frame_id: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "array", _freeze_array(self.array, "CapturedImage.array", allowed_ndim=(2, 3))
        )
        _check_enum(self.kind, ImageKind, "CapturedImage.kind")
        object.__setattr__(self, "camera_id", _check_count(self.camera_id, "camera_id"))
        object.__setattr__(
            self,
            "source_frame_id",
            _check_optional_count(self.source_frame_id, "source_frame_id"),
        )

    @classmethod
    def copy_of(
        cls,
        array: np.ndarray,
        camera_id: int,
        kind: ImageKind,
        source_frame_id: Optional[int] = None,
    ) -> "CapturedImage":
        """Copy-then-freeze, for a caller that cannot transfer ownership.

        Costs one host-side copy. Prefer handing over a staged buffer instead;
        see this module's array-ownership rule.
        """
        if not isinstance(array, np.ndarray):
            raise PreviewSchemaError(
                f"CapturedImage.copy_of expects a numpy.ndarray, got {type(array).__name__}"
            )
        return cls(np.array(array, copy=True), camera_id, kind, source_frame_id)

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def resolution(self) -> Tuple[int, int]:
        """``(width, height)``."""
        return (self.width, self.height)


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreviewEvent:
    """Common provenance carried by every canonical preview event.

    Deliberately carries **no** ``preview_seq``. The sequence number is assigned
    by the bus at publish time and lives on the envelope
    (``eligsir.viewers.bus.PreviewEnvelope``), so a producer cannot mint,
    reuse or reorder one.
    """

    iteration: int
    wall_time_s: float
    schema_version: int = field(default=PREVIEW_SCHEMA_VERSION, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "iteration", _check_count(self.iteration, "iteration"))
        wall = float(self.wall_time_s)
        if not math.isfinite(wall) or wall < 0.0:
            raise PreviewSchemaError(
                f"wall_time_s must be a finite, non-negative float, got {self.wall_time_s!r}"
            )
        object.__setattr__(self, "wall_time_s", wall)
        if int(self.schema_version) != PREVIEW_SCHEMA_VERSION:
            raise PreviewSchemaError(
                f"schema_version {self.schema_version!r} != this build's "
                f"{PREVIEW_SCHEMA_VERSION}"
            )


@dataclass(frozen=True)
class TrainingPairPreview(PreviewEvent):
    """THE primary event: one optimizer supervision operation's target + render.

    Exactly the target used for the loss and the render already computed for
    that same loss, for the same selected camera. Producing this event requires
    **no extra rasterization** — that is the point (design doc §13).

    Refused at construction (each of these is a real, tested case):

    * ``target.camera_id != render.camera_id``;
    * ``target`` / ``render`` shapes disagree;
    * both ``source_frame_id`` known and unequal;
    * ``target.kind`` is not ``TARGET`` or ``render.kind`` is not ``RENDER``
      (so the halves cannot be swapped);
    * a depth array whose shape does not match its RGB counterpart.
    """

    target: CapturedImage = None  # type: ignore[assignment]
    render: CapturedImage = None  # type: ignore[assignment]
    optimizer_step_id: int = 0
    supervision_view_count: int = 0
    camera_role: CameraRole = CameraRole.OTHER
    phase: MappingPhase = MappingPhase.OTHER
    n_gaussians: int = 0
    map_version: Optional[int] = None
    psnr_db: Optional[float] = None
    frames_seen: Optional[int] = None
    frames_ingested: Optional[int] = None
    keyframes_admitted: Optional[int] = None
    target_depth: Optional[np.ndarray] = None
    render_depth: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("target", "render"):
            if not isinstance(getattr(self, name), CapturedImage):
                raise PreviewSchemaError(
                    f"TrainingPairPreview.{name} must be a CapturedImage carrying "
                    f"its own camera identity; a bare array cannot express (and "
                    f"therefore cannot be checked for) a mismatched pair"
                )
        if self.target.kind is not ImageKind.TARGET:
            raise PreviewSchemaError(
                f"target half has kind {self.target.kind.value!r}, expected 'target'"
            )
        if self.render.kind is not ImageKind.RENDER:
            raise PreviewSchemaError(
                f"render half has kind {self.render.kind.value!r}, expected 'render'"
            )
        if self.target.camera_id != self.render.camera_id:
            raise PreviewSchemaError(
                "training-pair camera identity mismatch: target camera_id "
                f"{self.target.camera_id} != render camera_id {self.render.camera_id}. "
                "A pair whose halves came from different cameras is exactly the "
                "defect issue #570 exists to make unconstructible."
            )
        if self.target.array.shape[:2] != self.render.array.shape[:2]:
            raise PreviewSchemaError(
                "training-pair resolution mismatch: target "
                f"{self.target.resolution} != render {self.render.resolution}"
            )
        t_src, r_src = self.target.source_frame_id, self.render.source_frame_id
        if t_src is not None and r_src is not None and t_src != r_src:
            raise PreviewSchemaError(
                f"training-pair source_frame_id mismatch: {t_src} != {r_src}"
            )
        _check_enum(self.camera_role, CameraRole, "camera_role")
        _check_enum(self.phase, MappingPhase, "phase")
        for name in ("optimizer_step_id", "supervision_view_count", "n_gaussians"):
            object.__setattr__(self, name, _check_count(getattr(self, name), name))
        for name in ("map_version", "frames_seen", "frames_ingested", "keyframes_admitted"):
            object.__setattr__(
                self, name, _check_optional_count(getattr(self, name), name)
            )
        object.__setattr__(self, "psnr_db", _check_optional_finite(self.psnr_db, "psnr_db"))
        for name, ref in (("target_depth", self.target), ("render_depth", self.render)):
            depth = getattr(self, name)
            if depth is None:
                continue
            depth = _freeze_array(depth, name, allowed_ndim=(2,))
            if depth.shape[:2] != ref.array.shape[:2]:
                raise PreviewSchemaError(
                    f"{name} shape {depth.shape[:2]} does not match its RGB "
                    f"counterpart {ref.array.shape[:2]}"
                )
            object.__setattr__(self, name, depth)

    @property
    def camera_id(self) -> int:
        return self.target.camera_id

    @property
    def source_frame_id(self) -> Optional[int]:
        return (
            self.target.source_frame_id
            if self.target.source_frame_id is not None
            else self.render.source_frame_id
        )

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.target.resolution


@dataclass(frozen=True)
class RawInputPreview(PreviewEvent):
    """The newest source observation -- a DIFFERENT question from the pair.

    "What is the sensor seeing now?", not "how well is the map reconstructing
    the camera the optimizer just supervised?".

    Structurally distinct from :class:`TrainingPairPreview` by construction: no
    ``render`` field, and it does not accept a :class:`CapturedImage`, so no
    sink can silently treat one as the other. A UI juxtaposing them must label
    the ids/age itself (design doc §4.2).
    """

    source_frame_id: int = 0
    timestamp_s: float = 0.0
    rgb: np.ndarray = None  # type: ignore[assignment]
    depth: Optional[np.ndarray] = None
    camera_id: Optional[int] = None
    pose_c2w: Optional[np.ndarray] = None
    tracking_status: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.rgb, CapturedImage):
            raise PreviewSchemaError(
                "RawInputPreview.rgb must be a plain array, not a CapturedImage. "
                "Raw sensor input is not half of a matched supervision pair; "
                "keeping the types disjoint is what stops a sink implying they "
                "correspond."
            )
        object.__setattr__(
            self, "rgb", _freeze_array(self.rgb, "RawInputPreview.rgb", allowed_ndim=(2, 3))
        )
        object.__setattr__(
            self, "source_frame_id", _check_count(self.source_frame_id, "source_frame_id")
        )
        object.__setattr__(self, "camera_id", _check_optional_count(self.camera_id, "camera_id"))
        ts = float(self.timestamp_s)
        if not math.isfinite(ts):
            raise PreviewSchemaError(f"timestamp_s must be finite, got {self.timestamp_s!r}")
        object.__setattr__(self, "timestamp_s", ts)
        if self.depth is not None:
            depth = _freeze_array(self.depth, "RawInputPreview.depth", allowed_ndim=(2,))
            if depth.shape[:2] != self.rgb.shape[:2]:
                raise PreviewSchemaError(
                    f"depth shape {depth.shape[:2]} does not match rgb "
                    f"{self.rgb.shape[:2]}"
                )
            object.__setattr__(self, "depth", depth)
        if self.pose_c2w is not None:
            pose = _freeze_array(self.pose_c2w, "pose_c2w", allowed_ndim=(2,))
            if pose.shape != (4, 4):
                raise PreviewSchemaError(f"pose_c2w must be 4x4, got {pose.shape}")
            object.__setattr__(self, "pose_c2w", pose)
        if self.tracking_status is not None and not isinstance(self.tracking_status, str):
            raise PreviewSchemaError("tracking_status must be a string or None")


@dataclass(frozen=True)
class TrajectoryPreview(PreviewEvent):
    """Estimated (and optionally GT) camera trajectory, as ``(N, 4, 4)`` c2w."""

    poses_est: np.ndarray = None  # type: ignore[assignment]
    poses_gt: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        est = _freeze_array(self.poses_est, "poses_est", allowed_ndim=(3,))
        if est.shape[1:] != (4, 4):
            raise PreviewSchemaError(f"poses_est must be (N, 4, 4), got {est.shape}")
        object.__setattr__(self, "poses_est", est)
        if self.poses_gt is not None:
            gt = _freeze_array(self.poses_gt, "poses_gt", allowed_ndim=(3,))
            if gt.shape[1:] != (4, 4):
                raise PreviewSchemaError(f"poses_gt must be (N, 4, 4), got {gt.shape}")
            object.__setattr__(self, "poses_gt", gt)


@dataclass(frozen=True)
class KeyframePreview(PreviewEvent):
    """A newly admitted keyframe's pose + intrinsics."""

    keyframe_id: int = 0
    camera_id: int = 0
    pose_c2w: np.ndarray = None  # type: ignore[assignment]
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0
    source_frame_id: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        pose = _freeze_array(self.pose_c2w, "pose_c2w", allowed_ndim=(2,))
        if pose.shape != (4, 4):
            raise PreviewSchemaError(f"pose_c2w must be 4x4, got {pose.shape}")
        object.__setattr__(self, "pose_c2w", pose)
        for name in ("keyframe_id", "camera_id"):
            object.__setattr__(self, name, _check_count(getattr(self, name), name))
        object.__setattr__(
            self, "source_frame_id", _check_optional_count(self.source_frame_id, "source_frame_id")
        )
        for name in ("width", "height"):
            value = int(getattr(self, name))
            if value <= 0:
                raise PreviewSchemaError(f"{name} must be > 0, got {value}")
            object.__setattr__(self, name, value)
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise PreviewSchemaError(f"{name} must be finite, got {value!r}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MetricsPreview(PreviewEvent):
    """Compact scalar status. Non-finite values are refused, never coerced."""

    metrics: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        source: Sequence[Tuple[str, float]]
        if isinstance(self.metrics, Mapping):
            source = tuple(self.metrics.items())
        else:
            source = tuple(self.metrics)
        out = []
        for entry in source:
            try:
                key, value = entry
            except (TypeError, ValueError) as exc:
                raise PreviewSchemaError(
                    f"metrics entries must be (name, value) pairs, got {entry!r}"
                ) from exc
            if not isinstance(key, str):
                raise PreviewSchemaError(f"metric name must be a str, got {key!r}")
            value = float(value)
            if not math.isfinite(value):
                raise PreviewSchemaError(
                    f"metric {key!r} is {value!r}; an unmeasured scalar must be "
                    f"omitted, not emitted as a non-finite or placeholder number"
                )
            out.append((key, value))
        object.__setattr__(self, "metrics", tuple(out))

    def as_dict(self) -> dict:
        return dict(self.metrics)


@dataclass(frozen=True)
class RuntimeTelemetryPreview(PreviewEvent):
    """Read-only, CPU-assembled decision/freshness/GPU/residency panels."""

    panels: object = None

    def __post_init__(self) -> None:
        super().__post_init__()
        from eligsir.viewers.telemetry_panels import ViewerTelemetryPanels

        if not isinstance(self.panels, ViewerTelemetryPanels):
            raise PreviewSchemaError("panels must be ViewerTelemetryPanels")


@dataclass(frozen=True)
class MapPreviewSnapshot(PreviewEvent):
    """Bounded, versioned, immutable 3D map preview -- SCHEMA ONLY in #570.

    Deliberately **not** produced by anything in this PR. Design doc §4.4 is
    explicit that the safe snapshot/transfer implementation belongs with the
    transfer issue (#571): no sink may ever receive a ``GaussianMapReadView`` or
    a tensor that a concurrent topology mutation can resize or replace. What is
    defined here is only the shape such a snapshot must have once produced --
    detached CPU arrays, a ``map_version``, and an explicit sampled-vs-total
    accounting so a sink can never mistake a 100k sample for the whole map.
    """

    map_version: int = 0
    positions: np.ndarray = None  # type: ignore[assignment]
    scales: Optional[np.ndarray] = None
    quats: Optional[np.ndarray] = None
    opacities: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    n_total_gaussians: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        pos = _freeze_array(self.positions, "positions", allowed_ndim=(2,))
        if pos.shape[1] != 3:
            raise PreviewSchemaError(f"positions must be (N, 3), got {pos.shape}")
        object.__setattr__(self, "positions", pos)
        n = int(pos.shape[0])
        for name, cols in (
            ("scales", 3),
            ("quats", 4),
            ("opacities", None),
            ("colors", 3),
        ):
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = _freeze_array(arr, name, allowed_ndim=(1, 2))
            if arr.shape[0] != n:
                raise PreviewSchemaError(
                    f"{name} has {arr.shape[0]} rows but positions has {n}"
                )
            if cols is not None and (arr.ndim != 2 or arr.shape[1] != cols):
                raise PreviewSchemaError(f"{name} must be (N, {cols}), got {arr.shape}")
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "map_version", _check_count(self.map_version, "map_version"))
        total = _check_count(self.n_total_gaussians, "n_total_gaussians")
        if total < n:
            raise PreviewSchemaError(
                f"n_total_gaussians ({total}) < sampled rows ({n}); a snapshot must "
                f"state honestly how much of the map it represents"
            )
        object.__setattr__(self, "n_total_gaussians", total)

    @property
    def n_sampled(self) -> int:
        return int(self.positions.shape[0])

    @property
    def is_complete(self) -> bool:
        return self.n_sampled == self.n_total_gaussians


# ---------------------------------------------------------------------------
# channel routing -- the type IS the channel
# ---------------------------------------------------------------------------
EVENT_TYPE_BY_CHANNEL = {
    PreviewChannel.TRAINING_PAIR: TrainingPairPreview,
    PreviewChannel.RAW_INPUT: RawInputPreview,
    PreviewChannel.TRAJECTORY: TrajectoryPreview,
    PreviewChannel.KEYFRAME: KeyframePreview,
    PreviewChannel.METRICS: MetricsPreview,
    PreviewChannel.RUNTIME_TELEMETRY: RuntimeTelemetryPreview,
    PreviewChannel.MAP_SNAPSHOT: MapPreviewSnapshot,
}

_CHANNEL_BY_EVENT_TYPE = {v: k for k, v in EVENT_TYPE_BY_CHANNEL.items()}


def channel_for_event(event: PreviewEvent) -> PreviewChannel:
    """Resolve an event's bus channel from its exact type.

    Exact type, not ``isinstance``: a subclass of ``TrainingPairPreview`` would
    otherwise silently inherit the primary channel and could smuggle extra
    fields past a sink that only knows the base schema.
    """
    try:
        return _CHANNEL_BY_EVENT_TYPE[type(event)]
    except KeyError:
        raise PreviewSchemaError(
            f"{type(event).__name__} is not a canonical preview event type; "
            f"known types: {sorted(t.__name__ for t in _CHANNEL_BY_EVENT_TYPE)}"
        ) from None
