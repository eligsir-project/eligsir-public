"""One authoritative record of how a streaming run's resolution was resolved.

Issue #595. A real 2026-08-14 run (``issue290xsv2_sb20_obs``) printed::

    [streaming] auto-downscale gate: native 1920x1440 (2764800 px) exceeds
        --streaming_auto_downscale_max_pixels=1228800; forcing
        --streaming_resolution=2 (960x720) to bound render/backward memory
        regardless of ResolutionController's throughput-based tiering.

...and then trained at half linear resolution. Nothing machine-readable in
``report.json`` recorded that. Two runs launched from the same spec, on
machines with different memory limits, could therefore differ by 4x in pixel
budget while looking identical in every artifact a comparison reads.

Three DISTINCT mechanisms decide how big a training image is
-------------------------------------------------------------
They compose, and conflating them is the defect this module exists to fix.

1. **The hard startup gate** (``--streaming_auto_downscale_max_pixels``),
   a deterministic pre-flight check against the source's NATIVE resolution.
   It rewrites ``args.streaming_resolution`` (an integer divisor applied by
   the frame source, so every frame arrives already downscaled) exactly once,
   before any frame is loaded, and only when the run left
   ``--streaming_resolution`` at its default of 1. It is a memory-safety
   bound, not a quality/throughput decision.
2. **The camera-level resolution argument** (``args.resolution`` /
   ``resolution_scale``), applied per camera by
   ``utils.camera_utils._resolve_geometry``. ``StreamingScene.__init__``
   pins ``args.resolution = 1`` unless ``-r``/``--resolution`` appears in
   ``sys.argv``, so it is usually the identity on a streaming run -- but when
   it is not, it composes MULTIPLICATIVELY with (1). This is why the gate's
   own fields here are named ``hard_*`` and NOT ``trained_*``: a report
   claiming ``960x720`` while ``-r 2`` was also passed would be wrong by a
   further factor of two.
3. **The dynamic** :class:`~eligsir.mapping.resolution_controller.ResolutionController`,
   which reacts to measured throughput by switching the active cache tier
   between ``"high"`` and ``"low"`` DURING training. It never touches
   ``args.streaming_resolution``; it rebinds the camera's image to an
   already-cached tier.

Accordingly the record has two non-overlapping halves: the frozen
:class:`ResolutionProvenance` (mechanisms 1 and 2, decided at startup) and
the mutable :class:`ResolutionTierLog` (mechanism 3, observed during
training). ``startup_override_reason`` names which of (1)/(2) produced the
resolved startup divisor, so a hard memory cap is never mistaken for a
throughput-driven controller change or for an explicitly requested scale.

This module RECORDS; it never decides
-------------------------------------
Nothing here feeds back into the gate, the controller, or camera loading, and
no consumer may derive a resolution from it. #595 deliberately does not change
the gate's behaviour, thresholds or defaults. In particular the "should the
hard cap be expressed as ``max_pixels`` feeding the controller instead of as a
separate override" redesign question from #595 is NOT answered here.

The final trained size is OBSERVED, not derived
-----------------------------------------------
:meth:`ResolutionTierLog.observe` records the real
``viewpoint_cam.image_width/height`` the training loop just used. Re-deriving
it from ``args.resolution`` would duplicate ``_resolve_geometry``'s branch
logic (including its ``-1`` >1600px rule) in a second place, free to drift --
the exact defect class #596/#646 were filed about. So this module stays
torch-free and reports what the loop measured.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- startup_override_reason ------------------------------------------------

OVERRIDE_NONE = "none"
"""No override: the run asked for divisor 1 and got divisor 1."""

OVERRIDE_AUTO_DOWNSCALE_MAX_PIXELS = "auto_downscale_max_pixels"
"""The hard startup gate forced a divisor to bound render/backward memory.

This is a MEMORY/SAFETY cap, decided once before training from the source's
native resolution -- never a throughput decision. Its dynamic counterpart is
:class:`ResolutionTierLog`.
"""

OVERRIDE_REQUESTED_SCALE = "requested_scale"
"""``--streaming_resolution`` was already != 1 when the gate ran, so the gate
did not evaluate and the requested divisor stands.

Deliberately NOT named "explicit_request"/"user_request": since PR #618 a
``--config`` file or a ``--streaming_camera_profile`` can supply this value,
and this record cannot tell a human's flag from a profile's default. It
records WHICH mechanism won, not WHO intended it.
"""

STARTUP_OVERRIDE_REASONS = (
    OVERRIDE_NONE,
    OVERRIDE_AUTO_DOWNSCALE_MAX_PIXELS,
    OVERRIDE_REQUESTED_SCALE,
)

# --- where the native resolution came from ----------------------------------

NATIVE_SOURCE_INTRINSICS_JSON = "intrinsics_json"
"""``<source_path>/intrinsics.json`` -- what the hard gate itself reads."""

NATIVE_SOURCE_FRAME_SOURCE = "frame_source"
"""The frame source reported it (e.g. RealityScan's sfm.txt/*.xmp cameras,
which have no ``intrinsics.json`` and run their own equivalent gate)."""

NATIVE_SOURCE_UNKNOWN = "unknown"
"""Not determinable (no ``intrinsics.json``, source exposes nothing). The
hard gate cannot fire in this case either -- an unknown native resolution is
recorded as unknown, never guessed."""

# --- bounded tier history ---------------------------------------------------

DEFAULT_TIER_HISTORY_LIMIT = 256
"""Max recorded tier-change entries.

Bounded because a pathologically flapping controller could otherwise emit one
entry per iteration into ``report.json``. The FIRST N are kept, not the last N:
#595's question is "did a tier change follow the hard cap", which the earliest
transitions answer and a tail-truncating ring buffer would discard. The
unbounded ``tier_change_count`` and the ``tier_history_truncated`` flag keep
the count honest at any run length.
"""


def read_native_resolution(
    source_path: str,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Read a dataset's native frame size from ``intrinsics.json``.

    Extracted verbatim from ``train_streaming.py``'s hard-gate block so the
    gate and this record read exactly one implementation. The ``try`` covers
    precisely what the gate's own ``try`` covered (open + ``json.load`` + the
    two ``int()`` conversions) -- no wider, no narrower.

    Args:
        source_path: dataset root (``dataset.source_path``).

    Returns:
        ``(width, height, error)``. A missing ``intrinsics.json`` is
        ``(None, None, None)`` -- absence is not an error, and the gate
        prints nothing in that case. A read/parse failure is
        ``(None, None, "<exception text>")``; the caller owns the message.
    """
    path = os.path.join(source_path, "intrinsics.json")
    if not os.path.isfile(path):
        return None, None, None
    try:
        with open(path) as f:
            intr = json.load(f)
        return int(intr.get("width", 0)), int(intr.get("height", 0)), None
    except Exception as e:  # noqa: BLE001 -- mirrors the gate's own broad catch
        return None, None, str(e)


@dataclass(frozen=True)
class ResolutionProvenance:
    """Immutable record of the STARTUP half of the resolution decision.

    Every field is recorded from values the call site's already-executed code
    produced; nothing here recomputes a resolution.

    Attributes:
        native_width/native_height/native_pixels: the source's native frame
            size, or ``None`` when undeterminable (see ``native_source``).
        native_source: one of the ``NATIVE_SOURCE_*`` constants.
        native_read_error: ``intrinsics.json`` read/parse failure text, if any.
        requested_scale: ``--streaming_resolution`` as the gate saw it, i.e.
            AFTER ``--config``/camera-profile application and BEFORE the gate.
        hard_max_pixels: ``--streaming_auto_downscale_max_pixels`` (0 = off).
        hard_gate_disabled: ``--streaming_auto_downscale_disabled``.
        hard_gate_evaluated: whether the gate's pixel comparison actually ran
            (it is skipped when the requested scale is already != 1, when the
            gate is disabled, when the cap is 0, or when the native size is
            unknown).
        hard_gate_fired: whether the gate rewrote the divisor.
        hard_resolved_scale: the integer divisor the frame source applies --
            ``args.streaming_resolution`` after the gate.
        hard_resolved_width/height/pixels: native size divided by
            ``hard_resolved_scale``, i.e. the size frames arrive at from the
            frame source. **NOT necessarily the trained size** -- see
            ``camera_resolution_arg`` and the module docstring.
        startup_override_reason: one of :data:`STARTUP_OVERRIDE_REASONS`.
        camera_resolution_arg: ``args.resolution``, the SECOND, independent
            downscale applied per camera by ``_resolve_geometry``. 1 on a
            default streaming run (``StreamingScene.__init__`` pins it unless
            ``-r`` is in ``sys.argv``).
        camera_resolution_scale: ``args.resolution_scale``-equivalent (1.0
            on every streaming run today).
        camera_downscale_active: whether (2) is anything but the identity, in
            which case the real trained size is ``hard_resolved_*`` divided
            again -- read the observed size in the dynamic block instead of
            deriving it.
    """

    native_width: Optional[int]
    native_height: Optional[int]
    native_pixels: Optional[int]
    native_source: str
    native_read_error: Optional[str]
    requested_scale: int
    hard_max_pixels: int
    hard_gate_disabled: bool
    hard_gate_evaluated: bool
    hard_gate_fired: bool
    hard_resolved_scale: int
    hard_resolved_width: Optional[int]
    hard_resolved_height: Optional[int]
    hard_resolved_pixels: Optional[int]
    startup_override_reason: str
    camera_resolution_arg: Optional[int]
    camera_resolution_scale: float
    camera_downscale_active: bool

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serializable startup half of ``report.json["resolution"]``."""
        return asdict(self)

    def format_log(self) -> str:
        """The single authoritative startup summary.

        Formatted from the same fields as :meth:`as_dict`, so ``train.log``
        and run metadata cannot disagree by construction (the #596 pattern).
        """
        native = (
            f"{self.native_width}x{self.native_height} ({self.native_pixels} px)"
            if self.native_pixels is not None
            else f"unknown (source={self.native_source})"
        )
        resolved = (
            f"{self.hard_resolved_width}x{self.hard_resolved_height} "
            f"({self.hard_resolved_pixels} px)"
            if self.hard_resolved_pixels is not None
            else "unknown"
        )
        lines = [
            "[resolution] startup resolution provenance (authoritative; all overrides applied):",
            f"  native={native}  native_source={self.native_source}",
            f"  requested_scale={self.requested_scale}"
            f"  hard_max_pixels={self.hard_max_pixels}"
            f"  hard_gate_disabled={self.hard_gate_disabled}"
            f"  hard_gate_evaluated={self.hard_gate_evaluated}",
            f"  hard_resolved_scale={self.hard_resolved_scale}  frames arrive at {resolved}",
            f"  startup_override_reason={self.startup_override_reason}",
        ]
        if self.native_read_error:
            lines.append(f"  -> native resolution unreadable: {self.native_read_error}")
        if self.startup_override_reason == OVERRIDE_AUTO_DOWNSCALE_MAX_PIXELS:
            lines.append(
                "  -> this is a HARD MEMORY/SAFETY cap decided before training, not a "
                "ResolutionController throughput decision. This run did NOT train at "
                "its source's native resolution."
            )
        elif self.startup_override_reason == OVERRIDE_REQUESTED_SCALE:
            lines.append(
                f"  -> divisor {self.hard_resolved_scale} came from the REQUEST "
                "(--streaming_resolution, possibly via --config/--streaming_camera_profile); "
                "the hard gate did not evaluate."
            )
        if self.camera_downscale_active:
            lines.append(
                f"  -> WARNING: a SECOND, independent camera-level downscale is active "
                f"(--resolution={self.camera_resolution_arg}, "
                f"resolution_scale={self.camera_resolution_scale}). It composes with the "
                f"divisor above, so the trained size is SMALLER than {resolved}; read the "
                "observed size in the dynamic block."
            )
        return "\n".join(lines)


def build_resolution_provenance(
    *,
    native_width: Optional[int],
    native_height: Optional[int],
    native_source: str,
    native_read_error: Optional[str],
    requested_scale: int,
    hard_max_pixels: int,
    hard_gate_disabled: bool,
    hard_gate_evaluated: bool,
    hard_gate_fired: bool,
    resolved_scale: int,
    camera_resolution_arg: Optional[int],
    camera_resolution_scale: float = 1.0,
) -> ResolutionProvenance:
    """Assemble a :class:`ResolutionProvenance` from already-computed values.

    Performs no resolution decision: ``resolved_scale`` must be passed
    verbatim as the ``args.streaming_resolution`` the frame source was built
    with. The only things derived here are the attribution
    (``startup_override_reason``) and the arithmetic already implied by the
    divisor (``hard_resolved_*``).

    Args:
        native_width/native_height: from :func:`read_native_resolution` or the
            frame source; ``None``/0 when unknown.
        native_source: one of the ``NATIVE_SOURCE_*`` constants.
        native_read_error: ``intrinsics.json`` failure text, if any.
        requested_scale: ``--streaming_resolution`` before the gate ran.
        hard_max_pixels: ``--streaming_auto_downscale_max_pixels``.
        hard_gate_disabled: ``--streaming_auto_downscale_disabled``.
        hard_gate_evaluated: whether the gate's comparison ran at all.
        hard_gate_fired: whether the gate rewrote the divisor.
        resolved_scale: ``args.streaming_resolution`` after the gate.
        camera_resolution_arg: ``args.resolution`` (the independent
            camera-level downscale), or ``None`` if not yet resolved.
        camera_resolution_scale: the camera-level scale (1.0 everywhere today).
    """
    nw = int(native_width) if native_width else None
    nh = int(native_height) if native_height else None
    npx = nw * nh if (nw and nh) else None

    scale = max(1, int(resolved_scale))
    rw = max(1, nw // scale) if nw else None
    rh = max(1, nh // scale) if nh else None

    if hard_gate_fired:
        reason = OVERRIDE_AUTO_DOWNSCALE_MAX_PIXELS
    elif scale != 1:
        reason = OVERRIDE_REQUESTED_SCALE
    else:
        reason = OVERRIDE_NONE

    cam_arg = None if camera_resolution_arg is None else int(camera_resolution_arg)
    cam_scale = float(camera_resolution_scale)
    cam_active = bool(cam_arg is not None and cam_arg != 1) or cam_scale != 1.0

    return ResolutionProvenance(
        native_width=nw,
        native_height=nh,
        native_pixels=npx,
        native_source=str(native_source),
        native_read_error=native_read_error,
        requested_scale=int(requested_scale),
        hard_max_pixels=int(hard_max_pixels or 0),
        hard_gate_disabled=bool(hard_gate_disabled),
        hard_gate_evaluated=bool(hard_gate_evaluated),
        hard_gate_fired=bool(hard_gate_fired),
        hard_resolved_scale=scale,
        hard_resolved_width=rw,
        hard_resolved_height=rh,
        hard_resolved_pixels=(rw * rh) if (rw and rh) else None,
        startup_override_reason=reason,
        camera_resolution_arg=cam_arg,
        camera_resolution_scale=cam_scale,
        camera_downscale_active=cam_active,
    )


@dataclass
class ResolutionTierLog:
    """The DYNAMIC half: what
    :class:`~eligsir.mapping.resolution_controller.ResolutionController`
    actually did during training.

    Feed it :meth:`observe` once per training iteration with the tier
    ``decide()`` RETURNED -- not ``ResolutionController._tier``. Those two
    diverge permanently once ``mapping_done`` engages the controller's one-way
    lock: ``decide()`` returns ``"high"`` from then on while ``_tier`` keeps
    whatever value it last computed. The ``streaming_multires/
    active_tier_is_high`` TensorBoard scalar is written from the return value,
    so reading ``_tier`` here would make the report disagree with TB on every
    run that finished mapping in the ``low`` tier.

    ``controller_enabled`` is derived from the floors or explicit schedule:
    with ``--streaming_multires_it_v_floor 0 --streaming_multires_map_fps_floor
    0`` (which CLAUDE.md's standing guidance says to pass on every run)
    ``floor_breached`` can never be true, so the controller is structurally
    inert. Calling that "enabled" would be a lie of the same kind #595 is about.
    """

    it_v_floor: float = 0.0
    map_fps_floor: float = 0.0
    low_divisor: int = 8
    scheduled_low_until_iteration: int = 0
    history_limit: int = DEFAULT_TIER_HISTORY_LIMIT
    initial_tier: Optional[str] = None
    current_tier: Optional[str] = None
    tier_change_count: int = 0
    first_observed_width: Optional[int] = None
    first_observed_height: Optional[int] = None
    observed_width: Optional[int] = None
    observed_height: Optional[int] = None
    _history: List[Dict[str, Any]] = field(default_factory=list)
    _truncated: bool = False

    @property
    def controller_enabled(self) -> bool:
        """Whether a floor or explicit schedule can change the tier."""
        return (
            self.it_v_floor > 0.0
            or self.map_fps_floor > 0.0
            or self.scheduled_low_until_iteration > 0
        )

    def observe(
        self,
        iteration: int,
        tier: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Record one iteration's active tier and the camera size it used.

        Cheap by construction (a string compare plus, on a real change, one
        small dict append) so it can sit on the per-iteration path.
        """
        if width:
            self.observed_width = int(width)
            if self.first_observed_width is None:
                self.first_observed_width = int(width)
        if height:
            self.observed_height = int(height)
            if self.first_observed_height is None:
                self.first_observed_height = int(height)

        if tier == self.current_tier:
            return
        previous = self.current_tier
        self.current_tier = tier
        if previous is None:
            self.initial_tier = tier
            return
        self.tier_change_count += 1
        if len(self._history) < self.history_limit:
            self._history.append(
                {
                    "iteration": int(iteration),
                    "from_tier": previous,
                    "to_tier": tier,
                    "width": self.observed_width,
                    "height": self.observed_height,
                }
            )
        else:
            self._truncated = True

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serializable dynamic half of ``report.json["resolution"]``."""
        return {
            "controller_enabled": self.controller_enabled,
            "it_v_floor": float(self.it_v_floor),
            "map_fps_floor": float(self.map_fps_floor),
            "low_divisor": int(self.low_divisor),
            "scheduled_low_until_iteration": int(self.scheduled_low_until_iteration),
            "initial_tier": self.initial_tier,
            "current_tier": self.current_tier,
            "tier_change_count": int(self.tier_change_count),
            "tier_history": list(self._history),
            "tier_history_limit": int(self.history_limit),
            "tier_history_truncated": bool(self._truncated),
            # OBSERVED, never derived -- the real camera size the training
            # loop used (after BOTH downscale mechanisms and any tier rebind).
            "first_observed_train_size": (
                [self.first_observed_width, self.first_observed_height]
                if self.first_observed_width and self.first_observed_height
                else None
            ),
            "observed_train_size": (
                [self.observed_width, self.observed_height]
                if self.observed_width and self.observed_height
                else None
            ),
        }


def resolution_report_block(
    provenance: ResolutionProvenance, tier_log: ResolutionTierLog
) -> Dict[str, Any]:
    """Merge both halves into ``report.json["resolution"]``.

    The dynamic half is NESTED, not flattened, so a reader can never mistake a
    throughput-driven tier for the hard startup cap -- acceptance criterion 2
    of #595.
    """
    block = provenance.as_dict()
    block["dynamic"] = tier_log.as_dict()
    return block
