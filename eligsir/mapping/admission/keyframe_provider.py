"""Immutable proposal adapter for the existing keyframe admission semantics."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from eligsir.contracts.control import ControlSnapshot
from eligsir.contracts.control import DispositionKind, ObservationDisposition
from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeDecision,
)
from eligsir.mapping.scheduler.frame_quality import FrameQuality


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class KeyframeAdmissionProposal:
    """Evidence-only keyframe result; never a terminal disposition or permit."""

    provider: str
    admitted: bool
    reason: str
    trace: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.provider or not self.reason:
            raise ValueError("keyframe proposal provider and reason must be non-empty")
        if type(self.admitted) is not bool:
            raise TypeError("keyframe proposal admitted must be bool")
        frozen_trace = tuple((str(name), _freeze(value)) for name, value in self.trace)
        if any(not name for name, _value in frozen_trace):
            raise ValueError("keyframe proposal trace names must be non-empty")
        object.__setattr__(self, "trace", frozen_trace)


class KeyframeAdmitterProposalProvider:
    """Run the characterized legacy selector as an immutable proposal source."""

    def __init__(
        self,
        admitter: KeyframeAdmitter,
        *,
        frame_from_snapshot: Callable[[ControlSnapshot], Any],
        camera_from_snapshot: Callable[[ControlSnapshot], Any],
    ) -> None:
        if not isinstance(admitter, KeyframeAdmitter):
            raise TypeError("admitter must be KeyframeAdmitter")
        if not callable(frame_from_snapshot) or not callable(camera_from_snapshot):
            raise TypeError("snapshot adapters must be callable")
        self._admitter = admitter
        self._frame_from_snapshot = frame_from_snapshot
        self._camera_from_snapshot = camera_from_snapshot
        self._staged: dict[tuple[int, Any], tuple[KeyframeAdmitter, KeyframeAdmissionProposal]] = {}

    def propose(self, snapshot: ControlSnapshot) -> KeyframeAdmissionProposal:
        if not isinstance(snapshot, ControlSnapshot):
            raise TypeError("snapshot must be ControlSnapshot")
        frame = self._frame_from_snapshot(snapshot)
        camera = self._camera_from_snapshot(snapshot)
        candidate = copy.deepcopy(self._admitter)
        if snapshot.gaussian_count == 0:
            # A map-free bootstrap observation has no legitimate render-overlap
            # scalar yet.  Nominate it explicitly, then advance only the private
            # candidate copy; ObservationOwner still resolves modalities and
            # backpressure before commit can establish the accepted baseline.
            decision = KeyframeDecision(
                admitted=True,
                reason="bootstrap",
                score=1.0,
                cheap_score=1.0,
                render_overlap=None,
                trace={
                    "candidate_trigger": "bootstrap",
                },
            )
            candidate._record_admission(decision, frame)
        else:
            decision, _render_package = candidate.admit(camera, frame)
        trace = decision.trace or {}
        telemetry = {
            "gap": decision.gap,
            "pose_delta_m": decision.pose_delta_m,
            "pose_delta_deg": decision.pose_delta_deg,
            "render_overlap": decision.render_overlap,
            "quality": decision.quality,
            "novelty_forced": decision.novelty_forced,
            "forced": decision.forced,
            "controller_target_interval_s": decision.controller_target_interval_s,
            "translation_velocity_m_s": decision.translation_velocity_m_s,
            "rotation_velocity_deg_s": decision.rotation_velocity_deg_s,
            "translation_velocity_ema_m_s": decision.translation_velocity_ema_m_s,
            "rotation_velocity_ema_deg_s": decision.rotation_velocity_ema_deg_s,
            "source_dt_s": decision.source_dt_s,
            "source_frame_delta": decision.source_frame_delta,
            "timebase_source": decision.timebase_source,
            "implied_input_fps": decision.implied_input_fps,
            "starvation_age_source_s": decision.starvation_age_source_s,
            "starvation_age_source_frames": decision.starvation_age_source_frames,
        }
        proposal = KeyframeAdmissionProposal(
            provider=self.__class__.__name__,
            admitted=bool(decision.admitted),
            reason=str(decision.reason),
            trace=tuple(trace.items()) + tuple(telemetry.items()),
        )
        self._staged[(snapshot.snapshot_id, snapshot.source)] = (candidate, proposal)
        return proposal

    def commit(
        self, snapshot: ControlSnapshot, disposition: ObservationDisposition
    ) -> None:
        """Commit evaluated legacy state only when the terminal result agrees.

        Evaluation happens on a private copy.  This is the two-phase seam that
        keeps a capacity or modality veto from advancing the admitted pose,
        cadence controller, starvation counters, or keyframe history.
        """

        staged = self._staged.pop((snapshot.snapshot_id, snapshot.source), None)
        if staged is None:
            raise RuntimeError("keyframe proposal was not staged for terminal commit")
        candidate, proposal = staged
        finally_admitted = disposition.kind is DispositionKind.ADMIT
        if proposal.admitted and not finally_admitted:
            self._commit_candidate_state_without_admission(candidate)
            return
        if proposal.admitted != finally_admitted:
            return
        self._admitter.__dict__.clear()
        self._admitter.__dict__.update(candidate.__dict__)

    def _commit_candidate_state_without_admission(
        self, candidate: KeyframeAdmitter
    ) -> None:
        """Retain candidate clocks/telemetry while withholding admission state."""

        admission_fields = (
            "_keyframes_admitted",
            "_keyframe_indices",
            "_last_train_index",
            "_last_train_c2w",
            "_last_train_ts",
            "_last_admitted_luma",
            "_last_admitted_iter",
            "_last_admitted_iter_blur",
            "_novelty_force_decisive",
        )
        retained = {
            name: copy.deepcopy(getattr(self._admitter, name))
            for name in admission_fields
        }
        self._admitter.__dict__.clear()
        self._admitter.__dict__.update(candidate.__dict__)
        for name, value in retained.items():
            setattr(self._admitter, name, value)


def frame_info_from_snapshot(snapshot: ControlSnapshot) -> FrameInfo:
    """Transitional production adapter for pose/time evidence in a snapshot."""

    source = snapshot.source
    tracking = snapshot.tracking
    if source is None or tracking is None or tracking.pose is None:
        raise ValueError("keyframe admission requires source-bound tracking pose evidence")
    frame = FrameInfo(
        np.array(tracking.pose.matrix, dtype=np.float32, copy=True),
        source.source_ordinal,
        source.timestamp_seconds,
    )
    frame._control_snapshot = snapshot
    return frame


class _SnapshotAlpha:
    def __init__(self, overlap: float) -> None:
        self._overlap = max(0.0, min(1.0, float(overlap)))

    def detach(self) -> "_SnapshotAlpha":
        return self

    def __gt__(self, _threshold: float) -> "_SnapshotAlpha":
        return self

    def float(self) -> "_SnapshotAlpha":
        return self

    def mean(self) -> "_SnapshotAlpha":
        return self

    def item(self) -> float:
        return self._overlap


class _SnapshotKeyframeAdmitter(KeyframeAdmitter):
    """Bridge legacy coverage consumers to an immutable scalar snapshot fact."""

    def _coverage_new_voxel_frac(self, cam: Any, frame: Any) -> float | None:
        snapshot = getattr(frame, "_control_snapshot", cam)
        facts = {fact.name: fact.value for fact in snapshot.evidence}
        value = facts.get("coverage_gain")
        self._last_coverage_none_cause = None if value is not None else "unavailable"
        return None if value is None else float(value)

    def _median_parallax_deg(self) -> float | None:
        snapshot = getattr(self, "_active_control_snapshot", None)
        if snapshot is None:
            return super()._median_parallax_deg()
        value = {fact.name: fact.value for fact in snapshot.evidence}.get("parallax_deg")
        return None if value is None else float(value)

    def admit(self, cam: Any, frame: Any):
        self._active_control_snapshot = frame._control_snapshot
        try:
            return super().admit(cam, frame)
        finally:
            self._active_control_snapshot = None


def keyframe_admitter_proposal_provider(
    config,
) -> KeyframeAdmitterProposalProvider:
    """Build the proposal provider from an already resolved typed config."""

    from eligsir.mapping.scheduler.keyframes import KeyframeAdmitterConfig

    if not isinstance(config, KeyframeAdmitterConfig):
        raise TypeError("keyframe proposal provider requires KeyframeAdmitterConfig")
    if config.parallax_gate and not config.coverage_gain_gate:
        raise ValueError(
            "parallax gate requires coverage gain evidence in canonical production"
        )

    def _facts(frame: Any) -> dict[str, Any]:
        return {
            fact.name: fact.value
            for fact in frame._control_snapshot.evidence
        }

    def _render(snapshot: ControlSnapshot, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        facts = {fact.name: fact.value for fact in snapshot.evidence}
        return {"alpha": _SnapshotAlpha(float(facts.get("overlap", 0.0)))}

    def _quality(_rgb: np.ndarray, _depth: np.ndarray, frame: Any) -> FrameQuality:
        facts = _facts(frame)
        blur_value = facts.get("rgb_blur_laplacian")
        blur = 0.0 if blur_value is None else float(blur_value)
        clip = float(facts.get("rgb_clip_ratio", 0.0))
        depth_valid = float(facts.get("depth_valid_ratio", 1.0))
        reject_reason = None
        if blur < config.quality_min_blur:
            reject_reason = "blur_reject"
        elif clip > config.quality_max_clip_ratio:
            reject_reason = "exposure_reject"
        elif depth_valid < config.quality_min_depth_valid:
            reject_reason = "quality_reject"
        return FrameQuality(
            blur_laplacian=blur,
            gradient_energy=0.0,
            white_clip_ratio=clip,
            black_clip_ratio=0.0,
            depth_valid_ratio=depth_valid,
            depth_hole_ratio=max(0.0, 1.0 - depth_valid),
            depth_edge_invalid_ratio=0.0,
            quality_weight=float(facts.get("rgb_quality_score", 1.0)),
            reject_reason=reject_reason,
        )

    def _luma(frame: Any) -> float | None:
        value = _facts(frame).get("rgb_luma")
        return None if value is None else float(value)

    def _blur(frame: Any) -> float | None:
        facts = _facts(frame)
        value = facts.get(
            "rgb_standalone_blur_laplacian",
            facts.get("rgb_blur_laplacian"),
        )
        return 0.0 if value is None else float(value)

    admitter = _SnapshotKeyframeAdmitter(
        config,
        render_fn=_render,
        evaluate_quality=_quality,
        load_depth_m=lambda frame: np.ones((1, 1), dtype=np.float32),
        load_rgb=lambda frame: np.ones((1, 1, 3), dtype=np.float32),
        evaluate_luma=_luma,
        evaluate_blur=_blur,
        occupancy_fn=lambda *_args, **_kwargs: np.zeros((1,), dtype=bool),
    )
    return KeyframeAdmitterProposalProvider(
        admitter,
        frame_from_snapshot=frame_info_from_snapshot,
        camera_from_snapshot=lambda snapshot: snapshot,
    )


__all__ = [
    "KeyframeAdmissionProposal",
    "KeyframeAdmitterProposalProvider",
    "frame_info_from_snapshot",
    "keyframe_admitter_proposal_provider",
]
