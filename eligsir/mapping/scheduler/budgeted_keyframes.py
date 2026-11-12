"""Production #802 bounded-cohort keyframe admission.

This module extends the existing keyframe admitter without creating a second
camera/history owner.  When the explicit ``streaming_budgeted_admission`` run
configuration is disabled, behavior is exactly the base ``KeyframeAdmitter``.

When enabled, baseline-eligible candidates are evaluated under one immutable
map/topology/coordinate snapshot and held in the existing COLD/ICECOLD archive.
A bounded cohort owns the sole COMMIT/DEFER/DROP decision with a hard production
commit budget of one.  Deferred winners are reactivated through
``StreamingScene.promote_from_cache`` and then receive the same durable scene
bookkeeping an ordinary current-frame admission receives.

No tensor payload is copied into the cohort.  The payload is only the existing
frame reference, typed decision, render-package reference and small admission
state needed to replay the base admitter's admission-only side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from eligsir.mapping.admission.budgeted_controller import (
    AdmissionCandidate,
    BudgetedAdmissionController,
)
from eligsir.mapping.admission.cohort import CohortAction, CohortDecision
from eligsir.mapping.candidates.prefilter import (
    CandidatePrefilterResult,
    PrefilterDisposition,
)
from eligsir.mapping.evidence.canonical import (
    EvidenceAvailability,
    EvidenceIdentity,
    FrameEvidence,
    GaussianProbeEvidence,
    ScalarEvidence,
    SourceReliabilityEvidence,
)
from .keyframes import KeyframeAdmitter as _BaseKeyframeAdmitter


@dataclass(slots=True)
class _DeferredPayload:
    frame: Any
    decision: Any
    render_pkg: Any
    post_admission: dict[str, Any]
    score_map_version: int | None = None
    score_topology_version: int | None = None
    predicted_utility: float | None = None
    under_supported_mass_before: float | None = None


@dataclass(frozen=True, slots=True)
class CachedAdmission:
    camera: Any
    frame_id: int
    score_map_version: int
    score_topology_version: int | None
    predicted_utility: float | None
    under_supported_mass_before: float | None


def _scalar(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _optional_scalar(value, *, unit: str = "unitless") -> ScalarEvidence:
    if value is None:
        return ScalarEvidence.missing(
            EvidenceAvailability.UNEVALUABLE_NO_MAP_SUPPORT,
            unit=unit,
            reason="probe value unavailable for this candidate",
        )
    return ScalarEvidence(float(value), unit=unit)


class BudgetedCohortKeyframeAdmitter(_BaseKeyframeAdmitter):
    """Base keyframe policy plus one bounded deferred pre-commit cohort."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._budgeted_controller: BudgetedAdmissionController[_DeferredPayload] | None = None
        self._budgeted_snapshot: tuple[int, int, int] | None = None
        self._cached_admissions: list[CachedAdmission] = []
        self._budgeted_counts = {
            "candidate": 0,
            "defer": 0,
            "commit_current": 0,
            "commit_cached": 0,
            "drop": 0,
            "snapshot_flush": 0,
            "promotion_failed": 0,
        }

    # ------------------------------------------------------------------
    # Owner/config resolution
    # ------------------------------------------------------------------
    def _scene(self):
        # The trainer binds StreamingScene.check_occupancy as occupancy_fn.  Reuse
        # that existing bound owner rather than threading a second scene registry.
        owner = getattr(self._occupancy_fn, "__self__", None)
        if owner is None or not hasattr(owner, "promote_from_cache"):
            return None
        return owner

    def _enabled(self) -> bool:
        scene = self._scene()
        return bool(scene is not None and scene.args.streaming_budgeted_admission)

    def _controller(self) -> BudgetedAdmissionController[_DeferredPayload]:
        scene = self._scene()
        if scene is None:
            raise RuntimeError(
                "budgeted admission requires the canonical StreamingScene occupancy owner"
            )
        if self._budgeted_controller is None:
            max_candidates = max(
                1,
                int(scene.args.streaming_admission_cohort_max_candidates),
            )
            max_age = max(
                0,
                int(scene.args.streaming_admission_cohort_max_source_age),
            )
            self._budgeted_controller = BudgetedAdmissionController(
                max_candidates=max_candidates,
                max_source_age=max_age,
                # StreamingScene.ingest_next_frame has one durable result slot per
                # source event.  Keep the production hard budget explicit.
                commit_budget=1,
                rank_policy=str(scene.args.streaming_admission_rank_policy),
                rank_seed=int(scene.args.seed),
            )
        return self._budgeted_controller

    def _map_snapshot(self) -> tuple[int, int, int]:
        scene = self._scene()
        model = scene.gaussians
        backend = getattr(model, "_eligsir_map_backend", None)
        if backend is None:
            # Canonical adapters are idempotent per live model, so this does not
            # create a competing map owner.
            if hasattr(model, "params"):
                from eligsir.map.adapters.gsplat import GsplatMapBackend

                backend = GsplatMapBackend(model)
            else:
                from eligsir.map.adapters.legacy_graphdeco import (
                    LegacyGraphdecoMapBackend,
                )

                backend = LegacyGraphdecoMapBackend(model)
        return (
            int(getattr(backend, "map_content_version", 0)),
            int(getattr(backend, "topology_version", 0)),
            int(getattr(backend, "coordinate_frame_version", 0)),
        )

    # ------------------------------------------------------------------
    # Base-admitter admission-only state rollback/replay
    # ------------------------------------------------------------------
    def _admission_state_snapshot(self) -> dict[str, Any]:
        return {
            "keyframe_indices": list(self._keyframe_indices),
            "keyframes_admitted": int(self._keyframes_admitted),
            "last_admitted_luma": self._last_admitted_luma,
            "last_admitted_iter": int(self._last_admitted_iter),
            "last_admitted_iter_blur": int(self._last_admitted_iter_blur),
            "pending_luma": getattr(self, "_pending_luma", None),
            "last_train_index": self._last_train_index,
            "last_train_c2w": (
                None if self._last_train_c2w is None else self._last_train_c2w.copy()
            ),
            "last_train_ts": self._last_train_ts,
        }

    def _restore_admission_state(self, state: dict[str, Any]) -> None:
        self._keyframe_indices = list(state["keyframe_indices"])
        self._keyframes_admitted = int(state["keyframes_admitted"])
        self._last_admitted_luma = state["last_admitted_luma"]
        self._last_admitted_iter = int(state["last_admitted_iter"])
        self._last_admitted_iter_blur = int(state["last_admitted_iter_blur"])
        self._pending_luma = state["pending_luma"]
        self._last_train_index = state["last_train_index"]
        self._last_train_c2w = (
            None if state["last_train_c2w"] is None else state["last_train_c2w"].copy()
        )
        self._last_train_ts = state["last_train_ts"]

    def _post_admission_effect(self) -> dict[str, Any]:
        return {
            "last_admitted_luma": self._last_admitted_luma,
            "last_admitted_iter": int(self._last_admitted_iter),
            "last_admitted_iter_blur": int(self._last_admitted_iter_blur),
        }

    # ------------------------------------------------------------------
    # Durable reactivation of a previously deferred winner
    # ------------------------------------------------------------------
    def _apply_cached_commit(self, payload: _DeferredPayload) -> bool:
        scene = self._scene()
        idx = int(payload.frame.index)
        promoted = int(scene.promote_from_cache({idx}))
        if promoted != 1:
            self._budgeted_counts["promotion_failed"] += 1
            self._trace(
                payload.decision,
                action="drop",
                reason="promotion_failed",
                snapshot=list(self._budgeted_snapshot or ()),
            )
            return False

        cam = scene._active_cam_by_idx.get(idx)
        if cam is None:
            self._budgeted_counts["promotion_failed"] += 1
            self._trace(
                payload.decision,
                action="drop",
                reason="promotion_failed",
                snapshot=list(self._budgeted_snapshot or ()),
            )
            return False

        # Replay the base admitter's admission-only state.  Candidate-level
        # controller/velocity telemetry intentionally already advanced when the
        # candidate was first evaluated and is not replayed.
        payload.decision.admitted = True
        payload.decision.keyframe_index = idx
        self._keyframe_indices.append(idx)
        if len(self._keyframe_indices) > self._max_history:
            self._keyframe_indices = self._keyframe_indices[self._max_history // 2 :]
        self._keyframes_admitted += 1
        self._last_admitted_luma = payload.post_admission["last_admitted_luma"]
        self._last_admitted_iter = int(payload.post_admission["last_admitted_iter"])
        self._last_admitted_iter_blur = int(payload.post_admission["last_admitted_iter_blur"])
        self._pending_luma = None
        self.note_train_frame(payload.frame)

        # promote_from_cache owns camera materialization, active-index tracking,
        # multires-cache rebuilding and residency.  Complete only the admission
        # bookkeeping that method deliberately does not own because historical
        # partition-focus promotions are not ordinary admissions.
        scene._keyframes_admitted += 1
        scene._last_admitted_source_idx = idx
        scene.current_camera = cam
        scene._steps_since_new_frame = 0

        replay_size = int(getattr(scene.args, "streaming_replay_buffer", 512))
        if cam not in scene._replay_buffer:
            scene._replay_buffer.append(cam)
        if len(scene._replay_buffer) > replay_size:
            evict_idx = next(
                (
                    i
                    for i, candidate_cam in enumerate(scene._replay_buffer)
                    if not getattr(candidate_cam, "_is_bootstrap", False)
                ),
                0,
            )
            scene._replay_buffer.pop(evict_idx)

        scene._n_train_frames_ingested += 1
        reservoir_stride = int(scene.args.streaming_global_reservoir_stride)
        if (
            reservoir_stride > 0
            and scene._n_train_frames_ingested % reservoir_stride == 0
            and cam not in scene._global_reservoir
        ):
            scene._global_reservoir.append(cam)

        # Cohort commits are infrequent and already cross a source-event boundary;
        # perform the extent update immediately rather than inventing a second
        # deferred-extent owner.
        scene.update_cameras_extent()
        self._budgeted_counts["commit_cached"] += 1
        if payload.score_map_version is not None:
            self._cached_admissions.append(
                CachedAdmission(
                    camera=cam,
                    frame_id=idx,
                    score_map_version=payload.score_map_version,
                    score_topology_version=payload.score_topology_version,
                    predicted_utility=payload.predicted_utility,
                    under_supported_mass_before=payload.under_supported_mass_before,
                )
            )
        return True

    def record_deferred_shadow_observation(
        self,
        decision,
        *,
        score_map_version: int,
        score_topology_version: int | None,
        predicted_utility: float | None,
        under_supported_mass_before: float | None,
    ) -> bool:
        """Attach score identity to the still-pending candidate it describes."""
        controller = self._budgeted_controller
        if controller is None:
            return False
        for pending in controller.cohort.snapshot():
            payload = pending.payload.payload
            if payload.decision is decision:
                payload.score_map_version = int(score_map_version)
                payload.score_topology_version = (
                    None
                    if score_topology_version is None
                    else int(score_topology_version)
                )
                payload.predicted_utility = predicted_utility
                payload.under_supported_mass_before = under_supported_mass_before
                return True
        return False

    def drain_cached_admissions(self) -> tuple[CachedAdmission, ...]:
        """Return newly committed cached winners exactly once."""
        admissions = tuple(self._cached_admissions)
        self._cached_admissions.clear()
        return admissions

    # ------------------------------------------------------------------
    # Evidence + decision bookkeeping
    # ------------------------------------------------------------------
    def _trace(self, decision, **cohort) -> None:
        trace = dict(getattr(decision, "trace", None) or {})
        scene = self._scene()
        cohort.setdefault(
            "rank_policy",
            str(getattr(scene.args, "streaming_admission_rank_policy", "combined")),
        )
        trace["cohort"] = cohort
        decision.trace = trace

    def _evidence(self, cam, frame, decision, render_pkg, snapshot) -> FrameEvidence:
        map_v, topo_v, coord_v = snapshot
        ordinal = getattr(frame, "source_ordinal", None)
        if ordinal is None:
            ordinal = frame.index
        ts_ns = getattr(frame, "color_ts_ns", None)
        if ts_ns is None:
            ts_ns = int(round(_scalar(getattr(frame, "timestamp", 0.0)) * 1_000_000_000))

        overlap = max(
            0.0,
            min(1.0, _scalar(getattr(decision, "render_overlap", 0.0))),
        )
        rgb_l1 = None
        alpha_mean = None
        if render_pkg is not None:
            try:
                rendered = render_pkg.get("render")
                target = getattr(cam, "original_image", None)
                if rendered is not None and target is not None:
                    target = target[: rendered.shape[0]].to(rendered.device)
                    if target.shape == rendered.shape:
                        rgb_l1 = float((rendered.detach() - target).abs().mean().item())
                alpha = render_pkg.get("alpha")
                if alpha is not None:
                    alpha_mean = float(alpha.detach().mean().item())
            except Exception:
                rgb_l1 = None
                alpha_mean = None

        return FrameEvidence(
            identity=EvidenceIdentity(
                source_id=str(
                    getattr(
                        self._scene()._frame_source,
                        "source_id",
                        type(self._scene()._frame_source).__name__,
                    )
                ),
                source_ordinal=int(ordinal),
                timestamp_ns=int(ts_ns),
                map_version=map_v,
                topology_version=topo_v,
                coordinate_frame_version=coord_v,
            ),
            source=SourceReliabilityEvidence(
                translation_delta_m=ScalarEvidence(
                    _scalar(getattr(decision, "pose_delta_m", 0.0)), unit="m"
                ),
                rotation_delta_deg=ScalarEvidence(
                    _scalar(getattr(decision, "pose_delta_deg", 0.0)), unit="deg"
                ),
                tracking_confidence=ScalarEvidence.missing(
                    EvidenceAvailability.NOT_REQUESTED,
                    reason="cohort does not synthesize a tracking-confidence value",
                ),
            ),
            probe=GaussianProbeEvidence(
                rgb_l1_mean=_optional_scalar(rgb_l1),
                alpha_mean=_optional_scalar(alpha_mean),
                supported_pixel_fraction=ScalarEvidence(overlap),
                unexplained_pixel_fraction=ScalarEvidence(1.0 - overlap),
            ),
            rgb_eligible=True,
            depth_eligible=True,
            escalations=(
                ("baseline_force",)
                if bool(getattr(decision, "forced", False))
                else ()
            ),
        )

    def _process_decisions(
        self,
        decisions: Iterable[CohortDecision[AdmissionCandidate[_DeferredPayload]]],
        *,
        current_idx: int | None,
        flush_reason: str,
        snapshot: tuple[int, int, int],
    ) -> tuple[CohortAction | None, int | None, str | None, int | None]:
        current_action = None
        current_rank = None
        current_reason = None
        promoted_prior = None
        committed_seen = 0

        for item in decisions:
            payload = item.candidate.payload.payload
            idx = int(payload.frame.index)
            if item.action is CohortAction.COMMIT:
                committed_seen += 1
                if committed_seen > 1:
                    raise RuntimeError("#802 production cohort exceeded one hard commit slot")
                if current_idx is not None and idx == current_idx:
                    current_action = CohortAction.COMMIT
                    current_rank = item.rank
                    current_reason = item.reason
                else:
                    if self._apply_cached_commit(payload):
                        promoted_prior = idx
                    else:
                        self._budgeted_counts["drop"] += 1
                        continue
            elif item.action is CohortAction.DROP:
                self._budgeted_counts["drop"] += 1
                if current_idx is not None and idx == current_idx:
                    current_action = CohortAction.DROP
                    current_rank = item.rank
                    current_reason = item.reason

            self._trace(
                payload.decision,
                action=item.action.value,
                rank=item.rank,
                reason=item.reason,
                flush_reason=flush_reason,
                snapshot=list(snapshot),
            )

        return current_action, current_rank, current_reason, promoted_prior

    # ------------------------------------------------------------------
    # Public admit API
    # ------------------------------------------------------------------
    def admit(self, cam: Any, frame: Any):
        if not self._enabled():
            return super().admit(cam, frame)

        controller = self._controller()
        snapshot = self._map_snapshot()

        # Every prior candidate in the cohort is already in COLD/ICECOLD because
        # its ingest call returned DEFER.  If optimization or mutation changed the
        # visible map before the next candidate arrives, commit/drop the old
        # cohort first, then evaluate the new candidate against the new snapshot.
        if len(controller.cohort) and self._budgeted_snapshot != snapshot:
            self._budgeted_counts["snapshot_flush"] += 1
            prior_snapshot = self._budgeted_snapshot or snapshot
            decisions = controller.flush()
            self._process_decisions(
                decisions,
                current_idx=None,
                flush_reason="map_snapshot_changed",
                snapshot=prior_snapshot,
            )
            self._budgeted_snapshot = None

        before = self._admission_state_snapshot()
        decision, render_pkg = super().admit(cam, frame)
        if not bool(decision.admitted):
            self._trace(
                decision,
                action="drop",
                reason="baseline_prefilter_ineligible",
                snapshot=list(snapshot),
            )
            return decision, render_pkg

        self._budgeted_counts["candidate"] += 1
        post = self._post_admission_effect()
        evidence = self._evidence(cam, frame, decision, render_pkg, snapshot)
        disposition = (
            PrefilterDisposition.ESCALATE
            if bool(getattr(decision, "forced", False))
            else PrefilterDisposition.ELIGIBLE
        )
        candidate = AdmissionCandidate(
            payload=_DeferredPayload(frame, decision, render_pkg, post),
            prefilter=CandidatePrefilterResult(disposition, evidence),
            probe=evidence,
        )

        self._budgeted_snapshot = snapshot
        decisions = list(controller.submit(candidate))
        flush_reason = "cohort_pending"
        if controller.should_flush():
            flush_reason = "cohort_bound_or_escalation"
            decisions.extend(controller.flush())

        scene = self._scene()
        final_source = int(getattr(scene, "_source_idx", 0)) >= int(
            getattr(scene, "_frame_count", 0)
        )
        if final_source and len(controller.cohort):
            flush_reason = "end_of_source"
            decisions.extend(controller.flush())

        if not len(controller.cohort):
            self._budgeted_snapshot = None

        current_idx = int(frame.index)
        current_action, current_rank, current_reason, promoted_prior = self._process_decisions(
            decisions,
            current_idx=current_idx,
            flush_reason=flush_reason,
            snapshot=snapshot,
        )

        if current_action is CohortAction.COMMIT:
            self._budgeted_counts["commit_current"] += 1
            self._trace(
                decision,
                action="commit",
                rank=current_rank,
                reason=current_reason,
                flush_reason=flush_reason,
                promoted_prior_index=promoted_prior,
                snapshot=list(snapshot),
            )
            return decision, render_pkg

        # The base admitter already executed _record_admission().  A current
        # candidate that is still pending, or was dropped by the hard budget,
        # is not durable.  Roll back only admission-owned state; candidate-level
        # velocity/time/precedence telemetry intentionally remains advanced.
        self._restore_admission_state(before)
        decision.admitted = False
        decision.keyframe_index = None

        if current_action is CohortAction.DROP:
            decision.reason = "cohort_drop"
            action = CohortAction.DROP
            reason = current_reason or "hard_budget_exhausted"
        else:
            decision.reason = "cohort_defer"
            action = CohortAction.DEFER
            reason = "bounded_cohort_pending"
            self._budgeted_counts["defer"] += 1

        self._trace(
            decision,
            action=action.value,
            rank=current_rank,
            reason=reason,
            flush_reason=flush_reason,
            promoted_prior_index=promoted_prior,
            snapshot=list(snapshot),
        )
        return decision, render_pkg

    @property
    def budgeted_admission_report(self) -> dict:
        controller = self._budgeted_controller
        counts = dict(self._budgeted_counts)
        counts["selected"] = int(counts["commit_current"] + counts["commit_cached"])
        return {
            "enabled": self._enabled(),
            "counts": counts,
            "pending": 0 if controller is None else len(controller.cohort),
            "snapshot": list(self._budgeted_snapshot or ()),
            "commit_budget": 1,
            "rank_policy": (
                str(self._scene().args.streaming_admission_rank_policy)
                if controller is None
                else controller.rank_policy
            ),
            "rank_seed": int(self._scene().args.seed),
        }


__all__ = ["BudgetedCohortKeyframeAdmitter", "CachedAdmission"]
