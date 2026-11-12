"""Shadow integration point (plan section 11.5/11.6, Subagent E).

``ShadowUtilityPolicy`` is the single object ``train_streaming.py`` talks
to. It scores and logs a utility breakdown for every candidate frame that
reaches ``_keyframe_admission_closure`` -- **after** the real
``KeyframeAdmitter.admit()`` decision has already been made -- and tracks
realized gain for frames that decision actually admitted. It never returns
a value that overrides, vetoes, or reorders anything; every public method
either returns ``None`` or a plain logging record.

**Deliverable 1 (plan section 11.7): shadow mode does not alter keyframe
decisions, replay, optimization, or mutation.** Concretely:

- :meth:`observe_candidate` is called with the ``KeyframeDecision`` the
  real admitter already produced; it reads from it, never writes back into
  it or into ``render_pkg``/``stats``/``gaussians``.
- :meth:`observe_training_loss` is called with the loss the training loop
  already computed for its own purposes; it never mutates that loss, never
  calls ``.backward()``, never touches the optimizer.
- Every method is a no-op (single ``if self._enabled`` guard, no tensor
  work) when the policy is constructed disabled, matching the "near-zero
  cost default off" pattern ``EvidenceRequirements``/``EvidenceBuilder``
  already established in W3-05.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Optional

import torch

from eligsir.mapping.admission.realized_gain import RealizedGainRecord, RealizedGainTracker
from eligsir.mapping.admission.scoring import score_frame_utility
from eligsir.mapping.admission.utility_contracts import FrameUtilityScore, UtilityWeights
from eligsir.mapping.evidence.contracts import EvidenceMeta, GaussianEvidence, InputEvidence, RenderEvidence
from eligsir.mapping.evidence.gaussian_history import GaussianEvidenceHistory, StaleEvidenceError


def _score_to_json(score: FrameUtilityScore) -> dict:
    record = {
        "frame_id": score.frame_id,
        "utility": score.utility,
        "components": dataclasses.asdict(score.components),
        "weights": dataclasses.asdict(score.weights),
        "meta": dataclasses.asdict(score.meta),
    }
    return record


def _gain_to_json(record: RealizedGainRecord) -> dict:
    out = dataclasses.asdict(record)
    out["realized_gain"] = record.realized_gain
    out["under_supported_mass_delta"] = record.under_supported_mass_delta
    return out


class ShadowUtilityPolicy:
    """Scores + logs frame utility; never decides anything.

    ``log_dir`` (typically ``<model_path>/shadow_utility/``) receives two
    append-only JSONL files: ``scores.jsonl`` (every scored candidate,
    admitted or not) and ``realized_gain.jsonl`` (completed realized-gain
    records for admitted frames only).
    """

    #: Issue #118 fix-and-recheck. See arguments/__init__.py's
    #: ``streaming_shadow_utility_debt_normalization`` docstring for the
    #: full rationale. "raw" is the pre-existing, unmodified behavior.
    _VALID_DEBT_NORMALIZATIONS = ("raw", "per_gaussian")

    def __init__(
        self,
        enabled: bool,
        log_dir: Optional[str] = None,
        weights: Optional[UtilityWeights] = None,
        realized_gain_horizon: int = 200,
        gaussian_history: Optional[GaussianEvidenceHistory] = None,
        under_supported_threshold: float = 0.01,
        debt_normalization: str = "raw",
    ) -> None:
        self._enabled = bool(enabled)
        self.weights = weights or UtilityWeights()
        self.gaussian_history = gaussian_history
        self.under_supported_threshold = under_supported_threshold
        if debt_normalization not in self._VALID_DEBT_NORMALIZATIONS:
            raise ValueError(
                f"debt_normalization must be one of {self._VALID_DEBT_NORMALIZATIONS}, got {debt_normalization!r}"
            )
        self.debt_normalization = debt_normalization
        self.gain_tracker = RealizedGainTracker(horizon_iterations=realized_gain_horizon)
        # Most recent EvidenceSummary the training loop handed us (real-
        # opacity-backed under_supported_mass -- see observe_candidate's
        # docstring). ``None`` until the first population-energy-terms
        # computation happens after this policy is enabled.
        self.last_summary = None
        self._scores_path: Optional[str] = None
        self._gain_path: Optional[str] = None
        self._cost_samples_ms: list = []
        if self._enabled and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._scores_path = os.path.join(log_dir, "scores.jsonl")
            self._gain_path = os.path.join(log_dir, "realized_gain.jsonl")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    def observe_candidate(
        self,
        *,
        frame_id: int,
        camera_key: int,
        admitted: bool,
        iteration: int,
        pose_confidence: float,
        image_quality: float,
        depth_valid_fraction: float,
        n_gaussians: int,
        resolution_bucket: str = "default",
        render_pkg: Optional[dict] = None,
        gt_image: Optional[torch.Tensor] = None,
        redundancy: float = 0.0,
        useful_parallax: float = 0.0,
        new_surface: float = 0.0,
        under_supported_mass: float = 0.0,
        topology_version: int = 0,
        map_version: int = 0,
    ) -> Optional[FrameUtilityScore]:
        """Score one candidate frame, after the real admission decision has
        already been made. ``admitted`` is informational only (used to
        decide whether to also register realized-gain tracking) -- this
        method cannot change it.

        ``under_supported_mass`` is expected to be sourced from
        :class:`eligsir.mapping.evidence.contracts.EvidenceSummary`
        (built via ``EvidenceBuilder.build_summary``, which has access to
        the real per-Gaussian opacity ``energy_terms.alpha`` this module
        does not carry) -- passed in rather than recomputed here so this
        method never has to guess at an opacity proxy from fields
        :class:`GaussianEvidence` doesn't actually have.
        """
        if not self._enabled:
            return None
        t0 = time.monotonic()

        input_evidence = InputEvidence(
            pose_confidence=float(pose_confidence),
            image_quality=float(image_quality),
            depth_valid_fraction=float(depth_valid_fraction),
            geometry_reliability_mean=float(depth_valid_fraction),
            meta=EvidenceMeta(frame_id=frame_id, map_version=map_version, topology_version=topology_version),
        )

        render_evidence = None
        if render_pkg is not None and gt_image is not None and "render" in render_pkg:
            with torch.no_grad():
                rgb_residual = (render_pkg["render"].detach() - gt_image.detach()).detach()
            render_evidence = RenderEvidence(
                rgb_residual=rgb_residual,
                depth_residual=None,
                alpha=render_pkg.get("alpha"),
                rendered_depth=render_pkg.get("rendered_depth"),
                transmittance=None,
                meta=input_evidence.meta,
            )

        gaussian_evidence = self._current_gaussian_evidence(topology_version)

        # Issue #118: only the value fed into the *scoring formula* is
        # normalized -- `under_supported_mass` below (passed to
        # gain_tracker.register_admission a few lines down, and logged
        # verbatim into realized_gain.jsonl's under_supported_mass_before/
        # after diagnostic fields) stays raw regardless of this flag, since
        # that's a separate LSOM-style diagnostic, not a scoring input.
        scoring_under_supported_mass = under_supported_mass
        if self.debt_normalization == "per_gaussian" and n_gaussians > 0:
            scoring_under_supported_mass = under_supported_mass / float(n_gaussians)

        score = score_frame_utility(
            frame_id=frame_id,
            input_evidence=input_evidence,
            render_evidence=render_evidence,
            gaussian_evidence=gaussian_evidence,
            new_surface=new_surface,
            under_supported_mass=scoring_under_supported_mass,
            useful_parallax=useful_parallax,
            redundancy=redundancy,
            n_gaussians=n_gaussians,
            resolution_bucket=resolution_bucket,
            weights=self.weights,
            meta=input_evidence.meta,
        )
        self._log_score(score)

        if admitted:
            self.gain_tracker.register_admission(
                camera_key=camera_key,
                frame_id=frame_id,
                iteration=iteration,
                loss_before=float("nan"),  # filled in by the first observe_training_loss call
                score_map_version=score.meta.map_version,
                score_topology_version=score.meta.topology_version,
                predicted_utility=score.utility,
                under_supported_mass_before=under_supported_mass,
            )

        self._cost_samples_ms.append((time.monotonic() - t0) * 1000.0)
        return score

    def register_cached_admission(
        self,
        *,
        camera_key: int,
        frame_id: int,
        admitted_iteration: int,
        score_map_version: int,
        score_topology_version: Optional[int],
        predicted_utility: Optional[float],
        under_supported_mass_before: Optional[float],
    ) -> None:
        """Start gain tracking for a deferred score at its later commit."""
        if not self._enabled:
            return
        self.gain_tracker.register_admission(
            camera_key=camera_key,
            frame_id=frame_id,
            iteration=admitted_iteration,
            loss_before=float("nan"),
            score_map_version=score_map_version,
            score_topology_version=score_topology_version,
            predicted_utility=predicted_utility,
            under_supported_mass_before=under_supported_mass_before,
        )

    def observe_training_loss(
        self,
        camera_key: int,
        iteration: int,
        loss: float,
        under_supported_mass: Optional[float] = None,
    ) -> None:
        """Called once per training iteration with the loss the loop already
        computed. No-op unless ``camera_key`` is a tracked pending record
        (dict lookup only -- see module docstring re: near-zero cost)."""
        if not self._enabled:
            return
        record = self.gain_tracker._pending.get(camera_key)  # noqa: SLF001 -- same-package peek
        if record is not None and record.loss_after is None and record.loss_before != record.loss_before:
            # NaN sentinel from register_admission -- this is the first
            # training-loss observation for this camera post-admission.
            record.loss_before = float(loss)
        completed = self.gain_tracker.observe_loss(
            camera_key, iteration, loss, under_supported_mass=under_supported_mass
        )
        if completed is not None:
            self._log_gain(completed)

    # ------------------------------------------------------------------
    def cost_overhead_ms_summary(self) -> dict:
        """Mean/max wall-clock overhead of :meth:`observe_candidate` calls
        (deliverable 5, policy-cost measurement). Empty summary when never
        called or disabled.
        """
        if not self._cost_samples_ms:
            return {"n_samples": 0, "mean_ms": 0.0, "max_ms": 0.0}
        n = len(self._cost_samples_ms)
        return {
            "n_samples": n,
            "mean_ms": sum(self._cost_samples_ms) / n,
            "max_ms": max(self._cost_samples_ms),
        }

    def _current_gaussian_evidence(self, topology_version: int) -> Optional[GaussianEvidence]:
        if self.gaussian_history is None:
            return None
        try:
            return self.gaussian_history.get(topology_version)
        except StaleEvidenceError:
            return None

    def _log_score(self, score: FrameUtilityScore) -> None:
        if not self._scores_path:
            return
        with open(self._scores_path, "a") as f:
            f.write(json.dumps(_score_to_json(score)) + "\n")

    def _log_gain(self, record: RealizedGainRecord) -> None:
        if not self._gain_path:
            return
        with open(self._gain_path, "a") as f:
            f.write(json.dumps(_gain_to_json(record)) + "\n")
