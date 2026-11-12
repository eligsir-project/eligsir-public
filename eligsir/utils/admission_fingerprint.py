"""Small runtime admission-fingerprint adapter.

The streaming trainer needs a bounded report-only accumulator.  This module
keeps that runtime contract local to the publication export; campaign
reproduction and historical sidecar tooling are intentionally not included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Dict, List, Optional

import numpy as np


OBSERVATION_SAMPLE_CAP = 256
BACKSTOP_TRIGGERS = ("soft_max_gap", "hard_starvation_ceiling")
STARVATION_TRIGGERS = BACKSTOP_TRIGGERS + (
    "exposure_starvation",
    "blur_starvation",
    "low_overlap_starvation",
)
EVIDENCE_TRIGGERS = ("normal", "novelty_force")
BACKSTOP_OWNER_SHARE_MAX = 0.50
MIN_ADMITS_FOR_GATE = 20
MAX_UNCLASSIFIED_OWNER_SHARE = 0.10


class _BoundedReservoirSummary:
    __slots__ = ("cap", "_values", "count", "_min", "_max", "_rng")

    def __init__(self, cap: int = OBSERVATION_SAMPLE_CAP, seed: int = 0):
        self.cap = int(cap)
        self._values: List[float] = []
        self.count = 0
        self._min: Optional[float] = None
        self._max: Optional[float] = None
        self._rng = random.Random(seed)

    def add(self, value: float) -> None:
        value = float(value)
        self.count += 1
        self._min = value if self._min is None else min(self._min, value)
        self._max = value if self._max is None else max(self._max, value)
        if len(self._values) < self.cap:
            self._values.append(value)
            return
        index = self._rng.randrange(self.count)
        if index < self.cap:
            self._values[index] = value

    def __len__(self) -> int:
        return len(self._values)

    def summary(self) -> Dict[str, float]:
        if not self._values:
            return {}
        values = np.asarray(self._values, dtype=np.float64)
        return {
            "p10": float(np.percentile(values, 10)),
            "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)),
            "mean": float(np.mean(values)),
            "min": float(self._min),
            "max": float(self._max),
            "n": self.count,
            "sampled_n": len(self._values),
        }


def _seed(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")


def keyframe_decision_sidecar_fields(decision: object) -> Dict[str, Any]:
    """Return the stable sidecar fields shared by both decision sites."""
    def get(name: str, default: Any = None) -> Any:
        return getattr(decision, name, default)

    return {
        "starvation_age_source_s": get("starvation_age_source_s"),
        "starvation_age_source_frames": get("starvation_age_source_frames"),
        "starvation_age_basis": get("starvation_age_basis", "unavailable"),
        "soft_backstop_mode": get("soft_backstop_mode"),
        "soft_backstop_domain": get("soft_backstop_domain"),
        "hard_backstop_domain": get("hard_backstop_domain"),
        "soft_backstop_threshold_s": get("soft_backstop_threshold_s"),
        "soft_backstop_threshold_frames": get("soft_backstop_threshold_frames"),
        "hard_backstop_threshold_s": get("hard_backstop_threshold_s"),
        "hard_backstop_threshold_frames": get("hard_backstop_threshold_frames"),
        "soft_backstop_preemptable": int(bool(get("soft_backstop_preemptable", False))),
        "soft_backstop_triggered": int(bool(get("soft_backstop_triggered", False))),
        "hard_backstop_triggered": int(bool(get("hard_backstop_triggered", False))),
        "motion_trigger": get("motion_trigger", "none"),
        "translation_velocity_m_s": get("translation_velocity_m_s"),
        "translation_velocity_ema_m_s": get("translation_velocity_ema_m_s"),
        "rotation_velocity_deg_s": get("rotation_velocity_deg_s"),
        "rotation_velocity_ema_deg_s": get("rotation_velocity_ema_deg_s"),
        "source_dt_s": get("source_dt_s"),
        "source_frame_delta": get("source_frame_delta"),
        "implied_input_fps": get("implied_input_fps"),
        "effective_min_translation_m": get("effective_min_translation_m"),
        "controller_target_interval_s": get("controller_target_interval_s"),
        "controller_target_interval_source_frames": get("controller_target_interval_source_frames"),
        "timebase_source": get("timebase_source", "unavailable"),
        "threshold_clamp": get("threshold_clamp", "n/a"),
    }


@dataclass
class AdmissionFingerprint:
    candidates_evaluated: int = 0
    admitted_total: int = 0
    rejected_total: int = 0
    admit_owners: Dict[str, int] = field(default_factory=dict)
    reject_owners: Dict[str, int] = field(default_factory=dict)
    candidate_reject_owners: Dict[str, int] = field(default_factory=dict)
    prefilter_reject_owners: Dict[str, int] = field(default_factory=dict)
    prefilter_rejected_total: int = 0
    force_hard_count: int = 0
    force_soft_count: int = 0
    gate_bypasses: Dict[str, int] = field(default_factory=dict)
    novelty_gate_outcome_counts: Dict[str, int] = field(default_factory=dict)
    novelty_gate_unavailable_reason_counts: Dict[str, int] = field(default_factory=dict)
    novelty_gate_none_cause_counts: Dict[str, int] = field(default_factory=dict)
    novelty_force_outcome_counts: Dict[str, int] = field(default_factory=dict)
    novelty_force_unavailable_reason_counts: Dict[str, int] = field(default_factory=dict)
    novelty_force_none_cause_counts: Dict[str, int] = field(default_factory=dict)
    novelty_force_triggered: int = 0
    novelty_force_decisive: int = 0
    gap_histogram: Dict[int, int] = field(default_factory=dict)
    gaps_at_max_gap: int = 0
    max_gap: int = 0
    target_admit_fps: float = 0.0
    input_fps: float = 0.0
    realized_admit_fps: Optional[float] = None
    realized_source_frames_per_admit: Optional[float] = None
    gap_percentiles: Dict[str, float] = field(default_factory=dict)
    nominal_policy: str = ""
    admission_realized_mismatch: bool = False
    mismatch_reason: str = ""
    precedence_config_warnings: List[str] = field(default_factory=list)
    velocity_adaptive: bool = False
    timebase_source: str = "unknown"
    implied_input_fps: Optional[float] = None
    source_elapsed_s: float = 0.0
    controller_target_interval_s: Optional[float] = None
    threshold_evals: int = 0
    threshold_floor_hits: int = 0
    threshold_ceil_hits: int = 0
    realized_admit_fps_source_time: Optional[float] = None
    target_vs_realized_rate_error_fps: Optional[float] = None
    target_vs_realized_rate_rel_error: Optional[float] = None
    _observation_samples: Dict[str, Dict[str, _BoundedReservoirSummary]] = field(default_factory=dict, repr=False)
    _controller_samples: Dict[str, _BoundedReservoirSummary] = field(default_factory=dict, repr=False)
    _canonical_funnel: Optional[Dict[str, Any]] = field(default=None, repr=False)
    coverage_consumer_outcome_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    coverage_consumer_none_cause_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    admit_trigger_counts: Dict[str, int] = field(default_factory=dict)
    admit_motion_trigger_counts: Dict[str, int] = field(default_factory=dict)
    candidate_motion_trigger_counts: Dict[str, int] = field(default_factory=dict)
    soft_backstop_raw_crossings: int = 0
    hard_backstop_raw_crossings: int = 0
    starvation_age_basis_counts: Dict[str, int] = field(default_factory=dict)
    _admit_trigger_age_s: _BoundedReservoirSummary = field(
        default_factory=lambda: _BoundedReservoirSummary(seed=_seed("trigger_age_s")),
        repr=False,
    )
    _admit_trigger_age_frames: _BoundedReservoirSummary = field(
        default_factory=lambda: _BoundedReservoirSummary(seed=_seed("trigger_age_frames")),
        repr=False,
    )
    trigger_age_s_percentiles: Dict[str, float] = field(default_factory=dict)
    trigger_age_frames_percentiles: Dict[str, float] = field(default_factory=dict)
    admission_gate: Dict[str, object] = field(default_factory=dict)

    def record_candidate(self) -> None:
        self.candidates_evaluated += 1

    def record_admit(self, reason: str, gap: int, forced_hard: bool = False, forced_soft: bool = False) -> None:
        self.admitted_total += 1
        self.admit_owners[reason] = self.admit_owners.get(reason, 0) + 1
        self.force_hard_count += int(bool(forced_hard))
        self.force_soft_count += int(bool(forced_soft))
        self.gap_histogram[int(gap)] = self.gap_histogram.get(int(gap), 0) + 1
        if self.max_gap and self.max_gap <= int(gap) <= self.max_gap + 1:
            self.gaps_at_max_gap += 1

    def record_reject(self, reason: str, *, population: str = "candidate_evaluated") -> None:
        self.rejected_total += 1
        self.reject_owners[reason] = self.reject_owners.get(reason, 0) + 1
        if population == "candidate_evaluated":
            target = self.candidate_reject_owners
        elif population == "prefilter_evaluated":
            self.prefilter_rejected_total += 1
            target = self.prefilter_reject_owners
        else:
            raise ValueError(f"unknown rejection population: {population!r}")
        target[reason] = target.get(reason, 0) + 1

    def record_gate_bypass(self, gate_name: str) -> None:
        self.gate_bypasses[gate_name] = self.gate_bypasses.get(gate_name, 0) + 1

    def record_novelty_gate(self, trace: Optional[dict]) -> None:
        if not trace:
            return
        outcome = trace.get("novelty_gate_outcome")
        if not outcome or outcome == "not_applicable":
            return
        self.novelty_gate_outcome_counts[outcome] = self.novelty_gate_outcome_counts.get(outcome, 0) + 1
        reason = trace.get("novelty_gate_unavailable_reason")
        if reason:
            self.novelty_gate_unavailable_reason_counts[reason] = self.novelty_gate_unavailable_reason_counts.get(reason, 0) + 1
        cause = trace.get("novelty_gate_none_cause")
        if cause:
            self.novelty_gate_none_cause_counts[cause] = self.novelty_gate_none_cause_counts.get(cause, 0) + 1

    def record_novelty_force(self, trace: Optional[dict], admitted: bool) -> None:
        if not trace or not trace.get("novelty_force_outcome"):
            return
        outcome = trace["novelty_force_outcome"]
        self.novelty_force_outcome_counts[outcome] = self.novelty_force_outcome_counts.get(outcome, 0) + 1
        reason = trace.get("novelty_force_unavailable_reason")
        if reason:
            self.novelty_force_unavailable_reason_counts[reason] = self.novelty_force_unavailable_reason_counts.get(reason, 0) + 1
        cause = trace.get("novelty_force_none_cause")
        if cause:
            self.novelty_force_none_cause_counts[cause] = self.novelty_force_none_cause_counts.get(cause, 0) + 1
        self.novelty_force_triggered += int(outcome == "triggered")
        self.novelty_force_decisive += int(admitted and trace.get("novelty_force_decisive"))

    @staticmethod
    def _finite(value: Any) -> bool:
        try:
            return value is not None and bool(np.isfinite(value))
        except (TypeError, ValueError):
            return False

    def record_controller_state(self, decision: object) -> None:
        fields = {
            "translation_velocity_inst_m_s": "translation_velocity_m_s",
            "translation_velocity_ema_m_s": "translation_velocity_ema_m_s",
            "rotation_velocity_inst_deg_s": "rotation_velocity_deg_s",
            "rotation_velocity_ema_deg_s": "rotation_velocity_ema_deg_s",
            "adaptive_threshold_m": "effective_min_translation_m",
        }
        for output, attr in fields.items():
            value = getattr(decision, attr, None)
            if self._finite(value):
                sampler = self._controller_samples.setdefault(output, _BoundedReservoirSummary(seed=_seed(output)))
                sampler.add(float(value))
        clamp = getattr(decision, "threshold_clamp", "n/a")
        if clamp in ("none", "floor", "ceil"):
            self.threshold_evals += 1
            self.threshold_floor_hits += int(clamp == "floor")
            self.threshold_ceil_hits += int(clamp == "ceil")
        timebase = getattr(decision, "timebase_source", None)
        if timebase:
            self.timebase_source = str(timebase)
        implied = getattr(decision, "implied_input_fps", None)
        if self._finite(implied):
            self.implied_input_fps = float(implied)
        interval = getattr(decision, "controller_target_interval_s", None)
        if self._finite(interval):
            self.controller_target_interval_s = float(interval)
            self.velocity_adaptive = True
        dt = getattr(decision, "source_dt_s", None)
        if self._finite(dt) and float(dt) > 0:
            self.source_elapsed_s += float(dt)

    def record_decision_observation(self, decision: object) -> None:
        outcome = f"admit:{getattr(decision, 'reason', 'unknown')}" if bool(getattr(decision, "admitted", False)) else f"reject:{getattr(decision, 'reason', 'unknown')}"
        trace = getattr(decision, "trace", None) or {}
        observed = trace.get("observation_values") or {}
        values = {
            "translation_m": getattr(decision, "pose_delta_m", None),
            "rotation_deg": getattr(decision, "pose_delta_deg", None),
            "render_overlap": getattr(decision, "render_overlap", None),
            "coverage_new_voxel_fraction": observed.get("coverage_new_voxel_fraction", trace.get("novelty_gate_fraction")),
            "blur_laplacian": observed.get("blur_laplacian"),
            "exposure_delta": observed.get("exposure_delta"),
            "quality": observed.get("quality", getattr(decision, "quality", None)),
        }
        bucket = self._observation_samples.setdefault(outcome, {})
        for name, value in values.items():
            if self._finite(value):
                bucket.setdefault(name, _BoundedReservoirSummary(seed=_seed(f"{outcome}:{name}"))).add(float(value))
        for consumer, item in (trace.get("coverage_consumers") or {}).items():
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            if status:
                counts = self.coverage_consumer_outcome_counts.setdefault(consumer, {})
                counts[status] = counts.get(status, 0) + 1
            cause = item.get("none_cause")
            if cause:
                counts = self.coverage_consumer_none_cause_counts.setdefault(consumer, {})
                counts[cause] = counts.get(cause, 0) + 1

    def record_backstop_state(self, decision: object, admitted: bool) -> None:
        motion = getattr(decision, "motion_trigger", None)
        if motion:
            self.candidate_motion_trigger_counts[motion] = self.candidate_motion_trigger_counts.get(motion, 0) + 1
        self.soft_backstop_raw_crossings += int(bool(getattr(decision, "soft_backstop_triggered", False)))
        self.hard_backstop_raw_crossings += int(bool(getattr(decision, "hard_backstop_triggered", False)))
        basis = getattr(decision, "starvation_age_basis", None)
        if basis:
            self.starvation_age_basis_counts[basis] = self.starvation_age_basis_counts.get(basis, 0) + 1
        if not admitted:
            return
        trace = getattr(decision, "trace", None)
        trigger = trace.get("candidate_trigger") if isinstance(trace, dict) else None
        trigger = trigger or getattr(decision, "reason", None) or "unknown"
        self.admit_trigger_counts[trigger] = self.admit_trigger_counts.get(trigger, 0) + 1
        if motion:
            self.admit_motion_trigger_counts[motion] = self.admit_motion_trigger_counts.get(motion, 0) + 1
        age_s = getattr(decision, "starvation_age_source_s", None)
        if self._finite(age_s):
            self._admit_trigger_age_s.add(float(age_s))
        age_f = getattr(decision, "starvation_age_source_frames", None)
        if self._finite(age_f):
            self._admit_trigger_age_frames.add(float(age_f))

    @staticmethod
    def _percentiles(values: List[float]) -> Dict[str, float]:
        if not values:
            return {}
        arr = np.asarray(values, dtype=np.float64)
        return {"p10": float(np.percentile(arr, 10)), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "n": float(len(arr))}

    def attach_canonical_funnel(self, *, admission_cadence: Optional[dict], realized_cadence: Optional[dict], ingestion: Optional[dict]) -> None:
        clocks = (realized_cadence or {}).get("clocks") or {}
        unknown = "no authoritative producer is wired on this run; unknown is not zero"

        def stage(value: Any, source: str, reason: str) -> Dict[str, Any]:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return {"count": value, "status": "measured", "source": source}
            return {"count": None, "status": "unknown", "source": source, "reason": reason}

        self._canonical_funnel = {"stages": {
            "source_observed": {"count": None, "status": "unknown", "source": "unresolved #805.source_observed producer", "reason": "admission_cadence.source_observed is not an authoritative #805 source_observed producer; " + unknown},
            "source_frame_materialized_valid": {"count": None, "status": "unknown", "source": "unresolved #805.source_frame_materialized_valid producer", "reason": unknown},
            "source_released": stage(clocks.get("source_released"), "realized_cadence.clocks.source_released", "#289 source_released clock was unavailable"),
            "prefilter_evaluated": {"count": None, "status": "unknown", "source": "unresolved #805.prefilter_evaluated producer", "reason": unknown},
            "prefilter_eligible": {"count": None, "status": "unknown", "source": "unresolved #805.prefilter_eligible producer", "reason": unknown},
            "candidate_probed": {"count": None, "status": "unknown", "source": "unresolved #805.candidate_probed producer", "reason": unknown},
            "candidate_evaluated": stage(clocks.get("candidate_evaluated"), "realized_cadence.clocks.candidate_evaluated", "#289 candidate_evaluated clock was unavailable"),
            "keyframe_committed": stage(clocks.get("keyframe_admitted"), "realized_cadence.clocks.keyframe_admitted", "#289 keyframe_admitted clock was unavailable"),
        }, "source_decimation": dict((ingestion or {}).get("source_decimation") or {})}

    def _funnel(self) -> Dict[str, Any]:
        if self._canonical_funnel is None:
            self.attach_canonical_funnel(admission_cadence=None, realized_cadence=None, ingestion=None)
        return self._canonical_funnel or {}

    def _owner_shares(self, funnel: Dict[str, Any]) -> Dict[str, Any]:
        def share(values: Dict[str, int], stage_name: str) -> Dict[str, Any]:
            info = funnel["stages"][stage_name]
            denom = info.get("count")
            if info.get("status") == "measured" and denom:
                if sum(values.values()) > denom:
                    return {"denominator": stage_name, "denominator_count": denom, "status": "inconsistent_population", "values": None}
                return {"denominator": stage_name, "denominator_count": denom, "status": "measured", "values": {k: v / denom for k, v in values.items()}}
            return {"denominator": stage_name, "denominator_count": denom, "status": "unknown" if denom is None else "zero_denominator", "values": None}

        pre = funnel["stages"]["prefilter_eligible"]
        committed = funnel["stages"]["keyframe_committed"]
        eligible = pre.get("count")
        committed_count = committed.get("count")
        rate = committed_count / eligible if pre.get("status") == "measured" and committed.get("status") == "measured" and eligible else None
        return {
            "admit": share(self.admit_owners, "candidate_evaluated"),
            "candidate_reject": share(self.candidate_reject_owners, "candidate_evaluated"),
            "prefilter_reject": share(self.prefilter_reject_owners, "prefilter_evaluated"),
            "reject": {"denominator": None, "denominator_count": None, "status": "mixed_populations_no_share", "values": None},
            "keyframe_commit_rate_per_prefilter_eligible": {"denominator": "prefilter_eligible", "denominator_count": eligible, "numerator": "keyframe_committed", "numerator_count": committed_count, "value": rate, "status": "measured" if rate is not None else "unknown"},
        }

    def _compute_gate(self) -> None:
        admitted = self.admitted_total
        gate: Dict[str, Any] = {
            "threshold_backstop_owner_share_max": BACKSTOP_OWNER_SHARE_MAX,
            "min_admits_for_verdict": MIN_ADMITS_FOR_GATE,
            "nominal_policy": self.nominal_policy,
            "admitted_total": admitted,
            "admit_trigger_counts": dict(self.admit_trigger_counts),
            "admit_motion_trigger_counts": dict(self.admit_motion_trigger_counts),
            "candidate_motion_trigger_counts": dict(self.candidate_motion_trigger_counts),
            "soft_backstop_raw_crossings": self.soft_backstop_raw_crossings,
            "hard_backstop_raw_crossings": self.hard_backstop_raw_crossings,
            "starvation_age_basis_counts": dict(self.starvation_age_basis_counts),
            "trigger_age_source_s": dict(self.trigger_age_s_percentiles),
            "trigger_age_source_frames": dict(self.trigger_age_frames_percentiles),
        }
        if admitted <= 0:
            gate["verdict"] = "NO_ADMISSIONS"
            self.admission_gate = gate
            return
        share = lambda names: sum(self.admit_trigger_counts.get(n, 0) for n in names) / admitted
        backstop = share(BACKSTOP_TRIGGERS)
        starvation = share(STARVATION_TRIGGERS)
        evidence = share(EVIDENCE_TRIGGERS)
        gate.update(backstop_owner_share=backstop, starvation_owner_share=starvation, evidence_owner_share=evidence, unclassified_owner_share=max(0.0, 1 - starvation - evidence))
        motion_admits = sum(v for k, v in self.admit_motion_trigger_counts.items() if k != "none")
        if motion_admits:
            gate["motion_owner_decomposition"] = {"translation_only": self.admit_motion_trigger_counts.get("translation", 0) / motion_admits, "rotation_only": self.admit_motion_trigger_counts.get("rotation", 0) / motion_admits, "both": self.admit_motion_trigger_counts.get("both", 0) / motion_admits, "n_motion_admits": motion_admits}
        if self.nominal_policy == "fixed_stride":
            purity = self.admit_owners.get("fixed_stride", 0) / admitted
            gate["control_purity"] = purity
            gate["verdict"] = "CONTROL_PURE" if purity >= 1 else "CONTROL_LEAKED"
        elif self.nominal_policy in ("all", "", None):
            gate["verdict"] = "NOT_APPLICABLE_UNGATED"
        elif admitted < MIN_ADMITS_FOR_GATE:
            gate["verdict"] = "INSUFFICIENT_ADMITS"
        elif backstop >= BACKSTOP_OWNER_SHARE_MAX:
            gate["verdict"] = "BACKSTOP_DOMINATED"
        elif starvation + evidence <= 0:
            gate["verdict"] = "NOT_MEASURED"
            gate["verdict_note"] = "no admission carried a recognizable precedence trigger; backstop_owner_share == 0 means unmeasured, not clean"
        elif gate["unclassified_owner_share"] > MAX_UNCLASSIFIED_OWNER_SHARE:
            gate["verdict"] = "PARTIALLY_MEASURED"
        else:
            gate["verdict"] = "CLASSICAL_LIVE"
        self.admission_gate = gate

    def finalize(self, total_source_frames: Optional[int] = None) -> None:
        if self.admitted_total and self.gap_histogram:
            gaps = np.repeat(list(self.gap_histogram.keys()), list(self.gap_histogram.values())).astype(float)
            self.realized_source_frames_per_admit = float(np.mean(gaps))
            self.gap_percentiles = {"p10": float(np.percentile(gaps, 10)), "p50": float(np.percentile(gaps, 50)), "p90": float(np.percentile(gaps, 90)), "p95": float(np.percentile(gaps, 95)), "p99": float(np.percentile(gaps, 99)), "max": float(np.max(gaps)), "mean": float(np.mean(gaps))}
            if total_source_frames and self.input_fps > 0:
                self.realized_admit_fps = self.admitted_total / (total_source_frames / self.input_fps)
        if self.admitted_total and self.source_elapsed_s > 0:
            self.realized_admit_fps_source_time = self.admitted_total / self.source_elapsed_s
            if self.target_admit_fps > 0:
                self.target_vs_realized_rate_error_fps = self.realized_admit_fps_source_time - self.target_admit_fps
                self.target_vs_realized_rate_rel_error = self.target_vs_realized_rate_error_fps / self.target_admit_fps
        self.trigger_age_s_percentiles = self._admit_trigger_age_s.summary()
        self.trigger_age_frames_percentiles = self._admit_trigger_age_frames.summary()
        if self.nominal_policy in ("all", "", None):
            self.admission_realized_mismatch = False
            self.mismatch_reason = "NON_GATED_ADMISSION"
        elif self.admitted_total < MIN_ADMITS_FOR_GATE:
            self.admission_realized_mismatch = False
            self.mismatch_reason = f"INSUFFICIENT_ADMITS ({self.admitted_total} < {MIN_ADMITS_FOR_GATE})"
        else:
            force_share = (self.force_hard_count + self.force_soft_count) / self.admitted_total
            self.admission_realized_mismatch = force_share >= 0.70
            self.mismatch_reason = (f"FORCE_DOMINATED ({force_share:.1%} force-owned >= 70%, nominal={self.nominal_policy})" if self.admission_realized_mismatch else "NOMINAL_POLICY_REALIZED")
        self._compute_gate()

    def to_dict(self) -> Dict[str, Any]:
        funnel = self._funnel()
        return {
            "schema_version": 2,
            "admission_gate": dict(self.admission_gate),
            "nominal_policy": self.nominal_policy,
            "candidates_evaluated": self.candidates_evaluated,
            "admitted_total": self.admitted_total,
            "rejected_total": self.rejected_total,
            "admit_owners": dict(self.admit_owners),
            "reject_owners": dict(self.reject_owners),
            "candidate_reject_owners": dict(self.candidate_reject_owners),
            "prefilter_reject_owners": dict(self.prefilter_reject_owners),
            "prefilter_rejected_total": self.prefilter_rejected_total,
            "force_hard_count": self.force_hard_count,
            "force_soft_count": self.force_soft_count,
            "force_share": (self.force_hard_count + self.force_soft_count) / self.admitted_total if self.admitted_total else 0.0,
            "gate_bypasses": dict(self.gate_bypasses),
            "precedence_config_warnings": list(self.precedence_config_warnings),
            "novelty_gate": {"outcome_counts": dict(self.novelty_gate_outcome_counts), "unavailable_reason_counts": dict(self.novelty_gate_unavailable_reason_counts), "none_cause_counts": dict(self.novelty_gate_none_cause_counts)},
            "novelty_force": {"triggered": self.novelty_force_triggered, "decisive": self.novelty_force_decisive, "outcome_counts": dict(self.novelty_force_outcome_counts), "unavailable_reason_counts": dict(self.novelty_force_unavailable_reason_counts), "none_cause_counts": dict(self.novelty_force_none_cause_counts)},
            "coverage_consumers": {key: {"outcome_counts": dict(value), "none_cause_counts": dict(self.coverage_consumer_none_cause_counts.get(key, {}))} for key, value in self.coverage_consumer_outcome_counts.items()},
            "gap_histogram": dict(self.gap_histogram),
            "gaps_at_max_gap": self.gaps_at_max_gap,
            "gaps_at_max_gap_share": self.gaps_at_max_gap / self.admitted_total if self.admitted_total else 0.0,
            "max_gap": self.max_gap,
            "target_admit_fps": self.target_admit_fps,
            "input_fps": self.input_fps,
            "realized_admit_fps": self.realized_admit_fps,
            "realized_admit_fps_legacy_basis": "nominal_input_fps_over_total_source_frames",
            "realized_admit_fps_source_time": self.realized_admit_fps_source_time,
            "realized_source_frames_per_admit": self.realized_source_frames_per_admit,
            "gap_percentiles": dict(self.gap_percentiles),
            "observation_summary": {outcome: {metric: sampler.summary() for metric, sampler in metrics.items()} for outcome, metrics in self._observation_samples.items()},
            "observation_sample_cap_per_metric_per_outcome": OBSERVATION_SAMPLE_CAP,
            "controller_sample_cap_per_metric": OBSERVATION_SAMPLE_CAP,
            "funnel": funnel,
            "owner_shares": self._owner_shares(funnel),
            "admission_realized_mismatch": self.admission_realized_mismatch,
            "mismatch_reason": self.mismatch_reason,
            "controller": {
                "velocity_adaptive": self.velocity_adaptive,
                "translation_velocity_inst_m_s": self._controller_samples.get("translation_velocity_inst_m_s", _BoundedReservoirSummary()).summary(),
                "translation_velocity_ema_m_s": self._controller_samples.get("translation_velocity_ema_m_s", _BoundedReservoirSummary()).summary(),
                "rotation_velocity_inst_deg_s": self._controller_samples.get("rotation_velocity_inst_deg_s", _BoundedReservoirSummary()).summary(),
                "rotation_velocity_ema_deg_s": self._controller_samples.get("rotation_velocity_ema_deg_s", _BoundedReservoirSummary()).summary(),
                "adaptive_threshold_m": self._controller_samples.get("adaptive_threshold_m", _BoundedReservoirSummary()).summary(),
                "threshold_evals": self.threshold_evals,
                "threshold_floor_hits": self.threshold_floor_hits,
                "threshold_ceil_hits": self.threshold_ceil_hits,
                "threshold_floor_frac": self.threshold_floor_hits / self.threshold_evals if self.threshold_evals else None,
                "threshold_ceil_frac": self.threshold_ceil_hits / self.threshold_evals if self.threshold_evals else None,
                "timebase_source": self.timebase_source,
                "implied_input_fps": self.implied_input_fps,
                "configured_input_fps": self.input_fps,
                "source_elapsed_s": self.source_elapsed_s,
                "controller_target_interval_s": self.controller_target_interval_s,
                "realized_admit_fps_source_time": self.realized_admit_fps_source_time,
                "target_vs_realized_rate_error_fps": self.target_vs_realized_rate_error_fps,
                "target_vs_realized_rate_rel_error": self.target_vs_realized_rate_rel_error,
            },
        }
