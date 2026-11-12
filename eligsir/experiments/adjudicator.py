"""Automated quantitative experiment adjudication.

Reads two or more runs' artifacts and applies a predeclared decision profile
to produce a durable experiment_result.json with one definitive verdict.

See issue #636 and docs/architecture/2026-08-15-issue636-quantitative-adjudication.md
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from eligsir.experiments.decision_profile import (
    DecisionProfile,
    profile_hash,
)


# =============================================================================
# Constants and enums
# =============================================================================

VERDICT_CLEAR_WIN = "CLEAR_WIN"
VERDICT_CLEAR_LOSS = "CLEAR_LOSS"
VERDICT_EQUIVALENT = "EQUIVALENT"
VERDICT_TRADEOFF = "TRADEOFF"
VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS = "INCONCLUSIVE_NEEDS_MORE_RUNS"
VERDICT_QUALITATIVE_REVIEW_REQUIRED = "QUALITATIVE_REVIEW_REQUIRED"

VERDICTS = {
    VERDICT_CLEAR_WIN,
    VERDICT_CLEAR_LOSS,
    VERDICT_EQUIVALENT,
    VERDICT_TRADEOFF,
    VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS,
    VERDICT_QUALITATIVE_REVIEW_REQUIRED,
}

VALIDITY_VALID = "VALID"
VALIDITY_INVALID_COMPARISON = "INVALID_COMPARISON"
VALIDITY_MANIPULATION_FAILED = "MANIPULATION_FAILED"
VALIDITY_MISSING_REQUIRED_EVIDENCE = "MISSING_REQUIRED_EVIDENCE"
VALIDITY_NO_COMPATIBLE_NOISE_PROFILE = "NO_COMPATIBLE_NOISE_PROFILE"

VALIDITIES = {
    VALIDITY_VALID,
    VALIDITY_INVALID_COMPARISON,
    VALIDITY_MANIPULATION_FAILED,
    VALIDITY_MISSING_REQUIRED_EVIDENCE,
    VALIDITY_NO_COMPATIBLE_NOISE_PROFILE,
}


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RunMetrics:
    """Metrics extracted from a single run."""
    run_dir: str
    report_path: str
    report_json: dict
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PairedDelta:
    """Per-pair delta for a metric."""
    metric: str
    baseline_val: float
    candidate_val: float
    delta: float  # candidate - baseline


@dataclass
class MetricResult:
    """Result for evaluating a single metric."""
    metric: str
    direction: str
    paired_deltas: List[PairedDelta]
    mean_delta: float
    ci_lower: float
    ci_upper: float
    practical_margin: float
    noise_margin: Optional[float] = None
    win_margin: Optional[float] = None


@dataclass
class GuardrailResult:
    """Result for evaluating a single guardrail."""
    metric: str
    guardrail_type: str  # "hard" or "cost"
    paired_deltas: List[PairedDelta]
    mean_delta: float
    max_regression: Optional[float] = None
    max_relative_regression: Optional[float] = None
    passes: Optional[bool] = None
    reason: str = ""


@dataclass
class ExperimentResult:
    """Complete experiment adjudication result."""
    schema_version: int = 1
    profile_id: str = ""
    profile_hash: str = ""
    declared_profile_hash: str = ""
    profile_hash_matches: bool = True
    baseline_run_dir: str = ""
    candidate_run_dirs: List[str] = field(default_factory=list)
    n_pairs: int = 0
    validity_state: str = VALIDITY_VALID
    validity_reasons: List[str] = field(default_factory=list)
    holdout_verification_level: str = ""
    seed_fade_status: List[str] = field(default_factory=list)
    eval_valid_status: List[str] = field(default_factory=list)
    primary_metric_results: List[MetricResult] = field(default_factory=list)
    secondary_metric_results: List[MetricResult] = field(default_factory=list)
    hard_guardrail_results: List[GuardrailResult] = field(default_factory=list)
    cost_guardrail_results: List[GuardrailResult] = field(default_factory=list)
    conflicting_metrics: List[Tuple[str, str, str]] = field(default_factory=list)
    partitions_with_regression: List[str] = field(default_factory=list)
    quantitative_verdict: str = ""
    qualitative_review_reason: Optional[str] = None
    qualitative_review_triggers: List[int] = field(default_factory=list)
    tradeoff_description: str = ""
    evidence_level: str = ""
    promotion_action: str = ""
    bootstrap_random_seed: int = 0


# =============================================================================
# Utility functions
# =============================================================================


def _find_latest_report_json(model_path: str) -> Optional[str]:
    """Find the latest report.json under a run's comparison/ directory.

    Reuses logic from eligsir.utils.baseline_comparison._find_latest_report_json
    without importing it, since that logic is simple and we want to keep
    this module torch-free.
    """
    import glob
    import re

    pattern = os.path.join(model_path, "comparison", "iter_*", "report.json")
    candidates = []
    for path in glob.glob(pattern):
        dirname = os.path.basename(os.path.dirname(path))
        m = re.fullmatch(r"iter_(\d+)", dirname)
        if not m:
            continue
        candidates.append((int(m.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def _get_metric_from_nested_path(
    data: Dict[str, Any],
    path: str,
) -> Optional[float]:
    """Extract a metric from nested dict using dot-separated path.

    E.g., "geometry.seed_fade.detected" extracts from data["geometry"]["seed_fade"]["detected"].
    Returns None if any component is missing.
    """
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


def resolve_noise_margin(
    noise_provider: Optional[Any],
    metric_name: str,
) -> Tuple[Optional[float], str]:
    """Resolve a noise margin from the provider.

    Args:
        noise_provider: A callable, dict-like, or None. If callable, called
            as provider(metric_name). If dict-like, accessed as
            provider.get(metric_name) or provider[metric_name]. Must return
            an object with a "noise_margin" attribute/key, or None.
        metric_name: The metric to look up.

    Returns:
        Tuple of (margin_or_None, reason_string). If either component is absent,
        returns (None, reason).
    """
    if noise_provider is None:
        return None, "no_noise_provider_supplied"

    # Try calling as a function
    result = None
    try:
        result = noise_provider(metric_name)
    except TypeError:
        # Not callable; try dict-like access
        try:
            result = noise_provider.get(metric_name)
        except (AttributeError, TypeError):
            try:
                result = noise_provider[metric_name]
            except (KeyError, TypeError):
                return None, "noise_provider_missing_metric"

    if result is None:
        return None, f"noise_provider_returned_none_for_{metric_name}"

    # Extract margin from the result
    margin = None
    if isinstance(result, (int, float)):
        margin = float(result)
    elif isinstance(result, dict) and "noise_margin" in result:
        margin = result["noise_margin"]
    elif hasattr(result, "noise_margin"):
        margin = getattr(result, "noise_margin")

    if margin is None:
        return None, f"noise_profile_missing_noise_margin_for_{metric_name}"

    return float(margin), "noise_margin_resolved"


def paired_bootstrap_ci(
    deltas: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Compute a paired bootstrap confidence interval on a list of deltas.

    Args:
        deltas: Per-pair differences (candidate - baseline).
        confidence: Confidence level, e.g., 0.95 for 95% CI.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (ci_lower, ci_upper).
    """
    if not deltas:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    deltas_arr = np.array(deltas)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        resampled = rng.choice(deltas_arr, size=len(deltas_arr), replace=True)
        bootstrap_means.append(np.mean(resampled))

    bootstrap_means = np.array(bootstrap_means)
    alpha = 1.0 - confidence
    ci_lower = np.quantile(bootstrap_means, alpha / 2.0)
    ci_upper = np.quantile(bootstrap_means, 1.0 - alpha / 2.0)

    return float(ci_lower), float(ci_upper)


# =============================================================================
# Validity checks
# =============================================================================


def check_validity(
    baseline_metrics: RunMetrics,
    candidate_metrics: List[RunMetrics],
    profile: DecisionProfile,
) -> Tuple[str, List[str]]:
    """Validate the comparison before scoring any metrics.

    Returns (validity_state, reasons_list).
    """
    reasons = []

    # Check eval_valid
    baseline_eval_valid = baseline_metrics.report_json.get("eval_valid")
    if baseline_eval_valid is False:
        reasons.append("baseline_eval_valid_is_false")

    for i, cand in enumerate(candidate_metrics):
        cand_eval_valid = cand.report_json.get("eval_valid")
        if cand_eval_valid is False:
            reasons.append(f"candidate_{i}_eval_valid_is_false")

    if reasons:
        return VALIDITY_INVALID_COMPARISON, reasons

    # Check holdout hash consistency (scored > realized > n_test)
    baseline_scored_hash = baseline_metrics.report_json.get(
        "eval_metadata", {}
    ).get("scored_holdout_hash")
    baseline_realized_hash = baseline_metrics.report_json.get(
        "eval_metadata", {}
    ).get("realized_holdout_hash")
    baseline_n_test = baseline_metrics.report_json.get("n_test")

    for i, cand in enumerate(candidate_metrics):
        cand_scored_hash = cand.report_json.get("eval_metadata", {}).get(
            "scored_holdout_hash"
        )
        cand_realized_hash = cand.report_json.get("eval_metadata", {}).get(
            "realized_holdout_hash"
        )
        cand_n_test = cand.report_json.get("n_test")

        # Check scored hash first (strongest)
        if (baseline_scored_hash is not None and cand_scored_hash is not None):
            if baseline_scored_hash != cand_scored_hash:
                reasons.append(f"candidate_{i}_scored_holdout_hash_mismatch")
        # Then realized hash
        elif (baseline_realized_hash is not None and cand_realized_hash is not None):
            if baseline_realized_hash != cand_realized_hash:
                reasons.append(f"candidate_{i}_realized_holdout_hash_mismatch")
        # Finally n_test
        elif baseline_n_test is not None and cand_n_test is not None:
            if baseline_n_test != cand_n_test:
                reasons.append(f"candidate_{i}_n_test_mismatch")

    if reasons:
        return VALIDITY_INVALID_COMPARISON, reasons

    # Check seed_fade confounding
    baseline_seed_fade = baseline_metrics.report_json.get("seed_fade", {})
    if baseline_seed_fade.get("confounded") or baseline_metrics.report_json.get(
        "seed_fade_confounded"
    ):
        reasons.append("baseline_seed_fade_confounded")

    for i, cand in enumerate(candidate_metrics):
        cand_seed_fade = cand.report_json.get("seed_fade", {})
        if cand_seed_fade.get("confounded") or cand.report_json.get(
            "seed_fade_confounded"
        ):
            reasons.append(f"candidate_{i}_seed_fade_confounded")

    if reasons:
        return VALIDITY_INVALID_COMPARISON, reasons

    # Check seed_fade indeterminacy
    baseline_seed_fade_indet = baseline_metrics.report_json.get("seed_fade", {}).get(
        "indeterminate"
    )
    if baseline_seed_fade_indet:
        reasons.append("baseline_seed_fade_indeterminate")

    for i, cand in enumerate(candidate_metrics):
        cand_seed_fade_indet = cand.report_json.get("seed_fade", {}).get(
            "indeterminate"
        )
        if cand_seed_fade_indet:
            reasons.append(f"candidate_{i}_seed_fade_indeterminate")

    if reasons:
        return VALIDITY_INVALID_COMPARISON, reasons

    # Check n_test_metrics_failed (issue #437)
    baseline_failed = baseline_metrics.report_json.get("n_test_metrics_failed", 0)
    if baseline_failed > 0:
        reasons.append(f"baseline_n_test_metrics_failed_{baseline_failed}")

    for i, cand in enumerate(candidate_metrics):
        cand_failed = cand.report_json.get("n_test_metrics_failed", 0)
        if cand_failed > 0:
            reasons.append(f"candidate_{i}_n_test_metrics_failed_{cand_failed}")

    if reasons:
        return VALIDITY_INVALID_COMPARISON, reasons

    return VALIDITY_VALID, []


# =============================================================================
# Metric extraction and pairing
# =============================================================================


def extract_run_metrics(run_dir: str) -> Optional[RunMetrics]:
    """Load metrics from a run directory."""
    report_path = _find_latest_report_json(run_dir)
    if report_path is None:
        return None

    try:
        with open(report_path, "r") as f:
            report_json = json.load(f)
    except Exception:
        return None

    metrics: Dict[str, float] = {}

    # Extract standard metrics from report root
    for key in [
        "mean_test_psnr",
        "mean_train_psnr",
        "mean_test_lpips",
        "mean_train_lpips",
        "trajectory_fps",
        "min_test_psnr",
        "max_test_psnr",
        "n_test",
        "n_train",
    ]:
        if key in report_json and isinstance(report_json[key], (int, float)):
            metrics[key] = float(report_json[key])

    # Extract geometry metrics if present
    if "geometry" in report_json and isinstance(report_json["geometry"], dict):
        geometry = report_json["geometry"]
        for key in ["needle_frac_lt_0.3", "disk_frac", "surfel_frac"]:
            if key in geometry:
                metrics[f"geometry.{key}"] = float(geometry[key])

    # Extract map comparison metrics if present
    if "geometry" in report_json and "map_comparison" in report_json["geometry"]:
        mc = report_json["geometry"]["map_comparison"]
        if "floater_volume" in mc:
            metrics["geometry.map_comparison.floater_volume"] = float(mc["floater_volume"])

    return RunMetrics(
        run_dir=run_dir,
        report_path=report_path,
        report_json=report_json,
        metrics=metrics,
    )


def pair_metrics(
    baseline: RunMetrics,
    candidates: List[RunMetrics],
) -> List[Tuple[RunMetrics, RunMetrics]]:
    """Create baseline-candidate pairs."""
    pairs = []
    for cand in candidates:
        pairs.append((baseline, cand))
    return pairs


# =============================================================================
# Verdict computation
# =============================================================================


def compute_metric_effect(
    baseline_val: float,
    candidate_val: float,
    direction: str,
) -> float:
    """Compute effect size (candidate - baseline, signed appropriately)."""
    delta = candidate_val - baseline_val
    # For lower-is-better metrics, flip the sign so positive = improvement
    if direction == "lower":
        delta = -delta
    return delta


def evaluate_primary_metrics(
    pairs: List[Tuple[RunMetrics, RunMetrics]],
    profile: DecisionProfile,
    noise_provider: Optional[Any] = None,
    bootstrap_seed: int = 42,
) -> Tuple[List[MetricResult], List[str], str]:
    """Evaluate all primary metrics.

    Returns (results, validity_reasons, earliest_invalid_validity_state).
    """
    results = []
    validity_reasons = []
    validity_state = VALIDITY_VALID

    for primary_spec in profile.primary:
        metric_name = primary_spec.metric
        direction = primary_spec.direction
        practical_margin = primary_spec.practical_margin

        # Resolve noise margin
        noise_margin = None
        if primary_spec.noise_profile_metric:
            noise_margin, reason = resolve_noise_margin(
                noise_provider, primary_spec.noise_profile_metric
            )
            if noise_margin is None:
                validity_state = VALIDITY_NO_COMPATIBLE_NOISE_PROFILE
                validity_reasons.append(
                    f"primary_{metric_name}_noise_profile_unavailable: {reason}"
                )

        # Extract deltas
        paired_deltas = []
        for baseline, candidate in pairs:
            baseline_val = baseline.metrics.get(metric_name)
            candidate_val = candidate.metrics.get(metric_name)
            if baseline_val is None or candidate_val is None:
                validity_state = VALIDITY_MISSING_REQUIRED_EVIDENCE
                validity_reasons.append(
                    f"primary_{metric_name}_missing_from_one_or_both_runs"
                )
                break
            delta = compute_metric_effect(baseline_val, candidate_val, direction)
            paired_deltas.append(
                PairedDelta(
                    metric=metric_name,
                    baseline_val=baseline_val,
                    candidate_val=candidate_val,
                    delta=delta,
                )
            )

        if not paired_deltas or validity_state != VALIDITY_VALID:
            continue

        # Compute statistics
        deltas = [d.delta for d in paired_deltas]
        mean_delta = float(np.mean(deltas))
        ci_lower, ci_upper = paired_bootstrap_ci(
            deltas,
            confidence=profile.statistics.confidence if profile.statistics else 0.95,
            seed=bootstrap_seed,
        )

        # Compute win margin
        win_margin = practical_margin
        if noise_margin is not None:
            win_margin = max(noise_margin, practical_margin)

        result = MetricResult(
            metric=metric_name,
            direction=direction,
            paired_deltas=paired_deltas,
            mean_delta=mean_delta,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            practical_margin=practical_margin,
            noise_margin=noise_margin,
            win_margin=win_margin,
        )
        results.append(result)

    return results, validity_reasons, validity_state


def _get_metric_direction(metric_name: str) -> str:
    """Infer metric direction (higher-is-better or lower-is-better).

    Returns "higher" or "lower".
    """
    # Metrics where lower is better
    lower_is_better = {"lpips", "rmse", "mae", "error", "loss", "distance"}
    for pattern in lower_is_better:
        if pattern in metric_name.lower():
            return "lower"
    # Default: higher is better (PSNR, fps, etc.)
    return "higher"


def evaluate_guardrails(
    pairs: List[Tuple[RunMetrics, RunMetrics]],
    guardrails: List,
    guardrail_type: str,
    bootstrap_seed: int = 42,
) -> Tuple[List[GuardrailResult], List[str], str]:
    """Evaluate a list of guardrails (hard or cost).

    Returns (results, validity_reasons, validity_state).
    """
    results = []
    validity_reasons = []
    validity_state = VALIDITY_VALID

    for guardrail_spec in guardrails:
        metric_name = guardrail_spec.metric
        direction = _get_metric_direction(metric_name)

        # Extract deltas
        paired_deltas = []
        for baseline, candidate in pairs:
            baseline_val = baseline.metrics.get(metric_name)
            candidate_val = candidate.metrics.get(metric_name)
            if baseline_val is None or candidate_val is None:
                validity_state = VALIDITY_MISSING_REQUIRED_EVIDENCE
                validity_reasons.append(
                    f"guardrail_{metric_name}_missing_from_one_or_both_runs"
                )
                break

            # For guardrails, compute effect the same way as for primary metrics
            delta = compute_metric_effect(baseline_val, candidate_val, direction)
            paired_deltas.append(
                PairedDelta(
                    metric=metric_name,
                    baseline_val=baseline_val,
                    candidate_val=candidate_val,
                    delta=delta,
                )
            )

        if not paired_deltas or validity_state != VALIDITY_VALID:
            continue

        deltas = [d.delta for d in paired_deltas]
        mean_delta = float(np.mean(deltas))

        # Evaluate pass/fail
        # Negative delta = regression (got worse)
        # Positive delta = improvement (got better)
        passes = True
        reason = "passed"
        if guardrail_spec.max_regression is not None:
            # Check if regression exceeds the allowed amount
            if mean_delta < 0 and abs(mean_delta) > guardrail_spec.max_regression:
                passes = False
                reason = f"regression {abs(mean_delta):.4f} exceeds max {guardrail_spec.max_regression}"
            elif mean_delta < 0:
                reason = f"regressed by {abs(mean_delta):.4f} (allowed: {guardrail_spec.max_regression})"
            else:
                reason = f"improved by {mean_delta:.4f}"

        elif guardrail_spec.max_relative_regression is not None:
            # Relative regression: check abs(delta / baseline)
            if any(abs(bd.baseline_val) < 1e-8 for bd in paired_deltas):
                validity_state = VALIDITY_MISSING_REQUIRED_EVIDENCE
                validity_reasons.append(f"guardrail_{metric_name}_baseline_near_zero")
                continue
            relative_deltas = [
                pd.delta / pd.baseline_val for pd in paired_deltas if pd.baseline_val != 0
            ]
            mean_rel_delta = float(np.mean(relative_deltas))
            if mean_rel_delta < 0 and abs(mean_rel_delta) > guardrail_spec.max_relative_regression:
                passes = False
                reason = f"relative regression {abs(mean_rel_delta):.4f} exceeds max {guardrail_spec.max_relative_regression}"
            else:
                reason = f"relative change {mean_rel_delta:.4f}"

        result = GuardrailResult(
            metric=metric_name,
            guardrail_type=guardrail_type,
            paired_deltas=paired_deltas,
            mean_delta=mean_delta,
            max_regression=guardrail_spec.max_regression,
            max_relative_regression=guardrail_spec.max_relative_regression,
            passes=passes,
            reason=reason,
        )
        results.append(result)

    return results, validity_reasons, validity_state


def check_conflicting_metrics(
    primary_results: List[MetricResult],
) -> List[Tuple[str, str, str]]:
    """Check if primary metrics move in conflicting directions.

    Returns list of (metric_a, metric_b, conflict_type) tuples.
    """
    conflicts = []
    for i, res_a in enumerate(primary_results):
        for res_b in primary_results[i + 1 :]:
            # Check if CIs don't overlap and move in conflicting directions
            if res_a.ci_lower > 0 and res_b.ci_upper < 0:
                conflicts.append((res_a.metric, res_b.metric, "res_a_up_res_b_down"))
            elif res_a.ci_upper < 0 and res_b.ci_lower > 0:
                conflicts.append((res_a.metric, res_b.metric, "res_a_down_res_b_up"))
    return conflicts


def check_partition_regression(
    baseline_metrics: RunMetrics,
    candidate_metrics: RunMetrics,
) -> List[str]:
    """Check if any partition regresses while global metric improves."""
    regressions = []

    baseline_partition_psnr = baseline_metrics.report_json.get("partition_psnr", {})
    candidate_partition_psnr = candidate_metrics.report_json.get("partition_psnr", {})

    if not baseline_partition_psnr or not candidate_partition_psnr:
        return regressions

    threshold_regression_db = 0.30

    for partition_key in baseline_partition_psnr:
        baseline_val = baseline_partition_psnr.get(partition_key)
        candidate_val = candidate_partition_psnr.get(partition_key)
        if baseline_val is not None and candidate_val is not None:
            delta = candidate_val - baseline_val
            if delta < -threshold_regression_db:
                regressions.append(f"{partition_key}_regressed_{abs(delta):.2f}dB")

    return regressions


def compute_verdict(
    profile: DecisionProfile,
    validity_state: str,
    validity_reasons: List[str],
    n_pairs: int,
    primary_results: List[MetricResult],
    hard_guardrail_results: List[GuardrailResult],
    cost_guardrail_results: List[GuardrailResult],
    conflicting_metrics: List[Tuple[str, str, str]],
    partition_regressions: List[str],
) -> Tuple[str, Optional[str], List[int], str]:
    """Compute the final verdict.

    Returns (verdict, qualitative_reason, qualitative_triggers, tradeoff_description).
    """

    # If invalid, don't score metrics
    if validity_state != VALIDITY_VALID:
        return VERDICT_QUALITATIVE_REVIEW_REQUIRED, f"Validity: {validity_state}", [], ""

    # If not enough pairs, inconclusive
    if profile.replication and n_pairs < profile.replication.minimum_pairs:
        return (
            VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS,
            f"Only {n_pairs} pairs, need >= {profile.replication.minimum_pairs}",
            [],
            "",
        )

    # Check if any primary metric has a clear win
    primary_wins = 0
    primary_losses = 0
    primary_inconclusive = 0

    for primary_result in primary_results:
        if primary_result.win_margin is None:
            # No noise profile; use practical only
            win_margin = primary_result.practical_margin
        else:
            win_margin = primary_result.win_margin

        if primary_result.ci_lower > win_margin:
            # Clear win
            primary_wins += 1
        elif primary_result.ci_upper < -win_margin:
            # Clear loss
            primary_losses += 1
        else:
            # No clear win or loss - inconclusive
            primary_inconclusive += 1

    # Check hard guardrails
    hard_guardrails_pass = True
    failing_hard_guardrails = []
    for gr in hard_guardrail_results:
        if not gr.passes:
            hard_guardrails_pass = False
            failing_hard_guardrails.append(f"{gr.metric}: {gr.reason}")

    # Check cost guardrails
    cost_guardrails_pass = True
    failing_cost_guardrails = []
    for gr in cost_guardrail_results:
        if not gr.passes:
            cost_guardrails_pass = False
            failing_cost_guardrails.append(f"{gr.metric}: {gr.reason}")

    # Decide verdict
    if not hard_guardrails_pass:
        # Hard guardrail failure = CLEAR_LOSS
        qualitative_reason = (
            f"Hard guardrails failed: {'; '.join(failing_hard_guardrails)}"
        )
        return VERDICT_CLEAR_LOSS, qualitative_reason, [], ""

    if primary_losses > 0:
        return VERDICT_CLEAR_LOSS, "Primary metric regressed", [], ""

    if primary_wins > 0 and not failing_cost_guardrails:
        return VERDICT_CLEAR_WIN, "Primary metric improved", [], ""

    if primary_wins > 0 and failing_cost_guardrails:
        # Primary up, cost guardrail down = TRADEOFF
        tradeoff_msg = f"Quality improved but cost increased: {'; '.join(failing_cost_guardrails)}"
        return VERDICT_TRADEOFF, None, [], tradeoff_msg

    if primary_inconclusive > 0:
        # Check for conflicting metrics or partitions
        triggers = []
        if conflicting_metrics:
            triggers.append(3)  # Trigger 3: metrics disagree
        if partition_regressions:
            triggers.append(4)  # Trigger 4: global up, partition down
        if triggers:
            return (
                VERDICT_QUALITATIVE_REVIEW_REQUIRED,
                f"Conflicting metric evidence (triggers {triggers})",
                triggers,
                "",
            )
        # Check replication limit
        if profile.replication and n_pairs >= profile.replication.maximum_pairs:
            return (
                VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS,
                "At maximum replicas with inconclusive evidence",
                [],
                "",
            )
        else:
            return (
                VERDICT_INCONCLUSIVE_NEEDS_MORE_RUNS,
                "Insufficient evidence; need more runs",
                [],
                "",
            )

    # No clear evidence
    return VERDICT_EQUIVALENT, "No significant difference detected", [], ""


# =============================================================================
# Main adjudicator
# =============================================================================


def adjudicate(
    *,
    profile: DecisionProfile,
    declared_profile_hash: str,
    baseline_run_dir: str,
    candidate_run_dirs: List[str],
    noise_provider: Optional[Any] = None,
    bootstrap_seed: int = 42,
) -> ExperimentResult:
    """Execute the quantitative adjudication.

    Args:
        profile: The decision profile declaring success criteria.
        declared_profile_hash: The hash of the profile as declared before the outcome.
            Must match profile_hash(profile) or adjudication is refused.
        baseline_run_dir: Path to the baseline run directory.
        candidate_run_dirs: List of candidate run directories to compare against baseline.
        noise_provider: Optional callable/dict for resolving noise profiles.
        bootstrap_seed: Seed for reproducible bootstrap CI.

    Returns:
        ExperimentResult with complete adjudication details.
    """
    result = ExperimentResult(
        profile_id=profile.profile_id,
        baseline_run_dir=baseline_run_dir,
        candidate_run_dirs=candidate_run_dirs,
        bootstrap_random_seed=bootstrap_seed,
    )

    # Verify profile hash
    computed_hash = profile_hash(profile)
    result.profile_hash = computed_hash
    result.declared_profile_hash = declared_profile_hash
    result.profile_hash_matches = computed_hash == declared_profile_hash

    if not result.profile_hash_matches:
        result.validity_state = VALIDITY_INVALID_COMPARISON
        result.validity_reasons = ["profile_hash_mismatch"]
        result.quantitative_verdict = VERDICT_QUALITATIVE_REVIEW_REQUIRED
        return result

    # Extract metrics from all runs
    baseline_metrics = extract_run_metrics(baseline_run_dir)
    if baseline_metrics is None:
        result.validity_state = VALIDITY_MISSING_REQUIRED_EVIDENCE
        result.validity_reasons = ["baseline_report_not_found"]
        result.quantitative_verdict = VERDICT_QUALITATIVE_REVIEW_REQUIRED
        return result

    candidate_metrics_list = []
    for cand_dir in candidate_run_dirs:
        cand_metrics = extract_run_metrics(cand_dir)
        if cand_metrics is None:
            result.validity_state = VALIDITY_MISSING_REQUIRED_EVIDENCE
            result.validity_reasons.append(f"candidate_report_not_found: {cand_dir}")
            result.quantitative_verdict = VERDICT_QUALITATIVE_REVIEW_REQUIRED
            return result
        candidate_metrics_list.append(cand_metrics)

    result.n_pairs = len(candidate_metrics_list)

    # Validity checks
    validity_state, validity_reasons = check_validity(
        baseline_metrics,
        candidate_metrics_list,
        profile,
    )
    result.validity_state = validity_state
    result.validity_reasons = validity_reasons

    if validity_state != VALIDITY_VALID:
        result.quantitative_verdict = VERDICT_QUALITATIVE_REVIEW_REQUIRED
        return result

    # Create pairs
    pairs = pair_metrics(baseline_metrics, candidate_metrics_list)

    # Evaluate primary metrics
    primary_results, primary_validity_reasons, primary_validity_state = (
        evaluate_primary_metrics(pairs, profile, noise_provider, bootstrap_seed)
    )
    result.primary_metric_results = primary_results

    if primary_validity_state != VALIDITY_VALID:
        result.validity_state = primary_validity_state
        result.validity_reasons.extend(primary_validity_reasons)
        if primary_validity_state == VALIDITY_NO_COMPATIBLE_NOISE_PROFILE:
            result.quantitative_verdict = VERDICT_QUALITATIVE_REVIEW_REQUIRED
            return result

    # Evaluate guardrails
    hard_gr_results, hard_gr_validity_reasons, hard_gr_validity_state = (
        evaluate_guardrails(pairs, profile.hard_guardrails, "hard", bootstrap_seed)
    )
    result.hard_guardrail_results = hard_gr_results

    cost_gr_results, cost_gr_validity_reasons, cost_gr_validity_state = (
        evaluate_guardrails(pairs, profile.cost_guardrails, "cost", bootstrap_seed)
    )
    result.cost_guardrail_results = cost_gr_results

    if hard_gr_validity_state != VALIDITY_VALID or cost_gr_validity_state != VALIDITY_VALID:
        result.validity_state = hard_gr_validity_state or cost_gr_validity_state
        result.validity_reasons.extend(hard_gr_validity_reasons + cost_gr_validity_reasons)

    # Check for conflicting metrics
    conflicts = check_conflicting_metrics(primary_results)
    result.conflicting_metrics = conflicts

    # Check for partition regressions
    partitions = []
    for cand_metrics in candidate_metrics_list:
        partitions.extend(check_partition_regression(baseline_metrics, cand_metrics))
    result.partitions_with_regression = partitions

    # Compute verdict
    verdict, qualitative_reason, triggers, tradeoff_desc = compute_verdict(
        profile,
        result.validity_state,
        result.validity_reasons,
        result.n_pairs,
        primary_results,
        hard_gr_results,
        cost_gr_results,
        conflicts,
        partitions,
    )

    result.quantitative_verdict = verdict
    result.qualitative_review_reason = qualitative_reason
    result.qualitative_review_triggers = triggers
    result.tradeoff_description = tradeoff_desc

    return result


def write_experiment_result(result: ExperimentResult, output_path: str) -> None:
    """Write experiment result to a JSON file."""
    # Serialize complex objects
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (PairedDelta, MetricResult, GuardrailResult)):
            data = asdict(obj)
            # Remove paired_deltas from nested serialization
            if "paired_deltas" in data:
                data["paired_deltas"] = [
                    {
                        "metric": pd.metric,
                        "baseline_val": pd.baseline_val,
                        "candidate_val": pd.candidate_val,
                        "delta": pd.delta,
                    }
                    for pd in data["paired_deltas"]
                ]
            return data
        elif isinstance(obj, list):
            return [_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        return obj

    data = asdict(result)
    data = _serialize(data)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
