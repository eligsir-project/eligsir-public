"""A-vs-A noise profiles and paired/CRN comparison machinery (issue #635).

A noise profile measures irreducible run-to-run variance for a fixed capsule/config/
runtime class. It is NOT universal: a profile is only valid for the exact regime
it was measured in -- a lookup that does not match must REFUSE and return an
explicit ``NO_COMPATIBLE_NOISE_PROFILE`` state, never a silently substituted default.

Design principles:

- Noise profiles are keyed to (capsule_id, runtime_class, metric) triples. A profile
  is only valid for the exact regime it was measured in.
- Raw replicate values, empirical A/A deltas, and full distribution summaries are
  preserved (not collapsed to a single epsilon).
- Candidate runs are paired with a matched baseline seed; deltas are analyzed per-pair,
  not as two independent means.
- When the candidate changes the RNG-consumption graph, record which streams diverged
  and analyze as paired independent realizations rather than pretending identical
  downstream randomness.
- No implicit defaults: a lookup that does not match must REFUSE, returning
  ``NO_COMPATIBLE_NOISE_PROFILE`` rather than silently reusing a "close enough"
  profile.

This is pure-Python tooling, CPU-only, torch-free, with no import-side effects.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

__all__ = [
    "NoiseProfileID",
    "NoiseProfile",
    "PairedComparisonResult",
    "NoiseProfileNotFoundError",
    "RNGDivergence",
    "parse_noise_profile",
    "compute_noise_profile_from_replicates",
    "compare_paired_runs",
]


class NoiseProfileNotFoundError(Exception):
    """Raised when a noise profile lookup fails and no fallback is available."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"NO_COMPATIBLE_NOISE_PROFILE: {reason}")


@dataclass(frozen=True)
class NoiseProfileID:
    """Immutable key for a noise profile: (capsule_id, runtime_class, metric).

    This triple uniquely identifies the regime a profile was measured in. A lookup
    with a mismatched field MUST raise NoiseProfileNotFoundError, never fall back
    to a "close enough" profile.
    """

    capsule_id: str
    """Canonical capsule identity (e.g. 'tum_fr3_mapper_core_v1')."""

    runtime_class: str
    """Backend/hardware/deterministic-schedule class (e.g. 'rocm_gfx1151' or 'offline_mapper_core')."""

    metric: str
    """Metric name (e.g. 'mapping_end_psnr_db', 'final_psnr_db', 'ssim', 'lpips')."""

    def to_dict(self) -> dict[str, str]:
        """Serialize to JSON-compatible dict."""
        return {
            "capsule_id": self.capsule_id,
            "runtime_class": self.runtime_class,
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> NoiseProfileID:
        """Deserialize from dict."""
        return cls(
            capsule_id=d["capsule_id"],
            runtime_class=d["runtime_class"],
            metric=d["metric"],
        )


@dataclass
class RNGDivergence:
    """Record of which random streams diverged between baseline and candidate.

    When a candidate changes the control flow, some RNG streams will consume
    different numbers of draws. This records which streams and why, so the
    comparison can be analyzed as paired independent realizations rather than
    claiming perfect common-random-number parallelism.
    """

    streams_diverged: list[str] = field(default_factory=list)
    """List of stream names that diverged (e.g. 'python_rng', 'torch_rng')."""

    reason: str = ""
    """Human-readable explanation of why divergence occurred."""

    def __bool__(self) -> bool:
        """True if any stream diverged."""
        return bool(self.streams_diverged)


@dataclass
class NoiseProfile:
    """Null distribution summary for one (capsule, runtime, metric) regime.

    Stores raw replicate values, empirical A/A deltas, and distribution summaries.
    A profile is only valid for the exact regime it was measured in.
    """

    profile_version: int = 1
    """Schema version for forward compatibility."""

    profile_id: NoiseProfileID = field(default_factory=lambda: NoiseProfileID("", "", ""))
    """The (capsule_id, runtime_class, metric) identity this profile is valid for."""

    n_runs: int = 0
    """Number of independent replicates that produced this profile."""

    seeds: list[int] = field(default_factory=list)
    """The exact seeds used (e.g. [11, 22, 33, 44, 55])."""

    values: list[float] = field(default_factory=list)
    """Raw replicate values (one per seed), in order."""

    mean: Optional[float] = None
    """Arithmetic mean of values."""

    median: Optional[float] = None
    """Median of values."""

    std_dev: Optional[float] = None
    """Standard deviation (SD) of values."""

    mad: Optional[float] = None
    """Median absolute deviation of values."""

    iqr: Optional[float] = None
    """Interquartile range [Q1, Q3] of values."""

    pairwise_abs_deltas: list[float] = field(default_factory=list)
    """All pairwise absolute differences between replicates.

    For N replicates, this is N*(N-1)/2 values. Used to compute the empirical
    A/A null distribution without assuming normality.
    """

    pairwise_delta_p50: Optional[float] = None
    """Median of pairwise absolute deltas."""

    pairwise_delta_p90: Optional[float] = None
    """90th percentile of pairwise absolute deltas."""

    pairwise_delta_p95: Optional[float] = None
    """95th percentile of pairwise absolute deltas."""

    pairwise_delta_max: Optional[float] = None
    """Maximum pairwise absolute delta."""

    bootstrap_ci_lower: Optional[float] = None
    """Lower bound of bootstrap confidence interval (e.g. 2.5th percentile)."""

    bootstrap_ci_upper: Optional[float] = None
    """Upper bound of bootstrap confidence interval (e.g. 97.5th percentile)."""

    outlier_seeds: list[int] = field(default_factory=list)
    """Seeds of runs flagged as outliers or invalid, to exclude from analysis."""

    notes: str = ""
    """Free-form notes on measurement conditions, caveats, etc."""

    def validate(self) -> None:
        """Check internal consistency and refuse if incomplete.

        Raises NoiseProfileNotFoundError if required fields are missing or
        inconsistent, rather than producing a broken or default profile.

        When outliers are present, n_runs is the count of valid (non-outlier)
        runs used to compute summaries, so len(values) >= n_runs.
        """
        if not self.profile_id.capsule_id:
            raise NoiseProfileNotFoundError("capsule_id is empty")
        if not self.profile_id.runtime_class:
            raise NoiseProfileNotFoundError("runtime_class is empty")
        if not self.profile_id.metric:
            raise NoiseProfileNotFoundError("metric is empty")
        if self.n_runs <= 0:
            raise NoiseProfileNotFoundError(f"n_runs={self.n_runs} (must be > 0)")
        # When there are outliers, len(values) may be > n_runs (values includes
        # outliers, but n_runs is the count of valid runs)
        if len(self.values) < self.n_runs:
            raise NoiseProfileNotFoundError(
                f"len(values)={len(self.values)} < n_runs={self.n_runs}"
            )
        if len(self.seeds) != len(self.values):
            raise NoiseProfileNotFoundError(
                f"len(seeds)={len(self.seeds)} != len(values)={len(self.values)}"
            )
        if self.mean is None or self.median is None:
            raise NoiseProfileNotFoundError(
                "mean or median is None; profile may be incomplete"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        self.validate()
        return {
            "profile_version": self.profile_version,
            "profile_id": self.profile_id.to_dict(),
            "n_runs": self.n_runs,
            "seeds": self.seeds,
            "values": self.values,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "mad": self.mad,
            "iqr": self.iqr,
            "pairwise_abs_deltas": self.pairwise_abs_deltas,
            "pairwise_delta_p50": self.pairwise_delta_p50,
            "pairwise_delta_p90": self.pairwise_delta_p90,
            "pairwise_delta_p95": self.pairwise_delta_p95,
            "pairwise_delta_max": self.pairwise_delta_max,
            "bootstrap_ci_lower": self.bootstrap_ci_lower,
            "bootstrap_ci_upper": self.bootstrap_ci_upper,
            "outlier_seeds": self.outlier_seeds,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NoiseProfile:
        """Deserialize from dict and validate."""
        profile = cls(
            profile_version=d.get("profile_version", 1),
            profile_id=NoiseProfileID.from_dict(d["profile_id"]),
            n_runs=d["n_runs"],
            seeds=list(d["seeds"]),
            values=list(d["values"]),
            mean=d.get("mean"),
            median=d.get("median"),
            std_dev=d.get("std_dev"),
            mad=d.get("mad"),
            iqr=d.get("iqr"),
            pairwise_abs_deltas=list(d.get("pairwise_abs_deltas", [])),
            pairwise_delta_p50=d.get("pairwise_delta_p50"),
            pairwise_delta_p90=d.get("pairwise_delta_p90"),
            pairwise_delta_p95=d.get("pairwise_delta_p95"),
            pairwise_delta_max=d.get("pairwise_delta_max"),
            bootstrap_ci_lower=d.get("bootstrap_ci_lower"),
            bootstrap_ci_upper=d.get("bootstrap_ci_upper"),
            outlier_seeds=list(d.get("outlier_seeds", [])),
            notes=d.get("notes", ""),
        )
        profile.validate()
        return profile


@dataclass
class PairedComparisonResult:
    """Result of comparing a candidate against a baseline using CRN pairing.

    Analyzes paired per-seed deltas rather than two independent means, and
    records any RNG divergence that occurred.
    """

    comparable: bool = False
    """True if the runs can be safely compared (same seeds, same test set, etc.)."""

    reason_if_not_comparable: str = ""
    """Human-readable reason if comparable=False."""

    candidate_arm: str = ""
    """Identifier of the candidate arm being tested."""

    baseline_arm: str = ""
    """Identifier of the matched baseline arm."""

    seed: int = 0
    """The specific seed this pair used."""

    profile_id: NoiseProfileID = field(default_factory=lambda: NoiseProfileID("", "", ""))
    """The noise profile identity used for comparison."""

    profile_n_runs: int = 0
    """Number of replicates in the matched noise profile."""

    baseline_value: Optional[float] = None
    """Baseline arm's metric value for this seed."""

    candidate_value: Optional[float] = None
    """Candidate arm's metric value for this seed."""

    delta: Optional[float] = None
    """Candidate - baseline (None if comparison failed)."""

    rng_divergence: RNGDivergence = field(default_factory=RNGDivergence)
    """Record of which RNG streams diverged between baseline and candidate."""

    notes: str = ""
    """Free-form notes on this comparison."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "comparable": self.comparable,
            "reason_if_not_comparable": self.reason_if_not_comparable,
            "candidate_arm": self.candidate_arm,
            "baseline_arm": self.baseline_arm,
            "seed": self.seed,
            "profile_id": self.profile_id.to_dict(),
            "profile_n_runs": self.profile_n_runs,
            "baseline_value": self.baseline_value,
            "candidate_value": self.candidate_value,
            "delta": self.delta,
            "rng_divergence": {
                "streams_diverged": self.rng_divergence.streams_diverged,
                "reason": self.rng_divergence.reason,
            },
            "notes": self.notes,
        }


def parse_noise_profile(json_path: str) -> NoiseProfile:
    """Load and validate a noise profile from a JSON file.

    Raises NoiseProfileNotFoundError if the file is missing, malformed,
    or fails validation.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise NoiseProfileNotFoundError(f"File not found: {json_path}") from e
    except json.JSONDecodeError as e:
        raise NoiseProfileNotFoundError(f"JSON decode error in {json_path}: {e}") from e

    try:
        return NoiseProfile.from_dict(data)
    except KeyError as e:
        raise NoiseProfileNotFoundError(f"Missing required field in profile: {e}") from e


def compute_noise_profile_from_replicates(
    profile_id: NoiseProfileID,
    values: list[float],
    seeds: list[int],
    outlier_seeds: Optional[list[int]] = None,
    notes: str = "",
) -> NoiseProfile:
    """Compute a noise profile from A-vs-A replicate runs.

    Args:
        profile_id: The (capsule, runtime, metric) identity for this profile.
        values: Raw metric values from each replicate (in seed order).
        seeds: The random seeds used (must match len(values)).
        outlier_seeds: Seeds to exclude from distribution summaries (e.g. crashed runs).
        notes: Free-form measurement notes.

    Returns:
        A complete NoiseProfile with all distributional summaries computed.

    Raises:
        NoiseProfileNotFoundError: If inputs are malformed.
    """
    if len(values) != len(seeds):
        raise NoiseProfileNotFoundError(
            f"len(values)={len(values)} != len(seeds)={len(seeds)}"
        )
    if len(values) == 0:
        raise NoiseProfileNotFoundError("Empty values list")

    outlier_seeds = outlier_seeds or []

    # Filter out outliers for summary statistics
    valid_indices = [i for i, s in enumerate(seeds) if s not in outlier_seeds]
    if not valid_indices:
        raise NoiseProfileNotFoundError("All seeds marked as outliers")

    valid_values = [values[i] for i in valid_indices]

    # Compute summary statistics
    mean = sum(valid_values) / len(valid_values)
    sorted_vals = sorted(valid_values)
    n = len(sorted_vals)
    median = (
        sorted_vals[n // 2]
        if n % 2 == 1
        else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )

    # Standard deviation
    variance = sum((v - mean) ** 2 for v in valid_values) / len(valid_values)
    std_dev = variance ** 0.5

    # Median absolute deviation
    abs_devs = sorted(abs(v - median) for v in valid_values)
    mad = (
        abs_devs[n // 2]
        if n % 2 == 1
        else (abs_devs[n // 2 - 1] + abs_devs[n // 2]) / 2
    )

    # Interquartile range
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    iqr = sorted_vals[q3_idx] - sorted_vals[q1_idx] if n > 1 else 0

    # Pairwise absolute deltas (empirical null distribution)
    pairwise_deltas = []
    for i in range(len(valid_values)):
        for j in range(i + 1, len(valid_values)):
            pairwise_deltas.append(abs(valid_values[i] - valid_values[j]))

    pairwise_deltas.sort()
    pairwise_delta_p50 = (
        pairwise_deltas[len(pairwise_deltas) // 2]
        if pairwise_deltas
        else None
    )
    pairwise_delta_p90 = (
        pairwise_deltas[int(0.9 * len(pairwise_deltas))]
        if pairwise_deltas
        else None
    )
    pairwise_delta_p95 = (
        pairwise_deltas[int(0.95 * len(pairwise_deltas))]
        if pairwise_deltas
        else None
    )
    pairwise_delta_max = max(pairwise_deltas) if pairwise_deltas else None

    # Bootstrap confidence interval (simplified: percentile method)
    bootstrap_ci_lower = sorted_vals[max(0, int(0.025 * n))]
    bootstrap_ci_upper = sorted_vals[min(n - 1, int(0.975 * n))]

    # n_runs is the count of valid (non-outlier) runs used in the summaries
    profile = NoiseProfile(
        profile_version=1,
        profile_id=profile_id,
        n_runs=len(valid_values),
        seeds=seeds,
        values=values,
        mean=mean,
        median=median,
        std_dev=std_dev,
        mad=mad,
        iqr=iqr,
        pairwise_abs_deltas=pairwise_deltas,
        pairwise_delta_p50=pairwise_delta_p50,
        pairwise_delta_p90=pairwise_delta_p90,
        pairwise_delta_p95=pairwise_delta_p95,
        pairwise_delta_max=pairwise_delta_max,
        bootstrap_ci_lower=bootstrap_ci_lower,
        bootstrap_ci_upper=bootstrap_ci_upper,
        outlier_seeds=outlier_seeds,
        notes=notes,
    )
    profile.validate()
    return profile


def compare_paired_runs(
    profile: NoiseProfile,
    baseline_value: float,
    candidate_value: float,
    seed: int,
    candidate_arm: str = "",
    baseline_arm: str = "",
    rng_divergence: Optional[RNGDivergence] = None,
    notes: str = "",
) -> PairedComparisonResult:
    """Compare a baseline-candidate pair against a noise profile.

    Checks that the specified seed was part of the profile, then computes
    the paired delta and returns a comparison result.

    Args:
        profile: The noise profile for this metric/regime.
        baseline_value: The baseline arm's value for this seed.
        candidate_value: The candidate arm's value for this seed.
        seed: The random seed this pair used.
        candidate_arm: Identifier of candidate arm.
        baseline_arm: Identifier of baseline arm.
        rng_divergence: Record of RNG streams that diverged (if any).
        notes: Free-form notes.

    Returns:
        A PairedComparisonResult with comparable=True/False and the delta.

    Raises:
        NoiseProfileNotFoundError: If the profile is invalid.
    """
    profile.validate()
    rng_divergence = rng_divergence or RNGDivergence()

    # Check that this seed is in the profile
    if seed not in profile.seeds:
        return PairedComparisonResult(
            comparable=False,
            reason_if_not_comparable=(
                f"seed {seed} not in profile seeds {profile.seeds}"
            ),
            candidate_arm=candidate_arm,
            baseline_arm=baseline_arm,
            seed=seed,
            profile_id=profile.profile_id,
            profile_n_runs=profile.n_runs,
            rng_divergence=rng_divergence,
            notes=notes,
        )

    delta = candidate_value - baseline_value

    return PairedComparisonResult(
        comparable=True,
        candidate_arm=candidate_arm,
        baseline_arm=baseline_arm,
        seed=seed,
        profile_id=profile.profile_id,
        profile_n_runs=profile.n_runs,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        rng_divergence=rng_divergence,
        notes=notes,
    )
