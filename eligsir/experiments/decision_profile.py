"""Machine-readable decision profiles for automated experiment adjudication.

This module defines the schema and utilities for declaring experiment outcomes
before running them, enabling the adjudicator to enforce predeclared success
criteria.

See issue #636 and docs/architecture/2026-08-15-issue636-quantitative-adjudication.md
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# Schema definitions
# =============================================================================


@dataclass
class MetricSpec:
    """Specification for a single metric (primary or secondary)."""
    metric: str
    direction: str  # "higher" or "lower"
    practical_margin: float
    noise_profile_metric: Optional[str] = None

    def __post_init__(self) -> None:
        if self.direction not in ("higher", "lower"):
            raise ValueError(f"direction must be 'higher' or 'lower', got {self.direction}")
        if self.practical_margin < 0:
            raise ValueError(f"practical_margin must be non-negative, got {self.practical_margin}")


@dataclass
class GuardrailSpec:
    """Specification for a hard or cost guardrail."""
    metric: str
    max_regression: Optional[float] = None
    max_relative_regression: Optional[float] = None

    def __post_init__(self) -> None:
        # Exactly one of max_regression or max_relative_regression must be set
        if (self.max_regression is None) == (self.max_relative_regression is None):
            raise ValueError(
                "GuardrailSpec must have exactly one of max_regression or "
                "max_relative_regression set"
            )
        if self.max_regression is not None and self.max_regression < 0:
            raise ValueError(f"max_regression must be non-negative, got {self.max_regression}")
        if self.max_relative_regression is not None and self.max_relative_regression < 0:
            raise ValueError(
                f"max_relative_regression must be non-negative, got {self.max_relative_regression}"
            )


@dataclass
class ReplicationSpec:
    """Specification for replication/stopping rules."""
    minimum_pairs: int
    maximum_pairs: int
    sequential_rule: Optional[str] = None

    def __post_init__(self) -> None:
        if self.minimum_pairs < 1:
            raise ValueError(f"minimum_pairs must be >= 1, got {self.minimum_pairs}")
        if self.maximum_pairs < self.minimum_pairs:
            raise ValueError(
                f"maximum_pairs ({self.maximum_pairs}) must be >= "
                f"minimum_pairs ({self.minimum_pairs})"
            )


@dataclass
class StatisticsSpec:
    """Specification for statistical methods and thresholds."""
    paired: bool = True
    interval: str = "paired_bootstrap"  # "paired_bootstrap" or others
    confidence: float = 0.95
    equivalence_margins: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        if not (0 < self.confidence < 1):
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")


@dataclass
class DecisionProfile:
    """Complete decision profile for experiment adjudication."""
    schema_version: int
    profile_id: str
    experiment_family: str
    primary: List[MetricSpec] = field(default_factory=list)
    secondary: List[MetricSpec] = field(default_factory=list)
    hard_guardrails: List[GuardrailSpec] = field(default_factory=list)
    cost_guardrails: List[GuardrailSpec] = field(default_factory=list)
    replication: Optional[ReplicationSpec] = None
    statistics: Optional[StatisticsSpec] = None
    promotion_target: Optional[str] = None
    qualitative_trigger: Optional[str] = None

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError(f"schema_version must be >= 1, got {self.schema_version}")
        if not self.profile_id or not isinstance(self.profile_id, str):
            raise ValueError(f"profile_id must be a non-empty string, got {self.profile_id}")
        if not self.experiment_family or not isinstance(self.experiment_family, str):
            raise ValueError(
                f"experiment_family must be a non-empty string, got {self.experiment_family}"
            )


# =============================================================================
# Serialization and hashing
# =============================================================================


def to_dict(profile: DecisionProfile) -> dict:
    """Convert profile to a dictionary."""
    result = {}
    for key, value in asdict(profile).items():
        if isinstance(value, list):
            result[key] = [asdict(item) if hasattr(item, '__dataclass_fields__') else item
                          for item in value]
        elif value is not None and hasattr(value, '__dataclass_fields__'):
            result[key] = asdict(value)
        else:
            result[key] = value
    return result


def from_dict(data: dict) -> DecisionProfile:
    """Reconstruct profile from a dictionary."""
    # Convert metric specs
    primary = []
    for item in data.get("primary", []):
        primary.append(MetricSpec(**item))

    secondary = []
    for item in data.get("secondary", []):
        secondary.append(MetricSpec(**item))

    # Convert guardrails
    hard_guardrails = []
    for item in data.get("hard_guardrails", []):
        hard_guardrails.append(GuardrailSpec(**item))

    cost_guardrails = []
    for item in data.get("cost_guardrails", []):
        cost_guardrails.append(GuardrailSpec(**item))

    # Convert replication spec
    replication = None
    if "replication" in data and data["replication"] is not None:
        replication = ReplicationSpec(**data["replication"])

    # Convert statistics spec
    statistics = None
    if "statistics" in data and data["statistics"] is not None:
        statistics = StatisticsSpec(**data["statistics"])

    return DecisionProfile(
        schema_version=data["schema_version"],
        profile_id=data["profile_id"],
        experiment_family=data["experiment_family"],
        primary=primary,
        secondary=secondary,
        hard_guardrails=hard_guardrails,
        cost_guardrails=cost_guardrails,
        replication=replication,
        statistics=statistics,
        promotion_target=data.get("promotion_target"),
        qualitative_trigger=data.get("qualitative_trigger"),
    )


def profile_hash(profile: DecisionProfile) -> str:
    """Compute a canonical SHA256 hash of a decision profile.

    Uses canonical JSON serialization (sorted keys, minimal separators)
    to ensure deterministic hashing across load/dump round trips.
    """
    data = to_dict(profile)
    # Use canonical JSON: sorted keys, compact separators, avoid float repr surprises
    json_str = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def load(path: str) -> DecisionProfile:
    """Load a decision profile from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    return from_dict(data)


def save(profile: DecisionProfile, path: str) -> None:
    """Save a decision profile to a JSON file."""
    data = to_dict(profile)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# =============================================================================
# Common profile templates
# =============================================================================


def reconstruction_quality() -> DecisionProfile:
    """Template: primary quality metric at mapping-end or final checkpoint."""
    return DecisionProfile(
        schema_version=1,
        profile_id="reconstruction_quality_v1",
        experiment_family="reconstruction_quality",
        primary=[
            MetricSpec(
                metric="mean_test_psnr",
                direction="higher",
                practical_margin=0.30,
                noise_profile_metric="mean_test_psnr",
            ),
        ],
        hard_guardrails=[
            GuardrailSpec(metric="mean_train_lpips", max_regression=0.05),
        ],
        replication=ReplicationSpec(minimum_pairs=1, maximum_pairs=3),
        statistics=StatisticsSpec(paired=True, confidence=0.95),
    )


def online_continual_quality() -> DecisionProfile:
    """Template: quality during mapping phase + final quality."""
    return DecisionProfile(
        schema_version=1,
        profile_id="online_continual_quality_v1",
        experiment_family="online_continual_quality",
        primary=[
            MetricSpec(
                metric="mean_test_psnr",
                direction="higher",
                practical_margin=0.25,
                noise_profile_metric="mean_test_psnr",
            ),
        ],
        secondary=[
            MetricSpec(
                metric="mean_train_psnr",
                direction="higher",
                practical_margin=0.20,
            ),
        ],
        hard_guardrails=[
            GuardrailSpec(metric="mean_test_psnr", max_regression=0.50),
        ],
        replication=ReplicationSpec(minimum_pairs=1, maximum_pairs=3),
        statistics=StatisticsSpec(paired=True, confidence=0.95),
    )


def performance_quality_equivalent() -> DecisionProfile:
    """Template: throughput/latency improvement with quality equivalence."""
    return DecisionProfile(
        schema_version=1,
        profile_id="performance_quality_equivalent_v1",
        experiment_family="performance_quality_equivalent",
        primary=[
            MetricSpec(
                metric="trajectory_fps",
                direction="higher",
                practical_margin=2.0,
            ),
        ],
        hard_guardrails=[
            GuardrailSpec(
                metric="mean_test_psnr",
                max_regression=0.20,
            ),
        ],
        replication=ReplicationSpec(minimum_pairs=1, maximum_pairs=3),
        statistics=StatisticsSpec(paired=True, confidence=0.95),
    )
