"""Typed experiment manifest model.

Matches the YAML schema in section 3.3 of
``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``
exactly. Freeze-level requirements (which hashes are mandatory for a given
``frozen_level``) are in section 3.2 and are enforced by
:mod:`eligsir.experiments.validation`, not here — this module only
describes shape, not policy.

Uses plain dataclasses (no pydantic dependency) per repo house style, see
``eligsir/contracts/types.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FrozenLevel = Literal["A", "B", "C", "D"]

_VALID_FROZEN_LEVELS = ("A", "B", "C", "D")


@dataclass
class ArmSpec:
    """One experiment arm (baseline or a treatment).

    Args:
        name: Arm identifier, e.g. ``"baseline"`` or ``"treatment"``.
        changed_fields: Dotted config-path list of fields this arm changes
            relative to baseline, e.g. ``["training.lr"]``. Must be empty
            for the ``baseline`` arm itself and non-empty for every other
            arm (enforced by :mod:`eligsir.experiments.validation`).
    """

    name: str
    changed_fields: list[str] = field(default_factory=list)


@dataclass
class HardwareInfo:
    """Hardware/software target the experiment ran on.

    Args:
        gpu: GPU model string, e.g. ``"R9700"``.
        driver: GPU driver version string.
        torch: PyTorch version string.
        backend: ``"cuda"`` or ``"rocm"``.
    """

    gpu: str
    driver: str
    torch: str
    backend: Literal["cuda", "rocm"]


@dataclass
class ValidityInfo:
    """Run-validity flags surfaced for easy grep/CI parsing.

    Args:
        mapping_result_valid: Whether the mapping result should be treated
            as usable (mirrors the project's ``eval_valid``/
            ``map_reinit_invalidated`` conventions — see CLAUDE.md's Run
            Contracts section).
        source_frames_dropped: Count of source frames dropped during the run.
        map_reinitializations: Count of SLAM map re-initialization events
            (a nonzero count invalidates the run per the project's
            ORB-SLAM3 map re-init hard constraint).
    """

    mapping_result_valid: bool
    source_frames_dropped: int
    map_reinitializations: int


@dataclass
class ExperimentManifest:
    """Machine-readable experiment manifest (plan section 3.3).

    Args:
        experiment_id: Unique identifier for this experiment family.
        hypothesis: Free-text statement of what is being tested.
        independent_variable: Free-text description of the one thing being
            varied across arms.
        frozen_level: Freeze level this experiment was run at (``A``-``D``,
            see plan section 3.2). Higher letters freeze strictly more.
        repository_commit: Git commit SHA the run executed at.
        source_artifact_hash: SHA-256 hex digest of the frozen source
            artifact manifest (Level A requirement).
        pose_hash: SHA-256 hex digest of the frozen pose stream
            (Level A requirement).
        holdout_hash: SHA-256 hex digest of the holdout/guard-frame
            selection (Level A requirement).
        keyframe_hash: SHA-256 hex digest of the frozen keyframe policy
            (Level B requirement), or ``None`` below Level B.
        initial_checkpoint_hash: SHA-256 hex digest of the frozen initial
            Gaussian checkpoint (Level C requirement), or ``None`` below
            Level C.
        mutation_trace_hash: SHA-256 hex digest of the frozen mutation
            trace (Level D requirement), or ``None`` below Level D.
        arm_repository_commits: Optional per-arm Git SHAs for a code-layout
            A/B (e.g. W3-02's engine-OFF vs. engine-ON comparison, where the
            "arm" is which checkout produced the run, not a config flag).
            ``None`` for ordinary single-checkout experiments.
        config_hash: SHA-256 hex digest of the full run configuration
            (Level C requirement).
        seed_policy: Free-text description of the random-seed policy.
        hardware: Hardware/software target info.
        validity: Run-validity flags.
        arms: List of experiment arms, must include a ``"baseline"`` arm.
    """

    experiment_id: str
    hypothesis: str
    independent_variable: str
    frozen_level: FrozenLevel
    repository_commit: str
    source_artifact_hash: str
    pose_hash: str
    holdout_hash: str
    config_hash: str
    seed_policy: str
    hardware: HardwareInfo
    validity: ValidityInfo
    arms: list[ArmSpec] = field(default_factory=list)
    keyframe_hash: str | None = None
    initial_checkpoint_hash: str | None = None
    mutation_trace_hash: str | None = None
    # A regular single-checkout experiment only needs ``repository_commit``.
    # A code-layout A/B (such as W3-02) additionally records which checkout
    # produced each arm, so a report cannot silently compare the same code.
    arm_repository_commits: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain nested dict (JSON/YAML-ready)."""
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "independent_variable": self.independent_variable,
            "frozen_level": self.frozen_level,
            "repository_commit": self.repository_commit,
            "source_artifact_hash": self.source_artifact_hash,
            "pose_hash": self.pose_hash,
            "holdout_hash": self.holdout_hash,
            "keyframe_hash": self.keyframe_hash,
            "initial_checkpoint_hash": self.initial_checkpoint_hash,
            "mutation_trace_hash": self.mutation_trace_hash,
            "arm_repository_commits": (
                dict(sorted(self.arm_repository_commits.items()))
                if self.arm_repository_commits is not None
                else None
            ),
            "config_hash": self.config_hash,
            "seed_policy": self.seed_policy,
            "hardware": {
                "gpu": self.hardware.gpu,
                "driver": self.hardware.driver,
                "torch": self.hardware.torch,
                "backend": self.hardware.backend,
            },
            "validity": {
                "mapping_result_valid": self.validity.mapping_result_valid,
                "source_frames_dropped": self.validity.source_frames_dropped,
                "map_reinitializations": self.validity.map_reinitializations,
            },
            "arms": [
                {"name": arm.name, "changed_fields": list(arm.changed_fields)}
                for arm in self.arms
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentManifest":
        """Deserialize from a plain nested dict, e.g. loaded YAML/JSON."""
        hardware_data = data["hardware"]
        validity_data = data["validity"]
        return cls(
            experiment_id=data["experiment_id"],
            hypothesis=data["hypothesis"],
            independent_variable=data["independent_variable"],
            frozen_level=data["frozen_level"],
            repository_commit=data["repository_commit"],
            source_artifact_hash=data["source_artifact_hash"],
            pose_hash=data["pose_hash"],
            holdout_hash=data["holdout_hash"],
            keyframe_hash=data.get("keyframe_hash"),
            initial_checkpoint_hash=data.get("initial_checkpoint_hash"),
            mutation_trace_hash=data.get("mutation_trace_hash"),
            arm_repository_commits=(
                {str(name): str(commit) for name, commit in data["arm_repository_commits"].items()}
                if data.get("arm_repository_commits") is not None
                else None
            ),
            config_hash=data["config_hash"],
            seed_policy=data["seed_policy"],
            hardware=HardwareInfo(
                gpu=hardware_data["gpu"],
                driver=hardware_data["driver"],
                torch=hardware_data["torch"],
                backend=hardware_data["backend"],
            ),
            validity=ValidityInfo(
                mapping_result_valid=validity_data["mapping_result_valid"],
                source_frames_dropped=validity_data["source_frames_dropped"],
                map_reinitializations=validity_data["map_reinitializations"],
            ),
            arms=[
                ArmSpec(name=a["name"], changed_fields=list(a.get("changed_fields", [])))
                for a in data.get("arms", [])
            ],
        )
