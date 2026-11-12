"""Fixed-K ranking controls required by issue #500's canonical retest."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.analyze_frame_utility import _join_scores_to_gains
from eligsir.cli.config import ConfigError
from eligsir.cli.run_config import validate_config
from eligsir.mapping.admission.shadow_policy import ShadowUtilityPolicy
from eligsir.mapping.admission.budgeted_controller import (
    AdmissionCandidate,
    BudgetedAdmissionController,
)
from eligsir.mapping.admission.cohort import CohortAction
from eligsir.mapping.candidates.prefilter import (
    CandidatePrefilterResult,
    PrefilterDisposition,
)
from eligsir.mapping.evidence.canonical import (
    EvidenceIdentity,
    FrameEvidence,
    GaussianProbeEvidence,
    ScalarEvidence,
    SourceReliabilityEvidence,
)
from eligsir.mapping.scheduler.budgeted_keyframes import (
    BudgetedCohortKeyframeAdmitter,
)
from eligsir.mapping.scheduler.keyframes import FrameInfo, KeyframeAdmitterConfig


def _candidate(
    ordinal: int,
    *,
    translation: float,
    rotation: float,
    unexplained: float,
    residual: float,
) -> AdmissionCandidate[str]:
    evidence = FrameEvidence(
        identity=EvidenceIdentity(
            source_id="scene",
            source_ordinal=ordinal,
            timestamp_ns=ordinal * 100,
        ),
        source=SourceReliabilityEvidence(
            translation_delta_m=ScalarEvidence(translation, unit="m"),
            rotation_delta_deg=ScalarEvidence(rotation, unit="deg"),
        ),
        probe=GaussianProbeEvidence(
            unexplained_pixel_fraction=ScalarEvidence(unexplained),
            rgb_l1_mean=ScalarEvidence(residual),
        ),
    )
    return AdmissionCandidate(
        payload=f"frame-{ordinal}",
        prefilter=CandidatePrefilterResult(PrefilterDisposition.ELIGIBLE, evidence),
        probe=evidence,
    )


_CANDIDATES = (
    _candidate(10, translation=0.20, rotation=1.0, unexplained=0.10, residual=0.10),
    _candidate(11, translation=0.10, rotation=20.0, unexplained=0.20, residual=0.20),
    _candidate(12, translation=0.05, rotation=5.0, unexplained=0.90, residual=0.05),
    _candidate(13, translation=0.01, rotation=1.0, unexplained=0.30, residual=0.90),
)


def _flush(policy: str, *, commit_budget: int = 1, rank_seed: int = 17):
    controller = BudgetedAdmissionController[str](
        max_candidates=len(_CANDIDATES),
        max_source_age=100,
        commit_budget=commit_budget,
        rank_policy=policy,
        rank_seed=rank_seed,
    )
    for candidate in _CANDIDATES:
        controller.submit(candidate)
    return controller.flush()


@pytest.mark.parametrize(
    ("policy", "winner"),
    (
        ("combined", "frame-12"),
        ("motion", "frame-10"),
        ("coverage", "frame-12"),
        ("residual", "frame-13"),
        # SHA-256 order for seed 17 and the literal identities above.
        ("random", "frame-11"),
    ),
)
def test_each_rank_policy_selects_its_independent_signal(policy: str, winner: str) -> None:
    decisions = _flush(policy)

    committed = [
        item.candidate.payload.payload
        for item in decisions
        if item.action is CohortAction.COMMIT
    ]
    assert committed == [winner]


@pytest.mark.parametrize("policy", ("combined", "motion", "coverage", "residual", "random"))
@pytest.mark.parametrize("commit_budget", (0, 1, 2, 4))
def test_rank_policy_never_changes_the_hard_commit_budget(
    policy: str,
    commit_budget: int,
) -> None:
    decisions = _flush(policy, commit_budget=commit_budget)
    assert sum(item.action is CohortAction.COMMIT for item in decisions) == commit_budget


def test_default_policy_remains_byte_compatible_combined_ranking() -> None:
    default = BudgetedAdmissionController[str](
        max_candidates=4,
        max_source_age=100,
        commit_budget=1,
    )
    explicit = BudgetedAdmissionController[str](
        max_candidates=4,
        max_source_age=100,
        commit_budget=1,
        rank_policy="combined",
    )
    for candidate in _CANDIDATES:
        default.submit(candidate)
        explicit.submit(candidate)

    assert default.flush() == explicit.flush()


def test_unknown_rank_policy_fails_config_validation() -> None:
    with pytest.raises(ConfigError, match="streaming_admission_rank_policy"):
        validate_config({"streaming_admission_rank_policy": "not-a-policy"})


def test_rank_policy_is_a_validated_canonical_run_config_field() -> None:
    assert validate_config({"streaming_admission_rank_policy": "residual"}) == {
        "streaming_admission_rank_policy": "residual"
    }


class _Scene:
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            streaming_budgeted_admission=True,
            streaming_admission_cohort_max_candidates=4,
            streaming_admission_cohort_max_source_age=8,
            streaming_admission_rank_policy="residual",
            seed=23,
        )

    def promote_from_cache(self, _indices) -> int:
        return 0

    def check_occupancy(self, *_args, **_kwargs) -> bool:
        return False


def test_production_admitter_passes_rank_policy_and_seed_to_the_single_controller() -> None:
    scene = _Scene()
    admitter = BudgetedCohortKeyframeAdmitter.__new__(BudgetedCohortKeyframeAdmitter)
    admitter._occupancy_fn = scene.check_occupancy
    admitter._budgeted_controller = None

    controller = admitter._controller()

    assert controller.rank_policy == "residual"
    assert controller.rank_seed == 23


def test_budgeted_report_exposes_the_durable_selected_count() -> None:
    scene = _Scene()
    admitter = BudgetedCohortKeyframeAdmitter.__new__(BudgetedCohortKeyframeAdmitter)
    admitter._occupancy_fn = scene.check_occupancy
    admitter._budgeted_controller = None
    admitter._budgeted_snapshot = None
    admitter._budgeted_counts = {
        "candidate": 7,
        "defer": 4,
        "commit_current": 1,
        "commit_cached": 2,
        "drop": 4,
        "snapshot_flush": 0,
        "promotion_failed": 0,
    }

    report = admitter.budgeted_admission_report

    assert report["counts"]["candidate"] == 7
    assert report["counts"]["selected"] == 3
    assert report["pending"] == 0


class _LifecycleScene:
    def __init__(self, promotion: str = "success") -> None:
        self.args = SimpleNamespace(
            streaming_budgeted_admission=True,
            streaming_admission_cohort_max_candidates=2,
            streaming_admission_cohort_max_source_age=8,
            streaming_admission_rank_policy="motion",
            streaming_replay_buffer=8,
            streaming_global_reservoir_stride=0,
            seed=23,
        )
        backend = SimpleNamespace(
            map_content_version=1,
            topology_version=2,
            coordinate_frame_version=3,
        )
        self.gaussians = SimpleNamespace(_eligsir_map_backend=backend)
        self._frame_source = SimpleNamespace(source_id="issue500-scene")
        self._source_idx = 1
        self._frame_count = 100
        self._promotion = promotion
        self._cached_cams: dict[int, object] = {}
        self._active_cam_by_idx: dict[int, object] = {}
        self._replay_buffer: list[object] = []
        self._global_reservoir: list[object] = []
        self._keyframes_admitted = 0
        self._last_admitted_source_idx = -1
        self._steps_since_new_frame = 0
        self._n_train_frames_ingested = 0
        self.current_camera = None

    def check_occupancy(self, *_args, **_kwargs) -> bool:
        return False

    def promote_from_cache(self, indices: set[int]) -> int:
        idx = next(iter(indices))
        if self._promotion == "return_zero":
            return 0
        if self._promotion == "success":
            self._active_cam_by_idx[idx] = self._cached_cams[idx]
        return 1

    def update_cameras_extent(self) -> None:
        pass


def _frame(index: int, translation_m: float) -> SimpleNamespace:
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = translation_m
    return SimpleNamespace(
        index=index,
        source_ordinal=index,
        timestamp=float(index),
        color_ts_ns=index * 1_000_000_000,
        c2w=c2w,
    )


def _lifecycle_admitter(scene: _LifecycleScene) -> BudgetedCohortKeyframeAdmitter:
    admitter = BudgetedCohortKeyframeAdmitter(
        config=KeyframeAdmitterConfig(
            min_translation_m=0.05,
            min_rotation_deg=180.0,
            max_gap=1000,
            motion_pregate=False,
        ),
        occupancy_fn=scene.check_occupancy,
    )
    admitter.note_train_frame(
        FrameInfo(c2w=np.eye(4, dtype=np.float32), index=0, timestamp=0.0)
    )
    return admitter


def test_cached_winner_keeps_score_identity_and_uses_commit_iteration_for_gain(
    tmp_path: Path,
) -> None:
    scene = _LifecycleScene()
    admitter = _lifecycle_admitter(scene)
    original_cam = SimpleNamespace(name="original", original_image=None)
    promoted_cam = SimpleNamespace(name="promoted", original_image=None)
    first = _frame(1, 0.30)

    decision, _ = admitter.admit(original_cam, first)
    assert decision.admitted is False
    policy = ShadowUtilityPolicy(
        enabled=True,
        log_dir=str(tmp_path / "shadow_utility"),
        realized_gain_horizon=1,
    )
    score = policy.observe_candidate(
        frame_id=first.index,
        camera_key=id(original_cam),
        admitted=False,
        iteration=10,
        pose_confidence=1.0,
        image_quality=1.0,
        depth_valid_fraction=1.0,
        n_gaussians=100,
        topology_version=4,
        map_version=10,
        under_supported_mass=0.25,
    )
    assert score is not None
    assert admitter.record_deferred_shadow_observation(
        decision,
        score_map_version=score.meta.map_version,
        score_topology_version=score.meta.topology_version,
        predicted_utility=score.utility,
        under_supported_mass_before=0.25,
    )

    scene._cached_cams[first.index] = promoted_cam
    second_decision, _ = admitter.admit(
        SimpleNamespace(name="second", original_image=None),
        _frame(2, 0.10),
    )
    assert second_decision.admitted is False
    cached = admitter.drain_cached_admissions()
    assert len(cached) == 1
    admission = cached[0]
    policy.register_cached_admission(
        camera_key=id(admission.camera),
        frame_id=admission.frame_id,
        admitted_iteration=20,
        score_map_version=admission.score_map_version,
        score_topology_version=admission.score_topology_version,
        predicted_utility=admission.predicted_utility,
        under_supported_mass_before=admission.under_supported_mass_before,
    )
    policy.observe_training_loss(id(promoted_cam), 20, loss=1.0)
    policy.observe_training_loss(id(promoted_cam), 21, loss=0.5)

    shadow_dir = tmp_path / "shadow_utility"
    scores = [json.loads(line) for line in (shadow_dir / "scores.jsonl").read_text().splitlines()]
    gains = [
        json.loads(line)
        for line in (shadow_dir / "realized_gain.jsonl").read_text().splitlines()
    ]
    assert len(gains) == 1
    assert gains[0]["score_map_version"] == 10
    assert gains[0]["score_topology_version"] == 4
    assert gains[0]["admitted_iteration"] == 20
    paired = _join_scores_to_gains(scores, gains)
    assert len(paired) == 1
    assert paired[0]["score"]["meta"]["map_version"] == 10
    assert paired[0]["gain"]["followup_iteration"] == 21


@pytest.mark.parametrize("promotion", ("return_zero", "missing_camera"))
def test_cached_promotion_failure_has_one_drop_terminal_trace(promotion: str) -> None:
    scene = _LifecycleScene(promotion)
    admitter = _lifecycle_admitter(scene)
    first = _frame(1, 0.30)
    first_decision, _ = admitter.admit(
        SimpleNamespace(name="first", original_image=None),
        first,
    )
    scene._cached_cams[first.index] = SimpleNamespace(
        name="promoted",
        original_image=None,
    )

    admitter.admit(
        SimpleNamespace(name="second", original_image=None),
        _frame(2, 0.10),
    )

    report = admitter.budgeted_admission_report
    assert report["counts"]["selected"] == 0
    assert report["counts"]["commit_cached"] == 0
    assert report["counts"]["promotion_failed"] == 1
    assert report["counts"]["drop"] == 2
    assert report["counts"]["candidate"] == (
        report["counts"]["selected"] + report["counts"]["drop"] + report["pending"]
    )
    assert first_decision.trace["cohort"]["action"] == "drop"
    assert first_decision.trace["cohort"]["reason"] == "promotion_failed"
