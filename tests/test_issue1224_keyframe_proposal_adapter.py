from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from _fake_alpha import stub_render
from eligsir.contracts.control import (
    ControlLimits,
    ControlSnapshot,
    DispositionKind,
    EvidenceFact,
    FreshnessMode,
)
from eligsir.contracts.runtime import SourceIdentity, TrackingObservation, TrackingStatus
from eligsir.contracts.types import Pose
from eligsir.mapping.scheduler.backpressure import (
    BackpressureAction,
    BackpressureConfig,
    BackpressureDecision,
    BackpressureOwner,
)
from eligsir.mapping.scheduler.keyframes import (
    FrameInfo,
    KeyframeAdmitter,
    KeyframeAdmitterConfig,
    keyframe_admitter_config_from_args,
)
from eligsir.mapping.admission.keyframe_provider import (
    keyframe_admitter_proposal_provider,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures/issue1224/keyframe_precedence.json"


def keyframe_admitter_proposal_provider_from_args(args: SimpleNamespace):
    """Test-only bridge for parser-shaped historical fixture inputs."""

    return keyframe_admitter_proposal_provider(
        keyframe_admitter_config_from_args(args)
    )


def _frame(spec) -> FrameInfo:
    c2w = np.eye(4, dtype=np.float32)
    c2w[0, 3] = float(spec["x"])
    return FrameInfo(c2w, int(spec["index"]), float(spec["timestamp"]))


def _snapshot(frame: FrameInfo, *, gaussian_count: int = 1) -> ControlSnapshot:
    source = SourceIdentity("camera", 0, frame.index, round(frame.timestamp * 1e9))
    return ControlSnapshot(
        snapshot_id=frame.index,
        control_version=1,
        source=source,
        tracking=None,
        pending_sources=(source,),
        newest_frontier=source,
        freshness_mode=FreshnessMode.HEALTHY,
        pending_oldest_age_ns=0,
        ingress_queue_depth=0,
        candidate_queue_depth=0,
        transfer_queue_depth=0,
        gpu_queue_depth=0,
        map_content_version=0,
        topology_version=0,
        parameter_version=0,
        gaussian_count=gaussian_count,
        optimizer_steps_completed=frame.index,
        limits=ControlLimits(1, 0, 1, 0, 0, 0),
    )


def _tracked_snapshot(frame: FrameInfo, **facts) -> ControlSnapshot:
    snapshot = _snapshot(frame)
    tracking = TrackingObservation(
        source=snapshot.source,
        pose=Pose(frame.c2w),
        status=TrackingStatus.VALID,
        provider="fixture",
    )
    return ControlSnapshot(
        **{field: getattr(snapshot, field) for field in snapshot.__dataclass_fields__}
        | {
            "tracking": tracking,
            "evidence": tuple(
                EvidenceFact(name, value) for name, value in facts.items()
            ),
        }
    )


class _ProposalCapture:
    def __init__(self, provider):
        self.provider = provider
        self.latest = None

    def propose(self, snapshot):
        self.latest = self.provider.propose(snapshot)
        return self.latest

    def commit(self, snapshot, disposition):
        self.provider.commit(snapshot, disposition)


def _admitting_owner(provider):
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    capture = _ProposalCapture(provider)
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        ),
        semantic_providers=(capture,),
    )
    return owner, capture


@pytest.mark.parametrize("case", json.loads(FIXTURE_PATH.read_text()), ids=lambda case: case["name"])
def test_keyframe_proposal_adapter_preserves_legacy_precedence_fixture(case):
    from eligsir.mapping.admission.keyframe_provider import (
        KeyframeAdmitterProposalProvider,
    )

    config_values = dict(
        min_translation_m=0.05,
        min_rotation_deg=5.0,
        min_overlap=0.25,
        max_overlap=0.90,
        max_gap=10,
        motion_pregate=True,
        coverage_gain_gate=False,
        max_gap_require_novelty=False,
    )
    config_values.update(case.get("config", {}))
    config = KeyframeAdmitterConfig(**config_values)
    admitter = KeyframeAdmitter(config, render_fn=stub_render(case["overlap"]))
    if case["baseline"] is not None:
        admitter.note_train_frame(_frame(case["baseline"]))
    frames = {frame.index: frame for frame in map(_frame, case["candidates"])}
    provider = KeyframeAdmitterProposalProvider(
        admitter,
        frame_from_snapshot=lambda snapshot: frames[snapshot.source.source_ordinal],
        camera_from_snapshot=lambda _snapshot: object(),
    )
    owner, capture = _admitting_owner(provider)

    proposal = None
    for frame in frames.values():
        owner.decide(_snapshot(frame))
        proposal = capture.latest

    expected = case["expected"]
    assert proposal.admitted is expected["admitted"]
    assert proposal.reason == expected["reason"]
    assert dict(proposal.trace)["candidate_trigger"] == expected["trigger"]
    if "target_interval_s" in expected:
        assert dict(proposal.trace)["controller_target_interval_s"] == pytest.approx(
            expected["target_interval_s"]
        )
    assert not hasattr(proposal, "terminal")
    assert not hasattr(proposal, "materialization_permitted")
    with pytest.raises(FrozenInstanceError):
        proposal.reason = "changed"


def test_observation_owner_applies_keyframe_reason_before_capacity_and_alone_mints_permit():
    from eligsir.mapping.admission.keyframe_provider import KeyframeAdmissionProposal
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    frame = _frame({"index": 3, "x": 0.0, "timestamp": 0.1})
    snapshot = _snapshot(frame)

    class RejectingKeyframeProvider:
        def propose(self, _snapshot):
            return KeyframeAdmissionProposal(
                provider="KeyframeAdmitter",
                admitted=False,
                reason="no_motion",
                trace=(("candidate_trigger", "normal"),),
            )

    owner = ObservationOwner.from_backpressure(
        BackpressureOwner(BackpressureConfig(enabled=True, force_all_stress=True)),
        semantic_providers=(RejectingKeyframeProvider(),),
    )

    disposition = owner.decide(snapshot)

    assert disposition.reason == "no_motion"
    assert not disposition.materialization_permitted
    assert owner.permit(disposition) is None


def test_snapshot_adapter_runs_coverage_novelty_as_an_observation_owner_proposal():
    provider = keyframe_admitter_proposal_provider_from_args(
        SimpleNamespace(
            streaming_keyframe_max_gap=100,
            streaming_keyframe_hard_backstop_frames=400.0,
            streaming_keyframe_coverage_gain_gate=True,
            streaming_keyframe_max_gap_require_novelty=False,
            streaming_keyframe_novelty_force_admit=True,
            streaming_keyframe_novelty_force_require_fresh_pose=False,
            streaming_keyframe_min_new_voxel_frac=0.05,
            streaming_keyframe_novelty_force_frac=0.6,
            streaming_keyframe_velocity_adaptive=False,
        )
    )
    owner, capture = _admitting_owner(provider)
    owner.decide(
        _tracked_snapshot(_frame({"index": 0, "x": 0.0, "timestamp": 0.0}), overlap=0.5, coverage_gain=0.8)
    )

    owner.decide(
        _tracked_snapshot(_frame({"index": 1, "x": 0.0, "timestamp": 0.03}), overlap=0.95, coverage_gain=0.8)
    )
    proposal = capture.latest

    assert proposal.admitted
    assert proposal.reason == "novelty_admit"
    assert dict(proposal.trace)["candidate_trigger"] == "novelty_force"


@pytest.mark.parametrize(
    ("facts", "reason"),
    [
        ({"rgb_blur_laplacian": 0.0, "depth_valid_ratio": 1.0}, "blur_reject"),
        ({"rgb_blur_laplacian": 20.0, "depth_valid_ratio": 0.0}, "quality_reject"),
    ],
)
def test_snapshot_adapter_preserves_rgb_depth_quality_reason_codes(facts, reason):
    provider = keyframe_admitter_proposal_provider_from_args(
        SimpleNamespace(
            streaming_keyframe_max_gap=100,
            streaming_keyframe_hard_backstop_frames=400.0,
            streaming_keyframe_coverage_gain_gate=False,
            streaming_keyframe_max_gap_require_novelty=False,
            streaming_keyframe_quality_enabled=True,
            streaming_keyframe_velocity_adaptive=False,
        )
    )
    owner, capture = _admitting_owner(provider)
    common = {"overlap": 0.5, "coverage_gain": 0.0, "rgb_quality_score": 1.0}
    owner.decide(
        _tracked_snapshot(_frame({"index": 0, "x": 0.0, "timestamp": 0.0}), **common, **facts)
    )

    owner.decide(
        _tracked_snapshot(_frame({"index": 1, "x": 0.1, "timestamp": 0.03}), **common, **facts)
    )
    proposal = capture.latest

    assert not proposal.admitted
    assert proposal.reason == reason


def test_capacity_is_resolved_after_an_admitted_keyframe_proposal():
    from eligsir.mapping.admission.keyframe_provider import KeyframeAdmissionProposal
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    frame = _frame({"index": 3, "x": 0.1, "timestamp": 0.1})
    snapshot = _snapshot(frame)

    class AdmittingKeyframeProvider:
        def propose(self, _snapshot):
            return KeyframeAdmissionProposal("KeyframeAdmitter", True, "pose_delta")

    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.SKIP_CAPACITY,
            reason="skipped_capacity_exhausted",
        ),
        semantic_providers=(AdmittingKeyframeProvider(),),
    )

    disposition = owner.decide(snapshot)

    assert disposition.reason == "skipped_capacity_exhausted"
    assert not disposition.materialization_permitted
    assert owner.permit(disposition) is None


def test_capacity_veto_does_not_commit_keyframe_baseline_or_counters():
    from eligsir.mapping.admission.keyframe_provider import (
        KeyframeAdmitterProposalProvider,
    )
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    frames = {
        index: _frame({"index": index, "x": 0.0, "timestamp": index * 0.03})
        for index in (0, 1)
    }
    admitter = KeyframeAdmitter(
        KeyframeAdmitterConfig(max_gap=100), render_fn=stub_render(0.5)
    )
    provider = KeyframeAdmitterProposalProvider(
        admitter,
        frame_from_snapshot=lambda snapshot: frames[snapshot.source.source_ordinal],
        camera_from_snapshot=lambda _snapshot: object(),
    )
    calls = 0

    def capacity_then_admit(snapshot):
        nonlocal calls
        calls += 1
        admitted = calls > 1
        return BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=(
                BackpressureAction.ADMIT
                if admitted
                else BackpressureAction.SKIP_CAPACITY
            ),
            reason=(
                "soft_all_capacity_available"
                if admitted
                else "skipped_capacity_exhausted"
            ),
            admit_supervision=admitted,
            allow_geometry_insertion=admitted,
        )

    owner = ObservationOwner(capacity_then_admit, semantic_providers=(provider,))

    first = owner.decide(_snapshot(frames[0]))
    second = owner.decide(_snapshot(frames[1]))

    assert first.kind is not DispositionKind.ADMIT
    assert second.kind is DispositionKind.ADMIT
    assert second.reason == "new_content"
    assert admitter.keyframe_indices == [1]
    assert admitter.keyframes_admitted == 1


def test_capacity_veto_retains_candidate_clocks_without_advancing_admitted_baseline():
    from eligsir.mapping.admission.keyframe_provider import (
        KeyframeAdmitterProposalProvider,
    )
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    frames = {
        0: _frame({"index": 0, "x": 0.0, "timestamp": 0.0}),
        1: _frame({"index": 1, "x": 0.10, "timestamp": 0.1}),
        2: _frame({"index": 2, "x": 0.11, "timestamp": 0.2}),
    }
    admitter = KeyframeAdmitter(
        KeyframeAdmitterConfig(
            max_gap=100,
            velocity_adaptive=True,
            input_fps=10.0,
            target_admit_rate_fps=100.0,
            velocity_ema_alpha=1.0,
            exposure_gate_enabled=True,
            exposure_max_delta=1.0,
            blur_gate_enabled=True,
            blur_gate_min_laplacian=0.0,
        ),
        render_fn=stub_render(0.5),
        load_rgb=lambda _frame: np.ones((8, 8, 3), dtype=np.float32) * 0.5,
    )
    provider = KeyframeAdmitterProposalProvider(
        admitter,
        frame_from_snapshot=lambda snapshot: frames[snapshot.source.source_ordinal],
        camera_from_snapshot=lambda _snapshot: object(),
    )

    def capacity(snapshot):
        admitted = snapshot.source.source_ordinal != 1
        return BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=(
                BackpressureAction.ADMIT
                if admitted
                else BackpressureAction.SKIP_CAPACITY
            ),
            reason=(
                "soft_all_capacity_available"
                if admitted
                else "skipped_capacity_exhausted"
            ),
            admit_supervision=admitted,
            allow_geometry_insertion=admitted,
        )

    capture = _ProposalCapture(provider)
    owner = ObservationOwner(capacity, semantic_providers=(capture,))
    owner.decide(_snapshot(frames[0]))

    vetoed = owner.decide(_snapshot(frames[1]))

    assert vetoed.kind is DispositionKind.SKIP_CAPACITY
    assert admitter.keyframe_indices == [0]
    assert admitter.keyframes_admitted == 1
    assert admitter._last_train_index == 0
    assert admitter._prev_candidate_index == 1
    assert admitter._exposure_iter == 2
    assert admitter._blur_iter == 2
    assert admitter._last_admitted_iter == 1
    assert admitter._last_admitted_iter_blur == 1

    owner.decide(_snapshot(frames[2]))

    assert capture.latest.trace
    assert dict(capture.latest.trace)["translation_velocity_m_s"] == pytest.approx(0.1)
    assert admitter.keyframe_indices == [0, 2]


def _production_keyframe_facts(
    runtime, frame, rgb, depth, *, rgb_encoding="rgb8", **diagnostics
):
    from eligsir.contracts.runtime import BufferRef
    from eligsir.contracts.types import CameraIntrinsics

    snapshot = _snapshot(frame)
    observation = TrackingObservation(
        source=snapshot.source,
        pose=Pose(frame.c2w),
        status=TrackingStatus.VALID,
        provider="production-fixture",
        confidence=1.0,
        diagnostics=diagnostics,
    )
    intrinsics = CameraIntrinsics(
        fx=40.0,
        fy=40.0,
        cx=15.5,
        cy=15.5,
        width=32,
        height=32,
        depth_scale=1.0,
    )
    packet = SimpleNamespace(
        rgb=BufferRef.from_numpy(rgb, encoding=rgb_encoding),
        depth=(
            None
            if depth is None
            else BufferRef.from_numpy(depth, encoding="32FC1", units="m")
        ),
        intrinsics=intrinsics,
    )
    return observation, {
        fact.name: fact.value
        for fact in runtime._control_evidence(packet, observation)
    }


def _production_evidence_runtime(**args):
    from eligsir.runtime.production_application import (
        CanonicalMappingRuntime as ContinualMappingRuntime,
    )

    runtime = object.__new__(ContinualMappingRuntime)
    from eligsir.runtime.config import (
        CanonicalRuntimePolicy,
        DepthInsertionPolicy,
        MappingRuntimeConfig,
    )

    runtime._runtime_policy = CanonicalRuntimePolicy(
        blur_downsample=int(args.get("streaming_keyframe_blur_downsample", 4))
    )
    runtime.config = MappingRuntimeConfig(
        model_layout="gsplat",
        renderer="gsplat",
        numerical_mode="selective_adam",
        keyframe_policy={
            "parallax_gate": bool(args.get("streaming_kf_parallax_gate", False)),
            "quality_enabled": bool(
                args.get("streaming_keyframe_quality_enabled", False)
            ),
            "exposure_gate_enabled": bool(
                args.get("streaming_keyframe_exposure_gate", False)
            ),
            "blur_gate_enabled": bool(
                args.get("streaming_keyframe_blur_gate", False)
            ),
            "coverage_gain_depth_stride": int(
                args.get("streaming_keyframe_coverage_gain_depth_stride", 8)
            ),
        },
    )
    runtime._depth_insertion_policy = DepthInsertionPolicy(
        min_depth_m=float(args.get("streaming_min_depth", 0.1)),
        max_depth_m=float(args.get("streaming_max_depth", 8.0)),
    )
    runtime._last_admitted_pose = None
    runtime._motion_admission_thresholds = (0.05, 5.0)
    runtime._source_clock_ns = 0
    runtime._map_clock_ns = 0
    runtime._latest_gpu_transaction = None
    runtime._replay_modalities = {}
    return runtime


def test_disabled_rgb_quality_gates_publish_no_expensive_image_evidence():
    runtime = _production_evidence_runtime()
    frame = _frame({"index": 0, "timestamp": 0.0, "x": 0.0})

    _observation, evidence = _production_keyframe_facts(
        runtime,
        frame,
        np.full((32, 32, 3), 128, dtype=np.uint8),
        np.ones((32, 32), dtype=np.float32),
    )

    assert evidence["rgb_blur_laplacian"] is None
    assert evidence["rgb_standalone_blur_laplacian"] is None
    assert evidence["rgb_luma"] is None
    assert evidence["rgb_clip_ratio"] is None
    assert evidence["rgb_supervision_eligible"] is True
    assert evidence["depth_supervision_eligible"] is True


def test_production_control_evidence_publishes_candidate_index_shortlist_limit():
    from eligsir.mapping.scheduler.historical_candidate_index import (
        HistoricalCandidateConfig,
        HistoricalCandidateIndex,
    )

    runtime = _production_evidence_runtime()
    runtime._historical_candidate_index = HistoricalCandidateIndex(
        config=HistoricalCandidateConfig(
            shortlist_size=64,
            recent_lane_size=16,
            focus_lane_size=16,
            protected_lane_size=8,
            global_reservoir_size=32,
            global_floor=8,
        )
    )
    frame = _frame({"index": 0, "timestamp": 0.0, "x": 0.0})
    observation, evidence = _production_keyframe_facts(
        runtime,
        frame,
        np.full((32, 32, 3), 128, dtype=np.uint8),
        np.ones((32, 32), dtype=np.float32),
    )

    assert observation.source == _snapshot(frame).source
    assert evidence["replay_shortlist_configured_limit"] == 64


def test_real_rgbd_bootstrap_without_overlap_is_admitted_and_establishes_baseline():
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    runtime = _production_evidence_runtime()
    provider = keyframe_admitter_proposal_provider_from_args(
        SimpleNamespace(
            streaming_keyframe_max_gap=100,
            streaming_keyframe_max_gap_require_novelty=False,
            streaming_keyframe_coverage_gain_gate=False,
            streaming_keyframe_velocity_adaptive=False,
        )
    )

    class CaptureBeforeTerminalCommit:
        def __init__(self):
            self.proposal = None
            self.baseline_after_proposal = None

        def propose(self, snapshot):
            self.proposal = provider.propose(snapshot)
            self.baseline_after_proposal = provider._admitter._last_train_c2w
            return self.proposal

        def commit(self, snapshot, disposition):
            provider.commit(snapshot, disposition)

    capture = CaptureBeforeTerminalCommit()
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        ),
        semantic_providers=(capture,),
    )
    frame = _frame({"index": 0, "x": 0.0, "timestamp": 0.0})
    rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float32)
    observation, facts = _production_keyframe_facts(runtime, frame, rgb, depth)
    snapshot = replace(
        _tracked_snapshot(frame),
        gaussian_count=0,
        tracking=observation,
        evidence=tuple(EvidenceFact(name, value) for name, value in facts.items()),
    )

    disposition = owner.decide(snapshot)

    assert facts["overlap"] is None
    assert capture.proposal.admitted
    assert capture.proposal.reason == "bootstrap"
    proposal_trace = dict(capture.proposal.trace)
    assert proposal_trace["candidate_trigger"] == "bootstrap"
    assert proposal_trace["render_overlap"] is None
    assert "final_owner" not in proposal_trace
    assert capture.baseline_after_proposal is None
    assert disposition.kind is DispositionKind.ADMIT
    assert disposition.modality_eligibility.rgb_supervision
    assert disposition.modality_eligibility.rgb_quality == "qualified"
    assert disposition.modality_eligibility.depth_supervision
    assert disposition.modality_eligibility.depth_quality == "qualified"
    assert disposition.modality_eligibility.depth_insertion
    assert owner.permit(disposition) is not None
    assert provider._admitter.keyframe_indices == [0]
    assert provider._admitter.keyframes_admitted == 1
    np.testing.assert_array_equal(provider._admitter._last_train_c2w, frame.c2w)


def test_map_free_bootstrap_capacity_veto_does_not_commit_baseline_or_history():
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    provider = keyframe_admitter_proposal_provider_from_args(
        SimpleNamespace(
            streaming_keyframe_max_gap=100,
            streaming_keyframe_max_gap_require_novelty=False,
            streaming_keyframe_coverage_gain_gate=False,
            streaming_keyframe_velocity_adaptive=False,
        )
    )
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.SKIP_CAPACITY,
            reason="skipped_capacity_exhausted",
        ),
        semantic_providers=(provider,),
    )
    frame = _frame({"index": 0, "x": 0.0, "timestamp": 0.0})
    snapshot = replace(
        _tracked_snapshot(
            frame,
            rgb_supervision_eligible=True,
            rgb_quality="qualified",
            depth_supervision_eligible=True,
            depth_quality="qualified",
            depth_insertion_eligible=True,
        ),
        gaussian_count=0,
    )

    disposition = owner.decide(snapshot)

    assert disposition.kind is DispositionKind.SKIP_CAPACITY
    assert owner.permit(disposition) is None
    assert provider._admitter._last_train_c2w is None
    assert provider._admitter.keyframe_indices == []
    assert provider._admitter.keyframes_admitted == 0


def test_map_free_bootstrap_modality_veto_does_not_commit_baseline_or_history():
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    provider = keyframe_admitter_proposal_provider_from_args(
        SimpleNamespace(
            streaming_keyframe_max_gap=100,
            streaming_keyframe_max_gap_require_novelty=False,
            streaming_keyframe_coverage_gain_gate=False,
            streaming_keyframe_velocity_adaptive=False,
        )
    )
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        ),
        semantic_providers=(provider,),
    )
    frame = _frame({"index": 0, "x": 0.0, "timestamp": 0.0})
    snapshot = replace(
        _tracked_snapshot(
            frame,
            rgb_supervision_eligible=False,
            rgb_quality="invalid",
            depth_supervision_eligible=False,
            depth_quality="missing",
            depth_insertion_eligible=False,
        ),
        gaussian_count=0,
    )

    disposition = owner.decide(snapshot)

    assert disposition.kind is DispositionKind.SKIP_QUALITY
    assert disposition.reason == "modality_quality_rejected"
    assert owner.permit(disposition) is None
    assert provider._admitter._last_train_c2w is None
    assert provider._admitter.keyframe_indices == []
    assert provider._admitter.keyframes_admitted == 0


def test_production_packet_rgb_drives_standalone_blur_precedence_without_diagnostics():
    args = dict(
        streaming_keyframe_blur_gate=True,
        streaming_keyframe_blur_min_laplacian=20.0,
        streaming_keyframe_blur_downsample=1,
        streaming_keyframe_max_gap=100,
        streaming_keyframe_max_gap_require_novelty=False,
        streaming_keyframe_coverage_gain_gate=False,
        streaming_keyframe_velocity_adaptive=False,
    )
    runtime = _production_evidence_runtime(**args)
    provider = keyframe_admitter_proposal_provider_from_args(SimpleNamespace(**args))
    owner, _capture = _admitting_owner(provider)
    sharp = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
    sharp = np.repeat(sharp[..., None], 3, axis=-1)
    flat = np.full((32, 32, 3), 127, dtype=np.uint8)
    depth = np.ones((32, 32), dtype=np.float32)
    baseline = _frame({"index": 0, "x": 0.0, "timestamp": 0.0})
    observation, facts = _production_keyframe_facts(
        runtime, baseline, sharp, depth, overlap=0.5
    )
    owner.decide(
        replace(_tracked_snapshot(baseline), tracking=observation, evidence=tuple(
            EvidenceFact(name, value) for name, value in facts.items()
        ))
    )
    runtime._last_admitted_pose = baseline.c2w.copy()
    candidate = _frame({"index": 1, "x": 0.1, "timestamp": 0.03})
    observation, facts = _production_keyframe_facts(
        runtime, candidate, flat, depth, overlap=0.5
    )

    disposition = owner.decide(
        replace(_tracked_snapshot(candidate), tracking=observation, evidence=tuple(
            EvidenceFact(name, value) for name, value in facts.items()
        ))
    )

    assert facts["rgb_blur_laplacian"] is None
    assert facts["rgb_standalone_blur_laplacian"] == 0.0
    assert disposition.reason == "blur_reject"


def test_canonical_rgb32f_chw_packet_is_layout_normalized_before_blur_gate():
    args = dict(
        streaming_keyframe_blur_gate=True,
        streaming_keyframe_blur_min_laplacian=100_000.0,
        streaming_keyframe_blur_downsample=1,
        streaming_keyframe_max_gap=100,
        streaming_keyframe_max_gap_require_novelty=False,
        streaming_keyframe_coverage_gain_gate=False,
        streaming_keyframe_velocity_adaptive=False,
    )
    runtime = _production_evidence_runtime(**args)
    provider = keyframe_admitter_proposal_provider_from_args(SimpleNamespace(**args))
    owner, _capture = _admitting_owner(provider)
    checker = (np.indices((32, 32)).sum(axis=0) % 2).astype(np.float32)
    rgb_chw = np.repeat(checker[None, ...], 3, axis=0)
    depth = np.ones((32, 32), dtype=np.float32)
    baseline = _frame({"index": 0, "x": 0.0, "timestamp": 0.0})
    observation, facts = _production_keyframe_facts(
        runtime,
        baseline,
        rgb_chw,
        depth,
        rgb_encoding="rgb32f_chw",
        overlap=0.5,
    )
    owner.decide(
        replace(_tracked_snapshot(baseline), tracking=observation, evidence=tuple(
            EvidenceFact(name, value) for name, value in facts.items()
        ))
    )
    runtime._last_admitted_pose = baseline.c2w.copy()
    candidate = _frame({"index": 1, "x": 0.1, "timestamp": 0.03})
    observation, facts = _production_keyframe_facts(
        runtime,
        candidate,
        rgb_chw,
        depth,
        rgb_encoding="rgb32f_chw",
        overlap=0.5,
    )

    disposition = owner.decide(
        replace(_tracked_snapshot(candidate), tracking=observation, evidence=tuple(
            EvidenceFact(name, value) for name, value in facts.items()
        ))
    )

    assert facts["rgb_blur_laplacian"] is None
    assert facts["rgb_standalone_blur_laplacian"] > 100_000.0
    assert disposition.reason == "pose_delta"


def test_production_depth_pose_and_intrinsics_drive_parallax_precedence_without_diagnostics():
    args = dict(
        streaming_kf_parallax_gate=True,
        streaming_kf_parallax_deg=3.0,
        streaming_kf_covis_thresh=0.6,
        streaming_keyframe_coverage_gain_gate=True,
        streaming_keyframe_min_new_voxel_frac=0.05,
        streaming_keyframe_min_translation=0.05,
        streaming_max_depth=20.0,
        streaming_keyframe_max_gap=100,
        streaming_keyframe_max_gap_require_novelty=False,
        streaming_keyframe_velocity_adaptive=False,
    )
    runtime = _production_evidence_runtime(**args)
    provider = keyframe_admitter_proposal_provider_from_args(SimpleNamespace(**args))
    owner, _capture = _admitting_owner(provider)
    rgb = np.full((32, 32, 3), 127, dtype=np.uint8)
    depth = np.full((32, 32), 10.0, dtype=np.float32)
    baseline = _frame({"index": 0, "x": 0.0, "timestamp": 0.0})
    observation, facts = _production_keyframe_facts(
        runtime, baseline, rgb, depth, overlap=0.5, coverage_gain=0.1
    )
    owner.decide(
        replace(_tracked_snapshot(baseline), tracking=observation, evidence=tuple(
            EvidenceFact(name, value) for name, value in facts.items()
        ))
    )
    runtime._last_admitted_pose = baseline.c2w.copy()
    candidate = _frame({"index": 1, "x": 0.1, "timestamp": 0.03})
    observation, facts = _production_keyframe_facts(
        runtime, candidate, rgb, depth, overlap=0.5, coverage_gain=0.1
    )

    disposition = owner.decide(
        replace(_tracked_snapshot(candidate), tracking=observation, evidence=tuple(
            EvidenceFact(name, value) for name, value in facts.items()
        ))
    )

    assert 0.0 < facts["parallax_deg"] < 1.0
    assert disposition.reason == "rej_parallax"


def test_production_parallax_configuration_fails_when_coverage_is_disabled():
    with pytest.raises(ValueError, match="parallax.*coverage"):
        keyframe_admitter_proposal_provider_from_args(
            SimpleNamespace(
                streaming_kf_parallax_gate=True,
                streaming_keyframe_coverage_gain_gate=False,
            )
        )


def test_backpressure_admission_is_not_committed_after_modality_downgrade():
    from eligsir.mapping.admission.keyframe_provider import KeyframeAdmissionProposal
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    class AdmittingProvider:
        def propose(self, _snapshot):
            return KeyframeAdmissionProposal("keyframe", True, "new_content")

    backpressure = BackpressureOwner(
        BackpressureConfig(enabled=True, force_all_stress=True)
    )
    owner = ObservationOwner.from_backpressure(
        backpressure, semantic_providers=(AdmittingProvider(),)
    )
    first_frame = _frame({"index": 0, "x": 0.0, "timestamp": 0.0})
    first = _tracked_snapshot(
        first_frame,
        rgb_supervision_eligible=False,
        depth_supervision_eligible=False,
        depth_insertion_eligible=False,
    )

    downgraded = owner.decide(first)

    assert downgraded.kind is DispositionKind.SKIP_QUALITY
    assert backpressure.decision_for(0) is None

    second_frame = _frame({"index": 1, "x": 0.1, "timestamp": 0.03})
    second = _tracked_snapshot(
        second_frame,
        rgb_supervision_eligible=True,
        rgb_quality="qualified",
        depth_supervision_eligible=False,
        depth_quality="missing",
        depth_insertion_eligible=False,
    )
    admitted = owner.decide(second)

    assert admitted.kind is DispositionKind.ADMIT
    assert backpressure.decision_for(1) is BackpressureAction.ADMIT


def test_canonical_keyframe_config_builder_maps_every_opt_in_gate():
    from eligsir.mapping.scheduler.keyframes import (
        keyframe_admitter_config_from_args,
    )

    config = keyframe_admitter_config_from_args(
        SimpleNamespace(
            streaming_min_depth=0.2,
            streaming_max_depth=6.0,
            streaming_keyframe_max_history=19,
            streaming_kf_parallax_gate=True,
            streaming_kf_parallax_deg=4.0,
            streaming_kf_covis_thresh=0.7,
            streaming_keyframe_exposure_gate=True,
            streaming_keyframe_exposure_max_delta=0.2,
            streaming_keyframe_exposure_force_admit_iters=17,
            streaming_keyframe_blur_gate=True,
            streaming_keyframe_blur_min_laplacian=30.0,
            streaming_keyframe_blur_downsample=2,
            streaming_keyframe_blur_force_admit_iters=18,
            streaming_keyframe_quality_enabled=True,
        )
    )

    assert (config.depth_min, config.depth_max, config.max_history) == (0.2, 6.0, 19)
    assert (config.parallax_gate, config.parallax_min_deg, config.parallax_covis_max) == (
        True,
        4.0,
        0.7,
    )
    assert (
        config.exposure_gate_enabled,
        config.exposure_max_delta,
        config.exposure_force_admit_iters,
    ) == (True, 0.2, 17)
    assert (
        config.blur_gate_enabled,
        config.blur_gate_min_laplacian,
        config.blur_gate_downsample,
        config.blur_gate_force_admit_iters,
    ) == (True, 30.0, 2, 18)
    assert config.quality_enabled


@pytest.mark.parametrize(
    ("gate_args", "baseline_facts", "candidate_facts", "reason"),
    [
        (
            {"streaming_keyframe_blur_gate": True},
            {"rgb_blur_laplacian": 100.0, "rgb_luma": 0.5},
            {"rgb_blur_laplacian": 0.0, "rgb_luma": 0.5},
            "blur_reject",
        ),
        (
            {
                "streaming_keyframe_blur_gate": True,
                "streaming_keyframe_quality_enabled": True,
                "streaming_keyframe_quality_min_blur": 8.0,
            },
            {"rgb_blur_laplacian": 100.0, "rgb_luma": 0.5},
            {"rgb_blur_laplacian": 10.0, "rgb_luma": 0.5},
            "blur_reject",
        ),
        (
            {"streaming_keyframe_exposure_gate": True},
            {"rgb_blur_laplacian": 100.0, "rgb_luma": 0.5},
            {"rgb_blur_laplacian": 100.0, "rgb_luma": 0.9},
            "exposure_reject",
        ),
        (
            {
                "streaming_kf_parallax_gate": True,
                "streaming_keyframe_coverage_gain_gate": True,
            },
            {"coverage_gain": 0.8, "parallax_deg": 10.0},
            {"coverage_gain": 0.1, "parallax_deg": 1.0},
            "rej_parallax",
        ),
    ],
)
def test_snapshot_adapter_preserves_each_opt_in_gate_precedence(
    gate_args, baseline_facts, candidate_facts, reason
):
    from eligsir.mapping.admission.observation_owner import ObservationOwner

    provider = keyframe_admitter_proposal_provider_from_args(
        SimpleNamespace(
            streaming_keyframe_max_gap=100,
            streaming_keyframe_hard_backstop_frames=400.0,
            streaming_keyframe_max_gap_require_novelty=False,
            streaming_keyframe_velocity_adaptive=False,
            **gate_args,
        )
    )
    owner = ObservationOwner(
        lambda snapshot: BackpressureDecision(
            frame_id=snapshot.source.source_ordinal,
            action=BackpressureAction.ADMIT,
            reason="soft_all_capacity_available",
            admit_supervision=True,
            allow_geometry_insertion=True,
        ),
        semantic_providers=(provider,),
    )
    common = {
        "overlap": 0.5,
        "rgb_quality_score": 1.0,
        "depth_valid_ratio": 1.0,
        "rgb_supervision_eligible": True,
        "rgb_quality": "qualified",
        "depth_supervision_eligible": True,
        "depth_quality": "qualified",
        "depth_insertion_eligible": True,
    }
    owner.decide(
        _tracked_snapshot(
            _frame({"index": 0, "x": 0.0, "timestamp": 0.0}),
            **common,
            **baseline_facts,
        )
    )

    disposition = owner.decide(
        _tracked_snapshot(
            _frame({"index": 1, "x": 0.1, "timestamp": 0.03}),
            **common,
            **candidate_facts,
        )
    )

    assert disposition.reason == reason
