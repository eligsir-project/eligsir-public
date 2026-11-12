"""Tests for the mapping evidence substrate (Wave 3, PR W3-05).

Pure CPU torch — no GPU/container-specific behavior, matches this repo's
"CPU-only feasible" smoke-test convention (see CLAUDE.md).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from eligsir.mapping.evidence.builder import EvidenceBuilder
from eligsir.mapping.evidence.contracts import (
    EvidenceMeta,
    EvidenceSummary,
    GaussianEvidence,
    InputEvidence,
    RenderEvidence,
)
from eligsir.mapping.evidence.frame_history import (
    FrameEvidenceHistory,
    FrameEvidenceRecord,
)
from eligsir.mapping.evidence.gaussian_history import (
    GaussianEvidenceHistory,
    StaleEvidenceError,
)
from eligsir.mapping.evidence.reducers import (
    ema_update,
    masked_ema_update,
    scalar_fraction,
    scalar_mean,
    under_supported_mass,
)
from eligsir.mapping.evidence.requirements import EvidenceRequirements
from eligsir.mapping.evidence.serialization import EvidenceLogReader, EvidenceLogWriter


class _FakeEnergyTerms:
    """Minimal duck-typed stand-in for
    ``utils.strategies.mcmc.energy_policy.GaussianEnergyTerms``.
    """

    def __init__(self, n: int = 8):
        self.alpha = torch.rand(n)
        self.support = torch.rand(n)
        self.max_scale = torch.rand(n)
        self.min_scale = torch.rand(n) * 0.1
        self.anisotropy = self.max_scale / self.min_scale
        self.xyz_grad_norm = torch.rand(n)
        self.opacity_grad_abs = torch.rand(n)
        self.scale_grad_norm = torch.rand(n)
        self.visibility = torch.rand(n) > 0.5
        self.age_frac = torch.rand(n)
        self.has_anchor = None
        self.has_depth_conflict = None
        self.provisional = None
        self.geometry_confidence = torch.rand(n)
        self.surface_state = None
        self.normal_drift = None
        self.duplicate_mass = None
        self.free_space_conflict = torch.rand(n)
        self.surface_cell_overpopulation = None
        self.render_contribution_ema = torch.rand(n)


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def test_evidence_field_names_match_spec():
    """The four dataclasses must carry exactly the fields listed in plan
    section 10.3, plus the appended ``meta`` field — later PRs import these
    field names directly.
    """
    input_fields = {f.name for f in dataclasses.fields(InputEvidence)}
    assert input_fields == {
        "pose_confidence",
        "image_quality",
        "depth_valid_fraction",
        "geometry_reliability_mean",
        "meta",
    }

    render_fields = {f.name for f in dataclasses.fields(RenderEvidence)}
    assert render_fields == {
        "rgb_residual",
        "depth_residual",
        "alpha",
        "rendered_depth",
        "transmittance",
        "meta",
    }

    gaussian_fields = {f.name for f in dataclasses.fields(GaussianEvidence)}
    assert gaussian_fields == {
        "visibility",
        "projected_radius",
        "contribution",
        "xyz_grad_ema",
        "opacity_grad_ema",
        "support",
        "geometry_confidence",
        "free_space_conflict",
        "age",
        "meta",
    }

    summary_fields = {f.name for f in dataclasses.fields(EvidenceSummary)}
    assert summary_fields == {
        "new_surface_ratio",
        "persistent_residual",
        "under_supported_mass",
        "expected_gradient_value",
        "predicted_cost_ms",
        "meta",
    }


def test_input_evidence_and_summary_are_frozen():
    ev = InputEvidence(0.9, 0.8, 1.0, 0.7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.pose_confidence = 0.1

    summary = EvidenceSummary(0.1, 0.2, 0.3, 0.4, 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.new_surface_ratio = 1.0


# ---------------------------------------------------------------------------
# Requirements
# ---------------------------------------------------------------------------


def test_requirements_default_all_off():
    req = EvidenceRequirements()
    assert req.any_needed() is False
    assert req == EvidenceRequirements.none()


def test_requirements_everything_all_on():
    req = EvidenceRequirements.everything()
    assert req.any_needed() is True
    for f in dataclasses.fields(req):
        assert getattr(req, f.name) is True


# ---------------------------------------------------------------------------
# Builder: demand-driven, near-zero-cost when nothing is requested
# ---------------------------------------------------------------------------


def test_builder_returns_none_for_everything_when_nothing_requested():
    builder = EvidenceBuilder(EvidenceRequirements.none())
    terms = _FakeEnergyTerms()

    assert builder.build_input_evidence(
        pose_confidence=1.0, image_quality=1.0, depth_valid_fraction=1.0, geometry_reliability_mean=1.0
    ) is None
    assert builder.build_render_evidence({}) is None
    assert builder.build_gaussian_evidence(terms) is None
    assert builder.build_summary(terms) is None


def test_builder_builds_gaussian_evidence_when_requested():
    req = EvidenceRequirements(
        want_gaussian_evidence=True,
        want_visibility=True,
        want_support=True,
        want_age=True,
        want_contribution=True,
    )
    builder = EvidenceBuilder(req)
    terms = _FakeEnergyTerms(n=5)

    evidence = builder.build_gaussian_evidence(terms, topology_version=3, map_version=7)
    assert evidence is not None
    assert torch.equal(evidence.visibility, terms.visibility)
    assert torch.equal(evidence.support, terms.support)
    assert torch.equal(evidence.age, terms.age_frac)
    assert torch.equal(evidence.contribution, terms.render_contribution_ema)
    # Not requested -> zeros, not garbage/None (dataclass fields are non-Optional tensors).
    assert torch.equal(evidence.geometry_confidence, torch.zeros(5))
    assert evidence.meta.topology_version == 3
    assert evidence.meta.map_version == 7


def test_builder_gaussian_evidence_does_not_mutate_source_tensors():
    req = EvidenceRequirements(want_gaussian_evidence=True, want_visibility=True, want_support=True)
    builder = EvidenceBuilder(req)
    terms = _FakeEnergyTerms(n=4)
    support_before = terms.support.clone()

    evidence = builder.build_gaussian_evidence(terms, topology_version=0, map_version=0)
    evidence.support[0] = 999.0  # mutate the returned copy

    # Source tensor must be unaffected -- evidence is read-only observation,
    # never a live alias that a downstream mutation could corrupt.
    assert torch.equal(terms.support, support_before)


def test_builder_render_evidence_derived_fields():
    req = EvidenceRequirements(
        want_render_evidence=True, want_rgb_residual=True, want_alpha=True, want_transmittance=True
    )
    builder = EvidenceBuilder(req)
    render_pkg = {"render": torch.ones(3, 4, 4), "alpha": torch.full((1, 4, 4), 0.25)}
    gt_image = torch.zeros(3, 4, 4)

    evidence = builder.build_render_evidence(render_pkg, gt_image=gt_image)
    assert evidence is not None
    assert torch.allclose(evidence.rgb_residual, torch.ones(3, 4, 4))
    assert torch.allclose(evidence.alpha, torch.full((1, 4, 4), 0.25))
    assert torch.allclose(evidence.transmittance, torch.full((1, 4, 4), 0.75))
    assert evidence.depth_residual is None  # not requested
    assert evidence.rendered_depth is None  # not requested


def test_builder_gaussian_evidence_ema_accumulates_across_calls():
    req = EvidenceRequirements(want_gaussian_evidence=True, want_xyz_grad_ema=True)
    history = GaussianEvidenceHistory()
    builder = EvidenceBuilder(req, history=history, grad_ema_decay=0.5)
    terms = _FakeEnergyTerms(n=3)
    terms.xyz_grad_norm = torch.tensor([1.0, 1.0, 1.0])

    ev1 = builder.build_gaussian_evidence(terms, topology_version=1, map_version=1)
    assert torch.allclose(ev1.xyz_grad_ema, torch.tensor([1.0, 1.0, 1.0]))

    terms.xyz_grad_norm = torch.tensor([0.0, 0.0, 0.0])
    ev2 = builder.build_gaussian_evidence(terms, topology_version=1, map_version=2)
    # decay=0.5: 0.5*1.0 + 0.5*0.0 = 0.5
    assert torch.allclose(ev2.xyz_grad_ema, torch.tensor([0.5, 0.5, 0.5]))


def test_builder_gaussian_evidence_ema_resets_on_topology_change():
    req = EvidenceRequirements(want_gaussian_evidence=True, want_xyz_grad_ema=True)
    history = GaussianEvidenceHistory()
    builder = EvidenceBuilder(req, history=history, grad_ema_decay=0.5)

    terms = _FakeEnergyTerms(n=3)
    terms.xyz_grad_norm = torch.tensor([1.0, 1.0, 1.0])
    builder.build_gaussian_evidence(terms, topology_version=1, map_version=1)

    # Topology bumped (insert/prune) -> EMA state must cold-start, not blend
    # with the previous (now-mismatched-index) population's history.
    terms2 = _FakeEnergyTerms(n=5)
    terms2.xyz_grad_norm = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
    ev2 = builder.build_gaussian_evidence(terms2, topology_version=2, map_version=2)
    assert torch.allclose(ev2.xyz_grad_ema, torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0]))


# ---------------------------------------------------------------------------
# GaussianEvidenceHistory: stale-topology rejection (deliverable 3)
# ---------------------------------------------------------------------------


def test_gaussian_history_rejects_stale_topology_read():
    history = GaussianEvidenceHistory()
    evidence = GaussianEvidence(
        visibility=torch.zeros(3, dtype=torch.bool),
        projected_radius=torch.zeros(3),
        contribution=torch.zeros(3),
        xyz_grad_ema=torch.zeros(3),
        opacity_grad_ema=torch.zeros(3),
        support=torch.zeros(3),
        geometry_confidence=torch.zeros(3),
        free_space_conflict=torch.zeros(3),
        age=torch.zeros(3),
    )
    history.update(evidence, topology_version=5, map_version=10)

    # Reading at the SAME topology_version succeeds.
    assert history.get(5) is evidence

    # A structural mutation bumps topology_version. Reading with the OLD
    # version now must raise, not silently return the stale evidence.
    with pytest.raises(StaleEvidenceError):
        history.get(6)

    assert history.is_valid_for(5) is True
    assert history.is_valid_for(6) is False


def test_gaussian_history_begin_step_invalidates_stale_evidence():
    """Regression test: begin_step() bumping topology_version must also
    drop the previously-stored GaussianEvidence, not just the EMA state --
    otherwise is_valid_for(new_version) would report the OLD evidence as
    valid for the new topology, silently misapplying stale per-Gaussian
    data to a mutated population.
    """
    history = GaussianEvidenceHistory()
    evidence_a = GaussianEvidence(
        visibility=torch.zeros(3, dtype=torch.bool),
        projected_radius=torch.zeros(3),
        contribution=torch.zeros(3),
        xyz_grad_ema=torch.zeros(3),
        opacity_grad_ema=torch.zeros(3),
        support=torch.zeros(3),
        geometry_confidence=torch.zeros(3),
        free_space_conflict=torch.zeros(3),
        age=torch.zeros(3),
    )
    history.update(evidence_a, topology_version=5, map_version=10)
    assert history.is_valid_for(5) is True

    # Structural mutation happens; a caller advances the step before any
    # fresh evidence has been built for the new topology yet.
    history.begin_step(topology_version=6, map_version=11)

    assert history.is_valid_for(6) is False
    with pytest.raises(StaleEvidenceError):
        history.get(6)


def test_gaussian_history_invalidate_clears_state():
    history = GaussianEvidenceHistory()
    history.set_ema_state(torch.ones(3), torch.ones(3))
    history.update(
        GaussianEvidence(
            visibility=torch.zeros(3, dtype=torch.bool),
            projected_radius=torch.zeros(3),
            contribution=torch.zeros(3),
            xyz_grad_ema=torch.zeros(3),
            opacity_grad_ema=torch.zeros(3),
            support=torch.zeros(3),
            geometry_confidence=torch.zeros(3),
            free_space_conflict=torch.zeros(3),
            age=torch.zeros(3),
        ),
        topology_version=1,
        map_version=1,
    )
    history.invalidate()
    assert history.is_valid_for(1) is False
    with pytest.raises(StaleEvidenceError):
        history.get(1)
    assert history.ema_state() == (None, None)


# ---------------------------------------------------------------------------
# FrameEvidenceHistory
# ---------------------------------------------------------------------------


def test_frame_history_ring_buffer_bounds():
    history = FrameEvidenceHistory(max_frames=2)
    history.append(FrameEvidenceRecord(frame_id=0))
    history.append(FrameEvidenceRecord(frame_id=1))
    history.append(FrameEvidenceRecord(frame_id=2))

    assert len(history) == 2
    assert [r.frame_id for r in history] == [1, 2]
    assert history.latest().frame_id == 2
    assert history.get(1) is not None
    assert history.get(0) is None  # evicted


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def test_ema_update_cold_start_and_blend():
    obs1 = torch.tensor([1.0, 2.0])
    ema1 = ema_update(None, obs1, decay=0.9)
    assert torch.equal(ema1, obs1)
    assert ema1.data_ptr() != obs1.data_ptr()  # cloned, not aliased

    obs2 = torch.tensor([0.0, 0.0])
    ema2 = ema_update(ema1, obs2, decay=0.5)
    assert torch.allclose(ema2, torch.tensor([0.5, 1.0]))


def test_masked_ema_update_only_touches_masked_entries():
    prev = torch.tensor([10.0, 20.0, 30.0])
    obs = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.tensor([True, False, True])
    updated = masked_ema_update(prev, obs, mask, decay=0.5)
    assert torch.allclose(updated, torch.tensor([5.0, 20.0, 15.0]))


def test_scalar_mean_and_fraction_handle_empty():
    assert scalar_mean(torch.tensor([])) == 0.0
    assert scalar_fraction(torch.tensor([], dtype=torch.bool)) == 0.0
    assert scalar_mean(torch.tensor([1.0, 3.0])) == 2.0
    assert scalar_fraction(torch.tensor([True, False, True, False])) == 0.5


def test_under_supported_mass():
    support = torch.tensor([0.0, 0.5, 0.005])
    opacity = torch.tensor([1.0, 1.0, 1.0])
    mass = under_supported_mass(support, opacity, threshold=0.01)
    assert mass == pytest.approx(2.0)  # indices 0 and 2 are below threshold


# ---------------------------------------------------------------------------
# Serialization: frozen-log replay without rerunning rendering (deliverable 6)
# ---------------------------------------------------------------------------


def test_serialization_roundtrip(tmp_path):
    log_dir = str(tmp_path / "evidence_log")
    writer = EvidenceLogWriter(log_dir)

    input_ev = InputEvidence(0.9, 0.8, 1.0, 0.7, meta=EvidenceMeta(frame_id=1))
    writer.write_input_evidence(input_ev)

    render_ev = RenderEvidence(
        rgb_residual=torch.rand(3, 2, 2),
        depth_residual=None,
        alpha=torch.rand(1, 2, 2),
        rendered_depth=None,
        transmittance=None,
        meta=EvidenceMeta(frame_id=1, topology_version=2),
    )
    writer.write_render_evidence(render_ev)

    gaussian_ev = GaussianEvidence(
        visibility=torch.rand(4) > 0.5,
        projected_radius=torch.rand(4),
        contribution=torch.rand(4),
        xyz_grad_ema=torch.rand(4),
        opacity_grad_ema=torch.rand(4),
        support=torch.rand(4),
        geometry_confidence=torch.rand(4),
        free_space_conflict=torch.rand(4),
        age=torch.rand(4),
        meta=EvidenceMeta(frame_id=1, topology_version=2),
    )
    writer.write_gaussian_evidence(gaussian_ev)

    summary = EvidenceSummary(0.1, 0.2, 0.3, 0.4, 0.5, meta=EvidenceMeta(frame_id=1))
    writer.write_summary(summary)

    reader = EvidenceLogReader(log_dir)
    records = reader.read_all()
    assert len(records) == 4

    read_input, read_render, read_gaussian, read_summary = records
    assert read_input == input_ev

    assert torch.allclose(read_render.rgb_residual, render_ev.rgb_residual)
    assert torch.allclose(read_render.alpha, render_ev.alpha)
    assert read_render.depth_residual is None
    assert read_render.meta.topology_version == 2

    assert torch.equal(read_gaussian.visibility, gaussian_ev.visibility)
    assert torch.allclose(read_gaussian.support, gaussian_ev.support)
    assert read_gaussian.meta.topology_version == 2

    assert read_summary == summary


def test_serialization_replay_needs_no_torch_grad_or_render_call(tmp_path):
    """A replayed GaussianEvidence must be usable by policy code directly --
    no rendering, no autograd graph (everything loaded off disk is a plain
    leaf CPU tensor).
    """
    log_dir = str(tmp_path / "evidence_log2")
    writer = EvidenceLogWriter(log_dir)
    gaussian_ev = GaussianEvidence(
        visibility=torch.ones(3, dtype=torch.bool),
        projected_radius=torch.ones(3),
        contribution=torch.ones(3),
        xyz_grad_ema=torch.ones(3),
        opacity_grad_ema=torch.ones(3),
        support=torch.ones(3),
        geometry_confidence=torch.ones(3),
        free_space_conflict=torch.ones(3),
        age=torch.ones(3),
    )
    writer.write_gaussian_evidence(gaussian_ev)

    reader = EvidenceLogReader(log_dir)
    (replayed,) = reader.read_all()
    assert replayed.support.requires_grad is False
    assert replayed.support.device.type == "cpu"
    # A toy "policy" consuming this evidence with zero render/backward calls:
    utility = replayed.support * replayed.contribution
    assert torch.equal(utility, torch.ones(3))
