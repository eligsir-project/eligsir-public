"""Tests for ``GaussianBelief`` (issue #1365, canonical schema -- see
utils/gaussian_belief.py and
docs/architecture/2026-08-31-probabilistic-gaussian-belief-and-population-control.md).

Per ``utils/resolution_exposure.py``'s own precedent ("pure torch and
device-agnostic -- unit-testable on CPU against duck-typed stand-ins, with no
GsplatGaussianModel/CUDA dependency"), the bulk of this coverage exercises
``utils/gaussian_belief.py``'s functions directly against a lightweight duck-typed
stand-in rather than a real ``GsplatGaussianModel`` -- everything in that module
takes its device from its inputs (or an explicit ``device=`` override), so these run
on CPU with no skip needed, satisfying this phase's "fully verifiable via CPU
unit/regression tests, no GPU training" requirement. No test in this file performs a
full-belief device-to-host transfer (canonical §15.9/§16.8) except the CUDA-only
integration test's final checkpoint-equality assertions, which is test-only code
explicitly carved out by that constraint.

A CUDA-only integration test at the bottom exercises the real
``GsplatGaussianModel`` wiring end-to-end (mirrors
``tests/test_gaussian_lineage_model_wiring.py``'s pattern) -- skipped here, runs in
any GPU-enabled CI/container.
"""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from eligsir.utils.gaussian_belief import (
    FIELD_PROVENANCE,
    NEW_BUFFER_BYTES_PER_GAUSSIAN,
    NEW_BUFFER_NAMES,
    RELOCATION_POLICY,
    STRUCTURE_ONLY_FIELDS,
    TOTAL_NEW_BYTES_PER_GAUSSIAN,
    ExistenceBelief,
    GaussianBelief,
    GeometryBelief,
    ObservabilityBelief,
    ProvenanceBelief,
    RepresentationBelief,
    TemporalBelief,
    compute_p_exist,
    ensure_belief_buffers,
    growth_pad_belief_buffers,
    init_belief_buffers,
    new_row_belief_buffers,
    prune_belief_buffers,
    relocate_belief_buffers,
    reset_belief_buffers,
    sync_belief_buffer_lengths,
)


def _stub(n, device="cpu"):
    """Minimal duck-typed stand-in exposing only ``get_xyz`` -- what every
    buffer-management function in ``utils/gaussian_belief.py`` needs at minimum."""
    obj = SimpleNamespace()
    obj.get_xyz = torch.randn(n, 3, device=device)
    return obj


def _full_stub(n, device="cpu"):
    """Stand-in additionally carrying every ALIAS buffer ``GaussianBelief.from_gaussians``
    reads, so the accessor can be exercised without a real ``GsplatGaussianModel``."""
    obj = _stub(n, device)
    obj.gaussian_uid = torch.arange(n, dtype=torch.int64, device=device)
    obj.birth_iter = torch.zeros(n, dtype=torch.int32, device=device)
    obj.birth_method = torch.zeros(n, dtype=torch.int8, device=device)
    obj.parent_uid = torch.full((n,), -1, dtype=torch.int64, device=device)
    obj.root_uid = obj.gaussian_uid.clone()
    obj.generation = torch.zeros(n, dtype=torch.int32, device=device)
    obj.support_count = torch.zeros(n, dtype=torch.int16, device=device)
    obj.anchor_confidence = torch.zeros(n, dtype=torch.float32, device=device)
    obj.visibility_ema = torch.zeros(n, dtype=torch.float32, device=device)
    obj.utility_ema = torch.zeros(n, dtype=torch.float32, device=device)
    reset_belief_buffers(obj, n, device=device)
    return obj


# --------------------------------------------------------------------------
# compute_p_exist
# --------------------------------------------------------------------------


class TestComputePExist:
    def test_zero_evidence_defaults_to_one(self):
        alpha = torch.zeros(3)
        beta = torch.zeros(3)
        assert torch.equal(compute_p_exist(alpha, beta), torch.ones(3))

    def test_pure_support(self):
        alpha = torch.tensor([5.0])
        beta = torch.tensor([0.0])
        assert torch.allclose(compute_p_exist(alpha, beta), torch.tensor([1.0]))

    def test_pure_contradiction(self):
        alpha = torch.tensor([0.0])
        beta = torch.tensor([5.0])
        assert torch.allclose(compute_p_exist(alpha, beta), torch.tensor([0.0]))

    def test_mixed(self):
        alpha = torch.tensor([3.0])
        beta = torch.tensor([1.0])
        assert torch.allclose(compute_p_exist(alpha, beta), torch.tensor([0.75]))


# --------------------------------------------------------------------------
# Byte-budget self-consistency (canonical §15.6 requires a documented budget)
# --------------------------------------------------------------------------


def test_byte_budget_matches_buffer_specs_exactly():
    assert set(NEW_BUFFER_BYTES_PER_GAUSSIAN.keys()) == set(NEW_BUFFER_NAMES)
    assert TOTAL_NEW_BYTES_PER_GAUSSIAN == sum(NEW_BUFFER_BYTES_PER_GAUSSIAN.values())
    assert TOTAL_NEW_BYTES_PER_GAUSSIAN == 60


# --------------------------------------------------------------------------
# Buffer lifecycle: init / reset / ensure / new-row / growth / prune / relocate / sync
# --------------------------------------------------------------------------


def test_init_allocates_zero_length_buffers():
    obj = _stub(0)
    init_belief_buffers(obj, device="cpu")
    for name in NEW_BUFFER_NAMES:
        assert getattr(obj, name).shape[0] == 0


def test_reset_seeds_birth_anchor_to_current_position_and_sane_defaults():
    obj = _stub(5)
    reset_belief_buffers(obj, 5, device="cpu")
    assert torch.equal(obj.birth_anchor_xyz, obj.get_xyz)
    assert torch.equal(obj.p_exist, torch.ones(5, dtype=torch.float16))
    assert torch.equal(obj.surface_anchor_id, torch.full((5,), -1, dtype=torch.int64))
    assert torch.equal(obj.support_evidence, torch.zeros(5, dtype=torch.float16))
    assert torch.equal(obj.contradiction_evidence, torch.zeros(5, dtype=torch.float16))


def test_reset_zero_count_does_not_crash():
    obj = _stub(0)
    reset_belief_buffers(obj, 0, device="cpu")
    for name in NEW_BUFFER_NAMES:
        assert getattr(obj, name).shape[0] == 0


def test_ensure_repairs_mismatched_length():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    obj.get_xyz = torch.randn(6, 3)  # simulate growth without buffer update
    ensure_belief_buffers(obj)
    for name in NEW_BUFFER_NAMES:
        assert getattr(obj, name).shape[0] == 6
    assert torch.equal(obj.birth_anchor_xyz, obj.get_xyz)


def test_ensure_is_noop_when_already_synced():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    before = {name: getattr(obj, name).clone() for name in NEW_BUFFER_NAMES}
    ensure_belief_buffers(obj)
    for name in NEW_BUFFER_NAMES:
        assert torch.equal(getattr(obj, name), before[name])


def test_new_row_buffers_seed_birth_anchor_to_new_positions():
    new_means = torch.randn(3, 3)
    rows = new_row_belief_buffers(3, new_means)
    assert torch.equal(rows["birth_anchor_xyz"], new_means)
    assert set(rows.keys()) == set(NEW_BUFFER_NAMES)
    assert torch.equal(rows["p_exist"], torch.ones(3, dtype=torch.float16))
    assert torch.equal(rows["surface_anchor_id"], torch.full((3,), -1, dtype=torch.int64))


def test_growth_pad_extends_all_buffers_and_inherits_parent_anchor():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    obj.get_xyz = torch.cat([obj.get_xyz, torch.randn(2, 3)], dim=0)
    parent_idxs = torch.tensor([0, 1])
    growth_pad_belief_buffers(obj, old_count=4, added_count=2, parent_idxs=parent_idxs)
    for name in NEW_BUFFER_NAMES:
        assert getattr(obj, name).shape[0] == 6
    # The two clones' birth_anchor_xyz equals their sampled parents' anchors.
    assert torch.equal(obj.birth_anchor_xyz[4:], obj.birth_anchor_xyz[:4][parent_idxs])


def test_growth_pad_without_parent_idxs_falls_back_to_current_position():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    new_positions = torch.randn(2, 3)
    obj.get_xyz = torch.cat([obj.get_xyz, new_positions], dim=0)
    growth_pad_belief_buffers(obj, old_count=4, added_count=2, parent_idxs=None)
    assert torch.equal(obj.birth_anchor_xyz[4:], new_positions)


def test_growth_pad_skips_buffer_with_mismatched_length():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    obj.p_exist = obj.p_exist[:3]  # simulate a buffer already desynced elsewhere
    growth_pad_belief_buffers(obj, old_count=4, added_count=2)
    assert obj.p_exist.shape[0] == 3  # left untouched, not corrupted further
    assert obj.evidence_recency.shape[0] == 6  # unaffected buffers still extend normally


def test_prune_slices_all_buffers():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    mask = torch.tensor([True, False, True, False])
    valid = ~mask
    prune_belief_buffers(obj, mask, valid)
    for name in NEW_BUFFER_NAMES:
        assert getattr(obj, name).shape[0] == 2


def test_prune_skips_buffer_with_mismatched_length():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    obj.redundancy = obj.redundancy[:3]
    mask = torch.tensor([True, False, True, False])
    prune_belief_buffers(obj, mask, ~mask)
    assert obj.redundancy.shape[0] == 3  # untouched
    assert obj.evidence_recency.shape[0] == 2  # normally pruned


def test_relocate_applies_donor_and_reset_policies():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    obj.redundancy = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    obj.rgb_residual_ema = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    obj.depth_residual_ema = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    obj.expected_marginal_gain = torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32)
    obj.utility_ema = torch.tensor([8.0, 7.0, 6.0, 5.0], dtype=torch.float32)
    obj.p_exist = torch.tensor([0.9, 0.1, 0.5, 0.2], dtype=torch.float16)
    obj.surface_anchor_id = torch.tensor([10, 11, 12, 13], dtype=torch.int64)
    obj.birth_anchor_xyz = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]
    )
    dead_indices = torch.tensor([1, 3])
    sampled_idxs = torch.tensor([0, 2])
    relocate_belief_buffers(obj, dead_indices, sampled_idxs)
    # "reset" policy: redundancy is location-specific evidence and a relocated
    # slot is a fresh hypothesis at its new target, so it must not inherit the
    # donor's old-hypothesis evidence.
    assert torch.equal(obj.redundancy[dead_indices], torch.zeros(2, dtype=torch.float32))
    for name in ("rgb_residual_ema", "depth_residual_ema", "expected_marginal_gain", "utility_ema"):
        assert torch.equal(getattr(obj, name)[dead_indices], torch.zeros(2, dtype=torch.float32))
    assert torch.equal(
        obj.surface_anchor_id[dead_indices], torch.full((2,), -1, dtype=torch.int64)
    )
    assert torch.equal(
        obj.birth_anchor_xyz[dead_indices], obj.birth_anchor_xyz[sampled_idxs]
    )
    # "reset" policy: resets to the default (1.0), NOT the donor's value.
    assert torch.equal(obj.p_exist[dead_indices], torch.ones(2, dtype=torch.float16))


def test_relocation_policy_covers_every_new_buffer_with_a_valid_value():
    assert set(RELOCATION_POLICY.keys()) == set(NEW_BUFFER_NAMES)
    assert set(RELOCATION_POLICY.values()) <= {"donor", "reset"}


def test_sync_pads_and_truncates():
    obj = _stub(4)
    reset_belief_buffers(obj, 4, device="cpu")
    obj.get_xyz = torch.cat([obj.get_xyz, torch.randn(2, 3)], dim=0)
    sync_belief_buffer_lengths(obj)
    for name in NEW_BUFFER_NAMES:
        assert getattr(obj, name).shape[0] == 6
    obj.get_xyz = obj.get_xyz[:3]
    sync_belief_buffer_lengths(obj)
    for name in NEW_BUFFER_NAMES:
        assert getattr(obj, name).shape[0] == 3


# --------------------------------------------------------------------------
# GaussianBelief accessor
# --------------------------------------------------------------------------


def test_from_gaussians_aliases_are_same_tensor_objects_zero_copy():
    obj = _full_stub(4)
    belief = GaussianBelief.from_gaussians(obj)
    assert belief.provenance.gaussian_uid is obj.gaussian_uid
    assert belief.provenance.birth_update is obj.birth_iter
    assert belief.observability.independent_support_views is obj.support_count
    assert belief.geometry.p_surface_consistent is obj.anchor_confidence
    assert belief.representation.render_contribution is obj.visibility_ema
    assert belief.representation.gradient_utility_evidence is obj.utility_ema
    assert belief.n == 4


def test_from_gaussians_optional_fields_none_when_flag_off():
    obj = _full_stub(4)
    belief = GaussianBelief.from_gaussians(obj)
    assert belief.observability.meaningful_exposure_count is None
    assert belief.observability.highest_supervised_tier is None
    assert belief.observability.last_supported_update is None


def test_from_gaussians_picks_up_optional_buffers_when_present():
    obj = _full_stub(4)
    obj.exposure_count = torch.zeros(4, dtype=torch.int32)
    obj.supervised_tier_high_water = torch.full((4,), -1, dtype=torch.int8)
    obj.last_supervised_iter = torch.zeros(4, dtype=torch.int32)
    belief = GaussianBelief.from_gaussians(obj)
    assert belief.observability.meaningful_exposure_count is obj.exposure_count
    assert belief.observability.highest_supervised_tier is obj.supervised_tier_high_water
    assert belief.observability.last_supported_update is obj.last_supervised_iter


def test_from_gaussians_repairs_desynced_new_buffers():
    obj = _full_stub(4)
    obj.get_xyz = torch.cat([obj.get_xyz, torch.randn(2, 3)], dim=0)
    belief = GaussianBelief.from_gaussians(obj)  # calls ensure_belief_buffers internally
    assert belief.n == 6
    assert belief.existence.p_exist.shape[0] == 6
    assert belief.geometry.surface_anchor_id.shape[0] == 6


def test_geometry_belief_defaults_to_unassigned_anchor():
    obj = _full_stub(3)
    belief = GaussianBelief.from_gaussians(obj)
    assert torch.equal(
        belief.geometry.surface_anchor_id, torch.full((3,), -1, dtype=torch.int64)
    )


# --------------------------------------------------------------------------
# Schema self-consistency (documents the alias table as testable data)
# --------------------------------------------------------------------------


def test_field_provenance_covers_every_dataclass_field_exactly_once():
    all_names = set()
    for cls in (
        ProvenanceBelief, ObservabilityBelief, ExistenceBelief,
        GeometryBelief, RepresentationBelief, TemporalBelief,
    ):
        all_names.update(f.name for f in fields(cls))
    assert all_names == set(FIELD_PROVENANCE.keys())


def test_new_buffer_names_match_non_alias_field_provenance_entries():
    non_alias = {
        name for name, meta in FIELD_PROVENANCE.items() if meta["status"] != "alias"
    }
    assert non_alias == set(NEW_BUFFER_NAMES)


def test_structure_only_fields_match_field_provenance():
    derived = {
        name for name, meta in FIELD_PROVENANCE.items()
        if meta["status"] == "new_structure_only"
    }
    assert derived == STRUCTURE_ONLY_FIELDS
    # birth_anchor_xyz is the one non-alias field that IS wired -- confirm it's
    # deliberately excluded from "structure only", not merely absent by omission.
    assert FIELD_PROVENANCE["birth_anchor_xyz"]["status"] == "new_wired"
    assert "birth_anchor_xyz" not in STRUCTURE_ONLY_FIELDS


def test_alias_entries_name_a_real_buffer_used_by_from_gaussians():
    # Every "alias" entry's existing_buffer must be something _full_stub carries
    # (i.e. a buffer this phase reuses verbatim, not invented).
    stub_attrs = vars(_full_stub(1)).keys()
    for name, meta in FIELD_PROVENANCE.items():
        if meta["status"] == "alias":
            assert meta["existing_buffer"] in stub_attrs or meta["existing_buffer"] in (
                "exposure_count", "supervised_tier_high_water", "last_supervised_iter",
            ), f"{name} aliases {meta['existing_buffer']!r}, not present anywhere testable"


# --------------------------------------------------------------------------
# CUDA-only integration test: real GsplatGaussianModel wiring end-to-end.
# Mirrors tests/test_gaussian_lineage_model_wiring.py's pattern; skips on CPU CI
# per this repo's existing convention for GsplatGaussianModel (device="cuda" is
# hardcoded throughout scene/gsplat_model.py itself, not something this phase
# changes).
# --------------------------------------------------------------------------

pytest_cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GsplatGaussianModel is CUDA-only"
)


@pytest_cuda_only
def test_model_wiring_survives_growth_prune_relocate_and_checkpoint_roundtrip():
    from argparse import Namespace

    from scene.gsplat_model import GsplatGaussianModel
    from eligsir.utils.graphics_utils import BasicPointCloud
    from eligsir.utils.strategies.mcmc.proposal import add_energy_guided, reset_relocated_state

    def _training_args():
        return Namespace(
            iterations=100, position_lr_init=0.00016, position_lr_final=0.0000016,
            position_lr_delay_mult=0.01, position_lr_max_steps=20000,
            feature_lr=0.0025, feature_lr_final=0.00025, opacity_lr=0.05,
            opacity_lr_final=0.001, scaling_lr=0.005, scaling_lr_final=0.0005,
            rotation_lr=0.001, rotation_lr_final=0.0001, percent_dense=0.01,
            lambda_dssim=0.2, densification_interval=100, opacity_reset_interval=30000,
            densify_from_iter=500, densify_until_iter=15000,
            densify_grad_threshold=0.0002, random_background=False,
        )

    model = GsplatGaussianModel(sh_degree=0)
    n0 = 4
    pts = torch.randn(n0, 3)
    cols = torch.rand(n0, 3)
    pcd = BasicPointCloud(points=pts.numpy(), colors=cols.numpy(), normals=torch.zeros(n0, 3).numpy())
    model.create_from_pcd(pcd, spatial_lr_scale=1.0)
    model.training_setup(_training_args())

    for name in NEW_BUFFER_NAMES:
        assert getattr(model, name).shape[0] == n0
    assert torch.equal(model.birth_anchor_xyz, model.get_xyz.detach())

    # Depth-insertion append.
    new_pts = torch.randn(2, 3, device="cuda")
    new_cols = torch.rand(2, 3, device="cuda")
    added = model.add_points_as_gaussians(new_pts, new_cols, birth_iter=100)
    assert added == 2
    for name in NEW_BUFFER_NAMES:
        assert getattr(model, name).shape[0] == n0 + 2
    assert torch.equal(model.birth_anchor_xyz[-2:], new_pts)

    n1 = n0 + 2

    # MCMC growth: exercise the real proposal path, which expands parameters
    # and optimizer state before append_running_state wires belief buffers.
    added = add_energy_guided(
        model,
        _training_args(),
        utility=None,
        temperature=1.0,
        binoms=torch.ones((16, 16), device="cuda"),
        min_opacity=0.001,
        cap_max=n1 + 2,
        growth_factor=2.0,
        iteration=200,
    )
    assert added == 2
    for name in NEW_BUFFER_NAMES:
        assert getattr(model, name).shape[0] == n1 + 2
    assert all(param.shape[0] == n1 + 2 for param in model.params.values())
    assert all(
        optimizer.param_groups[0]["params"][0].shape[0] == n1 + 2
        for optimizer in model.optimizers.values()
    )
    n2 = n1 + 2

    # MCMC relocation.
    dead_indices = torch.tensor([0, 1], device="cuda")
    sampled_idxs = torch.tensor([2, 3], device="cuda")
    model.redundancy[sampled_idxs] = torch.tensor([9.0, 8.0], device="cuda")
    model.rgb_residual_ema[sampled_idxs] = torch.tensor([7.0, 6.0], device="cuda")
    model.depth_residual_ema[sampled_idxs] = torch.tensor([6.0, 5.0], device="cuda")
    model.expected_marginal_gain[sampled_idxs] = torch.tensor([5.0, 4.0], device="cuda")
    model.utility_ema[sampled_idxs] = torch.tensor([4.0, 3.0], device="cuda")
    model.surface_anchor_id[sampled_idxs] = torch.tensor([12, 13], device="cuda")
    reset_relocated_state(model, dead_indices, sampled_idxs, iteration=300)
    assert torch.equal(
        model.redundancy[dead_indices],
        torch.zeros(2, dtype=torch.float32, device="cuda"),
    )
    for name in ("rgb_residual_ema", "depth_residual_ema", "expected_marginal_gain", "utility_ema"):
        assert torch.equal(
            getattr(model, name)[dead_indices],
            torch.zeros(2, dtype=torch.float32, device="cuda"),
        )
    assert torch.equal(
        model.surface_anchor_id[dead_indices],
        torch.full((2,), -1, dtype=torch.int64, device="cuda"),
    )
    assert torch.equal(model.p_exist[dead_indices], torch.ones(2, dtype=torch.float16, device="cuda"))

    # Prune.
    mask = torch.zeros(n2, dtype=torch.bool, device="cuda")
    mask[0] = True
    model.prune_points(mask)
    n3 = n2 - 1
    for name in NEW_BUFFER_NAMES:
        assert getattr(model, name).shape[0] == n3

    # Checkpoint round-trip.
    blob = model.capture()
    restored = GsplatGaussianModel(sh_degree=0)
    restored.create_from_pcd(pcd, spatial_lr_scale=1.0)
    restored.training_setup(_training_args())
    restored.restore(blob, _training_args())
    for name in NEW_BUFFER_NAMES:
        assert torch.equal(getattr(restored, name).cpu(), getattr(model, name).cpu())
