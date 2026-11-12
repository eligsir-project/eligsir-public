"""Read-only row-state inventory coverage for the gsplat model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from scene.gsplat_model import GsplatGaussianModel
from eligsir.mapping.gaussian_belief_accumulator import (
    ensure_population_policy_e0_buffers,
)


_PARAMETER_SHAPES = {
    "means": (3,), "scales": (3,), "quats": (4,), "opacities": (),
    "sh0": (1, 3), "shN": (3, 3),
}
_REQUIRED_BUFFER_NAMES = (
    "max_radii2D", "xyz_gradient_accum", "denom", "visibility_ema",
    "birth_frame", "birth_iter", "support_count", "provisional", "anchor_xyz",
    "anchor_iter", "depth_conflict_count", "depth_last_support_uid",
    "depth_last_conflict_uid", "lifecycle_state", "utility_ema", "anchor_scale_log",
    "anchor_opacity_logit", "anchor_normal", "anchor_confidence", "geometry_source",
    "surface_state", "submap_id", "anchor_keyframe_id", "gaussian_uid", "parent_uid",
    "root_uid", "generation", "birth_method", "last_reseed_iter", "birth_anchor_xyz",
    "baseline_diversity", "last_exposed_update", "support_evidence",
    "contradiction_evidence", "p_exist", "surface_anchor_id",
    "surface_position_uncertainty", "surface_normal_uncertainty", "rgb_residual_ema",
    "depth_residual_ema", "redundancy", "expected_marginal_gain", "change_probability",
    "evidence_recency", "confidence_decay",
)
_VECTOR_BUFFER_NAMES = {
    "xyz_gradient_accum", "denom", "visibility_ema", "anchor_xyz", "anchor_scale_log",
    "anchor_normal", "birth_anchor_xyz",
}
_OPTIONAL_COMPATIBILITY_NAMES = (
    "seen_count", "consensus_support_count", "consensus_conflict_count",
    "consensus_last_seen_uid", "consensus_last_voter_center", "last_supervised_iter",
    "exposure_count", "exposure_last_view_uid", "supervised_tier_high_water",
)


def _cpu_model(rows: int = 3):
    model = SimpleNamespace()
    model.params = {
        name: torch.zeros((rows, *shape)) for name, shape in _PARAMETER_SHAPES.items()
    }
    model.optimizers = {
        name: SimpleNamespace(param_groups=[{"params": [param]}], state={})
        for name, param in model.params.items()
    }
    for name in _REQUIRED_BUFFER_NAMES:
        if name in {"anchor_xyz", "anchor_scale_log", "anchor_normal", "birth_anchor_xyz"}:
            shape = (rows, 3)
        elif name in _VECTOR_BUFFER_NAMES:
            shape = (rows, 1)
        else:
            shape = (rows,)
        setattr(model, name, torch.zeros(shape))
    return model


def _descriptor_map(registry):
    return {descriptor.name: descriptor for descriptor in registry.descriptors}


def test_build_row_state_registry_covers_every_required_live_family():
    registry = GsplatGaussianModel.build_row_state_registry(_cpu_model())

    descriptors = _descriptor_map(registry)
    assert {"params", "render", "provenance", "belief", "surface", "compatibility"} <= {
        descriptor.family for descriptor in registry.descriptors
    }
    assert {f"params.{name}" for name in _PARAMETER_SHAPES} <= registry.names
    assert not {name for name in registry.names if name.startswith("optimizer.")}
    assert {"belief.p_exist", "belief.birth_anchor_xyz", "surface.surface_state"} <= registry.names
    assert descriptors["render.visibility_ema"].checkpoint is True
    assert descriptors["render.visibility_ema"].shape_suffix == (1,)
    assert descriptors["render.max_radii2D"].checkpoint is False
    assert descriptors["provenance.gaussian_uid"].checkpoint is True


def test_build_row_state_registry_rejects_missing_required_buffer():
    model = _cpu_model()
    model.get_xyz = model.params["means"]
    model._reject_legacy_population_writer = lambda _writer: None
    del model.gaussian_uid

    with pytest.raises(ValueError, match="provenance.gaussian_uid.*missing"):
        GsplatGaussianModel.build_row_state_registry(model)


def test_build_row_state_registry_rejects_present_stale_compatibility_buffer():
    model = _cpu_model()
    model.get_xyz = model.params["means"]
    model.seen_count = torch.zeros(2)

    with pytest.raises(ValueError, match="compatibility.seen_count.*rows=2.*population=3"):
        GsplatGaussianModel.build_row_state_registry(model)


def test_legacy_prune_selects_registered_population_policy_source_seen_rows():
    """Shadow-only legacy pruning must retain the E0 ledger's row alignment."""

    model = _cpu_model()
    model.get_xyz = model.params["means"]
    model._reject_legacy_population_writer = lambda _writer: None
    ensure_population_policy_e0_buffers(model)
    model.population_policy_exposure_count[:] = torch.tensor([3, 5, 7])
    model.population_policy_support_count[:] = torch.tensor([1, 2, 3])
    model.population_policy_source_seen[0, 1] = True
    model.population_policy_source_seen[2, 2] = True

    GsplatGaussianModel.prune_points(model, torch.tensor([False, True, False]))
    del model.get_xyz

    registry = GsplatGaussianModel.build_row_state_registry(model)
    assert "belief.population_policy_source_seen" in registry.names
    assert model.population_policy_source_seen.shape == (2, 256)
    assert model.population_policy_source_seen[0, 1]
    assert model.population_policy_source_seen[1, 2]
    assert model.population_policy_exposure_count.tolist() == [3, 7]
    assert model.population_policy_support_count.tolist() == [1, 3]


def test_population_policy_reset_clears_counts_seen_rows_and_source_slots():
    """A reset cannot carry physical-source history into fresh hypotheses."""

    model = _cpu_model()
    model.get_xyz = model.params["means"]
    ensure_population_policy_e0_buffers(model)
    model.population_policy_exposure_count.fill_(9)
    model.population_policy_support_count.fill_(8)
    model.population_policy_source_seen.fill_(True)
    model._population_policy_source_slot_by_fingerprint["source-a"] = 0
    model._population_policy_source_identity_by_fingerprint["source-a"] = object()

    GsplatGaussianModel._reset_population_policy_e0_state(
        model, 3, device=model.params["means"].device
    )
    del model.get_xyz

    GsplatGaussianModel.build_row_state_registry(model)
    assert torch.count_nonzero(model.population_policy_exposure_count) == 0
    assert torch.count_nonzero(model.population_policy_support_count) == 0
    assert not torch.any(model.population_policy_source_seen)
    assert model._population_policy_source_slot_by_fingerprint == {}
    assert model._population_policy_source_identity_by_fingerprint == {}


def test_build_row_state_registry_rejects_stale_optimizer_momentum_rows():
    model = _cpu_model()
    model.optimizers["means"].state[model.params["means"]] = {"exp_avg": torch.zeros(2, 3)}

    with pytest.raises(ValueError, match="optimizer.means.exp_avg.*rows=2.*population=3"):
        GsplatGaussianModel.build_row_state_registry(model)


def test_build_row_state_registry_registers_live_optimizer_state_tensors_only():
    model = _cpu_model()
    model.optimizers["means"].state[model.params["means"]] = {
        "step": torch.tensor(1),
        "exp_avg": torch.zeros(3, 3),
        "exp_avg_sq": torch.zeros(3, 3),
    }

    descriptors = _descriptor_map(GsplatGaussianModel.build_row_state_registry(model))

    assert {"optimizer.means.exp_avg", "optimizer.means.exp_avg_sq"} <= descriptors.keys()
    assert descriptors["optimizer.means.exp_avg"].family == "optimizer"
    assert descriptors["optimizer.means.exp_avg"].shape_suffix == (3,)
    assert descriptors["optimizer.means.exp_avg"].checkpoint is True
    assert descriptors["optimizer.means.exp_avg"].default == "zero_moments"
    assert "optimizer.means.step" not in descriptors


def test_build_row_state_registry_rejects_unregistered_live_tensor_buffer():
    model = _cpu_model()
    model.unknown_row_state = torch.zeros(3, 2)

    with pytest.raises(ValueError, match="unregistered live row-aligned tensor.*unknown_row_state"):
        GsplatGaussianModel.build_row_state_registry(model)


def test_build_row_state_registry_ignores_non_row_scalar_and_non_tensor_attributes():
    model = _cpu_model()
    model.scalar_diagnostic = torch.tensor(1)
    model.python_diagnostic = {"population": 3}

    GsplatGaussianModel.build_row_state_registry(model)


def test_build_row_state_registry_reports_lineage_and_compatibility_lifecycle_metadata():
    descriptors = _descriptor_map(GsplatGaussianModel.build_row_state_registry(_cpu_model()))

    assert descriptors["provenance.gaussian_uid"].default == "fresh_monotonic_uid"
    assert descriptors["provenance.gaussian_uid"].relocate == "mint_fresh_hypothesis_uid"
    assert descriptors["provenance.parent_uid"].default == "no_parent_sentinel"
    assert descriptors["provenance.parent_uid"].grow == "donor_gaussian_uid"
    assert descriptors["provenance.root_uid"].grow == "donor_root_uid"
    assert descriptors["provenance.generation"].relocate == "donor_generation_plus_one"

    model = _cpu_model()
    model.consensus_last_voter_center = torch.full((3, 3), float("nan"))
    descriptors = _descriptor_map(GsplatGaussianModel.build_row_state_registry(model))
    assert descriptors["compatibility.submap_id"].default == "current_submap_id"
    assert descriptors["compatibility.consensus_last_voter_center"].default == "nan_vector"


def test_build_row_state_registry_has_concrete_metadata_for_every_canonical_buffer():
    descriptors = _descriptor_map(GsplatGaussianModel.build_row_state_registry(_cpu_model()))

    generic_defaults = {"streaming_default", "belief_schema_default", "surface_default"}
    for name in _REQUIRED_BUFFER_NAMES:
        descriptor = next(item for item in descriptors.values() if item.name.endswith(f".{name}"))
        assert descriptor.default not in generic_defaults
        assert descriptor.create not in {"birth_default", "belief_schema_default", "surface_default"}
        assert descriptor.grow not in {"birth_default", "belief_schema_default", "surface_default"}
        assert descriptor.relocate not in {"rebirth_default", "belief_schema_policy", "surface_policy"}


def test_build_row_state_registry_covers_active_optional_compatibility_inventory():
    model = _cpu_model()
    for name in _OPTIONAL_COMPATIBILITY_NAMES:
        shape = (3, 3) if name == "consensus_last_voter_center" else (3,)
        value = torch.full(shape, float("nan")) if name == "consensus_last_voter_center" else torch.zeros(shape)
        setattr(model, name, value)

    descriptors = _descriptor_map(GsplatGaussianModel.build_row_state_registry(model))

    assert {f"compatibility.{name}" for name in _OPTIONAL_COMPATIBILITY_NAMES} <= descriptors.keys()
    assert descriptors["compatibility.consensus_last_seen_uid"].default == "sentinel_minus_one"
    assert descriptors["compatibility.consensus_last_voter_center"].default == "nan_vector"


def test_build_row_state_registry_fails_when_canonical_table_omits_required_field(monkeypatch):
    table = dict(GsplatGaussianModel._ROW_STATE_BUFFER_DESCRIPTOR_TABLE)
    del table["birth_frame"]
    monkeypatch.setattr(GsplatGaussianModel, "_ROW_STATE_BUFFER_DESCRIPTOR_TABLE", table)

    with pytest.raises(ValueError, match="canonical descriptor table missing required field.*birth_frame"):
        GsplatGaussianModel.build_row_state_registry(_cpu_model())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GsplatGaussianModel initializes row buffers on CUDA")
def test_build_row_state_registry_reads_real_initialized_gsplat_model():
    model = GsplatGaussianModel(sh_degree=0)
    model.params = torch.nn.ParameterDict({
        name: torch.nn.Parameter(torch.zeros((2, *shape), device="cuda"))
        for name, shape in _PARAMETER_SHAPES.items()
    })
    model.optimizers = {
        name: torch.optim.Adam([param]) for name, param in model.params.items()
    }
    model._reset_running_state()

    descriptors = _descriptor_map(model.build_row_state_registry())

    assert descriptors["render.visibility_ema"].checkpoint is True
    assert descriptors["provenance.gaussian_uid"].shape_suffix == ()
