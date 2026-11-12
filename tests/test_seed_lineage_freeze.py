from __future__ import annotations

import copy
from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest
import torch


class _Model:
    def __init__(self, *, amsgrad: bool = False):
        self.params = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(
                    torch.tensor(
                        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]]
                    )
                ),
                "quats": torch.nn.Parameter(
                    torch.tensor(
                        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0],
                         [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
                    )
                ),
                "opacities": torch.nn.Parameter(torch.tensor([[-2.0], [-1.0], [0.0], [1.0]])),
                "scales": torch.nn.Parameter(torch.full((4, 3), -3.0)),
            }
        )
        self.optimizers = {
            name: torch.optim.Adam([param], lr=0.1, amsgrad=amsgrad)
            for name, param in self.params.items()
        }
        self.gaussian_uid = torch.tensor([10, 11, 20, 21], dtype=torch.int64)
        self.parent_uid = torch.tensor([-1, -1, -1, 10], dtype=torch.int64)
        self.root_uid = torch.tensor([10, 11, 20, 10], dtype=torch.int64)
        self.generation = torch.tensor([0, 0, 0, 1], dtype=torch.int32)
        self.birth_method = torch.tensor([0, 0, 1, 2], dtype=torch.int8)
        self.last_reseed_iter = torch.full((4,), -1, dtype=torch.int32)
        self.birth_iter = torch.tensor([0, 0, 1, 2], dtype=torch.int32)
        self._seed_uid_range = (10, 12)

    @property
    def get_xyz(self):
        return self.params["means"]

    @property
    def all_learnable_params(self):
        return list(self.params.values())

    @property
    def means_param(self):
        return self.params["means"]

    @property
    def raw_opacities(self):
        return self.params["opacities"]

    @property
    def _scaling(self):
        return self.params["scales"]

    def normalize_rotation_params(self, mask=None):
        with torch.no_grad():
            if mask is None:
                self.params["quats"].data = torch.nn.functional.normalize(
                    self.params["quats"].data, dim=-1
                )
            else:
                self.params["quats"].data[mask] = torch.nn.functional.normalize(
                    self.params["quats"].data[mask], dim=-1
                )


class _ReaderProbeOptimizer:
    def __init__(self, model):
        self.model = model

    def step(self, visibility=None):
        del visibility
        self.model.params["means"].data[0].add_(100.0)
        self.model.start_output_reader()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.model.optimizers.values():
            optimizer.zero_grad(set_to_none=set_to_none)


class _ReaderProbeModel(_Model):
    def __init__(self):
        super().__init__()
        self.params["sh0"] = torch.nn.Parameter(torch.zeros(4, 1, 3))
        self.optimizers["sh0"] = torch.optim.Adam([self.params["sh0"]], lr=0.1)
        self.optimizer = _ReaderProbeOptimizer(self)
        self._reader_backend = None
        self._reader_started = Event()
        self._reader_done = Event()
        self._reader_thread = None
        self.reader_observed = None
        self.reader_completed_during_transaction = None

    def arm_output_reader(self, backend):
        self._reader_backend = backend
        self._reader_started.clear()
        self._reader_done.clear()
        self.reader_observed = None
        self.reader_completed_during_transaction = None

    def start_output_reader(self):
        def _read():
            self._reader_started.set()
            sample = self._reader_backend.sample_output(
                max_gaussians=4, parameter_version=0
            )
            self.reader_observed = sample.positions[0].clone()
            self._reader_done.set()

        self._reader_thread = Thread(target=_read, daemon=True)
        self._reader_thread.start()

    def normalize_rotation_params(self, mask=None):
        assert self._reader_started.wait(timeout=1.0)
        self.reader_completed_during_transaction = self._reader_done.wait(timeout=0.2)
        super().normalize_rotation_params(mask=mask)

    def finish_output_reader(self):
        self._reader_thread.join(timeout=1.0)
        assert not self._reader_thread.is_alive()


def _args(*, from_iter: int = 3, layout: str = "gsplat", strategy: str = "gsplat_energy_mcmc"):
    return SimpleNamespace(
        streaming_seed_lineage_freeze_from_iter=from_iter,
        model_layout=layout,
        densification_strategy=strategy,
    )


def _prime_adam(model: _Model) -> None:
    for name, param in model.params.items():
        param.grad = torch.full_like(param, 0.25)
        model.optimizers[name].step()
        model.optimizers[name].zero_grad(set_to_none=True)


def test_default_off_is_behavior_neutral_even_without_lineage():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    control = copy.deepcopy(model)
    del model._seed_uid_range
    del control._seed_uid_range
    args = _args(from_iter=-1, layout="legacy", strategy="external")
    freeze = SeedLineageFreeze(model, args)
    model.params["means"].grad = torch.ones_like(model.params["means"])
    control.params["means"].grad = torch.ones_like(control.params["means"])
    with freeze.optimizer_guard(iteration=100):
        model.optimizers["means"].step()
    control.optimizers["means"].step()
    assert torch.equal(model.params["means"], control.params["means"])
    for key in ("exp_avg", "exp_avg_sq"):
        assert torch.equal(
            model.optimizers["means"].state[model.params["means"]][key],
            control.optimizers["means"].state[control.params["means"]][key],
        )
    assert freeze.report()["active"] is False


def test_activation_threshold_uses_exact_living_seed_lineage():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=3))
    assert freeze.active(2) is False
    assert freeze.active(3) is True
    assert freeze.report()["current_seed_lineage_count"] == 3
    assert freeze.report()["activation_iteration"] == 3


def test_causal_fork_treatment_captures_checkpoint_k_seed_bytes_immediately():
    from eligsir.checkpointing.exact_refinement_resume import (
        activate_causal_fork_treatment,
        build_causal_fork_contract,
    )
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=3))
    checkpoint_k_seed = model.params["means"].detach().clone()[[0, 1, 3]]
    contract = build_causal_fork_contract(
        checkpoint_iteration=3,
        seed_lineage_freeze_from_iter=-1,
        fork_boundary_flush={
            "iteration": 3,
            "pending_items": 0,
            "flushed_items": 0,
            "inserted_count": 0,
            "work_id": 0,
            "optimizer_step_id": 3,
            "mutation_event_id": 0,
        },
    )
    resumed_loop_iteration = 4

    activate_causal_fork_treatment(
        freeze,
        causal_fork_contract=contract,
        resume_seed_lineage_freeze_from_iter=3,
    )
    model.params["means"].data[[0, 1, 3]].add_(100.0)
    freeze.assert_integrity(iteration=3, context="causal-fork-test")

    assert torch.equal(model.params["means"].detach()[[0, 1, 3]], checkpoint_k_seed)
    assert freeze.report()["activation_iteration"] == 3
    assert freeze.report()["activation_iteration"] != resumed_loop_iteration


def test_optimizer_guard_zeros_adam_amsgrad_and_restores_seed_rows_only():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    frozen = _Model(amsgrad=True)
    control = copy.deepcopy(frozen)
    _prime_adam(frozen)
    _prime_adam(control)
    freeze = SeedLineageFreeze(frozen, _args(from_iter=0))
    seed = torch.tensor([True, True, False, True])

    for name in frozen.params:
        grad = torch.arange(frozen.params[name].numel(), dtype=torch.float32).reshape_as(
            frozen.params[name]
        ) + 1.0
        frozen.params[name].grad = grad.clone()
        control.params[name].grad = grad.clone()
    before = {name: param.detach().clone() for name, param in frozen.params.items()}

    with freeze.optimizer_guard(iteration=0):
        for optimizer in frozen.optimizers.values():
            optimizer.step()
    for optimizer in control.optimizers.values():
        optimizer.step()

    for name, param in frozen.params.items():
        assert torch.equal(param.detach()[seed], before[name][seed])
        assert torch.equal(param.detach()[~seed], control.params[name].detach()[~seed])
        state = frozen.optimizers[name].state[param]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            assert torch.count_nonzero(state[key][seed]) == 0


def test_optimizer_guard_restores_seed_quaternions_after_normalization():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    seed = torch.tensor([True, True, False, True])
    before = model.params["quats"].detach().clone()

    with freeze.optimizer_guard(iteration=0):
        with torch.no_grad():
            model.params["quats"].data = torch.nn.functional.normalize(
                model.params["quats"].data, dim=-1
            )

    assert torch.equal(model.params["quats"].detach()[seed], before[seed])
    assert torch.equal(
        model.params["quats"].detach()[~seed],
        torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    )


def test_mutation_filters_exclude_each_seed_role_and_zero_full_n_noise():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    all_rows = torch.ones(4, dtype=torch.bool)
    for role in (
        "relocation_target", "relocation_donor", "growth_parent", "reseed",
        "prune", "recycle", "decay",
    ):
        filtered = freeze.exclude_mask(all_rows, iteration=0, role=role)
        assert torch.equal(filtered, torch.tensor([False, False, True, False]))

    full_n_noise = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 1
    filtered_noise = freeze.zero_position_noise(full_n_noise, iteration=0)
    assert torch.count_nonzero(filtered_noise[[0, 1, 3]]) == 0
    assert torch.equal(filtered_noise[2], full_n_noise[2])
    report = freeze.report()
    assert report["excluded_counts"]["noise"] == 3
    assert report["excluded_counts"]["growth_parent"] == 3


def test_row_reindex_recomputes_mask_and_preserves_uid_hash():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    assert freeze.active(0)
    first_hash = freeze.report()["stable_seed_uid_hash"]
    order = torch.tensor([3, 2, 0, 1])
    for name in list(model.params):
        model.params[name] = torch.nn.Parameter(model.params[name].detach()[order].clone())
        model.optimizers[name] = torch.optim.Adam([model.params[name]], lr=0.1)
    for name in (
        "gaussian_uid", "parent_uid", "root_uid", "generation",
        "birth_method", "last_reseed_iter", "birth_iter",
    ):
        setattr(model, name, getattr(model, name)[order])

    filtered = freeze.exclude_mask(torch.ones(4, dtype=torch.bool), iteration=1, role="prune")
    assert torch.equal(filtered, torch.tensor([False, True, False, False]))
    assert freeze.report()["stable_seed_uid_hash"] == first_hash


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda model, args: setattr(args, "model_layout", "legacy"), "gsplat"),
        (lambda model, args: setattr(args, "densification_strategy", "external"), "strategy"),
        (lambda model, args: delattr(model, "_seed_uid_range"), "UNDETERMINED"),
    ],
)
def test_active_freeze_fails_closed_on_unsupported_contract(mutation, match):
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    args = _args(from_iter=0)
    mutation(model, args)
    with pytest.raises((ValueError, RuntimeError), match=match):
        SeedLineageFreeze(model, args).assert_integrity(iteration=0, context="test")


def test_unknown_row_aligned_optimizer_state_and_nonfinite_seed_fail_closed():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    _prime_adam(model)
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    param = model.params["means"]
    model.optimizers["means"].state[param]["mystery"] = torch.ones_like(param)
    param.grad = torch.ones_like(param)
    with pytest.raises(RuntimeError, match="unknown row-aligned optimizer state"):
        with freeze.optimizer_guard(iteration=0):
            pass
    assert freeze.report()["integrity_valid"] is False

    model = _Model()
    model.params["means"].data[0, 0] = float("nan")
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    with pytest.raises(RuntimeError, match="non-finite"):
        freeze.assert_integrity(iteration=0, context="test")


def test_default_energy_mcmc_refuses_seed_relocation_targets_and_donors(monkeypatch):
    gsplat_ops = pytest.importorskip("gsplat.strategy.ops")
    from eligsir.utils.strategies.mcmc.proposal import relocate_energy_guided

    model = _Model()
    args = _args(from_iter=0)
    before = {name: param.detach().clone() for name, param in model.params.items()}
    proposal_counts = []
    original_sample = gsplat_ops._multinomial_sample

    def _recording_sample(weights, n, replacement):
        proposal_counts.append(int(n))
        return original_sample(weights, n, replacement)

    monkeypatch.setattr(gsplat_ops, "_multinomial_sample", _recording_sample)
    relocated = relocate_energy_guided(
        model,
        args,
        dead_mask=torch.tensor([True, True, False, True]),
        utility=None,
        temperature=1.0,
        binoms=torch.ones(2),
        min_opacity=0.001,
        iteration=0,
    )

    assert relocated == 0
    for name in before:
        assert torch.equal(model.params[name].detach(), before[name])
    report = model._seed_lineage_freeze.report()
    assert report["excluded_counts"]["relocation_target"] == 3
    assert report["excluded_counts"]["reseed"] == 3
    assert report["excluded_counts"]["relocation_donor"] == 3
    assert proposal_counts == [3]


def test_default_energy_mcmc_refuses_seed_growth_parents_without_changing_rng(monkeypatch):
    gsplat_ops = pytest.importorskip("gsplat.strategy.ops")
    from eligsir.utils.strategies.mcmc.proposal import add_energy_guided

    model = _Model()
    model.root_uid[:] = torch.tensor([10, 11, 10, 11])
    args = _args(from_iter=0)
    weights = torch.sigmoid(model.params["opacities"].detach().flatten())
    original_sample = gsplat_ops._multinomial_sample
    proposal_counts = []

    def _recording_sample(sample_weights, n, replacement):
        proposal_counts.append(int(n))
        return original_sample(sample_weights, n, replacement)

    torch.manual_seed(716)
    original_sample(weights, 4, replacement=True)
    expected_next = torch.rand(())

    monkeypatch.setattr(gsplat_ops, "_multinomial_sample", _recording_sample)
    torch.manual_seed(716)
    added = add_energy_guided(
        model,
        args,
        utility=None,
        temperature=1.0,
        binoms=torch.ones(2),
        min_opacity=0.001,
        cap_max=8,
        growth_factor=2.0,
        iteration=0,
    )
    realized_next = torch.rand(())

    assert added == 0
    assert model.params["means"].shape[0] == 4
    assert model._seed_lineage_freeze.report()["excluded_counts"]["growth_parent"] == 4
    assert proposal_counts == [4]
    assert torch.equal(realized_next, expected_next)


def test_full_n_noise_draw_occurs_before_seed_mask_without_extra_rng_draws():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    torch.manual_seed(123)
    expected_noise = torch.randn(4, 3)
    expected_next = torch.rand(())

    torch.manual_seed(123)
    realized_noise = torch.randn(4, 3)
    masked = freeze.zero_position_noise(realized_noise, iteration=0)
    realized_next = torch.rand(())

    assert torch.equal(realized_noise, expected_noise)
    assert torch.equal(masked[2], expected_noise[2])
    assert torch.equal(realized_next, expected_next)


def test_activation_baseline_survives_later_mutation_and_final_report_fails_closed():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    activation_means = model.params["means"].detach().clone()

    with torch.no_grad():
        model.params["means"][0] = torch.tensor([90.0, 91.0, 92.0])

    with pytest.raises(RuntimeError, match="activation baseline"):
        freeze.report()
    assert torch.equal(model.params["means"].detach()[0], activation_means[0])
    assert freeze._integrity_valid is False


def test_uid_keyed_baseline_restores_correct_values_after_same_n_reorder_and_rebind():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model(amsgrad=True)
    _prime_adam(model)
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    activation_by_uid = {
        int(uid): model.params["means"].detach()[row].clone()
        for row, uid in enumerate(model.gaussian_uid)
    }
    order = torch.tensor([3, 2, 0, 1])
    for name in list(model.params):
        model.params[name] = torch.nn.Parameter(model.params[name].detach()[order].clone())
        model.optimizers[name] = torch.optim.Adam([model.params[name]], lr=0.1, amsgrad=True)
    for name in (
        "gaussian_uid", "parent_uid", "root_uid", "generation",
        "birth_method", "last_reseed_iter", "birth_iter",
    ):
        setattr(model, name, getattr(model, name)[order])
    with torch.no_grad():
        model.params["means"][0].add_(100.0)  # uid=21, frozen seed descendant

    with freeze.optimizer_guard(iteration=1):
        pass

    for row, uid in enumerate(model.gaussian_uid):
        if int(uid) in (10, 11, 21):
            assert torch.equal(
                model.params["means"].detach()[row], activation_by_uid[int(uid)]
            )


def test_same_n_reorder_and_parameter_rebind_inside_guard_restores_by_uid():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    activation_by_uid = {
        int(uid): model.params["means"].detach()[row].clone()
        for row, uid in enumerate(model.gaussian_uid)
    }
    order = torch.tensor([3, 2, 0, 1])

    with freeze.optimizer_guard(iteration=1):
        for name in list(model.params):
            model.params[name] = torch.nn.Parameter(
                model.params[name].detach()[order].clone()
            )
            model.optimizers[name] = torch.optim.Adam([model.params[name]], lr=0.1)
        for name in (
            "gaussian_uid", "parent_uid", "root_uid", "generation",
            "birth_method", "last_reseed_iter", "birth_iter",
        ):
            setattr(model, name, getattr(model, name)[order])
        model.params["means"].data[0].add_(100.0)

    for row, uid in enumerate(model.gaussian_uid):
        if int(uid) in (10, 11, 21):
            assert torch.equal(
                model.params["means"].detach()[row], activation_by_uid[int(uid)]
            )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda m: setattr(m, "_seed_uid_range", (12, 10)), "seed UID range"),
        (lambda m: setattr(m, "_seed_uid_range", (10, 10)), "seed UID range"),
        (lambda m: m.gaussian_uid.__setitem__(1, 10), "duplicate"),
        (lambda m: delattr(m, "birth_iter"), "birth_iter"),
    ],
)
def test_activation_rejects_invalid_range_duplicate_uid_and_missing_lineage(mutation, match):
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    mutation(model)
    with pytest.raises(RuntimeError, match=match):
        SeedLineageFreeze(model, _args(from_iter=0)).assert_integrity(
            iteration=0, context="activation"
        )


@pytest.mark.parametrize("field", ["_seed_uid_range", "root_uid", "birth_method"])
def test_active_freeze_rejects_range_root_or_cohort_remap(field):
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    if field == "_seed_uid_range":
        model._seed_uid_range = (10, 13)
    elif field == "root_uid":
        model.root_uid[0] = 20
    else:
        model.birth_method[0] = 2
    with pytest.raises(RuntimeError, match="range|lineage|baseline"):
        freeze.assert_integrity(iteration=1, context="remap")


def test_optimizer_exception_restores_activation_params_and_zeroes_post_state():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model(amsgrad=True)
    _prime_adam(model)
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    before = {name: p.detach().clone() for name, p in model.params.items()}
    for param in model.params.values():
        param.grad = torch.ones_like(param)

    with pytest.raises(ValueError, match="boom"):
        with freeze.optimizer_guard(iteration=1):
            for optimizer in model.optimizers.values():
                optimizer.step()
            raise ValueError("boom")

    seed = torch.tensor([True, True, False, True])
    for name, param in model.params.items():
        assert torch.equal(param.detach()[seed], before[name][seed])
        state = model.optimizers[name].state[param]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            assert torch.count_nonzero(state[key][seed]) == 0
    report = freeze.report()
    assert report["guarded_optimizer_steps"] == 1
    assert report["optimizer_post_state_verified"] is True


def test_optimizer_guard_rejects_unknown_row_state_added_during_step_after_restoring():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    _prime_adam(model)
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    before = model.params["means"].detach().clone()

    with pytest.raises(RuntimeError, match="unknown row-aligned optimizer state"):
        with freeze.optimizer_guard(iteration=0):
            param = model.params["means"]
            param.data.add_(3.0)
            model.optimizers["means"].state[param]["late_state"] = torch.ones_like(param)
            quats = model.params["quats"]
            model.optimizers["quats"].state[quats]["exp_avg"][[0, 1, 3]] = 5.0

    assert torch.equal(model.params["means"].detach()[[0, 1, 3]], before[[0, 1, 3]])
    assert torch.count_nonzero(
        model.optimizers["quats"].state[model.params["quats"]]["exp_avg"][[0, 1, 3]]
    ) == 0


def test_sanitizer_fails_before_correcting_a_frozen_activation_row():
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_sampled,
    )
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    model.params["scales"].data[0, 0] = 16.0

    with pytest.raises(RuntimeError, match="frozen seed.*sanitize"):
        _sanitize_gaussian_params_sampled(
            model,
            "test",
            1,
            protected_mask=freeze.current_mask(iteration=1),
            burn_in_iters=-1,
        )
    assert model.params["scales"].detach()[0, 0] == 16.0


def test_sanitizer_default_off_scaling_clamp_stays_in_place(monkeypatch):
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_branch_free,
    )

    model = _Model()
    model.params["scales"].data[0, 0] = 16.0
    storage_ptr = model.params["scales"].data_ptr()

    def _forbid_out_of_place_clamp(*_args, **_kwargs):
        raise AssertionError("default-off sanitizer allocated an out-of-place clamp")

    monkeypatch.setattr(torch, "clamp", _forbid_out_of_place_clamp)
    _sanitize_gaussian_params_branch_free(model, protected_mask=None)

    assert model.params["scales"].data_ptr() == storage_ptr
    assert model.params["scales"].detach()[0, 0] == 15.0


def test_sanitizer_protected_branch_masks_only_protected_rows():
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_branch_free,
    )

    model = _Model()
    model.params["scales"].data[[0, 2], 0] = 16.0
    _sanitize_gaussian_params_branch_free(
        model,
        protected_mask=torch.tensor([True, True, False, True]),
    )

    assert model.params["scales"].detach()[0, 0] == 16.0
    assert model.params["scales"].detach()[2, 0] == 15.0


def test_default_off_helpers_do_not_attach_controller_or_emit_report_state():
    from eligsir.utils.seed_lineage_freeze import get_seed_lineage_freeze

    model = _Model()
    controller = get_seed_lineage_freeze(model, _args(from_iter=-1))
    assert controller is None
    assert not hasattr(model, "_seed_lineage_freeze")


def test_hashing_occurs_only_at_activation_and_explicit_final_report(monkeypatch):
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    calls = []
    original = SeedLineageFreeze._uid_hash

    def _count_hash(uids):
        calls.append(1)
        return original(uids)

    monkeypatch.setattr(SeedLineageFreeze, "_uid_hash", staticmethod(_count_hash))
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    for iteration in range(1, 4):
        freeze.exclude_mask(
            torch.ones(4, dtype=torch.bool), iteration=iteration, role="prune"
        )
    freeze.report()
    assert len(calls) == 2


def test_active_hot_helpers_and_protected_sanitize_do_not_extract_device_scalars(monkeypatch):
    from eligsir.mapping.integration.loss_diagnostics import (
        _sanitize_gaussian_params_sampled,
    )
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model(amsgrad=True)
    _prime_adam(model)
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    protected = freeze.current_mask(iteration=0)
    for param in model.params.values():
        param.grad = torch.ones_like(param)

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("hot path extracted a device scalar")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", _forbidden)
        patch.setattr(torch.Tensor, "cpu", _forbidden)
        patch.setattr(torch.Tensor, "__bool__", _forbidden)
        patch.setattr(torch, "equal", _forbidden)
        freeze.exclude_mask(
            torch.ones(4, dtype=torch.bool), iteration=1, role="hot_mask"
        )
        freeze.filter_indices(
            torch.arange(4), iteration=1, role="hot_indices"
        )
        freeze.zero_position_noise(torch.ones(4, 3), iteration=1)
        token = freeze.prepare_optimizer_step(iteration=1)
        freeze.restore_optimizer_step(token, iteration=1)
        freeze.assert_integrity(iteration=1, context="hot-integrity")
        _sanitize_gaussian_params_sampled(
            model,
            "hot-sanitize",
            1,
            diag_interval=999,
            burn_in_iters=-1,
            protected_mask=protected,
        )

    report = freeze.report()
    assert report["integrity_valid"] is True
    assert report["excluded_counts"]["hot_mask"] == 3
    assert report["excluded_counts"]["hot_indices"] == 3


def test_active_freeze_rejects_global_gaussian_pose_correction():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    args = _args(from_iter=0)
    args.loop_closure_apply_correction = True
    with pytest.raises(RuntimeError, match="pose correction"):
        SeedLineageFreeze(model, args).validate_configuration()


def test_real_depth_lock_restore_cannot_replace_activation_values():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze
    from eligsir.utils.streaming_depth_lock import restore_depth_locked_centers

    model = _Model()
    model.anchor_xyz = model.params["means"].detach().clone()
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    activation = model.params["means"].detach().clone()
    model.anchor_xyz[[0, 2]] += 50.0

    with freeze.optimizer_guard(iteration=1):
        restore_depth_locked_centers(
            model, SimpleNamespace(), mask=torch.tensor([True, False, True, False])
        )

    assert torch.equal(model.params["means"].detach()[0], activation[0])
    assert torch.equal(model.params["means"].detach()[2], model.anchor_xyz[2])


def test_lifecycle_v2_demotion_filter_blocks_frozen_opacity_reset_only():
    from eligsir.utils.seed_lineage_freeze import exclude_frozen_seed_mask

    model = _Model()
    args = _args(from_iter=0)
    before = model.raw_opacities.detach().clone()
    demote = exclude_frozen_seed_mask(
        model,
        args,
        torch.ones(4, dtype=torch.bool),
        iteration=0,
        role="lifecycle_demote",
    )
    with torch.no_grad():
        model.raw_opacities[demote] = -9.0

    assert torch.equal(model.raw_opacities.detach()[[0, 1, 3]], before[[0, 1, 3]])
    assert model.raw_opacities.detach()[2].item() == -9.0


def test_real_selective_adam_state_is_zeroed_and_verified_without_kernel_step():
    SelectiveAdam = pytest.importorskip("gsplat.optimizers").SelectiveAdam
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    param = model.params["means"]
    optimizer = SelectiveAdam([param], eps=1e-15, betas=(0.9, 0.999))
    optimizer.state[param]["exp_avg"] = torch.ones_like(param)
    optimizer.state[param]["exp_avg_sq"] = torch.ones_like(param)
    model.optimizers["means"] = optimizer
    freeze = SeedLineageFreeze(model, _args(from_iter=0))

    with freeze.optimizer_guard(iteration=0):
        optimizer.state[param]["exp_avg"][[0, 1, 3]] = 7.0
        optimizer.state[param]["exp_avg_sq"][[0, 1, 3]] = 8.0

    assert torch.count_nonzero(optimizer.state[param]["exp_avg"][[0, 1, 3]]) == 0
    assert torch.count_nonzero(optimizer.state[param]["exp_avg_sq"][[0, 1, 3]]) == 0
    assert freeze.report()["optimizer_post_state_verified"] is True


def test_engine_exception_restores_activation_baseline_and_optimizer_state():
    from eligsir.optimization.gaussian.contracts import GradientPolicyContext
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    model = _Model()
    _prime_adam(model)
    args = _args(from_iter=0)
    before = model.params["means"].detach().clone()
    for param in model.params.values():
        param.grad = torch.ones_like(param)

    class _FailSession:
        def step(self, visibility=None):
            for optimizer in model.optimizers.values():
                optimizer.step()
            raise ValueError("engine boom")

        def zero_grad(self):
            pass

    class _Backend:
        def __init__(self):
            self.state_lock = RLock()
            model._eligsir_map_backend = self

        def begin_optimization(self):
            return _FailSession()

    engine = GaussianOptimizationEngine(_Backend())
    context = GradientPolicyContext(
        gaussians=model, args=args, iteration=0, n_frames_ingested=0
    )
    with pytest.raises(ValueError, match="engine boom"):
        engine.run_optimizer_step(
            model,
            args,
            SimpleNamespace(gsplat_sparse_grad=False),
            context,
            optimizer_type="adam",
        )

    assert torch.equal(model.params["means"].detach()[[0, 1, 3]], before[[0, 1, 3]])
    state = model.optimizers["means"].state[model.params["means"]]
    assert torch.count_nonzero(state["exp_avg"][[0, 1, 3]]) == 0


def test_engine_seed_transaction_blocks_output_reader_until_baseline_restore():
    from eligsir.map.adapters.gsplat import GsplatMapBackend
    from eligsir.optimization.gaussian.contracts import GradientPolicyContext
    from eligsir.optimization.gaussian.engine import GaussianOptimizationEngine

    model = _ReaderProbeModel()
    backend = GsplatMapBackend(model)
    model.arm_output_reader(backend)
    baseline = model.params["means"].detach()[0].clone()
    args = _args(from_iter=0)
    context = GradientPolicyContext(
        gaussians=model,
        args=args,
        iteration=0,
        n_frames_ingested=0,
    )

    GaussianOptimizationEngine(backend).run_optimizer_step(
        model,
        args,
        SimpleNamespace(gsplat_sparse_grad=True),
        context,
        optimizer_type="adam",
    )
    model.finish_output_reader()

    assert model.reader_completed_during_transaction is False
    assert torch.equal(model.reader_observed, baseline)


def test_inline_seed_transaction_blocks_output_reader_until_baseline_restore():
    from eligsir.map.adapters.gsplat import GsplatMapBackend
    from eligsir.utils.seed_lineage_freeze import seed_lineage_optimizer_guard

    model = _ReaderProbeModel()
    backend = GsplatMapBackend(model)
    model.arm_output_reader(backend)
    baseline = model.params["means"].detach()[0].clone()

    with seed_lineage_optimizer_guard(
        model, _args(from_iter=0), backend=backend, iteration=0
    ):
        model.optimizer.step()
        model.normalize_rotation_params()
    model.finish_output_reader()

    assert model.reader_completed_during_transaction is False
    assert torch.equal(model.reader_observed, baseline)


def test_seed_transaction_default_off_does_not_inspect_backend_lock():
    from eligsir.utils.seed_lineage_freeze import seed_lineage_optimizer_guard

    class _BackendMustNotBeInspected:
        @property
        def state_lock(self):
            raise AssertionError("disabled seed transaction inspected state_lock")

    with seed_lineage_optimizer_guard(
        _Model(),
        _args(from_iter=-1),
        backend=_BackendMustNotBeInspected(),
        iteration=0,
    ):
        pass


@pytest.mark.parametrize("canonical", [False, True])
def test_active_seed_transaction_fails_closed_without_canonical_lock(canonical):
    from eligsir.utils.seed_lineage_freeze import seed_lineage_optimizer_guard

    model = _Model()
    backend = SimpleNamespace(state_lock=None)
    if canonical:
        model._eligsir_map_backend = backend

    with pytest.raises(RuntimeError, match="canonical.*state_lock"):
        with seed_lineage_optimizer_guard(
            model, _args(from_iter=0), backend=backend, iteration=0
        ):
            pass


def test_final_report_zeros_all_known_state_before_rejecting_unknown_state():
    from eligsir.utils.seed_lineage_freeze import SeedLineageFreeze

    model = _Model()
    _prime_adam(model)
    freeze = SeedLineageFreeze(model, _args(from_iter=0))
    freeze.assert_integrity(iteration=0, context="activation")
    means = model.params["means"]
    quats = model.params["quats"]
    model.optimizers["means"].state[means]["late_state"] = torch.ones_like(means)
    model.optimizers["quats"].state[quats]["exp_avg"][[0, 1, 3]] = 5.0

    with pytest.raises(RuntimeError, match="unknown row-aligned optimizer state"):
        freeze.report()
    assert torch.count_nonzero(
        model.optimizers["quats"].state[quats]["exp_avg"][[0, 1, 3]]
    ) == 0
