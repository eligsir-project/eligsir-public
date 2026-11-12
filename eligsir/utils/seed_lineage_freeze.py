"""Exact, opt-in living-seed-lineage freeze for issue #346 causal runs."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Any, Iterator

import torch

from eligsir.utils.gaussian_lineage import (
    assert_lineage_consistent,
    seed_lineage_mask,
    seed_uid_range,
)


_KNOWN_ROW_ALIGNED_ADAM_STATE = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
_REQUIRED_LINEAGE_BUFFERS = (
    "gaussian_uid",
    "parent_uid",
    "root_uid",
    "generation",
    "birth_method",
    "last_reseed_iter",
    "birth_iter",
)


class SeedLineageFreeze:
    """Protect the exact living seed lineage at activation iteration ``K``."""

    def __init__(self, gaussians: Any, args: Any):
        self.gaussians = gaussians
        self.args = args
        self.requested_iteration = int(
            getattr(args, "streaming_seed_lineage_freeze_from_iter", -1)
        )
        self.activation_iteration = None
        self._stable_uid_hash = None
        self._stable_uid_count = None
        self._seed_uid_range = None
        self._baseline_uids: torch.Tensor | None = None
        self._baseline_params: dict[str, torch.Tensor] = {}
        self._baseline_lineage: dict[str, torch.Tensor] = {}
        self._lineage_available = None
        self._active = False
        self._last_iteration = None
        self._integrity_valid = True
        self._unknown_state_violations = 0
        self._topology_violations = 0
        self._parameter_delta_violations = 0
        self._guarded_optimizer_steps = 0
        self._optimizer_post_state_verified = False
        self._per_parameter: dict[str, dict[str, int]] = {}
        self._excluded_counts: dict[str, int] = {}
        self._device_integrity_violations: int | torch.Tensor = 0
        if self.requested_iteration >= 0:
            gaussians._seed_lineage_freeze = self

    def active(self, iteration: int) -> bool:
        if self.requested_iteration < 0 or int(iteration) < self.requested_iteration:
            return False
        self._last_iteration = int(iteration)
        if self.activation_iteration is None:
            self._activate(int(iteration))
        return True

    def validate_configuration(self) -> None:
        if self.requested_iteration >= 0:
            self._validate_supported_contract()

    def validate_pose_correction(self, active: bool) -> None:
        if self.requested_iteration >= 0 and bool(active):
            self._fail(
                "seed-lineage freeze does not support Gaussian pose correction; "
                "disable loop_closure_apply_correction"
            )

    def _fail(self, message: str, *, topology: bool = False, unknown_state: bool = False):
        self._integrity_valid = False
        if topology:
            self._topology_violations += 1
        if unknown_state:
            self._unknown_state_violations += 1
        raise RuntimeError(message)

    @staticmethod
    def _add_counter(mapping: dict, key: str, value: int | torch.Tensor) -> None:
        mapping[key] = mapping.get(key, 0) + value

    def _assert_async(
        self,
        predicate: torch.Tensor,
        message: str,
        *,
        topology: bool = False,
    ) -> None:
        violation = (~predicate).to(dtype=torch.int64)
        self._device_integrity_violations = self._device_integrity_violations + violation
        if topology:
            self._topology_violations = self._topology_violations + violation
        try:
            torch._assert_async(predicate, message)
        except RuntimeError:
            self._integrity_valid = False
            raise

    @staticmethod
    def _materialize_int(value: int | torch.Tensor) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.detach().item())
        return int(value)

    @classmethod
    def _materialize_counter_mapping(cls, mapping: dict) -> dict:
        return {
            key: (
                cls._materialize_counter_mapping(value)
                if isinstance(value, dict)
                else cls._materialize_int(value)
            )
            for key, value in mapping.items()
        }

    def _validate_supported_contract(self) -> None:
        layout = str(getattr(self.args, "model_layout", "gsplat")).lower()
        if layout != "gsplat":
            self._fail(f"seed-lineage freeze requires model_layout=gsplat; got {layout!r}")
        strategy = str(
            getattr(
                self.args,
                "_seed_lineage_freeze_realized_strategy",
                getattr(self.args, "densification_strategy", "gsplat_energy_mcmc"),
            )
        ).lower()
        if strategy != "gsplat_energy_mcmc":
            self._fail(
                "seed-lineage freeze supports only the project-default "
                f"gsplat_energy_mcmc strategy; got {strategy!r}"
            )
        self.validate_pose_correction(
            bool(getattr(self.args, "loop_closure_apply_correction", False))
        )

    @staticmethod
    def _uid_hash(uids: torch.Tensor) -> str:
        ordered = sorted(int(v) for v in uids.detach().cpu().tolist())
        payload = ",".join(str(v) for v in ordered).encode("ascii")
        return hashlib.sha256(payload).hexdigest()

    def _lineage_tensors(self, *, context: str, check_unique: bool) -> tuple[torch.Tensor, ...]:
        self._validate_supported_contract()
        n = int(self.gaussians.get_xyz.shape[0])
        try:
            assert_lineage_consistent(self.gaussians, context)
        except Exception as exc:
            self._lineage_available = False
            self._fail(str(exc), topology=True)
        tensors = []
        for name in _REQUIRED_LINEAGE_BUFFERS:
            value = getattr(self.gaussians, name, None)
            if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] != n:
                self._lineage_available = False
                self._fail(
                    f"required lineage buffer {name!r} is missing or inconsistent for N={n}",
                    topology=True,
                )
            tensors.append(value)
        uid = tensors[0]
        if check_unique and int(torch.unique(uid).numel()) != n:
            self._lineage_available = False
            self._fail("duplicate Gaussian UIDs make exact seed lineage unavailable", topology=True)
        try:
            lo, hi = seed_uid_range(self.gaussians)
        except Exception as exc:
            self._lineage_available = False
            self._fail(str(exc), topology=True)
        lo, hi = int(lo), int(hi)
        if lo >= hi:
            self._lineage_available = False
            self._fail(f"invalid or empty seed UID range [{lo}, {hi})", topology=True)
        if self._seed_uid_range is not None and (lo, hi) != self._seed_uid_range:
            self._fail(
                f"seed UID range drifted from {self._seed_uid_range} to {(lo, hi)}",
                topology=True,
            )
        self._lineage_available = True
        return (*tensors, torch.tensor([lo, hi], device=uid.device, dtype=torch.int64))

    def _activate(self, iteration: int) -> None:
        *lineage, bounds = self._lineage_tensors(
            context=f"seed-freeze-activation@{iteration}", check_unique=True
        )
        try:
            mask = seed_lineage_mask(self.gaussians).detach().to(dtype=torch.bool)
        except Exception as exc:
            self._fail(str(exc), topology=True)
        n = int(self.gaussians.get_xyz.shape[0])
        if mask.ndim != 1 or mask.shape[0] != n or not bool(mask.any()):
            self._fail(
                f"exact living seed lineage is empty or inconsistent at activation: "
                f"shape={tuple(mask.shape)}, N={n}",
                topology=True,
            )
        uid = lineage[0]
        sorted_uids, order = torch.sort(uid[mask])
        self._baseline_uids = sorted_uids.detach().clone()
        self._seed_uid_range = (int(bounds[0].item()), int(bounds[1].item()))
        self._stable_uid_count = int(sorted_uids.numel())
        self._stable_uid_hash = self._uid_hash(sorted_uids)
        self._baseline_lineage = {
            name: value[mask][order].detach().clone()
            for name, value in zip(_REQUIRED_LINEAGE_BUFFERS, lineage)
        }
        for name, param in self.gaussians.params.items():
            if param.ndim > 0 and param.shape[0] == n:
                self._baseline_params[name] = param.detach()[mask][order].clone()
        self.activation_iteration = int(iteration)
        self._active = True
        self._assert_finite_baseline()

    def _current_binding(
        self, *, iteration: int, context: str, checkpoint: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation_iteration is None:
            self._activate(iteration)
        *lineage, _bounds = self._lineage_tensors(context=context, check_unique=checkpoint)
        mask = seed_lineage_mask(self.gaussians).detach().to(dtype=torch.bool)
        n = int(self.gaussians.get_xyz.shape[0])
        if mask.ndim != 1 or mask.shape[0] != n:
            self._fail("seed-lineage mask shape changed", topology=True)
        current_uids = lineage[0][mask]
        baseline_uids = self._baseline_uids
        if baseline_uids is None or current_uids.numel() != baseline_uids.numel():
            self._fail("seed-lineage UID set changed after activation", topology=True)
        positions = torch.searchsorted(baseline_uids, current_uids)
        in_range = positions < baseline_uids.numel()
        safe_positions = positions.clamp(max=baseline_uids.numel() - 1)
        exact_uid_match = in_range & (baseline_uids[safe_positions] == current_uids)
        self._assert_async(
            exact_uid_match.all(),
            "seed-lineage UID set changed after activation",
            topology=True,
        )
        complete_once = (
            torch.bincount(safe_positions, minlength=baseline_uids.numel()) == 1
        ).all()
        self._assert_async(
            exact_uid_match.all() & complete_once,
            "duplicate or missing frozen Gaussian UID after activation",
            topology=True,
        )
        for name, value in zip(_REQUIRED_LINEAGE_BUFFERS, lineage):
            lineage_matches = (
                value[mask] == self._baseline_lineage[name][safe_positions]
            ).all()
            self._assert_async(
                exact_uid_match.all() & lineage_matches,
                f"seed-lineage metadata remap detected in {name!r} at {context}",
                topology=True,
            )
        if checkpoint:
            if self._uid_hash(current_uids) != self._stable_uid_hash:
                self._fail("seed-lineage UID hash changed after activation", topology=True)
        return mask, safe_positions

    def current_mask(self, *, iteration: int) -> torch.Tensor:
        if not self.active(iteration):
            return torch.zeros(
                int(self.gaussians.get_xyz.shape[0]),
                dtype=torch.bool,
                device=self.gaussians.get_xyz.device,
            )
        mask, _ = self._current_binding(
            iteration=int(iteration), context=f"seed-freeze-mask@{iteration}"
        )
        return mask

    def _assert_finite_baseline(self) -> None:
        for name, values in self._baseline_params.items():
            if not bool(torch.isfinite(values).all()):
                self._fail(f"seed-lineage activation baseline contains non-finite {name!r}")

    def _restore_activation_params(
        self, mask: torch.Tensor, positions: torch.Tensor, *, context: str, fail_on_delta: bool
    ) -> None:
        n = int(mask.shape[0])
        current_names = {
            name
            for name, param in self.gaussians.params.items()
            if param.ndim > 0 and param.shape[0] == n
        }
        if current_names != set(self._baseline_params):
            self._fail(
                "row-aligned learnable parameter set changed after activation",
                topology=True,
            )
        with torch.no_grad():
            for name, baseline in self._baseline_params.items():
                param = self.gaussians.params[name]
                expected = baseline[positions]
                changed = (param.detach()[mask] != expected).any()
                if fail_on_delta:
                    violation = changed.to(dtype=torch.int64)
                    self._parameter_delta_violations = (
                        self._parameter_delta_violations + violation
                    )
                    self._add_counter(
                        self._parameter_counters(name),
                        "post_step_parameter_delta_violations",
                        violation,
                    )
                param.data[mask] = expected
                restored = (param.detach()[mask] == expected).all()
                self._assert_async(
                    restored,
                    f"failed to restore activation baseline for {name!r} at {context}",
                )
                if fail_on_delta:
                    self._assert_async(
                        ~changed,
                        f"seed-lineage activation baseline changed at {context}: {name}",
                    )

    def assert_integrity(
        self, *, iteration: int, context: str, checkpoint: bool = False
    ) -> None:
        if not self.active(iteration):
            return
        mask, positions = self._current_binding(
            iteration=int(iteration), context=context, checkpoint=checkpoint
        )
        self._restore_activation_params(mask, positions, context=context, fail_on_delta=True)

    def exclude_mask(self, mask: torch.Tensor, *, iteration: int, role: str) -> torch.Tensor:
        if not self.active(iteration):
            return mask
        seed, _ = self._current_binding(iteration=int(iteration), context=role)
        if mask.ndim != 1 or mask.shape[0] != seed.shape[0]:
            self._fail(
                f"seed-lineage {role} mask shape mismatch: {tuple(mask.shape)} vs {tuple(seed.shape)}",
                topology=True,
            )
        mask = mask.to(dtype=torch.bool)
        self._add_counter(self._excluded_counts, role, (mask & seed).sum())
        return mask & ~seed

    def filter_indices(self, indices: torch.Tensor, *, iteration: int, role: str) -> torch.Tensor:
        if not self.active(iteration):
            return indices
        seed, _ = self._current_binding(iteration=int(iteration), context=role)
        if indices.numel() == 0:
            return indices
        valid = (indices >= 0) & (indices < seed.shape[0])
        safe_indices = indices.clamp(min=0, max=seed.shape[0] - 1)
        self._assert_async(
            valid.all(),
            f"seed-lineage {role} indices are out of range",
            topology=True,
        )
        protected = seed[safe_indices] & valid
        keep = valid & ~protected
        self._add_counter(self._excluded_counts, role, protected.sum())
        return indices[keep]

    def zero_position_noise(self, noise: torch.Tensor, *, iteration: int) -> torch.Tensor:
        if not self.active(iteration):
            return noise
        seed, _ = self._current_binding(iteration=int(iteration), context="noise")
        if noise.ndim == 0 or noise.shape[0] != seed.shape[0]:
            self._fail("full-N position noise shape does not match exact lineage", topology=True)
        result = noise.clone()
        result[seed] = 0
        self._add_counter(self._excluded_counts, "noise", seed.sum())
        return result

    def _optimizer_for(self, name: str):
        optimizers = getattr(self.gaussians, "optimizers", None)
        if isinstance(optimizers, dict):
            return optimizers.get(name)
        return getattr(self.gaussians, "optimizer", None)

    def _parameter_counters(self, name: str) -> dict[str, int]:
        return self._per_parameter.setdefault(
            name,
            {
                "grad_zeroed_rows": 0,
                "exp_avg_zeroed_rows": 0,
                "exp_avg_sq_zeroed_rows": 0,
                "max_exp_avg_sq_zeroed_rows": 0,
                "post_step_parameter_delta_violations": 0,
            },
        )

    def _mask_optimizer_state(
        self, name: str, param: torch.nn.Parameter, mask: torch.Tensor, *, verify: bool
    ) -> list[str]:
        optimizer = self._optimizer_for(name)
        state = {} if optimizer is None else optimizer.state.get(param, {})
        counters = self._parameter_counters(name)
        unknown = []
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            if value.shape[0] != mask.shape[0]:
                continue
            if key not in _KNOWN_ROW_ALIGNED_ADAM_STATE:
                unknown.append(f"{name}.{key}")
                continue
            value[mask] = 0
            self._add_counter(counters, f"{key}_zeroed_rows", mask.sum())
            if verify:
                self._assert_async(
                    torch.count_nonzero(value[mask]) == 0,
                    f"optimizer state {key!r} remained nonzero for frozen {name!r}",
                )
        return unknown

    @contextmanager
    def optimizer_guard(self, *, iteration: int) -> Iterator[None]:
        token = self.prepare_optimizer_step(iteration=iteration)
        try:
            yield
        finally:
            self.restore_optimizer_step(token, iteration=iteration)

    def prepare_optimizer_step(self, *, iteration: int):
        if not self.active(iteration):
            return None
        seed, _ = self._current_binding(
            iteration=int(iteration), context=f"optimizer-pre@{iteration}"
        )
        unknown = []
        with torch.no_grad():
            for name, param in self.gaussians.params.items():
                if param.ndim == 0 or param.shape[0] != seed.shape[0]:
                    continue
                counters = self._parameter_counters(name)
                if param.grad is not None:
                    if getattr(param.grad, "layout", torch.strided) != torch.strided:
                        param.grad = param.grad.to_dense().contiguous()
                    param.grad[seed] = 0
                    self._add_counter(counters, "grad_zeroed_rows", seed.sum())
                unknown.extend(self._mask_optimizer_state(name, param, seed, verify=False))
        if unknown:
            self._fail(
                f"unknown row-aligned optimizer state: {', '.join(unknown)}",
                unknown_state=True,
            )
        return True

    def restore_optimizer_step(self, token, *, iteration: int) -> None:
        if token is None:
            return
        seed, positions = self._current_binding(
            iteration=int(iteration), context=f"optimizer-post@{iteration}"
        )
        unknown = []
        try:
            self._restore_activation_params(
                seed, positions, context=f"optimizer-post@{iteration}", fail_on_delta=False
            )
        finally:
            with torch.no_grad():
                for name, param in self.gaussians.params.items():
                    if param.ndim > 0 and param.shape[0] == seed.shape[0]:
                        unknown.extend(
                            self._mask_optimizer_state(name, param, seed, verify=True)
                        )
            self._guarded_optimizer_steps += 1
            self._optimizer_post_state_verified = True
        if unknown:
            self._fail(
                f"unknown row-aligned optimizer state: {', '.join(unknown)}",
                unknown_state=True,
            )

    def report(self) -> dict:
        sanitizer_violations = getattr(
            self.gaussians, "_seed_lineage_sanitize_violations", 0
        )
        device_violations = self._materialize_int(self._device_integrity_violations)
        sanitizer_violation_count = self._materialize_int(sanitizer_violations)
        healthy_before_report = (
            self._integrity_valid
            and device_violations == 0
            and sanitizer_violation_count == 0
        )
        if healthy_before_report and self._active and self._last_iteration is not None:
            mask, positions = self._current_binding(
                iteration=self._last_iteration, context="final-report", checkpoint=True
            )
            self._restore_activation_params(
                mask, positions, context="final-report", fail_on_delta=True
            )
            unknown = []
            with torch.no_grad():
                for name, param in self.gaussians.params.items():
                    if param.ndim > 0 and param.shape[0] == mask.shape[0]:
                        unknown.extend(
                            self._mask_optimizer_state(name, param, mask, verify=True)
                        )
            if unknown:
                self._fail(
                    f"unknown row-aligned optimizer state: {', '.join(unknown)}",
                    unknown_state=True,
                )
        self._integrity_valid = bool(
            self._integrity_valid
            and self._materialize_int(self._device_integrity_violations) == 0
            and sanitizer_violation_count == 0
        )
        return {
            "schema_version": 2,
            "requested_activation_iteration": self.requested_iteration,
            "activation_iteration": self.activation_iteration,
            "active": bool(self._active),
            "exact_lineage_available": self._lineage_available,
            "seed_uid_range": None if self._seed_uid_range is None else list(self._seed_uid_range),
            "current_seed_lineage_count": self._stable_uid_count,
            "stable_seed_uid_hash": self._stable_uid_hash,
            "per_parameter": self._materialize_counter_mapping(self._per_parameter),
            "excluded_counts": self._materialize_counter_mapping(self._excluded_counts),
            "guarded_optimizer_steps": self._guarded_optimizer_steps,
            "optimizer_post_state_verified": self._optimizer_post_state_verified,
            "unknown_state_violations": self._materialize_int(
                self._unknown_state_violations
            ),
            "topology_violations": self._materialize_int(self._topology_violations),
            "post_step_parameter_delta_violations": self._materialize_int(
                self._parameter_delta_violations
            ),
            "sanitize_correction_violations": sanitizer_violation_count,
            "integrity_valid": bool(self._integrity_valid),
        }


def get_seed_lineage_freeze(gaussians: Any, args: Any) -> SeedLineageFreeze | None:
    if int(getattr(args, "streaming_seed_lineage_freeze_from_iter", -1)) < 0:
        return None
    controller = getattr(gaussians, "_seed_lineage_freeze", None)
    if controller is None:
        controller = SeedLineageFreeze(gaussians, args)
    return controller


@contextmanager
def seed_lineage_optimizer_guard(
    gaussians: Any, args: Any, *, backend: Any, iteration: int
) -> Iterator[None]:
    controller = get_seed_lineage_freeze(gaussians, args)
    if controller is None:
        yield
        return
    canonical_backend = getattr(gaussians, "_eligsir_map_backend", None)
    lock = None if canonical_backend is not backend else getattr(backend, "state_lock", None)
    if lock is None:
        controller._fail(
            "seed-lineage freeze requires the canonical map backend state_lock"
        )
    with lock:
        with controller.optimizer_guard(iteration=iteration):
            yield


def exclude_frozen_seed_mask(
    gaussians: Any,
    args: Any,
    mask: torch.Tensor,
    *,
    iteration: int,
    role: str,
) -> torch.Tensor:
    controller = get_seed_lineage_freeze(gaussians, args)
    if controller is None:
        return mask
    return controller.exclude_mask(mask, iteration=iteration, role=role)


def filter_frozen_seed_indices(
    gaussians: Any,
    args: Any,
    indices: torch.Tensor,
    *,
    iteration: int,
    role: str,
) -> torch.Tensor:
    controller = get_seed_lineage_freeze(gaussians, args)
    if controller is None:
        return indices
    return controller.filter_indices(indices, iteration=iteration, role=role)


def zero_frozen_seed_position_noise(
    gaussians: Any, args: Any, noise: torch.Tensor, *, iteration: int
) -> torch.Tensor:
    controller = get_seed_lineage_freeze(gaussians, args)
    if controller is None:
        return noise
    return controller.zero_position_noise(noise, iteration=iteration)


def assert_frozen_seed_integrity(
    gaussians: Any, args: Any, *, iteration: int, context: str
) -> None:
    controller = get_seed_lineage_freeze(gaussians, args)
    if controller is not None:
        controller.assert_integrity(
            iteration=iteration, context=context, checkpoint=True
        )


def frozen_seed_mask(gaussians: Any, args: Any, *, iteration: int) -> torch.Tensor | None:
    controller = get_seed_lineage_freeze(gaussians, args)
    return None if controller is None else controller.current_mask(iteration=iteration)
