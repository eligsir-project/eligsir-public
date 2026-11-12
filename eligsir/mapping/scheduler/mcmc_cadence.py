"""Exposure-driven cadence owner for canonical MCMC structural mutations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eligsir.contracts.work_clocks import RuntimeWorkClockSnapshot
from eligsir.mapping.population_control_contracts import MutationWorkToken


def _exact_non_negative(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _exact_positive(value: object, name: str) -> int:
    value = _exact_non_negative(value, name)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class MCMCCadenceDecision:
    grow_due: bool
    relocate_due: bool
    reason: str
    exposure_debt_pixels: int
    active_grow_interval_pixels: int
    active_relocate_interval_pixels: int
    optimizer_settled: bool
    _supervised_rgb_pixels: int = field(repr=False)
    _optimizer_updates: int = field(repr=False)
    _topology_version: int = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.grow_due) is not bool or type(self.relocate_due) is not bool:
            raise TypeError("MCMC due flags must be bool")
        if type(self.optimizer_settled) is not bool:
            raise TypeError("optimizer_settled must be bool")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty")
        for name in (
            "exposure_debt_pixels",
            "active_grow_interval_pixels",
            "active_relocate_interval_pixels",
            "_supervised_rgb_pixels",
            "_optimizer_updates",
            "_topology_version",
        ):
            if name in {
                "active_grow_interval_pixels",
                "active_relocate_interval_pixels",
            }:
                _exact_positive(getattr(self, name), name)
            else:
                _exact_non_negative(getattr(self, name), name)


class MCMCCadenceOwner:
    """Own one runtime's exposure debt and structural settling fence.

    Exposure debt is cumulative and is consumed only after a successful
    structural MCMC commit. A topology epoch from any other owner resets only
    the optimizer-settling baseline; it does not erase exposure.
    """

    _SCHEMA = "mcmc_cadence_v3"
    _LEGACY_SCHEMA = "mcmc_cadence_v2"
    _LEGACY_STATE_KEYS = {
        "owner_schema",
        "grow_interval_pixels",
        "post_ingestion_grow_interval_pixels",
        "relocate_interval_pixels",
        "post_ingestion_relocate_interval_pixels",
        "min_optimizer_updates_after_structural_mutation",
        "last_committed_grow_pixels",
        "last_committed_relocate_pixels",
        "last_structural_optimizer_updates",
        "observed_supervised_rgb_pixels",
        "observed_optimizer_updates",
        "observed_topology_version",
    }
    _STATE_KEYS = {
        *_LEGACY_STATE_KEYS,
        "last_population_authorization_rgb_pixels",
        "last_population_authorization_depth_valid_samples",
    }

    def __init__(
        self,
        *,
        grow_interval_pixels: int,
        relocate_interval_pixels: int,
        min_optimizer_updates_after_structural_mutation: int,
        post_ingestion_grow_interval_pixels: int = 0,
        post_ingestion_relocate_interval_pixels: int = 0,
    ) -> None:
        self.grow_interval_pixels = _exact_positive(
            grow_interval_pixels, "grow_interval_pixels"
        )
        self.relocate_interval_pixels = _exact_positive(
            relocate_interval_pixels, "relocate_interval_pixels"
        )
        self.post_ingestion_grow_interval_pixels = _exact_non_negative(
            post_ingestion_grow_interval_pixels,
            "post_ingestion_grow_interval_pixels",
        )
        self.post_ingestion_relocate_interval_pixels = _exact_non_negative(
            post_ingestion_relocate_interval_pixels,
            "post_ingestion_relocate_interval_pixels",
        )
        self.min_optimizer_updates_after_structural_mutation = _exact_non_negative(
            min_optimizer_updates_after_structural_mutation,
            "min_optimizer_updates_after_structural_mutation",
        )
        self._last_committed_grow_pixels = 0
        self._last_committed_relocate_pixels = 0
        self._last_structural_optimizer_updates = 0
        self._observed_supervised_rgb_pixels = 0
        self._observed_optimizer_updates = 0
        self._observed_topology_version = 0
        self._last_population_authorization_rgb_pixels = 0
        self._last_population_authorization_depth_valid_samples = 0

    def decide(
        self,
        snapshot: RuntimeWorkClockSnapshot,
        *,
        topology_version: int,
        growth_allowed: bool = True,
        relocation_allowed: bool = True,
    ) -> MCMCCadenceDecision:
        if not isinstance(snapshot, RuntimeWorkClockSnapshot):
            raise TypeError("snapshot must be RuntimeWorkClockSnapshot")
        topology_version = _exact_non_negative(topology_version, "topology_version")
        if type(growth_allowed) is not bool or type(relocation_allowed) is not bool:
            raise TypeError("MCMC permission flags must be bool")
        if snapshot.supervised_rgb_pixels < self._observed_supervised_rgb_pixels:
            raise ValueError("supervised RGB pixel counter regressed")
        if snapshot.optimizer_updates < self._observed_optimizer_updates:
            raise ValueError("optimizer update counter regressed")
        if topology_version < self._observed_topology_version:
            raise ValueError("topology version regressed")

        if topology_version > self._observed_topology_version:
            self._last_structural_optimizer_updates = snapshot.optimizer_updates
        self._observed_supervised_rgb_pixels = snapshot.supervised_rgb_pixels
        self._observed_optimizer_updates = snapshot.optimizer_updates
        self._observed_topology_version = topology_version

        grow_debt = max(
            0,
            snapshot.supervised_rgb_pixels - self._last_committed_grow_pixels,
        )
        relocate_debt = max(
            0,
            snapshot.supervised_rgb_pixels - self._last_committed_relocate_pixels,
        )
        optimizer_settled = (
            snapshot.optimizer_updates - self._last_structural_optimizer_updates
            >= self.min_optimizer_updates_after_structural_mutation
        )
        active_grow_interval_pixels = self.grow_interval_pixels
        if snapshot.source_complete and self.post_ingestion_grow_interval_pixels > 0:
            active_grow_interval_pixels = self.post_ingestion_grow_interval_pixels
        active_relocate_interval_pixels = self.relocate_interval_pixels
        if (
            snapshot.source_complete
            and self.post_ingestion_relocate_interval_pixels > 0
        ):
            active_relocate_interval_pixels = (
                self.post_ingestion_relocate_interval_pixels
            )
        grow_due = bool(
            growth_allowed
            and optimizer_settled
            and grow_debt >= active_grow_interval_pixels
        )
        relocate_due = bool(
            relocation_allowed
            and optimizer_settled
            and relocate_debt >= active_relocate_interval_pixels
        )
        if not optimizer_settled:
            reason = "optimizer settling after structural mutation"
        elif not growth_allowed and grow_debt >= active_grow_interval_pixels:
            reason = "growth debt retained by quality or work-budget brake"
        elif grow_due or relocate_due:
            reason = "semantic exposure interval due"
        else:
            reason = "semantic exposure interval not due"
        return MCMCCadenceDecision(
            grow_due=grow_due,
            relocate_due=relocate_due,
            reason=reason,
            exposure_debt_pixels=max(grow_debt, relocate_debt),
            active_grow_interval_pixels=active_grow_interval_pixels,
            active_relocate_interval_pixels=active_relocate_interval_pixels,
            optimizer_settled=optimizer_settled,
            _supervised_rgb_pixels=snapshot.supervised_rgb_pixels,
            _optimizer_updates=snapshot.optimizer_updates,
            _topology_version=topology_version,
        )

    def authorize_population_mutation(
        self,
        snapshot: RuntimeWorkClockSnapshot,
        *,
        topology_version: int,
        max_actions: int,
        geometry_opportunity: bool,
    ):
        """Issue a neutral live-policy budget without choosing an action.

        The legacy ``decide`` API retains independently named grow/relocate
        lanes for delegated MCMC compatibility.  The live population policy
        consumes only this token: cadence says whether bounded structural work
        is affordable, while policy evidence chooses CREATE, RELOCATE, RETIRE,
        or ABSTAIN.  Fresh depth/geometry evidence is an independent
        opportunity, so the live path is not exclusively RGB-clock driven.
        """

        max_actions = _exact_non_negative(max_actions, "max_actions")
        if type(geometry_opportunity) is not bool:
            raise TypeError("geometry_opportunity must be bool")
        decision = self.decide(snapshot, topology_version=topology_version)
        neutral_interval = min(
            decision.active_grow_interval_pixels,
            decision.active_relocate_interval_pixels,
        )
        rgb_due = (
            snapshot.supervised_rgb_pixels
            - self._last_population_authorization_rgb_pixels
            >= neutral_interval
        )
        geometry_due = bool(
            geometry_opportunity
            and snapshot.depth_valid_samples
            > self._last_population_authorization_depth_valid_samples
        )
        due = bool(
            decision.optimizer_settled and (rgb_due or geometry_due)
        )
        return MutationWorkToken(
            epoch=snapshot.optimizer_updates,
            units=max_actions if due else 0,
            geometry_opportunity=geometry_opportunity,
        )

    def record_population_mutation(
        self,
        snapshot: RuntimeWorkClockSnapshot,
        *,
        topology_version_after: int,
    ) -> None:
        """Consume one applied neutral live-policy authorization.

        This intentionally does not classify the committed structural action.
        Legacy grow/relocate exposure lanes remain intact for compatibility;
        the separate neutral baselines force the live population path to wait
        for fresh RGB debt or depth opportunity after its settling fence.
        """

        self.record_structural_mutation(
            snapshot,
            topology_version_after=topology_version_after,
        )
        self._last_population_authorization_rgb_pixels = (
            snapshot.supervised_rgb_pixels
        )
        self._last_population_authorization_depth_valid_samples = (
            snapshot.depth_valid_samples
        )

    def record_structural_mutation(
        self,
        snapshot: RuntimeWorkClockSnapshot,
        *,
        topology_version_after: int,
        grew: bool = False,
        relocated: bool = False,
    ) -> None:
        """Observe an exact backend mutation and consume only applied lanes.

        Depth insertion calls this with both action flags false. It advances
        the settling fence while preserving both MCMC exposure debts.
        """

        if not isinstance(snapshot, RuntimeWorkClockSnapshot):
            raise TypeError("snapshot must be RuntimeWorkClockSnapshot")
        topology_version_after = _exact_non_negative(
            topology_version_after, "topology_version_after"
        )
        if type(grew) is not bool or type(relocated) is not bool:
            raise TypeError("applied MCMC effects must be bool")
        if snapshot.supervised_rgb_pixels < self._observed_supervised_rgb_pixels:
            raise ValueError("supervised RGB pixel counter regressed")
        if snapshot.optimizer_updates < self._observed_optimizer_updates:
            raise ValueError("optimizer update counter regressed")
        if topology_version_after <= self._observed_topology_version:
            raise ValueError("structural mutation did not advance topology version")
        grow_debt = (
            snapshot.supervised_rgb_pixels - self._last_committed_grow_pixels
        )
        relocate_debt = (
            snapshot.supervised_rgb_pixels - self._last_committed_relocate_pixels
        )
        active_grow_interval_pixels = self.grow_interval_pixels
        if snapshot.source_complete and self.post_ingestion_grow_interval_pixels > 0:
            active_grow_interval_pixels = self.post_ingestion_grow_interval_pixels
        active_relocate_interval_pixels = self.relocate_interval_pixels
        if (
            snapshot.source_complete
            and self.post_ingestion_relocate_interval_pixels > 0
        ):
            active_relocate_interval_pixels = (
                self.post_ingestion_relocate_interval_pixels
            )
        if grew and grow_debt < active_grow_interval_pixels:
            raise ValueError("applied growth had no due exposure debt")
        if relocated and relocate_debt < active_relocate_interval_pixels:
            raise ValueError("applied relocation had no due exposure debt")
        if grew:
            self._last_committed_grow_pixels = snapshot.supervised_rgb_pixels
        if relocated:
            self._last_committed_relocate_pixels = snapshot.supervised_rgb_pixels
        self._last_structural_optimizer_updates = snapshot.optimizer_updates
        self._observed_supervised_rgb_pixels = snapshot.supervised_rgb_pixels
        self._observed_optimizer_updates = snapshot.optimizer_updates
        self._observed_topology_version = topology_version_after

    def state_dict(self) -> dict[str, Any]:
        return {
            "owner_schema": self._SCHEMA,
            "grow_interval_pixels": self.grow_interval_pixels,
            "post_ingestion_grow_interval_pixels": (
                self.post_ingestion_grow_interval_pixels
            ),
            "relocate_interval_pixels": self.relocate_interval_pixels,
            "post_ingestion_relocate_interval_pixels": (
                self.post_ingestion_relocate_interval_pixels
            ),
            "min_optimizer_updates_after_structural_mutation": (
                self.min_optimizer_updates_after_structural_mutation
            ),
            "last_committed_grow_pixels": self._last_committed_grow_pixels,
            "last_committed_relocate_pixels": self._last_committed_relocate_pixels,
            "last_structural_optimizer_updates": (
                self._last_structural_optimizer_updates
            ),
            "observed_supervised_rgb_pixels": self._observed_supervised_rgb_pixels,
            "observed_optimizer_updates": self._observed_optimizer_updates,
            "observed_topology_version": self._observed_topology_version,
            "last_population_authorization_rgb_pixels": (
                self._last_population_authorization_rgb_pixels
            ),
            "last_population_authorization_depth_valid_samples": (
                self._last_population_authorization_depth_valid_samples
            ),
        }

    def load_state_dict(self, payload: object) -> None:
        if not isinstance(payload, dict):
            raise ValueError("incompatible MCMC cadence state keys")
        schema = payload.get("owner_schema")
        if schema == self._SCHEMA and set(payload) == self._STATE_KEYS:
            values = {
                name: _exact_non_negative(payload[name], name)
                for name in self._STATE_KEYS
                if name != "owner_schema"
            }
        elif schema == self._LEGACY_SCHEMA and set(payload) == self._LEGACY_STATE_KEYS:
            values = {
                name: _exact_non_negative(payload[name], name)
                for name in self._LEGACY_STATE_KEYS
                if name != "owner_schema"
            }
            # A v2 checkpoint predates neutral live authorization.  Start it
            # conservatively: old RGB debt cannot mint a live token after
            # resume; only newly completed work may do so.
            values["last_population_authorization_rgb_pixels"] = values[
                "observed_supervised_rgb_pixels"
            ]
            values["last_population_authorization_depth_valid_samples"] = 0
        else:
            raise ValueError("incompatible MCMC cadence state keys")
        if schema not in {self._SCHEMA, self._LEGACY_SCHEMA}:
            raise ValueError("incompatible MCMC cadence state schema")
        for name, current in (
            ("grow_interval_pixels", self.grow_interval_pixels),
            (
                "post_ingestion_grow_interval_pixels",
                self.post_ingestion_grow_interval_pixels,
            ),
            ("relocate_interval_pixels", self.relocate_interval_pixels),
            (
                "post_ingestion_relocate_interval_pixels",
                self.post_ingestion_relocate_interval_pixels,
            ),
            (
                "min_optimizer_updates_after_structural_mutation",
                self.min_optimizer_updates_after_structural_mutation,
            ),
        ):
            if values[name] != current:
                raise ValueError(f"incompatible MCMC cadence {name}")
        if (
            values["last_committed_grow_pixels"]
            > values["observed_supervised_rgb_pixels"]
            or values["last_committed_relocate_pixels"]
            > values["observed_supervised_rgb_pixels"]
        ):
            raise ValueError("committed exposure baseline exceeds observed counter")
        if (
            values["last_structural_optimizer_updates"]
            > values["observed_optimizer_updates"]
        ):
            raise ValueError("structural update baseline exceeds observed counter")
        if (
            values["last_population_authorization_rgb_pixels"]
            > values["observed_supervised_rgb_pixels"]
        ):
            raise ValueError("population RGB baseline exceeds observed counter")
        if (
            values["observed_supervised_rgb_pixels"]
            < self._observed_supervised_rgb_pixels
            or values["observed_optimizer_updates"]
            < self._observed_optimizer_updates
            or values["observed_topology_version"]
            < self._observed_topology_version
        ):
            raise ValueError("MCMC cadence state rollback would regress live state")
        for name, current in (
            ("last_committed_grow_pixels", self._last_committed_grow_pixels),
            (
                "last_committed_relocate_pixels",
                self._last_committed_relocate_pixels,
            ),
            (
                "last_structural_optimizer_updates",
                self._last_structural_optimizer_updates,
            ),
            (
                "last_population_authorization_rgb_pixels",
                self._last_population_authorization_rgb_pixels,
            ),
            (
                "last_population_authorization_depth_valid_samples",
                self._last_population_authorization_depth_valid_samples,
            ),
        ):
            if values[name] < current:
                raise ValueError(f"{name} would regress live state")

        self._last_committed_grow_pixels = values["last_committed_grow_pixels"]
        self._last_committed_relocate_pixels = values[
            "last_committed_relocate_pixels"
        ]
        self._last_structural_optimizer_updates = values[
            "last_structural_optimizer_updates"
        ]
        self._observed_supervised_rgb_pixels = values[
            "observed_supervised_rgb_pixels"
        ]
        self._observed_optimizer_updates = values["observed_optimizer_updates"]
        self._observed_topology_version = values["observed_topology_version"]
        self._last_population_authorization_rgb_pixels = values[
            "last_population_authorization_rgb_pixels"
        ]
        self._last_population_authorization_depth_valid_samples = values[
            "last_population_authorization_depth_valid_samples"
        ]


__all__ = ["MCMCCadenceDecision", "MCMCCadenceOwner"]
