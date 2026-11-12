"""State machine for the mapping to post-mapping refinement boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    MAPPING = "mapping"
    REFINEMENT = "refinement"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PhaseTransition:
    from_phase: Phase
    to_phase: Phase
    iteration: int
    successful_optimization_steps: int


class StreamingPhaseController:
    """Count a fixed refinement budget after complete source disposition."""

    STATE_VERSION = 1

    def __init__(self, refinement_steps: int):
        if int(refinement_steps) < 0:
            raise ValueError("refinement_steps must be non-negative")
        self.refinement_steps = int(refinement_steps)
        self.phase = Phase.MAPPING
        self.mapping_end_iteration: int | None = None
        self.mapping_optimizer_steps: int | None = None
        self._last_successful_optimization_steps = 0

    @property
    def in_refinement(self) -> bool:
        return self.phase is Phase.REFINEMENT

    @property
    def refinement_steps_completed(self) -> int:
        if self.mapping_optimizer_steps is None:
            return 0
        return max(
            0,
            self._last_successful_optimization_steps - self.mapping_optimizer_steps,
        )

    def observe_source(
        self,
        source_exhausted: bool,
        dispositions_complete: bool,
        successful_optimization_steps: int,
        iteration: int,
    ) -> PhaseTransition | None:
        successful_steps = int(successful_optimization_steps)
        if successful_steps < self._last_successful_optimization_steps:
            raise ValueError("successful_optimization_steps must be monotonic")
        self._last_successful_optimization_steps = successful_steps

        if self.phase is Phase.MAPPING:
            if not (source_exhausted and dispositions_complete):
                return None
            self.mapping_end_iteration = int(iteration)
            self.mapping_optimizer_steps = successful_steps
            previous = self.phase
            self.phase = (
                Phase.COMPLETE if self.refinement_steps == 0 else Phase.REFINEMENT
            )
            return PhaseTransition(
                previous, self.phase, int(iteration), successful_steps
            )

        if (
            self.phase is Phase.REFINEMENT
            and self.refinement_steps_completed >= self.refinement_steps
        ):
            previous = self.phase
            self.phase = Phase.COMPLETE
            return PhaseTransition(
                previous, self.phase, int(iteration), successful_steps
            )
        return None

    def resume_state_dict(self) -> dict:
        return {
            "version": self.STATE_VERSION,
            "refinement_steps": self.refinement_steps,
            "phase": self.phase.value,
            "mapping_end_iteration": self.mapping_end_iteration,
            "mapping_optimizer_steps": self.mapping_optimizer_steps,
            "last_successful_optimization_steps": (
                self._last_successful_optimization_steps
            ),
        }

    def load_resume_state_dict(self, payload: dict, resolver=None) -> None:
        if (
            not isinstance(payload, dict)
            or set(payload) != {
                "version", "refinement_steps", "phase", "mapping_end_iteration",
                "mapping_optimizer_steps", "last_successful_optimization_steps",
            }
            or payload.get("version") != self.STATE_VERSION
        ):
            raise ValueError("unsupported streaming phase checkpoint state")
        if (
            isinstance(payload.get("refinement_steps"), bool)
            or not isinstance(payload.get("refinement_steps"), int)
            or payload["refinement_steps"] != self.refinement_steps
        ):
            raise ValueError("refinement_steps changed across resume")
        try:
            phase = Phase(payload["phase"])
            mapping_end_iteration = payload["mapping_end_iteration"]
            mapping_optimizer_steps = payload["mapping_optimizer_steps"]
            last_steps = payload["last_successful_optimization_steps"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid streaming phase checkpoint state") from exc
        if phase is not Phase.MAPPING and (
            mapping_end_iteration is None or mapping_optimizer_steps is None
        ):
            raise ValueError("refinement phase is missing its mapping boundary")
        for name, value in (
            ("mapping_end_iteration", mapping_end_iteration),
            ("mapping_optimizer_steps", mapping_optimizer_steps),
            ("last_successful_optimization_steps", last_steps),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"invalid {name}")
        if mapping_optimizer_steps is not None and last_steps < mapping_optimizer_steps:
            raise ValueError("last successful optimization steps precede mapping boundary")
        self.phase = phase
        self.mapping_end_iteration = mapping_end_iteration
        self.mapping_optimizer_steps = mapping_optimizer_steps
        self._last_successful_optimization_steps = last_steps
