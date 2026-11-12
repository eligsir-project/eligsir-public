"""Frame-utility scoring and admission control (Wave 3, PR W3-06 + W3-07,
plan sections 11-12).

W3-06 built the shadow frame-utility scorer (observe/log only, never
decides anything -- see ``shadow_policy.py``'s docstring). W3-07 builds
the graduated rollout policy that can act on those scores
(``rollout_policy.py``, ``decision_contracts.py``, ``budget_controller.py``)
under one of four modes (:class:`RolloutMode`): ``shadow`` (default for
every real run), ``advisory``, ``active``, ``active_with_budget``. The
existing ``_keyframe_admission_closure``/``KeyframeAdmitter`` path in
``train_streaming.py`` remains the fallback baseline decision-maker in
every mode -- see ``rollout_policy.py``'s module docstring for the hard
safety-gate invariant that holds regardless of rollout mode.
"""

from importlib import import_module

from eligsir.mapping.admission.budget_controller import (
    MappingBudgetController,
    MappingBudgetControllerConfig,
)
from eligsir.mapping.admission.cost_model import predict_gpu_cost
from eligsir.mapping.admission.decision_contracts import (
    AdmissionDecision,
    MappingBudgetRequest,
    RolloutMode,
)
from eligsir.mapping.admission.fixed_stride import (
    FixedStrideAdmitter,
    FixedStrideDecision,
)
from eligsir.mapping.admission.observation_owner import (
    DispositionConflict,
    DispositionLedger,
    MaterializationPermit,
    ObservationOwner,
    apply_disposition,
    disposition_from_backpressure,
)
from eligsir.mapping.admission.rollout_policy import AdmissionRolloutPolicy
from eligsir.mapping.admission.utility_contracts import (
    FrameUtilityComponents,
    FrameUtilityScore,
    UtilityWeights,
)


# Importing any admission submodule executes this package initializer first.
# Keep only torch-backed convenience exports lazy so dependency-light modules
# such as ``budgeted_controller`` remain importable in CPU-only environments.
_EXPORT_MODULES = {
    "RealizedGainRecord": ".realized_gain",
    "RealizedGainTracker": ".realized_gain",
    "score_frame_utility": ".scoring",
    "ShadowUtilityPolicy": ".shadow_policy",
}


def __getattr__(name: str):
    """Resolve a convenience export without importing unrelated policies."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "FrameUtilityComponents",
    "FrameUtilityScore",
    "UtilityWeights",
    "score_frame_utility",
    "predict_gpu_cost",
    "RealizedGainRecord",
    "RealizedGainTracker",
    "ShadowUtilityPolicy",
    "RolloutMode",
    "AdmissionDecision",
    "MappingBudgetRequest",
    "AdmissionRolloutPolicy",
    "MappingBudgetController",
    "MappingBudgetControllerConfig",
    "FixedStrideAdmitter",
    "FixedStrideDecision",
    "DispositionConflict",
    "DispositionLedger",
    "MaterializationPermit",
    "ObservationOwner",
    "apply_disposition",
    "disposition_from_backpressure",
]
