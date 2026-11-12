"""Experiment contract: typed manifests, hashing, and validation.

Pure-Python, CPU-only reproducibility scaffolding for causal ablations (see
``plans/2026-08-06-wave3-live-mapping-runtime-evidence-strategy-implementation-plan.md``,
section 3, "the golden rule of ablations"). This package must never import
the optimization loop, ``train_streaming.py``, or any GPU code.
"""

from __future__ import annotations

from eligsir.experiments.manifest import (
    ArmSpec,
    ExperimentManifest,
    FrozenLevel,
    HardwareInfo,
    ValidityInfo,
)

__all__ = [
    "ArmSpec",
    "ExperimentManifest",
    "FrozenLevel",
    "HardwareInfo",
    "ValidityInfo",
]
