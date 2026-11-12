"""Convenience re-exports for the gradient-policy contract.

Lets consumers write ``from
eligsir.optimization.gaussian.gradient_policies.base import
GradientPolicyBase`` (matching the file list proposed in plan section 7.5)
instead of reaching into ``..contracts`` directly. No new logic lives here.
"""

from __future__ import annotations

from ..contracts import (
    GradientPolicy,
    GradientPolicyBase,
    GradientPolicyContext,
    GradientPolicyResult,
)

__all__ = [
    "GradientPolicy",
    "GradientPolicyBase",
    "GradientPolicyContext",
    "GradientPolicyResult",
]
