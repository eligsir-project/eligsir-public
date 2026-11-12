"""Typed contracts for the shadow frame-utility score (plan section 11.3/11.5).

``U = pose_quality * image_quality *
      (w_new_surface * new_surface
       + w_residual * persistent_depth_supported_residual
       + w_debt * under_supported_mass
       + w_gradient * expected_gradient_value
       + w_parallax * useful_parallax)
      - w_redundancy * redundancy
      - w_specular * specular_risk
      - w_cost * predicted_gpu_cost``

Every component that feeds this formula is its own field on
:class:`FrameUtilityComponents`, logged separately (deliverable 2) rather
than only the final scalar -- this is what lets
``scripts/analyze_frame_utility.py`` diagnose *why* a frame scored the way
it did (e.g. "this frame scored high because of blur/pose-error/glass
residual, not real mapping value" -- deliverable 6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eligsir.mapping.evidence.contracts import EvidenceMeta


@dataclass(frozen=True)
class UtilityWeights:
    """Tunable weights for the formula above. Defaults are deliberately
    simple/uniform (not yet calibrated against real realized-gain data --
    that calibration is exactly what ``analyze_frame_utility.py`` /
    ``realized_gain.py`` exist to produce, for a *future* PR to consume).
    """

    w_new_surface: float = 1.0
    w_residual: float = 1.0
    w_debt: float = 1.0
    w_gradient: float = 1.0
    w_parallax: float = 0.5
    w_redundancy: float = 1.0
    w_specular: float = 1.0
    w_cost: float = 0.25


@dataclass(frozen=True)
class FrameUtilityComponents:
    """Every scalar input to the utility formula, logged individually.

    Field names match the formula's terms one-to-one (plan section 11.3)
    so a log consumer never has to reverse-engineer which value fed which
    term.
    """

    pose_quality: float
    image_quality: float
    new_surface: float
    persistent_depth_supported_residual: float
    under_supported_mass: float
    expected_gradient_value: float
    useful_parallax: float
    redundancy: float
    specular_risk: float
    predicted_gpu_cost: float


@dataclass(frozen=True)
class FrameUtilityScore:
    """The final scalar plus the full component breakdown and the weights
    that were applied -- self-contained enough that a log reader can
    recompute ``utility`` from ``components``/``weights`` alone as a sanity
    check (``scoring.recompute_utility`` does exactly that in tests).
    """

    frame_id: int
    utility: float
    components: FrameUtilityComponents
    weights: UtilityWeights
    meta: EvidenceMeta = field(default_factory=EvidenceMeta)
