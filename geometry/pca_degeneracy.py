"""Shared degeneracy guard for PCA-based local-normal estimation (issue #1149).

Both ``geometry.insertion.fuse_point_with_neighbors``/``fuse_points_with_knn``
(NumPy, used by ``integration_planner.py``) and
``geometry.device_fusion.fuse_points_with_knn_torch`` (PyTorch, device-native)
estimate a local surface normal via PCA on a k-NN neighbourhood's centered
covariance matrix: eigendecompose the ``3x3`` covariance, take the eigenvector
of the *smallest* eigenvalue as the normal (the direction of least variance,
i.e. "out of the local surface").

When the neighbourhood collapses to too few *distinct* point locations (near
-duplicate depth samples, or points that are exactly/near collinear -- common
in production, since a query's own neighbours are drawn from the same pool
it's matched against), the covariance becomes rank <= 1: two of its three
eigenvalues collapse toward zero together, and the "smallest" eigenvector is
no longer a well-defined direction -- it lives in an arbitrary 2D null space
rather than pointing along a unique normal. NumPy's ``eigh`` and PyTorch's
``eigh`` resolve that ambiguity differently (confirmed NOT a float32-vs-
float64 precision artifact by testing both implementations at float64 -- see
issue #1149), so they can and do disagree by up to ~1.5 rad (nearly
antiparallel) on the exact same input neighbourhood.

Both implementations MUST use the identical guard below so they can never
independently drift on this threshold again -- that consistency, not bit-
identical output in the degenerate case itself, is the actual point of this
module (see issue #1149's discussion of reject-vs-fallback: both sides fall
back to the same caller-supplied ``normal_hint``, so they agree exactly).

Threshold derivation
---------------------
For ascending eigenvalues ``e0 <= e1 <= e2`` of a covariance matrix (units of
squared length):

- A genuine, well-sampled, roughly-planar surface patch has ``e2`` and ``e1``
  both comparable in magnitude (the two in-plane directions), well separated
  from ``e0`` (the out-of-plane direction, ideally near zero -- *that's the
  whole point of PCA normal estimation*, so a small ``e0`` is expected and
  healthy, NOT a sign of degeneracy). The normal (``e0``'s eigenvector) is
  then unique up to sign.
- A degenerate neighbourhood (points collapsed onto <= 2 distinct locations,
  or exactly/near collinear -- rank <= 1) instead has ``e1`` collapse toward
  ``e0`` -- BOTH near zero relative to ``e2`` -- so the plane spanned by the
  two smallest eigenvectors is itself undefined, and whichever specific
  direction each solver picks within that plane (as its "``e0``
  eigenvector") is an artifact of the solver's tie-breaking, not the data.

So the discriminating quantity is ``ratio = e1 / e2`` (NOT ``e0``, which is
expected to be small in both the healthy and degenerate case): a healthy
patch has ``ratio`` typically O(1) (roughly isotropic in-plane spread); a
degenerate one has ``ratio`` collapse toward the solver's numerical noise
floor.

``EIGENVALUE_RATIO_EPS = 1e-3`` is set from that noise floor with margin:
eigenvalues are built from *squared* lengths, so a relative error of
``eps_dtype`` in the underlying coordinates shows up as a relative error of
roughly ``2 * eps_dtype`` in the eigenvalues, and eigenvector *directions*
near a degenerate eigenvalue are only stable (Davis-Kahan) once the relevant
eigengap clears roughly ``sqrt(eps_dtype)`` of the matrix norm. float32 (the
dtype the torch path runs at in production -- see
``geometry/device_fusion.py``) has ``eps_f32 ~= 1.19e-7``, so
``sqrt(eps_f32) ~= 3.4e-4``. ``1e-3`` sits about 3x above that float32
sqrt-eps floor (and about 3 orders of magnitude above float64's own
``sqrt(eps_f64) ~= 1.5e-8``), while remaining far below the ratio produced by
a real 2-distinct-point collapse (measured at issue #1149's own synthetic
repro: ``ratio`` on the order of ``1e-6`` or smaller, i.e. indistinguishable
from exact rank-1 up to floating noise) and far below any real, well-sampled
planar patch (``ratio`` normally O(1), see the docstring above). It is
deliberately NOT reused from ``needle_frac_lt_0.3``
(``utils/differentiable_energy_loss.py``'s Gaussian-*shape* needle
classifier, also an ``s2/s1`` ratio but over already-fit Gaussian scales, not
raw sample eigenvalues): a physically thin-but-real surfel can legitimately
have a small scale ratio and still be a perfectly well-defined, trustworthy
Gaussian, whereas this guard exists purely to detect *numerical*
under-determination of the PCA eigenbasis itself, which needs a much smaller
threshold to avoid falsely flagging real, if imperfectly isotropic, planar
patches.
"""

from __future__ import annotations

EIGENVALUE_RATIO_EPS = 1e-3


def eigvals_well_conditioned(e0: float, e1: float, e2: float, *, eps: float = EIGENVALUE_RATIO_EPS) -> bool:
    """Scalar guard for one neighbourhood's ascending eigenvalues ``e0<=e1<=e2``.

    Returns ``True`` iff the covariance has genuine rank >= 2 (a well-defined
    2D local surface patch), i.e. the PCA normal (``e0``'s eigenvector) is a
    numerically trustworthy direction rather than an arbitrary pick from a
    degenerate null space. See module docstring for the derivation.
    """
    if not (e2 == e2) or e2 <= 0.0:  # NaN or non-positive largest eigenvalue
        return False
    return (e1 / e2) >= eps


__all__ = ["EIGENVALUE_RATIO_EPS", "eigvals_well_conditioned"]
