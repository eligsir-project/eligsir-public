"""Gaussian map ownership boundary.

This package is the sole owner of live Gaussian-splat GPU state, per the
Wave 1 consolidation plan (see
the map ownership boundary.
It is intentionally placed at the top level, not nested under
``eligsir.mapping``, because the Gaussian map and its optimizer are
consumed by live mapping, replay mapping, offline reconstruction, offline
training, refinement, evaluation, and export alike.

Populated incrementally:

- ``views`` — :class:`GaussianMapReadView`, the borrowed device-resident
  read path (Wave 1 step 2).
- ``snapshots`` — :class:`MapSnapshot` / :class:`SnapshotRequest` /
  :class:`SnapshotProvider`, the owned host-side export path (Wave 1 step 2).
- ``contracts`` — the ``GaussianMapBackend`` protocol (Wave 1 step 3).
- ``mutation`` — ``GaussianMutationTransaction`` and friends (Wave 1 step 6).
- ``adapters`` — ``GsplatMapBackend`` (step 3), ``LegacyGraphdecoMapBackend``
  (step 4).
"""
