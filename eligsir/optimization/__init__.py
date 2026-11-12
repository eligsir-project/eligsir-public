"""Fixed-shape numerical refinement of the Gaussian map.

Wave 1 consolidation plan (see
the mapping optimization ownership boundary.
Distinct from :mod:`eligsir.mapping` (observation-driven: admission,
integration, mutation) and from :mod:`eligsir.map` (GPU-state
ownership): this package only concerns render -> loss -> backward ->
optimizer step, on a fixed parameter shape. It must never insert, prune,
relocate, or otherwise change topology — those are
:class:`eligsir.map.mutation.GaussianMutationTransaction` operations.
"""
