"""Narrow ownership stores extracted from `scene/streaming_scene.py`.

Wave 1 consolidation plan step 9 (see
the mapping store ownership boundary and
`docs/architecture/streaming_scene_decomposition.md` for the full
attribute-level migration map: KeyframeStore, TrajectoryStore,
FrameArchive, ReplayViewStore, OccupancyIndex, ObservationStore).

**Status (2026-08-05):** `TrajectoryStore` is the first real extraction,
landed with zero external API change (`StreamingScene` composes it
internally and keeps `_dense_trajectory`/`_dense_trajectory_flushed` as
property-backed aliases, so every existing external call site --
`train_streaming.py`'s `_flush_dense_trajectory`, both trajectory unit
test files -- keeps working unchanged). The other five stores in the
migration map are not yet extracted; this package is where they land,
following the same pattern.
"""
