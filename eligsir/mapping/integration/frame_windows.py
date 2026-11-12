"""Shared frame-collection/windowing prologue for the alternate trainer modes.

Both alternate whole-run trainers -- ``_run_rolling_seed``
(``rolling_seed_application.py``) and ``_run_submap_stitch``
(``submap_stitch_application.py``) -- opened with a byte-identical block of
orchestration: drain the frame source, pair the non-holdout source frames with
the ingested train cameras, and chunk the pairs into fixed-size windows. That
duplication is exactly the "trainer entrypoints hiding shared orchestration"
pattern eligsir#1075 names, so it lives here once instead of twice.

Scope, stated honestly
======================

This is a **pure extraction**. The body below is the two call sites' logic
verbatim -- same order of operations, same truncation rule, same chunking --
so both modes' observable behavior (which frames land in which window, how many
train cameras are collected, and therefore every downstream growth event) is
unchanged. ``tests/test_issue1075_alternate_trainer_windows.py`` pins that
equivalence against an independent transcription of the pre-extraction code.

**This is deliberately NOT a ``MappingSession``/``TrackingProvider``
migration.** Neither trainer performs any tracking: poses are already resolved
inside the ``Camera`` objects ``StreamingScene.ingest_next_frame()`` returns,
no frontend is constructed or driven here, and no ``SensorPacket``-shaped
observation exists at this layer. Routing this drain loop through a
``MappingSession`` would mean synthesizing packets purely to hand them to a
passthrough provider whose ledger nothing reads -- the "object nothing
consumes is decoration" failure #1075's own note explicitly refused. See
``docs/architecture/2026-08-23-issue1075-alternate-trainers-audit.md`` for the
full evidence and for what would have to exist first.
"""

from __future__ import annotations

from typing import Any, List, Tuple


def collect_frame_windows(
    streaming_scene: Any,
    args: Any,
    submap_frames: int,
) -> Tuple[List[Any], List[List[Tuple[Any, Any]]]]:
    """Drain the frame source and chunk (frame, camera) pairs into windows.

    Args:
        streaming_scene: The live :class:`~scene.streaming_scene.StreamingScene`.
            Its remaining frames are ingested here -- this call has the side
            effect of exhausting the source, which is what both trainers want
            before they start their own (non-streaming) processing.
        args: Parsed run arguments; only ``streaming_eval_hold`` is read.
        submap_frames: Window size. Callers already clamp this to ``>= 2``.

    Returns:
        ``(all_train_cameras, windows)`` where ``all_train_cameras`` is every
        train camera the scene holds after the drain (bootstrap cameras first,
        in ingest order), and ``windows`` is the list of ``submap_frames``-sized
        chunks of ``(frame, camera)`` pairs.

    Notes:
        Pairing is positional, not by identity: the ``i``-th non-holdout source
        frame is paired with the ``i``-th train camera, and the lists are
        truncated to the shorter of the two. That is the pre-existing contract
        (frames rejected by admission make the camera list shorter than the
        frame list), preserved verbatim rather than corrected here -- changing
        it would change which geometry each window sees.
    """
    # Collect all frames (use iter_all_frames to support lazy/live sources)
    all_frames = list(streaming_scene.iter_all_frames())
    all_train_cameras = list(streaming_scene.train_cameras)
    while streaming_scene.has_next_frame():
        result = streaming_scene.ingest_next_frame()
        if result is not None:
            cam, frame, is_train = result
            if is_train:
                all_train_cameras.append(cam)

    eval_hold = args.streaming_eval_hold
    train_frames = [f for f in all_frames if eval_hold <= 0 or f.index % eval_hold != 0]
    n_paired = min(len(train_frames), len(all_train_cameras))
    frame_cam_pairs = list(zip(train_frames[:n_paired], all_train_cameras[:n_paired]))

    windows = [
        frame_cam_pairs[i: i + submap_frames]
        for i in range(0, len(frame_cam_pairs), submap_frames)
    ]
    return all_train_cameras, windows


__all__ = ["collect_frame_windows"]
