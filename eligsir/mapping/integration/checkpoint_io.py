"""Streaming checkpoint / metadata / PLY-export IO helpers, split out of
``rgbd_integrator_impl.py`` (WS-E5) -- see that module's docstring for the
history of why this cluster ended up as one file.

Verbatim relocation, no math/logic change. ``rgbd_integrator_impl.py``
re-exports everything from here so every existing call site keeps working
unchanged.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from eligsir.io.snapshots import (
    snapshot_gaussians_for_ply as _snapshot_for_ply,
    subset_ply_snapshot as _subset_snapshot,
    write_ply_from_snapshot as _write_ply,
)
from eligsir.mapping.integration.frame_preparation import _timed


def _save_streaming_checkpoint(save_worker, gaussians, model_path: str, iteration: int,
                               n_frames_ingested: int, last_ingest_iter: int,
                               source_idx: int, keyframes_admitted: int = 0,
                               reason: str = "periodic", map_backend=None,
                               seed_fade_monitor=None,
                               per_source_supervision_counts=None,
                               coverage_ledger: dict | None = None,
                               extra_metadata: dict | None = None) -> str:
    """Capture + enqueue a streaming checkpoint save, atomically.

    Payload is a 3-tuple ``(state, iteration, streaming_meta)`` — the
    ``streaming_meta`` dict carries the frame-ingestion cursor
    (``n_frames_ingested``, ``last_ingest_iter``, ``source_idx``) so a resume
    can reposition the frame source instead of silently re-ingesting/
    re-admitting already-seen frames or re-running bootstrap (see the
    checkpoint-restore path above and the "Run Contracts" notes in
    CLAUDE.md). The write goes through a temp file + os.replace so a
    SIGKILL mid-write (e.g. terminationGracePeriodSeconds expiring on a
    preempted k8s pod) can never leave a truncated/corrupt chkpntN.pth behind
    — a partial write only ever clobbers the ``.tmp`` path, never the final
    one, so "latest chkpntN.pth present" is always a valid resume target.

    ``map_backend`` (Wave 2 backend-ownership pass, 2026-08-06): when given,
    the model capture goes through ``map_backend.checkpoint_state()`` rather
    than ``gaussians.capture()`` directly, and the resulting
    ``map_version``/``topology_version`` ride along in ``streaming_meta`` so
    a resumed run's backend reports the same versions instead of resetting
    to 0. Optional (defaults to the old direct-capture path) so this
    function still works for any caller that hasn't been threaded a backend
    yet.

    Returns the final checkpoint path (queued for async write; may not exist
    on disk yet when this returns).
    """
    print(f"\n[STEP {iteration}] Saving Checkpoint (reason={reason})", flush=True)
    with _timed("save_capture"):
        if map_backend is not None:
            blob = map_backend.checkpoint_state()
            state = blob.payload
            map_version, topology_version = blob.map_version, blob.topology_version
        else:
            state = gaussians.capture()
            map_version, topology_version = 0, 0
    streaming_meta = {
        "checkpoint_reason": str(reason),
        "n_frames_ingested": int(n_frames_ingested),
        "last_ingest_iter": int(last_ingest_iter),
        "source_idx": int(source_idx),
        # Cumulative, monotonic keyframe-admission count -- NOT restored
        # otherwise, silently resetting to 0 on resume and making the
        # trajectory_coverage/frames_admitted report metric (and the live
        # adm% progress-bar figure, both derived from this counter) read
        # near-zero for the remainder of any resumed run. Found via a
        # resumed scannetpp_partitionfocus_v2_focus run whose adm% climbed
        # normally to 15% pre-resume then dropped to 0% for the rest of
        # training (5105/5207 ticks), producing a reported frames_admitted
        # of 22 despite a real cumulative admission rate around 14-15%.
        "keyframes_admitted": int(keyframes_admitted),
        "map_version": int(map_version),
        "topology_version": int(topology_version),
    }
    if per_source_supervision_counts is not None:
        streaming_meta["per_source_supervision_counts"] = {
            int(source_idx): max(0, int(count))
            for source_idx, count in per_source_supervision_counts.items()
        }
    if coverage_ledger is not None:
        if not isinstance(coverage_ledger, dict):
            raise TypeError("coverage_ledger must be a dict")
        streaming_meta["coverage_ledger"] = dict(coverage_ledger)
    if seed_fade_monitor is not None:
        # Issue #290: the seed-fade detector's running max / episode state is
        # otherwise in-memory only, so a resumed run silently graded itself
        # against a running max that restarted from -inf mid-run. Absent from
        # older checkpoints, which is exactly why the resume path treats a
        # missing key as INDETERMINATE rather than as "no fade".
        try:
            streaming_meta["seed_fade"] = seed_fade_monitor.state_dict()
        except Exception as exc:  # never let telemetry break a checkpoint write
            print(f"[streaming] WARNING: seed-fade state not checkpointed: {exc}",
                  flush=True)
    if extra_metadata is not None:
        if not isinstance(extra_metadata, dict):
            raise TypeError("extra checkpoint metadata must be a dict")
        collisions = sorted(set(streaming_meta).intersection(extra_metadata))
        if collisions:
            raise ValueError(
                "extra checkpoint metadata may not override base metadata keys: "
                + ", ".join(collisions)
            )
        streaming_meta.update(extra_metadata)
    chk = os.path.join(model_path, f"chkpnt{iteration}.pth")

    def _do(s=state, i=iteration, meta=streaming_meta, p=chk):
        tmp = p + ".tmp"
        torch.save((s, i, meta), tmp)
        os.replace(tmp, p)  # atomic on POSIX — never leaves a half-written chkpntN.pth

    with _timed("save_enqueue"):
        save_worker.enqueue(_do)
    return chk


def _save_cameras_json(save_worker, out_dir: str, cameras) -> None:
    """Write a cameras.json next to an intermediate PLY export, without stalling.

    ``camera_to_JSON`` reads only CPU-side pose/intrinsics (camera.R/T are numpy and
    immutable after construction), so the serialization runs safely on the save-worker
    thread. The list is shallow-snapshotted on the caller thread (O(N) pointer copy,
    no GPU sync) and the actual JSON build + write happen asynchronously.
    """
    cams = list(cameras)
    if not cams:
        return

    def _do(cams=cams, out_dir=out_dir):
        try:
            import json as _json
            from eligsir.utils.camera_utils import camera_to_JSON
            entries = [camera_to_JSON(i, c) for i, c in enumerate(cams)]
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "cameras.json"), "w") as _f:
                _json.dump(entries, _f)
        except Exception as _e:  # never let a metadata save crash training
            print(f"[streaming] cameras.json save failed for {out_dir}: {_e}", flush=True)

    if save_worker is not None:
        with _timed("save_enqueue"):
            save_worker.enqueue(_do)
    else:
        _do()


def _save_test_cameras_json(save_worker, out_dir: str, cameras) -> None:
    """Write test_cameras.json: the deterministic real-holdout set, same schema
    as cameras.json. Without this, a post-hoc renderer has no way to reconstruct
    the exact test split used at training time -- only the admitted train
    keyframes were ever persisted (see _save_cameras_json above).
    """
    cams = list(cameras)
    if not cams:
        return

    def _do(cams=cams, out_dir=out_dir):
        try:
            import json as _json
            from eligsir.utils.camera_utils import camera_to_JSON
            entries = [camera_to_JSON(i, c) for i, c in enumerate(cams)]
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "test_cameras.json"), "w") as _f:
                _json.dump(entries, _f)
        except Exception as _e:
            print(f"[streaming] test_cameras.json save failed for {out_dir}: {_e}", flush=True)

    if save_worker is not None:
        with _timed("save_enqueue"):
            save_worker.enqueue(_do)
    else:
        _do()


# Ids already written by _save_holdout_gt this process -- holdout cameras never
# change after materialization, so each id's GT only needs saving once.
_HOLDOUT_GT_SAVED_IDS: set = set()


def _save_holdout_gt(save_worker, out_dir: str, cameras) -> None:
    """Persist GT color (+ sensor depth, if available) for holdout cameras.

    Enables a standalone post-hoc render/eval module to recompute real
    PSNR/LPIPS against the exact test split, without needing the live source
    (bag/camera) re-available. Idempotent by camera index -- only newly
    materialized holdout cams get their GT written; the holdout set is a
    small, deterministic fraction of frames (Run Contracts), so no throttling
    is needed.

    SCORE-BEARING, therefore EXACT (issue #639). Both RGB and depth are
    persisted through ``eligsir.io.holdout_gt_exact``, which writes the
    canonical representation bit-exactly (lossless PNG only when uint8 is
    verified exact for that array, else float32 ``.npy``) plus a sidecar
    carrying dtype/shape/layout/domain/sha256. RGB used to be written as JPEG
    q92, which measurably INFLATED post-hoc test PSNR by +0.175 dB on
    ``output/wave1_verify_fr3`` (8/8 views) -- see issue #630 and that module's
    docstring. Never reintroduce a lossy codec here.

    ENCODING CONTRACT for ``holdout_gt/<id>_depth.png`` (issue #358): the file
    is uint16 **millimetres, unconditionally**, whatever the source dataset's
    own ``depth_scale`` was. This is NOT a copy of the dataset's depth PNG --
    ``_get_sensor_depth`` has already divided the dataset scale out and returns
    metres. Readers must divide by
    ``eligsir.io.holdout_gt_depth.HOLDOUT_GT_DEPTH_SCALE`` (1000) and never
    by the dataset scale; doing the latter silently shrank issue #358's whole
    fr3 microcell lattice 5x and forced a public retraction of its Level A/B
    figures. See that module's docstring.
    """
    new_cams = [(i, c) for i, c in enumerate(cameras) if i not in _HOLDOUT_GT_SAVED_IDS]
    if not new_cams:
        return
    for i, _ in new_cams:
        _HOLDOUT_GT_SAVED_IDS.add(i)

    def _do(new_cams=new_cams, out_dir=out_dir):
        try:
            import cv2
            from eligsir.io.holdout_gt_depth import encode_holdout_depth_mm
            from eligsir.io.holdout_gt_exact import (
                save_holdout_depth, save_holdout_rgb,
            )
            from eligsir.utils.comparison_report import _gt_image, _get_sensor_depth
            gt_dir = os.path.join(out_dir, "holdout_gt")
            os.makedirs(gt_dir, exist_ok=True)
            for i, cam in new_cams:
                try:
                    # EXACT, score-preserving (issue #639): persist the canonical
                    # float32 CHW GT itself -- lossless PNG only when uint8 is
                    # verified bit-exact for this array, else float32 .npy, plus a
                    # sidecar carrying dtype/shape/layout/domain/sha256. The old
                    # JPEG q92 write measurably INFLATED post-hoc test PSNR
                    # (+0.175 dB on output/wave1_verify_fr3, 8/8 views -- issue #630).
                    save_holdout_rgb(gt_dir, i, _gt_image(cam))
                except Exception as _e:
                    print(f"[streaming] holdout_gt rgb save failed for cam {i}: {_e}", flush=True)
                try:
                    depth_t = _get_sensor_depth(cam, cam.image_height, cam.image_width)
                    if depth_t is not None:
                        depth_np = depth_t.detach().cpu().numpy().reshape(cam.image_height, cam.image_width)
                        # Exact float32 metres, the score-bearing representation.
                        save_holdout_depth(gt_dir, i, depth_np)
                        # LEGACY uint16-mm PNG (issue #358 contract) kept for its
                        # existing consumers (scripts/local_quality_*); readers
                        # prefer the .npy above.
                        depth_mm = encode_holdout_depth_mm(depth_np)
                        cv2.imwrite(os.path.join(gt_dir, f"{i}_depth.png"), depth_mm)
                except Exception as _e:
                    print(f"[streaming] holdout_gt depth save failed for cam {i}: {_e}", flush=True)
        except Exception as _e:
            print(f"[streaming] holdout_gt save failed for {out_dir}: {_e}", flush=True)

    if save_worker is not None:
        with _timed("save_enqueue"):
            save_worker.enqueue(_do)
    else:
        _do()


def _flush_dense_trajectory(save_worker, model_path: str, streaming_scene) -> None:
    """Append newly-recorded dense-trajectory entries to trajectory_full.jsonl.

    Append-mode, one JSON object per line, so a partial/killed run still leaves
    a valid, parseable prefix. Only entries recorded since the last flush are
    written (tracked via streaming_scene._dense_trajectory_flushed).
    """
    since = streaming_scene._dense_trajectory_flushed
    new_entries = streaming_scene.dense_trajectory_snapshot(since=since)
    if not new_entries:
        return
    streaming_scene._dense_trajectory_flushed = len(streaming_scene._dense_trajectory)

    def _do(new_entries=new_entries, model_path=model_path):
        try:
            import json as _json
            path = os.path.join(model_path, "trajectory_full.jsonl")
            with open(path, "a") as _f:
                for entry in new_entries:
                    _f.write(_json.dumps(entry) + "\n")
        except Exception as _e:
            print(f"[streaming] trajectory_full.jsonl append failed: {_e}", flush=True)

    if save_worker is not None:
        with _timed("save_enqueue"):
            save_worker.enqueue(_do)
    else:
        _do()


def _export_ply_snapshot(
    gaussians, streaming_scene, out_ply_path: str, save_worker, min_opacity: float = 0.0
) -> None:
    """Single PLY-export mechanism for the streaming loop.

    Snapshots the Gaussians and asynchronously writes the PLY plus a co-located
    cameras.json. This is the *one* place intermediate PLYs are exported — both the
    frame-milestone and iteration-interval save triggers call through here so the
    export (PLY + cameras.json) never drifts between the two paths.
    """
    with _timed("save_snapshot"):
        snap = _snapshot_for_ply(gaussians)
        if min_opacity > 0.0:
            _keep = torch.sigmoid(snap["opacities"].reshape(-1)) > min_opacity
            if not bool(_keep.all()):
                snap = _subset_snapshot(snap, _keep)
    with _timed("save_enqueue"):
        save_worker.enqueue(lambda s=snap, p=out_ply_path: _write_ply(s, p))
    _save_cameras_json(save_worker, os.path.dirname(out_ply_path), streaming_scene.train_cameras)
