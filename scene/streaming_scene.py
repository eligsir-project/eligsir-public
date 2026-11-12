"""
StreamingScene: manages arriving RGB-D frames for streaming-replay training.

Replaces the full-dataset Scene for --streaming_replay mode. Frames arrive
one at a time (controlled by FrameScheduler in train_streaming.py), each
converting to a Camera object and appending to the local keyframe window and
replay buffer. The Gaussian model is initialised from the first
streaming_initial_frames frames only; later frames insert Gaussians
incrementally via add_points_as_gaussians() (Phase 2).

Thread-safety: not re-entrant. All methods called from the training loop only.
"""
from __future__ import annotations

import os
import random
from collections import OrderedDict
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from geometry.surface_cells import SurfaceCellIndex
from eligsir.utils.holdout_rule import online_every_nth_holdout, online_holdout_rule_hash

if TYPE_CHECKING:
    from eligsir.utils.streaming_frames import StreamingRGBDFrame


def spatial_window_weights(
    centers: "torch.Tensor",
    viewdirs: "torch.Tensor",
    current_center: "torch.Tensor",
    current_viewdir: "torch.Tensor",
    radius: float,
    sigma: float,
) -> "torch.Tensor":
    """Pure, device-agnostic spatial pose-window weighting (Part B core).

    Given stacked candidate camera ``centers`` (N,3) and unit-ish world-space
    ``viewdirs`` (N,3), and the current camera's ``current_center`` (3,) /
    ``current_viewdir`` (3,), return a per-candidate weight (N,) combining:

      * a spatial pose-sphere: candidates with ``dist >= radius`` get weight 0,
      * a distance falloff ``exp(-dist / sigma)``,
      * a directional covisibility term ``max(0, dir . current_dir)``.

    Cameras that are near AND looking the same way score highest; far or
    opposite-facing cameras score ~0. No CUDA / no in-place ``.cuda()`` so this
    is unit-testable on CPU tensors.
    """
    if centers.numel() == 0:
        return centers.new_zeros((0,))
    sigma = max(float(sigma), 1e-6)
    diff = centers - current_center.reshape(1, 3)
    dist = torch.linalg.norm(diff, dim=1)
    within = (dist < float(radius)).to(centers.dtype)
    falloff = torch.exp(-dist / sigma)
    # Directional covisibility: normalize defensively (dirs may not be unit).
    cd = current_viewdir.reshape(3)
    cd = cd / torch.clamp(torch.linalg.norm(cd), min=1e-8)
    vn = viewdirs / torch.clamp(torch.linalg.norm(viewdirs, dim=1, keepdim=True), min=1e-8)
    cos = torch.clamp((vn * cd.reshape(1, 3)).sum(dim=1), min=0.0)
    return within * falloff * cos


def _external_seed_to_pcd(
    xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    voxel_size: float,
    max_points: int,
):
    """Convert an external SfM seed cloud (e.g. RealityScan's sparse point
    cloud) into a BasicPointCloud, ready for create_from_pcd().

    Pure/CPU-only so it's unit-testable without a full StreamingScene. Reuses
    StreamingScene._voxel_centroid_cluster's exact voxel-centroid algorithm
    (same cell hashing + mean-of-cluster averaging) rather than a separate
    downsampling implementation.
    """
    from eligsir.utils.graphics_utils import BasicPointCloud

    pts = np.asarray(xyz, dtype=np.float32)
    cols = np.asarray(rgb, dtype=np.float32) if rgb is not None else np.full_like(pts, 0.5)

    if voxel_size > 0.0 and pts.shape[0] > 0:
        pts, cols = StreamingScene._voxel_centroid_cluster(pts, cols, voxel_size)

    if max_points and pts.shape[0] > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts, cols = pts[idx], cols[idx]

    return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts, dtype=np.float32))


_RESOLUTION_ARGPARSE_DEFAULT = -1  # arguments/__init__.py ModelParams._resolution default


def resolve_streaming_resolution_pin(args, argv: Optional[List[str]] = None):
    """Decide whether StreamingScene should pin ``args.resolution`` to native (1).

    Pure/CPU-only (no torch, no StreamingScene construction) so it is directly
    unit-testable -- see ``offload_holdout_image`` above for the same
    module-level-on-purpose rationale.

    Background (issue #584): the pin below exists so the internal ``loadCam``
    scale used by streaming stays NATIVE by default -- with the argparse
    default ``args.resolution=-1``, ``loadCam`` auto-caps RGB at 1600px while
    depth insertion stays native, causing depth-supervision misalignment on
    >1600px sensors (commit 7690ba83). ``--streaming_resolution`` is the
    correct, dedicated streaming downscale knob (rgb+depth+intrinsics scaled
    together, upstream of ``loadCam``); this pin only governs ``loadCam``'s
    own internal divisor.

    The pin must still yield to a genuine user request for a non-default
    ``resolution``, but "did the user ask for a non-default value" cannot be
    answered by scanning ``sys.argv`` alone: ``--config <file>.json`` (or a
    YAML run-profile) can set ``args.resolution`` on the parsed namespace
    without the literal ``--resolution``/``-r`` flag ever appearing in
    ``sys.argv`` (`eligsir.cli.config.load_cli_config`, merged in
    ``train.py``'s ``__main__`` before any Scene is constructed) -- that
    config/profile value was previously silently clobbered back to 1 with no
    warning. Fixed by treating ANY value on ``args.resolution`` that already
    differs from the argparse default (``-1``) as an explicit request,
    regardless of where it came from, in addition to (not instead of) the
    original argv scan -- the argv scan is kept only to preserve the one
    narrow pre-existing edge case it alone can distinguish: an explicit
    ``--resolution -1``/``-r -1`` on the command line, which asks for the
    same numeric value as the default but is still a real user request (and
    was honored, not pinned, before this fix).

    Returns ``(effective_resolution, pinned, source)``:
      - ``effective_resolution``: the value ``args.resolution`` should end up
        at (native ``1`` when pinned, otherwise the value already on ``args``
        unchanged).
      - ``pinned``: whether the native-1 pin was applied.
      - ``source``: ``"cli"`` (explicit ``--resolution``/``-r`` in argv),
        ``"config_or_profile"`` (non-default value present on ``args`` with
        no matching argv flag -- e.g. ``--config``/run-profile), or
        ``"default"`` (no explicit request found anywhere; the pin applies).
    """
    import sys as _sys

    if argv is None:
        argv = _sys.argv[1:]
    explicit_cli = any(
        a in ("--resolution", "-r") or a.startswith(("--resolution=", "-r="))
        for a in argv
    )
    # Literal -1 (not the module constant) deliberately: scripts/audit_getattr_fallbacks.py
    # scans getattr(args, <flag>, <fallback>) sites for literal-vs-parser-default
    # drift, and a Name-node fallback is invisible to that diff (see
    # tests/test_getattr_fallback_freshness.py's _KNOWN_NONLITERAL_SITES). This
    # literal must stay equal to _RESOLUTION_ARGPARSE_DEFAULT / the real
    # ModelParams._resolution default (-1) -- the assert below pins that.
    assert _RESOLUTION_ARGPARSE_DEFAULT == -1
    current = getattr(args, "resolution", -1)
    if explicit_cli:
        return current, False, "cli"
    if current != _RESOLUTION_ARGPARSE_DEFAULT:
        return current, False, "config_or_profile"
    return 1, True, "default"


def offload_holdout_image(args, cam) -> bool:
    """Page a single holdout camera's GT image back to CPU RAM (issue #398).

    A holdout Camera is built EAGERLY (`_frame_to_camera(frame,
    load_image=True)`) -- deliberately, because a live/bag source releases a
    frame's pixels as the cursor advances, so a holdout's GT cannot be
    re-decoded later at report time -- and `Camera.__init__` allocates
    `original_image` on `data_device` ("cuda"). Nothing then evicted it: the
    hot-window eviction in `ingest_next_frame` only walks `train_cameras`. So
    every FUTURE evaluation image sat in VRAM for the whole mapping phase,
    growing as O(n_holdouts x W x H) with nothing bounding it (993 holdouts at
    960x720 float32 is several GiB of pixels nothing reads until report time).

    Holdout pixels are read ONLY inside an evaluation pass, and every such path
    fetches them through a helper that uploads a transient per-camera copy
    (`utils.comparison_report._gt_image`, plus the inline `.to("cuda")`/
    `.cuda()` call sites in `_run_test_eval` / `_render_streaming_snapshot`), so
    a CPU-resident tensor is read-compatible with all of them and the upload
    stays O(1) in the number of holdouts.

    Gated on the same `--streaming_offload_camera_images` flag as the train
    hot-window eviction, so `--no-streaming_offload_camera_images` reproduces
    the pre-#398 residency behavior exactly.

    Module-level (not a method) on purpose: the ingest and resume paths that
    call it are both exercised by unbound-method-on-a-fake-self test harnesses,
    which would otherwise each need a new stub attribute.

    Returns True if an offload was attempted (flag on and an image present).
    """
    if not bool(getattr(args, "streaming_offload_camera_images", True)):
        return False
    from scene.cameras import offload_camera_image
    return offload_camera_image(cam)


class StreamingScene:
    def __init__(
        self,
        args,
        gaussians,
        frame_source=None,
        resolution_scale: float = 1.0,
        *,
        canonical_initial_frames=None,
    ):
        if canonical_initial_frames is not None and frame_source is not None:
            raise ValueError(
                "canonical initial frames cannot be combined with a frame source"
            )
        if canonical_initial_frames is None and frame_source is None:
            raise ValueError("StreamingScene requires initial frames or a frame source")
        self.args = args
        self._device_transfer_owner = None
        # Pin the internal loadCam scale to NATIVE for streaming. With the default
        # args.resolution=-1, loadCam auto-caps RGB at 1600px while depth insertion
        # stays native -> depth-supervision misalignment on >1600px sensors, and it
        # forces the slow PIL-resize fallback. `streaming_resolution` is the proper
        # streaming downscale knob (rgb+depth+intrinsics scaled together, upstream).
        # Respect an explicit --resolution/-r override (CLI) OR a non-default
        # args.resolution already set some other way (--config JSON, YAML run
        # profile -- see resolve_streaming_resolution_pin's docstring, issue
        # #584: neither is visible to a sys.argv-only scan). Offline COLMAP
        # (Scene, not StreamingScene) is unaffected either way.
        _requested_resolution = getattr(args, "resolution", -1)  # see literal-vs-parser-default note above
        _effective_resolution, _pinned, _resolution_source = resolve_streaming_resolution_pin(args)
        print(
            f"[StreamingScene] resolution pin: requested={_requested_resolution} "
            f"effective={_effective_resolution} pinned={_pinned} source={_resolution_source}",
            flush=True,
        )
        if _pinned:
            try:
                args.resolution = _effective_resolution
            except Exception:
                pass
        self.gaussians = gaussians
        self.resolution_scale = resolution_scale
        self.model_path = args.model_path
        # External SfM seed cloud (e.g. RealityScan's sparse point cloud) --
        # set via set_external_seed(), consumed by _build_pcd_from_frames().
        # None for every source that doesn't provide one (TUM/ScanNet/generic
        # RGBD/ROS bags), so the depth-backprojection/random-scatter bootstrap
        # is unchanged for them.
        self._external_seed: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Owned by KeyframeStore (Wave 1 step 9,
        # eligsir/mapping/stores/keyframe_store.py): camera bookkeeping
        # (train_cameras, camera-position/viewdir caches, test/holdout
        # cameras, current_camera, cameras_extent, holdout/guard index
        # sets) and the five keyframe-admission counters, all property-
        # backed aliases onto this one instance below and further down --
        # no external API change from this extraction. Constructed first
        # (before any of those attributes are touched) since
        # train_cameras/cameras_extent etc. are used starting immediately
        # below.
        from eligsir.mapping.stores.keyframe_store import KeyframeStore
        self._keyframe_store = KeyframeStore()
        # CPU-cached camera centers (one .cpu() per camera) so update_cameras_extent
        # stays O(N) total instead of O(N^2) GPU syncs across the stream.
        # World-space optical axes (camera +Z in world) per camera, index-aligned
        # with _camera_centers_np / train_cameras. Populated lazily by the spatial
        # pose-window path only (see _ensure_camera_caches_np); kept empty when
        # streaming_spatial_window_enabled is off so the hot extent-update path
        # keeps its sync-free GPU cache. Previously READ by _spatial_local_cameras
        # but never assigned -> AttributeError when the window was enabled.
        self._extent_update_counter: int = 0
        # W3-48: outlier-rejection staging state for update_cameras_extent()'s
        # growth-ceiling+corroboration gate -- see that method's docstring.
        self._extent_pending_growth: Optional[float] = None
        self._extent_pending_corroborations: int = 0
        self._extent_pending_driver_idx: Optional[int] = None
        self._extent_outlier_rejections_total: int = 0
        # Issue #1135: count of RESERVED HOLDOUT frames rejected by the
        # `_odom_valid is False` guard in `ingest_next_frame` below -- i.e.
        # holdouts whose n_test/eval-camera reservation was knowingly NOT
        # honored because their pose is a stale/copied one (see that guard's
        # comment for the mechanism). Plain counter, not routed through
        # KeyframeStore, since it is read the same way
        # `frames_dropped_by_inner_queue` already is elsewhere in this
        # codebase: `getattr(streaming_scene, "_stale_pose_holdout_rejected",
        # 0)` from comparison_report.py, with no other consumer needing it
        # threaded through every write_post_training_report() call site.
        self._stale_pose_holdout_rejected: int = 0
        # Full dense ingested trajectory (every frame that reaches a real Camera,
        # i.e. passed the pose gate, regardless of keyframe-admission outcome).
        # Poses only, no images -- enables a post-hoc "dense video" render mode
        # over every source frame, not just the admitted keyframes kept in
        # train_cameras. See docs/superpowers/plans render-report plan, Part A.
        # Owned by TrajectoryStore (Wave 1 step 9,
        # eligsir/mapping/stores/trajectory_store.py); _dense_trajectory/
        # _dense_trajectory_flushed below are property-backed aliases onto it
        # for external callers (train_streaming.py, tests) -- no external API
        # change from this extraction.
        from eligsir.mapping.stores.trajectory_store import TrajectoryStore
        self._trajectory_store = TrajectoryStore()
        # Owned by ReplayViewStore (Wave 1 step 9,
        # eligsir/mapping/stores/replay_view_store.py): replay ring
        # buffer, global reservoir, partition-focus priority weighting, and
        # consolidation debt-weight cache. _priority_frame_indices/
        # _priority_frame_weights/_replay_buffer/_global_reservoir/
        # _consol_debt_* below and further down are property-backed aliases
        # onto this one instance -- no external API change.
        #
        # Partition-focus sampling (2026-07-27): source frame indices whose
        # partition was flagged as a statistical outlier (under-reconstructed
        # relative to the rest of THIS scene, via utils.partition_health.
        # detect_outlier_partitions -- see train_streaming.py's psnr-tick
        # call site, which owns the registry/partition_id resolution and
        # just hands this scene the concrete resulting indices). Empty until
        # the first health check runs / when the feature is off.
        #
        # Per-index sampling weight for partition-focus priority cameras
        # (source_frame_index -> weight, roughly proportional to how bad
        # that partition's PSNR deficit is). set_priority_frame_indices()
        # (uniform weight 1.0) is the simple back-compat entry point;
        # set_priority_frame_weights() is the graduated one train_streaming.
        # py's psnr-tick call site uses so a badly under-reconstructed
        # partition gets sampled more aggressively than a mildly weak one.
        from eligsir.mapping.stores.replay_view_store import ReplayViewStore
        self._replay_view_store = ReplayViewStore()
        # Tiered frame cache (Partition-Focus v2): three tiers --
        #   1. active   = self.train_cameras (populated by keyframe admission).
        #   2. cold     = _inactive_frame_cache: rejected-at-admission (or
        #      pregate-rejected) frames that fall within
        #      --streaming_cold_window source-index steps of the most
        #      recently admitted keyframe (i.e. genuinely "surrounding" an
        #      admitted keyframe, not just any rejected frame anywhere in
        #      the trajectory). Small, bounded LRU via
        #      --streaming_inactive_cache_size -- meant for fast, likely-
        #      relevant promotion candidates.
        #   3. icecold  = _icecold_frame_cache: the big, CPU-RAM-resident,
        #      long-term archive. Fed from two sources: (a) frames evicted
        #      from the cold tier's LRU (i.e. "dropped out and never
        #      called for a while" rather than discarded outright), and
        #      (b) frames belonging to train_cameras demoted out via
        #      demote_partitions() because their partition is doing
        #      unusually well relative to the rest of the scene. Much
        #      larger bound than the cold tier (--streaming_icecold_cache_
        #      size) -- this is the tier that keeps most of the original
        #      input accessible if a partition later needs it back.
        # Both cache tiers store StreamingRGBDFrame records, never a decoded
        # /GPU-resident Camera -- _frame_to_camera(frame) rebuilds a full
        # Camera on demand. For file-backed sources (RGBDSequence/TUM/
        # ScanNet-path) that record really is near-free path-only metadata.
        # ros2_live (and ScanNet's .sens in-memory path) have no seekable
        # file to re-decode from later, so their StreamingRGBDFrame already
        # carries the decoded pixels (_rgb_np/_depth_np, or _rgb_bytes/
        # _depth_bytes) -- see _frame_carries_pixel_data(). Those use the
        # separate, much smaller --streaming_live_inactive_cache_size /
        # --streaming_live_icecold_cache_size bounds instead. This is also
        # why demote_partitions needs no explicit GPU->CPU image transfer at
        # all: archiving the frame and dropping the Camera reference lets
        # the GPU tensor free via normal refcounting/the CUDA caching
        # allocator, rather than needing a synchronous per-frame
        # offload_camera_image() call (which is what the hot-window
        # recency eviction path still uses, since IT needs to keep the
        # Camera object itself, just page its image out).
        # Without a supply side (tiers 2/3), a priority weight/set alone is
        # close to a no-op on a partition with few/no admitted frames --
        # promote_from_cache() is what actually gives partition-focus
        # sampling new information to draw on.
        # Owned by FrameArchive (Wave 1 step 9,
        # eligsir/mapping/stores/frame_archive.py); _inactive_frame_cache/
        # _icecold_frame_cache below are property-backed aliases onto it --
        # no external API change from this extraction.
        from eligsir.mapping.stores.frame_archive import FrameArchive
        self._frame_archive = FrameArchive()
        self._active_source_indices: "set[int]" = set()
        self._active_cam_by_idx: "dict[int, object]" = {}
        # Optional FrameObservationDB (eligsir.mapping.observation_db) --
        # set post-construction via set_observation_db() once train_streaming.
        # py builds it (after StreamingScene itself is constructed). When
        # present, every tiered-cache transition (_cache_inactive_frame ->
        # COLD_CACHE, _archive_to_icecold -> ICECOLD_CACHE, promote_from_cache
        # -> MAPPING_KEYFRAME) calls db.set_buffer_stage() so the DB's record
        # of a frame's current location tracks the scene's actual tiers
        # instead of freezing at whatever stage the frame was first recorded
        # with. None (default) makes every call a no-op -- DB tracking is
        # optional, e.g. tests/lightweight callers that never construct one.
        #
        # Owned by ObservationStore (Wave 1 step 9,
        # eligsir/mapping/stores/observation_store.py) along with the
        # _obs_*/_iq_*/_scene_complexity_cache attributes set up further
        # below -- observation_db and those become property-backed aliases
        # onto this one instance; no external API change from this
        # extraction.
        from eligsir.mapping.stores.observation_store import ObservationStore
        self._observation_store = ObservationStore(
            obs_voxel=float(getattr(args, "streaming_insert_voxel_size", 0.02))
        )
        # Ring buffer of the last streaming_replay_buffer cameras -- owned
        # by ReplayViewStore, see the comment above where it's constructed.
        self._source_idx = 0
        self._frame_source = frame_source
        if canonical_initial_frames is not None:
            self._lazy_frames = False
            self._all_frames = list(canonical_initial_frames)
            self._frame_count = len(self._all_frames)
        else:
            self._lazy_frames = bool(getattr(frame_source, "lazy_frames", False))
            self._all_frames = None if self._lazy_frames else list(frame_source)
            self._frame_count = (
                len(frame_source) if self._lazy_frames else len(self._all_frames)
            )

        # P5: decode cache + optional background read-ahead. Offline replay is
        # single-thread synchronous, so CPU PIL decode stalls the GPU mapper, and
        # each frame is decoded 2-3x per iteration. Enable only for non-lazy offline
        # sources (TUM/ScanNet/RGBD/Replica); never for lazy/live ROS2 (which has
        # its own pump). Deterministic: keyed by frame.index, same order.
        self._decode_cache = None
        if args.streaming_prefetch and not self._lazy_frames and self._frame_count > 0:
            from eligsir.utils.streaming_frames import FrameDecodeCache, set_decode_cache
            ahead = int(getattr(args, "streaming_prefetch_depth", 3))
            n_workers = max(1, int(getattr(args, "streaming_prefetch_workers", 4)))
            self._decode_cache = FrameDecodeCache(maxsize=ahead + 3)
            set_decode_cache(self._decode_cache)
            if ahead > 0:
                self._decode_cache.start_prefetch(
                    lambda i: self._all_frames[i], self._frame_count, ahead,
                    n_workers=min(n_workers, ahead),
                )
            print(
                f"[streaming] decode prefetch ON (depth={ahead}, {self._frame_count} frames, "
                f"workers={min(n_workers, ahead) if ahead > 0 else 0})",
                flush=True,
            )
        else:
            # Defensive: clear any stale module-global cache from a prior run in the
            # same process so this scene's frames (unstamped) never route through it.
            from eligsir.utils.streaming_frames import set_decode_cache
            set_decode_cache(None)

        # Real-holdout test set: deterministic, admission-independent frame indices
        # reserved for evaluation (never trained / inserted). Computed up-front from
        # the source trajectory so the test set is identical across admission modes.
        # _guard_indices are neighbours withheld from training to widen the parallax
        # gap to each holdout (not evaluated). None => live/lazy source: fall back to
        # online every-Nth selection (no pose-gap filter) in ingest_next_frame.
        self._designate_test_holdouts()
        # Register the deterministic real-holdout set with a bounded lazy
        # frame source (e.g. ROS2LiveStreamingFrameSource) so its ring-buffer
        # eviction never reclaims a holdout frame the checkpoint-resume path
        # (_rematerialize_test_cameras_after_resume) may need to rebuild a
        # test Camera from later. See WS-D1 / "P0: live source buffering is
        # unbounded" -- no-op for sources without this optional method.
        if self._holdout_indices:
            _protect = getattr(self._frame_source, "set_protected_indices", None)
            if callable(_protect):
                _protect(self._holdout_indices)

        # Observed-surface accumulator for scene expressiveness (SV). Built
        # incrementally as frames arrive (depth still in hand) so it gives FULL
        # coverage for lazy rosbag/mcap sources too — not just the few frames whose
        # depth is still cached at report time. Keys are voxel(xN) x normal-bin at
        # multiple scales; position-only voxel sets feed the box-counting dimension.
        # Owned by ObservationStore (Wave 1 step 9,
        # eligsir/mapping/stores/observation_store.py); the _obs_*/_iq_*/
        # _scene_complexity_cache/observation_db attributes below (and
        # further down in __init__ for observation_db) are property-backed
        # aliases onto it -- no external API change from this extraction.
        #
        # Input-quality accumulators (sensor-noise vs tracking-error split).
        #  - sensor noise: running stats of per-frame PCA plane residual (intra-frame,
        #    pose-free), sampled on every Nth ingested frame to bound cost.
        #  - thickness: per-voxel diagonal Welford over subsampled world points
        #    (capped dict) -> RMS smallest-axis spread = combined surface thickness.
        #  - tracking: aggregated at report time from each frame's _odom_stats.
        # (self._observation_store is constructed once, above, before
        # _designate_test_holdouts() -- it used to be constructed a second
        # time here too, silently discarding the first, identically-built
        # instance; removed 2026-08-06.)

        # Persistent occupancy hash for incremental insertion. Owned by
        # OccupancyIndex (Wave 1 step 9,
        # eligsir/mapping/stores/occupancy_index.py); _occupancy_keys/
        # _occupancy_keys_gpu/occupancy_voxel_size below are property-backed
        # aliases onto it -- no external API change from this extraction.
        # occupancy_keys_gpu is an optional torch-resident mirror of the
        # sorted occupancy key array, maintained on the Gaussian device;
        # None until first torch update.
        from eligsir.mapping.stores.occupancy_index import OccupancyIndex
        self._occupancy_index = OccupancyIndex(voxel_size=0.02)
        self._surface_cell_index = SurfaceCellIndex(voxel_size=self.occupancy_voxel_size)

        # H8: warmup counter — steps since the last new frame arrived
        self._steps_since_new_frame: int = 0

        # H10: permanent global keyframe reservoir (never evicted, sampled
        # for replay) -- owned by ReplayViewStore, see the comment above
        # where it's constructed.
        # _n_train_frames_ingested (count of train frames excl. holdout) and
        # _bootstrap_count (set by initialize_from_frames, used by keyframe
        # window logic) -- owned by KeyframeStore, see the comment above
        # where it's constructed.

        # Stratified sampling support (Component D)
        # Hard-frame heap: list of (loss, camera) — capped at streaming_hard_frame_history
        self._hard_frames: List = []
        # Covisible sets: {cam_uid -> List[cam]} of recently-covisible cameras
        self._last_gaussian_ids: Optional[torch.Tensor] = None  # gaussian_ids from last render
        self._covisible_cache: List = []  # flat list of covisible cameras (refreshed periodically)

    def bind_device_transfer_owner(self, owner) -> None:
        """Bind the sole production pixel-transfer owner before camera creation."""

        if owner is None or not callable(getattr(owner, "prefetch_cameras", None)):
            raise TypeError("device transfer owner must provide prefetch_cameras")
        if self._device_transfer_owner is not None and self._device_transfer_owner is not owner:
            raise RuntimeError("StreamingScene already has a different transfer owner")
        self._device_transfer_owner = owner

    def _prepare_camera_for_render_owned(self, camera, *, device: str) -> None:
        from scene.cameras import prepare_camera_for_render

        owner = getattr(self, "_device_transfer_owner", None)
        if owner is not None:
            if getattr(camera, "has_pending_image", False):
                camera.materialize_pending_image()
            owner.prefetch_cameras((camera,))
        prepare_camera_for_render(camera, device=device)

    @staticmethod
    def _pose_gap(c2w_a: np.ndarray, c2w_b: np.ndarray) -> tuple:
        """Return (translation_m, rotation_deg) between two camera-to-world poses."""
        t = float(np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]))
        r_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]
        cos = (np.trace(r_rel) - 1.0) * 0.5
        ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return t, ang

    def _designate_test_holdouts(self) -> None:
        """Pre-compute the deterministic, admission-independent test holdout set.

        Every ``streaming_eval_hold``-th source frame (past the bootstrap range) is
        a candidate. A candidate is kept as a real holdout only if its viewpoint
        differs from the nearest *trained* neighbour (just outside an optional guard
        band) by at least ``streaming_eval_min_baseline_m`` (translation) OR
        ``streaming_eval_min_angle_deg`` (rotation); otherwise it is demoted back to
        train (a near-duplicate is a trivial interpolation view). If the filter would
        empty the test set, it falls back to the unfiltered every-Nth candidates so a
        run always has a non-empty test split.
        """
        eval_hold = int(self.args.streaming_eval_hold or 0)
        if eval_hold <= 0 or self._lazy_frames or not self._all_frames:
            # eval disabled, or live/lazy source (future poses unknown): leave
            # _holdout_indices=None -> ingest falls back to online every-Nth.
            return

        frames = self._all_frames
        n = len(frames)
        bootstrap = min(int(getattr(self.args, "streaming_initial_frames", 5)), n)
        guard = max(0, int(getattr(self.args, "streaming_eval_guard", 0)))
        min_t = float(getattr(self.args, "streaming_eval_min_baseline_m", 0.03))
        min_a = float(getattr(self.args, "streaming_eval_min_angle_deg", 2.0))

        def _c2w(p):
            c = getattr(frames[p], "c2w", None)
            if c is None:
                return None
            c = np.asarray(c, dtype=np.float32)
            return c if (c.shape == (4, 4) and np.isfinite(c).all()) else None

        candidates = [p for p in range(n)
                      if frames[p].index % eval_hold == 0 and frames[p].index >= bootstrap]
        cand_set = set(candidates)

        holdouts: set = set()
        guards: set = set()
        for p in candidates:
            cp = _c2w(p)
            # Guard band: withhold up to `guard` immediate neighbours from training.
            g_lo = [q for q in range(p - 1, max(p - guard - 1, -1), -1)]
            g_hi = [q for q in range(p + 1, min(p + guard + 1, n))]
            # Nearest trainable neighbour = first frame outside guard band that is
            # neither a candidate nor a guard frame.
            gap_t = gap_a = float("inf")
            if cp is not None:
                for q in range(p - guard - 1, -1, -1):
                    if q in cand_set:
                        continue
                    cq = _c2w(q)
                    if cq is not None:
                        gap_t, gap_a = self._pose_gap(cp, cq)
                        break
                for q in range(p + guard + 1, n):
                    if q in cand_set:
                        continue
                    cq = _c2w(q)
                    if cq is not None:
                        t2, a2 = self._pose_gap(cp, cq)
                        gap_t, gap_a = min(gap_t, t2), min(gap_a, a2)
                        break
            keep = (cp is None) or (gap_t >= min_t) or (gap_a >= min_a)
            if keep:
                holdouts.add(frames[p].index)
                guards.update(frames[q].index for q in (g_lo + g_hi))
            # else: demoted to train (no guard band withheld for it)

        if not holdouts:
            # Filter removed everything (e.g. near-static sequence): fall back to the
            # raw every-Nth split so the run still has a test set.
            holdouts = {frames[p].index for p in candidates}
            guards = set()
            if candidates:
                print(f"[streaming] eval holdout pose-gap filter removed all "
                      f"{len(candidates)} candidates; falling back to unfiltered "
                      f"every-{eval_hold} split.", flush=True)

        # Guard frames must never also be holdouts.
        guards -= holdouts
        self._holdout_indices = holdouts
        self._guard_indices = guards
        print(f"[streaming] eval holdouts: {len(holdouts)} test frames "
              f"(every-{eval_hold}, min_gap {min_t:.3f} m / {min_a:.1f} deg, "
              f"guard {guard}); {len(guards)} guard frames withheld from training.",
              flush=True)

    def eval_metadata(self) -> dict:
        """Self-documenting record of the holdout policy for report.json.

        The ``holdout_hash`` lets downstream comparison tooling confirm two runs
        used the *identical* test SELECTION RULE (the comparability contract) --
        NOT the realized set of test cameras a particular run happened to reach
        (that is ``realized_holdout_hash``, computed separately by
        ``utils/comparison_report.py`` from the actual materialized test
        cameras; a run's own ``eval_metadata()`` here never sees that value).

        issue #745 (2026-08-15): the live/lazy branch below used to hash the
        materialized ``self._test_cameras`` SET under the ``holdout_hash``
        name -- a genuine rule fingerprint on the offline (precomputed)
        branch, but on a live/bag run that made ``holdout_hash`` vary with
        how many frames the run had ingested by report time, i.e. the same
        measurement as ``realized_holdout_hash`` computed twice under two
        different names. That was the root cause of the 2026-07-08 floor3
        finding of three distinct ``holdout_hash`` values across four arms
        on the identical bag source. The live branch now hashes the RULE
        (eval_hold + the ordinal definition + guard/pose-gap parameters +
        source identity) instead, so ``holdout_hash`` means the same thing
        on both branches. See ``docs/architecture/
        2026-08-15-issue745-live-holdout-rule-hash.md`` for the full design
        and the acknowledged pre-fix-report backward-compatibility gap:
        ANY report.json written before this fix that carries
        ``holdout_selection == "online_every_nth_hashed"`` recorded the OLD
        realized-set meaning under ``holdout_hash`` and must not be
        reinterpreted as a rule hash -- the new live branch never writes
        that selection string again (it writes
        ``online_every_nth_ordinal_rule`` or, when holdout is disabled,
        the unchanged ``online_every_nth``), so the two are distinguishable
        by ``holdout_selection`` alone.

        The offline (precomputed) branch below is unchanged byte-for-byte
        from pre-#745 behavior -- it never had either defect (the full
        trajectory, and therefore the full holdout index set, is known up
        front)."""
        import hashlib
        eval_hold = int(self.args.streaming_eval_hold or 0)
        eval_min_baseline_m = float(getattr(self.args, "streaming_eval_min_baseline_m", 0.03))
        eval_min_angle_deg = float(getattr(self.args, "streaming_eval_min_angle_deg", 2.0))
        eval_guard = int(getattr(self.args, "streaming_eval_guard", 0))

        if self._holdout_indices is not None:
            # Offline/precomputed branch -- unchanged from pre-#745 behavior.
            idx = sorted(self._holdout_indices)
            selection = "precomputed"
            holdout_hash = None
            if idx:
                holdout_hash = hashlib.sha1(
                    ",".join(str(i) for i in idx).encode("utf-8")
                ).hexdigest()[:16]
            n_holdout = len(idx)
        else:
            # Live/lazy branch (issue #745 fix): hash the RULE, not the
            # realized set. The rule is fully determined by eval_hold, the
            # ordinal definition used to apply it (see
            # StreamingScene.ingest_next_frame's `source_ordinal`/`frame.index`
            # fallback and eligsir/io/sources/datasets/ros2_live.py's
            # `_ordinal_for_timestamp`), the guard/pose-gap parameters, and
            # which source trajectory this is -- none of which depend on how
            # many frames a given run actually ingested, so this hash is
            # available and stable even for a mid-run checkpoint report that
            # hasn't reached any holdout frame yet.
            n_holdout = sum(
                1 for c in self._test_cameras
                if getattr(c, "_streaming_frame", None) is not None
            )
            if eval_hold > 0:
                # issue #745 round 2 (2026-08-15): the ordinal-definition
                # version is now read from the frame source itself rather
                # than hardcoded, since ROS2LiveStreamingFrameSource can use
                # either its primary mechanism ("received_seq_v1", a real
                # monotonic message-sequence counter -- see that class's
                # `ordinal_definition_version` property) or, only when a
                # frame source doesn't populate `received_seq` at all, the
                # "ts_quantized_v1" timestamp-quantization fallback. Two
                # runs using different ordinal definitions must never
                # collide into the same rule fingerprint -- folding the
                # actual mechanism into the hash guarantees that. Falls back
                # to "ts_quantized_v1" (the pre-round-2 default) when
                # `self._frame_source` doesn't expose the property at all
                # (e.g. a non-ROS2-live lazy source, or a test harness).
                ordinal_version = getattr(
                    getattr(self, "_frame_source", None),
                    "ordinal_definition_version",
                    "ts_quantized_v1",
                )
                source_id = str(getattr(self.args, "source_path", "") or "")
                # issue #1302: delegates to utils.holdout_rule.online_holdout_rule_hash
                # -- same payload/hash this replaced, now shared with
                # eligsir-ros2's LiveHoldoutCollector-based reports so a
                # `holdout_hash` comparison is meaningful across both paths.
                holdout_hash = online_holdout_rule_hash(
                    eval_hold=eval_hold,
                    ordinal_definition_version=ordinal_version,
                    eval_guard=eval_guard,
                    eval_min_baseline_m=eval_min_baseline_m,
                    eval_min_angle_deg=eval_min_angle_deg,
                    source_id=source_id,
                )
                selection = "online_every_nth_ordinal_rule"
            else:
                holdout_hash = None
                selection = "online_every_nth"
        return {
            "eval_hold": eval_hold,
            "eval_min_baseline_m": eval_min_baseline_m,
            "eval_min_angle_deg": eval_min_angle_deg,
            "eval_guard": eval_guard,
            "n_holdout": n_holdout,
            "n_guard": len(self._guard_indices),
            "holdout_hash": holdout_hash,
            "holdout_selection": selection,
        }

    def update_cameras_extent(self) -> None:
        """Grow cameras_extent to encompass all arrived camera positions (never
        shrinks) -- gated by an outlier-rejection ceiling (W3-48, 2026-08-10).

        Root-caused in docs/architecture/2026-08-10-w3-30-init-seed-bootstrap-
        confidence-gate.md as the real floor2 scale-explosion amplifier: a
        single poorly-converged pose used to permanently inflate
        cameras_extent by 5-10x with zero outlier rejection, feeding
        densification scale/MCMC noise for the whole map for the rest of the
        run. A recomputed extent that jumps beyond
        streaming_extent_growth_ceiling_multiplier times the current extent is
        now withheld (the camera itself is still fully used for training --
        only the extent bookkeeping is gated) until it is corroborated by
        streaming_extent_outlier_corroboration further recompute windows that
        are each driven by a DIFFERENT camera than the one before.

        The "different camera" requirement (not just "still a large extent")
        matters because the position cache this method accumulates from
        (``_camera_centers_gpu``) is never pruned -- a single outlier camera's
        position stays in it forever once appended, so a naive "is the extent
        still large" check would trivially keep re-corroborating itself off
        that ONE stale cached point every subsequent window, regardless of
        whether any new evidence actually arrived. Requiring the argmax
        (extent-driving) camera index to change between corroborating windows
        means acceptance needs actual new frames independently supporting a
        large extent. A one-off pose glitch stays the sole driver forever (no
        new camera goes further than it) and is never corroborated; genuine
        sustained scene growth (e.g. a camera continuing down a long
        corridor) keeps handing the "furthest camera" title to increasingly
        recent frames, satisfying the distinct-driver requirement naturally.
        """
        if not self.train_cameras:
            return
        try:
            # Keep camera centers as GPU tensors and append new ones WITHOUT a
            # host sync (the old `.cpu().numpy()` here ran on every ingested
            # frame and stalled the GPU pipeline). The scalar extent — which only
            # feeds densification scale and never shrinks — is materialized with a
            # single `.item()` at most every 8 frames.
            import torch as _torch
            cache = self._camera_centers_gpu
            if len(cache) > len(self.train_cameras):  # defensive: cache desync
                cache.clear()
            for c in self.train_cameras[len(cache):]:
                cache.append(c.camera_center.detach().reshape(3))
            self._extent_update_counter += 1
            if self._extent_update_counter % 8 != 0 and self.cameras_extent > 1.0:
                return
            centers = _torch.stack(cache)  # [M, 3] on device
            centroid = centers.mean(dim=0)
            dists = _torch.linalg.norm(centers - centroid, dim=1)
            driver_idx = int(dists.argmax().item())
            new_extent = float(dists[driver_idx].item())

            ceiling_mult = float(getattr(self.args, "streaming_extent_growth_ceiling_multiplier", 1.5))
            required_corrob = int(getattr(self.args, "streaming_extent_outlier_corroboration", 2))
            current = self.cameras_extent
            # ceiling_mult<=0 or required_corrob<=0 disables the gate entirely
            # (pre-W3-48 behavior, escape hatch for scenes this ever mistunes).
            if ceiling_mult <= 0 or required_corrob <= 0 or current <= 1.0 or new_extent <= current * ceiling_mult:
                self.cameras_extent = max(current, new_extent, 1.0)
                self._extent_pending_growth = None
                self._extent_pending_corroborations = 0
                self._extent_pending_driver_idx = None
                return

            # new_extent is a suspect jump -- stage it rather than accepting
            # immediately. Only counts as corroboration if a camera OTHER
            # than the previously-staged driver is now the one producing the
            # large extent (see docstring for why "still large" alone isn't
            # sufficient).
            if (
                self._extent_pending_driver_idx is not None
                and driver_idx != self._extent_pending_driver_idx
            ):
                self._extent_pending_corroborations += 1
            else:
                self._extent_pending_corroborations = 1
            self._extent_pending_growth = new_extent
            self._extent_pending_driver_idx = driver_idx

            if self._extent_pending_corroborations >= required_corrob:
                self.cameras_extent = max(current, new_extent, 1.0)
                self._extent_pending_growth = None
                self._extent_pending_corroborations = 0
                self._extent_pending_driver_idx = None
            else:
                self._extent_outlier_rejections_total += 1
                if self._extent_outlier_rejections_total == 1 or self._extent_pending_corroborations == 1:
                    print(
                        f"[streaming] WARNING: update_cameras_extent withheld a suspect "
                        f"jump {current:.2f}m -> {new_extent:.2f}m "
                        f"({new_extent / current:.1f}x, ceiling={ceiling_mult:.1f}x) -- "
                        f"needs {required_corrob} corroborating window(s), have "
                        f"{self._extent_pending_corroborations}. cameras_extent "
                        f"stays at {current:.2f}m this cycle.",
                        flush=True,
                    )
        except Exception:
            pass

    def flush_deferred_extent_update(self) -> None:
        """Variant C: run a single update_cameras_extent() if a batched drain
        deferred one or more per-frame updates. No-op when nothing pending."""
        if getattr(self, "_pending_extent_update", False):
            self._pending_extent_update = False
            self.update_cameras_extent()

    def _xyz_to_keys(self, xyz_np: np.ndarray, voxel_size: float) -> np.ndarray:
        offset = 1000.0
        coords = np.floor((xyz_np + offset) / max(voxel_size, 1e-6)).astype(np.int64)
        return (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]

    def _gpu_occupancy_enabled(self) -> bool:
        return bool(getattr(getattr(self, "args", None), "gpu_occupancy", False))

    def maintain_occupancy_hash(self, voxel_size: float = 0.02):
        """Rebuild the occupancy array from current Gaussian positions."""
        # --gpu_occupancy: keep the occupancy key set on the GPU (single source
        # of truth). Avoids crossing the full Gaussian set to CPU + np.unique on
        # every rebuild. Surface-cell dedup (default off) still uses the numpy
        # index, rebuilt from CPU only when streaming_surface_cell_dedup is set.
        if self._gpu_occupancy_enabled():
            self.occupancy_voxel_size = voxel_size
            self._surface_cell_index = SurfaceCellIndex(voxel_size=voxel_size)
            self.maintain_occupancy_hash_torch(voxel_size)
            if bool(getattr(self.args, "streaming_surface_cell_dedup", False)):
                xyz = self.gaussians.get_xyz.detach().cpu().numpy()
                normals = getattr(self.gaussians, "anchor_normal", None)
                if normals is not None and normals.shape[0] == xyz.shape[0]:
                    self._surface_cell_index.rebuild(xyz, normals.detach().cpu().numpy())
            return
        self.occupancy_voxel_size = voxel_size
        self._surface_cell_index = SurfaceCellIndex(voxel_size=voxel_size)
        xyz = self.gaussians.get_xyz.detach().cpu().numpy()
        if xyz.shape[0] == 0:
            self._occupancy_keys = np.empty(0, dtype=np.int64)
            return
        keys = self._xyz_to_keys(xyz, voxel_size)
        self._occupancy_keys = np.sort(np.unique(keys))
        normals = getattr(self.gaussians, "anchor_normal", None)
        if normals is not None and normals.shape[0] == xyz.shape[0]:
            self._surface_cell_index.rebuild(xyz, normals.detach().cpu().numpy())

    def add_to_occupancy_hash(self, xyz: torch.Tensor):
        """Incrementally merge new points into the sorted occupancy array."""
        if xyz.shape[0] == 0:
            return
        # --gpu_occupancy: merge on-GPU (xyz is already a CUDA tensor at the
        # call sites), avoiding the per-insertion .cpu().numpy() + np.union1d.
        if self._gpu_occupancy_enabled():
            self.add_to_occupancy_hash_torch(xyz)
            return
        new_keys = np.sort(np.unique(self._xyz_to_keys(xyz.detach().cpu().numpy(), self.occupancy_voxel_size)))
        if not hasattr(self, "_occupancy_keys") or self._occupancy_keys.size == 0:
            self._occupancy_keys = new_keys
        else:
            self._occupancy_keys = np.union1d(self._occupancy_keys, new_keys)

    def add_to_surface_cells(self, xyz: torch.Tensor, normals: torch.Tensor):
        if xyz.shape[0] == 0 or normals.shape[0] != xyz.shape[0]:
            return
        self._surface_cell_index.add(
            xyz.detach().cpu().numpy(),
            normals.detach().cpu().numpy(),
        )

    def check_surface_cells(self, xyz_np: np.ndarray, normals_np: np.ndarray) -> np.ndarray:
        if xyz_np.shape[0] == 0 or normals_np.shape[0] != xyz_np.shape[0]:
            return np.zeros(xyz_np.shape[0], dtype=bool)
        return self._surface_cell_index.occupied(xyz_np, normals_np)

    def check_occupancy(self, xyz_np: np.ndarray, voxel_size: float, check_neighbors: bool = True) -> np.ndarray:
        """Check whether points fall into occupied voxels.

        voxel_size must equal self.occupancy_voxel_size; the hash is keyed by that
        size so a mismatched query would silently return wrong results.
        """
        assert abs(voxel_size - self.occupancy_voxel_size) < 1e-9, (
            f"check_occupancy voxel_size={voxel_size} must equal "
            f"occupancy_voxel_size={self.occupancy_voxel_size}"
        )
        # --gpu_occupancy: query against the GPU key set. Accepts numpy or tensor
        # input; returns numpy for numpy callers (the insert path / coverage gate
        # index numpy arrays with the result) and a tensor for tensor callers.
        if self._gpu_occupancy_enabled():
            dev = self.gaussians.get_xyz.device
            if torch.is_tensor(xyz_np):
                occ = self.check_occupancy_torch(xyz_np.to(dev), voxel_size, check_neighbors)
                return occ
            if xyz_np.shape[0] == 0:
                return np.zeros(0, dtype=bool)
            xyz_t = torch.as_tensor(np.ascontiguousarray(xyz_np), dtype=torch.float32, device=dev)
            return self.check_occupancy_torch(xyz_t, voxel_size, check_neighbors).cpu().numpy()
        keys_arr = getattr(self, "_occupancy_keys", None)
        if keys_arr is None or keys_arr.size == 0 or xyz_np.shape[0] == 0:
            return np.zeros(xyz_np.shape[0], dtype=bool)

        offset = 1000.0
        coords = np.floor((xyz_np + offset) / max(voxel_size, 1e-6)).astype(np.int64)
        keys = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]

        occupied = np.isin(keys, keys_arr)

        if check_neighbors and not occupied.all():
            # Pre-generate all 26 neighbour direction vectors once
            dirs = np.array(
                [[dx, dy, dz]
                 for dx in (-1, 0, 1)
                 for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1)
                 if not (dx == 0 and dy == 0 and dz == 0)],
                dtype=np.int64,
            )
            for d in dirs:
                remaining = ~occupied
                if not remaining.any():
                    break
                n_keys = ((coords[remaining, 0] + d[0]) << 42
                          | (coords[remaining, 1] + d[1]) << 21
                          | (coords[remaining, 2] + d[2]))
                occupied[remaining] |= np.isin(n_keys, keys_arr)
        return occupied

    # ------------------------------------------------------------------
    # Torch-resident occupancy hash (GPU). Bit-exact mirror of the numpy
    # versions above; kept side-by-side so either path can be selected via
    # the --gpu_occupancy flag. All methods stay on the input tensor's device
    # and avoid .cpu()/.numpy() round-trips.
    # ------------------------------------------------------------------
    def maintain_occupancy_hash_torch(self, voxel_size: float = 0.02):
        """Rebuild the torch occupancy key tensor from current Gaussians.

        Mirrors maintain_occupancy_hash but keeps the sorted-unique key array
        on the Gaussian device. The numpy _occupancy_keys is left untouched.
        """
        from eligsir.utils.torch_geometry import pack_voxel_keys_torch

        self.occupancy_voxel_size = voxel_size
        xyz = self.gaussians.get_xyz.detach()
        if xyz.shape[0] == 0:
            self._occupancy_keys_gpu = torch.empty(
                0, dtype=torch.int64, device=xyz.device
            )
            return
        keys = pack_voxel_keys_torch(xyz, voxel_size)
        self._occupancy_keys_gpu = torch.unique(keys, sorted=True)

    def add_to_occupancy_hash_torch(self, xyz: torch.Tensor):
        """Incrementally merge new points into the sorted torch key tensor.

        Mirrors add_to_occupancy_hash without any .cpu()/.numpy() transfer.
        """
        from eligsir.utils.torch_geometry import pack_voxel_keys_torch

        if xyz.shape[0] == 0:
            return
        new_keys = torch.unique(
            pack_voxel_keys_torch(xyz.detach(), self.occupancy_voxel_size),
            sorted=True,
        )
        if self._occupancy_keys_gpu is None or self._occupancy_keys_gpu.numel() == 0:
            self._occupancy_keys_gpu = new_keys
        else:
            existing = self._occupancy_keys_gpu.to(new_keys.device)
            self._occupancy_keys_gpu = torch.unique(
                torch.cat([existing, new_keys]), sorted=True
            )

    def check_occupancy_torch(
        self,
        xyz: torch.Tensor,
        voxel_size: float,
        check_neighbors: bool = True,
    ) -> torch.Tensor:
        """Torch port of check_occupancy. Returns a BoolTensor on xyz.device.

        Uses searchsorted against the sorted key tensor + equality test. The
        26-neighbour expansion matches the numpy operator-precedence packing
        exactly: ((c0+dx)<<42) | ((c1+dy)<<21) | (c2+dz).
        """
        assert abs(voxel_size - self.occupancy_voxel_size) < 1e-9, (
            f"check_occupancy_torch voxel_size={voxel_size} must equal "
            f"occupancy_voxel_size={self.occupancy_voxel_size}"
        )
        n = xyz.shape[0]
        keys_arr = self._occupancy_keys_gpu
        if keys_arr is None or keys_arr.numel() == 0 or n == 0:
            return torch.zeros(n, dtype=torch.bool, device=xyz.device)

        sorted_keys = keys_arr.to(xyz.device)
        offset = 1000.0
        vs = max(voxel_size, 1e-6)
        coords = torch.floor((xyz + offset) / vs).to(torch.int64)
        c0 = coords[:, 0]
        c1 = coords[:, 1]
        c2 = coords[:, 2]

        def _isin(query: torch.Tensor) -> torch.Tensor:
            idx = torch.searchsorted(sorted_keys, query)
            idx_clamped = idx.clamp(max=sorted_keys.shape[0] - 1)
            return sorted_keys[idx_clamped] == query

        keys = (c0 << 42) | (c1 << 21) | c2
        occupied = _isin(keys)

        if check_neighbors and not bool(occupied.all()):
            dirs = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]
            for dx, dy, dz in dirs:
                remaining = ~occupied
                if not bool(remaining.any()):
                    break
                n_keys = (
                    ((c0[remaining] + dx) << 42)
                    | ((c1[remaining] + dy) << 21)
                    | (c2[remaining] + dz)
                )
                occupied[remaining] |= _isin(n_keys)
        return occupied

    def iter_all_frames(self):
        """Iterate all source frames regardless of whether the source is lazy or eager."""
        for i in range(self._frame_count):
            yield self._get_source_frame(i)

    def drain_pending_odometry(self) -> None:
        """Force the live-odom producer to compute every source frame."""
        drain = getattr(self._frame_source, "drain_pending", None)
        if callable(drain):
            drain()

    def stop_pending_odometry(self) -> None:
        """Stop background live odometry without computing unseen frames."""
        stop = getattr(self._frame_source, "stop_pending", None)
        if callable(stop):
            stop()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def close_decode_cache(self) -> None:
        """Stop the prefetch thread and detach the module-level decode cache."""
        if self._decode_cache is not None:
            try:
                self._decode_cache.stop()
            except Exception:
                pass
            from eligsir.utils.streaming_frames import set_decode_cache, get_decode_cache
            if get_decode_cache() is self._decode_cache:
                set_decode_cache(None)
            self._decode_cache = None

    def _get_source_frame(self, index: int) -> "StreamingRGBDFrame":
        if self._decode_cache is not None:
            self._decode_cache.note_consumed(index)
        if self._lazy_frames:
            try:
                fr = self._frame_source.get_frame(index)
            except IndexError:
                # Live source exhausted early (idle-timeout / bag end). Clamp
                # _frame_count so has_next_frame() returns False from now on.
                actual = len(self._frame_source) if hasattr(self._frame_source, '__len__') else index
                self._frame_count = min(self._frame_count, actual)
                return None
        else:
            fr = self._all_frames[index]
        # Stamp the source position so load_frame_rgb/depth cache under the same key
        # the prefetch worker uses (decode cache is keyed by position, not frame.index).
        if self._decode_cache is not None and fr is not None:
            fr._cache_pos = index
        return fr

    def initialize_from_frames(
        self,
        n_frames: int,
        frame_filter=None,  # Callable[[StreamingRGBDFrame], bool] | None
        record_fn=None,  # Callable[[cam, frame, is_train, is_holdout], None] | None
    ) -> None:
        """
        Ingest the first n_frames as initial cameras and build the initial
        Gaussian point cloud from their depth data.

        Must be called BEFORE gaussians.training_setup() so the optimizer is
        built on the correct initial parameter tensors.

        Args:
            n_frames: Maximum number of frames to ingest.
            frame_filter: Optional callable. If provided, only frames for which
                it returns True are included in the bootstrap. This is used by
                online-SLAM bootstrap (Phase 1) to wait for SLAM poses to arrive
                before using a frame for the initial point cloud.
            record_fn: Optional callable, same signature/contract as
                ``ingest_next_frame``'s ``record_fn`` (multires-cache plan
                bugfix, 2026-07-26). Bootstrap frames bypass
                ``ingest_next_frame`` entirely, so without this they were
                never indexed into the observation DB -- every bootstrap
                camera's ``bind_resolution_tier`` lookup missed
                unconditionally (100% "record_none" in real-run validation),
                which matters disproportionately because
                ``_effective_keyframe_window`` deliberately keeps bootstrap
                frames in the active sampling window early in training.
        """
        n_frames = min(n_frames, self._frame_count)
        if n_frames == 0:
            raise RuntimeError("[streaming] No frames available for initialisation.")

        # Collect candidate frames, optionally filtered
        initial_frames = []
        candidate_count = 0
        for i in range(n_frames):
            frame = self._get_source_frame(i)
            if frame_filter is None or frame_filter(frame):
                initial_frames.append(frame)
            candidate_count += 1
            if len(initial_frames) >= n_frames:
                break

        if not initial_frames:
            raise RuntimeError(
                f"[streaming] Bootstrap: no frames passed the filter among first "
                f"{candidate_count} candidates. "
                + ("Is the SLAM node publishing poses?" if frame_filter else "Check dataset.")
            )

        print(
            f"[streaming] Bootstrapping from {len(initial_frames)} frame(s) "
            f"(filtered from {candidate_count} candidates)...",
            flush=True,
        )

        for frame in initial_frames:
            cam = self._frame_to_camera(frame)
            cam._is_bootstrap = True
            self.train_cameras.append(cam)
            self._replay_buffer.append(cam)
            StreamingScene._prepare_camera_for_render_owned(self, cam, device="cuda")
            if record_fn is not None:
                record_fn(cam, frame, is_train=True, is_holdout=False)

        self._bootstrap_count = n_frames
        self._source_idx = n_frames
        self.current_camera = self.train_cameras[-1]

        # Estimate scene extent from camera positions
        camera_extent = self._estimate_cameras_extent(initial_frames)

        # Build initial point cloud from depth images
        from eligsir.utils.graphics_utils import BasicPointCloud
        pcd = self._build_pcd_from_frames(initial_frames)

        # Incorporate PCD bbox into extent
        pcd_extent = 0.0
        if pcd.points.shape[0] > 0:
            mins = np.min(pcd.points, axis=0)
            maxs = np.max(pcd.points, axis=0)
            pcd_extent = np.linalg.norm(maxs - mins) * 0.5

        # Metric scale: enforce a minimum radius of 1.0 for stability
        self.cameras_extent = max(camera_extent, pcd_extent, 1.0)

        self.gaussians.create_from_pcd(
            pcd,
            self.cameras_extent,
            init_scale_mode=self.args.init_scale_mode,
            init_scale=getattr(self.args, "init_scale", 0.01),
            voxel_size=getattr(self.args, "pcd_voxel_size", 0.02),
        )
        self._export_initial_seed_pointcloud(pcd)
        self.maintain_occupancy_hash(voxel_size=getattr(self.args, "streaming_insert_voxel_size", 0.02))
        print(
            f"[streaming] Initialised {self.gaussians.get_xyz.shape[0]} Gaussians "
            f"from {n_frames} frames, scene radius={self.cameras_extent:.3f}",
            flush=True,
        )

    def initialize_from_canonical_frames(self, policy, record_fn=None) -> None:
        """Strict bootstrap for prevalidated MappingFrame observations only."""

        from eligsir.mapping.canonical_bootstrap import (
            canonical_bootstrap_point_cloud,
            canonical_camera_extent,
            canonical_camera_from_observation,
        )
        from eligsir.runtime.config import CanonicalBootstrapPolicy

        if not isinstance(policy, CanonicalBootstrapPolicy):
            raise TypeError("canonical scene bootstrap requires typed policy")
        frames = tuple(self._all_frames or ())
        if not frames:
            raise RuntimeError("canonical scene bootstrap requires MappingFrames")
        owner = self._device_transfer_owner
        if owner is None:
            raise RuntimeError("canonical scene bootstrap requires transfer ownership")
        pcd = canonical_bootstrap_point_cloud(frames, policy)
        for frame in frames:
            cam = canonical_camera_from_observation(frame, data_device=owner.device)
            cam._is_bootstrap = True
            self.train_cameras.append(cam)
            self._replay_buffer.append(cam)
            owner.prefetch_cameras((cam,))
            if record_fn is not None:
                record_fn(cam, frame, is_train=True, is_holdout=False)
        self._bootstrap_count = len(frames)
        self._source_idx = len(frames)
        self.current_camera = self.train_cameras[-1]
        self.cameras_extent = canonical_camera_extent(frames, pcd)
        self.gaussians.create_from_pcd(
            pcd,
            self.cameras_extent,
            init_scale_mode=policy.init_scale_mode,
            init_scale=policy.init_scale,
            voxel_size=policy.model_voxel_size_m,
        )
        self._export_initial_seed_pointcloud(pcd)
        self.maintain_occupancy_hash(voxel_size=policy.occupancy_voxel_size_m)

    def _export_initial_seed_pointcloud(self, pcd) -> None:
        """Write the bootstrap RGB-D point cloud into the run output for inspection."""
        try:
            from scene.readers.common import storePly

            seed_dir = os.path.join(self.model_path, "point_cloud", "initial_seed")
            os.makedirs(seed_dir, exist_ok=True)
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = np.clip(np.asarray(pcd.colors, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
            normals = np.asarray(getattr(pcd, "normals", np.zeros_like(points)), dtype=np.float32)
            storePly(os.path.join(seed_dir, "point_cloud.ply"), points, colors, normals=normals)
            storePly(os.path.join(self.model_path, "input_seed.ply"), points, colors, normals=normals)
            print(f"[streaming] Initial seed point cloud exported ({points.shape[0]} points)", flush=True)
        except Exception as exc:
            print(f"[streaming] Warning: initial seed point cloud export failed: {exc}", flush=True)

    def _estimate_cameras_extent(self, frames: List["StreamingRGBDFrame"]) -> float:
        """NeRF++ normalisation radius from a list of frames."""
        try:
            from scene.readers.common import CameraInfo, getNerfppNorm
            from eligsir.utils.graphics_utils import focal2fov
            from eligsir.utils.rgbd_frames import c2w_to_camera_rt

            infos = []
            for f in frames:
                R, T = c2w_to_camera_rt(f.c2w)
                infos.append(CameraInfo(
                    uid=f.index, R=R, T=T,
                    FovY=focal2fov(f.fy, f.height),
                    FovX=focal2fov(f.fx, f.width),
                    image=None, image_path="", image_name="",
                    width=f.width, height=f.height,
                ))
            norm = getNerfppNorm(infos)
            return max(float(norm["radius"]), 0.1)
        except Exception:
            return 1.0

    @staticmethod
    def _resize_nearest_np(arr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if arr.shape[:2] == (target_h, target_w):
            return arr
        src_h, src_w = arr.shape[:2]
        ys = np.clip(np.round(np.linspace(0, src_h - 1, target_h)).astype(np.int32), 0, src_h - 1)
        xs = np.clip(np.round(np.linspace(0, src_w - 1, target_w)).astype(np.int32), 0, src_w - 1)
        return arr[ys[:, None], xs[None, :]]

    @staticmethod
    def _stable_depth_samples(
        z: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        min_depth: float,
        max_depth: float,
        args,
    ) -> np.ndarray:
        if not args.streaming_depth_filter_enabled:
            return np.ones(xs.shape, dtype=bool)
        h, w = z.shape
        erode_px = max(0, int(getattr(args, "streaming_depth_filter_erode_px", 2)))
        window = max(1, int(getattr(args, "streaming_depth_filter_window", 5)))
        min_valid_ratio = float(getattr(args, "streaming_depth_filter_min_valid_ratio", 0.75))
        max_range = float(getattr(args, "streaming_depth_filter_max_range", 0.08))
        median_thresh = float(getattr(args, "streaming_depth_filter_median_thresh", 0.04))
        bottom_margin = float(getattr(args, "streaming_depth_filter_bottom_margin", 0.0))

        radius = window // 2
        local_depths = []
        for dy in range(-radius, radius + 1):
            yy = np.clip(ys + dy, 0, h - 1)
            for dx in range(-radius, radius + 1):
                xx = np.clip(xs + dx, 0, w - 1)
                local_depths.append(z[yy, xx])
        local = np.stack(local_depths, axis=0).astype(np.float32, copy=False)
        local_valid = np.isfinite(local) & (local > min_depth) & (local < max_depth)
        stable = local_valid.mean(axis=0) >= min_valid_ratio

        if max_range > 0:
            local_min = np.where(local_valid, local, np.inf).min(axis=0)
            local_max = np.where(local_valid, local, -np.inf).max(axis=0)
            stable &= np.isfinite(local_min) & np.isfinite(local_max) & ((local_max - local_min) <= max_range)

        if median_thresh > 0:
            from geometry.depth_filtering import fast_nanmedian_axis0
            local_nan = np.where(local_valid, local, np.nan)
            # np.nanmedian(..., axis=0) routes through numpy.ma's masked-array
            # machinery whenever NaNs are present, which profiled ~5x slower
            # than this bit-exact-equivalent sort-based replacement on this
            # method's realistic window-stack shape (issue #270/#276, see
            # docs/architecture/2026-08-13-issue270-accumulate-observed-surface-cpu-profile.md).
            local_median = fast_nanmedian_axis0(local_nan)
            stable &= np.isfinite(local_median) & (np.abs(z[ys, xs] - local_median) <= median_thresh)

        if erode_px > 0:
            valid_hw = np.isfinite(z) & (z > min_depth) & (z < max_depth)
            eroded = np.ones(xs.shape, dtype=bool)
            for dy in range(-erode_px, erode_px + 1):
                yy = np.clip(ys + dy, 0, h - 1)
                for dx in range(-erode_px, erode_px + 1):
                    xx = np.clip(xs + dx, 0, w - 1)
                    eroded &= valid_hw[yy, xx]
            stable &= eroded

        if bottom_margin > 0:
            stable &= ys < int(round(h * (1.0 - bottom_margin)))
        return stable

    @staticmethod
    def _temporal_depth_samples(
        z_v: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        frame,
        other_frame,
        other_depth_m: np.ndarray,
        d_fx: float,
        d_fy: float,
        d_cx: float,
        d_cy: float,
        thresh: float,
    ) -> Optional[np.ndarray]:
        if other_frame is None or other_depth_m is None or thresh <= 0:
            return None
        try:
            c2w = np.asarray(frame.c2w, dtype=np.float32)
            other_c2w = np.asarray(other_frame.c2w, dtype=np.float32)
            if c2w.shape != (4, 4) or other_c2w.shape != (4, 4):
                return None
            if not np.isfinite(c2w).all() or not np.isfinite(other_c2w).all():
                return None

            other_h, other_w = int(other_frame.height), int(other_frame.width)
            other_depth = StreamingScene._resize_nearest_np(
                np.asarray(other_depth_m, dtype=np.float32),
                (other_h, other_w),
            )

            x_c = (xs.astype(np.float32) - float(d_cx)) / float(d_fx) * z_v.astype(np.float32)
            y_c = (ys.astype(np.float32) - float(d_cy)) / float(d_fy) * z_v.astype(np.float32)
            pts_cam = np.stack([x_c, y_c, z_v.astype(np.float32)], axis=1)
            pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

            other_w2c = np.linalg.inv(other_c2w)
            pts_other = (other_w2c[:3, :3] @ pts_world.T).T + other_w2c[:3, 3]
            z_other = pts_other[:, 2]

            o_fx = other_frame.depth_fx if other_frame.depth_fx is not None else other_frame.fx
            o_fy = other_frame.depth_fy if other_frame.depth_fy is not None else other_frame.fy
            o_cx = other_frame.depth_cx if other_frame.depth_cx is not None else other_frame.cx
            o_cy = other_frame.depth_cy if other_frame.depth_cy is not None else other_frame.cy
            if other_frame.depth_fx is not None:
                _odw = getattr(other_frame, 'depth_width', None) or other_w
                _odh = getattr(other_frame, 'depth_height', None) or other_h
                if _odw != other_w or _odh != other_h:
                    o_fx = o_fx * other_w / _odw
                    o_fy = o_fy * other_h / _odh
                    o_cx = o_cx * other_w / _odw
                    o_cy = o_cy * other_h / _odh
            u = np.round((pts_other[:, 0] / np.maximum(z_other, 1e-6)) * float(o_fx) + float(o_cx)).astype(np.int32)
            v = np.round((pts_other[:, 1] / np.maximum(z_other, 1e-6)) * float(o_fy) + float(o_cy)).astype(np.int32)

            ok = (z_other > 0) & (u >= 0) & (u < other_w) & (v >= 0) & (v < other_h)
            out = np.zeros(z_v.shape[0], dtype=bool)
            if ok.any():
                lookup = other_depth[v[ok], u[ok]]
                out[ok] = np.isfinite(lookup) & (lookup > 0) & (np.abs(z_other[ok] - lookup) <= float(thresh))
            return out
        except Exception:
            return None

    def _build_pcd_from_frames(self, frames: List["StreamingRGBDFrame"]):
        """Backproject depth into a BasicPointCloud for create_from_pcd()."""
        from eligsir.utils.graphics_utils import BasicPointCloud
        from eligsir.utils.rgbd_frames import depth_to_meters, backproject_depth_pixels
        from PIL import Image as _Image

        depth_stride = getattr(self.args, "streaming_depth_stride", 8)
        min_depth = getattr(self.args, "streaming_min_depth", 0.1)
        max_depth = getattr(self.args, "streaming_max_depth", 8.0)
        max_pts = getattr(self.args, "rgbd_max_init_points", 250000)

        # External SfM seed cloud (e.g. RealityScan) takes precedence over
        # both the depth path and the disabled-depth random fallback below --
        # depth-less sources have no other real geometry to bootstrap from.
        if self._external_seed is not None:
            seed_pts, seed_cols = self._external_seed
            if seed_pts is not None and seed_pts.shape[0] > 0:
                n_before = seed_pts.shape[0]
                seed_voxel = float(self.args.streaming_seed_voxel_size)
                pcd = _external_seed_to_pcd(seed_pts, seed_cols, seed_voxel, max_pts)
                print(
                    f"[streaming] Bootstrap: using external SfM seed cloud "
                    f"({n_before} points -> {pcd.points.shape[0]} after voxel/cap).",
                    flush=True,
                )
                return pcd

        # When depth is globally disabled, skip to random bootstrap immediately
        if not getattr(self.args, "streaming_insert_from_depth", True):
            r = self.cameras_extent
            n = max(1000, self.args.cap_max // 10)
            pts = np.random.uniform(-r, r, (n, 3)).astype(np.float32)
            cols = np.random.uniform(0.3, 0.7, (n, 3)).astype(np.float32)
            print("[streaming] Bootstrap: depth disabled, using random init.", flush=True)
            return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))

        def _load_one(frame):
            if frame.depth_path is None and frame._depth_bytes is None and frame._depth_np is None:
                return None
            if frame.depth_path is not None and not os.path.exists(frame.depth_path) and frame._depth_bytes is None and frame._depth_np is None:
                return None
            try:
                from eligsir.utils.streaming_frames import load_frame_rgb, load_frame_depth_np
                depth_raw = load_frame_depth_np(frame)
                if depth_raw is None:
                    return None
                depth = np.asarray(depth_raw)
                _depth_native_h, _depth_native_w = depth.shape[:2]
                _target = (int(frame.height), int(frame.width))
                if depth.shape[:2] != _target:
                    depth = np.array(_Image.fromarray(depth).resize(
                        (_target[1], _target[0]), _Image.NEAREST))
                rgb_pil = load_frame_rgb(frame)
                if rgb_pil.size != (_target[1], _target[0]):
                    rgb_pil = rgb_pil.resize((_target[1], _target[0]), _Image.BILINEAR)
                rgb = np.array(rgb_pil).astype(np.float32) / 255.0
                return (frame, depth, rgb, depth_to_meters(depth, frame.depth_scale),
                        _depth_native_h, _depth_native_w)
            except Exception:
                return None

        import os as _os
        import time as _time
        n_workers = min(len(frames), max(1, _os.cpu_count() or 1))
        _t0 = _time.perf_counter()
        if n_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                loaded = list(ex.map(_load_one, frames))
        else:
            loaded = [_load_one(f) for f in frames]
        print(
            f"[streaming] PCD load: {sum(x is not None for x in loaded)}/{len(frames)} frames "
            f"in {_time.perf_counter() - _t0:.2f}s ({n_workers} workers)",
            flush=True,
        )

        points_all, colors_all = [], []
        seed_temporal = bool(self.args.streaming_depth_temporal_require_for_seed)
        temporal_thresh = float(getattr(self.args, "streaming_depth_temporal_thresh", 0.07))
        for idx, item in enumerate(loaded):
            if item is None:
                continue
            frame, depth, rgb, z, _depth_native_h, _depth_native_w = item
            _target = (int(frame.height), int(frame.width))
            h, w = z.shape
            ys, xs = np.mgrid[0:h:depth_stride, 0:w:depth_stride]
            z_v = z[ys, xs]
            valid = np.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)
            valid &= self._stable_depth_samples(z, xs, ys, min_depth, max_depth, self.args)
            if not valid.any():
                continue
            d_fx = frame.depth_fx if frame.depth_fx is not None else frame.fx
            d_fy = frame.depth_fy if frame.depth_fy is not None else frame.fy
            d_cx = frame.depth_cx if frame.depth_cx is not None else frame.cx
            d_cy = frame.depth_cy if frame.depth_cy is not None else frame.cy
            if frame.depth_fx is not None and (_depth_native_w != _target[1] or _depth_native_h != _target[0]):
                d_fx = d_fx * _target[1] / _depth_native_w
                d_fy = d_fy * _target[0] / _depth_native_h
                d_cx = d_cx * _target[1] / _depth_native_w
                d_cy = d_cy * _target[0] / _depth_native_h
            if seed_temporal and len(loaded) > 1:
                other_item = None
                for j in (idx - 1, idx + 1):
                    if 0 <= j < len(loaded) and loaded[j] is not None:
                        other_item = loaded[j]
                        break
                if other_item is not None:
                    other_frame, _, _, other_z, *_ = other_item
                    temporal = self._temporal_depth_samples(
                        z_v[valid].astype(np.float32),
                        xs[valid].astype(np.int32),
                        ys[valid].astype(np.int32),
                        frame,
                        other_frame,
                        other_z,
                        d_fx,
                        d_fy,
                        d_cx,
                        d_cy,
                        temporal_thresh,
                    )
                    if temporal is not None:
                        full_temporal = np.zeros_like(valid, dtype=bool)
                        full_temporal[valid] = temporal
                        valid &= full_temporal
                    else:
                        valid &= False
                else:
                    valid &= False
            if not valid.any():
                continue

            xs_v = xs[valid].astype(np.float32)
            ys_v = ys[valid].astype(np.float32)
            z_v = z_v[valid].astype(np.float32)
            pts_world, cols = backproject_depth_pixels(
                z_v, xs_v, ys_v, d_fx, d_fy, d_cx, d_cy, frame.c2w, rgb, (h, w))
            if frame.depth_fx is not None and (_depth_native_w != _target[1] or _depth_native_h != _target[0]):
                x_over_z = (xs_v - d_cx) / d_fx
                y_over_z = (ys_v - d_cy) / d_fy
                u_color = (x_over_z * frame.fx + frame.cx).round().astype(np.int32)
                v_color = (y_over_z * frame.fy + frame.cy).round().astype(np.int32)
                in_fov = (u_color >= 0) & (u_color < frame.width) & (v_color >= 0) & (v_color < frame.height)
                if not in_fov.all():
                    pts_world = pts_world[in_fov]
                    u_color = u_color[in_fov]
                    v_color = v_color[in_fov]
                rgb_h, rgb_w = rgb.shape[:2]
                uc = np.clip(u_color, 0, rgb_w - 1)
                vc = np.clip(v_color, 0, rgb_h - 1)
                cols = rgb[vc, uc].astype(np.float32)
            points_all.append(pts_world)
            colors_all.append(cols)

        if not points_all:
            # Fall back to random scatter inside the estimated scene extent
            r = self.cameras_extent
            n = max(1000, self.args.cap_max // 10)
            pts = np.random.uniform(-r, r, (n, 3)).astype(np.float32)
            cols = np.random.uniform(0.3, 0.7, (n, 3)).astype(np.float32)
            print("[streaming] Warning: no depth available for init, falling back to random point cloud.", flush=True)
            return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts))

        pts = np.concatenate(points_all, axis=0)
        cols = np.concatenate(colors_all, axis=0)

        # Intelligent seed clustering: collapse overlapping multi-frame points into
        # one voxel-centroid Gaussian per cell. Spatially-aware (unlike the random
        # max_pts subsample below), so the seed density is bounded BEFORE training
        # rather than left to MCMC to churn down at-cap. Applied first so the random
        # cap only ever trims an already-clustered cloud.
        seed_voxel = float(self.args.streaming_seed_voxel_size)
        if seed_voxel > 0.0 and pts.shape[0] > 0:
            n_before = pts.shape[0]
            pts, cols = self._voxel_centroid_cluster(pts, cols, seed_voxel)
            print(
                f"[streaming] Seed voxel clustering ({seed_voxel:.3f} m): "
                f"{n_before} -> {pts.shape[0]} points",
                flush=True,
            )

        if max_pts and pts.shape[0] > max_pts:
            rng = np.random.default_rng(42)
            idx = rng.choice(pts.shape[0], size=max_pts, replace=False)
            pts, cols = pts[idx], cols[idx]
        return BasicPointCloud(points=pts, colors=cols, normals=np.zeros_like(pts, dtype=np.float32))

    @staticmethod
    def _voxel_centroid_cluster(pts: np.ndarray, cols: np.ndarray, voxel_size: float):
        """Cluster points into voxels; return per-cell centroid position + mean colour.

        Averaging (rather than keep-first) denoises the seed and yields one clean
        Gaussian per occupied cell. O(N log N) via a single unique() over packed keys.
        """
        mins = pts.min(axis=0)
        cell = np.floor((pts - mins) / max(voxel_size, 1e-6)).astype(np.int64)
        maxc = cell.max(axis=0) + 1
        stride = np.array([maxc[1] * maxc[2], maxc[2], 1], dtype=np.int64)
        keys = cell @ stride
        _, inv = np.unique(keys, return_inverse=True)
        n_clusters = int(inv.max()) + 1 if inv.size else 0
        counts = np.bincount(inv, minlength=n_clusters).astype(np.float32)[:, None]
        sum_pts = np.zeros((n_clusters, 3), dtype=np.float32)
        np.add.at(sum_pts, inv, pts.astype(np.float32))
        sum_cols = np.zeros((n_clusters, 3), dtype=np.float32)
        np.add.at(sum_cols, inv, cols.astype(np.float32))
        return sum_pts / counts, sum_cols / counts

    @staticmethod
    def _pos_voxel_keys(pts: np.ndarray, voxel_size: float) -> np.ndarray:
        """Packed position-only voxel keys (for box-counting dimension)."""
        v = np.floor((pts + 1000.0) / max(voxel_size, 1e-9)).astype(np.int64)
        return (v[:, 0] << 42) ^ (v[:, 1] << 21) ^ v[:, 2]

    def accumulate_observed_surface(self, frame) -> None:
        """Fold one arrived frame's observed surface into the SV accumulator.

        Called at ingestion (depth still in hand) for EVERY arrived frame so SV gets
        full trajectory coverage even for lazy rosbag/mcap sources. Uses cheap
        finite-difference normals + the existing stable-depth filter; keys are hashed
        into compact sets at multiple voxel scales (raw points are never retained).

        Internally timed at sub-stage granularity when SPLAT_PROFILE_PHASES=1 (see
        train_streaming._timed / issue #274): "surface/materialize" (depth decode +
        cam-space normals), "surface/mask" (valid/stable-depth filtering),
        "surface/backproject" (world-space points+normals), "surface/voxelize" (the
        Python set-heavy multiscale key bookkeeping -- the region #276 would target
        first if this is ever the dominant cost), and "surface/quality_thickness" /
        "surface/quality_sensor_noise" split out separately because the latter is
        the ONE sub-region in this otherwise pure-numpy/CPU method that launches
        real GPU work (estimate_depth_normals_pca does an unfold + batched eigh on
        `cuda` when available) -- lumping it into one "quality_stats" bucket would
        hide that asymmetry. Workload counters (input pixels, valid samples, world
        points, unique voxel keys per scale) are recorded alongside via
        train_streaming.record_count() so latency can be normalized, not just read
        as raw ms -- see the "Required output" counters list in issue #274.
        """
        if not getattr(self.args, "streaming_track_scene_complexity", True):
            return
        from eligsir.utils.ingest_phase_profiler import _timed, record_count
        try:
            from geometry.normals import estimate_depth_normals_finite_difference
            from geometry.surface_cells import surface_cell_keys
            from eligsir.utils.rgbd_frames import depth_to_meters
            from eligsir.utils.streaming_frames import load_frame_depth_np

            with _timed("surface/materialize"):
                depth_raw = load_frame_depth_np(frame)
                if depth_raw is None:
                    return
                z = depth_to_meters(np.asarray(depth_raw), frame.depth_scale).astype(np.float32)
                c2w = np.asarray(getattr(frame, "c2w", None), dtype=np.float32)
                if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                    return
                fx = frame.depth_fx if frame.depth_fx is not None else frame.fx
                fy = frame.depth_fy if frame.depth_fy is not None else frame.fy
                cx = frame.depth_cx if frame.depth_cx is not None else frame.cx
                cy = frame.depth_cy if frame.depth_cy is not None else frame.cy
                fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
                min_depth = float(getattr(self.args, "streaming_min_depth", 0.1))
                max_depth = float(getattr(self.args, "streaming_max_depth", 8.0))
                # Coarser stride than insertion keeps this cheap in the ingest hot loop.
                stride = max(4, int(getattr(self.args, "streaming_depth_stride", 8)))

                cam_n, _ = estimate_depth_normals_finite_difference(
                    z, fx, fy, min_depth=min_depth, max_depth=max_depth)
                h, w = z.shape
                ys, xs = np.mgrid[0:h:stride, 0:w:stride]
                record_count("surface/input_pixels", h * w)
                record_count("surface/sampled_pixels", ys.size)

            with _timed("surface/mask"):
                z_v = z[ys, xs]
                valid = np.isfinite(z_v) & (z_v > min_depth) & (z_v < max_depth)
                valid &= self._stable_depth_samples(z, xs, ys, min_depth, max_depth, self.args)
                cn = cam_n[:, ys, xs]
                valid &= np.linalg.norm(cn, axis=0) > 1e-6
                record_count("surface/valid_samples", int(valid.sum()))
                if not valid.any():
                    return

            with _timed("surface/backproject"):
                xs_v = xs[valid].astype(np.float32)
                ys_v = ys[valid].astype(np.float32)
                zz = z_v[valid].astype(np.float32)
                pts_cam = np.stack([(xs_v - cx) / fx * zz, (ys_v - cy) / fy * zz, zz], axis=1)
                pw = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
                nw = (c2w[:3, :3] @ cn[:, valid]).T
                self._obs_n_points += int(pw.shape[0])
                record_count("surface/world_points", pw.shape[0])

            with _timed("surface/voxelize"):
                nb = int(getattr(self._surface_cell_index, "normal_bins", 8))
                for m in self._obs_scales:
                    d = self._obs_voxel * float(m)
                    keys_before = len(self._obs_keys[m])
                    new_keys = surface_cell_keys(pw, nw, voxel_size=d, normal_bins=nb).tolist()
                    self._obs_keys[m].update(new_keys)
                    self._obs_vox[m].update(self._pos_voxel_keys(pw, d).tolist())
                    record_count(f"surface/voxel_keys_new_scale{m}", len(self._obs_keys[m]) - keys_before)
                    record_count(f"surface/voxel_keys_total_scale{m}", len(self._obs_keys[m]))
                self._scene_complexity_cache = None  # invalidate

            # Input-quality: combined surface thickness (per-voxel diagonal Welford on
            # a subsample) + per-frame sensor-noise residual on every Nth frame.
            if getattr(self.args, "streaming_track_input_quality", True):
                self._iq_frame_counter += 1
                with _timed("surface/quality_thickness"):
                    self._accumulate_thickness(pw)
                if (self._iq_frame_counter % self._iq_sensor_stride) == 0:
                    # The only GPU-touching sub-region in this method (see
                    # docstring above) -- timed separately so its cost isn't
                    # blended into the otherwise pure-CPU/numpy stages above.
                    with _timed("surface/quality_sensor_noise"):
                        self._accumulate_sensor_noise(z, fx, fy, cx, cy, min_depth, max_depth)
                self._input_quality_cache = None
        except Exception:
            return

    def get_accumulated_sensor_points(self) -> np.ndarray:
        """Real, Gaussian-independent sensor point cloud for Tier D
        (``utils.scene_geometry_metrics.compute_offsurface_metrics``),
        decoded from the input-quality tracker's per-voxel accumulators
        (see ``sensor_points_from_iq_vox``'s docstring for why this is
        independent of the Gaussian model). Empty ``(0, 3)`` array if
        ``--streaming_track_input_quality`` is off or nothing has been
        ingested yet."""
        from eligsir.utils.scene_geometry_metrics import sensor_points_from_iq_vox
        return sensor_points_from_iq_vox(self._observation_store.iq_vox)

    def _accumulate_thickness(self, pw: np.ndarray, max_pts: int = 1500) -> None:
        """Per-voxel covariance sums on a subsample → smallest-eigenvalue (true
        out-of-plane) thickness at finalize. Sums: [n, Σx,Σy,Σz, Σxx,Σyy,Σzz, Σxy,Σxz,Σyz]."""
        if pw.shape[0] == 0:
            return
        if pw.shape[0] > max_pts:
            idx = np.random.default_rng(self._iq_frame_counter).choice(
                pw.shape[0], size=max_pts, replace=False)
            pw = pw[idx]
        keys = self._pos_voxel_keys(pw, self._obs_voxel)
        vox, cap = self._iq_vox, self._iq_vox_cap
        for k, p in zip(keys.tolist(), pw):
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            s = vox.get(k)
            if s is None:
                if len(vox) >= cap:
                    continue
                vox[k] = [1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z]
            else:
                s[0] += 1
                s[1] += x; s[2] += y; s[3] += z
                s[4] += x * x; s[5] += y * y; s[6] += z * z
                s[7] += x * y; s[8] += x * z; s[9] += y * z

    def _accumulate_sensor_noise(self, z, fx, fy, cx, cy, min_depth, max_depth) -> None:
        """Intra-frame PCA plane residual (m) — pose-free sensor-noise proxy."""
        from geometry.normals import NormalEstimationConfig, estimate_depth_normals_pca
        cfg = NormalEstimationConfig(mode="pca_plane", min_depth=min_depth, max_depth=max_depth)
        _, _, res = estimate_depth_normals_pca(z, fx, fy, cx, cy, cfg, return_residual=True)
        r = res[np.isfinite(res)]
        if r.size:
            self._iq_sensor_sum += float(r.sum())
            self._iq_sensor_sumsq += float((r * r).sum())
            self._iq_sensor_n += int(r.size)

    def scene_complexity(self, voxel_size: Optional[float] = None) -> dict:
        """Input-side scene expressiveness ``SV`` (surfel demand) — a scene constant.

        SV = unique surface cells (voxel x normal-bin) of the observed surface,
        accumulated incrementally across ALL arrived frames (full coverage, lazy +
        non-lazy alike). Box-counting ``complexity_dim`` from position-only voxels.
        Used to normalise Gaussian count: ``n_live / SV`` is size/growth-invariant.
        """
        if self._scene_complexity_cache is not None:
            return self._scene_complexity_cache

        base = self._obs_voxel if voxel_size is None else float(voxel_size)
        if self._obs_n_points == 0:
            result = {"sv": None, "voxel_size": base, "n_points": 0,
                      "note": "no observed surface accumulated"}
            self._scene_complexity_cache = result
            return result

        multiscale, deltas, boxes = {}, [], []
        for m in self._obs_scales:
            d = self._obs_voxel * float(m)
            sv_m = len(self._obs_keys[m])
            nbox = len(self._obs_vox[m])
            multiscale[f"x{m}"] = {"voxel_size": d, "sv": sv_m, "occupied_voxels": nbox}
            deltas.append(d)
            boxes.append(nbox)
        complexity_dim = None
        if len(deltas) >= 2 and min(boxes) > 0:
            complexity_dim = float(-np.polyfit(np.log(deltas), np.log(boxes), 1)[0])
        result = {
            "sv": multiscale[f"x{self._obs_scales[0]}"]["sv"],
            "voxel_size": self._obs_voxel,
            "normal_bins": int(getattr(self._surface_cell_index, "normal_bins", 8)),
            "n_points": self._obs_n_points,
            "sv_multiscale": multiscale,
            "complexity_dim": complexity_dim,
            "coverage_frames": len(self.train_cameras) + len(self._test_cameras),
        }
        self._scene_complexity_cache = result
        print(f"[streaming] scene complexity: SV={result['sv']} surfels "
              f"(voxel {self._obs_voxel:.3f} m, {result['n_points']} pts, dim={complexity_dim})",
              flush=True)
        return result

    def input_quality(self) -> dict:
        """Input-quality metric splitting sensor noise (intra-frame) from tracking
        error (inter-frame). See ``utils.scene_geometry_metrics.input_quality_stats``."""
        if self._input_quality_cache is not None:
            return self._input_quality_cache
        from eligsir.utils.scene_geometry_metrics import input_quality_stats

        # Sensor noise (intra-frame PCA residual).
        sensor_mean = sensor_std = None
        if self._iq_sensor_n > 0:
            sensor_mean = self._iq_sensor_sum / self._iq_sensor_n
            var = max(0.0, self._iq_sensor_sumsq / self._iq_sensor_n - sensor_mean * sensor_mean)
            sensor_std = float(np.sqrt(var))

        # Combined surface thickness = RMS smallest-eigenvalue (out-of-plane) spread
        # over multi-observed voxels — rotation-invariant (in-plane extent excluded).
        thickness_m = None
        min_eigs = []
        for s in self._iq_vox.values():
            n = s[0]
            if n < 4:
                continue
            mx, my, mz = s[1] / n, s[2] / n, s[3] / n
            cov = np.array([
                [s[4] / n - mx * mx, s[7] / n - mx * my, s[8] / n - mx * mz],
                [s[7] / n - mx * my, s[5] / n - my * my, s[9] / n - my * mz],
                [s[8] / n - mx * mz, s[9] / n - my * mz, s[6] / n - mz * mz],
            ], dtype=np.float64)
            lam = float(np.linalg.eigvalsh(cov)[0])  # smallest eigenvalue
            min_eigs.append(max(0.0, lam))
        if min_eigs:
            # Median (not mean): robust to depth-edge / mixed-surface voxels that
            # otherwise inflate the typical out-of-plane thickness.
            thickness_m = float(np.sqrt(np.median(min_eigs)))

        # Tracking error from per-frame odometry stats over arrived cameras.
        odom_stats = []
        for c in list(self.train_cameras) + list(self._test_cameras):
            f = getattr(c, "_streaming_frame", None)
            st = getattr(f, "_odom_stats", None) if f is not None else None
            if isinstance(st, dict):
                odom_stats.append(st)

        cplx = self.scene_complexity().get("complexity_dim")
        result = input_quality_stats(
            sensor_noise_m=sensor_mean, sensor_noise_std_m=sensor_std,
            sensor_n=self._iq_sensor_n, odom_stats=odom_stats,
            surface_thickness_m=thickness_m, complexity_dim=cplx,
            pose_source=self.args.pose_source,
        )
        self._input_quality_cache = result
        print(f"[streaming] input quality: sensor_noise={result.get('sensor_noise_m')} "
              f"tracking_rmse={result.get('tracking_inlier_rmse_m')} "
              f"thickness={result.get('surface_thickness_m')} verdict={result.get('quality_verdict')}",
              flush=True)
        return result

    # ------------------------------------------------------------------
    # Frame ingestion
    # ------------------------------------------------------------------

    def has_next_frame(self) -> bool:
        return self._source_idx < self._frame_count

    def has_ready_frame(self) -> bool:
        """Non-blocking: True only if the next frame is already buffered.

        For live sources (``buffered_count``) the online drain loop uses this so
        it never blocks the training thread waiting for the next ROS frame —
        when no frame is ready it returns False and the loop trains instead.
        For non-live sources every in-budget frame is "ready", so this matches
        :meth:`has_next_frame`.
        """
        if self._source_idx >= self._frame_count:
            return False
        bc = getattr(self._frame_source, "buffered_count", None)
        if callable(bc):
            return self._source_idx < bc()
        return True

    def queue_depth(self) -> Optional[int]:
        """Best-effort real pending-frame count at the frame source (F2
        fix, 2026-08-06, four-clocks telemetry): frames already buffered by
        the source but not yet consumed via ``ingest_next_frame``. Reuses
        the same ``buffered_count()``/``_source_idx`` plumbing
        :meth:`has_ready_frame` already relies on for live sources, rather
        than duplicating the WS-D1 ring-buffer work -- returns ``None``
        when the source doesn't expose a buffered count (e.g. a fully
        materialized offline dataset, where there is no real queue at all).
        """
        bc = getattr(self._frame_source, "buffered_count", None)
        if not callable(bc):
            return None
        return max(0, bc() - self._source_idx)

    def peek_next_frame(self) -> "Optional[StreamingRGBDFrame]":
        """Return the next source frame without consuming it.

        Used by the Milestone 3 BurstBudget to decide whether to drop
        a non-keyframe before paying the cost of
        :meth:`ingest_next_frame`. The returned frame is the same
        object the next :meth:`ingest_next_frame` call will yield.

        Returns:
            The next :class:`StreamingRGBDFrame`, or None when the
            source is exhausted.
        """
        if not self.has_next_frame():
            return None
        return self._get_source_frame(self._source_idx)

    def materialize_missed_holdouts(self) -> int:
        """Rebuild test cameras for holdout frames the cursor has already
        passed (checkpoint-resume path).

        The deterministic real-holdout indices (:meth:`_designate_test_holdouts`)
        are admission-independent, but the *Camera* objects in
        :attr:`_test_cameras` are only ever materialized as a side effect of
        :meth:`ingest_next_frame` actually being called for that frame index.
        A faithful resume repositions :attr:`_source_idx` directly (see
        train_streaming.py's checkpoint-restore path) specifically so it does
        NOT re-call ``ingest_next_frame`` for already-consumed indices --
        which means every holdout frame below the restored cursor would
        otherwise leave ``_test_cameras`` empty, silently producing an
        ``n_test == 0`` / ``eval_valid == false`` report on any resumed run.
        That violates this project's "Real test holdouts" contract (a report
        with n_test==0 must never be quoted), so this must be called right
        after the cursor is restored on resume.

        Only holdout frames are (re)built here -- never guard or train
        frames -- and they are appended to ``_test_cameras`` exactly as
        :meth:`ingest_next_frame` would have, without touching
        ``train_cameras``/the replay buffer/SV-accumulation bookkeeping
        (those are legitimately re-derived from the frames still ahead of
        the cursor as normal ingestion continues).

        Returns the number of test cameras (re)materialized.
        """
        if not self._holdout_indices:
            return 0
        already_have = {getattr(c, "_streaming_frame", None).index
                        for c in self._test_cameras
                        if getattr(c, "_streaming_frame", None) is not None}
        n_added = 0
        for idx in sorted(self._holdout_indices):
            if idx >= self._source_idx or idx in already_have:
                continue
            frame = self._get_source_frame(idx)
            if frame is None:
                continue
            cam = self._frame_to_camera(frame)
            self._test_cameras.append(cam)
            # Issue #398: same residency rule as the ingest path -- a resumed
            # run rebuilds EVERY already-passed holdout in one loop, so without
            # this the resume path alone spikes GPU residency by
            # O(n_holdouts x W x H) in a single burst.
            offload_holdout_image(self.args, cam)
            n_added += 1
        return n_added

    # Safety cushion kept resident behind the source cursor before a bounded
    # lazy frame source is told it may evict -- cheap (a few hundred MB at
    # most) insurance against any as-yet-undiscovered near-cursor re-read,
    # on top of the explicit holdout protection registered in __init__.
    _RELEASE_MARGIN = 32

    def _release_source_frames_upto_cursor(self) -> None:
        release = getattr(self._frame_source, "release_upto", None)
        if callable(release):
            release(max(0, self._source_idx - self._RELEASE_MARGIN))

    def skip_next_frame(self) -> int:
        """Advance ``_source_idx`` by one without ingesting the frame.

        Returns:
            The number of frames skipped (0 or 1).
        """
        if not self.has_next_frame():
            return 0
        self._source_idx += 1
        self._release_source_frames_upto_cursor()
        return 1

    # --- FrameArchive property-backed aliases (Wave 1 step 9) -------------
    # Preserve every pre-extraction attribute name so no external caller
    # (train_streaming.py, tests) needed to change. Getter-only: neither
    # dict is ever reassigned wholesale in this file, only mutated in
    # place.

    @property
    def _inactive_frame_cache(self) -> "OrderedDict":
        return self._frame_archive.inactive_frame_cache

    @property
    def _icecold_frame_cache(self) -> "OrderedDict":
        return self._frame_archive.icecold_frame_cache

    @property
    def _decode_cache(self):
        """Property-backed alias onto ``self._frame_archive.decode_cache``.
        Getter/setter both read/write straight through to the same stored
        instance -- identity-preserving, so ``get_decode_cache() is
        self._decode_cache`` continues to work exactly as before this
        extraction (Wave 2, closes the Wave 1 deferred-scope item)."""
        return self._frame_archive.decode_cache

    @_decode_cache.setter
    def _decode_cache(self, value) -> None:
        self._frame_archive.decode_cache = value

    @property
    def _lazy_frames(self) -> bool:
        return self._frame_archive.lazy_frames

    @_lazy_frames.setter
    def _lazy_frames(self, value: bool) -> None:
        self._frame_archive.lazy_frames = value

    @property
    def _all_frames(self):
        return self._frame_archive.all_frames

    @_all_frames.setter
    def _all_frames(self, value) -> None:
        self._frame_archive.all_frames = value

    # --- KeyframeStore property-backed aliases (Wave 1 step 9) ------------
    # Preserve every pre-extraction attribute name so no external caller
    # (train_streaming.py, tests) needed to change.

    @property
    def train_cameras(self) -> List:
        return self._keyframe_store.train_cameras

    @train_cameras.setter
    def train_cameras(self, value: List) -> None:
        self._keyframe_store.train_cameras = value

    @property
    def _test_cameras(self) -> List:
        return self._keyframe_store.test_cameras

    @_test_cameras.setter
    def _test_cameras(self, value: List) -> None:
        self._keyframe_store.test_cameras = value

    @property
    def _camera_centers_np(self) -> List:
        return self._keyframe_store.camera_centers_np

    @_camera_centers_np.setter
    def _camera_centers_np(self, value: List) -> None:
        self._keyframe_store.camera_centers_np = value

    @property
    def _camera_centers_gpu(self) -> List:
        return self._keyframe_store.camera_centers_gpu

    @_camera_centers_gpu.setter
    def _camera_centers_gpu(self, value: List) -> None:
        self._keyframe_store.camera_centers_gpu = value

    @property
    def _camera_viewdirs_np(self) -> List:
        return self._keyframe_store.camera_viewdirs_np

    @_camera_viewdirs_np.setter
    def _camera_viewdirs_np(self, value: List) -> None:
        self._keyframe_store.camera_viewdirs_np = value

    @property
    def current_camera(self):
        return self._keyframe_store.current_camera

    @current_camera.setter
    def current_camera(self, value) -> None:
        self._keyframe_store.current_camera = value

    @property
    def cameras_extent(self) -> float:
        return self._keyframe_store.cameras_extent

    @cameras_extent.setter
    def cameras_extent(self, value: float) -> None:
        self._keyframe_store.cameras_extent = value

    @property
    def _holdout_indices(self) -> Optional[set]:
        return self._keyframe_store.holdout_indices

    @_holdout_indices.setter
    def _holdout_indices(self, value: Optional[set]) -> None:
        self._keyframe_store.holdout_indices = value

    @property
    def _guard_indices(self) -> set:
        return self._keyframe_store.guard_indices

    @_guard_indices.setter
    def _guard_indices(self, value: set) -> None:
        self._keyframe_store.guard_indices = value

    @property
    def _keyframes_admitted(self) -> int:
        return self._keyframe_store.keyframes_admitted

    @_keyframes_admitted.setter
    def _keyframes_admitted(self, value: int) -> None:
        self._keyframe_store.keyframes_admitted = value

    @property
    def _keyframes_rejected(self) -> int:
        return self._keyframe_store.keyframes_rejected

    @_keyframes_rejected.setter
    def _keyframes_rejected(self, value: int) -> None:
        self._keyframe_store.keyframes_rejected = value

    @property
    def _last_admitted_source_idx(self) -> Optional[int]:
        return self._keyframe_store.last_admitted_source_idx

    @_last_admitted_source_idx.setter
    def _last_admitted_source_idx(self, value: Optional[int]) -> None:
        self._keyframe_store.last_admitted_source_idx = value

    @property
    def _bootstrap_count(self) -> int:
        return self._keyframe_store.bootstrap_count

    @_bootstrap_count.setter
    def _bootstrap_count(self, value: int) -> None:
        self._keyframe_store.bootstrap_count = value

    @property
    def _n_train_frames_ingested(self) -> int:
        return self._keyframe_store.n_train_frames_ingested

    @_n_train_frames_ingested.setter
    def _n_train_frames_ingested(self, value: int) -> None:
        self._keyframe_store.n_train_frames_ingested = value

    # --- OccupancyIndex property-backed aliases (Wave 1 step 9) -----------
    # Preserve every pre-extraction attribute name so no external caller
    # (train_streaming.py, tests) needed to change.

    @property
    def _occupancy_keys(self) -> np.ndarray:
        return self._occupancy_index.keys

    @_occupancy_keys.setter
    def _occupancy_keys(self, value: np.ndarray) -> None:
        self._occupancy_index.keys = value

    @property
    def _occupancy_keys_gpu(self):
        return self._occupancy_index.keys_gpu

    @_occupancy_keys_gpu.setter
    def _occupancy_keys_gpu(self, value) -> None:
        self._occupancy_index.keys_gpu = value

    @property
    def occupancy_voxel_size(self) -> float:
        return self._occupancy_index.voxel_size

    @occupancy_voxel_size.setter
    def occupancy_voxel_size(self, value: float) -> None:
        self._occupancy_index.voxel_size = value

    # --- ReplayViewStore property-backed aliases (Wave 1 step 9) ----------
    # Preserve every pre-extraction attribute name so no external caller
    # (train_streaming.py, tests) needed to change.

    @property
    def _replay_buffer(self) -> List:
        return self._replay_view_store.replay_buffer

    @_replay_buffer.setter
    def _replay_buffer(self, value: List) -> None:
        self._replay_view_store.replay_buffer = value

    @property
    def _global_reservoir(self) -> List:
        return self._replay_view_store.global_reservoir

    @property
    def _priority_frame_indices(self):
        return self._replay_view_store.priority_frame_indices

    @_priority_frame_indices.setter
    def _priority_frame_indices(self, value) -> None:
        self._replay_view_store.priority_frame_indices = value

    @property
    def _priority_frame_weights(self) -> dict:
        return self._replay_view_store.priority_frame_weights

    @_priority_frame_weights.setter
    def _priority_frame_weights(self, value: dict) -> None:
        self._replay_view_store.priority_frame_weights = value

    @property
    def _consol_debt_weights(self):
        return self._replay_view_store.consol_debt_weights

    @_consol_debt_weights.setter
    def _consol_debt_weights(self, value) -> None:
        self._replay_view_store.consol_debt_weights = value

    @property
    def _consol_debt_pool_len(self) -> int:
        return self._replay_view_store.consol_debt_pool_len

    @_consol_debt_pool_len.setter
    def _consol_debt_pool_len(self, value: int) -> None:
        self._replay_view_store.consol_debt_pool_len = value

    @property
    def _consol_debt_age(self) -> int:
        return self._replay_view_store.consol_debt_age

    @_consol_debt_age.setter
    def _consol_debt_age(self, value: int) -> None:
        self._replay_view_store.consol_debt_age = value

    # --- ObservationStore property-backed aliases (Wave 1 step 9) ---------
    # Preserve every pre-extraction attribute name so no external caller
    # (train_streaming.py, tests) needed to change.

    @property
    def observation_db(self):
        return self._observation_store.observation_db

    @observation_db.setter
    def observation_db(self, db) -> None:
        self._observation_store.observation_db = db

    @property
    def _obs_scales(self):
        return self._observation_store.obs_scales

    @property
    def _obs_voxel(self) -> float:
        return self._observation_store.obs_voxel

    @property
    def _obs_keys(self) -> dict:
        return self._observation_store.obs_keys

    @property
    def _obs_vox(self) -> dict:
        return self._observation_store.obs_vox

    @property
    def _obs_n_points(self) -> int:
        return self._observation_store.obs_n_points

    @_obs_n_points.setter
    def _obs_n_points(self, value: int) -> None:
        self._observation_store.obs_n_points = value

    @property
    def _scene_complexity_cache(self):
        return self._observation_store.scene_complexity_cache

    @_scene_complexity_cache.setter
    def _scene_complexity_cache(self, value) -> None:
        self._observation_store.scene_complexity_cache = value

    @property
    def _iq_sensor_sum(self) -> float:
        return self._observation_store.iq_sensor_sum

    @_iq_sensor_sum.setter
    def _iq_sensor_sum(self, value: float) -> None:
        self._observation_store.iq_sensor_sum = value

    @property
    def _iq_sensor_sumsq(self) -> float:
        return self._observation_store.iq_sensor_sumsq

    @_iq_sensor_sumsq.setter
    def _iq_sensor_sumsq(self, value: float) -> None:
        self._observation_store.iq_sensor_sumsq = value

    @property
    def _iq_sensor_n(self) -> int:
        return self._observation_store.iq_sensor_n

    @_iq_sensor_n.setter
    def _iq_sensor_n(self, value: int) -> None:
        self._observation_store.iq_sensor_n = value

    @property
    def _iq_sensor_stride(self) -> int:
        return self._observation_store.iq_sensor_stride

    @property
    def _iq_frame_counter(self) -> int:
        return self._observation_store.iq_frame_counter

    @_iq_frame_counter.setter
    def _iq_frame_counter(self, value: int) -> None:
        self._observation_store.iq_frame_counter = value

    @property
    def _iq_vox(self) -> dict:
        return self._observation_store.iq_vox

    @property
    def _iq_vox_cap(self) -> int:
        return self._observation_store.iq_vox_cap

    @property
    def _input_quality_cache(self):
        return self._observation_store.input_quality_cache

    @_input_quality_cache.setter
    def _input_quality_cache(self, value) -> None:
        self._observation_store.input_quality_cache = value

    @property
    def _dense_trajectory(self) -> List[dict]:
        """Property-backed alias onto ``self._trajectory_store.entries`` --
        preserves the pre-extraction external attribute name for
        ``train_streaming.py`` and existing tests."""
        return self._trajectory_store.entries

    @property
    def _dense_trajectory_flushed(self) -> int:
        return self._trajectory_store.flushed

    @_dense_trajectory_flushed.setter
    def _dense_trajectory_flushed(self, value: int) -> None:
        self._trajectory_store.flushed = value

    def _append_dense_trajectory(self, cam, frame, is_train: bool, is_holdout: bool) -> None:
        """Record this frame's pose in the full (unfiltered) trajectory buffer.

        Called for every frame that reaches a real Camera (i.e. passed the pose
        gate), regardless of keyframe-admission outcome -- so a post-hoc renderer
        can reconstruct a dense, every-frame trajectory. ``admitted`` means "kept
        as a training keyframe" (is_train and not is_holdout); guard/rejected
        frames and holdout frames are still recorded with admitted=False so their
        pose is available for dense playback even though they weren't trained on.

        Reads/writes ``self._dense_trajectory`` (the property alias onto
        ``self._trajectory_store.entries`` for real instances) rather than
        the store directly, deliberately -- this method is part of
        ``StreamingScene``'s public surface and existing tests exercise it
        via unbound-method assignment onto a plain fake-self object with its
        own ``_dense_trajectory`` list, not a real ``TrajectoryStore``; going
        through the property keeps both real and fake-self callers working.
        """
        try:
            from eligsir.utils.camera_utils import camera_to_JSON
            entry = camera_to_JSON(int(frame.index), cam)
            entry["timestamp"] = float(getattr(frame, "timestamp", 0.0) or 0.0)
            entry["admitted"] = bool(is_train and not is_holdout)
            entry["is_holdout"] = bool(is_holdout)
            self._dense_trajectory.append(entry)
        except Exception:
            pass

    def dense_trajectory_snapshot(self, since: int = 0) -> List[dict]:
        """Return dense-trajectory entries appended since index ``since``."""
        return list(self._dense_trajectory[since:])

    def holdout_status(self, frame) -> Tuple[bool, bool]:
        """``(is_holdout, is_guard)`` for *frame*, from index/ordinal alone.

        Cheap and side-effect free: no camera construction, no RGB/depth
        decode, no GPU work. Safe to call at peek time, before the source
        cursor has advanced.

        This is the single implementation of the holdout-reservation rule.
        :meth:`ingest_next_frame` calls it, and so must any pre-materialization
        gate that could otherwise skip a frame before it is ever realized as a
        test camera (issue #601): holdouts are reserved deterministically
        up-front and the selection must stay **identical across admission
        modes**, so no backpressure/burst/drop path may remove one. A frame in
        ``_holdout_indices`` ALWAYS becomes a test camera and is never trained,
        replayed, or depth-inserted, regardless of frame admission.

        For live/lazy sources (``_holdout_indices is None``) this falls back to
        online every-Nth selection, keyed to the drop-invariant
        ``source_ordinal`` when the source provides one (issue #745) and to
        ``frame.index`` otherwise.
        """
        if self._holdout_indices is not None:
            return (
                frame.index in self._holdout_indices,
                frame.index in self._guard_indices,
            )
        # issue #745 (2026-08-15, round 2): key the every-Nth rule to a
        # drop-invariant ordinal when the source provides one (today
        # only ROS2LiveStreamingFrameSource -- see
        # eligsir/io/sources/datasets/ros2_live.py's `_pump_loop`),
        # instead of `frame.index`, which for that source is a
        # resident-buffer arrival counter that renumbers every
        # subsequent frame whenever an earlier message was silently
        # dropped by the source's inner queue. That source's
        # `source_ordinal` is itself primarily a real monotonic
        # received-message sequence counter
        # (`eligsir/io/sources/ros2_live.py`'s on_rgb-assigned
        # `received_seq`, threaded through `utils/rgbd_frames.py`'s
        # `RGBDFrame.received_seq`) -- drop-invariant by construction,
        # not by timestamp approximation (that quantization scheme is
        # now only a fallback for a source that doesn't populate
        # `received_seq` at all). Sources that don't populate
        # `source_ordinal` (it defaults to None) fall back to
        # `frame.index` unchanged from pre-fix behavior -- safe there
        # because their `.index` isn't drop-sensitive in this way (e.g.
        # a fully-parsed rosbag source).
        eval_hold = int(self.args.streaming_eval_hold or 0)
        ordinal = getattr(frame, "source_ordinal", None)
        if ordinal is None:
            ordinal = frame.index
        # issue #1302: delegates to utils.holdout_rule.online_every_nth_holdout,
        # the same transport-neutral rule eligsir-ros2's lifecycle node
        # (which has no StreamingScene at all) can now also use via
        # utils.holdout_rule.LiveHoldoutCollector. Byte-identical arithmetic to
        # the inline form this replaced -- see tests/test_issue745_live_holdout_rule_hash.py.
        return (online_every_nth_holdout(ordinal, eval_hold), False)

    def ingest_frame(
        self,
        frame,
        admission_fn=None,
        defer_extent_update=False,
        visibility_fn=None,
        record_fn=None,
        pregate_fn=None,
        integrity_observer_fn=None,
    ) -> Optional[Tuple]:
        """Push one materialized observation through the sole ingest owner.

        This is the source-independent application seam. It delegates directly
        to :meth:`ingest_next_frame` with an already-supplied frame, so camera
        construction, admission, insertion and bookkeeping remain implemented
        once in the established ingestion path.
        """
        return self.ingest_next_frame(
            admission_fn=admission_fn,
            defer_extent_update=defer_extent_update,
            visibility_fn=visibility_fn,
            record_fn=record_fn,
            pregate_fn=pregate_fn,
            integrity_observer_fn=integrity_observer_fn,
            _provided_frame=frame,
        )

    def ingest_next_frame(
        self,
        admission_fn=None,
        defer_extent_update=False,
        visibility_fn=None,
        record_fn=None,
        pregate_fn=None,
        integrity_observer_fn=None,
        _provided_frame=None,
    ) -> Optional[Tuple]:
        """
        Load the next frame as a Camera and add it to the active window and
        replay buffer.  Returns (camera, StreamingRGBDFrame, is_train) or None
        when the source is exhausted.

        When ``defer_extent_update`` is True (Variant C batched ingest), the
        per-train-frame ``update_cameras_extent()`` call is skipped and a pending
        flag is set instead; the caller flushes it once after the drain batch via
        ``flush_deferred_extent_update()``. cameras_extent only grows and feeds
        densification scale, so deferring it within a single drain batch does not
        change dynamics (the readback is already every-8 gated internally).

        ``integrity_observer_fn(frame)``, when given, is called FIRST --
        unconditionally, on every materialized frame, before `pregate_fn`,
        before `admission_fn`, before even the odom-validity check below, and
        before any Camera is built (issue #806: safety/integrity observers
        must run before any discretionary prefilter can remove a frame from
        the decision pipeline; a re-init can happen on a frame the motion
        pregate would otherwise short-circuit past, and the observer must not
        depend on a frame ever reaching admission scoring). It must return
        ``None`` (no integrity event -- proceed normally) or an
        admission-stats-shaped rejection dict. A non-``None`` return takes
        precedence over `pregate_fn` for that frame: for any non-holdout,
        non-guard candidate, the integrity rejection is applied and
        `pregate_fn` is never even called (so its own rejection reason, e.g.
        "no_motion", can never mask an integrity veto). Exceptions raised by
        `integrity_observer_fn` are caught and logged (non-fatal) rather than
        crashing training -- but note this only guards against a crash in the
        observer itself, never against a real detection it successfully
        returns.

        ``pregate_fn(frame)``, when given, is called (after
        `integrity_observer_fn` reports no event) BEFORE the camera is
        built for any frame that is not a holdout/guard frame. It must
        return an admission-stats dict (rejection) or ``None`` (proceed
        normally). On rejection, `_frame_to_camera`/`visibility_fn` are
        skipped entirely for that frame -- so `record_fn` is also never
        called for it, meaning the observation DB does NOT get a record
        for pregate-rejected frames (accepted tradeoff: this is strictly
        the no-motion case admission_fn would reject anyway, see
        KeyframeAdmitter.pregate()'s docstring). `accumulate_observed_surface`
        still runs regardless, since it only needs `frame`.

        ``visibility_fn(cam, frame)``, when given, is called exactly once,
        unconditionally, right after `cam` is built and BEFORE any
        `admission_fn` invocation -- so admission scoring can see whatever it
        stashes on `cam` (e.g. `cam._observed_gaussian_uids`).

        ``record_fn(cam, frame, is_train, is_holdout)``, when given, is
        called at every exit point that has a real `cam` (guard-rejected,
        admission-rejected, and the final holdout/admitted return) so every
        candidate frame -- not just admitted/holdout ones -- gets indexed.
        A frame short-circuited by `pregate_fn` (or by `integrity_observer_fn`)
        never reaches a real `cam` and is therefore NOT passed to `record_fn`
        (see `pregate_fn` above). Both short-circuits are scoped to ordinary
        (non-holdout, non-guard) candidates, so a vetoed/pregated HOLDOUT or
        GUARD frame is unaffected: it still builds a real `cam` and is still
        passed to `record_fn` exactly as it would be with no event at all
        (issue #1125).
        """
        # Lazy import (runtime-only) of the env-gated phase profiler so m_ingest
        # sub-stages are attributed without a circular import at module load.
        from eligsir.utils.ingest_phase_profiler import _timed

        frame = _provided_frame
        if frame is None:
            if not self.has_next_frame():
                return None
            with _timed("ingest_get_source_frame"):
                frame = self._get_source_frame(self._source_idx)
            self._source_idx += 1
            self._release_source_frames_upto_cursor()
        if frame is None:
            return None  # source exhausted early (bag-end / idle-timeout)

        # Mandatory integrity observation (#806) -- runs on EVERY
        # materialized frame, unconditionally, before the odom-validity
        # check, before `pregate_fn`, before `admission_fn`, and before any
        # Camera is built. Not gated on is_holdout/is_guard: an integrity
        # event (e.g. an ORB-SLAM3 map re-init) must be observed regardless
        # of what kind of candidate this frame would otherwise be -- only
        # the REJECTION precedence below is scoped to ordinary (non-holdout,
        # non-guard) candidates, matching where `pregate_fn` itself applies.
        integrity_reject_stats = None
        if integrity_observer_fn is not None:
            with _timed("ingest_integrity_observer"):
                try:
                    integrity_reject_stats = integrity_observer_fn(frame)
                except Exception as _integrity_exc:  # noqa: BLE001 -- a crash in observation must never take down real training
                    print(
                        f"[integrity-observer] frame observation failed (non-fatal, "
                        f"observation only -- does NOT clear any veto already in "
                        f"effect): {_integrity_exc}",
                        flush=True,
                    )
                    integrity_reject_stats = None

        is_train = True
        # Real holdouts are reserved deterministically up-front (admission-independent):
        # a frame in _holdout_indices ALWAYS becomes a test camera and is never trained,
        # replayed, or depth-inserted — regardless of frame admission. For live/lazy
        # sources (_holdout_indices is None) fall back to online every-Nth selection.
        # Computed here (before the camera is built / pregate_fn runs) purely from
        # frame.index, since holdout/guard status must never be affected by, or
        # gate, camera construction. The rule itself lives in the public
        # `holdout_status()` predicate so a cheap pre-materialization caller
        # (issue #601's backpressure owner) can ask the SAME question without
        # a second, divergent copy of the rule.
        #
        # Moved ABOVE the odom-validity check (issue #1135) purely so that
        # check can attribute a stale-pose holdout rejection -- `holdout_status`
        # is cheap/side-effect-free (see its own docstring), so reordering it
        # changes no behavior for any other caller.
        is_holdout, is_guard = self.holdout_status(frame)

        # Skip frames where live odometry failed — inserting depth from a copied
        # (wrong) pose corrupts the Gaussian map.
        #
        # Issue #1135 (2026-08-23): unlike the #1125 integrity veto just above
        # this method's docstring, this rejection is NOT bypassed for holdouts.
        # #1125's bypass was safe because the map-reinit veto's harm channel is
        # map corruption via TRAINING/depth-insertion, which a holdout camera
        # never triggers -- so admitting a re-based-but-internally-coherent
        # pose into the test set only cost a possibly-mispositioned eval
        # camera, already flagged run-wide by `map_reinit_invalidated`.
        #
        # `_odom_valid is False` is a different failure mode: per
        # `eligsir/io/sources/datasets/orbbec_rosbag.py`'s live-odometry
        # fallback, a rejected frame's pose is literally a COPY of the last
        # successfully-tracked keyframe's pose (`curr_pose = key_pose.copy()`)
        # -- wrong in an unbounded, frame-specific way, not a coherent re-based
        # coordinate frame. A holdout eval camera built from it would render
        # the wrong part of the scene entirely and silently pollute
        # `mean_test_psnr`, with no run-level flag analogous to
        # `map_reinit_invalidated` to catch it. So the veto stays universal
        # here (holdouts/guards included) -- but the resulting reservation
        # shortfall is no longer silent: every holdout dropped this way is
        # counted in `self._stale_pose_holdout_rejected`, surfaced in
        # `report.json["geometry"]["trajectory_coverage"]` by
        # `compute_trajectory_coverage()` (see `utils/partition_eval.py`), so
        # a shrunken `n_test` is attributable instead of mysterious.
        if getattr(frame, "_odom_valid", None) is False:
            self._keyframes_rejected += 1
            if is_holdout:
                self._stale_pose_holdout_rejected += 1
            return None, frame, False

        # Integrity veto takes precedence over the cheap motion pregate
        # (#806): if `integrity_observer_fn` already flagged a hard-invalid
        # state for this frame (e.g. a permanent ORB-SLAM3 map-reinit veto),
        # apply that rejection -- but ONLY to ordinary (non-holdout,
        # non-guard) candidates, exactly like the `pregate_fn` stage it takes
        # precedence over.
        #
        # Issue #1125 (2026-08-23): this branch used to fire universally
        # ("including holdouts/guards"), which contradicted this method's own
        # docstring ("for any non-holdout, non-guard candidate, the integrity
        # rejection is applied") and `holdout_status()`'s documented rule
        # ("a frame in `_holdout_indices` ALWAYS becomes a test camera ...
        # regardless of frame admission"). Because it returned before
        # `_frame_to_camera` was ever called, an integrity event silently
        # shrank the realized holdout set: `n_test` dropped below what the
        # holdout rule reserved and `realized_holdout_hash` changed, making
        # `--baseline_run` comparisons refuse or mislead -- the exact failure
        # mode #601/#1084's pre-materialization bypass exists to prevent.
        #
        # Why the veto does not need to cover holdouts: its harm channel is
        # map corruption -- a pose from a re-based tracker coordinate frame
        # poisons the Gaussian map through TRAINING and depth insertion. A
        # holdout camera is never trained, never replayed, and never
        # depth-inserted, so that mechanism cannot fire through one. What
        # remains is a possibly-mispositioned eval camera, i.e. a polluted
        # `mean_test_psnr` -- which is precisely what
        # `report.json["geometry"]["trajectory_coverage"]["map_reinit_invalidated"]`
        # already flags for the whole run. A silently incomplete holdout set
        # is strictly worse than an attributable one.
        #
        # Guards are included in the bypass for the same reason and for
        # symmetry with `pregate_fn`: they are never trained and never
        # evaluated, they only need a real `cam` for dense-trajectory
        # bookkeeping.
        #
        # Note the observer itself still runs unconditionally, for every
        # frame, ahead of everything else (#806) -- only the REJECTION is
        # scoped here. Detection, logging, and the permanent run-level veto
        # in `MapReinitTracker` are untouched.
        if integrity_reject_stats is not None and not is_holdout and not is_guard:
            frame._streaming_admission_stats = integrity_reject_stats
            self._keyframes_rejected += 1
            self._cache_inactive_frame(frame)
            with _timed("ingest_accumulate_surface"):
                self.accumulate_observed_surface(frame)
            return None, frame, False

        # Cheap pregate: only for candidates that will actually go through
        # admission_fn (never for holdout/guard frames, which always need a
        # real `cam` regardless of motion). Rejects WITHOUT building `cam`,
        # so `visibility_fn`/`record_fn` never run for this frame (see the
        # docstring above for the accepted observation-DB tradeoff).
        if not is_holdout and not is_guard and pregate_fn is not None:
            with _timed("ingest_pregate"):
                pregate_stats = pregate_fn(frame)
            if pregate_stats is not None:
                frame._streaming_admission_stats = pregate_stats
                self._keyframes_rejected += 1
                self._cache_inactive_frame(frame)
                with _timed("ingest_accumulate_surface"):
                    self.accumulate_observed_surface(frame)
                return None, frame, False

        # D2 (Wave 2 consolidation): defer the RGB decode + H2D upload for
        # any frame that still has to pass admission_fn -- rejected
        # candidates then never pay that cost. Holdout/guard frames always
        # need a real image immediately (holdout -> eval camera, guard ->
        # dense-trajectory bookkeeping is cheap either way) so they stay
        # eager. `cam.materialize_pending_image()` is called once admission
        # is confirmed, below.
        with _timed("ingest_frame_to_camera"):
            cam = self._frame_to_camera(frame, load_image=(is_holdout or is_guard))

        # Task 5 fix (C1): visibility_fn must run BEFORE any admission_fn call
        # below so that admission scoring (e.g. DB-diversity scoring) can see
        # whatever it stashes on `cam` (e.g. `cam._observed_gaussian_uids`).
        if visibility_fn is not None:
            visibility_fn(cam, frame)

        # Fold this frame's observed surface into the SV accumulator (every arrived
        # frame, depth still in hand) — scene expressiveness is a property of the
        # observed scene, independent of admission / train-vs-holdout.
        with _timed("ingest_accumulate_surface"):
            self.accumulate_observed_surface(frame)

        if is_guard:
            # Withheld from training to widen the parallax gap to a holdout; not evaluated.
            self._keyframes_rejected += 1
            self._append_dense_trajectory(cam, frame, is_train=False, is_holdout=False)
            if record_fn is not None:
                record_fn(cam, frame, is_train=False, is_holdout=False)
            return None, frame, False
        if is_holdout:
            # Hold-out frame: add to test set only, skip training/replay/insertion.
            self._test_cameras.append(cam)
            is_train = False
        else:
            if admission_fn is not None:
                with _timed("ingest_admission_fn"):
                    admit, admission_stats, admission_render_pkg = admission_fn(cam, frame)
                frame._streaming_admission_stats = admission_stats
                if not admit:
                    self._keyframes_rejected += 1
                    self._cache_inactive_frame(frame)
                    self._append_dense_trajectory(cam, frame, is_train=False, is_holdout=False)
                    if record_fn is not None:
                        record_fn(cam, frame, is_train=False, is_holdout=False)
                    return None, frame, False
                cam._streaming_admission_stats = admission_stats
                cam._streaming_admission_render_pkg = admission_render_pkg
            # D2: this frame is now a confirmed keyframe -- materialize the
            # deferred RGB (no-op if `cam` was built eagerly, e.g. no
            # admission_fn wired). Must happen before `cam` is appended to
            # train_cameras/replay_buffer, both of which assume
            # `original_image` is present.
            cam.materialize_pending_image()
            self._keyframes_admitted += 1
            self.train_cameras.append(cam)
            from eligsir.utils.partition_eval import camera_source_index as _camera_source_index
            src_idx = _camera_source_index(cam)
            if src_idx is not None:
                self._active_source_indices.add(src_idx)
                self._active_cam_by_idx[src_idx] = cam
                self._last_admitted_source_idx = src_idx

            # Page the image of the camera that just fell out of the "hot" recency
            # window back to CPU. Without this, every admitted keyframe's full-res
            # image stays GPU-resident forever (see offload_camera_image docstring)
            # -- unbounded by cap_max, unbounded by anything except trajectory length
            # x resolution. `prepare_camera_for_render` re-uploads on demand right
            # before a camera is actually rendered/trained on.
            self._offload_evicted_camera()

            replay_size = getattr(self.args, "streaming_replay_buffer", 512)
            self._replay_buffer.append(cam)
            if len(self._replay_buffer) > replay_size:
                # Protect bootstrap frames: evict the oldest non-bootstrap frame first.
                # Only fall back to evicting bootstrap frames when the buffer contains
                # nothing but bootstrap frames.
                evict_idx = next(
                    (i for i, c in enumerate(self._replay_buffer) if not getattr(c, "_is_bootstrap", False)),
                    0,
                )
                self._replay_buffer.pop(evict_idx)

            # H10: update global reservoir (every Nth train frame kept permanently).
            # When hybrid_keyframe admission is active, default stride=1 so that
            # every admitted keyframe is retained — prevents early-scene fading when
            # the ring replay_buffer overflows.
            self._n_train_frames_ingested += 1
            admission_mode = getattr(self.args, "streaming_frame_admission", "all")
            default_stride = 1 if admission_mode == "hybrid_keyframe" else 0
            reservoir_stride = getattr(self.args, "streaming_global_reservoir_stride", default_stride)
            if reservoir_stride > 0 and self._n_train_frames_ingested % reservoir_stride == 0:
                self._global_reservoir.append(cam)

            # H8: reset warmup counter whenever a new train frame arrives
            self._steps_since_new_frame = 0
            if defer_extent_update:
                # Variant C: defer to one flush after the drain batch.
                self._pending_extent_update = True
            else:
                self.update_cameras_extent()

        self.current_camera = cam
        StreamingScene._prepare_camera_for_render_owned(self, cam, device="cuda")
        if is_holdout:
            # Issue #398: `prepare_camera_for_render` above has just (re-)placed
            # this holdout's GT image on the GPU, where -- unlike a train
            # camera, which the hot-window eviction pages out a few frames
            # later -- nothing would ever evict it again. Page it straight back
            # to CPU: holdout pixels are only ever read inside an evaluation
            # pass, which uploads its own transient per-camera copy. The small
            # `cam.gsplat` tensors built just above are deliberately kept (they
            # are geometry, ~a few hundred bytes, and are what makes the eval
            # render cheap). See `offload_holdout_image`.
            offload_holdout_image(self.args, cam)
        self._append_dense_trajectory(cam, frame, is_train=is_train, is_holdout=is_holdout)
        if record_fn is not None:
            record_fn(cam, frame, is_train=is_train, is_holdout=is_holdout)
        return cam, frame, is_train

    # ------------------------------------------------------------------
    # Camera sampling
    # ------------------------------------------------------------------

    def _effective_keyframe_window(self) -> int:
        """Return the effective keyframe window size.

        When streaming_keyframe_coverage > 0, the window grows to keep at least
        that fraction of all ingested train frames in the active window.  The
        fixed streaming_keyframe_window acts as a minimum floor.

        While the number of streamed frames is still less than the bootstrap
        span, the entire bootstrap is kept inside the local window so seed
        Gaussians receive recent-stratum gradient pressure.
        """
        # No getattr fallback for streaming_keyframe_window: it is an
        # unconditional StreamingParams default, always present on the real
        # parsed args (issue #368 -- old getattr(..., 8) fallback drifted
        # from the real default of 120; removed rather than re-stated).
        # streaming_keyframe_coverage's getattr(..., 0.0) fallback is left
        # as-is -- 0.0 is the documented "coverage growth off" sentinel
        # above, not a drifted copy of the real (nonzero) default.
        k_floor = self.args.streaming_keyframe_window
        coverage = float(self.args.streaming_keyframe_coverage)
        n_boot = self._bootstrap_count
        n_total = len(self.train_cameras)

        if coverage > 0 and n_total:
            k_adaptive = int(coverage * n_total)
        else:
            k_adaptive = 0

        bootstrap_in_window = getattr(self.args, "streaming_bootstrap_in_window", True)
        if bootstrap_in_window and n_boot > 0:
            n_streamed = max(0, n_total - n_boot)
            if n_streamed < n_boot:
                return max(k_floor, k_adaptive, n_boot)

        return max(k_floor, k_adaptive)

    def get_local_cameras(self) -> List:
        k = self._effective_keyframe_window()
        if not self.train_cameras:
            return []
        from eligsir.runtime.payload_residency import payload_ready_cameras

        return payload_ready_cameras(self.train_cameras[-k:])

    def get_nearest_cameras(self, cam, k: int = 4) -> List:
        """Return up to k cameras nearest to *cam* by Euclidean camera-center distance.

        Used by the consolidation burst to give newborn Gaussians immediate
        multi-view supervision from spatially nearby historical frames.
        Falls back to the most-recent cameras when fewer than k are available.
        """
        if not self.train_cameras or cam is None:
            return []
        from eligsir.runtime.payload_residency import payload_ready_cameras

        pool = payload_ready_cameras(self.train_cameras)
        if not pool:
            return []
        try:
            import torch as _t
            query = cam.camera_center.detach().reshape(1, 3)
            centers = _t.stack([c.camera_center.detach().reshape(3) for c in pool])
            dists = _t.cdist(query, centers).squeeze(0)  # [N]
            topk = min(k, len(pool))
            indices = _t.topk(dists, topk, largest=False).indices.tolist()
            return [pool[i] for i in indices]
        except Exception:
            return pool[-k:] if len(pool) >= k else list(pool)

    def recent_camera_baseline(self, n_frames: int = 5) -> float:
        """Translational baseline (metres) of the camera over the last n frames.

        Part D2: the parallax available to disambiguate a newly inserted depth
        point along its view ray. Returns the max pairwise distance among the
        last ``n_frames + 1`` camera centres (robust to a single jittered pose).
        ~0 when the camera is stalled (no parallax → depth highly uncertain).
        """
        cams = self.train_cameras
        if len(cams) < 2:
            return 0.0
        tail = cams[-(int(n_frames) + 1):]
        try:
            import torch as _torch
            centers = _torch.stack([c.camera_center.detach().reshape(3) for c in tail])
            d = _torch.cdist(centers, centers)
            return float(d.max().item())
        except Exception:
            return 0.0

    def _ensure_camera_caches_np(self) -> None:
        """Incrementally populate host-side center + view-direction caches.

        Index-aligned with ``train_cameras``. Each new camera's world-space
        center (``camera_center``) and optical axis (``R[:, 2]`` — cam +Z in
        world; ``R`` is camera-to-world in this codebase, see ``getWorld2View2``)
        are appended exactly once. Called on demand by the spatial pose-window
        path only, so the sync-free GPU extent cache stays the default hot path.
        Restores the assignment of ``_camera_viewdirs_np`` that was dropped on
        merge (the spatial window read it but nothing wrote it -> AttributeError).
        """
        cache = self._camera_centers_np
        dirs = self._camera_viewdirs_np
        n = len(self.train_cameras)
        # Defensive: rebuild from scratch if the caches desynced (e.g. cameras
        # were pruned/reset and the cache now overshoots the camera list).
        if len(cache) > n or len(dirs) > n or len(cache) != len(dirs):
            cache.clear()
            dirs.clear()
        for c in self.train_cameras[len(cache):]:
            try:
                cache.append(np.asarray(c.camera_center.detach().cpu().numpy(), dtype=np.float64))
                R = np.asarray(c.R, dtype=np.float64)
                dirs.append(R[:, 2])
            except Exception:
                # On any per-camera failure leave caches length-aligned but short;
                # _spatial_local_cameras treats a short cache as "disabled".
                break

    def _spatial_local_cameras(self) -> Tuple[List, Optional[List[float]]]:
        """Spatial pose-window: select nearby+aligned cameras around the current one.

        Returns ``(cameras, weights)`` where ``weights[i]`` is the spatial weight
        for ``cameras[i]``. Falls back to the temporal window (returning
        ``weights=None``) when disabled-precondition holds: no current camera, the
        camera hasn't moved, caches are empty/desynced, or no candidate lies
        within the radius. Caller treats ``weights=None`` as "use temporal kw".
        """
        cur = self.current_camera
        temporal = self.get_local_cameras()
        if cur is None or not self.train_cameras:
            return temporal, None
        # Lazily (re)build the host-side center / view-direction caches. The hot
        # update_cameras_extent path uses a sync-free GPU cache and intentionally
        # does NOT populate these np caches, so build them here on demand — only
        # the spatial-window path (this method) needs them, and it is opt-in.
        self._ensure_camera_caches_np()
        cache = self._camera_centers_np
        dirs = self._camera_viewdirs_np
        n = len(self.train_cameras)
        # Require fully-populated, index-aligned caches.
        if len(cache) != n or len(dirs) != n:
            return temporal, None
        try:
            cur_center = np.asarray(cur.camera_center.detach().cpu().numpy(), dtype=np.float64)
            cur_dir = np.asarray(cur.R, dtype=np.float64)[:, 2]
        except Exception:
            return temporal, None
        radius = float(getattr(self.args, "streaming_spatial_window_radius", 1.5))
        sigma = float(getattr(self.args, "streaming_spatial_window_sigma", 0.75))
        centers = torch.from_numpy(np.stack(cache))
        viewdirs = torch.from_numpy(np.stack(dirs))
        w = spatial_window_weights(
            centers,
            viewdirs,
            torch.from_numpy(cur_center),
            torch.from_numpy(cur_dir),
            radius,
            sigma,
        )
        nz = torch.nonzero(w > 0.0, as_tuple=False).flatten().tolist()
        if not nz:
            return temporal, None
        from eligsir.runtime.payload_residency import camera_payload_ready

        nz = [i for i in nz if camera_payload_ready(self.train_cameras[i])]
        cams = [self.train_cameras[i] for i in nz]
        weights = [float(w[i].item()) for i in nz]
        return cams, weights

    @staticmethod
    def _frame_carries_pixel_data(frame: "StreamingRGBDFrame") -> bool:
        """True when a cached frame already holds decoded pixels (ros2_live,
        ScanNet .sens) rather than a cheap path reference (RGBDSequence/TUM/
        ScanNet-path) -- see the tiered-cache comment in __init__. Source-
        agnostic: just checks which of the payload fields is populated, so
        it needs no knowledge of which FrameSource produced the frame."""
        return (
            getattr(frame, "_rgb_np", None) is not None
            or getattr(frame, "_depth_np", None) is not None
            or getattr(frame, "_rgb_bytes", None) is not None
            or getattr(frame, "_depth_bytes", None) is not None
        )

    def _cache_inactive_frame(self, frame: "StreamingRGBDFrame") -> None:
        """Stash a rejected (pregate- or admission-rejected) frame into the
        "cold" tier for possible later promotion via promote_from_cache() --
        but ONLY when it falls within --streaming_cold_window source-index
        steps of the most recently admitted keyframe. Cold tier is
        deliberately scoped to frames "surrounding" a selected keyframe, not
        every rejected frame in the trajectory (that would be most of the
        trajectory under a gated admission mode -- unbounded and mostly
        useless: a frame far from any keyframe is unlikely to be a good
        promotion candidate for a LOCAL partition-focus fix). Bounded LRU on
        top as a second safety net -- --streaming_inactive_cache_size for
        metadata-only frames (see _frame_to_camera), or the much smaller
        --streaming_live_inactive_cache_size when the frame already carries
        decoded pixels (_frame_carries_pixel_data)."""
        if self._frame_carries_pixel_data(frame):
            cache_size = int(getattr(self.args, "streaming_live_inactive_cache_size", 150))
        else:
            cache_size = int(getattr(self.args, "streaming_inactive_cache_size", 2000))
        if cache_size <= 0:
            return
        idx = getattr(frame, "index", None)
        if idx is None:
            return
        cold_window = int(getattr(self.args, "streaming_cold_window", 2))
        if cold_window >= 0 and self._last_admitted_source_idx is not None:
            if abs(idx - self._last_admitted_source_idx) > cold_window:
                return
        self._inactive_frame_cache[idx] = frame
        self._inactive_frame_cache.move_to_end(idx)
        from eligsir.mapping.observation_db import BufferStage
        self._mark_buffer_stage(idx, BufferStage.COLD_CACHE)
        # Cold-tier overflow spills into the icecold archive rather than being
        # discarded outright -- "dropped out and never called for a while"
        # should still be recoverable, just from the bigger/slower tier.
        while len(self._inactive_frame_cache) > cache_size:
            _evicted_idx, _evicted_frame = self._inactive_frame_cache.popitem(last=False)
            self._archive_to_icecold(_evicted_idx, _evicted_frame)

    def _archive_to_icecold(self, idx: int, frame: "StreamingRGBDFrame") -> None:
        """Insert a frame into the big, CPU-RAM-resident icecold archive tier
        (bounded LRU via --streaming_icecold_cache_size, or the smaller
        --streaming_live_icecold_cache_size when the frame already carries
        decoded pixels -- see _frame_carries_pixel_data). 0 disables the
        tier -- frames aged out of the cold tier or demoted from active are
        then simply discarded, as before this tier existed."""
        if self._frame_carries_pixel_data(frame):
            icecold_size = int(getattr(self.args, "streaming_live_icecold_cache_size", 1500))
        else:
            icecold_size = int(getattr(self.args, "streaming_icecold_cache_size", 20000))
        if icecold_size <= 0:
            return
        self._icecold_frame_cache[idx] = frame
        self._icecold_frame_cache.move_to_end(idx)
        from eligsir.mapping.observation_db import BufferStage
        self._mark_buffer_stage(idx, BufferStage.ICECOLD_CACHE)
        while len(self._icecold_frame_cache) > icecold_size:
            # Frame is aged out of the DB-tracked tiers entirely once evicted
            # from icecold too -- there's no colder tier to spill into, so
            # its DB record reverts to EVICTED_REDUNDANT (matches what a
            # never-cached rejected frame's record already looks like).
            _hard_evicted_idx, _ = self._icecold_frame_cache.popitem(last=False)
            self._mark_buffer_stage(_hard_evicted_idx, BufferStage.EVICTED_REDUNDANT)

    def cached_indices_in_partitions(self, partition_ids: "set[int]", max_source_index: int, k: int) -> "set[int]":
        """Return the set of source-frame indices currently sitting in
        either cache tier (cold or icecold) whose partition_id(idx,
        max_source_index, k) falls in ``partition_ids``.

        This is the missing supply-side lookup for train_streaming.py's
        psnr-tick call site: bad partitions are detected from held-out test
        PSNR over cameras already in train_cameras, but the frames that
        would actually IMPROVE a thin partition are the ones NOT yet
        active -- sitting in the cache tiers precisely because they were
        rejected at admission. Without this, set_priority_frame_weights()
        only ever assigns weight to indices already in train_cameras
        (promote_from_cache then has nothing new to promote for any of
        them, since they're already active) -- a silent no-op on exactly
        the partitions that need new frames the most."""
        from eligsir.utils.partition_eval import partition_id as _partition_id
        found = set()
        for idx in self._inactive_frame_cache.keys():
            if _partition_id(idx, max_source_index, k=k) in partition_ids:
                found.add(idx)
        for idx in self._icecold_frame_cache.keys():
            if _partition_id(idx, max_source_index, k=k) in partition_ids:
                found.add(idx)
        return found

    def _predecode_frames_parallel(self, frames: list) -> None:
        """Decode RGB+depth for multiple frames concurrently on a CPU thread
        pool, before any of them is turned into a Camera one at a time on
        the training thread. A promote_from_cache() call for a genuinely
        under-covered partition routinely materializes 100+ frames in one
        shot (observed live: 118-120 frames/tick on ScanNet++) -- doing
        that decode serially is exactly the per-frame CPU stall this
        mechanism exists to fix elsewhere.

        Real parallelism, not just GIL-serialized threads: Pillow's JPEG
        decode (libjpeg-turbo, the codec used for ScanNet++/most RGB
        sources) releases the GIL during the actual decode loop. Only the
        decode step is parallelized here -- _frame_to_camera's per-camera
        GPU upload still happens afterward, one at a time, on the main
        thread (loadCam is not batch-upload-capable without a deeper
        refactor of the camera-construction path; tracked as a follow-up,
        not attempted here). Populating frame._rgb_np lets _frame_to_camera
        take its existing fast path (skips the redundant PIL decode) when
        the frame's native size already matches its target size; when a
        downscale is pending, this still overlaps the disk read + JPEG
        decode across frames even though the final resize stays serial.
        Best-effort: any per-frame decode failure is swallowed here and
        surfaces normally (as a slower serial decode) inside
        _frame_to_camera instead."""
        if len(frames) < 2:
            return
        n_workers = min(8, max(1, (os.cpu_count() or 4) - 1), len(frames))
        from concurrent.futures import ThreadPoolExecutor
        from eligsir.utils.streaming_frames import load_frame_rgb, load_frame_depth_np

        def _decode_one(frame):
            try:
                if getattr(frame, "_rgb_np", None) is None:
                    frame._rgb_np = np.asarray(load_frame_rgb(frame))
            except Exception:
                pass
            try:
                load_frame_depth_np(frame)
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_decode_one, frames))

    def promote_from_cache(self, indices: "set[int]", resolution_cache_fn=None) -> int:
        """Materialize and splice cached-but-inactive frames whose source index
        is in ``indices`` back into train_cameras, so partition-focus sampling
        has real new information to draw on instead of just reweighting the
        same already-admitted views. Checks the cold tier first, then the
        (much larger) icecold archive. Both store lightweight frame metadata
        and rebuild a Camera via _frame_to_camera on promotion. Frames
        already active (in train_cameras) or present in neither tier are
        skipped. Decode for all promoted frames is warmed concurrently first
        (see _predecode_frames_parallel) before the (still serial, one GPU
        upload at a time) _frame_to_camera loop. Returns the number of
        frames promoted.

        ``resolution_cache_fn`` (issue #481(b), optional): ``fn(frame) ->
        dict`` producing the multires ``resolution_cache`` for a promoted
        frame -- in production ``train_streaming.build_frame_resolution_cache``,
        bound to the run's own low_res_divisor/depthless_rgb_cache/device.
        Every frame reachable here was recorded in the observation DB as a
        REJECTED candidate (``is_train=False``), and
        ``record_frame_observation`` only builds a cache for is_train frames --
        so promotion turns it into a real training camera whose DB record has
        an empty cache, making ``bind_resolution_tier`` return False
        ("cache_empty") for it for the rest of the run. That is what routes
        such a camera into ``depth_helpers._get_sensor_depth``'s lazy path
        (see that function's #384 comment block). Passing this closes the gap;
        omitting it reproduces the pre-#481 behaviour exactly. Best-effort:
        a build failure is swallowed (the camera is still promoted, just
        without a cache), matching the live-camera hard constraint that a
        telemetry/cache path must never take a run down."""
        from eligsir.mapping.observation_db import BufferStage
        to_promote = []
        for idx in list(indices):
            if idx in self._active_source_indices:
                continue
            frame = self._inactive_frame_cache.pop(idx, None)
            if frame is None:
                frame = self._icecold_frame_cache.pop(idx, None)
            if frame is None:
                continue
            to_promote.append((idx, frame))
        if not to_promote:
            return 0
        self._predecode_frames_parallel([f for _, f in to_promote])
        promoted = 0
        for idx, frame in to_promote:
            cam = self._frame_to_camera(frame)
            self.train_cameras.append(cam)
            self._active_source_indices.add(idx)
            self._active_cam_by_idx[idx] = cam
            self._mark_buffer_stage(idx, BufferStage.MAPPING_KEYFRAME)
            self._attach_promoted_resolution_cache(idx, frame, resolution_cache_fn)
            promoted += 1
        if promoted:
            # CRITICAL: without this, promoted cameras' freshly-uploaded
            # full-res images accumulate GPU-resident forever. The normal
            # admission path's per-append eviction (ingest_next_frame) never
            # runs for promoted cameras -- see _enforce_hot_window's
            # docstring. Root-caused 2026-07-27 from a live OOM: baseline
            # (no partition-focus) safely reached N=300079 with images
            # correctly hot-window-bounded; this mechanism OOM'd at a LOWER
            # N=261743 on the same card/config, from GPU-resident images
            # alone, not Gaussian count.
            self._enforce_hot_window()
        return promoted

    def _attach_promoted_resolution_cache(self, idx, frame, resolution_cache_fn) -> bool:
        """Build + attach the multires resolution_cache for one just-promoted
        frame (issue #481(b)). See ``promote_from_cache``'s docstring.

        Skipped entirely when no builder was supplied, no DB is attached, the
        frame was never recorded there, or a cache is already resident (a frame
        promoted, demoted and promoted again must not pay the rebuild twice).
        Returns True only when a cache was actually attached."""
        if resolution_cache_fn is None or self.observation_db is None or idx is None:
            return False
        try:
            record = self.observation_db.get_frame(idx)
            if record is None or record.resolution_cache:
                return False
            cache = resolution_cache_fn(frame)
            if not cache:
                return False
            return bool(self.observation_db.attach_resolution_cache(idx, cache))
        except Exception as exc:  # noqa: BLE001 -- must never take a live run down
            print(f"[streaming] promote_from_cache: resolution-cache rebuild failed "
                  f"for frame {idx} (non-fatal, camera still promoted): {exc}", flush=True)
            return False

    def _offload_camera_hot_window(self) -> int:
        """Return the raw hot-window size used by every offload decision.

        Deliberately the RAW ``--streaming_keyframe_window``, never
        ``_effective_keyframe_window()`` (the coverage-adaptive *sampling*
        window, which at the default ``--streaming_keyframe_coverage 0.6``
        reaches ~4800 on an 8000-frame trajectory). Every residency decision --
        the admission-path eviction, ``_enforce_hot_window``'s sweep, and
        ``hot_camera_ids``'s predicate -- must agree on one boundary, or a
        camera can be classified HOT by one and COLD by another. See issue #396
        and ``utils/camera_residency.py``'s module docstring.

        No getattr fallback: streaming_keyframe_window is an unconditional
        StreamingParams default, always present on the real parsed args
        (issue #368 -- an old getattr(..., 8) fallback drifted from the real
        default of 120; removed rather than re-stated).
        """
        return max(1, int(self.args.streaming_keyframe_window))

    def _offload_evicted_camera(self) -> bool:
        """Page out the one camera that just fell out of the hot recency window.

        Called once per admitted keyframe from ``ingest_next_frame``. Without
        this, every admitted keyframe's full-res image stays GPU-resident
        forever (see ``offload_camera_image``'s docstring) -- unbounded by
        cap_max, unbounded by anything except trajectory length x resolution.
        ``prepare_camera_for_render`` re-uploads on demand right before a camera
        is actually rendered/trained on.

        This is a monotonic single-index cursor: it visits index
        ``len(train_cameras) - hot_window - 1`` exactly once and never revisits
        it. That is precisely why replay page-ins need their own release path
        (issue #396) -- once the cursor is past a camera, this method can never
        page it out again. Extracted from the inline admission-path body so the
        cursor semantics are directly testable
        (``tests/test_replay_camera_residency.py``).

        Returns True if a camera was actually paged out.
        """
        if not bool(getattr(self.args, "streaming_offload_camera_images", True)):
            return False
        evict_at = len(self.train_cameras) - self._offload_camera_hot_window() - 1
        if evict_at < 0:
            return False
        from scene.cameras import offload_camera_image
        return bool(offload_camera_image(self.train_cameras[evict_at]))

    def hot_camera_ids(self) -> "set[int]":
        """``id()`` of every camera whose image may legitimately stay on the GPU.

        The HOT set is the tail ``_offload_camera_hot_window()`` of
        ``train_cameras`` -- the same boundary ``_offload_evicted_camera`` and
        ``_enforce_hot_window`` use -- plus ``current_camera``, which the H8
        new-frame warmup path returns directly for
        ``--streaming_new_frame_warmup_steps`` consecutive iterations (paging it
        out between those iterations would cost an H2D transfer every iteration
        for no memory benefit) and which can sit outside ``train_cameras``
        entirely on a holdout/non-admitted frame.

        Consumed by ``utils.camera_residency.CameraResidencyLedger.release``
        (issue #396). Returns identities, not cameras, so the caller holds no
        extra references and ``Camera`` needs no ``__eq__``/``__hash__``.
        """
        hot = {id(c) for c in self.train_cameras[-self._offload_camera_hot_window():]} \
            if self.train_cameras else set()
        cur = getattr(self, "current_camera", None)
        if cur is not None:
            hot.add(id(cur))
        return hot

    def retire_live_admitted_payloads(self) -> int:
        """Release non-replayable live RGB-D payloads, keeping pose records.

        Canonical preselected mapping deliberately retains every admitted
        camera for trajectory/accounting semantics.  The payload-retirement
        helper drops only old cameras that no active sampler can select; the
        current camera, hot window, replay ring, and other sampling pools stay
        reloadable.  This is intentionally separate from the tiered rejected
        frame archive, which is not a safe owner for preselected live payloads.
        """
        from eligsir.runtime.payload_residency import (
            retire_live_admitted_payloads,
        )

        return retire_live_admitted_payloads(self)

    def _enforce_hot_window(self) -> None:
        """Page every train_camera's image to CPU except the most recent
        --streaming_keyframe_window, via an explicit full sweep rather than
        the normal admission path's O(1) incremental position counter
        (ingest_next_frame's evict_at logic). That counter assumes
        train_cameras only ever grows by appending at the end; it silently
        stops being correct the moment anything mutates the list out of
        that order -- promote_from_cache appends without ever evicting
        anything, and demote_partitions' list.remove() shifts every later
        index down by one, desyncing the counter from the cameras it
        should have visited. Call this after any such mutation (promote_
        from_cache/demote_partitions) to restore the hot-window invariant
        from scratch; the cheap per-admission counter remains the fast path
        for the common case (a single new admitted frame)."""
        if not bool(getattr(self.args, "streaming_offload_camera_images", True)):
            return
        hot_window = self._offload_camera_hot_window()
        if len(self.train_cameras) <= hot_window:
            return
        from scene.cameras import offload_camera_image
        for cam in self.train_cameras[:-hot_window]:
            offload_camera_image(cam)

    # ------------------------------------------------------------------
    # Holdout/test image residency (issue #398)
    # ------------------------------------------------------------------
    def _offload_holdout_image(self, cam) -> bool:
        """Instance-level alias of the module-level `offload_holdout_image`.

        The ingest/resume paths deliberately call the MODULE-level function
        rather than this method: both are exercised by unbound-method-on-a-
        fake-self test harnesses (see tests/test_streaming_scene_dense_
        trajectory.py) that would each need a new stub attribute otherwise.
        """
        return offload_holdout_image(self.args, cam)

    def offload_test_camera_images(self) -> int:
        """Return holdout image residency to its pre-evaluation bound.

        An evaluation pass may page a holdout image onto the GPU (anything that
        calls `prepare_camera_for_render` on a test camera rebinds
        `cam.original_image` to a GPU tensor and leaves it there). Call this
        after any mid-training evaluation so residency does not ratchet upward
        over a run. No-op -- and safely so -- when nothing was paged in.

        Returns the number of cameras actually moved off the GPU.
        """
        if not bool(getattr(self.args, "streaming_offload_camera_images", True)):
            return 0
        from scene.cameras import offload_camera_images
        return offload_camera_images(self._test_cameras)

    def gpu_resident_test_image_count(self) -> int:
        """Number of holdout cameras whose GT image is currently GPU-resident.

        The measurable form of issue #398's acceptance criterion 1 ("training-
        time test-image residency is zero or explicitly bounded to a tiny
        cache"). Read-only.
        """
        from scene.cameras import gpu_resident_image_count
        return gpu_resident_image_count(self._test_cameras)

    def demote_partitions(self, indices: "set[int]") -> int:
        """Move active train_cameras whose source index is in ``indices``
        out of train_cameras and into the icecold archive tier (see class
        docstring), freeing active-buffer/sampling attention that a
        partition doing unusually well relative to the rest of the scene
        doesn't need. Never demotes a bootstrap frame or the current
        (most-recently-admitted) camera -- both are load-bearing for
        training stability independent of partition PSNR.

        Archives the camera's lightweight source frame (cam._streaming_
        frame, set by _frame_to_camera) rather than the Camera object
        itself, then drops the Camera reference -- its GPU-resident image
        tensor frees via normal refcounting/the CUDA caching allocator, no
        explicit (synchronous, per-frame) GPU->CPU transfer needed for the
        demoted cameras themselves. A separate _enforce_hot_window() sweep
        still runs once afterward (not per-frame) to restore the OTHER,
        still-active cameras' hot-window invariant -- train_cameras.remove()
        shifts later indices down by one, which would otherwise desync the
        admission path's O(1) eviction counter. Returns the number of
        cameras demoted."""
        demoted = 0
        for idx in list(indices):
            cam = self._active_cam_by_idx.get(idx)
            if cam is None:
                continue
            if getattr(cam, "_is_bootstrap", False) or cam is self.current_camera:
                continue
            frame = getattr(cam, "_streaming_frame", None)
            if frame is None:
                continue  # nothing to archive to -- leave it active rather than lose it
            try:
                self.train_cameras.remove(cam)
            except ValueError:
                continue
            self._active_source_indices.discard(idx)
            del self._active_cam_by_idx[idx]
            del cam  # drop the last local reference; GPU tensor frees via refcounting
            self._archive_to_icecold(idx, frame)
            demoted += 1
        if demoted:
            # list.remove() above shifts every later index down by one,
            # desyncing the normal admission path's position-based eviction
            # counter (see _enforce_hot_window's docstring) -- restore the
            # hot-window invariant from scratch so future admissions' O(1)
            # counter starts from a correct state again.
            self._enforce_hot_window()
        return demoted

    def set_external_seed(self, points: np.ndarray, colors: Optional[np.ndarray]) -> None:
        """Attach an external SfM seed cloud (e.g. RealityScan's sparse point
        cloud, Task 1's ``frame_source.seed_points``/``.seed_colors``) for
        depth-less sources. Consumed at the top of _build_pcd_from_frames,
        taking precedence over the depth-backprojection/random-scatter
        bootstrap. Must be called before initialize_from_frames()."""
        self._external_seed = (points, colors)

    def set_observation_db(self, db) -> None:
        """Attach a FrameObservationDB so tiered-cache transitions
        (_cache_inactive_frame/_archive_to_icecold/promote_from_cache/
        demote_partitions) keep its per-frame buffer_stage in sync with
        which tier the frame is actually in. See the __init__ docstring
        for observation_db."""
        self.observation_db = db

    def _mark_buffer_stage(self, idx: "int | None", stage) -> None:
        """Best-effort FrameObservationDB.set_buffer_stage() call -- no-op
        if no DB is attached or the frame was never recorded there (e.g.
        pregate-rejected frames never reach record_fn, see ingest_next_
        frame's docstring)."""
        if idx is None or self.observation_db is None:
            return
        try:
            self.observation_db.set_buffer_stage(idx, stage)
        except Exception:
            pass

    def set_priority_frame_indices(self, indices: "set[int]") -> None:
        """Update the set of source frame indices to bias training sampling
        toward, all at uniform weight 1.0. Simple back-compat entry point --
        prefer set_priority_frame_weights() for graduated (severity-
        proportional) weighting."""
        self.set_priority_frame_weights({idx: 1.0 for idx in indices})

    def set_priority_frame_weights(self, weights: "dict[int, float]", resolution_cache_fn=None) -> None:
        """Update the per-index sampling weight for partition-focus priority
        cameras (see utils.partition_health.detect_outlier_partitions and
        the periodic call site in train_streaming.py). Replaces the
        previous set/weights wholesale -- callers re-derive them fresh each
        check from the explicit role-safe control signal, so partitions that
        recover are naturally dropped again, not accumulated forever.

        Also promotes any newly-flagged index that is sitting in the
        cold or icecold cache tiers into train_cameras (see
        promote_from_cache) -- without this, biasing sampling toward a
        partition with few/no admitted frames has nothing new to draw on.

        ``resolution_cache_fn`` is forwarded verbatim to
        ``promote_from_cache`` (issue #481(b) -- see its docstring)."""
        self._priority_frame_weights = dict(weights)
        self._priority_frame_indices = set(weights.keys())
        if weights:
            self.promote_from_cache(set(weights.keys()), resolution_cache_fn=resolution_cache_fn)

    def sample_training_camera(self, model=None, render_fn=None, pipe=None, background=None):
        """Sample a camera for one training step.

        ``model`` (the Gaussian model) is only required when
        ``streaming_debt_replay`` is enabled; the debt sampler reads per-Gaussian
        support/visibility to weight replay frames. It is threaded through to the
        legacy/stratified samplers and tolerated as None everywhere else.
        """
        # H8: warmup — force current frame for K steps after each new arrival
        warmup_steps = getattr(self.args, "streaming_new_frame_warmup_steps", 0)
        if warmup_steps > 0 and self._steps_since_new_frame < warmup_steps and self.current_camera is not None:
            self._steps_since_new_frame += 1
            return self.current_camera
        self._steps_since_new_frame += 1

        # P0.4: two-regime sampling — once the stream idles (no admitted frame
        # for streaming_consolidation_idle_steps), the recency-biased ingest
        # sampler keeps hammering the tail window while the rest of the scene
        # starves. Switch to a global sampler over ALL train cameras instead.
        consol_mode = getattr(self.args, "streaming_consolidation_sampler", "off")
        if consol_mode != "off" and self.train_cameras:
            idle_steps = int(getattr(self.args, "streaming_consolidation_idle_steps", 500))
            if self._steps_since_new_frame > idle_steps:
                cam = self._sample_consolidation(consol_mode, model)
                if cam is not None:
                    return cam

        # Partition-focus resampling (opt-in, 0 = off): bias toward train
        # cameras in currently-flagged under-reconstructed partitions (see
        # utils.partition_health.detect_outlier_partitions and this scene's
        # set_priority_frame_indices, refreshed periodically by train_
        # streaming.py's psnr-tick). Distinct from the residual-weighted
        # branch below: that reacts to an admitted camera's OWN render
        # residual, this reacts to a role-separated REGIONAL control signal
        # and applies it to every train camera in that region, which
        # helps when a whole area is genuinely under-covered rather than
        # just individually hard to fit.
        partition_focus_weight = float(getattr(self.args, "streaming_partition_focus_weight", 0.0))
        if partition_focus_weight > 0.0 and self._priority_frame_indices and self.train_cameras:
            import random as _random
            if _random.random() < partition_focus_weight:
                from eligsir.utils.partition_eval import camera_source_index
                from eligsir.runtime.payload_residency import payload_ready_cameras
                priority_cams = []
                priority_weights = []
                for c in payload_ready_cameras(self.train_cameras):
                    src_idx = camera_source_index(c)
                    if src_idx in self._priority_frame_indices:
                        priority_cams.append(c)
                        priority_weights.append(self._priority_frame_weights.get(src_idx, 1.0))
                if priority_cams:
                    return self._weighted_choice_weighted(priority_cams, priority_weights)

        # Residual-weighted active-ingestion resampling (opt-in, 0 = off):
        # bias toward struggling (high-residual) admitted cameras DURING
        # active ingestion, not just once ingestion goes idle -- targets
        # seed-fade, where the origin view is outnumbered by newly-admitted
        # frames long before ingestion ever goes idle.
        residual_weight = float(getattr(self.args, "streaming_residual_replay_weight", 0.0))
        if residual_weight > 0.0 and self.train_cameras and render_fn is not None:
            import random as _random
            if _random.random() < residual_weight:
                interval = int(getattr(self.args, "streaming_residual_replay_interval", 200))
                # Cache validity depends only on draw-age, not on whether new
                # cameras have been admitted since the last recompute -- during
                # active ingestion train_cameras grows every steps_per_frame
                # iterations, so keying invalidation off pool-length equality
                # (as the debt-cache does, safe there since it only runs once
                # ingestion is idle and the pool is stable) would force a full
                # render sweep almost every check. Instead we sample from a
                # frozen SNAPSHOT of the pool taken at compute time
                # (_residual_camera_snapshot), so _residual_weights always
                # stays index-aligned with the list we actually sample from,
                # regardless of how much train_cameras has grown since.
                cache_valid = (
                    getattr(self, "_residual_weights", None) is not None
                    and getattr(self, "_residual_camera_snapshot", None) is not None
                    and getattr(self, "_residual_age", 0) < interval
                )
                if not cache_valid:
                    from eligsir.runtime.payload_residency import payload_ready_cameras
                    snapshot = payload_ready_cameras(self.train_cameras)
                    self._residual_weights = self._compute_residual_weights(
                        snapshot, model, render_fn, pipe, background
                    )
                    self._residual_camera_snapshot = snapshot
                    self._residual_age = 0
                self._residual_age = getattr(self, "_residual_age", 0) + 1
                if self._residual_weights is not None:
                    return self._weighted_choice_weighted(
                        self._residual_camera_snapshot, self._residual_weights
                    )

        sampling_mode = getattr(self.args, "streaming_sampling_mode", "legacy")
        if sampling_mode == "stratified":
            return self._sample_stratified(model)
        return self._sample_legacy(model)

    def _sample_consolidation(self, mode: str, model=None):
        """Global sampler for the idle/consolidation regime (P0.4).

        ``uniform``: uniform over all admitted train cameras.
        ``debt``: debt-weighted over all train cameras (weights cached and
        recomputed every 200 draws — the debt projection is O(N·V)).
        """
        from eligsir.runtime.payload_residency import payload_ready_cameras

        pool = payload_ready_cameras(self.train_cameras)
        if not pool:
            return None
        if mode == "debt" and model is not None:
            cache_valid = (
                getattr(self, "_consol_debt_weights", None) is not None
                and getattr(self, "_consol_debt_pool_len", -1) == len(pool)
                and getattr(self, "_consol_debt_age", 0) < 200
            )
            if not cache_valid:
                self._consol_debt_weights = self._compute_debt_weights(pool, model)
                self._consol_debt_pool_len = len(pool)
                self._consol_debt_age = 0
            self._consol_debt_age += 1
            if self._consol_debt_weights is not None:
                return self._weighted_choice_weighted(pool, self._consol_debt_weights)
        return random.choice(pool)

    @staticmethod
    def _weighted_choice(cameras, keyframe_weight: float):
        """Random choice from cameras, over-sampling bootstrap/keyframe entries."""
        if keyframe_weight <= 1.0 or not cameras:
            return random.choice(cameras)
        weights = [keyframe_weight if getattr(c, "_is_bootstrap", False) else 1.0 for c in cameras]
        total = sum(weights)
        r = random.random() * total
        cumulative = 0.0
        for cam, w in zip(cameras, weights):
            cumulative += w
            if r < cumulative:
                return cam
        return cameras[-1]

    @staticmethod
    def _weighted_choice_weighted(cameras, weights):
        """Random choice from cameras using an explicit per-camera weight array.

        Used by the spatial pose-window path. Falls back to uniform choice if
        weights are degenerate (sum <= 0).
        """
        if not cameras:
            return None
        total = float(sum(weights)) if weights else 0.0
        if total <= 0.0:
            return random.choice(cameras)
        r = random.random() * total
        cumulative = 0.0
        for cam, w in zip(cameras, weights):
            cumulative += float(w)
            if r < cumulative:
                return cam
        return cameras[-1]

    def _sample_local(self, kw: float):
        """Pick one local-window camera, honoring the spatial pose-window if enabled.

        Disabled path is byte-identical to the previous
        ``self._weighted_choice(local, kw) if local else self.current_camera``.
        """
        if getattr(self.args, "streaming_spatial_window_enabled", False):
            cams, weights = self._spatial_local_cameras()
            if weights is not None and cams:
                return self._weighted_choice_weighted(cams, weights)
            # Fallback: spatial precondition not met -> temporal behavior.
            return self._weighted_choice(cams, kw) if cams else self.current_camera
        local = self.get_local_cameras()
        return self._weighted_choice(local, kw) if local else self.current_camera

    def _compute_debt_weights(self, cameras, model) -> "list[float] | None":
        """Compute per-camera training debt weights for debt-based replay.

        For each camera, estimates the total unmet support among Gaussians
        visible in its frustum: sum of max(0, target - support_count_i).
        Returns softmax-normalized weights, or None if lifecycle unavailable.
        """
        import math

        support = getattr(model, "support_count", None)
        if support is None or support.shape[0] == 0:
            if not getattr(self, "_debt_fallback_warned", False):
                print("[streaming] debt_replay: support_count unavailable, falling back to uniform")
                self._debt_fallback_warned = True
            return None

        target = int(getattr(self.args, "streaming_debt_replay_target_support", 10))
        means = model.get_xyz  # (N, 3) world-space Gaussian centres

        import torch
        with torch.no_grad():
            support_f = support.float()
            deficit = (target - support_f).clamp(min=0.0)  # (N,)
            means_h = torch.cat([means, torch.ones(means.shape[0], 1, device=means.device)], dim=1)  # (N,4)

            raw_weights: list[float] = []
            for cam in cameras:
                w2c = cam.world_view_transform.T  # (4,4) col-major → row-major
                pts_cam = (w2c @ means_h.T).T  # (N,4) in camera space
                z = pts_cam[:, 2]
                znear = getattr(cam, "znear", 0.01)
                zfar = getattr(cam, "zfar", 100.0)
                in_depth = (z > znear) & (z < zfar)
                # Project to normalised image coords using FoV
                fx = cam.image_width / (2.0 * math.tan(cam.FoVx / 2.0))
                fy = cam.image_height / (2.0 * math.tan(cam.FoVy / 2.0))
                cx, cy = cam.image_width / 2.0, cam.image_height / 2.0
                u = pts_cam[:, 0] / z * fx + cx
                v = pts_cam[:, 1] / z * fy + cy
                in_image = (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height)
                visible = in_depth & in_image
                n_vis = int(visible.sum())
                # Normalise by visible count so weight = mean deficit per visible
                # Gaussian (≈ constant early in training) rather than raw sum.
                # Raw sum → softmax is degenerate: camera with most Gaussians
                # captures weight ≈ 1, starving all others.
                mean_def = float(deficit[visible].sum()) / max(n_vis, 1)
                raw_weights.append(mean_def)

            # Temperature-scaled softmax (T=4) keeps distribution soft even when
            # deficits differ substantially across cameras.
            w = torch.tensor(raw_weights, dtype=torch.float32)
            w = torch.softmax(w / 4.0, dim=0)
            return w.tolist()

    def _compute_residual_weights(self, cameras, gaussians, render_fn, pipe, background) -> "list[float] | None":
        """Per-camera training weights from render-vs-GT residual: higher
        residual (lower PSNR) -> higher sampling weight. Mirrors
        _compute_debt_weights' temperature-scaled-softmax shape but measures
        actual render quality instead of per-Gaussian support deficit.
        Fail-open: any rendering error returns None so callers fall back to
        existing sampling behavior rather than crashing training.
        """
        if not cameras:
            return None
        self._residual_replay_score_attempts = (
            int(getattr(self, "_residual_replay_score_attempts", 0)) + 1
        )
        import torch
        from eligsir.utils.score_contract import (
            apply_declared_score_clamp_pair,
            score_tensor_from_camera,
            scientific_psnr,
        )

        try:
            psnr_values = []
            clamp_records = []
            with torch.no_grad():
                for cam in cameras:
                    out = render_fn(cam, gaussians, pipe, background)
                    rendered = out["render"] if isinstance(out, dict) else out
                    gt = score_tensor_from_camera(cam, device=rendered.device)
                    rendered_score, gt_score, clamp_record = apply_declared_score_clamp_pair(
                        rendered,
                        gt,
                        context="streaming_scene:residual_replay",
                    )
                    clamp_records.append(clamp_record)
                    p = float(
                        scientific_psnr(
                            rendered_score,
                            gt_score,
                            context="streaming_scene:residual_replay",
                        ).mean()
                    )
                    psnr_values.append(p)
        except Exception as e:
            self._residual_replay_score_fallbacks = (
                int(getattr(self, "_residual_replay_score_fallbacks", 0)) + 1
            )
            self._residual_replay_last_failure = repr(e)
            print(f"[streaming] residual_replay: render failed ({e!r}), falling back to uniform")
            return None

        changed = sum(r["render"]["samples_changed"] for r in clamp_records)
        total = sum(r["render"]["samples_total"] for r in clamp_records)
        self._residual_replay_score_clamp = {
            "nonfinite_policy": "rejected before clamp",
            "samples_changed": changed,
            "samples_total": total,
            "fraction_changed": changed / max(1, total),
            "value_min": min(r["render"]["value_min"] for r in clamp_records),
            "value_max": max(r["render"]["value_max"] for r in clamp_records),
        }
        self._residual_replay_last_failure = None

        max_psnr = max(psnr_values)
        residuals = torch.tensor([max_psnr - p for p in psnr_values], dtype=torch.float32)
        weights = torch.softmax(residuals / 4.0, dim=0)
        return weights.tolist()

    def _sample_global_with_debt(self, pool: list, model, kw: float):
        """Sample from *pool* with debt weights when debt_replay is enabled."""
        if not pool:
            return None
        if not getattr(self.args, "streaming_debt_replay", False) or model is None:
            return self._weighted_choice(pool, kw)
        weights = self._compute_debt_weights(pool, model)
        if weights is None:
            return self._weighted_choice(pool, kw)
        return self._weighted_choice_weighted(pool, weights)

    def _sample_legacy(self, model=None):
        """Original ring-buffer + reservoir sampling."""
        kw = float(getattr(self.args, "streaming_keyframe_sampling_weight", 2.0))
        # No getattr fallback: streaming_global_replay_ratio is an
        # unconditional StreamingParams default, always present on the real
        # parsed args (issue #368 -- old getattr(..., 0.1) fallback drifted
        # from the real default of 0.3; removed rather than re-stated).
        replay_ratio = self.args.streaming_global_replay_ratio
        r = random.random()
        if replay_ratio > 0 and r < replay_ratio:
            reservoir_stride = self.args.streaming_global_reservoir_stride
            from eligsir.runtime.payload_residency import payload_ready_cameras
            reservoir = payload_ready_cameras(self._global_reservoir)
            replay = payload_ready_cameras(self._replay_buffer)
            if reservoir_stride > 0 and reservoir:
                return self._weighted_choice(reservoir, kw)
            if replay:
                cam = self._sample_global_with_debt(replay, getattr(self, "model", None), kw)
                if cam is not None:
                    return cam
            if replay:
                cam = self._sample_global_with_debt(replay, getattr(self, "model", None), kw)
                if cam is not None:
                    return cam
        return self._sample_local(kw)

    def _sample_stratified(self, model=None):
        """Four-strata sampling: recent / covisible / global-reservoir / hard-frames."""
        ratios_str = getattr(self.args, "streaming_sampling_ratios", "0.70,0.15,0.10,0.05")
        try:
            ratios = [float(x) for x in ratios_str.split(",")]
            if len(ratios) != 4:
                ratios = [0.70, 0.15, 0.10, 0.05]
        except Exception:
            ratios = [0.70, 0.15, 0.10, 0.05]
        r_recent, r_covis, r_reservoir, r_hard = ratios

        kw = float(getattr(self.args, "streaming_keyframe_sampling_weight", 2.0))
        r = random.random()
        local = self.get_local_cameras()

        if r < r_recent:
            return self._weighted_choice(local, kw) if local else self.current_camera

        if r < r_recent + r_covis:
            from eligsir.runtime.payload_residency import payload_ready_cameras
            covisible = payload_ready_cameras(self._covisible_cache)
            if covisible:
                return self._weighted_choice(covisible, kw)
            return self._weighted_choice(local, kw) if local else self.current_camera

        if r < r_recent + r_covis + r_reservoir:
            reservoir_stride = self.args.streaming_global_reservoir_stride
            from eligsir.runtime.payload_residency import payload_ready_cameras
            reservoir = payload_ready_cameras(self._global_reservoir)
            replay = payload_ready_cameras(self._replay_buffer)
            if reservoir_stride > 0 and reservoir:
                return self._weighted_choice(reservoir, kw)
            if replay:
                cam = self._sample_global_with_debt(replay, getattr(self, "model", None), kw)
                if cam is not None:
                    return cam
            if replay:
                cam = self._sample_global_with_debt(replay, getattr(self, "model", None), kw)
                if cam is not None:
                    return cam
            return self._sample_local(kw)

        # Hard frames stratum
        from eligsir.runtime.payload_residency import camera_payload_ready
        hard = []
        for item in self._hard_frames:
            candidate = item[1] if isinstance(item, tuple) and len(item) > 1 else item
            if camera_payload_ready(candidate):
                hard.append(item)
        if hard:
            item = random.choice(hard)
            return item[1] if isinstance(item, tuple) and len(item) > 1 else item
        return self._weighted_choice(local, kw) if local else self.current_camera

    def update_stratified_state(self, cam, loss_val: float, gaussian_ids=None):
        """Update hard-frame list and covisibility cache. Call once per training step."""
        if getattr(self.args, "streaming_sampling_mode", "legacy") != "stratified":
            return

        # Hard frames: keep the K highest-loss cameras
        max_hard = max(1, int(getattr(self.args, "streaming_hard_frame_history", 8)))
        self._hard_frames.append((loss_val, cam))
        self._hard_frames.sort(key=lambda x: -x[0])
        self._hard_frames = self._hard_frames[:max_hard]

        # Covisibility: cameras that share many Gaussians with the current view.
        # We build a flat covisible list from the replay buffer cameras closest
        # in index to the current train camera (cheap proxy for covisibility).
        if len(self.train_cameras) > 0 and cam in self.train_cameras:
            try:
                cam_idx = self.train_cameras.index(cam)
            except ValueError:
                cam_idx = len(self.train_cameras) - 1
            k_win = self._effective_keyframe_window()
            # Covisible = cameras within 2×window of current cam, excluding local window
            lo = max(0, cam_idx - 2 * k_win)
            hi = max(0, cam_idx - k_win)
            self._covisible_cache = self.train_cameras[lo:hi] if lo < hi else []

    # ------------------------------------------------------------------
    # Compatibility shims for code that calls scene.getTrainCameras()
    # ------------------------------------------------------------------

    def getTrainCameras(self, scale: float = 1.0) -> List:
        return self.train_cameras

    def getTestCameras(self, scale: float = 1.0) -> List:
        return self._test_cameras

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frame_to_camera(self, frame: "StreamingRGBDFrame", load_image: bool = True):
        """Build a GPU ``Camera`` for ``frame``.

        ``load_image=False`` (D2, Wave 2 consolidation) builds the camera's
        geometry only (R/T/FoV/intrinsics/width/height -- everything
        ``visibility_fn``, ``admission_fn``'s render, and
        ``camera_to_JSON``/dense-trajectory logging need) and defers the
        RGB decode + host->device upload behind ``Camera.
        materialize_pending_image()``. The deferred loader simply re-runs
        this same method with ``load_image=True`` and takes its
        ``original_image`` -- so the eager path below is the single source
        of truth for image decoding; the lazy path never duplicates its
        logic, only defers invoking it. Callers (``ingest_next_frame``)
        only materialize once a candidate frame is actually admitted, so a
        rejected candidate never pays the decode/upload cost.
        """
        from eligsir.utils.camera_utils import loadCam
        from eligsir.utils.graphics_utils import focal2fov
        from eligsir.utils.rgbd_frames import c2w_to_camera_rt
        from scene.readers.common import CameraInfo

        R, T = c2w_to_camera_rt(frame.c2w)
        _target = (int(frame.width), int(frame.height))

        if not load_image:
            orig_w, orig_h = _target
            cam_info = CameraInfo(
                uid=frame.index,
                R=R, T=T,
                FovY=focal2fov(frame.fy, orig_h),
                FovX=focal2fov(frame.fx, orig_w),
                image=None,
                image_path=frame.rgb_path,
                image_name=f"{frame.index:06d}",
                width=orig_w, height=orig_h,
                fx=float(frame.fx), fy=float(frame.fy),
                cx=float(frame.cx), cy=float(frame.cy),
            )

            def _pending_loader():
                real_cam = self._frame_to_camera(frame, load_image=True)
                # real_cam.original_image is already clamped and alpha-
                # multiplied (ones if no mask); returning gt_alpha_mask=None
                # makes materialize_pending_image's own clamp/multiply-by-
                # ones a no-op re-application, not a second real mask.
                return real_cam.original_image, None

            cam = loadCam(
                self.args, frame.index, cam_info, self.resolution_scale,
                rgb_uint8=None, build_image=False,
            )
            # loadCam's own default lazy loader assumes rgb_uint8/cam_info
            # captured at THIS call time; override with the delegator above
            # so a re-decode always sees frame's current state (in case
            # frame._rgb_np becomes available/changes between candidate
            # construction and admission -- unlikely within one ingest
            # call, but avoids relying on that assumption).
            # issue #333: installs the delegator as BOTH the one-shot lazy
            # loader and the retained native re-materializer. A bare
            # `cam._pending_image_loader = ...` would leave the latter on
            # loadCam's own stale-state closure.
            cam.set_image_loader(_pending_loader)
            cam._streaming_depth_path = frame.depth_path
            cam._streaming_depth_scale = frame.depth_scale
            cam._sensor_depth_cache = None
            cam._streaming_timestamp = float(frame.timestamp)
            cam._streaming_frame = frame
            return cam

        from PIL import Image

        from eligsir.utils.streaming_frames import load_frame_rgb
        # Fast path: when the decoded uint8 array is present and already at the
        # target size, derive geometry from its shape and SKIP building a PIL
        # Image. load_frame_rgb's per-frame Image.fromarray was only used to read
        # .size, and loadCam's uint8 path never touches cam_info.image — so the
        # PIL was pure waste on every live frame. loadCam lazily rebuilds a PIL
        # from rgb_uint8 only if an actual downscale forces the slow path, keeping
        # that path byte-identical. Offline / size-mismatch falls back to the PIL.
        _rgb_np = getattr(frame, "_rgb_np", None)
        image = None
        if (
            _rgb_np is not None
            and getattr(_rgb_np, "ndim", 0) == 3
            and (_rgb_np.shape[1], _rgb_np.shape[0]) == _target
        ):
            orig_w, orig_h = _target
        else:
            image = load_frame_rgb(frame)
            if image.size != _target:
                image = image.resize(_target, Image.BILINEAR)
            orig_w, orig_h = image.size
            _rgb_np = frame._rgb_np if (image.size == _target) else None
        cam_info = CameraInfo(
            uid=frame.index,
            R=R, T=T,
            FovY=focal2fov(frame.fy, orig_h),
            FovX=focal2fov(frame.fx, orig_w),
            image=image,
            image_path=frame.rgb_path,
            image_name=f"{frame.index:06d}",
            width=orig_w, height=orig_h,
            fx=float(frame.fx), fy=float(frame.fy),
            cx=float(frame.cx), cy=float(frame.cy),
        )
        # _rgb_np set above (fast path: the decoded uint8 array; slow path: None).
        # loadCam uses it for the GPU-resident upload and skips the PIL round-trip.
        cam = loadCam(self.args, frame.index, cam_info, self.resolution_scale, rgb_uint8=_rgb_np)
        # Attach depth source for streaming depth loss (loaded lazily at training time)
        cam._streaming_depth_path = frame.depth_path
        cam._streaming_depth_scale = frame.depth_scale
        cam._sensor_depth_cache = None  # populated on first access
        cam._streaming_timestamp = float(frame.timestamp)
        # Keep reference to frame for in-memory depth access (.sens source)
        cam._streaming_frame = frame
        return cam
