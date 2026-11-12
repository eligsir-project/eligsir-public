"""Per-subsystem GPU memory residency telemetry for streaming runs (issue #399).

Why this exists
---------------
Issue #202's P0-C lane ("memory/residency correctness for long trajectories")
is rooted at #399: three separate residency hypotheses -- #396 (replay cameras
never paged back out after a historical replay re-uploads their RGB), #398
(holdout/test images left GPU-resident during mapping), #333 (the multires
resolution cache holding a SECOND copy of native-resolution RGB alongside a
camera's own ``original_image``) -- were all argued from arithmetic
("993 test frames x 7.9 MiB ~= 7.7 GiB") rather than from measurement. Nothing
in the codebase could answer "who actually owns the 28 GiB?".

This module answers exactly that question and nothing else. **It is telemetry
only**: it never moves, frees, or offloads a tensor, and it changes no training
math. Fixing whatever it finds is #396/#398/#333's job.

Attribution scheme
------------------
Attribution is *real*, not modelled: every snapshot walks the live objects
(Gaussian model, its optimizer state, the camera lists owned by
``StreamingScene``, the ``FrameObservationDB``'s per-frame resolution cache)
and sums the **untyped storage** bytes of every tensor it finds on the target
device.

Two rules make the numbers reproducible:

1. **Dedup by storage, not by tensor.** ``tensor.data_ptr()`` on a view or a
   slice is an offset into a shared base storage, so keying on it both
   double-counts (two views of one storage) and under-sizes (a view's
   ``numel * element_size`` is not the storage's footprint). Every tensor is
   keyed on ``(device_index, untyped_storage().data_ptr())`` and sized with
   ``untyped_storage().nbytes()``.
2. **Fixed bucket priority; first claim wins.** The walk visits buckets in the
   order given by :data:`BUCKET_ORDER`, and a storage already claimed by an
   earlier bucket is skipped by every later one. So e.g. a parameter that the
   optimizer also references is billed to ``gaussian_params``, never twice.

Whatever the allocator reports as live but the walk did not reach is reported
as an explicit ``unattributed`` bucket -- never silently folded into a named
one. A small positive ``unattributed`` is *expected and correct*: the caching
allocator rounds block sizes up, autograd holds transient saved tensors, and
the render backend has its own workspaces. ``reserved_overhead``
(``memory_reserved() - memory_allocated()``) is reported as its own field and
is deliberately NOT part of ``unattributed`` -- it is allocator cache, not a
residency owner.

Discriminators
--------------
Bucket bytes alone do not discriminate the three hypotheses, so each snapshot
also carries the specific counters that do:

* ``cold_resident_count`` / ``cold_resident_bytes`` (aliased as
  ``resident_beyond_offload_bound_*``) (#396) -- train cameras *past the
  boundary the offload policy actually evicts at* whose ``original_image`` is
  still on the device. That boundary is the RAW ``--streaming_keyframe_window``
  (default 120), **not** ``_effective_keyframe_window()`` (which is the much
  larger, coverage-adaptive *sampling* window -- ~4800 on an 8000-frame
  ScanNet++ run at the default ``streaming_keyframe_coverage=0.6``). See
  ``_collect_cameras``' docstring: keying on the sampling window would classify
  a leaked, replay-paged-in camera at position -3000 as "hot" and never count
  it. Both numbers are reported (``offload_bound`` / ``sampling_window``).
  A steady value ~= the optimization batch size is healthy (#396 explicitly
  allows the cameras in the current batch to be resident); a value that GROWS
  with ``frames_ingested`` is the leak signature.
* ``holdout_resident_count`` / ``holdout_resident_bytes`` (#398) -- the same
  measurement over ``_test_cameras``. #398's acceptance criterion is literally
  "zero or explicitly bounded", so the count is emitted directly.
* ``multires_cached_frames`` / ``multires_double_represented`` (#333) -- how
  many frames hold a resolution cache (against
  ``--streaming_resolution_cache_max_frames``), and how many cameras hold BOTH
  a device-resident ``original_image`` AND a ``high``-tier cache handle. That
  second number is the "two copies of native-resolution RGB competing" #333
  names, and nothing else measures it.

Reading the numbers honestly
----------------------------
* **The snapshot is taken at a fixed point in the training loop (during loss
  construction, pre-backward), not at the allocator's high-water mark.** So
  ``peak_bucket_bytes`` / ``peak_allocated_bytes`` are peaks *over the sampled
  points*, and are legitimately lower than the true peak, which occurs inside
  backward / rasterization. ``torch_peak_allocated_bytes``
  (``torch.cuda.max_memory_allocated()``) is the honest whole-run peak; do not
  read the gap between the two as broken attribution.
* **``gaussian_grads`` reads ~0 on most runs.** ``zero_grad(set_to_none=True)``
  frees ``.grad`` between steps, and the snapshot point is before backward.
  That is not "no gradient memory"; it is where in the step we looked.

Cost / zero-overhead-when-off
-----------------------------
The walk touches **metadata only** -- ``.device``, ``.numel()``,
``.element_size()``, ``untyped_storage().data_ptr()/.nbytes()``. It runs no
tensor op, calls no ``.item()``/``.sum()``, and never calls
``torch.cuda.synchronize()`` or ``torch.cuda.empty_cache()``.
``torch.cuda.memory_allocated()``/``memory_reserved()``/
``max_memory_allocated()`` read the caching allocator's own bookkeeping and do
not synchronize either, so no host/device sync is introduced (cf. #424, and
the standing #480 invariant that observability must not backpressure
training). When telemetry is disabled the tracker is never constructed and the
cost is a single ``is None`` test per logging interval.

CPU-safety: this module is import-safe and callable on a CPU-only host (CI
collects tests without CUDA, #478). With no CUDA the tracker reports zeros
plus a ``reason``; pass ``device="cpu"`` to exercise the full attribution walk
against CPU tensors, which is what ``tests/test_gpu_memory_telemetry.py``
does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

# Bucket priority order. A storage claimed by an earlier bucket is never
# re-claimed by a later one, so this ordering fully determines attribution
# whenever two subsystems reference the same storage. Do not reorder without
# noting it in report.json consumers -- historical runs' numbers are only
# comparable under an identical order.
BUCKET_ORDER: Tuple[str, ...] = (
    "gaussian_params",
    "gaussian_grads",
    "gaussian_optimizer",
    "gaussian_buffers",
    "camera_hot_images",
    "camera_cold_images",
    "camera_holdout_images",
    "camera_depth_cache",
    "camera_gsplat",
    "multires_cache",
)

# Stable attribution contract consumed by #399 readouts.  The tracker measures
# ownership; it never changes residency.  Keep this explicit so a bucket's
# bytes cannot be mistaken for a policy decision or an unverified estimate.
RESIDENCY_OWNER_CONTRACT_VERSION = 1
_RESIDENCY_OWNERS = {
    "gaussian_params": ("GaussianMapBackend", "gaussians.params", "live tensor storage bytes"),
    "gaussian_grads": ("GaussianMapBackend", "parameter.grad", "live tensor storage bytes"),
    "gaussian_optimizer": ("GaussianMapBackend", "gaussians.optimizer.state", "live tensor storage bytes"),
    "gaussian_buffers": ("GaussianMapBackend", "gaussian model buffers", "live tensor storage bytes"),
    "camera_hot_images": ("StreamingScene", "train_cameras[-streaming_keyframe_window:]", "resident original_image storage bytes"),
    "camera_cold_images": ("StreamingScene", "train_cameras outside offload bound and replay pools", "resident original_image storage bytes"),
    "camera_holdout_images": ("StreamingScene", "_test_cameras", "resident original_image storage bytes"),
    "camera_depth_cache": ("Camera", "camera._sensor_depth_cache", "resident depth-cache storage bytes"),
    "camera_gsplat": ("Camera", "camera.gsplat tensors", "resident tensor storage bytes"),
    "multires_cache": ("FrameObservationDB", "frame resolution_cache handles", "resident cache tensor storage bytes"),
}


def residency_owner_contract() -> Dict[str, Any]:
    """Return the machine-readable owner/provenance contract for attribution."""
    return {
        "version": RESIDENCY_OWNER_CONTRACT_VERSION,
        "semantics": "telemetry_only_live_storage_first_claim_wins",
        "buckets": {
            name: {"owner": owner, "source": source, "measurement": measurement}
            for name, (owner, source, measurement) in _RESIDENCY_OWNERS.items()
        },
    }

# Legacy (`--model_layout legacy`) GaussianModel parameter attribute names.
_LEGACY_PARAM_ATTRS = (
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_opacity",
    "_scaling",
    "_rotation",
)

_MB = 1024.0 * 1024.0

# AMD DRM's GTT counters describe the whole device's shared system-memory
# aperture.  They are intentionally kept separate from allocator accounting:
# they cannot attribute memory to this process or to any residency bucket.
GTT_MEMORY_SEMANTICS = (
    "device_global_amd_drm_gtt_shared_system_memory_not_per_process"
)


def read_amd_drm_gtt_bytes() -> Optional[Tuple[int, int]]:
    """Return ``(used, total)`` from one readable AMD DRM GTT sysfs pair.

    Containers commonly cannot see these host files, and a system can expose
    DRM devices that have no GTT counters.  Both cases are unavailable data,
    not zero use.  Device selection is deliberately injectable at the tracker
    boundary so callers/tests with a known DRM node can provide an exact
    reader without turning this process-wide measurement into attribution.
    """
    for used_path in Path("/sys/class/drm").glob("card*/device/mem_info_gtt_used"):
        total_path = used_path.with_name("mem_info_gtt_total")
        try:
            used = int(used_path.read_text().strip())
            total = int(total_path.read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 <= used <= total:
            return used, total
    return None


class _StorageClaimer:
    """Accumulates per-bucket storage bytes with first-claim-wins dedup."""

    def __init__(self, device_type: str) -> None:
        self._device_type = device_type
        self._seen: set = set()
        self.bytes: Dict[str, int] = {b: 0 for b in BUCKET_ORDER}

    def claim(self, obj: Any, bucket: str) -> int:
        """Bill ``obj``'s storage to ``bucket`` if it is an unclaimed
        device-resident tensor. Returns the bytes actually billed (0 if the
        object is not a tensor, is on another device, or was already claimed
        by an earlier bucket)."""
        if not isinstance(obj, torch.Tensor):
            return 0
        if obj.device.type != self._device_type:
            return 0
        try:
            storage = obj.untyped_storage()
            key = (obj.device.index, storage.data_ptr())
            nbytes = int(storage.nbytes())
        except Exception:
            # Sparse/meta/exotic tensors have no single untyped storage. Fall
            # back to the logical size and key on the tensor's own data_ptr;
            # deliberately approximate rather than dropped, and rare enough
            # that it does not move the totals.
            try:
                key = (obj.device.index, obj.data_ptr())
                nbytes = int(obj.numel()) * int(obj.element_size())
            except Exception:
                return 0
        if nbytes <= 0 or key in self._seen:
            return 0
        self._seen.add(key)
        self.bytes[bucket] += nbytes
        return nbytes

    def claim_many(self, objs: Iterable[Any], bucket: str) -> int:
        return sum(self.claim(o, bucket) for o in objs)


def _is_device_resident(t: Any, device_type: str) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == device_type


def _walk_gaussians(claimer: _StorageClaimer, gaussians: Any) -> None:
    if gaussians is None:
        return
    params = getattr(gaussians, "params", None)
    param_tensors: List[Any] = []
    if params is not None:
        try:
            param_tensors = list(params.values())
        except Exception:
            param_tensors = []
    if not param_tensors:
        param_tensors = [getattr(gaussians, a, None) for a in _LEGACY_PARAM_ATTRS]
    claimer.claim_many(param_tensors, "gaussian_params")
    claimer.claim_many(
        (getattr(p, "grad", None) for p in param_tensors if isinstance(p, torch.Tensor)),
        "gaussian_grads",
    )

    optimizer = getattr(gaussians, "optimizer", None)
    state = getattr(optimizer, "state", None)
    if isinstance(state, dict):
        for per_param in state.values():
            if isinstance(per_param, dict):
                claimer.claim_many(per_param.values(), "gaussian_optimizer")

    # Everything else the model holds directly: lineage/streaming buffers
    # (birth_frame, gaussian_uid, anchor_xyz, ...), MCMC accumulators, etc.
    # Claimed last of the model buckets so a buffer that aliases a parameter
    # is billed to gaussian_params.
    try:
        attrs = list(vars(gaussians).values())
    except TypeError:
        attrs = []
    claimer.claim_many(attrs, "gaussian_buffers")


def _collect_cameras(
    streaming_scene: Any,
) -> Tuple[List[Any], List[Any], List[Any], int, int]:
    """Return (hot, cold, holdout, offload_bound, effective_sampling_window).

    **The hot/cold split is keyed on the OFFLOAD boundary, not the sampling
    window.** Two different "keyframe windows" exist and conflating them
    blinds the #396 discriminator on exactly the runs it was built for:

    * ``StreamingScene._effective_keyframe_window()`` is the *sampling*
      window, ``max(streaming_keyframe_window, streaming_keyframe_coverage *
      n_total)``. With the real default ``streaming_keyframe_coverage=0.6``
      this is ~4800 on an 8000-frame ScanNet++ trajectory.
    * The *offload/residency* boundary that
      ``StreamingScene.ingest_next_frame`` and
      ``enforce_hot_window_invariant`` actually evict against is the RAW
      ``args.streaming_keyframe_window`` flag (default 120):
      ``evict_at = len(train_cameras) - hot_window - 1``.

    A camera at position -3000 that a global replay paged back onto the GPU
    and never paged out is resident and *outside the offload bound* -- the
    exact #396 leak -- but would be classified "hot" under the sampling
    window and never counted. So ``hot`` is the tail inside the offload
    bound; everything else resident is the leak signal. The effective
    sampling window is still returned and reported, as context for sampling
    pressure, but it is not the residency invariant.
    """
    if streaming_scene is None:
        return [], [], [], 0, 0

    train_cams = list(getattr(streaming_scene, "train_cameras", None) or [])
    args = getattr(streaming_scene, "args", None)
    try:
        offload_bound = max(1, int(args.streaming_keyframe_window))
    except Exception:
        offload_bound = 0
    try:
        sampling_window = max(0, int(streaming_scene._effective_keyframe_window()))
    except Exception:
        sampling_window = offload_bound

    window = offload_bound
    hot = train_cams[-window:] if window else []
    hot_ids = {id(c) for c in hot}

    cold: List[Any] = []
    cold_ids: set = set()
    pools: List[Any] = [train_cams]
    for attr in ("_replay_buffer", "_global_reservoir"):
        pool = getattr(streaming_scene, attr, None)
        if pool:
            pools.append(list(pool))
    for pool in pools:
        for cam in pool:
            if id(cam) in hot_ids or id(cam) in cold_ids:
                continue
            cold_ids.add(id(cam))
            cold.append(cam)

    holdout = [
        c
        for c in (getattr(streaming_scene, "_test_cameras", None) or [])
        if id(c) not in hot_ids and id(c) not in cold_ids
    ]
    return hot, cold, holdout, offload_bound, sampling_window


def _walk_cameras(
    claimer: _StorageClaimer,
    device_type: str,
    hot: List[Any],
    cold: List[Any],
    holdout: List[Any],
) -> Dict[str, int]:
    # One pass per role (not one pass for images then another for depth/gsplat):
    # the camera walk is the only O(n_cameras) part of a snapshot, so halving
    # the number of Python loops over ~9000 cameras is the cheapest available
    # saving. Bucket priority is unaffected in practice -- a camera's
    # original_image / _sensor_depth_cache / gsplat tensors are always distinct
    # storages, so interleaving their claims cannot change any total.
    counts: Dict[str, int] = {}
    for name, cams in (
        ("camera_hot_images", hot),
        ("camera_cold_images", cold),
        ("camera_holdout_images", holdout),
    ):
        resident = 0
        billed = 0
        for cam in cams:
            img = getattr(cam, "original_image", None)
            if _is_device_resident(img, device_type):
                resident += 1
            billed += claimer.claim(img, name)
            claimer.claim(getattr(cam, "_sensor_depth_cache", None), "camera_depth_cache")
            gs = getattr(cam, "gsplat", None)
            if gs is not None:
                claimer.claim(getattr(gs, "viewmat", None), "camera_gsplat")
                claimer.claim(getattr(gs, "K", None), "camera_gsplat")
                claimer.claim(getattr(gs, "camera_center", None), "camera_gsplat")
        counts[name + "_resident"] = resident
        counts[name + "_bytes_billed"] = billed
    return counts


def _walk_multires(
    claimer: _StorageClaimer,
    device_type: str,
    observation_db: Any,
    cameras: List[Any],
) -> Dict[str, int]:
    out = {"multires_cached_frames": 0, "multires_double_represented": 0}
    frames = getattr(observation_db, "_frames", None)
    if not isinstance(frames, dict):
        return out
    high_cached_ids: set = set()
    for frame_id, record in frames.items():
        cache = getattr(record, "resolution_cache", None)
        if not cache:
            continue
        out["multires_cached_frames"] += 1
        for tier, handle in cache.items():
            claimer.claim(getattr(handle, "rgb", None), "multires_cache")
            claimer.claim(getattr(handle, "depth", None), "multires_cache")
            if tier == "high":
                high_cached_ids.add(frame_id)

    if high_cached_ids:
        for cam in cameras:
            frame = getattr(cam, "_streaming_frame", None)
            idx = getattr(frame, "index", None)
            if idx in high_cached_ids and _is_device_resident(
                getattr(cam, "original_image", None), device_type
            ):
                out["multires_double_represented"] += 1
    return out


class GPUMemoryResidencyTracker:
    """Walks live tensors and attributes device memory to named subsystems.

    Parameters
    ----------
    device:
        Device whose residency is measured. ``None`` resolves to ``"cuda"``
        when available, else ``"cpu"``.
    total_bytes_fn / reserved_bytes_fn / peak_bytes_fn:
        Injectable allocator readers (default: the ``torch.cuda`` ones). Tests
        pass their own so the full attribution path -- including the
        ``unattributed`` residual -- is exercised without a GPU.
    max_history:
        Hard cap on retained samples. Once exceeded the series is decimated
        in place (every other sample dropped, stride doubled) so the telemetry
        itself stays bounded as frame history grows -- it must not reproduce
        the unbounded-growth bug class it exists to measure.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        total_bytes_fn: Optional[Callable[[], int]] = None,
        reserved_bytes_fn: Optional[Callable[[], int]] = None,
        peak_bytes_fn: Optional[Callable[[], int]] = None,
        gtt_bytes_fn: Optional[Callable[[], Optional[Tuple[int, int]]]] = None,
        max_history: int = 512,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_type = torch.device(device).type
        self._cuda = self.device_type == "cuda"
        self._total_fn = total_bytes_fn or (
            torch.cuda.memory_allocated if self._cuda else (lambda: 0)
        )
        self._reserved_fn = reserved_bytes_fn or (
            torch.cuda.memory_reserved if self._cuda else (lambda: 0)
        )
        self._peak_fn = peak_bytes_fn or (
            torch.cuda.max_memory_allocated if self._cuda else (lambda: 0)
        )
        self._gtt_bytes_fn = gtt_bytes_fn or read_amd_drm_gtt_bytes
        self.max_history = max(4, int(max_history))
        self._history: List[Dict[str, Any]] = []
        self._stride = 1
        self._peak_buckets: Dict[str, int] = {b: 0 for b in BUCKET_ORDER}
        self._peak_buckets["unattributed"] = 0
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._peak_gtt_used: Optional[int] = None
        self._n_snapshots = 0
        self._last: Optional[Dict[str, Any]] = None

    # -- measurement --------------------------------------------------
    def snapshot(
        self,
        *,
        iteration: int = 0,
        frames_ingested: int = 0,
        gaussians: Any = None,
        streaming_scene: Any = None,
        observation_db: Any = None,
    ) -> Dict[str, Any]:
        """Measure once. Pure metadata walk -- no tensor ops, no host sync."""
        claimer = _StorageClaimer(self.device_type)
        _walk_gaussians(claimer, gaussians)
        hot, cold, holdout, offload_bound, sampling_window = _collect_cameras(
            streaming_scene
        )
        cam_counts = _walk_cameras(claimer, self.device_type, hot, cold, holdout)
        if observation_db is None and streaming_scene is not None:
            observation_db = getattr(streaming_scene, "observation_db", None)
        multires = _walk_multires(
            claimer, self.device_type, observation_db, hot + cold + holdout
        )

        buckets = dict(claimer.bytes)
        attributed = sum(buckets.values())
        allocated = int(self._total_fn())
        reserved = int(self._reserved_fn())
        try:
            gtt = self._gtt_bytes_fn()
        except Exception:
            gtt = None
        try:
            if (
                not isinstance(gtt, tuple)
                or len(gtt) != 2
                or any(isinstance(value, bool) for value in gtt)
            ):
                gtt = None
            elif int(gtt[0]) < 0 or int(gtt[1]) < int(gtt[0]):
                gtt = None
            gtt_used = None if gtt is None else int(gtt[0])
            gtt_total = None if gtt is None else int(gtt[1])
        except (TypeError, ValueError):
            gtt_used = gtt_total = None
            gtt = None
        buckets["unattributed"] = max(0, allocated - attributed)

        sample: Dict[str, Any] = {
            "iteration": int(iteration),
            "frames_ingested": int(frames_ingested),
            "allocated_bytes": allocated,
            "reserved_bytes": reserved,
            "reserved_overhead_bytes": max(0, reserved - allocated),
            "gtt_used_bytes": gtt_used,
            "gtt_total_bytes": gtt_total,
            "gtt_available": gtt is not None,
            "gtt_memory_semantics": GTT_MEMORY_SEMANTICS,
            "attributed_bytes": attributed,
            # Signed residual. `buckets["unattributed"]` is clamped at 0 so it
            # reads as a byte count; this field is NOT, because a NEGATIVE
            # residual means the walk over-attributed (a storage counted twice,
            # or a tensor the allocator does not account for) -- i.e. the
            # diagnostic itself is wrong, exactly the moment the clamp would
            # hide it.
            "residual_bytes_signed": allocated - attributed,
            "buckets": buckets,
            # #396 discriminator. `offload_bound` is the RAW
            # --streaming_keyframe_window, which is what
            # StreamingScene.ingest_next_frame / enforce_hot_window_invariant
            # actually evict against; `sampling_window` is the (much larger,
            # coverage-adaptive) _effective_keyframe_window() and is context
            # only -- see _collect_cameras' docstring for why conflating them
            # blinds this counter.
            #
            # EXPECTED FLOOR: cold_resident_count is not expected to be exactly
            # 0 even under a correct policy. The camera(s) participating in the
            # current optimizer step are legitimately paged in when the
            # snapshot is taken, and #396's own acceptance criterion allows
            # "the HOT set plus the cameras actively participating in the
            # current optimization batch". A steady value ~= batch size is
            # healthy; a value that GROWS with frames_ingested is the leak.
            "offload_bound": offload_bound,
            "sampling_window": sampling_window,
            "n_hot_cameras": len(hot),
            "n_cold_cameras": len(cold),
            "n_holdout_cameras": len(holdout),
            "hot_resident_count": cam_counts["camera_hot_images_resident"],
            "cold_resident_count": cam_counts["camera_cold_images_resident"],
            "cold_resident_bytes": buckets["camera_cold_images"],
            # Explicit, unambiguous aliases: "resident despite being past the
            # boundary the offload policy evicts at" is the #396 signature.
            "resident_beyond_offload_bound_count": cam_counts[
                "camera_cold_images_resident"
            ],
            "resident_beyond_offload_bound_bytes": buckets["camera_cold_images"],
            # #398
            "holdout_resident_count": cam_counts["camera_holdout_images_resident"],
            "holdout_resident_bytes": buckets["camera_holdout_images"],
            # #333
            "multires_cached_frames": multires["multires_cached_frames"],
            "multires_double_represented": multires["multires_double_represented"],
            "multires_bytes": buckets["multires_cache"],
        }

        self._n_snapshots += 1
        self._last = sample
        self._peak_allocated = max(self._peak_allocated, allocated)
        self._peak_reserved = max(self._peak_reserved, reserved)
        if gtt_used is not None:
            self._peak_gtt_used = max(self._peak_gtt_used or 0, gtt_used)
        for k, v in buckets.items():
            self._peak_buckets[k] = max(self._peak_buckets.get(k, 0), v)
        self._append_history(sample)
        return sample

    def phase_snapshot(
        self,
        phase: str,
        *,
        iteration: int = 0,
        frames_ingested: int = 0,
        gaussians: Any = None,
        streaming_scene: Any = None,
        observation_db: Any = None,
    ) -> Dict[str, Any]:
        """Record one labelled current allocator sample without reading a peak.

        A phase sample is deliberately not a phase peak: it is the allocator's
        bookkeeping at one existing report/refinement boundary.  The tracker
        retains its own sampled peak while real global CUDA peaks remain the
        ordinary ``report_block`` concern; neither is reset here.
        """
        if not phase:
            raise ValueError("phase must be non-empty")
        sample = self.snapshot(
            iteration=iteration,
            frames_ingested=frames_ingested,
            gaussians=gaussians,
            streaming_scene=streaming_scene,
            observation_db=observation_db,
        )
        return {
            "phase": str(phase),
            "status": "measured",
            "current_allocated_bytes": sample["allocated_bytes"],
            "current_reserved_bytes": sample["reserved_bytes"],
            "current_reserved_overhead_bytes": sample["reserved_overhead_bytes"],
            "sampled_peak_allocated_bytes": self._peak_allocated,
            "sampled_peak_reserved_bytes": self._peak_reserved,
            "torch_global_peak_allocated_bytes": None,
            "torch_global_peak_reason": "not_read_by_phase_sample",
        }

    def _append_history(self, sample: Dict[str, Any]) -> None:
        if self._n_snapshots % self._stride != 1 and self._stride > 1:
            return
        self._history.append(sample)
        if len(self._history) > self.max_history:
            # Decimate in place: keep every other retained sample, double the
            # stride. Bounded memory, full-run coverage, monotonic iteration
            # order preserved.
            self._history = self._history[::2]
            self._stride *= 2

    # -- reporting ----------------------------------------------------
    def tb_scalars(self, sample: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Flat ``{tag: value}`` for TensorBoard. MiB for byte quantities so
        the curves are readable next to each other."""
        s = sample if sample is not None else self._last
        if s is None:
            return {}
        out: Dict[str, float] = {}
        for name, val in s["buckets"].items():
            out[f"memory/bucket_mib/{name}"] = val / _MB
        out["memory/allocated_mib"] = s["allocated_bytes"] / _MB
        out["memory/reserved_mib"] = s["reserved_bytes"] / _MB
        out["memory/reserved_overhead_mib"] = s["reserved_overhead_bytes"] / _MB
        if s["gtt_used_bytes"] is not None:
            out["memory/gtt_used_mib"] = s["gtt_used_bytes"] / _MB
        if s["gtt_total_bytes"] is not None:
            out["memory/gtt_total_mib"] = s["gtt_total_bytes"] / _MB
        out["memory/attributed_mib"] = s["attributed_bytes"] / _MB
        out["memory/attributed_frac"] = (
            s["attributed_bytes"] / s["allocated_bytes"] if s["allocated_bytes"] else 0.0
        )
        out["memory/residual_signed_mib"] = s["residual_bytes_signed"] / _MB
        for k in (
            "hot_resident_count",
            "cold_resident_count",
            "resident_beyond_offload_bound_count",
            "holdout_resident_count",
            "multires_cached_frames",
            "multires_double_represented",
            "n_cold_cameras",
            "n_holdout_cameras",
            "offload_bound",
            "sampling_window",
        ):
            out[f"memory/residency/{k}"] = float(s[k])
        return out

    def growth_stats(self) -> Dict[str, Any]:
        """Growth of allocated/per-bucket bytes against frames ingested.

        Least-squares slope over the retained series, in bytes per ingested
        frame. This is the number that discriminates "bounded residency" from
        "residency proportional to trajectory length".
        """
        pts = [
            (float(s["frames_ingested"]), s)
            for s in self._history
            if s["frames_ingested"] > 0
        ]
        if len(pts) < 2:
            return {"available": False, "reason": "fewer than 2 samples with frames_ingested>0"}
        xs = [p[0] for p in pts]
        n = float(len(xs))
        mx = sum(xs) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom <= 0:
            return {"available": False, "reason": "no variation in frames_ingested"}

        def slope(vals: List[float]) -> float:
            mv = sum(vals) / n
            return sum((x - mx) * (v - mv) for x, v in zip(xs, vals)) / denom

        out: Dict[str, Any] = {
            "available": True,
            "n_points": int(n),
            "frames_ingested_first": int(xs[0]),
            "frames_ingested_last": int(xs[-1]),
            "allocated_bytes_per_frame": slope([p[1]["allocated_bytes"] for p in pts]),
            "per_bucket_bytes_per_frame": {
                b: slope([float(p[1]["buckets"].get(b, 0)) for p in pts])
                for b in list(BUCKET_ORDER) + ["unattributed"]
            },
            "cold_resident_count_per_frame": slope(
                [float(p[1]["cold_resident_count"]) for p in pts]
            ),
            "holdout_resident_count_per_frame": slope(
                [float(p[1]["holdout_resident_count"]) for p in pts]
            ),
        }
        return out

    def report_block(self) -> Dict[str, Any]:
        """``report.json["gpu_memory"]`` payload (issue #399).

        Always emitted once telemetry is enabled -- a run with zero snapshots
        records ``n_snapshots: 0`` rather than nothing at all, since "the
        walk never ran" and "the walk found nothing" must stay
        distinguishable (the lesson of the ``seed_fade`` ``NO_SAMPLES``
        audit, #282).

        Issue #879: with zero snapshots taken, every attribution field below
        (``peak_allocated_bytes``, ``peak_reserved_bytes``,
        ``peak_bucket_bytes.*``) is still emitted at its ``__init__``-time
        default of ``0`` -- indistinguishable, key-for-key, from a genuine
        measurement of zero (this is exactly the #879 symptom: the engine
        arm silently never called ``snapshot()``, yet ``report.json`` looked
        like a fully-populated, all-zero measurement). Following this
        module's own ``growth_stats()`` precedent (``available``/``reason``)
        and the ``seed_cohort``/``measurement_incomplete`` precedent used
        elsewhere in the reporting layer, the block now also carries a
        top-level ``"available"`` boolean (``n_snapshots > 0``) plus a
        verbatim ``"reason"`` string whenever it is ``False`` -- a consumer
        must check ``available`` before trusting any attribution field, the
        same way it must check ``seed_cohort["available"]`` before trusting
        a cohort statistic. (``"reason"`` pre-dates #879 as a narrower key
        describing only ``final is None``; since ``self._last is None`` iff
        ``self._n_snapshots == 0``, the two conditions coincide exactly, so
        #879 reuses the same key with a fuller message instead of adding a
        second, redundant one.) All pre-existing keys are unchanged in
        value for runs that DID sample."""
        block: Dict[str, Any] = {
            "enabled": True,
            "available": self._n_snapshots > 0,
            "device": self.device_type,
            "n_snapshots": self._n_snapshots,
            "bucket_order": list(BUCKET_ORDER),
            "residency_owner_contract": residency_owner_contract(),
            "peak_allocated_bytes": self._peak_allocated,
            "peak_reserved_bytes": self._peak_reserved,
            "peak_gtt_used_bytes": self._peak_gtt_used,
            "gtt_memory_semantics": GTT_MEMORY_SEMANTICS,
            "peak_bucket_bytes": dict(self._peak_buckets),
            "history_len": len(self._history),
            "history_stride": self._stride,
            "history_max": self.max_history,
        }
        if self._n_snapshots == 0:
            block["reason"] = (
                "no snapshot taken -- attribution fields (peak_allocated_bytes, "
                "peak_reserved_bytes, peak_bucket_bytes.*) are unmeasured "
                "__init__ defaults, not a real measurement of zero"
            )
        if self._cuda:
            try:
                block["torch_peak_allocated_bytes"] = int(self._peak_fn())
            except Exception:
                pass
        if self._last is not None:
            block["final"] = dict(self._last)
        else:
            block["final"] = None
            # `self._last is None` iff `self._n_snapshots == 0` (both are
            # only ever set together, inside snapshot()), so `block["reason"]`
            # was already set above -- do not overwrite it with a narrower
            # string here.
        block["growth"] = self.growth_stats()
        block["series"] = [
            {
                "iteration": s["iteration"],
                "frames_ingested": s["frames_ingested"],
                "allocated_bytes": s["allocated_bytes"],
                "gtt_used_bytes": s["gtt_used_bytes"],
                "gtt_total_bytes": s["gtt_total_bytes"],
                "cold_resident_count": s["cold_resident_count"],
                "holdout_resident_count": s["holdout_resident_count"],
                "multires_cached_frames": s["multires_cached_frames"],
                "buckets": s["buckets"],
            }
            for s in self._history
        ]
        return block


def disabled_report_block(reason: str = "telemetry disabled") -> Dict[str, Any]:
    """Block written when the tracker was never constructed.

    Distinguishes "off" from "on but never sampled" from "absent key = report
    predates #399", mirroring ``utils/duty_cycle.disabled_report_block()`` and
    the #282 ``NO_SAMPLES`` lesson: an absent key means the report predates the
    feature; ``enabled: false`` means telemetry was off; ``n_snapshots: 0`` means
    it was on but collected no samples.

    All structural keys from the enabled arm are present in unavailable form,
    so a consumer can read the same paths on both arms and branch on the
    *value*, not on key presence. This prevents latent KeyErrors when accessing
    ``block["bucket_order"]``, ``block["final"]``, ``block["growth"]``,
    ``block["series"]`` etc.

    Key assignment logic:
    - Structural/schema keys (``bucket_order``, ``peak_bucket_bytes``) carry
      their real value: schema is static and correct, buckets is genuinely
      empty when no snapshot was taken.
    - Count-like keys (``history_len``, ``n_snapshots``, ``series``) carry
      truthful zero/empty values: no history, no samples, no data points.
    - Measurement keys (``peak_allocated_bytes``, ``peak_reserved_bytes``,
      ``peak_gtt_used_bytes``, ``history_stride``, ``history_max``) carry ``None`` because zero is a
      valid measurement and would silently misrepresent an unmeasured value as
      a real zero peak — indistinguishable to a downstream consumer without
      checking ``enabled`` first, which is the anti-pattern this structure
      exists to prevent.
    - ``available``: carries ``False`` because telemetry never ran; a consumer
      must check this before trusting any attribution field (same contract as
      ``seed_cohort["available"]``).
    - ``device``: carries ``None`` because no tracker existed so the device is
      genuinely unknown; following the module's own stated convention that
      unmeasured measurement keys carry ``None`` rather than a plausible-looking
      real value."""
    return {
        "enabled": False,
        "available": False,
        "device": None,
        "reason": reason,
        "n_snapshots": 0,
        "bucket_order": list(BUCKET_ORDER),
        "residency_owner_contract": residency_owner_contract(),
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "peak_gtt_used_bytes": None,
        "gtt_memory_semantics": GTT_MEMORY_SEMANTICS,
        "peak_bucket_bytes": {},
        "history_len": 0,
        "history_stride": None,
        "history_max": None,
        "final": None,
        "growth": {"available": False, "reason": reason},
        "series": [],
    }
