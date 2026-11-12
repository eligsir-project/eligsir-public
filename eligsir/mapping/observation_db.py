"""Frame<->Gaussian observation database (keyframe DB design, 2026-07-25).

Stores, per frame, which Gaussian UIDs its view frustum saw (see
docs/superpowers/specs/2026-07-25-keyframe-database-design.md). Backed by a
sparse torch matrix (rows=frame_id, cols=gaussian_uid) so overlap queries
against the FULL observation history cost one sparse matrix-vector product,
not a Python loop over history. No external index library -- this project
must run identically on ROCm and CUDA, which rules out FAISS/RAPIDS/
cuCollections (NVIDIA-only); native torch sparse ops lower to the correct
backend automatically.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from eligsir.mapping.hot_tier_cache import (
    MAX_RETAINED_TIERS_PER_FRAME,
    HotTierCache,
    entry_bytes,
)
from eligsir.mapping.stores.cold_payload_store import (
    BoundedColdPayloadStore,
    ColdPayloadDiskCache,
    ColdPayloadSpillReceipt,
    ColdPayloadSpillRequest,
)


class BufferStage(Enum):
    """Which historical-buffer stage a frame currently belongs to.

    Set explicitly at FrameRecord creation time (MAPPING_KEYFRAME/
    HOLDOUT_TEST/EVICTED_REDUNDANT/GOLDEN). COLD_CACHE and ICECOLD_CACHE
    (Partition-Focus v2, 2026-07-27) ARE actively transitioned after
    creation -- StreamingScene's tiered frame cache
    (_cache_inactive_frame/_archive_to_icecold/promote_from_cache/
    demote_partitions) calls FrameObservationDB.set_buffer_stage() on every
    tier move a frame makes, so the DB's record of a frame's current
    location stays consistent with the scene's actual tiers instead of
    freezing at whatever stage the frame was first recorded with.
    """

    MAPPING_KEYFRAME = "mapping_keyframe"
    HOLDOUT_TEST = "holdout_test"
    EVICTED_REDUNDANT = "evicted_redundant"
    GOLDEN = "golden"
    COLD_CACHE = "cold_cache"
    ICECOLD_CACHE = "icecold_cache"


@dataclass
class CacheHandle:
    """One resolution tier's cached, ready-to-render frame data.

    rgb/depth are already-resized tensors (see resize_rgbd_frame -- depth
    is NEVER interpolated, only nearest-neighbor decimated). Stored
    GPU-resident per the multires-cache design's explicit choice; this
    wrapper exists so a later CPU-offload fallback (if VRAM pressure is
    too high) is a swap of this handle's storage, not a redesign -- no
    offload policy is implemented yet.

    depth is Optional: depth-less sources (e.g. RealityScan Mobile scenes --
    RGB + poses + sparse SfM cloud, no per-frame depth, see
    eligsir/io/sources/datasets/realityscan.py) still get real rgb tiers
    built (2026-08-12 fix) since the low/high tier DECISION itself
    (ResolutionController.decide) is driven purely by ingestion throughput,
    not by depth content -- but there is no depth to cache, so this stays
    None for those tiers. Callers (bind_resolution_tier) must NOT
    synthesize a depth tensor here and must leave a camera's existing
    depth-availability state (cam._sensor_depth_cache) untouched when this
    is None, preserving the pre-existing "no depth for this source" signal
    that depth-supervision loss code already relies on.
    """

    rgb: torch.Tensor
    depth: Optional[torch.Tensor]
    intrinsics: object
    score_provenance: object = None


def _payload_device_type(payload: object) -> Optional[str]:
    """Return a tensor-like payload's device type without importing its owner."""
    device = getattr(payload, "device", None)
    if device is None:
        return None
    return str(getattr(device, "type", device)).split(":", 1)[0]


def _payload_nbytes(payload: object) -> int:
    """Measure one resident CPU payload without materializing or transferring it."""
    if payload is None:
        return 0
    if isinstance(payload, torch.Tensor):
        return int(payload.untyped_storage().nbytes())
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, memoryview):
        return int(payload.nbytes)
    if isinstance(payload, (bytes, bytearray)):
        return len(payload)
    nbytes = getattr(payload, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    raise TypeError(
        "bounded cold payload retention requires a measurable CPU payload"
    )


@dataclass(frozen=True, slots=True)
class ColdPixelHandle:
    """CPU/cold ownership for native pixels or a deterministic reload token.

    The handle deliberately stays separate from :class:`FrameRecord`'s compact
    metadata and visibility fields.  It may borrow already-materialized CPU
    arrays/tensors, retain compressed bytes, or name a source offset through
    ``reload_token``.  Device tensors are rejected: historical full-frame GPU
    ownership belongs exclusively to the bounded active-supervision cache.
    """

    source_identity: str
    rgb: object
    depth: Optional[object]
    intrinsics: object
    reload_token: Optional[object] = None
    payload_evicted: bool = False
    # Prepared COLD page-in must retain the original raw byte semantics.  The
    # compact labels are sufficient to rebuild BufferRefs without retaining a
    # whole original SensorPacket or inventing a second archive.
    rgb_encoding: Optional[str] = None
    depth_encoding: Optional[str] = None
    depth_units: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source_identity:
            raise ValueError("source_identity must be non-empty")
        if (
            self.rgb is None
            and self.depth is None
            and self.reload_token is None
            and not self.payload_evicted
        ):
            raise ValueError("cold pixels require payload data or a reload_token")
        if self.payload_evicted and (self.rgb is not None or self.depth is not None):
            raise ValueError("an evicted cold handle cannot retain resident payloads")
        for name, payload in (("rgb", self.rgb), ("depth", self.depth)):
            device_type = _payload_device_type(payload)
            if device_type not in {None, "cpu"}:
                raise ValueError(
                    f"{name} payload must remain CPU/cold, got device={device_type!r}"
                )
        for name in ("rgb_encoding", "depth_encoding", "depth_units"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a non-empty string or None")

    @property
    def has_resident_payload(self) -> bool:
        return self.rgb is not None or self.depth is not None

    @property
    def reloadable(self) -> bool:
        return self.reload_token is not None

    @property
    def can_materialize(self) -> bool:
        return self.has_resident_payload or self.reloadable

    @property
    def payload_nbytes(self) -> int:
        return _payload_nbytes(self.rgb) + _payload_nbytes(self.depth)

    def without_resident_payload(self) -> "ColdPixelHandle":
        """Keep reload/provenance while releasing resident CPU pixel ownership."""
        return ColdPixelHandle(
            source_identity=self.source_identity,
            rgb=None,
            depth=None,
            intrinsics=self.intrinsics,
            reload_token=self.reload_token,
            payload_evicted=self.reload_token is None,
            rgb_encoding=self.rgb_encoding,
            depth_encoding=self.depth_encoding,
            depth_units=self.depth_units,
        )


@dataclass(frozen=True, slots=True)
class ModalityEligibility:
    """Historical uses independently authorized for one observation."""

    rgb_supervision: bool = False
    depth_supervision: bool = False
    depth_insertion: bool = False


def _normalize_modality_eligibility(value: object) -> ModalityEligibility:
    """Copy any matching immutable control contract onto the DB-owned schema."""
    if isinstance(value, ModalityEligibility):
        return value
    fields = {}
    for name in ("rgb_supervision", "depth_supervision", "depth_insertion"):
        item = getattr(value, name, None)
        if not isinstance(item, bool):
            raise TypeError(
                "modality_eligibility must expose boolean supervision fields"
            )
        fields[name] = item
    return ModalityEligibility(**fields)


@dataclass(frozen=True, slots=True)
class FrameMetadataSnapshot:
    """Immutable CPU-only metadata view, independent of pixel residency."""

    frame_id: int
    pose_c2w: Tuple[Tuple[float, ...], ...] | None
    timestamp: float
    submap_id: int
    buffer_stage: BufferStage
    source_identity: Optional[str] = None
    source_ordinal: Optional[int] = None


@dataclass(frozen=True, slots=True)
class FrameResidencySnapshot:
    """Latest CPU-readable residency state for one historical frame."""

    frame_id: int
    cold_pixels_available: bool
    active_tiers: Tuple[str, ...]
    device_resident_tiers: Tuple[str, ...]
    visibility_device: str
    cold_payload_resident: bool = False
    reloadable: bool = False


@dataclass(frozen=True, slots=True)
class ResidentSupervisionSnapshot:
    """One validated exact-tier HOT payload borrowed for one transaction.

    The observation DB remains the sole owner.  This snapshot carries no
    camera and creates no second residency policy; its tensor references stay
    valid only while the owning record remains in the bounded active LRU.
    """

    frame_id: int
    source_identity: str
    tier: str
    rgb: torch.Tensor
    depth: Optional[torch.Tensor]
    intrinsics: object
    pose_c2w: Tuple[Tuple[float, ...], ...]
    score_provenance: object = None


@dataclass(frozen=True, slots=True)
class HistoricalCandidateSnapshot:
    """Pixel-free selection view over one retained historical identity."""

    frame_id: int
    source_identity: Optional[str]
    source_ordinal: Optional[int]
    timestamp: float
    submap_id: int
    buffer_stage: BufferStage
    modality_eligibility: ModalityEligibility
    can_materialize: bool
    cold_payload_resident: bool
    reloadable: bool
    disposition_snapshot_id: Optional[int] = None
    semantic_action: Optional[str] = None


class ActiveSupervisionCache:
    """Sole bounded residency policy for full-frame supervision tiers.

    Payload dictionaries remain on the compatibility ``FrameRecord`` surface;
    this owner controls which records are HOT.  Keeping policy and payload
    membership separate lets the observation-DB façade preserve existing tier
    binding while making the history-independent bound explicit.
    """

    def __init__(self, max_frames: int) -> None:
        if int(max_frames) < 0:
            raise ValueError("max active supervision frames must be >= 0")
        self.max_frames = int(max_frames)
        self._order: "OrderedDict[int, None]" = OrderedDict()

    def admit(self, frame_id: int) -> Tuple[int, ...]:
        self._order[int(frame_id)] = None
        self._order.move_to_end(int(frame_id))
        evicted = []
        if self.max_frames > 0:
            while len(self._order) > self.max_frames:
                evicted_id, _ = self._order.popitem(last=False)
                evicted.append(evicted_id)
        return tuple(evicted)

    def touch(self, frame_id: int) -> bool:
        if frame_id not in self._order:
            return False
        self._order.move_to_end(frame_id)
        return True

    def remove(self, frame_id: int) -> bool:
        if frame_id not in self._order:
            return False
        del self._order[frame_id]
        return True

    @property
    def frame_ids(self) -> Tuple[int, ...]:
        return tuple(self._order)

    def __len__(self) -> int:
        return len(self._order)


@dataclass
class FrameRecord:
    """One frame's observation-DB entry.

    Attributes:
        frame_id: The source frame's own index (frame.index), NOT the
            training loop's n_frames_ingested counter -- n_frames_ingested
            only advances on a non-rejected ingest_next_frame() return, so
            using it here would collide multiple rejected/guard frames onto
            the same row/key in FrameObservationDB's sparse matrix.
        pose_c2w: 4x4 camera-to-world at capture time.
        timestamp: Source timestamp (seconds).
        submap_id: Which submap this frame belongs to.
        buffer_stage: See BufferStage. Schema only, not auto-transitioned.
        visible_gaussian_uids: int64 tensor of gaussian_uid values this
            frame's frustum saw (from the fast projection pass, Task 3).
        resolution_cache: tier-name -> cache handle. Empty until a future
            extension populates it (schema only, per the design's
            non-goals -- no cache producer in this plan).
        resolution_cache_evicted: True once this record's resolution_cache was
            cleared by ``FrameObservationDB._register_resolution_cache``'s
            bounded-LRU policy (issue #333). An empty ``resolution_cache`` is
            otherwise ambiguous: it could mean "the configured memory bound
            did its job" (expected) or "the cache was never built for this
            frame" (a real defect, e.g. a decode failure). Bind accounting
            reads this flag to keep the two apart instead of reporting every
            eviction as a bind failure -- which is what made
            ``issue109exp_E2`` report 84266 "high_fail" binds. Reset to False
            if the cache is ever repopulated for this frame.
    """

    frame_id: int
    pose_c2w: np.ndarray | None
    timestamp: float
    submap_id: int
    buffer_stage: BufferStage
    visible_gaussian_uids: torch.Tensor
    resolution_cache: Dict[str, object] = field(default_factory=dict)
    resolution_cache_evicted: bool = False
    # Which individual tiers the per-tier retention policy dropped while the
    # frame stayed HOT. `resolution_cache_evicted` is deliberately left as the
    # frame-level "holds nothing at all" flag it has always been: a partial
    # eviction must not make a record that still owns another tier look like a
    # fully-evicted one to `tier_bind._cache_cause`.
    evicted_resolution_tiers: Tuple[str, ...] = ()
    # issue #333: why an EMPTY resolution_cache is empty, when the producer
    # knows (``record_frame_observation``): "not_train_frame" /
    # "depth_load_failed_transient" / "depthless_rgb_cache_disabled" (all BY
    # DESIGN -- see that function's own branch comments) or "build_exception".
    # None means either "the cache was really built" or "an older producer that
    # did not record a reason" -- never assume it implies a defect.
    resolution_cache_absent_reason: Optional[str] = None
    # Canonical transport/source identity is separate from ``frame_id``. The
    # latter is an internal monotonically-increasing sparse-matrix row and may
    # differ from ``source_ordinal`` after a stream epoch/source restart.
    source_identity: Optional[str] = None
    source_ordinal: Optional[int] = None
    historical_eligible: bool = False
    # Optional train-only display-referred exposure/white-balance state.  This
    # belongs to the durable source record so rebinding a tier or rebuilding a
    # cold replay camera cannot silently reset a fitted training-frame value.
    # It is intentionally absent for holdout records.
    photometric_evwb: Optional[torch.Tensor] = None
    modality_eligibility: ModalityEligibility = field(
        default_factory=ModalityEligibility
    )
    disposition_snapshot_id: Optional[int] = None
    semantic_action: Optional[str] = None
    # Prepared-camera bootstrap retains attested calibration and the bounded
    # ready-source route instead of manufacturing a CPU/cold pixel handle.
    intrinsics: object = None
    prepared_source_reference: object = None
    # An admitted prepared COMMIT that could not obtain bounded FRESH GPU
    # residency still owes one first supervision.  The obligation is owned by
    # this durable historical record, not by a bounded runtime context cache:
    # evicting it without a terminal retirement would make a valid admitted
    # source permanently undiscoverable.  The value is deliberately opaque to
    # the generic observation DB; the prepared-runtime owner validates its
    # immutable control/tracking payload at read/write boundaries.
    prepared_commit_obligation: object = None
    # The same compact authority remains with the durable source after first
    # supervision so a later COLD page-in can rebuild replay residency.
    prepared_commit_context: object = None
    # DEFER promotion needs the same durable source-local authority.  Unlike
    # the runtime's bounded active-context table, this survives until the
    # historical observation is explicitly retired.
    prepared_defer_context: object = None
    # One accepted absolute pose correction, if a source-local producer has
    # prepared and published it.  The immutable correction owns only compact
    # camera geometry; pixels remain solely with their lease/cold owners.
    accepted_pose_correction: object = None


class FrameObservationDB:
    """In-RAM/VRAM Frame<->Gaussian observation store with fast overlap queries.

    ``device`` controls where the sparse matrix lives -- "cpu" for unit
    tests without a GPU, "cuda" in production (matches the live model's
    gaussian_uid buffer device).
    """

    def __init__(
        self,
        device: str = "cuda",
        max_resolution_cache_frames: int = 0,
        resolution_cache_rebuilder: Optional[Callable[[object, int], object]] = None,
        *,
        max_active_supervision_frames: Optional[int] = None,
        max_cold_payload_frames: Optional[int] = None,
        max_cold_payload_bytes: Optional[int] = None,
        max_hot_tier_bytes: Optional[int] = None,
        max_tiers_per_frame: int = MAX_RETAINED_TIERS_PER_FRAME,
        tier_prepare_cost_provider: Optional[Callable[[str], Optional[float]]] = None,
        cold_payload_spill_root: object | None = None,
        cold_payload_spill_max_bytes: Optional[int] = None,
    ) -> None:
        self._device = device
        self._frames: Dict[int, FrameRecord] = {}
        self._cold_pixels: Dict[int, ColdPixelHandle] = {}
        self._cold_payloads: BoundedColdPayloadStore[ColdPixelHandle] = (
            BoundedColdPayloadStore(
                max_frames=max_cold_payload_frames,
                max_bytes=max_cold_payload_bytes,
            )
        )
        self._cold_payload_disk = (
            None
            if cold_payload_spill_root is None
            else ColdPayloadDiskCache(
                cold_payload_spill_root, max_bytes=cold_payload_spill_max_bytes
            )
        )
        self._pending_cold_payload_spills: "OrderedDict[int, None]" = OrderedDict()
        self._inflight_cold_payload_spills: set[int] = set()
        self._cold_payload_spill_failures: list[ColdPayloadSpillReceipt] = []
        self._cold_payload_spill_lock = threading.RLock()
        self._source_identity_to_frame_id: Dict[str, int] = {}
        # This is the authoritative service order for durable COMMIT
        # obligations.  Normal terminal handling and retirement remove entries
        # through ``clear_prepared_commit_obligation``.
        self._prepared_commit_obligation_sources: OrderedDict[str, object] = OrderedDict()
        self._historical_rgb_materializable_generation = 0
        # Observers notified when a retained identity loses its LAST route back
        # to real pixels (bounded-cold eviction of a payload with no reload
        # token). See `_retire_unmaterializable`.
        self._payload_eviction_observers: list[Callable[[int], None]] = []
        self._unmaterializable_announced: set[int] = set()
        self._unmaterializable_evictions = 0
        self._resolution_cache_rebuilder = resolution_cache_rebuilder
        # BUG FIX (2026-08-04, ScanNet++ VRAM blowup): record.resolution_cache
        # (multires-cache plan) holds GPU-resident, native-resolution RGB+depth
        # tensors built unconditionally for EVERY is_train frame
        # (record_frame_observation -> build_resolution_cache, divisor=1 for
        # the "high" tier regardless of the ResolutionController's active
        # tier or any --streaming_auto_downscale_max_pixels cap). FrameRecord
        # itself is kept forever in self._frames (no eviction), so this cache
        # accumulated ~20MB/frame at ScanNet++'s 1920x1440 native resolution
        # with zero bound -- ~1300 admitted keyframes by iter 4500 accounted
        # for the observed 34GB VRAM growth while N (Gaussian count) grew
        # only modestly. The "hot window" offload_camera_image() eviction in
        # StreamingScene only frees a camera's *currently bound*
        # original_image reference, not these separately-held cache tensors.
        # Bound retention to the most recently-cached `max_resolution_cache_
        # frames` frames (LRU by insertion order); 0 = unbounded (test
        # default / back-compat).
        if max_active_supervision_frames is None:
            active_bound = int(max_resolution_cache_frames)
        else:
            active_bound = int(max_active_supervision_frames)
            legacy_bound = int(max_resolution_cache_frames)
            if legacy_bound not in {0, active_bound}:
                raise ValueError(
                    "max_resolution_cache_frames and max_active_supervision_frames "
                    "must not specify different residency policies"
                )
        self._max_resolution_cache_frames = active_bound
        self._active_supervision_cache = ActiveSupervisionCache(active_bound)
        # Compatibility alias for issue #333 tests/telemetry. This is the same
        # OrderedDict owned by ActiveSupervisionCache, not a second policy.
        self._resolution_cache_order = self._active_supervision_cache._order
        # Per-tier residency inside the count-bounded HOT frame set above.
        # ``ActiveSupervisionCache`` bounds how many FRAMES are hot; it knows
        # nothing about bytes, and a native tier costs ~16x a quarter tier for
        # the same frame. This policy bounds the extra multi-tier retention in
        # real measured bytes and evicts by value, not by age alone. See
        # ``eligsir/mapping/hot_tier_cache.py``.
        self._hot_tier_cache = HotTierCache(
            max_tiers_per_frame=max_tiers_per_frame,
            max_bytes=max_hot_tier_bytes,
            prepare_cost_provider=tier_prepare_cost_provider,
        )
        # issue #333: how many evictions the bounded policy performed. This is
        # the denominator that makes "cache_empty" binds interpretable.
        self.resolution_cache_evictions = 0
        self.resolution_cache_reconstruction_attempts = 0
        self.resolution_cache_reconstructions = 0
        self.resolution_cache_reconstruction_failures = 0
        # BUG FIX (2026-08-04, follow-up to the resolution_cache VRAM fix
        # above): this used to maintain permanent Python-list copies
        # (self._rows/self._cols, one Python int object per (frame, visible
        # uid) pair) purely to feed the COO/CSR rebuild in _ensure_csr(),
        # duplicating data ALREADY held compactly, forever, in each
        # FrameRecord.visible_gaussian_uids tensor. Measured at realistic
        # ScanNet++ scale (7040 ingested candidate frames, N ramping to
        # 302898, 15% visible fraction -- see tests/test_observation_db.py):
        # those two lists alone reached ~12.7GB RSS, spiking to ~19GB during
        # a single CSR rebuild, fully independent of Gaussian count N. The
        # per-frame tensors were never the problem; the redundant list
        # accumulation was. _ensure_csr() below now builds the COO indices
        # transiently from self._frames.values() each time it is dirty --
        # same full-history correctness (no frame's visibility history is
        # ever dropped), zero permanent duplicate storage.
        self._max_uid_seen = -1
        self._max_frame_id_seen = -1
        self._dirty = True
        self._csr: Optional[torch.Tensor] = None
        # PERF FIX (2026-08-04, follow-up to the d0a5f36 dedup fix): that fix
        # removed the permanent _rows/_cols leak but left _ensure_csr()
        # rebuilding the ENTIRE sparse matrix (walk self._frames.values(),
        # re-cat every frame's uids, coalesce/sort all of it) on every dirty
        # call -- and overlap_query() is called once per admission candidate
        # in keyframes.py's coverage_gain_gate path, immediately followed by
        # add_frame() for that same frame (see scene/streaming_scene.py's
        # ingest_next_frame: visibility_fn -> admission_fn -> record_fn), so
        # the DB is dirty again before the next query. That is a full
        # O(total observations so far) rebuild on essentially every
        # candidate frame. Measured host-repro at realistic ScanNet++ scale
        # (F=7040, N->302898, 15% visible): a single rebuild at the end of
        # that range takes ~1.5s and is ~100% of overlap_query's cost (the
        # actual spmm is ~20ms) -- growing per-call as the run progresses,
        # not a one-time cost.
        #
        # Fix: add_frame() only ever APPENDS a new row (frame_id is the
        # source frame's own increasing index -- verified via train_streaming
        # .py's call sites, always increasing: bootstrap frames 0..k-1, then
        # ingest_next_frame's visibility_fn/admission_fn/record_fn triple
        # runs once per candidate in ingestion order). That means the CSR's
        # crow_indices (row pointers) and col_indices/values can be extended
        # incrementally -- new column indices appended to a capacity-doubling
        # buffer (O(1) amortized per add_frame, like a growable vector), new
        # crow boundaries appended to a small O(n_rows) Python list -- with
        # NO re-sort/coalesce of prior data required (torch's sparse CSR
        # does not require column indices to be sorted within a row for
        # sparse.mm correctness, verified empirically). _ensure_csr() then
        # just wraps the buffer's filled prefix into a CSR tensor (a view,
        # not a copy, for col/val) instead of rebuilding from scratch.
        #
        # This relies on frame_id strictly increasing across add_frame
        # calls, which holds for every real call site. If that invariant
        # is ever violated (out-of-order or duplicate frame_id -- has never
        # been observed, but is not provably impossible for every future
        # caller), _incremental_valid flips False and _ensure_csr() falls
        # back to the old full-rebuild-from-self._frames path permanently
        # for the life of this DB instance -- correctness is preserved
        # either way, only the perf win is lost in that fallback case.
        self._incremental_valid = True
        self._crow_built_upto_row = -1
        self._crow: list = [0]
        self._buf_col: Optional[torch.Tensor] = None
        self._buf_len = 0

    @property
    def cold_payload_disk_enabled(self) -> bool:
        return self._cold_payload_disk is not None

    def configure_cold_payload_spill(
        self, root: object, *, max_bytes: Optional[int] = None
    ) -> None:
        """Enable the prepared local spill tier once, during runtime setup."""
        if self._cold_payload_disk is not None:
            raise RuntimeError("cold payload spill is already configured")
        self._cold_payload_disk = ColdPayloadDiskCache(root, max_bytes=max_bytes)

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def historical_rgb_materializable_generation(self) -> int:
        """Monotonic membership version for materializable RGB history."""

        return self._historical_rgb_materializable_generation

    def add_frame(self, record: FrameRecord) -> None:
        self._frames[record.frame_id] = record
        uids = record.visible_gaussian_uids
        if uids.numel() > 0:
            self._max_uid_seen = max(self._max_uid_seen, int(uids.max().item()))
        self._max_frame_id_seen = max(self._max_frame_id_seen, record.frame_id)
        self._dirty = True
        if record.resolution_cache:
            self._register_resolution_cache(record.frame_id)
        self._try_incremental_append(record)

    @staticmethod
    def _source_identity_key(source_identity: object) -> str:
        fingerprint = getattr(source_identity, "fingerprint", None)
        key = str(fingerprint() if callable(fingerprint) else source_identity)
        if not key:
            raise ValueError("source identity must be non-empty")
        return key

    def _allocate_frame_id(self, requested_id: int, source_key: str) -> int:
        """Preserve safe source ordinals and remap restarts to monotonic rows."""
        if source_key in self._source_identity_to_frame_id:
            raise ValueError(f"source identity already recorded: {source_key}")
        requested_id = int(requested_id)
        if requested_id > self._max_frame_id_seen and requested_id not in self._frames:
            return requested_id
        return self._max_frame_id_seen + 1

    def frame_id_for_source(self, source_identity: object) -> Optional[int]:
        """Resolve a canonical source identity to its internal visibility row."""
        return self._source_identity_to_frame_id.get(
            self._source_identity_key(source_identity)
        )

    def record_prepared_commit_obligation(
        self, source_identity: object, obligation: object
    ) -> None:
        """Attach one unresolved prepared-COMMIT obligation to durable history.

        This is intentionally not an LRU/cache entry.  Its lifetime is the
        source's semantic obligation: it remains until successful FRESH READY
        re-entry or an explicit historical retirement removes the durable
        source itself.
        """

        if obligation is None:
            raise TypeError("prepared COMMIT obligation must not be None")
        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            raise ValueError("prepared COMMIT obligation requires durable source")
        record = self._frames.get(frame_id)
        if record is None or not record.historical_eligible:
            raise RuntimeError("prepared COMMIT obligation has no eligible durable record")
        if record.semantic_action != "commit":
            raise ValueError("prepared COMMIT obligation requires COMMIT provenance")
        existing = record.prepared_commit_obligation
        if existing is not None and existing != obligation:
            raise ValueError("prepared COMMIT obligation cannot change")
        context = record.prepared_commit_context
        if context is not None and context != obligation:
            raise ValueError("prepared COMMIT context cannot change")
        record.prepared_commit_context = obligation
        record.prepared_commit_obligation = obligation
        source_key = self._source_identity_key(source_identity)
        self._prepared_commit_obligation_sources[source_key] = source_identity

    def next_prepared_commit_obligation_source(self) -> object | None:
        """Return the oldest unresolved prepared-COMMIT identity in O(1)."""

        if not self._prepared_commit_obligation_sources:
            return None
        source_key, source = next(
            iter(self._prepared_commit_obligation_sources.items())
        )
        frame_id = self._source_identity_to_frame_id.get(source_key)
        record = None if frame_id is None else self._frames.get(frame_id)
        if record is None or record.prepared_commit_obligation is None:
            raise RuntimeError("prepared COMMIT service index is inconsistent")
        return source

    def prepared_commit_obligation_for_source(self, source_identity: object) -> object | None:
        """Return the unresolved durable prepared-COMMIT obligation, if any."""

        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            return None
        record = self._frames.get(frame_id)
        return None if record is None else record.prepared_commit_obligation

    def prepared_commit_context_for_source(self, source_identity: object) -> object | None:
        """Return durable COMMIT authority for later replay residency."""

        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            return None
        record = self._frames.get(frame_id)
        return None if record is None else record.prepared_commit_context

    def record_prepared_defer_context(
        self, source_identity: object, context: object
    ) -> None:
        """Attach immutable DEFER page-in authority to durable history."""

        if context is None:
            raise TypeError("prepared DEFER context must not be None")
        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            raise ValueError("prepared DEFER context requires durable source")
        record = self._frames.get(frame_id)
        if record is None or not record.historical_eligible:
            raise RuntimeError("prepared DEFER context has no eligible durable record")
        if record.semantic_action != "defer":
            raise ValueError("prepared DEFER context requires DEFER provenance")
        existing = record.prepared_defer_context
        if existing is not None and existing != context:
            raise ValueError("prepared DEFER context cannot change")
        record.prepared_defer_context = context

    def prepared_defer_context_for_source(self, source_identity: object) -> object | None:
        """Return durable DEFER page-in authority, if any."""

        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            return None
        record = self._frames.get(frame_id)
        return None if record is None else record.prepared_defer_context

    def clear_prepared_commit_obligation(
        self, source_identity: object, *, expected: object | None = None
    ) -> bool:
        """Clear a durable first-supervision obligation after terminal handling."""

        source_key = self._source_identity_key(source_identity)
        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            self._prepared_commit_obligation_sources.pop(source_key, None)
            return False
        record = self._frames.get(frame_id)
        if record is None or record.prepared_commit_obligation is None:
            self._prepared_commit_obligation_sources.pop(source_key, None)
            return False
        if expected is not None and record.prepared_commit_obligation != expected:
            raise ValueError("prepared COMMIT obligation differs from terminal item")
        record.prepared_commit_obligation = None
        self._prepared_commit_obligation_sources.pop(source_key, None)
        return True

    def record_candidate(
        self,
        cam: object,
        frame: object,
        *,
        source_identity: object,
        is_train: bool,
        is_holdout: bool,
        visible_uids: Optional[torch.Tensor] = None,
        submap_id: int = 0,
        historical_eligible: bool = False,
        modality_eligibility: Optional[object] = None,
    ) -> FrameRecord:
        """Record metadata, cold pixels and visibility without building HOT tiers.

        This is the reusable ``StreamingScene`` ``record_fn`` boundary.  It
        intentionally performs no resize, transfer or resolution-cache build;
        root/application wiring pages selected frames through
        ``DeviceTransferOwner`` and then calls :meth:`activate_supervision`.
        """
        if is_holdout:
            stage = BufferStage.HOLDOUT_TEST
        elif is_train:
            stage = BufferStage.MAPPING_KEYFRAME
        else:
            stage = BufferStage.EVICTED_REDUNDANT

        source_key = self._source_identity_key(source_identity)
        source_ordinal = getattr(source_identity, "source_ordinal", None)
        requested_frame_id = int(
            source_ordinal if source_ordinal is not None else getattr(frame, "index")
        )
        frame_id = self._allocate_frame_id(requested_frame_id, source_key)
        source_timestamp = getattr(source_identity, "timestamp_seconds", None)
        if callable(source_timestamp):
            source_timestamp = source_timestamp()
        timestamp = float(
            source_timestamp
            if source_timestamp is not None
            else getattr(frame, "timestamp", frame_id)
        )
        if visible_uids is None:
            visible_uids = torch.zeros(0, dtype=torch.int64)
        if not isinstance(visible_uids, torch.Tensor):
            visible_uids = torch.as_tensor(visible_uids, dtype=torch.int64)
        if modality_eligibility is None:
            modality_eligibility = ModalityEligibility()
        else:
            modality_eligibility = _normalize_modality_eligibility(
                modality_eligibility
            )

        record = FrameRecord(
            frame_id=frame_id,
            pose_c2w=np.asarray(getattr(cam, "c2w", np.eye(4)), dtype=np.float32),
            timestamp=timestamp,
            submap_id=int(submap_id),
            buffer_stage=stage,
            visible_gaussian_uids=visible_uids,
            resolution_cache={},
            resolution_cache_absent_reason="not_activated",
            source_identity=source_key,
            source_ordinal=(
                int(source_ordinal) if source_ordinal is not None else requested_frame_id
            ),
            historical_eligible=bool(historical_eligible),
            modality_eligibility=modality_eligibility,
        )

        mapping_frame = getattr(frame, "mapping_frame", None)
        if mapping_frame is None:
            mapping_frame = getattr(frame, "_mapping_frame", None)
        cold_handle = None
        if mapping_frame is not None:
            if str(getattr(mapping_frame, "device", "cpu")) != "cpu":
                raise ValueError("record_candidate MappingFrame must be CPU-owned")
            rgb_ref = getattr(mapping_frame, "rgb", None)
            depth_ref = getattr(mapping_frame, "depth", None)
            cold_handle = ColdPixelHandle(
                source_identity=source_key,
                rgb=getattr(rgb_ref, "data", rgb_ref),
                depth=None if depth_ref is None else getattr(depth_ref, "data", depth_ref),
                intrinsics=getattr(mapping_frame, "intrinsics", None),
                reload_token=getattr(mapping_frame, "reload_token", None),
            )
        else:
            rgb = getattr(frame, "_rgb_np", None)
            if rgb is None:
                rgb = getattr(frame, "_rgb_bytes", None)
            depth = getattr(frame, "_depth_np", None)
            if depth is None:
                depth = getattr(frame, "_depth_bytes", None)
            rgb_path = getattr(frame, "rgb_path", None)
            depth_path = getattr(frame, "depth_path", None)
            reload_token = (rgb_path, depth_path) if rgb_path or depth_path else None
            if rgb is not None or depth is not None or reload_token is not None:
                intrinsics = tuple(
                    getattr(frame, name, None)
                    for name in ("width", "height", "fx", "fy", "cx", "cy")
                )
                cold_handle = ColdPixelHandle(
                    source_identity=source_key,
                    rgb=rgb,
                    depth=depth,
                    intrinsics=intrinsics,
                    reload_token=reload_token,
                )
        # Publish metadata/visibility and its optional cold owner only after
        # every ownership/domain validation above succeeds.
        self.add_frame(record)
        self._source_identity_to_frame_id[source_key] = frame_id
        if cold_handle is not None:
            self.store_cold_pixels(frame_id, cold_handle)
        return record

    def record_prepared_candidate(
        self,
        *,
        source_identity: object,
        pose_c2w: object,
        timestamp: float,
        intrinsics: object,
        ready_source_reference: object,
        modality_eligibility: object,
        is_train: bool,
        submap_id: int = 0,
    ) -> FrameRecord:
        """Register an attested device camera without inventing a cold owner."""

        source_key = self._source_identity_key(source_identity)
        source_ordinal = getattr(source_identity, "source_ordinal", None)
        if source_ordinal is None:
            raise ValueError("prepared source identity requires source_ordinal")
        pose = np.asarray(pose_c2w, dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("prepared candidate pose must be a finite 4x4 matrix")
        required_intrinsics = ("fx", "fy", "cx", "cy", "width", "height")
        if any(getattr(intrinsics, name, None) is None for name in required_intrinsics):
            raise ValueError("prepared candidate requires authoritative intrinsics")
        if float(intrinsics.fx) <= 0.0 or float(intrinsics.fy) <= 0.0:
            raise ValueError("prepared candidate intrinsics must be positive")
        if int(intrinsics.width) <= 0 or int(intrinsics.height) <= 0:
            raise ValueError("prepared candidate intrinsics geometry must be positive")
        if getattr(ready_source_reference, "source_identity", None) != source_identity:
            raise ValueError("prepared source reference identity mismatch")
        if getattr(ready_source_reference, "control_token", None) is None:
            raise ValueError("prepared source reference requires its control token")
        existing_frame_id = self.frame_id_for_source(source_identity)
        if existing_frame_id is not None:
            record = self._frames.get(existing_frame_id)
            if record is None:
                raise RuntimeError("prepared source lookup has no frame record")
            if record.source_identity != source_key:
                raise RuntimeError("prepared source lookup changed source identity")
            if not record.historical_eligible:
                raise ValueError("source identity already recorded")
            if not np.array_equal(record.pose_c2w, pose):
                raise ValueError("prepared source pose differs from durable record")
            if record.intrinsics != intrinsics:
                raise ValueError("prepared source intrinsics differ from durable record")
            # The durable COLD record is authoritative for future page-in.
            # Do not retain this READY lease here: its borrowed device payloads
            # remain owned solely by the bounded ReadySourceOwner cache.
            return record
        frame_id = self._allocate_frame_id(int(source_ordinal), source_key)
        record = FrameRecord(
            frame_id=frame_id,
            pose_c2w=np.array(pose, copy=True),
            timestamp=float(timestamp),
            submap_id=int(submap_id),
            buffer_stage=(BufferStage.MAPPING_KEYFRAME if is_train else BufferStage.EVICTED_REDUNDANT),
            visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
            resolution_cache={},
            resolution_cache_absent_reason="prepared_device_native",
            source_identity=source_key,
            source_ordinal=int(source_ordinal),
            # A ready binding owns borrowed device payloads only for its
            # selected lease lifetime.  It is provenance, not a durable
            # materializer: never advertise it to the historical catalog.
            historical_eligible=False,
            modality_eligibility=_normalize_modality_eligibility(modality_eligibility),
            semantic_action="prepared_device_transient",
            intrinsics=intrinsics,
            prepared_source_reference=ready_source_reference,
        )
        self.add_frame(record)
        self._source_identity_to_frame_id[source_key] = frame_id
        return record

    def prepared_source_reference_for_source(self, source_identity: object) -> object | None:
        """Return the attested ready-source route retained for one bootstrap ID."""

        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            return None
        record = self._frames.get(frame_id)
        return None if record is None else record.prepared_source_reference

    def prepared_pose_correction_for_source(self, source_identity: object) -> object | None:
        """Return the one accepted immutable correction for a retained source."""

        from eligsir.runtime.prepared_pose_correction import PreparedPoseCorrection

        frame_id = self.frame_id_for_source(source_identity)
        record = None if frame_id is None else self._frames.get(frame_id)
        correction = None if record is None else record.accepted_pose_correction
        if correction is None or isinstance(correction, PreparedPoseCorrection):
            return correction
        raise ValueError("malformed accepted pose correction")

    def commit_prepared_pose_correction(
        self, correction: object, *, expected_revision: int
    ) -> bool:
        """Compare-and-publish one completed next-revision pose correction."""

        from eligsir.runtime.prepared_pose_correction import PreparedPoseCorrection

        if type(expected_revision) is not int or expected_revision < 0:
            return False
        if not isinstance(correction, PreparedPoseCorrection):
            return False
        try:
            frame_id = self.frame_id_for_source(correction.source_identity)
            record = None if frame_id is None else self._frames.get(frame_id)
            if (
                record is None
                or frame_id in self._unmaterializable_announced
                or record.source_identity != self._source_identity_key(correction.source_identity)
            ):
                return False
            accepted = record.accepted_pose_correction
            if accepted is not None and not isinstance(accepted, PreparedPoseCorrection):
                return False
            current_revision = 0 if accepted is None else accepted.revision
            if (
                expected_revision != current_revision
                or correction.revision != current_revision + 1
                or not correction.completion_event.query()
            ):
                return False
            pose = np.asarray(correction.pose_c2w, dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                return False
        except (AttributeError, TypeError, ValueError):
            return False
        record.pose_c2w = np.array(pose, copy=True)
        record.accepted_pose_correction = correction
        return True

    def record_historical_candidate(
        self,
        *,
        source_identity: object,
        pose_c2w: object,
        timestamp: float,
        intrinsics: object,
        cold_pixels: ColdPixelHandle,
        modality_eligibility: object,
        disposition_snapshot_id: Optional[int] = None,
        semantic_action: str = "defer",
        submap_id: int = 0,
    ) -> FrameRecord:
        """Record a deferred source without constructing a camera or GPU tier."""

        return self._record_durable_candidate(
            source_identity=source_identity,
            pose_c2w=pose_c2w,
            timestamp=timestamp,
            intrinsics=intrinsics,
            cold_pixels=cold_pixels,
            modality_eligibility=modality_eligibility,
            disposition_snapshot_id=disposition_snapshot_id,
            semantic_action=semantic_action,
            submap_id=submap_id,
            allowed_semantic_actions=("defer",),
        )

    def record_prepared_historical_candidate(
        self,
        *,
        source_identity: object,
        pose_c2w: object,
        timestamp: float,
        intrinsics: object,
        cold_pixels: ColdPixelHandle,
        modality_eligibility: object,
        disposition_snapshot_id: int,
        semantic_action: str,
        submap_id: int = 0,
    ) -> FrameRecord:
        """Record an admitted or promoted prepared source as durable COLD data.

        This is a catalog operation only: it owns CPU/reload pixels and the
        immutable source facts before a separate READY cache offer.  It never
        creates a device camera or changes HOT residency.
        """

        return self._record_durable_candidate(
            source_identity=source_identity,
            pose_c2w=pose_c2w,
            timestamp=timestamp,
            intrinsics=intrinsics,
            cold_pixels=cold_pixels,
            modality_eligibility=modality_eligibility,
            disposition_snapshot_id=disposition_snapshot_id,
            semantic_action=semantic_action,
            submap_id=submap_id,
            allowed_semantic_actions=("commit", "defer"),
        )

    def record_nontraining_cache(
        self,
        *,
        source_identity: object,
        pose_c2w: object,
        timestamp: float,
        intrinsics: object,
        cold_pixels: ColdPixelHandle,
        role: str,
    ) -> FrameRecord:
        """Cache a non-training source without making it replay eligible."""
        if role not in {"holdout", "guard", "rejected", "terminal"}:
            raise ValueError("non-training cache role is invalid")
        source_key = self._source_identity_key(source_identity)
        source_ordinal = getattr(source_identity, "source_ordinal", None)
        if source_ordinal is None:
            raise ValueError("non-training cache source identity requires source_ordinal")
        if cold_pixels.source_identity != source_key:
            raise ValueError("non-training cache cold handle source identity mismatch")
        pose = None if pose_c2w is None else np.asarray(pose_c2w, dtype=np.float32)
        if pose is None and role != "terminal":
            raise ValueError("non-training cache requires a finite accepted pose")
        if pose is not None and (pose.shape != (4, 4) or not np.isfinite(pose).all()):
            raise ValueError("non-training cache requires a finite accepted pose")
        if cold_pixels.intrinsics != intrinsics:
            raise ValueError("non-training cache cold handle intrinsics mismatch")
        self._preflight_cold_handle_record(cold_pixels)
        frame_id = self._allocate_frame_id(int(source_ordinal), source_key)
        record = FrameRecord(
            frame_id=frame_id,
            pose_c2w=None if pose is None else np.array(pose, copy=True),
            timestamp=float(timestamp),
            submap_id=0,
            buffer_stage=(
                BufferStage.HOLDOUT_TEST
                if role == "holdout"
                else BufferStage.EVICTED_REDUNDANT
            ),
            visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
            resolution_cache={},
            resolution_cache_absent_reason="nontraining_cache",
            source_identity=source_key,
            source_ordinal=int(source_ordinal),
            historical_eligible=False,
            modality_eligibility=ModalityEligibility(),
            intrinsics=intrinsics,
            semantic_action=role,
        )
        self.add_frame(record)
        self._source_identity_to_frame_id[source_key] = frame_id
        self.store_cold_pixels(frame_id, cold_pixels)
        return record

    def _preflight_cold_handle_record(self, handle: ColdPixelHandle) -> None:
        """Refuse a record before metadata publication when RAM is unavailable."""
        if self._cold_payload_disk is None or not handle.has_resident_payload:
            return
        with self._cold_payload_spill_lock:
            if not self._prepared_cold_payload_ready_locked(handle.payload_nbytes):
                raise RuntimeError(
                    "cold payload cache capacity was not reserved before recording"
                )

    def _record_durable_candidate(
        self,
        *,
        source_identity: object,
        pose_c2w: object,
        timestamp: float,
        intrinsics: object,
        cold_pixels: ColdPixelHandle,
        modality_eligibility: object,
        disposition_snapshot_id: Optional[int],
        semantic_action: str,
        submap_id: int,
        allowed_semantic_actions: tuple[str, ...],
    ) -> FrameRecord:
        """Shared implementation for COLD catalog records.

        The public entry points make the semantic authority explicit; this
        helper keeps their storage behavior identical.
        """

        if str(semantic_action) not in allowed_semantic_actions:
            if allowed_semantic_actions == ("defer",):
                raise ValueError("historical candidates require DEFER semantic provenance")
            raise ValueError(
                "durable candidates require semantic provenance in "
                + repr(allowed_semantic_actions)
            )
        if disposition_snapshot_id is None:
            raise ValueError("historical candidates require disposition_snapshot_id")
        disposition_snapshot_id = int(disposition_snapshot_id)
        if disposition_snapshot_id < 0:
            raise ValueError("disposition_snapshot_id must be >= 0")
        source_key = self._source_identity_key(source_identity)
        source_ordinal = getattr(source_identity, "source_ordinal", None)
        if source_ordinal is None:
            raise ValueError("historical source identity requires source_ordinal")
        if cold_pixels.source_identity != source_key:
            raise ValueError("historical cold handle source identity mismatch")
        pose = np.asarray(pose_c2w, dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("historical candidate pose must be a finite 4x4 matrix")
        if cold_pixels.intrinsics != intrinsics:
            raise ValueError("historical cold handle intrinsics mismatch")
        self._preflight_cold_handle_record(cold_pixels)
        frame_id = self._allocate_frame_id(int(source_ordinal), source_key)
        record = FrameRecord(
            frame_id=frame_id,
            pose_c2w=np.array(pose, copy=True),
            timestamp=float(timestamp),
            submap_id=int(submap_id),
            buffer_stage=BufferStage.COLD_CACHE,
            visible_gaussian_uids=torch.zeros(0, dtype=torch.int64),
            resolution_cache={},
            resolution_cache_absent_reason="deferred_not_activated",
            source_identity=source_key,
            source_ordinal=int(source_ordinal),
            historical_eligible=True,
            modality_eligibility=_normalize_modality_eligibility(
                modality_eligibility
            ),
            intrinsics=intrinsics,
            disposition_snapshot_id=disposition_snapshot_id,
            semantic_action=str(semantic_action),
        )
        self.add_frame(record)
        self._source_identity_to_frame_id[source_key] = frame_id
        self.store_cold_pixels(frame_id, cold_pixels)
        return record

    def _buf_append(self, uids: torch.Tensor) -> None:
        """Append `uids` to the growable column-index buffer, doubling
        capacity when exceeded (amortized O(1) per call, like a growable
        vector) instead of a fresh allocation+copy of ALL prior data every
        time -- see the PERF FIX note in __init__."""
        n = int(uids.numel())
        if n == 0:
            return
        need = self._buf_len + n
        cur_cap = 0 if self._buf_col is None else int(self._buf_col.numel())
        if need > cur_cap:
            new_cap = max(need, max(1024, cur_cap * 2))
            new_buf = torch.empty(new_cap, dtype=torch.int64, device=self._device)
            if self._buf_col is not None and self._buf_len > 0:
                new_buf[: self._buf_len] = self._buf_col[: self._buf_len]
            self._buf_col = new_buf
        self._buf_col[self._buf_len : self._buf_len + n] = uids.to(
            device=self._device, dtype=torch.int64
        )
        self._buf_len += n

    def _try_incremental_append(self, record: FrameRecord) -> None:
        """Extend the incremental CSR buffer with `record`'s observations,
        or permanently disable the incremental path if frame_id ordering
        ever breaks the append-only assumption (see PERF FIX note)."""
        if not self._incremental_valid:
            return
        fid = record.frame_id
        if fid <= self._crow_built_upto_row:
            self._incremental_valid = False
            return
        # Rows strictly between the last built row and this one (e.g. a
        # pregate-short-circuited frame that never reaches record_fn, see
        # scene/streaming_scene.py's ingest_next_frame) are empty -- give
        # them a zero-width crow entry so row indexing stays aligned with
        # frame_id, matching the full-rebuild path's n_rows convention.
        for _ in range(fid - self._crow_built_upto_row - 1):
            self._crow.append(self._buf_len)
        self._buf_append(record.visible_gaussian_uids)
        self._crow.append(self._buf_len)
        self._crow_built_upto_row = fid

    def _register_resolution_cache(self, frame_id: int) -> None:
        """Track insertion order of GPU-resident resolution_cache tensors and
        evict the oldest once over `_max_resolution_cache_frames` (see the
        BUG FIX note in __init__). Evicting just clears the FrameRecord's
        resolution_cache dict -- the record itself (pose, visibility,
        buffer_stage) is untouched and stays queryable forever, only the
        heavy cached image tensors are dropped.

        Issue #333: an eviction is recorded on the FrameRecord itself
        (``resolution_cache_evicted``) and counted here, so a later bind can
        tell "the bounded LRU policy dropped this, as configured" apart from
        "this frame's cache was never built" -- previously indistinguishable,
        and reported as a bind failure either way."""
        _record = self._frames.get(frame_id)
        if _record is not None and _record.resolution_cache:
            # (Re)populated for this frame -- it is resident again, so the
            # stale "was evicted" mark must not survive.
            _record.resolution_cache_evicted = False
        for _evicted_id in self._active_supervision_cache.admit(frame_id):
            _evicted_record = self._frames.get(_evicted_id)
            self.resolution_cache_evictions += 1
            self._hot_tier_cache.discard(_evicted_id)
            if _evicted_record is not None:
                _evicted_record.resolution_cache = {}
                _evicted_record.resolution_cache_evicted = True

    def touch_resolution_cache(self, frame_id: int) -> bool:
        """Refresh recency on a cache HIT (issue #333 phase 3).

        This is called from bind_resolution_tier when a resident cache hit
        occurs, to update the LRU order and keep hot/frequently-used frames
        resident instead of evicting them solely because they are old in
        source insertion order.

        Does NOT trigger eviction (unlike _register_resolution_cache). Only
        moves the frame to the end of the LRU order if it is already resident.

        Args:
            frame_id: frame to refresh in the LRU order

        Returns:
            True if the frame was found, resident, and refreshed; False if
            the frame was not found, not resident, or not in the cache order.
        """
        if frame_id not in self._resolution_cache_order:
            return False
        record = self._frames.get(frame_id)
        if record is None or not record.resolution_cache:
            return False
        # Frame is resident and in the cache order -- move it to the end
        # (most recently used) to delay its eviction.
        self._hot_tier_cache.touch(frame_id)
        return self._active_supervision_cache.touch(frame_id)

    def touch_resolution_tier(self, frame_id: int, tier: str) -> bool:
        """Refresh recency/frequency for ONE materialized tier of a HOT frame.

        Frame-level ``touch_resolution_cache`` credits every retained tier
        equally; a per-tier hit is what the value-based eviction policy needs
        in order to tell a repeatedly-replayed quarter tier apart from a native
        tier that has been sitting unused on the same frame.
        """

        if not self.touch_resolution_cache(frame_id):
            return False
        return self._hot_tier_cache.touch(int(frame_id), str(tier))

    @property
    def hot_tier_cache(self) -> HotTierCache:
        """The per-tier residency policy inside the HOT frame set."""

        return self._hot_tier_cache

    @property
    def hot_tier_cache_bytes(self) -> int:
        """Measured bytes retained by multi-tier HOT residency."""

        return self._hot_tier_cache.total_bytes

    @property
    def resolution_cache_occupancy(self) -> int:
        """How many frames currently hold a resident resolution_cache (issue
        #333 telemetry -- bounded by ``_max_resolution_cache_frames``)."""
        return len(self._active_supervision_cache)

    @property
    def active_supervision_occupancy(self) -> int:
        """Number of HOT full-frame entries, bounded independently of history."""
        return len(self._active_supervision_cache)

    @property
    def active_supervision_frame_ids(self) -> Tuple[int, ...]:
        """HOT frame identities in least-to-most-recently-used order."""
        return self._active_supervision_cache.frame_ids

    @property
    def cold_pixel_handle_count(self) -> int:
        """Number of CPU/cold handles; these never contribute to VRAM residency."""
        with self._cold_payload_spill_lock:
            return len(self._cold_pixels) + len(self._cold_payloads)

    @property
    def cold_payload_occupancy(self) -> int:
        """Resident non-reloadable CPU payload count under the explicit bound."""
        with self._cold_payload_spill_lock:
            return len(self._cold_payloads)

    @property
    def cold_payload_bytes(self) -> int:
        """Measured bytes owned by the bounded live CPU payload archive."""
        with self._cold_payload_spill_lock:
            return self._cold_payloads.total_bytes

    def get_frame(self, frame_id: int) -> Optional[FrameRecord]:
        return self._frames.get(frame_id)

    def get_metadata(self, frame_id: int) -> Optional[FrameMetadataSnapshot]:
        """Return an immutable CPU metadata snapshot without pixel references."""
        record = self._frames.get(frame_id)
        if record is None:
            return None
        pose = None if record.pose_c2w is None else np.asarray(record.pose_c2w)
        return FrameMetadataSnapshot(
            frame_id=int(record.frame_id),
            pose_c2w=None if pose is None else tuple(tuple(float(v) for v in row) for row in pose),
            timestamp=float(record.timestamp),
            submap_id=int(record.submap_id),
            buffer_stage=record.buffer_stage,
            source_identity=record.source_identity,
            source_ordinal=record.source_ordinal,
        )

    def resident_supervision_for_source(
        self, source_identity: object, tier: str
    ) -> Optional[ResidentSupervisionSnapshot]:
        """Borrow one exact, validated HOT tier and refresh its LRU recency.

        A tier mismatch or stale/malformed handle is a cache miss.  Callers
        then use the deterministic cold/reload-token path; this method never
        substitutes another tier, repairs payloads, or transfers tensors.
        """

        tier = str(tier)
        if not tier:
            return None
        source_key = self._source_identity_key(source_identity)
        frame_id = self._source_identity_to_frame_id.get(source_key)
        if frame_id is None:
            return None
        record = self._frames.get(frame_id)
        if (
            record is None
            or record.source_identity != source_key
            or frame_id not in self._resolution_cache_order
        ):
            return None
        handle = record.resolution_cache.get(tier)
        if not isinstance(handle, CacheHandle):
            return None
        if handle.score_provenance is None:
            return None
        rgb, depth = handle.rgb, handle.depth
        if not isinstance(rgb, torch.Tensor) or rgb.ndim != 3:
            return None
        if depth is not None and (
            not isinstance(depth, torch.Tensor) or depth.ndim != 2
        ):
            return None
        try:
            width, height, fx, fy, cx, cy = tuple(handle.intrinsics)
            width, height = int(width), int(height)
            calibration = np.asarray((fx, fy, cx, cy), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if (
            width <= 0
            or height <= 0
            or tuple(rgb.shape[-2:]) != (height, width)
            or (depth is not None and tuple(depth.shape) != (height, width))
            or not np.isfinite(calibration).all()
            or float(fx) <= 0.0
            or float(fy) <= 0.0
        ):
            return None
        target_device = str(torch.device(self._device).type)
        if _payload_device_type(rgb) != target_device or (
            depth is not None and _payload_device_type(depth) != target_device
        ):
            return None
        pose = np.asarray(record.pose_c2w, dtype=np.float32)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            return None
        # Credit the exact tier that hit, not just the frame: the value-based
        # retention policy needs per-tier reuse evidence to decide which of a
        # frame's tiers is worth keeping.
        if not self.touch_resolution_tier(frame_id, tier):
            return None
        return ResidentSupervisionSnapshot(
            frame_id=int(frame_id),
            source_identity=source_key,
            tier=tier,
            rgb=rgb,
            depth=depth,
            intrinsics=handle.intrinsics,
            pose_c2w=tuple(tuple(float(value) for value in row) for row in pose),
            score_provenance=handle.score_provenance,
        )

    def store_cold_pixels(self, frame_id: int, handle: ColdPixelHandle) -> bool:
        """Attach CPU/cold pixel ownership to an existing metadata identity."""
        with self._cold_payload_spill_lock:
            return self._store_cold_pixels_locked(frame_id, handle)

    def _store_cold_pixels_locked(self, frame_id: int, handle: ColdPixelHandle) -> bool:
        """Store COLD ownership while serializing cache bookkeeping only."""
        record = self._frames.get(int(frame_id))
        if record is None:
            return False
        if not isinstance(handle, ColdPixelHandle):
            raise TypeError("handle must be a ColdPixelHandle")
        if (
            record.source_identity is not None
            and record.source_identity != handle.source_identity
        ):
            raise ValueError("cold handle source identity mismatch")
        frame_id = int(frame_id)
        payload_bytes = handle.payload_nbytes if handle.has_resident_payload else 0
        if self._cold_payload_disk is not None:
            self._cold_payloads.eviction_ids_for_incoming(
                payload_bytes=payload_bytes
            )
            if not self._cold_payloads.can_put_without_eviction(
                payload_bytes=payload_bytes
            ):
                raise RuntimeError(
                    "cold payload cache capacity was not reserved before recording"
                )
        self._historical_rgb_materializable_generation += 1
        self._cold_payloads.remove(frame_id)
        self._cold_pixels.pop(frame_id, None)

        # A configured prepared cache begins with cache-owned RAM even if
        # ingress supplied a source token.  Only the local spill token may
        # later replace these arrays; legacy users retain token-only behavior.
        if self._cold_payload_disk is not None and handle.has_resident_payload:
            self._cold_payloads.put(
                frame_id, handle, payload_bytes=payload_bytes, evict=False
            )
            return True

        # Deterministically reloadable sources retain only their cheap token;
        # native arrays remain the source owner's concern until page-in.
        if handle.reloadable:
            self._cold_pixels[frame_id] = handle.without_resident_payload()
            return True
        if not handle.has_resident_payload:
            self._cold_pixels[frame_id] = handle
            # A non-reloadable handle with no resident payload can never be
            # materialized again; it must not stay historically eligible.
            self._retire_unmaterializable(frame_id)
            return True

        evicted = self._cold_payloads.put(
            frame_id,
            handle,
            payload_bytes=payload_bytes,
        )
        for item in evicted:
            demoted = item.payload.without_resident_payload()
            self._cold_pixels[item.frame_id] = demoted
            evicted_record = self._frames.get(item.frame_id)
            if evicted_record is not None:
                evicted_record.buffer_stage = BufferStage.ICECOLD_CACHE
            # A live (non-reloadable) source has no deterministic route back to
            # its pixels once the bounded cold store releases them. Retire the
            # identity here, at the exact moment its last route disappears, so
            # no scheduler lane can plan supervision the runtime then cannot
            # honour (issue #1289).
            if not demoted.can_materialize:
                self._retire_unmaterializable(item.frame_id)
        return True

    def _schedule_cold_payload_spills(self, *, incoming_payload_bytes: int = 0) -> bool:
        """Queue the bounded available LRU prefix without I/O or waiting."""
        try:
            frame_ids = self._cold_payloads.eviction_ids_for_incoming(
                payload_bytes=incoming_payload_bytes
            )
        except ValueError:
            return False
        for frame_id in frame_ids:
            if (
                frame_id not in self._pending_cold_payload_spills
                and frame_id not in self._inflight_cold_payload_spills
            ):
                self._pending_cold_payload_spills[frame_id] = None
        return True

    def _prepared_cold_payload_ready_locked(self, payload_bytes: int) -> bool:
        """Reserve bounded spill capacity for one prospective prepared payload."""
        if not self._schedule_cold_payload_spills(
            incoming_payload_bytes=payload_bytes
        ):
            raise ValueError("prepared cold payload cannot fit configured resident bounds")
        return not (
            self._pending_cold_payload_spills
            or self._inflight_cold_payload_spills
            or not self._cold_payloads.can_put_without_eviction(
                payload_bytes=payload_bytes
            )
        )

    @property
    def pending_cold_payload_spill_count(self) -> int:
        with self._cold_payload_spill_lock:
            return len(self._pending_cold_payload_spills) + len(self._inflight_cold_payload_spills)

    def write_pending_cold_payload_spills(
        self, *, max_items: int
    ) -> tuple[ColdPayloadSpillReceipt, ...]:
        """Worker-only write phase; it cannot mutate semantic history."""
        if self._cold_payload_disk is None:
            return ()
        if type(max_items) is not int or max_items < 1:
            raise ValueError("max_items must be a positive integer")
        requests = []
        with self._cold_payload_spill_lock:
            while self._pending_cold_payload_spills and len(requests) < max_items:
                frame_id, _ = self._pending_cold_payload_spills.popitem(last=False)
                handle = self._cold_payloads.get(frame_id, touch=False)
                if handle is None or not handle.has_resident_payload:
                    continue
                self._inflight_cold_payload_spills.add(frame_id)
                requests.append(ColdPayloadSpillRequest(frame_id, handle.source_identity, handle))
        return tuple(self._cold_payload_disk.write(request) for request in requests)

    def commit_cold_payload_spill_receipts(
        self, receipts: tuple[ColdPayloadSpillReceipt, ...]
    ) -> int:
        """Mapper-only publication swaps resident arrays for local-cache tokens."""
        committed = 0
        with self._cold_payload_spill_lock:
            for receipt in receipts:
                if not isinstance(receipt, ColdPayloadSpillReceipt):
                    raise TypeError("cold payload spill receipts must be typed")
                frame_id = int(receipt.frame_id)
                if frame_id not in self._inflight_cold_payload_spills:
                    continue
                self._inflight_cold_payload_spills.remove(frame_id)
                handle = self._cold_payloads.get(frame_id, touch=False)
                if (
                    receipt.error is not None
                    or receipt.token is None
                    or handle is None
                    or handle.source_identity != receipt.source_identity
                ):
                    self._cold_payload_spill_failures.append(receipt)
                    continue
                self._cold_payloads.remove(frame_id)
                self._cold_pixels[frame_id] = ColdPixelHandle(
                    source_identity=handle.source_identity,
                    rgb=None,
                    depth=None,
                    intrinsics=handle.intrinsics,
                    reload_token=receipt.token,
                    rgb_encoding=handle.rgb_encoding,
                    depth_encoding=handle.depth_encoding,
                    depth_units=handle.depth_units,
                )
                record = self._frames.get(frame_id)
                if record is not None:
                    record.buffer_stage = BufferStage.ICECOLD_CACHE
                committed += 1
            self._schedule_cold_payload_spills()
        return committed

    def take_cold_payload_spill_failures(self) -> tuple[ColdPayloadSpillReceipt, ...]:
        """Surface terminal storage faults without faking retirement or retry."""
        with self._cold_payload_spill_lock:
            failures, self._cold_payload_spill_failures = self._cold_payload_spill_failures, []
        return tuple(failures)

    def load_cold_payload_from_token(self, token: object) -> tuple[np.ndarray, np.ndarray | None]:
        """Prefetch-worker-only page-in of the local cache; no ingress fallback."""
        from eligsir.contracts.runtime import PixelReloadToken

        if not isinstance(token, PixelReloadToken) or self._cold_payload_disk is None:
            raise ValueError("prepared cold disk cache is unavailable")
        return self._cold_payload_disk.load(token)

    def can_retain_cold_handle(self, handle: ColdPixelHandle) -> bool:
        """Check a new COLD handle without releasing any existing route."""
        if not isinstance(handle, ColdPixelHandle):
            raise TypeError("handle must be a ColdPixelHandle")
        with self._cold_payload_spill_lock:
            return (
                handle.reloadable
                or not handle.has_resident_payload
                or self._cold_payloads.can_put_without_eviction(
                    payload_bytes=handle.payload_nbytes
                )
            )

    def can_accept_prepared_cold_payload(self, *, payload_bytes: int) -> bool:
        """Whether one packet can enter without bypassing active spill pressure."""
        if type(payload_bytes) is not int or payload_bytes < 0:
            raise ValueError("prepared cold payload bytes must be a non-negative integer")
        if self._cold_payload_disk is None:
            return True
        with self._cold_payload_spill_lock:
            return self._prepared_cold_payload_ready_locked(payload_bytes)

    def add_payload_eviction_observer(
        self, observer: Callable[[int], None]
    ) -> None:
        """Register a callback for identities that lost their last pixel route.

        The DB is authoritative for eligibility, but the scheduler keeps its own
        bounded replay lanes keyed by source identity. Without this notification
        those lanes keep proposing an identity whose pixels are already gone.
        """

        if not callable(observer):
            raise TypeError("payload eviction observer must be callable")
        self._payload_eviction_observers.append(observer)

    @property
    def unmaterializable_evictions(self) -> int:
        """Count of identities retired because their last pixel route vanished."""

        return self._unmaterializable_evictions

    def _retire_unmaterializable(self, frame_id: int) -> None:
        """Announce that one identity lost its last route back to real pixels.

        Provenance is deliberately NOT deleted and ``historical_eligible`` is
        deliberately NOT flipped: ``historical_candidates`` /
        ``historical_candidate_for_source`` already exclude an unmaterializable
        record through ``materializable_only``, and
        ``materializable_only=False`` remains the full historical/provenance
        domain. What was missing is the *notification*: scheduler lanes outside
        this DB keep their own identity views and would otherwise go on
        proposing supervision for pixels that no longer exist (issue #1289).
        """

        frame_id = int(frame_id)
        record = self._frames.get(frame_id)
        if record is None or frame_id in self._unmaterializable_announced:
            return
        self._unmaterializable_announced.add(frame_id)
        self._unmaterializable_evictions += 1
        for observer in tuple(self._payload_eviction_observers):
            observer(frame_id)

    def get_cold_pixels(self, frame_id: int) -> Optional[ColdPixelHandle]:
        """Return the CPU/cold source used for deterministic supervision paging."""
        with self._cold_payload_spill_lock:
            resident = self._cold_payloads.get(frame_id)
            if resident is not None:
                return resident
            return self._cold_pixels.get(frame_id)

    def set_historical_eligibility(
        self,
        frame_id: int,
        *,
        eligible: bool,
        modality_eligibility: Optional[object] = None,
    ) -> bool:
        """Update future-use semantics without changing pixels or HOT residency."""
        record = self._frames.get(int(frame_id))
        if record is None:
            return False
        self._historical_rgb_materializable_generation += 1
        record.historical_eligible = bool(eligible)
        if modality_eligibility is not None:
            record.modality_eligibility = _normalize_modality_eligibility(
                modality_eligibility
            )
        return True

    def historical_candidates(
        self,
        *,
        require_rgb: bool = False,
        require_depth: bool = False,
        require_depth_insertion: bool = False,
        materializable_only: bool = True,
    ) -> Tuple[HistoricalCandidateSnapshot, ...]:
        """Enumerate retained identities without exposing or paging pixels."""
        candidates = []
        for record in self._frames.values():
            modalities = record.modality_eligibility
            if not record.historical_eligible:
                continue
            if require_rgb and not modalities.rgb_supervision:
                continue
            if require_depth and not modalities.depth_supervision:
                continue
            if require_depth_insertion and not modalities.depth_insertion:
                continue
            with self._cold_payload_spill_lock:
                handle = self._cold_payloads.get(record.frame_id, touch=False)
                if handle is None:
                    handle = self._cold_pixels.get(record.frame_id)
                cold_payload_resident = self._cold_payloads.contains(record.frame_id)
            can_materialize = bool(handle is not None and handle.can_materialize)
            if materializable_only and not can_materialize:
                continue
            candidates.append(
                HistoricalCandidateSnapshot(
                    frame_id=int(record.frame_id),
                    source_identity=record.source_identity,
                    source_ordinal=record.source_ordinal,
                    timestamp=float(record.timestamp),
                    submap_id=int(record.submap_id),
                    buffer_stage=record.buffer_stage,
                    modality_eligibility=modalities,
                    can_materialize=can_materialize,
                    cold_payload_resident=cold_payload_resident,
                    reloadable=bool(handle is not None and handle.reloadable),
                    disposition_snapshot_id=record.disposition_snapshot_id,
                    semantic_action=record.semantic_action,
                )
            )
        return tuple(candidates)

    def historical_candidate_for_source(
        self,
        source_identity: object,
        *,
        require_rgb: bool = False,
        require_depth: bool = False,
        require_depth_insertion: bool = False,
        materializable_only: bool = True,
    ) -> Optional[HistoricalCandidateSnapshot]:
        """Resolve one durable historical identity in O(1), without a scan."""

        frame_id = self.frame_id_for_source(source_identity)
        if frame_id is None:
            return None
        record = self._frames.get(frame_id)
        if record is None or not record.historical_eligible:
            return None
        modalities = record.modality_eligibility
        if require_rgb and not modalities.rgb_supervision:
            return None
        if require_depth and not modalities.depth_supervision:
            return None
        if require_depth_insertion and not modalities.depth_insertion:
            return None
        with self._cold_payload_spill_lock:
            handle = self._cold_payloads.get(frame_id, touch=False)
            if handle is None:
                handle = self._cold_pixels.get(frame_id)
            cold_payload_resident = self._cold_payloads.contains(frame_id)
        can_materialize = bool(handle is not None and handle.can_materialize)
        if materializable_only and not can_materialize:
            return None
        return HistoricalCandidateSnapshot(
            frame_id=int(record.frame_id),
            source_identity=record.source_identity,
            source_ordinal=record.source_ordinal,
            timestamp=float(record.timestamp),
            submap_id=int(record.submap_id),
            buffer_stage=record.buffer_stage,
            modality_eligibility=modalities,
            can_materialize=can_materialize,
            cold_payload_resident=cold_payload_resident,
            reloadable=bool(handle is not None and handle.reloadable),
            disposition_snapshot_id=record.disposition_snapshot_id,
            semantic_action=record.semantic_action,
        )

    def activate_supervision(self, frame_id: int, cache: Dict[str, object]) -> bool:
        """Promote one frame into the bounded HOT working set.

        Upload/materialization is intentionally outside this storage owner and
        must be performed by ``DeviceTransferOwner`` before this call.  The
        method only atomically publishes the completed tier handles.

        The published tiers ACCUMULATE onto whatever the frame already holds
        (bounded and value-evicted by ``HotTierCache``) rather than replacing
        them, so an oscillating tier choice re-uses an already-materialized
        representation instead of re-paying resize + staging + H2D every
        switch.  Nothing is materialized speculatively: a tier is retained only
        because a transaction actually asked for it.
        """
        target_device = str(torch.device(self._device).type)
        for tier, handle in cache.items():
            for payload in (getattr(handle, "rgb", handle), getattr(handle, "depth", None)):
                payload_device = _payload_device_type(payload)
                if payload_device is not None and payload_device != target_device:
                    raise ValueError(
                        "active supervision cache tensor device does not match "
                        f"visibility/runtime device: tier={tier!r} "
                        f"payload={payload_device!r} target={target_device!r}"
                    )
        return self.attach_resolution_cache(frame_id, cache)

    def demote_supervision(self, frame_id: int) -> bool:
        """Drop HOT tier ownership while retaining metadata, pixels and visibility."""
        record = self._frames.get(frame_id)
        if record is None or not record.resolution_cache:
            return False
        self._active_supervision_cache.remove(frame_id)
        self._hot_tier_cache.discard(frame_id)
        record.resolution_cache = {}
        record.resolution_cache_evicted = True
        return True

    def residency_snapshot(self, frame_id: int) -> Optional[FrameResidencySnapshot]:
        """Publish independently-readable current residency for one identity."""
        record = self._frames.get(frame_id)
        if record is None:
            return None
        active_tiers = tuple(sorted(str(tier) for tier in record.resolution_cache))
        device_tiers = []
        for tier, handle in record.resolution_cache.items():
            tensors = (getattr(handle, "rgb", handle), getattr(handle, "depth", None))
            if any(_payload_device_type(value) not in {None, "cpu"} for value in tensors):
                device_tiers.append(str(tier))
        with self._cold_payload_spill_lock:
            cold_handle = self._cold_payloads.get(frame_id, touch=False)
            if cold_handle is None:
                cold_handle = self._cold_pixels.get(frame_id)
            cold_payload_resident = self._cold_payloads.contains(frame_id)
        return FrameResidencySnapshot(
            frame_id=int(frame_id),
            cold_pixels_available=bool(
                cold_handle is not None and cold_handle.can_materialize
            ),
            active_tiers=active_tiers,
            device_resident_tiers=tuple(sorted(device_tiers)),
            visibility_device=str(torch.device(self._device).type),
            cold_payload_resident=cold_payload_resident,
            reloadable=bool(cold_handle is not None and cold_handle.reloadable),
        )

    def set_buffer_stage(self, frame_id: int, stage: BufferStage) -> bool:
        """Update a known frame's buffer_stage in place (e.g. a tiered-cache
        transition: active -> cold -> icecold -> active again, or active ->
        icecold via demotion -- see StreamingScene). Does NOT touch
        visible_gaussian_uids/the sparse overlap matrix -- only the record's
        stage label changes; observation history is immutable. No-op
        (returns False) if frame_id was never recorded via add_frame (e.g.
        DB tracking disabled, or the frame predates DB construction)."""
        record = self._frames.get(frame_id)
        if record is None:
            return False
        record.buffer_stage = stage
        return True

    def attach_resolution_cache(
        self, frame_id: int, cache: Dict[str, object], *, replace: bool = False
    ) -> bool:
        """Attach a (re)built ``resolution_cache`` to a KNOWN frame, in place.

        Issue #481(b). ``record_frame_observation`` only builds a
        ``resolution_cache`` for ``is_train`` frames, so a frame first seen as a
        REJECTED candidate is recorded with an empty one
        (``resolution_cache_absent_reason == "not_train_frame"``). When
        ``StreamingScene.promote_from_cache`` later splices that frame back into
        ``train_cameras`` it becomes a real training camera, but nothing ever
        built its cache -- ``bind_resolution_tier`` then fails ``cache_empty``
        for it forever.

        Deliberately a narrow in-place mutator in the ``set_buffer_stage`` idiom
        rather than a second ``add_frame``: ``add_frame`` re-appends the record's
        visible-uid row to the incremental CSR buffer (double-counting that
        frame's observations) and replaces ``visible_gaussian_uids`` wholesale,
        which would DESTROY the real visibility captured at ingest. Only the
        cache field moves here; observation history stays immutable.

        Returns False (no-op) if the frame was never recorded via ``add_frame``,
        or if ``cache`` is empty -- an empty dict must not clear a cache that is
        already resident, nor reset ``resolution_cache_evicted``.

        MULTI-TIER RETENTION: this MERGES ``cache`` into whatever tiers the
        record already holds instead of replacing the dict wholesale.  The
        wholesale replacement was the specific reason a HOT frame could hold
        only one materialized tier: the canonical path publishes exactly
        ``{chosen_tier: handle}`` per transaction, so a quarter -> half ->
        quarter oscillation dropped and re-prepared each tier every time.  Which
        tiers actually survive the merge is decided by ``HotTierCache`` (bounded
        per frame, byte-budgeted across the HOT set, value-evicted) -- never by
        unbounded accumulation here.  ``replace=True`` restores the old
        semantics for a caller that genuinely republishes a frame's whole tier
        set.
        """
        record = self._frames.get(frame_id)
        if record is None or not cache:
            return False
        frame_id = int(frame_id)
        if replace:
            for tier in tuple(record.resolution_cache):
                if tier not in cache:
                    self._hot_tier_cache.discard(frame_id, tier)
            record.resolution_cache = dict(cache)
        else:
            merged = dict(record.resolution_cache)
            merged.update(cache)
            record.resolution_cache = merged
        record.resolution_cache_absent_reason = None
        if record.evicted_resolution_tiers:
            record.evicted_resolution_tiers = tuple(
                tier for tier in record.evicted_resolution_tiers if tier not in cache
            )
        self._register_resolution_cache(frame_id)
        for tier, handle in cache.items():
            for eviction in self._hot_tier_cache.admit(
                frame_id, str(tier), entry_bytes(handle)
            ):
                self._drop_cached_tier(eviction.frame_id, eviction.tier)
        return bool(record.resolution_cache)

    def _drop_cached_tier(self, frame_id: int, tier: str) -> None:
        """Mirror one per-tier retention eviction into the record's storage."""

        record = self._frames.get(int(frame_id))
        if record is None:
            return
        if record.resolution_cache.pop(str(tier), None) is None:
            return
        self.resolution_cache_evictions += 1
        record.evicted_resolution_tiers = tuple(
            sorted(set(record.evicted_resolution_tiers) | {str(tier)})
        )
        if not record.resolution_cache:
            # The frame now holds no tier at all, which is exactly what the
            # pre-existing frame-level flag means. A PARTIAL eviction is not
            # reported through it: `_cache_cause` only reads the flag for an
            # empty cache, and a record still holding another tier must keep
            # reporting `tier_handle_missing`, not `expected_eviction`.
            record.resolution_cache_evicted = True
            self._active_supervision_cache.remove(int(frame_id))

    def rebuild_resolution_cache(
        self,
        frame_id: int,
        source_frame: object,
        required_tier: Optional[str] = None,
    ) -> bool:
        """Rebuild an evicted derived tier through the canonical producer.

        The callback is supplied by the training boundary so this storage
        owner does not import ``train_streaming``. It may return either the
        cache dict or ``(cache, absent_reason)``. Failed reconstruction keeps
        the prior camera/cache state untouched and therefore fails closed.
        """
        record = self._frames.get(frame_id)
        if record is None or self._resolution_cache_rebuilder is None:
            return False
        self.resolution_cache_reconstruction_attempts += 1
        try:
            rebuilt = self._resolution_cache_rebuilder(source_frame, frame_id)
        except Exception:
            self.resolution_cache_reconstruction_failures += 1
            return False
        absent_reason = None
        if isinstance(rebuilt, tuple) and len(rebuilt) == 2:
            rebuilt, absent_reason = rebuilt
        if (
            not isinstance(rebuilt, dict)
            or not rebuilt
            or (required_tier is not None and required_tier not in rebuilt)
        ):
            if absent_reason:
                record.resolution_cache_absent_reason = str(absent_reason)
            self.resolution_cache_reconstruction_failures += 1
            return False
        if not self.attach_resolution_cache(frame_id, rebuilt):
            self.resolution_cache_reconstruction_failures += 1
            return False
        self.resolution_cache_reconstructions += 1
        return True

    def _ensure_csr(self) -> None:
        if not self._dirty and self._csr is not None:
            return
        if self._incremental_valid:
            self._rebuild_csr_from_buffer()
            self._dirty = False
            return
        self._full_rebuild_csr()

    def _rebuild_csr_from_buffer(self) -> None:
        """Wrap the incrementally-maintained buffer into a CSR tensor --
        O(n_rows) for the crow_indices list->tensor conversion, O(1) for
        col/values (contiguous-prefix views, no copy) -- instead of
        _full_rebuild_csr()'s O(total observations) walk+cat+coalesce. See
        the PERF FIX note in __init__."""
        if self._buf_len == 0:
            self._csr = None
            return
        n_rows = self._max_frame_id_seen + 1
        n_cols = self._max_uid_seen + 1
        crow = torch.tensor(self._crow, dtype=torch.int64, device=self._device)
        col = self._buf_col[: self._buf_len]
        val = torch.ones(self._buf_len, dtype=torch.float32, device=self._device)
        self._csr = torch.sparse_csr_tensor(crow, col, val, size=(n_rows, n_cols), device=self._device)

    def _full_rebuild_csr(self) -> None:
        """Fallback path, only reachable if _incremental_valid was disabled
        (out-of-order/duplicate frame_id -- see PERF FIX note in __init__).
        Build COO indices transiently from the per-frame tensors already
        held (forever, by design) in self._frames -- see the BUG FIX note
        in __init__ for why this replaced a permanent duplicate
        self._rows/self._cols Python-list accumulation. frame_ids/counts
        stay small (one entry per frame, not per observation); only the
        concatenated uid tensor scales with total observations, and it is
        freed once this method returns (only the resulting CSR is kept).
        """
        frame_ids: list = []
        uid_chunks: list = []
        counts: list = []
        for record in self._frames.values():
            uids = record.visible_gaussian_uids
            n = uids.numel()
            if n == 0:
                continue
            frame_ids.append(record.frame_id)
            uid_chunks.append(uids)
            counts.append(n)
        if not uid_chunks:
            self._csr = None
            self._dirty = False
            return
        n_rows = self._max_frame_id_seen + 1
        n_cols = self._max_uid_seen + 1
        cols = torch.cat([c.to(dtype=torch.int64) for c in uid_chunks]).to(self._device)
        frame_ids_t = torch.tensor(frame_ids, dtype=torch.int64)
        counts_t = torch.tensor(counts, dtype=torch.int64)
        rows = torch.repeat_interleave(frame_ids_t, counts_t).to(self._device)
        coo = torch.sparse_coo_tensor(
            indices=torch.stack([rows, cols]),
            values=torch.ones(cols.numel(), dtype=torch.float32, device=self._device),
            size=(n_rows, n_cols),
            device=self._device,
        ).coalesce()
        self._csr = coo.to_sparse_csr()
        self._dirty = False

    def overlap_query(self, candidate_uids: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Return (frame_ids, overlap_counts) sorted by descending overlap.

        overlap_counts[i] = number of gaussian_uids shared between
        candidate_uids and frame frame_ids[i]. Empty arrays if the DB has
        no frames yet or none of candidate_uids fall within the indexed
        UID range.
        """
        self._ensure_csr()
        if self._csr is None:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        n_cols = self._csr.shape[1]
        in_range = candidate_uids[candidate_uids < n_cols]
        if in_range.numel() == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        onehot = torch.zeros(n_cols, dtype=torch.float32, device=self._device)
        onehot[in_range] = 1.0
        counts = torch.sparse.mm(self._csr, onehot.unsqueeze(1)).squeeze(1)
        nonzero = counts.nonzero(as_tuple=True)[0]
        if nonzero.numel() == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        vals = counts[nonzero]
        order = torch.argsort(vals, descending=True)
        return nonzero[order].cpu().numpy(), vals[order].to(torch.int64).cpu().numpy()
