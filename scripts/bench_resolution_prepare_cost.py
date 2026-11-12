"""Measure T_plan_to_ready: the cost of preparing a supervision resolution tier.

Standalone diagnostic (dev-only, UNMANAGED -- not a ResearchFlow experiment).
It imports the REAL production functions and drives them on real decoded
frames.  It adds no timing hooks to production code and starts no training run.

The measured path is `production_application._execute_transaction_work`
(eligsir/runtime/production_application.py, the `materialize_resolution_tier`
call site):

    observation_db.resident_supervision_for_source(source, tier)
      hit  -> _bind_resident_supervision              [condition A: HOT]
      miss -> _restore_camera_payload_from_cold(...)  [condition C only]
              camera.materialize_pending_image()
              materialize_resolution_tier(camera, tier)   # GPU F.interpolate
                                                          #  RGB area / depth nearest
    transfer_owner.prefetch_cameras(cameras)          # pinned slab + batched H2D
      -> PinnedTensorPool.acquire  -> host[i].copy_(cam.<attr>)  (SYNCHRONOUS loop)
      -> DeviceTransferStream.h2d  -> non_blocking copy + cuda.Event
    prepare_camera_for_render(camera)                 # compute stream waits
    observation_db.activate_supervision(frame_id, {tier: handle})

Usage (inside the container):
    python scripts/bench_resolution_prepare_cost.py \
        --source /data/fr1_xyz --repeats 30 --out bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from eligsir.contracts.control import ResolutionTier  # noqa: E402
from eligsir.io.sources.decode_cache import (  # noqa: E402
    StreamingRGBDFrame,
    load_frame_rgb,
    load_frame_depth_np,
)
from eligsir.mapping.resolution_materialization import (  # noqa: E402
    materialize_resolution_tier,
)
from eligsir.runtime.transfers import (  # noqa: E402
    DeviceTransferOwner,
    DeviceTransferStream,
)


# ---------------------------------------------------------------- timing utils
def _sync(device: str) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """Accumulate named wall-clock spans (ms)."""

    def __init__(self, device: str):
        self.device = device
        self.spans: Dict[str, float] = {}
        self._name = ""
        self._sync_on_exit = False
        self._t0 = 0.0

    def __call__(self, name: str, sync: bool = False):
        self._name = name
        self._sync_on_exit = sync
        return self

    def __enter__(self):
        if self._sync_on_exit:
            _sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._sync_on_exit:
            _sync(self.device)
        self.spans[self._name] = self.spans.get(self._name, 0.0) + (
            time.perf_counter() - self._t0
        ) * 1e3
        return False


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "p50": round(statistics.median(s), 4),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 4),
        "mean": round(statistics.fmean(s), 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
    }


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for r in rows for k in r})
    return {k: _stats([r[k] for r in rows if k in r]) for k in keys}


# ---------------------------------------------------------------- frame loading
def load_tum_frames(source: str, count: int) -> List[StreamingRGBDFrame]:
    """Minimal TUM rgb.txt/depth.txt reader -- avoids constructing the whole
    streaming scene just to obtain a handful of real, on-disk frames."""

    def _read(path):
        out = []
        with open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    out.append((float(parts[0]), parts[1]))
        return out

    rgb = _read(os.path.join(source, "rgb.txt"))
    depth = _read(os.path.join(source, "depth.txt"))
    frames: List[StreamingRGBDFrame] = []
    di = 0
    for i, (ts, rp) in enumerate(rgb):
        while di + 1 < len(depth) and abs(depth[di + 1][0] - ts) < abs(depth[di][0] - ts):
            di += 1
        if abs(depth[di][0] - ts) > 0.02:
            continue
        frames.append(
            StreamingRGBDFrame(
                index=i, timestamp=ts,
                rgb_path=os.path.join(source, rp),
                depth_path=os.path.join(source, depth[di][1]),
                fx=517.3, fy=516.5, cx=318.6, cy=255.3,
                width=640, height=480, depth_scale=5000.0,
            )
        )
        if len(frames) >= count:
            break
    return frames


def decode(frame: StreamingRGBDFrame):
    """Full reload: PIL/PNG decode -> numpy, both streams (condition C's
    `reload_ms`, i.e. `_restore_camera_payload_from_cold` +
    `materialize_pending_image` in production terms)."""
    rgb = np.asarray(load_frame_rgb(frame))
    depth = load_frame_depth_np(frame)
    return rgb, depth


# ---------------------------------------------------------------- camera stub
class BenchCamera:
    """Duck-typed camera carrying exactly the attributes the measured
    production functions read/write (`materialize_resolution_tier`,
    `DeviceTransferOwner.prefetch`, `prepare_camera_for_render`)."""

    def __init__(self, rgb_chw: torch.Tensor, depth_hw: Optional[torch.Tensor],
                 fx, fy, cx, cy):
        self.original_image = rgb_chw
        self._sensor_depth_cache = depth_hw
        self.image_height = self.height = int(rgb_chw.shape[1])
        self.image_width = self.width = int(rgb_chw.shape[2])
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.score_provenance = None
        self.gsplat = None
        self.has_pending_image = False
        # prepare_camera_for_render / make_gsplat_camera_tensors inputs
        self.world_view_transform = torch.eye(4)
        self.full_proj_transform = torch.eye(4)
        self.camera_center = torch.zeros(3)
        self.FoVx = self.FoVy = 1.0
        self.znear, self.zfar = 0.01, 100.0


def make_cpu_native_camera(rgb_np, depth_np, frame) -> BenchCamera:
    """CPU-resident native camera in training format (CHW float32 RGB in
    [0,1], HW float32 depth) -- what `materialize_pending_image` yields."""
    rgb = (torch.from_numpy(np.ascontiguousarray(rgb_np))
           .permute(2, 0, 1).float().div_(255.0).contiguous())
    depth = None
    if depth_np is not None:
        depth = torch.from_numpy(
            np.ascontiguousarray(depth_np.astype(np.float32) / frame.depth_scale)
        ).contiguous()
    return BenchCamera(rgb, depth, frame.fx, frame.fy, frame.cx, frame.cy)


def to_device(cam: BenchCamera, device: str) -> BenchCamera:
    cam.original_image = cam.original_image.to(device).contiguous()
    if cam._sensor_depth_cache is not None:
        cam._sensor_depth_cache = cam._sensor_depth_cache.to(device).contiguous()
    return cam


# ---------------------------------------------------------------- measurement
def measure_prepare(cams: List[BenchCamera], tier: ResolutionTier, owner, device: str,
                    timer: Timer, do_materialize: bool = True) -> None:
    """One `_execute_transaction_work` preparation step over a C-camera batch."""
    if do_materialize:
        with timer("resize_ms", sync=True):
            for cam in cams:
                materialize_resolution_tier(cam, tier)
    else:
        timer.spans["resize_ms"] = 0.0

    if owner is None:  # CPU device: no staging/H2D exists
        timer.spans["prefetch_ms"] = 0.0
        timer.spans["h2d_wait_ms"] = 0.0
        return

    # transfers.py `DeviceTransferOwner.prefetch_cameras` ->
    # `_CameraBatchPrefetcher.prefetch_and_bind`: bucket by shape/dtype,
    # acquire a pinned slot, run the SYNCHRONOUS `host[index].copy_(...)`
    # staging loop, issue the batched async h2d, hand the event to compute
    # and rebind the camera tensors. Timed as one span (`prefetch_ms`); the
    # `prefetch_internals` block below attributes it to stage/issue/wait.
    with timer("prefetch_ms", sync=False):
        owner.prefetch_cameras(cams)
    with timer("h2d_wait_ms", sync=True):
        pass


def stage_vs_submit_split(cams: List[BenchCamera], owner, device: str) -> Dict[str, float]:
    """Attribute `prefetch`'s internals: how much of it is the synchronous
    host-side staging copy vs. issuing the async H2D. Measured by timing the
    same host[i].copy_ loop the production code runs, against the same pinned
    pool, immediately before a real prefetch."""
    payloads = [c.original_image for c in cams if c.original_image.device.type == "cpu"]
    if not payloads:
        return {}
    shape = (len(payloads),) + tuple(payloads[0].shape)
    key = (tuple(payloads[0].shape), payloads[0].dtype)
    pool = owner._camera_prefetcher._pools.get(key)
    if pool is None:
        return {}
    slot, host = pool.acquire(shape, payloads[0].dtype)
    t0 = time.perf_counter()
    for i, p in enumerate(payloads):
        host[i].copy_(p)
    stage_ms = (time.perf_counter() - t0) * 1e3
    _sync(device)
    t1 = time.perf_counter()
    dev, event = owner._transfer.h2d(host)
    submit_ms = (time.perf_counter() - t1) * 1e3
    t2 = time.perf_counter()
    event.wait()
    _sync(device)
    wait_ms = (time.perf_counter() - t2) * 1e3
    pool.release_now(slot)
    return {
        "stage_loop_ms": round(stage_ms, 4),
        "h2d_issue_ms": round(submit_ms, 4),
        "h2d_event_wait_ms": round(wait_ms, 4),
    }


# ---------------------------------------------------------------- main
def run(args) -> Dict:
    device = args.device
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        print("[bench] no accelerator -> CPU-only timings (resize axis only)")
        device = "cpu"
    accel = torch.device(device).type != "cpu"
    if accel and torch.device(device).index is None:
        # DeviceTransferOwner.prefetch compares `payload.device ==
        # transfer.device` to skip already-resident tensors; a bare "cuda"
        # (index None) never equals "cuda:0" and every resident payload would
        # be misrouted into a staging copy. Production resolves the index.
        device = f"{torch.device(device).type}:{torch.cuda.current_device()}"

    frames = load_tum_frames(args.source, max(args.frames, max(args.batch)))
    if not frames:
        raise SystemExit(f"no frames under {args.source}")
    pool_raw = [(f,) + decode(f) for f in frames]

    if args.synthetic_hw:
        # Second resolution point (e.g. ScanNet++ iPhone 1920x1440) WITHOUT
        # reading the NFS-backed dataset -- a different storage backend would
        # confound every timing here (CLAUDE.md: "never compare speed across
        # storage backends"). Only shape/dtype drive resize/stage/H2D cost, so
        # nearest-upsampling real frames is faithful for those axes. It is NOT
        # a stand-in for decode cost, so `cold` conditions are skipped.
        import cv2 as _cv2
        sh, sw = args.synthetic_hw
        pool_raw = [
            (f,
             _cv2.resize(r, (sw, sh), interpolation=_cv2.INTER_NEAREST),
             _cv2.resize(d, (sw, sh), interpolation=_cv2.INTER_NEAREST))
            for (f, r, d) in pool_raw
        ]
        print(f"[bench] SYNTHETIC upsample to {sw}x{sh}")

    nh, nw = pool_raw[0][1].shape[:2]
    print(f"[bench] {len(pool_raw)} frames, native {nw}x{nh}, device={device}")

    owner = None
    if accel:
        owner = DeviceTransferOwner(
            device, max_pinned_bytes=int(args.max_pinned_mib) * 1024 * 1024
        )

    tiers = [ResolutionTier(t) for t in args.tiers]
    results: Dict = {
        "meta": {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if accel else None,
            "torch": torch.__version__,
            "source": args.source,
            "native_hw": [nh, nw],
            "synthetic_hw": args.synthetic_hw,
            "tiers": args.tiers,
            "batch_sizes": args.batch,
            "repeats": args.repeats,
            "max_pinned_mib": args.max_pinned_mib,
        },
        "payload_bytes": {},
        "conditions": {},
        "prefetch_internals": {},
    }

    # ----- measured payload accounting (per frame, per tier) -----
    for tier in tiers:
        R = tier.downsample_factor
        h, w = max(1, nh // R), max(1, nw // R)
        rgb_b = h * w * 3 * 4          # CHW float32 training format
        d_b = h * w * 4                # HW float32 metres
        results["payload_bytes"][tier.value] = {
            "hw": [h, w],
            "rgb_float32_bytes": rgb_b,
            "depth_float32_bytes": d_b,
            "rgbd_total_bytes": rgb_b + d_b,
            "rgbd_total_mib": round((rgb_b + d_b) / 2**20, 4),
            "rgb_only_mib": round(rgb_b / 2**20, 4),
        }

    conditions = ("A_hot_exact_hit", "B_cpu_resident_tier_miss",
                  "C_cold_reload_miss", "D_gpu_native_resident_tier_miss")
    if args.synthetic_hw:
        conditions = ("A_hot_exact_hit", "B_cpu_resident_tier_miss",
                      "D_gpu_native_resident_tier_miss")

    order = [(t, C, m, cond)
             for t in tiers for C in args.batch
             for m in ("rgbd", "rgb_only") for cond in conditions]

    rows: Dict[str, List[Dict[str, float]]] = {}
    rng = random.Random(20260829)

    for rep in range(args.repeats + args.warmup):
        rng.shuffle(order)
        record = rep >= args.warmup
        for (tier, C, mode, cond) in order:
            sel = [pool_raw[(rep + i) % len(pool_raw)] for i in range(C)]
            timer = Timer(device)

            if cond == "C_cold_reload_miss":
                with timer("reload_ms"):
                    sel = [(f,) + decode(f) for (f, _, _) in sel]
            else:
                timer.spans["reload_ms"] = 0.0

            with timer("host_format_ms"):
                cams = [make_cpu_native_camera(r, (None if mode == "rgb_only" else d), f)
                        for (f, r, d) in sel]

            if cond == "A_hot_exact_hit":
                # Already GPU-resident AND already bound to this exact tier:
                # `resident_supervision_for_source` hit -> `_bind_resident_
                # supervision`. Modelled by the idempotent re-call, which is
                # `materialize_resolution_tier`'s own early-return fast path.
                for cam in cams:
                    to_device(cam, device)
                    materialize_resolution_tier(cam, tier)
                _sync(device)
                measure_prepare(cams, tier, owner, device, timer)
                timer.spans["host_format_ms"] = 0.0   # not paid on a hit
            elif cond == "D_gpu_native_resident_tier_miss":
                # The camera's NATIVE payload is already GPU-resident but bound
                # to a different tier: `materialize_resolution_tier` re-derives
                # from `_canonical_native_resolution_state` with a GPU
                # F.interpolate, and `prefetch_cameras` is then a no-op (already
                # on-device). This is what a retained-native multi-tier cache
                # would actually cost -- not a production state today, because
                # the cold path materializes on CPU before the upload.
                for cam in cams:
                    to_device(cam, device)
                _sync(device)
                measure_prepare(cams, tier, owner, device, timer)
                timer.spans["host_format_ms"] = 0.0
            elif cond == "B_cpu_resident_tier_miss":
                measure_prepare(cams, tier, owner, device, timer)
            else:
                measure_prepare(cams, tier, owner, device, timer)

            timer.spans["total_prepare_ms"] = sum(
                v for k, v in timer.spans.items() if k != "reload_ms"
            )
            timer.spans["total_plan_to_ready_ms"] = (
                timer.spans["total_prepare_ms"] + timer.spans["reload_ms"]
            )
            # per-camera normalisation, so C=1/4/8 are directly comparable
            timer.spans["per_camera_prepare_ms"] = timer.spans["total_prepare_ms"] / C
            if record:
                rows.setdefault(f"{cond}/{tier.value}/C{C}/{mode}", []).append(timer.spans)

    # ----- attribute prefetch's internals (stage loop vs h2d issue vs wait) -----
    if owner is not None:
        for tier in tiers:
            for C in args.batch:
                acc: List[Dict[str, float]] = []
                for rep in range(args.repeats):
                    sel = [pool_raw[(rep + i) % len(pool_raw)] for i in range(C)]
                    cams = [make_cpu_native_camera(r, d, f) for (f, r, d) in sel]
                    for cam in cams:
                        materialize_resolution_tier(cam, tier)
                    owner.prefetch_cameras(cams)  # ensure the pool/bucket exists
                    cams2 = [make_cpu_native_camera(r, d, f) for (f, r, d) in sel]
                    for cam in cams2:
                        materialize_resolution_tier(cam, tier)
                    s = stage_vs_submit_split(cams2, owner, device)
                    if s:
                        acc.append(s)
                if acc:
                    results["prefetch_internals"][f"{tier.value}/C{C}"] = _aggregate(acc)

    for k, v in rows.items():
        results["conditions"][k] = _aggregate(v)
    if owner is not None:
        results["transfer_stats"] = owner.stats.snapshot()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--tiers", nargs="+", default=["native", "half", "quarter"])
    ap.add_argument("--synthetic-hw", dest="synthetic_hw", type=int, nargs=2,
                    default=None, metavar=("H", "W"))
    ap.add_argument("--max-pinned-mib", dest="max_pinned_mib", type=int,
                    default=2048,
                    help="DeviceTransferOwner pinned budget; matches the "
                         "current production default of 2048 MiB / 2GiB "
                         "(--streaming_max_transfer_bytes, arguments/__init__.py; "
                         "raised from a stale 256 MiB by issue #1259 -- see "
                         "issue #1254, where this flag's old 256 MiB default "
                         "hard-errored on a batch the real production budget "
                         "already stages fine). Lower to measure a batch a "
                         "SMALLER budget -- e.g. a profile override such as "
                         "configs/profiles/live/canonical_mapping.yml's 640 "
                         "MiB -- cannot actually stage.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args)
    text = json.dumps(res, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"[bench] wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
