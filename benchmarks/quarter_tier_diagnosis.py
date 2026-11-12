"""Diagnostic harness for the reported non-monotonic resolution-tier cost ordering.

Background
----------
A prior RTX 3090 true-camera (C4) kernel benchmark reported
``T_half (2.41 ms) < T_native (3.75 ms) < T_quarter (7.02 ms)`` --- an ordering
that would make raster resolution a non-monotonic cost axis.  This script exists
to test that claim under controlled conditions.  It does three things the
original harness did not:

1. **Dumps the realized per-tier camera intrinsics** (``W, H, fx, fy, cx, cy``)
   so intrinsics-scaling correctness is verifiable from the artifact alone.
2. **Extracts gsplat packed-mode occupancy counters** from the *production*
   render path (``render_true_camera_batch`` -> ``rasterization(..., packed=True)``):
   visible camera/Gaussian pairs (``nnz``), total tile intersections, tile
   counts, per-tile list-length percentiles, projected-radius percentiles, and
   the fraction of projected radii sitting on the ``eps2d`` covariance floor.
   These are *deterministic counts*, not timings, so they are unaffected by
   GPU co-tenancy.
3. **Interleaves and randomizes tier order across repeat blocks**, and runs a
   fixed reference workload at the head of every block, so GPU contention and
   DVFS/ordering artifacts are separable from real per-tier cost.

Timing hygiene
--------------
Every block records the reference workload's own timing.  If the reference
drifts between blocks, the block is contaminated by something outside the
benchmark and its per-tier numbers must not be compared against another
block's.  This is the specific failure mode that produced the original
inverted ordering.

Evidence class: UNMANAGED engineering diagnostic.  Produces no quality claims.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.true_camera_scaling import (  # noqa: E402
    _build_cameras,
    _build_pipe,
    _build_render_model,
    _load_checkpoint_state,
    _zero_model_grads,
    load_camera_manifest,
)

PASS_MODES = ("forward", "forward+backward")


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summarize(samples: Sequence[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    ordered = sorted(samples)
    return {
        "n": len(samples),
        "min_ms": float(ordered[0]),
        "p50_ms": float(statistics.median(samples)),
        "p95_ms": _percentile(samples, 0.95),
        "max_ms": float(ordered[-1]),
        "mean_ms": float(statistics.fmean(samples)),
        "stdev_ms": float(statistics.stdev(samples)) if len(samples) > 1 else 0.0,
        # A run free of external interference is tight around its own floor.
        # `clean_fraction` is the share of samples within 15% of the observed
        # minimum; a value well below 1.0 means the mean is a mixture of a
        # clean mode and a contended mode and must not be quoted.
        "clean_fraction": float(
            sum(1 for value in samples if value < 1.15 * ordered[0]) / len(samples)
        ),
        "clean_mean_ms": float(
            statistics.fmean([v for v in samples if v < 1.15 * ordered[0]])
        ),
    }


def _intrinsics_rows(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        rows.append(
            {
                "img_name": entry.get("img_name"),
                "width": int(entry["width"]),
                "height": int(entry["height"]),
                "fx": float(entry["fx"]),
                "fy": float(entry["fy"]),
                "cx": float(entry["cx"]),
                "cy": float(entry["cy"]),
            }
        )
    return rows


def _check_intrinsics_scaling(
    native: Sequence[dict[str, Any]], tier: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compare a tier's intrinsics against the native tier's expected scaling.

    Two conventions exist for the principal point under a resize by ``s``:
    the pixel-center-correct ``(c + 0.5) * s - 0.5`` and the naive ``c * s``.
    Both are reported; the difference between them is a fraction of a pixel and
    is not a plausible cause of a multi-millisecond cost difference, but a tier
    that failed to scale ``fx``/``fy`` at all would be a cropped/zoomed camera
    and *would* explain one.
    """
    if not native or not tier:
        return {"checked": False, "reason": "missing_manifest"}
    scale_w = float(tier[0]["width"]) / float(native[0]["width"])
    scale_h = float(tier[0]["height"]) / float(native[0]["height"])
    worst_focal = 0.0
    worst_center_exact = 0.0
    worst_center_naive = 0.0
    for base, scaled in zip(native, tier):
        worst_focal = max(
            worst_focal,
            abs(float(scaled["fx"]) - float(base["fx"]) * scale_w),
            abs(float(scaled["fy"]) - float(base["fy"]) * scale_h),
        )
        worst_center_exact = max(
            worst_center_exact,
            abs(float(scaled["cx"]) - ((float(base["cx"]) + 0.5) * scale_w - 0.5)),
            abs(float(scaled["cy"]) - ((float(base["cy"]) + 0.5) * scale_h - 0.5)),
        )
        worst_center_naive = max(
            worst_center_naive,
            abs(float(scaled["cx"]) - float(base["cx"]) * scale_w),
            abs(float(scaled["cy"]) - float(base["cy"]) * scale_h),
        )
    return {
        "checked": True,
        "scale_w": scale_w,
        "scale_h": scale_h,
        "max_focal_abs_error_px": worst_focal,
        "max_center_abs_error_px_exact_convention": worst_center_exact,
        "max_center_abs_error_px_naive_convention": worst_center_naive,
        # Focal length is the load-bearing one: an unscaled fx/fy shrinks the
        # raster without shrinking the projected Gaussian footprints.
        "focal_correctly_scaled": worst_focal < 1e-3,
        "center_convention": (
            "exact"
            if worst_center_exact <= worst_center_naive
            else "naive_c_times_s"
        ),
    }


def _occupancy_counters(meta: dict[str, Any], cameras: Sequence[Any], tile_size: int) -> dict[str, Any]:
    """Derive tile-occupancy / intersection-density evidence from gsplat meta."""
    import torch

    out: dict[str, Any] = {"meta_keys": sorted(str(k) for k in meta.keys())}

    def _get(name: str):
        value = meta.get(name)
        return value if isinstance(value, torch.Tensor) else None

    radii = _get("radii")
    flatten_ids = _get("flatten_ids")
    isect_offsets = _get("isect_offsets")
    gaussian_ids = _get("gaussian_ids")

    if radii is not None:
        # packed mode: [nnz] or [nnz, 2]
        flat_radii = radii.reshape(radii.shape[0], -1).float()
        per_gaussian = flat_radii.max(dim=-1).values
        out["nnz_projected_pairs"] = int(radii.shape[0])
        values = per_gaussian.detach().cpu().tolist()
        positive = [v for v in values if v > 0.0]
        out["n_radii_positive"] = len(positive)
        out["radius_px_p50"] = _percentile(positive, 0.50)
        out["radius_px_p95"] = _percentile(positive, 0.95)
        out["radius_px_max"] = max(positive) if positive else None
        # gsplat adds eps2d to the 2D covariance diagonal, so the smallest
        # achievable projected sigma is sqrt(eps2d) and the smallest radius is
        # ~3*sqrt(eps2d).  That floor is resolution-INDEPENDENT: it does not
        # shrink when the raster shrinks, so at low resolution a growing share
        # of Gaussians sit on it and stop getting cheaper.
        for threshold in (1.0, 2.0, 3.0, 4.0):
            out[f"radius_frac_le_{threshold:g}px"] = (
                float(sum(1 for v in positive if v <= threshold) / len(positive))
                if positive
                else None
            )
    if gaussian_ids is not None:
        out["nnz_gaussian_ids"] = int(gaussian_ids.shape[0])

    n_tiles_total = 0
    for camera in cameras:
        width = int(getattr(camera, "image_width"))
        height = int(getattr(camera, "image_height"))
        n_tiles_total += math.ceil(width / tile_size) * math.ceil(height / tile_size)
    out["tile_size"] = int(tile_size)
    out["n_tiles_total"] = int(n_tiles_total)

    if flatten_ids is not None:
        n_isect = int(flatten_ids.shape[0])
        out["n_intersections"] = n_isect
        out["isects_per_tile"] = float(n_isect / n_tiles_total) if n_tiles_total else None
        if radii is not None and out.get("nnz_projected_pairs"):
            out["tiles_touched_per_visible_gaussian"] = float(
                n_isect / out["nnz_projected_pairs"]
            )

    if isect_offsets is not None and flatten_ids is not None:
        offsets = isect_offsets.reshape(-1).long()
        total = int(flatten_ids.shape[0])
        boundaries = torch.cat(
            [offsets, torch.tensor([total], dtype=offsets.dtype, device=offsets.device)]
        )
        per_tile = (boundaries[1:] - boundaries[:-1]).clamp(min=0).float()
        counts = per_tile.detach().cpu().tolist()
        out["n_tiles_from_offsets"] = len(counts)
        out["tile_list_len_p50"] = _percentile(counts, 0.50)
        out["tile_list_len_p95"] = _percentile(counts, 0.95)
        out["tile_list_len_max"] = max(counts) if counts else None
        out["tile_list_len_mean"] = float(statistics.fmean(counts)) if counts else None
        out["empty_tile_fraction"] = (
            float(sum(1 for c in counts if c == 0) / len(counts)) if counts else None
        )
        # Load imbalance across tiles is what turns a small raster into a
        # latency problem: the rasterizer's block per tile runs as long as its
        # own list, so p95/p50 is a direct proxy for tail-dominated occupancy.
        p50 = out["tile_list_len_p50"] or 0.0
        out["tile_list_imbalance_p95_over_p50"] = (
            float((out["tile_list_len_p95"] or 0.0) / p50) if p50 > 0 else None
        )
    return out


def build_native_manifest_from_rgbd_source(
    source: Path, frame_ids: Sequence[int]
) -> list[dict[str, Any]]:
    """Build a ``camera_to_JSON``-schema manifest from a generic RGBDSequence.

    Reads ``intrinsics.json`` (native, unscaled) and the requested rows of
    ``frames.jsonl`` (whose ``c2w`` is a 4x4 camera-to-world matrix).  The
    manifest schema wants the C2W rotation and the camera centre, which are
    exactly the rotation block and translation column of ``c2w`` --- see the
    convention note in ``render_report._camera_json_to_minicam``.
    """
    intrinsics = json.loads((source / "intrinsics.json").read_text())
    wanted = {int(i) for i in frame_ids}
    rows: dict[int, dict[str, Any]] = {}
    with (source / "frames.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            index = int(record["id"])
            if index in wanted:
                rows[index] = record
                if len(rows) == len(wanted):
                    break
    missing = sorted(wanted - set(rows))
    if missing:
        raise SystemExit(f"frames.jsonl is missing requested ids: {missing}")

    manifest: list[dict[str, Any]] = []
    for position, index in enumerate(sorted(wanted)):
        c2w = rows[index]["c2w"]
        manifest.append(
            {
                "id": position,
                "img_name": f"{index:06d}",
                "width": int(intrinsics["width"]),
                "height": int(intrinsics["height"]),
                "fx": float(intrinsics["fx"]),
                "fy": float(intrinsics["fy"]),
                "cx": float(intrinsics["cx"]),
                "cy": float(intrinsics["cy"]),
                "rotation": [[float(v) for v in c2w[r][:3]] for r in range(3)],
                "position": [float(c2w[r][3]) for r in range(3)],
            }
        )
    return manifest


def derive_tier_manifest(
    native: Sequence[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    """Downscale a native manifest to ``width x height``.

    Uses the SAME convention as production
    (``eligsir/mapping/resolution_materialization.py``): focal lengths scale
    linearly, and the principal point uses the pixel-centre-exact form
    ``c' = (c + 0.5) * s - 0.5``.  At very low resolutions the difference from
    the naive ``c * s`` is a larger *fraction* of the raster, so getting this
    right matters more the further down the ladder one goes.
    """
    derived: list[dict[str, Any]] = []
    for entry in native:
        scale_x = width / float(entry["width"])
        scale_y = height / float(entry["height"])
        row = dict(entry)
        row["width"] = int(width)
        row["height"] = int(height)
        row["fx"] = float(entry["fx"]) * scale_x
        row["fy"] = float(entry["fy"]) * scale_y
        row["cx"] = (float(entry["cx"]) + 0.5) * scale_x - 0.5
        row["cy"] = (float(entry["cy"]) + 0.5) * scale_y - 0.5
        derived.append(row)
    return derived


class GPUTenancyMonitor:
    """Samples GPU utilization and compute-process count for the whole run.

    The RTX 3090 result this script exists to diagnose was corrupted by a
    co-tenant that was present the entire time but invisible to a
    before-launch-only check.  Sampling *throughout* is the only way to assert
    exclusivity for the measurement window rather than for one instant.
    """

    def __init__(self, interval_s: float = 2.0) -> None:
        self._interval = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self.backend, self.reason = self._detect()

    @staticmethod
    def _detect() -> tuple[str | None, str | None]:
        if shutil.which("nvidia-smi"):
            return "nvidia-smi", None
        if shutil.which("rocm-smi"):
            return "rocm-smi", None
        return None, "neither nvidia-smi nor rocm-smi is available in this container"

    def _sample(self) -> dict[str, Any] | None:
        try:
            if self.backend == "nvidia-smi":
                util = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                procs = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                ).stdout.strip().splitlines()
                return {
                    "t": time.time(),
                    "utilization_percent": int(util[0]) if util and util[0].isdigit() else None,
                    "compute_process_count": len([p for p in procs if p.strip()]),
                }
            raw = subprocess.run(
                ["rocm-smi", "--showuse"], capture_output=True, text=True, timeout=15
            ).stdout
            percent = None
            for line in raw.splitlines():
                if "use (%)" in line.lower():
                    tail = line.rsplit(":", 1)[-1].strip()
                    if tail.isdigit():
                        percent = int(tail)
            return {"t": time.time(), "utilization_percent": percent, "compute_process_count": None}
        except Exception as exc:  # pragma: no cover - best-effort telemetry
            return {"t": time.time(), "error": f"{type(exc).__name__}: {exc}"}

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            sample = self._sample()
            if sample is not None:
                self.samples.append(sample)

    def start(self) -> None:
        if self.backend is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.backend is None:
            return {"available": False, "reason": self.reason}
        counts = [
            s["compute_process_count"]
            for s in self.samples
            if isinstance(s.get("compute_process_count"), int)
        ]
        utils = [
            s["utilization_percent"]
            for s in self.samples
            if isinstance(s.get("utilization_percent"), int)
        ]
        return {
            "available": True,
            "backend": self.backend,
            "n_samples": len(self.samples),
            "utilization_percent_min": min(utils) if utils else None,
            "utilization_percent_mean": (statistics.fmean(utils) if utils else None),
            "utilization_percent_max": max(utils) if utils else None,
            "compute_process_count_min": min(counts) if counts else None,
            "compute_process_count_max": max(counts) if counts else None,
            # This process itself is one compute app, so anything above 1 at any
            # point during the window means a foreign tenant shared the device.
            "exclusive_for_whole_window": (max(counts) <= 1) if counts else None,
            "exclusivity_reason": (
                None if counts else "compute-process enumeration unavailable on this backend"
            ),
            "samples": self.samples,
        }


def _invoke(cameras, model, pipe, background, *, render_mode: str, pass_mode: str):
    from eligsir.mapping.renderer.true_camera_batch import render_true_camera_batch
    from eligsir.mapping.renderer.gsplat_backend import render

    render_depth = render_mode == "RGB+ED"
    if len(cameras) == 1:
        result = render(
            cameras[0], model, pipe, background,
            update_sh_rest=True, render_depth=render_depth,
        )
        tensors = [result["render"]]
        if render_depth:
            tensors.append(result["rendered_depth_ed"])
        meta = result.get("meta", {})
    else:
        result = render_true_camera_batch(
            cameras, model, pipe, background,
            update_sh_rest=True, render_depth=render_depth,
        )
        tensors = [result.renders]
        if render_depth and result.rendered_depth_ed is not None:
            tensors.append(result.rendered_depth_ed)
        meta = result.meta
    if pass_mode == "forward+backward":
        loss = sum(tensor.float().mean() for tensor in tensors)
        loss.backward()
    return meta


def _time_block(
    cameras, model, pipe, background, *, render_mode: str, pass_mode: str,
    warmups: int, repetitions: int,
) -> list[float]:
    import torch

    def once() -> None:
        _zero_model_grads(model)
        if pass_mode == "forward":
            with torch.no_grad():
                _invoke(cameras, model, pipe, background,
                        render_mode=render_mode, pass_mode=pass_mode)
        else:
            _invoke(cameras, model, pipe, background,
                    render_mode=render_mode, pass_mode=pass_mode)

    for _ in range(warmups):
        once()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        once()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tier",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Tier label and its camera manifest, e.g. native=/path/cameras_native.json",
    )
    parser.add_argument(
        "--rgbd-source",
        type=Path,
        help="Generic RGBDSequence directory (frames.jsonl + intrinsics.json) to build "
             "the NATIVE camera manifest from, instead of passing --tier native=...",
    )
    parser.add_argument(
        "--frame-ids",
        help="Comma-separated frame ids from --rgbd-source to use as the camera batch. "
             "Defaults to --cameras frames spread evenly across the trajectory.",
    )
    parser.add_argument(
        "--derive-tier",
        action="append",
        default=[],
        metavar="LABEL=WxH",
        help="Derive a tier from the native manifest by downscaling to WxH using the "
             "production intrinsics convention, e.g. eighth=240x180. Repeatable.",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=2.0,
        help="Seconds between GPU tenancy samples taken for the whole run (0 disables).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--render-mode", default="RGB+ED")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--radius-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pass-modes",
        default=",".join(PASS_MODES),
        help="Comma-separated subset of: forward, forward+backward",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import torch

    args = build_parser().parse_args(argv)
    rng = random.Random(args.seed)

    tiers: dict[str, Path] = {}
    for item in args.tier:
        label, _, raw = item.partition("=")
        if not label or not raw:
            raise SystemExit(f"--tier expects LABEL=PATH, got {item!r}")
        tiers[label] = Path(raw)

    derived_specs: list[tuple[str, int, int]] = []
    for item in args.derive_tier:
        label, _, raw = item.partition("=")
        width_raw, _, height_raw = raw.lower().partition("x")
        if not label or not width_raw.isdigit() or not height_raw.isdigit():
            raise SystemExit(f"--derive-tier expects LABEL=WxH, got {item!r}")
        derived_specs.append((label, int(width_raw), int(height_raw)))

    if not tiers and args.rgbd_source is None:
        raise SystemExit("pass --tier LABEL=PATH and/or --rgbd-source")

    pass_modes = tuple(m.strip() for m in args.pass_modes.split(",") if m.strip())
    for mode in pass_modes:
        if mode not in PASS_MODES:
            raise SystemExit(f"unsupported pass mode {mode!r}")

    state, checkpoint_provenance = _load_checkpoint_state(args.checkpoint)
    mature_n = int(state["params"]["means"].shape[0])
    indices = torch.arange(mature_n, dtype=torch.long)

    pipe = _build_pipe()
    pipe.render_mode = args.render_mode
    pipe.radius_clip = float(args.radius_clip)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    properties = torch.cuda.get_device_properties(0)
    payload: dict[str, Any] = {
        "schema_version": "eligsir.quarter_tier_diagnosis.v1",
        "evidence_class": "UNMANAGED",
        "evidence_scope": "engineering_kernel_diagnosis",
        "quality_claims_permitted": False,
        "checkpoint": {**checkpoint_provenance, "mature_n": mature_n},
        "config": {
            "cameras": int(args.cameras),
            "blocks": int(args.blocks),
            "repetitions": int(args.repetitions),
            "warmups": int(args.warmups),
            "render_mode": args.render_mode,
            "sh_degree": int(args.sh_degree),
            "radius_clip": float(args.radius_clip),
            "seed": int(args.seed),
            "pass_modes": list(pass_modes),
            "mode": "packed+sparse",
        },
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "multiprocessor_count": int(properties.multi_processor_count),
            "total_memory_bytes": int(properties.total_memory),
        },
        "software": {
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "intrinsics": {},
        "occupancy": {},
        "timing_blocks": [],
        "timing_summary": {},
    }

    try:
        import gsplat

        payload["software"]["gsplat"] = getattr(gsplat, "__version__", None)
    except Exception:  # pragma: no cover - provenance only
        payload["software"]["gsplat"] = None

    # ---- camera + model construction -------------------------------------
    manifests: dict[str, list[dict[str, Any]]] = {}
    built: dict[str, Any] = {}
    model = _build_render_model(state, indices, sh_degree=args.sh_degree)

    if args.rgbd_source is not None:
        if args.frame_ids:
            frame_ids = [int(v) for v in args.frame_ids.split(",") if v.strip()]
        else:
            total = sum(1 for _ in (args.rgbd_source / "frames.jsonl").open())
            step = max(1, total // args.cameras)
            frame_ids = [min(total - 1, i * step) for i in range(args.cameras)]
        manifests["native"] = build_native_manifest_from_rgbd_source(
            args.rgbd_source, frame_ids
        )
        payload["config"]["rgbd_source"] = str(args.rgbd_source)
        payload["config"]["frame_ids"] = frame_ids

    for label, path in tiers.items():
        entries, _ = load_camera_manifest(path)
        manifests[label] = list(entries[: args.cameras])

    if derived_specs:
        if "native" not in manifests:
            raise SystemExit("--derive-tier requires a native manifest (--rgbd-source or --tier native=...)")
        for label, width, height in derived_specs:
            manifests[label] = derive_tier_manifest(manifests["native"], width, height)

    for label, subset in manifests.items():
        payload["intrinsics"][label] = _intrinsics_rows(subset)
        built[label] = _build_cameras(subset)

    native = manifests.get("native")
    if native is not None:
        payload["intrinsics_scaling_check"] = {
            label: _check_intrinsics_scaling(native, rows)
            for label, rows in manifests.items()
        }

    monitor = GPUTenancyMonitor(args.gpu_monitor_interval)
    if args.gpu_monitor_interval > 0:
        monitor.start()

    # ---- Phase A: deterministic occupancy counters ------------------------
    tile_size = int(getattr(pipe, "tile_size", 16))
    for label, cameras in built.items():
        _zero_model_grads(model)
        with torch.no_grad():
            meta = _invoke(
                cameras, model, pipe, background,
                render_mode=args.render_mode, pass_mode="forward",
            )
        payload["occupancy"][label] = _occupancy_counters(meta, cameras, tile_size)
    torch.cuda.synchronize()

    # ---- Phase B: randomized, interleaved timing blocks -------------------
    for block in range(args.blocks):
        order = list(built.keys())
        rng.shuffle(order)
        for pass_mode in pass_modes:
            for label in order:
                samples = _time_block(
                    built[label], model, pipe, background,
                    render_mode=args.render_mode, pass_mode=pass_mode,
                    warmups=args.warmups, repetitions=args.repetitions,
                )
                payload["timing_blocks"].append(
                    {
                        "block": block,
                        "tier": label,
                        "pass_mode": pass_mode,
                        "order_in_block": order.index(label),
                        "tier_order": list(order),
                        "samples_ms": samples,
                        **_summarize(samples),
                    }
                )

    payload["gpu_tenancy"] = monitor.stop() if args.gpu_monitor_interval > 0 else {
        "available": False, "reason": "monitoring disabled (--gpu-monitor-interval 0)"
    }

    for pass_mode in pass_modes:
        for label in built:
            rows = [
                r
                for r in payload["timing_blocks"]
                if r["tier"] == label and r["pass_mode"] == pass_mode
            ]
            payload["timing_summary"][f"{label}|{pass_mode}"] = {
                "per_block_clean_mean_ms": [r["clean_mean_ms"] for r in rows],
                "per_block_min_ms": [r["min_ms"] for r in rows],
                "per_block_clean_fraction": [r["clean_fraction"] for r in rows],
                "pooled": _summarize([v for r in rows for v in r["samples_ms"]]),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"[quarter-tier-diagnosis] wrote {args.output}")
    tenancy = payload["gpu_tenancy"]
    if tenancy.get("available"):
        print(
            f"  GPU tenancy: {tenancy['n_samples']} samples, "
            f"compute processes max={tenancy['compute_process_count_max']}, "
            f"exclusive_for_whole_window={tenancy['exclusive_for_whole_window']}"
        )
    else:
        print(f"  GPU tenancy: UNVERIFIED ({tenancy.get('reason')})")
    for key, value in sorted(payload["timing_summary"].items()):
        print(
            f"  {key:34s} per-block clean mean ms = "
            + ", ".join(f"{v:.3f}" for v in value["per_block_clean_mean_ms"])
            + "  clean_frac = "
            + ", ".join(f"{v:.2f}" for v in value["per_block_clean_fraction"])
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
