"""Residency-first payload budget resolver (pure, probe-injectable).

Computes frame and byte bounds from runtime demand (source resolution, keyframe window,
batch size) and available supply (device memory, GTT, host RAM) without wiring into
the runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class ResidencySupply:
    """Probed memory supply snapshot."""
    device_free_bytes: int | None
    device_total_bytes: int | None
    gtt_total_bytes: int | None
    gtt_used_bytes: int | None
    host_available_bytes: int | None


@dataclass(frozen=True)
class ResidencyDemand:
    """Payload budget demand derived from the recipe."""
    native_width: int
    native_height: int
    tier_divisor: int
    window_frames: int
    batch_size: int
    tiers_per_frame: int = 2


@dataclass(frozen=True)
class ResidencyBudget:
    """Resolved budget with all intermediate values for logging."""
    hot_tier_max_bytes: int
    cold_payload_max_bytes: int
    cold_payload_max_frames: int
    admitted_payload_max_frames: int
    resident_frames: int
    window_effective: int
    prefetch_entries: int
    starved: bool
    metadata: dict


def read_meminfo_available(path: str = "/proc/meminfo") -> int | None:
    """Read MemAvailable from /proc/meminfo in bytes.

    Args:
        path: Path to meminfo file (default /proc/meminfo).

    Returns:
        MemAvailable in bytes, or None if not found or unreadable.
    """
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # Format: "MemAvailable:   12345 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            kb = int(parts[1])
                            return kb * 1024
                        except (ValueError, IndexError):
                            pass
        return None
    except (OSError, IOError):
        return None


def probe_supply() -> ResidencySupply:
    """Probe real device and host memory supply.

    Returns:
        ResidencySupply with all probes wrapped; returns None on failure.
    """
    device_free = None
    device_total = None
    gtt_total = None
    gtt_used = None
    host_available = None

    # Probe torch device memory
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            device_free = int(free_bytes)
            device_total = int(total_bytes)
    except Exception:
        pass

    # Probe AMD GTT
    try:
        from eligsir.utils.gpu_memory_telemetry import read_amd_drm_gtt_bytes
        gtt_result = read_amd_drm_gtt_bytes()
        if gtt_result is not None:
            gtt_used, gtt_total = gtt_result
            gtt_used = int(gtt_used)
            gtt_total = int(gtt_total)
    except Exception:
        pass

    # Probe host memory
    try:
        host_available = read_meminfo_available()
    except Exception:
        pass

    return ResidencySupply(
        device_free_bytes=device_free,
        device_total_bytes=device_total,
        gtt_total_bytes=gtt_total,
        gtt_used_bytes=gtt_used,
        host_available_bytes=host_available,
    )


def resolve_residency_budget(
    demand: ResidencyDemand,
    supply: ResidencySupply,
    *,
    device_headroom_frac: float = 0.5,
    host_headroom_frac: float = 0.25,
    hot_share: float = 0.4,
    cold_share: float = 0.5,
) -> ResidencyBudget:
    """Resolve residency budget from demand and supply.

    Args:
        demand: ResidencyDemand with resolution, tier, window, batch size.
        supply: ResidencySupply with probed memory values.
        device_headroom_frac: Fraction of device free to reserve (default 0.5).
        host_headroom_frac: Fraction of host available to reserve (default 0.25).
        hot_share: Fraction of device supply for hot tier (default 0.4).
        cold_share: Fraction of host supply for cold tier (default 0.5).

    Returns:
        ResidencyBudget with all resolved values and metadata.
    """
    # Compute native and tier bytes
    native_bytes = demand.native_width * demand.native_height * 16
    tier_bytes = native_bytes // (demand.tier_divisor ** 2)

    # Compute demand
    demand_hot = demand.window_frames * tier_bytes * demand.tiers_per_frame
    demand_cold = demand.window_frames * native_bytes
    demand_pool = demand.window_frames

    metadata = {
        "native_width": demand.native_width,
        "native_height": demand.native_height,
        "native_bytes": native_bytes,
        "tier_divisor": demand.tier_divisor,
        "tier_bytes": tier_bytes,
        "window_frames": demand.window_frames,
        "batch_size": demand.batch_size,
        "tiers_per_frame": demand.tiers_per_frame,
        "demand_hot": demand_hot,
        "demand_cold": demand_cold,
        "demand_pool": demand_pool,
    }

    # Compute device supply with GTT logic
    device_supply = 0
    if supply.device_free_bytes is not None:
        device_supply = supply.device_free_bytes
    if supply.gtt_total_bytes is not None and supply.gtt_used_bytes is not None:
        gtt_available = supply.gtt_total_bytes - supply.gtt_used_bytes
        if device_supply > 0:
            device_supply = min(device_supply, gtt_available)
        else:
            device_supply = gtt_available

    device_supply_after_headroom = int(device_supply * (1 - device_headroom_frac))

    # Compute host supply
    host_supply = 0
    if supply.host_available_bytes is not None:
        host_supply = supply.host_available_bytes
    host_supply_after_headroom = int(host_supply * (1 - host_headroom_frac))

    metadata["device_free"] = supply.device_free_bytes
    metadata["device_total"] = supply.device_total_bytes
    metadata["gtt_used"] = supply.gtt_used_bytes
    metadata["gtt_total"] = supply.gtt_total_bytes
    metadata["host_available"] = supply.host_available_bytes
    metadata["device_supply"] = device_supply_after_headroom
    metadata["host_supply"] = host_supply_after_headroom

    # Handle unknown supplies
    if device_supply == 0:
        metadata["device_supply"] = "unknown"
        device_supply_after_headroom_for_resident = None
    else:
        device_supply_after_headroom_for_resident = device_supply_after_headroom

    if host_supply == 0:
        metadata["host_supply"] = "unknown"
        host_supply_after_headroom_for_resident = None
    else:
        host_supply_after_headroom_for_resident = host_supply_after_headroom

    # Resolve hot and cold bytes
    if device_supply_after_headroom_for_resident is not None:
        hot_bytes = min(demand_hot, int(device_supply_after_headroom_for_resident * hot_share))
    else:
        hot_bytes = demand_hot  # Unbounded

    if host_supply_after_headroom_for_resident is not None:
        cold_bytes = min(demand_cold, int(host_supply_after_headroom_for_resident * cold_share))
    else:
        cold_bytes = demand_cold  # Unbounded

    metadata["hot_bytes"] = hot_bytes
    metadata["cold_bytes"] = cold_bytes

    # Compute frames
    cold_frames = cold_bytes // native_bytes
    pool_frames = max(demand_pool, cold_frames)

    metadata["cold_frames"] = cold_frames
    metadata["pool_frames"] = pool_frames

    # Compute resident frames
    if device_supply_after_headroom_for_resident is not None and host_supply_after_headroom_for_resident is not None:
        # Both supplies known: compute normally
        resident = min(
            hot_bytes // (tier_bytes * demand.tiers_per_frame),
            cold_frames,
        )
    elif device_supply_after_headroom_for_resident is not None:
        # Only device known: use device constraint
        resident = hot_bytes // (tier_bytes * demand.tiers_per_frame)
    elif host_supply_after_headroom_for_resident is not None:
        # Only host known: use host constraint
        resident = cold_frames
    else:
        # Neither known: unbounded
        resident = demand.window_frames

    metadata["resident_frames"] = resident

    # Compute effective window and starvation
    starved = resident < demand.window_frames
    if starved:
        window_effective = max(8, resident)
    else:
        window_effective = demand.window_frames

    metadata["starved"] = starved
    metadata["window_effective"] = window_effective

    # Compute prefetch entries
    prefetch_entries = max(8, 2 * demand.batch_size)

    metadata["prefetch_entries"] = prefetch_entries

    return ResidencyBudget(
        hot_tier_max_bytes=hot_bytes,
        cold_payload_max_bytes=cold_bytes,
        cold_payload_max_frames=cold_frames,
        admitted_payload_max_frames=pool_frames,
        resident_frames=resident,
        window_effective=window_effective,
        prefetch_entries=prefetch_entries,
        starved=starved,
        metadata=metadata,
    )


def format_budget_line(budget: ResidencyBudget) -> str:
    """Format a single log line with budget resolution.

    Format: [payload-budget] policy=residency native=WxH tier=half window=200
    demand_hot=...MiB demand_cold=...MiB device_free=.. host_avail=.. gtt=..
    -> hot=..MiB cold=..frames pool=..frames resident=.. window_eff=..

    Args:
        budget: Resolved ResidencyBudget.

    Returns:
        Single formatted log line.
    """
    meta = budget.metadata
    native_res = f"{meta['native_width']}x{meta['native_height']}"
    tier_name = {1: "native", 2: "half", 4: "quarter", 8: "eighth"}.get(
        meta["tier_divisor"], f"tier_div_{meta['tier_divisor']}"
    )
    window = meta["window_frames"]
    demand_hot_mib = meta["demand_hot"] / (1024 * 1024)
    demand_cold_mib = meta["demand_cold"] / (1024 * 1024)

    # Format supply
    device_free_str = (
        f"{meta['device_free'] // (1024*1024)}" if isinstance(meta["device_free"], int) else "unknown"
    )
    host_avail_str = (
        f"{meta['host_available'] // (1024*1024)}" if isinstance(meta["host_available"], int) else "unknown"
    )
    gtt_str = (
        f"total={meta['gtt_total'] // (1024*1024)},used={meta['gtt_used'] // (1024*1024)}"
        if isinstance(meta["gtt_total"], int) and isinstance(meta["gtt_used"], int)
        else "N/A"
    )

    # Format resolution
    hot_mib = meta["hot_bytes"] / (1024 * 1024)
    cold_frames = meta["cold_frames"]
    pool_frames = meta["pool_frames"]
    resident = meta["resident_frames"]
    window_eff = meta["window_effective"]

    parts = [
        "[payload-budget] policy=residency",
        f"native={native_res}",
        f"tier={tier_name}",
        f"window={window}",
        f"demand_hot={demand_hot_mib:.1f}MiB",
        f"demand_cold={demand_cold_mib:.1f}MiB",
        f"device_free={device_free_str}MiB",
        f"host_avail={host_avail_str}MiB",
        f"gtt={gtt_str}",
        "->",
        f"hot={hot_mib:.1f}MiB",
        f"cold={cold_frames}frames",
        f"pool={pool_frames}frames",
        f"resident={resident}",
        f"window_eff={window_eff}",
    ]

    line = " ".join(parts)
    if budget.starved:
        line += " STARVATION"

    return line


# --- Decode-worker sizing (Task 3: --causal_decode_workers_policy) ---
#
# Calibration points for the linear per-frame decode-time-vs-bytes model used
# by ``estimate_decode_ms_per_frame``. Both are WARM (cache-hit) measurements
# from docs/superpowers/specs/2026-09-08-residency-budget-design.md,
# "Addendum: decode-worker starvation":
#   - TUM (640x480, 640*480*5 = 1,536,000 "bytes"): ~7ms warm decode.
#   - ScanNet (1296x968, 1296*968*5 = 6,272,640 "bytes"): ~19ms warm decode,
#     itself the sum of color (7ms) + depth (4ms) + register (8ms) measured in
#     this investigation's own local decode test. The cold-cache figures
#     (180ms / 11ms split) are explicitly out of scope for this estimate.
_DECODE_CALIBRATION_TUM_BYTES = 640 * 480 * 5
_DECODE_CALIBRATION_TUM_MS = 7.0
_DECODE_CALIBRATION_SCANNET_BYTES = 1296 * 968 * 5
_DECODE_CALIBRATION_SCANNET_MS = 19.0


def estimate_decode_ms_per_frame(native_width: int, native_height: int) -> float:
    """Estimate per-frame RGB-D decode cost in milliseconds (derived, not measured).

    Fits a linear model ``ms = a + b * bytes`` (``bytes = width*height*5``, a
    rough per-pixel proxy for color+depth+register work) through the two warm
    calibration points documented above, then evaluates it at the given
    resolution. This is a startup-time estimate only -- no benchmarking is
    performed live.

    Args:
        native_width: Source native frame width in pixels.
        native_height: Source native frame height in pixels.

    Returns:
        Estimated warm decode time in milliseconds for one frame.
    """
    bytes_estimate = native_width * native_height * 5
    slope = (_DECODE_CALIBRATION_SCANNET_MS - _DECODE_CALIBRATION_TUM_MS) / (
        _DECODE_CALIBRATION_SCANNET_BYTES - _DECODE_CALIBRATION_TUM_BYTES
    )
    intercept = _DECODE_CALIBRATION_TUM_MS - slope * _DECODE_CALIBRATION_TUM_BYTES
    return intercept + slope * bytes_estimate


def resolve_decode_workers(
    *,
    policy: str,
    fixed_workers: int,
    cpu_count: int | None,
    source_hz: float,
    per_frame_decode_ms_estimate: float,
) -> tuple[int, dict]:
    """Resolve the decode ThreadPoolExecutor worker count.

    ``policy == "fixed"`` returns ``fixed_workers`` unchanged -- this is the
    pre-existing, byte-identical default behavior. ``policy == "cpu_scaled"``
    derives a worker count from how many decode-workers are needed to keep up
    with the pacing clock at the estimated per-frame decode cost, bounded
    below by ``fixed_workers`` (never fewer than the explicit/default floor)
    and above by a CPU-headroom cap (never more than half the available
    cores, minimum 2).

    Args:
        policy: "fixed" or "cpu_scaled".
        fixed_workers: args.causal_decode_workers; always the "fixed" answer
            and the floor under "cpu_scaled".
        cpu_count: os.cpu_count() or an injected value; None falls back to 4.
        source_hz: Pacing rate (frames/sec) the producer must keep up with.
        per_frame_decode_ms_estimate: Estimated decode cost per frame in ms
            (see ``estimate_decode_ms_per_frame``).

    Returns:
        Tuple of (resolved worker count, metadata dict for logging).
    """
    if policy == "fixed":
        return fixed_workers, {"policy": "fixed", "source": "explicit_or_default"}

    frame_budget_ms = 1000.0 / max(source_hz, 1e-6)
    needed = max(1, math.ceil(per_frame_decode_ms_estimate / frame_budget_ms))
    cap = max(2, (cpu_count or 4) // 2)
    workers = max(fixed_workers, min(needed, cap))

    metadata = {
        "policy": "cpu_scaled",
        "fixed_workers": fixed_workers,
        "cpu_count": cpu_count,
        "source_hz": source_hz,
        "per_frame_decode_ms_estimate": per_frame_decode_ms_estimate,
        "frame_budget_ms": frame_budget_ms,
        "needed": needed,
        "cap": cap,
        "workers": workers,
    }
    return workers, metadata
