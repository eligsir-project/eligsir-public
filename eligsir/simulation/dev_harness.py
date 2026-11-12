"""Fast deterministic simulation for implementation qualification.

This harness intentionally does **not** emulate reconstruction quality. It drives
real transport-free EliGSiR contracts through ``MappingSession`` and the
reference driver cadence using tiny synthetic RGB-D/pose observations. The goal
is to prove software semantics cheaply and deterministically without reserving a
GPU.

The harness is valid development evidence for ordering, materialization,
tracking rejection, bounded drain cadence, policy skips, zero-frame ticks, and
basic ownership/device contracts. It is not scientific evidence and does not
replace device qualification for device-dependent claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from eligsir.contracts.materialize import (
    ColorPolicy,
    DepthPolicy,
    MaterializationPolicy,
    numpy_backend,
    torch_backend,
)
from eligsir.contracts.runtime import BufferRef, SensorPacket, SourceIdentity
from eligsir.contracts.types import CameraIntrinsics
from eligsir.runtime.driver import (
    DEFAULT_DRAIN_CAP,
    CadenceViolation,
    IterationTick,
    check_cadence_invariant,
    drain_tick,
    total_mapped,
)
from eligsir.runtime.session import MappingSession
from eligsir.tracking.providers import ExternalTrackingProvider, observation_from_pose_matrix

BackendName = Literal["numpy", "torch"]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuration for one deterministic development simulation."""

    frame_count: int = 24
    width: int = 32
    height: int = 24
    drain_cap: int = DEFAULT_DRAIN_CAP
    skip_every: int = 5
    untracked_every: int = 7
    backend: BackendName = "numpy"
    zero_frame_tail_ticks: int = 2

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width/height must be positive")
        if self.drain_cap <= 0:
            raise ValueError("drain_cap must be positive")
        if self.skip_every < 0 or self.untracked_every < 0:
            raise ValueError("skip_every/untracked_every must be >= 0")
        if self.zero_frame_tail_ticks <= 0:
            raise ValueError("zero_frame_tail_ticks must be positive")
        if self.backend not in {"numpy", "torch"}:
            raise ValueError(f"unsupported backend {self.backend!r}")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Compact machine-readable result of one simulation run."""

    schema_version: int
    backend: str
    source_frames: int
    policy_skips: int
    submitted: int
    mapped: int
    rejected: int
    tick_count: int
    zero_frame_ticks: int
    mapped_ordinals: tuple[int, ...]
    mapped_devices: tuple[str, ...]
    event_actions: tuple[str, ...]
    rgb_dtypes: tuple[str, ...]
    depth_dtypes: tuple[str, ...]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _Cursor:
    packets: list[SensorPacket]
    index: int = 0
    policy_skips: int = 0

    def ready(self) -> bool:
        return self.index < len(self.packets)

    def next(self, *, skip_every: int) -> SensorPacket | None:
        packet = self.packets[self.index]
        self.index += 1
        if skip_every > 0 and packet.source.source_ordinal > 0:
            if packet.source.source_ordinal % skip_every == 0:
                self.policy_skips += 1
                return None
        return packet


def _packet(config: SimulationConfig, ordinal: int) -> SensorPacket:
    source = SourceIdentity(
        source_id="sim-camera",
        stream_epoch=0,
        source_ordinal=ordinal,
        timestamp_ns=1_000_000_000 + ordinal * 33_333_333,
        pair_id=f"sim-{ordinal:06d}",
    )
    rgb = np.empty((config.height, config.width, 3), dtype=np.uint8)
    rgb[..., 0] = ordinal % 251
    rgb[..., 1] = (ordinal * 3) % 251
    rgb[..., 2] = (ordinal * 7) % 251
    depth = np.full(
        (config.height, config.width),
        1.0 + ordinal * 0.01,
        dtype=np.float32,
    )
    intrinsics = CameraIntrinsics(
        fx=float(config.width),
        fy=float(config.width),
        cx=(config.width - 1) / 2.0,
        cy=(config.height - 1) / 2.0,
        width=config.width,
        height=config.height,
    )

    tracking = None
    should_drop_tracking = (
        config.untracked_every > 0
        and ordinal > 0
        and ordinal % config.untracked_every == 0
    )
    if not should_drop_tracking:
        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 3] = ordinal * 0.01
        tracking = observation_from_pose_matrix(source, c2w, provider="simulation")

    return SensorPacket(
        source=source,
        rgb=BufferRef.from_numpy(rgb, encoding="rgb8"),
        depth=BufferRef.from_numpy(depth, encoding="32FC1", units="m"),
        intrinsics=intrinsics,
        tracking=tracking,
    )


def _dtype_name(payload: Any) -> str:
    dtype = getattr(payload, "dtype", None)
    return str(dtype) if dtype is not None else type(payload).__name__


def _backend(name: BackendName):
    if name == "numpy":
        return numpy_backend()
    import torch

    return torch_backend(torch)


def run_dev_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    """Run a tiny deterministic transport->session->materialization trace.

    The simulation deliberately includes driver-level skips and untracked
    packets. It then appends zero-frame ticks so the caller-owned optimizer
    cadence/refinement-tail contract is exercised as well.
    """

    config = config or SimulationConfig()
    packets = [_packet(config, ordinal) for ordinal in range(config.frame_count)]
    cursor = _Cursor(packets)

    mapped_ordinals: list[int] = []
    mapped_devices: list[str] = []
    rgb_dtypes: list[str] = []
    depth_dtypes: list[str] = []

    def mapper(frame, observation) -> None:
        assert observation.source == frame.source
        mapped_ordinals.append(frame.source.source_ordinal)
        mapped_devices.append(frame.device)
        rgb_dtypes.append(_dtype_name(frame.rgb.data))
        if frame.depth is not None:
            depth_dtypes.append(_dtype_name(frame.depth.data))

    policy = MaterializationPolicy(
        device="cpu",
        color=ColorPolicy.NORMALIZED_FLOAT32,
        depth=DepthPolicy.METERS_FLOAT32,
    )
    session = MappingSession(
        ExternalTrackingProvider("simulation"),
        mapper=mapper,
        materialization_policy=policy,
        materialization_backend=_backend(config.backend),
    )

    ticks: list[IterationTick] = []
    iteration = 0
    while cursor.ready():
        ticks.append(
            drain_tick(
                session,
                iteration=iteration,
                has_ready_frame=cursor.ready,
                next_packet=lambda: cursor.next(skip_every=config.skip_every),
                drain_cap=config.drain_cap,
            )
        )
        iteration += 1

    for _ in range(config.zero_frame_tail_ticks):
        ticks.append(
            drain_tick(
                session,
                iteration=iteration,
                has_ready_frame=cursor.ready,
                next_packet=lambda: cursor.next(skip_every=config.skip_every),
                drain_cap=config.drain_cap,
            )
        )
        iteration += 1

    try:
        check_cadence_invariant(
            ticks,
            drain_cap=config.drain_cap,
            expect_zero_frame_ticks=True,
        )
    except CadenceViolation as exc:
        raise RuntimeError("development simulation violated driver cadence") from exc
    session.flush()
    session.close()

    actions = tuple(event.action.value for event in session.events())
    tick_mapped = total_mapped(ticks)
    if tick_mapped != session.mapped_count:
        raise RuntimeError(
            "development simulation mapped-count mismatch: "
            f"ticks={tick_mapped}, session={session.mapped_count}"
        )
    zero_ticks = sum(tick.submitted == 0 for tick in ticks)
    payload = {
        "backend": config.backend,
        "source_frames": config.frame_count,
        "policy_skips": cursor.policy_skips,
        "submitted": session.submitted_count,
        "mapped": tick_mapped,
        "rejected": session.rejected_count,
        "ticks": [
            {
                "iteration": tick.iteration,
                "submitted": tick.submitted,
                "mapped": tick.mapped,
                "skipped": tick.skipped_by_policy,
                "deferred": tick.deferred,
                "sequences": list(tick.sequences),
            }
            for tick in ticks
        ],
        "mapped_ordinals": mapped_ordinals,
        "actions": actions,
        "devices": mapped_devices,
        "rgb_dtypes": rgb_dtypes,
        "depth_dtypes": depth_dtypes,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return SimulationResult(
        schema_version=1,
        backend=config.backend,
        source_frames=config.frame_count,
        policy_skips=cursor.policy_skips,
        submitted=session.submitted_count,
        mapped=tick_mapped,
        rejected=session.rejected_count,
        tick_count=len(ticks),
        zero_frame_ticks=zero_ticks,
        mapped_ordinals=tuple(mapped_ordinals),
        mapped_devices=tuple(mapped_devices),
        event_actions=actions,
        rgb_dtypes=tuple(rgb_dtypes),
        depth_dtypes=tuple(depth_dtypes),
        signature=signature,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--drain-cap", type=int, default=2)
    parser.add_argument("--skip-every", type=int, default=5)
    parser.add_argument("--untracked-every", type=int, default=7)
    args = parser.parse_args(argv)

    result = run_dev_simulation(
        SimulationConfig(
            frame_count=args.frames,
            width=args.width,
            height=args.height,
            drain_cap=args.drain_cap,
            skip_every=args.skip_every,
            untracked_every=args.untracked_every,
            backend=args.backend,
        )
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
