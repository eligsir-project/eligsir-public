"""Optional coarse-to-fine depth sampling schedule for map insertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DepthSparsityScheduleConfig:
    mode: str = "fixed"
    start_stride: int = 16
    end_stride: int = 8
    ramp_frames: int = 500

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "coarse_to_fine"}:
            raise ValueError("mode must be 'fixed' or 'coarse_to_fine'")
        if self.start_stride < 1 or self.end_stride < 1 or self.ramp_frames < 1:
            raise ValueError("schedule strides and ramp_frames must be positive")


def config_from_values(
    mode: str,
    base_stride: int,
    start_stride: int,
    end_stride_raw: int,
    ramp_frames: int,
) -> Optional[DepthSparsityScheduleConfig]:
    if mode == "fixed":
        return None
    end_stride = end_stride_raw if end_stride_raw > 0 else base_stride
    return DepthSparsityScheduleConfig(
        mode=str(mode),
        start_stride=int(start_stride),
        end_stride=int(end_stride),
        ramp_frames=int(ramp_frames),
    )


def config_from_args(args) -> Optional[DepthSparsityScheduleConfig]:
    return config_from_values(
        str(getattr(args, "streaming_depth_sparsity_schedule", "fixed")),
        int(getattr(args, "streaming_depth_stride", 8)),
        int(getattr(args, "streaming_depth_sparsity_start_stride", 16)),
        int(getattr(args, "streaming_depth_sparsity_end_stride", 0)),
        int(getattr(args, "streaming_depth_sparsity_ramp_frames", 500)),
    )


def config_from_policy(policy, base_stride: int) -> Optional[DepthSparsityScheduleConfig]:
    return config_from_values(
        str(getattr(policy, "depth_sparsity_schedule_mode", "fixed")),
        int(base_stride),
        int(getattr(policy, "depth_sparsity_start_stride", 16)),
        int(getattr(policy, "depth_sparsity_end_stride", 0)),
        int(getattr(policy, "depth_sparsity_ramp_frames", 500)),
    )


def resolve_effective_depth_stride(
    config: Optional[DepthSparsityScheduleConfig],
    frames_ingested: int,
    base_stride: int,
) -> int:
    if config is None or config.mode == "fixed":
        return int(base_stride)
    fraction = min(1.0, max(0, int(frames_ingested)) / float(config.ramp_frames))
    stride = config.start_stride * (
        (config.end_stride / config.start_stride) ** fraction
    )
    return max(1, int(round(stride)))
