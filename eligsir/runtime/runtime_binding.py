"""Torch-free boundary between composition and an accelerator runtime."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from eligsir.contracts.materialize import DepthPolicy, MaterializationPolicy, numpy_backend
from eligsir.runtime.config import RuntimeBinding


def _default_cold_payload_spill_root(args: Any) -> str:
    """Choose one run-local backing root without assuming model output is local."""
    from eligsir.runtime.output_staging import (
        _is_tmpfs_mount,
        is_network_filesystem,
        resolve_staging_root,
    )

    model_path = getattr(args, "model_path", None)
    candidate = (
        Path(model_path) / "cold_payloads"
        if isinstance(model_path, str) and model_path
        else None
    )
    if (
        candidate is not None
        and os.access(candidate.parent, os.W_OK)
        and not is_network_filesystem(str(candidate))
        and not _is_tmpfs_mount(str(candidate))
    ):
        return str(candidate)
    root = resolve_staging_root(getattr(args, "output_staging_root", "") or None)
    label = "mapping" if candidate is None else candidate.parent.name
    return str(Path(root) / f"cold-payloads-{label}-{uuid.uuid4().hex}")


def runtime_binding_from_runtime(runtime: Any, *, args: Any = None) -> RuntimeBinding:
    """Expose every actuator owned by one already-constructed runtime.

    ``args`` (optional, the parsed CLI/config args) is consulted only for the
    ``--streaming_gpu_native_depth`` opt-in (issue #1348, docs/architecture/
    2026-08-31-gpu-native-decode-and-compressed-ingress.md, priority 1).
    Without the opt-in, the binding explicitly exposes the canonical CPU
    NumPy backend.  This is behaviorally identical to the historical session
    fallback, but is now observable by out-of-band production consumers
    (notably holdout-camera construction) so they cannot silently choose a
    different materialization implementation.

    The historical-retirement wrapper is installed here, after runtime
    construction, so FrameObservationDB's already-registered physical-eviction
    callback keeps its original semantics while later dynamic RGB-supervision
    withdrawals can be modality-aware.
    """
    from eligsir.runtime.historical_retirement import (
        install_modality_aware_historical_retirement,
    )

    install_modality_aware_historical_retirement(runtime)

    configure_spill = getattr(runtime, "configure_prepared_cold_payload_spill", None)
    if args is not None and callable(configure_spill):
        configure_spill(_default_cold_payload_spill_root(args))

    materialization_backend = numpy_backend()
    if bool(getattr(args, "streaming_gpu_native_depth", False)):
        from eligsir.contracts.materialize import torch_gpu_depth_backend

        materialization_backend = torch_gpu_depth_backend(device="cuda")
    return RuntimeBinding(
        mapper_sink=runtime.map_frame,
        optimizer_tick=runtime.tick,
        source_complete_hook=runtime.mark_source_complete,
        flush_hook=runtime.flush,
        close_hook=runtime.close,
        observation_owner=runtime.observation_owner,
        control_snapshot_provider=runtime.control_snapshot,
        disposition_sink=runtime.record_disposition,
        packet_capture=getattr(runtime, "capture_packet_for_history", None),
        terminal_packet_sink=getattr(runtime, "cache_terminal_packet", None),
        source_consumed_observer=getattr(runtime, "record_source_consumed", None),
        materialization_policy=MaterializationPolicy(depth=DepthPolicy.METERS_FLOAT32),
        materialization_backend=materialization_backend,
    )
