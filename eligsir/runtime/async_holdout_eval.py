"""Async, snapshot-safe mid-mapping holdout evaluation (issue #1314).

``--streaming_mapping_holdout_curve`` renders the frozen causal holdout while
mapping is live so ``partition_psnr_over_time.svg`` has real mapping-phase
samples, not only a single post-source-completion endpoint. Doing that render
synchronously on the mapping thread (the historical/default behaviour, still
used when ``--streaming_mapping_holdout_curve_async`` is off) pauses the
optimizer loop for the render+score duration, which is why that flag is
documented as invalidating throughput evidence for the whole run.

This module implements the async alternative:

1. At the eval-cadence tick, clone the rendering-relevant Gaussian parameter
   tensors (means/scales/quats/opacities/sh) onto a dedicated CUDA/HIP
   snapshot stream, ordered with ``snapshot_stream.wait_stream(current)`` so
   the clone only reads state the default stream has already produced.
2. Record an event on the snapshot stream once the clone kernels are queued.
3. Render + score the holdout off that snapshot, entirely on a background
   thread (reusing :class:`~eligsir.runtime.save_worker.AsyncSaveWorker`'s
   queue+daemon-thread plumbing) -- never touching the live model.
4. Close the REVERSE-direction gap the progress-video mechanism in
   ``train.py`` (``_pv_stream``) does not have: the copy only orders
   "clone happens after everything already queued", it does nothing to stop
   the *next* in-place mutation (optimizer step / MCMC relocation / append)
   from writing into the same parameter storage while the clone kernels are
   still in flight on the snapshot stream. :class:`MutationBarrier` closes
   this by recording the copy-done event and making the caller's own default
   stream wait on it immediately before that next mutation runs --
   ``wait_for_pending_copy()`` is called from the single choke point every
   in-place mutation passes through in the causal-mapping execution path
   (``binding.optimizer_tick``, wrapped in
   :mod:`eligsir.runtime.causal_entry`).

The only cost this imposes on the mapping thread is the wall-clock time to
*enqueue* the clone (a handful of kernel launches) plus, once per tick, the
already-cheap event wait before the next mutation -- both measured and
reported, never assumed.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Sequence

import torch

from eligsir.runtime.save_worker import AsyncSaveWorker

# The exact rendering-relevant raw (pre-activation, in-place-mutated) leaf
# tensors shared by both model layouts (`GsplatGaussianModel`,
# `GaussianModel`) -- see scene/gsplat_model.py and scene/gaussian_model.py,
# both of which expose these identical attribute names with identical
# activation formulas (exp / normalize / sigmoid).
_SNAPSHOT_RAW_ATTRS: tuple[str, ...] = (
    "_xyz",
    "_scaling",
    "_rotation",
    "_opacity",
    "_features_dc",
    "_features_rest",
)


class GaussianRenderSnapshot:
    """Frozen, render-only clone of one model's rendering-relevant params.

    Exposes exactly the attribute surface ``gaussian_renderer`` reads from a
    live model (``get_xyz``, ``get_scaling``, ``get_rotation``,
    ``get_opacity``, ``get_features``, ``_rotation``, ``_features_dc``,
    ``_features_rest``, ``active_sh_degree``, ``max_sh_degree``) so the
    renderer needs no knowledge that this is not the live model. Never
    mutated after construction -- every property recomputes its activation
    from tensors captured once at snapshot time.
    """

    __slots__ = (
        "_xyz_t",
        "_scaling_raw",
        "_rotation_raw",
        "_opacity_raw",
        "_features_dc_t",
        "_features_rest_t",
        "active_sh_degree",
        "max_sh_degree",
    )

    def __init__(
        self,
        *,
        xyz: torch.Tensor,
        scaling_raw: torch.Tensor,
        rotation_raw: torch.Tensor,
        opacity_raw: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        active_sh_degree: int,
        max_sh_degree: int,
    ) -> None:
        self._xyz_t = xyz
        self._scaling_raw = scaling_raw
        self._rotation_raw = rotation_raw
        self._opacity_raw = opacity_raw
        self._features_dc_t = features_dc
        self._features_rest_t = features_rest
        self.active_sh_degree = int(active_sh_degree)
        self.max_sh_degree = int(max_sh_degree)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz_t

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling_raw)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation_raw, dim=1, eps=1e-8)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity_raw)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc_t, self._features_rest_t), dim=1)

    # Layout-agnostic raw accessors: the renderer's sparse-grad fast path
    # reads these directly instead of the (extra-op) get_* properties above.
    @property
    def _rotation(self) -> torch.Tensor:
        return self._rotation_raw

    @property
    def _scaling(self) -> torch.Tensor:
        return self._scaling_raw

    @property
    def _opacity(self) -> torch.Tensor:
        return self._opacity_raw

    @property
    def _features_dc(self) -> torch.Tensor:
        return self._features_dc_t

    @property
    def _features_rest(self) -> torch.Tensor:
        return self._features_rest_t


def snapshot_gaussians_on_stream(
    gaussians: Any, snapshot_stream: "torch.cuda.Stream"
) -> tuple[GaussianRenderSnapshot, "torch.cuda.Event", float]:
    """Enqueue a device-to-device clone of the rendering-relevant params.

    Returns ``(snapshot, copy_done_event, mapping_thread_stall_s)``.
    ``mapping_thread_stall_s`` is the wall-clock time THIS call occupied the
    calling (mapping) thread -- issuing the clone kernels, never waiting for
    them to complete. Waiting for completion happens later, off-thread, in
    :meth:`AsyncHoldoutEvaluator.submit`'s enqueued task.
    """

    device = gaussians.get_xyz.device
    current_stream = torch.cuda.current_stream(device)
    started = time.perf_counter()
    with torch.cuda.stream(snapshot_stream):
        snapshot_stream.wait_stream(current_stream)
        cloned = {
            attr: getattr(gaussians, attr).detach().clone()
            for attr in _SNAPSHOT_RAW_ATTRS
        }
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(snapshot_stream)
    stall_s = time.perf_counter() - started
    snapshot = GaussianRenderSnapshot(
        xyz=cloned["_xyz"],
        scaling_raw=cloned["_scaling"],
        rotation_raw=cloned["_rotation"],
        opacity_raw=cloned["_opacity"],
        features_dc=cloned["_features_dc"],
        features_rest=cloned["_features_rest"],
        active_sh_degree=int(gaussians.active_sh_degree),
        max_sh_degree=int(gaussians.max_sh_degree),
    )
    return snapshot, copy_done_event, stall_s


class MutationBarrier:
    """The reverse-direction guard the ``_pv_stream`` progress-video path lacks.

    ``arm()`` records the event the next in-place mutation must wait for;
    ``wait_and_clear()`` is the single choke point every in-place mutation
    (optimizer step / MCMC relocation / append) passes through, and blocks
    the DEFAULT stream (not the snapshot stream) until the outstanding
    snapshot copy has finished reading the pre-mutation tensors. A missing
    call here is exactly a torn-read hazard: the mutation could start
    overwriting parameter storage the snapshot-stream clone has not yet
    finished copying out of.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_event: "torch.cuda.Event | None" = None

    def arm(self, event: "torch.cuda.Event") -> None:
        with self._lock:
            self._pending_event = event

    def wait_and_clear(self) -> None:
        with self._lock:
            event = self._pending_event
            self._pending_event = None
        if event is not None:
            torch.cuda.current_stream().wait_event(event)


def _sequence_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"max": 0.0, "mean": 0.0, "samples": 0}
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "samples": len(values),
    }


class AsyncHoldoutEvaluator:
    """Owns the snapshot stream, the mutation barrier, and the render worker.

    Reuses :class:`AsyncSaveWorker`'s queue+daemon-thread plumbing (per the
    issue's explicit instruction to reuse it rather than build new
    thread-management code) for the render+score work itself.
    """

    def __init__(self, *, join_timeout_s: float = 120.0) -> None:
        self._snapshot_stream = torch.cuda.Stream()
        self._barrier = MutationBarrier()
        self._worker = AsyncSaveWorker(join_timeout_s=join_timeout_s)
        self._lock = threading.Lock()
        self._stall_samples_s: list[float] = []
        self._copy_wait_samples_s: list[float] = []

    def wait_for_pending_copy(self) -> None:
        """Call immediately before the next in-place Gaussian mutation."""

        self._barrier.wait_and_clear()

    def submit(self, *, gaussians: Any, evaluate_fn: Callable[[GaussianRenderSnapshot], None]) -> float:
        """Snapshot ``gaussians`` async and enqueue ``evaluate_fn(snapshot)``.

        Arms the mutation barrier with the copy-done event so the next call
        to :meth:`wait_for_pending_copy` blocks the mutation until the clone
        has actually finished reading the pre-mutation tensors. Returns the
        measured mapping-thread stall in seconds -- the only cost this
        imposes on the caller.
        """

        snapshot, event, stall_s = snapshot_gaussians_on_stream(
            gaussians, self._snapshot_stream
        )
        self._barrier.arm(event)
        with self._lock:
            self._stall_samples_s.append(stall_s)

        def _task() -> None:
            # Runs entirely on the background thread: block THIS thread
            # (never the mapping thread) until the copy has actually
            # completed, measure the real wait, then render+score off the
            # now-safe-to-read snapshot.
            waited_start = time.perf_counter()
            event.synchronize()
            copy_wait_s = time.perf_counter() - waited_start
            with self._lock:
                self._copy_wait_samples_s.append(copy_wait_s)
            evaluate_fn(snapshot)

        self._worker.enqueue(_task)
        return stall_s

    def drain(self) -> None:
        """Block until every already-submitted evaluate has completed.

        Must be called before any synchronous evaluation/report step that
        writes to the same on-disk artifacts (psnr_history, eval ticks,
        plots) an in-flight async task might still be writing to.
        """

        self._worker.join()

    def stats(self) -> dict[str, float]:
        with self._lock:
            stalls = list(self._stall_samples_s)
            copy_waits = list(self._copy_wait_samples_s)
        stall_stats = _sequence_stats(stalls)
        copy_stats = _sequence_stats(copy_waits)
        return {
            "mapping_thread_stall_ms_max": stall_stats["max"] * 1000.0,
            "mapping_thread_stall_ms_mean": stall_stats["mean"] * 1000.0,
            "mapping_thread_stall_samples": stall_stats["samples"],
            "snapshot_copy_wait_ms_max": copy_stats["max"] * 1000.0,
            "snapshot_copy_wait_ms_mean": copy_stats["mean"] * 1000.0,
        }


__all__ = [
    "AsyncHoldoutEvaluator",
    "GaussianRenderSnapshot",
    "MutationBarrier",
    "snapshot_gaussians_on_stream",
]
