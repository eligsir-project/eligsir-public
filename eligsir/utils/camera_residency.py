"""Bounded GPU residency for replay-paged camera image tensors (issue #396).

Replay views are retained until the next iteration boundary so no consumer can
lose a tensor during a B>1 backward pass.  At release, real torch camera images
are grouped and offloaded through the shared transfer layer instead of issuing
one D2H per camera.  Duck-typed/non-tensor test cameras retain the established
single-camera fallback.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set


class CameraResidencyLedger:
    """Track camera images paged onto the GPU during the current iteration.

    ``release`` runs at the top of the next iteration, after all consumers of
    the previous view set are done.  This keeps residency bounded while making
    the complete B-view set owner-safe.  ``enabled=False`` makes the ledger a
    no-op, preserving the existing offload-policy escape hatch.
    """

    __slots__ = (
        "enabled",
        "_pending",
        "_batch_offloader",
        "_transfer_owner",
        "cumulative_paged_in",
        "cumulative_paged_out",
        "cumulative_retained_hot",
        "max_concurrent_paged_in",
        "last_paged_out",
    )

    def __init__(self, enabled: bool = True, *, transfer_owner=None) -> None:
        """Create an empty per-iteration residency ledger."""

        self.enabled = bool(enabled)
        self._pending: Dict[int, object] = {}
        self._batch_offloader = None
        self._transfer_owner = transfer_owner
        self.cumulative_paged_in = 0
        self.cumulative_paged_out = 0
        self.cumulative_retained_hot = 0
        self.max_concurrent_paged_in = 0
        self.last_paged_out = 0

    def bind_transfer_owner(self, transfer_owner) -> None:
        """Route subsequent live page-outs through the canonical owner."""

        if transfer_owner is None:
            raise ValueError("transfer_owner must not be None")
        if self._transfer_owner is not None and self._transfer_owner is not transfer_owner:
            raise RuntimeError("CameraResidencyLedger transfer owner is already bound")
        self._transfer_owner = transfer_owner

    def note_paged_in(self, cam) -> None:
        """Register a camera whose image was paged onto the accelerator."""

        if not self.enabled or cam is None:
            return
        key = id(cam)
        if key not in self._pending:
            self._pending[key] = cam
            self.cumulative_paged_in += 1
            self.max_concurrent_paged_in = max(
                self.max_concurrent_paged_in, len(self._pending)
            )

    @staticmethod
    def _batchable(cam) -> bool:
        image = getattr(cam, "original_image", None)
        return bool(
            image is not None
            and hasattr(image, "device")
            and hasattr(image, "shape")
            and hasattr(image, "dtype")
            and getattr(image.device, "type", None) != "cpu"
        )

    def release(self, hot_ids: Optional[Iterable[int]] = None) -> int:
        """Page out registered non-hot cameras and clear the iteration ledger.

        Real torch tensors are grouped by device/shape/dtype and submitted via
        :class:`eligsir.runtime.transfers.CameraBatchOffloader`; legacy
        duck-typed cameras use ``offload_camera_image`` individually.  The
        method returns only after each selected camera owns CPU storage, which
        preserves the pre-batching release contract.
        """

        if not self.enabled:
            return 0
        if not self._pending:
            self.last_paged_out = 0
            return 0

        hot: Set[int] = set(hot_ids) if hot_ids is not None else set()
        batchable = []
        fallback = []
        for key, cam in self._pending.items():
            if key in hot:
                self.cumulative_retained_hot += 1
            elif self._batchable(cam):
                batchable.append(cam)
            else:
                fallback.append(cam)

        n_out = 0
        if batchable:
            if self._transfer_owner is not None:
                n_out += int(self._transfer_owner.offload_cameras(batchable))
            else:
                if self._batch_offloader is None:
                    from eligsir.runtime.transfers import CameraBatchOffloader

                    self._batch_offloader = CameraBatchOffloader()
                n_out += int(self._batch_offloader.offload(batchable))

        if fallback:
            from scene.cameras import offload_camera_image

            n_out += sum(1 for cam in fallback if offload_camera_image(cam))

        self._pending.clear()
        self.cumulative_paged_out += n_out
        self.last_paged_out = n_out
        return n_out

    def pending_count(self) -> int:
        """Return the number of distinct page-ins held for this iteration."""

        return len(self._pending)

    def transfer_stats(self) -> Dict[str, int]:
        """Return grouped camera page-out transfer counters."""

        if self._batch_offloader is None:
            return {
                "h2d_ops": 0,
                "h2d_bytes": 0,
                "d2h_ops": 0,
                "d2h_bytes": 0,
                "pool_waits": 0,
                "pool_drops": 0,
            }
        return dict(self._batch_offloader.stats())

    def stats(self) -> Dict[str, int]:
        """Return per-attempt page-in/page-out event counters."""

        return {
            "paged_in": self.cumulative_paged_in,
            "paged_out": self.cumulative_paged_out,
            "retained_hot": self.cumulative_retained_hot,
            "max_concurrent_paged_in": self.max_concurrent_paged_in,
        }
