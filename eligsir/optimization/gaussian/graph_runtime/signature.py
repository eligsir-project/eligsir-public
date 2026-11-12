"""``GraphSignature`` — the capture-invalidation key (plan section 17.5).

Any change to any field means the captured graph's pointers/shapes/control
flow are no longer valid for the new step and a fresh capture is required.
This module contains no torch/CUDA dependency — it is deliberately pure
Python (frozen dataclass of hashable primitives) so it can be unit-tested
without a GPU (see ``tests/test_w3_12_graph_signature.py``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphSignature:
    """Minimum field set from plan section 17.5, in that order.

    ``active_shape`` is a tuple (not a bare int) so it can represent either
    an active-*count* (``(n_active,)``) or an active-*mask* shape
    (``(capacity,)``) depending on which the caller's storage model uses —
    the two are not interchangeable invalidation keys (a mask-shaped
    signature must recapture on capacity change even if the count of
    ``True`` entries happens to match a prior capture's count).
    """

    device_backend: str
    capacity_bucket: int
    active_shape: tuple[int, ...]
    resolution_bucket: tuple[int, int]
    sh_degree: int
    renderer_mode: str
    packed_mode: bool
    sparse_grad_mode: bool
    optimizer_mode: str
    loss_policy_id: str
    gradient_policy_id: str

    def describe_diff(self, other: "GraphSignature") -> list[str]:
        """Human-readable list of which fields differ from ``other``.

        Used by capture-invalidation logging/tests so a recapture always
        has a stated reason rather than a silent "signature changed".
        """
        diffs = []
        for field_name in self.__dataclass_fields__:
            a = getattr(self, field_name)
            b = getattr(other, field_name)
            if a != b:
                diffs.append(f"{field_name}: {b!r} -> {a!r}")
        return diffs
