"""Issue #326 -- the (only) torch-touching half of the operation tracer.

Imported LAZILY from inside a ``if tracer is not None:`` guard, never on the
disabled path. Every function here performs device->host syncs (``.cpu()`` /
``.item()``) and is therefore *expensive by construction*; that is acceptable
precisely because it can only run when tracing was explicitly switched on.

Rows are matched between snapshots **by ``gaussian_uid``, not by row index** --
growth appends and pruning compacts, so index identity is not preserved across
a mutation and an index-matched diff would silently attribute one Gaussian's
state change to a different Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class OpSnapshot:
    """Host-side state of the watched UID subset at one instant."""

    uids: list[int]
    xyz: list[list[float]]
    opacity: list[float]
    scale: list[float]
    n_population: int

    def by_uid(self) -> dict[int, tuple[list[float], float, float]]:
        return {u: (self.xyz[i], self.opacity[i], self.scale[i]) for i, u in enumerate(self.uids)}


def _watched_index(gaussians, uid_stride: int):
    """Row indices whose ``gaussian_uid % uid_stride == 0`` (design invariant 2)."""
    uid = getattr(gaussians, "gaussian_uid", None)
    n = gaussians.get_xyz.shape[0]
    if uid is None or uid.shape[0] != n:
        # No lineage buffers (PLY-loaded model, pre-#373 checkpoint). Fall back to
        # ROW index so the trace is still internally consistent, and say so.
        idx = torch.arange(0, n, uid_stride, device=gaussians.get_xyz.device)
        return idx, idx.to(torch.int64), False
    sel = (uid % int(uid_stride)) == 0
    idx = sel.nonzero(as_tuple=True)[0]
    return idx, uid[idx], True


def snapshot(gaussians, uid_stride: int) -> tuple[OpSnapshot, bool]:
    """Capture the watched subset. Returns ``(snapshot, uids_are_real)``.

    ``uids_are_real`` is False when the model carries no ``gaussian_uid`` buffer
    (PLY-loaded / pre-#373 checkpoint), in which case the "uid" is a row index
    and is NOT stable across a topology change -- the recipe therefore declares
    ``--start_checkpoint``/PLY restore unsupported for UID-keyed tracing.
    """
    with torch.no_grad():
        idx, uids, real = _watched_index(gaussians, uid_stride)
        xyz = gaussians.get_xyz[idx].detach().float().cpu().tolist()
        try:
            opacity = gaussians.get_opacity[idx].detach().float().flatten().cpu().tolist()
        except Exception:
            opacity = [float("nan")] * len(xyz)
        try:
            scale = gaussians.get_scaling[idx].detach().float().max(dim=-1).values.cpu().tolist()
        except Exception:
            scale = [float("nan")] * len(xyz)
        return OpSnapshot(
            uids=[int(u) for u in uids.detach().cpu().tolist()],
            xyz=xyz, opacity=opacity, scale=scale,
            n_population=int(gaussians.get_xyz.shape[0]),
        ), real


def emit_uid_deltas(
    tracer,
    *,
    before: OpSnapshot,
    after: OpSnapshot,
    iteration: int,
    op: str,
    mechanism: str,
    phase: str,
    topology_version: int,
    evidence: dict[str, Any] | None = None,
    eps: float = 0.0,
) -> int:
    """Emit one uid-scoped event per watched UID whose state actually changed.

    UIDs present in only one snapshot are skipped here (a birth has no
    ``state_before`` by construction and a death has no ``state_after``); those
    are reported as aggregate ``grow``/``retire`` events by their own hook,
    which is the honest split -- see ``UNATTRIBUTABLE['mcmc/grow']``.
    """
    b = before.by_uid()
    a = after.by_uid()
    n = 0
    for uid, (xyz_b, op_b, sc_b) in b.items():
        aft = a.get(uid)
        if aft is None:
            continue
        xyz_a, op_a, sc_a = aft
        d_xyz = [float(x2 - x1) for x1, x2 in zip(xyz_b, xyz_a)]
        d_op = float(op_a - op_b)
        d_sc = float(sc_a - sc_b)
        moved = any(abs(v) > eps for v in d_xyz)
        if not (moved or abs(d_op) > eps or abs(d_sc) > eps):
            continue
        applied: dict[str, Any] = {}
        if moved:
            applied["xyz"] = d_xyz
        if abs(d_op) > eps:
            applied["opacity"] = d_op
        if abs(d_sc) > eps:
            applied["scale"] = d_sc
        tracer.record(
            iteration=iteration, op=op, mechanism=mechanism, phase=phase,
            scope="uid", topology_version=topology_version, uid=int(uid),
            state_before={"xyz": xyz_b, "opacity": op_b, "scale": sc_b},
            state_after={"xyz": xyz_a, "opacity": op_a, "scale": sc_a},
            applied_delta=applied,
            evidence=dict(evidence or {}),
        )
        n += 1
    return n


def emit_candidate_set(
    tracer,
    gaussians,
    mask,
    *,
    iteration: int,
    op: str,
    mechanism: str,
    topology_version: int,
    reason: str = "",
    max_uids: int = 4096,
) -> int:
    """Record the WATCHED subset of a candidate mask (death set, growth parents).

    Candidate sets are what make death-candidate OVERLAP measurable: an applied
    retire only ever has one owner, so nominations must be recorded separately
    from executions.
    """
    with torch.no_grad():
        if mask is None:
            return 0
        if mask.dtype == torch.bool:
            idx = mask.nonzero(as_tuple=True)[0]
        else:
            idx = mask.reshape(-1).to(torch.int64)
        if idx.numel() == 0:
            tracer.record(
                iteration=iteration, op=op, mechanism=mechanism, phase="candidate",
                scope="aggregate", topology_version=topology_version, n_rows=0,
                evidence={"reason": reason},
            )
            return 0
        uid = getattr(gaussians, "gaussian_uid", None)
        n = int(gaussians.get_xyz.shape[0])
        if uid is not None and uid.shape[0] == n:
            uids = uid[idx]
        else:
            uids = idx
        watched = uids[(uids % int(tracer.uid_stride)) == 0][:max_uids]
        out = [int(u) for u in watched.detach().cpu().tolist()]
        tracer.record(
            iteration=iteration, op=op, mechanism=mechanism, phase="candidate",
            scope="aggregate", topology_version=topology_version,
            n_rows=int(idx.numel()), candidate_uids=out,
            evidence={"reason": reason, "n_watched_of_candidates": len(out)},
        )
        return len(out)
