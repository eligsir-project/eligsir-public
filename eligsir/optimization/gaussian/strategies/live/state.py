"""Per-Gaussian aligned state buffers for the live population strategy
(Wave 3, PR W3-09, plan section 14.5).

Mirrors the version-gating convention already established by
``eligsir.mapping.evidence.gaussian_history.GaussianEvidenceHistory``:
per-Gaussian tensors are only meaningful for the ``topology_version`` they
were captured under, and reading them against a different version raises
rather than silently applying stale/misaligned state.

Fields (plan section 14.5, "at minimum"):
    geometry lifecycle state, appearance lifecycle state, support count,
    contribution EMA, RGB residual EMA, depth residual EMA,
    position-gradient EMA, opacity-gradient EMA, free-space-conflict EMA,
    geometry confidence, last visible step, birth frame, source surface
    cell.

A few extra bookkeeping fields (``_suspect_streak``, ``_recover_streak``,
``_stable_streak``) are carried beyond that minimum list to make the
"persistent conflict"/"geometry convergence" language in plan section 14.6
implementable as more than a single-step flicker -- see ``live/lifecycle.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from eligsir.optimization.gaussian.strategies.contracts import GaussianLifecycleState

_STATE_TO_CODE: dict[GaussianLifecycleState, int] = {
    GaussianLifecycleState.PROVISIONAL: 0,
    GaussianLifecycleState.ACTIVE: 1,
    GaussianLifecycleState.SUSPECT: 2,
    GaussianLifecycleState.STABLE: 3,
    GaussianLifecycleState.RETIRED: 4,
    GaussianLifecycleState.RESEEDED: 5,
}
_CODE_TO_STATE: dict[int, GaussianLifecycleState] = {v: k for k, v in _STATE_TO_CODE.items()}


def state_to_code(state: GaussianLifecycleState) -> int:
    return _STATE_TO_CODE[state]


def code_to_state(code: int) -> GaussianLifecycleState:
    return _CODE_TO_STATE[int(code)]


class StaleLiveStateError(RuntimeError):
    """Raised when live per-Gaussian state is read against a
    ``topology_version`` that no longer matches the store's current one."""


@dataclass
class LivePopulationState:
    """Version-gated, aligned per-Gaussian buffers.

    One instance persists for the lifetime of a
    :class:`~eligsir.optimization.gaussian.strategies.live.strategy.EliGSiRLiveGaussianStrategy`.
    Not thread-safe (matches the rest of the mapping stack's single
    training-thread assumption).
    """

    topology_version: int = -1
    n: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    geometry_lifecycle: Optional[torch.Tensor] = field(default=None, repr=False)
    appearance_lifecycle: Optional[torch.Tensor] = field(default=None, repr=False)
    support_count: Optional[torch.Tensor] = field(default=None, repr=False)
    contribution_ema: Optional[torch.Tensor] = field(default=None, repr=False)
    rgb_residual_ema: Optional[torch.Tensor] = field(default=None, repr=False)
    depth_residual_ema: Optional[torch.Tensor] = field(default=None, repr=False)
    pos_grad_ema: Optional[torch.Tensor] = field(default=None, repr=False)
    opacity_grad_ema: Optional[torch.Tensor] = field(default=None, repr=False)
    free_space_conflict_ema: Optional[torch.Tensor] = field(default=None, repr=False)
    geometry_confidence: Optional[torch.Tensor] = field(default=None, repr=False)
    last_visible_step: Optional[torch.Tensor] = field(default=None, repr=False)
    birth_frame: Optional[torch.Tensor] = field(default=None, repr=False)
    source_surface_cell: Optional[torch.Tensor] = field(default=None, repr=False)

    # Streak bookkeeping (implementation detail beyond the "at minimum" list).
    _suspect_streak: Optional[torch.Tensor] = field(default=None, repr=False)
    _recover_streak: Optional[torch.Tensor] = field(default=None, repr=False)
    _stable_streak: Optional[torch.Tensor] = field(default=None, repr=False)

    def is_valid_for(self, topology_version: int) -> bool:
        return self.geometry_lifecycle is not None and self.topology_version == topology_version

    def align(
        self,
        n: int,
        topology_version: int,
        *,
        iteration: int,
        device: Optional[torch.device] = None,
        pure_append: bool = False,
    ) -> None:
        """Ensure buffers exist and match ``n``/``topology_version``.

        A ``topology_version`` mismatch is treated as a structural change
        (insert/prune/relocate happened): buffers are cold-started fresh
        at size ``n`` (every Gaussian re-enters as ``PROVISIONAL`` with
        ``birth_frame=iteration``). This is deliberately conservative --
        matching ``GaussianEvidenceHistory.begin_step``'s "no automatic
        index remapping" stance (see that module's docstring) -- rather
        than guessing which old slot maps to which new one.

        ``pure_append=True`` (W3-39) is the caller's explicit assertion
        that the only structural change since the last call was a pure
        append (rows added at the end, nothing pruned/reindexed) -- see
        :meth:`grow_append`. When set, and buffers already exist and
        ``n >= self.n``, this delegates to :meth:`grow_append` instead of
        cold-starting, so existing rows' lifecycle/EMA/streak state
        survives the call untouched. Any other case (first-ever call,
        ``n < self.n``, or ``pure_append=False``) falls through to the
        cold-start path unchanged -- callers that cannot distinguish pure
        append from a structural change keep today's conservative
        behavior by simply not passing ``pure_append=True``.
        """
        device = device or self.device
        if self.is_valid_for(topology_version) and self.n == n:
            return
        if pure_append and self.geometry_lifecycle is not None and n >= self.n:
            self.grow_append(n, topology_version, iteration=iteration, device=device)
            return
        self.topology_version = topology_version
        self.n = n
        self.device = device
        long_zeros = lambda: torch.zeros(n, dtype=torch.int64, device=device)  # noqa: E731
        float_zeros = lambda: torch.zeros(n, dtype=torch.float32, device=device)  # noqa: E731
        self.geometry_lifecycle = torch.full((n,), state_to_code(GaussianLifecycleState.PROVISIONAL), dtype=torch.int8, device=device)
        self.appearance_lifecycle = torch.full((n,), state_to_code(GaussianLifecycleState.PROVISIONAL), dtype=torch.int8, device=device)
        self.support_count = float_zeros()
        self.contribution_ema = float_zeros()
        self.rgb_residual_ema = float_zeros()
        self.depth_residual_ema = float_zeros()
        self.pos_grad_ema = float_zeros()
        self.opacity_grad_ema = float_zeros()
        self.free_space_conflict_ema = float_zeros()
        self.geometry_confidence = float_zeros()
        self.last_visible_step = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.birth_frame = torch.full((n,), int(iteration), dtype=torch.int64, device=device)
        self.source_surface_cell = torch.full((n,), -1, dtype=torch.int64, device=device)
        self._suspect_streak = long_zeros()
        self._recover_streak = long_zeros()
        self._stable_streak = long_zeros()

    def grow_append(
        self,
        n: int,
        topology_version: int,
        *,
        iteration: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Append-preserving growth path (W3-39, plan follow-up to W3-09/
        W3-10/W3-17/W3-41).

        Extends every buffer to size ``n`` by appending ``n - self.n``
        fresh ``PROVISIONAL`` rows (via :meth:`append_rows`), then records
        the new ``topology_version`` -- **every existing row's
        lifecycle/EMA/streak/support-count/etc. value is left exactly as
        it was**, unlike :meth:`align`'s cold-start path, which
        re-initializes the entire buffer set from scratch. This is the
        mechanism that lets retirement/recovery evidence built up over
        many steps actually survive an ordinary Phase-2 depth-insertion
        growth event instead of restarting from PROVISIONAL every time
        (see W3-41's measured ~26-nonzero/~24-zero churn pattern, this
        module's own docstring, and
        ``docs/architecture/2026-08-09-w3-39-topology-version-growth-fix.md``).

        Callers are responsible for having already established that the
        growth between the previous call and this one was a *pure*
        append -- no prune/reseed/relocate-index-shift happened in
        between. This method itself does not (and cannot, from ``n``
        alone) verify that; it only refuses the cases it can detect as
        definitely wrong (``n < self.n``, or no buffers yet to extend).
        See :meth:`align`'s ``pure_append`` kwarg for the gated entry
        point most callers should use instead of calling this directly.
        """
        device = device or self.device
        if self.geometry_lifecycle is None:
            raise StaleLiveStateError(
                "grow_append() requires buffers to already exist -- call align() "
                "at least once (any topology_version) before the first grow_append()."
            )
        if n < self.n:
            raise ValueError(
                f"grow_append() requires n ({n}) >= current n ({self.n}); a shrink is a "
                "structural change (prune/reseed) -- use align() (pure_append=False) instead."
            )
        added = n - self.n
        if added > 0:
            self.append_rows(added, iteration=iteration)
        self.topology_version = topology_version
        self.device = device

    def require_valid(self, topology_version: int) -> None:
        if not self.is_valid_for(topology_version):
            raise StaleLiveStateError(
                f"LivePopulationState holds topology_version={self.topology_version}, "
                f"requested {topology_version}. Call align() first."
            )

    @staticmethod
    def _ema(prev: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, decay: float) -> torch.Tensor:
        updated = decay * prev + (1.0 - decay) * obs
        return torch.where(mask, updated, prev)

    def update_from_observation(
        self,
        *,
        visible_mask: torch.Tensor,
        contribution: Optional[torch.Tensor] = None,
        rgb_residual: Optional[torch.Tensor] = None,
        depth_residual: Optional[torch.Tensor] = None,
        pos_grad: Optional[torch.Tensor] = None,
        opacity_grad: Optional[torch.Tensor] = None,
        free_space_conflict: Optional[torch.Tensor] = None,
        geometry_confidence: Optional[torch.Tensor] = None,
        iteration: int,
        decay: float = 0.9,
    ) -> None:
        """Blend one step's per-Gaussian observation into the running EMAs,
        restricted to ``visible_mask`` (non-visible Gaussians keep their
        previous value). Pure bookkeeping -- never touches model tensors.
        """
        n = self.n
        if visible_mask.shape[0] != n:
            raise ValueError(f"visible_mask has {visible_mask.shape[0]} entries, expected {n}")
        vis = visible_mask.to(torch.bool)

        if contribution is not None:
            self.contribution_ema = self._ema(self.contribution_ema, contribution.detach(), vis, decay)
        if rgb_residual is not None:
            self.rgb_residual_ema = self._ema(self.rgb_residual_ema, rgb_residual.detach(), vis, decay)
        if depth_residual is not None:
            self.depth_residual_ema = self._ema(self.depth_residual_ema, depth_residual.detach(), vis, decay)
        if pos_grad is not None:
            self.pos_grad_ema = self._ema(self.pos_grad_ema, pos_grad.detach(), vis, decay)
        if opacity_grad is not None:
            self.opacity_grad_ema = self._ema(self.opacity_grad_ema, opacity_grad.detach(), vis, decay)
        if free_space_conflict is not None:
            self.free_space_conflict_ema = self._ema(self.free_space_conflict_ema, free_space_conflict.detach(), vis, decay)
        if geometry_confidence is not None:
            self.geometry_confidence = torch.where(vis, geometry_confidence.detach(), self.geometry_confidence)

        self.support_count = torch.where(vis, self.support_count + 1.0, self.support_count)
        self.last_visible_step = torch.where(vis, torch.full_like(self.last_visible_step, int(iteration)), self.last_visible_step)

    def apply_lifecycle_batch(self, batch) -> None:
        """Persist a proposed
        :class:`~eligsir.optimization.gaussian.strategies.contracts.LifecycleUpdateBatch`
        into this strategy's own bookkeeping buffers.

        This writes only to ``self`` (``LivePopulationState`` — the live
        strategy's private lifecycle tracking), never to any gaussian model
        tensor; it is the mechanism by which the strategy's *own* state
        actually advances across steps (PROVISIONAL -> ACTIVE -> ... etc.),
        distinct from (and not to be confused with) mutating the map, which
        this whole package never does. Called by
        ``live/strategy.py``'s ``propose()`` after ``build_mutation_plan``
        computes the batch.
        """
        if batch.geometry_indices.numel():
            self.geometry_lifecycle[batch.geometry_indices] = batch.geometry_new_state
        if batch.appearance_indices.numel():
            self.appearance_lifecycle[batch.appearance_indices] = batch.appearance_new_state

    def persist_streaks(
        self,
        *,
        geometry_suspect_streak: torch.Tensor,
        geometry_recover_streak: torch.Tensor,
        geometry_stable_streak: torch.Tensor,
    ) -> None:
        """Persist the geometry track's streak counters computed this step.
        (Appearance-track streaks currently share the same underlying
        counters as a documented simplification — see
        ``mutation_plan.build_mutation_plan``'s note.)"""
        self._suspect_streak = geometry_suspect_streak
        self._recover_streak = geometry_recover_streak
        self._stable_streak = geometry_stable_streak

    def prune_rows(self, keep_mask: torch.Tensor) -> None:
        """Physically drop rows where ``keep_mask`` is ``False`` from every
        aligned buffer this state owns (Wave 3, PR W3-10). Used by
        ``live/backend.py``'s ``AlignedBufferRegistry`` immediately after
        the paired model-side ``prune_points(~keep_mask)`` call, so this
        state's buffers and the model's tensors never disagree about which
        row index means what, for even one line of caller code -- there is
        no window where one has been pruned and the other has not.

        Does **not** bump ``topology_version`` itself (the caller, i.e. the
        registry, owns that decision) and does **not** reset any surviving
        row's values -- this is a pure compaction, distinct from
        :meth:`reset_rows` which explicitly reinitializes specific rows
        in place without changing ``n``.
        """
        if keep_mask.shape[0] != self.n:
            raise ValueError(f"keep_mask has {keep_mask.shape[0]} entries, expected {self.n}")
        mask = keep_mask.to(torch.bool)
        self.n = int(mask.sum().item())
        self.geometry_lifecycle = self.geometry_lifecycle[mask]
        self.appearance_lifecycle = self.appearance_lifecycle[mask]
        self.support_count = self.support_count[mask]
        self.contribution_ema = self.contribution_ema[mask]
        self.rgb_residual_ema = self.rgb_residual_ema[mask]
        self.depth_residual_ema = self.depth_residual_ema[mask]
        self.pos_grad_ema = self.pos_grad_ema[mask]
        self.opacity_grad_ema = self.opacity_grad_ema[mask]
        self.free_space_conflict_ema = self.free_space_conflict_ema[mask]
        self.geometry_confidence = self.geometry_confidence[mask]
        self.last_visible_step = self.last_visible_step[mask]
        self.birth_frame = self.birth_frame[mask]
        self.source_surface_cell = self.source_surface_cell[mask]
        self._suspect_streak = self._suspect_streak[mask]
        self._recover_streak = self._recover_streak[mask]
        self._stable_streak = self._stable_streak[mask]

    def append_rows(self, count: int, *, iteration: int) -> None:
        """Append ``count`` fresh ``PROVISIONAL`` rows (Wave 3, PR W3-10),
        mirroring the model-side ``add_points_as_gaussians`` row count so
        the registry can keep both aligned after an insert. New rows start
        with the exact same zeroed/PROVISIONAL values ``align()`` would
        give a cold-started buffer -- see that method's field-by-field
        defaults, duplicated here per-row rather than reused directly
        because this call must *extend* existing rows, not replace them.
        """
        if count <= 0:
            return
        device = self.device
        long_zeros = lambda: torch.zeros(count, dtype=torch.int64, device=device)  # noqa: E731
        float_zeros = lambda: torch.zeros(count, dtype=torch.float32, device=device)  # noqa: E731
        cat = torch.cat
        self.geometry_lifecycle = cat([self.geometry_lifecycle, torch.full((count,), state_to_code(GaussianLifecycleState.PROVISIONAL), dtype=torch.int8, device=device)])
        self.appearance_lifecycle = cat([self.appearance_lifecycle, torch.full((count,), state_to_code(GaussianLifecycleState.PROVISIONAL), dtype=torch.int8, device=device)])
        self.support_count = cat([self.support_count, float_zeros()])
        self.contribution_ema = cat([self.contribution_ema, float_zeros()])
        self.rgb_residual_ema = cat([self.rgb_residual_ema, float_zeros()])
        self.depth_residual_ema = cat([self.depth_residual_ema, float_zeros()])
        self.pos_grad_ema = cat([self.pos_grad_ema, float_zeros()])
        self.opacity_grad_ema = cat([self.opacity_grad_ema, float_zeros()])
        self.free_space_conflict_ema = cat([self.free_space_conflict_ema, float_zeros()])
        self.geometry_confidence = cat([self.geometry_confidence, float_zeros()])
        self.last_visible_step = cat([self.last_visible_step, torch.full((count,), -1, dtype=torch.int64, device=device)])
        self.birth_frame = cat([self.birth_frame, torch.full((count,), int(iteration), dtype=torch.int64, device=device)])
        self.source_surface_cell = cat([self.source_surface_cell, torch.full((count,), -1, dtype=torch.int64, device=device)])
        self._suspect_streak = cat([self._suspect_streak, long_zeros()])
        self._recover_streak = cat([self._recover_streak, long_zeros()])
        self._stable_streak = cat([self._stable_streak, long_zeros()])
        self.n = self.n + count

    def snapshot(self) -> dict:
        """Read-only dict snapshot for logging/testing (detached clones)."""
        return {
            "topology_version": self.topology_version,
            "n": self.n,
            "geometry_lifecycle": self.geometry_lifecycle.clone(),
            "appearance_lifecycle": self.appearance_lifecycle.clone(),
            "support_count": self.support_count.clone(),
            "contribution_ema": self.contribution_ema.clone(),
            "rgb_residual_ema": self.rgb_residual_ema.clone(),
            "depth_residual_ema": self.depth_residual_ema.clone(),
            "pos_grad_ema": self.pos_grad_ema.clone(),
            "opacity_grad_ema": self.opacity_grad_ema.clone(),
            "free_space_conflict_ema": self.free_space_conflict_ema.clone(),
            "geometry_confidence": self.geometry_confidence.clone(),
            "last_visible_step": self.last_visible_step.clone(),
            "birth_frame": self.birth_frame.clone(),
            "source_surface_cell": self.source_surface_cell.clone(),
        }
