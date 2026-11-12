"""Single-owner facade for legacy/coupled MCMC strategy executors.

The existing MCMC implementations perform coupled model/optimizer state surgery
that cannot be decomposed into simple insert/prune tensors without changing the
method.  Production therefore treats one complete ``step_post_backward`` call
as a delegated mutation proposal: scheduling/conflict order is owned by
``GaussianMutationArbiter`` while the numerical executor remains the proven
strategy implementation.

This is intentionally a transition boundary. New mutation mechanisms should
emit typed ``PruneProposal``/``RelocationProposal``/``SurfaceInsertionProposal``
directly rather than adding another delegated executor.
"""
from __future__ import annotations

from typing import Any

import torch

from eligsir.map.arbiter import (
    DelegatedMutationEffect,
    DelegatedMutationProposal,
    GaussianMutationArbiter,
    ProposalKey,
)


def _resync_gaussian_lifecycle_buffers(gaussians) -> None:
    """Pad/truncate EVERY per-Gaussian lifecycle/lineage buffer to the
    model's current population size after a strategy mutation that did not
    maintain them itself.

    Issue #1259: strategies that delegate straight to gsplat's own upstream
    ``MCMCStrategy``/``DefaultStrategy`` ("gsplat_mcmc", "gsplat_default")
    mutate only ``gaussians.params``/``.optimizers`` and have zero awareness
    of this repo's ~25 parallel per-Gaussian buffers (see
    ``GsplatGaussianModel._STREAMING_BUFFER_NAMES``); ``GsplatEnergyMCMCStrategy``
    keeps them in sync itself via ``utils/strategies/mcmc/proposal.py``'s
    ``append_running_state`` (which additionally assigns real lineage --
    fresh ``gaussian_uid``, parent/root/generation from ``parent_idxs``).
    This function does NOT have that per-row provenance (it only knows the
    delta in COUNT, not which existing row each new row came from), so it
    intentionally trades lineage fidelity for crash-safety: new rows get
    zero/False/-1 sentinel-shaped fills rather than real lineage. Three real
    crashes were found chasing this gap on 2026-08-31 (config-field
    ``AttributeError``, then a ``provisional`` shape mismatch this function's
    first version fixed via ``utils.streaming_state_sync._sync_streaming_state_lengths``,
    then a ``birth_iter`` shape mismatch because that helper's own hand-picked
    name list has silently drifted out of sync with
    ``_STREAMING_BUFFER_NAMES`` -- it never covered the lineage buffers
    (``gaussian_uid``, ``parent_uid``, ``root_uid``, ``generation``,
    ``birth_method``, ``birth_iter``, etc.) at all) -- so this covers the
    FULL class list directly instead of trusting that helper's coverage.
    Acceptable for gsplat_mcmc/gsplat_default today because neither strategy
    is (yet) validated for anything lineage-sensitive (seed-cohort analysis,
    consensus voting); a future strategy that needs real per-row lineage for
    its OWN clones must still call ``append_running_state`` itself, the same
    way ``GsplatEnergyMCMCStrategy`` does.
    """
    from scene.gsplat_model import GsplatGaussianModel

    count = int(gaussians.get_xyz.shape[0])
    guid = getattr(gaussians, "gaussian_uid", None)
    if guid is not None and 0 < guid.shape[0] < count:
        # gaussian_uid must never be zero/placeholder-padded (collides many
        # unrelated new rows onto the same fake uid=0) -- allocate real,
        # unique fresh uids the same way append_running_state does.
        added = count - guid.shape[0]
        next_uid = int(getattr(gaussians, "_next_gaussian_uid", guid.shape[0]))
        new_uids = torch.arange(
            next_uid, next_uid + added, dtype=guid.dtype, device=guid.device
        )
        gaussians._next_gaussian_uid = next_uid + added
        gaussians.gaussian_uid = torch.cat([guid, new_uids], dim=0)
    for name in GsplatGaussianModel._STREAMING_BUFFER_NAMES:
        if name == "gaussian_uid":
            continue
        tensor = getattr(gaussians, name, None)
        if tensor is None or tensor.shape[0] == count:
            continue
        if tensor.shape[0] > count:
            setattr(gaussians, name, tensor[:count].contiguous())
            continue
        pad_shape = (count - tensor.shape[0], *tensor.shape[1:])
        if tensor.dtype == torch.bool:
            fill = False
        elif name in (
            "depth_last_support_uid",
            "depth_last_conflict_uid",
        ):
            fill = -1
        elif name == "lifecycle_state":
            # 0=PROVISIONAL, 1=YOUNG, 2=MATURE, 3=FROZEN (see
            # append_running_state's own comment on this exact point) --
            # new rows are non-provisional, matching `provisional=False`.
            fill = 1
        else:
            fill = 0
        pad = torch.full(pad_shape, fill, dtype=tensor.dtype, device=tensor.device)
        setattr(gaussians, name, torch.cat([tensor, pad], dim=0))


def _canonical_backend(gaussians):
    if getattr(gaussians, "uses_gsplat_layout", False) or hasattr(gaussians, "params"):
        from eligsir.map.adapters.gsplat import GsplatMapBackend
        return GsplatMapBackend(gaussians)
    from eligsir.map.adapters.legacy_graphdeco import LegacyGraphdecoMapBackend
    return LegacyGraphdecoMapBackend(gaussians)


class ArbiterOwnedMCMCStrategy:
    """Delegate all non-topology hooks, arbitrate the mutating tail exactly once."""

    def __init__(self, strategy: Any, gaussians: Any, args: Any) -> None:
        self._strategy = strategy
        self._backend = _canonical_backend(gaussians)
        self._args = args
        self.name = getattr(strategy, "name", type(strategy).__name__)

    def __getattr__(self, name: str):
        return getattr(self._strategy, name)

    def initialize_state(self, *args, **kwargs):
        return self._strategy.initialize_state(*args, **kwargs)

    def step_pre_backward(self, *args, **kwargs):
        return self._strategy.step_pre_backward(*args, **kwargs)

    def inject_noise(self, *args, **kwargs):
        # Fixed-shape noise called through this hook is optimizer-like and does
        # not resize topology. The historical mutating MCMC tail remains owned
        # by step_post_backward below.
        return self._strategy.inject_noise(*args, **kwargs)

    def step_post_backward(self, *args, **kwargs):
        proposal = self.delegated_post_backward_proposal(*args, **kwargs)
        max_population = getattr(self._args, "cap_max", None)
        arbiter = GaussianMutationArbiter(
            self._backend,
            max_population=(None if max_population is None else int(max_population)),
        )
        return arbiter.commit((proposal,))

    def delegated_post_backward_proposal(self, *args, **kwargs):
        """Build, but do not commit, the sole canonical MCMC proposal."""

        iteration = int(kwargs.get("iteration", 0))
        passed_backend = kwargs.get("map_backend")
        if passed_backend is not None and passed_backend is not self._backend:
            raise ValueError("MCMC map_backend must be the model's canonical backend")
        kwargs["map_backend"] = self._backend

        result_box: dict[str, Any] = {}
        gaussians_for_resync = kwargs.get("gaussians")

        def _execute():
            result_box["value"] = self._strategy.step_post_backward(*args, **kwargs)
            # Issue #1259: strategies that delegate straight to gsplat's own
            # upstream MCMCStrategy/DefaultStrategy ("gsplat_mcmc",
            # "gsplat_default") mutate only `gaussians.params`/`.optimizers` --
            # they have zero awareness of this repo's ~25 parallel per-Gaussian
            # lifecycle/lineage buffers (`provisional`, `birth_iter`,
            # `anchor_xyz`, etc., see `GsplatGaussianModel._STREAMING_BUFFER_NAMES`).
            # `GsplatEnergyMCMCStrategy` keeps those in sync itself as part of
            # its own relocate/grow logic; these two do not, so any growth/
            # relocation event desyncs them from the true population size.
            # Found 2026-08-31 via a real canonical-runtime crash:
            # `RuntimeError: The size of tensor a (1190) must match the size
            # of tensor b (1134)` in objective.py's `_visible | _provisional`
            # mask assembly, one growth event after "gsplat_mcmc" was
            # unlocked on this runtime. `utils/streaming_state_sync.py`'s
            # `_sync_streaming_state_lengths` (moved out of `train_streaming.py`
            # during the #1364 audit -- this file's original lazy import of it
            # was what broke the #1224 canonical/legacy routing boundary)
            # already solves exactly this
            # (pads/truncates every such buffer to the current population,
            # with position-aware padding for anchor_xyz/anchor_scale_log/
            # anchor_opacity_logit) for its own strategy dispatch; reused
            # here via a lazy import rather than duplicated, since it has no
            # dependency on anything else in that module. Applied
            # unconditionally (idempotent no-op when already in sync) rather
            # than name-gated, so this invariant holds regardless of which
            # strategy is selected and cannot silently bit-rot if a future
            # strategy is added with the same gap.
            if gaussians_for_resync is not None and getattr(
                gaussians_for_resync, "uses_gsplat_layout", False
            ):
                # Position/param-aware padding (anchor_xyz/anchor_scale_log/
                # anchor_opacity_logit/anchor_normal) first, for the buffers
                # where a real value beats a zero sentinel; then a generic
                # length-only pass over the FULL buffer list catches
                # everything that helper does not cover (see
                # `_resync_gaussian_lifecycle_buffers`'s own docstring for
                # why both are needed).
                from eligsir.utils.streaming_state_sync import _sync_streaming_state_lengths

                _sync_streaming_state_lengths(gaussians_for_resync)
                _resync_gaussian_lifecycle_buffers(gaussians_for_resync)
            value = result_box["value"]
            if isinstance(value, DelegatedMutationEffect):
                return value
            if isinstance(value, dict):
                relocated = max(0, int(value.get("relocated", 0)))
                added = max(0, int(value.get("added", 0)))
                corrected = max(0, int(value.get("corrected", 0)))
                content_changed = bool(
                    value.get(
                        "content_changed",
                        relocated > 0 or added > 0 or corrected > 0,
                    )
                )
                return DelegatedMutationEffect(
                    content_changed=content_changed,
                    topology_changed=bool(relocated > 0 or added > 0),
                    relocated=relocated,
                    corrected=corrected,
                )
            return value

        return DelegatedMutationProposal(
            key=ProposalKey(
                stream_epoch=0,
                source_ordinal=iteration,
                timestamp_ns=0,
                source_id=f"mcmc:{self.name}",
                local_ordinal=0,
            ),
            executor=_execute,
            # MCMC may relocate/noise even when population count is unchanged.
            content_may_change=True,
        )


def ensure_arbiter_owned(strategy: Any, gaussians: Any, args: Any):
    if isinstance(strategy, ArbiterOwnedMCMCStrategy):
        return strategy
    return ArbiterOwnedMCMCStrategy(strategy, gaussians, args)


__all__ = ["ArbiterOwnedMCMCStrategy", "ensure_arbiter_owned"]
