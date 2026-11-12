"""Per-Gaussian evidence history, versioned by topology (plan section 10.4,
"Add topology-version validity").

Per-Gaussian evidence tensors (visibility EMA, gradient EMAs, support, ...)
are indexed by Gaussian slot ``[0, N)``. That index is only meaningful for
the ``topology_version`` it was captured under: an insert appends new
slots, a prune/relocate can reorder or reuse slots entirely. Reading
last-topology-version evidence at this-topology-version indices without
checking would silently apply one Gaussian's history to a different
Gaussian after a structural mutation — exactly the failure mode this
module exists to make impossible to do by accident.

``GaussianEvidenceHistory`` is a thin, explicit version-gated store: every
read declares the topology version it expects, and a mismatch raises
:class:`StaleEvidenceError` rather than returning silently-wrong data.
There is no automatic index remapping here (that would require the actual
insert/prune permutation, which this observation-only module does not
own) — a caller that wants to carry evidence across a mutation must
explicitly call :meth:`GaussianEvidenceHistory.invalidate` (dropping the
stale state, cold-starting fresh) or a future PR's remap-aware update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from eligsir.mapping.evidence.contracts import EvidenceMeta, GaussianEvidence


class StaleEvidenceError(RuntimeError):
    """Raised when per-Gaussian evidence is read/written against a
    topology_version that no longer matches the history's current one.
    """


@dataclass
class GaussianEvidenceHistory:
    """Holds the most recent :class:`GaussianEvidence` plus running EMA
    state, gated by topology version.

    Not thread-safe (matches the rest of the mapping stack's single
    training-thread assumption — see ``eligsir/mapping/stores/*``).
    """

    topology_version: int = -1
    map_version: int = 0
    _evidence: Optional[GaussianEvidence] = field(default=None, repr=False)
    _xyz_grad_ema: Optional[torch.Tensor] = field(default=None, repr=False)
    _opacity_grad_ema: Optional[torch.Tensor] = field(default=None, repr=False)

    def is_valid_for(self, topology_version: int) -> bool:
        return self._evidence is not None and self.topology_version == topology_version

    def invalidate(self) -> None:
        """Drop all stored state. Call this whenever a structural mutation
        (insert/prune/relocate) is known to have happened and no
        index-remap is being performed for evidence specifically.
        """
        self.topology_version = -1
        self._evidence = None
        self._xyz_grad_ema = None
        self._opacity_grad_ema = None

    def get(self, topology_version: int) -> GaussianEvidence:
        """Return the current evidence, iff it was captured under
        ``topology_version``. Raises :class:`StaleEvidenceError` otherwise
        — callers must not catch this and silently fall back to stale data;
        the correct response is to rebuild evidence fresh for the current
        topology (cold start).
        """
        if not self.is_valid_for(topology_version):
            raise StaleEvidenceError(
                f"GaussianEvidenceHistory holds topology_version="
                f"{self.topology_version}, requested {topology_version}. "
                "Per-Gaussian evidence indices are not valid across a "
                "structural mutation; call invalidate() and rebuild."
            )
        assert self._evidence is not None
        return self._evidence

    def begin_step(self, topology_version: int, map_version: int) -> None:
        """Advance this history to ``topology_version``/``map_version``.

        Must be called **before** any EMA reducer reads
        :meth:`ema_state` for the step — if ``topology_version`` differs
        from what this history currently holds, prior EMA state is
        discarded here (cold start) so a reducer never blends an EMA
        computed against a different, mis-indexed Gaussian population.
        ``_evidence`` is discarded on the same branch: a mismatched
        ``topology_version`` bumps ``self.topology_version`` to the new
        value, and leaving ``_evidence`` (captured under the OLD version)
        in place would make :meth:`is_valid_for`/:meth:`get` report it as
        valid for the new version — exactly the stale-index-misapplication
        bug this whole module exists to prevent (see ``StaleEvidenceError``
        docstring). Idempotent within a step: calling it again with the
        same ``topology_version`` before :meth:`update` is a no-op on both
        the EMA and evidence state (only ``map_version`` is refreshed).
        """
        if self.topology_version != topology_version:
            self._xyz_grad_ema = None
            self._opacity_grad_ema = None
            self._evidence = None
        self.topology_version = topology_version
        self.map_version = map_version

    def update(
        self,
        evidence: GaussianEvidence,
        topology_version: int,
        map_version: int,
    ) -> None:
        """Store new evidence for ``topology_version``.

        Calls :meth:`begin_step` first for callers that skip it (e.g. a
        caller that never reads EMA state), so this remains safe to call
        standalone. A caller that *does* use EMA reducers should call
        :meth:`begin_step` itself before reading :meth:`ema_state`, then
        call this method last to publish the finished evidence without
        re-triggering a (now redundant, but harmless — same version) reset.
        """
        self.begin_step(topology_version, map_version)
        self._evidence = evidence

    def ema_state(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Current ``(xyz_grad_ema, opacity_grad_ema)`` state tensors, for a
        reducer (see ``reducers.py``) to update in place at the caller's
        discretion. Returned tensors are the live references — treat as
        read-only; use :meth:`set_ema_state` to publish updates.
        """
        return self._xyz_grad_ema, self._opacity_grad_ema

    def set_ema_state(
        self,
        xyz_grad_ema: Optional[torch.Tensor],
        opacity_grad_ema: Optional[torch.Tensor],
    ) -> None:
        self._xyz_grad_ema = xyz_grad_ema
        self._opacity_grad_ema = opacity_grad_ema

    def meta(self) -> EvidenceMeta:
        return EvidenceMeta(
            map_version=self.map_version,
            topology_version=self.topology_version,
        )
