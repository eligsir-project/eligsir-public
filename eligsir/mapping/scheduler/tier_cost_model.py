"""Hardware-local, bounded per-tier transaction cost calibration.

Resolution *quality* is ordered by construction (native > half > quarter > eighth).
Runtime cost and final reconstruction quality are not ordered by resolution;
each legal tier needs local measurement. EIGHTH availability therefore makes no
cost, quality, or backend-qualification claim. What is **not** predictable is
the magnitude of the saving between two tiers::

    T(R) = F_device_workload + T_resolution_dependent(R)

Cost has a fixed floor that belongs to the device and workload rather than to
resolution. Measured on this project's production ROCm backend (R9700), the
half->quarter step buys roughly 29% under
forward+backward, but essentially nothing under forward-only; on an RTX 3090
dev box the same step buys about 0%, because the fixed floor is over half the
transaction.  (See the ongoing investigation in PR #1249 and
``docs/architecture/2026-08-29-quarter-tier-slowdown-diagnosis.md``; an earlier
report of a genuinely *inverted* ordering there was traced to GPU co-tenancy
contamination and retracted.)

No measured comparison establishes an EIGHTH ordering. A ScanNet++ sweep at
223k Gaussians found a U-shaped cost curve whose ascending
limb starts around 240x180 and below, driven by gsplat ``isects_per_tile``
crossing roughly 500-750 rather than by resolution or Gaussian count directly;
a matched TUM sweep at 9.5k Gaussians never inverts at the same pixel counts.
The selector only picks a tier with valid local measurements, so EIGHTH remains
unavailable to a decision until its actual cost is observed. A future map large
enough to push any tier into the ascending limb is handled by the same evidence
gate without a special case. (See
``docs/architecture/2026-08-29-low-resolution-cost-inversion.md``.)

Two consequences shape this module:

* **Cost must be measured, not assumed.**  A scheduler that hard-codes "one
  tier down is meaningfully cheaper" will spend supervision quality for a
  saving that may be zero on the machine it is actually running on.
* **A calibration is hardware-local and workload-local.**  Cost is a function
  of ``(device, Gaussian-count regime, camera-count regime, render mode, pass
  mode, tier)``.  A CUDA dev-box measurement must never reach a ROCm
  production decision, and a forward-only evidence render must never be
  mistaken for a forward+backward optimization transaction.

The selector answers one question: among the legal tiers with valid recent
evidence for *this* context, which is the highest-quality tier whose measured
cost satisfies the transaction budget?
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from eligsir.contracts.control import (
    MappingTransactionClass,
    ResolutionTier,
)

#: Coarse Gaussian-count regimes.  A measurement taken at 9.5k Gaussians must
#: not be treated as representative at 100k, but a high-dimensional
#: performance model is explicitly out of scope for P0.
GAUSSIAN_REGIME_BOUNDS: tuple[int, int] = (25_000, 100_000)

#: Coarse render-concurrency regimes (C).
CAMERA_REGIME_BOUNDS: tuple[int, int] = (2, 5)

#: Legal live supervision tiers. The work-budget owner selects among these from
#: measured cost and freshness; listing ``eighth`` does not force its use.
LIVE_TIERS: tuple[ResolutionTier, ...] = (
    ResolutionTier.NATIVE,
    ResolutionTier.HALF,
    ResolutionTier.QUARTER,
    ResolutionTier.EIGHTH,
)

#: Pass modes with materially different cost profiles.  Forward-only evidence
#: renders and forward+backward optimization transactions are never pooled.
FORWARD_ONLY = "forward_only"
FORWARD_BACKWARD = "forward_backward"

UNKNOWN_DEVICE = "unknown-device"

#: Supervision-residency classes.  A first-use materialization (resize + stage
#: + H2D, then render) and a subsequent HOT-cache hit (render only, prepare
#: ~0.02ms) are the SAME tier at wildly different cost -- measured 5.06-12.29ms
#: vs 0.021-0.022ms for a C=8 640x480 RGB-D transaction (PR #1252).  Blending
#: them into one average makes a tier look expensive when what was actually
#: observed is a one-time miss that amortizes away, so they are folded into
#: separate windows.
RESIDENCY_HOT_HIT = "hot_hit"
RESIDENCY_COLD_MISS = "cold_miss"
RESIDENCY_MIXED = "mixed"
RESIDENCY_UNKNOWN = "unknown"

RESIDENCY_CLASSES: tuple[str, ...] = (
    RESIDENCY_HOT_HIT,
    RESIDENCY_COLD_MISS,
    RESIDENCY_MIXED,
    RESIDENCY_UNKNOWN,
)


def residency_class(sample: "TierCostSample") -> str:
    """Classify one transaction by how its supervision payloads were sourced.

    ``mixed`` is a real and common state for a C>1 batch where some identities
    were HOT and others were not; it is kept distinct rather than rounded to
    either pure class, because its cost is genuinely between them.
    """

    hits = int(sample.cache_hits)
    misses = int(sample.cache_misses)
    if hits <= 0 and misses <= 0:
        return RESIDENCY_UNKNOWN
    if misses <= 0:
        return RESIDENCY_HOT_HIT
    if hits <= 0:
        return RESIDENCY_COLD_MISS
    return RESIDENCY_MIXED


def gaussian_regime(gaussian_count: int) -> str:
    """Bucket a Gaussian count into a coarse cost regime label."""

    low, high = GAUSSIAN_REGIME_BOUNDS
    if gaussian_count < low:
        return "g<25k"
    if gaussian_count < high:
        return "g25k-100k"
    return "g>100k"


def camera_regime(camera_count: int) -> str:
    """Bucket render concurrency (C) into a coarse cost regime label."""

    low, high = CAMERA_REGIME_BOUNDS
    if camera_count < low:
        return "c1"
    if camera_count < high:
        return "c2-4"
    return "c5+"


def quality_rank(tier: ResolutionTier) -> int:
    """Fixed supervision-quality ordering: native > half > quarter > eighth.

    Higher is better.  This ordering is a property of the supervision signal
    and is never derived from, or traded against, measured cost.
    """

    return -int(tier.downsample_factor)


def _finite_non_negative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _exact_non_negative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class TierCostContext:
    """The workload identity a cost measurement belongs to.

    Two transactions share cost evidence only when every field here matches.
    ``device_id`` in particular partitions the calibration by backend so a
    CUDA measurement can never inform a ROCm decision.
    """

    device_id: str = UNKNOWN_DEVICE
    gaussian_count: int = 0
    camera_count: int = 1
    render_mode: str = "default"
    pass_mode: str = FORWARD_BACKWARD

    def __post_init__(self) -> None:
        for name in ("device_id", "render_mode", "pass_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        _exact_non_negative(self.gaussian_count, "gaussian_count")
        if isinstance(self.camera_count, bool) or not isinstance(self.camera_count, int):
            raise TypeError("camera_count must be an exact integer")
        if self.camera_count < 1:
            raise ValueError("camera_count must be positive")

    def bucket(self) -> tuple[str, str, str, str, str]:
        """The coarse, bounded key this context calibrates against."""

        return (
            self.device_id,
            gaussian_regime(self.gaussian_count),
            camera_regime(self.camera_count),
            self.render_mode,
            self.pass_mode,
        )


@dataclass(frozen=True, slots=True)
class TierCostSample:
    """One completed transaction's separately identifiable cost components.

    ``kernel_ms`` (pure render/backward), ``prepare_ms`` (reload/resize/
    transfer/barrier) and ``transaction_wall_ms`` (the critical path of the
    whole transaction) stay distinct on purpose: conflating them makes
    rasterizer, tier-materialization and scheduler behaviour
    indistinguishable, and it is exactly the fixed ``F`` term of the cost
    model above that hides in ``prepare_ms``.
    """

    context: TierCostContext
    tier: ResolutionTier
    transaction_class: Optional[MappingTransactionClass] = None
    prepare_ms: float = 0.0
    optimization_ms: float = 0.0
    transaction_wall_ms: float = 0.0
    kernel_ms: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.context, TierCostContext):
            raise TypeError("context must be TierCostContext")
        if not isinstance(self.tier, ResolutionTier):
            raise TypeError("tier must be ResolutionTier")
        if self.transaction_class is not None and not isinstance(
            self.transaction_class, MappingTransactionClass
        ):
            raise TypeError("transaction_class must be MappingTransactionClass or None")
        for name in ("cache_hits", "cache_misses"):
            _exact_non_negative(getattr(self, name), name)
        for name in ("prepare_ms", "optimization_ms", "transaction_wall_ms"):
            object.__setattr__(
                self, name, _finite_non_negative(getattr(self, name), name)
            )
        if self.kernel_ms is not None:
            object.__setattr__(
                self, "kernel_ms", _finite_non_negative(self.kernel_ms, "kernel_ms")
            )


@dataclass(frozen=True, slots=True)
class TierCostState:
    """Bounded rolling cost evidence for one (context bucket, tier) pair."""

    tier: ResolutionTier
    bucket: tuple[str, str, str, str, str]
    completed_samples: int
    recent_mean_ms: float
    recent_p95_ms: float
    last_sample_epoch: int

    @property
    def estimate_ms(self) -> float:
        """Conservative point estimate used for feasibility.

        p95 rather than the mean: a tier that usually fits but regularly blows
        the budget is not a tier that fits.  Both are retained for telemetry.
        """

        return self.recent_p95_ms


@dataclass(frozen=True, slots=True)
class TierSelection:
    """The single authoritative tier choice for one planned transaction."""

    tier: ResolutionTier
    reason: str
    probe: bool = False
    budget_ms: Optional[float] = None
    measured_tiers: tuple[ResolutionTier, ...] = field(default_factory=tuple)
    feasible_tiers: tuple[ResolutionTier, ...] = field(default_factory=tuple)


class _RollingWindow:
    """Fixed-capacity sample window with O(1) memory per keyed slot."""

    __slots__ = ("_capacity", "_samples", "_epoch")

    def __init__(self, capacity: int) -> None:
        self._capacity = int(capacity)
        self._samples: list[float] = []
        self._epoch = 0

    def append(self, value: float, *, epoch: int) -> None:
        self._samples.append(float(value))
        if len(self._samples) > self._capacity:
            del self._samples[0]
        self._epoch = int(epoch)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def mean(self) -> float:
        return sum(self._samples) / float(len(self._samples))

    @property
    def p95(self) -> float:
        ordered = sorted(self._samples)
        index = int(math.ceil(0.95 * len(ordered))) - 1
        return ordered[max(0, min(len(ordered) - 1, index))]


class TierCostEstimator:
    """Bounded online cost calibration plus the sole tier-selection algorithm.

    Selection order:

    1. Restrict to the legal tier set (``quarter``/``half``/``native`` by
       default; the configured quality floor may narrow it further) and to the
       evidence-supplied quality ceiling.
    2. Walk that set from highest quality to lowest and take the first tier
       whose **measured** estimate for the current context fits the budget.
       Cost is never inferred from pixel count or ordinal position, so a tier
       whose saving turns out to be negligible on this device does not get
       chosen just because it is nominally smaller.
    3. An unmeasured tier is never selected while any measured tier exists,
       unless the caller explicitly allows probing (``allow_probe=True``).
       Bounded probing is unconditional for replay-only/post-refinement
       work; for fresh/mixed work it is the caller's responsibility to only
       allow it while the pipeline is genuinely healthy (see
       ``WorkBudgetOwner._resolve_transaction_resolution``, issue #1259) --
       this estimator itself does not know or enforce that condition.
    4. If nothing fits, take the measured lowest-cost allowed tier and let other
       work-budget actuators respond; resolution alone is not expected to
       solve overload.
    5. If nothing at all is measured for this context, fall back to the
       ceiling.  This is the only bootstrap path and is reported as such.
    """

    def __init__(
        self,
        *,
        legal_tiers: Iterable[ResolutionTier] = LIVE_TIERS,
        window: int = 16,
        evidence_ttl: int = 64,
        probe_headroom_fraction: float = 0.5,
    ) -> None:
        tiers = tuple(dict.fromkeys(legal_tiers))
        if not tiers:
            raise ValueError("legal_tiers must not be empty")
        if not all(isinstance(tier, ResolutionTier) for tier in tiers):
            raise TypeError("legal_tiers must contain ResolutionTier values")
        if isinstance(window, bool) or int(window) < 1:
            raise ValueError("window must be a positive integer")
        if isinstance(evidence_ttl, bool) or int(evidence_ttl) < 1:
            raise ValueError("evidence_ttl must be a positive integer")
        fraction = float(probe_headroom_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("probe_headroom_fraction must be between zero and one")
        # Highest quality first.
        self._legal_tiers = tuple(sorted(tiers, key=quality_rank, reverse=True))
        self._window = int(window)
        self._evidence_ttl = int(evidence_ttl)
        self._probe_headroom_fraction = fraction
        self._windows: dict[
            tuple[tuple[str, str, str, str, str], ResolutionTier], _RollingWindow
        ] = {}
        # Residency-split evidence. These windows are DELIBERATELY not read by
        # `select`: they exist so a first-use materialization and a HOT-cache
        # hit stay separately attributable (and so the HOT tier cache can price
        # what a retained tier actually saves). Selection continues to read the
        # single blended wall-cost window above, unchanged by this addition.
        self._prepare_windows: dict[
            tuple[tuple[str, str, str, str, str], ResolutionTier, str], _RollingWindow
        ] = {}
        self._residency_wall_windows: dict[
            tuple[tuple[str, str, str, str, str], ResolutionTier, str], _RollingWindow
        ] = {}
        self._epoch = 0
        self._last_sample: TierCostSample | None = None

    @property
    def legal_tiers(self) -> tuple[ResolutionTier, ...]:
        return self._legal_tiers

    @property
    def last_sample(self) -> TierCostSample | None:
        return self._last_sample

    def record(self, sample: TierCostSample) -> None:
        """Fold one completed transaction into bounded keyed evidence."""

        if not isinstance(sample, TierCostSample):
            raise TypeError("sample must be TierCostSample")
        self._last_sample = sample
        if sample.tier not in self._legal_tiers:
            # A tier outside the legal set (post-refinement native under a
            # narrowed policy, say) is observed but never becomes selectable.
            return
        self._epoch += 1
        key = (sample.context.bucket(), sample.tier)
        window = self._windows.get(key)
        if window is None:
            window = _RollingWindow(self._window)
            self._windows[key] = window
        window.append(sample.transaction_wall_ms, epoch=self._epoch)
        residency = residency_class(sample)
        residency_key = (sample.context.bucket(), sample.tier, residency)
        prepare_window = self._prepare_windows.get(residency_key)
        if prepare_window is None:
            prepare_window = _RollingWindow(self._window)
            self._prepare_windows[residency_key] = prepare_window
        prepare_window.append(sample.prepare_ms, epoch=self._epoch)
        wall_window = self._residency_wall_windows.get(residency_key)
        if wall_window is None:
            wall_window = _RollingWindow(self._window)
            self._residency_wall_windows[residency_key] = wall_window
        wall_window.append(sample.transaction_wall_ms, epoch=self._epoch)

    def prepare_state(
        self,
        tier: ResolutionTier,
        *,
        context: TierCostContext,
        residency: str,
    ) -> TierCostState | None:
        """Bounded PREPARE-cost evidence for one (bucket, tier, residency).

        Separate from :meth:`state`, which reports the blended whole-transaction
        wall cost the selector reads.  A caller asking "what does having this
        tier already resident save?" wants the difference between the
        ``cold_miss`` and ``hot_hit`` prepare estimates here, never the blended
        number.
        """

        return self._residency_state(
            self._prepare_windows, tier, context=context, residency=residency
        )

    def residency_state(
        self,
        tier: ResolutionTier,
        *,
        context: TierCostContext,
        residency: str,
    ) -> TierCostState | None:
        """Bounded WALL-cost evidence split by supervision residency."""

        return self._residency_state(
            self._residency_wall_windows, tier, context=context, residency=residency
        )

    def _residency_state(
        self,
        windows: dict[
            tuple[tuple[str, str, str, str, str], ResolutionTier, str], _RollingWindow
        ],
        tier: ResolutionTier,
        *,
        context: TierCostContext,
        residency: str,
    ) -> TierCostState | None:
        if not isinstance(context, TierCostContext):
            raise TypeError("context must be TierCostContext")
        if residency not in RESIDENCY_CLASSES:
            raise ValueError(f"unknown residency class {residency!r}")
        bucket = context.bucket()
        window = windows.get((bucket, tier, residency))
        if window is None or window.count == 0:
            return None
        return TierCostState(
            tier=tier,
            bucket=bucket,
            completed_samples=window.count,
            recent_mean_ms=window.mean,
            recent_p95_ms=window.p95,
            last_sample_epoch=window.epoch,
        )

    def prepare_saving_ms(
        self, tier: ResolutionTier, *, context: TierCostContext
    ) -> float | None:
        """Measured preparation time a HOT hit on ``tier`` avoids, or ``None``.

        This is the ``T_prepare_saved`` term of ``HotTierCache``'s eviction
        value.  It is a MEASURED difference (cold-miss prepare minus hot-hit
        prepare) inside one calibration bucket -- never inferred from pixel
        count or tier ordinal, because measured preparation cost is not
        monotonic in tier (half costs more to prepare than quarter: the CPU
        area-resize tracks output pixels, and native skips the resize
        altogether).  ``None`` means "no evidence yet", which callers must
        treat as unknown rather than as zero.
        """

        miss = self._residency_state(
            self._prepare_windows, tier, context=context, residency=RESIDENCY_COLD_MISS
        )
        if miss is None:
            miss = self._residency_state(
                self._prepare_windows, tier, context=context, residency=RESIDENCY_MIXED
            )
        if miss is None:
            return None
        hit = self._residency_state(
            self._prepare_windows, tier, context=context, residency=RESIDENCY_HOT_HIT
        )
        hit_ms = 0.0 if hit is None else hit.recent_mean_ms
        return max(0.0, miss.recent_mean_ms - hit_ms)

    def state(
        self, tier: ResolutionTier, *, context: TierCostContext
    ) -> TierCostState | None:
        """Bounded evidence for ``tier`` in ``context``'s calibration bucket."""

        if not isinstance(context, TierCostContext):
            raise TypeError("context must be TierCostContext")
        bucket = context.bucket()
        window = self._windows.get((bucket, tier))
        if window is None or window.count == 0:
            return None
        return TierCostState(
            tier=tier,
            bucket=bucket,
            completed_samples=window.count,
            recent_mean_ms=window.mean,
            recent_p95_ms=window.p95,
            last_sample_epoch=window.epoch,
        )

    def states(self, *, context: TierCostContext) -> dict[ResolutionTier, TierCostState]:
        resolved = {}
        for tier in self._legal_tiers:
            state = self.state(tier, context=context)
            if state is not None:
                resolved[tier] = state
        return resolved

    def _valid_estimates(
        self, *, context: TierCostContext
    ) -> dict[ResolutionTier, float]:
        estimates: dict[ResolutionTier, float] = {}
        for tier, state in self.states(context=context).items():
            if self._epoch - state.last_sample_epoch > self._evidence_ttl:
                continue  # aged out; re-measure before trusting it again
            estimates[tier] = state.estimate_ms
        return estimates

    def measured_estimates(
        self, *, context: TierCostContext
    ) -> dict[ResolutionTier, float]:
        """Current, non-aged estimates per measured legal tier (ms) -- the
        same evidence ``select`` decides on, exposed read-only.

        The work budget compares these against its strict per-transaction
        budget to drive its closed-loop budget relaxation: while freshness is
        healthy and some allowed tier is still unaffordable, the budget is
        relaxed one bounded step (see ``WorkBudgetOwner``). Records nothing.
        """

        return dict(self._valid_estimates(context=context))

    def select(
        self,
        *,
        ceiling: ResolutionTier,
        budget_ms: Optional[float],
        context: TierCostContext,
        allow_probe: bool = False,
    ) -> TierSelection:
        """Resolve the one final tier for a planned transaction."""

        if not isinstance(ceiling, ResolutionTier):
            raise TypeError("ceiling must be ResolutionTier")
        if not isinstance(context, TierCostContext):
            raise TypeError("context must be TierCostContext")
        if budget_ms is not None:
            budget_ms = _finite_non_negative(budget_ms, "budget_ms")
        if type(allow_probe) is not bool:
            raise TypeError("allow_probe must be bool")
        clamped_ceiling = self._clamp_to_legal(ceiling)
        estimates = self._valid_estimates(context=context)
        if not estimates:
            return TierSelection(
                tier=clamped_ceiling,
                reason=(
                    "no valid measured tier evidence for this device/workload "
                    "bucket; holding the evidence-proposed ceiling"
                ),
                budget_ms=budget_ms,
            )
        allowed = tuple(
            tier
            for tier in self._legal_tiers
            if tier in estimates and quality_rank(tier) <= quality_rank(clamped_ceiling)
        )
        if not allowed:
            # Every measured tier is above the ceiling; the ceiling is a
            # quality cap, so hold it and wait for evidence at that quality.
            return TierSelection(
                tier=clamped_ceiling,
                reason=(
                    "no measured evidence at or below the proposed quality "
                    "ceiling; holding the ceiling"
                ),
                budget_ms=budget_ms,
                measured_tiers=tuple(estimates),
            )
        if budget_ms is None:
            feasible = allowed
        else:
            feasible = tuple(tier for tier in allowed if estimates[tier] <= budget_ms)
        if feasible:
            # `allowed` is already ordered highest quality first, so the first
            # feasible entry is the highest-quality tier that actually fits.
            chosen = feasible[0]
            reason = (
                f"highest-quality feasible measured tier ({estimates[chosen]:.3f}ms"
                + ("" if budget_ms is None else f" <= {budget_ms:.3f}ms budget")
                + ")"
            )
        else:
            chosen = min(allowed, key=estimates.__getitem__)
            reason = (
                "no legal tier satisfies the transaction budget "
                f"({budget_ms:.3f}ms); holding the measured lowest-cost tier "
                f"({estimates[chosen]:.3f}ms) for other work-budget actuators"
            )
        if allow_probe and budget_ms is not None:
            probe_tier = self._probe_candidate(
                chosen=chosen, estimates=estimates, budget_ms=budget_ms
            )
            if probe_tier is not None:
                return TierSelection(
                    tier=probe_tier,
                    reason=(
                        f"bounded exploration probe of {probe_tier.value}: this "
                        f"device/workload bucket has no recent evidence for it "
                        f"and {estimates[chosen]:.3f}ms leaves headroom under "
                        f"the {budget_ms:.3f}ms budget"
                    ),
                    probe=True,
                    budget_ms=budget_ms,
                    measured_tiers=tuple(estimates),
                    feasible_tiers=feasible,
                )
        return TierSelection(
            tier=chosen,
            reason=reason,
            budget_ms=budget_ms,
            measured_tiers=tuple(estimates),
            feasible_tiers=feasible,
        )

    def _clamp_to_legal(self, tier: ResolutionTier) -> ResolutionTier:
        if tier in self._legal_tiers:
            return tier
        # A proposal outside the legal set (``eighth`` under the quarter
        # quality floor) is clamped to the nearest legal quality.
        floor = self._legal_tiers[-1]
        if quality_rank(tier) < quality_rank(floor):
            return floor
        return self._legal_tiers[0]

    def _probe_candidate(
        self,
        *,
        chosen: ResolutionTier,
        estimates: Mapping[ResolutionTier, float],
        budget_ms: float,
    ) -> ResolutionTier | None:
        """At most one bounded exploration, gated entirely by the caller's
        `allow_probe` flag -- this method has no opinion on which transaction
        classes may reach it.

        The probe deliberately ignores the quality ceiling and may also point
        *below* an already-feasible tier: its purpose is to obtain the missing
        measurement, and evidence for a lower tier is exactly what lets the
        no-feasible-tier fallback step down later. It costs one transaction's
        supervision quality (whichever class the caller allowed to probe).
        Selection is a pure proposal over completed evidence. Until a matching
        transaction records evidence, repeated planning may propose this tier
        again; the one-candidate execution owner decides what actually runs.
        """

        if estimates[chosen] > self._probe_headroom_fraction * budget_ms:
            return None
        for tier in self._legal_tiers:
            if tier is chosen or tier in estimates:
                continue
            return tier
        return None


def resolve_device_identity() -> str:
    """Best-effort accelerator identity used to partition the calibration."""

    try:  # torch is optional in contract-level test contexts
        import torch
    except Exception:  # pragma: no cover - torch is present in production
        return UNKNOWN_DEVICE
    try:
        if not torch.cuda.is_available():
            return "cpu"
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = getattr(properties, "gcnArchName", "") or ""
        name = getattr(properties, "name", "") or ""
        identity = f"{name}:{arch}".strip(":")
        return identity or UNKNOWN_DEVICE
    except Exception:  # pragma: no cover - defensive
        return UNKNOWN_DEVICE


__all__ = [
    "CAMERA_REGIME_BOUNDS",
    "FORWARD_BACKWARD",
    "FORWARD_ONLY",
    "GAUSSIAN_REGIME_BOUNDS",
    "LIVE_TIERS",
    "UNKNOWN_DEVICE",
    "TierCostContext",
    "TierCostEstimator",
    "TierCostSample",
    "TierCostState",
    "TierSelection",
    "resolve_device_identity",
]
