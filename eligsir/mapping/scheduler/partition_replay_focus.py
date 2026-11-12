"""CPU-only replay weighting from same-render partition PSNR evidence."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import math
from statistics import median
from typing import Any, Mapping, Sequence

from eligsir.contracts.runtime import SourceIdentity
from eligsir.plot_style import svg_output_path
from eligsir.utils.partition_eval import PARTITIONS, partition_id


@dataclass(frozen=True, slots=True)
class PartitionReplayFocusDecision:
    """One immutable tail-refinement replay-weight publication."""

    active: bool
    reason: str
    weights: tuple[tuple[SourceIdentity, float], ...]
    partition_weights: tuple[float, ...]
    history_samples: int


class PartitionReplayFocusController:
    """Bias replay toward weak/fading trajectory regions without exclusion.

    Every identity starts at weight 1.0. Low current PSNR and regression from
    the partition's observed peak can only increase that weight, so the
    weighted full-cache provider retains trajectory-wide coverage. Evidence
    is ignored until several same-holdout samples exist and undersampled
    partitions remain neutral.
    """

    def __init__(
        self,
        *,
        min_history: int = 3,
        min_views_per_partition: int = 2,
        low_psnr_gain: float = 0.5,
        regression_gain: float = 1.0,
        regression_tolerance_db: float = 0.25,
        max_weight: float = 8.0,
    ) -> None:
        if isinstance(min_history, bool) or int(min_history) < 1:
            raise ValueError("min_history must be a positive integer")
        if (
            isinstance(min_views_per_partition, bool)
            or int(min_views_per_partition) < 1
        ):
            raise ValueError("min_views_per_partition must be a positive integer")
        for name, value in (
            ("low_psnr_gain", low_psnr_gain),
            ("regression_gain", regression_gain),
            ("regression_tolerance_db", regression_tolerance_db),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(max_weight, bool) or not math.isfinite(float(max_weight)):
            raise ValueError("max_weight must be finite")
        if float(max_weight) < 1.0:
            raise ValueError("max_weight must be at least 1")
        self._min_history = int(min_history)
        self._min_views = int(min_views_per_partition)
        self._low_gain = float(low_psnr_gain)
        self._regression_gain = float(regression_gain)
        self._regression_tolerance = float(regression_tolerance_db)
        self._max_weight = float(max_weight)
        self._history_samples = 0
        self._partition_peaks: list[float | None] = []

    @staticmethod
    def _neutral(
        sources: Sequence[SourceIdentity],
        *,
        reason: str,
        history_samples: int,
        partition_count: int = 0,
    ) -> PartitionReplayFocusDecision:
        return PartitionReplayFocusDecision(
            active=False,
            reason=reason,
            weights=tuple((source, 1.0) for source in sources),
            partition_weights=(1.0,) * max(0, int(partition_count)),
            history_samples=int(history_samples),
        )

    def update(
        self,
        record: Mapping[str, Any],
        sources: Sequence[SourceIdentity],
    ) -> PartitionReplayFocusDecision:
        """Update replay weights from partition PSNR evidence."""
        sources = tuple(sources)
        if not all(isinstance(source, SourceIdentity) for source in sources):
            raise TypeError("replay focus sources must be SourceIdentity values")
        try:
            k = int(record["partition_k"])
            denominator = int(record["partition_denominator"])
            counts = tuple(int(value) for value in record["partition_n"])
            raw_values = tuple(record["partition_psnr_db"])
        except (KeyError, TypeError, ValueError):
            return self._neutral(
                sources,
                reason="invalid_partition_evidence",
                history_samples=self._history_samples,
            )
        if (
            k < 2
            or denominator <= 0
            or len(counts) != k
            or len(raw_values) != k
            or any(count < 0 for count in counts)
        ):
            return self._neutral(
                sources,
                reason="invalid_partition_evidence",
                history_samples=self._history_samples,
                partition_count=max(0, k),
            )
        raw_supervision_counts = record.get("partition_supervision_n")
        if raw_supervision_counts is None:
            supervision_counts = (0,) * k
        else:
            try:
                supervision_counts = tuple(int(value) for value in raw_supervision_counts)
            except (TypeError, ValueError):
                return self._neutral(
                    sources,
                    reason="invalid_partition_evidence",
                    history_samples=self._history_samples,
                    partition_count=k,
                )
            if len(supervision_counts) != k or any(value < 0 for value in supervision_counts):
                return self._neutral(
                    sources,
                    reason="invalid_partition_evidence",
                    history_samples=self._history_samples,
                    partition_count=k,
                )
        values: list[float | None] = []
        for value in raw_values:
            if value is None:
                values.append(None)
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                parsed = math.nan
            values.append(parsed if math.isfinite(parsed) else None)

        coverage_target = max(supervision_counts, default=0)
        coverage_weights = tuple(
            min(
                self._max_weight,
                1.0 + (coverage_target - count) / max(1, count),
            )
            if coverage_target > count
            else 1.0
            for count in supervision_counts
        )
        measured = [
            value
            for value, count in zip(values, counts, strict=True)
            if value is not None and count >= self._min_views
        ]
        if len(measured) < 2:
            source_weights = tuple(
                (
                    source,
                    coverage_weights[
                        partition_id(source.source_ordinal, denominator, k=k)
                    ],
                )
                for source in sources
            )
            if any(weight > 1.0 for _, weight in source_weights):
                return PartitionReplayFocusDecision(
                    active=True,
                    reason="partition_coverage_focus",
                    weights=source_weights,
                    partition_weights=coverage_weights,
                    history_samples=self._history_samples,
                )
            return self._neutral(
                sources,
                reason="insufficient_partition_support",
                history_samples=self._history_samples,
                partition_count=k,
            )

        if len(self._partition_peaks) != k:
            self._partition_peaks = [None] * k
            self._history_samples = 0
        previous_peaks = tuple(self._partition_peaks)
        for index, (value, count) in enumerate(zip(values, counts, strict=True)):
            if value is None or count < self._min_views:
                continue
            peak = self._partition_peaks[index]
            self._partition_peaks[index] = value if peak is None else max(peak, value)
        self._history_samples += 1
        if self._history_samples < self._min_history:
            return self._neutral(
                sources,
                reason="partition_history_warmup",
                history_samples=self._history_samples,
                partition_count=k,
            )

        # The absolute-quality target must be "what this scene demonstrably
        # achieves elsewhere" (the best-observed partition), not the group
        # median. A median target is only a robust "central tendency" when
        # weak partitions are a minority outlier group; on a real Replica
        # room0 run four of six partitions sat within ~0.5dB of each other
        # near 18dB while only two sat near 25-27dB, so the median landed
        # INSIDE the weak cluster itself (18.39dB) and every weak partition's
        # low_gap collapsed to ~0-0.4dB -- producing near-zero requested
        # focus weight for the very partitions the controller exists to lift,
        # exactly reproducing the "gained uniformly, spread unchanged"
        # real-run finding this fix addresses. Using the best-observed
        # partition as target instead measures every partition against what
        # the scene has already proven achievable, so a majority-weak
        # distribution still yields a large low_gap for the weak group.
        target = float(max(measured))
        partition_weights = []
        for index, (value, count) in enumerate(zip(values, counts, strict=True)):
            if value is None or count < self._min_views:
                partition_weights.append(coverage_weights[index])
                continue
            low_gap = max(0.0, target - value)
            previous_peak = previous_peaks[index]
            regression_gap = (
                0.0
                if previous_peak is None
                else max(0.0, previous_peak - value - self._regression_tolerance)
            )
            requested = (
                1.0
                + self._low_gain * low_gap
                + self._regression_gain * regression_gap
            )
            partition_weights.append(
                max(
                    coverage_weights[index],
                    min(self._max_weight, requested),
                )
            )

        source_weights = tuple(
            (
                source,
                partition_weights[
                    partition_id(source.source_ordinal, denominator, k=k)
                ],
            )
            for source in sources
        )
        active = any(weight > 1.0 for _, weight in source_weights)
        return PartitionReplayFocusDecision(
            active=active,
            reason=("partition_psnr_focus" if active else "partition_psnr_balanced"),
            weights=source_weights,
            partition_weights=tuple(partition_weights),
            history_samples=self._history_samples,
        )


class TrainingPartitionPsnrAccumulator:
    """Bounded per-partition history of already-produced training PSNR."""

    def __init__(
        self,
        *,
        n_source_frames: int,
        window_per_partition: int = 16,
        partition_count: int = PARTITIONS,
    ) -> None:
        if isinstance(n_source_frames, bool) or int(n_source_frames) < 2:
            raise ValueError("n_source_frames must be at least 2")
        if isinstance(window_per_partition, bool) or int(window_per_partition) < 1:
            raise ValueError("window_per_partition must be a positive integer")
        if isinstance(partition_count, bool) or int(partition_count) < 2:
            raise ValueError("partition_count must be an integer >= 2")
        self._denominator = int(n_source_frames) - 1
        # Deliberately the SAME fixed partition count the post-training
        # report/holdout accounting uses (utils.partition_eval.PARTITIONS,
        # report.json's "p0".."p5"), NOT the trajectory-length-scaled
        # utils.partition_eval.partition_count() used elsewhere (16-way on
        # a 2000-frame Replica scene). Issue #1552 root-caused a real
        # verification failure to this mismatch: the controller's weight
        # elevations were computed over a partitioning the report/holdout
        # accounting never sees, so "did the worst reported partition
        # improve" could not even in principle be validated against what
        # the controller actually did. Overridable only for tests that need
        # to probe a specific k explicitly; production call sites must not
        # pass a k derived from partition_count(n_source_frames) here.
        self._k = int(partition_count)
        self._samples = tuple(
            deque(maxlen=int(window_per_partition)) for _ in range(self._k)
        )
        self._supervision_counts = [0] * self._k

    def _record(self) -> dict[str, Any]:
        counts = [len(samples) for samples in self._samples]
        means = [
            (sum(samples) / len(samples) if samples else None)
            for samples in self._samples
        ]
        return {
            "partition_k": self._k,
            "partition_denominator": self._denominator,
            "partition_n": counts,
            "partition_supervision_n": list(self._supervision_counts),
            "partition_psnr_db": means,
            "score_scope": "training_primary_pre_update",
        }

    def record_successful_supervision(
        self, sources: Sequence[SourceIdentity]
    ) -> dict[str, Any]:
        """Account every completed view without requesting another render score."""

        for source in sources:
            if not isinstance(source, SourceIdentity):
                raise TypeError("successful supervision sources must be SourceIdentity")
            self._supervision_counts[
                partition_id(source.source_ordinal, self._denominator, k=self._k)
            ] += 1
        return self._record()

    def observe(self, source: SourceIdentity, psnr_db: float) -> dict[str, Any]:
        """Record a training PSNR sample from a source."""
        if not isinstance(source, SourceIdentity):
            raise TypeError("training PSNR source must be a SourceIdentity")
        if isinstance(psnr_db, bool) or not math.isfinite(float(psnr_db)):
            raise ValueError("training PSNR must be finite")
        pid = partition_id(
            source.source_ordinal,
            self._denominator,
            k=self._k,
        )
        self._samples[pid].append(float(psnr_db))
        return self._record()


@dataclass(frozen=True, slots=True)
class SpatialCellQuality:
    """One training-render score attached to an immutable world microcell."""

    cell_key: int
    psnr_db: float
    n_samples: int

    def __post_init__(self) -> None:
        if isinstance(self.cell_key, bool) or not isinstance(self.cell_key, int):
            raise TypeError("cell_key must be an integer")
        if isinstance(self.psnr_db, bool) or not math.isfinite(float(self.psnr_db)):
            raise ValueError("psnr_db must be finite")
        if isinstance(self.n_samples, bool) or int(self.n_samples) < 1:
            raise ValueError("n_samples must be a positive integer")
        object.__setattr__(self, "psnr_db", float(self.psnr_db))
        object.__setattr__(self, "n_samples", int(self.n_samples))


@dataclass(frozen=True, slots=True)
class SpatialMicrocellQualitySample:
    """Training-only per-cell evidence from one already-rendered replay view."""

    source: SourceIdentity
    cell_size_m: float
    cells: tuple[SpatialCellQuality, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        if (
            isinstance(self.cell_size_m, bool)
            or not math.isfinite(float(self.cell_size_m))
            or float(self.cell_size_m) <= 0.0
        ):
            raise ValueError("cell_size_m must be finite and positive")
        cells = tuple(self.cells)
        if not cells or not all(isinstance(cell, SpatialCellQuality) for cell in cells):
            raise ValueError("cells must contain SpatialCellQuality values")
        keys = tuple(cell.cell_key for cell in cells)
        if len(set(keys)) != len(keys):
            raise ValueError("a spatial sample cannot repeat a cell key")
        object.__setattr__(self, "cell_size_m", float(self.cell_size_m))
        object.__setattr__(self, "cells", cells)


def _spatial_cell_reduction(*, camera, tracker, sensor_depth, pixel_values=None):
    """Return compact ``(cell_key, value_sum, pixel_count)`` rows on CPU."""
    import torch

    from eligsir.utils.local_quality import camera_pixel_cell_keys

    mapped = camera_pixel_cell_keys(
        camera,
        tracker.grid,
        sensor_depth,
        tracker.min_depth_m,
        tracker.max_depth_m,
    )
    if mapped is None:
        return ()
    keys, valid = mapped
    unique_keys, inverse, counts = torch.unique(
        keys, sorted=True, return_inverse=True, return_counts=True
    )
    sums = torch.zeros_like(unique_keys, dtype=torch.float32)
    if pixel_values is not None:
        values = pixel_values.to(device=keys.device, dtype=torch.float32).reshape(-1)
        if values.numel() != valid.numel():
            raise ValueError("pixel values and sensor depth must share one pixel grid")
        sums.scatter_add_(0, inverse, torch.where(valid, values, 0.0))
    host_keys = unique_keys.detach().cpu().tolist()
    host_stats = (
        torch.stack((sums, counts.to(torch.float32)), dim=1)
        .detach()
        .cpu()
        .tolist()
    )
    if -2 in host_keys:
        raise ValueError("cell index is outside the packed int64 range")
    return tuple(
        (int(key), float(stats[0]), int(stats[1]))
        for key, stats in zip(host_keys, host_stats, strict=True)
        if key >= 0
    )


def spatial_microcell_quality_from_render(
    *,
    source: SourceIdentity,
    camera: Any,
    tracker: Any,
    render_image: Any,
    gt_image: Any,
    sensor_depth: Any,
) -> SpatialMicrocellQualitySample | None:
    """Reduce one existing training render into fixed world-space cells.

    Cell identity is derived only from the camera pose and sensor depth.  The
    current map contributes RGB residuals, never geometry, so a degrading map
    cannot move its own evidence between cells.  This helper deliberately
    consumes an already-produced render and therefore adds no render pass. Its
    pixel-to-cell tensors are transaction-scoped: training cameras are mostly
    one-shot here, so retaining their dense maps would make device memory grow
    with every sampled source. Only the compact touched-cell summary crosses
    to the CPU control plane.
    """

    if not isinstance(source, SourceIdentity):
        raise TypeError("source must be SourceIdentity")
    if sensor_depth is None:
        return None
    expected_shape = tuple(sensor_depth.shape[-2:])
    if (
        tuple(render_image.shape[-2:]) != expected_shape
        or tuple(gt_image.shape[-2:]) != expected_shape
    ):
        raise ValueError("render, GT, and sensor depth must share one pixel grid")
    if (
        render_image.device != sensor_depth.device
        or gt_image.device != sensor_depth.device
    ):
        raise ValueError("training spatial tensors must already share one device")
    squared_error = (
        (render_image.float() - gt_image.float()) ** 2
    ).reshape(3, -1).sum(dim=0)
    measured = _spatial_cell_reduction(
        camera=camera,
        tracker=tracker,
        sensor_depth=sensor_depth,
        pixel_values=squared_error,
    )
    if not measured:
        return None
    cells = tuple(
        SpatialCellQuality(
            cell_key=int(key),
            psnr_db=-10.0 * math.log10(max(float(sse) / (3.0 * float(count)), 1e-12)),
            n_samples=3 * int(count),
        )
        for key, sse, count in measured
    )
    return SpatialMicrocellQualitySample(
        source=source,
        cell_size_m=float(tracker.grid.cell_size),
        cells=cells,
    )


@dataclass(frozen=True, slots=True)
class SpatialMicrocellReplayFocusDecision:
    """Immutable spatial microcell replay-weight decision and telemetry."""

    active: bool
    reason: str
    weights: tuple[tuple[SourceIdentity, float], ...]
    cell_psnr_db: tuple[tuple[int, float], ...]
    cell_drop_from_peak_db: tuple[tuple[int, float], ...]
    history_samples: int
    score_scope: str = "training_primary_spatial_pre_update"
    # Cell-debt telemetry (issue #1259 spatial-focus follow-up). Populated
    # regardless of which actuation mode is active, so the same jsonl trace
    # can classify a persistently-bad cell as a scheduler problem (low
    # exposure) vs. a mobility/representation problem (high exposure, no
    # gain) without needing a second, incompatible run.
    cell_cumulative_samples: tuple[tuple[int, int], ...] = ()
    cell_gain_since_first_observed_db: tuple[tuple[int, float], ...] = ()
    actuation_mode: str = "frame_averaged"


class SpatialMicrocellReplayFocusController:
    """Weight full-cache replay from training-only world-space cell debt.

    Microcell identity comes from fixed sensor depth and pose.  The controller
    only raises weights above 1.0; frames with no measured incidence remain
    neutral and therefore retain the candidate provider's global coverage.
    """

    def __init__(
        self,
        *,
        min_history: int = 3,
        minimum_cell_samples: int = 64,
        window_per_cell: int = 16,
        low_psnr_gain: float = 0.5,
        regression_gain: float = 1.0,
        regression_tolerance_db: float = 0.25,
        max_weight: float = 8.0,
        cell_debt_actuation: bool = False,
    ) -> None:
        if isinstance(min_history, bool) or int(min_history) < 1:
            raise ValueError("min_history must be a positive integer")
        if isinstance(minimum_cell_samples, bool) or int(minimum_cell_samples) < 1:
            raise ValueError("minimum_cell_samples must be a positive integer")
        if isinstance(window_per_cell, bool) or int(window_per_cell) < 1:
            raise ValueError("window_per_cell must be a positive integer")
        for name, value in (
            ("low_psnr_gain", low_psnr_gain),
            ("regression_gain", regression_gain),
            ("regression_tolerance_db", regression_tolerance_db),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            isinstance(max_weight, bool)
            or not math.isfinite(float(max_weight))
            or float(max_weight) < 1.0
        ):
            raise ValueError("max_weight must be finite and at least 1")
        self._min_history = int(min_history)
        self._minimum_samples = int(minimum_cell_samples)
        self._window = int(window_per_cell)
        self._low_gain = float(low_psnr_gain)
        self._regression_gain = float(regression_gain)
        self._regression_tolerance = float(regression_tolerance_db)
        self._max_weight = float(max_weight)
        # Off by default (issue #1259): default remains the original
        # per-source (frame-averaged) actuation. When True, a candidate
        # replay frame's weight is instead the sample-count-weighted average
        # of its VISIBLE CELLS' individual debt, so one catastrophically bad
        # cell inside an otherwise-healthy frame is no longer diluted away
        # by that frame's good cells before the debt comparison happens.
        self._cell_debt_actuation = bool(cell_debt_actuation)
        self._cell_size_m: float | None = None
        self._histories: dict[tuple[SourceIdentity, int], deque[float]] = {}
        self._peaks: dict[tuple[SourceIdentity, int], float] = {}
        self._source_cells: dict[SourceIdentity, dict[int, int]] = {}
        self._source_quality: dict[SourceIdentity, float] = {}
        self._pair_regression_debt: dict[tuple[SourceIdentity, int], float] = {}
        self._pair_drops: dict[tuple[SourceIdentity, int], float] = {}
        self._history_samples = 0
        # Telemetry-only state (always tracked, independent of actuation
        # mode): cumulative supervised sample count per cell and the PSNR
        # at first observation, so gain_per_exposure can be derived without
        # a second run.
        self._cell_cumulative_samples: dict[int, int] = {}
        self._cell_first_psnr: dict[int, float] = {}

    def active_source_cell_keys(self) -> dict[SourceIdentity, tuple[int, ...]]:
        """Covered cell keys for every source with at least one live sample.

        Design note (#1723): only covers sources that have actually been
        rendered as a training-primary sample at least once (``_source_cells``
        is populated by ``update()``, never by a geometry-only lookup), so it
        is cheap -- a dict copy, no camera resolution or depth fetch -- but it
        cannot answer for a cached/evicted source that predates this
        controller's own history. That gap is intentional: a cached source's
        coverage has to come from a geometry-only resolver elsewhere (e.g.
        ``CanonicalMappingRuntime.covered_cell_keys_for_sources``), since this
        controller has no camera/tracker access of its own.
        """
        return {
            source: tuple(sorted(cells))
            for source, cells in self._source_cells.items()
        }

    def current_cell_quality(self) -> tuple[dict[int, float], dict[int, int]]:
        """Read-only snapshot of ``(cell_psnr, cell_view_counts)`` (#1723).

        Same "established" gate and per-pair -> per-cell median reduction
        ``update()`` uses internally (see its ``current_pairs``/``cell_values``
        block), exposed standalone so an external actuation decision (e.g.
        ``decide_spatial_partition_focus``) can read this controller's live
        cell-quality state without needing a fresh sample to trigger it.
        Callable any time; never mutates controller state.
        """
        current_pairs = {
            pair: history[-1]
            for pair, history in self._histories.items()
            if len(history) >= self._min_history
        }
        cell_values: dict[int, list[float]] = {}
        for (_source, key), value in current_pairs.items():
            cell_values.setdefault(key, []).append(value)
        cell_psnr = {key: float(median(values)) for key, values in cell_values.items()}
        cell_view_counts = {key: len(values) for key, values in cell_values.items()}
        return cell_psnr, cell_view_counts

    def update(
        self,
        sample: SpatialMicrocellQualitySample,
        sources: Sequence[SourceIdentity],
    ) -> SpatialMicrocellReplayFocusDecision:
        """Update spatial microcell replay weights from quality sample."""
        if not isinstance(sample, SpatialMicrocellQualitySample):
            raise TypeError("sample must be SpatialMicrocellQualitySample")
        sources = tuple(sources)
        if not all(isinstance(source, SourceIdentity) for source in sources):
            raise TypeError("replay focus sources must be SourceIdentity values")
        if sample.source not in sources:
            raise ValueError("spatial sample source must remain replay eligible")
        if self._cell_size_m is None:
            self._cell_size_m = sample.cell_size_m
        elif not math.isclose(self._cell_size_m, sample.cell_size_m):
            raise ValueError("spatial replay focus cell size cannot change")

        # Prune per-(source, cell) and per-source state for identities no
        # longer in the replay-candidate pool. Unlike the sibling temporal
        # PartitionReplayFocusController -- whose peak/debt state is a
        # fixed-size list indexed by partition id, inherently bounded --
        # this controller's state is keyed by SourceIdentity (one entry per
        # unique frame ever observed), which grows without bound over a
        # long run if never pruned. Left unpruned, quality_target =
        # median(self._source_quality.values()) is biased forever by every
        # stale, long-retired source instead of reflecting the current
        # candidate pool, and the four (source, cell)-keyed dicts leak
        # memory indefinitely. Cell-keyed telemetry (_cell_cumulative_
        # samples, _cell_first_psnr) is intentionally NOT pruned here --
        # cells are a fixed spatial grid, not a growing per-frame identity,
        # and that state is meant to accumulate for the life of the run.
        eligible_sources = frozenset(sources)
        stale_sources = self._source_quality.keys() - eligible_sources
        if stale_sources:
            for source in stale_sources:
                self._source_quality.pop(source, None)
                self._source_cells.pop(source, None)
            for pair in tuple(self._histories):
                if pair[0] in stale_sources:
                    del self._histories[pair]
                    self._peaks.pop(pair, None)
                    self._pair_regression_debt.pop(pair, None)
                    self._pair_drops.pop(pair, None)

        self._source_cells[sample.source] = {
            cell.cell_key: cell.n_samples for cell in sample.cells
        }
        supported_sample = []
        for cell in sample.cells:
            # Telemetry: track cumulative exposure and first-observed PSNR
            # for every cell that ever gets a supported measurement, so
            # gain_per_exposure = (current - first) / cumulative_samples is
            # derivable from the jsonl trace alone. Independent of whether
            # this cell clears minimum_cell_samples on THIS update.
            if cell.n_samples >= self._minimum_samples:
                self._cell_cumulative_samples[cell.cell_key] = (
                    self._cell_cumulative_samples.get(cell.cell_key, 0)
                    + cell.n_samples
                )
                self._cell_first_psnr.setdefault(cell.cell_key, cell.psnr_db)
            if cell.n_samples < self._minimum_samples:
                continue
            pair = (sample.source, cell.cell_key)
            history = self._histories.setdefault(
                pair, deque(maxlen=self._window)
            )
            history.append(cell.psnr_db)
            supported_sample.append(cell)
        if supported_sample:
            sample_count = float(sum(cell.n_samples for cell in supported_sample))
            self._source_quality[sample.source] = sum(
                cell.psnr_db * cell.n_samples for cell in supported_sample
            ) / sample_count
        self._history_samples += 1

        current_pairs = {
            # React immediately to a fading cell once it is established;
            # history is the establishment/hysteresis gate, not a moving
            # average that would hide the very regression being controlled.
            pair: history[-1]
            for pair, history in self._histories.items()
            if len(history) >= self._min_history
        }
        for pair, value in current_pairs.items():
            self._peaks[pair] = max(self._peaks.get(pair, value), value)
            drop = max(0.0, self._peaks[pair] - value)
            self._pair_drops[pair] = drop
            self._pair_regression_debt[pair] = self._regression_gain * max(
                0.0, drop - self._regression_tolerance
            )

        cell_values: dict[int, list[float]] = {}
        cell_drops: dict[int, list[float]] = {}
        for (source, key), value in current_pairs.items():
            cell_values.setdefault(key, []).append(value)
            cell_drops.setdefault(key, []).append(
                self._pair_drops.get((source, key), 0.0)
            )
        cell_current: dict[int, float] = {
            key: float(median(values)) for key, values in cell_values.items()
        }

        # Absolute frame difficulty may still shift replay toward a weak view,
        # but temporal regression is never compared across cameras.  The same
        # physical cell can have very different visibility/texture from two
        # viewpoints; only (source, cell) is a valid fading identity.
        quality_target = (
            float(median(self._source_quality.values()))
            if self._source_quality
            else 0.0
        )
        # Cell-level target for cell-debt actuation: the median across
        # currently-observed CELLS rather than across per-source averages.
        # A persistently-bad cell's own debt is measured against the same
        # distribution the frame-averaged mode uses, so switching modes is a
        # pure actuation-granularity change, not also a re-definition of
        # "weak".
        cell_quality_target = (
            float(median(cell_current.values())) if cell_current else quality_target
        )

        weights = []
        for source in sources:
            incidence = self._source_cells.get(source, {})
            supported = tuple(
                (key, count)
                for key, count in incidence.items()
                if (source, key) in self._pair_regression_debt
            )
            if self._cell_debt_actuation and supported:
                # Per-cell debt = low-gap (this cell's own current value vs.
                # the cell-level target) + regression debt (already
                # per-pair). Averaging DEBT across the frame's visible cells
                # -- rather than averaging QUALITY first and computing one
                # frame-level debt -- keeps a single catastrophically bad
                # cell's contribution intact instead of diluting it against
                # co-visible good cells (issue #1259 finding 3).
                total = float(sum(count for _, count in supported))
                debt_sum = 0.0
                for key, count in supported:
                    pair_value = self._histories[(source, key)][-1]
                    low_gap = self._low_gain * max(
                        0.0, cell_quality_target - pair_value
                    )
                    debt_sum += (
                        low_gap + self._pair_regression_debt[(source, key)]
                    ) * count
                weight = min(self._max_weight, 1.0 + debt_sum / total)
            else:
                regression_debt = 0.0
                if supported:
                    total = float(sum(count for _, count in supported))
                    regression_debt = sum(
                        self._pair_regression_debt[(source, key)] * count
                        for key, count in supported
                    ) / total
                low_debt = self._low_gain * max(
                    0.0,
                    quality_target - self._source_quality.get(source, quality_target),
                )
                weight = min(self._max_weight, 1.0 + low_debt + regression_debt)
            weights.append((source, float(weight)))

        active = any(weight > 1.0 for _, weight in weights)
        return SpatialMicrocellReplayFocusDecision(
            active=active,
            reason=("spatial_microcell_focus" if active else "spatial_microcell_balanced"),
            weights=tuple(weights),
            cell_psnr_db=tuple(
                (key, value) for key, value in sorted(cell_current.items())
            ),
            cell_drop_from_peak_db=tuple(
                (key, round(float(max(values)), 12))
                for key, values in sorted(cell_drops.items())
            ),
            history_samples=self._history_samples,
            cell_cumulative_samples=tuple(
                sorted(self._cell_cumulative_samples.items())
            ),
            cell_gain_since_first_observed_db=tuple(
                (key, round(cell_current[key] - self._cell_first_psnr[key], 12))
                for key in sorted(cell_current)
                if key in self._cell_first_psnr
            ),
            actuation_mode=(
                "cell_debt" if self._cell_debt_actuation else "frame_averaged"
            ),
        )


####################################################################
# Issue #1318: fast source-complete cell-debt replay focus
#
# Canonical design note: docs/architecture/2026-08-30-spatial-replay-
# focus-fast-refinement.md (sections A-F). This is a NEW, additive
# mechanism alongside SpatialMicrocellReplayFocusController above (which
# is left untouched, including its existing tests): rather than further
# overloading that controller's two actuation modes, the direct
# cell-debt design gets its own incidence index, debt ledger, and
# controller so the two mechanisms cannot silently interact and every
# existing assertion about the frame-averaged/cell_debt_actuation
# controller keeps holding exactly as before.
#
# IMPLEMENTED-tier scope (issue #202 policy): this section is a
# deterministic software contract only. It does not claim any
# quality/performance improvement -- that requires a separate
# ResearchFlow-managed matched experiment (issue #1318's "Required
# experiment" section).
####################################################################

from collections import OrderedDict, deque as _deque


@dataclass(frozen=True, slots=True)
class WorldCellIncidenceSample:
    """One source's visible world-microcells, derived from pose+depth only.

    Never touches RGB, PSNR, holdout data, or current Gaussian geometry --
    see :func:`world_cell_incidence_from_pose_depth`. This is the causal
    basis for section A: the incidence graph can be built the moment a
    source frame's pose+depth are ingested, independent of whether/when a
    training render happens to be scored against it.
    """

    source: SourceIdentity
    cell_size_m: float
    cell_keys: tuple[int, ...]
    valid_pixel_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceIdentity):
            raise TypeError("source must be SourceIdentity")
        if (
            isinstance(self.cell_size_m, bool)
            or not math.isfinite(float(self.cell_size_m))
            or float(self.cell_size_m) <= 0.0
        ):
            raise ValueError("cell_size_m must be finite and positive")
        keys = tuple(int(key) for key in self.cell_keys)
        counts = tuple(int(count) for count in self.valid_pixel_counts)
        if len(keys) != len(counts):
            raise ValueError("cell_keys and valid_pixel_counts must be parallel")
        if len(set(keys)) != len(keys):
            raise ValueError("an incidence sample cannot repeat a cell key")
        if any(count < 0 for count in counts):
            raise ValueError("valid_pixel_counts must be non-negative")
        object.__setattr__(self, "cell_size_m", float(self.cell_size_m))
        object.__setattr__(self, "cell_keys", keys)
        object.__setattr__(self, "valid_pixel_counts", counts)


def world_cell_incidence_from_pose_depth(
    *,
    source: SourceIdentity,
    tracker: Any,
    camera: Any,
    sensor_depth: Any,
) -> WorldCellIncidenceSample | None:
    """Section A: source -> visible world-microcell incidence from pose+depth.

    Uses the same fixed sensor/GT-depth + fixed pose basis as
    ``spatial_microcell_quality_from_render`` and never touches a render,
    ground-truth image, holdout data, or current Gaussian geometry. Dense
    pixel maps are transaction-scoped rather than cached per source.
    """

    if not isinstance(source, SourceIdentity):
        raise TypeError("source must be SourceIdentity")
    measured = _spatial_cell_reduction(
        camera=camera, tracker=tracker, sensor_depth=sensor_depth
    )
    if not measured:
        return None
    return WorldCellIncidenceSample(
        source=source,
        cell_size_m=float(tracker.grid.cell_size),
        cell_keys=tuple(key for key, _sum, _count in measured),
        valid_pixel_counts=tuple(count for _key, _sum, count in measured),
    )


class WorldCellIncidenceIndex:
    """Bounded CPU-side ``source <-> world microcell`` candidate graph.

    Section A. Built only from :class:`WorldCellIncidenceSample` values
    (pose+depth derived). Both directions are kept bounded so the index
    cannot grow without limit over an arbitrarily long run:

    - ``source -> cells`` is an LRU-bounded map keyed by source identity
      (one entry per unique frame ever ingested);
    - ``cell -> candidate sources`` keeps only the most recently observed
      ``max_candidate_sources_per_cell`` sources per physical cell -- a
      bounded candidate list is sufficient for scoring/telemetry and does
      not need perfect historical completeness.
    """

    def __init__(
        self,
        *,
        max_sources: int = 8192,
        max_candidate_sources_per_cell: int = 64,
    ) -> None:
        if isinstance(max_sources, bool) or int(max_sources) < 1:
            raise ValueError("max_sources must be a positive integer")
        if (
            isinstance(max_candidate_sources_per_cell, bool)
            or int(max_candidate_sources_per_cell) < 1
        ):
            raise ValueError(
                "max_candidate_sources_per_cell must be a positive integer"
            )
        self._max_sources = int(max_sources)
        self._max_candidates = int(max_candidate_sources_per_cell)
        self._source_cells: "OrderedDict[SourceIdentity, tuple[int, ...]]" = (
            OrderedDict()
        )
        self._cell_sources: dict[int, _deque[SourceIdentity]] = {}
        self._cell_source_members: dict[int, set[SourceIdentity]] = {}

    def record(self, sample: WorldCellIncidenceSample) -> None:
        """Record a source's world cell incidence."""
        if not isinstance(sample, WorldCellIncidenceSample):
            raise TypeError("sample must be WorldCellIncidenceSample")
        self._source_cells.pop(sample.source, None)
        self._source_cells[sample.source] = sample.cell_keys
        self._source_cells.move_to_end(sample.source)
        while len(self._source_cells) > self._max_sources:
            evicted, _ = self._source_cells.popitem(last=False)
            for key in self._cell_sources:
                self._cell_source_members.get(key, set()).discard(evicted)
        for key in sample.cell_keys:
            queue = self._cell_sources.setdefault(
                key, _deque(maxlen=self._max_candidates)
            )
            members = self._cell_source_members.setdefault(key, set())
            if sample.source not in members:
                if len(queue) == queue.maxlen:
                    dropped = queue[0]
                    members.discard(dropped)
                queue.append(sample.source)
                members.add(sample.source)

    def prune_source(self, source: SourceIdentity) -> None:
        """Drop a retired source from the forward index (mirrors the
        sibling SpatialMicrocellReplayFocusController's eligible-source
        pruning). The bounded reverse (cell -> candidate sources) index is
        left as-is: it is a candidate list, not an actuation identity
        list, and callers should intersect it against the current
        replay-eligible set at query time."""

        self._source_cells.pop(source, None)

    def cells_for_source(self, source: SourceIdentity) -> tuple[int, ...]:
        """Return the world cell keys for a source."""
        return self._source_cells.get(source, ())

    def sources_for_cell(
        self,
        cell_key: int,
        eligible: "frozenset[SourceIdentity] | None" = None,
    ) -> tuple[SourceIdentity, ...]:
        """Return candidate sources for a world cell, optionally filtered by eligibility."""
        queue = self._cell_sources.get(int(cell_key), ())
        if eligible is None:
            return tuple(queue)
        return tuple(source for source in queue if source in eligible)

    def known_source_count(self) -> int:
        """Return the number of sources in the index."""
        return len(self._source_cells)

    def known_cell_count(self) -> int:
        """Return the number of cells in the index."""
        return len(self._cell_sources)


@dataclass
class _CellDebtState:
    """One physical cell's direct-debt bookkeeping (Section C). Owned
    exclusively by :class:`CellDebtLedger` -- see :class:`_CellExposureCounters`
    for the disjoint set of fields :class:`SpatialCellDebtReplayFocusController`
    tracks about the same cell."""

    psnr_current: float | None = None
    psnr_peak: float | None = None


class CellDebtLedger:
    """Bounded per-cell direct debt (Section C):

        D_c = D_abs(c) + gamma * D_reg(c)
        D_abs(c) = max(0, P_target - P_c)
        D_reg(c) = max(0, P_peak_c - P_c - tolerance)

    ``target_psnr_db`` is a fixed training-side policy constant supplied
    by the caller (never a holdout-derived or adaptive threshold), so a
    cell that has never been good still accrues debt purely from
    ``D_abs`` -- there is no cross-cell averaging step that could dilute
    or hide a single persistently-bad cell.

    State is an LRU-bounded map keyed by physical cell, capped at
    ``max_cells`` so total memory/work stays bounded no matter how long a
    run's history grows (issue #1318 acceptance criterion 9); this
    mirrors the exact class of unbounded-per-identity-history bug PR
    #1712 fixed for SpatialMicrocellReplayFocusController's per-source
    state.
    """

    def __init__(
        self,
        *,
        target_psnr_db: float,
        gamma: float = 1.0,
        tolerance_db: float = 0.25,
        max_debt: float = 20.0,
        max_cells: int = 20000,
    ) -> None:
        if isinstance(target_psnr_db, bool) or not math.isfinite(
            float(target_psnr_db)
        ):
            raise ValueError("target_psnr_db must be finite")
        for name, value in (("gamma", gamma), ("tolerance_db", tolerance_db)):
            if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(max_debt, bool) or not math.isfinite(float(max_debt)) or max_debt <= 0:
            raise ValueError("max_debt must be finite and positive")
        if isinstance(max_cells, bool) or int(max_cells) < 1:
            raise ValueError("max_cells must be a positive integer")
        self._target = float(target_psnr_db)
        self._gamma = float(gamma)
        self._tolerance = float(tolerance_db)
        self._max_debt = float(max_debt)
        self._max_cells = int(max_cells)
        self._states: "OrderedDict[int, _CellDebtState]" = OrderedDict()

    def _touch(self, cell_key: int) -> _CellDebtState:
        key = int(cell_key)
        state = self._states.pop(key, None)
        if state is None:
            state = _CellDebtState()
        self._states[key] = state
        while len(self._states) > self._max_cells:
            self._states.popitem(last=False)
        return state

    def peek(self, cell_key: int) -> _CellDebtState | None:
        """Return the debt state for a cell, or None if unknown."""
        return self._states.get(int(cell_key))

    def known_cell_count(self) -> int:
        """Return the number of cells with known debt state."""
        return len(self._states)

    def update_quality(self, cell_key: int, psnr_db: float) -> None:
        """Update the quality peak and current value for a cell."""
        if isinstance(psnr_db, bool) or not math.isfinite(float(psnr_db)):
            raise ValueError("psnr_db must be finite")
        state = self._touch(cell_key)
        value = float(psnr_db)
        state.psnr_peak = value if state.psnr_peak is None else max(state.psnr_peak, value)
        state.psnr_current = value

    def debt(self, cell_key: int) -> float:
        """0.0 for a cell with no quality observation yet (neutral/unknown,
        never falsely healthy nor falsely maximal -- see
        :meth:`bootstrap_seed`, which is how section F avoids leaving a
        known-but-unscored cell at this neutral zero for thousands of
        updates)."""

        state = self._states.get(int(cell_key))
        if state is None or state.psnr_current is None:
            return 0.0
        abs_debt = max(0.0, self._target - state.psnr_current)
        peak = state.psnr_peak if state.psnr_peak is not None else state.psnr_current
        reg_debt = max(0.0, peak - state.psnr_current - self._tolerance)
        return min(self._max_debt, abs_debt + self._gamma * reg_debt)

    def bootstrap_seed(self, cell_key: int, psnr_db: float) -> None:
        """Section F: seed a cell's debt from an early training residual
        (e.g. the first post-source-complete render that happens to cover
        it) instead of waiting for the normal window/history warm-up this
        ledger otherwise needs none of -- a single observation is already
        sufficient for :meth:`debt` to return a non-zero value. Must never
        be called with holdout/evaluation evidence."""

        self.update_quality(cell_key, psnr_db)


def aggregate_cell_debt(
    debts: Sequence[float],
    *,
    method: str = "mean",
    top_k: int = 1,
) -> float:
    """Bounded aggregation ``A`` over one source's visible-cell debts
    (Section D). ``top_k``/``max`` keep one severe cell from being diluted
    by an otherwise-good frame average; ``mean`` is the more conservative
    default."""

    values = [float(value) for value in debts]
    if not values:
        return 0.0
    if method == "mean":
        return sum(values) / len(values)
    if method == "max":
        return max(values)
    if method == "top_k":
        k = max(1, min(int(top_k), len(values)))
        return sum(sorted(values, reverse=True)[:k]) / k
    raise ValueError(f"unknown cell-debt aggregation method {method!r}")


def cell_debt_source_weight(
    cell_keys: Sequence[int],
    ledger: CellDebtLedger,
    *,
    lambda_: float,
    aggregation: str = "mean",
    top_k: int = 1,
    max_weight: float = 8.0,
) -> float:
    """``W(v) = 1 + lambda * A({D_c | c visible in v})`` (Section D)."""

    if isinstance(lambda_, bool) or not math.isfinite(float(lambda_)) or lambda_ < 0:
        raise ValueError("lambda_ must be finite and non-negative")
    debts = [ledger.debt(key) for key in cell_keys]
    aggregated = aggregate_cell_debt(debts, method=aggregation, top_k=top_k)
    return min(float(max_weight), 1.0 + float(lambda_) * aggregated)


def mixed_replay_probability(
    p_coverage: float,
    p_cell_debt: float,
    *,
    lam: float,
    uniform_probability: float | None = None,
    cap_multiplier: float | None = 2.0,
) -> float:
    """``p(v) = (1 - lambda) * p_coverage(v) + lambda * p_cell_debt(v)``
    (Section E), with an optional conservative cap (default ~2x a
    uniform-equivalent probability, matching the design doc's initial
    qualification guidance) so cell-debt bias cannot make one source
    dominate the coverage-preserving base distribution regardless of how
    large its debt is.
    """

    if isinstance(lam, bool) or not math.isfinite(float(lam)) or not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    for name, value in (("p_coverage", p_coverage), ("p_cell_debt", p_cell_debt)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    mixed = (1.0 - float(lam)) * float(p_coverage) + float(lam) * float(p_cell_debt)
    if uniform_probability is not None and cap_multiplier is not None:
        if (
            isinstance(uniform_probability, bool)
            or not math.isfinite(float(uniform_probability))
            or uniform_probability < 0.0
        ):
            raise ValueError("uniform_probability must be finite and non-negative")
        if (
            isinstance(cap_multiplier, bool)
            or not math.isfinite(float(cap_multiplier))
            or cap_multiplier < 1.0
        ):
            raise ValueError("cap_multiplier must be finite and >= 1")
        mixed = min(mixed, float(cap_multiplier) * float(uniform_probability))
    return mixed


@dataclass(frozen=True, slots=True)
class CellExposureRecord:
    """Section B per-cell exposure/quality telemetry, one row per known cell."""

    cell_key: int
    candidate_sources_covering_cell: int
    selected_replay_views_covering_cell: int
    cumulative_supervised_pixels: int
    optimizer_transactions_touching_cell: int
    optimizer_updates_since_last_supervision: int
    cumulative_focus_weight: float
    psnr_before_exposure: float | None
    psnr_after_exposure: float | None
    debt: float


@dataclass(frozen=True, slots=True)
class SpatialCellDebtFocusDecision:
    """Immutable spatial cell-debt replay-weight decision and telemetry."""

    active: bool
    reason: str
    weights: tuple[tuple[SourceIdentity, float], ...]
    cell_exposure: tuple[CellExposureRecord, ...]
    history_updates: int
    score_scope: str = "training_primary_spatial_pre_update"


@dataclass
class _CellExposureCounters:
    """One physical cell's Section B exposure bookkeeping, owned exclusively
    by :class:`SpatialCellDebtReplayFocusController`. Deliberately disjoint
    from :class:`_CellDebtState` (which owns quality/peak/debt): a cell's
    debt is a property of the shared :class:`CellDebtLedger`, not of this
    controller's own per-cell counters."""

    psnr_first: float | None = None
    psnr_current: float | None = None
    selected_replay_views: int = 0
    cumulative_supervised_pixels: int = 0
    optimizer_transactions: int = 0
    updates_since_last_supervision: int = 0
    cumulative_focus_weight: float = 0.0


class SpatialCellDebtReplayFocusController:
    """Section D/E/F controller: cell-debt-derived, incidence-bounded replay
    source weighting.

    This is additive to (not a replacement for) the existing
    ``SpatialMicrocellReplayFocusController`` above. It consumes:

    - a :class:`WorldCellIncidenceIndex` (Section A), built independently
      from pose+depth at ingestion time -- the controller never builds
      its own incidence, so it can be fed a graph that is already
      complete by source completion;
    - training-only :class:`SpatialMicrocellQualitySample` observations
      (the same evidence type ``SpatialMicrocellReplayFocusController``
      consumes), never holdout evidence.

    Every identity starts at weight 1.0; cell debt can only raise it,
    preserving the existing full-cache provider's coverage guarantee the
    same way the sibling controller does.
    """

    def __init__(
        self,
        *,
        incidence: WorldCellIncidenceIndex,
        target_psnr_db: float,
        gamma: float = 1.0,
        tolerance_db: float = 0.25,
        lam: float = 0.5,
        aggregation: str = "top_k",
        top_k: int = 1,
        max_weight: float = 8.0,
        max_debt: float = 20.0,
        max_tracked_cells: int = 20000,
    ) -> None:
        if not isinstance(incidence, WorldCellIncidenceIndex):
            raise TypeError("incidence must be WorldCellIncidenceIndex")
        if isinstance(lam, bool) or not math.isfinite(float(lam)) or not (0.0 <= lam <= 1.0):
            raise ValueError("lam must be in [0, 1]")
        if aggregation not in ("mean", "max", "top_k"):
            raise ValueError(f"unknown cell-debt aggregation method {aggregation!r}")
        self._incidence = incidence
        self._lam = float(lam)
        self._aggregation = aggregation
        self._top_k = int(top_k)
        self._max_weight = float(max_weight)
        self._ledger = CellDebtLedger(
            target_psnr_db=target_psnr_db,
            gamma=gamma,
            tolerance_db=tolerance_db,
            max_debt=max_debt,
            max_cells=max_tracked_cells,
        )
        self._exposure: "OrderedDict[int, _CellExposureCounters]" = OrderedDict()
        self._max_tracked_cells = int(max_tracked_cells)
        self._history_updates = 0

    def _exposure_state(self, cell_key: int) -> _CellExposureCounters:
        key = int(cell_key)
        state = self._exposure.pop(key, None)
        if state is None:
            state = _CellExposureCounters()
        self._exposure[key] = state
        while len(self._exposure) > self._max_tracked_cells:
            self._exposure.popitem(last=False)
        return state

    def known_cell_count(self) -> int:
        """Return the number of cells with known exposure state."""
        return len(self._exposure)

    def bootstrap_from_incidence(
        self, early_residuals: Mapping[int, float]
    ) -> None:
        """Section F: seed debt for every cell already known from the
        incidence graph using the earliest available TRAINING residual
        observation, instead of waiting for organic replay warm-up. Cells
        with no residual yet are left at the ledger's neutral/unknown
        state (0.0 debt), not falsely marked healthy or maximal -- they
        simply have not been reached yet."""

        for cell_key, psnr_db in early_residuals.items():
            if isinstance(psnr_db, bool) or not math.isfinite(float(psnr_db)):
                raise ValueError("bootstrap residual PSNR must be finite")
            self._ledger.bootstrap_seed(cell_key, float(psnr_db))
            state = self._exposure_state(cell_key)
            if state.psnr_first is None:
                state.psnr_first = float(psnr_db)
            state.psnr_current = float(psnr_db)

    def update(
        self,
        sample: SpatialMicrocellQualitySample,
        sources: Sequence[SourceIdentity],
        *,
        holdout: bool = False,
    ) -> SpatialCellDebtFocusDecision:
        """Update spatial cell debt replay weights from quality sample."""
        if holdout:
            # Section F / acceptance criterion 8: holdout evidence must
            # never enter this actuation path. Fail loudly rather than
            # silently ignoring the flag.
            raise ValueError(
                "holdout/evaluation evidence must never enter cell-debt "
                "replay actuation"
            )
        if not isinstance(sample, SpatialMicrocellQualitySample):
            raise TypeError("sample must be SpatialMicrocellQualitySample")
        sources = tuple(sources)
        if not all(isinstance(source, SourceIdentity) for source in sources):
            raise TypeError("replay focus sources must be SourceIdentity values")
        if sample.source not in sources:
            raise ValueError("spatial sample source must remain replay eligible")

        self._history_updates += 1
        touched = set()
        for cell in sample.cells:
            key = int(cell.cell_key)
            touched.add(key)
            self._ledger.update_quality(key, cell.psnr_db)
            state = self._exposure_state(key)
            state.selected_replay_views += 1
            state.cumulative_supervised_pixels += int(cell.n_samples)
            state.optimizer_transactions += 1
            state.updates_since_last_supervision = 0
            if state.psnr_first is None:
                state.psnr_first = float(cell.psnr_db)
            state.psnr_current = float(cell.psnr_db)

        # Age every OTHER already-known cell's since-last-supervision
        # counter. Bounded by known_cell_count(), not by run length.
        for key, state in self._exposure.items():
            if key not in touched:
                state.updates_since_last_supervision += 1

        eligible = frozenset(sources)
        weights = []
        for source in sources:
            cells = self._incidence.cells_for_source(source)
            if not cells:
                weights.append((source, 1.0))
                continue
            weight = cell_debt_source_weight(
                cells,
                self._ledger,
                lambda_=self._lam,
                aggregation=self._aggregation,
                top_k=self._top_k,
                max_weight=self._max_weight,
            )
            weights.append((source, weight))
            for key in cells:
                state = self._exposure_state(key)
                state.cumulative_focus_weight += weight

        active = any(weight > 1.0 for _, weight in weights)
        exposure = tuple(
            CellExposureRecord(
                cell_key=key,
                candidate_sources_covering_cell=len(
                    self._incidence.sources_for_cell(key, eligible)
                ),
                selected_replay_views_covering_cell=state.selected_replay_views,
                cumulative_supervised_pixels=state.cumulative_supervised_pixels,
                optimizer_transactions_touching_cell=state.optimizer_transactions,
                optimizer_updates_since_last_supervision=(
                    state.updates_since_last_supervision
                ),
                cumulative_focus_weight=state.cumulative_focus_weight,
                psnr_before_exposure=state.psnr_first,
                psnr_after_exposure=state.psnr_current,
                debt=self._ledger.debt(key),
            )
            for key, state in sorted(self._exposure.items())
        )
        return SpatialCellDebtFocusDecision(
            active=active,
            reason=("spatial_cell_debt_focus" if active else "spatial_cell_debt_balanced"),
            weights=tuple(weights),
            cell_exposure=exposure,
            history_updates=self._history_updates,
        )


_CELL_DEBT_EXPOSURE_FILENAME = "spatial_cell_debt_focus.jsonl"


def record_cell_debt_exposure(
    decision: SpatialCellDebtFocusDecision,
    *,
    iteration: int,
    model_path: "str",
) -> "Any":
    """Append one Section B telemetry row for ``decision`` to
    ``spatial_cell_debt_focus.jsonl`` under ``model_path``, mirroring the
    existing ``spatial_replay_focus.jsonl`` convention used by
    ``SpatialMicrocellReplayFocusController``/``plot_spatial_replay_focus``.

    This is a standalone, opt-in writer: it is not called by the causal
    runtime's ingestion/replay loop in this change (see the #1318 PR body
    for what remains as follow-up wiring). It exists so the mechanism's
    telemetry is genuinely machine-readable and plottable end-to-end
    on its own, ahead of that wiring.
    """

    from pathlib import Path as _Path

    weights = tuple(float(weight) for _, weight in decision.weights)
    row = {
        "iteration": int(iteration),
        "active": bool(decision.active),
        "reason": decision.reason,
        "score_scope": decision.score_scope,
        "history_updates": int(decision.history_updates),
        "replay_population": len(weights),
        "weight_min": min(weights, default=1.0),
        "weight_max": max(weights, default=1.0),
        "weight_mean": sum(weights) / len(weights) if weights else 1.0,
        "cells": [
            {
                "cell_key": int(row_.cell_key),
                "candidate_sources_covering_cell": int(
                    row_.candidate_sources_covering_cell
                ),
                "selected_replay_views_covering_cell": int(
                    row_.selected_replay_views_covering_cell
                ),
                "cumulative_supervised_pixels": int(
                    row_.cumulative_supervised_pixels
                ),
                "optimizer_transactions_touching_cell": int(
                    row_.optimizer_transactions_touching_cell
                ),
                "optimizer_updates_since_last_supervision": int(
                    row_.optimizer_updates_since_last_supervision
                ),
                "cumulative_focus_weight": float(row_.cumulative_focus_weight),
                "psnr_before_exposure": row_.psnr_before_exposure,
                "psnr_after_exposure": row_.psnr_after_exposure,
                "debt": float(row_.debt),
            }
            for row_ in decision.cell_exposure
        ],
    }
    path = _Path(model_path) / _CELL_DEBT_EXPOSURE_FILENAME
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def plot_spatial_cell_debt_exposure(model_path: "str"):
    """Section B diagnostic: exposure vs. debt/gain scatter, answering the
    design doc's discriminator table (section 11):

        low exposure + high debt  -> scheduler/replay allocation problem
        high exposure + high debt -> mobility/representation problem
        high exposure + low debt  -> controller productive

    Reads ``spatial_cell_debt_focus.jsonl`` (see :func:`record_cell_debt_
    exposure`) and plots each cell's most recent row exactly once, capped
    for readability the same way ``plot_spatial_replay_focus`` caps its
    heatmap rows.
    """

    from pathlib import Path as _Path

    root = _Path(model_path)
    trace = root / _CELL_DEBT_EXPOSURE_FILENAME
    if not trace.is_file():
        return None
    rows = [
        json.loads(line) for line in trace.read_text().splitlines() if line.strip()
    ]
    latest_by_cell: dict[int, dict] = {}
    for row in rows:
        for cell in row.get("cells", ()):
            latest_by_cell[int(cell["cell_key"])] = cell
    if not latest_by_cell:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_points = 2000
    cells = list(latest_by_cell.values())[:max_points]
    exposure = [
        int(cell["selected_replay_views_covering_cell"]) for cell in cells
    ]
    debt = [float(cell["debt"]) for cell in cells]

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(exposure, debt, alpha=0.6, s=14)
    axis.set_xlabel("selected replay views covering cell (cumulative)")
    axis.set_ylabel("current cell debt D_c (dB-equivalent)")
    axis.set_title(
        f"Cell-debt exposure diagnostic (n={len(cells)} cells, #1318)"
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    output = root / "spatial_cell_debt_exposure.svg"
    fig.savefig(svg_output_path(output), dpi=150)
    plt.close(fig)
    return output


__all__ = [
    "PartitionReplayFocusController",
    "PartitionReplayFocusDecision",
    "TrainingPartitionPsnrAccumulator",
    "SpatialCellQuality",
    "SpatialMicrocellQualitySample",
    "SpatialMicrocellReplayFocusController",
    "SpatialMicrocellReplayFocusDecision",
    "spatial_microcell_quality_from_render",
    "WorldCellIncidenceSample",
    "world_cell_incidence_from_pose_depth",
    "WorldCellIncidenceIndex",
    "CellDebtLedger",
    "aggregate_cell_debt",
    "cell_debt_source_weight",
    "mixed_replay_probability",
    "CellExposureRecord",
    "SpatialCellDebtFocusDecision",
    "SpatialCellDebtReplayFocusController",
    "record_cell_debt_exposure",
    "plot_spatial_cell_debt_exposure",
]
