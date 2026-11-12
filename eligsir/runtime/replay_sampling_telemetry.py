"""Post-run replay sampling diagnostics derived from the opt-in decision trace."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from eligsir.plot_style import svg_output_path


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _normalized_entropy(counts: list[int]) -> float:
    total = sum(counts)
    nonzero = [count for count in counts if count > 0]
    if total <= 0 or len(nonzero) <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log(count / total) for count in nonzero
    )
    return float(entropy / math.log(len(nonzero)))


def _gini(counts: list[float]) -> float:
    ordered = sorted(max(0, count) for count in counts)
    total = sum(ordered)
    size = len(ordered)
    if size == 0 or total == 0:
        return 0.0
    weighted = sum((index + 1) * count for index, count in enumerate(ordered))
    return float((2.0 * weighted) / (size * total) - (size + 1.0) / size)


def _source_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": str(record["fingerprint"]),
        "source_ordinal": int(record["source_ordinal"]),
        "timestamp_ns": int(record["timestamp_ns"]),
    }


def _write_heatmap(
    path: Path,
    samples: list[tuple[int, int, bool]],
    frame_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    if samples:
        ordinals = np.asarray([sample[0] for sample in samples], dtype=np.float64)
        iterations = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        ordinal_bins = min(128, max(8, len({int(value) for value in ordinals})))
        iteration_bins = min(128, max(8, len({int(value) for value in iterations})))
        histogram, x_edges, y_edges = np.histogram2d(
            ordinals,
            iterations,
            bins=(ordinal_bins, iteration_bins),
        )
        mesh = axes[0].pcolormesh(
            x_edges,
            y_edges,
            np.log1p(histogram.T),
            shading="flat",
            cmap="magma",
        )
        figure.colorbar(mesh, ax=axes[0], label="log(1 + replay selections)")
    else:
        axes[0].text(0.5, 0.5, "No realized replay selections", ha="center", va="center")
    axes[0].set_xlabel("source frame ordinal")
    axes[0].set_ylabel("optimizer iteration")
    axes[0].set_title("Replay sampling over trajectory and optimization time")

    if frame_rows:
        ordinals = [row["source_ordinal"] for row in frame_rows]
        active = [row["active_replay_count"] for row in frame_rows]
        idle = [row["idle_replay_count"] for row in frame_rows]
        axes[1].bar(ordinals, active, label="new-frame tick", width=1.0)
        axes[1].bar(ordinals, idle, bottom=active, label="idle/replay-only tick", width=1.0)
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No retained frame identities", ha="center", va="center")
    axes[1].set_xlabel("source frame ordinal")
    axes[1].set_ylabel("realized replay selections")
    axes[1].set_title("Per-frame sampling distribution")
    coverage = summary.get("full_replay_coverage_fraction")
    catalog_gini = summary.get("catalog_sampling_gini")
    coverage_text = "unavailable" if coverage is None else f"{coverage:.1%}"
    gini_text = "unavailable" if catalog_gini is None else f"{catalog_gini:.3f}"
    figure.suptitle(
        f"Full retained-catalog replay coverage {coverage_text}, "
        f"catalog sampling Gini {gini_text}"
    )
    figure.savefig(svg_output_path(path), dpi=150)
    plt.close(figure)


def write_replay_sampling_telemetry(
    trace_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate realized replay selections without touching the live hot path."""

    trace = Path(trace_path)
    destination = trace.parent if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    retained: dict[str, dict[str, Any]] = {}
    samples_by_source: dict[str, list[int]] = defaultdict(list)
    sample_ticks_by_source: dict[str, list[int]] = defaultdict(list)
    active_counts: Counter[str] = Counter()
    idle_counts: Counter[str] = Counter()
    weights_by_source: dict[str, list[float]] = defaultdict(list)
    eligible_tick_counts: Counter[str] = Counter()
    eligible_ticks_by_source: dict[str, list[int]] = defaultdict(list)
    uniform_expected_counts: Counter[str] = Counter()
    allocation_counts: Counter[str] = Counter()
    requested_allocation_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    planned_resolution_counts: Counter[str] = Counter()
    freshness_counts: Counter[str] = Counter()
    mutation_totals: Counter[str] = Counter()
    control_phase_counts: Counter[str] = Counter()
    ingress_queue_depths: list[float] = []
    mutation_iterations: list[float] = []
    executed_ticks = 0
    optimizer_steps_total = 0
    supervision_views_total = 0
    mcmc_requested_ticks = 0
    mcmc_effective_ticks = 0
    mcmc_structural_epochs = 0
    live_depth_structural_epochs = 0
    replay_depth_structural_epochs = 0
    mutation_by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    samples: list[tuple[int, int, bool]] = []
    trace_rows = 0
    final_iteration = 0
    catalog_population_samples: list[int] = []
    catalog_sequence_mismatch_rows = 0
    catalog_population_position_ambiguous_rows = 0
    catalog_disposition_rows_without_population = 0
    disposition_membership_provenance: set[str] = set()
    disposition_identity_conflicts = 0
    shortlist_configured_limits: set[int] = set()
    shortlist_realized_populations: list[float] = []
    shortlist_membership_mismatch_ticks = 0
    eligibility_population_mismatch_ticks = 0
    executed_membership_rows = 0
    executed_membership_exact_rows = 0
    candidate_sources_seen: set[str] = set()
    selected_sources_seen: set[str] = set()

    if trace.is_file():
        with trace.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                trace_rows += 1
                iteration = int(row.get("iteration", 0))
                final_iteration = max(final_iteration, iteration)
                snapshot = row.get("snapshot") or {}
                ingress_queue_depths.append(
                    float(snapshot.get("ingress_queue_depth", 0))
                )
                freshness_counts[str(snapshot.get("freshness", "unknown"))] += 1
                evidence = snapshot.get("evidence") or {}
                retained_catalog_evidence = row.get("retained_catalog")
                catalog_population_position = None
                if isinstance(retained_catalog_evidence, dict):
                    catalog_population = retained_catalog_evidence.get("population")
                    catalog_population_position = retained_catalog_evidence.get(
                        "position"
                    )
                else:
                    catalog_population = snapshot.get("retained_catalog_population")
                    if catalog_population is None:
                        catalog_population = evidence.get("replay_catalog_population")
                    catalog_population_position = snapshot.get(
                        "retained_catalog_population_position"
                    )
                if catalog_population is not None:
                    catalog_population_samples.append(int(catalog_population))
                configured_limit = snapshot.get("replay_shortlist_configured_limit")
                if configured_limit is None:
                    configured_limit = evidence.get(
                        "replay_shortlist_configured_limit"
                    )
                if configured_limit is not None:
                    shortlist_configured_limits.add(int(configured_limit))
                realized_population = snapshot.get(
                    "replay_shortlist_realized_population",
                    snapshot.get("replay_candidate_population"),
                )
                if realized_population is not None:
                    shortlist_realized_populations.append(float(realized_population))
                if row.get("resolution_tier") is not None:
                    planned_resolution_counts[str(row["resolution_tier"])] += 1
                mutation = row.get("mutation") or {}
                for name in ("candidates", "inserted", "pruned", "relocated", "corrected"):
                    mutation_totals[name] += int(mutation.get(name, 0))
                mutation_totals["structural_epochs"] += int(
                    bool(mutation.get("structural_mutation", False))
                )

                executed = bool(row.get("executed", False)) and int(
                    row.get("optimizer_steps", 0)
                ) > 0
                if executed:
                    executed_ticks += 1
                    executed_tick = executed_ticks
                    optimizer_steps_total += int(row.get("optimizer_steps", 0))
                    supervision_views_total += int(row.get("supervision_views", 0))
                    reasons = tuple(str(item) for item in row.get("reason_trace", ()))
                    phase = (
                        "post_refinement"
                        if any("post-refinement" in item for item in reasons)
                        else "online_mapping"
                    )
                    control_phase_counts[phase] += 1
                    mutation_requested = int(row.get("mutation_request", 0)) > 0
                    if mutation_requested:
                        mcmc_requested_ticks += 1
                    mutation_events = tuple(mutation.get("events") or ())
                    if mutation_events:
                        for event in mutation_events:
                            kind = str(event.get("kind", "unknown"))
                            for name in (
                                "candidates",
                                "inserted",
                                "pruned",
                                "relocated",
                                "corrected",
                            ):
                                mutation_by_kind[kind][name] += int(
                                    event.get(name, 0)
                                )
                            structural = any(
                                int(event.get(name, 0)) > 0
                                for name in (
                                    "inserted",
                                    "pruned",
                                    "relocated",
                                    "corrected",
                                )
                            )
                            mutation_by_kind[kind]["structural_epochs"] += int(
                                structural
                            )
                            if kind == "mcmc":
                                mcmc_effective_ticks += 1
                                mcmc_structural_epochs += int(structural)
                            elif kind == "live_depth":
                                live_depth_structural_epochs += int(structural)
                            elif kind == "replay_depth":
                                replay_depth_structural_epochs += int(structural)
                    elif mutation_requested:
                        # Compatibility for traces written before exact lane
                        # attribution existed.
                        mcmc_effective_ticks += 1
                    if bool(mutation.get("structural_mutation", False)):
                        mutation_iterations.append(float(iteration))
                        if not mutation_events:
                            if mutation_requested and int(mutation.get("candidates", 0)) == 0:
                                mcmc_structural_epochs += 1
                            elif int(mutation.get("candidates", 0)) > 0:
                                live_depth_structural_epochs += 1
                    if row.get("resolution_tier") is not None:
                        resolution_counts[str(row["resolution_tier"])] += int(
                            row.get("optimizer_steps", 0)
                        )
                    replay_records = [
                        _source_record(record)
                        for record in row.get("replay_source_records", ())
                    ]
                    selected_fingerprints = {
                        record["fingerprint"] for record in replay_records
                    }
                    selected_sources_seen.update(selected_fingerprints)
                    candidate_population = int(
                        (row.get("replay_sampling") or {}).get(
                            "population",
                            snapshot.get(
                                "replay_shortlist_realized_population",
                                snapshot.get("replay_candidate_population", 0),
                            ),
                        )
                    )
                    candidate_records_present = "replay_candidate_records" in snapshot
                    candidate_records = [
                        _source_record(record)
                        for record in snapshot.get("replay_candidate_records", ())
                    ]
                    candidate_fingerprints = tuple(
                        record["fingerprint"] for record in candidate_records
                    )
                    candidate_membership_exact = (
                        candidate_records_present
                        and len(set(candidate_fingerprints))
                        == len(candidate_fingerprints)
                        and candidate_population == len(candidate_fingerprints)
                        and selected_fingerprints.issubset(candidate_fingerprints)
                    )
                    if candidate_records_present:
                        executed_membership_rows += 1
                        candidate_sources_seen.update(candidate_fingerprints)
                        declared_shortlist_population = snapshot.get(
                            "replay_shortlist_realized_population",
                            snapshot.get("replay_candidate_population"),
                        )
                        if (
                            declared_shortlist_population is not None
                            and int(declared_shortlist_population)
                            != len(candidate_fingerprints)
                        ):
                            shortlist_membership_mismatch_ticks += 1
                    if not candidate_membership_exact:
                        eligibility_population_mismatch_ticks += 1
                    else:
                        executed_membership_exact_rows += 1
                    selected_slots = len(replay_records)
                    uniform_probability = (
                        min(1.0, selected_slots / candidate_population)
                        if candidate_population > 0
                        else 0.0
                    )
                    if candidate_membership_exact:
                        for fingerprint in candidate_fingerprints:
                            eligible_tick_counts[fingerprint] += 1
                            eligible_ticks_by_source[fingerprint].append(executed_tick)
                            uniform_expected_counts[fingerprint] += uniform_probability
                    has_new = bool(row.get("supervision_source_records", ()))
                    new_count = len(row.get("supervision_source_records", ()))
                    allocation_counts[
                        f"new={new_count};replay={selected_slots}"
                    ] += 1
                    allocation_request = row.get("allocation_request") or {}
                    requested_allocation_counts[
                        "new="
                        f"{int(allocation_request.get('supervision_slots', 0))};"
                        "replay="
                        f"{int(allocation_request.get('replay_slots', 0))}"
                    ] += 1
                    selected_weights = list(
                        (row.get("replay_sampling") or {}).get(
                            "selected_weights", ()
                        )
                    )
                    for index, record in enumerate(replay_records):
                        fingerprint = record["fingerprint"]
                        samples_by_source[fingerprint].append(iteration)
                        sample_ticks_by_source[fingerprint].append(executed_tick)
                        (active_counts if has_new else idle_counts)[fingerprint] += 1
                        samples.append(
                            (record["source_ordinal"], iteration, has_new)
                        )
                        if index < len(selected_weights):
                            weights_by_source[fingerprint].append(
                                float(selected_weights[index])
                            )

                dispositions = tuple(row.get("dispositions", ()))
                for disposition in dispositions:
                    action = disposition.get("semantic_action")
                    historical = disposition.get("historical_eligibility")
                    provenance = "explicit_v3"
                    if action is None:
                        kind = str(disposition.get("kind", ""))
                        modalities = disposition.get("modalities") or {}
                        if (
                            kind == "admit"
                            and bool(disposition.get("admit_supervision", False))
                        ):
                            action = "commit"
                            historical = "eligible"
                            provenance = "contract_inference_v2"
                        elif (
                            kind == "skip_redundant"
                            and str(disposition.get("residency", "")) == "cold"
                            and not bool(disposition.get("mapping_candidate", False))
                            and (
                                bool(modalities.get("rgb_supervision", False))
                                or bool(modalities.get("depth_supervision", False))
                            )
                        ):
                            action = "defer"
                            historical = "eligible"
                            provenance = "contract_inference_v2"
                    if action not in {"commit", "defer"} or historical != "eligible":
                        continue
                    fingerprint = str(disposition["source"])
                    disposition_membership_provenance.add(provenance)
                    candidate_record = {
                        "fingerprint": fingerprint,
                        "source_ordinal": int(
                            disposition.get("source_ordinal", -1)
                        ),
                        "timestamp_ns": int(disposition.get("timestamp_ns", -1)),
                        "semantic_action": action,
                        "historical_eligibility": historical,
                        "first_terminal_iteration": iteration,
                        "first_admission_iteration": (
                            iteration if action == "commit" else None
                        ),
                    }
                    prior = retained.get(fingerprint)
                    if prior is not None and prior["semantic_action"] != action:
                        disposition_identity_conflicts += 1
                        continue
                    record = retained.setdefault(
                        fingerprint,
                        candidate_record,
                    )
                if catalog_population is None:
                    if dispositions:
                        catalog_disposition_rows_without_population += 1
                elif catalog_population_position == "post_dispositions":
                    if int(catalog_population) != len(retained):
                        catalog_sequence_mismatch_rows += 1
                else:
                    catalog_population_position_ambiguous_rows += 1

    total_selections = sum(len(values) for values in samples_by_source.values())
    explicit_catalog_population = max(catalog_population_samples, default=0)
    retained_catalog_population = max(len(retained), explicit_catalog_population)
    unknown_catalog_sources = (
        candidate_sources_seen | selected_sources_seen
    ).difference(retained)
    catalog_membership_exact = bool(retained) and not (
        disposition_identity_conflicts
        or catalog_sequence_mismatch_rows
        or catalog_population_position_ambiguous_rows
        or unknown_catalog_sources
        or explicit_catalog_population > len(retained)
    )
    catalog_membership_status = (
        "available_exact_terminal_union"
        if catalog_membership_exact
        else "unavailable"
    )
    eligibility_membership_exact = (
        catalog_membership_exact
        and executed_ticks > 0
        and executed_membership_exact_rows > 0
        and executed_membership_exact_rows == executed_ticks
        and eligibility_population_mismatch_ticks == 0
        and shortlist_membership_mismatch_ticks == 0
    )
    eligibility_metrics_status = (
        "available_exact_membership"
        if eligibility_membership_exact
        else "unavailable"
    )

    frame_rows: list[dict[str, Any]] = []
    for fingerprint, record in retained.items():
        frame_samples = sorted(samples_by_source.get(fingerprint, ()))
        weights = weights_by_source.get(fingerprint, ())
        eligible_ticks = int(eligible_tick_counts[fingerprint])
        uniform_expected = float(uniform_expected_counts[fingerprint])
        eligible_tick_sequence = eligible_ticks_by_source.get(fingerprint, ())
        selected_tick_set = set(sample_ticks_by_source.get(fingerprint, ()))
        longest_starvation = 0
        current_starvation = 0
        for eligible_tick in eligible_tick_sequence:
            if eligible_tick in selected_tick_set:
                current_starvation = 0
            else:
                current_starvation += 1
                longest_starvation = max(longest_starvation, current_starvation)
        frame_rows.append({
            **record,
            "replay_count": len(frame_samples),
            "active_replay_count": int(active_counts[fingerprint]),
            "idle_replay_count": int(idle_counts[fingerprint]),
            "selection_share": (
                len(frame_samples) / total_selections if total_selections else 0.0
            ),
            "first_replay_iteration": frame_samples[0] if frame_samples else None,
            "last_replay_iteration": frame_samples[-1] if frame_samples else None,
            "max_starvation_gap_iterations": None,
            "max_starvation_gap_eligible_ticks": (
                longest_starvation if eligibility_membership_exact else None
            ),
            "mean_selected_weight": (
                sum(weights) / len(weights) if weights else None
            ),
            "eligible_tick_count": (
                eligible_ticks if eligibility_membership_exact else None
            ),
            "selection_rate_per_1000_eligible_ticks": (
                1000.0 * len(frame_samples) / eligible_ticks
                if eligibility_membership_exact and eligible_ticks > 0
                else None
            ),
            "uniform_expected_replay_count": (
                uniform_expected if eligibility_membership_exact else None
            ),
            "realized_to_uniform_expected_ratio": (
                len(frame_samples) / uniform_expected
                if eligibility_membership_exact and uniform_expected > 0.0
                else None
            ),
        })
    frame_rows.sort(key=lambda row: (row["source_ordinal"], row["timestamp_ns"]))

    admitted_rows = [
        row for row in frame_rows if row["semantic_action"] == "commit"
    ]
    deferred_rows = [
        row for row in frame_rows if row["semantic_action"] == "defer"
    ]
    replayed_admitted = sum(row["replay_count"] > 0 for row in admitted_rows)
    replayed_deferred = sum(row["replay_count"] > 0 for row in deferred_rows)
    replayed_retained = sum(row["replay_count"] > 0 for row in frame_rows)
    counts = [int(row["replay_count"]) for row in frame_rows]
    starvation_gaps = [
        float(row["max_starvation_gap_eligible_ticks"])
        for row in frame_rows
        if row["max_starvation_gap_eligible_ticks"] is not None
    ]
    normalized_rates = [
        float(row["selection_rate_per_1000_eligible_ticks"])
        for row in frame_rows
        if row["selection_rate_per_1000_eligible_ticks"] is not None
    ]
    uniform_ratios = [
        float(row["realized_to_uniform_expected_ratio"])
        for row in frame_rows
        if row["realized_to_uniform_expected_ratio"] is not None
    ]
    mutation_gaps = [
        right - left
        for left, right in zip(mutation_iterations, mutation_iterations[1:])
    ]
    full_replay_coverage = (
        replayed_retained / len(frame_rows)
        if catalog_membership_exact and frame_rows
        else None
    )
    admitted_replay_coverage = (
        replayed_admitted / len(admitted_rows)
        if catalog_membership_exact and admitted_rows
        else None
    )
    deferred_replay_coverage = (
        replayed_deferred / len(deferred_rows)
        if catalog_membership_exact and deferred_rows
        else None
    )
    configured_shortlist_limit = (
        next(iter(shortlist_configured_limits))
        if len(shortlist_configured_limits) == 1
        else None
    )
    catalog_entropy = (
        _normalized_entropy(counts) if catalog_membership_exact else None
    )
    catalog_gini = _gini(counts) if catalog_membership_exact else None
    summary: dict[str, Any] = {
        "schema": "eligsir.replay-sampling-telemetry/v3",
        "trace_file": trace.name,
        "trace_rows": trace_rows,
        "executed_ticks": executed_ticks,
        "optimizer_steps_total": optimizer_steps_total,
        "supervision_views_total": supervision_views_total,
        "final_optimizer_iteration": final_iteration,
        "total_replay_selections": total_selections,
        "active_replay_selections": int(sum(active_counts.values())),
        "idle_replay_selections": int(sum(idle_counts.values())),
        "retained_catalog_population": retained_catalog_population,
        "retained_catalog_admitted_population": len(admitted_rows),
        "retained_catalog_deferred_population": len(deferred_rows),
        "retained_catalog_identity_membership_status": catalog_membership_status,
        "retained_catalog_identity_membership_source": (
            "terminal_admit_defer_union"
            if catalog_membership_exact
            else "unavailable"
        ),
        "retained_catalog_population_evidence_max": explicit_catalog_population,
        "terminal_retained_identity_population": len(retained),
        "catalog_sequence_mismatch_rows": catalog_sequence_mismatch_rows,
        "catalog_population_position_ambiguous_rows": (
            catalog_population_position_ambiguous_rows
        ),
        "catalog_disposition_rows_without_population": (
            catalog_disposition_rows_without_population
        ),
        "catalog_disposition_membership_provenance": sorted(
            disposition_membership_provenance
        ),
        "catalog_unknown_candidate_or_selected_identities": len(
            unknown_catalog_sources
        ),
        "unique_admitted_frames": len(admitted_rows),
        "unique_replayed_frames": sum(row["replay_count"] > 0 for row in frame_rows),
        "full_replay_coverage_fraction": full_replay_coverage,
        "admitted_replay_coverage_fraction": admitted_replay_coverage,
        "deferred_replay_coverage_fraction": deferred_replay_coverage,
        "replay_coverage_fraction": full_replay_coverage,
        "sampling_count_min": min(counts) if counts and catalog_membership_exact else None,
        "sampling_count_median": (
            float(median(counts)) if counts and catalog_membership_exact else None
        ),
        "sampling_count_p90": (
            _quantile([float(value) for value in counts], 0.9)
            if counts and catalog_membership_exact
            else None
        ),
        "sampling_count_max": max(counts) if counts and catalog_membership_exact else None,
        "catalog_normalized_sampling_entropy": catalog_entropy,
        "catalog_sampling_gini": catalog_gini,
        "replay_shortlist_configured_limit": configured_shortlist_limit,
        "replay_shortlist_configured_limit_status": (
            "available_exact"
            if configured_shortlist_limit is not None
            else "unavailable"
        ),
        "replay_shortlist_realized_population_min": (
            min(shortlist_realized_populations)
            if shortlist_realized_populations
            else None
        ),
        "replay_shortlist_realized_population_median": (
            _quantile(shortlist_realized_populations, 0.5)
            if shortlist_realized_populations
            else None
        ),
        "replay_shortlist_realized_population_p90": (
            _quantile(shortlist_realized_populations, 0.9)
            if shortlist_realized_populations
            else None
        ),
        "replay_shortlist_realized_population_max": (
            max(shortlist_realized_populations)
            if shortlist_realized_populations
            else None
        ),
        "replay_shortlist_realized_population_final": (
            shortlist_realized_populations[-1]
            if shortlist_realized_populations
            else None
        ),
        "replay_shortlist_membership_mismatch_ticks": (
            shortlist_membership_mismatch_ticks
        ),
        "executed_ticks_with_shortlist_membership": executed_membership_rows,
        "executed_ticks_with_exact_sampler_membership": (
            executed_membership_exact_rows
        ),
        "eligibility_metrics_status": eligibility_metrics_status,
        "eligibility_metrics_unavailable_reason": (
            None
            if eligibility_membership_exact
            else (
                "exact post-filter replay candidate membership is not available "
                "for every executed tick"
            )
        ),
        "eligibility_population_mismatch_ticks": (
            eligibility_population_mismatch_ticks
        ),
        "eligibility_normalized_sampling_gini": (
            _gini(normalized_rates) if eligibility_membership_exact else None
        ),
        "realized_to_uniform_expected_ratio_p10": (
            _quantile(uniform_ratios, 0.1) if eligibility_membership_exact else None
        ),
        "realized_to_uniform_expected_ratio_median": (
            _quantile(uniform_ratios, 0.5) if eligibility_membership_exact else None
        ),
        "realized_to_uniform_expected_ratio_p90": (
            _quantile(uniform_ratios, 0.9) if eligibility_membership_exact else None
        ),
        "starvation_gap_iterations_p50": None,
        "starvation_gap_iterations_p90": None,
        "starvation_gap_iterations_p99": None,
        "starvation_gap_iterations_max": None,
        "starvation_gap_eligible_ticks_p50": (
            _quantile(starvation_gaps, 0.5) if eligibility_membership_exact else None
        ),
        "starvation_gap_eligible_ticks_p90": (
            _quantile(starvation_gaps, 0.9) if eligibility_membership_exact else None
        ),
        "starvation_gap_eligible_ticks_p99": (
            _quantile(starvation_gaps, 0.99) if eligibility_membership_exact else None
        ),
        "starvation_gap_eligible_ticks_max": (
            max(starvation_gaps, default=0.0)
            if eligibility_membership_exact
            else None
        ),
        "allocation_constellations": dict(sorted(allocation_counts.items())),
        "requested_allocation_constellations": dict(
            sorted(requested_allocation_counts.items())
        ),
        "resolution_tiers": dict(sorted(resolution_counts.items())),
        "planned_resolution_tiers": dict(
            sorted(planned_resolution_counts.items())
        ),
        "control_phases": dict(sorted(control_phase_counts.items())),
        "ingress_queue_depth_p50": _quantile(ingress_queue_depths, 0.5),
        "ingress_queue_depth_p90": _quantile(ingress_queue_depths, 0.9),
        "ingress_queue_depth_p99": _quantile(ingress_queue_depths, 0.99),
        "ingress_queue_depth_max": max(ingress_queue_depths, default=0.0),
        "mcmc_requested_ticks": mcmc_requested_ticks,
        "mcmc_effective_ticks": mcmc_effective_ticks,
        "mcmc_structural_epochs": mcmc_structural_epochs,
        "live_depth_structural_epochs": live_depth_structural_epochs,
        "replay_depth_structural_epochs": replay_depth_structural_epochs,
        "mutation_by_kind": {
            kind: dict(sorted(counts.items()))
            for kind, counts in sorted(mutation_by_kind.items())
        },
        "structural_mutation_gap_iterations_p50": _quantile(
            mutation_gaps, 0.5
        ),
        "structural_mutation_gap_iterations_p90": _quantile(
            mutation_gaps, 0.9
        ),
        "freshness_modes": dict(sorted(freshness_counts.items())),
        "mutation_totals": dict(sorted(mutation_totals.items())),
        "interpretation": (
            "Counts describe realized replay selections over the full terminal "
            "ADMIT+DEFER retained catalog. The replay shortlist is a bounded "
            "selection input, not a cache or catalog capacity. Eligibility-normalized "
            "and starvation metrics are published only when exact per-tick shortlist "
            "membership and sampler population agree; the uniform ratio remains a "
            "neutral reference for a non-uniform weighted draw."
        ),
    }
    (destination / "replay_sampling_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with (destination / "replay_sampling_counts.jsonl").open("w") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_heatmap(
        destination / "replay_sampling_heatmap.svg",
        samples,
        frame_rows,
        summary,
    )
    return summary


__all__ = ["write_replay_sampling_telemetry"]
