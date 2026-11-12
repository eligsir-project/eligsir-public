#!/usr/bin/env python3
"""Analyze deterministic true-camera scaling JSONL without conflating selectors."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

from eligsir.plot_style import svg_output_path

ANALYSIS_SCHEMA_VERSION = "eligsir.true_camera_scaling.analysis.v1"


def load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
                rows.append(row)
    return rows


def _platform(row: dict[str, Any]) -> str:
    hardware = row.get("hardware") or {}
    parts = (
        hardware.get("gpu_label"),
        hardware.get("device_name"),
        hardware.get("gpu_uuid"),
        hardware.get("gpu_arch"),
        hardware.get("pci_bus_id"),
        hardware.get("total_memory_bytes"),
    )
    return "|".join(str(part) for part in parts if part is not None) or "unknown"


def _experiment_key(row: dict[str, Any]) -> tuple[Any, ...]:
    checkpoint = row.get("checkpoint") or {}
    manifest = row.get("camera_manifest") or {}
    subset = row.get("gaussian_subset") or {}
    return (
        checkpoint.get("sha256"),
        manifest.get("sha256"),
        subset.get("indices_sha256"),
        row.get("code_sha"),
        row.get("rng_seed"),
        row.get("warmups"),
        row.get("repetitions"),
        row.get("requested_n"),
        row.get("realized_n"),
        row.get("render_mode"),
        row.get("sh_mode"),
        row.get("realized_sh_degree"),
        row.get("pass_mode"),
        row.get("mode"),
    )


def _series_key(row: dict[str, Any]) -> tuple[Any, ...]:
    hardware = row.get("hardware") or {}
    software = row.get("software") or {}
    source = row.get("source_provenance") or {}
    return (
        json.dumps(
            _physical_hardware_identity(hardware),
            sort_keys=True,
            separators=(",", ":"),
        ),
        json.dumps(
            _software_identity(software), sort_keys=True, separators=(",", ":")
        ),
        json.dumps(source, sort_keys=True, separators=(",", ":")),
        *_experiment_key(row),
    )


def _physical_hardware_identity(hardware: dict[str, Any]) -> dict[str, Any]:
    """Keep physical identity while dropping per-container process metadata."""
    ephemeral_fields = {"hostname", "container_hostname", "device_index"}
    return {
        field: value
        for field, value in hardware.items()
        if field not in ephemeral_fields
    }


def _software_identity(software: dict[str, Any]) -> dict[str, Any]:
    """Retain every software field so new stable build fields split series."""
    return dict(software)


def _stable_host_identity_available(row: dict[str, Any]) -> bool:
    hardware = row.get("hardware") or {}
    host_id = hardware.get("host_machine_id")
    source = hardware.get("host_machine_id_source")
    return (
        isinstance(host_id, str)
        and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", host_id) is not None
        and source in {
            "explicit_env_sha256",
            "dmi_product_uuid_sha256",
            "linux_boot_id_sha256",
        }
    )


def _stable_software_identity_available(row: dict[str, Any]) -> bool:
    software = row.get("software") or {}
    digest = software.get("container_image_digest")
    backend_versions = (
        software.get("torch_cuda"),
        software.get("torch_hip"),
    )
    return (
        isinstance(digest, str)
        and re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest) is not None
        and bool(str(software.get("torch") or "").strip())
        and bool(str(software.get("gsplat") or "").strip())
        and sum(bool(str(value or "").strip()) for value in backend_versions) == 1
    )


def _recompute_scaling_efficiency(rows: Sequence[dict[str, Any]]) -> None:
    """Overwrite per-process eta after the complete cross-file series is known."""
    baselines = [
        float(row["mean_transaction_ms"])
        for row in rows
        if row.get("status") == "ok"
        and int(row.get("requested_c", 0)) == 1
        and int(row.get("realized_c", 0)) == 1
        and row.get("mean_transaction_ms") is not None
        and float(row["mean_transaction_ms"]) > 0
    ]
    baseline = statistics.fmean(baselines) if baselines else None
    for row in rows:
        row["scaling_efficiency_source"] = "analyzer_recomputed_grouped_series"
        requested_c = int(row.get("requested_c", 0))
        realized_c = int(row.get("realized_c", 0))
        transaction_ms = row.get("mean_transaction_ms")
        if baseline is None:
            row["scaling_efficiency"] = None
            row["scaling_efficiency_reason"] = "matching_C1_baseline_unavailable"
        elif (
            row.get("status") != "ok"
            or requested_c != realized_c
            or transaction_ms is None
            or float(transaction_ms) <= 0
        ):
            row["scaling_efficiency"] = None
            row["scaling_efficiency_reason"] = "point_not_successful_or_fully_realized"
        else:
            row["scaling_efficiency"] = (
                realized_c * baseline / float(transaction_ms)
            )
            row["scaling_efficiency_reason"] = None


def _series_description(key: tuple[Any, ...]) -> dict[str, Any]:
    hardware = json.loads(key[0])
    software = json.loads(key[1])
    source = json.loads(key[2])
    return {
        "platform": hardware.get("gpu_label"),
        "device_name": hardware.get("device_name"),
        "gpu_uuid": hardware.get("gpu_uuid"),
        "gpu_arch": hardware.get("gpu_arch"),
        "pci_bus_id": hardware.get("pci_bus_id"),
        "total_memory_bytes": hardware.get("total_memory_bytes"),
        "python": software.get("python"),
        "software_platform": software.get("platform"),
        "kernel_release": software.get("kernel_release"),
        "torch": software.get("torch"),
        "torch_cuda": software.get("torch_cuda"),
        "torch_hip": software.get("torch_hip"),
        "gsplat": software.get("gsplat"),
        "container_image_digest": software.get("container_image_digest"),
        "hardware": hardware,
        "software": software,
        "source_provenance": source,
        **_experiment_description(key[3:]),
    }


def _experiment_description(key: tuple[Any, ...]) -> dict[str, Any]:
    names = (
        "checkpoint_sha256",
        "camera_manifest_sha256",
        "gaussian_indices_sha256",
        "code_sha",
        "rng_seed",
        "warmups",
        "repetitions",
        "requested_n",
        "realized_n",
        "render_mode",
        "sh_mode",
        "realized_sh_degree",
        "pass_mode",
        "mode",
    )
    return dict(zip(names, key))


def _weighted_affine(xs: list[float], ys: list[float], weights: list[float]) -> tuple[float, float]:
    weight_sum = sum(weights)
    x_bar = sum(w * x for w, x in zip(weights, xs)) / weight_sum
    y_bar = sum(w * y for w, y in zip(weights, ys)) / weight_sum
    denominator = sum(w * (x - x_bar) ** 2 for w, x in zip(weights, xs))
    if denominator <= 0:
        raise ValueError("fit requires at least two distinct C values")
    slope = sum(
        w * (x - x_bar) * (y - y_bar)
        for w, x, y in zip(weights, xs, ys)
    ) / denominator
    intercept = y_bar - slope * x_bar
    return intercept, slope


def fit_affine(
    rows: Sequence[dict[str, Any]],
    *,
    max_c: int = 16,
    weighting: str = "unweighted",
) -> dict[str, Any]:
    """Fit T(C)=T0+t_camera*C over successful realized points C<=max_c."""
    if weighting not in {"unweighted", "inverse_sample_variance"}:
        raise ValueError("weighting must be unweighted or inverse_sample_variance")
    cells: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("mean_transaction_ms") is None:
            continue
        requested_c = int(row["requested_c"])
        realized_c = int(row["realized_c"])
        if requested_c != realized_c or realized_c > int(max_c):
            continue
        cells.setdefault(realized_c, []).append(row)
    xs = sorted(cells)
    ys = [statistics.fmean(float(row["mean_transaction_ms"]) for row in cells[c]) for c in xs]
    if len(xs) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer_than_two_distinct_successful_C_values_in_fit_domain",
            "domain": {
                "min_c": min(xs) if xs else None,
                "max_c": max(xs) if xs else None,
                "n_points": len(xs),
            },
            "configured_domain": {"max_c": int(max_c)},
            "weighting": weighting,
            "r2_weighting": weighting,
            "t0_ms": None,
            "t_camera_ms": None,
            "r2": None,
            "residuals_ms": [],
            "t0_standard_error_ms": None,
            "t_camera_standard_error_ms": None,
            "uncertainty_reason": "requires_at_least_three_distinct_C_values",
        }
    if weighting == "unweighted":
        weights = [1.0] * len(xs)
    else:
        weights = []
        for c in xs:
            samples = [
                float(value)
                for row in cells[c]
                for value in (row.get("samples_transaction_ms") or [])
            ]
            variance = statistics.variance(samples) if len(samples) >= 2 else 0.0
            weights.append(1.0 / max(variance, 1e-12))
    intercept, slope = _weighted_affine(
        [float(c) for c in xs], ys, weights
    )
    predicted = [intercept + slope * c for c in xs]
    residual_values = [actual - estimate for actual, estimate in zip(ys, predicted)]
    weight_sum = sum(weights)
    y_bar = sum(weight * value for weight, value in zip(weights, ys)) / weight_sum
    sse = sum(weight * residual * residual for weight, residual in zip(weights, residual_values))
    sst = sum(weight * (value - y_bar) ** 2 for weight, value in zip(weights, ys))
    r2 = None if sst == 0 else 1.0 - sse / sst
    t0_se = slope_se = None
    if len(xs) > 2:
        sigma2 = sse / (len(xs) - 2)
        s0 = weight_sum
        s1 = sum(weight * c for weight, c in zip(weights, xs))
        s2 = sum(weight * c * c for weight, c in zip(weights, xs))
        determinant = s0 * s2 - s1 * s1
        if determinant > 0:
            slope_se = math.sqrt(max(0.0, sigma2 * s0 / determinant))
            t0_se = math.sqrt(max(0.0, sigma2 * s2 / determinant))
    return {
        "status": "ok",
        "reason": None,
        "model": "T(C)=T0+t_camera*C",
        "domain": {"min_c": min(xs), "max_c": max(xs), "n_points": len(xs)},
        "configured_domain": {"max_c": int(max_c)},
        "weighting": weighting,
        "r2_weighting": weighting,
        "t0_ms": intercept,
        "t_camera_ms": slope,
        "r2": r2,
        "residuals_ms": [
            {"c": c, "observed_ms": actual, "predicted_ms": estimate, "residual_ms": residual}
            for c, actual, estimate, residual in zip(xs, ys, predicted, residual_values)
        ],
        "t0_standard_error_ms": t0_se,
        "t_camera_standard_error_ms": slope_se,
        "uncertainty_reason": None if len(xs) > 2 else "requires_at_least_three_distinct_C_values",
    }


def cross_platform_crossover(
    platform_a: str,
    fit_a: dict[str, Any],
    platform_b: str,
    fit_b: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "platform_a": platform_a,
        "platform_b": platform_b,
        "status": "unavailable",
        "c_star": None,
        "reason": None,
    }
    if fit_a.get("status") != "ok" or fit_b.get("status") != "ok":
        result["reason"] = "one_or_both_affine_fits_unavailable"
        return result
    t0_a, slope_a = float(fit_a["t0_ms"]), float(fit_a["t_camera_ms"])
    t0_b, slope_b = float(fit_b["t0_ms"]), float(fit_b["t_camera_ms"])
    if min(t0_a, t0_b) < 0 or min(slope_a, slope_b) <= 0:
        result["reason"] = "non_positive_fit_coefficients_make_crossover_not_meaningful"
        return result
    denominator = slope_a - slope_b
    if math.isclose(denominator, 0.0, rel_tol=1e-9, abs_tol=1e-12):
        result["reason"] = "parallel_or_numerically_indistinguishable_slopes"
        return result
    c_star = (t0_b - t0_a) / denominator
    if not math.isfinite(c_star) or c_star <= 0:
        result["reason"] = "crossover_is_not_positive_and_finite"
        return result
    result.update({"status": "ok", "c_star": c_star, "reason": None})
    domains = (fit_a.get("domain") or {}, fit_b.get("domain") or {})
    if all(domain.get("min_c") is not None and domain.get("max_c") is not None for domain in domains):
        low = max(float(domain["min_c"]) for domain in domains)
        high = min(float(domain["max_c"]) for domain in domains)
        result["within_overlapping_fit_domain"] = low <= c_star <= high
    else:
        result["within_overlapping_fit_domain"] = None
    return result


def choose_winners(
    rows: Sequence[dict[str, Any]],
    *,
    freshness_budget_ms: float,
    peak_memory_budget_bytes: int,
) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("cameras_per_s") is not None
        and int(row.get("requested_c", 0)) == int(row.get("realized_c", -1))
    ]
    offline = None
    if successful:
        chosen = max(
            successful,
            key=lambda row: (float(row["cameras_per_s"]), -int(row["realized_c"])),
        )
        offline = {
            "chosen_c": int(chosen["realized_c"]),
            "realized_cameras_per_s": float(chosen["cameras_per_s"]),
            "objective": "max_realized_cameras_per_s",
            "reason": "highest measured realized cameras/s among successful points",
        }
    eligible_live = []
    for row in successful:
        p95 = row.get("p95_transaction_ms")
        memory = row.get("peak_memory_allocated_bytes")
        if p95 is None or memory is None:
            continue
        if float(p95) < float(freshness_budget_ms) and int(memory) <= int(peak_memory_budget_bytes):
            useful = int(row.get("useful_realized_cameras", row["realized_c"]))
            rate = 1000.0 * useful / float(row["mean_transaction_ms"])
            eligible_live.append((rate, row))
    live = None
    if eligible_live:
        rate, chosen = max(eligible_live, key=lambda item: (item[0], -int(item[1]["realized_c"])))
        live = {
            "chosen_c": int(chosen["realized_c"]),
            "useful_realized_supervision_per_s": rate,
            "objective": "max_useful_realized_supervision_per_s",
            "constraints": {
                "freshness_budget_ms": float(freshness_budget_ms),
                "p95_transaction_ms": float(chosen["p95_transaction_ms"]),
                "p95_strictly_below_freshness_budget": True,
                "peak_memory_budget_bytes": int(peak_memory_budget_bytes),
                "peak_memory_allocated_bytes": int(chosen["peak_memory_allocated_bytes"]),
                "peak_memory_within_budget": True,
            },
            "reason": "highest useful realized supervision/s among points satisfying both live constraints",
        }
    return {
        "offline": offline or {
            "chosen_c": None,
            "objective": "max_realized_cameras_per_s",
            "reason": "no successful fully realized points",
        },
        "live": live or {
            "chosen_c": None,
            "objective": "max_useful_realized_supervision_per_s",
            "constraints": {
                "freshness_budget_ms": float(freshness_budget_ms),
                "peak_memory_budget_bytes": int(peak_memory_budget_bytes),
            },
            "reason": "no successful point satisfies strict freshness and peak-memory constraints",
        },
    }


def _slug(description: dict[str, Any]) -> str:
    title = _series_title(description)
    digest = hashlib.sha256(
        json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", title).strip("-")[:100]
    return f"{text}-{digest}"


def _series_title(description: dict[str, Any]) -> str:
    """Human title for a C curve; full provenance remains in analysis JSON."""
    platform = description.get("platform") or "unknown GPU"
    architecture = description.get("gpu_arch") or "unknown arch"
    n_value = description.get("realized_n") or description.get("requested_n")
    sh_mode = description.get("sh_mode")
    sh_degree = description.get("realized_sh_degree")
    sh_text = f"SH={sh_mode}" + (f"/{sh_degree}" if sh_degree is not None else "")
    return " | ".join(
        str(value)
        for value in (
            f"{platform} {architecture}",
            f"N={n_value}",
            description.get("render_mode"),
            sh_text,
            description.get("pass_mode"),
            description.get("mode"),
        )
        if value is not None
    )


def write_plots(series: list[dict[str, Any]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for item in series:
        rows = sorted(item["rows"], key=lambda row: int(row["realized_c"]))
        if not rows:
            continue
        c_values = [int(row["realized_c"]) for row in rows]
        transaction = [float(row["mean_transaction_ms"]) for row in rows]
        per_camera = [float(row["ms_per_camera"]) for row in rows]
        throughput = [float(row["cameras_per_s"]) for row in rows]
        efficiency = [row.get("scaling_efficiency") for row in rows]
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        panels = (
            (axes[0, 0], transaction, "T(C)", "transaction ms"),
            (axes[0, 1], per_camera, "T(C)/C", "ms/camera"),
            (axes[1, 0], throughput, "C/T(C)", "cameras/s"),
            (axes[1, 1], efficiency, "eta(C)=C*T(1)/T(C)", "efficiency"),
        )
        for axis, values, title, ylabel in panels:
            present = [(c, value) for c, value in zip(c_values, values) if value is not None]
            if present:
                axis.plot([point[0] for point in present], [point[1] for point in present], marker="o")
            axis.set_xscale("log", base=2)
            axis.set_xlabel("realized C")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
        path = output_dir / f"{_slug(item['series'])}.svg"
        figure.suptitle(_series_title(item["series"]))
        figure.savefig(svg_output_path(path), dpi=140)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def analyze(
    rows: Sequence[dict[str, Any]],
    *,
    fit_max_c: int,
    weighting: str,
    freshness_budget_ms: float,
    peak_memory_budget_bytes: int,
) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    excluded_rows = []
    for row_index, row in enumerate(rows):
        if row.get("status") != "ok":
            continue
        source = row.get("source_provenance") or {}
        resolved_head = source.get("resolved_git_head")
        resolved_tree = source.get("resolved_git_tree")
        entrypoint_sha = source.get("entrypoint_sha256")
        verified = (
            source.get("verification_status") == "verified_clean_git_head"
            and source.get("comparison_eligible") is True
            and source.get("working_tree_state") == "clean"
            and isinstance(resolved_head, str)
            and re.fullmatch(r"[0-9a-fA-F]{40}", resolved_head) is not None
            and isinstance(resolved_tree, str)
            and re.fullmatch(r"[0-9a-fA-F]{40}", resolved_tree) is not None
            and isinstance(entrypoint_sha, str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", entrypoint_sha) is not None
            and row.get("code_sha") == resolved_head
        )
        if not verified:
            excluded_rows.append(
                {
                    "row_index": row_index,
                    "reason": "source_provenance_not_verified_for_comparison",
                    "verification_status": source.get("verification_status") or "missing",
                }
            )
            continue
        if not _stable_host_identity_available(row):
            excluded_rows.append(
                {
                    "row_index": row_index,
                    "reason": "stable_host_identity_unavailable",
                    "verification_status": source.get("verification_status") or "missing",
                }
            )
            continue
        if not _stable_software_identity_available(row):
            excluded_rows.append(
                {
                    "row_index": row_index,
                    "reason": "software_identity_unavailable",
                    "verification_status": source.get("verification_status") or "missing",
                }
            )
            continue
        grouped.setdefault(_series_key(row), []).append(row)
    series = []
    fits_by_comparable: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        group_rows = [dict(row) for row in group_rows]
        _recompute_scaling_efficiency(group_rows)
        fit = fit_affine(group_rows, max_c=fit_max_c, weighting=weighting)
        description = _series_description(key)
        series.append(
            {
                "series": description,
                "rows": sorted(group_rows, key=lambda row: int(row["realized_c"])),
                "fit": fit,
                "selectors": choose_winners(
                    group_rows,
                    freshness_budget_ms=freshness_budget_ms,
                    peak_memory_budget_bytes=peak_memory_budget_bytes,
                ),
            }
        )
        comparable_key = _experiment_key(group_rows[0])
        fits_by_comparable.setdefault(comparable_key, []).append(
            {
                "platform_id": _platform(group_rows[0]),
                "provenance": {
                    "series": description,
                    "hardware": dict(group_rows[0].get("hardware") or {}),
                    "software": dict(group_rows[0].get("software") or {}),
                    "source_provenance": dict(
                        group_rows[0].get("source_provenance") or {}
                    ),
                },
                "fit": fit,
            }
        )
    crossovers = []
    for comparable_key, platform_fits in sorted(
        fits_by_comparable.items(), key=lambda item: str(item[0])
    ):
        ordered_fits = sorted(platform_fits, key=lambda item: item["platform_id"])
        for side_a, side_b in itertools.combinations(ordered_fits, 2):
            if side_a["platform_id"] == side_b["platform_id"]:
                continue
            crossovers.append(
                {
                    "series": _experiment_description(comparable_key),
                    "platform_a_provenance": side_a["provenance"],
                    "platform_b_provenance": side_b["provenance"],
                    **cross_platform_crossover(
                        side_a["platform_id"],
                        side_a["fit"],
                        side_b["platform_id"],
                        side_b["fit"],
                    ),
                }
            )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "fit_contract": {
            "model": "T(C)=T0+t_camera*C",
            "configured_domain": {"max_c": int(fit_max_c)},
            "reported_domain": "actual min/max realized C included in each fit",
            "weighting": weighting,
        },
        "selector_contract": {
            "offline": "maximize realized cameras/s",
            "live": "maximize useful realized supervision/s subject to p95 < freshness budget and peak memory <= budget",
            "freshness_budget_ms": float(freshness_budget_ms),
            "peak_memory_budget_bytes": int(peak_memory_budget_bytes),
        },
        "input_row_count": len(rows),
        "comparison_eligible_row_count": sum(len(group) for group in grouped.values()),
        "excluded_rows": excluded_rows,
        "series": series,
        "cross_platform_crossover": crossovers,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots-dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--fit-max-c", type=int, default=16)
    parser.add_argument(
        "--weighting",
        choices=("unweighted", "inverse_sample_variance"),
        default="unweighted",
    )
    parser.add_argument("--freshness-budget-ms", type=float, required=True)
    parser.add_argument("--peak-memory-budget-bytes", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fit_max_c < 1:
        raise ValueError("--fit-max-c must be >= 1")
    if args.freshness_budget_ms <= 0 or args.peak_memory_budget_bytes <= 0:
        raise ValueError("freshness and peak-memory budgets must be > 0")
    rows = load_rows(args.inputs)
    result = analyze(
        rows,
        fit_max_c=args.fit_max_c,
        weighting=args.weighting,
        freshness_budget_ms=args.freshness_budget_ms,
        peak_memory_budget_bytes=args.peak_memory_budget_bytes,
    )
    if not args.no_plots:
        plots_dir = args.plots_dir or args.output.with_suffix("").with_name(args.output.stem + "-plots")
        result["plots"] = write_plots(result["series"], plots_dir)
    else:
        result["plots"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "series": len(result["series"]), "plots": len(result["plots"])}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"analyze_true_camera_scaling: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
