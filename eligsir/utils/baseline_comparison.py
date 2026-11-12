"""``--baseline_run`` curve-level comparison (torch-free).

Split out of ``utils/comparison_report.py`` (issue #423). That module
imports torch at module scope for its rendering/contact-sheet work, which
made ``tests/test_baseline_comparison.py`` and
``tests/test_scored_subset_hash.py`` uncollectable in the CPU-only smoke
image (``Dockerfile.ci-cpu`` deliberately installs no torch) -- they went
red on main the moment #427 added them to the smoke allowlist.

Nothing here touches a tensor: it reads two runs' ``report.json`` /
``psnr_history.jsonl`` off disk and compares the curves. ``comparison_report``
re-exports every public name, so existing import sites are unchanged.

See CLAUDE.md's "Compare PSNR curves, not just the final number" run contract.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from bisect import bisect_right
from typing import Optional



# ---------------------------------------------------------------------------
# --baseline_run comparison (opt-in; see CLAUDE.md's "Compare PSNR curves,
# not just the final number" run contract)
# ---------------------------------------------------------------------------

_BASELINE_SMALL_DELTA_DB = 1.0   # within this: treated as run-to-run noise
_BASELINE_LARGE_DELTA_DB = 3.0   # beyond this: a real, notable difference
_BASELINE_MIN_ALIGNED_POINTS = 3

_PHASE_RELATIVE_ALIGNMENT = (
    "piecewise-linear interpolation on the union of observed knots "
    "within shared post-mapping support"
)


def _read_valid_psnr_rows(model_path: str) -> list[dict]:
    """Read score-bearing PSNR rows without discarding semantic clocks.

    ``utils.psnr_history_io.read_psnr_history`` intentionally returns only
    iteration and score for legacy callers.  Cross-throughput comparison must
    retain ``mapping_active_s`` and ``supervision_views`` from the same durable
    row, so it uses this equally torch-free reader instead of attempting to
    join independent telemetry streams after the fact.
    """
    path = os.path.join(model_path, "psnr_history.jsonl")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if row.get("eval_valid", True) is not True:
                continue
            if row.get("mean_test_psnr") is None:
                continue
            copied = dict(row)
            copied["_line_number"] = line_number
            rows.append(copied)
    from eligsir.utils.psnr_history_io import psnr_reduction

    psnr_reduction(rows, source=path)
    return rows


def _mapping_endpoint_iteration(model_path: str) -> tuple[int | None, str | None]:
    """Return the authoritative mapping endpoint or an explicit failure.

    Both production streaming implementations write
    ``streaming_phase_metadata.json``.  The legacy trainer uses
    ``mapping_end_iteration`` while the canonical application stores the same
    value as ``mapping_endpoint.iteration``.  Inferring an endpoint from the
    first PSNR row would turn a missing artifact into a plausible-looking but
    invalid comparison, so absence or ambiguity fails closed.
    """
    path = os.path.join(model_path, "streaming_phase_metadata.json")
    if not os.path.isfile(path):
        return None, f"missing streaming_phase_metadata.json under {model_path}"
    try:
        with open(path) as handle:
            metadata = json.load(handle)
    except Exception as exc:
        return None, f"failed to read streaming_phase_metadata.json: {exc!r}"
    if not isinstance(metadata, dict):
        return None, "streaming_phase_metadata.json root is not an object"

    direct = metadata.get("mapping_end_iteration")
    endpoint_payload = metadata.get("mapping_endpoint")
    if endpoint_payload is not None and not isinstance(endpoint_payload, dict):
        return None, "streaming_phase_metadata.json mapping_endpoint is not an object"
    nested = (endpoint_payload or {}).get("iteration")
    if direct is not None and nested is not None and direct != nested:
        return None, (
            "ambiguous mapping endpoint: mapping_end_iteration="
            f"{direct!r} differs from mapping_endpoint.iteration={nested!r}"
        )
    value = direct if direct is not None else nested
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, (
            "streaming_phase_metadata.json has no valid non-negative integer "
            "mapping_end_iteration or mapping_endpoint.iteration"
        )
    return int(value), None


def _phase_curve(
    rows: list[dict],
    endpoint_iteration: int,
    axis_name: str,
    arm_name: str,
) -> tuple[dict | None, str | None]:
    """Build one arm's post-mapping curve on an endpoint-relative axis."""
    endpoint_rows = [row for row in rows if row.get("iteration") == endpoint_iteration]
    semantic_endpoint_rows = [
        row for row in endpoint_rows if row.get("evaluation_phase") == "mapping_endpoint"
    ]
    if semantic_endpoint_rows:
        endpoint_rows = semantic_endpoint_rows
    if len(endpoint_rows) != 1:
        return None, (
            f"{arm_name}: expected exactly one score-bearing psnr_history row at "
            f"mapping endpoint iteration {endpoint_iteration}, found {len(endpoint_rows)}"
        )
    endpoint_row = endpoint_rows[0]
    post_rows = []
    for row in rows:
        iteration = row.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            return None, (
                f"{arm_name}: psnr_history line {row.get('_line_number')} has no "
                "valid integer iteration"
            )
        if iteration == endpoint_iteration and row is not endpoint_row:
            continue
        if iteration >= endpoint_iteration:
            post_rows.append(row)
    post_rows.sort(key=lambda row: row["iteration"])
    if not post_rows:
        return None, f"{arm_name}: no post-mapping psnr_history rows"
    if axis_name == "mapping_active_s":
        invalid_rows = [
            row
            for row in post_rows
            if row.get("throughput_evidence_valid") is False
        ]
        if invalid_rows:
            statuses = sorted(
                {
                    str(
                        row.get(
                            "throughput_evidence_status",
                            "invalid_throughput_evidence",
                        )
                    )
                    for row in invalid_rows
                }
            )
            return None, (
                f"{arm_name}: mapping_active_s throughput comparison refused: "
                + ", ".join(statuses)
            )
    if len({row["iteration"] for row in post_rows}) != len(post_rows):
        return None, f"{arm_name}: duplicate post-mapping optimizer iterations"

    endpoint_axis = endpoint_row.get(axis_name)
    if isinstance(endpoint_axis, bool) or not isinstance(endpoint_axis, (int, float)):
        return None, (
            f"{arm_name}: mapping endpoint row is missing {axis_name}"
        )
    endpoint_axis = float(endpoint_axis)
    if not math.isfinite(endpoint_axis) or endpoint_axis < 0.0:
        return None, f"{arm_name}: mapping endpoint {axis_name} is invalid"

    relative_axis = []
    psnr_values = []
    for row in post_rows:
        value = row.get(axis_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, (
                f"{arm_name}: psnr_history row at iteration {row['iteration']} "
                f"is missing {axis_name}"
            )
        value = float(value)
        psnr = row.get("mean_test_psnr")
        if (
            not math.isfinite(value)
            or value < endpoint_axis
            or isinstance(psnr, bool)
            or not isinstance(psnr, (int, float))
            or not math.isfinite(float(psnr))
        ):
            return None, (
                f"{arm_name}: invalid or pre-endpoint {axis_name}/PSNR at "
                f"iteration {row['iteration']}"
            )
        relative_axis.append(value - endpoint_axis)
        psnr_values.append(float(psnr))
    if relative_axis[0] != 0.0:
        return None, f"{arm_name}: normalized {axis_name} does not start at zero"
    if any(b <= a for a, b in zip(relative_axis, relative_axis[1:])):
        return None, (
            f"{arm_name}: post-mapping {axis_name} must be strictly increasing"
        )
    return {
        "axis": relative_axis,
        "psnr": psnr_values,
        "endpoint": {
            "iteration": endpoint_iteration,
            "mapping_active_s": endpoint_row.get("mapping_active_s"),
            "supervision_views": endpoint_row.get("supervision_views"),
            "psnr_db": float(endpoint_row["mean_test_psnr"]),
        },
    }, None


def _interpolate_piecewise_linear(xs: list[float], ys: list[float], x: float) -> float:
    """Deterministically interpolate a curve within its observed support."""
    index = bisect_right(xs, x)
    if index > 0 and xs[index - 1] == x:
        return ys[index - 1]
    if index == 0 or index >= len(xs):
        raise ValueError(f"axis value {x} is outside interpolation support")
    left_x, right_x = xs[index - 1], xs[index]
    fraction = (x - left_x) / (right_x - left_x)
    return ys[index - 1] + fraction * (ys[index] - ys[index - 1])


def _align_phase_curves(current: dict, baseline: dict):
    """Align on all observed knots inside the intersection of both supports."""
    shared_start = max(current["axis"][0], baseline["axis"][0])
    shared_end = min(current["axis"][-1], baseline["axis"][-1])
    if shared_end < shared_start:
        return [], None
    knots = sorted(
        {
            value
            for value in current["axis"] + baseline["axis"]
            if shared_start <= value <= shared_end
        }
    )
    aligned = []
    for value in knots:
        current_psnr = _interpolate_piecewise_linear(
            current["axis"], current["psnr"], value
        )
        baseline_psnr = _interpolate_piecewise_linear(
            baseline["axis"], baseline["psnr"], value
        )
        aligned.append(
            (value, current_psnr, baseline_psnr, current_psnr - baseline_psnr)
        )
    return aligned, [shared_start, shared_end]


def _first_axis_to_gain(xs: list[float], ys: list[float], gain_db: float):
    target = ys[0] + gain_db
    for index in range(1, len(xs)):
        previous_y, current_y = ys[index - 1], ys[index]
        if current_y < target:
            continue
        if current_y == previous_y:
            return xs[index]
        fraction = (target - previous_y) / (current_y - previous_y)
        return xs[index - 1] + fraction * (xs[index] - xs[index - 1])
    return None


def _fast_growth_summary(curve: dict) -> dict:
    xs, ys = curve["axis"], curve["psnr"]
    first_axis = xs[1] if len(xs) > 1 else None
    first_gain = (ys[1] - ys[0]) if len(ys) > 1 else None
    return {
        "endpoint_psnr_db": ys[0],
        "final_psnr_db": ys[-1],
        "final_gain_db": ys[-1] - ys[0],
        "first_post_mapping_axis_value": first_axis,
        "gain_to_first_post_mapping_sample_db": first_gain,
        "initial_gain_rate_db_per_axis_unit": (
            first_gain / first_axis
            if first_axis is not None and first_axis > 0.0
            else None
        ),
        "axis_to_plus_1db": _first_axis_to_gain(xs, ys, 1.0),
        "axis_to_plus_2db": _first_axis_to_gain(xs, ys, 2.0),
    }


def _unavailable_phase_axis(axis_name: str, reason: str) -> dict:
    unit = "s" if axis_name == "mapping_active_s" else "supervised_views"
    return {
        "status": "UNAVAILABLE",
        "reason": reason,
        "curve_shape": None,
        "axis_provenance": {
            "axis": f"post_mapping_{axis_name}",
            "unit": unit,
            "normalization": (
                f"subtract each run's own mapping endpoint {axis_name} before alignment"
            ),
            "alignment": _PHASE_RELATIVE_ALIGNMENT,
            "primary": True,
        },
    }


def _compare_phase_axis(
    *,
    axis_name: str,
    current_rows: list[dict],
    baseline_rows: list[dict],
    current_endpoint: int,
    baseline_endpoint: int,
) -> dict:
    current, current_error = _phase_curve(
        current_rows, current_endpoint, axis_name, "current"
    )
    baseline, baseline_error = _phase_curve(
        baseline_rows, baseline_endpoint, axis_name, "baseline"
    )
    errors = [error for error in (current_error, baseline_error) if error]
    if errors:
        return _unavailable_phase_axis(axis_name, "; ".join(errors))
    assert current is not None and baseline is not None
    aligned, shared_support = _align_phase_curves(current, baseline)
    if shared_support is None:
        return _unavailable_phase_axis(
            axis_name, "current and baseline have no shared post-mapping support"
        )
    deltas = [row[3] for row in aligned]
    unit = "s" if axis_name == "mapping_active_s" else "supervised_views"
    return {
        "status": "VALID",
        "reason": None,
        "curve_shape": _classify_curve_shape(aligned),
        "axis_provenance": {
            "axis": f"post_mapping_{axis_name}",
            "unit": unit,
            "normalization": (
                f"subtract each run's own mapping endpoint {axis_name} before alignment"
            ),
            "endpoint_source": (
                "streaming_phase_metadata.json endpoint iteration joined to the exact "
                "psnr_history.jsonl row"
            ),
            "alignment": _PHASE_RELATIVE_ALIGNMENT,
            "shared_support_policy": "intersection of observed post-mapping support",
            "primary": True,
        },
        "shared_support": shared_support,
        "sample_counts": {
            "current_post_mapping": len(current["axis"]),
            "baseline_post_mapping": len(baseline["axis"]),
            "aligned": len(aligned),
        },
        "endpoint_psnr_db": {
            "current": current["psnr"][0],
            "baseline": baseline["psnr"][0],
            "delta": current["psnr"][0] - baseline["psnr"][0],
        },
        "fast_growth": {
            "current": _fast_growth_summary(current),
            "baseline": _fast_growth_summary(baseline),
        },
        "curve_mean_delta_db": sum(deltas) / len(deltas) if deltas else None,
        "curve_max_abs_delta_db": max((abs(value) for value in deltas), default=None),
        "aligned_curve": [
            {
                "axis_value": axis_value,
                "current_psnr_db": current_psnr,
                "baseline_psnr_db": baseline_psnr,
                "delta_db": delta,
            }
            for axis_value, current_psnr, baseline_psnr, delta in aligned
        ],
        "_current_endpoint": current["endpoint"],
        "_baseline_endpoint": baseline["endpoint"],
    }


def _compare_phase_relative_psnr(
    current_model_path: str, baseline_run: str
) -> dict:
    current_rows = _read_valid_psnr_rows(current_model_path)
    baseline_rows = _read_valid_psnr_rows(baseline_run)
    current_endpoint, current_endpoint_error = _mapping_endpoint_iteration(
        current_model_path
    )
    baseline_endpoint, baseline_endpoint_error = _mapping_endpoint_iteration(
        baseline_run
    )
    endpoint_errors = [
        f"current: {current_endpoint_error}" if current_endpoint_error else None,
        f"baseline: {baseline_endpoint_error}" if baseline_endpoint_error else None,
    ]
    endpoint_errors = [error for error in endpoint_errors if error]
    if endpoint_errors:
        axes = {
            axis_name: _unavailable_phase_axis(
                axis_name, "; ".join(endpoint_errors)
            )
            for axis_name in ("mapping_active_s", "supervision_views")
        }
        return {
            "status": "UNAVAILABLE",
            "reason": "; ".join(endpoint_errors),
            "authoritative": True,
            "primary_axes": ["mapping_active_s", "supervision_views"],
            "mapping_endpoints": {"current": None, "baseline": None},
            "axes": axes,
        }

    assert current_endpoint is not None and baseline_endpoint is not None
    axes = {
        axis_name: _compare_phase_axis(
            axis_name=axis_name,
            current_rows=current_rows,
            baseline_rows=baseline_rows,
            current_endpoint=current_endpoint,
            baseline_endpoint=baseline_endpoint,
        )
        for axis_name in ("mapping_active_s", "supervision_views")
    }
    valid = all(axis["status"] == "VALID" for axis in axes.values())
    endpoint_source = next(
        (
            axis
            for axis in axes.values()
            if axis["status"] == "VALID"
        ),
        None,
    )
    mapping_endpoints = {
        "current": (
            dict(endpoint_source["_current_endpoint"]) if endpoint_source else {
                "iteration": current_endpoint
            }
        ),
        "baseline": (
            dict(endpoint_source["_baseline_endpoint"]) if endpoint_source else {
                "iteration": baseline_endpoint
            }
        ),
    }
    for axis in axes.values():
        axis.pop("_current_endpoint", None)
        axis.pop("_baseline_endpoint", None)
    reasons = [axis["reason"] for axis in axes.values() if axis["status"] != "VALID"]
    return {
        "status": "VALID" if valid else "UNAVAILABLE",
        "reason": None if valid else "; ".join(reasons),
        "authoritative": True,
        "primary_axes": ["mapping_active_s", "supervision_views"],
        "mapping_endpoints": mapping_endpoints,
        "axes": axes,
    }


def _label_legacy_iteration_comparison(result: dict) -> None:
    """Keep the old schema while making its non-primary standing explicit."""
    result["curve_shape_provenance"] = {
        "axis": "absolute_optimizer_iteration",
        "status": "LEGACY_NON_PRIMARY",
        "primary": False,
        "reason": (
            "absolute optimizer iterations do not align mapping phases across "
            "batch sizes, throughput levels, or different mapping endpoints"
        ),
    }
    legacy_fields = (
        "curve_shape",
        "n_aligned_checkpoints",
        "curve_mean_delta",
        "curve_max_abs_delta",
        "curve_min_delta",
        "curve_max_delta",
        "curve_deltas_by_iteration",
    )
    result["legacy_iteration_comparison"] = {
        "status": "LEGACY_NON_PRIMARY",
        "primary": False,
        "axis": "absolute_optimizer_iteration",
        **{field: result.get(field) for field in legacy_fields},
    }


def _align_psnr_curves(cur_iters, cur_vals, base_iters, base_vals):
    """Nearest-iteration alignment between two psnr_history curves.

    Real checkpoints on two different runs rarely land on identical
    iterations (different --streaming_psnr_chart_interval, different
    rolling-idle-stop endpoints, etc.), so this matches each current-run
    point to its nearest baseline point and only keeps the pair if the two
    iterations are within half of the SMALLER run's own median checkpoint
    spacing -- close enough to be "the same point in training", not a
    fabricated interpolation across a large gap. Returns a list of
    (iteration, cur_val, base_val, delta) tuples, cur - base.
    """
    if not cur_iters or not base_iters:
        return []

    def _median_gap(iters):
        if len(iters) < 2:
            return None
        gaps = sorted(b - a for a, b in zip(iters[:-1], iters[1:]) if b > a)
        return gaps[len(gaps) // 2] if gaps else None

    cur_gap = _median_gap(cur_iters)
    base_gap = _median_gap(base_iters)
    gaps = [g for g in (cur_gap, base_gap) if g]
    tolerance = (min(gaps) / 2.0) if gaps else 500.0

    aligned = []
    for ci, cv in zip(cur_iters, cur_vals):
        best_bi, best_bv, best_dist = None, None, None
        for bi, bv in zip(base_iters, base_vals):
            dist = abs(bi - ci)
            if best_dist is None or dist < best_dist:
                best_bi, best_bv, best_dist = bi, bv, dist
        if best_dist is not None and best_dist <= tolerance:
            aligned.append((ci, cv, best_bv, cv - best_bv))
    return aligned


def _classify_curve_shape(aligned) -> str:
    """Simple, evidence-grounded shape classification (not a fancy
    curve-shape classifier -- three categories, each directly matching a
    real pattern seen in this project's own fr3 investigation, 2026-08-10):

    - "superimposable": delta stays small (<= _BASELINE_SMALL_DELTA_DB)
      across the ENTIRE aligned range, not just at the end -- e.g. R1/R2/R4/
      R5 vs. baseline in docs/architecture/2026-08-10-w3-66-fr3-admission-
      density-root-cause.md, the decisive evidence that four tested flags
      never touched the real mechanism at any point in training.
    - "converges": the curves were meaningfully apart somewhere mid-run
      (max |delta| > _BASELINE_LARGE_DELTA_DB) but end up close
      (|final delta| <= _BASELINE_SMALL_DELTA_DB) -- e.g. the baseline/R1
      arms' late-spike recovery relative to an already-healthy run, per
      docs/architecture/2026-08-10-w3-66-followup-assimilation-debt.md.
    - "diverges": the curves end up meaningfully apart
      (|final delta| > _BASELINE_LARGE_DELTA_DB), regardless of earlier shape.
    - "mixed": doesn't cleanly fit any of the above (small-but-not-tiny
      deltas throughout, or a final delta between the two thresholds) --
      still reported with real numbers, not forced into a category.
    - "insufficient_overlap": fewer than _BASELINE_MIN_ALIGNED_POINTS aligned
      checkpoints -- too little overlap to say anything about shape.
    """
    if len(aligned) < _BASELINE_MIN_ALIGNED_POINTS:
        return "insufficient_overlap"
    deltas = [d for _, _, _, d in aligned]
    max_abs_delta = max(abs(d) for d in deltas)
    final_delta = aligned[-1][3]
    if max_abs_delta <= _BASELINE_SMALL_DELTA_DB:
        return "superimposable"
    if abs(final_delta) <= _BASELINE_SMALL_DELTA_DB and max_abs_delta > _BASELINE_LARGE_DELTA_DB:
        return "converges"
    if abs(final_delta) > _BASELINE_LARGE_DELTA_DB:
        return "diverges"
    return "mixed"


def _find_latest_report_json(model_path: str) -> Optional[str]:
    """Find the most-complete real report.json under a run's comparison/ dir.

    Real reports live at ``<model_path>/comparison/iter_{N}/report.json``
    (see this module's own docstring), never directly at
    ``<model_path>/report.json`` -- that top-level path was never written by
    any real run and was a bug in the first version of
    ``compare_against_baseline`` (found 2026-08-11, silently made every
    ``--baseline_run`` comparison this session report ``comparable: False``).

    Picks the numerically-largest ``iter_{N}`` directory that has a
    ``report.json``, skipping non-numeric-suffixed variants like
    ``iter_52000_bootstrap_views`` (debug artifacts per this project's own
    run contract -- never a real generalization result) and using the
    directory NAME as the sort key, not file mtime, since the sentinel
    ``iter_2147483647`` used by online_mode runs must always win regardless
    of when it happened to be written.
    """
    pattern = os.path.join(model_path, "comparison", "iter_*", "report.json")
    candidates = []
    for path in glob.glob(pattern):
        dirname = os.path.basename(os.path.dirname(path))
        m = re.fullmatch(r"iter_(\d+)", dirname)
        if not m:
            continue
        candidates.append((int(m.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


def _admission_policy_snapshot(fingerprint) -> dict:
    """Extract the #795 contract fields from one report without recomputing.

    The persisted fingerprint is the authoritative realization record.  This
    intentionally copies it into both arms of ``--baseline_run`` output rather
    than deriving a second vocabulary or silently comparing only PSNR.
    """
    if not isinstance(fingerprint, dict):
        return {
            "available": False,
            "fingerprint": None,
            "owner_shares": None,
            "realized": None,
            "nominal_vs_realized": None,
            "controller": None,
        }
    controller = fingerprint.get("controller") or {}
    nominal_vs_realized = {
        "nominal_policy": fingerprint.get("nominal_policy"),
        "admission_realized_mismatch": fingerprint.get(
            "admission_realized_mismatch"
        ),
        "mismatch_reason": fingerprint.get("mismatch_reason"),
    }
    return {
        "available": True,
        "fingerprint": fingerprint,
        "owner_shares": fingerprint.get("owner_shares"),
        "realized": {
            "realized_admit_fps": fingerprint.get("realized_admit_fps"),
            "realized_admit_fps_source_time": fingerprint.get(
                "realized_admit_fps_source_time"
            ),
            "gap_percentiles": fingerprint.get("gap_percentiles"),
            "gap_histogram": fingerprint.get("gap_histogram"),
        },
        "nominal_vs_realized": nominal_vs_realized,
        "controller": {
            "velocity_adaptive": controller.get("velocity_adaptive"),
            "timebase_source": controller.get("timebase_source"),
            "controller_target_interval_s": controller.get(
                "controller_target_interval_s"
            ),
        },
    }


def _classify_admission_arm(snapshot: dict) -> str:
    """Classify one arm's nominal-vs-realized admission evidence.

    This is deliberately arm-local: an A/B treatment can legitimately change
    nominal policy, controller settings, owner shares, realized FPS, and gap
    distribution.  None of those observed or declared treatment differences
    are evidence that either arm failed to realize *its own* nominal policy.
    """
    if not snapshot["available"]:
        return "UNKNOWN"

    verdict = snapshot["nominal_vs_realized"]
    nominal_policy = verdict["nominal_policy"]
    mismatch = verdict["admission_realized_mismatch"]
    reason = verdict["mismatch_reason"]
    if not nominal_policy or not isinstance(reason, str):
        return "UNKNOWN"
    if mismatch is True:
        return "MISMATCH"
    if reason.startswith("INSUFFICIENT_ADMITS"):
        return "UNKNOWN"
    if nominal_policy == "all":
        return (
            "NOT_APPLICABLE"
            if mismatch is False and reason == "NON_GATED_ADMISSION"
            else "UNKNOWN"
        )
    if mismatch is False and reason == "NOMINAL_POLICY_REALIZED":
        return "REALIZED"
    return "UNKNOWN"


def _admission_claim_eligibility(current_status: str, baseline_status: str) -> dict:
    """Fail closed only for admission-policy-derived claims.

    Holdout/quality reporting remains governed by its own evidence ladder.
    A non-gated ``NOT_APPLICABLE`` arm is not an invalid realization; an
    experiment can intentionally compare it with a realized gated treatment.
    """
    statuses = {"current": current_status, "baseline": baseline_status}
    mismatch_arms = [arm for arm, status in statuses.items() if status == "MISMATCH"]
    if mismatch_arms:
        return {
            "eligible": False,
            "status": "BLOCKED_MISMATCH",
            "reason": (
                "admission-policy claims are blocked because "
                f"{', '.join(mismatch_arms)} arm realization is MISMATCH"
            ),
            "blocking_arms": mismatch_arms,
        }
    unknown_arms = [arm for arm, status in statuses.items() if status == "UNKNOWN"]
    if unknown_arms:
        return {
            "eligible": False,
            "status": "UNKNOWN",
            "reason": (
                "admission-policy claims are unavailable because "
                f"{', '.join(unknown_arms)} arm realization is UNKNOWN"
            ),
            "blocking_arms": unknown_arms,
        }
    return {
        "eligible": True,
        "status": "ELIGIBLE",
        "reason": "each arm has realized admission evidence or is non-gated",
        "blocking_arms": [],
    }


def _comparison_semantic_verdict(current_status: str, baseline_status: str) -> str:
    """Compatibility summary derived from independent arm statuses.

    It is intentionally *not* an equality verdict.  Observed outcomes and
    declared treatment axes are retained side by side, and can be assessed
    for A/A equivalence only under a separately predeclared tolerance.
    """
    statuses = {current_status, baseline_status}
    if "MISMATCH" in statuses:
        return "MISMATCH"
    if "UNKNOWN" in statuses:
        return "UNKNOWN"
    if statuses == {"NOT_APPLICABLE"}:
        return "NOT_APPLICABLE"
    return "REALIZED"


def _compare_admission_policy(current_fingerprint, baseline_fingerprint) -> dict:
    """Report both arms while assessing nominal realization independently."""
    current = _admission_policy_snapshot(current_fingerprint)
    baseline = _admission_policy_snapshot(baseline_fingerprint)
    current_status = _classify_admission_arm(current)
    baseline_status = _classify_admission_arm(baseline)
    # Surface the independent verdict inside each persisted-arm view as well
    # as at the comparison level, so downstream report readers do not have
    # to infer it from cross-arm equality.
    current["semantic_status"] = current_status
    baseline["semantic_status"] = baseline_status
    return {
        "current": current,
        "baseline": baseline,
        "current_status": current_status,
        "baseline_status": baseline_status,
        "semantic_verdict": _comparison_semantic_verdict(
            current_status, baseline_status
        ),
        "claim_eligibility": _admission_claim_eligibility(
            current_status, baseline_status
        ),
        # These are declared treatment axes, not equality gates.  Keep both
        # values in the canonical comparison report so A/B readers can see
        # what changed without a second telemetry vocabulary.
        "declared_treatment_axes": {
            "nominal_policy": {
                "current": current["nominal_vs_realized"].get("nominal_policy")
                if current["available"] else None,
                "baseline": baseline["nominal_vs_realized"].get("nominal_policy")
                if baseline["available"] else None,
            },
            "controller_velocity_adaptive": {
                "current": current["controller"].get("velocity_adaptive")
                if current["available"] else None,
                "baseline": baseline["controller"].get("velocity_adaptive")
                if baseline["available"] else None,
            },
            "controller_timebase_source": {
                "current": current["controller"].get("timebase_source")
                if current["available"] else None,
                "baseline": baseline["controller"].get("timebase_source")
                if baseline["available"] else None,
            },
            "controller_target_interval_s": {
                "current": current["controller"].get("controller_target_interval_s")
                if current["available"] else None,
                "baseline": baseline["controller"].get("controller_target_interval_s")
                if baseline["available"] else None,
            },
        },
    }


def compare_against_baseline(
    baseline_run: str,
    current_summary: dict,
    current_model_path: str,
) -> dict:
    """Compare the current run against a previously-completed baseline run.

    Opt-in (only called when --baseline_run is set). Never raises -- any
    failure (missing path, missing report.json, unreadable psnr_history)
    produces a ``{"comparable": False, "reason": ...}`` dict instead of
    interrupting the caller's real report-writing. Compares the FULL
    psnr_history curve, not just the final PSNR, per CLAUDE.md's "Compare
    PSNR curves, not just the final number" run contract -- see
    ``_classify_curve_shape`` for why a final-number-only diff is
    insufficient (superimposable-but-endpoint-only-checked flags, and a
    late-spike-recovery that read as "gradual" by endpoint alone, both real
    cases from this project's own fr3 investigation).

    Holdout comparability (see GitHub issue #199 -- ``holdout_hash`` matching
    does NOT by itself guarantee comparability, since it hashes the holdout
    *selection rule*, not the test set actually realized/evaluated; three
    real fr3 runs shared one ``holdout_hash`` while realizing n_test=52/62/
    102). Issue #420 extended this with a further distinction: even an
    identical realized POOL is not the set metrics were computed over --
    ``write_post_training_report`` subsamples the pool down to
    ``--streaming_report_test_metrics_max`` (and drops render failures)
    before averaging PSNR/SSIM/LPIPS. Layers, strongest available wins:

    1. ``realized_holdout_hash`` matches on both sides AND
       ``eval_metadata["scored_holdout_hash"]`` (issue #420: hash of the
       cameras actually scored, after cap/subsampling) is present on both
       and matches ⇒ ``verification_level="scored_hash"``,
       ``holdout_verified=True``. This is the ONLY configuration that now
       earns ``holdout_verified is True``. A scored-hash mismatch under a
       matching pool hash is refused outright (``"scored_hash_mismatch"``).
       Note the scored check TIGHTENS the pool check rather than replacing
       it -- it is only consulted once the pool hash already matched, so a
       report carrying a scored hash but no pool hash can never be promoted.
    2. ``realized_holdout_hash`` matches but the scored subset's exact
       provenance is unavailable on at least one side (a legacy report) ⇒
       DOWNGRADED to ``verification_level="realized_hash"``,
       ``holdout_verified="realized_hash"`` (truthy, but deliberately not
       ``True``). ``n_test_metrics`` acts as a weak legacy guard here: if
       both sides carry it and it differs, the comparison is refused
       (``"n_test_metrics_mismatch"``) -- two runs launched with different
       ``--streaming_report_test_metrics_max`` caps averaged over different
       subset sizes. Differing pool hashes are still refused outright
       (``"realized_hash_mismatch"``).
    3. One or both sides predate ``realized_holdout_hash`` (a legacy
       report.json): falls back to comparing ``n_test`` directly. Differing
       ⇒ refused (``"n_test_mismatch"``, the issue #199 scenario). Equal,
       with both sides also carrying an equal ``n_test_metrics`` ⇒
       ``"n_test_metrics_only"``; equal ``n_test`` with an unavailable
       ``n_test_metrics`` on either side ⇒ ``"n_test_only"``. Both are
       honest about ruling out only *size* mismatches -- two equal-size
       sets covering different trajectory segments still can't be told
       apart at these levels. An equal ``n_test`` with a DIFFERING
       ``n_test_metrics`` is refused (``"n_test_metrics_mismatch"``).
    4. Neither ``realized_holdout_hash`` nor a comparable ``n_test`` is
       available: falls back to the original selection-rule ``holdout_hash``
       check alone (``verification_level="hash_only"`` if it matches,
       ``"unverified"`` otherwise) -- reproduces this function's pre-#199
       behavior exactly, so old reports missing every new field still
       produce a caveated reference point (never a false "incomparable").

    Backward compatibility: a report missing every new field degrades down
    this ladder rather than crashing, and can never be promoted to
    ``holdout_verified is True``.

    Issue #745 follow-up (2026-08-22): a report whose ``eval_metadata
    ["holdout_selection"]`` is the legacy live/bag marker
    ``"online_every_nth_hashed"`` predates the #745 fix -- its
    ``holdout_hash`` is not a selection-RULE hash at all, it is the
    REALIZED test-camera-set hash (see ``scene/streaming_scene.py``'s
    ``eval_metadata()`` docstring), written under the rule-hash field name.
    A bare ``holdout_hash`` equality against (or between) such reports proves
    nothing about the selection rule -- it can only ever be a coincidental
    realized-set match, exactly the "never a naive ``holdout_hash`` equality
    check implying rule-level comparability" gap #745 named. Whenever either
    side carries this marker, tiers 3-4 above (the ones that would otherwise
    grant truthy ``holdout_verified`` credit purely from ``cur_hash ==
    base_hash``) are downgraded: no hash-based credit is granted, and
    ``verification_level`` is instead ``"legacy_live_realized_set_hash"``
    when the (spurious) hash still happens to match, so a caller can see WHY
    the ladder refused to trust it rather than reading a plain
    ``"unverified"``. Tiers 1-2 (``realized_holdout_hash``/
    ``scored_holdout_hash``) are unaffected -- they never depended on
    ``holdout_hash`` for their credit. A report written by the post-#745
    live branch carries ``holdout_selection ==
    "online_every_nth_ordinal_rule"`` (or ``"online_every_nth"``/
    ``"precomputed"``) and is never downgraded by this check.

    If BOTH selection-rule hashes are present and DIFFER, the comparison is
    refused outright regardless of the above -- CLAUDE.md's existing "Real
    test holdouts" contract already treats differing hashes as flatly
    incomparable.

    Issue #745 follow-up, second gap (2026-08-22): a ``ros2://live`` bag-
    replay run's ``eval_metadata["holdout_hash"]`` rule payload folds in
    ``args.source_path`` for scene identity -- but that field is the LITERAL
    CONSTANT STRING ``"ros2://live"`` for every such run (the real bag is
    chosen by an external ``ros2 bag play`` process, invisible to this
    process's own args; see ``utils/comparison_report.py``'s
    ``_scene_fingerprint`` docstring, "Robust to run names and bag paths
    (which never reach cfg_args on the ros2://live replay path)"). Two runs
    of entirely DIFFERENT bags sharing the same ``--streaming_eval_hold``/
    ``--streaming_eval_guard`` therefore collide into the same rule hash --
    the opposite failure from #745's original defect (a false MATCH instead
    of a false mismatch), and the same "floor3 vs. handheld bag" scene-
    identity drift ``_scene_fingerprint`` (a content hash of the actual GT
    pixels, immune to this) already exists to catch for a different
    contract -- but ``compare_against_baseline`` never consulted it. This
    function now does: whenever both sides carry a ``scene_fingerprint`` and
    they differ, the comparison is refused outright (``"reason"`` names it
    explicitly), same standing as a ``holdout_hash`` mismatch. A missing
    ``scene_fingerprint`` on either side (older reports, or a report with no
    test/train cameras to fingerprint) skips this check entirely --
    behavior-neutral for every report written before this addition. This is
    deliberately an ADDITIVE new check reusing an existing, independently-
    computed field, not a change to how ``holdout_hash`` itself is computed
    -- folding a normalized scene identity into the rule hash directly was
    considered and rejected: ``SCENE_SOURCE`` (the one candidate identifier
    that IS set to the real bag path) is a raw, unnormalized absolute path
    that is not guaranteed stable across machines/mounts (this project's own
    ``utils/partition_eval.py`` registry lookup already needs an explicit
    exact-path-then-basename-fallback scheme for exactly this reason), so
    using it directly in the hash would trade today's false-match risk for a
    false-mismatch risk across a multi-machine ablation -- a worse failure
    direction for #745's own stated goal of making bag-source comparisons
    satisfiable. Left for human judgment on whether a normalized identifier
    is worth adding later.
    """
    result = {
        "baseline_run": baseline_run,
        "comparable": False,
        # Explicit dimension for callers that must not mistake admission
        # realization evidence for holdout/quality-curve comparability.
        "holdout_quality_comparable": False,
        "holdout_verified": False,
        "verification_level": "unverified",
        "reason": None,
    }

    if not baseline_run:
        result["reason"] = "no baseline_run specified"
        return result

    try:
        if not os.path.isdir(baseline_run):
            result["reason"] = f"baseline_run path does not exist: {baseline_run}"
            return result
        baseline_report_path = _find_latest_report_json(baseline_run)
        if baseline_report_path is None:
            result["reason"] = (
                f"no comparison/iter_*/report.json found under baseline_run: {baseline_run}"
            )
            return result
        with open(baseline_report_path) as f:
            baseline_report = json.load(f)
    except Exception as e:
        result["reason"] = f"failed to load baseline report.json: {e!r}"
        return result

    admission_policy_comparison = _compare_admission_policy(
        current_summary.get("admission_fingerprint"),
        baseline_report.get("admission_fingerprint"),
    )
    result["admission_policy_comparison"] = admission_policy_comparison
    result["admission_policy_claim_eligible"] = admission_policy_comparison[
        "claim_eligibility"
    ]["eligible"]

    # Issue #745 follow-up, second gap: refuse outright if both sides carry a
    # scene_fingerprint (content hash of actual GT pixels, see
    # utils/comparison_report.py's _scene_fingerprint) and it differs -- a
    # ros2://live run's holdout_hash rule payload cannot itself discriminate
    # between two different bags (see this function's docstring). This is a
    # top-level check like the holdout_hash mismatch below: two genuinely
    # different scenes are incomparable no matter what any hash-ladder tier
    # below would otherwise conclude.
    cur_fp = current_summary.get("scene_fingerprint")
    base_fp = baseline_report.get("scene_fingerprint")
    if cur_fp and base_fp and cur_fp != base_fp:
        result["reason"] = (
            f"scene_fingerprint mismatch: this run={cur_fp} baseline={base_fp} -- "
            "runs were evaluated on different scenes (this content hash of the actual "
            "GT pixels is immune to ros2://live's source_path=='ros2://live' identity "
            "gap, see issue #745). Comparison refused (not just uncertain)."
        )
        result["verification_level"] = "scene_fingerprint_mismatch"
        return result

    cur_meta = current_summary.get("eval_metadata") or {}
    base_meta = baseline_report.get("eval_metadata") or {}
    cur_hash = cur_meta.get("holdout_hash")
    base_hash = base_meta.get("holdout_hash")
    # Issue #745 follow-up: see this function's docstring, "Issue #745
    # follow-up" section. A report predating the #745 fix records the
    # REALIZED test-camera-set hash under the `holdout_hash` field name (the
    # exact defect #745 named), distinguishable only via `holdout_selection`.
    _LEGACY_LIVE_REALIZED_SET_MARKER = "online_every_nth_hashed"
    _legacy_live_hash = (
        cur_meta.get("holdout_selection") == _LEGACY_LIVE_REALIZED_SET_MARKER
        or base_meta.get("holdout_selection") == _LEGACY_LIVE_REALIZED_SET_MARKER
    )
    if cur_hash and base_hash and cur_hash != base_hash:
        result["reason"] = (
            f"holdout_hash mismatch: this run={cur_hash} baseline={base_hash} -- "
            "runs use different test splits, comparison refused (not just uncertain)."
        )
        result["verification_level"] = "hash_mismatch"
        return result

    cur_realized = cur_meta.get("realized_holdout_hash")
    base_realized = base_meta.get("realized_holdout_hash")
    cur_scored = cur_meta.get("scored_holdout_hash")
    base_scored = base_meta.get("scored_holdout_hash")
    cur_n_test = current_summary.get("n_test")
    base_n_test = baseline_report.get("n_test")
    # NOTE: n_test_metrics lives at the TOP LEVEL of the report/summary
    # (summary["n_test_metrics"]), not inside eval_metadata -- reading it
    # from cur_meta/base_meta would make this guard silently always-None.
    cur_n_metrics = current_summary.get("n_test_metrics")
    base_n_metrics = baseline_report.get("n_test_metrics")

    holdout_verified = False
    verification_level = "unverified"
    refuse_reason = None

    def _scored_subset_check():
        """Issue #420, applied once the realized POOL is already verified.

        Returns (refuse_reason, verification_level, holdout_verified).
        Deliberately tightens rather than replaces the pool check: it is
        only ever reached under a matching realized_holdout_hash, so a
        report carrying a scored hash but no pool hash can never be
        promoted past the weaker tiers below.
        """
        if cur_scored and base_scored:
            if cur_scored != base_scored:
                return (
                    f"scored_holdout_hash mismatch: this run={cur_scored} "
                    f"baseline={base_scored} (realized pool hash MATCHES on both "
                    f"sides: {cur_realized}; n_test={cur_n_test}/{base_n_test}, "
                    f"n_test_metrics={cur_n_metrics}/{base_n_metrics}) -- PSNR/SSIM/"
                    "LPIPS were averaged over different camera subsets of the same "
                    "holdout pool (see --streaming_report_test_metrics_max "
                    "subsampling, issue #420). Comparison refused (not just "
                    "uncertain).",
                    "scored_hash_mismatch",
                    False,
                )
            return (None, "scored_hash", True)
        # Exact scored-subset provenance unavailable on at least one side.
        # Fall back to n_test_metrics as a weak size-only guard: it only
        # discriminates when the two runs used different
        # --streaming_report_test_metrics_max caps (whenever
        # n_test <= cap, n_test_metrics == n_test and it adds nothing over
        # the n_test check). Mismatch refuses, consistent with the existing
        # n_test-mismatch refusal.
        if cur_n_metrics is not None and base_n_metrics is not None \
                and cur_n_metrics != base_n_metrics:
            return (
                f"n_test_metrics differs: this run={cur_n_metrics} "
                f"baseline={base_n_metrics} -- mean_test_psnr was averaged over "
                "different numbers of views despite a matching realized holdout "
                "pool (see issue #420); neither report carries a "
                "scored_holdout_hash so this is a fallback size check, not a "
                "definitive one. Comparison refused (not just uncertain).",
                "n_test_metrics_mismatch",
                False,
            )
        # Pool verified, scored subset unverifiable -> DOWNGRADED from the
        # top level. `holdout_verified` becomes the narrower truthy marker
        # "realized_hash" (same convention as the pre-existing
        # "n_test_only") rather than True: True is now reserved for a
        # verified scored subset.
        return (None, "realized_hash", "realized_hash")

    if cur_realized and base_realized:
        # Pool-level signal: hash of the realized holdout set.
        if cur_realized != base_realized:
            refuse_reason = (
                f"realized_holdout_hash mismatch: this run={cur_realized} "
                f"baseline={base_realized} (n_test this run={cur_n_test} "
                f"baseline={base_n_test}) -- runs realized different test sets despite "
                "a matching selection-rule holdout_hash; see issue #199. "
                "Comparison refused (not just uncertain)."
            )
            verification_level = "realized_hash_mismatch"
        else:
            refuse_reason, verification_level, holdout_verified = _scored_subset_check()
    elif cur_n_test is not None and base_n_test is not None:
        # Graceful degradation: one or both reports predate
        # realized_holdout_hash. n_test equality rules out the *size*
        # mismatch issue #199 found, but cannot rule out two equal-size
        # realized sets covering different trajectory segments -- report
        # that honestly instead of a false "verified".
        if cur_n_test != base_n_test:
            refuse_reason = (
                f"n_test differs: this run={cur_n_test} baseline={base_n_test} -- "
                "realized test sets are not guaranteed comparable even though "
                "holdout_hash matches (see issue #199); one or both reports predate "
                "realized_holdout_hash so this is a fallback size check, not a "
                "definitive one. Comparison refused (not just uncertain)."
            )
            verification_level = "n_test_mismatch"
        elif cur_n_metrics is not None and base_n_metrics is not None \
                and cur_n_metrics != base_n_metrics:
            # Issue #420: equal n_test but a different number of views was
            # actually scored (different --streaming_report_test_metrics_max).
            refuse_reason = (
                f"n_test_metrics differs: this run={cur_n_metrics} "
                f"baseline={base_n_metrics} (n_test matches at {cur_n_test}) -- "
                "mean_test_psnr was averaged over different numbers of views; see "
                "issue #420. Comparison refused (not just uncertain)."
            )
            verification_level = "n_test_metrics_mismatch"
        else:
            # Issue #745 follow-up: never grant hash-based credit from a
            # legacy live/bag realized-set hash masquerading as a rule hash.
            _hash_ok = bool(cur_hash and base_hash and cur_hash == base_hash) \
                and not _legacy_live_hash
            if cur_n_metrics is not None and base_n_metrics is not None:
                # Both sides at least agree on how many views were scored --
                # a notch above bare n_test equality, still far below a
                # scored-subset hash (equal counts say nothing about WHICH
                # cameras were scored).
                verification_level = "n_test_metrics_only"
                holdout_verified = "n_test_metrics_only" if _hash_ok else False
            else:
                verification_level = "n_test_only"
                holdout_verified = "n_test_only" if _hash_ok else False
    else:
        _hashes_match = bool(cur_hash and base_hash and cur_hash == base_hash)
        if _hashes_match and _legacy_live_hash:
            # Issue #745 follow-up: the hashes match, but at least one side's
            # `holdout_hash` is a legacy live/bag REALIZED-set hash under the
            # rule-hash field name -- a match here is coincidental, not
            # evidence of a shared selection rule. Never promote to
            # `holdout_verified`, and say so explicitly rather than reading
            # as a plain "unverified" (which would look like no signal at
            # all, rather than a signal that was deliberately discounted).
            holdout_verified = False
            verification_level = "legacy_live_realized_set_hash"
        else:
            holdout_verified = _hashes_match and not _legacy_live_hash
            verification_level = "hash_only" if holdout_verified else "unverified"

    if refuse_reason:
        result["reason"] = refuse_reason
        result["verification_level"] = verification_level
        return result

    # Issue #420: the top level is now "scored_hash" (verified scored subset),
    # so a bare "realized_hash" (pool verified, scored subset unverifiable --
    # e.g. a legacy baseline) correctly falls into this caveat print.
    if verification_level != "scored_hash":
        print(
            f"[report] baseline_comparison: verification_level={verification_level!r} "
            f"(this run: hash={cur_hash!r} realized={cur_realized!r} "
            f"scored={cur_scored!r} n_test={cur_n_test!r} n_test_metrics={cur_n_metrics!r}; "
            f"baseline: hash={base_hash!r} realized={base_realized!r} "
            f"scored={base_scored!r} n_test={base_n_test!r} n_test_metrics={base_n_metrics!r}) "
            "-- proceeding as a caveated reference point, not a certain apples-to-apples "
            "comparison.",
            flush=True,
        )

    try:
        from eligsir.utils.psnr_history_io import psnr_reduction, read_psnr_history
        base_iters, base_vals = read_psnr_history(baseline_run)
        current_psnr_iters, current_psnr_vals = read_psnr_history(current_model_path)
        current_rows = _read_valid_psnr_rows(current_model_path)
        baseline_rows = _read_valid_psnr_rows(baseline_run)
        current_reduction = psnr_reduction(
            current_rows,
            source=os.path.join(current_model_path, "psnr_history.jsonl"),
        )
        baseline_reduction = psnr_reduction(
            baseline_rows,
            source=os.path.join(baseline_run, "psnr_history.jsonl"),
        )
    except Exception as e:
        result["reason"] = f"failed to read psnr_history.jsonl: {e!r}"
        return result
    if (
        current_reduction is not None
        and baseline_reduction is not None
        and current_reduction != baseline_reduction
    ):
        result["reason"] = (
            "PSNR reduction mismatch: "
            f"this run={current_reduction!r} baseline={baseline_reduction!r}; "
            "comparison refused"
        )
        result["verification_level"] = "psnr_reduction_mismatch"
        return result

    result["comparable"] = True
    result["holdout_quality_comparable"] = True
    result["holdout_verified"] = holdout_verified
    result["verification_level"] = verification_level
    result["baseline_holdout_hash"] = base_hash
    result["baseline_realized_holdout_hash"] = base_realized
    result["baseline_scored_holdout_hash"] = base_scored
    result["baseline_n_test_metrics"] = base_n_metrics
    result["baseline_mean_test_psnr"] = baseline_report.get("mean_test_psnr")
    result["baseline_eval_valid"] = baseline_report.get("eval_valid")
    result["baseline_n_test"] = baseline_report.get("n_test")
    if current_reduction is not None:
        result["psnr_reduction"] = current_reduction

    # Issue #346, INFORMATIONAL ONLY: surface both sides' seed-fade severity so
    # a reader (or downstream tool) can see that one/both sides of this
    # comparison are seed-fade-confounded. Deliberately does NOT influence
    # `comparable`: flipping that on seed fade would silently invalidate every
    # existing --baseline_run comparison, a semantics change well outside this
    # issue's reporting-only scope. The current side's grade is already in
    # current_summary["seed_fade"] (graded in write_post_training_report); the
    # baseline's is graded here from its own report + psnr_history on disk.
    try:
        from eligsir.utils.seed_fade_severity import grade_seed_fade_for_run

        _cur_sf = current_summary.get("seed_fade") or {}
        result["seed_fade_severity"] = _cur_sf.get("severity")
        result["seed_fade_confounded"] = bool(_cur_sf.get("confounded", False))
        _base_sf = grade_seed_fade_for_run(baseline_report.get("seed_fade"), baseline_run)
        result["baseline_seed_fade_severity"] = _base_sf.get("severity")
        result["baseline_seed_fade_confounded"] = bool(_base_sf.get("confounded", False))
        result["either_side_seed_fade_confounded"] = bool(
            result["seed_fade_confounded"] or result["baseline_seed_fade_confounded"]
        )
        # Issue #290: a resumed run on EITHER side means the severity numbers
        # just compared are graded off a running max that restarted mid-run.
        # Same treatment as the confounded pair above -- surfaced, and
        # deliberately not allowed to flip `comparable`. The baseline side is
        # read from its own report (both the top-level mirror and the nested
        # key, since only one may exist on an older report).
        result["seed_fade_indeterminate"] = bool(_cur_sf.get("indeterminate", False))
        result["baseline_seed_fade_indeterminate"] = bool(
            baseline_report.get("seed_fade_indeterminate")
            or (baseline_report.get("seed_fade") or {}).get("indeterminate")
        )
        result["either_side_seed_fade_indeterminate"] = bool(
            result["seed_fade_indeterminate"] or result["baseline_seed_fade_indeterminate"]
        )
    except Exception:
        pass

    cur_final = current_summary.get("mean_test_psnr")
    base_final = result["baseline_mean_test_psnr"]
    result["final_psnr_delta"] = (
        (cur_final - base_final) if (cur_final is not None and base_final is not None) else None
    )

    # Primary cross-batch/cross-throughput evidence.  Each arm is normalized
    # to its own durable mapping endpoint before alignment; absolute optimizer
    # iteration is retained below only for backward-compatible diagnostics.
    result["phase_relative_comparison"] = _compare_phase_relative_psnr(
        current_model_path, baseline_run
    )

    if not base_iters:
        result["reason"] = "baseline_run has no psnr_history.jsonl entries -- endpoint-only comparison"
        result["curve_shape"] = "insufficient_overlap"
        _label_legacy_iteration_comparison(result)
        return result
    if not current_psnr_iters:
        result["reason"] = "this run has no psnr_history entries yet -- endpoint-only comparison"
        result["curve_shape"] = "insufficient_overlap"
        _label_legacy_iteration_comparison(result)
        return result

    aligned = _align_psnr_curves(current_psnr_iters, current_psnr_vals, base_iters, base_vals)
    result["n_aligned_checkpoints"] = len(aligned)
    result["curve_shape"] = _classify_curve_shape(aligned)
    if aligned:
        deltas = [d for _, _, _, d in aligned]
        result["curve_mean_delta"] = sum(deltas) / len(deltas)
        result["curve_max_abs_delta"] = max(abs(d) for d in deltas)
        result["curve_min_delta"] = min(deltas)
        result["curve_max_delta"] = max(deltas)
        # Full aligned series for anyone who wants to re-plot it -- kept
        # compact (iteration + delta only, not both raw values, since those
        # are already in each run's own psnr_history.jsonl).
        result["curve_deltas_by_iteration"] = {str(it): d for it, _, _, d in aligned}
    _label_legacy_iteration_comparison(result)
    return result


# ---------------------------------------------------------------------------
# Realized/scored holdout-camera identity hashing (issues #199 / #420).
#
# Pure-python (hashlib + getattr over camera objects), so it lives here rather
# than in the torch-importing comparison_report module -- see this file's
# top-level docstring. Re-exported from comparison_report for existing callers.
# ---------------------------------------------------------------------------

def _camera_set_hash(cams) -> Optional[str]:
    """Order-independent content hash of an arbitrary camera list.

    Shared implementation behind both ``_realized_holdout_hash`` (the
    realized holdout *pool*) and ``scored_holdout_hash`` (issue #420: the
    subset of that pool metrics were actually computed over, after the
    ``--streaming_report_test_metrics_max`` cap and after dropping any
    camera whose render raised). Both need the exact same identifier
    policy, so it lives in one place rather than being reimplemented and
    left to drift.

    See ``_realized_holdout_hash``'s docstring for the identifier policy
    (``_streaming_frame.index`` → ``image_name`` → withhold entirely) and
    why it deliberately refuses to fall back to ``cam.uid``.
    """
    import hashlib

    if not cams:
        return None
    ids = []
    for c in cams:
        frame = getattr(c, "_streaming_frame", None)
        if frame is not None and getattr(frame, "index", None) is not None:
            ids.append(f"idx:{frame.index}")
            continue
        name = getattr(c, "image_name", None)
        if name is not None:
            ids.append(f"name:{name}")
            continue
        return None
    ids.sort()
    return hashlib.sha1(",".join(ids).encode("utf-8")).hexdigest()[:16]


def _realized_holdout_hash(test_cams) -> Optional[str]:
    """Content hash of the test cameras *actually realized/evaluated* by this
    run -- distinct from ``eval_metadata["holdout_hash"]``, which hashes the
    holdout *selection rule* (e.g. the full every-Nth candidate set computed
    from the whole known trajectory) and can be IDENTICAL across two runs
    whose realized test sets differ in size (see GitHub issue #199: three
    fr3 runs differing only in ``--iterations`` shared one ``holdout_hash``
    while realizing n_test=52/62/102 respectively -- for a streaming run,
    admission is iteration-driven, so a shorter run reaches fewer of the
    reserved holdout frames even though the selection RULE is identical).

    Hashes the sorted list of stable per-camera identifiers -- preferring
    ``cam._streaming_frame.index`` (the source-dataset frame index, stable
    across runs of the same scene) and falling back to ``cam.image_name``
    for non-streaming (COLMAP/offline) cameras that never carry a
    ``_streaming_frame`` (``image_name`` is the on-disk source filename,
    stable across runs of the identical scene). Deliberately does NOT fall
    back further to ``cam.uid`` -- in some code paths ``uid`` is only an
    arrival-order counter, not a scene-stable identifier, and hashing it
    could manufacture a false mismatch between two runs that actually
    realized the same test set. If a camera has neither a usable
    ``_streaming_frame.index`` nor an ``image_name``, the whole hash is
    withheld (``None``) rather than computed from an unreliable id --
    callers already treat ``None`` as "not verifiable at this level" and
    fall back to the ``n_test`` check, which is a safer degradation than a
    hash that looks authoritative but isn't.

    NOTE (issue #420): this hashes the realized *pool*, which is NOT the set
    metrics were computed over -- ``write_post_training_report`` subsamples
    the pool to ``--streaming_report_test_metrics_max`` before scoring. The
    scored subset is fingerprinted separately as
    ``eval_metadata["scored_holdout_hash"]``.
    """
    return _camera_set_hash(test_cams)
