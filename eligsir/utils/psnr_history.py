"""Lightweight, always-on test-PSNR-over-time tracking for streaming/online runs.

Unlike write_post_training_report (contact sheets, mp4, geometry metrics --
minute-scale), this only renders the held-out test cameras and computes PSNR:
cheap enough to run synchronously on the training thread at a fixed interval.
Per CLAUDE.md's Run Contracts, a psnr-over-time chart must always be produced
-- even for online_mode runs -- not gated behind an opt-in flag the way the
heavy mid-training report (--streaming_refinement_eval_interval) is.

Scientific PSNR is computed only through :mod:`utils.score_contract`: the
render and GT must have matching float score domains, explicit camera
provenance is honored when present, and visualization/serialization lineage
is rejected before a value can enter the history.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import torch

from eligsir.utils.psnr_history_io import (
    PSNR_REDUCTION_RGB_PER_VIEW_V2,
    psnr_history_tail_reduction,
)
from eligsir.plot_style import svg_output_path
from eligsir.utils.score_contract import (
    apply_declared_score_clamp_pair,
    canonical_score_batch,
    scientific_psnr,
)


@dataclass
class PsnrTickResult:
    """One PSNR-tick's worth of per-view results, GPU-resident."""

    cams: List
    psnr_tensor: "torch.Tensor | None"
    mean_psnr: Optional[float]
    score_domain_clamp: Optional[dict] = None
    edge_psnr_tensor: "torch.Tensor | None" = None
    mean_edge_psnr: Optional[float] = None
    temporal_flicker_rms: Optional[float] = None
    temporal_flicker_psnr_db: Optional[float] = None
    temporal_flicker_valid_pairs: int = 0
    temporal_flicker_status: str = "not_computed"
    rasterize_mode: str = "classic"
    gaussian_count: Optional[int] = None
    training_resolution_tiers: tuple[str, ...] = ()
    evaluation_resolutions: tuple[tuple[int, int], ...] = ()
    psnr_reduction: str = PSNR_REDUCTION_RGB_PER_VIEW_V2


def score_per_view_rgb_psnr(render: torch.Tensor, gt: torch.Tensor, *, context: str) -> torch.Tensor:
    """Score one canonical CHW image as one NCHW RGB image."""
    return scientific_psnr(
        canonical_score_batch(render, context=f"{context}:render_batch"),
        canonical_score_batch(gt, context=f"{context}:gt_batch"),
        context=context,
    )


def compute_per_view_test_psnr_gpu(
    gaussians,
    test_cams: List,
    render_fn,
    pipe,
    background,
    per_view_hook=None,
    clamp_records_out: Optional[List[dict]] = None,
    score_fn=score_per_view_rgb_psnr,
):
    """Render every test camera once, returning ``(cams, psnr_tensor)``.

    ``psnr_tensor`` remains on-device. Any render, GT, or score-contract
    failure rejects the whole tick so the curve cannot silently average a
    failure-selected subset of views.
    """
    if not test_cams:
        return [], None
    from eligsir.utils.comparison_report import _render_image, _gt_image

    cams: List = []
    vals: List[torch.Tensor] = []
    with torch.no_grad():
        for cam in test_cams:
            render_img = _render_image(cam, gaussians, render_fn, pipe, background)
            gt = _gt_image(cam)
            render_score, gt_score, clamp_record = apply_declared_score_clamp_pair(
                render_img,
                gt,
                context="psnr_history:test_holdout",
            )
            vals.append(
                score_fn(
                    render_score,
                    gt_score,
                    context="psnr_history:test_holdout",
                ).mean()
            )
            cams.append(cam)
            if clamp_records_out is not None:
                clamp_records_out.append(clamp_record)
            if per_view_hook is not None:
                try:
                    per_view_hook(cam, render_score, gt_score)
                except Exception:
                    # Auxiliary telemetry must not alter an already-computed
                    # canonical score when its hook fails.
                    pass
    if not vals:
        return [], None
    return cams, torch.stack(vals)


def compute_per_view_test_psnr(gaussians, test_cams: List, render_fn, pipe, background) -> List[tuple]:
    """Render every test camera once, returning ``[(cam, psnr_float), ...]``."""
    cams, psnr_tensor = compute_per_view_test_psnr_gpu(gaussians, test_cams, render_fn, pipe, background)
    if psnr_tensor is None:
        return []
    return list(zip(cams, (float(v) for v in psnr_tensor.detach().cpu().tolist())))


def compute_mean_test_psnr(gaussians, test_cams: List, render_fn, pipe, background) -> Optional[float]:
    _cams, psnr_tensor = compute_per_view_test_psnr_gpu(gaussians, test_cams, render_fn, pipe, background)
    if psnr_tensor is None:
        return None
    return float(psnr_tensor.mean().item())


def _aggregate_clamp_records(records: List[dict]) -> dict:
    result = {}
    for side in ("render", "gt"):
        side_records = [record[side] for record in records]
        changed = sum(record["samples_changed"] for record in side_records)
        total = sum(record["samples_total"] for record in side_records)
        result[side] = {
            "range": [0.0, 1.0],
            "nonfinite_policy": "rejected before clamp",
            "samples_changed": changed,
            "samples_total": total,
            "fraction_changed": changed / max(1, total),
            "value_min": min((record["value_min"] for record in side_records), default=0.0),
            "value_max": max((record["value_max"] for record in side_records), default=0.0),
        }
    return result


def append_psnr_history(
    model_path: str,
    iteration: int,
    mean_psnr: float,
    score_domain_clamp: Optional[dict] = None,
    *,
    mean_edge_psnr: float | None = None,
    rasterize_mode: str = "classic",
    evaluation_resolutions: list[list[int]] | None = None,
    metric_context: Mapping[str, Any] | None = None,
    temporal_flicker_rms: float | None = None,
    temporal_flicker_psnr_db: float | None = None,
    temporal_flicker_valid_pairs: int = 0,
    temporal_flicker_status: str = "not_computed",
) -> None:
    path = os.path.join(model_path, "psnr_history.jsonl")
    prior_reduction = psnr_history_tail_reduction(model_path)
    if prior_reduction not in (None, PSNR_REDUCTION_RGB_PER_VIEW_V2):
        raise ValueError(
            f"cannot append {PSNR_REDUCTION_RGB_PER_VIEW_V2} to "
            f"{prior_reduction}; use a fresh model/output path"
        )
    row = {
        "iteration": iteration,
        "mean_test_psnr": mean_psnr,
        "psnr_reduction": PSNR_REDUCTION_RGB_PER_VIEW_V2,
    }
    if score_domain_clamp is not None:
        row["score_domain_clamp"] = score_domain_clamp
    row["rasterize_mode"] = str(rasterize_mode)
    row["edge_score_scope"] = "sobel_gradient_same_evaluation_resolution"
    row["mean_edge_psnr"] = mean_edge_psnr
    row["temporal_flicker_rms"] = temporal_flicker_rms
    row["temporal_flicker_psnr_db"] = temporal_flicker_psnr_db
    row["temporal_flicker_valid_pairs"] = int(temporal_flicker_valid_pairs)
    row["temporal_flicker_status"] = str(temporal_flicker_status)
    if evaluation_resolutions is not None:
        row["evaluation_resolutions"] = evaluation_resolutions
    if metric_context is not None:
        allowed = {
            "training_resolution_tiers",
            "optimizer_updates",
            "supervision_views",
            "mapping_active_s",
            "gaussian_count",
            "rasterizer_gpu_ms",
            "rasterizer_timing_status",
            "evaluation_phase",
            "evaluation_observer_execution",
            "mapping_execution_paused",
            "throughput_evidence_valid",
            "throughput_evidence_status",
            "evaluation_cadence_clock",
        }
        unknown = set(metric_context).difference(allowed)
        if unknown:
            raise ValueError(
                "unsupported PSNR metric context keys: " + ", ".join(sorted(unknown))
            )
        row.update(metric_context)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


from eligsir.utils.psnr_history_io import read_psnr_history, _read_psnr_history  # noqa: F401,E402


def check_psnr_plateau(model_path: str, patience_ticks: int, min_delta_db: float) -> bool:
    """True once the recent smoothed PSNR trend is within ``min_delta_db``."""
    if patience_ticks <= 0:
        return False
    _iters, vals = _read_psnr_history(model_path)
    if len(vals) < patience_ticks:
        return False
    window = vals[-patience_ticks:]
    half = len(window) // 2
    if half < 1:
        return False
    first_half_mean = sum(window[:half]) / half
    second_half_mean = sum(window[-half:]) / half
    return abs(second_half_mean - first_half_mean) <= min_delta_db


def plot_psnr_history(model_path: str, out_name: str = "psnr_over_time.svg") -> Optional[str]:
    iters, vals = _read_psnr_history(model_path)
    if not iters:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        history_path = os.path.join(model_path, "psnr_history.jsonl")
        with open(history_path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        valid_rows = [row for row in rows if row.get("mean_test_psnr") is not None]
        use_mapping_time = bool(valid_rows) and all(
            row.get("mapping_active_s") is not None for row in valid_rows
        )
        x_values = (
            [float(row["mapping_active_s"]) for row in valid_rows]
            if use_mapping_time
            else [int(row["iteration"]) for row in valid_rows]
        )
        y_values = [float(row["mean_test_psnr"]) for row in valid_rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_values, y_values, marker="o", color="C0", label="holdout RGB PSNR")
        edge_x = []
        edge_y = []
        for x_value, row in zip(x_values, valid_rows, strict=True):
            edge_value = row.get("mean_edge_psnr")
            if edge_value is not None:
                edge_x.append(x_value)
                edge_y.append(float(edge_value))
        if edge_x:
            ax.plot(edge_x, edge_y, marker="x", color="C1", label="same-res edge PSNR")
        previous_tiers = None
        previous_mode = None
        for x_value, row in zip(x_values, valid_rows, strict=True):
            tiers = tuple(row.get("training_resolution_tiers") or ())
            mode = str(row.get("rasterize_mode", "classic"))
            if tiers and tiers != previous_tiers:
                ax.axvline(x_value, color="0.45", linestyle=":", alpha=0.55)
                ax.annotate(
                    "/".join(tiers),
                    (x_value, 1.0),
                    xycoords=("data", "axes fraction"),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=7,
                )
            if tiers:
                previous_tiers = tiers
            if mode != previous_mode:
                ax.axvline(x_value, color="C2", linestyle="--", alpha=0.4)
                ax.annotate(
                    mode,
                    (x_value, 0.0),
                    xycoords=("data", "axes fraction"),
                    rotation=90,
                    va="bottom",
                    ha="left",
                    fontsize=7,
                    color="C2",
                )
                previous_mode = mode
        ax.set_xlabel("mapping-active time (s)" if use_mapping_time else "iteration")
        ax.set_ylabel("PSNR (dB)")
        modes = sorted({str(row.get("rasterize_mode", "classic")) for row in valid_rows})
        mode_label = "/".join(modes)
        ax.set_title(
            f"Test-holdout PSNR over time ({len(valid_rows)} samples, {mode_label})"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        out_path = str(svg_output_path(os.path.join(model_path, out_name)))
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path
    except Exception as e:
        print(f"[psnr_history] plot failed: {e}", flush=True)
        return None


def record_and_plot(model_path: str, iteration: int, gaussians, test_cams: List,
                     render_fn, pipe, background, per_view_hook=None,
                     metric_context: Mapping[str, Any] | None = None,
                     metric_context_resolver=None) -> Optional[PsnrTickResult]:
    """Compute, append and plot one canonical held-out PSNR tick."""
    try:
        clamp_records: List[dict] = []
        from eligsir.mapping.renderer.gsplat_backend import validate_rasterize_mode
        from eligsir.utils.live_antialiasing_metrics import (
            camera_pair_temporal_flicker,
            edge_psnr,
            summarize_temporal_flicker,
        )

        edge_values: List[torch.Tensor] = []
        edge_errors: List[str] = []
        temporal_values = []
        temporal_errors: List[str] = []
        previous = None

        def _metric_hook(cam, render_score, gt_score):
            nonlocal previous
            try:
                edge_values.append(edge_psnr(render_score, gt_score))
            except Exception as exc:
                edge_errors.append(repr(exc))
            if previous is not None:
                try:
                    temporal_values.append(
                        camera_pair_temporal_flicker(
                            previous[0], previous[1], previous[2],
                            cam, render_score, gt_score,
                        )
                    )
                except Exception as exc:
                    temporal_errors.append(repr(exc))
            previous = (cam, render_score, gt_score)
            if per_view_hook is not None:
                per_view_hook(cam, render_score, gt_score)

        cams, psnr_tensor = compute_per_view_test_psnr_gpu(
            gaussians,
            test_cams,
            render_fn,
            pipe,
            background,
            per_view_hook=_metric_hook,
            clamp_records_out=clamp_records,
        )
        if psnr_tensor is None:
            return None
        mean_psnr = float(psnr_tensor.mean().item())
        edge_psnr_tensor = (
            torch.stack(edge_values) if len(edge_values) == len(cams) else None
        )
        mean_edge_psnr = (
            None
            if edge_psnr_tensor is None
            else float(edge_psnr_tensor.mean().item())
        )
        if mean_edge_psnr is not None and not math.isfinite(mean_edge_psnr):
            mean_edge_psnr_for_json = None
        else:
            mean_edge_psnr_for_json = mean_edge_psnr
        temporal_valid_pairs = len(temporal_values)
        if temporal_values:
            try:
                (
                    temporal_flicker_rms,
                    temporal_flicker_psnr_db,
                    temporal_valid_pairs,
                ) = summarize_temporal_flicker(temporal_values)
                temporal_flicker_status = (
                    "valid_motion_compensated_perfect_match"
                    if temporal_flicker_rms == 0.0
                    else "valid_motion_compensated"
                )
            except Exception as exc:
                temporal_errors.append(repr(exc))
                temporal_flicker_rms = None
                temporal_flicker_psnr_db = None
                temporal_valid_pairs = 0
                temporal_flicker_status = "unavailable_invalid_aggregate"
        else:
            temporal_flicker_rms = None
            temporal_flicker_psnr_db = None
            temporal_flicker_status = (
                "unavailable_pose_depth_overlap" if temporal_errors else "insufficient_camera_pairs"
            )
        score_domain_clamp = _aggregate_clamp_records(clamp_records)
        rasterize_mode = validate_rasterize_mode(
            getattr(pipe, "rasterize_mode", "classic")
        )
        evaluation_resolutions = [
            list(resolution)
            for resolution in sorted(
                {
                    (
                        int(getattr(cam, "image_width", 0)),
                        int(getattr(cam, "image_height", 0)),
                    )
                    for cam in cams
                }
            )
        ]
        if metric_context_resolver is not None:
            metric_context = metric_context_resolver(metric_context)
        append_psnr_history(
            model_path,
            iteration,
            mean_psnr,
            score_domain_clamp=score_domain_clamp,
            mean_edge_psnr=mean_edge_psnr_for_json,
            rasterize_mode=rasterize_mode,
            evaluation_resolutions=evaluation_resolutions,
            metric_context=metric_context,
            temporal_flicker_rms=temporal_flicker_rms,
            temporal_flicker_psnr_db=temporal_flicker_psnr_db,
            temporal_flicker_valid_pairs=temporal_valid_pairs,
            temporal_flicker_status=temporal_flicker_status,
        )
        plot_psnr_history(model_path)
        print(f"[psnr_history] iter={iteration} mean_test_psnr={mean_psnr:.2f}dB "
              f"(n_test={len(cams)})", flush=True)
        return PsnrTickResult(
            cams=cams,
            psnr_tensor=psnr_tensor,
            mean_psnr=mean_psnr,
            score_domain_clamp=score_domain_clamp,
            edge_psnr_tensor=edge_psnr_tensor,
            mean_edge_psnr=mean_edge_psnr_for_json,
            temporal_flicker_rms=temporal_flicker_rms,
            temporal_flicker_psnr_db=temporal_flicker_psnr_db,
            temporal_flicker_valid_pairs=temporal_valid_pairs,
            temporal_flicker_status=temporal_flicker_status,
            rasterize_mode=rasterize_mode,
            gaussian_count=(
                None
                if metric_context is None
                or metric_context.get("gaussian_count") is None
                else int(metric_context["gaussian_count"])
            ),
            training_resolution_tiers=tuple(
                ()
                if metric_context is None
                else metric_context.get("training_resolution_tiers") or ()
            ),
            evaluation_resolutions=tuple(
                (int(width), int(height))
                for width, height in evaluation_resolutions
            ),
        )
    except Exception as e:
        path = os.path.join(model_path, "psnr_history.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps({
                "iteration": iteration,
                "mean_test_psnr": None,
                "eval_valid": False,
                "psnr_reduction": PSNR_REDUCTION_RGB_PER_VIEW_V2,
                "n_attempted": len(test_cams),
                "error": repr(e),
            }) + "\n")
        print(f"[psnr_history] record_and_plot failed at iter={iteration}: {e}", flush=True)
        return None
