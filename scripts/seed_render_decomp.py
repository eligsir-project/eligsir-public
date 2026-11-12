#!/usr/bin/env python3
"""Exact fixed-view render decomposition for the seed-fade investigation.

The four passes are derived only from checkpoint-persisted Gaussian identity:

``full``
    Every checkpoint row.
``seed_root``
    Surviving original bootstrap roots from ``seed_cohort_mask``.
``seed_lineage``
    Bootstrap roots and every descendant from ``seed_lineage_mask``.
``nonseed``
    The exact complement of ``seed_lineage``.

This is a diagnostic artifact generator, not canonical ResearchFlow evidence.
It deliberately refuses PLY input, approximate geometric masks, missing lineage,
legacy layouts, mismatched checkpoint names, and an existing output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arguments import get_default_pipeline_params
from scene.gsplat_model import GsplatGaussianModel
from eligsir.utils.gaussian_lineage import (
    CheckpointLineageView,
    seed_cohort_mask,
    seed_lineage_mask,
)
from eligsir.utils.image_utils import psnr
from eligsir.utils.score_contract import (
    declare_camera_score_provenance,
    declare_canonical_score_tensor,
    score_tensor_from_camera,
)


RENDER_PARAM_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")
PASS_NAMES = ("full", "seed_root", "seed_lineage", "nonseed")
_CHECKPOINT_RE = re.compile(r"chkpnt(?P<iteration>\d+)\.pth")


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    state: dict
    saved_iteration: int
    streaming_meta: dict
    tuple_arity: int
    lineage_view: CheckpointLineageView
    checkpoint_sha256: str


@dataclass(frozen=True)
class FixedSeedView:
    camera: Any
    gt: torch.Tensor
    entry: dict
    camera_index: int
    camera_json_path: Path
    camera_file_sha256: str
    camera_entry_sha256: str
    gt_sha256: str
    cfg_args: argparse.Namespace


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _infer_max_sh_degree(params: Mapping[str, torch.Tensor]) -> int:
    n_coeffs = 1 + int(params["shN"].shape[1])
    root = math.isqrt(n_coeffs)
    if root * root != n_coeffs:
        raise ValueError(
            f"params['shN'] implies {n_coeffs} SH coefficients, not a square count"
        )
    return root - 1


def _validate_checkpoint_state(state: Any) -> None:
    if not isinstance(state, dict):
        raise ValueError("checkpoint state must be a dict")
    if state.get("layout") != "gsplat":
        raise ValueError(
            f"checkpoint layout must be 'gsplat', got {state.get('layout')!r}"
        )
    params = state.get("params")
    if not isinstance(params, Mapping):
        raise ValueError("checkpoint state has no parameter mapping at key 'params'")
    missing = [name for name in RENDER_PARAM_KEYS if name not in params]
    if missing:
        raise ValueError(f"checkpoint is missing required render parameter(s): {missing}")
    for name in RENDER_PARAM_KEYS:
        if not torch.is_tensor(params[name]):
            raise ValueError(f"checkpoint render parameter {name!r} is not a tensor")

    means = params["means"]
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"params['means'] must have shape [N,3], got {tuple(means.shape)}")
    n = int(means.shape[0])
    for name in RENDER_PARAM_KEYS[1:]:
        if params[name].ndim == 0 or int(params[name].shape[0]) != n:
            got = 0 if params[name].ndim == 0 else int(params[name].shape[0])
            raise ValueError(
                f"checkpoint parameter {name!r} has {got} rows but means has {n} rows"
            )
    expected_shapes = {
        "scales": (n, 3),
        "quats": (n, 4),
        "opacities": (n,),
        "sh0": (n, 1, 3),
    }
    for name, expected in expected_shapes.items():
        if tuple(params[name].shape) != expected:
            raise ValueError(
                f"params[{name!r}] must have shape {expected}, got {tuple(params[name].shape)}"
            )
    if params["shN"].ndim != 3 or int(params["shN"].shape[2]) != 3:
        raise ValueError(
            f"params['shN'] must have shape [N,K,3], got {tuple(params['shN'].shape)}"
        )
    max_sh_degree = _infer_max_sh_degree(params)
    active = state.get("active_sh_degree")
    if not isinstance(active, (int, np.integer)) or not 0 <= int(active) <= max_sh_degree:
        raise ValueError(
            f"active_sh_degree must be an integer in [0,{max_sh_degree}], got {active!r}"
        )
    streaming_state = state.get("streaming_state")
    if streaming_state is not None and not isinstance(streaming_state, Mapping):
        raise ValueError("checkpoint streaming_state must be a mapping when present")


def load_checkpoint_state(path: os.PathLike[str] | str) -> LoadedCheckpoint:
    """Load and validate a 2/3-tuple checkpoint on CPU, never a PLY."""
    checkpoint_path = Path(path)
    match = _CHECKPOINT_RE.fullmatch(checkpoint_path.name)
    if match is None:
        raise ValueError(
            f"checkpoint filename must match chkpnt<iteration>.pth, got {checkpoint_path.name!r}"
        )
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(payload, tuple) or len(payload) not in (2, 3):
        raise ValueError("checkpoint payload must be an exact 2- or 3-tuple")
    state, saved_iteration = payload[:2]
    if isinstance(saved_iteration, bool) or not isinstance(
        saved_iteration, numbers.Integral
    ):
        raise ValueError("checkpoint saved iteration must be an integer")
    saved_iteration = int(saved_iteration)
    filename_iteration = int(match.group("iteration"))
    if filename_iteration != saved_iteration:
        raise ValueError(
            f"checkpoint filename iteration {filename_iteration} does not match "
            f"saved iteration {saved_iteration}"
        )
    streaming_meta = payload[2] if len(payload) == 3 else {}
    if streaming_meta is None:
        streaming_meta = {}
    if not isinstance(streaming_meta, dict):
        raise ValueError("checkpoint streaming metadata must be a dict")
    _validate_checkpoint_state(state)
    return LoadedCheckpoint(
        path=checkpoint_path,
        state=state,
        saved_iteration=saved_iteration,
        streaming_meta=streaming_meta,
        tuple_arity=len(payload),
        lineage_view=CheckpointLineageView(state),
        checkpoint_sha256=file_sha256(checkpoint_path),
    )


def exact_partition_masks(view: CheckpointLineageView) -> dict[str, torch.Tensor]:
    """Return exact root, lineage, and lineage-complement masks fail-closed."""
    seed_root = seed_cohort_mask(view).to(dtype=torch.bool)
    seed_lineage = seed_lineage_mask(view).to(dtype=torch.bool)
    n = int(view.get_xyz.shape[0])
    if seed_root.ndim != 1 or seed_lineage.ndim != 1:
        raise RuntimeError("canonical lineage masks must be one-dimensional")
    if int(seed_root.shape[0]) != n or int(seed_lineage.shape[0]) != n:
        raise RuntimeError(
            f"canonical lineage mask length mismatch: N={n}, "
            f"seed_root={seed_root.shape[0]}, seed_lineage={seed_lineage.shape[0]}"
        )
    if bool((seed_root & ~seed_lineage).any()):
        raise RuntimeError("seed_root must be an exact subset of seed_lineage")
    nonseed = ~seed_lineage
    if bool((seed_lineage & nonseed).any()):
        raise RuntimeError("seed_lineage and nonseed masks are not disjoint")
    union_count = int((seed_lineage | nonseed).sum().item())
    if union_count != n:
        raise RuntimeError(
            f"seed_lineage/nonseed union covers {union_count} rows, expected {n}"
        )
    return {
        "seed_root": seed_root,
        "seed_lineage": seed_lineage,
        "nonseed": nonseed,
    }


def build_render_view(
    state: dict,
    mask: torch.Tensor | None = None,
    *,
    device: str | torch.device = "cuda",
) -> GsplatGaussianModel:
    """Build an optimizer-free, non-aliasing gsplat view for one render pass."""
    _validate_checkpoint_state(state)
    params = state["params"]
    n = int(params["means"].shape[0])
    if mask is not None:
        if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("render mask must be a one-dimensional bool tensor")
        if int(mask.shape[0]) != n:
            raise ValueError(f"render mask has {mask.shape[0]} rows, expected {n}")
        mask = mask.detach().to(device="cpu")

    # GsplatGaussianModel.__init__ allocates CUDA-resident training buffers.
    # Rendering only reads params plus these two degree fields, so bypassing the
    # training constructor is both smaller and avoids optimizer/running-state
    # mutation. The instance remains the canonical model type and uses its real
    # property accessors (get_xyz/get_scaling/get_rotation/get_opacity/features).
    render_view = object.__new__(GsplatGaussianModel)
    render_view.active_sh_degree = int(state["active_sh_degree"])
    render_view.max_sh_degree = _infer_max_sh_degree(params)
    render_view.params = {}
    for name in RENDER_PARAM_KEYS:
        source = params[name]
        selected = source if mask is None else source[mask]
        render_view.params[name] = (
            selected.detach().clone().to(device=device, non_blocking=True).contiguous()
        )
    return render_view


def load_fixed_seed_view(
    model_path: os.PathLike[str] | str,
    scene_path: os.PathLike[str] | str,
    *,
    device: str | torch.device = "cuda",
) -> FixedSeedView:
    """Load saved ``cameras.json[0]`` and its exact canonical source-frame GT."""
    from render_report import (
        _camera_json_to_minicam,
        _load_cfg_args,
        attach_source_gt,
    )

    model_path = Path(model_path)
    scene_path = Path(scene_path)
    camera_json_path = model_path / "cameras.json"
    if not camera_json_path.is_file():
        raise FileNotFoundError(f"saved camera file not found: {camera_json_path}")
    with camera_json_path.open() as handle:
        entries = json.load(handle)
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        raise ValueError("cameras.json must be a non-empty list of camera entries")
    entry = entries[0]
    camera = _camera_json_to_minicam(entry, device=str(device))
    cfg_args = _load_cfg_args(str(model_path))
    stats = attach_source_gt([camera], cfg_args, source_path=str(scene_path))
    if stats.get("n_matched") != 1 or not hasattr(camera, "original_image"):
        raise RuntimeError(
            "saved cameras.json[0] could not be resolved to its canonical source "
            f"frame under {scene_path}: {stats.get('reason') or stats}"
        )
    expected_shape = (3, int(entry["height"]), int(entry["width"]))
    if tuple(camera.original_image.shape) != expected_shape:
        raise RuntimeError(
            f"saved camera/GT shape mismatch: camera entry expects {expected_shape}, "
            f"resolved GT is {tuple(camera.original_image.shape)}"
        )
    declare_camera_score_provenance(
        camera, context="seed_render_decomp:cameras.json[0] source GT"
    )
    gt = score_tensor_from_camera(camera)
    return FixedSeedView(
        camera=camera,
        gt=gt,
        entry=entry,
        camera_index=0,
        camera_json_path=camera_json_path,
        camera_file_sha256=file_sha256(camera_json_path),
        camera_entry_sha256=_json_sha256(entry),
        gt_sha256=_tensor_sha256(gt),
        cfg_args=cfg_args,
    )


def _alpha_chw(alpha: torch.Tensor) -> torch.Tensor:
    if alpha.ndim == 3 and alpha.shape[-1] == 1:
        return alpha.permute(2, 0, 1).contiguous()
    if alpha.ndim == 3 and alpha.shape[0] == 1:
        return alpha.contiguous()
    raise ValueError(f"alpha must have shape [H,W,1] or [1,H,W], got {tuple(alpha.shape)}")


def _depth_chw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 3 and depth.shape[0] == 1:
        return depth.contiguous()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return depth.permute(2, 0, 1).contiguous()
    raise ValueError(
        f"expected depth must have shape [1,H,W] or [H,W,1], got {tuple(depth.shape)}"
    )


def raw_rgb_diagnostics(rgb: torch.Tensor) -> dict[str, Any]:
    """Describe the renderer's raw tensor before any score-domain transform."""
    if not torch.is_tensor(rgb):
        raise TypeError("renderer RGB must be a torch tensor")
    finite = torch.isfinite(rgb)
    finite_count = int(finite.sum().item())
    finite_values = rgb[finite]
    return {
        "dtype": str(rgb.dtype),
        "shape": [int(value) for value in rgb.shape],
        "element_count": int(rgb.numel()),
        "finite_count": finite_count,
        "nonfinite_count": int(rgb.numel()) - finite_count,
        "below_zero_count": int(((rgb < 0) & finite).sum().item()),
        "above_one_count": int(((rgb > 1) & finite).sum().item()),
        "finite_min": (
            float(finite_values.min().item()) if finite_count else None
        ),
        "finite_max": (
            float(finite_values.max().item()) if finite_count else None
        ),
    }


def _compute_rgb_metrics(rgb: torch.Tensor, gt: torch.Tensor) -> dict[str, Any]:
    if tuple(rgb.shape) != tuple(gt.shape):
        raise ValueError(f"render/GT shape mismatch: {tuple(rgb.shape)} vs {tuple(gt.shape)}")
    diagnostics = raw_rgb_diagnostics(rgb)
    score_domain = {
        "status": "available",
        "input": "raw_renderer_rgb",
        "transform": "clamp_to_[0,1]",
        "provenance_boundary": "declare_canonical_score_tensor",
        "canonical_domain": "float_RGB_CHW_[0,1]",
    }
    if diagnostics["nonfinite_count"]:
        score_domain["status"] = "unavailable_nonfinite_raw_rgb"
        return {
            "raw_rgb_range": diagnostics,
            "score_domain": score_domain,
            "psnr_clamped_db": None,
            "mean_abs_error_clamped": None,
        }
    clamped, _ = declare_canonical_score_tensor(
        rgb.clamp(0.0, 1.0),
        context="seed_render_decomp:explicit clamp-to-[0,1] render boundary",
    )
    return {
        "raw_rgb_range": diagnostics,
        "score_domain": score_domain,
        "psnr_clamped_db": float(psnr(clamped, gt).mean().item()),
        "mean_abs_error_clamped": float((clamped - gt).abs().mean().item()),
    }


def compute_pass_metrics(
    rgb: torch.Tensor,
    gt: torch.Tensor,
    alpha: torch.Tensor,
    expected_depth: torch.Tensor,
    *,
    alpha_threshold: float,
) -> dict[str, Any]:
    """Score one isolated pass; ED summaries use only alpha-covered pixels."""
    if not 0.0 <= float(alpha_threshold) < 1.0:
        raise ValueError("alpha_threshold must be in [0,1)")
    alpha_chw = _alpha_chw(alpha)
    depth_chw = _depth_chw(expected_depth)
    if tuple(alpha_chw.shape[-2:]) != tuple(rgb.shape[-2:]):
        raise ValueError("alpha spatial shape does not match RGB")
    if tuple(depth_chw.shape) != tuple(alpha_chw.shape):
        raise ValueError("expected-depth shape does not match alpha")
    covered = alpha_chw > float(alpha_threshold)
    depth_valid = covered & torch.isfinite(depth_chw) & (depth_chw > 0)
    depth_mean = (
        float(depth_chw[depth_valid].mean().item()) if bool(depth_valid.any()) else None
    )
    return {
        **_compute_rgb_metrics(rgb, gt),
        "mean_alpha": float(alpha_chw.mean().item()),
        "alpha_coverage_fraction": float(covered.float().mean().item()),
        "expected_depth_mean_m": depth_mean,
        "expected_depth_valid_fraction": float(depth_valid.float().mean().item()),
    }


def compute_overlap_metrics(
    seed_alpha: torch.Tensor,
    seed_expected_depth: torch.Tensor,
    nonseed_alpha: torch.Tensor,
    nonseed_expected_depth: torch.Tensor,
    *,
    alpha_threshold: float,
) -> dict[str, float | str | None]:
    """Approximate overlap/order from isolated passes, never pixel attribution."""
    seed_alpha = _alpha_chw(seed_alpha)
    nonseed_alpha = _alpha_chw(nonseed_alpha)
    seed_depth = _depth_chw(seed_expected_depth)
    nonseed_depth = _depth_chw(nonseed_expected_depth)
    shapes = {
        tuple(seed_alpha.shape),
        tuple(nonseed_alpha.shape),
        tuple(seed_depth.shape),
        tuple(nonseed_depth.shape),
    }
    if len(shapes) != 1:
        raise ValueError(f"overlap inputs must have one shared shape, got {sorted(shapes)}")
    overlap = (seed_alpha > float(alpha_threshold)) & (
        nonseed_alpha > float(alpha_threshold)
    )
    comparable = (
        overlap
        & torch.isfinite(seed_depth)
        & torch.isfinite(nonseed_depth)
        & (seed_depth > 0)
        & (nonseed_depth > 0)
    )
    overlap_count = int(overlap.sum().item())
    comparable_count = int(comparable.sum().item())
    if comparable_count:
        seed_front = float((seed_depth[comparable] < nonseed_depth[comparable]).float().mean().item())
        nonseed_front = float((nonseed_depth[comparable] < seed_depth[comparable]).float().mean().item())
        tied = float((seed_depth[comparable] == nonseed_depth[comparable]).float().mean().item())
    else:
        seed_front = nonseed_front = tied = None
    return {
        "semantics": "approximate_isolated_pass_overlap",
        "alpha_threshold": float(alpha_threshold),
        "image_pixel_count": int(overlap.numel()),
        "alpha_overlap_pixel_count": overlap_count,
        "alpha_overlap_fraction": float(overlap.float().mean().item()),
        "depth_comparable_pixel_count": comparable_count,
        "depth_comparable_fraction_of_image": float(comparable.float().mean().item()),
        "depth_comparable_fraction_within_alpha_overlap": (
            float(comparable_count / overlap_count) if overlap_count else None
        ),
        "seed_in_front_fraction_within_comparable_overlap": seed_front,
        "nonseed_in_front_fraction_within_comparable_overlap": nonseed_front,
        "depth_tie_fraction_within_comparable_overlap": tied,
    }


def create_output_dir(path: os.PathLike[str] | str) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"output already exists; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .numpy()
    )
    Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB").save(path)


def _save_scalar_png(path: Path, tensor: torch.Tensor, vmax: float) -> None:
    array = tensor.detach().to(device="cpu", dtype=torch.float32).squeeze().numpy()
    valid = np.isfinite(array) & (array >= 0)
    scaled = np.zeros(array.shape, dtype=np.uint16)
    if vmax > 0:
        scaled[valid] = np.rint(np.clip(array[valid] / vmax, 0.0, 1.0) * 65535.0).astype(
            np.uint16
        )
    Image.fromarray(scaled).save(path)


def _save_raw_npy(path: Path, tensor: torch.Tensor) -> None:
    """Persist the tensor without clamping, quantization, or dtype conversion."""
    array = tensor.detach().to(device="cpu").contiguous().numpy()
    np.save(path, array, allow_pickle=False)


def write_checkpoint_artifacts(
    out_dir: os.PathLike[str] | str,
    iteration: int,
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    gt: torch.Tensor,
    *,
    production_full_rgb: torch.Tensor,
) -> dict[str, dict[str, str]]:
    """Write exact arrays plus visual PNGs for one checkpoint."""
    if tuple(outputs) not in (PASS_NAMES, ("full",)):
        raise ValueError(
            f"outputs must contain {PASS_NAMES} or the parity-gated full pass, "
            f"got {tuple(outputs)}"
        )
    root = Path(out_dir)
    checkpoint_dir = root / f"iteration_{int(iteration):06d}"
    checkpoint_dir.mkdir(parents=False, exist_ok=False)
    _save_rgb(checkpoint_dir / "gt_rgb_clamped.png", gt)

    production_dir = checkpoint_dir / "production_full_no_depth"
    production_dir.mkdir()
    production_raw_path = production_dir / "raw_rgb.npy"
    production_png_path = production_dir / "rgb_clamped.png"
    _save_raw_npy(production_raw_path, production_full_rgb)
    _save_rgb(production_png_path, production_full_rgb)

    positive_depths = []
    for output in outputs.values():
        depth = _depth_chw(output["expected_depth"]).detach().cpu().numpy()
        valid = depth[np.isfinite(depth) & (depth > 0)]
        if valid.size:
            positive_depths.append(valid)
    depth_vmax = (
        float(np.quantile(np.concatenate(positive_depths), 0.99))
        if positive_depths
        else 1.0
    )

    artifact_paths: dict[str, dict[str, str]] = {
        "production_full_no_depth": {
            "raw_rgb_npy": production_raw_path.relative_to(root).as_posix(),
            "rgb_clamped_png": production_png_path.relative_to(root).as_posix(),
        }
    }
    for name in outputs:
        output = outputs[name]
        pass_dir = checkpoint_dir / name
        pass_dir.mkdir()
        rgb = output["rgb"]
        alpha = _alpha_chw(output["alpha"])
        depth = _depth_chw(output["expected_depth"])
        error = (rgb.clamp(0.0, 1.0) - gt).abs()
        rgb_raw_path = pass_dir / "raw_rgb.npy"
        rgb_path = pass_dir / "rgb_clamped.png"
        alpha_raw_path = pass_dir / "raw_alpha.npy"
        alpha_path = pass_dir / "alpha.png"
        depth_npy_path = pass_dir / "expected_depth.npy"
        depth_png_path = pass_dir / "expected_depth.png"
        error_path = pass_dir / "absolute_clamped_rgb_error.png"
        _save_raw_npy(rgb_raw_path, rgb)
        _save_rgb(rgb_path, rgb)
        _save_raw_npy(alpha_raw_path, alpha)
        _save_scalar_png(alpha_path, alpha, vmax=1.0)
        _save_raw_npy(depth_npy_path, depth)
        _save_scalar_png(depth_png_path, depth, vmax=depth_vmax)
        _save_rgb(error_path, error)
        artifact_paths[name] = {
            "raw_rgb_npy": rgb_raw_path.relative_to(root).as_posix(),
            "rgb_clamped_png": rgb_path.relative_to(root).as_posix(),
            "raw_alpha_npy": alpha_raw_path.relative_to(root).as_posix(),
            "alpha_png": alpha_path.relative_to(root).as_posix(),
            "expected_depth_npy": depth_npy_path.relative_to(root).as_posix(),
            "expected_depth_png": depth_png_path.relative_to(root).as_posix(),
            "absolute_clamped_rgb_error_png": error_path.relative_to(root).as_posix(),
        }
    return artifact_paths


def _flatten_value(row: dict[str, Any], prefix: str, value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_value(row, f"{prefix}_{key}", child)
    else:
        row[prefix] = value


def _flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "iteration": record.get("iteration"),
        "checkpoint_sha256": record.get("checkpoint_sha256"),
    }
    for name, count in (record.get("counts") or {}).items():
        row[f"{name}_count"] = count
    for name, value in (record.get("mean_gaussian_opacity") or {}).items():
        row[f"{name}_mean_gaussian_opacity"] = value
    for key, value in (record.get("production_full_no_depth") or {}).items():
        _flatten_value(row, f"production_full_no_depth_{key}", value)
    for name, metrics in (record.get("passes") or {}).items():
        for key, value in metrics.items():
            _flatten_value(row, f"{name}_{key}", value)
    for key, value in (record.get("overlap") or {}).items():
        _flatten_value(row, f"overlap_{key}", value)
    for key, value in (record.get("full_rgb_mode_parity") or {}).items():
        _flatten_value(row, f"full_rgb_mode_parity_{key}", value)
    return row


def write_series(
    out_dir: os.PathLike[str] | str, records: list[dict[str, Any]]
) -> None:
    root = Path(out_dir)
    jsonl_path = root / "series.jsonl"
    jsonl_temporary = jsonl_path.with_suffix(".jsonl.tmp")
    with jsonl_temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(jsonl_temporary, jsonl_path)
    rows = [_flatten_record(record) for record in records]
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    csv_path = root / "series.csv"
    csv_temporary = csv_path.with_suffix(".csv.tmp")
    with csv_temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(csv_temporary, csv_path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def write_manifest(out_dir: os.PathLike[str] | str, manifest: dict[str, Any]) -> None:
    path = Path(out_dir) / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _partition_statistics(
    state: dict, masks: Mapping[str, torch.Tensor]
) -> tuple[dict[str, int], dict[str, float | None]]:
    opacity = torch.sigmoid(state["params"]["opacities"].detach().cpu())
    full = torch.ones(opacity.shape[0], dtype=torch.bool)
    pass_masks = {"full": full, **masks}
    counts = {name: int(mask.sum().item()) for name, mask in pass_masks.items()}
    means = {
        name: (float(opacity[mask].mean().item()) if bool(mask.any()) else None)
        for name, mask in pass_masks.items()
    }
    return counts, means


def _empty_cuda_cache(device: str | torch.device) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def render_checkpoint(
    state: dict,
    masks: Mapping[str, torch.Tensor],
    camera: Any,
    gt: torch.Tensor,
    pipe: Any,
    background: torch.Tensor,
    *,
    alpha_threshold: float,
    full_rgb_tolerance: float,
    device: str | torch.device = "cuda",
    render_fn: Callable[..., Mapping[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Render production full RGB, then four sequential RGB+ED passes."""
    if full_rgb_tolerance < 0:
        raise ValueError("full RGB mode tolerance must be non-negative")
    if render_fn is None:
        from gaussian_renderer import render as render_fn

    gt_device = score_tensor_from_camera(camera, device=device)
    production_view = build_render_view(state, mask=None, device=device)
    with torch.no_grad():
        production_package = render_fn(
            camera,
            production_view,
            pipe,
            background,
            render_depth=False,
        )
    if "render" not in production_package:
        raise RuntimeError("production full pass did not return required key 'render'")
    production_rgb = production_package["render"]
    production_metrics = _compute_rgb_metrics(production_rgb, gt_device)
    production_rgb_cpu = production_rgb.detach().to(device="cpu", copy=True)
    del production_package, production_rgb, production_view
    _empty_cuda_cache(device)

    outputs: dict[str, dict[str, torch.Tensor]] = {}
    pass_metrics: dict[str, dict[str, Any]] = {}
    pass_masks = {
        "full": None,
        "seed_root": masks["seed_root"],
        "seed_lineage": masks["seed_lineage"],
        "nonseed": masks["nonseed"],
    }
    parity: dict[str, Any] | None = None
    for name in PASS_NAMES:
        render_view = build_render_view(state, mask=pass_masks[name], device=device)
        with torch.no_grad():
            package = render_fn(
                camera,
                render_view,
                pipe,
                background,
                render_depth=True,
            )
        for key in ("render", "alpha", "rendered_depth_ed"):
            if key not in package:
                raise RuntimeError(f"renderer pass {name!r} did not return required key {key!r}")
        rgb = package["render"]
        alpha = _alpha_chw(package["alpha"])
        depth = _depth_chw(package["rendered_depth_ed"])
        pass_metrics[name] = compute_pass_metrics(
            rgb,
            gt_device,
            alpha,
            depth,
            alpha_threshold=alpha_threshold,
        )
        outputs[name] = {
            "rgb": rgb.detach().cpu(),
            "alpha": alpha.detach().cpu(),
            "expected_depth": depth.detach().cpu(),
        }
        if name == "full":
            full_delta = (
                production_rgb_cpu - outputs["full"]["rgb"]
            ).abs()
            if bool(torch.isfinite(full_delta).all()):
                max_delta = float(full_delta.max().item())
                parity_status = (
                    "passed" if max_delta <= full_rgb_tolerance else "failed"
                )
            else:
                max_delta = None
                parity_status = "failed_nonfinite_delta"
            parity = {
                "status": parity_status,
                "raw_rgb_max_abs_delta": max_delta,
                "tolerance": float(full_rgb_tolerance),
                "production_call_render_depth": False,
                "decomposition_call_render_depth": True,
            }
        del package, rgb, alpha, depth, render_view
        _empty_cuda_cache(device)
        if parity is not None and parity["status"] != "passed":
            break
    assert parity is not None
    overlap = (
        compute_overlap_metrics(
            outputs["seed_lineage"]["alpha"],
            outputs["seed_lineage"]["expected_depth"],
            outputs["nonseed"]["alpha"],
            outputs["nonseed"]["expected_depth"],
            alpha_threshold=alpha_threshold,
        )
        if tuple(outputs) == PASS_NAMES
        else None
    )
    result = {
        "production_full_no_depth": {
            "rgb": production_rgb_cpu,
            "metrics": production_metrics,
        },
        "outputs": outputs,
        "passes": pass_metrics,
        "overlap": overlap,
        "full_rgb_mode_parity": parity,
    }
    return result


def _pipeline_from_cfg(cfg_args: argparse.Namespace) -> Any:
    pipe = get_default_pipeline_params()
    for name in vars(pipe):
        if hasattr(cfg_args, name):
            setattr(pipe, name, getattr(cfg_args, name))
    # Streaming training injects this optimization-owned field dynamically
    # immediately after argument extraction (train_streaming.py). It changes
    # whether the renderer receives raw or normalized quaternions.
    pipe.gsplat_sparse_grad = bool(getattr(cfg_args, "gsplat_sparse_grad", False))
    return pipe


def _plain_pipeline_dict(pipe: Any) -> dict[str, Any]:
    result = {}
    for key, value in vars(pipe).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _tool_identity(explicit_commit: str | None) -> dict[str, Any]:
    environment_commit = os.environ.get("ELIGSIR_GIT_COMMIT")
    git_head = _git_output("rev-parse", "HEAD")
    if explicit_commit is not None:
        commit, source = explicit_commit, "cli"
    elif environment_commit is not None:
        commit, source = environment_commit, "environment"
    else:
        commit, source = git_head, "git_head"
    if not commit or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", commit):
        raise RuntimeError(
            "tool commit must be a full 40- or 64-hex SHA; pass --tool-commit "
            "or export ELIGSIR_GIT_COMMIT"
        )
    commit = commit.lower()
    if git_head is None:
        verification = {
            "status": "declared_unverified_git_unavailable",
            "resolved_commit": None,
            "script_blob": None,
        }
    else:
        resolved = _git_output("rev-parse", f"{commit}^{{commit}}")
        if resolved is None or resolved.lower() != commit:
            raise RuntimeError(
                f"declared tool commit {commit} cannot be verified against local git"
            )
        committed_script_blob = _git_output(
            "rev-parse", f"{commit}:scripts/seed_render_decomp.py"
        )
        running_script_blob = _git_output("hash-object", str(Path(__file__).resolve()))
        if (
            committed_script_blob is None
            or running_script_blob is None
            or committed_script_blob != running_script_blob
        ):
            raise RuntimeError(
                "running seed_render_decomp.py does not match the declared commit blob"
            )
        verification = {
            "status": "verified_commit_and_script_blob",
            "resolved_commit": resolved.lower(),
            "script_blob": committed_script_blob,
        }
    status = _git_output("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "commit_identity_source": source,
        "commit_verification": verification,
        "branch": os.environ.get("ELIGSIR_GIT_BRANCH")
        or _git_output("branch", "--show-current"),
        "tracked_tree_dirty": None if status is None else bool(status),
        "script": "scripts/seed_render_decomp.py",
        "script_sha256": file_sha256(__file__),
    }


_REFERENCE_FIELDS = (
    ("schema_version",),
    ("analysis",),
    ("tool", "script"),
    ("tool", "script_sha256"),
    ("run", "source_git_dirty"),
    ("camera", "index"),
    ("camera", "entry"),
    ("camera", "entry_sha256"),
    ("camera", "gt_tensor_sha256"),
    ("camera", "gt_dtype"),
    ("camera", "gt_shape"),
    ("camera", "resolution_hw"),
    ("render", "production_full_pass"),
    ("render", "decomposition_passes"),
    ("render", "decomposition_render_depth"),
    ("render", "expected_depth_key"),
    ("render", "full_rgb_mode_tolerance"),
    ("render", "background_rgb"),
    ("render", "alpha_threshold"),
    ("render", "pipeline"),
    ("render", "canonical_rgb_metric_contract"),
    ("checkpoint_iteration_grid",),
)


_MISSING = object()


def _nested_value(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return _MISSING
        value = value[key]
    return value


def validate_reference_compatibility(
    current: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    source_commit_mismatch_reason: str | None = None,
) -> dict[str, Any]:
    """Fail closed on the small set of axes required for aligned comparison."""
    mismatches = []
    for path in _REFERENCE_FIELDS:
        current_value = _nested_value(current, path)
        reference_value = _nested_value(reference, path)
        if (
            current_value is _MISSING
            or reference_value is _MISSING
            or current_value != reference_value
        ):
            mismatches.append(".".join(path))

    grid = _nested_value(reference, ("checkpoint_iteration_grid",))
    if not isinstance(grid, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in grid
    ):
        mismatches.append("checkpoint_iteration_grid:type")
        grid = []
    if reference.get("expected_record_count") != len(grid):
        mismatches.append("expected_record_count")
    if reference.get("realized_record_count") != len(grid):
        mismatches.append("realized_record_count")
    if reference.get("realized_iterations") != grid:
        mismatches.append("realized_iterations")
    if current.get("expected_record_count") != len(grid):
        mismatches.append("current.expected_record_count")
    for label, manifest in (("current", current), ("reference", reference)):
        checkpoints = manifest.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != len(grid):
            mismatches.append(f"{label}.checkpoints")
            continue
        for expected_iteration, checkpoint in zip(grid, checkpoints):
            if (
                not isinstance(checkpoint, Mapping)
                or checkpoint.get("iteration") != expected_iteration
                or checkpoint.get("filename") != f"chkpnt{expected_iteration}.pth"
                or not isinstance(checkpoint.get("sha256"), str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint["sha256"]) is None
            ):
                mismatches.append(f"{label}.checkpoints")
                break
    if mismatches:
        raise ValueError(
            "reference manifest incompatible on: " + ", ".join(mismatches)
        )

    current_commit = _nested_value(current, ("run", "source_git_commit"))
    reference_commit = _nested_value(reference, ("run", "source_git_commit"))
    if not isinstance(current_commit, str) or not isinstance(reference_commit, str):
        raise ValueError("reference manifest incompatible on: run.source_git_commit")
    reason = (
        source_commit_mismatch_reason.strip()
        if isinstance(source_commit_mismatch_reason, str)
        else None
    )
    if current_commit != reference_commit:
        if not reason:
            raise ValueError(
                "source git commit mismatch requires "
                "--acknowledge-reference-source-commit-mismatch REASON"
            )
        return {
            "status": "acknowledged_diagnostic_source_commit_mismatch",
            "current_source_git_commit": current_commit,
            "reference_source_git_commit": reference_commit,
            "reason": reason,
        }
    if source_commit_mismatch_reason is not None:
        raise ValueError(
            "source commit mismatch acknowledgement was provided but commits match"
        )
    return {
        "status": "matched",
        "current_source_git_commit": current_commit,
        "reference_source_git_commit": reference_commit,
        "reason": None,
    }


def load_authenticated_reference_records(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    relative = _nested_value(manifest, ("series", "jsonl"))
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or PurePosixPath(relative).is_absolute()
        or any(part in ("", ".", "..") for part in relative.split("/"))
    ):
        raise ValueError("reference series must use a safe relative path")
    root = manifest_path.parent.resolve()
    series_path = (root / PurePosixPath(relative)).resolve()
    if not series_path.is_relative_to(root) or not series_path.is_file():
        raise ValueError("reference series must use a safe relative path to a file")

    declared = _nested_value(manifest, ("output_integrity", relative))
    if not isinstance(declared, Mapping):
        raise ValueError("reference series has no declared output integrity")
    actual = {
        "size_bytes": series_path.stat().st_size,
        "sha256": file_sha256(series_path),
    }
    if (
        declared.get("size_bytes") != actual["size_bytes"]
        or declared.get("sha256") != actual["sha256"]
    ):
        raise ValueError(
            f"reference series integrity mismatch: declared={dict(declared)} actual={actual}"
        )

    records = _read_jsonl(series_path)
    grid = manifest["checkpoint_iteration_grid"]
    checkpoints = manifest["checkpoints"]
    if len(records) != len(grid):
        raise ValueError("reference record count does not match declared iteration grid")
    for record, iteration, checkpoint in zip(records, grid, checkpoints):
        if record.get("iteration") != iteration:
            raise ValueError("reference record iteration does not match declared grid")
        if (
            record.get("checkpoint_filename") != checkpoint["filename"]
            or record.get("checkpoint_sha256") != checkpoint["sha256"]
        ):
            raise ValueError(
                f"reference record checkpoint identity mismatch at iteration {iteration}"
            )
    return records, {"relative_path": relative, **actual}


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaves(child, child_prefix))
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        leaves[prefix] = float(value)
    return leaves


def compute_reference_metric_deltas(
    current_records: list[dict[str, Any]], reference_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    current_grid = [record.get("iteration") for record in current_records]
    reference_grid = [record.get("iteration") for record in reference_records]
    if current_grid != reference_grid:
        raise ValueError(
            "reference record iteration grid does not align with current records"
        )
    result = []
    for current, reference in zip(current_records, reference_records):
        current_metrics: dict[str, float] = {}
        reference_metrics: dict[str, float] = {}
        for root in ("production_full_no_depth", "passes", "overlap"):
            if root in current:
                current_metrics.update(_numeric_leaves(current[root], root))
            if root in reference:
                reference_metrics.update(_numeric_leaves(reference[root], root))
        if set(current_metrics) != set(reference_metrics):
            raise ValueError(
                f"reference metrics do not align at iteration {current['iteration']}"
            )
        deltas = {
            key: current_metrics[key] - reference_metrics[key]
            for key in current_metrics
        }
        result.append(
            {
                "iteration": current["iteration"],
                "metric_deltas_current_minus_reference": deltas,
            }
        )
    return result


def collect_output_integrity(out_dir: os.PathLike[str] | str) -> dict[str, dict[str, Any]]:
    root = Path(out_dir)
    identities = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".tmp":
            continue
        identities[path.relative_to(root).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return identities


def _preflight_checkpoints(
    model_path: Path, iterations: list[int]
) -> list[dict[str, Any]]:
    identities = []
    for iteration in iterations:
        checkpoint = load_checkpoint_state(model_path / f"chkpnt{iteration}.pth")
        masks = exact_partition_masks(checkpoint.lineage_view)
        counts, opacity_means = _partition_statistics(checkpoint.state, masks)
        identities.append(
            {
                "iteration": checkpoint.saved_iteration,
                "filename": checkpoint.path.name,
                "sha256": checkpoint.checkpoint_sha256,
                "tuple_arity": checkpoint.tuple_arity,
                "counts": counts,
                "mean_gaussian_opacity": opacity_means,
            }
        )
        del masks, checkpoint
    return identities


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="run directory with chkpntN.pth")
    parser.add_argument("--scene", required=True, help="original canonical frame source")
    parser.add_argument("--iterations", required=True, nargs="+", type=int)
    parser.add_argument("--out", required=True, help="fresh output directory")
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument(
        "--full-rgb-mode-tolerance",
        type=float,
        default=1e-5,
        help="maximum raw RGB delta between production no-depth and full ED calls",
    )
    parser.add_argument(
        "--acknowledge-checkpoint-iteration-indeterminate",
        action="store_true",
        required=True,
        help=(
            "required acknowledgement that pre-fix async checkpoints may alias "
            "live state and their iteration labels are point-in-time indeterminate "
            "(issue #1172)"
        ),
    )
    parser.add_argument(
        "--reference-manifest",
        default=None,
        help="optional COMPLETE decomposition manifest for exact-axis comparison",
    )
    parser.add_argument(
        "--acknowledge-reference-source-commit-mismatch",
        metavar="REASON",
        default=None,
        help=(
            "narrow diagnostic acknowledgement for differing current/reference "
            "source commits; all other compatibility axes remain fail-closed"
        ),
    )
    parser.add_argument(
        "--tool-commit",
        default=None,
        help="tool git commit when linked-worktree container mounting hides .git",
    )
    return parser


def main(
    argv: list[str] | None = None,
    *,
    device: str | torch.device = "cuda",
    render_fn: Callable[..., Mapping[str, torch.Tensor]] | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    model_path = Path(args.model_path).resolve()
    scene_path = Path(args.scene).resolve()
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"output already exists; refusing to overwrite: {output_path}"
        )
    if len(set(args.iterations)) != len(args.iterations):
        raise ValueError("--iterations must not contain duplicates")
    if args.iterations != sorted(args.iterations):
        raise ValueError("--iterations must be in increasing order")
    if any(iteration < 0 for iteration in args.iterations):
        raise ValueError("--iterations must be non-negative")
    if not 0.0 <= args.alpha_threshold < 1.0:
        raise ValueError("--alpha-threshold must be in [0,1)")
    if args.full_rgb_mode_tolerance < 0:
        raise ValueError("--full-rgb-mode-tolerance must be non-negative")
    if (
        args.acknowledge_reference_source_commit_mismatch is not None
        and args.reference_manifest is None
    ):
        raise ValueError(
            "--acknowledge-reference-source-commit-mismatch requires "
            "--reference-manifest"
        )
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("exact RGB/alpha/depth decomposition requires one available GPU")

    tool = _tool_identity(args.tool_commit)
    create_output_dir(output_path)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "completion_state": "IN_PROGRESS",
        "analysis": "exact_seed_lineage_fixed_view_render_decomposition",
        "evidence_class": "UNMANAGED_DIAGNOSTIC",
        "canonical_scientific_evidence": False,
        "tool": tool,
        "checkpoint_iteration_grid": list(args.iterations),
        "expected_record_count": len(args.iterations),
        "realized_record_count": 0,
        "realized_iterations": [],
        "iteration_identity": {
            "point_in_time": "indeterminate",
            "acknowledged": bool(
                args.acknowledge_checkpoint_iteration_indeterminate
            ),
            "reason": (
                "pre-fix asynchronous checkpoint capture may serialize live tensor "
                "aliases after the labeled iteration"
            ),
            "issue": "https://github.com/eligsir-project/eligsir/issues/1172",
            "tensorboard_step_scalar_comparison": "prohibited",
        },
        "limitations": [
            "checkpoint labels are not exact point-in-time identities for pre-#1172 files",
            "isolated alpha/depth overlap is approximate, not in-full pixel attribution",
            "isolated RGB renders cannot be subtracted for exact compositing attribution",
            "this static decomposition does not establish a causal production fix",
        ],
        "output_integrity": {},
        "exception": None,
    }
    write_manifest(output_path, manifest)
    records: list[dict[str, Any]] = []
    try:
        fixed = load_fixed_seed_view(model_path, scene_path, device=device)
        pipe = _pipeline_from_cfg(fixed.cfg_args)
        background_values = [1.0, 1.0, 1.0] if bool(
            getattr(fixed.cfg_args, "white_background", False)
        ) else [0.0, 0.0, 0.0]
        background = torch.tensor(
            background_values, dtype=torch.float32, device=device
        )
        checkpoint_identities = _preflight_checkpoints(model_path, args.iterations)
        manifest.update(
            {
                "run": {
                    "model_path": str(model_path),
                    "source_git_commit": getattr(fixed.cfg_args, "git_commit", None),
                    "source_git_dirty": getattr(fixed.cfg_args, "git_dirty", None),
                    "source_model_path_recorded": getattr(
                        fixed.cfg_args, "model_path", None
                    ),
                },
                "scene_path": str(scene_path),
                "camera": {
                    "index": fixed.camera_index,
                    "entry": fixed.entry,
                    "cameras_json": str(fixed.camera_json_path),
                    "cameras_json_sha256": fixed.camera_file_sha256,
                    "entry_sha256": fixed.camera_entry_sha256,
                    "gt_tensor_sha256": fixed.gt_sha256,
                    "gt_dtype": str(fixed.gt.dtype),
                    "gt_shape": list(fixed.gt.shape),
                    "resolution_hw": [
                        int(fixed.gt.shape[-2]),
                        int(fixed.gt.shape[-1]),
                    ],
                },
                "render": {
                    "production_full_pass": {"render_depth": False},
                    "decomposition_passes": list(PASS_NAMES),
                    "decomposition_render_depth": True,
                    "expected_depth_key": "rendered_depth_ed",
                    "full_rgb_mode_tolerance": args.full_rgb_mode_tolerance,
                    "background_rgb": background_values,
                    "alpha_threshold": args.alpha_threshold,
                    "pipeline": _plain_pipeline_dict(pipe),
                    "canonical_rgb_metric_contract": {
                        "input": "raw_renderer_rgb",
                        "transform": "clamp_to_[0,1]",
                        "provenance_boundary": "declare_canonical_score_tensor",
                        "metric_name": "psnr_clamped_db",
                    },
                },
                "checkpoints": checkpoint_identities,
                "series": {"jsonl": "series.jsonl", "csv": "series.csv"},
            }
        )

        reference_records: list[dict[str, Any]] | None = None
        if args.reference_manifest:
            reference_path = Path(args.reference_manifest).resolve()
            with reference_path.open() as handle:
                reference_manifest = json.load(handle)
            if not isinstance(reference_manifest, dict):
                raise ValueError("reference manifest must be a JSON object")
            if reference_manifest.get("completion_state") != "COMPLETE":
                raise ValueError("reference manifest must have completion_state COMPLETE")
            source_commit_policy = validate_reference_compatibility(
                manifest,
                reference_manifest,
                source_commit_mismatch_reason=(
                    args.acknowledge_reference_source_commit_mismatch
                ),
            )
            reference_records, reference_series_identity = (
                load_authenticated_reference_records(
                    reference_path, reference_manifest
                )
            )
            manifest["reference_comparison"] = {
                "status": "compatible_pending_metrics",
                "manifest_path": str(reference_path),
                "manifest_sha256": file_sha256(reference_path),
                "source_commit_policy": source_commit_policy,
                "reference_series": reference_series_identity,
                "comparison_jsonl": "reference_comparison.jsonl",
            }
        else:
            manifest["reference_comparison"] = {"status": "not_requested"}
        write_manifest(output_path, manifest)

        for identity in checkpoint_identities:
            iteration = int(identity["iteration"])
            checkpoint = load_checkpoint_state(model_path / identity["filename"])
            if checkpoint.checkpoint_sha256 != identity["sha256"]:
                raise RuntimeError(
                    f"checkpoint {identity['filename']} changed after preflight; refusing to render"
                )
            masks = exact_partition_masks(checkpoint.lineage_view)
            rendered = render_checkpoint(
                checkpoint.state,
                masks,
                fixed.camera,
                fixed.gt,
                pipe,
                background,
                alpha_threshold=args.alpha_threshold,
                full_rgb_tolerance=args.full_rgb_mode_tolerance,
                device=device,
                render_fn=render_fn,
            )
            artifact_paths = write_checkpoint_artifacts(
                output_path,
                iteration,
                rendered["outputs"],
                fixed.gt,
                production_full_rgb=rendered["production_full_no_depth"]["rgb"],
            )
            record = {
                "iteration": iteration,
                "checkpoint_filename": identity["filename"],
                "checkpoint_sha256": identity["sha256"],
                "counts": identity["counts"],
                "mean_gaussian_opacity": identity["mean_gaussian_opacity"],
                "production_full_no_depth": rendered["production_full_no_depth"][
                    "metrics"
                ],
                "passes": rendered["passes"],
                "overlap": rendered["overlap"],
                "full_rgb_mode_parity": rendered["full_rgb_mode_parity"],
                "gt_artifact": f"iteration_{iteration:06d}/gt_rgb_clamped.png",
                "artifacts": artifact_paths,
            }
            records.append(record)
            write_series(output_path, records)
            manifest["realized_iterations"] = [
                int(item["iteration"]) for item in records
            ]
            manifest["realized_record_count"] = len(records)
            manifest["output_integrity"] = collect_output_integrity(output_path)
            write_manifest(output_path, manifest)

            score_nodes = [record["production_full_no_depth"], *record["passes"].values()]
            if any(node["score_domain"]["status"] != "available" for node in score_nodes):
                raise RuntimeError(
                    f"non-finite raw renderer RGB at checkpoint label {iteration}; "
                    "raw artifacts were preserved and canonical scoring was withheld"
                )
            if record["full_rgb_mode_parity"]["status"] != "passed":
                raise RuntimeError(
                    "production/decomposition RGB parity failed at checkpoint label "
                    f"{iteration}: {record['full_rgb_mode_parity']}"
                )
            print(
                f"[seed_render_decomp] checkpoint_label={iteration} "
                f"full_psnr_clamped={record['passes']['full']['psnr_clamped_db']:.6f} "
                f"counts={identity['counts']}",
                flush=True,
            )
            del rendered, masks, checkpoint

        if reference_records is not None:
            comparison = compute_reference_metric_deltas(records, reference_records)
            _write_jsonl(output_path / "reference_comparison.jsonl", comparison)
            manifest["reference_comparison"]["status"] = "complete"
            manifest["reference_comparison"]["aligned_iterations"] = [
                item["iteration"] for item in comparison
            ]
        manifest["completion_state"] = "COMPLETE"
        manifest["output_integrity"] = collect_output_integrity(output_path)
        write_manifest(output_path, manifest)
        print(f"[seed_render_decomp] wrote fresh bundle: {output_path}", flush=True)
        return 0
    except BaseException as exc:
        manifest["completion_state"] = "FAILED"
        manifest["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["realized_iterations"] = [
            int(item["iteration"]) for item in records
        ]
        manifest["realized_record_count"] = len(records)
        manifest["output_integrity"] = collect_output_integrity(output_path)
        write_manifest(output_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
