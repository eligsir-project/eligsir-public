#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Post-hoc PSNR/SSIM/LPIPS evaluation of a rendered checkpoint (issue #640).

**This script no longer reads PNGs.** Until 2026-08, ``render.py`` wrote the
float32 renderer output and the float32 ground truth through
``torchvision.utils.save_image(..., ".png")`` -- an irreversible
float32 -> uint8 quantization (and an *implicit* clamp to [0, 1], which
``save_image`` performs silently) -- and this file reopened those files with
PIL before computing metrics. PNG compression is lossless, but the samples
being compressed had already been destroyed: the scores were computed on
8-bit data, giving this path different evaluation semantics from every other
metric path in the project.

The score path now consumes **exact score artifacts**: the unmodified float32
tensors, persisted by ``render.py`` next to the PNGs with dtype/shape/SHA-256
metadata (see :func:`save_score_tensor`). The PNGs still exist, but they are
visualization artifacts only and nothing here reads them. There is
deliberately **no PNG fallback** -- a run directory rendered before this
change has no exact artifacts, and silently scoring its 8-bit PNGs is exactly
the defect being retired. Re-run ``render.py`` on such a directory.

Clamping is now an explicitly declared part of the evaluation protocol rather
than a side effect of image serialization: it is applied here, recorded in
``results.json`` under ``provenance``, and reported per method.

The post-hoc and inline report paths both use the shared canonical scorer and
persist enough exact evidence for deterministic parity validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Exact score-artifact contract (the canonical representation for this path)
# --------------------------------------------------------------------------

#: Bumped whenever the on-disk payload layout changes incompatibly.
SCORE_ARTIFACT_FORMAT_VERSION = 2

#: Sub-directory (under a ``ours_<iter>`` method directory) holding the exact
#: float tensors. Siblings ``renders/`` and ``gt/`` hold the PNGs, which are
#: visualization artifacts only.
SCORE_ARTIFACT_DIRNAME = "score_tensors"
SCORE_ARTIFACT_SUFFIX = ".pt"

#: Canonical score-space representation for this path.
SCORE_DTYPE = torch.float32
SCORE_DOMAIN = "rgb_float32_chw_unit_interval"

#: Era marker: distinguishes post-#640 exact-tensor scores from legacy 8-bit-PNG
#: round-trip scores. Recorded in per-tensor metadata and results.json provenance.
SCORE_ARTIFACT_ERA = "issue_640_exact_float32_tensor"

#: Declared score-domain clamp. This is a sample-changing transform, not a
#: representation conversion -- it is applied explicitly and recorded.
SCORE_CLAMP_MIN = 0.0
SCORE_CLAMP_MAX = 1.0

# Same-device, same-tensor parity tolerances for inline versus post-hoc scoring.
SCORE_PARITY_TOLERANCES = {"psnr": 1e-6, "ssim": 1e-7, "lpips": 1e-7}
SCORE_ARTIFACT_ROLES = frozenset({
    "inline_scientific_metric_inputs",
    "posthoc_scientific_metric_inputs",
})


class ScoreArtifactError(RuntimeError):
    """Raised when exact score artifacts are missing, malformed or corrupt."""


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA-256 over the exact bytes of a CPU, contiguous tensor."""
    cpu = tensor.detach().to("cpu").contiguous()
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def score_tensor_metadata(tensor: torch.Tensor, *, kind: str, view_name: str) -> dict:
    """Provenance metadata describing one exact score tensor."""
    cpu = tensor.detach().to(SCORE_DTYPE).to("cpu").contiguous()
    return {
        "format_version": SCORE_ARTIFACT_FORMAT_VERSION,
        "era": SCORE_ARTIFACT_ERA,
        "kind": kind,  # "render" | "gt"
        "view": view_name,
        "domain": SCORE_DOMAIN,
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": _tensor_sha256(cpu),
        # Recorded, NOT applied at save time: the persisted samples are the
        # renderer's/GT's exact values, clamp included or not as they came.
        "clamp_applied_at_save": False,
        "declared_score_clamp": [SCORE_CLAMP_MIN, SCORE_CLAMP_MAX],
        "value_min": float(cpu.min().item()) if cpu.numel() else 0.0,
        "value_max": float(cpu.max().item()) if cpu.numel() else 0.0,
    }


def save_score_tensor(tensor: torch.Tensor, path, *, kind: str, view_name: str) -> dict:
    """Persist ``tensor`` exactly (float32, no quantization) with metadata.

    Returns the metadata dict that was stored alongside the tensor.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = tensor.detach().to(SCORE_DTYPE).to("cpu").contiguous()
    meta = score_tensor_metadata(cpu, kind=kind, view_name=view_name)
    torch.save({"tensor": cpu, "meta": meta}, path)
    return meta


def load_score_tensor(path):
    """Load an exact score tensor, verifying dtype/shape/hash.

    Returns ``(tensor, meta)``. Raises :class:`ScoreArtifactError` on any
    mismatch -- a corrupt artifact must never be silently scored.
    """
    path = Path(path)
    if not path.is_file():
        raise ScoreArtifactError(f"missing score artifact: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only kwarg
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "tensor" not in payload or "meta" not in payload:
        raise ScoreArtifactError(f"malformed score artifact (missing keys): {path}")
    tensor = payload["tensor"]
    meta = payload["meta"]
    if not isinstance(tensor, torch.Tensor):
        raise ScoreArtifactError(f"malformed score artifact (not a tensor): {path}")
    if meta.get("format_version") != SCORE_ARTIFACT_FORMAT_VERSION:
        raise ScoreArtifactError(
            f"score artifact format {meta.get('format_version')} != expected "
            f"{SCORE_ARTIFACT_FORMAT_VERSION}: {path}"
        )
    if str(tensor.dtype) != meta.get("dtype"):
        raise ScoreArtifactError(f"score artifact dtype mismatch: {path}")
    if list(tensor.shape) != list(meta.get("shape", [])):
        raise ScoreArtifactError(f"score artifact shape mismatch: {path}")
    if _tensor_sha256(tensor) != meta.get("sha256"):
        raise ScoreArtifactError(f"score artifact hash mismatch (corrupt): {path}")
    return tensor, meta


def save_score_pair(
    render: torch.Tensor,
    gt: torch.Tensor,
    root,
    *,
    artifact_name: str,
    view_name: str,
) -> dict:
    """Persist one exact render/GT pair and return its hash ledger row."""
    root = Path(root)
    render_meta = save_score_tensor(
        render, root / "renders" / artifact_name,
        kind="render", view_name=view_name,
    )
    gt_meta = save_score_tensor(
        gt, root / "gt" / artifact_name,
        kind="gt", view_name=view_name,
    )
    return {
        "artifact": artifact_name,
        "view_name": view_name,
        "render_sha256": render_meta["sha256"],
        "gt_sha256": gt_meta["sha256"],
        "render_dtype": render_meta["dtype"],
        "gt_dtype": gt_meta["dtype"],
        "render_shape": render_meta["shape"],
        "gt_shape": gt_meta["shape"],
    }


def _canonical_json_sha256(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_artifact_set_sha256(entries) -> str:
    """Canonical content-set hash, independent of artifact layout/order."""
    content = [
        {key: value for key, value in entry.items() if key != "artifact"}
        for entry in sorted(entries, key=lambda row: row["view_name"])
    ]
    return _canonical_json_sha256(content)


def score_holdout_hash(entries) -> str:
    """Match the offline scored-holdout hash policy using stable view names."""
    names = sorted(f"name:{entry['view_name']}" for entry in entries)
    return hashlib.sha1(",".join(names).encode("utf-8")).hexdigest()[:16]


def mean_score_values(values) -> float | None:
    values = [float(value) for value in values]
    return math.fsum(values) / len(values) if values else None


def write_score_set_manifest(
    root,
    *,
    entries,
    split: str,
    iteration: int,
    role: str,
    score_provenance=None,
    metrics=None,
    tolerances=None,
    extra=None,
) -> dict:
    """Write a self-verifying manifest for a set of exact score inputs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda row: row["artifact"])
    payload = {
        "format_version": SCORE_ARTIFACT_FORMAT_VERSION,
        "era": SCORE_ARTIFACT_ERA,
        "score_domain": SCORE_DOMAIN,
        "role": role,
        "split": split,
        "iteration": int(iteration),
        "n_views": len(ordered),
        "views": ordered,
        "artifact_set_sha256": score_artifact_set_sha256(ordered),
        "holdout_hash": score_holdout_hash(ordered),
        "score_provenance": score_provenance,
        "metrics": metrics,
        "tolerances": tolerances,
    }
    if extra:
        overlap = set(payload).intersection(extra)
        if overlap:
            raise ValueError(f"score manifest extra cannot replace contract fields: {sorted(overlap)}")
        payload.update(dict(extra))
    payload["manifest_payload_sha256"] = _canonical_json_sha256(payload)
    with open(root / "manifest.json", "w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
    return payload


def reset_score_artifact_set(root) -> None:
    """Replace one producer-owned exact score set before a fresh write."""
    root = Path(root)
    if root.is_symlink():
        raise ScoreArtifactError(f"refusing to replace symlinked score artifact set: {root}")
    if root.exists() and not root.is_dir():
        raise ScoreArtifactError(f"score artifact set is not a directory: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)


def _score_pair_ledger(root: Path) -> tuple[list[dict], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    renders_dir = root / "renders"
    gt_dir = root / "gt"
    if not renders_dir.is_dir() or not gt_dir.is_dir():
        raise ScoreArtifactError(f"missing exact score-pair directories under: {root}")
    render_names = sorted(p.name for p in renders_dir.glob(f"*{SCORE_ARTIFACT_SUFFIX}"))
    gt_names = sorted(p.name for p in gt_dir.glob(f"*{SCORE_ARTIFACT_SUFFIX}"))
    if not render_names or render_names != gt_names:
        raise ScoreArtifactError(
            f"render/GT artifact name mismatch under {root}: "
            f"renders={render_names} gt={gt_names}"
        )
    entries = []
    pairs = {}
    seen_views = set()
    for name in render_names:
        render, render_meta = load_score_tensor(renders_dir / name)
        gt, gt_meta = load_score_tensor(gt_dir / name)
        if render.shape != gt.shape:
            raise ScoreArtifactError(
                f"render/GT shape mismatch for {name}: "
                f"{tuple(render.shape)} != {tuple(gt.shape)}"
            )
        view_name = str(render_meta.get("view"))
        if view_name != str(gt_meta.get("view")):
            raise ScoreArtifactError(f"render/GT view identity mismatch for {name}")
        if not view_name or view_name == "None" or view_name in seen_views:
            raise ScoreArtifactError(f"missing or duplicate score view identity: {view_name!r}")
        seen_views.add(view_name)
        entries.append({
            "artifact": name,
            "view_name": view_name,
            "render_sha256": render_meta["sha256"],
            "gt_sha256": gt_meta["sha256"],
            "render_dtype": render_meta["dtype"],
            "gt_dtype": gt_meta["dtype"],
            "render_shape": render_meta["shape"],
            "gt_shape": gt_meta["shape"],
        })
        pairs[name] = (render, gt)
    return entries, pairs


def score_exact_pairs(
    entries,
    pairs,
    *,
    device: str,
    lpips_fn,
) -> tuple[dict, dict, dict]:
    """Recompute canonical NCHW score evidence from exact raw tensors."""
    from eligsir.utils.loss_utils import evaluation_ssim
    from eligsir.utils.score_contract import (
        apply_declared_score_clamp_pair,
        declare_canonical_score_tensor,
        score_canonical_pair,
    )

    rows = []
    score_provenance = None
    clamp_totals = {"samples_changed": 0, "samples_total": 0, "samples_nonfinite": 0}
    for entry in entries:
        raw_render, raw_gt = pairs[entry["artifact"]]
        render = raw_render[:3]
        gt = raw_gt[:3]
        if render.ndim == 3:
            render = render.unsqueeze(0)
            gt = gt.unsqueeze(0)
        render = render.to(device)
        gt, _ = declare_canonical_score_tensor(
            gt.to(device), context=f"score_artifact_gt:{entry['view_name']}"
        )
        render, gt, audit = apply_declared_score_clamp_pair(
            render, gt, context=f"score_artifact_pair:{entry['view_name']}"
        )
        provenance = audit["score_provenance"]
        for clamp_record in (audit["render"], audit["gt"]):
            for key in clamp_totals:
                clamp_totals[key] += clamp_record[key]
        if score_provenance is None:
            score_provenance = provenance
        elif score_provenance != provenance:
            raise ScoreArtifactError("score provenance changed across exact views")
        scores = score_canonical_pair(
            render,
            gt,
            ssim_fn=evaluation_ssim,
            lpips_fn=lpips_fn,
            context=f"score_artifact_pair:{entry['view_name']}",
        )
        if scores.ssim is None or scores.lpips is None:
            raise ScoreArtifactError("PSNR/SSIM/LPIPS evidence is required")
        rows.append({
            "artifact": entry["artifact"],
            "view_name": entry["view_name"],
            "psnr": float(scores.psnr.mean().item()),
            "ssim": float(scores.ssim.item()),
            "lpips": float(scores.lpips.item()),
        })
    aggregate = {
        metric: mean_score_values(row[metric] for row in rows)
        for metric in ("psnr", "ssim", "lpips")
    }
    return (
        json.loads(json.dumps(score_provenance)),
        {"per_view": rows, "aggregate": aggregate},
        clamp_totals,
    )


def score_persisted_pairs(root, *, device: str, lpips_fn):
    """Load, verify, and score one persisted exact-tensor set."""
    entries, pairs = _score_pair_ledger(Path(root))
    provenance, score_metrics, clamp_totals = score_exact_pairs(
        entries, pairs, device=device, lpips_fn=lpips_fn
    )
    return entries, provenance, score_metrics, clamp_totals


def _assert_metric_close(label: str, claimed, actual, tolerance: float) -> None:
    delta = abs(float(claimed) - float(actual))
    if delta > tolerance:
        raise ScoreArtifactError(
            f"{label} claim mismatch: delta={delta} tolerance={tolerance}"
        )


def _load_complete_score_manifest(
    root: Path,
    actual_entries: list[dict],
    actual_pairs,
    *,
    device: str,
    lpips_fn,
) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise ScoreArtifactError(f"missing score artifact manifest: {path}")
    try:
        manifest = json.loads(path.read_text())
    except Exception as exc:
        raise ScoreArtifactError(f"malformed score artifact manifest: {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ScoreArtifactError(f"malformed score artifact manifest: {path}")
    claimed_payload_hash = manifest.get("manifest_payload_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_payload_sha256", None)
    if not claimed_payload_hash or claimed_payload_hash != _canonical_json_sha256(unhashed):
        raise ScoreArtifactError(f"score artifact manifest hash mismatch: {path}")
    if manifest.get("n_views") != len(actual_entries) or manifest.get("views") != actual_entries:
        raise ScoreArtifactError(f"score artifact manifest view/hash ledger mismatch: {path}")
    if manifest.get("artifact_set_sha256") != score_artifact_set_sha256(actual_entries):
        raise ScoreArtifactError(f"score artifact aggregate hash mismatch: {path}")
    expected_contract = {
        "format_version": SCORE_ARTIFACT_FORMAT_VERSION,
        "era": SCORE_ARTIFACT_ERA,
        "score_domain": SCORE_DOMAIN,
        "split": "test",
    }
    for field, expected in expected_contract.items():
        if manifest.get(field) != expected:
            raise ScoreArtifactError(f"score artifact manifest invalid {field}: {path}")
    if manifest.get("role") not in SCORE_ARTIFACT_ROLES:
        raise ScoreArtifactError(f"score artifact manifest invalid role: {path}")
    if not isinstance(manifest.get("iteration"), int) or manifest["iteration"] < 0:
        raise ScoreArtifactError(f"score artifact manifest invalid iteration: {path}")
    if manifest.get("holdout_hash") != score_holdout_hash(actual_entries):
        raise ScoreArtifactError(f"score artifact holdout hash mismatch: {path}")
    if not isinstance(manifest.get("score_provenance"), dict) or not manifest["score_provenance"]:
        raise ScoreArtifactError(f"score artifact manifest missing provenance: {path}")
    score_metrics = manifest.get("metrics")
    if not isinstance(score_metrics, dict):
        raise ScoreArtifactError(f"score artifact manifest missing metrics: {path}")
    per_view = score_metrics.get("per_view")
    aggregate = score_metrics.get("aggregate")
    if not isinstance(per_view, list) or len(per_view) != len(actual_entries):
        raise ScoreArtifactError(f"score artifact manifest missing per-view metrics: {path}")
    if not isinstance(aggregate, dict):
        raise ScoreArtifactError(f"score artifact manifest missing aggregate metrics: {path}")
    tolerances = manifest.get("tolerances")
    if tolerances != SCORE_PARITY_TOLERANCES:
        raise ScoreArtifactError(f"score artifact manifest missing tolerances: {path}")
    expected_names = [entry["artifact"] for entry in actual_entries]
    if [row.get("artifact") for row in per_view if isinstance(row, dict)] != expected_names:
        raise ScoreArtifactError(f"score artifact metric/view mismatch: {path}")
    expected_views = [entry["view_name"] for entry in actual_entries]
    if [row.get("view_name") for row in per_view] != expected_views:
        raise ScoreArtifactError(f"score artifact metric/view identity mismatch: {path}")
    for metric_name in ("psnr", "ssim", "lpips"):
        tolerance = tolerances.get(metric_name)
        if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
            raise ScoreArtifactError(
                f"score artifact manifest missing tolerance for {metric_name}: {path}"
            )
        values = [aggregate.get(metric_name)] + [row.get(metric_name) for row in per_view]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ScoreArtifactError(
                f"score artifact manifest missing finite {metric_name} metrics: {path}"
            )
    actual_provenance, actual_metrics, _ = score_exact_pairs(
        actual_entries,
        actual_pairs,
        device=device,
        lpips_fn=lpips_fn,
    )
    if manifest["score_provenance"] != actual_provenance:
        raise ScoreArtifactError(f"score provenance claim mismatch: {path}")
    for claimed_row, actual_row in zip(per_view, actual_metrics["per_view"]):
        for metric_name in ("psnr", "ssim", "lpips"):
            _assert_metric_close(
                f"per-view {metric_name}",
                claimed_row[metric_name],
                actual_row[metric_name],
                float(tolerances[metric_name]),
            )
    for metric_name in ("psnr", "ssim", "lpips"):
        claimed_mean = mean_score_values(row[metric_name] for row in per_view)
        _assert_metric_close(
            f"aggregate {metric_name}",
            aggregate[metric_name],
            claimed_mean,
            float(tolerances[metric_name]),
        )
        _assert_metric_close(
            f"recomputed aggregate {metric_name}",
            aggregate[metric_name],
            actual_metrics["aggregate"][metric_name],
            float(tolerances[metric_name]),
        )
    return manifest


def compare_score_artifact_sets(
    left_root,
    right_root,
    *,
    device=None,
    lpips_fn=None,
) -> dict:
    """Require bit-exact identity of two persisted score-input sets.

    Every tensor is hash-verified while loading. Views are aligned by their
    stable identity; any view, dtype, shape, sample, or claim mismatch fails.
    """
    left_root = Path(left_root)
    right_root = Path(right_root)
    left_entries, left_pairs = _score_pair_ledger(left_root)
    right_entries, right_pairs = _score_pair_ledger(right_root)
    if lpips_fn is None:
        try:
            from lpipsPyTorch import lpips
        except ImportError as exc:
            raise ScoreArtifactError("LPIPS is required to verify score evidence") from exc
        lpips_fn = lambda render, gt: lpips(render, gt, net_type="vgg")
    device = _resolve_device(device)
    left_by_view = {
        entry["view_name"]: (entry, left_pairs[entry["artifact"]])
        for entry in left_entries
    }
    right_by_view = {
        entry["view_name"]: (entry, right_pairs[entry["artifact"]])
        for entry in right_entries
    }
    left_names = sorted(left_by_view)
    right_names = sorted(right_by_view)
    if left_names != right_names:
        raise ScoreArtifactError(
            f"score artifact view sets differ: left={left_names} right={right_names}"
        )
    for name in left_names:
        _, (left_render, left_gt) = left_by_view[name]
        _, (right_render, right_gt) = right_by_view[name]
        if not torch.equal(left_render, right_render):
            raise ScoreArtifactError(f"render tensor mismatch for view {name}")
        if not torch.equal(left_gt, right_gt):
            raise ScoreArtifactError(f"GT tensor mismatch for view {name}")
    left_manifest = _load_complete_score_manifest(
        left_root, left_entries, left_pairs, device=device, lpips_fn=lpips_fn
    )
    right_manifest = _load_complete_score_manifest(
        right_root, right_entries, right_pairs, device=device, lpips_fn=lpips_fn
    )
    left_hash = score_artifact_set_sha256(left_entries)
    right_hash = score_artifact_set_sha256(right_entries)
    if left_hash != right_hash:
        raise ScoreArtifactError("score artifact content-set hash mismatch")
    if left_manifest["iteration"] != right_manifest["iteration"]:
        raise ScoreArtifactError("score iteration mismatch")
    if {left_manifest["role"], right_manifest["role"]} != SCORE_ARTIFACT_ROLES:
        raise ScoreArtifactError("comparison requires one inline and one posthoc manifest")
    if left_manifest["score_provenance"] != right_manifest["score_provenance"]:
        raise ScoreArtifactError("score provenance mismatch")
    if left_manifest["tolerances"] != right_manifest["tolerances"]:
        raise ScoreArtifactError("score tolerance contract mismatch")
    tolerances = left_manifest["tolerances"]
    left_metrics = left_manifest["metrics"]
    right_metrics = right_manifest["metrics"]
    left_rows = {row["view_name"]: row for row in left_metrics["per_view"]}
    right_rows = {row["view_name"]: row for row in right_metrics["per_view"]}
    for view_name in left_names:
        for metric_name in ("psnr", "ssim", "lpips"):
            _assert_metric_close(
                f"per-view {metric_name} parity for {view_name}",
                left_rows[view_name][metric_name],
                right_rows[view_name][metric_name],
                float(tolerances[metric_name]),
            )
    for metric_name in ("psnr", "ssim", "lpips"):
        _assert_metric_close(
            f"aggregate {metric_name} parity",
            left_metrics["aggregate"][metric_name],
            right_metrics["aggregate"][metric_name],
            float(tolerances[metric_name]),
        )
    return {
        "match": True,
        "n_views": len(left_names),
        "left_artifact_set_sha256": left_hash,
        "right_artifact_set_sha256": right_hash,
    }


def apply_score_clamp(tensor: torch.Tensor):
    """Apply the declared score-domain clamp, reporting whether it bit.

    Returns ``(clamped_tensor, record)``. ``record`` is what gets written into
    ``results.json`` so that a sample-changing operation in the score path is
    never invisible.
    """
    from eligsir.utils.score_contract import apply_declared_score_clamp

    return apply_declared_score_clamp(
        tensor,
        context="metrics:declared_score_clamp",
    )


def score_artifact_dir(method_dir) -> Path:
    return Path(method_dir) / SCORE_ARTIFACT_DIRNAME


def read_score_pairs(method_dir, device: str = "cpu"):
    """Read the (render, gt) exact-tensor pairs for one ``ours_<iter>`` dir.

    Returns ``(renders, gts, names, metas)`` with each tensor as a 4-D
    ``(1, 3, H, W)`` float32 batch on ``device``. No PNG is read anywhere.
    """
    from eligsir.utils.score_contract import declare_canonical_score_tensor

    method_dir = Path(method_dir)
    root = score_artifact_dir(method_dir)
    renders_dir = root / "renders"
    gts_dir = root / "gt"
    if not renders_dir.is_dir() or not gts_dir.is_dir():
        raise ScoreArtifactError(
            f"no exact score artifacts under {root}.\n"
            "This run directory was rendered before issue #640 (renders were "
            "persisted only as 8-bit PNGs, which are visualization artifacts "
            "and must not be scored). Re-run render.py on this checkpoint to "
            "produce exact float32 score artifacts."
        )

    names = sorted(p.name for p in renders_dir.glob(f"*{SCORE_ARTIFACT_SUFFIX}"))
    if not names:
        raise ScoreArtifactError(f"score artifact directory is empty: {renders_dir}")

    renders, gts, metas = [], [], []
    for name in names:
        render_t, render_meta = load_score_tensor(renders_dir / name)
        gt_t, gt_meta = load_score_tensor(gts_dir / name)
        if render_t.shape[-2:] != gt_t.shape[-2:]:
            raise ScoreArtifactError(
                f"render/gt resolution mismatch for {name}: "
                f"{tuple(render_t.shape)} vs {tuple(gt_t.shape)}"
            )
        renders.append(render_t[:3].unsqueeze(0).to(device))
        gt_score, _ = declare_canonical_score_tensor(
            gt_t[:3].unsqueeze(0).to(device),
            context=f"metrics:score_artifact_gt:{name}",
        )
        gts.append(gt_score)
        metas.append({"render": render_meta, "gt": gt_meta})
    return renders, gts, names, metas


def _resolve_device(device=None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model_paths, device=None):
    """Score every ``test/ours_<iter>`` method directory of each model path."""
    # Heavy imports are deferred so that the score-artifact contract above can
    # be imported (and tested) without pulling in LPIPS/fused-SSIM.
    from lpipsPyTorch import lpips
    device = _resolve_device(device)

    full_dict = {}
    per_view_dict = {}
    failures = []
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in sorted(os.listdir(test_dir)):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                method_dir = test_dir / method
                score_root = score_artifact_dir(method_dir)
                (
                    exact_entries,
                    score_provenance,
                    score_metrics,
                    clamp_totals,
                ) = score_persisted_pairs(
                    score_root,
                    device=device,
                    lpips_fn=lambda render, gt: lpips(render, gt, net_type="vgg"),
                )
                aggregate = score_metrics["aggregate"]
                print("  SSIM : {:>12.7f}".format(aggregate["ssim"]))
                print("  PSNR : {:>12.7f}".format(aggregate["psnr"]))
                print("  LPIPS: {:>12.7f}".format(aggregate["lpips"]))
                print("")

                score_manifest = write_score_set_manifest(
                    score_root,
                    entries=exact_entries,
                    split="test",
                    iteration=int(method.rsplit("_", 1)[-1]),
                    role="posthoc_scientific_metric_inputs",
                    score_provenance=score_provenance,
                    metrics=score_metrics,
                    tolerances=SCORE_PARITY_TOLERANCES,
                    extra={"png_role": "visualization_only"},
                )

                full_dict[scene_dir][method].update(
                    {
                        "SSIM": aggregate["ssim"],
                        "PSNR": aggregate["psnr"],
                        "LPIPS": aggregate["lpips"],
                        "provenance": {
                            "metric_input": "exact_score_tensors",
                            "era": SCORE_ARTIFACT_ERA,
                            "score_artifact_format_version": SCORE_ARTIFACT_FORMAT_VERSION,
                            "score_domain": SCORE_DOMAIN,
                            "score_contract": score_provenance,
                            "artifact_set_sha256": score_manifest["artifact_set_sha256"],
                            "manifest_payload_sha256": score_manifest["manifest_payload_sha256"],
                            "png_roundtrip": False,
                            "n_views": len(exact_entries),
                            "device": device,
                            "score_domain_clamp": {
                                "applied": True,
                                "range": [SCORE_CLAMP_MIN, SCORE_CLAMP_MAX],
                                "samples_changed": clamp_totals["samples_changed"],
                                "samples_total": clamp_totals["samples_total"],
                                "fraction_changed": (
                                    clamp_totals["samples_changed"]
                                    / clamp_totals["samples_total"]
                                    if clamp_totals["samples_total"] else 0.0
                                ),
                                "nonfinite_policy": "rejected before clamp",
                                "samples_nonfinite": clamp_totals["samples_nonfinite"],
                            },
                        },
                    }
                )
                per_view_dict[scene_dir][method].update(
                    {
                        metric.upper(): {
                            row["artifact"]: row[metric]
                            for row in score_metrics["per_view"]
                        }
                        for metric in ("ssim", "psnr", "lpips")
                    }
                )

            with open(os.path.join(scene_dir, "results.json"), "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(os.path.join(scene_dir, "per_view.json"), "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as exc:  # noqa: BLE001 -- reported, never silenced
            # The original code used a bare `except:` printing only "Unable to
            # compute metrics", which would have hidden the
            # missing-score-artifact error this file now raises.
            print(f"Unable to compute metrics for model {scene_dir}: "
                  f"{type(exc).__name__}: {exc}")
            failures.append(scene_dir)

    return failures


if __name__ == "__main__":
    parser = ArgumentParser(description="Metric evaluation parameters")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for metric evaluation (default: cuda if available, else cpu)",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    failed = evaluate(args.model_paths, device=device)
    if failed:
        raise SystemExit(f"metric evaluation failed for: {', '.join(failed)}")
