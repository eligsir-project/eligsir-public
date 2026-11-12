"""Producer-side artifact manifest generator for EliGSiR.

Provides explicit artifact classification:
- scientific_evidence: config, provenance, metrics, trajectory, final model/map, reports (must hash SHA-256)
- reproducibility_artifact: logs, reproduction scripts, selected milestone checkpoints
- diagnostic: contact sheets, trajectory videos, error heatmaps, psnr curves
- ephemeral: intermediate checkpoints, dense per-frame intermediate dumps, caches (metadata only / DEFERRED hash)

Computes Merkle group hashes and root manifest hash for instant O(1) verification.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "eligsir.artifact-manifest/v1"
MANIFEST_FILENAME = "artifact_manifest.json"


def _hash_file(path: Path | str, chunk_size: int = 1048576) -> tuple[str, int]:
    """Compute SHA-256 hash and byte size of a file."""
    p = Path(path)
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj: Any) -> str:
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def classify_artifact(
    file_path: Path | str,
    relative_path: str,
    is_final_checkpoint: bool = False,
) -> tuple[str, str, str]:
    """Classify an artifact file into (artifact_kind, scientific_role, hash_policy).

    Returns:
        artifact_kind: e.g. "summary", "report", "checkpoint", "log", "image", "video", "provenance"
        scientific_role: "scientific_evidence" | "reproducibility_artifact" | "diagnostic" | "ephemeral"
        hash_policy: "IMMEDIATE" | "DEFERRED"
    """
    p = Path(file_path)
    name_lower = p.name.lower()
    rel_lower = str(relative_path).lower().replace("\\", "/")

    # 1. Canonical Scientific Evidence (Must hash immediately)
    if name_lower in (
        "run_summary.json",
        "summary.json",
        "run_completion_state.json",
        "completion_state.json",
        "report.json",
        "experiment_origin.json",
        "experiment_declaration.json",
        "experiment_declaration_receipt.json",
        "cfg_args",
        "args.json",
        "params.json",
        "timing_summary.json",
        "reproduce.sh",
        "input_seed.ply",
    ) or (rel_lower.startswith("trajectory_eval/") and name_lower.endswith((".json", ".txt"))):
        kind = "provenance" if "origin" in name_lower or "decl" in name_lower else (
            "summary" if "summary" in name_lower else (
                "report" if "report" in name_lower else "config"
            )
        )
        return kind, "scientific_evidence", "IMMEDIATE"

    # Final iteration point cloud / model
    if name_lower == "point_cloud.ply" or is_final_checkpoint or "final" in name_lower:
        return "point_cloud", "scientific_evidence", "IMMEDIATE"

    # 2. Reproducibility Artifacts (Must hash immediately)
    if name_lower.endswith(".log") or name_lower in ("cameras.json", "test_cameras.json"):
        kind = "log" if name_lower.endswith(".log") else "config"
        return kind, "reproducibility_artifact", "IMMEDIATE"

    # 3. Diagnostic Artifacts (Hash immediately)
    if (
        "contact_sheet" in name_lower
        or name_lower.endswith(".mp4")
        or name_lower in (
            "psnr_history.jsonl",
            "psnr_over_time.svg",
            "partition_psnr_over_time.svg",
            "evaluation_ticks.jsonl",
            "keyframe_admission_log.jsonl",
            "replay_sampling_summary.json",
            "replay_sampling_counts.jsonl",
            "replay_sampling_heatmap.svg",
            # PR #1783 follow-up: default-on per-run system-performance trace
            # (eligsir/telemetry/system_sampler.py) -- same diagnostic
            # tier as its psnr_history/replay_sampling siblings above (a
            # timestamped .jsonl trace, its plot, and a small derived
            # summary), so it is discoverable from this manifest rather than
            # silently absent from it.
            "system_performance.jsonl",
            "system_performance.svg",
            "system_performance_summary.json",
        )
    ):
        kind = "video" if name_lower.endswith(".mp4") else ("metrics" if name_lower.endswith(".jsonl") else "image")
        return kind, "diagnostic", "IMMEDIATE"

    # 4. Ephemeral / Heavyweight Intermediate (DEFERRED hash)
    # Intermediate PyTorch weights, dense evaluation frames, intermediate point clouds
    if (
        re.match(r"^chkpnt\d+\.pth$", name_lower)
        or "holdout_gt/" in rel_lower
        or (name_lower.endswith((".png", ".jpg")) and "renders/" in rel_lower)
        or name_lower.startswith("events.out.tfevents")
    ):
        kind = "checkpoint" if name_lower.endswith(".pth") else ("telemetry" if "tfevents" in name_lower else "image")
        return kind, "ephemeral", "DEFERRED"

    # Default: checkpoint if ply, otherwise diagnostic/ephemeral based on extension
    if name_lower.endswith((".ply", ".pth", ".pt", ".bin")):
        return "checkpoint", "reproducibility_artifact", "IMMEDIATE"

    return "artifact", "diagnostic", "IMMEDIATE"


def build_artifact_manifest(
    model_path: str | Path,
    compute_deferred: bool = False,
) -> dict[str, Any]:
    """Scan model directory and build a canonical artifact_manifest.json.

    Args:
        model_path: Directory containing the model and artifacts.
        compute_deferred: If True, calculates SHA-256 for ephemeral/deferred artifacts too.

    Returns:
        Dictionary representation of the artifact manifest.
    """
    root = Path(model_path).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Model path directory does not exist: {model_path}")

    # Discover files
    all_files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != MANIFEST_FILENAME and not p.name.endswith(".tmp"):
            all_files.append(p)

    # Determine latest iteration for point_cloud / checkpoint
    point_cloud_dirs: list[int] = []
    for p in all_files:
        if p.name == "point_cloud.ply":
            parent_name = p.parent.name
            m = re.search(r"iteration_(\d+)", parent_name)
            if m:
                point_cloud_dirs.append(int(m.group(1)))
    max_pc_iter = max(point_cloud_dirs) if point_cloud_dirs else None

    artifacts_list: list[dict[str, Any]] = []
    group_items: dict[str, list[dict[str, Any]]] = {
        "scientific_evidence": [],
        "reproducibility_artifact": [],
        "diagnostic": [],
        "ephemeral": [],
    }

    for p in all_files:
        rel_path = str(p.relative_to(root)).replace("\\", "/")
        is_final = False
        if max_pc_iter is not None and f"iteration_{max_pc_iter}/point_cloud.ply" in rel_path:
            is_final = True

        kind, role, policy = classify_artifact(p, rel_path, is_final_checkpoint=is_final)

        stat = p.stat()
        size_bytes = stat.st_size
        mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))

        if policy == "IMMEDIATE" or compute_deferred:
            sha, actual_size = _hash_file(p)
            hash_status = "VERIFIED"
            size_bytes = actual_size
        else:
            sha = None
            hash_status = "DEFERRED"

        item = {
            "relative_path": rel_path,
            "artifact_kind": kind,
            "scientific_role": role,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "sha256": sha,
            "hash_status": hash_status,
        }
        artifacts_list.append(item)
        group_items[role].append(item)

    # Merkle group hashes
    groups_manifest: dict[str, str] = {}
    for role, items in group_items.items():
        # Compute deterministic group hash from sorted relative paths, sizes, and available shas
        group_repr = [
            {"path": it["relative_path"], "size": it["size_bytes"], "sha256": it["sha256"], "status": it["hash_status"]}
            for it in items
        ]
        groups_manifest[f"{role}_manifest_sha256"] = _sha256_json(group_repr)

    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    manifest_data: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": created_at,
        "model_path": str(root),
        "total_files": len(artifacts_list),
        "groups": groups_manifest,
        "artifacts": artifacts_list,
    }
    manifest_data["root_manifest_sha256"] = _sha256_json(groups_manifest)

    # Write atomically
    target = root / MANIFEST_FILENAME
    tmp = root / f"{MANIFEST_FILENAME}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, target)

    return manifest_data


def load_artifact_manifest(model_path: str | Path) -> dict[str, Any] | None:
    """Load artifact_manifest.json if it exists and matches schema."""
    target = Path(model_path) / MANIFEST_FILENAME
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == MANIFEST_SCHEMA:
            return data
        return None
    except Exception:
        return None
