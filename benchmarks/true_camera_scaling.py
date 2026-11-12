#!/usr/bin/env python3
"""Deterministic true-camera gsplat scaling benchmark.

Dry-run planning is intentionally torch-free.  GPU execution imports the
production true-camera adapter lazily so a plan can be validated on any host.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import numbers
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SCHEMA_VERSION = "eligsir.true_camera_scaling.row.v1"
PLAN_SCHEMA_VERSION = "eligsir.true_camera_scaling.plan.v1"
CAMERA_COUNTS = (1, 2, 4, 8, 16, 32, 64)
GAUSSIAN_SIZES = (8192, 30000, 100000, "mature")
RENDER_MODES = ("RGB", "RGB+ED")
SH_MODES = ("0", "production")
PASS_MODES = ("forward", "forward+backward")
GPU_LABELS = ("8060S", "R9700", "RTX3090")

_CAMERA_FIELDS = (
    "width",
    "height",
    "fx",
    "fy",
    "cx",
    "cy",
    "position",
    "rotation",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def camera_identity(entry: dict[str, Any]) -> str:
    """Return the manifest's stable source identity without inventing one."""
    for key in ("source_identity", "source_id", "img_name", "id"):
        value = entry.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    raise ValueError(
        "every camera needs source_identity, source_id, img_name, or id"
    )


def load_camera_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("camera manifest must be a non-empty JSON list")
    entries: list[dict[str, Any]] = []
    identities: list[str] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"camera manifest entry {index} must be an object")
        missing = [field for field in _CAMERA_FIELDS if field not in entry]
        if missing:
            raise ValueError(
                f"camera manifest entry {index} is missing required fields {missing}"
            )
        if int(entry["width"]) <= 0 or int(entry["height"]) <= 0:
            raise ValueError(f"camera manifest entry {index} has invalid resolution")
        if len(entry["position"]) != 3:
            raise ValueError(f"camera manifest entry {index} position must have 3 values")
        rotation = entry["rotation"]
        if (
            not isinstance(rotation, list)
            or len(rotation) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in rotation)
        ):
            raise ValueError(f"camera manifest entry {index} rotation must be 3x3")
        numeric_values = [
            entry["fx"],
            entry["fy"],
            entry["cx"],
            entry["cy"],
            *entry["position"],
            *(value for row in rotation for value in row),
        ]
        try:
            finite = all(math.isfinite(float(value)) for value in numeric_values)
        except (TypeError, ValueError):
            finite = False
        if not finite or float(entry["fx"]) <= 0 or float(entry["fy"]) <= 0:
            raise ValueError(f"camera manifest entry {index} has invalid numeric geometry")
        identity = camera_identity(entry)
        entries.append(entry)
        identities.append(identity)
    duplicate_ids = sorted(
        identity for identity in set(identities) if identities.count(identity) > 1
    )
    if duplicate_ids:
        raise ValueError(f"camera manifest contains duplicate source identities: {duplicate_ids}")
    return entries, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "entry_count": len(entries),
        "identity_sha256": hashlib.sha256(_canonical_json(identities)).hexdigest(),
    }


@dataclass(frozen=True)
class CameraSelection:
    requested_c: int
    cameras: tuple[dict[str, Any], ...]
    identities: tuple[str, ...]
    eligible_count: int
    status: str
    reason: str | None

    @property
    def realized_c(self) -> int:
        return len(self.cameras)


def select_distinct_cameras(
    entries: Sequence[dict[str, Any]],
    requested_c: int,
    *,
    eligible_identities: Iterable[str] | None = None,
) -> CameraSelection:
    """Select an ordered distinct prefix; never wait and never duplicate-pad."""
    requested_c = int(requested_c)
    if requested_c < 1:
        raise ValueError("requested C must be >= 1")
    eligible = None if eligible_identities is None else set(eligible_identities)
    selected: list[dict[str, Any]] = []
    identities: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        identity = camera_identity(entry)
        if identity in seen or (eligible is not None and identity not in eligible):
            continue
        seen.add(identity)
        selected.append(entry)
        identities.append(identity)
        if len(selected) == requested_c:
            break
    eligible_count = sum(
        1
        for entry in entries
        if eligible is None or camera_identity(entry) in eligible
    )
    if len(selected) < requested_c:
        return CameraSelection(
            requested_c=requested_c,
            cameras=tuple(selected),
            identities=tuple(identities),
            eligible_count=eligible_count,
            status="skip",
            reason="insufficient_distinct_manifest_cameras",
        )
    return CameraSelection(
        requested_c=requested_c,
        cameras=tuple(selected),
        identities=tuple(identities),
        eligible_count=eligible_count,
        status="eligible",
        reason=None,
    )


def build_plan(
    entries: Sequence[dict[str, Any]],
    *,
    camera_counts: Sequence[int] = CAMERA_COUNTS,
    gaussian_sizes: Sequence[str] = GAUSSIAN_SIZES,
    render_modes: Sequence[str] = RENDER_MODES,
    sh_modes: Sequence[str] = SH_MODES,
    pass_modes: Sequence[str] = PASS_MODES,
    gpu_labels: Sequence[str] = GPU_LABELS,
) -> list[dict[str, Any]]:
    selections = {
        int(c): select_distinct_cameras(entries, int(c)) for c in camera_counts
    }
    points: list[dict[str, Any]] = []
    for gpu, c, n, render_mode, sh_mode, pass_mode in itertools.product(
        gpu_labels,
        camera_counts,
        gaussian_sizes,
        render_modes,
        sh_modes,
        pass_modes,
    ):
        selection = selections[int(c)]
        normalized_n = "mature" if str(n) == "mature" else int(n)
        points.append(
            {
                "gpu_label": gpu,
                "requested_c": int(c),
                "realized_c": selection.realized_c,
                "camera_identities": list(selection.identities),
                "requested_n": normalized_n,
                "render_mode": render_mode,
                "sh_mode": sh_mode,
                "pass_mode": pass_mode,
                "mode": "packed+sparse",
                "plan_status": selection.status,
                "skip_reason": selection.reason,
            }
        )
    return points


def deterministic_subset_indices(
    total_n: int, requested_n: int | str, *, seed: int
):
    """Return exact sorted indices for one immutable Gaussian snapshot."""
    import numpy as np

    total_n = int(total_n)
    if total_n < 1:
        raise ValueError("mature population must contain at least one Gaussian")
    if str(requested_n) == "mature":
        return np.arange(total_n, dtype=np.int64)
    count = int(requested_n)
    if count < 1:
        raise ValueError("requested N must be >= 1")
    if count > total_n:
        raise ValueError(
            f"requested N={count} exceeds mature population N={total_n}"
        )
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_n, size=count, replace=False).astype(np.int64))


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("timing samples must not be empty")
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summarize_timings(
    samples_ms: Sequence[float], *, realized_c: int
) -> dict[str, float]:
    if int(realized_c) < 1:
        raise ValueError("realized C must be >= 1")
    samples = [float(sample) for sample in samples_ms]
    if not samples or any(not math.isfinite(sample) or sample <= 0 for sample in samples):
        raise ValueError("timing samples must be finite and > 0")
    mean_ms = statistics.fmean(samples)
    return {
        "p50_transaction_ms": _percentile(samples, 50.0),
        "p95_transaction_ms": _percentile(samples, 95.0),
        "mean_transaction_ms": mean_ms,
        "ms_per_camera": mean_ms / int(realized_c),
        "cameras_per_s": 1000.0 * int(realized_c) / mean_ms,
    }


def _scaling_key(row: dict[str, Any]) -> tuple[Any, ...]:
    hardware = row.get("hardware") or {}
    checkpoint = row.get("checkpoint") or {}
    manifest = row.get("camera_manifest") or {}
    subset = row.get("gaussian_subset") or {}
    return (
        hardware.get("gpu_label"),
        hardware.get("device_name"),
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


def add_scaling_efficiency(rows: Sequence[dict[str, Any]]) -> None:
    """Annotate eta(C)=C*T(1)/T(C) within exact-comparability groups."""
    baselines: dict[tuple[Any, ...], float] = {}
    for row in rows:
        source = row.get("source_provenance") or {}
        if (
            row.get("status", "ok") == "ok"
            and int(row.get("realized_c", 0)) == 1
            and row.get("mean_transaction_ms") is not None
            and source.get("comparison_eligible") is True
            and source.get("verification_status") == "verified_clean_git_head"
            and row.get("code_sha") == source.get("resolved_git_head")
            and re.fullmatch(
                r"[0-9a-fA-F]{40}", str(source.get("resolved_git_tree") or "")
            )
            and re.fullmatch(
                r"[0-9a-fA-F]{64}", str(source.get("entrypoint_sha256") or "")
            )
        ):
            baselines[_scaling_key(row)] = float(row["mean_transaction_ms"])
    for row in rows:
        if row.get("status", "ok") != "ok" or row.get("mean_transaction_ms") is None:
            row["scaling_efficiency"] = None
            row["scaling_efficiency_reason"] = "point_not_successful"
            continue
        source = row.get("source_provenance") or {}
        if (
            source.get("comparison_eligible") is not True
            or source.get("verification_status") != "verified_clean_git_head"
            or row.get("code_sha") != source.get("resolved_git_head")
            or re.fullmatch(
                r"[0-9a-fA-F]{40}", str(source.get("resolved_git_tree") or "")
            )
            is None
            or re.fullmatch(
                r"[0-9a-fA-F]{64}", str(source.get("entrypoint_sha256") or "")
            )
            is None
        ):
            row["scaling_efficiency"] = None
            row["scaling_efficiency_reason"] = (
                "source_provenance_not_verified_for_comparison"
            )
            continue
        baseline = baselines.get(_scaling_key(row))
        if baseline is None:
            row["scaling_efficiency"] = None
            row["scaling_efficiency_reason"] = "matching_C1_baseline_unavailable"
            continue
        realized_c = int(row["realized_c"])
        row["scaling_efficiency"] = (
            realized_c * baseline / float(row["mean_transaction_ms"])
        )
        row["scaling_efficiency_reason"] = None


def profiler_evidence_unavailable(reason: str) -> dict[str, Any]:
    """Explicit nulls for counters that this harness cannot truthfully infer."""
    capability = {"available": False, "reason": str(reason)}
    return {
        "capabilities": {
            "stage_timings": dict(capability),
            "hardware_counters": dict(capability),
            "kernel_launch_count": dict(capability),
        },
        "stage_timings_ms": {
            "projection": None,
            "spherical_harmonics": None,
            "intersection_sort": None,
            "raster_forward": None,
            "raster_backward": None,
        },
        "gfx_active_percent": None,
        "memory_bandwidth_bytes_per_s": None,
        "kernel_launch_count": None,
    }


@dataclass(frozen=True)
class ObservedGPUIdentity:
    """Physical accelerator evidence resolved on the benchmark runner."""

    device_name: str
    architecture: str | None
    backend: str | None
    pci_vendor_id: int | None
    pci_device_id: int | None
    pci_domain_id: int | None
    pci_bus_id: int | None
    pci_bdf: str | None
    total_memory_bytes: int


@dataclass(frozen=True)
class _ExpectedGPUIdentity:
    architecture: str
    backend: str
    pci_vendor_id: int
    pci_device_id: int
    nominal_memory_bytes: int
    memory_tolerance_bytes: int
    supporting_name_tokens: tuple[str, ...]


_GIB = 1024**3
_EXPECTED_GPU_IDENTITIES = {
    "8060S": _ExpectedGPUIdentity(
        architecture="gfx1151",
        backend="ROCm",
        pci_vendor_id=0x1002,
        pci_device_id=0x1586,
        nominal_memory_bytes=124 * _GIB,
        memory_tolerance_bytes=2 * _GIB,
        supporting_name_tokens=("amd", "radeon"),
    ),
    "R9700": _ExpectedGPUIdentity(
        architecture="gfx1201",
        backend="ROCm",
        pci_vendor_id=0x1002,
        pci_device_id=0x7551,
        nominal_memory_bytes=32 * _GIB,
        memory_tolerance_bytes=1 * _GIB,
        supporting_name_tokens=("amd", "radeon"),
    ),
    "RTX3090": _ExpectedGPUIdentity(
        architecture="sm_86",
        backend="CUDA",
        pci_vendor_id=0x10DE,
        pci_device_id=0x2204,
        nominal_memory_bytes=24 * _GIB,
        memory_tolerance_bytes=1 * _GIB,
        supporting_name_tokens=("nvidia", "geforce", "rtx3090"),
    ),
}


def resolve_host_machine_identity(
    *,
    environ: Mapping[str, str] | None = None,
    dmi_uuid_path: Path = Path("/sys/class/dmi/id/product_uuid"),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
) -> dict[str, Any]:
    """Resolve a privacy-safe stable physical-host identity when defensible."""
    environment = os.environ if environ is None else environ
    explicit = str(environment.get("ELIGSIR_HOST_MACHINE_ID") or "").strip()
    if explicit:
        source = "explicit_env_sha256"
        value = explicit
    else:
        source = ""
        value = ""
        for candidate_path, candidate_source in (
            (dmi_uuid_path, "dmi_product_uuid_sha256"),
            (boot_id_path, "linux_boot_id_sha256"),
        ):
            try:
                candidate = candidate_path.read_text().strip().lower()
            except OSError:
                continue
            compact = re.sub(r"[^0-9a-f]", "", candidate)
            if (
                re.fullmatch(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                    candidate,
                )
                is not None
                and compact not in {"0" * 32, "f" * 32}
            ):
                source = candidate_source
                value = candidate
                break
        if not source:
            return {
                "available": False,
                "host_machine_id": None,
                "host_machine_id_source": None,
                "reason": "stable physical host identity unavailable",
            }
    digest = hashlib.sha256(
        b"eligsir-host-identity-v1\0"
        + source.encode("utf-8")
        + b"\0"
        + value.lower().encode("utf-8")
    ).hexdigest()
    return {
        "available": True,
        "host_machine_id": f"sha256:{digest}",
        "host_machine_id_source": source,
        "reason": None,
    }


def validate_gpu_label(gpu_label: str, observed: ObservedGPUIdentity) -> None:
    """Verify a declared platform from typed physical identity evidence."""
    if not isinstance(observed, ObservedGPUIdentity):
        raise TypeError("GPU validation requires a typed physical identity")
    try:
        expected = _EXPECTED_GPU_IDENTITIES[gpu_label]
    except KeyError as exc:
        raise ValueError(f"unknown declared GPU label {gpu_label!r}") from exc
    if observed.architecture != expected.architecture:
        raise ValueError(
            f"GPU architecture mismatch for {gpu_label}: expected "
            f"{expected.architecture}, observed {observed.architecture!r}"
        )
    if observed.backend != expected.backend:
        raise ValueError(
            f"GPU backend mismatch for {gpu_label}: expected {expected.backend}, "
            f"observed {observed.backend!r}"
        )
    if observed.pci_bdf is None:
        raise ValueError(f"PCI BDF is unavailable for declared GPU {gpu_label}")
    if observed.pci_domain_id is None or observed.pci_bus_id is None:
        raise ValueError(f"PCI location is incomplete for declared GPU {gpu_label}")
    bdf_match = re.fullmatch(
        r"([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):[0-9a-fA-F]{2}\.[0-7]",
        observed.pci_bdf,
    )
    if (
        bdf_match is None
        or int(bdf_match.group(1), 16) != int(observed.pci_domain_id)
        or int(bdf_match.group(2), 16) != int(observed.pci_bus_id)
    ):
        raise ValueError(
            f"PCI BDF {observed.pci_bdf!r} is inconsistent with observed "
            f"domain/bus for {gpu_label}"
        )
    if observed.pci_vendor_id != expected.pci_vendor_id:
        raise ValueError(
            f"PCI vendor mismatch for {gpu_label}: expected "
            f"0x{expected.pci_vendor_id:04x}, observed {observed.pci_vendor_id!r}"
        )
    if observed.pci_device_id != expected.pci_device_id:
        raise ValueError(
            f"PCI device mismatch for {gpu_label}: expected "
            f"0x{expected.pci_device_id:04x}, observed {observed.pci_device_id!r}"
        )
    memory_delta = abs(
        int(observed.total_memory_bytes) - expected.nominal_memory_bytes
    )
    if memory_delta > expected.memory_tolerance_bytes:
        raise ValueError(
            f"GPU memory mismatch for {gpu_label}: expected "
            f"{expected.nominal_memory_bytes} +/- {expected.memory_tolerance_bytes} "
            f"bytes, observed {observed.total_memory_bytes}"
        )
    normalized_name = re.sub(r"[^a-z0-9]", "", observed.device_name.lower())
    if not any(token in normalized_name for token in expected.supporting_name_tokens):
        raise ValueError(
            f"device name {observed.device_name!r} is inconsistent supporting "
            f"evidence for {gpu_label}"
        )


def _resolve_pci_identity(
    *, pci_domain_id: int | None, pci_bus_id: int | None
) -> tuple[str, int, int]:
    """Resolve the exact display-controller function from Linux sysfs."""
    if pci_domain_id is None or pci_bus_id is None:
        raise RuntimeError("GPU PCI domain/bus properties are unavailable")
    root = Path("/sys/bus/pci/devices")
    candidates: list[Path] = []
    pattern = f"{int(pci_domain_id):04x}:{int(pci_bus_id):02x}:*"
    for candidate in sorted(root.glob(pattern)):
        try:
            class_code = (candidate / "class").read_text().strip().lower()
        except OSError:
            continue
        if class_code.startswith("0x03"):
            candidates.append(candidate)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one display-controller PCI function for {pattern}, "
            f"found {len(candidates)}"
        )
    candidate = candidates[0]
    try:
        vendor_id = int((candidate / "vendor").read_text().strip(), 16)
        device_id = int((candidate / "device").read_text().strip(), 16)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read PCI identity for {candidate.name}") from exc
    return candidate.name, vendor_id, device_id


def _observe_gpu_identity(torch, properties, device_name: str) -> ObservedGPUIdentity:
    raw_architecture = getattr(properties, "gcnArchName", None)
    if getattr(torch.version, "hip", None):
        backend = "ROCm"
        architecture = (
            str(raw_architecture).split(":", 1)[0].lower()
            if raw_architecture
            else None
        )
    elif getattr(torch.version, "cuda", None):
        backend = "CUDA"
        major = getattr(properties, "major", None)
        minor = getattr(properties, "minor", None)
        architecture = (
            f"sm_{int(major)}{int(minor)}"
            if major is not None and minor is not None
            else None
        )
    else:
        backend = None
        architecture = None
    pci_domain_id = getattr(properties, "pci_domain_id", None)
    pci_bus_id = getattr(properties, "pci_bus_id", None)
    pci_bdf, pci_vendor_id, pci_device_id = _resolve_pci_identity(
        pci_domain_id=pci_domain_id,
        pci_bus_id=pci_bus_id,
    )
    return ObservedGPUIdentity(
        device_name=str(device_name),
        architecture=architecture,
        backend=backend,
        pci_vendor_id=pci_vendor_id,
        pci_device_id=pci_device_id,
        pci_domain_id=int(pci_domain_id),
        pci_bus_id=int(pci_bus_id),
        pci_bdf=pci_bdf,
        total_memory_bytes=int(properties.total_memory),
    )


def _dirty_tree_fingerprint(repo_root: Path, status: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"git-status-v1-z\0")
    digest.update(status)
    digest.update(b"git-diff-binary-head\0")
    digest.update(
        subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--"], cwd=repo_root
        )
    )
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
    )
    for raw_path in sorted(path for path in untracked.split(b"\0") if path):
        relative = os.fsdecode(raw_path)
        path = repo_root / relative
        digest.update(b"untracked-path\0")
        digest.update(raw_path)
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.fsencode(os.readlink(path)))
        else:
            digest.update(b"file\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def resolve_source_provenance(
    repo_root: Path, declared_sha: str | None
) -> dict[str, Any]:
    """Resolve a declared SHA against the runner's actual Git/tree state."""
    declared = str(declared_sha).strip().lower() if declared_sha else None
    if declared is not None and re.fullmatch(r"[0-9a-f]{40}", declared) is None:
        raise ValueError("declared code SHA must be exactly 40 hexadecimal characters")
    repo_root = Path(repo_root).resolve()
    try:
        resolved_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().lower()
        resolved_tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().lower()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {
            "verification_status": "unverified_git_metadata_unavailable",
            "comparison_eligible": False,
            "reason": "runner could not resolve Git HEAD and working-tree state",
            "declared_code_sha": declared,
            "resolved_git_head": None,
            "resolved_git_tree": None,
            "code_sha": None,
            "working_tree_state": "unknown",
            "working_tree_fingerprint_sha256": None,
            "entrypoint_sha256": sha256_file(Path(__file__)),
        }
    if declared is not None and declared != resolved_head:
        raise ValueError(
            f"declared code SHA {declared} does not match resolved Git HEAD "
            f"{resolved_head}"
        )
    dirty = bool(status)
    return {
        "verification_status": (
            "dirty_git_tree" if dirty else "verified_clean_git_head"
        ),
        "comparison_eligible": not dirty,
        "reason": (
            "working tree differs from resolved Git HEAD" if dirty else None
        ),
        "declared_code_sha": declared,
        "resolved_git_head": resolved_head,
        "resolved_git_tree": resolved_tree,
        "code_sha": None if dirty else resolved_head,
        "working_tree_state": "dirty" if dirty else "clean",
        "working_tree_fingerprint_sha256": (
            _dirty_tree_fingerprint(repo_root, status) if dirty else None
        ),
        "entrypoint_sha256": sha256_file(Path(__file__)),
    }


def _hardware_software_provenance(gpu_label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm accelerator is unavailable")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    device_name = torch.cuda.get_device_name(device)
    observed_identity = _observe_gpu_identity(torch, properties, device_name)
    validate_gpu_label(gpu_label, observed_identity)
    hardware = {
        "gpu_label": gpu_label,
        "device_index": device,
        "device_name": device_name,
        "gpu_uuid": str(getattr(properties, "uuid", "unknown")),
        "gpu_arch": getattr(properties, "gcnArchName", None),
        "pci_bus_id": getattr(properties, "pci_bus_id", None),
        "gpu_architecture": observed_identity.architecture,
        "accelerator_backend": observed_identity.backend,
        "pci_domain_id": observed_identity.pci_domain_id,
        "pci_bdf": observed_identity.pci_bdf,
        "pci_vendor_id": observed_identity.pci_vendor_id,
        "pci_device_id": observed_identity.pci_device_id,
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
        "machine": platform.machine(),
    }
    host_identity = resolve_host_machine_identity()
    hardware["stable_host_identity_available"] = host_identity["available"]
    hardware["stable_host_identity_reason"] = host_identity["reason"]
    if host_identity["available"]:
        hardware["host_machine_id"] = host_identity["host_machine_id"]
        hardware["host_machine_id_source"] = host_identity[
            "host_machine_id_source"
        ]
    try:
        import gsplat

        gsplat_version = getattr(gsplat, "__version__", None)
        gsplat_file = getattr(gsplat, "__file__", None)
    except Exception as exc:
        gsplat_version = None
        gsplat_file = None
        hardware["gsplat_import_error"] = f"{type(exc).__name__}: {exc}"
    software = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "gsplat": gsplat_version,
        "gsplat_file": gsplat_file,
        "kernel_release": platform.release(),
        "container_image_digest": os.environ.get("ELIGSIR_IMAGE_DIGEST") or None,
        "container_image_digest_reason": (
            None
            if os.environ.get("ELIGSIR_IMAGE_DIGEST")
            else "ELIGSIR_IMAGE_DIGEST was not supplied"
        ),
    }
    return hardware, software


def _load_checkpoint_state(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, tuple) or len(payload) not in (2, 3):
        raise ValueError("checkpoint payload must be an exact 2- or 3-tuple")
    state, saved_iteration = payload[:2]
    if isinstance(saved_iteration, bool) or not isinstance(saved_iteration, numbers.Integral):
        raise ValueError("checkpoint saved iteration must be an integer")
    if not isinstance(state, dict) or state.get("layout") != "gsplat":
        raise ValueError("checkpoint must contain a gsplat-layout capture state")
    params = state.get("params")
    required = ("means", "scales", "quats", "opacities", "sh0", "shN")
    if not isinstance(params, dict) or any(name not in params for name in required):
        raise ValueError(f"checkpoint params must contain {required}")
    total_n = int(params["means"].shape[0])
    if total_n < 1 or any(int(params[name].shape[0]) != total_n for name in required):
        raise ValueError("checkpoint render parameters are not row-aligned")
    expected_shapes = {
        "means": (total_n, 3),
        "scales": (total_n, 3),
        "quats": (total_n, 4),
        "opacities": (total_n,),
        "sh0": (total_n, 1, 3),
    }
    for name, expected in expected_shapes.items():
        if tuple(params[name].shape) != expected:
            raise ValueError(
                f"checkpoint param {name!r} must have shape {expected}, "
                f"got {tuple(params[name].shape)}"
            )
    if params["shN"].ndim != 3 or int(params["shN"].shape[2]) != 3:
        raise ValueError("checkpoint param 'shN' must have shape [N,K,3]")
    coefficient_count = 1 + int(params["shN"].shape[1])
    coefficient_root = math.isqrt(coefficient_count)
    if coefficient_root * coefficient_root != coefficient_count:
        raise ValueError("checkpoint SH coefficient count is not a square")
    active_degree = int(state.get("active_sh_degree", 0))
    max_degree = coefficient_root - 1
    if active_degree < 0 or active_degree > max_degree:
        raise ValueError(
            f"checkpoint active SH degree {active_degree} is outside [0,{max_degree}]"
        )
    return state, {
        "saved_iteration": int(saved_iteration),
        "tuple_arity": len(payload),
        "mature_n": total_n,
        "production_sh_degree": active_degree,
    }


def _write_subset_manifest(
    indices, *, requested_n: int | str, seed: int, total_n: int, output_dir: Path
) -> dict[str, Any]:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"N-{requested_n}-seed-{seed}.npy"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Gaussian index manifest: {path}")
    np.save(path, indices, allow_pickle=False)
    file_digest = sha256_file(path)
    indices_digest = hashlib.sha256(
        indices.astype("<i8", copy=False).tobytes(order="C")
    ).hexdigest()
    return {
        "path": str(path.resolve()),
        "file_sha256": file_digest,
        "indices_sha256": indices_digest,
        "algorithm": "identity_all" if requested_n == "mature" else "numpy_pcg64_choice_without_replacement_sorted",
        "seed": int(seed),
        "population_n": int(total_n),
        "realized_n": int(len(indices)),
        "dtype": str(indices.dtype),
    }


def _build_render_model(state: dict[str, Any], indices, *, sh_degree: int):
    import torch
    from scene.gsplat_model import GsplatGaussianModel

    model = object.__new__(GsplatGaussianModel)
    model.active_sh_degree = int(sh_degree)
    coeffs = 1 + int(state["params"]["shN"].shape[1])
    model.max_sh_degree = math.isqrt(coeffs) - 1
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    model.params = {}
    for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
        value = state["params"][name].detach().index_select(0, index_tensor).clone()
        model.params[name] = value.to(device="cuda", non_blocking=True).contiguous().requires_grad_(True)
    return model


def _build_cameras(entries: Sequence[dict[str, Any]]):
    from render_report import _camera_json_to_minicam

    resolutions = {(int(entry["width"]), int(entry["height"])) for entry in entries}
    if len(resolutions) != 1:
        raise ValueError("true camera batches require one homogeneous manifest resolution")
    return [_camera_json_to_minicam(entry, device="cuda") for entry in entries]


def _build_pipe():
    from arguments import get_default_pipeline_params

    pipe = get_default_pipeline_params()
    pipe.sh_impl = "gsplat"
    pipe.gsplat_sparse_grad = True
    pipe.render_mode = "RGB"
    return pipe


def _zero_model_grads(model) -> None:
    for parameter in model.params.values():
        parameter.grad = None


def _invoke_production_transaction(
    cameras, model, pipe, background, *, render_mode: str, pass_mode: str
):
    import torch

    render_depth = render_mode == "RGB+ED"
    if len(cameras) == 1:
        from eligsir.mapping.renderer.gsplat_backend import render

        result = render(
            cameras[0], model, pipe, background,
            update_sh_rest=True,
            render_depth=render_depth,
        )
        tensors = [result["render"]]
        if render_depth:
            tensors.append(result["rendered_depth_ed"])
    else:
        from eligsir.mapping.renderer.true_camera_batch import render_true_camera_batch

        result = render_true_camera_batch(
            cameras, model, pipe, background,
            update_sh_rest=True,
            render_depth=render_depth,
        )
        tensors = [result.renders]
        if render_depth and result.rendered_depth_ed is not None:
            tensors.append(result.rendered_depth_ed)
    if pass_mode == "forward+backward":
        loss = sum(tensor.float().mean() for tensor in tensors)
        loss.backward()
    return result


def _measure_point(
    *,
    state: dict[str, Any],
    indices,
    cameras: Sequence[Any],
    sh_degree: int,
    render_mode: str,
    pass_mode: str,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    import torch

    model = _build_render_model(state, indices, sh_degree=sh_degree)
    pipe = _build_pipe()
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    def invoke() -> None:
        _zero_model_grads(model)
        if pass_mode == "forward":
            with torch.no_grad():
                _invoke_production_transaction(
                    cameras, model, pipe, background,
                    render_mode=render_mode, pass_mode=pass_mode,
                )
        else:
            _invoke_production_transaction(
                cameras, model, pipe, background,
                render_mode=render_mode, pass_mode=pass_mode,
            )

    for _ in range(warmups):
        invoke()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    return {
        "samples_transaction_ms": samples,
        "peak_memory_allocated_bytes": peak_allocated,
        "peak_memory_reserved_bytes": peak_reserved,
    }


def _parse_csv(value: str, *, cast=str) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, default=Path("true_camera_scaling.jsonl"))
    parser.add_argument("--list-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-label", choices=GPU_LABELS)
    parser.add_argument("--camera-counts", default=",".join(map(str, CAMERA_COUNTS)))
    parser.add_argument("--gaussian-sizes", default=",".join(map(str, GAUSSIAN_SIZES)))
    parser.add_argument("--render-modes", default=",".join(RENDER_MODES))
    parser.add_argument("--sh-modes", default=",".join(SH_MODES))
    parser.add_argument("--pass-modes", default=",".join(PASS_MODES))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--code-sha",
        help=(
            "Declared expected code SHA; verified against runner Git HEAD when "
            "metadata is available, otherwise recorded as unverified"
        ),
    )
    return parser


def _plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    entries, manifest = load_camera_manifest(args.camera_manifest)
    gpu_labels = GPU_LABELS if args.gpu_label is None else (args.gpu_label,)
    camera_counts = _parse_csv(args.camera_counts, cast=int)
    gaussian_sizes = _parse_csv(args.gaussian_sizes)
    render_modes = _parse_csv(args.render_modes)
    sh_modes = _parse_csv(args.sh_modes)
    pass_modes = _parse_csv(args.pass_modes)
    requested_axes = (
        ("camera counts", camera_counts, set(CAMERA_COUNTS)),
        ("Gaussian sizes", gaussian_sizes, {str(value) for value in GAUSSIAN_SIZES}),
        ("render modes", render_modes, set(RENDER_MODES)),
        ("SH modes", sh_modes, set(SH_MODES)),
        ("pass modes", pass_modes, set(PASS_MODES)),
    )
    for name, values, allowed in requested_axes:
        comparable = set(values) if name == "camera counts" else {str(value) for value in values}
        if not values or not comparable.issubset(allowed):
            raise ValueError(f"unsupported {name}: {values}; allowed={sorted(allowed, key=str)}")
    points = build_plan(
        entries,
        camera_counts=camera_counts,
        gaussian_sizes=gaussian_sizes,
        render_modes=render_modes,
        sh_modes=sh_modes,
        pass_modes=pass_modes,
        gpu_labels=gpu_labels,
    )
    declared_sha = args.code_sha or os.environ.get(
        "ELIGSIR_REPOSITORY_COMMIT", ""
    ).strip() or None
    source_provenance = resolve_source_provenance(_REPO_ROOT, declared_sha)
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "evidence_class": "UNMANAGED",
        "benchmark_kind": "kernel",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": sha256_file(args.checkpoint),
        },
        "camera_manifest": manifest,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "code_sha": source_provenance["code_sha"],
        "source_provenance": source_provenance,
        "points": points,
    }


def _skip_row(
    point: dict[str, Any],
    *,
    reason: str,
    common: dict[str, Any],
    realized_n: int | None = None,
) -> dict[str, Any]:
    row = {
        **common,
        **point,
        "schema_version": SCHEMA_VERSION,
        "status": "skip",
        "skip_reason": reason,
        "realized_n": realized_n,
        "useful_realized_cameras": 0,
        "samples_transaction_ms": [],
        "p50_transaction_ms": None,
        "p95_transaction_ms": None,
        "mean_transaction_ms": None,
        "ms_per_camera": None,
        "cameras_per_s": None,
        "peak_memory_allocated_bytes": None,
        "peak_memory_reserved_bytes": None,
    }
    row.update(profiler_evidence_unavailable("benchmark point did not execute"))
    return row


def build_execution_common_row(
    *,
    source_provenance: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    camera_manifest: dict[str, Any],
    hardware: dict[str, Any],
    software: dict[str, Any],
    rng_seed: int,
    warmups: int,
    repetitions: int,
    created_at_utc: str,
    container_hostname: str,
) -> dict[str, Any]:
    """Build execution-wide row fields without importing or invoking torch."""
    host_available = hardware.get("stable_host_identity_available")
    host_reason = hardware.get("stable_host_identity_reason")
    if not isinstance(host_available, bool):
        raise ValueError("hardware provenance lacks stable host availability")
    if host_available and (
        not hardware.get("host_machine_id")
        or not hardware.get("host_machine_id_source")
    ):
        raise ValueError("available stable host provenance lacks identity or source")
    if not host_available and not host_reason:
        raise ValueError("unavailable stable host provenance lacks a reason")
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "UNMANAGED",
        "evidence_scope": "engineering_kernel_benchmark",
        "quality_claims_permitted": False,
        "created_at_utc": str(created_at_utc),
        "code_sha": source_provenance["code_sha"],
        "source_provenance": source_provenance,
        "rng_seed": int(rng_seed),
        "warmups": int(warmups),
        "repetitions": int(repetitions),
        "checkpoint": {**checkpoint, **checkpoint_metadata},
        "camera_manifest": camera_manifest,
        "hardware": hardware,
        "software": software,
        "row_provenance": {
            "container_hostname": str(container_hostname),
            "stable_host_identity_available": host_available,
            "stable_host_identity_reason": host_reason,
        },
    }


def _execute(args: argparse.Namespace, plan_payload: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    if args.gpu_label is None:
        raise ValueError("--gpu-label is required for execution; one process measures one GPU")
    source_provenance = plan_payload["source_provenance"]
    if source_provenance["verification_status"] == "dirty_git_tree":
        raise ValueError(
            "refusing benchmark execution from a dirty Git tree; commit the exact "
            "source or run where Git metadata is unavailable and retain the explicit "
            "unverified comparison-ineligible provenance label"
        )
    if args.output_jsonl.exists():
        raise FileExistsError(f"refusing to overwrite benchmark output: {args.output_jsonl}")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    state, checkpoint_meta = _load_checkpoint_state(args.checkpoint)
    entries, manifest_meta = load_camera_manifest(args.camera_manifest)
    checkpoint_sha = plan_payload["checkpoint"]["sha256"]
    hardware, software = _hardware_software_provenance(args.gpu_label)
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    common = build_execution_common_row(
        source_provenance=source_provenance,
        checkpoint=plan_payload["checkpoint"],
        checkpoint_metadata=checkpoint_meta,
        camera_manifest=manifest_meta,
        hardware=hardware,
        software=software,
        rng_seed=args.seed,
        warmups=args.warmups,
        repetitions=args.repetitions,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        container_hostname=socket.gethostname(),
    )
    points = build_plan(
        entries,
        camera_counts=_parse_csv(args.camera_counts, cast=int),
        gaussian_sizes=_parse_csv(args.gaussian_sizes),
        render_modes=_parse_csv(args.render_modes),
        sh_modes=_parse_csv(args.sh_modes),
        pass_modes=_parse_csv(args.pass_modes),
        gpu_labels=(args.gpu_label,),
    )

    subset_cache: dict[int | str, tuple[Any, dict[str, Any]]] = {}
    subset_dir = args.output_jsonl.parent / f"{args.output_jsonl.stem}.subsets"
    for requested_n in dict.fromkeys(point["requested_n"] for point in points):
        try:
            indices = deterministic_subset_indices(
                checkpoint_meta["mature_n"], requested_n, seed=args.seed
            )
        except ValueError:
            continue
        subset_cache[requested_n] = (
            indices,
            _write_subset_manifest(
                indices,
                requested_n=requested_n,
                seed=args.seed,
                total_n=checkpoint_meta["mature_n"],
                output_dir=subset_dir,
            ),
        )

    rows: list[dict[str, Any]] = []
    for point in points:
        requested_n = point["requested_n"]
        if point["plan_status"] != "eligible":
            rows.append(_skip_row(point, reason=point["skip_reason"], common=common))
            continue
        if requested_n not in subset_cache:
            rows.append(
                _skip_row(
                    point,
                    reason="requested_n_exceeds_mature_population",
                    common=common,
                )
            )
            continue
        indices, subset_manifest = subset_cache[requested_n]
        production_degree = checkpoint_meta["production_sh_degree"]
        realized_degree = 0 if point["sh_mode"] == "0" else production_degree
        if point["sh_mode"] == "production" and production_degree == 0:
            rows.append(
                _skip_row(
                    point,
                    reason="production_sh_degree_duplicates_degree_zero",
                    common=common,
                    realized_n=len(indices),
                )
            )
            rows[-1]["realized_sh_degree"] = 0
            rows[-1]["gaussian_subset"] = subset_manifest
            continue
        selection = select_distinct_cameras(entries, point["requested_c"])
        camera_subset = {
            "identities": list(selection.identities),
            "sha256": hashlib.sha256(_canonical_json(list(selection.identities))).hexdigest(),
        }
        row = {
            **common,
            **point,
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "skip_reason": None,
            "realized_n": int(len(indices)),
            "realized_sh_degree": int(realized_degree),
            "camera_subset": camera_subset,
            "gaussian_subset": subset_manifest,
            "useful_realized_cameras": int(selection.realized_c),
        }
        try:
            cameras = _build_cameras(selection.cameras)
            measured = _measure_point(
                state=state,
                indices=indices,
                cameras=cameras,
                sh_degree=realized_degree,
                render_mode=point["render_mode"],
                pass_mode=point["pass_mode"],
                warmups=args.warmups,
                repetitions=args.repetitions,
            )
            row.update(measured)
            row.update(
                summarize_timings(
                    measured["samples_transaction_ms"],
                    realized_c=selection.realized_c,
                )
            )
            row.update(
                profiler_evidence_unavailable(
                    "no supported external kernel profiler/counter collector was configured; "
                    "transaction timing is from accelerator events and is not decomposed"
                )
            )
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "useful_realized_cameras": 0,
                    "samples_transaction_ms": [],
                    "p50_transaction_ms": None,
                    "p95_transaction_ms": None,
                    "mean_transaction_ms": None,
                    "ms_per_camera": None,
                    "cameras_per_s": None,
                    "peak_memory_allocated_bytes": None,
                    "peak_memory_reserved_bytes": None,
                }
            )
            row.update(profiler_evidence_unavailable("benchmark point failed"))
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        rows.append(row)
        torch.cuda.empty_cache()

    add_scaling_efficiency(rows)
    if sha256_file(args.checkpoint) != checkpoint_sha:
        raise RuntimeError("checkpoint changed during benchmark execution")
    if sha256_file(args.camera_manifest) != manifest_meta["sha256"]:
        raise RuntimeError("camera manifest changed during benchmark execution")
    if source_provenance["comparison_eligible"]:
        refreshed_source = resolve_source_provenance(
            _REPO_ROOT, source_provenance["resolved_git_head"]
        )
        if (
            refreshed_source["verification_status"] != "verified_clean_git_head"
            or refreshed_source["resolved_git_tree"]
            != source_provenance["resolved_git_tree"]
        ):
            raise RuntimeError("source tree changed during benchmark execution")
    with args.output_jsonl.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path = args.output_jsonl.with_suffix(".run.json")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "eligsir.true_camera_scaling.run.v1",
                "output_jsonl": str(args.output_jsonl.resolve()),
                "row_count": len(rows),
                "status_counts": {
                    status: sum(row["status"] == status for row in rows)
                    for status in ("ok", "skip", "error")
                },
                "common_provenance": common,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.warmups <= 100 or not 1 <= args.repetitions <= 1000:
        raise ValueError("warmups must be in [0,100] and repetitions in [1,1000]")
    payload = _plan_payload(args)
    if args.list_plan or args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    rows = _execute(args, payload)
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("ok", "skip", "error")
    }
    print(json.dumps({"output_jsonl": str(args.output_jsonl), "status_counts": counts}))
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"true_camera_scaling: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
