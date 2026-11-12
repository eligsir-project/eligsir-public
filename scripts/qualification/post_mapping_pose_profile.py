#!/usr/bin/env python3
"""Fail-closed launcher contract for a bounded post-mapping pose-fit trace.

This is intentionally a *diagnostic* launcher.  It runs an immutable historical
mapper on the first contiguous source prefix, which preserves the ordinary
zero-origin warm-start recurrence without presenting it as a full trajectory or
a current-main comparison.  It writes only provenance, timing, and profiler
receipts; the train command disables final presentation work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


INPUT_SCHEMA = "eligsir.post-mapping-pose-profile-input/v1"
PROFILE_MODES = {"untraced", "trace"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _load_manifest(path: Path, expected_sha256: str, prefix_frames: int) -> list[dict[str, Any]]:
    if _sha256(path) != expected_sha256:
        raise ValueError("input manifest SHA-256 mismatch")
    payload = json.loads(path.read_text())
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError("unsupported input manifest schema")
    if payload.get("cache_owned") is not True:
        raise ValueError("diagnostic input must be cache-owned/staged")
    if payload.get("prefix_start_ordinal") != 0:
        raise ValueError("only a zero-origin contiguous prefix preserves warm start")
    identities = payload.get("source_identities")
    if not isinstance(identities, list) or len(identities) < prefix_frames:
        raise ValueError("input manifest does not contain the requested prefix")
    selected = identities[:prefix_frames]
    for ordinal, identity in enumerate(selected):
        if not isinstance(identity, dict) or identity.get("ordinal") != ordinal:
            raise ValueError("input prefix ordinals must be contiguous from zero")
        if not isinstance(identity.get("source_id"), str) or not identity["source_id"]:
            raise ValueError("input identity is missing source_id")
        for key in ("rgb_timestamp", "depth_timestamp", "rgb_rel", "depth_rel"):
            if not isinstance(identity.get(key), str) or not identity[key]:
                raise ValueError(f"input identity is missing {key}")
        for key in ("rgb_sha256", "depth_sha256", "payload_sha256"):
            value = identity.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"input identity is missing {key}")
    return selected


def _pinned_tum_associations(
    source_root: Path, scene_path: Path, prefix_frames: int
) -> list[dict[str, Any]]:
    """Read the staged prefix through the pinned mapper's actual TUM reader."""
    saved_modules = {
        name: sys.modules.pop(name)
        for name in tuple(sys.modules)
        if name == "scene" or name.startswith("scene.")
    }
    sys.path.insert(0, str(source_root))
    try:
        importlib.invalidate_caches()
        from scene.readers.tum import readTUMCameras

        _, _, associations = readTUMCameras(
            str(scene_path), max_frames=prefix_frames, sequence="freiburg3"
        )
        return associations
    finally:
        for name in tuple(sys.modules):
            if name == "scene" or name.startswith("scene."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path.pop(0)


def _validate_staged_scene(
    source_root: Path, scene_path: Path, identities: list[dict[str, Any]]
) -> list[tuple[str, str, str, str]]:
    """Bind every declared RGB-D payload and order to the mounted TUM scene."""
    if not scene_path.is_dir():
        raise ValueError("staged scene path is not a directory")
    for identity in identities:
        rgb_rel, depth_rel = Path(identity["rgb_rel"]), Path(identity["depth_rel"])
        if rgb_rel.is_absolute() or depth_rel.is_absolute() or ".." in rgb_rel.parts or ".." in depth_rel.parts:
            raise ValueError("staged payload path must remain inside the staged scene")
        rgb = scene_path / rgb_rel
        depth = scene_path / depth_rel
        if not rgb.is_file() or not depth.is_file():
            raise ValueError("staged scene is missing a declared RGB-D payload")
        rgb_bytes, depth_bytes = rgb.read_bytes(), depth.read_bytes()
        if hashlib.sha256(rgb_bytes).hexdigest() != identity["rgb_sha256"]:
            raise ValueError("staged RGB bytes do not match the declared manifest")
        if hashlib.sha256(depth_bytes).hexdigest() != identity["depth_sha256"]:
            raise ValueError("staged depth bytes do not match the declared manifest")
        if hashlib.sha256(rgb_bytes + depth_bytes).hexdigest() != identity["payload_sha256"]:
            raise ValueError("staged RGB-D payload does not match the declared manifest")
    associations = _pinned_tum_associations(source_root, scene_path, len(identities))
    if len(associations) != len(identities):
        raise ValueError("pinned TUM reader did not consume the declared prefix")
    order = []
    for identity, association in zip(identities, associations, strict=True):
        if (
            association["rgb_rel"] != identity["rgb_rel"]
            or association["depth_rel"] != identity["depth_rel"]
            or not math.isclose(
                float(association["rgb_t"]), float(identity["rgb_timestamp"]), abs_tol=1e-6
            )
            or not math.isclose(
                float(association["depth_t"]), float(identity["depth_timestamp"]), abs_tol=1e-6
            )
        ):
            raise ValueError("pinned TUM reader order does not match the declared manifest")
        order.append(
            (
                f"{association['rgb_t']:.6f}",
                association["rgb_rel"],
                f"{association['depth_t']:.6f}",
                association["depth_rel"],
            )
        )
    return order


def preflight(
    *, source_root: Path, source_commit: str, input_manifest: Path,
    input_manifest_sha256: str, prefix_frames: int, pose_steps: int,
    profile_mode: str, image_digest: str, scene_path: Path,
    source_verified_by_launcher: bool = False,
) -> dict[str, Any]:
    """Validate immutable source/input facts before any GPU process begins."""
    if len(source_commit) != 40:
        raise ValueError("source_commit must be a full immutable Git SHA")
    if not source_verified_by_launcher:
        if _git(source_root, "rev-parse", "HEAD") != source_commit:
            raise ValueError("source checkout commit mismatch")
        if _git(source_root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError("source checkout is not clean")
    if not (source_root / "train.py").is_file():
        raise ValueError("source checkout has no train.py")
    if prefix_frames < 1:
        raise ValueError("prefix_frames must be positive")
    if pose_steps != 50:
        raise ValueError("pose_steps must be exactly 50 for this same-work diagnostic")
    if profile_mode not in PROFILE_MODES:
        raise ValueError(f"profile_mode must be one of {sorted(PROFILE_MODES)}")
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        raise ValueError("image_digest must be an immutable sha256 reference")
    identities = _load_manifest(input_manifest, input_manifest_sha256, prefix_frames)
    association_order = _validate_staged_scene(source_root, scene_path, identities)
    return {
        "schema": "eligsir.post-mapping-pose-profile-receipt/v1",
        "diagnostic_only": True,
        "not_scientific_evidence": True,
        "source": {"checkout": str(source_root), "commit": source_commit},
        "image_digest": image_digest,
        "checkpoint_strategy": "zero_origin_contiguous_prefix",
        "input": {
            "manifest": str(input_manifest),
            "manifest_sha256": input_manifest_sha256,
            "scene_path": str(scene_path),
            "cache_owned": True,
            "source_identities": [identity["source_id"] for identity in identities],
            "payload_sha256": [identity["payload_sha256"] for identity in identities],
            "reader_association_order": association_order,
        },
        "fit": {"prefix_frames": prefix_frames, "pose_steps": pose_steps, "warm_start": "on"},
        "profiling": {
            "mode": profile_mode,
            "requires_native_launch_trace": profile_mode == "trace",
            "separate_from_uninstrumented_wall_time": True,
        },
    }


def train_command(receipt: dict[str, Any], *, model_path: str, scene_path: str) -> list[str]:
    """Construct the historical command without changing fit mathematics."""
    source = receipt["source"]
    fit = receipt["fit"]
    return [
        "python", str(Path(source["checkout"]) / "train.py"), "-s", scene_path, "-m", model_path,
        "--tum_sequence", "freiburg3", "--execution_trainer", "causal_mapping",
        "--tum_max_frames", str(fit["prefix_frames"]),
        "--streaming_frame_admission", "preselected",
        "--gpu_frame_prep", "--gpu_frame_resident",
        "--streaming_post_mapping_pose_align_steps", str(fit["pose_steps"]),
        "--streaming_post_mapping_pose_align_tier", "half",
        "--streaming_post_mapping_pose_align_warm_start", fit["warm_start"],
        "--streaming_post_mapping_pose_align_stride", "1",
        "--streaming_post_mapping_pose_align_max_frames", "0",
        "--streaming_post_mapping_pose_align_time_budget_s", "0",
        "--streaming_post_mapping_pose_align_propagate_holdout", "none",
        "--streaming_benchmark_mode", "--no-streaming_report_final",
        "--streaming_eval_hold", "0", "--disable_progress_bar",
    ]


def _write_receipt(output_dir: Path, receipt: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "post_mapping_pose_profile_receipt.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained qualification receipt: {path}")
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def _run_pinned_train(
    command: list[str],
    *,
    trace_path: Path | None,
    allow_cpu_trace: bool = False,
    fit_call_observer: Callable[[tuple[Any, ...], dict[str, Any]], None] | None = None,
) -> int:
    """Run old ``train.py`` and, when requested, trace the first real pose fit.

    The historical runtime does not expose a post-mapping pose profiler.  Its
    ordinary tick profiler ends before the terminal alignment pass, so patching
    only that existing diagnostic would attribute the wrong workload.  This
    local launcher replaces the pinned module's callable before ``train.py``
    imports it; no mapper code or fit arguments change.  ``allow_cpu_trace``
    exists only for the deterministic CPU composition contract.  ``main``
    never selects it: device diagnostics use the default native-event check
    and therefore fail closed without a real kernel-launch trace.
    """
    source_root = Path(command[1]).parent
    sys.path.insert(0, str(source_root))
    fitter_module = None
    original_fitter = None
    if trace_path is not None or fit_call_observer is not None:
        import torch
        import eligsir.utils.eval_pose_align as fitter_module

        traced = False
        try:
            original_fitter = fitter_module._fit_pose_delta
        except AttributeError as error:
            raise RuntimeError("pinned utils.eval_pose_align lacks _fit_pose_delta") from error

        def trace_first_fit(*args: Any, **kwargs: Any) -> Any:
            nonlocal traced
            if fit_call_observer is not None:
                fit_call_observer(args, kwargs)
            if trace_path is None:
                return original_fitter(*args, **kwargs)
            if traced:
                return original_fitter(*args, **kwargs)
            traced = True
            activities = [torch.profiler.ProfilerActivity.CPU]
            if not allow_cpu_trace and torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False) as profiler:
                result = original_fitter(*args, **kwargs)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace_path))
            events = [str(event.name) for event in profiler.events()]
            if not allow_cpu_trace and not any(
                name in {"hipLaunchKernel", "cudaLaunchKernel"} for name in events
            ):
                raise RuntimeError("native pose-fit kernel launch was not recorded")
            return result

        fitter_module._fit_pose_delta = trace_first_fit
    old_argv = sys.argv
    try:
        sys.argv = command[1:]
        runpy.run_path(command[1], run_name="__main__")
    except SystemExit as error:
        return int(error.code or 0)
    finally:
        sys.argv = old_argv
        if fitter_module is not None and original_fitter is not None:
            fitter_module._fit_pose_delta = original_fitter
        if sys.path and sys.path[0] == str(source_root):
            sys.path.pop(0)
    return 0


def _fit_receipt_row(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Capture CPU camera metadata only; never materialize device tensors."""
    cam = args[0]
    k = getattr(cam, "K", None)
    return {
        "camera_identity": next((str(getattr(cam, name)) for name in ("_streaming_source_id", "image_name", "uid") if getattr(cam, name, None) is not None), None),
        "height": getattr(cam, "image_height", None),
        "width": getattr(cam, "image_width", None),
        "K_metadata_source": "camera.K.shape" if hasattr(k, "shape") else "unavailable",
        "K_shape": list(k.shape) if hasattr(k, "shape") else None,
        "steps": int(args[7]),
        "warm_start": kwargs.get("init_delta6") is not None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--prefix-frames", type=int, required=True)
    parser.add_argument("--pose-steps", type=int, default=50)
    parser.add_argument("--profile-mode", choices=sorted(PROFILE_MODES), required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-path", default="/data/scene")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--source-verified-by-launcher", action="store_true",
        help="use only after the outer managed launcher verified the mounted source checkout",
    )
    args = parser.parse_args(argv)
    receipt = preflight(
        source_root=args.source_root, source_commit=args.source_commit,
        input_manifest=args.input_manifest, input_manifest_sha256=args.input_manifest_sha256,
        prefix_frames=args.prefix_frames, pose_steps=args.pose_steps,
        profile_mode=args.profile_mode, image_digest=args.image_digest,
        scene_path=Path(args.scene_path),
        source_verified_by_launcher=args.source_verified_by_launcher,
    )
    command = train_command(receipt, model_path=str(args.output_dir), scene_path=args.scene_path)
    receipt["train_command"] = command
    if args.preflight_only:
        receipt["result"] = "preflight_only"
        _write_receipt(args.output_dir, receipt)
        return 0
    trace_path = args.output_dir / "pose_fit_torch_profile.json" if args.profile_mode == "trace" else None
    fitted = []
    started = time.monotonic()
    exit_code = _run_pinned_train(
        command, trace_path=trace_path,
        fit_call_observer=lambda fit_args, fit_kwargs: fitted.append(_fit_receipt_row(fit_args, fit_kwargs)),
    )
    wall_key = "profiled_wall_time_s" if args.profile_mode == "trace" else "unprofiled_wall_time_s"
    receipt[wall_key] = time.monotonic() - started
    # The hook runs before each fitter invocation, so it proves only observed
    # calls.  It has no authoritative selection join and cannot turn camera
    # metadata into selected SourceIdentities.
    receipt["fit_observed"] = {
        "fit_call_count": len(fitted),
        "selected_source_count": None,
        "selected_source_identity_status": "unavailable_no_authoritative_selection_join",
        "cameras": fitted,
    }
    # This is written only after the historical launcher returns.  A raised
    # cancellation/error does not manufacture a completed-run receipt.
    receipt["run"] = {
        "launcher_returned": True,
        "exit_code": exit_code,
        "successful": exit_code == 0,
    }
    if args.profile_mode == "trace":
        if trace_path is None or not trace_path.is_file():
            raise SystemExit("native kernel launch trace missing")
        receipt["native_kernel_launch_trace"] = str(trace_path)
    _write_receipt(args.output_dir, receipt)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
