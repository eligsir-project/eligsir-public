#!/usr/bin/env python3
"""CLI entry point for the Wave 3 live-ingestion pipeline benchmark suite.

Runs one (or all) of the required benchmark arms (plan section 6.5,
deliverable #4) against the deterministic synthetic RGB-D source + fake SLAM
sidecar (``eligsir/io/sources/synthetic.py``), collects GPU-timed spans
and queue metrics via ``eligsir.runtime.profiling`` and writes a JSON
report via ``eligsir.experiments.report.write_experiment_report`` (falls
back to plain ``json.dump`` if that module isn't available yet -- see
``eligsir/runtime/profiling/benchmark_scenarios.py``'s module docstring
for why these sibling-subagent imports are best-effort at this stage).

Usage:
    python scripts/benchmark_live_pipeline.py --scenario two_fps_accepted
    python scripts/benchmark_live_pipeline.py --scenario all --output report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eligsir.experiments.hashing import content_hash
from eligsir.experiments.manifest import (
    ArmSpec,
    ExperimentManifest,
    HardwareInfo,
    ValidityInfo,
)
from eligsir.experiments.report import write_experiment_report
from eligsir.runtime.profiling.benchmark_scenarios import (
    SCENARIO_NAMES,
    ScenarioResult,
    build_scenario,
    run_scenario,
)
from eligsir.runtime.profiling.events import set_profiling_enabled
from eligsir.io.sources.synthetic import frozen_artifact_hashes


def _result_to_dict(result: ScenarioResult) -> dict:
    """``dataclasses.asdict()`` only serializes real fields, not the
    ``mapping_result_valid`` computed property, so add it explicitly --
    see ``ScenarioResult``'s docstring (review Fix 7) for the three
    validity dimensions this represents."""
    payload = dataclasses.asdict(result)
    payload["mapping_result_valid"] = result.mapping_result_valid
    return payload


def _repository_commit() -> str:
    """Read the current commit SHA without shelling out to `git` -- the
    `train` container image doesn't ship a git binary, but a checked-out
    repo's .git/HEAD (plus, for an unpacked ref, .git/<ref-path>) is enough
    to resolve it directly."""
    supplied = os.environ.get("ELIGSIR_REPOSITORY_COMMIT", "").strip()
    if supplied:
        return supplied
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1]
        ).decode().strip()
    except Exception:
        pass
    try:
        repo_root = Path(__file__).resolve().parents[1]
        git_path = repo_root / ".git"
        if git_path.is_file():
            # Worktree checkout: .git is a file pointing at the real
            # gitdir, which may live outside this container's mounted
            # volume -- if so, this raises and falls through to "unknown"
            # below rather than crashing.
            gitdir_line = git_path.read_text().strip()
            git_dir = Path(gitdir_line.split(":", 1)[1].strip())
        else:
            git_dir = git_path
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref_path = head.split(" ", 1)[1].strip()
        # A worktree's own gitdir only stores refs it has independently
        # checked out; shared branch refs live in the common gitdir
        # (recorded in `commondir`, relative to this worktree's gitdir).
        candidates = [git_dir / ref_path]
        commondir_file = git_dir / "commondir"
        if commondir_file.exists():
            common_dir = (git_dir / commondir_file.read_text().strip()).resolve()
            candidates.append(common_dir / ref_path)
            candidates.append(common_dir / "packed-refs")
        for candidate in candidates:
            if candidate.name == "packed-refs" and candidate.exists():
                for line in candidate.read_text().splitlines():
                    if line.endswith(f" {ref_path}"):
                        return line.split(" ", 1)[0]
                continue
            if candidate.exists():
                return candidate.read_text().strip()
        return "unknown"
    except Exception:
        return "unknown"


def _hardware_info() -> HardwareInfo:
    gpu = driver = "unknown"
    backend = "cuda"
    torch_version = "unknown"
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            hip_version = getattr(torch.version, "hip", None)
            if hip_version:
                backend = "rocm"
                driver = hip_version
            else:
                backend = "cuda"
                driver = getattr(torch.version, "cuda", None) or "unknown"
    except Exception:
        pass
    if driver == "unknown":
        # Best-effort fallback via a driver CLI -- not over-engineered, just
        # try the tool matching whichever backend we detected and accept a
        # missing/failing binary silently (containers may lack either).
        probe_cmd = ["rocm-smi", "--showdriverversion"] if backend == "rocm" else [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
        try:
            output = subprocess.check_output(probe_cmd, stderr=subprocess.DEVNULL, timeout=5).decode().strip()
            if output:
                driver = output.splitlines()[0].strip()
        except Exception:
            pass
    return HardwareInfo(gpu=gpu, driver=driver, torch=torch_version, backend=backend)


def _build_manifest(args: argparse.Namespace, results: list[ScenarioResult]) -> ExperimentManifest:
    """Level-A manifest: the synthetic source is fully deterministic and
    seeded, so its content IS the frozen source artifact -- hash the
    scenario names + seed + num_frames actually run as the source artifact
    hash (a real bag/camera run would hash the recorded frame manifest
    instead; see docs/experiments/FROZEN_ABLATION_RULE.md)."""
    artifact_hashes = frozen_artifact_hashes(seed=args.seed, num_frames=args.num_frames)
    # Per-scenario `mapping_result_valid` (see ScenarioResult docstring, review Fix 7)
    # already excludes EXPECTED faults (e.g. slam_failure's deliberately injected
    # SlamFailure) -- so an unexpected fault in any scenario invalidates the whole
    # manifest, but the slam_failure scenario firing its fault as designed does not.
    any_unexpected_fault = any(not r.mapping_result_valid for r in results)
    return ExperimentManifest(
        experiment_id="w3-01-live-pipeline-baseline",
        hypothesis="Baseline live-ingestion pipeline telemetry before Wave 3 architecture changes.",
        independent_variable="none (baseline measurement, not a comparison)",
        frozen_level="A",
        repository_commit=_repository_commit(),
        source_artifact_hash=artifact_hashes["source_artifact_hash"],
        pose_hash=artifact_hashes["pose_hash"],
        holdout_hash=content_hash({"holdout": "none"}),
        config_hash=content_hash(dataclasses.asdict(args) if dataclasses.is_dataclass(args) else vars(args)),
        seed_policy=f"fixed seed={args.seed}",
        hardware=_hardware_info(),
        validity=ValidityInfo(
            mapping_result_valid=(not any_unexpected_fault and all(r.production_services_used for r in results)),
            source_frames_dropped=0,
            map_reinitializations=0,
        ),
        arms=[ArmSpec(name="baseline", changed_fields=[])],
    )


def _write_report(args: argparse.Namespace, results: list[ScenarioResult], out_path: Path) -> None:
    manifest = _build_manifest(args, results)
    if manifest.repository_commit == "unknown":
        raise RuntimeError("ELIGSIR_REPOSITORY_COMMIT must provide a concrete commit SHA")
    if not manifest.validity.mapping_result_valid:
        raise RuntimeError(
            "refusing a production benchmark report: supply real admission, mutation, and optimization services"
        )
    payload = {"scenarios": [_result_to_dict(r) for r in results]}
    write_experiment_report(manifest, payload, out_path, strict=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIO_NAMES, "all"),
        default="all",
        help="Which benchmark arm to run (default: all).",
    )
    parser.add_argument("--num-frames", type=int, default=90, help="Frame budget for finite scenarios.")
    parser.add_argument("--seed", type=int, default=0, help="Synthetic source seed.")
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="Disable real sleeping (fast, non-wall-clock-accurate run, for smoke testing).",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("output/benchmarks/live_pipeline_report.json"),
        help="Output JSON report path (docs/experiments/LIVE_PIPELINE_BENCHMARK.md calls this "
        "--output; --out is accepted as a backward-compatible alias).",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Wall-clock duration cap for scenarios that would otherwise run until stopped "
        "explicitly. NOT YET IMPLEMENTED for early-stopping -- every scenario here has an "
        "intrinsically finite frame count (--num-frames) already, so this is accepted for "
        "forward CLI-contract compatibility with the docs but currently has no effect beyond "
        "a printed warning. See docs/experiments/LIVE_PIPELINE_BENCHMARK.md and review Fix 6.",
    )
    parser.add_argument(
        "--profiling",
        choices=("on", "off"),
        default="on",
        help="Enable/disable eligsir.runtime.profiling instrumentation for this process "
        "(default: on -- unlike production code paths, which default off via "
        "SPLAT_PROFILE_RUNTIME, this benchmark's whole point is to measure, so profiling-off "
        "must be opted into explicitly, not the other way around).",
    )
    args = parser.parse_args(argv)

    set_profiling_enabled(args.profiling == "on")
    if args.duration_s is not None:
        print(
            f"[benchmark_live_pipeline] WARNING: --duration-s={args.duration_s} was passed but "
            "is not yet implemented as an early-stopping mechanism -- every scenario here runs "
            "to its full --num-frames budget regardless. See docs/experiments/"
            "LIVE_PIPELINE_BENCHMARK.md and review Fix 6."
        )

    names = list(SCENARIO_NAMES) if args.scenario == "all" else [args.scenario]
    results: list[ScenarioResult] = []
    for name in names:
        spec = build_scenario(
            name,
            num_frames=args.num_frames,
            realtime=not args.no_realtime,
            seed=args.seed,
        )
        print(f"[benchmark_live_pipeline] running scenario={name!r} ({spec.description})")
        result = run_scenario(spec)
        print(
            f"[benchmark_live_pipeline] scenario={name!r} "
            f"frames_seen={result.frames_seen} frames_accepted={result.frames_accepted} "
            f"slam_failures={result.slam_failures} wall_seconds={result.wall_seconds:.3f}"
        )
        results.append(result)

    _write_report(args, results, args.output)
    print(f"[benchmark_live_pipeline] wrote report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
