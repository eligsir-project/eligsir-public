"""Opt-in runtime evidence emission for the W3-02 fixed-topology A/B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eligsir.experiments.hashing import content_hash


# W3-17 (2026-08-10): pinned span definition for optimization_gpu_ms.
#
# docs/architecture/w3_02_completion.md's 2026-08-09 "round 3" investigation
# found the W3-02 throughput gate's entire 12.08%/17.84% "regression" across
# two prior rounds was not a real code-performance difference -- it was a
# measurement-window bug: the engine (treatment) arm's begin()/end() bracket
# and the inline (baseline) arm's begin()/end() bracket were NOT measuring
# the same logical span (the inline arm's `.end()` fired before its own MCMC
# noise injection and both `_sanitize_gaussian_params` hook calls; the engine
# arm's `.end()` fired after all of that). Fixed in commit `4b2e22e` by
# moving the inline arm's `.end()` call site (train_streaming.py) to match
# the engine arm's. That doc's own explicit recommendation, not yet acted on
# until this task: "any such gate must pin down the span definition... as
# part of its specification, since that is now proven to be the thing that
# actually moves the number."
#
# SPAN_DEFINITION_VERSION bumps whenever either arm's begin()/end() call
# sites (train_streaming.py) move relative to each other or to the render/
# MCMC/sanitize sequence they bracket. scripts/w3_02_level_d_report.py
# refuses to compute a throughput comparison across two evidence files with
# different versions -- exactly the class of silent apples-to-oranges bug
# that produced the false 12.08%/17.84% numbers.
SPAN_DEFINITION_VERSION = 2
SPAN_DEFINITION = (
    "begin(iteration) immediately before the per-step render dispatch; "
    "end() after render + backward + optimizer step + MCMC noise injection "
    "+ both _sanitize_gaussian_params hook calls ('after_optstep', "
    "'after_mcmc_noise') -- identical logical span on both the "
    "--streaming_optimization_engine ON (engine) and OFF (inline) arms. "
    "Version 1 (all evidence from before commit 4b2e22e) had the inline "
    "arm's end() firing before its own MCMC-noise/sanitize calls -- narrower "
    "than the engine arm's span by ~0.6ms/step -- and must never be compared "
    "against version 2 evidence."
)


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class W3_02OptimizationTimingRecorder:
    """Record complete fixed-topology step spans with CUDA/HIP events.

    Events are resolved once at finalization, never in the hot loop.  A run
    without a CUDA/HIP device writes no samples and is intentionally rejected
    by the evidence validator for a performance claim.
    """

    def __init__(self, torch_module) -> None:
        self._torch = torch_module
        self._pending: list[tuple[int, Any, Any]] = []
        self._open: tuple[int, Any] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._torch.cuda.is_available())

    def begin(self, iteration: int) -> None:
        if not self.enabled:
            return
        start = self._torch.cuda.Event(enable_timing=True)
        start.record()
        self._open = (iteration, start)

    def end(self) -> None:
        if self._open is None:
            return
        iteration, start = self._open
        self._open = None
        end = self._torch.cuda.Event(enable_timing=True)
        end.record()
        self._pending.append((iteration, start, end))

    def finalize(
        self,
        *,
        model_path: str | Path,
        manifest_path: str | Path,
        arm: str,
        git_commit: str,
        clean_checkout: bool,
        seed: int,
        checkpoint_path: str | Path,
        mutation_trace_path: str | Path,
        frozen_artifact_paths: dict[str, str | Path],
        mutation_trace_summary: dict[str, Any] | None = None,
    ) -> Path:
        manifest_file = Path(manifest_path)
        manifest = json.loads(manifest_file.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("W3-02 manifest must be a JSON object")
        if self._pending:
            self._torch.cuda.synchronize()
        rows = [
            {"iteration": iteration, "gpu_ms": float(start.elapsed_time(end))}
            for iteration, start, end in self._pending
        ]
        root = Path(model_path)
        artifacts = {
            name: _file_hash(root / name)
            for name in ("cfg_args", "reproduce.sh", "psnr_history.jsonl")
            if (root / name).is_file()
        }
        # Only real, numeric-iteration report dirs -- excludes debug variants
        # like "iter_2000_bootstrap_views" (degenerate_split=True, train-view
        # only; see CLAUDE.md's Run Contracts: never read as generalization),
        # which the naive glob used to crash on (found running a real W3-02
        # Level-D fr3 A/B: ValueError parsing "2000_bootstrap_views" as int).
        reports = sorted(
            (
                report for report in root.glob("comparison/iter_*/report.json")
                if report.parent.name.removeprefix("iter_").isdigit()
            ),
            key=lambda report: int(report.parent.name.removeprefix("iter_")),
        )
        if reports:
            artifacts["final_report"] = _file_hash(reports[-1])
            artifact_paths = {name: str(root / name) for name in ("cfg_args", "reproduce.sh", "psnr_history.jsonl")}
            artifact_paths["final_report"] = str(reports[-1])
        else:
            artifact_paths = {name: str(root / name) for name in ("cfg_args", "reproduce.sh", "psnr_history.jsonl")}
        frozen = {name: {"path": str(path), "sha256": _file_hash(path)} for name, path in frozen_artifact_paths.items()}
        payload = {
            "manifest_hash": content_hash(manifest),
            "arm": arm,
            "optimization_gpu_events": rows,
            "runtime": {
                "git_commit": git_commit,
                "span_definition_version": SPAN_DEFINITION_VERSION,
                "span_definition": SPAN_DEFINITION,
                "clean_checkout": bool(clean_checkout),
                "seed": int(seed),
                "initial_checkpoint_hash": _file_hash(checkpoint_path),
                "mutation_trace_hash": _file_hash(mutation_trace_path),
                # mutation_trace_summary comes from
                # eligsir/optimization/gaussian/mutation_trace.py's
                # MutationTraceRecorder.finalize(): for the baseline (record)
                # arm this is always False (nothing to verify against yet);
                # for the treatment (verify) arm it is True only if every
                # recorded baseline mutation event at the shared,
                # flag-independent call sites (depth-insertion flush, MCMC
                # step_post_backward) was matched exactly by this run, with
                # zero divergences and zero missing events -- see
                # MutationTraceRecorder.wrap()/finalize(). A caller that
                # doesn't pass mutation_trace_summary (i.e. isn't running
                # --w3_02_mutation_trace_mode) gets the same honest False the
                # salvage harness always wrote.
                "mutation_trace_applied": bool(
                    (mutation_trace_summary or {}).get("mutation_trace_applied", False)
                ),
                "mutation_trace_summary": mutation_trace_summary,
                # The container-writable JSON below is not a cryptographic
                # attestation: it is produced by the same untrusted training
                # process whose output it describes, with no host-side
                # signing key, no reproducible-build proof, and no protection
                # against a compromised or modified container image silently
                # writing a fabricated payload. It DOES give real, checkable
                # provenance -- git commit + a clean-checkout attestation flag
                # (both host-injected via --w3_02_git_commit/--w3_02_clean_checkout,
                # not computed inside the container) plus sha256 hashes of
                # every artifact this report depends on (see artifact_hashes/
                # frozen_artifacts below), so a reviewer can independently
                # re-hash the same files and catch post-hoc tampering or a
                # stale run directory. What it cannot do is prove the
                # container that produced these bytes actually ran the code
                # at that commit rather than some other code -- that would
                # require a signed host-side attestation (e.g. an out-of-band
                # build provenance system), which is out of scope for this
                # PR. Do not flip this to True without adding that mechanism.
                "trustworthy_attestation": False,
                "artifact_hashes": artifacts,
                "artifact_paths": artifact_paths,
                "frozen_artifacts": frozen,
            },
        }
        output = root / "w3_02_runtime_evidence.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return output
