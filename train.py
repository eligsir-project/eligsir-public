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

# Issue #524 (remaining item 3): dispatch --print-effective-config before torch import
# (Guard is written as one expression so the __main__ block the completion
# contract tests anchor on stays the real one at the bottom of this file.)
import sys as _sys
if __name__ == "__main__" and {"--print-effective-config", "--print_effective_config"} & set(_sys.argv):
    from eligsir.cli.effective_config import main as _pec_main
    raise SystemExit(_pec_main(_sys.argv[1:]))

import os
import json
import math
import shutil
import time
import warnings
import torch
from random import randint
from eligsir.utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render

import sys
from scene import Scene, GaussianModel, GsplatGaussianModel
from eligsir.utils.general_utils import safe_state
from eligsir.utils.git_metadata import get_git_metadata
import uuid
from tqdm import tqdm
from eligsir.utils.image_utils import psnr
from argparse import Namespace
from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    StreamingParams,
    apply_renderer_optimization_params,
)
from eligsir.cli.config import load_cli_config, load_config as _load_config
from eligsir.cli.parser import parse_training_args
from eligsir.utils.mcmc_schedule import MCMCScheduleConfig, get_mcmc_schedule
from eligsir.utils.energy_mcmc import (
    compute_effective_count_loss,
    compute_opacity_entropy_loss,
    compute_gaussian_utility,
)
from eligsir.utils.compiled_kernels import configure_torch_compile, set_compile_iteration
from eligsir.utils.score_contract import apply_declared_score_clamp_pair
from eligsir.utils.geometry_metrics import (
    update_visibility_ema,
    compute_geometry_dashboard,
)
from eligsir.utils.taming_3dgs import (
    compute_edge_map,
    compute_taming_scores,
    get_taming_budget,
    get_taming_count_array,
    get_taming_score_weights,
    sample_taming_cameras,
)
from eligsir.utils.strategies import make_mcmc_strategy
from eligsir.runtime.save_worker import AsyncSaveWorker
from eligsir.runtime.finalize import begin_offline_completion
from eligsir.io.snapshots import (
    capture_checkpoint as _capture_checkpoint,
    mutate_and_capture_iteration_outputs as _mutate_and_capture_iteration_outputs,
    write_ply_from_snapshot as _write_ply_from_snapshot,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

warnings.filterwarnings("once", message=".*HIPBLAS_STATUS_NOT_SUPPORTED.*")


def _load_legacy_web_viewer():
    """Load the legacy web viewer only when explicitly enabled."""
    try:
        from eligsir.legacy import web_viewer as web_viewer_mod
    except Exception:
        return None
    return web_viewer_mod


def _load_web_viewer_media():
    """Load optional web-viewer media helpers only for cache/video features."""
    try:
        from eligsir.legacy.web_viewer_media import (
            MultiCameraVideoRecorder,
            ScenePlyCacheWriter,
            parse_video_camera_indices,
        )
    except Exception:
        return None, None, None
    return MultiCameraVideoRecorder, ScenePlyCacheWriter, parse_video_camera_indices

class TeeStream:
    # Console-side batching interval/size when output staging is active --
    # matches PersistenceWorker's log-tailing cadence closely enough that
    # batching the console write doesn't meaningfully delay a real console
    # viewer, while sparing a network-backed `primary` from a tiny synchronous
    # write on every training-loop print(). The local `log_file` write below
    # is NEVER batched -- that's the file the persistence worker tails.
    _FLUSH_INTERVAL_S = 2.0
    _FLUSH_SIZE_BYTES = 64 * 1024

    def __init__(self, primary, log_file, staging_active=False):
        self.primary = primary
        self.log_file = log_file
        self.staging_active = staging_active
        self._buffer = []
        self._buffer_size = 0
        self._last_flush_time = time.monotonic()

    def write(self, text):
        self.log_file.write(text)
        if not self.staging_active:
            self.primary.write(text)
            return
        self._buffer.append(text)
        self._buffer_size += len(text)
        now = time.monotonic()
        if (
            self._buffer_size >= self._FLUSH_SIZE_BYTES
            or now - self._last_flush_time >= self._FLUSH_INTERVAL_S
        ):
            self._flush_buffer(now)

    def _flush_buffer(self, now):
        if self._buffer:
            self.primary.write("".join(self._buffer))
            self._buffer = []
            self._buffer_size = 0
        self._last_flush_time = now

    def flush(self):
        if self.staging_active:
            self._flush_buffer(time.monotonic())
        self.primary.flush()
        self.log_file.flush()


def _output_staging_env_override(args, flag_name, env_name):
    """Env wins over the argparse default, but an explicit CLI flag wins over
    env. argparse has already resolved args.<flag_name> to its default by the
    time this runs, so "not passed on the command line" is detected via
    sys.argv rather than via argparse itself (no existing precedent in this
    codebase for distinguishing explicit-vs-default post-parse).
    """
    env_value = os.environ.get(env_name)
    if not env_value:
        return
    positive = f"--{flag_name}"
    negative = f"--no-{flag_name}"
    prefix = f"{positive}="
    for token in sys.argv:
        if token == positive or token == negative or token.startswith(prefix):
            return
    setattr(args, flag_name, env_value)


def ensure_model_path(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    if getattr(args, "_output_staging_manager", None) is not None:
        # Already staged on this exact args object -- idempotent re-entry.
        # (Also skips re-running the preflight/GPU-snapshot below against
        # what is now the remapped LOCAL staging path rather than the
        # original persist target -- both only make sense to run once,
        # against the real user-specified path, below.)
        return

    # Mandatory pre-training preflight (issue #293): assert this run's
    # persistent output path is genuinely backed by a distinct, mounted
    # volume -- not merely writable -- BEFORE any training work happens.
    # Catches a missing/dropped bind mount (e.g. rtx3090's runner losing
    # its configured persistent output mount
    # entry) in well under a second, instead of after a full training
    # budget silently discards every artifact on container exit. Runs on
    # the raw, user-specified `args.model_path` -- BEFORE
    # OutputStagingManager below may remap it to a local staging path --
    # since that's the path this run's results are ultimately meant to
    # persist to. Scoped to known shared-mount prefixes only (default
    # configured mount prefixes; a no-op for
    # ordinary relative `output/...` paths.
    from eligsir.runtime.output_staging import assert_persistent_mount
    assert_persistent_mount(args.model_path)

    os.makedirs(args.model_path, exist_ok=True)

    # Validate and record Scene Replay Capsule metadata (issue #632 criterion 8).
    # Must happen after model_path exists but before any GPU work.
    # FAIL CLOSED: if --replay_capsule is passed, validation failure aborts the run.
    # Fallback literal must match arguments/__init__.py's real default (''),
    # not None (issue #368): tests/test_output_staging_wiring.py reaches
    # training() with a partial Namespace, so the fallback is genuinely
    # load-bearing here and cannot simply be dropped.
    if getattr(args, "replay_capsule", ""):
        from eligsir.experiments.capsule_run_metadata import (
            load_and_validate_capsule,
            write_replay_capsule_sidecar,
        )

        capsule, capsule_hash = load_and_validate_capsule(args.replay_capsule)
        if capsule is None or capsule_hash is None:
            raise RuntimeError(
                f"[capsule] FATAL: could not load/validate capsule from {args.replay_capsule}. "
                f"--replay_capsule was explicitly passed, so its absence or invalidity is fatal. "
                f"Check the path, file permissions, schema version, and validation errors above."
            )

        write_replay_capsule_sidecar(args.model_path, capsule, capsule_hash)
        print(
            f"[capsule] recorded metadata from {args.replay_capsule} "
            f"(id={capsule.capsule_id}, hash={capsule_hash[:8]}...)",
            flush=True,
        )

    # Best-effort GPU co-residency snapshot (issue #295 fix item 4): record
    # how many processes are resident on the GPU right now, so a run that
    # later turns out to have shared the card with another training
    # process is self-identifying from its own report.json instead of
    # requiring forensics. Never blocks/slows startup meaningfully and
    # never raises.
    try:
        from eligsir.utils.gpu_coresidency import record_start_snapshot
        record_start_snapshot(args.model_path)
    except Exception as e:
        print(f"[gpu-coresidency] start snapshot skipped: {e!r}", flush=True)

    from eligsir.runtime.output_staging import OutputStagingManager, PersistenceWorker

    _output_staging_env_override(args, "output_staging", "ELIGSIR_OUTPUT_STAGING")
    _output_staging_env_override(args, "output_staging_root", "ELIGSIR_OUTPUT_STAGING_ROOT")

    manager = OutputStagingManager(
        persist_model_path=args.model_path,
        mode=getattr(args, "output_staging", "auto"),
        staging_root=getattr(args, "output_staging_root", "") or None,
    )
    manager.setup()
    # Always set, even when staging is inactive (persist == local), so
    # downstream code has one consistent attribute to read regardless of mode.
    args.persist_model_path = manager.persist_path
    args.model_path = manager.local_path
    args._output_staging_manager = manager

    worker = PersistenceWorker(
        manager,
        log_sync_interval=float(getattr(args, "output_log_sync_interval", 2.0)),
        artifact_sync_interval=float(getattr(args, "output_sync_interval", 300.0)),
    )
    # Start as early as possible so train.log mirrors from its first line.
    worker.start()
    args._persistence_worker = worker


def start_output_log(args):
    if getattr(args, "_run_log_started", False):
        return
    ensure_model_path(args)
    log_path = os.path.join(args.model_path, "train.log")
    log_file = open(log_path, "a", buffering=1)
    staging_active = bool(getattr(args, "_output_staging_manager", None) and args._output_staging_manager.active)
    sys.stdout = TeeStream(sys.stdout, log_file, staging_active=staging_active)
    sys.stderr = TeeStream(sys.stderr, log_file, staging_active=staging_active)
    args._run_log_started = True
    args.run_log_path = log_path
    # Keep the file alive for the process lifetime.
    args._run_log_file = log_file


class OutputPersistenceError(RuntimeError):
    """Raised when final output-staging persistence to the canonical
    (network) path fails or times out -- the run must not report success
    (exit 0) while its results only exist in local scratch."""


def _finalize_output_staging(args, timeout=600.0):
    """Block until the persistence worker confirms the local staging dir is
    fully mirrored onto the canonical `persist_model_path`, then write the
    `.COMPLETE`/`.FAILED` marker. A no-op when staging was never active.

    `.COMPLETE` is written here, by the parent process, only once
    `.finalize()` has confirmed success -- never by the worker itself, so
    the authority to declare a run done lives in one place.
    """
    manager = getattr(args, "_output_staging_manager", None)
    if manager is None or not manager.active:
        return

    worker = args._persistence_worker
    persist_path = args.persist_model_path
    local_path = args.model_path

    if worker.finalize(timeout=timeout):
        open(os.path.join(persist_path, ".COMPLETE"), "w").close()
        worker.shutdown()
        if not getattr(args, "output_keep_local", False):
            try:
                shutil.rmtree(local_path)
            except Exception as e:
                print(
                    f"[output-staging] warning: failed to remove local scratch dir "
                    f"{local_path}: {e}",
                    flush=True,
                )
        return

    msg = (
        "final persistence to network storage timed out/failed; "
        f"local results preserved at {local_path}"
    )
    with open(os.path.join(persist_path, ".FAILED"), "w") as f:
        f.write(msg + "\n")
    print(f"[output-staging] FAILED: {msg}", flush=True)
    worker.shutdown()
    raise OutputPersistenceError(msg)


def public_namespace(args):
    return Namespace(**{k: v for k, v in vars(args).items() if not k.startswith("_")})


# Issue #228: every parallelism profile owns the SAME set of optimizer fields,
# so selecting a profile fully determines them. Before this table existed only
# "safe" assigned anything and "off" was a silent no-op, leaving the sparse
# argparse defaults (selective_adam / gsplat_sparse_grad=True, set
# unconditionally in arguments/__init__.py) in place -- i.e. "off" never
# actually turned the sparse path off.
#
# `sh_update_interval` is deliberately NOT in this table. It is owned by the
# separate #239/#246 mechanism below (an explicit --sh_update_interval wins over
# the "safe" profile's 16). Adding an `off` row for it -- i.e. "off" also
# selecting sh_update_interval=1 -- was option (a) in the closed PR #235's
# decision table and is rejected here: specs/issue106_inline_batch_fixed.json,
# specs/issue106_inline_batch_b1dense.json and specs/issue106_computematched_b1.json
# all pass `off` + both optimizer flags and NO --sh_update_interval, and their
# three completed arms all ran at 16, so an `off` row would silently flip a
# requeue of any of them from 16 to 1 and break their mutual comparability.
# Under "off" the field simply keeps its argparse default (16); see
# docs/architecture/2026-08-13-issue228-fix-impact.md.
PARALLELISM_PROFILES = {
    "safe": {
        "optimizer_type": "selective_adam",
        "gsplat_sparse_grad": True,
    },
    "off": {
        "optimizer_type": "adam",
        "gsplat_sparse_grad": False,
    },
}


def profile_explicit_dests(cli_explicit, config_keys):
    """Which dests outrank --parallelism_profile. Precedence: CLI > profile > config.

    A ``--config``/``--run_profile`` key beats the profile only when the profile
    itself came from a config too (same tier -- the more specific field wins).
    When the profile was named on the CLI it outranks every config key, matching
    the repo-wide "CLI flags win over config" rule: otherwise a config carrying
    ``optimizer_type: selective_adam`` (exactly what
    ``configs/profiles/defaults.reference.yml`` holds) would silently defeat an
    explicit ``--parallelism_profile off`` -- issue #228 surviving through the
    config door.
    """
    cli_explicit = set(cli_explicit or ())
    if "parallelism_profile" in cli_explicit:
        return cli_explicit
    return cli_explicit | set(config_keys or ())


def apply_parallelism_profile(args, explicit_dests=None):
    """Resolve the effective parallelism/optimizer settings for a run.

    ``explicit_dests`` (optional) is the set of argparse ``dest`` names the
    user actually typed on the command line (see
    ``eligsir.cli.run_config.cli_explicit_dests``, already computed by
    every real call site for CLI-vs-config precedence). When supplied, an
    explicit ``--sh_update_interval`` on the command line wins over the
    ``safe`` profile's default of 16 -- mirroring how ``--parallelism_profile
    off`` already leaves an explicit ``--sh_update_interval`` untouched (see
    issue #239).

    Issue #228 extends the same rule to ``optimizer_type`` /
    ``gsplat_sparse_grad``: both profiles now assign both fields (see
    ``PARALLELISM_PROFILES``), so ``--parallelism_profile off`` really does
    select dense Adam instead of silently keeping the sparse argparse
    defaults -- unless the user explicitly passed one of those flags, which
    still wins in both directions.

    When ``explicit_dests`` is omitted (``None``), no field is treated as
    explicit, so the profile wins outright -- the pre-#239 behavior for
    ``sh_update_interval``. Every real call site in this repo passes it.
    """
    profile = getattr(args, "parallelism_profile", "off").lower()
    if profile not in PARALLELISM_PROFILES:
        raise ValueError(
            f"Unsupported --parallelism_profile '{profile}'. Expected 'off' or 'safe'."
        )
    explicit = set(explicit_dests or ())
    for _key, _value in PARALLELISM_PROFILES[profile].items():
        if _key not in explicit:
            setattr(args, _key, _value)

    if profile == "safe":
        sh_explicit = explicit_dests is not None and "sh_update_interval" in explicit_dests
        if sh_explicit:
            # NOTE: this runs before start_output_log() installs the TeeStream
            # (both real call sites resolve the profile in their startup
            # prologue, ahead of ensure_model_path/start_output_log), so this
            # print reaches container stdout only, not yet <model_path>/train.log.
            # `args._sh_update_interval_override_warned` re-triggers the same
            # notice from inside training()'s existing post-tee "[parallelism]"
            # startup line, which IS captured in train.log -- that's the one to
            # grep. Kept here too as a belt-and-braces immediate signal.
            print(
                f"[parallelism_profile] WARNING: --sh_update_interval="
                f"{getattr(args, 'sh_update_interval', None)} was explicitly passed on the "
                f"command line; the 'safe' profile's default of 16 is NOT overriding it "
                f"(see issue #239). Since issue #228 the same explicit-wins rule "
                f"applies to optimizer_type/gsplat_sparse_grad as well.",
                flush=True,
            )
            args._sh_update_interval_override_warned = True
        else:
            args.sh_update_interval = 16

    args.optimizer_type = getattr(args, "optimizer_type", "adam").lower()
    if args.optimizer_type == "default":
        args.optimizer_type = "adam"
    if args.optimizer_type not in {"adam", "selective_adam"}:
        raise ValueError(
            f"Unsupported --optimizer_type '{args.optimizer_type}'. "
            "Expected 'adam' or 'selective_adam'."
        )

    args.sparse_policy = getattr(args, "sparse_policy", "active_set").lower()
    if args.sparse_policy != "active_set":
        raise ValueError(
            f"Unsupported --sparse_policy '{args.sparse_policy}'. Expected 'active_set'."
        )

    args.sh_impl = getattr(args, "sh_impl", "gsplat").lower()
    if args.sh_impl not in {"python", "gsplat"}:
        raise ValueError(
            f"Unsupported --sh_impl '{args.sh_impl}'. "
            "Expected 'python' or 'gsplat'."
        )

    args.model_layout = getattr(args, "model_layout", "gsplat").lower()
    if args.model_layout not in {"gsplat", "legacy"}:
        raise ValueError(
            f"Unsupported --model_layout '{args.model_layout}'. Expected 'gsplat' or 'legacy'."
        )
    args.densification_strategy = getattr(args, "densification_strategy", "gsplat_energy_mcmc").lower()
    gsplat_strategies = {"gsplat_mcmc", "gsplat_energy_mcmc", "gsplat_default", "gsplat_live_strategy"}
    if args.model_layout == "gsplat" and args.densification_strategy not in gsplat_strategies:
        raise ValueError(
            "--model_layout gsplat currently supports only --densification_strategy "
            "gsplat_mcmc, gsplat_energy_mcmc, or gsplat_default. "
            "Use --model_layout legacy for mcmc, hybrid, or taming."
        )
    if args.model_layout == "legacy" and args.densification_strategy in gsplat_strategies:
        raise ValueError(
            f"--densification_strategy {args.densification_strategy} requires --model_layout gsplat."
        )

    # Issue #1353: gsplat_default (vanilla ADC) requires dense gradients on the
    # 2D screen-space means tensor, which sparse-gradient mode doesn't populate.
    # gsplat's DefaultStrategy.clone/split heuristic needs .grad populated
    # post-backward; sparse gradients break this. Force dense Adam when selected.
    if args.densification_strategy == "gsplat_default":
        if "gsplat_sparse_grad" in explicit and bool(
            getattr(args, "gsplat_sparse_grad", True)
        ):
            raise ValueError(
                "--gsplat_sparse_grad is incompatible with "
                "--densification_strategy gsplat_default (gsplat's DefaultStrategy "
                "requires dense gradients on the 2D means tensor). Pass "
                "--no-gsplat_sparse_grad as well, or switch to "
                "--densification_strategy gsplat_energy_mcmc or gsplat_mcmc."
            )
        args.gsplat_sparse_grad = False
        # Also force dense Adam optimizer if not explicitly specified otherwise
        if "optimizer_type" not in explicit:
            args.optimizer_type = "adam"

    # selective_adam genuinely requires sparse gradients. Before issue #228
    # this silently flipped an explicit --no-gsplat_sparse_grad back to
    # True (so `--parallelism_profile off --no-gsplat_sparse_grad` looked
    # accepted but ran sparse); now that combination is a loud error
    # instead of a silent override.
    if args.optimizer_type == "selective_adam":
        if "gsplat_sparse_grad" in explicit and not bool(
            getattr(args, "gsplat_sparse_grad", True)
        ):
            raise ValueError(
                "--no-gsplat_sparse_grad is incompatible with "
                "--optimizer_type selective_adam (selective_adam assumes a "
                "sparse per-iteration visibility set). Pass --optimizer_type "
                "adam as well, or --parallelism_profile off (which selects "
                "both)."
            )
        # Issue #1353: gsplat_default's dense-gradient requirement (forced
        # above) and selective_adam's sparse-gradient requirement (forced
        # here) are mutually exclusive. Without this check, this branch's
        # unconditional `= True` silently undid the gsplat_default block's
        # `= False` for `--densification_strategy gsplat_default
        # --optimizer_type selective_adam` (sparse_grad left implicit),
        # reintroducing the exact crash this issue's guard exists to reject.
        if args.densification_strategy == "gsplat_default":
            raise ValueError(
                "--optimizer_type selective_adam is incompatible with "
                "--densification_strategy gsplat_default (gsplat_default "
                "requires dense gradients; selective_adam requires sparse "
                "ones). Pass --optimizer_type adam, or switch to "
                "--densification_strategy gsplat_energy_mcmc or gsplat_mcmc."
            )
        args.gsplat_sparse_grad = True

    args.sh_update_interval = max(1, int(getattr(args, "sh_update_interval", 1)))
    args.parallelism_profile = profile


def start_benchmark_log(args):
    benchmark_dir = getattr(args, "benchmark_dir", "")
    if not benchmark_dir:
        return
    if benchmark_dir in {"model", "model_path"}:
        benchmark_dir = args.model_path
    os.makedirs(benchmark_dir, exist_ok=True)
    args.benchmark_dir = benchmark_dir
    args._benchmark_file = open(
        os.path.join(benchmark_dir, "timings.jsonl"),
        "a",
        buffering=1,
    )


def begin_stage_timer(sync=False):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    stage_times = {}
    start = time.perf_counter()
    last = start

    def mark(name):
        nonlocal last
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        stage_times[name] = stage_times.get(name, 0.0) + now - last
        last = now

    def finish():
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        stage_times["total_wall"] = time.perf_counter() - start

    return stage_times, mark, finish


def log_stage_times(tb_writer, benchmark_file, iteration, stage_times, args, scene, densification_strategy, num_visible=0):
    # No getattr fallback: scalar_log_interval is an unconditional
    # OptimizationParams default, always present on the real parsed args
    # (issue #368 -- the old getattr(..., 10) fallback literal had drifted
    # from the real default of 100; removed rather than re-stated).
    scalar_log_interval = max(1, int(args.scalar_log_interval))
    if tb_writer and (iteration % scalar_log_interval == 0 or iteration == getattr(args, "iterations", iteration)):
        for name, seconds in stage_times.items():
            tb_writer.add_scalar(f"timing/{name}_ms", seconds * 1000.0, iteration)
        tb_writer.add_scalar("parallelism/gsplat_sparse_grad", int(getattr(args, "gsplat_sparse_grad", False)), iteration)
        tb_writer.add_scalar("parallelism/sh_update_interval", int(getattr(args, "sh_update_interval", 1)), iteration)
        tb_writer.add_scalar("parallelism/selective_adam", int(getattr(args, "optimizer_type", "adam") == "selective_adam"), iteration)

    if benchmark_file is not None:
        row = {
            "iteration": iteration,
            "model_path": getattr(args, "model_path", ""),
            "strategy": densification_strategy,
            "optimizer_type": getattr(args, "optimizer_type", "adam"),
            "parallelism_profile": getattr(args, "parallelism_profile", "off"),
            "gsplat_sparse_grad": bool(getattr(args, "gsplat_sparse_grad", False)),
            "sh_update_interval": int(getattr(args, "sh_update_interval", 1)),
            "active_sh_degree": int(scene.gaussians.active_sh_degree),
            "num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
            # No getattr fallback: get_git_metadata() unconditionally sets
            # args.git_branch/git_commit (to "unknown" on failure, same value
            # as this fallback literal) in __main__ before training() ever
            # runs, so the attribute always exists on the real, live args
            # object log_stage_times is called with (issue #368 triage).
            "git_branch": args.git_branch,
            "git_commit": args.git_commit,
            "timing_ms": {name: seconds * 1000.0 for name, seconds in stage_times.items()},
        }
        if num_visible > 0:
            row["num_visible"] = num_visible
            row["active_fraction"] = num_visible / max(row["num_gaussians"], 1)
        benchmark_file.write(json.dumps(row) + "\n")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sh_degree_schedule, run_args=None):
    if run_args is None:
        raise ValueError("training() requires run_args so feature flags are explicit and non-global.")
    args = run_args

    from eligsir.utils.experiment_origin import validate_researchflow_origin
    validate_researchflow_origin(args)

    if args.streaming_replay:
        raise SystemExit(
            "--streaming_replay is retired for production runs. Remove it and "
            "select --execution_trainer causal_mapping with a causal_mapping profile."
        )
    if (
        bool(getattr(args, "streaming_train_evwb_enabled", False))
        and args.execution_trainer != "causal_mapping"
    ):
        raise SystemExit(
            "--streaming_train_evwb_enabled requires --execution_trainer "
            "causal_mapping so its exact train-role reference can be validated."
        )
    if args.execution_trainer == "causal_mapping" and checkpoint:
        # issue #1552 bug 1: causal_entry.py has checkpoint-writing logic but
        # no checkpoint-reading/restore logic, so --start_checkpoint used to
        # be silently ignored here -- a resumed run cold-started from scratch
        # with no error or warning. Fail closed instead of doing the wrong
        # thing silently; real resume support for causal_mapping is a
        # separate feature (state, source cursor, replay-focus controller).
        raise SystemExit(
            "--start_checkpoint is not supported with --execution_trainer "
            "causal_mapping: causal_entry.py has no checkpoint-restore path, "
            "so this would silently cold-start instead of resuming. Remove "
            "--start_checkpoint, or use a non-causal_mapping trainer that "
            "supports it."
        )
    if args.execution_trainer == "causal_mapping":
        from eligsir.runtime.causal_entry import run_causal_mapping
        try:
            result = run_causal_mapping(dataset=dataset, opt=opt, pipe=pipe, args=args)
        except BaseException as training_error:
            # Issue #1278/#1284: causal_mapping crashes used to leave no
            # run_completion_state.json at all (that machine-readable
            # termination contract only ever got wired into the legacy
            # offline_reconstruction path below). Record the crash -- with
            # the FIRST/real exception, before any teardown code below that
            # could itself raise and mask it -- so a consumer never has to
            # reconstruct the root cause from train.log's tail.
            try:
                from eligsir.utils.run_completion_state import RunCompletionState

                RunCompletionState(args.model_path).mark_training_crashed(training_error)
            except Exception as state_error:
                print(
                    "[train] run_completion_state crash record FAILED (original "
                    f"error still propagating): {state_error}",
                    flush=True,
                )
            # Issue #1306: this must ALSO run on the exception path, not only
            # on success (it used to sit only after a successful return). A
            # crash inside run_causal_mapping (a real CUDA OOM has been
            # observed mid-refinement in production) previously skipped it
            # entirely -- the traceback that WAS written to the in-container
            # train.log (TeeStream mirrors stderr too) then depended on the
            # periodic PersistenceWorker background sync winning a race
            # against process exit to ever reach the persisted copy under the
            # shared mount, which it lost at least once. Forcing one last
            # sync here (including the crash traceback) before the container
            # can be torn down is exactly what the success path already did.
            #
            # `_finalize_output_staging` can itself raise
            # (`OutputPersistenceError`, e.g. the final sync timing out) --
            # a bare `finally:` would let THAT replace the original OOM
            # traceback, reintroducing the exact masking problem this fix
            # exists to close. Swallow (log only) a staging failure here so
            # the original exception always wins.
            try:
                _finalize_output_staging(args)
            except Exception as staging_error:
                print(
                    "[train] output-staging finalize FAILED while handling a "
                    f"crash (original error still propagating): {staging_error}",
                    flush=True,
                )
            raise
        _finalize_output_staging(args)
        # Causal mapping returns before the legacy offline report tail below.
        # Its application has already flush-saved the map.
        return result
    if bool(getattr(args, "system_telemetry", True)):
        from eligsir.telemetry.system_sampler import (
            start_system_telemetry,
            stop_system_telemetry,
            write_system_performance_report,
        )

        _system_sampler = start_system_telemetry(
            args.model_path,
            interval_s=float(getattr(args, "system_telemetry_interval_s", 1.0)),
        )
    else:
        _system_sampler = None
    try:
        return _train_offline(
            dataset,
            opt,
            pipe,
            testing_iterations,
            saving_iterations,
            checkpoint_iterations,
            checkpoint,
            debug_from,
            sh_degree_schedule,
            run_args,
        )
    finally:
        if _system_sampler is not None:
            stop_system_telemetry(_system_sampler)
            write_system_performance_report(args.model_path)

def _train_offline(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, sh_degree_schedule, run_args):
    """Legacy (non-``causal_mapping``) offline reconstruction training loop.

    Split out of ``training()`` (PR #1783 system-telemetry follow-up) purely
    so the caller can wrap this body's full lifetime in a
    start/stop-system-telemetry ``try/finally`` without reindenting the
    ~1400 existing lines below -- moved verbatim, same indentation, into a
    new top-level function whose own body sits at the same nesting depth
    this code already had as part of ``training()``.
    """
    args = run_args
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, run_args=run_args)
    model_layout = getattr(args, "model_layout", getattr(dataset, "model_layout", "gsplat")).lower()
    model_cls = GsplatGaussianModel if model_layout == "gsplat" else GaussianModel
    gaussians = model_cls(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    save_worker = AsyncSaveWorker()

    # Host resource monitor (SPLAT_RESOURCE_MONITOR=1, utils/resource_monitor.py).
    # Constructed only when enabled -- zero cost when off. See
    # train_streaming.py's identical wiring / that module's docstring for
    # how the container reaches the host's Glances instance.
    _resource_monitor = None
    if bool(os.environ.get("SPLAT_RESOURCE_MONITOR")):
        from eligsir.utils.resource_monitor import ResourceMonitor

        _resource_monitor = ResourceMonitor(args.model_path)
        _resource_monitor.start()

    import signal as _signal
    _emergency_state: dict = {}

    def _emergency_finalize_output_staging():
        # Shorter timeout than the normal-completion path (600s): this runs
        # inside a preemption grace window, not an unbounded budget. Never
        # raises -- os._exit below must fire regardless of outcome -- but
        # still writes the .COMPLETE/.FAILED marker so a gracefully-stopped
        # staged run isn't left local-only with no explanation.
        try:
            _finalize_output_staging(args, timeout=60.0)
        except OutputPersistenceError:
            pass
        except Exception as e:
            print(f"[train] output-staging finalize FAILED during emergency "
                  f"shutdown: {e}", flush=True)

    def _emergency_checkpoint_and_exit(sig, frame):
        # Mirrors train_streaming.py's SIGTERM handler: a preempted k8s pod
        # (kai-batch-low is the lowest KAI Scheduler priority and can be
        # preempted at any time) gets SIGTERM before terminationGracePeriodSeconds
        # expires and SIGKILL follows. Write a checkpoint synchronously (not
        # just enqueued to the async worker) so a resumable checkpoint lands
        # on durable storage inside the grace window.
        sig_name = _signal.Signals(sig).name
        st = _emergency_state
        if not st:
            print(f"\n[train] {sig_name} received before first checkpoint-eligible "
                  "iteration — nothing to checkpoint, exiting.", flush=True)
            _emergency_finalize_output_staging()
            os._exit(1)
        print(f"\n[train] {sig_name} received at iter={st.get('iteration')} — "
              "writing emergency checkpoint synchronously...", flush=True)
        try:
            state = _capture_checkpoint(st["gaussians"])
            chk = os.path.join(st["model_path"], f"chkpnt{st['iteration']}.pth")
            tmp = chk + ".tmp"
            torch.save((state, st["iteration"]), tmp)
            os.replace(tmp, chk)
            print(f"[train] emergency checkpoint written: {chk}", flush=True)
        except Exception as e:
            print(f"[train] emergency checkpoint FAILED: {e}", flush=True)
        try:
            save_worker.shutdown()
        except Exception:
            pass
        _emergency_finalize_output_staging()
        os._exit(143 if sig == _signal.SIGTERM else 130)

    _orig_sigterm = _signal.getsignal(_signal.SIGTERM)
    _signal.signal(_signal.SIGTERM, _emergency_checkpoint_and_exit)

    # Preload camera tensors to GPU to avoid per-render DeviceCopy overhead.
    from scene.cameras import prepare_camera_for_render
    for cam in scene.getTrainCameras():
        prepare_camera_for_render(cam, device="cuda")
    for cam in scene.getTestCameras():
        prepare_camera_for_render(cam, device="cuda")
    mcmc_cfg = MCMCScheduleConfig(
        start_iter=opt.densify_from_iter,
        stop_growth_iter=getattr(opt, "mcmc_stop_growth_iter", 12_000),
        stop_reloc_iter=opt.densify_until_iter,
        growth_factor_start=getattr(opt, "mcmc_growth_factor_start", 1.05),
        growth_factor_min=getattr(opt, "mcmc_growth_factor_min", 1.002),
        growth_factor_tau=getattr(opt, "mcmc_growth_factor_tau", 0.35),
        grow_interval_min=getattr(opt, "mcmc_grow_interval_min", 100),
        grow_interval_max=getattr(opt, "mcmc_grow_interval_max", 2000),
        grow_tau=getattr(opt, "mcmc_grow_tau", 0.35),
        relocate_interval_min=getattr(opt, "mcmc_relocate_interval_min", 50),
        relocate_interval_max=getattr(opt, "mcmc_relocate_interval_max", 500),
        relocate_tau=getattr(opt, "mcmc_relocate_tau", 0.65),
        cap_growth_power=getattr(opt, "mcmc_cap_growth_power", 2.0),
        cap_interval_strength=getattr(opt, "mcmc_cap_interval_strength", 4.0),
        cap_interval_power=getattr(opt, "mcmc_cap_interval_power", 2.0),
        cap_stop_ratio=getattr(opt, "mcmc_cap_stop_ratio", 0.98),
        dead_opacity_start=getattr(opt, "mcmc_dead_opacity_start", 0.003),
        dead_opacity_end=getattr(opt, "mcmc_dead_opacity_end", 0.010),
        dead_opacity_power=getattr(opt, "mcmc_dead_opacity_power", 1.5),
        use_target_deficit=getattr(opt, "mcmc_use_target_deficit", False),
        target_splat_end=getattr(opt, "mcmc_target_splat_end", 150000),
        target_q_start=getattr(opt, "mcmc_target_q_start", 0.05),
        target_q_end=getattr(opt, "mcmc_target_q_end", 0.85),
        target_tau=getattr(opt, "mcmc_target_tau", 0.45),
    )
    densification_strategy = getattr(opt, "densification_strategy", "gsplat_energy_mcmc").lower()
    valid_strategies = {
        "mcmc", "taming", "hybrid", "gsplat_mcmc", "gsplat_energy_mcmc", "gsplat_default",
        "gsplat_live_strategy",
    }
    if densification_strategy not in valid_strategies:
        raise ValueError(
            f"Unsupported --densification_strategy '{densification_strategy}'. "
            f"Expected one of: {sorted(valid_strategies)}"
        )
    if dataset.cap_max == -1 and not (
        densification_strategy == "taming" and getattr(args, "taming_budget", -1.0) > 0
    ):
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    apply_renderer_optimization_params(pipe, opt)
    mcmc_strategy = make_mcmc_strategy(densification_strategy, gaussians=gaussians, args=args)
    optimizer_type = getattr(opt, "optimizer_type", "adam").lower()
    sh_update_interval = max(1, int(getattr(opt, "sh_update_interval", 1)))
    benchmark_file = getattr(args, "_benchmark_file", None)
    benchmark_sync = benchmark_file is not None
    sh_override_warned = bool(getattr(run_args, "_sh_update_interval_override_warned", False))
    print(
        f"[parallelism] profile={getattr(opt, 'parallelism_profile', 'off')} "
        f"model_layout={model_layout} "
        f"optimizer={optimizer_type} sparse_grad={pipe.gsplat_sparse_grad} "
        f"sparse_policy={getattr(args, 'sparse_policy', 'active_set')} "
        f"sh_impl={getattr(pipe, 'sh_impl', 'gsplat')} "
        f"sh_update_interval={sh_update_interval}"
        + (
            " [EXPLICIT: this run's --sh_update_interval overrode the 'safe' "
            "profile default of 16 -- see issue #239]"
            if sh_override_warned
            else ""
        ),
        flush=True,
    )
    allow_dense_grads = bool(getattr(args, "selective_adam_allow_dense_grads", False))
    sparse_active_set = optimizer_type == "selective_adam" and not allow_dense_grads
    if sparse_active_set:
        print(
            "[sparse] active-set training enabled: sparse grads preserved, "
            "active-set regularizers, active-set energy losses, active-set MCMC noise",
            flush=True,
        )
    elif optimizer_type == "selective_adam" and allow_dense_grads:
        print("[sparse] selective_adam with dense grads fallback (allow_dense_grads=True)", flush=True)

    # Growth-aware compile activation: never compile while N is still changing.
    # MCMC stops growth at mcmc_stop_growth_iter; taming/hybrid at densify_until_iter.
    if getattr(args, "compile_mode", "off") != "off":
        if densification_strategy in {"mcmc", "gsplat_mcmc", "gsplat_energy_mcmc", "gsplat_live_strategy"}:
            growth_stop = int(getattr(opt, "mcmc_stop_growth_iter", 12000))
        else:
            growth_stop = int(getattr(opt, "densify_until_iter", 25000))
        margin = int(getattr(args, "compile_growth_margin", 500))
        old_after = int(getattr(args, "compile_after_iter", 0))
        args.compile_after_iter = max(old_after, growth_stop + margin)
        print(
            f"[compiled-kernels] growth-aware compile_after_iter={args.compile_after_iter} "
            f"(strategy={densification_strategy}, growth_stop={growth_stop}, margin={margin})",
            flush=True,
        )

    configure_torch_compile(args)
    from eligsir.utils.loss_utils import configure_training_ssim
    configure_training_ssim(args)
    taming_enabled = densification_strategy in {"taming", "hybrid"}
    use_energy_mcmc = args.energy_mcmc and densification_strategy in {
        "mcmc", "hybrid", "gsplat_energy_mcmc", "gsplat_live_strategy",
    }
    taming_weights = get_taming_score_weights(args) if taming_enabled else None
    taming_counts = None
    taming_densify_step = 0
    taming_stats_logged = False
    taming_edge_maps = {}
    strategy_log_interval = max(1, getattr(args, "strategy_log_interval", 500))
    mcmc_control_log_interval = max(1, getattr(args, "mcmc_control_log_interval", 500))

    def should_log_strategy(iteration):
        return iteration % strategy_log_interval == 0 or iteration == opt.iterations

    _resumed_from_iteration = None
    if checkpoint:
        from eligsir.utils.checkpoint_io import unpack_checkpoint_payload
        model_params, first_iter = unpack_checkpoint_payload(
            torch.load(checkpoint, weights_only=False)
        )
        gaussians.restore(model_params, opt)
        _resumed_from_iteration = first_iter
    mcmc_strategy.initialize_state(gaussians=gaussians, args=args)

    lpips_fn = None
    if opt.lambda_lpips > 0:
        from lpipsPyTorch import LPIPS
        lpips_fn = LPIPS(net_type='vgg').to("cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_disabled = getattr(args, "disable_progress_bar", False) or getattr(args, "quiet", False)
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", disable=progress_disabled)
    progress_log_last_iter = first_iter
    progress_log_last_time = time.perf_counter()
    first_iter += 1

    if taming_enabled:
        for cam in scene.getTrainCameras():
            taming_edge_maps[id(cam)] = compute_edge_map(cam.original_image).detach().cpu()

    # Persistent web viewer and media cache. The viewer may already be running
    # from __main__ so users can connect while scene loading is still in flight.
    _web_viewer_mod = _load_legacy_web_viewer() if (
        bool(args.web_viewer_enabled)
        and int(getattr(args, "web_viewer_port", 6010)) > 0
    ) or bool(getattr(args, "record_video", False)) else None
    web_viewer = _web_viewer_mod.get_web_viewer() if _web_viewer_mod is not None else None
    web_viewer_enabled = bool(args.web_viewer_enabled)
    web_viewer_port = int(getattr(args, "web_viewer_port", 6010))
    web_viewer_image_interval = max(1, getattr(args, "web_viewer_image_interval", 100))
    web_viewer_fixed_camera = bool(getattr(args, "web_viewer_fixed_camera", False))
    web_viewer_cache_dir = getattr(args, "web_viewer_cache_dir", "") or os.path.join(scene.model_path, "web_viewer_cache")
    web_viewer_backend = str(getattr(args, "web_viewer_backend", "process")).lower()
    if (not web_viewer_enabled) or web_viewer_port <= 0:
        web_viewer_backend = "off"
    if web_viewer_backend != "off" and _web_viewer_mod is not None and web_viewer is None:
        _train_cams = scene.getTrainCameras()
        web_viewer = _web_viewer_mod.start_web_viewer(
            port=web_viewer_port,
            host=getattr(args, "web_viewer_host", "0.0.0.0"),
            image_interval=web_viewer_image_interval,
            total_cams=len(_train_cams),
            viewer_cam_idx=0,
            backend=web_viewer_backend,
            cache_dir=web_viewer_cache_dir,
            model_path=scene.model_path,
        )
    elif web_viewer is not None:
        web_viewer.update_camera_info(len(scene.getTrainCameras()), viewer_cam_idx=0)
    elif web_viewer_backend != "off":
        print("[web-viewer] Disabled - web viewer module failed to import.", flush=True)

    # Optional ROS2/RVizSplat live Gaussian map publisher. It is deliberately
    # separate from the render-image web viewer and only active when requested.
    _viewer_name = getattr(args, "viewer", "none")
    _live_viewer = None
    if _viewer_name == "ros2_rviz":
        from eligsir.viewers import make_viewer
        _live_viewer = make_viewer(
            _viewer_name,
            topic=getattr(args, "viewer_rviz_topic", "/eligsir/gaussian_splats"),
            publish_interval=getattr(args, "viewer_rviz_publish_interval", 100),
            min_publish_period_s=getattr(args, "viewer_rviz_min_publish_period_s", 2.0),
            camera_publish_interval=getattr(args, "viewer_rviz_camera_interval", 1),
            publish_clock=getattr(args, "viewer_rviz_publish_clock", True),
            max_gaussians=getattr(args, "viewer_rviz_max_gaussians", 0),
            only_when_map_version_changed=getattr(args, "viewer_rviz_only_when_map_version_changed", False),
            world_frame=getattr(args, "viewer_rviz_world_frame", "map"),
            transport=args.viewer_rviz_transport,
            blob_topic=getattr(args, "viewer_rviz_blob_topic", "/eligsir/splat_map/blob_chunks"),
            tile_topic=getattr(args, "viewer_rviz_tile_topic", "/eligsir/splat_map/tile_chunks"),
            session_id=getattr(args, "viewer_rviz_session_id", "eligsir"),
            chunk_max_bytes=getattr(args, "viewer_rviz_chunk_max_bytes", 1048576),
            tile_size_m=getattr(args, "viewer_rviz_tile_size_m", 1.0),
        )

    scene_cache_writer = None
    scene_cache_interval = max(0, int(args.web_viewer_scene_cache_interval))
    scene_cache_keep = max(0, int(getattr(args, "web_viewer_scene_cache_keep", 3)))
    if web_viewer_enabled and scene_cache_interval > 0:
        _, ScenePlyCacheWriter, _ = _load_web_viewer_media()
        if ScenePlyCacheWriter is None:
            print("[web-viewer-cache] Disabled - media helpers failed to import.", flush=True)
        else:
            scene_cache_writer = ScenePlyCacheWriter(web_viewer_cache_dir, keep=scene_cache_keep, viewer=web_viewer)

    video_recorder = None
    video_camera_indices = []
    if bool(getattr(args, "record_video", False)) and _web_viewer_mod is not None:
        MultiCameraVideoRecorder, _, parse_video_camera_indices = _load_web_viewer_media()
        if MultiCameraVideoRecorder is None or parse_video_camera_indices is None:
            print("[web-viewer-video] Disabled - media helpers failed to import.", flush=True)
        else:
            video_camera_indices = parse_video_camera_indices(
                getattr(args, "record_video_cameras", ""),
                len(scene.getTrainCameras()),
            )
            if video_camera_indices:
                video_recorder = MultiCameraVideoRecorder(
                    cache_dir=web_viewer_cache_dir,
                    camera_indices=video_camera_indices,
                    fps=max(1, int(getattr(args, "record_video_fps", 30))),
                    crf=max(0, int(getattr(args, "record_video_crf", 23))),
                    preset=str(getattr(args, "record_video_preset", "veryfast")),
                    viewer=web_viewer,
                )
                print(
                    f"[web-viewer-video] recording cameras {video_camera_indices} "
                    f"every {max(1, int(getattr(args, 'record_video_interval', 100)))} iterations",
                    flush=True,
                )
    elif bool(getattr(args, "record_video", False)):
        print("[web-viewer-video] Disabled - web viewer image helpers failed to import.", flush=True)

    _profile_path = getattr(run_args, "profile", None) if run_args else None
    if _profile_path:
        _prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=200, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        _prof.__enter__()
        print(f"[profile] Profiling enabled → {_profile_path}", flush=True)
    else:
        _prof = None

    # Persistent CUDA stream for async GPU→CPU viewer image copies.
    # Keeps the copy from blocking the main training stream.
    _viewer_stream = torch.cuda.Stream() if web_viewer is not None else None
    _video_stream = torch.cuda.Stream() if video_recorder is not None else None

    _progress_video_interval = max(0, int(args.progress_video_interval))
    _progress_video_fps = int(getattr(args, "progress_video_fps", 10))
    _progress_cams: list = []     # 3 equally-spaced fixed cameras (locked at first trigger)
    _progress_frames: list = []   # accumulated side-by-side uint8 HWC numpy frames

    import threading as _threading
    import queue as _queue_mod
    _pv_stream = torch.cuda.Stream()
    _pv_cpu_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=2)

    def _pv_cpu_worker():
        import numpy as _np2
        while True:
            item = _pv_cpu_queue.get()
            if item is None:
                break
            _gpu_panels, _pv_iter, _pv_total = item
            _panels = []
            for _t in _gpu_panels:
                _arr = _t.cpu().numpy()
                _arr = _np2.transpose(_arr, (1, 2, 0))
                _arr = (_arr * 255 + 0.5).astype(_np2.uint8)
                _panels.append(_arr)
            _sep = _np2.ones((_panels[0].shape[0], 2, 3), dtype=_np2.uint8) * 80
            _combined = _np2.concatenate([_panels[0], _sep, _panels[1], _sep, _panels[2]], axis=1)
            try:
                import cv2 as _cv2
                _lbl = f"iter {_pv_iter:>6d} / {_pv_total}"
                _cv2.putText(_combined, _lbl, (8, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, _cv2.LINE_AA)
                _cv2.putText(_combined, _lbl, (8, 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, _cv2.LINE_AA)
            except Exception:
                pass
            _progress_frames.append(_combined)

    _pv_thread = _threading.Thread(target=_pv_cpu_worker, daemon=True)
    _pv_thread.start()

    for iteration in range(first_iter, opt.iterations + 1):
        _emergency_state["gaussians"] = gaussians
        _emergency_state["iteration"] = iteration
        _emergency_state["model_path"] = scene.model_path
        set_compile_iteration(iteration)
        if _resource_monitor is not None:
            _resource_monitor.set_iteration(iteration)
        stage_times, mark_stage, finish_stage = begin_stage_timer(sync=benchmark_sync)

        iter_start.record()

        xyz_lr = gaussians.update_learning_rate(iteration)

        # Increase SH degree at configured schedule (default: 1000, 2000, 3000)
        if iteration in sh_degree_schedule:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        update_sh_rest = (
            gaussians.active_sh_degree == 0
            or sh_update_interval <= 1
            or iteration % sh_update_interval == 0
        )
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            update_sh_rest=update_sh_rest,
        )
        image = render_pkg["render"]
        # Cache a copy on CPU for the live web viewer.
        # Every image_interval iterations, render from the viewer's chosen camera
        # so the view stays stable (the image shows the same camera every time).
        # Uses a dedicated CUDA stream + pinned memory for async GPU→CPU copy so
        # the main training stream is not blocked by the transfer.
        _viewer_image_arr = None
        _viewer_render_time_ms = 0.0
        _viewer_pinned_buf = None  # non-None when async copy is in-flight
        _video_frames = []
        if web_viewer is not None and web_viewer.image_interval > 0 and iteration % web_viewer_image_interval == 0:
            _viewer_cam = scene.getTrainCameras()[web_viewer.viewer_cam_idx]
            if viewpoint_cam is _viewer_cam or not web_viewer_fixed_camera:
                # Training happened to render from the viewer's camera — reuse it.
                # By default the viewer shows the current training-camera render
                # to avoid a second rasterization pass during training.
                # Record a CUDA event so the viewer stream waits for the render to complete.
                _render_ready = torch.cuda.Event()
                _render_ready.record()
                _viewer_pinned_buf = _web_viewer_mod.encode_render_image_async_start(
                    image, _viewer_stream, wait_event=_render_ready,
                )
            else:
                # Do an extra render pass from the viewer's fixed camera
                _viewer_render_start = torch.cuda.Event(enable_timing=True)
                _viewer_render_end = torch.cuda.Event(enable_timing=True)
                _viewer_render_start.record()
                with torch.no_grad():
                    _viewer_pkg = render(_viewer_cam, gaussians, pipe, bg, update_sh_rest=update_sh_rest)
                    # Record event on default stream after render is queued
                    _render_ready = torch.cuda.Event()
                    _render_ready.record()
                    # Start async GPU→CPU copy on viewer stream (waits for render first)
                    _viewer_pinned_buf = _web_viewer_mod.encode_render_image_async_start(
                        _viewer_pkg["render"], _viewer_stream, wait_event=_render_ready,
                    )
                _viewer_render_end.record()
                _viewer_render_end.synchronize()
                _viewer_render_time_ms = _viewer_render_start.elapsed_time(_viewer_render_end)
        if video_recorder is not None and iteration % max(1, int(getattr(args, "record_video_interval", 100))) == 0:
            for _video_cam_idx in video_camera_indices:
                _video_cam = scene.getTrainCameras()[_video_cam_idx]
                if viewpoint_cam is _video_cam:
                    _render_ready = torch.cuda.Event()
                    _render_ready.record()
                    _video_buf = _web_viewer_mod.encode_render_image_async_start(
                        image, _video_stream, wait_event=_render_ready, reuse_buffer=False,
                    )
                else:
                    with torch.no_grad():
                        _video_pkg = render(_video_cam, gaussians, pipe, bg, update_sh_rest=update_sh_rest)
                        _render_ready = torch.cuda.Event()
                        _render_ready.record()
                        _video_buf = _web_viewer_mod.encode_render_image_async_start(
                            _video_pkg["render"], _video_stream, wait_event=_render_ready, reuse_buffer=False,
                        )
                _video_stream.synchronize()
                _video_frames.append((_video_cam_idx, _web_viewer_mod.encode_render_image_finish(_video_buf).copy()))
        mark_stage("forward")

        if _progress_video_interval > 0 and iteration % _progress_video_interval == 0:
            if not _progress_cams:
                _all_cams = scene.getTrainCameras()
                if len(_all_cams) >= 3:
                    _n = len(_all_cams)
                    _progress_cams = [_all_cams[0], _all_cams[(_n - 1) // 2], _all_cams[_n - 1]]
            if _progress_cams:
                _gpu_panels = []
                with torch.cuda.stream(_pv_stream):
                    _pv_stream.wait_stream(torch.cuda.current_stream())
                    with torch.no_grad():
                        for _pvc in _progress_cams:
                            _pv_pkg = render(_pvc, gaussians, pipe, background)
                            _gpu_panels.append(_pv_pkg["render"].detach().clamp(0, 1))
                try:
                    _pv_cpu_queue.put_nowait((_gpu_panels, iteration, opt.iterations))
                except _queue_mod.Full:
                    pass

        # Loss — capture intermediate values for logging (no recompute)
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        _ssim_val = ssim(image, gt_image)  # capture for logging (already computed for loss)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - _ssim_val)

        if lpips_fn is not None and iteration % opt.lpips_interval == 0:
            # LPIPS expects [B, 3, H, W] in [0, 1]
            # render and gt_image are [3, H, W]
            l_lpips = lpips_fn(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = loss + opt.lambda_lpips * l_lpips

        # Phase 5: differentiable energy loss flag
        # When energy_loss_enabled=True, replace the active_reg + entropy + eff_count
        # call sites with compute_differentiable_loss. Default False keeps legacy path.
        _energy_loss_enabled = getattr(args, "energy_loss_enabled", False)
        if _energy_loss_enabled:
            from eligsir.utils.differentiable_energy_loss import compute_differentiable_loss
            _active_mask_p5 = render_pkg["visibility_filter"].detach() if sparse_active_set else None
            _diff_loss_result = compute_differentiable_loss(
                gaussians=gaussians,
                iteration=iteration,
                cap_max=args.cap_max,
                active_mask=_active_mask_p5,
                include_effective_count=use_energy_mcmc,
                w_opacity_reg=args.opacity_reg,
                w_scale_reg=args.scale_reg,
                w_opacity_entropy=args.lambda_opacity_entropy,
                w_effective_count=args.lambda_eff_count,
                w_anisotropy_reg=float(args.anisotropy_reg),
                anisotropy_soft_limit=float(getattr(args, "anisotropy_soft_limit", 10.0)),
                w_needle_reg=float(getattr(args, "needle_reg_weight", 0.0)),
                needle_threshold=float(getattr(args, "needle_reg_threshold", 0.3)),
                scale_reg_mode=str(getattr(args, "scale_reg_mode", "l1")),
                max_iterations=opt.iterations,
            )
            loss = loss + _diff_loss_result.total
            # Gate on scalar_log_interval like the rest of the per-iteration TB
            # scalars (see should_log_scalars below): to_tensorboard_dict() calls
            # .item() on 7 tensors (forcing 7 GPU syncs) and then fires 7
            # add_scalar()s, unconditionally on every iteration when
            # energy_loss_enabled=True. Un-gated, that's both a sync-point stall
            # and the same per-call protobuf-serialization cost fixed elsewhere.
            # No getattr fallback: scalar_log_interval is an unconditional
            # OptimizationParams default, always present on the real parsed
            # args (issue #368 -- old getattr(..., 10) fallback drifted from
            # the real default of 100; removed rather than re-stated).
            _energy_tb_interval = max(1, int(args.scalar_log_interval))
            if tb_writer and (iteration % _energy_tb_interval == 0 or iteration == opt.iterations):
                for _k, _v in _diff_loss_result.to_tensorboard_dict().items():
                    tb_writer.add_scalar(_k, _v, iteration)
            # Placeholders for logging compatibility (energy path doesn't use these separately)
            L_eff = None
            L_entropy = None
            N_eff = None
            N_target = None
        else:
            # Active-set regularization: in sparse mode, only regularize visible Gaussians.
            # This avoids dense all-Gaussian gradient traffic that defeats sparse training.
            from eligsir.utils.compiled_kernels import active_reg_core
            # Birth-local opacity_reg exemption: exclude Gaussians younger than
            # streaming_young_reg_exempt_iters from the regularizer. Local version
            # of global retention relaxation — avoids the E2c failure mode.
            _young_exempt_iters = int(args.streaming_young_reg_exempt_iters)
            _young_exempt_mask: "torch.Tensor | None" = None
            if _young_exempt_iters > 0:
                _birth_iter_t = getattr(gaussians, "birth_iter", None)
                if _birth_iter_t is not None and _birth_iter_t.shape[0] == gaussians.get_opacity.shape[0]:
                    _young_exempt_mask = (iteration - _birth_iter_t.to(dtype=torch.int64)) < _young_exempt_iters

            if sparse_active_set:
                _active_reg = render_pkg["visibility_filter"].detach()
                if _young_exempt_mask is not None:
                    _active_reg = _active_reg & ~_young_exempt_mask
                _reg_loss = active_reg_core(
                    gaussians.get_opacity[_active_reg],
                    gaussians.get_scaling[_active_reg],
                    w_opacity=args.opacity_reg,
                    w_scale=args.scale_reg,
                )
            else:
                if _young_exempt_mask is not None:
                    _old_mask = ~_young_exempt_mask
                    _reg_loss = active_reg_core(
                        gaussians.get_opacity[_old_mask],
                        gaussians.get_scaling[_old_mask],
                        w_opacity=args.opacity_reg,
                        w_scale=args.scale_reg,
                    )
                else:
                    _reg_loss = active_reg_core(
                        gaussians.get_opacity,
                        gaussians.get_scaling,
                        w_opacity=args.opacity_reg,
                        w_scale=args.scale_reg,
                    )
            loss = loss + _reg_loss

            # Energy-guided MCMC losses (Stage A: soft loss steering)
            # Backward compatibility: only active when --energy_mcmc is enabled (default)
            if use_energy_mcmc:
                # In sparse mode, run differentiable energy losses only periodically as
                # global maintenance steps. Normal sparse steps skip them to preserve
                # the active-set profile (dense global opacity gradients defeat the
                # selective-optimizer benefit). When they do run, they operate on the
                # FULL opacity tensor (dense, all Gaussians) since they are global
                # population-control signals, not per-view losses.
                _energy_global_interval = getattr(args, "sparse_energy_global_interval", 500)
                _run_energy = True
                if sparse_active_set and _energy_global_interval > 0:
                    _run_energy = iteration % _energy_global_interval == 0

                if _run_energy:
                    # Effective count loss: steer toward target population curve
                    L_eff, N_eff, N_target = compute_effective_count_loss(
                        gaussians._opacity,  # raw, full (global maintenance)
                        iteration=iteration,
                        cap_max=args.cap_max,
                        max_iterations=opt.iterations,
                        target_splat_end=args.mcmc_target_splat_end,
                        q_start=getattr(args, "mcmc_target_q_start", 0.05),
                        q_end=getattr(args, "mcmc_target_q_end", 0.85),
                        tau_N=getattr(args, "mcmc_target_tau", 0.45),
                    )
                    loss = loss + args.lambda_eff_count * L_eff

                    # Opacity entropy loss: encourage decisive alive/dead opacities
                    L_entropy = compute_opacity_entropy_loss(gaussians._opacity)  # raw, full
                    loss = loss + args.lambda_opacity_entropy * L_entropy
                else:
                    # Normal sparse step: skip differentiable energy losses
                    L_eff = None
                    L_entropy = None
                    N_eff = None
                    N_target = None
            else:
                L_eff = None
                L_entropy = None
                N_eff = None
                N_target = None

        # Per-iteration TB logging of loss components (captured values, no recompute)
        # No getattr fallback: scalar_log_interval is an unconditional
        # OptimizationParams default, always present on the real parsed args
        # (issue #368 -- old getattr(..., 10) fallback drifted from the real
        # default of 100; removed rather than re-stated).
        scalar_log_interval = max(1, int(args.scalar_log_interval))
        should_log_scalars = tb_writer and (iteration % scalar_log_interval == 0 or iteration == opt.iterations)
        if should_log_scalars:
            tb_writer.add_scalar('train_loss_patches/ssim_loss', (1.0 - _ssim_val).item(), iteration)
            # Recompute separate reg terms for logging (no graph cost — not in backward path)
            with torch.no_grad():
                if sparse_active_set:
                    _active_reg = render_pkg["visibility_filter"].detach()
                    _opacity_reg = args.opacity_reg * torch.abs(gaussians.get_opacity[_active_reg]).mean()
                    _scale_reg = args.scale_reg * torch.abs(gaussians.get_scaling[_active_reg]).mean()
                else:
                    _opacity_reg = args.opacity_reg * torch.abs(gaussians.get_opacity).mean()
                    _scale_reg = args.scale_reg * torch.abs(gaussians.get_scaling).mean()
            tb_writer.add_scalar('train_loss_patches/opacity_reg_term', _opacity_reg.item(), iteration)
            tb_writer.add_scalar('train_loss_patches/scale_reg_term', _scale_reg.item(), iteration)
            if lpips_fn is not None:
                tb_writer.add_scalar('train_loss_patches/lpips_loss', l_lpips.item(), iteration)
            if use_energy_mcmc and L_eff is not None:
                tb_writer.add_scalar('train_loss_patches/eff_count_loss', args.lambda_eff_count * L_eff.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/opacity_entropy_loss', args.lambda_opacity_entropy * L_entropy.item(), iteration)
                tb_writer.add_scalar('train_count/effective_count_N', N_eff.item(), iteration)
                tb_writer.add_scalar('train_count/target_count_N', int(N_target), iteration)

        mark_stage("loss")
        # Non-finite loss guard: a single NaN/Inf loss would otherwise propagate
        # through backward + optimizer.step and corrupt the model irrecoverably.
        # Mirrors train_streaming.py's guard (see its "Non-finite loss guard"
        # comment) — offline training previously had no equivalent protection
        # (found via the #164 operation inventory, docs/architecture/
        # 2026-08-12-issue164-operation-inventory.md).
        _loss_finite = bool(torch.isfinite(loss))
        if _loss_finite:
            mcmc_strategy.step_pre_backward(
                gaussians=gaussians,
                args=args,
                iteration=iteration,
                render_pkg=render_pkg,
                loss=loss,
            )
            loss.backward()
            # Sparse safety: zero invisible rows in strided grads so SelectiveAdam
            # never sees stale non-zero values for invisible Gaussians.
            # (Active-set regularizers / energy losses produce strided grads;
            #  PyTorch autograd converts sparse + strided → strided in accumulation.)
            if sparse_active_set and getattr(args, "selective_adam_zero_invisible_grads", True):
                _mask_grad = render_pkg["visibility_filter"].detach()
                for group in gaussians.optimizer.param_groups:
                    p = group["params"][0]
                    if p.grad is not None and getattr(p.grad, "layout", torch.strided) == torch.strided:
                        p.grad[~_mask_grad] = 0.0
            # One-line gradient-layout snapshot at first iteration
            if sparse_active_set and iteration == first_iter:
                _layouts = {
                    group["name"]: str(getattr(group["params"][0].grad, "layout", "None"))
                    for group in gaussians.optimizer.param_groups
                }
                print(f"[sparse] first-iter grad layouts: {_layouts}", flush=True)
        else:
            _nan_skip_count = locals().get("_nan_skip_count", 0) + 1
            if _nan_skip_count <= 5:
                print(
                    f"[nan-guard] iter={iteration} SKIPPING backward+step — non-finite loss."
                    f" Ll1={float(Ll1.detach()):.4g}",
                    flush=True,
                )
            if tb_writer is not None:
                tb_writer.add_scalar("train/nan_skip", 1.0, iteration)
            gaussians.optimizer.zero_grad(set_to_none=True)
        mark_stage("backward")

        # Update visibility EMA for geometry dashboard
        with torch.no_grad():
            update_visibility_ema(gaussians, render_pkg["is_used"])
            if taming_enabled and iteration < opt.densify_until_iter:
                visibility_filter = render_pkg["visibility_filter"]
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    render_pkg["radii"][visibility_filter].to(gaussians.max_radii2D.dtype),
                )

        iter_end.record()

        # Compute utility between backward and step (reads gradient norms)
        if use_energy_mcmc and _loss_finite:
            utility = compute_gaussian_utility(
                gaussians=gaussians,
                render_pkg=render_pkg,
                iteration=iteration,
                w_alpha=args.energy_w_alpha,
                w_vis=args.energy_w_vis,
                w_grad=args.energy_w_grad,
                w_scale=args.energy_w_scale,
                w_dead=args.energy_w_dead,
                w_support=getattr(args, "energy_w_support", 2.0),
                beta_opacity=getattr(args, "energy_beta_opacity", 1.0),
                beta_scale=getattr(args, "energy_beta_scale", 0.5),
                # No getattr fallback: energy_alpha_dead is an unconditional
                # OptimizationParams default, always present on the real parsed
                # args (issue #368 -- the old getattr(..., 0.005) fallback
                # literal had drifted from the real default of 0.05, the same
                # stale value that already caused a confirmed live bug once
                # before the parser default was raised; removed rather than
                # re-stated so it can't silently drift again).
                alpha_dead=args.energy_alpha_dead,
                w_aniso_utility=float(getattr(args, "utility_aniso_weight", 0.1)),
                aniso_soft_limit=float(getattr(args, "utility_aniso_soft_limit", 10.0)),
                args=args,
            )

            # Temperature annealing for birth/death sampling
            u_temp = min(iteration / 30000.0, 1.0)
            tau_t = max(getattr(args, "energy_temp_tau", 0.4), 1e-6)
            denom_t = 1.0 - math.exp(-1.0 / tau_t)
            alpha_t = (1.0 - math.exp(-u_temp / tau_t)) / denom_t
            temperature = (
                args.energy_temp_min
                + (args.energy_temp_start - args.energy_temp_min)
                * (1.0 - alpha_t)
            )
            if should_log_scalars:
                tb_writer.add_scalar('mcmc/temperature', temperature, iteration)
                # Phase 3: log population energy stats when mode is active
                if tb_writer and getattr(args, "energy_mode", "legacy_utility") == "population_energy":
                    from eligsir.utils.gaussian_energy_policy import GaussianEnergyPolicy, energy_config_from_args
                    # issue #385: this telemetry config used to pass only 2 of the
                    # 14 weights compute_population_energy() reads, so
                    # energy/population_mean and energy/population_p95 were logged
                    # from a DIFFERENT energy function than the one actually
                    # driving utility at utils/energy_mcmc.py's population_energy
                    # branch. Now built from the same factory as that branch, so
                    # the scalar reports the energy that is really in use.
                    # No getattr fallback for energy_alpha_dead -- see comment
                    # above at this same flag's other call site (issue #368).
                    _log_pop_cfg = energy_config_from_args(
                        args,
                        alpha_dead=float(args.energy_alpha_dead),
                        energy_mode="population_energy",
                    )
                    _log_policy = GaussianEnergyPolicy(_log_pop_cfg)
                    _log_terms = _log_policy.compute_terms(gaussians, render_pkg, iteration)
                    with torch.no_grad():
                        _log_energy = _log_policy.compute_population_energy(_log_terms)
                    tb_writer.add_scalar("energy/population_mean", _log_energy.population_mean, iteration)
                    tb_writer.add_scalar("energy/population_p95", _log_energy.population_p95, iteration)
        else:
            utility = None
            temperature = 1.0

        # Mask utility to active set: non-visible Gaussians keep only their
        # opacity-based score so MCMC decisions are driven by visible Gaussians.
        if sparse_active_set and utility is not None:
            _active_util = render_pkg["visibility_filter"].detach()
            utility[~_active_util] = (
                getattr(args, "energy_w_alpha", 1.0)
                * gaussians.get_opacity.squeeze(-1)[~_active_util]
            )
        mark_stage("utility")

        taming_scores = None
        run_taming_growth = (
            taming_enabled
            and opt.densify_from_iter < iteration < opt.densify_until_iter
            and iteration % max(1, args.taming_score_interval or opt.densification_interval) == 0
        )
        if run_taming_growth:
            if taming_counts is None:
                taming_budget = get_taming_budget(args)
                taming_counts = get_taming_count_array(
                    start_count=gaussians.get_xyz.shape[0],
                    budget=taming_budget,
                    opt=args,
                    mode=getattr(args, "taming_budget_mode", "final_count"),
                )
                print(
                    f"[taming-budget] start={taming_counts[0]} "
                    f"final={taming_counts[-1]} "
                    f"steps={len(taming_counts) - 1} "
                    f"mode={getattr(args, 'taming_budget_mode', 'final_count')}",
                    flush=True,
                )

            if not taming_stats_logged:
                print(
                    "[taming-stats] requiring exact renderer accumulators; "
                    "scoring raises on approximation fallback",
                    flush=True,
                )
                taming_stats_logged = True

            camlist = sample_taming_cameras(scene.getTrainCameras(), getattr(args, "taming_cams", 3))
            edge_maps = [taming_edge_maps[id(cam)].to(device=gaussians.get_xyz.device) for cam in camlist]
            with torch.no_grad():
                taming_scores = compute_taming_scores(
                    scene=scene,
                    camlist=camlist,
                    edge_maps=edge_maps,
                    gaussians=gaussians,
                    pipe=pipe,
                    bg=bg,
                    weights=taming_weights,
                    opt=opt,
                )
        mark_stage("taming_scoring")

        # Optimizer step (must come BEFORE MCMC mutations)
        if iteration < opt.iterations and _loss_finite:
            if optimizer_type == "selective_adam":
                visible = render_pkg["visibility_filter"].detach().to(dtype=torch.bool).contiguous()
                gaussians.prepare_selective_adam_step(allow_dense_grads=allow_dense_grads)
                mark_stage("selective_adam_prepare")
                gaussians.optimizer.step(visibility=visible)
                mark_stage("selective_adam_step")
            else:
                gaussians.optimizer.step()
            if pipe.gsplat_sparse_grad:
                if sparse_active_set:
                    gaussians.normalize_rotation_params(mask=visible)
                else:
                    gaussians.normalize_rotation_params()
                mark_stage("rotation_normalize")
            gaussians.optimizer.zero_grad(set_to_none=True)

            if densification_strategy in {
                "mcmc", "hybrid", "gsplat_mcmc", "gsplat_energy_mcmc", "gsplat_live_strategy",
            }:
                mcmc_strategy.inject_noise(
                    gaussians=gaussians,
                    args=args,
                    xyz_lr=xyz_lr,
                    visible=visible if optimizer_type == "selective_adam" else None,
                    sparse_active_set=sparse_active_set,
                    iteration=iteration,
                    tb_writer=tb_writer,
                )
        mark_stage("optimizer")

        # Track active-set statistics
        _num_visible = int(render_pkg["visibility_filter"].sum().item())
        _active_fraction = _num_visible / max(gaussians.get_xyz.shape[0], 1)

        # Progress bar, logging, geometry dashboard
        with torch.no_grad():
            # Cache scalar values ONCE — all downstream code uses these (avoids redundant syncs)
            _Ll1_val = Ll1.item()
            _loss_val = loss.item()
            _iter_time_ms = iter_start.elapsed_time(iter_end)

            ema_loss_for_log = 0.4 * _loss_val + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            training_report(
                tb_writer,
                iteration,
                _Ll1_val,
                _loss_val,
                l1_loss,
                _iter_time_ms,
                testing_iterations,
                scene,
                render,
                (pipe, background),
                no_empty_cache=getattr(run_args, 'no_empty_cache', False),
                log_scalars=bool(should_log_scalars),
            )
            if progress_disabled and (iteration % 500 == 0 or iteration == opt.iterations):
                now = time.perf_counter()
                delta_iter = max(1, iteration - progress_log_last_iter)
                it_s = delta_iter / max(now - progress_log_last_time, 1e-9)
                print(
                    f"[progress] iter={iteration} loss={ema_loss_for_log:.7f} "
                    f"it/s={it_s:.2f} sh={gaussians.active_sh_degree}",
                    flush=True,
                )
                progress_log_last_iter = iteration
                progress_log_last_time = now
                if should_log_scalars:
                    tb_writer.add_scalar('iterations_per_sec', it_s, iteration)
                    tb_writer.add_scalar('total_points', gaussians.get_xyz.shape[0], iteration)
                    if optimizer_type == "selective_adam":
                        tb_writer.add_scalar('active_set/num_visible', _num_visible, iteration)
                        tb_writer.add_scalar('active_set/active_fraction', _active_fraction, iteration)
                        # Periodic grad-layout dashboard: track which parameters retain
                        # sparse COO layouts vs. dense strided after backward.
                        _sparse_count = 0
                        _none_count = 0
                        for group in gaussians.optimizer.param_groups:
                            name = group["name"]
                            p = group["params"][0]
                            g = p.grad
                            if g is None:
                                _none_count += 1
                                tb_writer.add_scalar(f"grad_layout/{name}", -1.0, iteration)
                            elif getattr(g, "layout", torch.strided) != torch.strided:
                                _sparse_count += 1
                                tb_writer.add_scalar(f"grad_layout/{name}", 1.0, iteration)
                            else:
                                tb_writer.add_scalar(f"grad_layout/{name}", 0.0, iteration)
                        tb_writer.add_scalar("grad_layout/sparse_count", _sparse_count, iteration)
                        tb_writer.add_scalar("grad_layout/none_count", _none_count, iteration)
                    _score_image, _score_gt, _score_transform = apply_declared_score_clamp_pair(
                        image,
                        gt_image,
                        context="offline_training_progress_psnr",
                    )
                    tb_writer.add_scalar(
                        'train_loss_patches/psnr',
                        psnr(_score_image, _score_gt).mean().item(),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        'train_loss_patches/score_clamp_fraction',
                        _score_transform['render']['fraction_changed'],
                        iteration,
                    )
            # Geometry failure dashboard — configurable to avoid sync collisions.
            geometry_log_interval = max(1, int(getattr(args, "geometry_log_interval", 500)))
            sfm_anchor_interval = max(1, int(getattr(args, "sfm_anchor_interval", 2000)))
            if iteration % geometry_log_interval == 0:
                with torch.no_grad():
                    # Load SfM points once, but only on the slower SfM-anchor cadence.
                    include_sfm_anchor = iteration % sfm_anchor_interval == 0
                    if include_sfm_anchor and not hasattr(scene, "_sfm_points"):
                        try:
                            import numpy as np
                            from scene.dataset_readers import fetchPly
                            sfm_pcd = fetchPly(os.path.join(dataset.source_path, "sparse", "0", "points3D.ply"))
                            scene._sfm_points = torch.tensor(np.asarray(sfm_pcd.points), dtype=torch.float32, device="cuda")
                        except Exception:
                            scene._sfm_points = None

                    geo = compute_geometry_dashboard(
                        gaussians,
                        sfm_points=getattr(scene, "_sfm_points", None) if include_sfm_anchor else None,
                    )
                    gaussians._last_geo = geo

                    for k, v in geo.items():
                        if should_log_scalars:
                            tb_writer.add_scalar(f"geometry/{k}", v, iteration)

                    print(
                        f"[geo] iter={iteration} "
                        f"N={int(geo['num_gaussians'])} "
                        f"LSOM={geo.get('low_support_opacity_mass', 0.0):.4f} "
                        f"OSF={geo.get('opacity_scale_floater_score', 0.0):.4f} "
                        f"mean_sup={geo.get('mean_support', 0.0):.4f}",
                        flush=True,
                    )
        mark_stage("reporting")

        # Log CUDA memory allocator stats every 50 iterations for debugging stutters
        if should_log_scalars and getattr(args, "log_memory", False) and torch.cuda.is_available():
            try:
                mem_stats = torch.cuda.memory_stats()
                tb_writer.add_scalar('memory/allocated_bytes', mem_stats.get('allocated_bytes.all.current', 0), iteration)
                tb_writer.add_scalar('memory/reserved_bytes', mem_stats.get('reserved_bytes.all.current', 0), iteration)
                tb_writer.add_scalar('memory/active_bytes', mem_stats.get('active_bytes.all.current', 0), iteration)
                tb_writer.add_scalar('memory/num_alloc_retries', mem_stats.get('num_alloc_retries', 0), iteration)
                tb_writer.add_scalar('memory/num_segments', mem_stats.get('num_segments', 0), iteration)
                tb_writer.add_scalar('memory/num_segments_reclaimed', mem_stats.get('num_segments_reclaimed', 0), iteration)
                tb_writer.add_scalar('memory/cached_events', mem_stats.get('cuda_events.max', 0), iteration)
                # Log # of GPU kernel launches in last iteration (from the profiler viewpoint)
                tb_writer.add_scalar('memory/oversize_allocations', mem_stats.get('oversize_allocations.current', 0), iteration)
            except Exception:
                pass  # memory stats may not be available on all ROCm versions

        # Push metrics and image to the live web viewer
        if web_viewer is not None:
            _it_s = 1000.0 / max(_iter_time_ms, 0.001) if _iter_time_ms > 0 else 0.0
            web_viewer.push_metrics({
                "iteration": iteration,
                "loss": _loss_val,
                "l1": _Ll1_val,
                "num_gaussians": gaussians.get_xyz.shape[0],
                "sh_degree": gaussians.active_sh_degree,
                "iter_time_ms": _iter_time_ms,
                "viewer_render_ms": _viewer_render_time_ms,
                "it_s": _it_s,
            })
            # Finalize async GPU→CPU copy (pinned buffer + separate stream avoids blocking main stream)
            if _viewer_pinned_buf is not None:
                _viewer_stream.synchronize()
                _viewer_image_arr = _web_viewer_mod.encode_render_image_finish(_viewer_pinned_buf)
                _viewer_pinned_buf = None
                if _viewer_image_arr is not None:
                    web_viewer.push_image(_viewer_image_arr, iteration)
        if _live_viewer is not None:
            from eligsir.map.views import read_view_from_model as _read_view_from_model
            _live_viewer.log_gaussian_model(_read_view_from_model(gaussians), iteration)
        if video_recorder is not None and _video_frames:
            for _video_cam_idx, _video_arr in _video_frames:
                video_recorder.write(_video_cam_idx, _video_arr)

        # Compute MCMC schedule
        sched = get_mcmc_schedule(
            iteration=iteration,
            current_n=gaussians.get_xyz.shape[0],
            cap_max=args.cap_max,
            cfg=mcmc_cfg,
        )

        # Log schedule metrics to TensorBoard
        if should_log_scalars:
            tb_writer.add_scalar('mcmc/growth_factor', sched['growth_factor'], iteration)
            tb_writer.add_scalar('mcmc/capacity_ratio_rho', sched['rho'], iteration)
            tb_writer.add_scalar('mcmc/dead_opacity_threshold', sched['dead_opacity_threshold'], iteration)
            tb_writer.add_scalar('mcmc/relocate_interval', sched['relocate_interval'], iteration)
            tb_writer.add_scalar('mcmc/grow_interval', sched['grow_interval'], iteration)
            tb_writer.add_scalar('mcmc/allow_growth', int(sched['allow_growth']), iteration)
            tb_writer.add_scalar('mcmc/allow_relocation', int(sched['allow_relocation']), iteration)
            tb_writer.add_scalar('train_loss_patches/xyz_lr', xyz_lr, iteration)
            tb_writer.add_scalar('timing/viewer_render_ms', _viewer_render_time_ms, iteration)

        # Optional closed-loop: suppress growth if LSOM is rising
        if use_energy_mcmc and iteration > opt.densify_from_iter:
            geo = getattr(gaussians, "_last_geo", {})
            lsom = geo.get("low_support_opacity_mass", 0.0)
            if lsom > 0.15:
                # Too many unsupported splats: halve growth factor
                sched["growth_factor"] = max(1.0, sched["growth_factor"] * 0.5)
                if iteration % mcmc_control_log_interval == 0 or iteration == opt.iterations:
                    print(f"[mcmc-control] LSOM={lsom:.3f} > 0.15, suppressing growth", flush=True)
        mark_stage("schedule")

        _should_save = (
            iteration in saving_iterations
            or (args.save_interval > 0 and iteration % args.save_interval == 0)
        )
        _should_checkpoint = (
            iteration in checkpoint_iterations
            or (args.checkpoint_interval > 0 and iteration % args.checkpoint_interval == 0)
        )

        def _apply_iteration_mutations():
            nonlocal taming_densify_step
            # Training owns loss/backward/optimizer; strategy owns population edits.
            mcmc_strategy.step_post_backward(
                gaussians=gaussians,
                args=args,
                sched=sched,
                iteration=iteration,
                utility=utility,
                temperature=temperature,
                use_energy_mcmc=use_energy_mcmc,
                tb_writer=tb_writer,
                should_log_strategy=should_log_strategy,
                render_pkg=render_pkg,
                lr=xyz_lr,
            )
            if taming_enabled and run_taming_growth and taming_scores is not None:
                with torch.no_grad():
                    before = gaussians.get_xyz.shape[0]
                    target_idx = min(taming_densify_step + 1, len(taming_counts) - 1)
                    target_count = taming_counts[target_idx]
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    result = gaussians.densify_with_taming_scores(
                        scores=taming_scores,
                        target_count=target_count,
                        extent=scene.cameras_extent,
                        min_opacity=getattr(args, "taming_min_opacity", 0.005),
                        max_screen_size=size_threshold,
                        iteration=iteration,
                        prune_stop_iter=getattr(args, "taming_prune_stop_iter", 3200),
                        grad_threshold=opt.densify_grad_threshold,
                    )
                    taming_densify_step += 1
                    after = gaussians.get_xyz.shape[0]
                    if tb_writer:
                        tb_writer.add_scalar('mcmc/growth_delta_N', after - before, iteration)
                        tb_writer.add_scalar('taming/cloned_count', result['cloned'], iteration)
                        tb_writer.add_scalar('taming/split_count', result['split'], iteration)
                        tb_writer.add_scalar('taming/pruned_count', result['pruned'], iteration)
                        tb_writer.add_scalar('taming/target_count', target_count, iteration)
                    if should_log_strategy(iteration):
                        print(
                            f"[{densification_strategy}-taming-grow] iter={iteration} "
                            f"target={target_count} "
                            f"clone={result['cloned']} split={result['split']} "
                            f"prune={result['pruned']} N={before}->{after}",
                            flush=True,
                        )
            mark_stage("mutation")

        _iteration_outputs = _mutate_and_capture_iteration_outputs(
            gaussians,
            mutate=_apply_iteration_mutations,
            capture_ply=_should_save,
            capture_checkpoint_state=_should_checkpoint,
        )

        if _should_save:
            print("\n[ITER {}] Saving Gaussians".format(iteration), flush=True)
            _ply_path = os.path.join(
                scene.model_path,
                "point_cloud/iteration_{}".format(iteration),
                "point_cloud.ply",
            )
            save_worker.enqueue(
                lambda s=_iteration_outputs["ply"], p=_ply_path: _write_ply_from_snapshot(s, p)
            )

        if scene_cache_writer is not None and scene_cache_interval > 0 and iteration % scene_cache_interval == 0:
            _cache_start = time.perf_counter()
            scene_cache_writer.save(gaussians, iteration, final=False)
            stage_times["web_viewer_scene_cache"] = stage_times.get("web_viewer_scene_cache", 0.0) + (
                time.perf_counter() - _cache_start
            )

        if _should_checkpoint:
            print("\n[ITER {}] Saving Checkpoint".format(iteration), flush=True)
            _chk_state = _iteration_outputs["checkpoint"]
            _chk_iter = iteration
            _chk_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"

            def _save_chk(s=_chk_state, i=_chk_iter, p=_chk_path):
                # Atomic write: temp file + os.replace, so a SIGKILL mid-write
                # (e.g. terminationGracePeriodSeconds expiring on a preempted
                # k8s pod) can never leave a truncated chkpntN.pth behind.
                tmp = p + ".tmp"
                torch.save((s, i), tmp)
                os.replace(tmp, p)

            save_worker.enqueue(_save_chk)
            _emergency_state["gaussians"] = gaussians
            _emergency_state["iteration"] = iteration
            _emergency_state["model_path"] = scene.model_path
        mark_stage("checkpoint")
        finish_stage()
        log_stage_times(tb_writer, benchmark_file, iteration, stage_times, args, scene, densification_strategy, num_visible=_num_visible)

        if _prof is not None:
            _prof.step()

    # --- Profiler export (after loop ends) ---
    if _prof is not None:
        _prof.__exit__(None, None, None)
        _prof.export_chrome_trace(_profile_path)
        print(f"[profile] Trace saved to {_profile_path}.", flush=True)
    if scene_cache_writer is not None:
        scene_cache_writer.save(gaussians, opt.iterations, final=True)

    from eligsir.utils.run_completion_state import (
        RunCompletionState,
        FINAL_REPORT_CONTRACT,
        TRAJECTORY_EVAL_CONTRACT,
        PhaseContract as _PhaseContract,
    )

    _run_state = RunCompletionState(args.model_path)
    begin_offline_completion(
        save_worker,
        _run_state,
        final_iteration=opt.iterations,
        orig_sigterm=_orig_sigterm,
    )

    _pv_cpu_queue.put(None)
    _pv_thread.join(timeout=120)
    if _progress_frames:
        from eligsir.utils.comparison_report import _write_mp4
        _prog_path = os.path.join(args.model_path, "training_progress.mp4")
        with _run_state.phase(_PhaseContract(
            name="training_progress_video",
            label=f"training_progress.mp4 ({len(_progress_frames)} frames)",
            required=False,
        )):
            print(f"[train] Writing training_progress.mp4 ({len(_progress_frames)} frames @ {_progress_video_fps}fps)…", flush=True)
            _err = _write_mp4(_prog_path, iter(_progress_frames), fps=_progress_video_fps)
            if _err is not None:
                raise RuntimeError(f"training_progress.mp4 write failed: {_err}")
            print(f"[train] training_progress.mp4 → {_prog_path}", flush=True)

    _benchmark_mode = getattr(args, "streaming_benchmark_mode", False)
    _report_final = getattr(args, "streaming_report_final", True) and not _benchmark_mode

    # Stop the resource-monitor thread before the report reads its summary
    # (or, if no report is written, simply to shut the background thread
    # down cleanly) -- mirrors train_streaming.py's identical wiring.
    if _resource_monitor is not None:
        _resource_monitor.stop()

    # Mandatory post-training report: test PSNR + comparison renders + trajectory MP4.
    if _report_final:
        with _run_state.phase(FINAL_REPORT_CONTRACT) as _ph:
            from eligsir.utils.comparison_report import (
                write_post_training_report,
                git_provenance_from_args,
                determinism_from_args,
            )
            from eligsir.utils.source_attestation import (
                source_attestation_from_args as _source_attestation_from_args,
            )
            from eligsir.utils.experiment_origin import experiment_origin_from_args as _experiment_origin_from_args
            from eligsir.utils.execution_semantics import (
                build_execution_semantics_block as _execution_semantics_from_args,
            )
            from eligsir.runtime.deterministic_digest import (
                report_with_global_digest as _report_with_global_digest,
            )
            from eligsir.utils.presentation_ssaa import presentation_ssaa_kwargs_from_args
            from eligsir.utils.trajectory_video import presentation_trajectory_interpolate_kwargs_from_args
            _ph.record_output(
                write_post_training_report(
                    model_path=args.model_path,
                    iteration=opt.iterations,
                    gaussians=gaussians,
                    train_cams=list(scene.getTrainCameras()),
                    test_cams=list(scene.getTestCameras()),
                    render_fn=render,
                    pipe=pipe,
                    background=background,
                    tb_writer=tb_writer,
                    log_prefix="report",
                    depth_max_m=float(args.rgbd_max_depth),
                    report_videos=bool(getattr(args, "streaming_report_videos", True)),
                    resource_monitor=(
                        _resource_monitor.summary() if _resource_monitor is not None else None
                    ),
                    **presentation_ssaa_kwargs_from_args(args),
                    **presentation_trajectory_interpolate_kwargs_from_args(args),
                    resumed_from_iteration=_resumed_from_iteration,
                    reference_map_path=getattr(args, "reference_map", None),
                    map_comparison_align=getattr(args, "map_comparison_align", True),
                    baseline_run=getattr(args, "baseline_run", None),
                    git_provenance=git_provenance_from_args(args),
                    determinism=_report_with_global_digest(
                        determinism_from_args(args)
                    ),
                    # Issue #1302: same passive-record mechanism as
                    # git_provenance/determinism above; see
                    # utils/source_attestation.py.
                    source_attestation=_source_attestation_from_args(args),
                    # Issue #511: WHY this run exists / which task owns it. Pure
                    # accessor -- the object was resolved once at startup and is the
                    # same one written to <model_path>/experiment_origin.json.
                    experiment_origin=_experiment_origin_from_args(args),
                    # Issue #168: resolved run-mode axes (report-only). Present on the
                    # offline path too, so an offline run's report.json positively
                    # asserts trainer=offline rather than leaving it to be inferred.
                    execution_semantics=_execution_semantics_from_args(args),
                    # Issue #203: declared experiment validity class (report-only).
                    # This is the legacy (non-streaming) path's single end-of-run
                    # report, hence is_final=True.
                    experiment_validity_class=getattr(args, "experiment_validity_class", None),
                    experiment_validity_is_final=True,
                    # Issue #524 acceptance criterion 3: declared run intent + resolved
                    # profile stack + effective config. Assigned unconditionally on
                    # args in __main__ above (mirrors experiment_origin's own
                    # bare-access convention, see tests/test_getattr_fallback_freshness.py) --
                    # a bare access here, not getattr, is deliberate: any future
                    # dispatch path that reaches this line without that assignment
                    # having run is a real bug this should surface loudly, not degrade
                    # from silently.
                    run_intent=args.run_intent_report,
                    # Issue #705: exact context validated once at startup. The
                    # fallback preserves direct unit callers of training(); every real
                    # entrypoint assigns this before dispatch.
                    experiment_context=getattr(args, "experiment_context", None),
                    # Issue #397: trajectory.mp4 resolution policy. StreamingParams
                    # is registered for every trainer invocation (see
                    # eligsir/cli/parser.py), so these attrs exist even on a
                    # non-streaming (--streaming_replay off) run.
                    report_video_min_width=getattr(args, "streaming_report_video_min_width", None),
                    report_video_max_width=getattr(args, "streaming_report_video_max_width", None),
                    mp4_fps=getattr(args, "streaming_report_video_fps", 30),
                )
            )

        # Trajectory export (trajectory_tum.txt at model_path root): mirrors
        # train_streaming.py's TRAJECTORY_EVAL_CONTRACT block. Needed by any
        # consumer that reads back a eligsir run's estimated camera path
        # in TUM format (e.g. gs-slam-bench's adapter.py, which unconditionally
        # requires "<model_path>/trajectory_tum.txt" and previously only ever
        # got it from streaming runs -- every legacy/offline `train.py --eval`
        # benchmark arm failed post-training with FileNotFoundError on this
        # exact path; see issue #1013 champion-reproduction verification).
        if not _benchmark_mode:
            with _run_state.phase(TRAJECTORY_EVAL_CONTRACT):
                from eligsir.utils.trajectory_eval import run_trajectory_eval, discover_gt_trajectory
                _gt_tum_path = discover_gt_trajectory(getattr(args, "source_path", "")) or None
                run_trajectory_eval(
                    output_dir=args.model_path,
                    train_cams=list(scene.getTrainCameras()),
                    test_cams=list(scene.getTestCameras()),
                    gt_tum_path=_gt_tum_path,
                )
    else:
        _run_state.mark_not_requested(
            FINAL_REPORT_CONTRACT,
            reason=(
                "--streaming_benchmark_mode" if _benchmark_mode
                else "--no-streaming_report_final"
            ),
        )
        print(
            f"[report] final report disabled by "
            f"{'--streaming_benchmark_mode' if _benchmark_mode else '--no-streaming_report_final'}",
            flush=True,
        )

    if video_recorder is not None:
        video_recorder.close()
    if web_viewer is not None:
        web_viewer.set_status("finished", "training complete")
        web_viewer.close()
    if _live_viewer is not None:
        _live_viewer.close()

    # Issue #697 / #998: authoritative, machine-readable run state.
    # Persisted BEFORE output staging moves the directory to the canonical path,
    # and fail-closed raise comes AFTER output staging so a failed report phase
    # costs the run its exit code, never its outputs.
    _run_state.write()
    _run_state.print_summary()

    # Issue #997 / #1013: root-level run_summary.json canonical generic handoff
    from eligsir.utils.run_summary import write_run_summary
    write_run_summary(
        args.model_path,
        completion_state=_run_state.to_dict(),
        evidence_class="MANAGED" if getattr(args, "scientific_mode", False) else "UNMANAGED",
        args=args,
    )

    # Issue #1020: root-level artifact_manifest.json fast-ingest contract
    try:
        from eligsir.utils.artifact_manifest import build_artifact_manifest
        build_artifact_manifest(args.model_path)
    except Exception as _e:
        print(f"[artifact-manifest] Warning: failed to build artifact manifest: {_e}")

    # Gate a successful exit on the local staging dir actually having landed
    # on the canonical path -- raises OutputPersistenceError on failure,
    # which propagates uncaught up through __main__ (this file's existing
    # idiom for a hard training failure) and produces a non-zero exit code.
    _finalize_output_staging(args)

    # Issue #998 fail-closed gate: raises RunIncompleteError if any required phase failed
    # or required artifacts are missing, preventing the unqualified 'Training complete.' print.
    _run_state.raise_if_incomplete()

    # Keep emergency checkpoint/staging active through the persistence drain,
    # report generation, and canonical output finalization.
    _signal.signal(_signal.SIGTERM, _orig_sigterm)

    if web_viewer is not None and bool(getattr(args, "web_viewer_keep_alive", True)) and web_viewer_backend == "process":
        print(
            f"[web-viewer] Training complete. Viewer remains available at "
            f"http://127.0.0.1:{web_viewer_port}. Press Ctrl+C to stop this container/process.",
            flush=True,
        )
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("[web-viewer] Keep-alive stopped.", flush=True)

def prepare_output_and_logger(args, run_args=None):
    # `args` here is a ModelParams-extracted view (dataset), not the full
    # run_args ensure_model_path() was called on -- its model_path is already
    # resolved (to local staging, if active) by the caller, so just ensure
    # the directory exists rather than re-running staging setup on this
    # second, distinct object (which would spawn a second PersistenceWorker).
    os.makedirs(args.model_path, exist_ok=True)

    # cfg_args/run_args/reproduce.sh are one-time small writes at run start
    # (not a hot per-iteration path) and are ablator/reproducibility metadata
    # -- always record the canonical (persist) path, never the ephemeral
    # local staging path, per the Global Constraint in the output-staging
    # design.
    persist_path = (getattr(run_args, "persist_model_path", None) if run_args is not None else None) or args.model_path
    os.makedirs(persist_path, exist_ok=True)

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    from eligsir.utils.cfg_args_io import write_cfg_args
    cfg_args_source = run_args if run_args is not None else args
    cfg_args_ns = Namespace(**vars(cfg_args_source))
    cfg_args_ns.model_path = persist_path
    write_cfg_args(os.path.join(persist_path, "cfg_args"), cfg_args_ns)
    if run_args is not None:
        with open(os.path.join(persist_path, "run_args"), 'w') as run_args_f:
            run_args_f.write(str(public_namespace(run_args)))

    # Write reproduce.sh (includes docker compose wrapper when running in container)
    from eligsir.utils.comparison_report import write_reproduce_sh
    write_reproduce_sh(os.path.join(persist_path, "reproduce.sh"), sys.argv)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        # See train_streaming.py's SummaryWriter call for why max_queue is
        # raised: default 10 lets NFS write latency on model_path block the
        # main training thread inside add_scalar().
        tb_writer = SummaryWriter(args.model_path, max_queue=1000, flush_secs=30)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1_val, loss_val, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, no_empty_cache=False, log_scalars=True):
    from eligsir.utils.score_contract import score_tensor_from_camera

    if tb_writer and log_scalars:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1_val, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss_val, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('active_sh_degree', scene.gaussians.active_sh_degree, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        if not no_empty_cache:
            torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                score_clamp_changed = 0
                score_clamp_total = 0
                score_raw_max = float("-inf")
                for idx, viewpoint in enumerate(config['cameras']):
                    image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    gt_image = score_tensor_from_camera(viewpoint, device="cuda")
                    score_image, score_gt, score_transform = apply_declared_score_clamp_pair(
                        image,
                        gt_image,
                        context=f"offline_{config['name']}_eval_psnr",
                    )
                    score_clamp_changed += score_transform["render"]["samples_changed"]
                    score_clamp_total += score_transform["render"]["samples_total"]
                    score_raw_max = max(
                        score_raw_max, score_transform["render"]["value_max"]
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(score_image, score_gt).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/score_clamp_fraction',
                        score_clamp_changed / max(1, score_clamp_total),
                        iteration,
                    )
                    tb_writer.add_scalar(
                        config['name'] + '/score_raw_max', score_raw_max, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        if not no_empty_cache:
            torch.cuda.empty_cache()

def load_config(config_file):
    return _load_config(config_file)

# get_git_metadata() lives in utils/git_metadata.py (moved 2026-08-14) so it
# can be unit-tested without pulling in this module's full gsplat/torch
# import chain. Imported at module top (see the utils.git_metadata import
# near the other `from eligsir.utils...` imports) and re-exported here under its
# original name so existing call sites (this file's __main__, any external
# reference to `train.get_git_metadata`) are unaffected.

if __name__ == "__main__":
    args, parser_bundle = parse_training_args(sys.argv[1:], description="Training script parameters")
    lp = parser_bundle.model_params
    op = parser_bundle.optimization_params
    pp = parser_bundle.pipeline_params

    # Issue #524 (deferred half): resolve --run_intent to the run_profile it
    # names BEFORE the config merge below, so a run declaring
    # --run_intent paper_registered actually gets that profile's stack, not
    # just a report-side label. Raises loudly (rather than picking one
    # silently) if --run_intent and an explicit --run_profile disagree on
    # which profile to load. No effect on any run that doesn't pass
    # --run_intent -- see utils/run_intent.py's module docstring.
    from eligsir.utils.run_intent import (
        RunIntentConflictError,
        resolve_run_intent_and_profile,
        run_intent_report_from_args,
    )
    try:
        _effective_run_profile = resolve_run_intent_and_profile(
            getattr(args, "run_intent", None), getattr(args, "run_profile", None)
        )
    except RunIntentConflictError as e:
        raise SystemExit(f"[run_intent] {e}")

    from eligsir.cli.config import resolve_profile_stack
    from eligsir.cli.run_config import cli_explicit_dests, validate_config
    merged = validate_config(
        load_cli_config(
            config_files=[args.config] if args.config else None,
            run_profile=_effective_run_profile,
        )
    )
    explicit = cli_explicit_dests(parser_bundle.parser, sys.argv[1:])
    for key, value in merged.items():
        if key not in explicit:        # CLI flags win over config
            setattr(args, key, value)

    # M9 must inspect the raw command line after profile/config precedence has
    # resolved whether this is a live population run.  The downstream runtime
    # adapter intentionally strips legacy MCMC scheduling fields, so validating
    # there would make an explicit delegated-MCMC override undiscoverable.
    from eligsir.population_control_deprecation import (
        reject_raw_live_population_conflicts,
    )
    try:
        reject_raw_live_population_conflicts(args, sys.argv[1:])
    except ValueError as error:
        raise SystemExit(f"[population-control] {error}")

    # Issue #524 acceptance criterion 3: every run persists its declared run
    # intent, resolved profile stack, and effective (profile-derived) config --
    # built here (BEFORE apply_parallelism_profile()/git-metadata/etc. so it
    # reflects exactly what the profile stack itself set) and threaded into
    # both trainer paths' post-training report below and in
    # train_streaming.py's _streaming_report_kwargs(). Never None -- an
    # undeclared run still records declared: "UNDECLARED", the same
    # least-permissive convention --experiment_validity_class already uses.
    args.run_intent_report = run_intent_report_from_args(
        args,
        effective_config=merged,
        profile_stack=resolve_profile_stack(
            config_files=[args.config] if args.config else None,
            run_profile=_effective_run_profile,
        ),
        cli_explicit=explicit,
    )

    # Keep explicit negative CLI flags authoritative even when --config is used.

    apply_parallelism_profile(
        args, explicit_dests=profile_explicit_dests(explicit, merged)
    )

    for key, value in get_git_metadata().items():
        setattr(args, key, value)

    # Issue #511 (lane P0-A of #202) / #1013: resolve the run's task/issue provenance
    # EXACTLY ONCE, here, before any output directory exists. git_provenance
    # (above) records what executed; this records WHY it executed and which
    # tracked task owns the result. Malformed references raise loudly rather
    # than producing provenance that points at the wrong task.
    from eligsir.utils.experiment_origin import resolve_experiment_origin_from_args, validate_researchflow_origin
    args.experiment_origin = resolve_experiment_origin_from_args(args)
    validate_researchflow_origin(args)

    args.save_iterations.extend(i for i in args.early_output_iterations if 0 < i <= args.iterations)
    args.save_iterations.append(args.iterations)
    args.save_iterations = sorted(set(args.save_iterations))

    ensure_model_path(args)
    start_benchmark_log(args)
    start_output_log(args)

    # Issue #705: consume and freeze the Ablator declaration only after the
    # output-staging manager and train.log TeeStream exist. Managed ambiguity
    # raises here before any training/GPU work; an absent declaration remains
    # an explicitly non-gradeable direct run.
    from eligsir.utils.experiment_declaration import initialize_experiment_context
    args.experiment_context = initialize_experiment_context(args.model_path)

    # Issue #670: the git-provenance capture above runs before the TeeStream
    # exists, so its degradation warning never reached train.log. Re-emit it
    # here, now that stdout is teed, so a run whose provenance degraded carries
    # the reason in its own durable artifacts.
    from eligsir.utils.git_metadata import emit_deferred_git_provenance_warning
    emit_deferred_git_provenance_warning()

    # Issue #511: write the resolved origin to the run output ROOT, so the
    # directory stays self-describing after being copied off this machine.
    # Written after start_output_log() so the banner + any provenance warning
    # land in train.log. Mirrors cfg_args by writing to the persist path too
    # when output staging is active.
    try:
        from eligsir.utils.experiment_origin import (
            format_origin_line, write_experiment_origin_file)
        write_experiment_origin_file(
            args.model_path, args.experiment_origin,
            persist_path=getattr(args, "persist_model_path", None))
        print(format_origin_line(args.experiment_origin), flush=True)
        for _w in args.experiment_origin.get("warnings", []):
            print(f"[origin] WARNING: {_w}", flush=True)
    except Exception as e:
        print(f"[origin] failed to write experiment_origin.json: {e!r}", flush=True)

    early_web_viewer = None
    _early_web_viewer_mod = None
    if bool(args.web_viewer_enabled) and int(getattr(args, "web_viewer_port", 6010)) > 0:
        _early_web_viewer_mod = _load_legacy_web_viewer()
    if _early_web_viewer_mod is not None:
        early_web_viewer = _early_web_viewer_mod.start_web_viewer(
            port=int(getattr(args, "web_viewer_port", 6010)),
            host=getattr(args, "web_viewer_host", "0.0.0.0"),
            image_interval=max(1, int(getattr(args, "web_viewer_image_interval", 100))),
            total_cams=1,
            viewer_cam_idx=0,
            backend=str(getattr(args, "web_viewer_backend", "process")).lower(),
            cache_dir=getattr(args, "web_viewer_cache_dir", "") or os.path.join(args.model_path, "web_viewer_cache"),
            model_path=args.model_path,
        )
        if early_web_viewer is not None:
            early_web_viewer.set_status("loading", "loading scene and cameras")

    # --- Issue #168: resolved execution semantics -----------------------------
    # Printed HERE, in __main__ before training() dispatches, so all four
    # trainer paths (offline, streaming, rolling_seed, submap_stitch)
    # self-identify from one place and the banner lands in train.log. Purely
    # diagnostic except for the provably-impossible-combination refusal below.
    try:
        from eligsir.utils.execution_semantics import (
            build_execution_semantics_block,
            format_execution_semantics,
        )

        _exec_sem = build_execution_semantics_block(args)
        print(format_execution_semantics(_exec_sem), flush=True)
        if _exec_sem.get("errors"):
            if bool(getattr(args, "allow_invalid_execution_semantics", False)):
                print(
                    "[execution] --allow_invalid_execution_semantics set: "
                    "continuing despite the errors above.",
                    flush=True,
                )
            else:
                raise SystemExit(
                    "[execution] Refusing to run: the resolved execution "
                    "semantics contain provably-impossible axis combinations "
                    "(listed above). Pass "
                    "--allow_invalid_execution_semantics to override."
                )
    except SystemExit:
        raise
    except Exception as _e:  # pragma: no cover - never block a run on reporting
        print(f"[execution] resolved-semantics banner failed: {_e}", flush=True)

    print("Optimizing " + args.model_path)

    # Issue #1392: fail fast and legibly if this image's baked ROCm arch
    # doesn't match the GPU actually visible, instead of a first-matmul
    # hipBLASLt "Cannot read TensileLibrary_lazy_<arch>.dat" that reads like a
    # mislabeled image.
    from eligsir.utils.gpu_arch_guard import check_rocm_arch_match

    check_rocm_arch_match()

    # Initialize system state (RNG)
    safe_state(
        args.quiet,
        seed=getattr(args, "seed", 1),
        strict_deterministic=getattr(args, "strict_deterministic", False),
    )

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.sh_degree_schedule, run_args=args)

    # All done
    print("\nTraining complete.")
