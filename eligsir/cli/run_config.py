"""Pydantic validation and runtime application for merged run configs."""

from __future__ import annotations

from argparse import ArgumentParser
import os
from typing import Any, Optional

from pydantic import ConfigDict, create_model

from arguments import ModelParams, OptimizationParams, PipelineParams, StreamingParams
from eligsir.cli.config import ConfigError
from eligsir.mapping.admission.budgeted_controller import RANK_POLICIES


# Configuration-first method-freeze controls. They are consumed only by the
# canonical streaming path and are launched primarily from ResearchFlow YAML.
# train.py copies every validated key onto args and persists the merged config,
# so these have the same provenance as ParamGroup-backed settings without adding
# another dozen legacy CLI switches.
_CANONICAL_EXPERIMENT_FIELDS: dict[str, type] = {
    # Canonical root-loop selection.  Historical streaming_replay configs are
    # migrated explicitly; this field is the authoritative new declaration.
    "execution_trainer": str,
    # Canonical SH-degree cadence is cumulative completed RGB-pixel exposure.
    # The legacy iteration schedule remains an offline-only parser option.
    "sh_degree_schedule_rgb_pixels": list[int],
    # #802 bounded pre-commit keyframe cohort.
    "streaming_budgeted_admission": bool,
    "streaming_admission_cohort_max_candidates": int,
    "streaming_admission_cohort_max_source_age": int,
    "streaming_admission_rank_policy": str,
    # #109/#192 sparse reliable source-depth seeding.
    "streaming_sparse_seed_min_confidence": float,
    "streaming_sparse_seed_voxel_size": float,
    "streaming_sparse_seed_max_per_frame": int,
    # #116/#953 historical replay/reactivation policy.
    "streaming_historical_repair_mode": str,
    "streaming_historical_repair_event_ordinal": int,
    "streaming_historical_repair_extra_step_cap": int,
    "streaming_historical_repair_frontier_fraction": float,
    # #110/#214 bounded assimilation-pressure controller.
    "streaming_assimilation_control": bool,
    "streaming_assimilation_dwell_ticks": int,
    "streaming_assimilation_max_extra_steps": int,
    "streaming_assimilation_max_replay_bias": float,
    # Canonical causal partition control uses bounded non-oracle probes. These
    # switches live on the shared parser for legacy/research compatibility.
    "streaming_partition_focus_signal_source": str,
    "streaming_partition_focus_control_probe_n": int,
    # #1279/residency-budget decode/thread/payload policy switches
    # (eligsir/cli/parser.py, build_training_parser). Added directly to
    # the shared parser rather than one of the four ParamGroup classes
    # _arg_fields() introspects above, so any profile YAML setting one of
    # these (e.g. configs/profiles/causal_replay/scannet.yml's
    # causal_thread_pinning_policy: pinned) failed RunConfig validation with
    # "Extra inputs are not permitted" until listed here explicitly, the
    # same way every other parser-direct field in this dict already is.
    "streaming_payload_budget_policy": str,
    "causal_decode_workers_policy": str,
    "causal_thread_pinning_policy": str,
    "streaming_psnr_chart_interval": int,
}


_METHOD_FREEZE_DEFAULTS = {
    "streaming_budgeted_admission": False,
    "streaming_admission_cohort_max_candidates": 4,
    "streaming_admission_cohort_max_source_age": 8,
    "streaming_admission_rank_policy": "combined",
    "streaming_sparse_seed_min_confidence": 0.0,
    "streaming_sparse_seed_voxel_size": 0.02,
    "streaming_sparse_seed_max_per_frame": 1000,
    "streaming_historical_repair_mode": "targeted_history",
    "streaming_historical_repair_event_ordinal": 0,
    "streaming_historical_repair_extra_step_cap": 0,
    "streaming_historical_repair_frontier_fraction": 0.5,
    "streaming_assimilation_control": False,
    "streaming_assimilation_dwell_ticks": 3,
    "streaming_assimilation_max_extra_steps": 2,
    "streaming_assimilation_max_replay_bias": 2.0,
}


def _arg_fields() -> dict[str, tuple]:
    """Introspect ParamGroups plus explicit canonical experiment fields."""
    parser = ArgumentParser()
    for group_cls in (ModelParams, OptimizationParams, PipelineParams, StreamingParams):
        group_cls(parser)
    fields: dict[str, tuple] = {}
    for action in parser._actions:
        dest = action.dest
        if dest in ("help", "config", "run_profile"):
            continue
        py_type = action.type if callable(action.type) else Any
        fields[dest] = (Optional[py_type], None)
    for name, py_type in _CANONICAL_EXPERIMENT_FIELDS.items():
        fields[name] = (Optional[py_type], None)
    return fields


def build_run_config_model():
    """Dynamically create the Pydantic RunConfig model with merged field definitions."""
    return create_model(
        "RunConfig",
        __config__=ConfigDict(extra="forbid"),
        **_arg_fields(),
    )


def _resolved_method_freeze_config(merged: dict) -> dict:
    resolved = dict(_METHOD_FREEZE_DEFAULTS)
    for key in resolved:
        if key in merged and merged[key] is not None:
            resolved[key] = merged[key]
    return resolved


def _install_replay_digest_reporting() -> None:
    """Reset the control-plane digest before dispatching a validated run."""
    from eligsir.runtime.deterministic_digest import reset_global

    reset_global()


def _configure_method_freeze_runtime(merged: dict) -> None:
    """Install validated method policy before either trainer is dispatched.

    Assimilation control has a direct typed runtime API. Historical repair still
    has an older env-shaped internal read in ``utils.partition_control``; until
    that monolith is split, overwrite those four compatibility variables from
    the validated config on every invocation. External environment values are
    therefore never authoritative and cannot alter a ResearchFlow/paper run.
    """
    cfg = _resolved_method_freeze_config(merged)

    from eligsir.utils.assimilation_pressure import configure_assimilation_control
    from eligsir.utils.partition_control import reset_historical_repair_state

    reset_historical_repair_state()

    configure_assimilation_control(
        enabled=bool(cfg["streaming_assimilation_control"]),
        dwell_ticks=int(cfg["streaming_assimilation_dwell_ticks"]),
        max_extra_steps=int(cfg["streaming_assimilation_max_extra_steps"]),
        max_replay_bias=float(cfg["streaming_assimilation_max_replay_bias"]),
    )

    # Compatibility transport only; source of truth is cfg above. Explicitly
    # overwrite rather than setdefault so a shell variable can never win.
    os.environ["ELIGSIR_HISTORICAL_REPAIR_MODE"] = str(
        cfg["streaming_historical_repair_mode"]
    )
    os.environ["ELIGSIR_HISTORICAL_REPAIR_EVENT_ORDINAL"] = str(
        int(cfg["streaming_historical_repair_event_ordinal"])
    )
    os.environ["ELIGSIR_HISTORICAL_REPAIR_EXTRA_STEP_CAP"] = str(
        int(cfg["streaming_historical_repair_extra_step_cap"])
    )
    os.environ["ELIGSIR_HISTORICAL_REPAIR_FRONTIER_FRACTION"] = str(
        float(cfg["streaming_historical_repair_frontier_fraction"])
    )


_MODEL = None


def validate_config(merged: dict) -> dict:
    """Validate merged YAML and install its canonical runtime policy."""
    global _MODEL
    if _MODEL is None:
        _MODEL = build_run_config_model()
    try:
        _MODEL(**merged)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"Invalid run-config: {exc}") from exc
    if (
        "streaming_admission_rank_policy" in merged
        and merged["streaming_admission_rank_policy"] not in RANK_POLICIES
    ):
        raise ConfigError(
            "Invalid run-config: streaming_admission_rank_policy must be one of "
            f"{sorted(RANK_POLICIES)}"
        )
    if "execution_trainer" in merged and merged["execution_trainer"] not in {
        "offline_reconstruction",
        "causal_mapping",
    }:
        raise ConfigError(
            "Invalid run-config: execution_trainer must be offline_reconstruction "
            "or causal_mapping"
        )
    validated = {k: v for k, v in merged.items()}
    _install_replay_digest_reporting()
    _configure_method_freeze_runtime(validated)
    return validated


def cli_explicit_dests(parser, argv) -> set[str]:
    """Return dests whose option string was explicitly present in argv."""
    tokens = {tok.split("=", 1)[0] for tok in argv}
    explicit = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in tokens:
                explicit.add(action.dest)
                break
            if opt.startswith("--") and ("--no-" + opt[2:]) in tokens:
                explicit.add(action.dest)
                break
    return explicit


__all__ = [
    "build_run_config_model",
    "cli_explicit_dests",
    "validate_config",
]
