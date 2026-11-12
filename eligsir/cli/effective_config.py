"""Print effective configuration and exit without importing torch. Issue #524."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import yaml

from eligsir.cli.config import (
    load_cli_config,
    resolve_profile_stack,
)
from eligsir.cli.parser import build_training_parser
from eligsir.cli.run_config import cli_explicit_dests
from eligsir.utils.run_intent import (
    RunIntentConflictError,
    resolve_run_intent_and_profile,
    run_intent_report_from_args,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_file(path: Path) -> dict:
    """Load a YAML file without using the loader's other machinery."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {}


def resolve_profile_provenance(
    config_files: list[str] | None = None,
    run_profile: str | None = None,
) -> dict[str, str]:
    """Return per-key provenance (profile file paths) from the resolved config stack.

    Returns a dict mapping argparse dest names to their provenance strings:
    - "default" for keys that came from argparse defaults (not in any profile)
    - "profile:<repo-relative-path>" for keys from the given profile layer
    - Later profiles override earlier ones; last-writer-wins

    Walks the exact same path as load_cli_config() / resolve_profile_stack() to
    avoid #591-style drift (issue #524 acceptance criterion 3).
    """
    from eligsir.cli.config import (
        _resolve_config_paths,
        _is_default_only,
        _expand_paths,
    )

    paths, config_files_normalized = _resolve_config_paths(config_files, run_profile)
    if _is_default_only(paths, run_profile, config_files_normalized):
        # No profile was merged on a plain no-flag run; every key is a default
        return {}

    # Walk the layers in order, tracking where each key came from
    provenance: dict[str, str] = {}
    for p in paths:
        for layer_path in _expand_paths(p, []):
            data = _load_yaml_file(layer_path)
            # Skip loader directives
            data.pop("extends", None)
            # Record this layer's contribution, letting later layers override
            try:
                rel_path = str(layer_path.relative_to(_REPO_ROOT))
            except ValueError:
                rel_path = str(layer_path)
            for key in data:
                provenance[key] = f"profile:{rel_path}"

    return provenance


def _effective_config_with_provenance(
    args: Namespace,
    merged: dict[str, Any],
    profile_stack: list[str],
    cli_explicit: set[str],
    all_defaults: dict[str, Any],
) -> dict[str, dict]:
    """Build per-option output with value and provenance.

    Returns a dict mapping option name to {value, provenance} dict, sorted by
    option name for deterministic output.
    """
    profile_provenance = resolve_profile_provenance(
        config_files=[args.config] if args.config else None,
        run_profile=getattr(args, "_effective_run_profile", None),
    )

    output = {}

    # Include all options that have a current value in args (after all precedence applied)
    args_dict = vars(args)
    all_keys = set(merged.keys()) | set(args_dict.keys()) | set(all_defaults.keys())

    # Filter out parser-internal keys and run-intent-specific keys
    exclude_keys = {"_effective_run_profile", "print_effective_config"}

    for key in sorted(all_keys):
        if key in exclude_keys:
            continue

        # Get value from args (which has CLI overrides applied), or from merged if not in args
        if key in args_dict:
            value = args_dict[key]
        else:
            value = merged.get(key, all_defaults.get(key))

        # Determine provenance
        if key in cli_explicit:
            prov = "cli"
        elif key in profile_provenance:
            prov = profile_provenance[key]
        else:
            prov = "default"

        output[key] = {
            "value": value,
            "provenance": prov,
        }

    return output


def format_effective_config_output(
    args: Namespace,
    merged: dict[str, Any],
    profile_stack: list[str],
    cli_explicit: set[str],
    all_defaults: dict[str, Any],
    run_intent: str | None = None,
) -> str:
    """Format effective config for output as YAML.

    Returns a string with sections for:
    - run_intent (if declared)
    - profile_stack
    - effective_config (with per-option provenance)
    """
    output_dict: dict[str, Any] = {}

    # Add run intent block if available
    if run_intent:
        run_intent_block = run_intent_report_from_args(
            args,
            effective_config=merged,
            profile_stack=profile_stack,
            cli_explicit=cli_explicit,
        )
        output_dict["run_intent"] = run_intent_block

    # Add profile stack
    output_dict["profile_stack"] = profile_stack

    # Add per-option effective config with provenance
    effective = _effective_config_with_provenance(
        args, merged, profile_stack, cli_explicit, all_defaults
    )
    output_dict["effective_config"] = effective

    # Format as YAML with deterministic ordering
    return yaml.safe_dump(
        output_dict,
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=True,
    )


def main(argv: list[str]) -> int:
    """Parse args, resolve config, and print effective configuration.

    Exits 0 on success, with YAML output to stdout.
    Raises SystemExit with non-zero code on error.
    """
    # Parse arguments using the same parser as the training paths
    bundle = build_training_parser()
    args = bundle.parser.parse_args(argv)

    # Resolve --run_intent to --run_profile before config merge
    try:
        _effective_run_profile = resolve_run_intent_and_profile(
            getattr(args, "run_intent", None), getattr(args, "run_profile", None)
        )
    except RunIntentConflictError as e:
        raise SystemExit(f"[run_intent] {e}") from e

    # Store for provenance tracking
    args._effective_run_profile = _effective_run_profile

    # Load and merge configuration
    merged = load_cli_config(
        config_files=[args.config] if args.config else None,
        run_profile=_effective_run_profile,
    )

    # Determine which options were explicitly passed on CLI
    explicit = cli_explicit_dests(bundle.parser, argv)

    # Collect default values (for options not in merged config)
    from scripts.dump_default_config import collect_defaults
    all_defaults = collect_defaults()

    # Resolve profile stack for reporting
    profile_stack = resolve_profile_stack(
        config_files=[args.config] if args.config else None,
        run_profile=_effective_run_profile,
    )

    # Get declared run intent (if any)
    declared_intent = getattr(args, "run_intent", None)

    # Format and print output
    output = format_effective_config_output(
        args, merged, profile_stack, explicit, all_defaults, run_intent=declared_intent
    )
    print(output, end="")

    return 0
