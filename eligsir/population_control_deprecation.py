"""M9 compatibility metadata for the retired population-control families.

This module deliberately describes a command/configuration boundary only.  It
does not alter a legacy executor: registered reproductions retain their exact
arguments until the ResearchFlow horizon closes.
"""

from __future__ import annotations

from typing import Any


LEGACY_POPULATION_CONTROLS_STATUS = (
    "reproduction_only_pending_researchflow_horizon"
)

LEGACY_POPULATION_CONTROL_CLI_GUIDANCE = """
M9 legacy population controls: lifecycle, post-mapping mature mobility,
depth-lock, direct-depth insertion, and delegated-MCMC controls are
reproduction-only pending the ResearchFlow horizon.  Registered legacy
reproductions retain them unchanged.  A live probabilistic population run
rejects conflicting controls; use its owned policy configuration instead.
Relevant legacy controls include --streaming_depth_lock_centers and
--densification_strategy=gsplat_mcmc.
""".strip()

# Each Boolean entry is both recorded in MappingRuntimeConfig.static_options
# and forbidden when the live probabilistic owner is requested.  Numerical
# depth-lock tuning is intentionally absent: without one of these activation
# flags it has no competing owner to disable.
LEGACY_POPULATION_CONTROL_DESTINATIONS = (
    ("legacy_lifecycle_enabled", "streaming_lifecycle_enabled", "--streaming-lifecycle-enabled"),
    (
        "legacy_post_mapping_mature_mobility",
        "streaming_post_mapping_mature_mobility",
        "--streaming-post-mapping-mature-mobility",
    ),
    (
        "legacy_depth_lock_centers",
        "streaming_depth_lock_centers",
        "--streaming-depth-lock-centers",
    ),
    (
        "legacy_depth_lock_bootstrap",
        "streaming_depth_lock_bootstrap",
        "--streaming-depth-lock-bootstrap",
    ),
    (
        "legacy_depth_lock_insertions",
        "streaming_depth_lock_insertions",
        "--streaming-depth-lock-insertions",
    ),
    (
        "legacy_depth_lock_prune_conflicts",
        "streaming_depth_lock_prune_conflicts",
        "--streaming-depth-lock-prune-conflicts",
    ),
    (
        "legacy_insert_from_depth",
        "streaming_insert_from_depth",
        "--streaming-insert-from-depth",
    ),
)

# The launcher sees raw argv and can reject the entire delegated-MCMC family,
# including an explicit value equal to a parser default.  That removes the
# ambiguity that a resolved Namespace cannot recover.
LEGACY_POPULATION_CONTROL_ARG_PREFIXES = (
    "--streaming_lifecycle_",
    "--streaming_post_mapping_mature_mobility",
    "--streaming_depth_lock_",
    "--streaming_insert_from_depth",
    "--densification_strategy",
    "--mcmc_",
    "--streaming_mcmc_",
    "--energy_mcmc",
    "--no-energy_mcmc",
)


def legacy_population_control_values(args: Any) -> dict[str, bool]:
    """Return the M9 activation values preserved in the frozen config."""

    values = {
        metadata_name: bool(getattr(args, destination))
        for metadata_name, destination, _option in LEGACY_POPULATION_CONTROL_DESTINATIONS
    }
    values["legacy_delegated_mcmc"] = (
        str(getattr(args, "densification_strategy")).lower() == "gsplat_mcmc"
    )
    return values


def active_legacy_population_controls(args: Any) -> tuple[str, ...]:
    """Return live-incompatible M9 controls, in stable CLI-display order."""

    values = legacy_population_control_values(args)
    active = [
        option
        for metadata_name, _destination, option in LEGACY_POPULATION_CONTROL_DESTINATIONS
        if values[metadata_name]
    ]
    if values["legacy_delegated_mcmc"]:
        active.append("--densification-strategy gsplat_mcmc")
    return tuple(active)


def raw_legacy_population_controls(argv: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Return raw M9 controls before resolved-config adapters discard them."""

    return tuple(
        argument
        for argument in argv
        if argument.startswith(LEGACY_POPULATION_CONTROL_ARG_PREFIXES)
    )


def reject_raw_live_population_conflicts(args: Any, argv: tuple[str, ...] | list[str]) -> None:
    """Reject raw legacy control tokens whenever live population is enabled."""

    if not bool(getattr(args, "streaming_population_live_enabled", False)):
        return
    conflicts = raw_legacy_population_controls(argv)
    if conflicts:
        raise ValueError(
            "live probabilistic population is incompatible with raw legacy "
            "population controls: " + ", ".join(conflicts)
        )
