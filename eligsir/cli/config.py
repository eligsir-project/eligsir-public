"""Config and run-profile loading for EliGSiR CLI entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config/profile cannot be resolved or loaded."""


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_ROOT = _REPO_ROOT / "configs" / "profiles"

_PROFILE_ALIASES = {
    # `live/orbbec_ros2_rviz` was removed by the #1076 ROS2 hard cut, which
    # deleted configs/profiles/live/orbbec_ros2_rviz.yml along with the core
    # ros2_rviz viewer it configured. The alias outlived its file, so
    # known_run_profiles() advertised a profile that could only ever raise
    # ConfigError. ROS/RViz live runs are launched from eligsir-ros2.
    "offline_reconstruction/tum_rgbd": (
        _PROFILE_ROOT / "offline_reconstruction" / "tum_rgbd.yml"
    ),
    "offline_reconstruction/tum_rgbd.yml": (
        _PROFILE_ROOT / "offline_reconstruction" / "tum_rgbd.yml"
    ),
    "causal_replay/tum_rgbd": _PROFILE_ROOT / "causal_replay" / "tum_rgbd.yml",
    "causal_replay/tum_rgbd.yml": _PROFILE_ROOT / "causal_replay" / "tum_rgbd.yml",
    "causal_replay/scannet": _PROFILE_ROOT / "causal_replay" / "scannet.yml",
    "causal_replay/scannet.yml": _PROFILE_ROOT / "causal_replay" / "scannet.yml",
    "causal_replay/replica": _PROFILE_ROOT / "causal_replay" / "replica.yml",
    "causal_replay/replica.yml": _PROFILE_ROOT / "causal_replay" / "replica.yml",
    "causal_replay/scannetpp": _PROFILE_ROOT / "causal_replay" / "scannetpp.yml",
    "causal_replay/scannetpp.yml": _PROFILE_ROOT / "causal_replay" / "scannetpp.yml",
    "live/canonical_mapping": _PROFILE_ROOT / "live" / "canonical_mapping.yml",
    "live/canonical_mapping.yml": _PROFILE_ROOT / "live" / "canonical_mapping.yml",
    "bench/r9700_streaming": _PROFILE_ROOT / "bench" / "r9700_streaming.yml",
    "bench/r9700_streaming.yml": _PROFILE_ROOT / "bench" / "r9700_streaming.yml",
}


# Removed runtime names are tombstones, never executable compatibility aliases
# (#1073 hard-cut amendment to issue #168).
_REMOVED_PROFILE_ALIASES = {
    "offline/tum_rgbd": "causal_replay/tum_rgbd",
    "offline/tum_rgbd.yml": "causal_replay/tum_rgbd",
}


def _removed_profile_error(profile: str, replacement: str) -> ConfigError:
    return ConfigError(
        f"Run profile {profile!r} was removed because 'offline' falsely named "
        "a streaming causal-replay path. Use "
        f"--run_profile {replacement} for causal replay, or "
        "--run_profile offline_reconstruction/tum_rgbd for true all-data-at-start "
        "reconstruction. Historical reproduction must use the original pinned revision."
    )


def known_run_profiles() -> tuple[str, ...]:
    """Return stable run-profile aliases accepted by ``--run_profile``."""

    return tuple(sorted(k for k in _PROFILE_ALIASES if not k.endswith(".yml")))


def known_execution_profiles() -> tuple[str, ...]:
    """Executable semantics leaves; ``run_intent/*`` reporting mixins excluded."""

    return known_run_profiles()


def migrate_historical_execution_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return an explicit execution-trainer migration for a historical config.

    This helper is deliberately *not* used by the loader or CLI.  Historical
    ``streaming_replay`` configurations must be reviewed and re-authored, not
    silently reinterpreted while launching a run.  It only performs the
    mechanical trainer-name conversion required by the canonical runtime cut.
    """

    migrated = dict(config)
    legacy = migrated.pop("streaming_replay", None)
    if legacy is None:
        return migrated
    if not isinstance(legacy, bool):
        raise ConfigError("historical streaming_replay must be boolean to migrate")
    migrated["execution_trainer"] = (
        "causal_mapping" if legacy else "offline_reconstruction"
    )
    return migrated


def resolve_config_path(config_file: str | None = None, run_profile: str | None = None) -> Path | None:
    """Resolve ``--config`` or ``--run_profile`` to a concrete config path."""

    if config_file and run_profile:
        raise ConfigError("Use either --config or --run_profile, not both.")
    if run_profile:
        profile = str(run_profile).strip()
        replacement = _REMOVED_PROFILE_ALIASES.get(profile)
        if replacement is not None:
            raise _removed_profile_error(profile, replacement)
        path = _PROFILE_ALIASES.get(profile)
        if path is None:
            candidate = Path(profile).expanduser()
            path = candidate if candidate.suffix else _PROFILE_ROOT / f"{profile}.yml"
        return path
    if config_file:
        raw = str(config_file).replace("\\", "/")
        if raw.endswith("configs/profiles/offline/tum_rgbd.yml"):
            raise _removed_profile_error(
                "configs/profiles/offline/tum_rgbd.yml",
                "causal_replay/tum_rgbd",
            )
        return Path(config_file).expanduser()
    return None


def load_yaml(path) -> dict:
    """Load a single YAML config file as a mapping."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    if not p.exists():
        raise ConfigError(f"Config file does not exist: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {p}")
    return data


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge ``over`` onto ``base``; later wins. Pure (no mutation)."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand(path, _stack):
    p = Path(path).expanduser().resolve()
    if p in _stack:
        chain = " -> ".join(str(s) for s in _stack + [p])
        raise ConfigError(f"extends cycle detected: {chain}")
    data = load_yaml(p)
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    layers = []
    for parent in parents:
        parent_path = (p.parent / parent) if not Path(parent).is_absolute() else Path(parent)
        layers.extend(_expand(parent_path, _stack + [p]))
    layers.append(data)  # parents first (lower priority), then this file
    return layers


def resolve_layers(config_paths) -> dict:
    """Resolve extends: chains for each path and deep-merge all layers.

    Order low->high: parents before children; earlier --config before later.
    """
    merged: dict = {}
    for cp in config_paths:
        for layer in _expand(cp, []):
            merged = deep_merge(merged, layer)
    return merged


_DEFAULT_BASE = _PROFILE_ROOT / "default.yml"


def _resolve_config_paths(config_files=None, run_profile: str | None = None):
    """Ordered top-level config paths (before ``extends`` expansion) that a
    ``load_cli_config(config_files, run_profile)`` call would consider:
    implicit ``default.yml`` base, then ``--run_profile``, then each
    ``--config`` file, in that order. ``resolve_config_path()`` results are
    guarded (``None`` is never appended).

    Single shared walk for ``load_cli_config()`` (which merges these paths)
    and ``resolve_profile_stack()`` (which reports them) -- see issue #591:
    the two used to duplicate this walk and had already drifted (an
    unguarded append here vs. a guarded one there). Returns
    ``(paths, normalized_config_files)``; the caller also needs the
    normalized list to evaluate ``_is_default_only()``.
    """
    if isinstance(config_files, (str, Path)):
        config_files = [config_files]
    config_files = config_files or []

    paths: list[Path] = []
    if _DEFAULT_BASE.exists():
        paths.append(_DEFAULT_BASE)
    if run_profile:
        rp = resolve_config_path(run_profile=run_profile)
        if rp is not None:
            paths.append(rp)
    for cf in config_files:
        if cf:
            cp = resolve_config_path(config_file=cf)
            if cp is not None:
                paths.append(cp)
    return paths, config_files


def _is_default_only(paths, run_profile, config_files) -> bool:
    """True iff ``paths`` is exactly the implicit ``default.yml`` base (or
    empty, if no default.yml exists) with no ``--run_profile``/``--config``
    given -- i.e. nothing the caller actually asked for. Both
    ``load_cli_config()`` and ``resolve_profile_stack()`` treat this as "no
    layer was merged", so this predicate must be shared, not duplicated
    (issue #591): a plain no-flag run reports an EMPTY profile stack next to
    an empty effective_config, rather than claiming default.yml was applied
    when it deliberately was not merged.
    """
    has_base = _DEFAULT_BASE.exists()
    return len(paths) == (1 if has_base else 0) and not run_profile and not config_files


def _expand_paths(path, _stack):
    """Same extends-chain walk as ``_expand``, but returning the ordered list
    of resolved ``Path`` objects (parents first) rather than loaded data.

    Deliberately a SEPARATE walk rather than a refactor of ``_expand`` itself:
    keeps this addition zero-risk to ``load_cli_config()``'s existing,
    extensively-tested merge behavior (issue #524 acceptance criterion 3 needs
    only the path list, for ``report.json["run_intent"]["profile_stack"]``).
    """
    p = Path(path).expanduser().resolve()
    if p in _stack:
        chain = " -> ".join(str(s) for s in _stack + [p])
        raise ConfigError(f"extends cycle detected: {chain}")
    data = load_yaml(p)
    parents = data.get("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    layers = []
    for parent in parents:
        parent_path = (p.parent / parent) if not Path(parent).is_absolute() else Path(parent)
        layers.extend(_expand_paths(parent_path, _stack + [p]))
    layers.append(p)
    return layers


def resolve_profile_stack(config_files=None, run_profile: str | None = None) -> list[str]:
    """Return the ordered list of resolved config layer paths (parents before
    children, repo-relative strings where possible) that
    ``load_cli_config(config_files, run_profile)`` would merge for the SAME
    arguments -- mirrors that function's own path-resolution exactly (the
    unconditional ``default.yml`` prepend, ``resolve_config_path()`` per
    argument, then each layer's ``extends`` chain) without touching or
    duplicating its merge logic. Used to populate
    ``report.json["run_intent"]["profile_stack"]`` (issue #524 acceptance
    criterion 3).

    Returns an empty list on the "default-only" no-flag path -- see
    ``_is_default_only()`` -- since ``load_cli_config()`` deliberately does
    NOT merge ``default.yml`` in that case; reporting it as a layer here
    would claim a merge that never happened (issue #591).
    """
    paths, config_files = _resolve_config_paths(config_files, run_profile)
    if _is_default_only(paths, run_profile, config_files):
        return []

    stack: list = []
    for p in paths:
        for layer_path in _expand_paths(p, []):
            if layer_path not in stack:
                stack.append(layer_path)

    out: list[str] = []
    for lp in stack:
        try:
            out.append(str(lp.relative_to(_REPO_ROOT)))
        except ValueError:
            out.append(str(lp))
    return out


def load_config(config_file: str | Path) -> dict[str, Any]:
    """Load a single YAML config file as a mapping.

    JSON profiles were dropped in favor of layered YAML; ``.json`` paths raise.
    """

    path = Path(config_file).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".json":
        raise ConfigError(
            f"JSON config profiles are no longer supported: {path}. "
            "Migrate to a layered YAML profile (.yml) under configs/profiles."
        )
    if suffix and suffix not in {".yaml", ".yml"}:
        raise ConfigError(f"Unsupported config extension for {path}; expected .yml/.yaml.")
    return load_yaml(path)


def load_cli_config(config_files=None, run_profile: str | None = None) -> dict[str, Any]:
    """Load + merge config layers.

    Implicit ``default.yml`` base, then ``--run_profile`` (with its extends
    chain), then each ``--config`` file in order; later wins.

    On a plain no-flag run (no ``--run_profile``/``--config``, only
    ``default.yml`` resolvable), deliberately returns ``{}`` rather than
    merging ``default.yml`` -- see ``_is_default_only()`` for why, and
    ``resolve_profile_stack()`` for the reporting counterpart that must
    (and now does) agree with this.
    """

    paths, config_files = _resolve_config_paths(config_files, run_profile)
    if _is_default_only(paths, run_profile, config_files):
        return {}
    return resolve_layers(paths)
