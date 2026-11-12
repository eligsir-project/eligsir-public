"""Config-diff calculator: cross-check declared vs. actual changed fields.

Given a baseline config dict and a treatment config dict, computes the set
of dotted key paths that actually differ (e.g. ``"training.lr"``). This is
compared against a manifest arm's declared ``changed_fields`` to catch the
golden-rule violation this whole PR exists to prevent: a manifest that
UNDER-declares what it changed (the real diff has fields the manifest
doesn't mention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_MISSING = object()


def _flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into {dotted.path: leaf_value}."""
    flat: dict[str, Any] = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and value:
            flat.update(_flatten(value, path))
        else:
            flat[path] = value
    return flat


def diff_config_paths(baseline: dict[str, Any], treatment: dict[str, Any]) -> set[str]:
    """Return the set of dotted key paths that differ between two configs.

    Handles nested dicts (recursing into them) and top-level keys added or
    removed on either side. A leaf value that differs, or that exists on
    only one side, counts as one changed path.
    """
    flat_baseline = _flatten(baseline)
    flat_treatment = _flatten(treatment)
    all_paths = set(flat_baseline) | set(flat_treatment)
    changed: set[str] = set()
    for path in all_paths:
        left = flat_baseline.get(path, _MISSING)
        right = flat_treatment.get(path, _MISSING)
        if left is _MISSING or right is _MISSING or left != right:
            changed.add(path)
    return changed


@dataclass
class ChangedFieldsAudit:
    """Result of cross-checking declared vs. actual config changes.

    Args:
        actual_changed: Dotted paths that really differ between the two
            configs.
        declared_changed: Dotted paths the manifest arm claims it changed.
        under_declared: Real diffs missing from ``declared_changed`` — the
            golden-rule violation (manifest hides a real change).
        over_declared: Declared paths that don't actually differ — not a
            golden-rule violation but worth surfacing (stale/no-op entry).
    """

    actual_changed: set[str]
    declared_changed: set[str]
    under_declared: set[str]
    over_declared: set[str]

    @property
    def is_under_declared(self) -> bool:
        return bool(self.under_declared)


def audit_changed_fields(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    declared_changed_fields: list[str],
) -> ChangedFieldsAudit:
    """Cross-check a manifest arm's declared changes against a real config diff."""
    actual = diff_config_paths(baseline, treatment)
    declared = set(declared_changed_fields)
    return ChangedFieldsAudit(
        actual_changed=actual,
        declared_changed=declared,
        under_declared=actual - declared,
        over_declared=declared - actual,
    )
