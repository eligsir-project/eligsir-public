"""Validation of :class:`eligsir.experiments.manifest.ExperimentManifest`.

Enforces the golden rule of ablations (plan section 3): every non-baseline
arm must declare what it changed, and the manifest must carry the hashes
required by its declared freeze level. Violations raise loudly — this is a
CI gate, not a warning.
"""

from __future__ import annotations

import re
from typing import Any

from eligsir.experiments.frozen_artifact import audit_changed_fields
from eligsir.experiments.manifest import ExperimentManifest

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

# Fields that, when a manifest is validated with strict=True, must not be
# the literal placeholder string "unknown" -- a manifest intended to be
# quotable (e.g. for a real experiment report) must have actually resolved
# these, not silently fallen back.
_STRICT_NO_UNKNOWN_FIELDS: tuple[str, ...] = ("repository_commit",)

_VALID_FROZEN_LEVELS = ("A", "B", "C", "D")

# Level -> additional hash fields required beyond the previous level's set.
_LEVEL_REQUIRED_HASHES: dict[str, tuple[str, ...]] = {
    "A": ("source_artifact_hash", "pose_hash", "holdout_hash"),
    "B": ("keyframe_hash",),
    "C": ("initial_checkpoint_hash", "config_hash"),
    "D": ("mutation_trace_hash",),
}


class ManifestValidationError(ValueError):
    """Raised when an experiment manifest fails validation."""


def _cumulative_required_hashes(frozen_level: str) -> tuple[str, ...]:
    idx = _VALID_FROZEN_LEVELS.index(frozen_level)
    required: list[str] = []
    for level in _VALID_FROZEN_LEVELS[: idx + 1]:
        required.extend(_LEVEL_REQUIRED_HASHES[level])
    return tuple(required)


def _is_sha256_hex(value: str | None) -> bool:
    return isinstance(value, str) and bool(_SHA256_HEX_RE.match(value))


def validate_manifest(
    manifest: ExperimentManifest,
    arm_configs: dict[str, dict[str, Any]] | None = None,
    strict: bool = False,
) -> None:
    """Validate ``manifest``. Raises :class:`ManifestValidationError` on failure.

    Checks:
      - ``frozen_level`` is one of A/B/C/D.
      - every hash required by that freeze level (cumulative, per plan
        section 3.2) is present and looks like a 64-hex-char SHA-256 digest.
      - a ``baseline`` arm exists with empty ``changed_fields``.
      - every non-baseline arm declares a non-empty ``changed_fields``.
      - no two arms declare identical ``changed_fields`` (a manifest where
        a "treatment" arm's changed_fields matches baseline's, i.e. both
        empty, or matches another arm's exactly, hides a real difference).
      - if ``arm_configs`` is given (``{arm_name: actual_config_dict}``),
        cross-checks each non-baseline arm's declared ``changed_fields``
        against the real config diff vs. the ``baseline`` arm's config
        (:func:`eligsir.experiments.frozen_artifact.audit_changed_fields`)
        and raises if any field is under-declared -- the golden-rule
        violation this whole module exists to catch. Arms missing from
        ``arm_configs`` (or missing the ``"baseline"`` key) are skipped,
        since not every caller has actual configs available.

    Args:
        manifest: The manifest to validate.
        arm_configs: Optional ``{arm_name: actual_config_dict}`` map used to
            audit declared vs. actual changed fields.
        strict: When True, additionally rejects manifests where a field
            meant to be quotable (currently just ``repository_commit``) is
            still the unresolved placeholder string ``"unknown"``. Use this
            for any manifest/report intended to be cited as real evidence;
            leave it False for smoke tests / dev runs where "unknown" is an
            acceptable degraded value.
    """
    if manifest.frozen_level not in _VALID_FROZEN_LEVELS:
        raise ManifestValidationError(
            f"frozen_level must be one of {_VALID_FROZEN_LEVELS}, got {manifest.frozen_level!r}"
        )

    if strict:
        for field_name in _STRICT_NO_UNKNOWN_FIELDS:
            if getattr(manifest, field_name, None) == "unknown":
                raise ManifestValidationError(
                    f"strict validation: {field_name!r} is the unresolved placeholder "
                    "'unknown' -- this manifest cannot be treated as quotable evidence"
                )

    manifest_dict = manifest.to_dict()
    required_hashes = _cumulative_required_hashes(manifest.frozen_level)
    missing = [
        field_name
        for field_name in required_hashes
        if not _is_sha256_hex(manifest_dict.get(field_name))
    ]
    if missing:
        raise ManifestValidationError(
            f"manifest at frozen_level={manifest.frozen_level!r} is missing required "
            f"sha256 hash field(s): {missing}"
        )

    if not manifest.arms:
        raise ManifestValidationError("manifest must declare at least a 'baseline' arm")

    arm_names = [arm.name for arm in manifest.arms]
    if "baseline" not in arm_names:
        raise ManifestValidationError("manifest must declare an arm named 'baseline'")

    seen_changed_fields: dict[tuple[str, ...], str] = {}
    for arm in manifest.arms:
        changed_key = tuple(sorted(arm.changed_fields))

        if arm.name == "baseline":
            if arm.changed_fields:
                raise ManifestValidationError(
                    f"'baseline' arm must have empty changed_fields, got {arm.changed_fields!r}"
                )
        else:
            if not arm.changed_fields:
                raise ManifestValidationError(
                    f"arm {arm.name!r} is not 'baseline' but declares no changed_fields; "
                    "every non-baseline arm must declare what it changed"
                )

        if changed_key in seen_changed_fields:
            other = seen_changed_fields[changed_key]
            raise ManifestValidationError(
                f"arm {arm.name!r} declares changed_fields identical to arm {other!r} "
                f"({list(changed_key)!r}); this hides a real difference between arms"
            )
        seen_changed_fields[changed_key] = arm.name

    if arm_configs and "baseline" in arm_configs:
        baseline_config = arm_configs["baseline"]
        for arm in manifest.arms:
            if arm.name == "baseline" or arm.name not in arm_configs:
                continue
            audit = audit_changed_fields(baseline_config, arm_configs[arm.name], arm.changed_fields)
            if audit.is_under_declared:
                raise ManifestValidationError(
                    f"arm {arm.name!r} under-declares changed_fields: actual config diff "
                    f"includes {sorted(audit.under_declared)!r} which is not in "
                    f"declared changed_fields {arm.changed_fields!r} -- this hides a real "
                    "config change (golden-rule violation)"
                )
