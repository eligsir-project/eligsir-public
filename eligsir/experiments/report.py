"""Experiment report writer.

Bundles a manifest + arbitrary results dict into one JSON report, with
``validity`` surfaced at the top level so CI/grep tooling doesn't need to
know the manifest's internal shape to gate on run validity -- mirrors the
project's convention of surfacing ``eval_valid``/``map_reinit_invalidated``
directly in ``report.json`` (see CLAUDE.md's Run Contracts section).

``write_experiment_report`` validates the manifest (via
:func:`eligsir.experiments.validation.validate_manifest`) BEFORE
writing, by default -- a report must never be written from a manifest that
fails the golden-rule check (e.g. under-declared changed_fields). Callers
that genuinely need to skip validation (e.g. writing a known-invalid
manifest for a test fixture) must use
:func:`write_experiment_report_unchecked` explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eligsir.experiments.manifest import ExperimentManifest
from eligsir.experiments.validation import validate_manifest


def build_experiment_report(manifest: ExperimentManifest, results: dict[str, Any]) -> dict[str, Any]:
    """Build the report dict without writing it (useful for tests/composition)."""
    manifest_dict = manifest.to_dict()
    return {
        "manifest": manifest_dict,
        "results": results,
        "validity": manifest_dict["validity"],
    }


def write_experiment_report_unchecked(
    manifest: ExperimentManifest,
    results: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a JSON experiment report to ``output_path`` WITHOUT validating.

    Skips :func:`eligsir.experiments.validation.validate_manifest`
    entirely -- only use this when the caller has an explicit reason to
    write a report from a manifest it knows/expects may be invalid (e.g. a
    test fixture exercising a downstream reader). Prefer
    :func:`write_experiment_report` for all real experiment runs.
    """
    report = build_experiment_report(manifest, results)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return out_path


def write_experiment_report(
    manifest: ExperimentManifest,
    results: dict[str, Any],
    output_path: str | Path,
    arm_configs: dict[str, dict[str, Any]] | None = None,
    strict: bool = False,
) -> Path:
    """Validate ``manifest`` and write a JSON experiment report to ``output_path``.

    Raises :class:`eligsir.experiments.validation.ManifestValidationError`
    (fails loudly, writes nothing) if the manifest doesn't pass
    ``validate_manifest``. Pass ``arm_configs`` to additionally audit
    declared vs. actual changed fields per arm, and ``strict=True`` to
    reject unresolved-placeholder fields (e.g. ``repository_commit ==
    "unknown"``) -- see :func:`eligsir.experiments.validation.validate_manifest`.

    Returns the path written.
    """
    validate_manifest(manifest, arm_configs=arm_configs, strict=strict)
    return write_experiment_report_unchecked(manifest, results, output_path)
