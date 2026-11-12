"""Source attestation record (issue #1302).

A small, narrowly-scoped escape hatch for historical RGB-D bags (kitchen1,
floor2) that predate ``eligsir-rgbd-sync``'s capture-provenance gate and
cannot be re-recorded through it. See
``docs/architecture/2026-08-30-issue1302-legacy-attestation-design.md`` for
the full cross-repo design.

This module is deliberately the same shape as
``utils.comparison_report.git_provenance_from_args`` /
``determinism_from_args``: a passive record built once from ``args`` at
process startup, never a gate. It does not (and cannot, on its own) verify
anything about the actual capture -- it only makes an operator's explicit
declaration machine-readable and durable in ``report.json``, and caps
``evidence_class`` so that declaration can never silently launder into a
canonical/managed result.

Valid statuses:

- ``"unknown"`` (default): no attestation information at all. This is the
  bottom-of-lattice default (mirrors ``utils.experiment_validity``'s
  ``UNKNOWN``/``UNDECLARED`` convention) -- every run that doesn't pass
  ``--source_attestation_status`` reads back exactly this, so existing runs
  are completely unaffected by this feature's addition.
- ``"legacy_unattested"``: an operator has explicitly declared this run
  consumes a historical, unattested capture (e.g. kitchen1/floor2 bags with
  no ``CaptureProvenance``/``validated_source`` signal at all). Forces
  ``evidence_class`` to ``"UNMANAGED"`` regardless of ResearchFlow
  registration -- see ``cap_evidence_class`` below.
- ``"validated"``: reserved for a future ROS2-side producer that has
  actually confirmed hardware capture-provenance attestation end to end.
  Nothing in this repo sets this today; a caller passing it gets it recorded
  verbatim, but this module does not itself verify the claim.
"""

from __future__ import annotations

from typing import Any, Optional

VALID_STATUSES = ("unknown", "legacy_unattested", "validated")

# Statuses that must never be allowed to make a run's evidence_class MANAGED,
# no matter what ResearchFlow registration/scientific_mode says.
_NON_CANONICAL_STATUSES = ("legacy_unattested", "unknown")


def source_attestation_from_args(args: Any) -> Optional[dict]:
    """Build the ``source_attestation`` kwarg for
    ``write_post_training_report()`` from an argparse-style ``args``
    namespace carrying ``source_attestation_status`` /
    ``source_attestation_bag_id`` / ``source_attestation_reason`` (the
    ``--source_attestation_*`` CLI flags, ``arguments/__init__.py``).

    Returns ``None`` (never partial) if ``args`` lacks the status attribute
    entirely, so a call site built from a hand-rolled namespace degrades to
    the pre-existing (absent-key) behavior rather than writing a misleading
    default -- mirrors ``git_provenance_from_args``'s contract.

    An invalid/unrecognized status string is coerced to ``"unknown"`` rather
    than raising, with the original value preserved in
    ``invalid_status_value`` so a malformed launch is visible in the report
    instead of crashing a multi-hour run over a typo.
    """
    if args is None:
        return None
    status = getattr(args, "source_attestation_status", None)
    if status is None:
        return None
    result: dict = {}
    if status not in VALID_STATUSES:
        result["status"] = "unknown"
        result["invalid_status_value"] = status
    else:
        result["status"] = status
    bag_id = getattr(args, "source_attestation_bag_id", "") or None
    reason = getattr(args, "source_attestation_reason", "") or None
    if bag_id:
        result["legacy_source_bag_id"] = bag_id
    if reason:
        result["reason"] = reason
    return result


def cap_evidence_class(evidence_class: str, source_attestation: Optional[dict]) -> str:
    """Cap ``evidence_class`` at ``"UNMANAGED"`` whenever the resolved
    ``source_attestation`` status is not ``"validated"``.

    Mirrors ``utils.experiment_validity``'s ``effective = more conservative
    of declared and achieved`` pattern: this is the run's OWN ceiling, not a
    replacement for whatever ``utils.run_summary.write_run_summary`` already
    derived -- it can only push the class down (to ``UNMANAGED``), never up.
    A run with no ``source_attestation`` at all (``None``, i.e. this repo's
    pre-#1302 behavior) is passed through unchanged.
    """
    if not isinstance(source_attestation, dict):
        return evidence_class
    status = source_attestation.get("status")
    if status in _NON_CANONICAL_STATUSES:
        return "UNMANAGED"
    return evidence_class
