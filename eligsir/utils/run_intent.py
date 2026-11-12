"""Run-intent resolution + report persistence -- issue #524, acceptance criterion 3.

PR #550 (issue #524's declarative half) shipped the three canonical run
intents as resolvable, behaviour-neutral config profiles
(``configs/profiles/run_intent/{smoke_dev,canonical_online,paper_registered}.yml``,
selectable today via ``--run_profile run_intent/<name>``) plus their
non-argparse metadata in ``docs/inventory/run_intents.yml``. What was
explicitly deferred (see that issue's last comment, "What remains") was:

  1. a first-class ``--run_intent {smoke_dev,canonical_online,paper_registered}``
     flag, so a run does not have to spell out the ``run_intent/<name>`` profile
     alias by hand;
  2. persisting the resolved intent, its full profile stack, and its effective
     configuration into ``report.json["run_intent"]``, so a run answers "what
     evidence grade do you claim to be" from its own artifacts rather than by
     the reader reconstructing the CLI string that launched it.

This module implements both, and only both -- criteria 4-7 of #524 (a paper-run
VALIDATOR that can reject an ad-hoc intent, the non-argparse
``PAPER_REGISTERED`` guarantees, and genuinely composable ``run_intent`` +
``execution`` + ``sensor`` layers) are explicitly out of scope; see this
module's own functions for what each one does and does not do.

**No default changes here.** ``--run_intent`` is a NEW, opt-in flag (default
``None``) that, when set, resolves to exactly the same ``run_intent/<name>``
profile alias ``--run_profile`` already accepted before this module existed --
it does not add, remove, or alter a single key in any of the three profile
files. A run that never passes ``--run_intent`` gets
``report.json["run_intent"]["declared"] == "UNDECLARED"`` (the least
permissive value, matching ``--experiment_validity_class``'s own convention),
with ``profile_stack``/``effective_config`` reflecting whatever profile stack
(if any) that run actually resolved -- i.e. this module is a REPORTING
addition for every run, not a behavior change for any run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

# Kept in sync with tests/test_run_intent_profiles.py::CANONICAL_INTENTS and
# docs/inventory/run_intents.yml's `intents:` keys -- there is deliberately no
# import-time cross-check against either (both are pure data/test files with no
# stable importable constant), so a mismatch would show up as an
# --run_intent value with no docs/inventory/run_intents.yml metadata, handled
# gracefully below rather than raising.
KNOWN_RUN_INTENTS = ("smoke_dev", "canonical_online", "paper_registered")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUN_INTENTS_METADATA_PATH = _REPO_ROOT / "docs" / "inventory" / "run_intents.yml"


class RunIntentConflictError(ValueError):
    """Raised when --run_intent and an explicit --run_profile disagree."""


def run_intent_run_profile(run_intent: Optional[str]) -> Optional[str]:
    """Map an ``--run_intent`` value to the ``run_profile`` alias it resolves to.

    ``None``/empty maps to ``None`` (no profile implied by intent alone).
    """
    if not run_intent:
        return None
    return f"run_intent/{run_intent}"


def resolve_run_intent_and_profile(
    run_intent: Optional[str], run_profile: Optional[str]
) -> Optional[str]:
    """Reconcile ``--run_intent`` with an (optional) explicit ``--run_profile``.

    Returns the effective ``run_profile`` string to hand to
    ``load_cli_config()``/``resolve_profile_stack()``. When ``run_intent`` is
    unset, this is just ``run_profile`` unchanged (identical to every call site
    before this function existed). When both are set, they must agree -- e.g.
    ``--run_intent smoke_dev --run_profile run_intent/smoke_dev`` is redundant
    but harmless, while ``--run_intent smoke_dev --run_profile
    run_intent/paper_registered`` is refused (which intent should the run
    report itself as?) rather than silently picking one.
    """
    intent_profile = run_intent_run_profile(run_intent)
    if intent_profile is None:
        return run_profile
    if run_profile and run_profile != intent_profile:
        raise RunIntentConflictError(
            f"--run_intent {run_intent!r} resolves to --run_profile "
            f"{intent_profile!r}, which conflicts with the explicitly-passed "
            f"--run_profile {run_profile!r}. Pass only one, or make them agree."
        )
    return intent_profile


def _intent_metadata(run_intent: str) -> dict:
    """Best-effort lookup into docs/inventory/run_intents.yml. Never raises --
    a missing/malformed metadata file degrades to an empty dict so a report
    can still be written (the metadata is documentation, not a gate)."""
    try:
        if not _RUN_INTENTS_METADATA_PATH.exists():
            return {}
        data = yaml.safe_load(_RUN_INTENTS_METADATA_PATH.read_text()) or {}
        return dict((data.get("intents") or {}).get(run_intent) or {})
    except Exception:
        return {}


def run_intent_report_from_args(
    args: Any,
    *,
    effective_config: dict,
    profile_stack: Iterable[str],
    cli_explicit: Optional[Iterable[str]] = None,
) -> dict:
    """Build the ``report.json["run_intent"]`` block for this run.

    ``effective_config``: the merged profile-stack dict this run actually
    resolved (i.e. what ``load_cli_config()`` returned for the
    ``run_profile``/``config_files`` this run used) -- NOT the full
    ``vars(args)`` snapshot. This is deliberate: ``effective_config`` answers
    "what did the profile stack itself set", which is what #524 asks a run to
    self-report as its policy; the full resolved argument set (profile +
    argparse defaults + explicit CLI) is already recoverable from
    ``cfg_args``/``run_args``, so duplicating all ~700 argparse dests into this
    block would bury the profile-specific signal in noise.

    ``profile_stack``: ordered list of the config layer paths (repo-relative
    strings, parents before children) this run's ``effective_config`` was
    merged from -- e.g.
    ``["configs/profiles/default.yml", "configs/profiles/run_intent/paper_registered.yml"]``
    for a paper-registered run (``paper_registered.yml`` itself ``extends``
    ``canonical_online.yml``, so that layer appears too).

    ``cli_explicit``: the set of argparse dests that were passed explicitly on
    the command line for this run (as already computed by
    ``eligsir.cli.run_config.cli_explicit_dests`` at every call site). Used
    only to record ``cli_overridden_keys`` -- which profile-set keys this run's
    explicit CLI flags then overrode, per #524's "command-line overrides remain
    explicit and provenance-tracked" requirement. An empty/``None`` value
    degrades to an empty list, never an error.

    Never returns ``None`` -- an undeclared run still gets a block (``declared:
    "UNDECLARED"``), matching ``--experiment_validity_class``'s existing
    convention (see ``build_execution_semantics_block``/``experiment_validity``
    call sites) that "not declared" is a recorded, least-permissive state, not
    an absent field.
    """
    declared = getattr(args, "run_intent", None) or None
    meta = _intent_metadata(declared) if declared else {}
    cli_explicit = set(cli_explicit or ())
    overridden = sorted(k for k in effective_config if k in cli_explicit)
    return {
        "declared": declared or "UNDECLARED",
        "resolved_run_profile": run_intent_run_profile(declared),
        # docs/inventory/run_intents.yml's evidence_grade for this intent, or
        # "undeclared" when no --run_intent was passed at all. A declared
        # intent whose metadata is missing/unreadable (should not happen for
        # the three canonical intents, but this module never raises on it)
        # reads "unknown" rather than silently omitting the field.
        "evidence_grade": (
            "undeclared" if declared is None else meta.get("evidence_grade", "unknown")
        ),
        "paper_registered": bool(meta.get("paper_registered", False)),
        "profile_stack": list(profile_stack or ()),
        "effective_config": dict(effective_config or {}),
        "cli_overridden_keys": overridden,
    }
