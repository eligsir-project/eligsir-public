"""`docs/inventory/promotion_policy.yml` -- issue #524's graduation taxonomy.

WHAT THESE TESTS PROTECT

Issue #524's core principle is that **a merged implementation is not
automatically a default-on implementation**. That principle only has teeth if
it is mechanically checkable, otherwise "promoted" degrades into "somebody once
wrote it in a doc". These tests are the mechanical form of the four rules #524's
own Tests section names:

  * a feature cannot claim a `*_default` class while its promotion gate is
    unresolved;
  * a METHOD-CHANGING feature cannot graduate on smoke-only evidence (the
    measurement ladder is not available to it);
  * evidence invalidation can DOWNGRADE a previously-promoted feature without
    deleting the historical record;
  * the generated inventory column agrees with the registry, so the artifact
    cannot silently drift from the policy it claims to encode -- the identical
    staleness class #182 was filed about.

RELATIONSHIP TO `status` (#182). The inventory's pre-existing `status` column is
pinned at `unclassified` for every row by
`tests/test_option_inventory_freshness.py::test_status_is_never_guessed`, and
this file does NOT relax that. `graduation_state` is a separate column with its
own, stricter rule: it may be non-default ONLY for options this registry names,
and every registry entry must carry a citation. Neither column supersedes the
other.

Torch-free by construction (yaml + json + `arguments/`, which imports no torch),
so it executes in the lean CPU CI image rather than being silently excluded by
`tests/conftest.py` -- see issue #536.
"""
from __future__ import annotations

import json

import yaml

from scripts.build_option_inventory import (
    OUT_DIR,
    PROMOTION_POLICY_PATH,
    UNASSIGNED,
)
from scripts.dump_default_config import collect_defaults

INVENTORY_PATH = OUT_DIR / "option_inventory.json"
SUMMARY_PATH = OUT_DIR / "option_inventory_summary.json"
REGEN_COMMAND = "python3 scripts/build_option_inventory.py"

#: Classes that ASSERT the thing is default-on somewhere. Kept here as well as
#: in the YAML so a policy edit that quietly flips `promoted` on a class has to
#: face this list too.
PROMOTED_CLASSES = {
    "canonical_universal",
    "smoke_default",
    "canonical_online_default",
    "paper_default",
}

#: Evidence levels that are "smoke only" -- insufficient for a method-changing
#: feature to graduate under #202's ladder.
SMOKE_ONLY_LEVELS = {"none", "wiring_proof", "smoke"}


def _policy() -> dict:
    return yaml.safe_load(PROMOTION_POLICY_PATH.read_text())


def _entries() -> list[dict]:
    return _policy()["entries"]


def _inventory() -> list[dict]:
    return json.loads(INVENTORY_PATH.read_text())


# --------------------------------------------------------------------------- #
# The taxonomy itself
# --------------------------------------------------------------------------- #

def test_taxonomy_declares_every_class_issue_524_names():
    """#524's body lists the graduation states; none may quietly go missing."""
    required = {
        "canonical_universal",
        "smoke_default",
        "canonical_online_default",
        "paper_default",
        "profile_only",
        "telemetry_only",
        "experimental",
        "provisional_retest",
        "negative_result",
        "deprecated",
        "compatibility",
        "dark_unwired",
        "lost_reimplement",
    }
    classes = set(_policy()["promotion_classes"])
    missing = sorted(required - classes)
    assert not missing, (
        f"promotion_policy.yml dropped promotion classes issue #524 requires: {missing}"
    )
    assert UNASSIGNED in classes, (
        f"the default state {UNASSIGNED!r} must itself be a declared class, "
        "otherwise most inventory rows carry a state the taxonomy does not define"
    )


def test_every_class_declares_whether_it_asserts_a_default():
    """`promoted` is what the gate rules below key on; it cannot be optional."""
    for name, spec in _policy()["promotion_classes"].items():
        assert isinstance(spec.get("promoted"), bool), (
            f"promotion class {name!r} does not declare `promoted: true/false`"
        )
        assert spec.get("summary"), f"promotion class {name!r} has no summary"
    promoted = {n for n, s in _policy()["promotion_classes"].items() if s["promoted"]}
    assert promoted == PROMOTED_CLASSES, (
        f"the set of default-asserting classes changed: {sorted(promoted)} vs "
        f"{sorted(PROMOTED_CLASSES)}. That is a policy change, not a refactor -- "
        "update this test deliberately."
    )


def test_both_evidence_ladders_exist_and_are_distinguished():
    """#524's 'important distinction': measurement and method use DIFFERENT gates."""
    ladders = _policy()["gate_ladders"]
    assert set(ladders) == {"measurement_provenance", "method_changing"}
    assert len(ladders["measurement_provenance"]["rungs"]) == 5
    # #202's ladder, all eight rungs.
    assert len(ladders["method_changing"]["rungs"]) == 8
    assert (
        set(ladders["method_changing"]["smoke_only_validation_levels"])
        == SMOKE_ONLY_LEVELS
    )


# --------------------------------------------------------------------------- #
# Entry well-formedness
# --------------------------------------------------------------------------- #

def test_every_entry_carries_the_issue_524_schema():
    required = {
        "feature_or_option",
        "kind",
        "owner_issue",
        "training_semantics_changed",
        "live_gpu_cost_class",
        "validation_level",
        "required_gate",
        "graduation_state",
        "supersedes",
        "evidence",
    }
    for entry in _entries():
        missing = sorted(required - set(entry))
        assert not missing, (
            f"promotion_policy.yml entry {entry.get('feature_or_option')!r} is "
            f"missing schema fields {missing}"
        )
        assert entry["kind"] in ("option", "feature")
        assert isinstance(entry["training_semantics_changed"], bool)
        assert entry["live_gpu_cost_class"] in _policy()["live_gpu_cost_class"]
        assert entry["validation_level"] in _policy()["validation_levels"]
        assert entry["graduation_state"] in _policy()["promotion_classes"]


def test_every_entry_is_cited():
    """A graduation state without evidence is exactly the false authority #182 forbids."""
    uncited = [
        e["feature_or_option"]
        for e in _entries()
        if not str(e.get("evidence", "")).strip()
    ]
    assert not uncited, (
        f"promotion_policy.yml entries with no `evidence` citation: {uncited}. "
        "Record the run/issue/doc that justifies the state, or leave the option "
        "out of the registry entirely (it then reads `unassigned`, which is the "
        "honest answer for 'nobody has looked')."
    )


def test_option_entries_name_real_argparse_dests():
    """A typo'd option name would silently classify nothing at all."""
    dests = set(collect_defaults())
    bogus = [
        e["feature_or_option"]
        for e in _entries()
        if e["kind"] == "option" and e["feature_or_option"] not in dests
    ]
    assert not bogus, (
        f"promotion_policy.yml classifies {bogus} as `kind: option`, but they are "
        "not argparse dests. Either fix the name or set `kind: feature` (a "
        "capability with no CLI knob yet)."
    )


def test_feature_entries_name_a_blocking_issue():
    """`kind: feature` means 'not wired yet' -- it MUST say what unblocks it."""
    for entry in _entries():
        if entry["kind"] != "feature":
            continue
        assert str(entry["required_gate"]).strip(), (
            f"feature entry {entry['feature_or_option']!r} declares no "
            "`required_gate`. #524 requires a pending feature to name the issue "
            "that gates it, otherwise 'pending' is indistinguishable from 'forgotten'."
        )
        assert "#" in str(entry["required_gate"]), (
            f"feature entry {entry['feature_or_option']!r} has required_gate "
            f"{entry['required_gate']!r}, which names no issue number"
        )


# --------------------------------------------------------------------------- #
# The graduation rules #524's Tests section names
# --------------------------------------------------------------------------- #

def test_a_promoted_class_cannot_be_claimed_while_a_gate_is_unresolved():
    """#524: "A feature cannot be marked canonical_online_default while its
    promotion gate is unresolved." Generalized to every default-asserting class.
    """
    offenders = [
        (e["feature_or_option"], e["graduation_state"], e["required_gate"])
        for e in _entries()
        if e["graduation_state"] in PROMOTED_CLASSES
        and str(e["required_gate"]).strip()
    ]
    assert not offenders, (
        f"these entries claim a default-asserting class while still recording an "
        f"unresolved gate: {offenders}. Either the gate passed (clear "
        "`required_gate` and record the evidence) or the class is premature."
    )


def test_a_method_changing_feature_cannot_graduate_on_smoke_evidence():
    """#524: "A method-changing feature cannot graduate based only on smoke evidence."

    The measurement ladder is unavailable to anything that changes admission,
    replay, optimizer exposure, Gaussian topology, loss terms, MCMC behaviour,
    insertion, or representation resolution -- those must clear #202's full
    ladder, and its first three rungs are explicitly not enough.
    """
    offenders = [
        (e["feature_or_option"], e["validation_level"], e["graduation_state"])
        for e in _entries()
        if e["training_semantics_changed"]
        and e["graduation_state"] in PROMOTED_CLASSES
        and e["validation_level"] in SMOKE_ONLY_LEVELS
    ]
    assert not offenders, (
        f"these method-changing entries claim a default class on smoke-only "
        f"evidence: {offenders}. #202's ladder requires at minimum a canonical "
        "baseline comparison on a real scene."
    )


def test_the_four_issue_493_downgrades_are_recorded_as_provisional_retest():
    """#524's "Required default-status corrections", verbatim.

    This is the evidence-INVALIDATION direction: each of these is still the
    live default and its value is unchanged; only the CLAIM is withdrawn. The
    historical record survives in each entry's `evidence` field, which is what
    #524 means by "without deleting historical records".
    """
    by_name = {e["feature_or_option"]: e for e in _entries()}
    expected_gates = {
        # young protection + the maturity/lifecycle bundle, classified individually
        "streaming_young_prot_iters": "#188",
        "streaming_young_reg_exempt_iters": "#188",
        "streaming_maturity_reg_ramp": "#188",
        "streaming_consolidation_burst": "#188",
        "streaming_lifecycle_enabled": "#188",
        # corrected energy-guided MCMC
        "densification_strategy": "#187",
        # restored rolling starvation floor
        "streaming_keyframe_rolling_window_size": "#313",
        "streaming_keyframe_rolling_window_min_admits": "#313",
    }
    for name, gate in expected_gates.items():
        assert name in by_name, (
            f"#524 requires {name!r} to carry a default-status correction; it is "
            "absent from promotion_policy.yml"
        )
        entry = by_name[name]
        assert entry["graduation_state"] == "provisional_retest", (
            f"{name!r} is graded {entry['graduation_state']!r}; #493/#524 require "
            "`provisional_retest` until its retest lands"
        )
        assert gate in entry["required_gate"], (
            f"{name!r} must name {gate} as its retest gate, got "
            f"{entry['required_gate']!r}"
        )
        assert entry["evidence"].strip(), (
            f"{name!r} is a downgrade, so the superseded evidence must be RECORDED, "
            "not deleted"
        )


def test_downgrading_changes_no_value():
    """A `provisional_retest` grade must never have been used to flip a default.

    This is the load-bearing safety property of the whole downgrade mechanism:
    it withdraws a CLAIM. If grading something provisional also changed what a
    run does, nobody could ever afford to record an honest downgrade.

    HONEST SCOPE: this currently passes VACUOUSLY. All eight `provisional_retest`
    options are method-changing keys, and `test_no_intent_sets_a_training_semantics_changing_key`
    already forbids an intent from setting any of those, so none could have been
    set here regardless. It is kept as a forward guard for the case a
    semantics-neutral option is later downgraded -- do not read it as evidence
    that the eight values are pinned. What actually pins them is that this PR
    edits no argparse default and no `default.yml` value at all.
    """
    inventory = {r["name"]: r for r in _inventory()}
    for entry in _entries():
        if entry["kind"] != "option" or entry["graduation_state"] != "provisional_retest":
            continue
        row = inventory[entry["feature_or_option"]]
        assert not any(
            "profiles/run_intent/" in p.replace("\\", "/")
            for p in row["profile_overrides"]
        ), (
            f"{entry['feature_or_option']!r} is graded provisional_retest but a "
            "run-intent profile sets it. A run intent must not carry a value whose "
            "evidence has been withdrawn -- that would smuggle an unproven default "
            "in under a new name."
        )


# --------------------------------------------------------------------------- #
# Inventory <-> registry consistency (the #182-style freshness half)
# --------------------------------------------------------------------------- #

def test_inventory_graduation_column_matches_the_registry():
    rows = {r["name"]: r for r in _inventory()}
    for entry in _entries():
        if entry["kind"] != "option":
            continue
        name = entry["feature_or_option"]
        assert name in rows, f"{name!r} vanished from the inventory"
        assert rows[name]["graduation_state"] == entry["graduation_state"], (
            f"{name!r}: inventory says {rows[name]['graduation_state']!r}, "
            f"promotion_policy.yml says {entry['graduation_state']!r}. "
            f"Regenerate with:\n    {REGEN_COMMAND}"
        )


def test_no_inventory_row_invents_a_graduation_state():
    """A state may appear in the artifact ONLY via the cited registry."""
    named = {e["feature_or_option"] for e in _entries() if e["kind"] == "option"}
    bad = sorted(
        r["name"]
        for r in _inventory()
        if r["graduation_state"] != UNASSIGNED and r["name"] not in named
    )
    assert not bad, (
        f"{bad} carry a graduation_state with no entry in promotion_policy.yml. "
        f"Regenerate with:\n    {REGEN_COMMAND}"
    )


def test_every_inventory_state_is_in_the_taxonomy():
    classes = set(_policy()["promotion_classes"])
    bad = sorted({r["graduation_state"] for r in _inventory()} - classes)
    assert not bad, f"inventory carries undeclared graduation states: {bad}"


def test_summary_pending_features_match_the_registry():
    summary = json.loads(SUMMARY_PATH.read_text())
    from_summary = {p["feature"] for p in summary["pending_features"]}
    from_registry = {e["feature_or_option"] for e in _entries() if e["kind"] == "feature"}
    assert from_summary == from_registry, (
        f"summary pending_features drifted from the registry.\n"
        f"Regenerate with:\n    {REGEN_COMMAND}"
    )
