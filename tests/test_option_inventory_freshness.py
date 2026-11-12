"""`docs/inventory/option_inventory.json` must cover every current argparse option.

The inventory is the machine-readable artifact issue #182's "required repo-wide
audit" asks for. It was committed by #184 (part 1) with no regeneration
enforcement at all -- which is the *identical* silent-drift class #182 was filed
about for `defaults.reference.yml`. Measured 2026-08-14 (#182 part 2): the
committed inventory described 739 options while argparse declared 761, i.e. 22
options had been added since and were absent from the audit with nothing
reporting it.

SCOPE -- deliberately the option NAME SET, not the whole file.

The inventory also carries `production_readers` / `test_doc_readers` grep counts,
which move whenever *any* production file mentions an option name in passing.
A byte-identity or full-value test on those would fail on nearly every unrelated
PR, get muted, and protect nothing. The name set is the stable invariant that
actually encodes "every option is accounted for in the audit", so that is what is
enforced here. Refreshing the counts is a maintenance action
(`python3 scripts/build_option_inventory.py`), not a correctness gate.

Torch-free by construction (json + `arguments/`, which imports no torch), so this
runs in the lean CPU CI image rather than being silently excluded by
`tests/conftest.py` -- see issue #536.
"""
from __future__ import annotations

import json

from scripts.build_option_inventory import OUT_DIR
from scripts.dump_default_config import collect_defaults

INVENTORY_PATH = OUT_DIR / "option_inventory.json"
SUMMARY_PATH = OUT_DIR / "option_inventory_summary.json"

REGEN_COMMAND = "python3 scripts/build_option_inventory.py"


def _inventory() -> list[dict]:
    return json.loads(INVENTORY_PATH.read_text())


def test_inventory_covers_every_argparse_option():
    """No option may exist in argparse without a row in the audit."""
    rows = _inventory()
    inventoried = {row["name"] for row in rows}
    current = set(collect_defaults())

    missing = sorted(current - inventoried)
    extra = sorted(inventoried - current)
    assert not missing and not extra, (
        f"{INVENTORY_PATH.name} is stale relative to arguments/__init__.py.\n"
        f"Regenerate with:\n    {REGEN_COMMAND}\n"
        f"Options MISSING from the inventory ({len(missing)}): {missing}\n"
        f"Options in the inventory but no longer in argparse ({len(extra)}): {extra}"
    )


def test_inventory_rows_carry_the_issue_182_schema_fields():
    """The audit schema #182 specifies, so a field can't silently disappear."""
    required = {
        "name",
        "owner_group",
        "runtime_default",
        "profile_overrides",
        "profile_divergences",
        "spec_usage",
        "production_readers",
        "test_doc_readers",
        "reader_scope",
        "last_changed",
        "status",
    }
    rows = _inventory()
    assert rows, f"{INVENTORY_PATH.name} is empty"
    missing = required - set(rows[0])
    assert not missing, (
        f"{INVENTORY_PATH.name} rows are missing schema fields {sorted(missing)}. "
        f"Regenerate with:\n    {REGEN_COMMAND}"
    )


def test_status_is_never_guessed():
    """#182 forbids inventing a promotion status without its evidence gate.

    Statuses (canonical_default / experimental / negative_result / ...) require
    matched-seed evidence per option. Until that exists per row, every status
    stays explicitly `unclassified` rather than being guessed -- a guessed status
    would manufacture false authority in the audit artifact.
    """
    bad = sorted(
        {row["name"] for row in _inventory() if row["status"] != "unclassified"}
    )
    assert not bad, (
        f"{INVENTORY_PATH.name} assigns a non-`unclassified` status to {bad} "
        f"without the per-option evidence issue #182 requires. If a status is "
        f"genuinely earned, record the evidence in the write-up doc and relax "
        f"this test deliberately."
    )


def test_summary_is_consistent_with_the_inventory():
    """The derived aggregates must describe the same option set."""
    rows = _inventory()
    summary = json.loads(SUMMARY_PATH.read_text())
    assert summary["n_options"] == len(rows), (
        f"{SUMMARY_PATH.name} describes {summary['n_options']} options but "
        f"{INVENTORY_PATH.name} has {len(rows)}. Regenerate with:\n    {REGEN_COMMAND}"
    )
    by_scope = {row["name"]: row["reader_scope"] for row in rows}
    for name in summary["zero_production_readers"]:
        assert by_scope.get(name) == "none", (
            f"{SUMMARY_PATH.name} lists {name} as having no readers, but the "
            f"inventory scopes it {by_scope.get(name)!r}. Regenerate with:\n"
            f"    {REGEN_COMMAND}"
        )
