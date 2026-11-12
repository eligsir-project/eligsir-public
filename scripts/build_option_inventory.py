#!/usr/bin/env python3
"""Mechanical CLI-option inventory for issue #182's "required repo-wide audit".

Emits, for every argparse option declared in arguments/__init__.py, only what can
be computed WITHOUT judgement:

    name                    argparse dest
    owner_group             ModelParams / PipelineParams / OptimizationParams / StreamingParams
    runtime_default         the bare-CLI default
    profile_overrides       {profile_path: value} across configs/**.yml
    profile_divergences     the subset of the above whose value != the argparse
                            default (#182's "profile values contradicting argparse")
    spec_usage              {spec_path: value} across specs/*.json
    production_readers      grep hits in production code (see caveats below)
    test_doc_readers        grep hits under tests/ and docs/
    reader_scope            production | test_or_doc_only | none -- separates
                            "referenced only by tests/docs" from "referenced
                            nowhere", two distinct #182 audit findings
    last_changed            YYYY-MM-DD from git blame of the declaration line
    status                  ALWAYS "unclassified" -- see below

Plus issue #524's promotion schema, joined on from the two files #524 owns
(`docs/inventory/promotion_policy.yml`, `configs/profiles/run_intent/*.yml`):

    graduation_state        #524's `current_status` slot, renamed so it cannot
                            be confused with `status` above. "unassigned" for
                            every option promotion_policy.yml does not name.
    owner_issue             issue that owns this option's promotion decision
    training_semantics_changed  which of the two evidence ladders applies
    live_gpu_cost_class     cost on the ACTIVE TRAINING GPU (final_eval_only is
                            a real answer, per #524's core principle)
    validation_level        strongest evidence recorded for the current value
    required_gate           the unresolved gate, if any
    supersedes              what this entry replaces
    run_intent_defaults     {intent: effective value} for SMOKE_DEV /
                            CANONICAL_ONLINE / PAPER_REGISTERED, resolved
                            through the real profile layering

NONE of these is guessed either: an option gets a graduation_state only when
promotion_policy.yml names it WITH a citation, and
`tests/test_promotion_policy.py` refuses a promoted class while an unresolved
`required_gate` is recorded.

A second artifact, `option_inventory_summary.json`, carries the derived
aggregates from #182's "audit in particular for" list (zero-production-reader
options, test/doc-only options, profile-vs-argparse contradictions, defaults
repeatedly overridden by specs). See `summarize()`.

`status` (canonical_default / experimental / negative_result / deprecated /
dark-unwired / lost) is deliberately NOT assigned here. It is a judgement call
requiring evidence review per option, and issue #182 gates behavioral promotions
behind real matched-seed GPU A/Bs. Emitting a guessed status would manufacture
false authority. A small hand-classified sample lives in the write-up doc
(docs/architecture/2026-08-12-issue182-defaults-freshness-and-inventory-part1.md),
not here.

CAVEATS on `production_readers` -- it is a LOWER BOUND and a rough signal only:
  * counted by literal name occurrence, so `getattr(args, "flag", default)`,
    f-string/`setattr` construction, and prefix-shared names are miscounted;
  * arguments/ is excluded (that's the declaration site, not a reader), as are
    tests/, docs/, specs/, configs/ and scripts/ (this file included);
  * a count of 0 is a *candidate* dark/unwired arg worth a human look, NOT proof.

Usage:
    python3 scripts/build_option_inventory.py            # -> docs/inventory/... .json + .csv
    python3 scripts/build_option_inventory.py --stdout   # JSON to stdout
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from scripts.dump_default_config import _GROUPS, _group_keys, collect_defaults  # noqa: E402

OUT_DIR = _REPO_ROOT / "docs" / "inventory"

#: Issue #524's promotion-class taxonomy + per-entry graduation registry.
PROMOTION_POLICY_PATH = OUT_DIR / "promotion_policy.yml"
#: Issue #524's three canonical run intents, as resolvable profiles.
RUN_INTENT_DIR = _REPO_ROOT / "configs" / "profiles" / "run_intent"

#: Value of `graduation_state` for any option the registry does not name.
#: Mirrors the pre-existing `status: unclassified` pin: "nobody has looked",
#: NOT "reviewed and found unremarkable".
UNASSIGNED = "unassigned"

# Directories searched for production readers. Deliberately excludes arguments/
# (declaration), tests/, docs/, specs/, configs/, scripts/.
_PRODUCTION_PATHS = [
    "train.py",
    "train_streaming.py",
    "train_ros_node.py",
    "render.py",
    "metrics.py",
    "eligsir",
    "utils",
    "scene",
    "streaming",
    "slam",
    "gaussian_renderer",
    "viz",
    "contracts",
    "loop_closure",
]


def _owner_groups() -> dict[str, str]:
    owner: dict[str, str] = {}
    for name, cls in _GROUPS:
        for key in _group_keys(cls):
            owner.setdefault(key, name)
    return owner


def _yaml_overrides() -> dict[str, dict[str, object]]:
    """{option: {profile_relpath: value}} across configs/**.yml."""
    out: dict[str, dict[str, object]] = {}
    for path in sorted((_REPO_ROOT / "configs").rglob("*.yml")):
        if path.name == "defaults.reference.yml":
            continue  # generated mirror of argparse, not an override
        try:
            data = yaml.safe_load(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        rel = str(path.relative_to(_REPO_ROOT))
        for key, value in data.items():
            if isinstance(key, str):
                out.setdefault(key, {})[rel] = value
    return out


def _run_intent_defaults() -> dict[str, dict[str, object]]:
    """{option: {intent_name: resolved_value}} for the three #524 run intents.

    Resolves each intent through the SAME layering a real run gets --
    ``configs/profiles/default.yml`` is prepended unconditionally by
    ``load_cli_config()``, and ``paper_registered`` additionally ``extends``
    ``canonical_online`` -- so this reports the effective value, not the value
    literally typed in the intent file. Testing the file in isolation would
    assert something no run ever sees.

    Deliberately imports ``eligsir.cli.config`` rather than re-implementing
    the merge: it is torch-free (verified), so this stays runnable in the lean
    CPU CI image, and a divergence between the audit's idea of layering and the
    trainer's would be exactly the silent-drift class #182 exists to stop.
    """
    from eligsir.cli import config as cli_config  # torch-free

    out: dict[str, dict[str, object]] = {}
    for name in sorted(RUN_INTENT_DIR.glob("*.yml")):
        intent = name.stem
        # Resolved through the `run_profile=` entry point specifically -- the same
        # one a real `--run_profile run_intent/<name>` run uses, and the same one
        # tests/test_run_intent_profiles.py asserts against. (`config_files=` is
        # equivalent here and was measured byte-identical, but keeping the audit
        # on the run path means it cannot silently diverge from it later.)
        merged = cli_config.load_cli_config(run_profile=f"run_intent/{intent}")
        for key, value in merged.items():
            out.setdefault(key, {})[intent] = value
    return out


def _promotion_policy() -> dict:
    """The #524 taxonomy + per-entry graduation registry, or {} if absent."""
    if not PROMOTION_POLICY_PATH.exists():
        return {}
    data = yaml.safe_load(PROMOTION_POLICY_PATH.read_text())
    return data if isinstance(data, dict) else {}


def _spec_usage() -> dict[str, dict[str, object]]:
    """{option: {spec_relpath: value}} across specs/*.json.

    Specs store options both as JSON keys and inside CLI-string fields
    (`extra_args`, `args`, ...), so both forms are scanned.
    """
    out: dict[str, dict[str, object]] = {}
    # `(?!--)` on the value group keeps a bare flag followed by another flag
    # (`--online_mode --cap_max 5000`) from capturing `--cap_max` as its value;
    # the `no-` prefix is argparse's BooleanOptionalAction negation, so it must
    # record False, not True.
    flag_re = re.compile(r"--(no-)?([A-Za-z0-9_]+)(?:[= ]+(?!--)(\S+))?")

    def record(option: str, rel: str, value: object) -> None:
        out.setdefault(option, {}).setdefault(rel, value)

    def walk(node: object, rel: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and not isinstance(value, (dict, list)):
                    record(key, rel, value)
                walk(value, rel)
        elif isinstance(node, list):
            for item in node:
                walk(item, rel)
        elif isinstance(node, str) and "--" in node:
            for negated, name, value in flag_re.findall(node):
                if value:
                    record(name, rel, value)
                else:
                    record(name, rel, not negated)

    for path in sorted((_REPO_ROOT / "specs").glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        walk(data, str(path.relative_to(_REPO_ROOT)))
    return out


#: Non-production trees, counted separately so "referenced only by tests/docs"
#: can be told apart from "referenced nowhere at all" -- two different findings
#: in issue #182's audit list, previously collapsed into one zero.
_NON_PRODUCTION_PATHS = ["tests", "docs"]


def _grep_counts(names: list[str], paths: list[str]) -> dict[str, int]:
    """Literal-occurrence counts under `paths`. Lower bound -- see module docstring.

    ``docs/inventory/`` is excluded because it is THIS SCRIPT'S OWN OUTPUT: the
    generated .json/.csv name every option, so counting them gave every single
    option >=3 self-hits and made ``reader_scope == "none"`` unreachable. That
    artifact briefly turned "18 options have no production reader" into the false
    reading "zero options are referenced nowhere" -- the scan was measuring
    itself. Deliberately NOT fixed by restricting to ``*.py``: genuine prose
    mentions in docs/*.md are exactly what this count is for.
    """
    existing = [p for p in paths if (_REPO_ROOT / p).exists()]
    counts: dict[str, int] = {}
    for name in names:
        if not existing:
            counts[name] = 0
            continue
        proc = subprocess.run(
            [
                "grep",
                "-rIoh",
                f"--exclude-dir={OUT_DIR.name}",
                "-F",
                "-e",
                name,
                *existing,
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        counts[name] = len(proc.stdout.splitlines())
    return counts


def _reader_counts(names: list[str]) -> dict[str, int]:
    """Production-code reader counts (unchanged behaviour; kept for compatibility)."""
    existing = [p for p in _PRODUCTION_PATHS if (_REPO_ROOT / p).exists()]
    counts: dict[str, int] = {}
    for name in names:
        proc = subprocess.run(
            ["grep", "-rIoh", "--include=*.py", "-F", "-e", name, *existing],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        counts[name] = len(proc.stdout.splitlines())
    return counts


def _last_changed() -> dict[str, str]:
    """{option: YYYY-MM-DD} from `git blame` of its declaration line.

    Fills issue #182's `last_changed` schema field. One blame pass over
    arguments/__init__.py is mapped to the `self.<name> = ...` line that declares
    each option, so this is the date that option's DEFAULT last moved (or the
    date it was introduced) -- not the date its behaviour last changed elsewhere.
    Options declared indirectly (not via a literal `self.<name> =`) get "".
    """
    decl_re = re.compile(r"^\s*self\.(_?)([A-Za-z0-9_]+)\s*=")
    proc = subprocess.run(
        ["git", "blame", "--line-porcelain", "--", "arguments/__init__.py"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {}

    out: dict[str, str] = {}
    author_time = ""
    for line in proc.stdout.splitlines():
        if line.startswith("author-time "):
            author_time = line.split(" ", 1)[1].strip()
        elif line.startswith("\t"):
            match = decl_re.match(line[1:])
            if match and author_time:
                date = datetime.datetime.fromtimestamp(
                    int(author_time), datetime.timezone.utc
                ).strftime("%Y-%m-%d")
                # Last declaration wins if a name is assigned more than once.
                out[match.group(2)] = date
    return out


def build() -> list[dict]:
    defaults = collect_defaults()
    owner = _owner_groups()
    overrides = _yaml_overrides()
    specs = _spec_usage()
    names = sorted(defaults)
    readers = _reader_counts(names)
    non_prod = _grep_counts(names, _NON_PRODUCTION_PATHS)
    changed = _last_changed()
    intents = _run_intent_defaults()
    policy = _promotion_policy()
    registry = {
        e["feature_or_option"]: e
        for e in policy.get("entries", [])
        if isinstance(e, dict) and e.get("feature_or_option")
    }

    rows = []
    for name in names:
        prod = readers[name]
        # A profile setting the SAME value as argparse is documentation; a
        # profile setting a DIFFERENT value is a real behavioural divergence and
        # is what issue #182's "profile values contradicting argparse" asks for.
        divergences = {
            path: value
            for path, value in overrides.get(name, {}).items()
            if value != defaults[name]
        }
        if prod > 0:
            scope = "production"
        elif non_prod[name] > 0:
            scope = "test_or_doc_only"
        else:
            scope = "none"
        entry = registry.get(name, {})
        rows.append(
            {
                "name": name,
                "owner_group": owner.get(name, "unknown"),
                "runtime_default": defaults[name],
                "profile_overrides": overrides.get(name, {}),
                "profile_divergences": divergences,
                "spec_usage": specs.get(name, {}),
                "production_readers": prod,
                "test_doc_readers": non_prod[name],
                "reader_scope": scope,
                "last_changed": changed.get(name, ""),
                "status": "unclassified",
                # --- issue #524's promotion schema ---------------------------
                # `graduation_state` fills #524's `current_status` slot under a
                # different name so it cannot be confused with the `status`
                # column above, which #182 pins at "unclassified" for every row.
                # Populated ONLY from docs/inventory/promotion_policy.yml, which
                # requires a citation per entry; everything else stays
                # "unassigned" rather than being guessed.
                "graduation_state": entry.get("graduation_state", UNASSIGNED),
                "owner_issue": entry.get("owner_issue", ""),
                # Which of the two evidence ladders applies. "" = no judgement
                # recorded -- deliberately NOT defaulted to False, which would
                # assert the permissive answer for every unreviewed option.
                "training_semantics_changed": entry.get("training_semantics_changed", ""),
                "live_gpu_cost_class": entry.get("live_gpu_cost_class", "unknown"),
                "validation_level": entry.get("validation_level", "none"),
                "required_gate": entry.get("required_gate", ""),
                "supersedes": entry.get("supersedes", ""),
                # Effective value under each canonical run intent, resolved
                # through the real layering (default.yml -> intent -> extends).
                "run_intent_defaults": intents.get(name, {}),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    """The derived aggregates issue #182's "audit in particular for" list asks for.

    Every field here is MECHANICAL. None of it is a status verdict: an option
    appearing under `zero_production_readers` is a CANDIDATE for a human look
    (the grep is a lower bound -- getattr/setattr/f-string construction are all
    miscounted), never proof that it is dark or unwired.
    """
    def named(pred):
        return sorted(r["name"] for r in rows if pred(r))

    # #182's "profile values contradicting argparse" means DRIFT -- a leaf
    # profile that silently disagrees with the parser. Issue #524's run-intent
    # profiles also diverge from argparse, but deliberately and with a written
    # rationale, so folding them into the same list would turn a drift report
    # into a mixed bag and train readers to ignore it. Split, not hidden: each
    # row's raw `profile_divergences` field still lists every source file.
    def _is_intent(path: str) -> bool:
        return "profiles/run_intent/" in path.replace("\\", "/")

    contradictions = {}
    intent_declared = {}
    for r in rows:
        drift = {k: v for k, v in r["profile_divergences"].items() if not _is_intent(k)}
        declared = {k: v for k, v in r["profile_divergences"].items() if _is_intent(k)}
        if drift:
            contradictions[r["name"]] = drift
        if declared:
            intent_declared[r["name"]] = declared

    # #524 promotion aggregates.
    by_state: dict[str, list[str]] = {}
    for r in rows:
        by_state.setdefault(r["graduation_state"], []).append(r["name"])
    policy = _promotion_policy()
    pending_features = [
        {
            "feature": e["feature_or_option"],
            "owner_issue": e.get("owner_issue", ""),
            "required_gate": e.get("required_gate", ""),
            "graduation_state": e.get("graduation_state", UNASSIGNED),
        }
        for e in policy.get("entries", [])
        if isinstance(e, dict) and e.get("kind") == "feature"
    ]
    # Specs overriding a default in >=5 files is a signal the default may be
    # wrong for real runs -- #182's "spec/champion values that repeatedly
    # override defaults". Still a signal, not a verdict.
    repeated_spec_overrides = {
        r["name"]: len(r["spec_usage"])
        for r in rows
        if len(r["spec_usage"]) >= 5
    }
    return {
        "n_options": len(rows),
        # Echoed so a reader of this file alone can tell "below threshold" apart
        # from "absent from specs entirely".
        "repeated_spec_override_threshold": 5,
        "zero_production_readers": named(lambda r: r["reader_scope"] == "none"),
        "test_or_doc_readers_only": named(lambda r: r["reader_scope"] == "test_or_doc_only"),
        "profile_values_contradicting_argparse": contradictions,
        # Issue #524: deliberate, documented intent policy -- NOT drift.
        "run_intent_declared_divergences": intent_declared,
        "run_intents": sorted(p.stem for p in RUN_INTENT_DIR.glob("*.yml")),
        # Counts for every class; NAMES only for the classified minority. Listing
        # all ~740 `unassigned` names would bury the 18 rows that carry a real,
        # cited judgement under the 97% that carry none.
        "graduation_state_counts": {k: len(v) for k, v in sorted(by_state.items())},
        "graduation_states": {
            k: sorted(v) for k, v in sorted(by_state.items()) if k != UNASSIGNED
        },
        "n_options_with_graduation_state": sum(
            1 for r in rows if r["graduation_state"] != UNASSIGNED
        ),
        # Features #524 wants default-on somewhere but which have no argparse
        # dest yet. Declared here so "pending behind issue N" is machine-readable
        # rather than only prose in a profile comment.
        "pending_features": pending_features,
        "repeated_spec_overrides": dict(
            sorted(repeated_spec_overrides.items(), key=lambda kv: -kv[1])
        ),
        "caveat": (
            "Mechanical/derivable only. Reader counts are literal-grep lower "
            "bounds; a zero is a candidate for review, not proof of dark code. "
            "No row carries a promotion status -- issue #182 gates those behind "
            "matched-seed GPU evidence."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true", help="print JSON instead of writing files")
    args = ap.parse_args()

    rows = build()
    if args.stdout:
        json.dump(rows, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "option_inventory.json"
    csv_path = OUT_DIR / "option_inventory.csv"
    summary_path = OUT_DIR / "option_inventory_summary.json"
    json_path.write_text(json.dumps(rows, indent=2, default=str, sort_keys=False) + "\n")
    summary_path.write_text(
        json.dumps(summarize(rows), indent=2, default=str, sort_keys=False) + "\n"
    )
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            [
                "name",
                "owner_group",
                "runtime_default",
                "profile_override_count",
                "profile_overrides",
                "profile_divergences",
                "spec_usage_count",
                "spec_usage",
                "production_readers",
                "test_doc_readers",
                "reader_scope",
                "last_changed",
                "status",
                "graduation_state",
                "owner_issue",
                "training_semantics_changed",
                "live_gpu_cost_class",
                "validation_level",
                "required_gate",
                "supersedes",
                "run_intent_defaults",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["name"],
                    row["owner_group"],
                    row["runtime_default"],
                    len(row["profile_overrides"]),
                    ";".join(f"{k}={v}" for k, v in row["profile_overrides"].items()),
                    ";".join(f"{k}={v}" for k, v in row["profile_divergences"].items()),
                    len(row["spec_usage"]),
                    ";".join(f"{k}={v}" for k, v in row["spec_usage"].items()),
                    row["production_readers"],
                    row["test_doc_readers"],
                    row["reader_scope"],
                    row["last_changed"],
                    row["status"],
                    row["graduation_state"],
                    row["owner_issue"],
                    row["training_semantics_changed"],
                    row["live_gpu_cost_class"],
                    row["validation_level"],
                    row["required_gate"],
                    row["supersedes"],
                    ";".join(f"{k}={v}" for k, v in row["run_intent_defaults"].items()),
                ]
            )
    print(f"wrote {json_path.relative_to(_REPO_ROOT)}, {csv_path.relative_to(_REPO_ROOT)} "
          f"and {summary_path.relative_to(_REPO_ROOT)} ({len(rows)} options)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
