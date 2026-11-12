#!/usr/bin/env python3
"""Enforcement gate for the `DO-NOT-MERGE-YET` self-labelling convention.

Every subagent dispatched in this repo is instructed to prefix a PR title
(or its head-commit subject) with `DO-NOT-MERGE-YET:` when it is uncertain
about correctness -- typically because a change lands most of a fix but
leaves one or more decisions that need a real run plus a human call
(see issue #408 for the canonical example this gate exists to catch).

Before this script/workflow existed, that convention was a title string
with no enforcement: three `DO-NOT-MERGE-YET:`-titled PRs merged to `main`
in one night (issue #474), including one that landed six behaviour-changing
`getattr` fallback value-decisions issue #408 explicitly said "should not be
'fixed' by guessing".

This script is the check itself; `.github/workflows/do-not-merge-gate.yml`
wires it into CI as a required PR status check. It looks for the marker,
case-insensitively, in either the PR title or the head commit's subject line
-- either one carrying the marker fails the gate, since either is how the
convention has actually been used.

Three placements count as carrying the marker, all of which have been seen
on real PRs in this repo: a leading token (`DO-NOT-MERGE-YET: fix ...`), a
bracketed or parenthesised tag anywhere (`fix ... [DO-NOT-MERGE-YET]`), and
the marker as the final token. A bare mid-sentence mention does NOT fire, so
that a PR discussing the convention is not blocked by describing it.

Usage:
    python scripts/check_do_not_merge_marker.py --title "<pr title>" \
        --commit-subject "<head commit subject>" [--require-commit-subject]

Exit codes:
    0 -- gate passes: neither string starts with the marker.
    1 -- gate fails: the marker was found.
    3 -- gate COULD NOT BE EVALUATED: `--require-commit-subject` was passed
         but no non-empty head commit subject was supplied.

The distinction between 1 and 3 is deliberate. The first real run of the CI
workflow that wraps this script failed with exit 127 (`git: not found` in a
container image with no git binary), which in a branch-protection UI is
indistinguishable from a genuine marker hit. Worse, the mirror-image bug --
silently treating an unretrieved subject as an empty, marker-free one -- is
the fail-open class issue #487 was just fixed for. So callers that can
guarantee a real subject should pass `--require-commit-subject`, and get a
loud, separately-identifiable refusal instead of either failure mode.

`--commit-subject` therefore defaults to `None` ("not supplied at all"),
which is a different state from `--commit-subject ""` ("supplied and
explicitly empty"); a real git commit cannot have an empty subject line, so
under `--require-commit-subject` both are refused. Without the flag, both
are treated as marker-free, preserving the permissive behaviour for any
caller that genuinely has no commit to check (e.g. a title-only check).
"""

from __future__ import annotations

import argparse
import re
import sys

# Pattern that matches the marker with flexible spacing: DO NOT MERGE (optional YET)
# tolerates spaces, underscores, or hyphens as separators, so that DO NOT MERGE,
# DO-NOT-MERGE, DO_NOT_MERGE, and all variations with optional YET suffix all fire.
# This prevents spelling variations from trivially defeating the gate (issue #912).
MARKER_PATTERN = r"DO[\s_-]*NOT[\s_-]*MERGE(?:[\s_-]*YET)?"

# Exit code for "the gate could not be evaluated", kept distinct from both 1
# (marker found) and argparse's own 2 (usage error).
EXIT_COULD_NOT_EVALUATE = 3

# Matches the marker as a leading token, optionally followed by ':' or '-',
# case-insensitively, allowing leading whitespace. This intentionally does
# NOT require an exact-case match or a trailing colon -- the three PRs cited
# in issue #474 all used `DO-NOT-MERGE-YET:` but the convention itself
# (per the subagent instructions this gate is meant to back up) is just the
# marker word, and a gate that only fires on one exact spelling is trivially
# defeated by a harmless-looking typo. The pattern now also tolerates spelling
# variations (spaces vs hyphens vs underscores) to prevent evasion (issue #912).
_MARKER_LEADING_RE = re.compile(r"^\s*" + MARKER_PATTERN + r"\b", re.IGNORECASE)

# Matches the marker used as a standalone *tag* rather than a leading token:
# wrapped in brackets or parentheses anywhere in the string, or standing as
# the final token of the string. Both are real spellings of the convention
# that the leading-token match alone silently passed.
#
# Found live (2026-08-16) on open PR #371, whose title is
#   `perf(#358): Level A fp64 scatter_add_ ... [DO-NOT-MERGE-YET]`
# -- a PR held open for an unresolved correctness question, which this gate
# reported as marker-free and would have allowed to merge. That is precisely
# the failure issue #474 created this gate to prevent, reintroduced by a
# different placement of the same marker.
#
# The mid-sentence case is deliberately still NOT a hit, so that a PR which
# merely *discusses* the convention (e.g. "docs: explain the
# DO-NOT-MERGE-YET convention") is not blocked by mentioning it -- see
# test_marker_mid_sentence_is_not_a_hit. A bracketed tag or a trailing final
# token is never prose; a bare mid-sentence occurrence usually is.
_MARKER_TAG_RE = re.compile(
    r"[\[(]\s*" + MARKER_PATTERN + r"\b[^\])]*[\])]",
    re.IGNORECASE,
)
_MARKER_TRAILING_RE = re.compile(
    r"\b" + MARKER_PATTERN + r"\s*[:\-]?\s*$",
    re.IGNORECASE,
)


def carries_marker(text: str | None) -> bool:
    if not text:
        return False
    return bool(
        _MARKER_LEADING_RE.match(text)
        or _MARKER_TAG_RE.search(text)
        or _MARKER_TRAILING_RE.search(text)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="", help="PR title")
    parser.add_argument(
        "--commit-subject",
        default=None,
        help=(
            "Head commit subject line. Omitting it entirely is a distinct "
            "state from passing an empty string -- see --require-commit-"
            "subject."
        ),
    )
    parser.add_argument(
        "--require-commit-subject",
        action="store_true",
        help=(
            "Refuse to evaluate (exit 3) unless a non-empty commit subject "
            "was supplied, instead of silently treating an unretrieved "
            "subject as marker-free."
        ),
    )
    args = parser.parse_args(argv)

    if args.require_commit_subject and not args.commit_subject:
        how = (
            "was not passed at all"
            if args.commit_subject is None
            else "was passed as an empty string"
        )
        print(
            "[do-not-merge-gate] COULD NOT EVALUATE: --require-commit-subject "
            f"was set but --commit-subject {how}.\n"
            "\n"
            "A real git commit cannot have an empty subject line, so this "
            "means the head commit subject was never successfully retrieved. "
            "The gate is refusing to evaluate rather than pass on a subject "
            "it never saw -- an unretrieved subject looks marker-free to any "
            "text check, which would silently fail this gate OPEN (the same "
            "class of defect as issue #487).\n"
            "\n"
            "This is an infrastructure problem with the caller, NOT a "
            "DO-NOT-MERGE-YET marker hit (that exits 1): fix whatever was "
            "supposed to supply the subject -- in CI, the "
            "`Resolve head commit subject` step of "
            ".github/workflows/do-not-merge-gate.yml.",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_EVALUATE

    hits = []
    if carries_marker(args.title):
        hits.append(f"PR title: {args.title!r}")
    if carries_marker(args.commit_subject):
        hits.append(f"head commit subject: {args.commit_subject!r}")

    if not hits:
        print("[do-not-merge-gate] OK: no DO-NOT-MERGE-YET marker found.")
        return 0

    print(
        "[do-not-merge-gate] FAILED: this PR is marked DO-NOT-MERGE-YET and "
        "must not be merged in this state.\n"
        "\n"
        "Found the marker in:\n  - " + "\n  - ".join(hits) + "\n"
        "\n"
        "Why this fired: the DO-NOT-MERGE-YET marker is this repo's convention "
        "for an agent (or human) flagging genuine, unresolved uncertainty about "
        "correctness -- e.g. a value decision that needs a real run plus a "
        "human call, not a guess (see issue #408 / #474 for the incident that "
        "motivated making this an enforced gate instead of a title convention "
        "nothing actually checked).\n"
        "\n"
        "How to clear this check: once the uncertainty is actually resolved "
        "(the decision made, verified on a real run per this repo's Run "
        "Contracts -- not just merged), remove the `DO-NOT-MERGE-YET` marker "
        "from both the PR title and, if present, the head commit's subject "
        "line, then push. Do not remove the marker just to get this check to "
        "pass while the underlying uncertainty is still open.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
