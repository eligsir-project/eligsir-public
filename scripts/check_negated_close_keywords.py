#!/usr/bin/env python3
"""Guard against GitHub's auto-close bug on negated/descriptive closing keywords.

GitHub's issue-closing-keyword parser has no negation model and no intent
model. A PR body or commit message that carefully documents what a change
does NOT fix, or merely describes a closing event that already happened
elsewhere, will silently close the very issues it's talking about --
because keywords like `close`, `fix`, `resolve` (+ inflections) immediately
followed by `#N` are a closing directive regardless of what precedes them,
including negation tokens, line breaks, and past-tense narration.

This script has two independent rule sets, both scanning the same
keyword-adjacent-to-`#N` surface:

1. NEGATION rule (original, issue #827 first pass): flags a closing keyword
   preceded by a negation token (`not`/`never`/`without`/`neither`/`n't`)
   within a ~60-char clause-bounded window.

   Real incidents:
     - PR #826 / commit 58acd56a: "it does NOT resolve\\n#793 ..."
     - PR #616 / commit 3452af6a: "Does not close #593 -- see PR body ..."
     - PR #391 / commit c5f9ed5b: "does not, by itself, close #374's item 3"

2. DESCRIPTIVE rule (issue #827 second scope gap, found within seconds of
   the negation-only guard merging): flags a closing keyword + `#N` that is
   NOT a genuine, isolated closing directive -- i.e. inline mid-sentence
   prose, past-tense narration of a different PR's close, or a keyword
   carrying a trailing qualifier that contradicts a clean close. Unlike the
   negation rule, this one is default-ALLOW: a closing keyword + `#N` that
   forms its own clause (bounded by start-of-message / a clause terminator
   on both sides, e.g. a bare `Closes #123` line) is exempt. Everything else
   adjacent to a keyword is flagged.

   Real incidents (neither caught by the negation rule -- no negation token
   present in either):
     - PR #829 body: "PR #377 also closed #374, but with an unqualified..."
       -- past-tense narration, auto-closed #374 a second time.
     - PR #828 body: "-- closes #614 ledger item 3's passivity half"
       -- a scope description, not a directive; auto-closed #614.
     - PR #788 body (docs-only retrospective!): quoting "Fixes #489 (phase
       1)" and "Closes #515 (Phase 2 complete; Phases 1/3/4 remain)" as
       *examples of what had gone wrong* -- GitHub does not exempt quoted
       text (only backtick code spans), so this PR's own merge silently
       re-closed both #489 and #515 the moment it landed, seconds after the
       PR's summary claimed to have reopened them.

Empirically measured against this repo's real corpus (150 recent commits +
60 recent PR bodies, see the PR that introduced this rule for the full
table): the descriptive rule's exemption ("own clause, both sides bounded")
passes every observed genuine single-issue close and flags every observed
incident, at a false-positive cost of 2 messages out of 210 -- both
one-topic closes with a same-clause trailing elaboration ("Closes #632
criterion 8: enables ...", "(fixes #548 antipattern)"), both fixable by
moving the elaboration to a new sentence/clause.

WARNING -- this is the DOMINANT failure mode for the class of PR this
project writes constantly, not a minor edge case. Any syntactically bare
`close[s]/fix[es]/resolve[s] #N`-shaped string that forms a clause of its
own will pass this script's descriptive-rule exemption REGARDLESS of
whether it is wrapped in quotes, backticks, or nothing at all -- because
the exemption is a text-shape check, not a "will GitHub actually act on
this" check. GitHub's own behavior on markdown code spans is confirmed
safe in exactly one documented case (a PR explaining this convention
inside backticks did not close the issue it named), but that is one
data point, not a general guarantee this script can rely on for every
markdown form (nesting, multi-line spans, etc.) -- and quoted-but-not-
backticked text is confirmed UNSAFE: a retrospective docs PR quoting two
earlier incidents' exact strings (inside quotation marks) re-closed both
on its own merge. There are now THREE real occurrences of this shape in
this repo's history, including one caught only by hand during review of
the PR that shipped this very rule -- an earlier draft of that PR's own
body and commit message reproduced the incident it was documenting,
narrowly avoided only because a reviewer independently re-scanned it with
a maximally conservative pattern (no clause model, no code-span
stripping at all) before merge:

    import re
    PAT = re.compile(r"(?i)\\b(close[sd]?|fix(?:e[sd])?|resolve[sd]?)\\b\\s*:?\\s*#(\\d+)")
    # any match in a PR body/commit message that isn't an INTENTIONAL,
    # currently-desired close is a real risk -- rewrite it, don't rely on
    # this script's (or GitHub's own) code-span/quote handling.

Decision: this script KEEPS the code-span exemption (matches the one
confirmed-safe case, and removing it would regress the
`test_code_span_github_refs_example` regression test covering that real
false positive) rather than refusing all backticked directives outright.
The tradeoff this creates is explicit, not silent: **any PR that documents
or audits a past auto-close incident -- i.e. any PR whose own text is
expected to be dense with closing-keyword-shaped strings describing OTHER
issues -- must be hand-verified with the conservative pattern above before
merge.** This script's exemptions (clause-boundary, code-span stripping)
are calibrated to minimize false positives on ordinary engineering PRs;
they are not, and are not intended to be, a safety net for a PR whose
entire purpose is to reproduce examples of the bug it is fixing.

This script recommends writing scope exclusions and descriptive references
verb-free instead:
  - BAD:  "This does NOT fix #123"
  - BAD:  "PR #99 also closed #123"
  - GOOD: "out of scope: #123" or "#123 is unaffected by this PR"
  - GOOD: "#123 was auto-closed by PR #99"

Usage:
    python scripts/check_negated_close_keywords.py \\
        --title "<pr title>" --body "<pr body>" --commit-messages <msg> [<msg> ...]

Exit codes:
    0 -- gate passes: no violations found.
    1 -- gate fails: a negated or descriptive closing keyword was found.
    3 -- gate COULD NOT EVALUATE: input parsing failed.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass


# Closing keywords (GitHub's auto-close triggers, case-insensitive)
_CLOSING_KEYWORDS = [
    "close", "closes", "closed",
    "fix", "fixes", "fixed",
    "resolve", "resolves", "resolved",
]

# Negation tokens to search for in the preceding window
_NEGATION_TOKENS = [
    "not", "never", "without", "neither",
    r"n't",  # contraction regex
]

# Build the regex pattern. After normalizing whitespace and stripping code spans:
#   1. Look for a negation token (case-insensitive)
#   2. Allow up to ~60 chars within clause boundaries (word chars, spaces, commas only)
#      This prevents matching across sentence/clause breaks, avoiding false positives
#      like "Behavior is not changed by this refactor. Fixes #123."
#   3. Match a closing keyword immediately followed by optional space/colon and #digits
#
# Clause boundary rationale: forbid `.` `;` `:` `--` `!` `?` or paragraph breaks
# between negation and keyword, as these indicate separate clauses. The gap pattern
# `[\w ,]{0,60}?` rejects characters outside the set; since newlines are already
# normalized to spaces by normalize_text(), a blank-line boundary becomes two spaces
# and matches this class, so we cannot detect it post-normalization. This is documented
# as a known limitation.
_NEGATION_RE = re.compile(
    r"(" + "|".join(_NEGATION_TOKENS) + r")"  # negation token
    r"[\w ,]{0,60}?"  # up to 60 chars within clause boundary
    r"(" + "|".join(_CLOSING_KEYWORDS) + r")"  # closing keyword
    r"(?=\s*[:#]*\s*#)"  # lookahead: must be followed by space/colon and #
    r"\s*[:#]*\s*"  # optional whitespace, colon
    r"(#\d+)",  # issue number
    re.IGNORECASE | re.DOTALL
)

# Any closing keyword immediately followed (optional colon/space) by an
# issue number -- used by the descriptive-rule scan as the universe of
# candidates, independent of whether a negation token is nearby.
_KEYWORD_ADJACENCY_RE = re.compile(
    r"\b(" + "|".join(_CLOSING_KEYWORDS) + r")\b"
    r"\s*:?\s*"
    r"(#\d+)",
    re.IGNORECASE,
)

# A markdown heading line ("## Foo", "### Closes", ...). Treated as its own
# clause on both sides so a heading like "## Closes" (common convention:
# heading naming the section, directive on the next line) does not fuse
# with the line that follows it once whitespace is normalized away.
_HEADING_LINE_RE = re.compile(r"^#{1,6}(\s|$)")

# CANONICAL / exempt form for the descriptive rule: a closing keyword +
# one-or-more comma-separated issue refs that forms its OWN clause --
# bounded on the left by start-of-message or a clause terminator
# (. ; : ! ? or --), and on the right by a clause terminator or
# end-of-message, with nothing else (no trailing qualifier, no leading
# prose) attached to it. A leading "-"/"*" bullet is allowed. This is
# deliberately the ONLY exempt shape; a genuine "Closes #632 criterion 8:
# ..." elaboration on the same clause is NOT exempt (measured: 2/210 real
# messages hit this, both fixable with a period/rewrite) because "closes
# #N, but only partially" is exactly the incident shape this rule exists
# to catch (see PR #788's "Fixes #489 (phase 1)" quoting itself into a
# second real incident).
_CANONICAL_CLOSE_RE = re.compile(
    r"(?:^|(?<=[.;:!?])|(?<=--))\s*"
    r"(?:[-*]\s*)?"
    r"(?:" + "|".join(_CLOSING_KEYWORDS) + r")"
    r"\s*:?\s*"
    r"#\d+(?:\s*,\s*#\d+)*"
    r"\s*(?=[.;:!?]|--|$)",
    re.IGNORECASE,
)

EXIT_COULD_NOT_EVALUATE = 3


@dataclass
class Violation:
    """A single closing-keyword violation (negated or descriptive)."""
    text: str  # the full matched span
    issue_num: str  # the issue number (e.g., "#123")
    surface: str  # the surface for display (title/body/commit message N)
    kind: str = "negated"  # "negated" or "descriptive"


def strip_code_spans(text: str) -> str:
    """Strip fenced code blocks and inline code spans.

    Replaces each span with a single space to avoid fusing words.
    GitHub's auto-close parser does not honor closing keywords inside
    code spans, so we shouldn't flag them either. See real false positive:
    PR #817 body contains ``not attempting (hence `Refs #806`, not
    `Closes #806`)`` — GitHub did not close #806 because the keyword
    is in backticks. This check should match GitHub's behavior.
    """
    if not text:
        return ""

    # Strip fenced code blocks (triple backticks) first
    text = re.sub(r"```[^`]*```", " ", text, flags=re.DOTALL)

    # Strip inline code spans (single backticks)
    text = re.sub(r"`[^`]*`", " ", text)

    return text


def normalize_text(text: str) -> str:
    """Normalize whitespace and newlines for matching."""
    if not text:
        return ""
    # Collapse multiple whitespace/newlines into single spaces
    return re.sub(r"\s+", " ", text)


def normalize_for_canonical(text: str) -> str:
    """Paragraph/heading-aware normalization for the descriptive-rule scan.

    Unlike normalize_text() (which collapses ALL whitespace including blank
    lines and is a documented limitation of the negation rule), this
    normalization needs to know where clauses actually break, because the
    descriptive rule's whole exemption is "does this keyword+#N form its own
    clause". A blank line (paragraph break) or a markdown heading line is a
    real clause boundary and is turned into an explicit " . " terminator
    before whitespace collapsing -- otherwise "## Closes\nCloses #809" would
    read as one fused, non-canonical clause after naive normalization
    (measured false positive, fixed before this shipped).
    """
    if not text:
        return ""
    parts = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "":
            parts.append(" . ")
        elif _HEADING_LINE_RE.match(stripped):
            parts.append(" . " + stripped + " . ")
        else:
            parts.append(stripped + " ")
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def find_violations(text: str) -> list[Violation]:
    """Find all negated closing keyword violations in a text."""
    if not text:
        return []

    # Strip code spans before matching, as GitHub doesn't honor keywords in code
    stripped = strip_code_spans(text)

    # Normalize whitespace and newlines for matching
    normalized = normalize_text(stripped)
    violations = []

    # Find all matches
    for match in _NEGATION_RE.finditer(normalized):
        violation = Violation(
            text=match.group(0),
            issue_num=match.group(3),
            surface="",  # will be filled by caller
            kind="negated",
        )
        violations.append(violation)

    return violations


def find_descriptive_violations(text: str) -> list[Violation]:
    """Find closing-keyword+#N occurrences that are NOT a clean, isolated
    closing directive (the "second scope gap" from issue #827).

    Any occurrence already caught by find_violations() (the negation rule)
    is skipped here to avoid reporting the same span twice under two
    different messages -- the negation message is more specific about why
    that particular case is dangerous.
    """
    if not text:
        return []

    stripped = strip_code_spans(text)
    already_negated = {v.issue_num for v in find_violations(text)}

    normalized = normalize_for_canonical(stripped)
    canonical_spans = list(_CANONICAL_CLOSE_RE.finditer(normalized))

    violations = []
    for match in _KEYWORD_ADJACENCY_RE.finditer(normalized):
        issue_num = match.group(2)
        if issue_num in already_negated:
            continue
        is_canonical = any(
            cm.start() <= match.start() < cm.end() for cm in canonical_spans
        )
        if is_canonical:
            continue
        violations.append(
            Violation(
                text=match.group(0),
                issue_num=issue_num,
                surface="",  # will be filled by caller
                kind="descriptive",
            )
        )

    return violations


def check_all_inputs(
    title: str | None,
    body: str | None,
    commit_messages: list[str] | None,
) -> list[Violation]:
    """Check title, body, and all commit messages for violations."""
    all_violations = []

    # Check title
    if title:
        for v in find_violations(title):
            v.surface = "PR title"
            all_violations.append(v)
        for v in find_descriptive_violations(title):
            v.surface = "PR title"
            all_violations.append(v)

    # Check body
    if body:
        for v in find_violations(body):
            v.surface = "PR body"
            all_violations.append(v)
        for v in find_descriptive_violations(body):
            v.surface = "PR body"
            all_violations.append(v)

    # Check each commit message
    if commit_messages:
        for i, msg in enumerate(commit_messages, 1):
            for v in find_violations(msg):
                v.surface = f"commit message ({i})"
                all_violations.append(v)
            for v in find_descriptive_violations(msg):
                v.surface = f"commit message ({i})"
                all_violations.append(v)

    return all_violations


def main(argv: list[str] | None = None) -> int:
    import json
    import os

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--title", default="", help="PR title")
    parser.add_argument("--body", default="", help="PR body")
    parser.add_argument(
        "--commit-messages-json",
        default="[]",
        help="JSON array of commit messages (from env var, not positional args)",
    )
    args = parser.parse_args(argv)

    try:
        # Parse commit messages from JSON to avoid argparse issues with messages
        # starting with `-` character
        try:
            commit_messages = json.loads(args.commit_messages_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in commit messages: {e}")

        violations = check_all_inputs(
            title=args.title or None,
            body=args.body or None,
            commit_messages=commit_messages or None,
        )
    except Exception as e:
        print(
            f"[negated-close-keyword-gate] COULD NOT EVALUATE: {e}",
            file=sys.stderr,
        )
        return EXIT_COULD_NOT_EVALUATE

    if not violations:
        print("[negated-close-keyword-gate] OK: no negated or descriptive closing keywords found.")
        return 0

    negated = [v for v in violations if v.kind == "negated"]
    descriptive = [v for v in violations if v.kind == "descriptive"]

    sections = []

    if negated:
        details = [
            f"  {v.surface}: found negated closing keyword near {v.issue_num}\n"
            f"    Offending text: {v.text!r}"
            for v in negated
        ]
        sections.append(
            "NEGATED closing keywords (a negation token precedes the keyword):\n"
            + "\n".join(details) + "\n"
            "\n"
            "Why this fires: GitHub's auto-close parser has no negation model. "
            "A PR that documents what it does NOT fix will silently close the very "
            "issues it disclaims, because closing keywords like 'close', 'fix', "
            "'resolve' (+ inflections) immediately followed by '#N' are a closing "
            "directive regardless of negation in the preceding context.\n"
            "\n"
            "Real incidents (all reopened by hand after merging):\n"
            "  - PR #826 (58acd56a): 'it does NOT resolve\\n#793 ...'\n"
            "  - PR #616 (3452af6a): 'Does not close #593 -- see PR body ...'\n"
            "  - PR #391 (c5f9ed5b): 'does not, by itself, close #374's item 3'"
        )

    if descriptive:
        details = [
            f"  {v.surface}: found unqualified/descriptive closing keyword near {v.issue_num}\n"
            f"    Offending text: {v.text!r}"
            for v in descriptive
        ]
        sections.append(
            "DESCRIPTIVE/UNQUALIFIED closing keywords (not a negation, but not a "
            "clean isolated directive either -- inline prose, past-tense narration, "
            "or a trailing qualifier on the same clause as the reference):\n"
            + "\n".join(details) + "\n"
            "\n"
            "Why this fires: GitHub's auto-close parser also has no INTENT model. "
            "Prose that merely mentions or narrates a closing keyword next to '#N' "
            "closes it exactly as if it were a real directive -- quotes do not "
            "protect you (only backtick code spans do).\n"
            "\n"
            "Real incidents (none contain a negation token, so the negation rule "
            "above did not catch them; all closed within seconds of merge):\n"
            "  - PR #829 body: 'PR #377 also closed #374, but with an unqualified...'\n"
            "  - PR #828 body: '-- closes #614 ledger item 3's passivity half'\n"
            "  - PR #788 body (itself a retrospective doc!) quoting 'Fixes #489 "
            "(phase 1)' and 'Closes #515 (Phase 2 complete; Phases 1/3/4 remain)' "
            "as examples of what went wrong -- and closed both again on merge."
        )

    print(
        "[negated-close-keyword-gate] FAILED: this PR contains closing-keyword "
        "text that would spuriously close GitHub issues.\n"
        "\n"
        "Found violations:\n\n" + "\n\n".join(sections) + "\n"
        "\n"
        "How to clear this check: rewrite so the keyword+issue-number is EITHER\n"
        "  (a) a clean, isolated directive on its own -- nothing else attached to\n"
        "      the same clause -- e.g. a bare 'Closes #123' or 'Closes #1, #2'\n"
        "      line/sentence, optionally after a '-'/'*' bullet, OR\n"
        "  (b) rewritten VERB-FREE so no closing keyword is adjacent to '#N' at all.\n"
        "\n"
        "  Instead of:        'This does NOT fix #123 because ...'\n"
        "  Instead of:        'PR #99 also closed #123, but only partially'\n"
        "  Instead of:        'Closes #123 (phase 1 only, phases 2-3 remain)'\n"
        "  Write one of:      'out of scope: #123 (reason)'\n"
        "                     'unaffected by this PR: #123 (reason)'\n"
        "                     '#123 is unaffected by this PR'\n"
        "                     '#123 was auto-closed by PR #99'\n"
        "                     'see #123 for scope' / 'Refs #123 (not closed by this)'\n"
        "                     'Closes #123.' as its own sentence, with the caveat\n"
        "                       moved to a separate sentence/line\n"
        "\n"
        "Verb-free phrasings are safe because GitHub's closing-keyword parser "
        "requires the keyword to be immediately adjacent to the issue number -- "
        "other phrasing patterns (like 'Related:', 'Refs:', or bare issue "
        "mentions) do not trigger auto-close.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
