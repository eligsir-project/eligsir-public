"""Tests for scripts/check_do_not_merge_marker.py (issue #474).

Covers the DO-NOT-MERGE-YET CI enforcement gate: the marker must fail the
check whether it appears in the PR title, the head commit subject, or both;
a clean title/commit must pass; and matching must be prefix-based and
case-insensitive so a near-miss spelling doesn't silently defeat the gate.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_do_not_merge_marker.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location(
        "check_do_not_merge_marker", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


checker = _load_checker_module()


# ------------------------------------------------------------- carries_marker

def test_clean_title_and_commit_no_marker():
    assert not checker.carries_marker("fix(#474): add CI enforcement gate")


def test_empty_and_none_are_clean():
    assert not checker.carries_marker("")
    assert not checker.carries_marker(None)


def test_marker_with_colon_prefix():
    assert checker.carries_marker(
        "DO-NOT-MERGE-YET: fix(#408) resolve 6 of 8 pairs")


def test_marker_case_insensitive_no_colon():
    assert checker.carries_marker("do-not-merge-yet fix something")


def test_marker_with_leading_whitespace():
    assert checker.carries_marker("   DO-NOT-MERGE-YET: wip")


def test_marker_mid_sentence_is_not_a_hit():
    # A PR that merely *mentions* the marker mid-sentence (e.g. referencing
    # this very gate) must not be blocked by it. Only placements that read as
    # a self-applied flag -- leading token, bracketed tag, or trailing final
    # token -- should fire.
    assert not checker.carries_marker(
        "docs: explain the DO-NOT-MERGE-YET convention")
    assert not checker.carries_marker(
        "ci: make the DO-NOT-MERGE-YET gate catch bracketed titles")


def test_marker_as_bracketed_suffix_tag():
    # Regression guard for the real hole found on open PR #371 (2026-08-16):
    # a correctness-held PR whose marker sits in a trailing bracketed tag was
    # reported marker-free and would have been allowed to merge.
    assert checker.carries_marker(
        "perf(#358): Level A fp64 scatter_add_ measured on real GPU "
        "(no dtype change) + hoist frozen per-camera accumulators "
        "[DO-NOT-MERGE-YET]")


def test_marker_as_parenthesised_tag():
    assert checker.carries_marker("fix(#123): something (DO-NOT-MERGE-YET)")


def test_marker_as_bracketed_tag_not_at_the_end():
    assert checker.carries_marker("[DO-NOT-MERGE-YET] fix(#123): something")


def test_marker_as_bracketed_tag_with_trailing_reason():
    assert checker.carries_marker(
        "fix(#123): something [DO-NOT-MERGE-YET: GPU validation pending]")


def test_marker_as_trailing_final_token_unbracketed():
    assert checker.carries_marker("fix(#123): something DO-NOT-MERGE-YET")
    assert checker.carries_marker("fix(#123): something DO-NOT-MERGE-YET:")


def test_bracketed_marker_is_case_insensitive_too():
    assert checker.carries_marker("fix(#123): something [do-not-merge-yet]")


# ------------------------------------------------- spelling variants (issue #912)

def test_marker_with_spaces_instead_of_hyphens_leading():
    # Regression guard for issue #912: the gate should tolerate spaces
    # instead of hyphens, so "DO NOT MERGE" (with spaces) is caught.
    assert checker.carries_marker("DO NOT MERGE: fix(#408) something")


def test_marker_with_spaces_no_yet_suffix():
    # Issue #912: "DO NOT MERGE" without the "YET" suffix should also fire.
    assert checker.carries_marker("DO NOT MERGE (freeze): progress-bar spec")


def test_marker_with_underscores_instead_of_hyphens():
    # Issue #912: underscores between words should also be caught.
    assert checker.carries_marker("do_not_merge: fix something")


def test_marker_spaced_variant_in_bracketed_tag():
    # Issue #912: spaced variant should also fire in bracketed placement.
    assert checker.carries_marker("fix(#123): something [DO NOT MERGE]")


def test_marker_spaced_variant_mid_sentence_is_still_not_a_hit():
    # Issue #912: even with spaced variant, mid-sentence mentions should
    # NOT fire, so a PR that merely discusses the convention isn't blocked.
    assert not checker.carries_marker(
        "docs: explain the DO NOT MERGE convention")


# --------------------------------------------------------------------- main()

def test_main_exits_1_when_title_carries_marker(capsys):
    rc = checker.main([
        "--title", "DO-NOT-MERGE-YET: fix(#408) resolve pairs",
        "--commit-subject", "fix(#408) resolve pairs",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "FAILED" in err
    assert "PR title" in err


def test_main_exits_1_when_commit_subject_carries_marker(capsys):
    rc = checker.main([
        "--title", "clean title",
        "--commit-subject", "DO-NOT-MERGE-YET: wip",
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "head commit subject" in err


def test_main_exits_0_when_both_clean(capsys):
    rc = checker.main([
        "--title", "fix(#474): add CI enforcement gate",
        "--commit-subject", "fix(#474): add CI enforcement gate",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out


def test_main_exits_0_when_both_empty():
    rc = checker.main(["--title", "", "--commit-subject", ""])
    assert rc == 0


def test_main_exits_0_when_commit_subject_omitted_entirely():
    # Without --require-commit-subject the permissive behaviour is preserved
    # for callers that genuinely have no commit to check.
    rc = checker.main(["--title", "fix(#474): add CI enforcement gate"])
    assert rc == 0


# ------------------------------------------- could-not-retrieve-subject (#474)
#
# The workflow's first real run failed with exit 127 (`git: not found`), which
# a branch-protection UI cannot tell apart from a genuine marker hit -- and
# the opposite handling (treat an unretrieved subject as marker-free) would
# fail the gate OPEN, the issue #487 class. Both must be impossible.

def test_require_commit_subject_refuses_when_flag_omitted(capsys):
    rc = checker.main([
        "--title", "fix(#474): add CI enforcement gate",
        "--require-commit-subject",
    ])
    assert rc == checker.EXIT_COULD_NOT_EVALUATE
    err = capsys.readouterr().err
    assert "COULD NOT EVALUATE" in err
    assert "was not passed at all" in err


def test_require_commit_subject_refuses_on_empty_string(capsys):
    # An empty COMMIT_SUBJECT env var (the shape an unset step output takes
    # in the workflow) is refused just as hard as an absent flag, and the
    # message distinguishes the two so the cause is diagnosable from the log.
    rc = checker.main([
        "--title", "fix(#474): add CI enforcement gate",
        "--commit-subject", "",
        "--require-commit-subject",
    ])
    assert rc == checker.EXIT_COULD_NOT_EVALUATE
    err = capsys.readouterr().err
    assert "COULD NOT EVALUATE" in err
    assert "was passed as an empty string" in err


def test_could_not_evaluate_exit_code_is_distinct_from_marker_hit():
    # The whole point of the separate code: an infrastructure failure must be
    # distinguishable from a real DO-NOT-MERGE-YET hit (1) and from an
    # argparse usage error (2).
    assert checker.EXIT_COULD_NOT_EVALUATE not in (0, 1, 2)


def test_require_commit_subject_still_passes_with_a_real_subject(capsys):
    rc = checker.main([
        "--title", "fix(#474): add CI enforcement gate",
        "--commit-subject", "fix(#474): add CI enforcement gate",
        "--require-commit-subject",
    ])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_leading_dash_values_are_parsed_as_values_not_options():
    # A title/subject legitimately starting with `-` must not be swallowed by
    # argparse as an option (which exits 2 and blocks a clean PR). The
    # workflow passes `--flag=$VALUE` for exactly this reason.
    rc = checker.main([
        "--title=-- WIP: something",
        "--commit-subject=-- WIP: something",
        "--require-commit-subject",
    ])
    assert rc == 0


def test_leading_dash_value_still_catches_the_marker():
    rc = checker.main([
        "--title=-x",
        "--commit-subject=DO-NOT-MERGE-YET: -x",
        "--require-commit-subject",
    ])
    assert rc == 1


def test_require_commit_subject_still_fails_on_a_real_marker(capsys):
    rc = checker.main([
        "--title", "clean title",
        "--commit-subject", "DO-NOT-MERGE-YET: wip",
        "--require-commit-subject",
    ])
    assert rc == 1
    assert "head commit subject" in capsys.readouterr().err
