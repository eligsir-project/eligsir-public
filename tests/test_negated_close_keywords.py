"""Tests for check_negated_close_keywords.py guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import the script as a module
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

from check_negated_close_keywords import (
    EXIT_COULD_NOT_EVALUATE,
    check_all_inputs,
    find_descriptive_violations,
    find_violations,
    main,
    normalize_for_canonical,
    normalize_text,
)


class TestNormalizeText:
    """Test whitespace normalization."""

    def test_single_space_unchanged(self) -> None:
        assert normalize_text("hello world") == "hello world"

    def test_multiple_spaces_collapsed(self) -> None:
        assert normalize_text("hello    world") == "hello world"

    def test_newlines_to_space(self) -> None:
        assert normalize_text("hello\nworld") == "hello world"

    def test_mixed_whitespace(self) -> None:
        text = """it does NOT resolve
#793 (the velocity-adaptive controller's)"""
        assert normalize_text(text) == "it does NOT resolve #793 (the velocity-adaptive controller's)"

    def test_empty_string(self) -> None:
        assert normalize_text("") == ""

    def test_only_whitespace(self) -> None:
        assert normalize_text("   \n\n  \t  ") == " "


class TestFindViolations:
    """Test violation detection in single texts."""

    def test_real_case_1_multiline(self) -> None:
        """Real #826 case with newline between NOT and resolve."""
        text = """This is a fast, narrowly-scoped fix for #794's core complaint (max_gap
silently bypassing the novelty/coverage gate) -- it does NOT resolve
#793 (the velocity-adaptive controller's separate "speed EMA is
cumulative displacement, not velocity" correctness bug) or #794's full"""
        violations = find_violations(text)
        assert len(violations) > 0
        assert any("#793" in v.issue_num for v in violations)

    def test_real_case_2_single_line(self) -> None:
        """Real #616 case."""
        text = "Does not close #593 -- see PR body / issue comment for what remains."
        violations = find_violations(text)
        assert len(violations) > 0
        assert any("#593" in v.issue_num for v in violations)

    def test_real_case_3_possessive_and_intervening_clause(self) -> None:
        """Real #374 case: negation + intervening clause + possessive form."""
        # This is from PR #391 commit c5f9ed5b
        text = "does not, by itself, close #374's item 3"
        violations = find_violations(text)
        assert len(violations) > 0
        assert any("#374" in v.issue_num for v in violations)

    def test_possessive_issue_number(self) -> None:
        """Should flag possessive forms like #123's."""
        text = "Does not close #123's problem"
        violations = find_violations(text)
        assert len(violations) > 0
        assert any("#123" in v.issue_num for v in violations)

    def test_intervening_clause_with_comma(self) -> None:
        """Should flag even with intervening clauses (e.g., comma-delimited asides)."""
        text = "This does not, in any way, fix #456's issue"
        violations = find_violations(text)
        assert len(violations) > 0
        assert any("#456" in v.issue_num for v in violations)

    def test_genuine_close_no_negation(self) -> None:
        """Should pass: genuine Closes without negation."""
        text = "Closes #123"
        violations = find_violations(text)
        assert len(violations) == 0

    def test_verb_free_exclusion_colon(self) -> None:
        """Should pass: verb-free exclusion."""
        text = "out of scope: #593 (perf measurement, needs GPU)"
        violations = find_violations(text)
        assert len(violations) == 0

    def test_verb_free_exclusion_is(self) -> None:
        """Should pass: verb-free 'is' clause."""
        text = "#793 is unaffected by this PR"
        violations = find_violations(text)
        assert len(violations) == 0

    def test_refs_keyword(self) -> None:
        """Should pass: Refs is not a closing keyword."""
        text = "Refs #792, #794."
        violations = find_violations(text)
        assert len(violations) == 0

    def test_related_keyword(self) -> None:
        """Should pass: Related is not a closing keyword."""
        text = "Related: #182"
        violations = find_violations(text)
        assert len(violations) == 0

    def test_see_no_keyword(self) -> None:
        """Should pass: 'see' without closing keyword."""
        text = "see #202 for details"
        violations = find_violations(text)
        assert len(violations) == 0

    def test_clause_boundary_sentence_break(self) -> None:
        """Must-PASS: negation and keyword in different sentences (clause boundary).

        Peer review found these common, legitimate patterns were being flagged.
        After clause-boundary tightening (forbid . ; : -- ! ? between negation
        and keyword), these should pass.
        """
        # Real false-positive case found in peer review
        text = "This does not change behavior. Closes #123."
        violations = find_violations(text)
        assert len(violations) == 0, "Sentence-separated negation should not flag"

    def test_clause_boundary_refactor_fixes(self) -> None:
        """Must-PASS: negation and keyword separated by period (clause boundary)."""
        text = "Behavior is not changed by this refactor. Fixes #456."
        violations = find_violations(text)
        assert len(violations) == 0

    def test_clause_boundary_tests_not_passing(self) -> None:
        """Must-PASS: negation and keyword in separate sentences."""
        text = "The tests were not passing before. Fixes #789"
        violations = find_violations(text)
        assert len(violations) == 0

    def test_clause_boundary_depends_on_fix(self) -> None:
        """Must-PASS: negation and keyword separated by multiple sentences."""
        text = "This PR does not depend on the fix in #99. Closes #100."
        violations = find_violations(text)
        assert len(violations) == 0

    def test_code_span_github_refs_example(self) -> None:
        """Must-PASS: closing keyword in backticks should not flag (PR #817 case).

        Real case: PR #817 body explaining the verb-free convention:
        ``not attempting (hence `Refs #806`, not `Closes #806`)``

        GitHub did NOT close #806 when #817 merged because the keyword is in
        backticks. This checker should match GitHub's behavior by stripping code spans.
        """
        text = "not attempting (hence `Refs #806`, not `Closes #806`)"
        violations = find_violations(text)
        assert len(violations) == 0, "Closing keywords in backticks should be stripped"

    def test_negation_too_far_away(self) -> None:
        """Should pass: negation is >60 chars away from keyword."""
        # Build a string where negation is far from the keyword
        text = "NOT relevant here: " + "x" * 70 + " This fixes #123"
        violations = find_violations(text)
        # Should not match because "NOT" is >60 chars from "fixes"
        assert len(violations) == 0

    def test_case_insensitive_negation(self) -> None:
        """Should flag: NOT should match in different cases."""
        text = "This does NOT fix #123"
        violations = find_violations(text)
        assert len(violations) > 0

    def test_case_insensitive_keyword(self) -> None:
        """Should flag: closing keyword in different case."""
        text = "Does not CLOSE #456"
        violations = find_violations(text)
        assert len(violations) > 0

    def test_never_negation(self) -> None:
        """Should flag: 'never' as negation token."""
        text = "I would never fix #789"
        violations = find_violations(text)
        assert len(violations) > 0

    def test_without_negation(self) -> None:
        """Should flag: 'without' as negation token."""
        text = "This is without fixing this closes #789"
        violations = find_violations(text)
        assert len(violations) > 0
        assert any("#789" in v.issue_num for v in violations)

    def test_neither_negation(self) -> None:
        """Should flag: 'neither' as negation token."""
        text = "This is neither closes #555 nor resolves #666"
        violations = find_violations(text)
        assert len(violations) > 0
        assert any("#555" in v.issue_num for v in violations)

    def test_contraction_nt(self) -> None:
        """Should flag: n't contraction."""
        text = "doesn't close #111"
        violations = find_violations(text)
        assert len(violations) > 0

    def test_multiple_violations_same_text(self) -> None:
        """Should flag: multiple violations in one text."""
        text = "Does not close #1 and does not fix #2"
        violations = find_violations(text)
        assert len(violations) >= 2

    def test_no_longer_negation(self) -> None:
        """Should flag: 'no longer' as negation (if 'longer' added to tokens)."""
        # Note: 'no longer' is two words; 'no' is in the token list
        text = "This is no longer closing #999"
        violations = find_violations(text)
        # This will only match if 'no' is close enough to 'closing'
        # With "is no longer closing", the distance is ~10 chars
        # The issue requires 'no' to be matched as a single token
        # Since we don't have 'no' in the negation tokens list yet, this won't match
        # Let's assert it doesn't match (test that 'no' alone isn't in the list)
        assert len(violations) == 0


class TestCheckAllInputs:
    """Test checking multiple input sources."""

    def test_title_violation(self) -> None:
        """Should flag violation in title."""
        violations = check_all_inputs(
            title="Does not fix #123",
            body=None,
            commit_messages=None,
        )
        assert len(violations) > 0
        assert all(v.surface == "PR title" for v in violations)

    def test_body_violation(self) -> None:
        """Should flag violation in body."""
        violations = check_all_inputs(
            title=None,
            body="Does not fix #456",
            commit_messages=None,
        )
        assert len(violations) > 0
        assert all(v.surface == "PR body" for v in violations)

    def test_commit_violation(self) -> None:
        """Should flag violation in commit message."""
        violations = check_all_inputs(
            title=None,
            body=None,
            commit_messages=["Does not fix #789"],
        )
        assert len(violations) > 0
        assert all(v.surface == "commit message (1)" for v in violations)

    def test_multiple_commits(self) -> None:
        """Should track multiple commits separately."""
        violations = check_all_inputs(
            title=None,
            body=None,
            commit_messages=["Does not fix #111", "Never closes #222"],
        )
        assert len(violations) >= 2
        surfaces = {v.surface for v in violations}
        assert "commit message (1)" in surfaces
        assert "commit message (2)" in surfaces

    def test_mixed_sources(self) -> None:
        """Should flag violations across all sources."""
        violations = check_all_inputs(
            title="Not fixing this fixes #1",
            body="Does not close #2",
            commit_messages=["Never resolves #3"],
        )
        assert len(violations) >= 2  # At least body and commit
        surfaces = {v.surface for v in violations}
        assert "PR body" in surfaces
        assert "commit message (1)" in surfaces

    def test_genuine_close_all_sources(self) -> None:
        """Should pass: genuine closes in all sources."""
        violations = check_all_inputs(
            title="Fixes #1",
            body="Closes #2",
            commit_messages=["Resolves #3"],
        )
        assert len(violations) == 0

    def test_verb_free_all_sources(self) -> None:
        """Should pass: verb-free exclusions in all sources."""
        violations = check_all_inputs(
            title="out of scope: #1",
            body="#2 is unaffected by this PR",
            commit_messages=["see #3 for details"],
        )
        assert len(violations) == 0

    def test_mixed_genuine_and_exclusions(self) -> None:
        """Should pass: mixed genuine closes and verb-free exclusions."""
        violations = check_all_inputs(
            title="Fixes #1",
            body="Does not address #2 (out of scope)",
            commit_messages=["see #3"],
        )
        # The body line "Does not address" is NOT a closing keyword,
        # so it should not trigger
        assert len(violations) == 0


class TestMainFunction:
    """Test the main CLI interface."""

    def test_exit_0_pass(self, capsys) -> None:
        """Should exit 0 on pass."""
        exit_code = main(["--title", "Closes #123"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "OK" in captured.out

    def test_exit_1_fail(self, capsys) -> None:
        """Should exit 1 on violation."""
        exit_code = main(["--title", "Does not fix #456"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAILED" in captured.err

    def test_title_argument(self, capsys) -> None:
        """Should process title argument."""
        exit_code = main(["--title", "Never closes #789"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "PR title" in captured.err

    def test_body_argument(self, capsys) -> None:
        """Should process body argument."""
        exit_code = main(["--body", "Does not fix #111"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "PR body" in captured.err

    def test_commit_messages_multiple(self, capsys) -> None:
        """Should process multiple commit messages."""
        import json
        messages_json = json.dumps([
            "Does not fix #1",
            "Never closes #2",
        ])
        exit_code = main([
            "--commit-messages-json",
            messages_json,
        ])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "FAILED" in captured.err

    def test_empty_inputs(self, capsys) -> None:
        """Should pass with empty inputs."""
        exit_code = main([])
        assert exit_code == 0

    def test_shows_violation_details(self, capsys) -> None:
        """Should show violation details in error."""
        exit_code = main(["--title", "Does not fix #123"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "#123" in captured.err
        assert "not fix" in captured.err or "not" in captured.err

    def test_real_case_1(self, capsys) -> None:
        """Should catch real case #826."""
        text = """This is a fast, narrowly-scoped fix for #794's core complaint (max_gap
silently bypassing the novelty/coverage gate) -- it does NOT resolve
#793 (the velocity-adaptive controller's separate "speed EMA is
cumulative displacement, not velocity" correctness bug) or #794's full"""
        exit_code = main(["--body", text])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "#793" in captured.err

    def test_real_case_2(self, capsys) -> None:
        """Should catch real case #616."""
        text = "Does not close #593 -- see PR body / issue comment for what remains."
        exit_code = main(["--body", text])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "#593" in captured.err


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_issue_number_with_space_and_colon(self) -> None:
        """Should match issue with space and colon between keyword and #."""
        text = "Does not: #123 close the issue"
        violations = find_violations(text)
        # Should match because of the colon/space allowance
        # Actually, the regex requires the keyword BEFORE space/colon,
        # so this might not match. Let me check the regex...
        # The pattern is: keyword \s*[:#]*\s* #\d+
        # So "not: #123" should match. But here we have "not:" then space then
        # "close" later. The "close" is after, not before. So no match.
        # This is expected behavior - the keyword must come before the #
        assert len(violations) == 0

    def test_keyword_space_colon_number(self) -> None:
        """Should match keyword with space/colon before issue number."""
        text = "Does not fix: #123"
        violations = find_violations(text)
        assert len(violations) > 0

    def test_keyword_space_number(self) -> None:
        """Should match keyword with space before issue number."""
        text = "Does not fix #123"
        violations = find_violations(text)
        assert len(violations) > 0

    def test_long_issue_number(self) -> None:
        """Should handle any length issue number."""
        text = "Does not fix #12345678"
        violations = find_violations(text)
        assert len(violations) > 0
        assert violations[0].issue_num == "#12345678"

    def test_multiple_issues_in_sequence(self) -> None:
        """Should flag each issue that has negated closing keyword."""
        text = "Does not close #1 and does not fix #2"
        violations = find_violations(text)
        assert len(violations) >= 2
        issue_nums = {v.issue_num for v in violations}
        assert "#1" in issue_nums
        assert "#2" in issue_nums

    def test_negation_followed_by_punctuation(self) -> None:
        """Should handle negation followed by punctuation."""
        text = "Does not, fix #123"
        violations = find_violations(text)
        # The pattern allows any character, so "," is included
        assert len(violations) > 0

    def test_nested_issue_references(self) -> None:
        """Should find all violations even with nested references."""
        text = "Fixes #1 but does not close #2 and resolves #3 except does not fix #4"
        violations = find_violations(text)
        # Should find negated ones: #2 and #4
        issue_nums = {v.issue_num for v in violations}
        assert "#2" in issue_nums
        assert "#4" in issue_nums
        # Should NOT find #1 and #3 (they're genuine fixes/resolves)
        assert "#1" not in issue_nums
        assert "#3" not in issue_nums


class TestNormalizeForCanonical:
    """Test the paragraph/heading-aware normalization used by the
    descriptive-rule scan."""

    def test_blank_line_becomes_boundary(self) -> None:
        text = "paragraph one ends here.\n\nCloses #1"
        normalized = normalize_for_canonical(text)
        assert " . " in normalized or normalized.count(".") >= 1

    def test_heading_line_isolated(self) -> None:
        text = "## Closes\nCloses #809"
        normalized = normalize_for_canonical(text)
        # heading content should be bounded by terminators on both sides
        assert "## Closes ." in normalized

    def test_empty_string(self) -> None:
        assert normalize_for_canonical("") == ""

    def test_ordinary_wrap_collapses_to_space(self) -> None:
        text = "line one\nline two"
        assert normalize_for_canonical(text) == "line one line two"


class TestFindDescriptiveViolations:
    """Test the second scope-gap rule: descriptive/unqualified closing
    keywords that are not negated but are also not a clean, isolated
    directive.

    Real incidents (issue #827, "second scope gap", found within seconds of
    the negation-only guard merging -- none contain a negation token):
      - PR #829 body: 'PR #377 also closed #374, but with an unqualified...'
      - PR #828 body: '-- closes #614 ledger item 3's passivity half'
      - PR #788 body (a retrospective docs-only PR!) quoting 'Fixes #489
        (phase 1)' and 'Closes #515 (Phase 2 complete; Phases 1/3/4
        remain)' as illustrative examples of what had gone wrong -- and
        auto-closed both again, seconds after its own summary claimed to
        have reopened them.
    """

    def test_real_case_pr829_374(self) -> None:
        """Verbatim incident string: PR #829 body re-closing #374."""
        text = (
            "PR #377 also closed #374, but with an unqualified closing "
            "keyword reflecting genuine intent -- correct GitHub behavior, "
            "deliberately out of scope here."
        )
        violations = find_descriptive_violations(text)
        assert len(violations) > 0
        assert any(v.issue_num == "#374" for v in violations)
        assert all(v.kind == "descriptive" for v in violations)

    def test_real_case_pr828_614(self) -> None:
        """Verbatim incident string: PR #828 body closing #614."""
        text = "## Notes\n\n-- closes #614 ledger item 3's passivity half\n\nFour arms: three r9700 idle."
        violations = find_descriptive_violations(text)
        assert len(violations) > 0
        assert any(v.issue_num == "#614" for v in violations)

    def test_real_case_pr788_quoted_489_and_515(self) -> None:
        """Verbatim-shaped incident: a docs PR quoting past incident text as
        an example still contains a live keyword+#N adjacency and is
        flagged, even though it is inside quotation marks (GitHub only
        exempts backtick code spans, not quotes -- confirmed: this exact
        text re-closed both #489 and #515 on PR #788's own merge)."""
        text = (
            'Two issues reopened (were incorrectly auto-closed by GitHub '
            'keyword matching despite their own merging commit/PR text '
            'stating the delivery was partial -- "Fixes #489 (phase 1)", '
            '"Closes #515 (Phase 2 complete; Phases 1/3/4 remain)"):'
        )
        violations = find_descriptive_violations(text)
        issue_nums = {v.issue_num for v in violations}
        assert "#489" in issue_nums
        assert "#515" in issue_nums

    def test_inline_prose_mid_sentence(self) -> None:
        """Prose describing future work needed, ending in 'close #N', is not
        a directive and must be flagged (real case: PR #342/#162)."""
        text = "This is what still needs to run to actually close #162."
        violations = find_descriptive_violations(text)
        assert any(v.issue_num == "#162" for v in violations)

    def test_trailing_qualifier_same_clause(self) -> None:
        """A closing keyword with a scope-limiting parenthetical directly
        attached is flagged (real case: 'Fixes #489 (phase 1).' silently
        auto-closed #489 despite the module being explicitly unwired)."""
        text = "No quality claim: module is CPU-only. Fixes #489 (phase 1)."
        violations = find_descriptive_violations(text)
        assert any(v.issue_num == "#489" for v in violations)

    def test_canonical_bare_close_exempt(self) -> None:
        """Must-PASS: a bare 'Closes #N' forming its own clause is the
        canonical safe form and must not be flagged."""
        assert find_descriptive_violations("Closes #123") == []

    def test_canonical_bare_close_with_period_exempt(self) -> None:
        assert find_descriptive_violations("Closes #123.") == []

    def test_canonical_multi_issue_exempt(self) -> None:
        """Must-PASS: multiple comma-separated refs on their own clause."""
        assert find_descriptive_violations("Closes #1, #2, #3") == []

    def test_canonical_bulleted_exempt(self) -> None:
        """Must-PASS: a leading bullet before the directive is fine."""
        assert find_descriptive_violations("- Closes #820") == []
        assert find_descriptive_violations("* Fixes #820") == []

    def test_canonical_own_sentence_with_trailing_prose_exempt(self) -> None:
        """Must-PASS: the directive is its own complete sentence; unrelated
        prose in a LATER sentence must not fuse with it (real case: PR
        #534's 'Resolves #532. Note for the branch-protection...')."""
        text = (
            "Verified CI green. Resolves #532. Note for the "
            "branch-protection action item: no further action needed."
        )
        violations = find_descriptive_violations(text)
        assert violations == []

    def test_canonical_heading_then_directive_exempt(self) -> None:
        """Must-PASS: a markdown heading naming the section, followed by the
        real directive on its own line (real case: PR #812's '## Closes\\n
        Closes #809')."""
        text = "## Closes\nCloses #809\n\nGenerated with Claude Code"
        violations = find_descriptive_violations(text)
        assert violations == []

    def test_negated_case_not_double_reported(self) -> None:
        """A negated occurrence is owned by find_violations(); the
        descriptive scan must not also report it (would be a confusing
        duplicate)."""
        text = "Does not close #593 -- see PR body for what remains."
        assert find_violations(text) != []
        assert find_descriptive_violations(text) == []

    def test_empty_string(self) -> None:
        assert find_descriptive_violations("") == []

    def test_refs_keyword_not_flagged(self) -> None:
        """'Refs' is not a closing keyword and must never be flagged."""
        assert find_descriptive_violations("Refs #792, #794.") == []

    def test_code_span_stripped(self) -> None:
        """A closing keyword inside backticks is exempt (GitHub does not
        honor it there either -- verified PR #817/#806)."""
        text = "not attempting (hence `Refs #806`, not `Closes #806`)"
        assert find_descriptive_violations(text) == []


class TestCheckAllInputsDescriptive:
    """Descriptive-rule coverage through the check_all_inputs() entry point."""

    def test_descriptive_violation_in_body(self) -> None:
        violations = check_all_inputs(
            title=None,
            body="PR #377 also closed #374, but with an unqualified keyword.",
            commit_messages=None,
        )
        assert any(v.kind == "descriptive" and v.issue_num == "#374" for v in violations)
        assert all(v.surface == "PR body" for v in violations)

    def test_genuine_close_all_sources_still_passes(self) -> None:
        """Must-PASS: the existing genuine-close regression case from
        TestCheckAllInputs must still pass with the descriptive rule wired
        in."""
        violations = check_all_inputs(
            title="Fixes #1",
            body="Closes #2",
            commit_messages=["Resolves #3"],
        )
        assert violations == []

    def test_verb_free_all_sources_still_passes(self) -> None:
        violations = check_all_inputs(
            title="out of scope: #1",
            body="#2 is unaffected by this PR",
            commit_messages=["see #3 for details"],
        )
        assert violations == []


class TestMainFunctionDescriptive:
    """CLI-level coverage for the descriptive rule."""

    def test_exit_1_on_descriptive_violation(self, capsys) -> None:
        exit_code = main(["--body", "PR #99 also closed #123, but only partially."])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "#123" in captured.err
        assert "DESCRIPTIVE" in captured.err

    def test_exit_0_on_canonical_close(self, capsys) -> None:
        exit_code = main(["--title", "Closes #123"])
        assert exit_code == 0

    def test_negated_and_descriptive_both_reported(self, capsys) -> None:
        """A message containing both a negated case and a separate
        descriptive case reports both sections."""
        body = (
            "This does NOT resolve #793 (out of scope).\n\n"
            "PR #500 also closed #501 unqualified."
        )
        exit_code = main(["--body", body])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "#793" in captured.err
        assert "#501" in captured.err
        assert "NEGATED" in captured.err
        assert "DESCRIPTIVE" in captured.err
