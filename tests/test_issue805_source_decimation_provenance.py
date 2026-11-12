"""CPU tests for issue #805: persisting composed frame-decimation provenance.

Scope is deliberately narrow: recording both native (frame_stride) and generic
(source_stride) frame decimation stages and their composition into effective stride,
with explicit notation of whether frames_before is a real count or a live-stream
placeholder.

Complementary to ``test_strided_frame_source.py`` (which pins the *behavior* of
StridedFrameSource, the generic wrapping stage) and ``test_issue302_*`` (which pins
ingestion-mode provenance). This test only records what #805 adds: the decimation
stack and its composition.

What is tested here:

  1. ``resolve_source_decimation`` computes the correct effective_stride
     (frame_stride × source_stride) and composed flag for every combination.
  2. The composed flag is True iff both input strides are > 1.
  3. The function is additive: adding source_decimation to report.json["ingestion"]
     does not clobber existing keys.
  4. Frames counts (before/after) are recorded verbatim, with provenance tracking
     for placeholder vs. real counts.
"""
from __future__ import annotations

import pytest

from eligsir.mapping.scheduler import resolve_source_decimation


# --- resolve_source_decimation behavior -----------------------------------

def test_stride_1x1_not_composed():
    result = resolve_source_decimation(1, 1)
    assert result["frame_stride"] == 1
    assert result["source_stride"] == 1
    assert result["effective_stride"] == 1
    assert result["composed"] is False


def test_stride_2x1_not_composed():
    result = resolve_source_decimation(2, 1)
    assert result["frame_stride"] == 2
    assert result["source_stride"] == 1
    assert result["effective_stride"] == 2
    assert result["composed"] is False


def test_stride_1x3_not_composed():
    result = resolve_source_decimation(1, 3)
    assert result["frame_stride"] == 1
    assert result["source_stride"] == 3
    assert result["effective_stride"] == 3
    assert result["composed"] is False


def test_stride_2x3_composed():
    """Issue #834: composed strides require explicit legacy_composition_mode."""
    # Canonical mode: raises on conflict
    import pytest
    from eligsir.mapping.scheduler import ConflictingStrideError
    with pytest.raises(ConflictingStrideError):
        resolve_source_decimation(2, 3)

    # Legacy mode: allows multiplication
    result = resolve_source_decimation(2, 3, legacy_composition_mode=True)
    assert result["frame_stride"] == 2
    assert result["source_stride"] == 3
    assert result["effective_stride"] == 6
    assert result["composed"] is True
    assert result["legacy_composition_active"] is True


def test_stride_4x5_composed():
    """Issue #834: composed strides require explicit legacy_composition_mode."""
    import pytest
    from eligsir.mapping.scheduler import ConflictingStrideError
    with pytest.raises(ConflictingStrideError):
        resolve_source_decimation(4, 5)

    # Legacy mode: allows multiplication
    result = resolve_source_decimation(4, 5, legacy_composition_mode=True)
    assert result["frame_stride"] == 4
    assert result["source_stride"] == 5
    assert result["effective_stride"] == 20
    assert result["composed"] is True
    assert result["legacy_composition_active"] is True


# --- Input sanitization (matching the pattern at arguments line 745-746) ----

def test_none_stride_treated_as_1():
    """None or falsy stride is clamped to 1, matching the getattr pattern."""
    result = resolve_source_decimation(None, None)
    assert result["effective_stride"] == 1
    assert result["composed"] is False

    result = resolve_source_decimation(0, 0)
    assert result["effective_stride"] == 1
    assert result["composed"] is False

    # Mixed cases
    result = resolve_source_decimation(2, None)
    assert result["effective_stride"] == 2
    assert result["composed"] is False

    result = resolve_source_decimation(None, 3)
    assert result["effective_stride"] == 3
    assert result["composed"] is False


# --- Frame count provenance ---

def test_frames_before_and_after_recorded_verbatim():
    """Frame counts are recorded as supplied, with None allowed."""
    result = resolve_source_decimation(1, 1, frames_before=1000, frames_after=900)
    assert result["frames_before_source_stride"] == 1000
    assert result["frames_after_source_stride"] == 900

    # None is allowed for unknown counts
    result = resolve_source_decimation(1, 1, frames_before=None, frames_after=None)
    assert result["frames_before_source_stride"] is None
    assert result["frames_after_source_stride"] is None


def test_frames_source_provenance_placeholder():
    """frames_source='placeholder' sets the placeholder flag."""
    result = resolve_source_decimation(
        1, 1, frames_before=999999, frames_source="placeholder"
    )
    assert result["frames_before_is_placeholder"] is True

    result = resolve_source_decimation(
        1, 1, frames_before=999999, frames_source="pre_stride"
    )
    assert result["frames_before_is_placeholder"] is False

    result = resolve_source_decimation(1, 1, frames_source="unknown")
    assert result["frames_before_is_placeholder"] is False


# --- Block structure and additivity ---

def test_return_dict_has_all_required_keys():
    """The block includes all keys, even when counts are None."""
    # Use legacy mode for composed strides (issue #834)
    result = resolve_source_decimation(2, 3, legacy_composition_mode=True)
    required_keys = {
        "frame_stride",
        "source_stride",
        "effective_stride",
        "composed",
        "frames_before_source_stride",
        "frames_after_source_stride",
        "frames_before_is_placeholder",
    }
    assert required_keys.issubset(result.keys())
    # New #834 keys should also be present
    assert "sampling_owner" in result.keys()
    assert "legacy_composition_active" in result.keys()
    assert "conflict" in result.keys()


def test_block_is_additive_to_ingestion_report():
    """Adding source_decimation to ingestion report does not clobber existing keys."""
    # Simulate the existing _ingestion_report from train_streaming.py
    existing_ingestion_report = {
        "mode": "iter_based",
        "reason": "offline_default",
        "requested": "auto",
        "online_mode": False,
        "steps_per_frame_honored": True,
    }

    # Simulate adding the new block (use legacy mode for composed strides, issue #834)
    source_decimation_block = resolve_source_decimation(
        2, 3, frames_before=1000, frames_after=333, legacy_composition_mode=True
    )

    # Build the merged report as train_streaming.py does
    merged_report = {**existing_ingestion_report, "source_decimation": source_decimation_block}

    # Verify all existing keys are still there, unchanged
    assert merged_report["mode"] == "iter_based"
    assert merged_report["reason"] == "offline_default"
    assert merged_report["requested"] == "auto"
    assert merged_report["online_mode"] is False
    assert merged_report["steps_per_frame_honored"] is True

    # Verify the new block is there
    assert "source_decimation" in merged_report
    assert merged_report["source_decimation"]["effective_stride"] == 6
    assert merged_report["source_decimation"]["composed"] is True


# --- Exemplary run scenarios ---

def test_offline_tum_stride_1_scenario():
    """Offline TUM replay with default strides (no decimation)."""
    result = resolve_source_decimation(
        frame_stride=1,
        source_stride=1,
        frames_before=2525,  # TUM fr3 actual frame count
        frames_after=2525,
        frames_source="pre_stride",
    )
    assert result["effective_stride"] == 1
    assert result["composed"] is False
    assert result["frames_before_is_placeholder"] is False


def test_offline_tum_reader_stride_2_scenario():
    """Offline TUM with reader-side stride (native skip)."""
    result = resolve_source_decimation(
        frame_stride=2,
        source_stride=1,
        frames_before=2525,
        frames_after=1263,  # ~2525/2
        frames_source="pre_stride",
    )
    assert result["effective_stride"] == 2
    assert result["composed"] is False


def test_offline_ros2_live_placeholder_frames_scenario():
    """Offline ros2://live bag replay with placeholder frame count."""
    result = resolve_source_decimation(
        frame_stride=1,
        source_stride=2,
        frames_before=999999,  # --streaming_max_frames placeholder
        frames_after=500000,   # actual after wrapping
        frames_source="placeholder",
    )
    assert result["effective_stride"] == 2
    assert result["composed"] is False
    assert result["frames_before_is_placeholder"] is True


def test_composed_decimation_scenario():
    """Both strides > 1: issue #834 requires explicit legacy_composition_mode."""
    # Use legacy mode to test composed decimation behavior
    result = resolve_source_decimation(
        frame_stride=2,
        source_stride=3,
        frames_before=1000,
        frames_after=167,  # 1000 / 2 / 3 ≈ 167
        frames_source="pre_stride",
        legacy_composition_mode=True,
    )
    assert result["frame_stride"] == 2
    assert result["source_stride"] == 3
    assert result["effective_stride"] == 6
    assert result["composed"] is True
    assert result["legacy_composition_active"] is True
    assert result["frames_before_is_placeholder"] is False


def test_notice_text_mentions_both_strides():
    """COMPOSED_DECIMATION_NOTICE includes frame_stride, source_stride, and product."""
    from eligsir.mapping.scheduler import COMPOSED_DECIMATION_NOTICE

    # The notice is a format string; pin its load-bearing substrings
    assert "{frame_stride}" in COMPOSED_DECIMATION_NOTICE
    assert "{source_stride}" in COMPOSED_DECIMATION_NOTICE
    assert "{effective_stride}" in COMPOSED_DECIMATION_NOTICE
    # It must name the mechanism so neither stride can be misread as the sole rate
    assert "native stride" in COMPOSED_DECIMATION_NOTICE
    assert "generic stride" in COMPOSED_DECIMATION_NOTICE
    assert "composition" in COMPOSED_DECIMATION_NOTICE.lower()


# --- AST-based regression tests over train_streaming.py wiring ---------------
# (issue #805: verify the production code path is actually wired, not just
# that the helper exists and works in isolation. Pattern from PR #787 #333.)

@pytest.fixture
def ast_parse_file():
    """Fixture: parse a Python file with the ast module."""
    import ast
    def parse(path: str):
        with open(path, "r") as f:
            return ast.parse(f.read(), filename=path)
    return parse


@pytest.fixture
def post_fix_train_streaming_path() -> str:
    """Path to the current (post-fix) train_streaming.py in the worktree."""
    import os
    path = "train_streaming.py"
    if not os.path.exists(path):
        pytest.skip(f"{path} not found in current directory")
    return path


@pytest.fixture
def pre_fix_train_streaming_path(tmp_path) -> str:
    """Path to the pre-fix train_streaming.py from before the #805 wiring was introduced.

    Materializes the file from git history (merge base with origin/main) via subprocess.
    If the file cannot be resolved or the pre-fix baseline is no longer available, skips the test.
    """
    import subprocess
    import os

    # Determine the merge base (the commit where this PR diverged from main)
    try:
        merge_base = subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=os.getcwd(),
            stderr=subprocess.PIPE,
            text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("Could not determine merge base with origin/main (git unavailable or not a git repo)")

    # Try to materialize train_streaming.py from the merge base
    try:
        pre_fix_source = subprocess.check_output(
            ["git", "show", f"{merge_base}:train_streaming.py"],
            cwd=os.getcwd(),
            stderr=subprocess.PIPE,
            text=True
        )
    except subprocess.CalledProcessError:
        pytest.skip(f"Could not retrieve train_streaming.py from merge base {merge_base}")

    # Write to a temporary file
    temp_file = tmp_path / "pre_fix_train_streaming.py"
    temp_file.write_text(pre_fix_source)

    # Sanity check: verify the pre-fix baseline doesn't already contain #805 wiring
    # If it does, the negative-control tests would pass vacuously (not discriminating)
    if "source_decimation" in pre_fix_source and "resolve_source_decimation" in pre_fix_source:
        pytest.skip(
            "Pre-fix baseline from git already contains #805 wiring — "
            "the pre-fix baseline is no longer resolvable (merge base has the fix)"
        )

    return str(temp_file)


def _find_dict_key_in_assignment(tree, dict_var_name: str, key: str) -> bool:
    """Check if a dict assignment contains a key entry via AST walk."""
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == dict_var_name:
                    if isinstance(node.value, ast.Dict):
                        for dict_key in node.value.keys:
                            if isinstance(dict_key, ast.Constant) and dict_key.value == key:
                                return True
    return False


def _find_import_name(tree, name: str) -> bool:
    """Check if name is imported or referenced in the AST."""
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.names:
                for alias in node.names:
                    if alias.name == name or (alias.asname and alias.asname == name):
                        return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == name or (alias.asname and alias.asname == name):
                    return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == name:
                return True
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
    return False


def _contains_gated_print_with_format_string(tree, gate_pattern: str) -> bool:
    """Check if there's a gated print that uses COMPOSED_DECIMATION_NOTICE.

    Originally (issue #805) looked only for:
        if _native_frame_stride > 1 and _explicit_source_stride > 1:
            ...
            print(COMPOSED_DECIMATION_NOTICE.format(...), flush=True)

    Issue #834 completed the wiring this gate existed to protect: rather than
    re-deriving "both requested strides are non-1" ad hoc at this print site,
    the gate now reads the SAME resolved-and-enforced sampling policy that
    also drove the actual frame-source construction a few lines above
    (`if _sampling_owner == "both":` -- true only under the explicit
    legacy_compose escape hatch, matching resolve_source_sampling_policy's
    own "both" owner value). Both gate shapes are accepted here so this
    helper still recognizes the pre-#834-completion structure on a historical
    file (see the pre_fix test below) as well as the current one.
    """
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            has_both_strides = (
                "_native_frame_stride" in test_names and "_explicit_source_stride" in test_names
            )
            has_owner_both_gate = "_sampling_owner" in test_names and any(
                isinstance(n, ast.Constant) and n.value == "both"
                for n in ast.walk(node.test)
            )

            if has_both_strides or has_owner_both_gate:
                # Check if body contains a print call using COMPOSED_DECIMATION_NOTICE
                for body_node in ast.walk(node):
                    if isinstance(body_node, ast.Call):
                        if isinstance(body_node.func, ast.Name) and body_node.func.id == "print":
                            # Check if print's args reference COMPOSED_DECIMATION_NOTICE
                            for arg in ast.walk(body_node):
                                if isinstance(arg, ast.Attribute):
                                    if isinstance(arg.value, ast.Name):
                                        if arg.value.id == "COMPOSED_DECIMATION_NOTICE":
                                            return True
                                if isinstance(arg, ast.Name):
                                    if arg.id == "COMPOSED_DECIMATION_NOTICE":
                                        return True
    return False


def _is_unconditionally_assigned(tree, var_name: str) -> bool:
    """Check if variable is assigned (simple heuristic via ast.walk)."""
    import ast
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return True
    return False


def test_source_decimation_key_wired_post_fix(ast_parse_file, post_fix_train_streaming_path):
    """POST-FIX: source_decimation must be in _ingestion_report."""
    tree = ast_parse_file(post_fix_train_streaming_path)
    assert _find_dict_key_in_assignment(tree, "_ingestion_report", "source_decimation"), (
        "source_decimation key NOT FOUND in _ingestion_report dict. "
        "Check if the key was deleted or misnamed."
    )


def test_source_decimation_key_missing_pre_fix(ast_parse_file, pre_fix_train_streaming_path):
    """PRE-FIX: source_decimation must NOT exist (proves test is meaningful)."""
    tree = ast_parse_file(pre_fix_train_streaming_path)
    assert not _find_dict_key_in_assignment(tree, "_ingestion_report", "source_decimation"), (
        "Pre-fix file ALREADY HAS source_decimation key — test baseline violated."
    )


def test_resolve_source_decimation_imported_post_fix(ast_parse_file, post_fix_train_streaming_path):
    """POST-FIX: resolve_source_decimation must be imported or called."""
    tree = ast_parse_file(post_fix_train_streaming_path)
    assert _find_import_name(tree, "resolve_source_decimation"), (
        "resolve_source_decimation NOT FOUND in imports or body."
    )


def test_resolve_source_decimation_missing_pre_fix(ast_parse_file, pre_fix_train_streaming_path):
    """PRE-FIX: resolve_source_decimation must NOT be present."""
    tree = ast_parse_file(pre_fix_train_streaming_path)
    assert not _find_import_name(tree, "resolve_source_decimation"), (
        "Pre-fix file ALREADY HAS resolve_source_decimation — baseline violated."
    )


def test_composed_decimation_notice_gated_post_fix(ast_parse_file, post_fix_train_streaming_path):
    """POST-FIX: notice must be inside a gated (if) block, not always printing."""
    tree = ast_parse_file(post_fix_train_streaming_path)
    # Look for if blocks containing "composed" or stride-related condition
    found_gated_print = _contains_gated_print_with_format_string(tree, "composed")
    assert found_gated_print, (
        "Composed-decimation notice print NOT FOUND inside gated if block. "
        "It may be unconditional, misplaced, or deleted."
    )


def test_composed_decimation_notice_not_in_pre_fix(ast_parse_file, pre_fix_train_streaming_path):
    """PRE-FIX: notice must NOT exist (proves test is meaningful)."""
    tree = ast_parse_file(pre_fix_train_streaming_path)
    found = _contains_gated_print_with_format_string(tree, "composed")
    assert not found, "Pre-fix file ALREADY HAS gated notice — baseline violated."


def test_native_frame_stride_captured_post_fix(ast_parse_file, post_fix_train_streaming_path):
    """POST-FIX: the requested native frame stride must exist in the AST
    (unconditionally assigned).

    Issue #834 renamed `_native_frame_stride` to `_requested_frame_stride`
    (sourced from `resolve_source_sampling_application(...)
    ["requested_frame_stride"]`, the normalized copy of the originally
    requested `--streaming_frame_stride`) as part of completing the #805
    wiring into a real single-owner resolution -- the old name no longer
    exists in the file at all (see the AST-based diff audit in this PR's own
    verification for why grep/AST no longer finds it), so this test now
    checks for its direct successor instead.
    """
    tree = ast_parse_file(post_fix_train_streaming_path)
    assert _is_unconditionally_assigned(tree, "_requested_frame_stride"), (
        "_requested_frame_stride NOT FOUND in AST. "
        "Check if it was deleted or if the capture was moved inside a conditional."
    )


def test_native_frame_stride_missing_pre_fix(ast_parse_file, pre_fix_train_streaming_path):
    """PRE-FIX: _native_frame_stride must NOT exist."""
    tree = ast_parse_file(pre_fix_train_streaming_path)
    assert not _is_unconditionally_assigned(tree, "_native_frame_stride"), (
        "Pre-fix file ALREADY HAS _native_frame_stride — baseline violated."
    )


# --- Hermetic negative controls (do not depend on git, never skip) -------
# These tests verify the AST helpers correctly identify ABSENCE of #805 wiring.
# They build synthetic Python source strings that resemble the relevant regions
# but deliberately lack the #805 mechanisms. This allows the tests to run
# everywhere (CI, local, all environments) without skipping, even when git
# is unavailable or scratchpad paths don't exist.

def test_ast_helper_dict_key_absent_in_synthetic_code(ast_parse_file, tmp_path):
    """Hermetic test: _find_dict_key_in_assignment correctly returns False on code lacking the key.

    Builds a synthetic _ingestion_report dict with OTHER keys (proving the helper
    can find keys when they exist) but deliberately omits 'source_decimation'.
    """
    synthetic_code = '''
_ingestion_report = {
    "mode": "iter_based",
    "reason": "offline_default",
    "online_mode": False,
    "steps_per_frame_honored": True,
}
'''
    tmp_file = tmp_path / "synthetic_no_decimation_key.py"
    tmp_file.write_text(synthetic_code)

    tree = ast_parse_file(str(tmp_file))

    # Should return True for existing keys (proves the helper works)
    assert _find_dict_key_in_assignment(tree, "_ingestion_report", "mode"), (
        "Helper should find existing 'mode' key in dict"
    )
    assert _find_dict_key_in_assignment(tree, "_ingestion_report", "reason"), (
        "Helper should find existing 'reason' key in dict"
    )

    # Should return False for the missing key (the test itself)
    assert not _find_dict_key_in_assignment(tree, "_ingestion_report", "source_decimation"), (
        "Helper should return False when 'source_decimation' key is absent"
    )


def test_ast_helper_import_absent_in_synthetic_code(ast_parse_file, tmp_path):
    """Hermetic test: _find_import_name correctly returns False on code lacking the import.

    Builds a synthetic file with OTHER imports (proving the helper can find them)
    but deliberately omits 'resolve_source_decimation'.
    """
    synthetic_code = '''
import logging
from pathlib import Path
from typing import Optional

def some_function():
    logging.info("example")
    return Path.cwd()
'''
    tmp_file = tmp_path / "synthetic_no_import.py"
    tmp_file.write_text(synthetic_code)

    tree = ast_parse_file(str(tmp_file))

    # Should return True for existing imports (proves the helper works)
    assert _find_import_name(tree, "logging"), (
        "Helper should find imported 'logging'"
    )
    assert _find_import_name(tree, "Path"), (
        "Helper should find imported 'Path'"
    )

    # Should return False for the missing import (the test itself)
    assert not _find_import_name(tree, "resolve_source_decimation"), (
        "Helper should return False when 'resolve_source_decimation' import is absent"
    )


def test_ast_helper_gated_print_absent_in_synthetic_code(ast_parse_file, tmp_path):
    """Hermetic test: _contains_gated_print_with_format_string correctly returns False on plain code.

    Builds a synthetic file with OTHER if-blocks and prints (proving the helper can find them)
    but deliberately omits the gated print using COMPOSED_DECIMATION_NOTICE.
    """
    synthetic_code = '''
import logging

_frame_stride = 2
_source_stride = 3

# Regular if-block, unrelated to composition
if _frame_stride > 1:
    logging.info("frame stride is greater than 1")
    print("Frame stride enabled", flush=True)

# If-block using different variables, not the composition gate
if _source_stride > 1:
    notice = "Only source stride is active"
    print(notice.format(), flush=True)

# Correctly-structured gate but using a different constant
if _frame_stride > 1 and _source_stride > 1:
    OTHER_NOTICE = "This is not COMPOSED_DECIMATION_NOTICE"
    print(OTHER_NOTICE.format(native="val1", generic="val2"), flush=True)
'''
    tmp_file = tmp_path / "synthetic_no_gated_print.py"
    tmp_file.write_text(synthetic_code)

    tree = ast_parse_file(str(tmp_file))

    # Should return False (no gated print using COMPOSED_DECIMATION_NOTICE)
    assert not _contains_gated_print_with_format_string(tree, "composed"), (
        "Helper should return False when gated print using COMPOSED_DECIMATION_NOTICE is absent"
    )


def test_ast_helper_variable_assignment_absent_in_synthetic_code(ast_parse_file, tmp_path):
    """Hermetic test: _is_unconditionally_assigned correctly returns False for missing assignment.

    Builds a synthetic file with OTHER assignments (proving the helper can find them)
    but deliberately omits '_native_frame_stride' assignment.
    """
    synthetic_code = '''
_frame_stride = 2  # Different variable with similar name
_source_stride = 3  # Another variable
_timestamp = time.time()  # Completely different variable

def compute_stride():
    # Compute some values but never assign _native_frame_stride
    effective = _frame_stride * _source_stride
    return effective
'''
    tmp_file = tmp_path / "synthetic_no_var_assign.py"
    tmp_file.write_text(synthetic_code)

    tree = ast_parse_file(str(tmp_file))

    # Should return True for existing assignments (proves the helper works)
    assert _is_unconditionally_assigned(tree, "_frame_stride"), (
        "Helper should find existing '_frame_stride' assignment"
    )
    assert _is_unconditionally_assigned(tree, "_source_stride"), (
        "Helper should find existing '_source_stride' assignment"
    )

    # Should return False for the missing variable (the test itself)
    assert not _is_unconditionally_assigned(tree, "_native_frame_stride"), (
        "Helper should return False when '_native_frame_stride' assignment is absent"
    )
