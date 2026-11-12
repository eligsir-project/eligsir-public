"""Verification that all MetricEntry producer fields are resolvable.

This test ensures that every producer field in the canonical metric registry
('file:function:line' format) actually points to a real, verified location in
the codebase. A failing test indicates a stale or incorrect producer declaration.

Issue #704 requires this as a recurrence guard: producers were broken before
(4 of 5 entries had unresolvable or wrong locations), and this test prevents
that from happening again silently.
"""

import ast
import pathlib
from typing import Optional, Tuple, Set

import pytest

from eligsir.utils.metric_registry import CANONICAL_METRICS


def _parse_producer(producer: str) -> Tuple[str, str, int]:
    """Parse a producer string into (file_path, function_name, line_number).

    Expected format: 'file/path.py:function_name:123'

    Raises ValueError if the format is invalid.
    """
    parts = producer.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Producer format invalid (expected 'file:function:line'): {producer!r}"
        )
    file_path, func_name, line_str = parts
    try:
        line_num = int(line_str)
    except ValueError:
        raise ValueError(
            f"Producer line number not an integer: {line_str!r} in {producer!r}"
        )
    return file_path, func_name, line_num


def _resolve_file(file_path: str) -> Optional[pathlib.Path]:
    """Resolve a file path relative to the repo root.

    Returns the absolute path if it exists, or None if not found.
    """
    # This test runs from within tests/, so we need to find the repo root
    repo_root = pathlib.Path(__file__).parent.parent
    full_path = repo_root / file_path
    return full_path if full_path.exists() else None


def _find_function_in_file(
    file_path: pathlib.Path, func_name: str
) -> Optional[Tuple[int, int]]:
    """Find a function in a file and return (start_line, end_line).

    Returns None if the function is not found. Line numbers are 1-indexed.
    """
    try:
        with open(file_path, "r") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except Exception:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # lineno is 1-indexed; end_lineno may be None for older Python versions
            start = node.lineno
            end = node.end_lineno if hasattr(node, "end_lineno") else start + 10
            return (start, end)

    return None


def _line_text(file_path: pathlib.Path, line_num: int) -> Optional[str]:
    """The raw text of a 1-indexed line, or None if unreadable/out of range."""
    try:
        lines = file_path.read_text().splitlines()
    except Exception:
        return None
    if not (1 <= line_num <= len(lines)):
        return None
    return lines[line_num - 1]


def _is_not_a_statement(line_text: str) -> bool:
    """True for a line carrying no code at all: blank, or comment-only.

    A line whose code is followed by a trailing comment (``x = 1  # note``) is
    a statement and passes.
    """
    stripped = line_text.strip()
    return not stripped or stripped.startswith("#")


def test_all_producers_resolvable():
    """Test that every producer field resolves to a real code location.

    Each producer must:
    1. Point to a file that exists in the repo
    2. Name a function that exists in that file
    3. Have a line number that falls within that function's body
    4. Have that line be an actual statement -- not a blank line or a
       comment (issue #1112; see the check itself for why)
    """
    failures = []

    for metric_name, entry in CANONICAL_METRICS.items():
        producer = entry.producer

        # Explicit, narrow exemption: "PENDING" is a genuine not-yet-implemented
        # sentinel (see each such entry's own `notes`, e.g. "NOT YET IMPLEMENTED.
        # Requires ... instrumentation") -- not a phantom/incorrect reference.
        # This does NOT weaken the fail-closed guard for any entry that claims a
        # real producer: every non-PENDING producer must still resolve to a real
        # file:function:line exactly as before.
        if producer == "PENDING":
            continue

        # Parse the producer string
        try:
            file_path_str, func_name, line_num = _parse_producer(producer)
        except ValueError as e:
            failures.append(f"{metric_name}: {e}")
            continue

        # Resolve the file path
        file_path = _resolve_file(file_path_str)
        if file_path is None:
            failures.append(
                f"{metric_name}: file not found: {file_path_str!r}"
            )
            continue

        # Find the function in the file
        func_range = _find_function_in_file(file_path, func_name)
        if func_range is None:
            failures.append(
                f"{metric_name}: function {func_name!r} not found in {file_path_str}"
            )
            continue

        # Verify the line number is within the function
        start_line, end_line = func_range
        if not (start_line <= line_num <= end_line):
            failures.append(
                f"{metric_name}: line {line_num} is outside function "
                f"{func_name!r} (defined lines {start_line}–{end_line}) in {file_path_str}"
            )
            continue

        # 4. The pinned line must be an actual statement, not a blank line or a
        #    comment. Issue #1112: `outer_loop_iteration_rate_hz` pinned
        #    `streaming_training:4907`, which was a *comment* line deep inside an
        #    unrelated nested helper -- and passed, because the containing
        #    function is large enough to span it. A producer pin that resolves to
        #    a comment documents nothing and silently survives the very drift
        #    this test exists to catch, so reject it explicitly.
        #
        #    Deliberately narrow: this cannot verify a pin points at the
        #    *intended* statement (that needs an expected-text field in the
        #    registry -- a convention change, tracked separately). It only
        #    forecloses the "pinned at nothing at all" case.
        line_text = _line_text(file_path, line_num)
        if line_text is not None and _is_not_a_statement(line_text):
            failures.append(
                f"{metric_name}: line {line_num} in {file_path_str} is not a "
                f"statement (blank or comment-only): {line_text.strip()!r}"
            )
            continue

    if failures:
        pytest.fail(
            "Producer verification failed for the following entries:\n"
            + "\n".join(f"  • {f}" for f in failures)
        )


def test_no_phantom_functions():
    """Ensure no producer references non-existent functions like '_progress_bar_update'.

    This is a specific regression guard for issue #704, where three entries
    pointed to '_progress_bar_update', a function that doesn't exist anywhere
    in the codebase.
    """
    phantom_functions = {"_progress_bar_update", "_placeholder", "_unknown"}

    failures = []
    for metric_name, entry in CANONICAL_METRICS.items():
        try:
            _file_path, func_name, _line = _parse_producer(entry.producer)
        except ValueError:
            continue  # Already caught by test_all_producers_resolvable

        if func_name in phantom_functions:
            failures.append(
                f"{metric_name}: references phantom function {func_name!r}"
            )

    if failures:
        pytest.fail(
            "Found references to phantom functions:\n"
            + "\n".join(f"  • {f}" for f in failures)
        )


def test_outer_loop_rate_pin_targets_progress_bar_construction():
    """The exact producer pin identifies the tqdm rate tracker itself.

    This pin has drifted twice in one session (issue #1112 originally, then
    again as unrelated `streaming_training` edits shifted lines): a bare
    ``_line_text(...) == "progress_bar = tqdm("`` check just compares one
    hardcoded number (the registry's) against another (this test's), and a
    stale registry number silently rots until something else notices. Search
    the function body for the anchor statement instead of hardcoding a second
    expectation, so a future drift fails here with the exact corrected line
    number to paste into utils/metric_registry.py, rather than requiring
    someone to grep for it by hand.
    """
    entry = CANONICAL_METRICS["outer_loop_iteration_rate_hz"]
    file_path_str, function_name, line_num = _parse_producer(entry.producer)
    assert function_name == "streaming_training"
    file_path = _resolve_file(file_path_str)
    assert file_path is not None

    start_line, end_line = _find_function_in_file(file_path, function_name)
    anchor = "progress_bar = tqdm("
    lines = file_path.read_text().splitlines()
    matches = [
        lineno
        for lineno in range(start_line, end_line + 1)
        if lines[lineno - 1].strip() == anchor
    ]
    assert matches, (
        f"no {anchor!r} statement found in {function_name!r} "
        f"({file_path_str}:{start_line}-{end_line}) at all -- the tqdm "
        "construction itself may have been removed or renamed"
    )
    assert len(matches) == 1, (
        f"{anchor!r} appears at multiple lines in {function_name!r}: {matches} "
        "-- the producer pin is ambiguous"
    )
    assert line_num == matches[0], (
        f"outer_loop_iteration_rate_hz's producer pin is stale: registry says "
        f"line {line_num}, but {anchor!r} is actually at line {matches[0]}. "
        f"Update utils/metric_registry.py's producer to "
        f"'{file_path_str}:{function_name}:{matches[0]}'."
    )


@pytest.mark.parametrize(
    ("metric_name", "expected_producer", "expected_statement"),
    (
        (
            "keyframe_admission_rate_hz",
                "utils/execution_semantics.py:compute_progress_postfix:1010",
            '"mapFPS": f"{_adm / _elapsed:.1f}",  # keyframes admitted per wall-second',
        ),
        (
            "per_view_iteration_budget",
                "utils/execution_semantics.py:compute_progress_postfix:1009",
            '"it/V": it_per_view,  # iterations per view (training budget)',
        ),
    ),
)
def test_progress_metric_pins_target_exact_current_producer_statements(
    metric_name, expected_producer, expected_statement
):
    entry = CANONICAL_METRICS[metric_name]
    assert entry.producer == expected_producer
    file_path_str, function_name, line_num = _parse_producer(entry.producer)
    assert function_name == "compute_progress_postfix"
    file_path = _resolve_file(file_path_str)
    assert file_path is not None
    assert _line_text(file_path, line_num).strip() == expected_statement


@pytest.mark.parametrize(
    ("metric_name", "result_field"),
    (
        ("optimization_step_count", "optimizer_step_count"),
        ("supervision_view_count", "supervision_view_count"),
        ("rasterizer_call_count", "rasterizer_call_count"),
        ("supervised_pixel_count", "supervised_pixel_count"),
        ("loss_objective_call_count", "loss_objective_call_count"),
        ("backward_call_count", "backward_call_count"),
    ),
)
def test_engine_work_pins_resolve_to_the_reported_counter(metric_name, result_field):
    file_name, function_name, line = _parse_producer(CANONICAL_METRICS[metric_name].producer)
    assert function_name == "run_batched_step"
    path = _resolve_file(file_name)
    assert path is not None
    assert _line_text(path, line).strip().startswith(f"result.{result_field} =")


if __name__ == "__main__":
    # Allow running this test standalone for debugging
    pytest.main([__file__, "-v"])
