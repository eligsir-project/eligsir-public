"""Tests for timing field registry and semantics (issue #702).

This module contains two types of tests:

1. Registry tests (pure Python data structure validation) — torch-free
2. Real-source AST tests (verify the actual PhaseTimer.write_summary() implementation)
   — torch-free, reads code via ast.parse, follows issue #548 precedent (test_issue729)
"""

import ast
import pathlib
from eligsir.utils import timing_fields

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_STREAMING = REPO_ROOT / "train_streaming.py"


class TestTimingFieldsRegistry:
    """Registry consistency and lookup tests."""

    def test_registry_no_errors(self):
        """Validate the timing_fields registry structure."""
        errors = timing_fields.validate_registry()
        assert errors == [], f"Registry validation failed:\n{chr(10).join(errors)}"

    def test_canonical_name_resolution(self):
        """Test that aliases resolve to canonical names correctly."""
        # Legacy aliases should resolve to canonical fields.
        assert timing_fields.get_canonical_name("iter_ms_mean") == "outer_loop_iteration_wall_ms_mean"
        assert timing_fields.get_canonical_name("optimization_step_ms_mean") == "outer_loop_iteration_wall_ms_mean"
        assert timing_fields.get_canonical_name("insert_ms_mean") == "depth_insertion_wall_ms_mean"

    def test_canonical_name_unchanged(self):
        """Test that canonical names resolve to themselves."""
        canonical = "outer_loop_iteration_wall_ms_mean"
        assert timing_fields.get_canonical_name(canonical) == canonical

    def test_unknown_field_returns_none(self):
        """Test that unknown field names return None."""
        assert timing_fields.get_canonical_name("nonexistent_field") is None
        assert timing_fields.get_field("nonexistent_field") is None

    def test_get_field_returns_entry(self):
        """Test that get_field returns the full registry entry."""
        entry = timing_fields.get_field("outer_loop_iteration_wall_ms_mean")
        assert entry is not None
        assert entry["name"] == "outer_loop_iteration_wall_ms_mean"
        assert entry["unit"] == "wall_ms"
        assert entry["aggregation"] == "mean"

    def test_implementation_status(self):
        """Test that implemented vs not-implemented fields are correctly identified."""
        # Implemented fields (produced by PhaseTimer).
        assert timing_fields.is_implemented("outer_loop_iteration_wall_ms_mean")
        assert timing_fields.is_implemented("iter_ms_mean")  # alias of implemented
        assert timing_fields.is_implemented("depth_insertion_wall_ms_mean")

        # Not implemented fields.
        assert not timing_fields.is_implemented("optimizer_update_wall_ms_mean")
        assert not timing_fields.is_implemented("render_wall_ms_mean")
        assert not timing_fields.is_implemented("optimization_transaction_wall_ms_mean")


# ============================================================================
# AST-based tests: verify the real PhaseTimer.write_summary() implementation
# ============================================================================
# Following the pattern from test_issue729_streaming_report_scope.py (PR #731),
# these tests read the actual source code via ast.parse() rather than using mocks.
# This ensures we test the REAL behavior, not a copy.
# ============================================================================


def _find_class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    """Find a method inside a class. Raises AssertionError if not found."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(
        f"Method {class_name}.{method_name}() not found in {TRAIN_STREAMING.name}"
    )


def _find_summary_dict_assignments(method: ast.FunctionDef) -> dict[str, str | None]:
    """Extract all assignments to `summary[<key>]` or `summary["key"]`.

    Returns:
        dict mapping key names to either:
        - the variable name assigned (e.g. "outer_loop_mean"), or
        - None if assigned a direct value (not a variable).
    """
    assignments = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            # Check if this is: summary[...] = <value>
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    if target.value.id == "summary":
                        # Get the key (subscript)
                        key_node = target.slice
                        if isinstance(key_node, ast.Constant):
                            key = key_node.value
                        elif isinstance(key_node, ast.Index) and isinstance(key_node.value, ast.Constant):
                            # Python 3.8 compatibility (ast.Index was removed in 3.9)
                            key = key_node.value.value
                        else:
                            continue

                        # Get the RHS value
                        if isinstance(node.value, ast.Name):
                            assignments[key] = node.value.id  # Variable assignment
                        else:
                            assignments[key] = None  # Direct value (subscript, call, etc.)
    return assignments


class TestPhaseTimerRealSource:
    """AST-based tests of the real PhaseTimer.write_summary() implementation.

    These tests read the actual train_streaming.py source and verify that:
    1. Canonical keys are assigned
    2. Deprecated aliases are still assigned
    3. Deprecated keys are assigned FROM the canonical keys (i.e., aliases, not drift)

    This follows the pattern from test_issue729_streaming_report_scope.py and is the
    fix for issue #548's "test fake drift" antipattern.
    """

    @staticmethod
    def module_tree() -> ast.Module:
        return ast.parse(TRAIN_STREAMING.read_text())

    @staticmethod
    def write_summary_method(module_tree) -> ast.FunctionDef:
        return _find_class_method(module_tree, "PhaseTimer", "write_summary")

    def test_canonical_outer_loop_keys_assigned(self):
        """Test that canonical outer-loop keys are assigned in the real code."""
        tree = ast.parse(TRAIN_STREAMING.read_text())
        method = _find_class_method(tree, "PhaseTimer", "write_summary")
        assignments = _find_summary_dict_assignments(method)

        # Canonical keys must be present.
        assert "outer_loop_iteration_wall_ms_mean" in assignments, (
            "Canonical key 'outer_loop_iteration_wall_ms_mean' not assigned "
            "in PhaseTimer.write_summary(). "
            "See issue #702."
        )
        assert "outer_loop_iteration_wall_ms_p95" in assignments, (
            "Canonical key 'outer_loop_iteration_wall_ms_p95' not assigned "
            "in PhaseTimer.write_summary(). "
            "See issue #702."
        )

    def test_deprecated_outer_loop_keys_assigned(self):
        """Test that deprecated outer-loop aliases are still assigned."""
        tree = ast.parse(TRAIN_STREAMING.read_text())
        method = _find_class_method(tree, "PhaseTimer", "write_summary")
        assignments = _find_summary_dict_assignments(method)

        # All deprecated aliases must still be present for backward compatibility.
        for key in ["optimization_step_ms_mean", "optimization_step_ms_p95",
                   "iter_ms_mean", "iter_ms_p95"]:
            assert key in assignments, (
                f"Deprecated alias '{key}' not assigned in PhaseTimer.write_summary(). "
                "Breaking change! Backward compatibility requires these keys."
            )

    def test_canonical_insertion_key_assigned(self):
        """Test that canonical depth-insertion key is assigned."""
        tree = ast.parse(TRAIN_STREAMING.read_text())
        method = _find_class_method(tree, "PhaseTimer", "write_summary")
        assignments = _find_summary_dict_assignments(method)

        assert "depth_insertion_wall_ms_mean" in assignments, (
            "Canonical key 'depth_insertion_wall_ms_mean' not assigned "
            "in PhaseTimer.write_summary(). "
            "See issue #702."
        )

    def test_deprecated_insertion_key_assigned(self):
        """Test that deprecated insertion alias is still assigned."""
        tree = ast.parse(TRAIN_STREAMING.read_text())
        method = _find_class_method(tree, "PhaseTimer", "write_summary")
        assignments = _find_summary_dict_assignments(method)

        assert "insert_ms_mean" in assignments, (
            "Deprecated alias 'insert_ms_mean' not assigned in PhaseTimer.write_summary(). "
            "Breaking change! Backward compatibility requires this key."
        )

    def test_deprecated_keys_alias_canonical_outer_loop(self):
        """Test that deprecated outer-loop keys are assigned FROM canonical keys (aliases).

        This is the critical test: it verifies that deprecated keys are simple aliases
        that reference the canonical variable, not independently recomputed values.
        This prevents formula drift (issue #548 antipattern).
        """
        tree = ast.parse(TRAIN_STREAMING.read_text())
        method = _find_class_method(tree, "PhaseTimer", "write_summary")
        assignments = _find_summary_dict_assignments(method)

        # Canonical keys are assigned to variables (e.g., outer_loop_mean, outer_loop_p95)
        canonical_mean_var = assignments["outer_loop_iteration_wall_ms_mean"]
        canonical_p95_var = assignments["outer_loop_iteration_wall_ms_p95"]

        assert canonical_mean_var is not None, (
            "Canonical key should be assigned to a variable, not a direct value"
        )
        assert canonical_p95_var is not None, (
            "Canonical p95 key should be assigned to a variable"
        )

        # Deprecated aliases must be assigned FROM those same variables.
        for deprecated_key, expected_var in [
            ("optimization_step_ms_mean", canonical_mean_var),
            ("optimization_step_ms_p95", canonical_p95_var),
            ("iter_ms_mean", canonical_mean_var),
            ("iter_ms_p95", canonical_p95_var),
        ]:
            actual_var = assignments[deprecated_key]
            assert actual_var == expected_var, (
                f"Deprecated key '{deprecated_key}' should be assigned from "
                f"'{expected_var}', but is assigned from '{actual_var}'. "
                "This suggests formula drift (issue #548 antipattern). "
                "Deprecated keys must be simple aliases to the canonical value."
            )

    def test_deprecated_insertion_key_aliases_canonical(self):
        """Test that deprecated insertion key is assigned FROM the canonical key."""
        tree = ast.parse(TRAIN_STREAMING.read_text())
        method = _find_class_method(tree, "PhaseTimer", "write_summary")
        assignments = _find_summary_dict_assignments(method)

        # Canonical insertion key is assigned to a variable
        canonical_insert_var = assignments["depth_insertion_wall_ms_mean"]

        # Deprecated alias must be assigned FROM that same variable
        deprecated_insert_var = assignments["insert_ms_mean"]
        assert deprecated_insert_var == canonical_insert_var, (
            f"Deprecated key 'insert_ms_mean' should be assigned from "
            f"'{canonical_insert_var}', but is assigned from '{deprecated_insert_var}'. "
            "This suggests formula drift (issue #548 antipattern)."
        )


class TestTimingFieldsNaming:
    """Naming consistency and semantic checks."""

    def test_optimizer_update_fields_narrow_bracket(self):
        """Test that fields with 'optimizer_update' describe narrow brackets, not outer-loop."""
        for field_name, entry in timing_fields.TIMING_FIELDS.items():
            if "optimizer_update" in field_name:
                start = entry["start_bracket"].lower()
                # Should mention optimizer.step(), not "iteration" or "outer".
                assert "optimizer" in start or "NOT_IMPLEMENTED" in entry["producers"], \
                    f"Field {field_name} has 'optimizer_update' but start_bracket doesn't mention optimizer: {start}"

    def test_all_aliases_have_canonical(self):
        """Test that every alias points to a documented canonical field."""
        for field_name, entry in timing_fields.TIMING_FIELDS.items():
            alias_of = entry.get("alias_of")
            if alias_of is not None:
                assert alias_of in timing_fields.TIMING_FIELDS, \
                    f"Alias {field_name} points to non-existent canonical {alias_of}"

    def test_required_keys_all_fields(self):
        """Test that all fields have the required metadata keys."""
        required = {"name", "unit", "aggregation", "producers", "start_bracket",
                   "end_bracket", "included_phases", "excluded_phases", "window",
                   "alias_of", "notes"}
        for field_name, entry in timing_fields.TIMING_FIELDS.items():
            missing = required - set(entry.keys())
            assert not missing, \
                f"Field {field_name} missing keys: {missing}"

    def test_unit_values_valid(self):
        """Test that unit values are from a known set."""
        valid_units = {"wall_ms", "gpu_ms", "count"}
        for field_name, entry in timing_fields.TIMING_FIELDS.items():
            unit = entry.get("unit")
            assert unit in valid_units, \
                f"Field {field_name} has invalid unit: {unit}"

    def test_aggregation_values_valid(self):
        """Test that aggregation values are from a known set."""
        valid_aggs = {"mean", "p95", "sum"}  # sum for count fields
        for field_name, entry in timing_fields.TIMING_FIELDS.items():
            agg = entry.get("aggregation")
            assert agg in valid_aggs, \
                f"Field {field_name} has invalid aggregation: {agg}"
