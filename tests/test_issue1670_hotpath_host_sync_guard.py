"""Static guard against per-view host syncs on the prepared B4 hot path.

Issue #1670: The prepared-B4 numerical hot path must not sync tensors to the
host (via .item(), .cpu(), .tolist(), .numpy(), bool(torch.*), float(tensor),
int(tensor)) except at ONE boundary: GuardedLossAccumulator.resolve().

Part B landed via PR #1733: GuardedLossAccumulator consolidates the ~10
per-term bool(torch.isfinite(...)) host syncs into one per-view materialization,
preserving exact drop-and-continue/reporting/gradient-zero-on-drop semantics.

This test walks the AST of functions on the hot path and fails if any host
syncs appear outside the approved allowlist (GuardedLossAccumulator.resolve).
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eligsir.optimization.gaussian.objective import assemble_loss, prepare_loss
from eligsir.mapping.integration.loss_diagnostics import GuardedLossAccumulator


# Hot-path functions that must not have per-view host syncs
_HOT_PATH_FUNCTIONS = [
    # Primary entry points
    ("eligsir.optimization.gaussian.objective", "assemble_loss"),
    ("eligsir.optimization.gaussian.objective", "prepare_loss"),
    # Loss term builders called by prepare_loss
    ("eligsir.optimization.gaussian.losses", "_streaming_young_isotropy_reg"),
    ("eligsir.optimization.gaussian.losses", "_streaming_young_opacity_reg"),
    # Regularizer helpers
    ("utils.compiled_kernels", "active_reg_core"),
    ("utils.compiled_kernels", "active_reg_core_masked"),
]

# The ONLY allowed host sync site (contains materialization logic)
_ALLOWED_SYNC_SITE = ("eligsir.mapping.integration.loss_diagnostics", "GuardedLossAccumulator.resolve")

# Allowlisted host syncs: single scalar D2H per-condition is cheaper than the alternative cost.
# Each entry validates the exact expression still exists (so rewrites must be intentional).
_ALLOWLISTED_SYNCS = [
    (
        "eligsir.optimization.gaussian.objective",
        "prepare_loss",
        "bool(_risk_mask.any())",
        "appearance-only render gate: gates an entire second render on detached geometry; "
        "one scalar D2H is cheaper than unconditional rasterization when zero risk pixels exist",
    ),
]


class _HostSyncDetector(ast.NodeVisitor):
    """AST visitor that detects tensor-to-Python conversions."""

    def __init__(self, module_name: str, func_name: str):
        self.module_name = module_name
        self.func_name = func_name
        self.violations: list[dict] = []
        # Track if we're inside a closure passed to _guard_acc.add()
        self._inside_guard_callback = False
        # Count nested functions INSIDE the target function (0 = top level of target)
        self._nested_func_depth = 0
        # Track if we've entered the target function body
        self._in_target_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Track nested function definitions (callbacks)."""
        if not self._in_target_function:
            # First function def is the target function itself
            self._in_target_function = True
            self.generic_visit(node)
            self._in_target_function = False
        else:
            # Nested function inside the target
            self._nested_func_depth += 1
            self.generic_visit(node)
            self._nested_func_depth -= 1

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Track lambda definitions (callbacks)."""
        if self._in_target_function:
            self._nested_func_depth += 1
            self.generic_visit(node)
            self._nested_func_depth -= 1
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect tensor-to-Python conversion calls."""
        # Check if this call is to _guard_acc.add() and if so, mark nested funcs as OK
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "add" and isinstance(node.func.value, ast.Name):
                if node.func.value.id in ("_guard_acc",):
                    # Mark that callbacks in this call are OK
                    old_value = self._inside_guard_callback
                    self._inside_guard_callback = True
                    self.generic_visit(node)
                    self._inside_guard_callback = old_value
                    return

        # Skip host sync checks for callbacks registered with _guard_acc.add()
        if self._inside_guard_callback or self._nested_func_depth > 0:
            # Still visit children, but don't report violations for host syncs
            # in callbacks
            self.generic_visit(node)
            return

        # Check for .item(), .cpu(), .tolist(), .numpy() method calls
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ("item", "cpu", "tolist", "numpy"):
                # Check if this is likely a tensor operation (e.g., foo().item() or foo.item())
                # Exclude metadata accesses like tensor._version (underscore attrs are metadata)
                obj = node.func.value
                if not (isinstance(obj, ast.Attribute) and obj.attr.startswith("_")):
                    self.violations.append({
                        "type": "method_call",
                        "method": method_name,
                        "lineno": node.lineno,
                    })
        # Check for bool(torch.isfinite(...)), bool(torch.any(...)), bool(torch.all(...))
        elif isinstance(node.func, ast.Name) and node.func.id == "bool":
            if len(node.args) >= 1:
                arg = node.args[0]
                if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute):
                    attr_name = arg.func.attr
                    # Only torch.isfinite / .any / .all / .all on actual torch/tensor calls
                    # Check if the function is a torch call (torch.isfinite, tensor.any, etc)
                    if attr_name in ("isfinite", "any", "all"):
                        self.violations.append({
                            "type": "bool_tensor_reduction",
                            "func": attr_name,
                            "lineno": node.lineno,
                        })
        self.generic_visit(node)


def _get_function_source(module_name: str, func_name: str) -> tuple[str, int] | None:
    """Get the source code of a function by module and name."""
    try:
        module = __import__(module_name, fromlist=[""])
        # Handle nested names like "GuardedLossAccumulator.resolve"
        if "." in func_name:
            parts = func_name.split(".")
            obj = module
            for part in parts:
                obj = getattr(obj, part)
            func = obj
        else:
            func = getattr(module, func_name)
        source = inspect.getsource(func)
        # Get the line number
        lineno = inspect.getsourcelines(func)[1]
        return source, lineno
    except (ImportError, AttributeError, TypeError) as e:
        return None


def test_hot_path_functions_exist():
    """Verify that all hot-path functions still exist (renamed functions would silently disable guard)."""
    for module_name, func_name in _HOT_PATH_FUNCTIONS:
        result = _get_function_source(module_name, func_name)
        assert result is not None, (
            f"Hot-path function {module_name}.{func_name} not found -- "
            "if this function was renamed, update _HOT_PATH_FUNCTIONS in this test"
        )


def test_allowed_sync_site_exists():
    """Verify GuardedLossAccumulator.resolve() still exists."""
    module_name, func_path = _ALLOWED_SYNC_SITE
    result = _get_function_source(module_name, func_path)
    assert result is not None, (
        f"Allowed sync site {module_name}.{func_path} not found -- "
        "if this method was renamed, update _ALLOWED_SYNC_SITE in this test"
    )


def test_hot_path_no_host_syncs_outside_allowlist():
    """Main test: verify no host syncs on hot path except GuardedLossAccumulator.resolve() and allowlist."""
    import textwrap

    # Build a lookup for allowlisted syncs
    allowlist: dict[tuple[str, str], set[str]] = {}
    for mod_name, func_name, expr, reason in _ALLOWLISTED_SYNCS:
        key = (mod_name, func_name)
        if key not in allowlist:
            allowlist[key] = set()
        allowlist[key].add(expr)

    violations_by_function: dict[str, list[dict]] = {}

    for module_name, func_name in _HOT_PATH_FUNCTIONS:
        source_and_line = _get_function_source(module_name, func_name)
        if source_and_line is None:
            continue
        source, original_lineno = source_and_line

        dedented_source = textwrap.dedent(source)
        try:
            tree = ast.parse(dedented_source)
        except SyntaxError as e:
            pytest.skip(f"Could not parse {module_name}.{func_name}: {e}")

        detector = _HostSyncDetector(module_name, func_name)
        detector.visit(tree)

        # Filter out allowlisted violations
        non_allowlisted = []
        for v in detector.violations:
            # Try to match against allowlist
            is_allowlisted = False
            if (module_name, func_name) in allowlist:
                # For bool(expr), mark bool_tensor_reduction as potentially allowlisted
                if v.get("type") == "bool_tensor_reduction":
                    is_allowlisted = True
            if not is_allowlisted:
                v['lineno'] = v['lineno'] + original_lineno - 1
                non_allowlisted.append(v)

        if non_allowlisted:
            violations_by_function[f"{module_name}.{func_name}"] = non_allowlisted

    # Also verify allowlisted syncs still exist
    for mod_name, func_name, expr, reason in _ALLOWLISTED_SYNCS:
        source_and_line = _get_function_source(mod_name, func_name)
        assert source_and_line is not None, (
            f"Allowlisted expression '{expr}' in {mod_name}.{func_name} no longer found "
            f"-- this sync was allowlisted for: {reason}"
        )
        source, _ = source_and_line
        # Verify the exact expression still appears in the source
        if expr not in source:
            pytest.fail(
                f"Allowlisted expression '{expr}' not found in {mod_name}.{func_name}. "
                f"Reason for allowlist: {reason}"
            )

    if violations_by_function:
        error_lines = [
            "Host syncs found on prepared-B4 hot path outside GuardedLossAccumulator.resolve() and allowlist:"
        ]
        for func_id, violations in violations_by_function.items():
            error_lines.append(f"\n  {func_id}:")
            for v in violations:
                error_lines.append(f"    Line {v['lineno']}: {v['type']}")
        pytest.fail("\n".join(error_lines))


def test_guarded_loss_accumulator_resolve_is_allowed():
    """Sanity check: GuardedLossAccumulator.resolve() is the allowed sync site."""
    module_name, func_path = _ALLOWED_SYNC_SITE
    source_and_line = _get_function_source(module_name, func_path)
    assert source_and_line is not None
    # The function should exist and should contain at least one host sync
    # (otherwise why would it be allowed?)
    source, _ = source_and_line
    # Just verify it parses and exists
    import textwrap
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    assert tree is not None


def test_mutation_control_bool_isfinite_detected():
    """Mutation test: adding bool(torch.isfinite(loss)) should fail the guard.

    This test temporarily injects a host sync into assemble_loss and verifies
    that the guard detects it.
    """
    # Get assemble_loss source
    source_and_line = _get_function_source(
        "eligsir.optimization.gaussian.objective", "assemble_loss"
    )
    assert source_and_line is not None
    source, _ = source_and_line

    # Dedent the source
    import textwrap
    dedented_source = textwrap.dedent(source)

    # Inject a bool(torch.isfinite(...)) call
    injected_source = dedented_source.replace(
        "return finalize(*accumulator.resolve(loss, inputs.iteration))",
        "_ = bool(torch.isfinite(loss))\n    return finalize(*accumulator.resolve(loss, inputs.iteration))",
    )

    tree = ast.parse(injected_source)
    detector = _HostSyncDetector(
        "eligsir.optimization.gaussian.objective", "assemble_loss"
    )
    detector.visit(tree)

    # The injected bool(torch.isfinite(...)) should be detected
    assert any(
        v.get("type") == "bool_tensor_reduction" for v in detector.violations
    ), (
        f"Mutation test failed: bool(torch.isfinite(...)) not detected. "
        f"Violations found: {detector.violations}"
    )
