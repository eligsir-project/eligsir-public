"""Issue #718: GC-safety guard on graph-capture window.

PR #718 (`feat/issue698-eval-tick-record`) triggered SIGSEGVs (rc=139) on the
ROCm training image when pytest runs the full suite. Role B reproduced the
crash THREE times at different main heads, with byte-identical stack traces
pointing into torch's pytree flattening code, fired during garbage collection
inside an open CUDA graph capture context.

This test validates:
1. AST-based: The GC guard (gc.disable/enable) is present and wired correctly
2. Behavioral: GC is actually disabled during _capture(), and prior state restored

Both tests run CPU-only or skip if torch is unavailable, fitting the CI image
that cannot run GPU operations.
"""

from __future__ import annotations

import ast
import gc
import inspect
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_PATH = _REPO_ROOT / "eligsir" / "optimization" / "gaussian" / "graph_runtime" / "runtime.py"


def _read_source(path: Path) -> str:
    with open(path, "r") as f:
        return f.read()


def _parse_ast(source: str) -> ast.Module:
    return ast.parse(source)


class TestGraphCaptureGCSafetyAST:
    """CPU-only AST validation that GC guard is present in the _capture method."""

    def test_capture_method_has_gc_import(self):
        """gc module is imported in _capture."""
        source = _read_source(_RUNTIME_PATH)
        tree = _parse_ast(source)

        # Find _capture method
        method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_capture":
                method = node
                break

        assert method is not None, "_capture method not found"

        # Check for gc import inside the method
        has_gc_import = False
        for node in ast.walk(method):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "gc":
                        has_gc_import = True
                        break

        assert has_gc_import, "_capture must import gc module"

    def test_capture_method_has_gc_isenabled_call(self):
        """gc.isenabled() is called to check prior state."""
        source = _read_source(_RUNTIME_PATH)
        tree = _parse_ast(source)

        # Find _capture method
        method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_capture":
                method = node
                break

        assert method is not None, "_capture method not found"

        # Check for gc.isenabled() call
        has_isenabled = False
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "gc"
                        and node.func.attr == "isenabled"
                    ):
                        has_isenabled = True
                        break

        assert has_isenabled, "_capture must call gc.isenabled() to check prior state"

    def test_capture_method_has_gc_disable_call(self):
        """gc.disable() is called to protect the capture window."""
        source = _read_source(_RUNTIME_PATH)
        tree = _parse_ast(source)

        # Find _capture method
        method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_capture":
                method = node
                break

        assert method is not None, "_capture method not found"

        # Check for gc.disable() call
        has_disable = False
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "gc"
                        and node.func.attr == "disable"
                    ):
                        has_disable = True
                        break

        assert has_disable, "_capture must call gc.disable() before the capture window"

    def test_capture_method_has_gc_enable_call(self):
        """gc.enable() is called in finally block to restore prior state."""
        source = _read_source(_RUNTIME_PATH)
        tree = _parse_ast(source)

        # Find _capture method
        method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_capture":
                method = node
                break

        assert method is not None, "_capture method not found"

        # Check for gc.enable() call in try-finally
        has_enable = False
        has_finally = False

        for node in ast.walk(method):
            if isinstance(node, ast.Try):
                # Check if this try has a finally block
                if node.finalbody:
                    has_finally = True
                    # Check for gc.enable() in the finally block
                    for finally_node in ast.walk(node):
                        if isinstance(finally_node, ast.Call):
                            if isinstance(finally_node.func, ast.Attribute):
                                if (
                                    isinstance(finally_node.func.value, ast.Name)
                                    and finally_node.func.value.id == "gc"
                                    and finally_node.func.attr == "enable"
                                ):
                                    has_enable = True
                                    break

        assert has_finally, "_capture must have a try-finally block"
        assert has_enable, "_capture must call gc.enable() in finally block to restore state"

    def test_capture_method_has_conditional_gc_disable(self):
        """gc.disable() is called conditionally only if GC was enabled."""
        source = _read_source(_RUNTIME_PATH)
        tree = _parse_ast(source)

        # Find _capture method
        method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_capture":
                method = node
                break

        assert method is not None, "_capture method not found"

        # Check for conditional gc.disable()
        has_conditional_disable = False
        for node in ast.walk(method):
            if isinstance(node, ast.If):
                # Look for condition checking a gc variable
                if isinstance(node.test, ast.Name) and "gc" in node.test.id:
                    # Check if gc.disable() is in the if body
                    for if_node in ast.walk(node):
                        if isinstance(if_node, ast.Call):
                            if isinstance(if_node.func, ast.Attribute):
                                if (
                                    isinstance(if_node.func.value, ast.Name)
                                    and if_node.func.value.id == "gc"
                                    and if_node.func.attr == "disable"
                                ):
                                    has_conditional_disable = True
                                    break

        assert has_conditional_disable, (
            "_capture must conditionally call gc.disable() based on gc.isenabled() check"
        )


class TestGraphCaptureGCBehavior:
    """Behavioral tests that GC is actually disabled/restored during _capture."""

    def test_gc_disabled_during_warmup_and_capture(self):
        """Verify GC is disabled during the _capture call."""
        pytest.importorskip("torch")

        from eligsir.optimization.gaussian.graph_runtime.runtime import (
            TorchAcceleratorGraphRuntime,
        )
        from eligsir.optimization.gaussian.graph_runtime.signature import (
            GraphSignature,
        )

        # Track whether GC was disabled inside step_fn
        gc_disabled_during_step = []

        def mock_step_fn():
            # Record whether GC is disabled during step_fn execution
            gc_disabled_during_step.append(not gc.isenabled())

        runtime = TorchAcceleratorGraphRuntime(warmup_iters=1)
        signature = GraphSignature(
            device_backend="cuda",
            capacity_bucket=100,
            active_shape=(100,),
            resolution_bucket=(0, 0),
            sh_degree=0,
            renderer_mode="none",
            packed_mode=False,
            sparse_grad_mode=False,
            optimizer_mode="adam",
            loss_policy_id="test",
            gradient_policy_id="test",
        )

        # Ensure GC is enabled before we start
        gc.enable()

        # Capture should disable GC during execution
        try:
            runtime._capture(signature, mock_step_fn)
        except Exception:
            pass  # Capture may fail (e.g., no GPU), but we still check the GC state

        # After _capture returns, GC should be re-enabled
        assert gc.isenabled(), "GC should be re-enabled after _capture returns"

        # During the step_fn calls (warmup + capture), GC should have been disabled
        if gc_disabled_during_step:
            assert all(gc_disabled_during_step), (
                "GC should be disabled during step_fn execution inside _capture"
            )

    def test_gc_state_restored_on_exception(self):
        """Verify GC state is restored even if _capture raises an exception."""
        pytest.importorskip("torch")

        from eligsir.optimization.gaussian.graph_runtime.runtime import (
            TorchAcceleratorGraphRuntime,
            GraphCaptureUnavailable,
        )
        from eligsir.optimization.gaussian.graph_runtime.signature import (
            GraphSignature,
        )

        def failing_step_fn():
            raise RuntimeError("Intentional test failure")

        runtime = TorchAcceleratorGraphRuntime(warmup_iters=1)
        signature = GraphSignature(
            device_backend="cuda",
            capacity_bucket=100,
            active_shape=(100,),
            resolution_bucket=(0, 0),
            sh_degree=0,
            renderer_mode="none",
            packed_mode=False,
            sparse_grad_mode=False,
            optimizer_mode="adam",
            loss_policy_id="test",
            gradient_policy_id="test",
        )

        # Ensure GC is enabled before we start
        gc.enable()
        initial_state = gc.isenabled()

        # _capture should raise GraphCaptureUnavailable
        with pytest.raises(GraphCaptureUnavailable):
            runtime._capture(signature, failing_step_fn)

        # Even after exception, GC state should be restored to initial
        assert gc.isenabled() == initial_state, (
            "GC state must be restored in finally block, even after exception"
        )

    def test_gc_state_preserved_if_initially_disabled(self):
        """Verify GC remains disabled if it was disabled before _capture."""
        pytest.importorskip("torch")

        from eligsir.optimization.gaussian.graph_runtime.runtime import (
            TorchAcceleratorGraphRuntime,
            GraphCaptureUnavailable,
        )
        from eligsir.optimization.gaussian.graph_runtime.signature import (
            GraphSignature,
        )

        def failing_step_fn():
            raise RuntimeError("Intentional test failure")

        runtime = TorchAcceleratorGraphRuntime(warmup_iters=1)
        signature = GraphSignature(
            device_backend="cuda",
            capacity_bucket=100,
            active_shape=(100,),
            resolution_bucket=(0, 0),
            sh_degree=0,
            renderer_mode="none",
            packed_mode=False,
            sparse_grad_mode=False,
            optimizer_mode="adam",
            loss_policy_id="test",
            gradient_policy_id="test",
        )

        # Disable GC before calling _capture
        gc.disable()

        try:
            # _capture should raise GraphCaptureUnavailable
            with pytest.raises(GraphCaptureUnavailable):
                runtime._capture(signature, failing_step_fn)

            # GC should still be disabled after _capture returns
            assert not gc.isenabled(), (
                "GC should remain disabled if it was disabled before _capture"
            )
        finally:
            # Re-enable for other tests
            gc.enable()
