"""Keep optimization independent of application entry points.

Temporary compatibility adapters may live under ``mapping.integration``;
the optimization package itself must not import ``train_streaming``.
"""

from __future__ import annotations

import ast
from pathlib import Path


def test_optimization_has_no_runtime_train_streaming_imports():
    root = Path(__file__).resolve().parents[1]
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name == "train_streaming" for alias in node.names):
                    violations.append(str(path))
            elif isinstance(node, ast.ImportFrom) and node.module == "train_streaming":
                violations.append(str(path))
    assert not violations, f"optimization may not import train_streaming: {violations}"
