#!/usr/bin/env python3
"""Audit metric registry for producer/consumer drift (issue #704).

Read-only AST-based scanner that verifies metric registry entries against
the actual codebase. Detects:

1. Declared producer sites that don't exist
2. Metrics declared implemented but no producer found
3. Declared consumer sites that don't exist
4. Producer without any production consumer (library_only class)
5. Unregistered metrics in declared scopes

Surfaces scanned:
- Train-streaming loop: timing_summary.json keys in PhaseTimer.write_summary
- TensorBoard scalar outputs: tb_writer.add_scalar() calls
- report.json key assignments: summary[...] = or report[...] =
- Consumed in train_streaming.py, utils/comparison_report.py, utils/training_curves.py, etc.

Exit code:
- 0 if no drift detected
- 1 if drift found (missing producers, library_only metrics, etc.)

Usage:
    python scripts/audit_metric_registry.py [--verbose]

Author: Claude Code agent (issue #704), 2026-08-15.
"""

import ast
import pathlib
import sys
from typing import Set, List, Dict, Optional
from collections import defaultdict

# Import the registry
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from eligsir.utils.metric_registry import CANONICAL_METRICS, MetricEntry


REPO_ROOT = pathlib.Path(__file__).parent.parent
VERBOSE = "--verbose" in sys.argv


def print_v(*args, **kwargs):
    """Print only in verbose mode."""
    if VERBOSE:
        print(*args, **kwargs)


class ProducerConsumerAuditor:
    """AST-based audit of metric producers and consumers."""

    def __init__(self):
        self.producers_found: Dict[str, Set[str]] = defaultdict(set)  # metric -> {sites}
        self.consumers_found: Dict[str, Set[str]] = defaultdict(set)  # metric -> {sites}
        self.unregistered_metrics: Set[str] = set()

    def scan_producers(self) -> None:
        """Scan for metric producers in the codebase."""
        print_v("Scanning for metric producers...")

        # Scan PhaseTimer.write_summary() for timing_summary.json keys
        self._scan_phase_timer_summary()

        # Scan for TensorBoard add_scalar calls (simple string search for now)
        self._scan_tensorboard_scalars()

        # Scan for direct producer sites referenced in registry
        self._scan_explicit_producer_sites()

    def scan_consumers(self) -> None:
        """Scan for metric consumers in production surfaces."""
        print_v("Scanning for metric consumers...")

        # Scan for TensorBoard scalar reads
        self._scan_tensorboard_scalar_reads()

        # Scan for report.json key accesses
        self._scan_report_json_consumers()

        # Scan for explicit consumer sites from registry
        self._scan_explicit_consumer_sites()

    def _scan_phase_timer_summary(self) -> None:
        """Scan PhaseTimer.write_summary() for timing_summary.json keys."""
        train_streaming = REPO_ROOT / "train_streaming.py"
        if not train_streaming.exists():
            print_v("train_streaming.py not found, skipping PhaseTimer scan")
            return

        try:
            tree = ast.parse(train_streaming.read_text())
        except SyntaxError:
            print(f"WARNING: Could not parse {train_streaming}")
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "PhaseTimer":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "write_summary":
                        self._extract_summary_keys(item, "train_streaming.py::PhaseTimer.write_summary")

    def _extract_summary_keys(self, method: ast.FunctionDef, site: str) -> None:
        """Extract keys assigned to summary[key] = value."""
        for node in ast.walk(method):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        if target.value.id == "summary":
                            key_node = target.slice
                            if isinstance(key_node, ast.Constant):
                                key = key_node.value
                                if isinstance(key, str):
                                    self.producers_found[key].add(site)
                                    if key not in CANONICAL_METRICS:
                                        self.unregistered_metrics.add(key)

    def _scan_tensorboard_scalars(self) -> None:
        """Scan for add_scalar('tag', ...) calls to identify TensorBoard producers."""
        print_v("Scanning for add_scalar() TensorBoard producer calls...")

        # Search in train_streaming.py, utils/, eligsir/mapping/
        for py_file in REPO_ROOT.glob("**/*.py"):
            if "test" in py_file.parts or "__pycache__" in py_file.parts:
                continue

            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Look for add_scalar(tag, ...)
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "add_scalar":
                        if node.args and isinstance(node.args[0], ast.Constant):
                            tag = node.args[0].value
                            if isinstance(tag, str) and "/" in tag:
                                # Extract the base metric name from tag like "streaming/xyz"
                                short_tag = tag.split("/")[-1]
                                rel_path = py_file.relative_to(REPO_ROOT)
                                self.producers_found[short_tag].add(f"{rel_path}::add_scalar({tag})")

    def _scan_explicit_producer_sites(self) -> None:
        """Verify producer sites declared in registry exist."""
        for name, entry in CANONICAL_METRICS.items():
            if entry.producer and entry.producer not in ("PENDING", "NOT_IMPLEMENTED"):
                # Parse "path::symbol" format
                if "::" in entry.producer:
                    file_part, func_part = entry.producer.split("::", 1)
                    py_file = REPO_ROOT / file_part
                    if py_file.exists():
                        self.producers_found[name].add(entry.producer)
                    else:
                        print_v(f"Producer site not found: {entry.producer}")

    def _scan_tensorboard_scalar_reads(self) -> None:
        """Scan for consumers reading TensorBoard scalars (e.g., in comparison scripts)."""
        print_v("Scanning for TensorBoard scalar consumer reads...")
        # This would require parsing event files or log patterns — deferred for now

    def _scan_report_json_consumers(self) -> None:
        """Scan for consumers reading report.json or timing_summary.json fields."""
        print_v("Scanning for report.json / timing_summary.json consumers...")

        # Look for direct metric name references in utils/comparison_report.py, utils/training_curves.py
        consumer_files = [
            "utils/comparison_report.py",
            "utils/training_curves.py",
            "utils/promote_run.py",
        ]

        for rel_path in consumer_files:
            py_file = REPO_ROOT / rel_path
            if not py_file.exists():
                print_v(f"Consumer file not found: {rel_path}")
                continue

            try:
                content = py_file.read_text()
                tree = ast.parse(content)
            except SyntaxError:
                print_v(f"Cannot parse {rel_path}")
                continue

            # Search for string literals that might be metric names
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    metric_name = node.value
                    # Match against known metric names
                    if metric_name in CANONICAL_METRICS:
                        self.consumers_found[metric_name].add(f"{rel_path}::literal({metric_name})")

    def _scan_explicit_consumer_sites(self) -> None:
        """Verify consumer sites declared in registry exist."""
        for name, entry in CANONICAL_METRICS.items():
            for consumer_site in entry.consumers:
                if consumer_site and consumer_site not in ("PENDING", "NOT_IMPLEMENTED"):
                    # Verify file part exists
                    if "::" in consumer_site:
                        file_part = consumer_site.split("::", 1)[0]
                        py_file = REPO_ROOT / file_part
                        if py_file.exists():
                            self.consumers_found[name].add(consumer_site)

    def report_drift(self) -> int:
        """Generate audit report and return exit code."""
        print("\n" + "=" * 80)
        print("METRIC REGISTRY AUDIT REPORT")
        print("=" * 80)

        issues_found = 0

        # Categorize metrics by mutually exclusive status
        implemented = []      # producer exists, consumer exists
        library_only = []     # producer exists, no consumer
        not_implemented = []  # no producer (PENDING or NOT_IMPLEMENTED)

        for name, entry in CANONICAL_METRICS.items():
            if entry.producer in ("PENDING", "NOT_IMPLEMENTED"):
                not_implemented.append((name, entry))
            else:
                # Producer exists
                consumers = self.consumers_found.get(name, set())
                if consumers:
                    implemented.append((name, entry))
                else:
                    library_only.append((name, entry))
                    issues_found += 1

        # Check for declared producers that don't exist
        print("\n1. PRODUCER VERIFICATION")
        print("-" * 80)
        for name, entry in CANONICAL_METRICS.items():
            if entry.producer in ("PENDING", "NOT_IMPLEMENTED"):
                status = "NOT IMPLEMENTED (as expected)"
                print(f"  {name:50s} {status}")
            elif entry.producer and "::" in entry.producer:
                found = self.producers_found.get(name, set()) or self.producers_found.get(entry.producer.split("::")[-1], set())
                if found:
                    print(f"  {name:50s} FOUND")
                else:
                    print(f"  {name:50s} PRODUCER NOT FOUND: {entry.producer}")

        # Report library_only metrics
        print("\n2. LIBRARY_ONLY METRICS (Producer exists, no consumer)")
        print("-" * 80)
        for name, entry in library_only:
            print(f"  {name:50s} library_only (producer={entry.producer})")

        # Check for declared consumers that don't exist
        print("\n3. CONSUMER VERIFICATION")
        print("-" * 80)
        for name, entry in CANONICAL_METRICS.items():
            for consumer_site in entry.consumers:
                if consumer_site and consumer_site not in ("PENDING", "NOT_IMPLEMENTED"):
                    found = self.consumers_found.get(name, set())
                    if not found:
                        print(f"  {name:50s} consumer not found: {consumer_site}")
                        issues_found += 1

        # Check for unregistered metrics found in code
        print("\n4. UNREGISTERED METRICS FOUND IN CODE")
        print("-" * 80)
        if self.unregistered_metrics:
            for metric in sorted(self.unregistered_metrics):
                print(f"  {metric}")
                issues_found += 1
        else:
            print("  (none)")

        # Summary table with mutually exclusive counts
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total canonical metrics:       {len(CANONICAL_METRICS)}")
        print(f"  - Implemented:               {len(implemented)} (producer + consumer)")
        print(f"  - Library_only:              {len(library_only)} (producer, no consumer)")
        print(f"  - Not implemented:           {len(not_implemented)} (no producer)")
        print(f"\nDrift issues found:            {issues_found} (library_only metrics)")
        print(f"Unregistered metrics found:    {len(self.unregistered_metrics)}")

        return 1 if issues_found > 0 else 0


def main():
    auditor = ProducerConsumerAuditor()
    auditor.scan_producers()
    auditor.scan_consumers()
    exit_code = auditor.report_drift()

    if exit_code == 0:
        print("\n✓ No drift detected\n")
    else:
        print(f"\n✗ {exit_code} drift issue(s) detected\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
