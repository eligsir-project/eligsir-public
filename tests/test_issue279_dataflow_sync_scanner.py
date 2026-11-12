"""Tests for the issue #279 CPU<->GPU transfer/sync site scanner.

**Deliberately torch-free** (issue #536): the CPU CI image excludes any test file
that imports torch at module scope, so a test written that way would report green
without ever executing. Nothing here imports torch, directly or transitively --
the scanner under test is pure-stdlib AST by design, which is what makes that
possible.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

_SCANNER = Path(__file__).resolve().parent.parent / "scripts" / "audit_dataflow_sync_sites.py"
_MODNAME = "audit_dataflow_sync_sites"


def _load():
    if _MODNAME in sys.modules:
        return sys.modules[_MODNAME]
    spec = importlib.util.spec_from_file_location(_MODNAME, _SCANNER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register BEFORE exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which raises AttributeError otherwise.
    sys.modules[_MODNAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _scan_source(mod, source: str, rel: str = "train_streaming.py"):
    import ast

    tree = ast.parse(textwrap.dedent(source))
    v = mod._Visitor(rel, textwrap.dedent(source).splitlines())
    v.visit(tree)
    return v.sites


def test_scanner_module_imports_no_torch():
    """The whole point of the pure-AST design (issue #536)."""
    text = _SCANNER.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "from torch" not in text


def test_detects_d2h_and_h2d_kinds():
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def f(t):
            a = t.cpu()
            b = t.item()
            c = t.numpy()
            d = t.tolist()
            e = t.cuda()
            g = t.to(device="cuda")
            return a, b, c, d, e, g
        """,
    )
    kinds = {s.kind for s in sites}
    assert {".cpu()", ".item()", ".numpy()", ".tolist()", ".cuda()", ".to(device)"} <= kinds
    dirs = {s.kind: s.direction for s in sites}
    assert dirs[".cpu()"] == "D2H"
    assert dirs[".cuda()"] == "H2D"


def test_unguarded_site_in_iteration_loop_is_hot():
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def streaming_training(loss):
            for iteration in range(10):
                v = loss.item()
            return v
        """,
    )
    item = [s for s in sites if s.kind == ".item()"]
    assert len(item) == 1
    assert item[0].classification == "HOT"
    assert item[0].in_loop is True


def test_cadence_guarded_site_is_amortized_and_records_the_guard():
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def streaming_training(loss, tb_writer, scalar_log_interval):
            for iteration in range(10):
                if iteration % scalar_log_interval == 0:
                    v = loss.item()
            return v
        """,
    )
    item = [s for s in sites if s.kind == ".item()"]
    assert len(item) == 1
    assert item[0].classification == "AMORTIZED"
    # the guard text is recorded verbatim so a reader can re-derive the label
    assert any("scalar_log_interval" in g for g in item[0].guards)


def test_cold_module_wins_over_loop_nesting():
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def anything(loss):
            for iteration in range(10):
                v = loss.item()
            return v
        """,
        rel="utils/comparison_report.py",
    )
    assert [s.classification for s in sites if s.kind == ".item()"] == ["COLD"]


def test_cold_function_name_marker_wins_over_loop_nesting():
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def write_report(loss):
            for iteration in range(10):
                v = loss.item()
            return v
        """,
    )
    assert [s.classification for s in sites if s.kind == ".item()"] == ["COLD"]


def test_non_blocking_kwarg_is_recorded():
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def streaming_training(t):
            for iteration in range(10):
                a = t.to(device="cuda", non_blocking=True)
                b = t.to(device="cuda")
            return a, b
        """,
    )
    to_sites = [s for s in sites if s.kind == ".to(device)"]
    assert len(to_sites) == 2
    assert [s.non_blocking for s in to_sites] == [True, False]


def test_bare_literal_float_call_is_not_flagged():
    """float(0.5) is not a host readback; only tensor-shaped expressions are."""
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def streaming_training(cfg):
            for iteration in range(10):
                a = float(cfg.value)
                b = float(cfg.t.sum())
            return a, b
        """,
    )
    floats = [s for s in sites if s.kind == "float(tensor)"]
    assert len(floats) == 1
    assert ".sum(" in floats[0].snippet


def test_hand_verified_overrides_are_counted_and_reachable():
    """Every override key must be well-formed, so a stale one is visible."""
    mod = _load()
    for key, (cls, reason) in mod.HAND_VERIFIED_OVERRIDES.items():
        assert "::" in key or ":" in key, key
        assert cls in {"HOT", "AMORTIZED", "COLD", "HOT_REACHABLE", "UNCLASSIFIED"}, key
        assert reason, key


def test_summarize_reports_every_classification_bucket():
    mod = _load()
    sites = _scan_source(
        mod,
        """
        def streaming_training(loss):
            for iteration in range(10):
                v = loss.item()
            return v
        """,
    )
    summary = mod.summarize(sites)
    assert summary["total"] == len(sites)
    assert summary["by_classification"]["HOT"] == 1
    assert summary["by_direction"]["D2H"] == 1
    assert "hand_verified_overrides_applied" in summary
