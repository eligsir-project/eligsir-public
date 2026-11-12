"""Tests for issue #259's git_provenance recording.

Issue #259: ablator runners (and, per this session's own coordinator
findings, concurrently-dispatched agents) execute jobs from a mutable
shared git checkout with nothing pinning a job to a commit. Live evidence
from queue.jsonl (2026-08-13): jobs `w3106inlinefix_b2fixed` and
`issue118_raw_control` were quarantined with a real drift_warning
(r9700 at commit 38145db7 vs main's checkout last observed at 39fb3aaf),
and `issue278cadence_admit_all`/`fr3postmerge_shui16_prefix_control` ran
to completion with drift already flagged (r9700 at 3bfd346 vs main's
checkout observed at 87954a33). `cfg_args`/`report.json` recorded only
*arguments* before this change -- never the commit that actually ran --
so a drifted run was silently indistinguishable from a clean one in its
own artifacts.

This module tests the eligsir-side half of the fix:
`utils.comparison_report.git_provenance_from_args()` (build the record
from an args namespace) and `write_post_training_report()`'s
`git_provenance` kwarg (write it into report.json). The ablator-side half
(pin/isolate each job's checkout, e.g. via a per-job worktree) is a
separate, out-of-repo proposal -- see
docs/architecture/2026-08-13-issue259-ablator-checkout-isolation.md.
"""

from __future__ import annotations

import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True, scope="module")
def _mock_lpips_import():
    had_original = "lpipsPyTorch" in sys.modules
    original = sys.modules.get("lpipsPyTorch")
    sys.modules["lpipsPyTorch"] = MagicMock()
    try:
        yield
    finally:
        if had_original:
            sys.modules["lpipsPyTorch"] = original
        else:
            sys.modules.pop("lpipsPyTorch", None)


from eligsir.utils.comparison_report import git_provenance_from_args, write_post_training_report


class _FakeGaussian:
    def get_xyz(self):
        import torch
        return torch.zeros(0, 3)


class _FakePipe:
    pass


class _FakeCam:
    def __init__(self, name):
        import torch
        self.image_name = name
        self.original_image = torch.zeros(3, 64, 64)


def _minimal_render_fn(cam, gaussians, pipe, background):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"render": torch.zeros(3, 64, 64, device=device)}


# --- git_provenance_from_args() ---------------------------------------------


def test_git_provenance_from_args_extracts_all_three_fields():
    args = SimpleNamespace(git_commit="abc123", git_branch="main", git_dirty=False)
    result = git_provenance_from_args(args)
    assert result == {"git_commit": "abc123", "git_branch": "main", "git_dirty": False}


def test_git_provenance_from_args_none_input_returns_none():
    assert git_provenance_from_args(None) is None


def test_git_provenance_from_args_missing_attrs_returns_none():
    """An args namespace that never went through get_git_metadata() (e.g. a
    hand-built test fixture, or a call path that predates this feature)
    must not produce a misleading empty/partial dict -- degrade to None so
    the caller's existing (pre-#259) behavior is preserved exactly."""
    args = SimpleNamespace(model_path="foo")
    assert git_provenance_from_args(args) is None


def test_git_provenance_from_args_partial_attrs_fills_unknown():
    """If exactly one of the three is set (defensive -- get_git_metadata()
    always sets all three or none), the others read back as 'unknown'
    rather than silently omitted."""
    args = SimpleNamespace(git_commit="abc123")
    result = git_provenance_from_args(args)
    assert result == {"git_commit": "abc123", "git_branch": "unknown", "git_dirty": "unknown"}


def test_git_provenance_from_args_includes_unavailable_reason_when_nonempty():
    """When git provenance capture failed, unavailable_reason is recorded
    in the result to distinguish the failure mode (issue #784)."""
    reason = "[git_binary_missing] FileNotFoundError: git not found"
    args = SimpleNamespace(
        git_commit="unknown",
        git_branch="unknown",
        git_dirty="unknown",
        git_provenance_unavailable_reason=reason,
    )
    result = git_provenance_from_args(args)
    assert result["unavailable_reason"] == reason


def test_git_provenance_from_args_omits_empty_unavailable_reason():
    """When git provenance succeeds, unavailable_reason is empty and
    is not included in the result (backward compatibility)."""
    args = SimpleNamespace(
        git_commit="abc123",
        git_branch="main",
        git_dirty=False,
        git_provenance_unavailable_reason="",
    )
    result = git_provenance_from_args(args)
    assert "unavailable_reason" not in result
    assert result == {"git_commit": "abc123", "git_branch": "main", "git_dirty": False}


# --- write_post_training_report(git_provenance=...) -------------------------


def test_write_post_training_report_records_git_provenance():
    provenance = {"git_commit": "deadbeef", "git_branch": "agent/issue259", "git_dirty": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=100,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
            git_provenance=provenance,
        )
        assert summary["git_provenance"] == provenance

        report_path = f"{tmpdir}/comparison/iter_100/report.json"
        with open(report_path) as f:
            on_disk = json.load(f)
        assert on_disk["git_provenance"] == provenance


def test_write_post_training_report_omits_git_provenance_when_not_provided():
    """Omitting the kwarg (every call site before this feature existed, and
    any call site that hasn't been wired yet) reproduces the exact
    report.json from before this parameter existed -- no 'git_provenance'
    key at all, not a null placeholder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = write_post_training_report(
            model_path=tmpdir,
            iteration=100,
            gaussians=_FakeGaussian(),
            train_cams=[_FakeCam("a")],
            test_cams=[_FakeCam("t")],
            render_fn=_minimal_render_fn,
            pipe=_FakePipe(),
            background=None,
        )
        assert "git_provenance" not in summary


def test_every_write_post_training_report_call_site_is_wired():
    """Issue #259 follow-up: the original PR wired 5 of the 7 real call sites
    and explicitly flagged the other two (``rolling_seed_application`` /
    ``submap_stitch_application``) as unwired -- so those two modes wrote a
    ``report.json`` with no record of the commit that produced it, which is
    exactly the auditing hole #259 exists to close.

    Source-text check (this repo's existing wiring-test convention, see
    ``tests/test_streaming_optimization_root_wiring.py``): importing
    ``train_streaming``/the mapping-integration modules needs a full
    CUDA/gsplat environment this suite does not assume. Every module that
    CALLS ``write_post_training_report(`` must also mention
    ``git_provenance``, either directly or via a shared kwargs helper.

    ``render_report.py`` is a deliberate, named exception: it re-renders an
    already-trained checkpoint offline, so it has no ``args.git_commit``
    (that is captured by ``train.py``'s ``__main__`` at training time) and
    the commit that re-rendered a PLY is not the commit that produced it.
    Named here rather than silently skipped, so adding a genuinely-unwired
    training call site still fails this test.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[1]
    callers = []
    for path in list(repo.glob("*.py")) + list((repo / "eligsir").rglob("*.py")):
        text = path.read_text()
        if "write_post_training_report(" not in text:
            continue
        if "def write_post_training_report(" in text:
            continue  # the definition itself
        callers.append((path.relative_to(repo), text))

    assert callers, "no call sites found -- the glob or the API name changed"
    exempt = {"render_report.py"}
    unwired = [
        str(p) for p, text in callers
        if "git_provenance" not in text and str(p) not in exempt
    ]
    assert not unwired, f"call sites missing git_provenance wiring: {unwired}"
