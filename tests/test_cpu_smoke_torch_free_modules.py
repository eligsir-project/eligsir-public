"""Issue #423 / #434: the modules the CPU smoke suite depends on must stay
importable and *functional* without torch.

`Dockerfile.ci-cpu` deliberately installs no torch ("Keep this image
intentionally light ... without pulling ROCm, Torch, ROS2, Open3D, or
viewers"). Two distinct failure modes have to be guarded, and only the first
is loud:

1. **Import-time**: a module-scope `import torch` makes pytest abort
   *collection* of the whole file. That is how `main` went red after #427 —
   and because collection was interrupted, the other 20 allowlisted files
   silently stopped running too.

2. **Silent degradation** — the dangerous one. Several call sites wrap a lazy
   import in a bare `except Exception` and return a "nothing to report"
   sentinel on failure. Under a torch-importing dependency, a missing torch
   turns into a *wrong answer* rather than an error:
   `compare_against_baseline` would report `comparable=False`, and
   `seed_fade_severity._final_report_path` would report every run as
   ungradeable. No traceback, no red test — just quietly empty results.

These tests pin the torch-free dependency closure so a future refactor cannot
reintroduce either failure mode without a red test.

**Known limitation — do not over-trust this guard.** It is a closed-world check:
it catches an import of a module on `TORCH_IMPORTING_MODULES` from a module on
`TORCH_FREE_MODULES`. A fourth site importing some *other* torch-carrying module
would not fire, and neither list is auto-derived. The third instance fixed under
#434 (`utils/mapping_phase_backfill.py`, carrying this bug on `main` since #359)
was in fact found by merge-testing #425, not by this guard — the guard was then
widened to cover it. Treat these lists as a ratchet over known carriers, not as
proof that the closure is torch-free.

The FOURTH instance (#449, ``utils/seed_fade_checker.py``) confirms the
limitation is real and not theoretical: it survived #435 untouched and was
found by a manual repo-wide AST sweep, not by this guard. Two consequences,
both now encoded below: a module and the carrier it must avoid have to be
added to these lists **together** (adding only the module makes the guard pass
vacuously), and ``PARTIAL_TORCH_FREE_MODULES`` exists because the defect also
occurs in modules that legitimately use torch elsewhere — pulling a torch-free
helper out of a torch carrier is the bug regardless.
"""
from __future__ import annotations

import ast
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modules that MUST NOT import torch, transitively or otherwise, because the
# CPU-smoke-allowlisted tests exercise real behavior through them.
TORCH_FREE_MODULES = [
    "utils/baseline_comparison.py",
    "utils/psnr_history_io.py",
    "utils/seed_fade_severity.py",
    # Reached lazily from seed_fade_severity.grade_seed_fade_for_run, again
    # inside a bare `except` — so a torch dependency here silently deletes the
    # report.json branch instead of failing (found while merge-testing #425).
    "utils/mapping_phase_backfill.py",
    # #449, the FOURTH instance of this class: check_run_dir's lazy
    # `_find_latest_report_json` import sat under `except Exception` and
    # reported "could not locate report.json: No module named 'torch'" as a
    # verdict. Four tests in tests/test_seed_fade_checker.py /
    # tests/test_seed_fade_resume.py were skipped in the CPU image because of
    # it (tests/conftest.py's _TORCH_VIA_COMPARISON_REPORT_REASON, now gone).
    "utils/seed_fade_checker.py",
    # Same helper, same defect, found by the #449 sweep: an offline tabulator
    # that only ever reads report.json off disk.
    "scripts/w3_13_tabulate.py",
    # #449: torch-free home for ScanNet's numpy-only pose reader, so
    # utils/trajectory_eval.py's GT auto-discovery no longer reaches
    # scene.readers.scannet under a bare `except`.
    "utils/scannet_pose_io.py",
    # Issue #397: trajectory-video resolution/pacing pure functions, split
    # out of utils/comparison_report.py for the same torch-free-testability
    # reason baseline_comparison.py was.
    "utils/trajectory_video.py",
]

# Modules that legitimately use torch themselves, but must not reach a
# torch-carrying module for a helper that is torch-free -- the same
# silent-degradation hazard, in a module that cannot be fully torch-free.
PARTIAL_TORCH_FREE_MODULES = [
    # utils/trajectory_eval.py imports torch only inside _cams_to_c2w; its
    # ScanNet GT auto-discovery path (_materialize_scannet_gt) is pure numpy
    # and runs under `except Exception: return None` (#449).
    "utils/trajectory_eval.py",
]

# Modules known to import torch at module scope. Nothing in TORCH_FREE_MODULES
# may reference these, at module scope OR in a lazy/function-local import.
TORCH_IMPORTING_MODULES = {
    "utils.comparison_report",
    "utils.psnr_history",
    "utils.image_utils",
    # Carries torch via scene.readers.common / utils.graphics_utils, and (as a
    # member of the `scene` package) via scene/__init__.py's model imports.
    # Added with utils/scannet_pose_io.py in #449 -- per this file's
    # closed-world caveat, a carrier and the module that must avoid it belong
    # in these lists TOGETHER or the guard silently passes.
    "scene.readers.scannet",
}


def _iter_imported_names(path: str):
    """Yield every module name imported by `path`, including function-local
    (lazy) imports -- which is the whole point: the silent-degradation bug
    lived in a lazy import inside a bare `except Exception`."""
    with open(os.path.join(_REPO_ROOT, path)) as fh:
        tree = ast.parse(fh.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                yield node.module


@pytest.mark.parametrize("module_path", TORCH_FREE_MODULES)
def test_module_does_not_import_torch(module_path):
    imported = set(_iter_imported_names(module_path))
    offenders = {n for n in imported if n == "torch" or n.startswith("torch.")}
    assert not offenders, (
        f"{module_path} imports {sorted(offenders)}. It must stay torch-free so "
        f"the CPU smoke image (Dockerfile.ci-cpu, no torch) can use it -- see #434."
    )


@pytest.mark.parametrize("module_path", TORCH_FREE_MODULES)
def test_module_does_not_reach_torch_via_a_known_torch_importer(module_path):
    """The subtle half. A lazy `from eligsir.utils.comparison_report import X` looks
    harmless and passes the test above, but drags torch in at call time -- and
    when it is wrapped in `except Exception`, the result is a silent wrong
    answer rather than a crash."""
    imported = set(_iter_imported_names(module_path))
    offenders = imported & TORCH_IMPORTING_MODULES
    assert not offenders, (
        f"{module_path} imports {sorted(offenders)}, which import torch at module "
        f"scope. Use the torch-free equivalent (utils.baseline_comparison / "
        f"utils.psnr_history_io) instead -- see #434."
    )


@pytest.mark.parametrize("module_path", PARTIAL_TORCH_FREE_MODULES)
def test_partially_torch_free_module_does_not_reach_a_known_torch_importer(module_path):
    """These modules may `import torch` for their own torch-shaped work, but
    must still not pull a TORCH-FREE helper out of a torch-carrying module --
    that is the silent-degradation half of #434/#449, independent of whether
    the importing module is itself fully torch-free."""
    imported = set(_iter_imported_names(module_path))
    offenders = imported & TORCH_IMPORTING_MODULES
    assert not offenders, (
        f"{module_path} imports {sorted(offenders)} for a helper that is "
        f"torch-free. Use the torch-free equivalent (utils.baseline_comparison "
        f"/ utils.psnr_history_io / utils.scannet_pose_io) -- see #449."
    )


def test_scannet_reader_still_reexports_the_moved_pose_reader():
    """`from scene.readers.scannet import _read_scannet_pose` is used by
    eligsir/io/sources/datasets/scannet.py and must keep working."""
    src = os.path.join(_REPO_ROOT, "scene", "readers", "scannet.py")
    with open(src) as fh:
        text = fh.read()
    assert "_read_scannet_pose" in text, (
        "scene/readers/scannet.py no longer exposes _read_scannet_pose"
    )


def test_scannet_pose_reader_works_without_torch(tmp_path):
    """The moved reader must actually read a pose, from the torch-free module."""
    from eligsir.utils.scannet_pose_io import _read_scannet_pose

    p = tmp_path / "0.txt"
    p.write_text("1 0 0 0.5\n0 1 0 0\n0 0 1 0\n0 0 0 1\n")
    pose = _read_scannet_pose(str(p))
    assert pose is not None and pose.shape == (4, 4)
    assert abs(float(pose[0, 3]) - 0.5) < 1e-6
    assert _read_scannet_pose(str(tmp_path / "missing.txt")) is None


def test_comparison_report_still_reexports_the_moved_names():
    """The split must stay source-compatible: production code and older tests
    import these from `utils.comparison_report`, and that must keep working."""
    src = os.path.join(_REPO_ROOT, "utils", "comparison_report.py")
    with open(src) as fh:
        text = fh.read()
    for name in (
        "compare_against_baseline",
        "_align_psnr_curves",
        "_classify_curve_shape",
        "_find_latest_report_json",
        "_camera_set_hash",
        "_realized_holdout_hash",
    ):
        assert name in text, f"utils/comparison_report.py no longer exposes {name}"


def test_psnr_history_still_reexports_the_moved_readers():
    src = os.path.join(_REPO_ROOT, "utils", "psnr_history.py")
    with open(src) as fh:
        text = fh.read()
    for name in ("read_psnr_history", "_read_psnr_history"):
        assert name in text, f"utils/psnr_history.py no longer exposes {name}"


def test_seed_fade_severity_grades_a_run_without_torch(tmp_path):
    """End-to-end on the silent-degradation path: `_final_report_path` must
    actually FIND the report, not swallow an ImportError into `None`."""
    from eligsir.utils.seed_fade_severity import _final_report_path

    run_dir = tmp_path / "run1"
    d = run_dir / "comparison" / "iter_7000"
    d.mkdir(parents=True)
    (d / "report.json").write_text("{}")

    found = _final_report_path(str(run_dir))
    assert found is not None, (
        "_final_report_path returned None for a run that plainly has a report -- "
        "its lazy import is reaching a torch-importing module and the bare "
        "`except Exception` is hiding the failure (#434)."
    )
    assert found.endswith(os.path.join("comparison", "iter_7000", "report.json"))
