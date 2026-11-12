"""Tests for the production-reachability contract gate (2026-08-23).

**Deliberately torch-free** (issue #536): the CPU CI image excludes any test file
that imports torch at module scope, so a test written that way would report green
without ever executing. Nothing here imports torch, directly or transitively --
the checker under test is pure-stdlib AST by design.

These tests are the gate's own teeth-check, kept permanently rather than run once
by hand. They exercise BOTH ratchet directions on synthetic trees:

* a new unwired primitive with no ledger entry must FAIL (the discovery half --
  without this, the gate is opt-in decoration);
* a ledgered entry that has BECOME reachable must FAIL with "delete this entry"
  (the half that decays silently otherwise, because nothing else ever prompts
  anyone to clean the ledger up).

plus the negative controls that stop it crying wolf: a production caller, a
production caller *inside a conditional* (the staged feature-flag case, which is
legitimately fine), and a script-only caller (which must NOT launder a primitive
into looking wired).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_CHECKER = (
    Path(__file__).resolve().parent.parent / "scripts" / "check_production_reachability.py"
)
_MODNAME = "check_production_reachability"
_REPO = Path(__file__).resolve().parent.parent


def _load():
    if _MODNAME in sys.modules:
        return sys.modules[_MODNAME]
    spec = importlib.util.spec_from_file_location(_MODNAME, _CHECKER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register BEFORE exec: @dataclass resolves annotations via
    # sys.modules[cls.__module__], which raises AttributeError otherwise.
    sys.modules[_MODNAME] = mod
    spec.loader.exec_module(mod)
    return mod


def _tree(root: Path, files: dict) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")


def _manifest(root: Path, *, ledger=None, grandfathered=None) -> dict:
    manifest = {
        "schema_version": 1,
        # synthetic trees are deliberately tiny -- the real tree-completeness
        # sentinel is exercised by its own test below, not by every fixture
        "config": {"exclude_dirs": [".git"], "tree_sentinels": []},
        "ledger": ledger or [],
        "grandfathered": grandfathered or [],
    }
    (root / "contract").mkdir(parents=True, exist_ok=True)
    (root / "contract" / "production_reachability.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _ledger_entry(key, kind, **kw):
    entry = {
        "id": "synthetic",
        "key": key,
        "kind": kind,
        "status": "ledgered",
        "tracking_issue": None,
        "reason": "synthetic test fixture",
    }
    entry.update(kw)
    return entry


# --------------------------------------------------------------------------
# design invariants
# --------------------------------------------------------------------------


def test_checker_imports_no_torch():
    """The whole point of the pure-AST design (issue #536)."""
    text = _CHECKER.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "from torch" not in text


def test_scripts_and_tests_are_not_production():
    mod = _load()
    assert mod.classify("scripts/promote_champion.py") == "script"
    assert mod.classify("tests/test_x.py") == "test"
    assert mod.classify("tests/conftest.py") == "test"
    assert mod.classify("utils/foo/test_helper.py") == "test"
    assert mod.classify("geometry/device_fusion.py") == "prod"
    assert mod.classify("train_streaming.py") == "prod"


# --------------------------------------------------------------------------
# direction 1: discovery -- a new unwired primitive must FAIL
# --------------------------------------------------------------------------


def test_unwired_module_with_no_ledger_entry_fails(tmp_path):
    """The device_fusion.py shape: test-imported module, zero production importers."""
    mod = _load()
    _tree(
        tmp_path,
        {
            "geometry/device_fusion.py": "def fuse_points_with_knn_torch(x):\n    return x\n",
            "train.py": "print('entrypoint')\n",
            "tests/test_fusion.py": (
                "from geometry.device_fusion import fuse_points_with_knn_torch\n"
                "def test_it():\n    assert fuse_points_with_knn_torch(1) == 1\n"
            ),
        },
    )
    manifest = _manifest(tmp_path)
    report, violations, _ = mod.check(tmp_path, manifest)
    assert "module:geometry/device_fusion.py" in {v.key for v in violations}
    assert not report.ok
    joined = "\n".join(report.failures)
    assert "UNWIRED PRIMITIVE" in joined
    assert "geometry/device_fusion.py" in joined
    # the failure must name both legitimate ways forward, not just "fix it"
    assert "WIRE IT" in joined and "LEDGER IT" in joined


def test_unwired_symbol_inside_a_live_module_fails(tmp_path):
    """The make_gsplat_camera_batch shape: new symbol hiding in a live module."""
    mod = _load()
    _tree(
        tmp_path,
        {
            "scene/cameras.py": (
                "def make_gsplat_camera_tensors(cam):\n    return cam\n\n"
                "def make_gsplat_camera_batch(cams):\n    return list(cams)\n"
            ),
            "train.py": (
                "from scene.cameras import make_gsplat_camera_tensors\n"
                "make_gsplat_camera_tensors(None)\n"
            ),
            "tests/test_cams.py": (
                "from scene.cameras import make_gsplat_camera_batch\n"
                "def test_it():\n    assert make_gsplat_camera_batch([]) == []\n"
            ),
        },
    )
    manifest = _manifest(tmp_path)
    report, violations, _ = mod.check(tmp_path, manifest)
    keys = {v.key for v in violations}
    assert "symbol:scene/cameras.py::make_gsplat_camera_batch" in keys
    # the sibling that production DOES call must not be flagged
    assert "symbol:scene/cameras.py::make_gsplat_camera_tensors" not in keys
    assert not report.ok


def test_ledgering_the_unwired_primitive_reaches_a_passing_state(tmp_path):
    mod = _load()
    _tree(
        tmp_path,
        {
            "geometry/device_fusion.py": "def fuse(x):\n    return x\n",
            "tests/test_fusion.py": "from geometry.device_fusion import fuse\n",
        },
    )
    manifest = _manifest(
        tmp_path,
        ledger=[_ledger_entry("module:geometry/device_fusion.py", "module")],
    )
    report, _, _ = mod.check(tmp_path, manifest)
    assert report.ok, report.failures
    assert any("LEDGERED" in p for p in report.pending)


def test_grandfathered_entry_is_accepted(tmp_path):
    mod = _load()
    _tree(
        tmp_path,
        {
            "utils/old_helper.py": "def helper(x):\n    return x\n",
            "tests/test_old.py": "from eligsir.utils.old_helper import helper\n",
        },
    )
    manifest = _manifest(tmp_path, grandfathered=["module:utils/old_helper.py"])
    report, _, _ = mod.check(tmp_path, manifest)
    assert report.ok, report.failures


# --------------------------------------------------------------------------
# direction 2: the ratchet -- a ledgered entry that got wired must FAIL
# --------------------------------------------------------------------------


def test_ledgered_entry_that_became_reachable_fails(tmp_path):
    """The direction that decays silently: nothing else prompts a ledger cleanup."""
    mod = _load()
    _tree(
        tmp_path,
        {
            "geometry/device_fusion.py": "def fuse(x):\n    return x\n",
            # production now calls it -- the gap is closed
            "train_streaming.py": "from geometry.device_fusion import fuse\nfuse(1)\n",
            "tests/test_fusion.py": "from geometry.device_fusion import fuse\n",
        },
    )
    manifest = _manifest(
        tmp_path,
        ledger=[_ledger_entry("module:geometry/device_fusion.py", "module")],
    )
    report, violations, _ = mod.check(tmp_path, manifest)
    assert "module:geometry/device_fusion.py" not in {v.key for v in violations}
    assert not report.ok
    joined = "\n".join(report.failures)
    assert "LEDGER ENTRY RESOLVED" in joined
    assert "Delete its entry" in joined


def test_ledgered_symbol_that_became_reachable_fails(tmp_path):
    mod = _load()
    _tree(
        tmp_path,
        {
            "scene/cameras.py": "def make_batch(c):\n    return c\n",
            "train.py": "from scene.cameras import make_batch\nmake_batch(1)\n",
            "tests/test_cams.py": "from scene.cameras import make_batch\n",
        },
    )
    manifest = _manifest(
        tmp_path, ledger=[_ledger_entry("symbol:scene/cameras.py::make_batch", "symbol")]
    )
    report, _, _ = mod.check(tmp_path, manifest)
    assert not report.ok
    assert "LEDGER ENTRY RESOLVED" in "\n".join(report.failures)


def test_ledgered_entry_not_yet_landed_is_pending_not_a_failure(tmp_path):
    """A gap may be ledgered from the branch that will introduce it."""
    mod = _load()
    _tree(tmp_path, {"train.py": "pass\n"})
    manifest = _manifest(
        tmp_path, ledger=[_ledger_entry("module:geometry/not_yet.py", "module")]
    )
    report, _, _ = mod.check(tmp_path, manifest)
    assert report.ok, report.failures
    assert any("NOT LANDED" in p for p in report.pending)


def test_grandfathered_entry_that_got_wired_is_resolved_not_a_failure(tmp_path):
    """Deleting or wiring old code must never fail a PR -- only nag."""
    mod = _load()
    _tree(
        tmp_path,
        {
            "utils/old_helper.py": "def helper(x):\n    return x\n",
            "train.py": "from eligsir.utils.old_helper import helper\nhelper(1)\n",
            "tests/test_old.py": "from eligsir.utils.old_helper import helper\n",
        },
    )
    manifest = _manifest(tmp_path, grandfathered=["module:utils/old_helper.py"])
    report, _, _ = mod.check(tmp_path, manifest)
    assert report.ok, report.failures
    assert "module:utils/old_helper.py" in report.resolved


# --------------------------------------------------------------------------
# negative controls -- the gate must not cry wolf
# --------------------------------------------------------------------------


def test_production_caller_behind_a_feature_flag_is_not_a_violation(tmp_path):
    """A staged, default-off primitive has a real call site -- just inside an `if`.

    Zero *unconditional* callers is fine (see #8's ``streaming_insert_surface_fusion``
    -gated path); zero callers is not. This is the case a transitive-reachability
    walk would get wrong and a reference scan gets right.
    """
    mod = _load()
    _tree(
        tmp_path,
        {
            "geometry/staged.py": "def staged_fuse(x):\n    return x\n",
            "train_streaming.py": (
                "from geometry.staged import staged_fuse\n"
                "def step(args, pts):\n"
                "    if getattr(args, 'streaming_insert_surface_fusion', False):\n"
                "        return staged_fuse(pts)\n"
                "    return pts\n"
            ),
            "tests/test_staged.py": "from geometry.staged import staged_fuse\n",
        },
    )
    manifest = _manifest(tmp_path)
    report, violations, _ = mod.check(tmp_path, manifest)
    assert violations == []
    assert report.ok, report.failures


def test_script_only_caller_does_not_count_as_production(tmp_path):
    """A one-off probe script must not launder an unwired primitive."""
    mod = _load()
    _tree(
        tmp_path,
        {
            "scene/cameras.py": (
                "def make_single(c):\n    return c\n\n"
                "def make_batch(c):\n    return c\n"
            ),
            # the module itself is live -- only the batched symbol is not
            "train.py": "from scene.cameras import make_single\nmake_single(1)\n",
            "scripts/issue106_camera_batch_probe.py": (
                "from scene.cameras import make_batch\nmake_batch(1)\n"
            ),
            "tests/test_cams.py": "from scene.cameras import make_batch\n",
        },
    )
    manifest = _manifest(tmp_path)
    _report, violations, _ = mod.check(tmp_path, manifest)
    assert "symbol:scene/cameras.py::make_batch" in {v.key for v in violations}


def test_symbol_with_no_test_caller_at_all_is_not_a_violation(tmp_path):
    """Ordinary unused public API is none of this gate's business."""
    mod = _load()
    _tree(
        tmp_path,
        {
            "utils/misc.py": "def never_used(x):\n    return x\n",
            "train.py": "pass\n",
        },
    )
    manifest = _manifest(tmp_path)
    _report, violations, _ = mod.check(tmp_path, manifest)
    assert violations == []


def test_attribute_style_production_use_counts(tmp_path):
    """``import m`` + ``m.fn()`` is a real production caller."""
    mod = _load()
    _tree(
        tmp_path,
        {
            "geometry/fusion.py": "def fuse(x):\n    return x\n",
            "train.py": "import geometry.fusion\ngeometry.fusion.fuse(1)\n",
            "tests/test_f.py": "from geometry.fusion import fuse\n",
        },
    )
    manifest = _manifest(tmp_path)
    _report, violations, _ = mod.check(tmp_path, manifest)
    assert violations == []


# --------------------------------------------------------------------------
# manifest hygiene
# --------------------------------------------------------------------------


def test_ledger_entry_without_a_reason_is_rejected(tmp_path):
    mod = _load()
    _tree(tmp_path, {"train.py": "pass\n"})
    entry = _ledger_entry("module:geometry/x.py", "module")
    entry["reason"] = "   "
    manifest = _manifest(tmp_path, ledger=[entry])
    report, _, _ = mod.check(tmp_path, manifest)
    assert not report.ok
    assert "must carry a non-empty 'reason'" in "\n".join(report.failures).lower().replace(
        "MUST", "must"
    ) or "reason" in "\n".join(report.failures)


def test_null_tracking_issue_is_allowed_but_absent_field_is_not(tmp_path):
    """The gate must reach a passing state without anyone filing an issue first."""
    mod = _load()
    _tree(tmp_path, {"train.py": "pass\n"})
    ok_entry = _ledger_entry("module:geometry/x.py", "module")
    assert ok_entry["tracking_issue"] is None
    report, _, _ = mod.check(tmp_path, _manifest(tmp_path, ledger=[ok_entry]))
    assert report.ok, report.failures

    bad = _ledger_entry("module:geometry/x.py", "module")
    del bad["tracking_issue"]
    report, _, _ = mod.check(tmp_path, _manifest(tmp_path, ledger=[bad]))
    assert not report.ok
    assert "tracking_issue" in "\n".join(report.failures)


def test_unenforceable_entry_must_have_null_key(tmp_path):
    mod = _load()
    _tree(tmp_path, {"train.py": "pass\n"})
    good = _ledger_entry(None, "unenforceable")
    report, _, _ = mod.check(tmp_path, _manifest(tmp_path, ledger=[good]))
    assert report.ok, report.failures

    bad = _ledger_entry("module:utils/general_utils.py", "unenforceable")
    report, _, _ = mod.check(tmp_path, _manifest(tmp_path, ledger=[bad]))
    assert not report.ok


def test_key_in_both_ledger_and_baseline_is_rejected(tmp_path):
    mod = _load()
    _tree(tmp_path, {"train.py": "pass\n"})
    manifest = _manifest(
        tmp_path,
        ledger=[_ledger_entry("module:geometry/x.py", "module")],
        grandfathered=["module:geometry/x.py"],
    )
    report, _, _ = mod.check(tmp_path, manifest)
    assert not report.ok
    assert "BOTH" in "\n".join(report.failures)


# --------------------------------------------------------------------------
# the real repository
# --------------------------------------------------------------------------


def test_incomplete_tree_fails_loudly_instead_of_inventing_violations(tmp_path):
    """A partial container COPY must not read as "this primitive is unwired".

    Measured 2026-08-23: the ci-cpu image shipped train.py/train_streaming.py but
    not render.py/render_report.py/metrics.py, and the gate reported 5 invented
    failures for genuinely-wired code (e.g. holdout_gt_exact.py::load_holdout_depth,
    whose only production caller is render_report.py).
    """
    mod = _load()
    _tree(
        tmp_path,
        {
            "train.py": "pass\n",
            "geometry/thing.py": "def fn(x):\n    return x\n",
            "tests/test_thing.py": "from geometry.thing import fn\n",
        },
    )
    manifest = _manifest(tmp_path)
    manifest["config"]["tree_sentinels"] = ["train.py", "render_report.py"]
    report, violations, _ = mod.check(tmp_path, manifest)
    assert not report.ok
    joined = "\n".join(report.failures)
    assert "INCOMPLETE TREE" in joined
    assert "render_report.py" in joined
    # and it must NOT also emit phantom per-primitive verdicts from a tree it
    # has just declared untrustworthy
    assert violations == []
    assert "UNWIRED PRIMITIVE" not in joined


def test_ci_cpu_image_ships_every_tree_sentinel():
    """The gate is only load-bearing if the image CI runs it in has the full tree."""
    mod = _load()
    dockerfile = (_REPO / "Dockerfile.ci-cpu").read_text(encoding="utf-8")
    copied = " ".join(
        line for line in dockerfile.splitlines() if line.startswith("COPY ")
    )
    manifest = mod.load_manifest(_REPO / mod.MANIFEST_REL)
    sentinels = manifest.get("config", {}).get(
        "tree_sentinels", list(mod.DEFAULT_TREE_SENTINELS)
    )
    for sentinel in sentinels:
        assert sentinel in copied, (
            f"Dockerfile.ci-cpu does not COPY {sentinel}, which the "
            "production-reachability gate needs to avoid inventing violations"
        )
    assert "COPY contract " in dockerfile, "Dockerfile.ci-cpu must COPY contract/"


def test_real_repo_manifest_is_valid_and_the_gate_passes():
    mod = _load()
    manifest_path = _REPO / mod.MANIFEST_REL
    assert manifest_path.exists(), f"missing {mod.MANIFEST_REL}"
    manifest = mod.load_manifest(manifest_path)
    assert mod.validate_manifest(manifest) == []
    report, violations, _ = mod.check(_REPO, manifest)
    assert violations, "the scanner found nothing at all -- it is probably misconfigured"
    assert report.ok, "production-reachability gate FAILED:\n" + "\n".join(report.failures)


def test_real_repo_ledger_carries_the_2026_08_23_findings_that_are_still_open():
    """Regression guard on the motivating incident's own record.

    ``device-knn-surface-fusion`` was one of the three original findings and is
    deliberately NOT required here any more: PR #1142 wired
    ``fuse_points_with_knn_torch`` into the production device-native candidate
    producer, so the checker itself reports that entry as ``LEDGER ENTRY
    RESOLVED`` and instructs that it be deleted. Requiring it to persist would
    put this test in direct conflict with the gate it guards -- a ledger entry
    resolving is the success case, and the ledger is not an append-only log.

    The batched camera helper has likewise been removed from the ledger because
    this PR wires it into the production render path. What replaces the broad
    device-fusion entry is the narrower, still-genuinely-unwired sibling
    ``knn-isotropic-log-scale``, asserted below so the shrink is a real
    reclassification rather than a silent loss of coverage.
    """
    mod = _load()
    manifest = mod.load_manifest(_REPO / mod.MANIFEST_REL)
    ids = {e["id"] for e in manifest["ledger"]}
    assert "strict-deterministic-mode" in ids
    assert "batched-gsplat-camera-tensors" not in ids, (
        "the batched camera helper is production-wired now; its ledger entry "
        "must be deleted, not kept"
    )
    assert "device-knn-surface-fusion" not in ids, (
        "device_fusion is wired on the production path now; its ledger entry "
        "must be deleted, not kept (the checker reports it RESOLVED)."
    )
    assert "knn-isotropic-log-scale" in ids, (
        "the still-unwired sibling symbol must remain accounted for"
    )
    for entry in manifest["ledger"]:
        assert entry["reason"].strip(), entry["id"]
        assert "tracking_issue" in entry, entry["id"]


def test_baseline_contains_no_entry_that_is_also_ledgered():
    mod = _load()
    manifest = mod.load_manifest(_REPO / mod.MANIFEST_REL)
    ledger_keys = {e["key"] for e in manifest["ledger"] if e.get("key")}
    assert not (ledger_keys & set(manifest["grandfathered"]))
