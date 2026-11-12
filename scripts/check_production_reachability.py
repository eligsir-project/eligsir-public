#!/usr/bin/env python3
"""Production-reachability contract check (the "test-only primitive" gate).

Motivating incident (2026-08-23)
--------------------------------
A repo audit found the same structural failure three times in one seven-minute
window of commits: a correct, real, unit-tested primitive was written to replace
an existing production path -- and production kept calling the old path. The new
primitive had ONLY test callers, silently, forever. Tests passed. Coverage looked
fine. The training loop never benefited.

* ``geometry/device_fusion.py`` (``0f25f584``) -- an all-on-device batched KNN
  surface fusion whose own docstring says it "replaces the old GPU KNN -> CPU
  indices -> Python/NumPy PCA shape". The old NumPy path
  (``geometry/insertion.py::fuse_points_with_knn``) is what
  ``eligsir/mapping/integration/integration_planner.py`` actually calls, CPU
  round-trip and all. The new module's only references in the entire repository
  are three lines of ``tests/test_owner_transfer_sequence.py``.
* ``scene/cameras.py::make_gsplat_camera_batch`` (+ ``GsplatCameraBatch``,
  ``7eae5170``) -- a batched sibling of the already-shipping single-camera
  ``make_gsplat_camera_tensors``. Only caller: the same one test file.
* ``utils/general_utils.py::safe_state(..., strict_deterministic=)``
  (``6d165182``) -- a real determinism mode that no CLI flag, config key, or call
  site can ever set to ``True``.

None of the three was old, rotting code: all three landed the same day. This is
not a "dead code" problem, it is a **wiring** problem, and the window in which it
is cheap to catch is the PR that introduces the primitive.

What this checker enforces
--------------------------
Two purely static, AST-derived violation kinds, over the repository as a whole:

``module:<path>``
    A production-classified module that at least one file under ``tests/``
    imports, and that **no other production file imports at all**. This is
    exactly the ``device_fusion.py`` shape.

``symbol:<path>::<Name>``
    A public symbol imported **by name** from a production module in at least one
    test file, whose name is referenced by **no production file other than its own
    defining module**. This is exactly the ``make_gsplat_camera_batch`` shape --
    a new unwired symbol hiding inside a module that is otherwise live.

Both rules are deliberately keyed on "has a test caller". A public symbol with no
callers anywhere is ordinary unused API surface and is none of this gate's
business; a symbol with *tests* and no production caller is precisely the
dead-weight-that-looks-alive pattern above, and is also the shape that a coverage
report cannot distinguish from healthy code.

Why "is it referenced at all", not a call-graph walk
----------------------------------------------------
A transitive reachability walk from ``train.py`` / ``train_streaming.py`` /
``render.py`` / ``metrics.py`` was considered and rejected. This codebase reaches
live code through property-backed aliases (``scene/streaming_scene.py``),
string-keyed strategy dispatch (``--densification_strategy``), ``getattr``
fallbacks (see ``scripts/audit_getattr_fallbacks.py``, which exists because there
are many), ``nn.ParameterDict`` indirection and lazy in-function imports. A walk
would report false "unreachable" verdicts on genuinely-live code, and a gate that
cries wolf gets silenced. Reference-existence is conservative in the safe
direction: it can miss a gap, it will not invent one.

This also gives the **feature-flag case** the right answer for free. A staged
primitive gated behind a default-off flag (e.g. ``--streaming_insert_surface_fusion``)
*does* have a production call site, just one inside an ``if``. An AST reference
scan sees it and passes. Zero *unconditional* callers is fine; zero callers is not.

Non-production for this gate's purposes: ``tests/`` and ``scripts/``. A one-off
probe script (``scripts/issue106_camera_batch_probe.py``) must not be able to
launder an unwired primitive into looking wired.

Ledger and baseline (the ratchet)
---------------------------------
``contract/production_reachability.json`` carries two lists, and the asymmetry
between them is the whole mechanism:

``ledger``
    Hand-written, reviewed entries: a primitive that is knowingly not wired yet,
    with a ``reason`` and a ``tracking_issue`` (``null`` is allowed -- an entry
    must never be blocked on filing an issue, or the gate cannot reach a passing
    state). A ledgered entry that is still unreachable is PENDING and
    non-blocking. A ledgered entry that has **become reachable FAILS**, telling
    you to delete it. That is the direction that decays silently otherwise: the
    ledger would otherwise fill up with stale entries describing gaps that were
    closed months ago, and stop meaning anything.

    A ledger entry whose module/symbol does not exist in this checkout at all is
    also PENDING -- so a gap can be ledgered from the branch that will introduce
    it, and the entry starts enforcing the moment the code lands.

``grandfathered``
    A machine-generated snapshot (``--write-baseline``) of the violations that
    already existed when this gate was introduced. These are accepted without a
    reason: hand-auditing them was measured at 60 module-level + 449 symbol-level
    entries on ``origin/main`` @ ``8a140c72``, which is not a thing anyone would
    honestly do in one PR, and a gate nobody can reach a passing state on is a
    gate that gets deleted. The baseline may only shrink: an entry that stops
    being a violation is reported as RESOLVED (non-blocking -- deleting code must
    not fail a PR) with an instruction to regenerate.

Anything that is a violation and appears in **neither** list is a FAIL. That is
the discovery half, and it is what makes this gate load-bearing rather than
decorative: the next ``device_fusion.py`` fails CI in the PR that introduces it,
with a message naming the two legitimate ways forward (wire it, or ledger it).

Known limitation, stated plainly
--------------------------------
The third motivating instance -- ``strict_deterministic`` -- is a *parameter
value*, not a symbol. Its enclosing function ``safe_state`` is called from
production; nobody ever passes the keyword. No symbol-or-module reachability rule
can see that, and this checker does not claim to. It is carried in the ledger with
``kind: "unenforceable"`` as documentation only, never resolution-checked. Widening
the gate to "public keyword-only parameters never passed by any production caller"
is plausible future work and is not attempted here.

Stdlib-only, pure ``ast``, no torch -- so it runs in the lean CPU CI image, in the
same idiom as ``scripts/check_import_fences.py``, ``scripts/audit_dataflow_sync_sites.py``
and ``scripts/gaussian_writer_registry.py``.

Usage::

    python3 scripts/check_production_reachability.py            # check (exit 1 on FAIL)
    python3 scripts/check_production_reachability.py --json
    python3 scripts/check_production_reachability.py --write-baseline
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_REL = "contract/production_reachability.json"

#: Directories never walked. ``ablator``/``submodules`` are vendored third-party
#: trees with their own lifecycles; ``output``/``experiments`` are run artifacts;
#: ``ros2_ws`` is a separately-built colcon workspace.
DEFAULT_EXCLUDE_DIRS = (
    ".git",
    ".claude",
    ".venv",
    "ablator",
    "build",
    "docs",
    "experiments",
    "node_modules",
    "output",
    "ros2_ws",
    "submodules",
)

VALID_STATUS = ("ledgered",)
VALID_KIND = ("module", "symbol", "unenforceable")


# --------------------------------------------------------------------------
# file classification
# --------------------------------------------------------------------------

TEST = "test"
SCRIPT = "script"
PROD = "prod"


def classify(rel_path: str) -> str:
    """Classify a repo-relative ``.py`` path as test / script / production.

    ``scripts/`` is NOT production on purpose: some scripts there are real
    tooling and some are one-off probes, and letting a probe count as a
    production caller is exactly how an unwired primitive would slip the gate.
    """
    parts = rel_path.split("/")
    base = parts[-1]
    if parts[0] == "tests" or base.startswith("test_") or base == "conftest.py":
        return TEST
    if parts[0] == "scripts":
        return SCRIPT
    return PROD


def module_name(rel_path: str) -> str:
    mod = rel_path[:-3].replace("/", ".")
    if mod.endswith(".__init__"):
        mod = mod[: -len(".__init__")]
    return mod


def iter_python_files(root: Path, exclude_dirs) -> list[str]:
    excluded = set(exclude_dirs)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in excluded and not d.startswith(".")
        ]
        for name in filenames:
            if name.endswith(".py"):
                out.append(os.path.relpath(os.path.join(dirpath, name), root))
    return sorted(out)


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------


@dataclass
class Index:
    """Static reference index over the whole tree."""

    files: list[str] = field(default_factory=list)
    modules: dict[str, str] = field(default_factory=dict)  # module name -> rel path
    #: module name -> {class -> set(importing rel paths)}
    module_imports: dict = field(default_factory=lambda: collections.defaultdict(
        lambda: collections.defaultdict(set)))
    #: (module name, symbol) -> {class -> set(rel paths)} from ``from M import S``
    symbol_imports: dict = field(default_factory=lambda: collections.defaultdict(
        lambda: collections.defaultdict(set)))
    #: bare name -> {class -> set(rel paths)} from any Name/Attribute node
    name_refs: dict = field(default_factory=lambda: collections.defaultdict(
        lambda: collections.defaultdict(set)))
    #: rel path -> set of public top-level def/class names
    public_defs: dict = field(default_factory=dict)
    parse_errors: list = field(default_factory=list)


def build_index(root: Path, exclude_dirs) -> Index:
    idx = Index()
    idx.files = iter_python_files(root, exclude_dirs)
    for rel in idx.files:
        idx.modules[module_name(rel)] = rel

    for rel in idx.files:
        try:
            source = (root / rel).read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            idx.parse_errors.append((rel, str(exc)))
            continue

        cls = classify(rel)
        publics = set()
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and not node.name.startswith("_"):
                publics.add(node.name)
            # module-level ``NAME = ...`` public constants count too: a
            # test-only constant is the same wiring smell as a test-only helper.
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                        publics.add(tgt.id)
        idx.public_defs[rel] = publics

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    idx.module_imports[alias.name][cls].add(rel)
                    # ``import a.b.c`` also makes a.b reachable
                    parts = alias.name.split(".")
                    for i in range(1, len(parts)):
                        idx.module_imports[".".join(parts[:i])][cls].add(rel)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # relative import -- resolve against the importing package
                    pkg = module_name(rel).split(".")
                    # a package __init__ is its own package; a module is in its parent
                    if not rel.endswith("__init__.py"):
                        pkg = pkg[:-1]
                    base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                    target = ".".join([*base, node.module] if node.module else base)
                else:
                    target = node.module or ""
                if target:
                    idx.module_imports[target][cls].add(rel)
                    for alias in node.names:
                        idx.symbol_imports[(target, alias.name)][cls].add(rel)
                        # ``from pkg import mod`` is a module import of pkg.mod
                        idx.module_imports[f"{target}.{alias.name}"][cls].add(rel)
            elif isinstance(node, ast.Name):
                idx.name_refs[node.id][cls].add(rel)
            elif isinstance(node, ast.Attribute):
                idx.name_refs[node.attr][cls].add(rel)
    return idx


# --------------------------------------------------------------------------
# violation detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    key: str
    kind: str  # "module" | "symbol"
    path: str
    symbol: str | None
    test_refs: tuple  # sorted rel paths


def _module_importers(idx: Index, mod: str, own_path: str, cls: str) -> set:
    hits = {f for f in idx.module_imports.get(mod, {}).get(cls, ()) if f != own_path}
    return hits


def _is_referenced_by_other_prod_file(idx: Index, mod: str, sym: str, rel: str) -> bool:
    """Is ``sym`` (defined in ``rel``, module name ``mod``) referenced -- by bare
    name or by ``from mod import sym`` -- from any OTHER production file?"""
    if any(f != rel for f in idx.name_refs.get(sym, {}).get(PROD, ())):
        return True
    if any(f != rel for f in idx.symbol_imports.get((mod, sym), {}).get(PROD, ())):
        return True
    return False


#: (root, rel path) -> {top-level def/class name -> names it references that are
#: ALSO top-level defs/classes in the same file}. Lazily built and cached per
#: file: only the handful of files with an otherwise-unresolved symbol
#: violation ever need it. Keyed on ``root`` too, not just ``rel`` -- a single
#: process (e.g. the pytest suite exercising this module against many
#: synthetic trees) can call this with the SAME relative path (a synthetic
#: fixture reusing a real-looking name like ``scene/cameras.py``) resolved
#: against DIFFERENT roots; a rel-only key would return a wrong, stale graph
#: from an unrelated tree instead of parsing this one.
_LOCAL_CALL_GRAPH_CACHE: dict[tuple, dict] = {}


def _local_call_graph(root: Path, rel: str) -> dict:
    """Same-module call graph among top-level defs/classes in ``rel``.

    Exists to answer one question precisely: a helper with no reference outside
    its own file can still be genuinely production-reachable if some OTHER
    top-level symbol in that same file -- one a different production module
    actually imports and calls -- calls it internally. ``config_from_args``
    calling ``PresentationSSAAConfig`` from inside ``utils/presentation_ssaa.py``
    is exactly this shape: the file-level bare-name/import scan in
    ``find_violations`` deliberately excludes same-file references (to avoid a
    module marking its own internals as "externally used"), so without this
    step a symbol that is reached ONLY via an intra-module call from a real
    entry point reads as unwired even though production genuinely exercises it.
    """
    cache_key = (str(root), rel)
    if cache_key in _LOCAL_CALL_GRAPH_CACHE:
        return _LOCAL_CALL_GRAPH_CACHE[cache_key]
    graph: dict = {}
    try:
        source = (root / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        _LOCAL_CALL_GRAPH_CACHE[cache_key] = graph
        return graph
    top_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    top_names = {node.name for node in top_defs}
    for node in top_defs:
        refs: set = set()
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, ast.Name):
                refs.add(child.id)
            elif isinstance(child, ast.Attribute):
                refs.add(child.attr)
        graph[node.name] = refs & top_names
    _LOCAL_CALL_GRAPH_CACHE[cache_key] = graph
    return graph


def _is_reachable_via_local_entry_point(
    idx: Index, root: Path, mod: str, sym: str, rel: str
) -> bool:
    """Is ``sym`` reachable, within its own file's call graph, from some other
    top-level def/class in that file that IS referenced by another production
    file (i.e. a real entry point)? See ``_local_call_graph`` for why this is
    needed on top of the direct external-reference check."""
    graph = _local_call_graph(root, rel)
    if sym not in graph:
        return False
    entry_points = {
        name for name in graph if _is_referenced_by_other_prod_file(idx, mod, name, rel)
    }
    if not entry_points:
        return False
    seen: set = set()
    frontier = list(entry_points)
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if cur == sym:
            return True
        frontier.extend(n for n in graph.get(cur, ()) if n not in seen)
    return False


def find_violations(idx: Index, root: Path | None = None) -> list[Violation]:
    out: list[Violation] = []
    module_violation_paths: set[str] = set()

    # --- module-level ---------------------------------------------------
    for mod, rel in sorted(idx.modules.items()):
        if classify(rel) != PROD:
            continue
        prod_hits = _module_importers(idx, mod, rel, PROD)
        if prod_hits:
            continue
        test_hits = _module_importers(idx, mod, rel, TEST)
        if not test_hits:
            continue
        out.append(
            Violation(
                key=f"module:{rel}",
                kind="module",
                path=rel,
                symbol=None,
                test_refs=tuple(sorted(test_hits)),
            )
        )
        module_violation_paths.add(rel)

    # --- symbol-level ---------------------------------------------------
    for (mod, sym), by_cls in sorted(idx.symbol_imports.items()):
        rel = idx.modules.get(mod)
        if rel is None or classify(rel) != PROD:
            continue
        if rel in module_violation_paths:
            continue  # already reported at module granularity
        if sym.startswith("_") or sym == "*":
            continue
        if sym not in idx.public_defs.get(rel, ()):
            continue  # re-export / submodule, not a symbol defined here
        test_hits = {f for f in by_cls.get(TEST, ()) if f != rel}
        if not test_hits:
            continue
        # any production reference to the bare name outside its own module?
        if _is_referenced_by_other_prod_file(idx, mod, sym, rel):
            continue
        # reachable purely intra-module, but from a real cross-module entry point?
        if root is not None and _is_reachable_via_local_entry_point(idx, root, mod, sym, rel):
            continue
        out.append(
            Violation(
                key=f"symbol:{rel}::{sym}",
                kind="symbol",
                path=rel,
                symbol=sym,
                test_refs=tuple(sorted(test_hits)),
            )
        )
    return out


def key_exists(idx: Index, key: str) -> bool:
    """Does the module/symbol a ledger key names exist in this checkout?"""
    if key.startswith("module:"):
        return key[len("module:"):] in idx.public_defs
    if key.startswith("symbol:"):
        rest = key[len("symbol:"):]
        rel, _, sym = rest.partition("::")
        return sym in idx.public_defs.get(rel, ())
    return False


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_manifest(manifest: dict) -> list[str]:
    problems: list[str] = []
    seen: set[str] = set()
    for i, entry in enumerate(manifest.get("ledger", [])):
        where = entry.get("id") or f"ledger[{i}]"
        if "reason" not in entry or not str(entry.get("reason", "")).strip():
            problems.append(f"{where}: a ledger entry MUST carry a non-empty 'reason'")
        if entry.get("status") not in VALID_STATUS:
            problems.append(
                f"{where}: status={entry.get('status')!r} is not one of {VALID_STATUS}"
            )
        kind = entry.get("kind")
        if kind not in VALID_KIND:
            problems.append(f"{where}: kind={kind!r} is not one of {VALID_KIND}")
        if "tracking_issue" not in entry:
            problems.append(
                f"{where}: must declare 'tracking_issue' (null is allowed and means "
                "'pending triage' -- the field being absent is not)"
            )
        key = entry.get("key")
        if kind == "unenforceable":
            if key is not None:
                problems.append(
                    f"{where}: kind='unenforceable' entries are documentation only "
                    "and must set 'key': null"
                )
            continue
        if not key or not key.startswith(("module:", "symbol:")):
            problems.append(
                f"{where}: key={key!r} must be 'module:<path>' or 'symbol:<path>::<Name>'"
            )
            continue
        if key in seen:
            problems.append(f"{where}: duplicate key {key!r}")
        seen.add(key)
    for key in manifest.get("grandfathered", []):
        if key in seen:
            problems.append(
                f"{key!r} appears in BOTH 'ledger' and 'grandfathered'; a curated "
                "ledger entry supersedes the baseline -- remove it from 'grandfathered'"
            )
    return problems


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


@dataclass
class Report:
    failures: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    resolved: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


#: Files whose absence would make the scan lie rather than merely be incomplete.
#: Each is a real production caller of something, so dropping it turns a wired
#: primitive into a phantom violation. Measured 2026-08-23: running the gate
#: against the ci-cpu image tree (which shipped train.py/train_streaming.py but
#: not render.py/render_report.py/metrics.py) produced 5 invented failures, e.g.
#: eligsir/io/holdout_gt_exact.py::load_holdout_depth, whose only production
#: caller is render_report.py. Fail loudly on an incomplete tree instead: a
#: sparse checkout or a partial container COPY is an operator error, and one
#: actionable message beats a storm of false positives that trains people to
#: distrust the gate.
DEFAULT_TREE_SENTINELS = (
    "train.py",
    "train_streaming.py",
    "render.py",
    "render_report.py",
    "metrics.py",
)


def check(root: Path, manifest: dict) -> tuple[Report, list[Violation], Index]:
    report = Report()
    sentinels = manifest.get("config", {}).get(
        "tree_sentinels", list(DEFAULT_TREE_SENTINELS)
    )
    missing = [s for s in sentinels if not (root / s).exists()]
    if missing:
        report.failures.append(
            f"INCOMPLETE TREE: {missing} missing under {root}.\n"
            "    This gate walks the whole repository to decide what production\n"
            "    references a primitive. With a real production caller absent it does\n"
            "    not merely under-report -- it INVENTS violations for wired code.\n"
            "    Scan a full checkout, or add the missing paths to the container COPY\n"
            "    list (see Dockerfile.ci-cpu)."
        )
        return report, [], Index()
    exclude = manifest.get("config", {}).get("exclude_dirs", list(DEFAULT_EXCLUDE_DIRS))
    idx = build_index(root, exclude)
    for rel, err in idx.parse_errors:
        report.notes.append(f"unparseable (skipped): {rel}: {err}")

    for problem in validate_manifest(manifest):
        report.failures.append(f"manifest invalid: {problem}")

    ledger = {
        e["key"]: e
        for e in manifest.get("ledger", [])
        if e.get("kind") != "unenforceable" and e.get("key")
    }
    grandfathered = set(manifest.get("grandfathered", []))

    violations = find_violations(idx, root)
    live = {v.key: v for v in violations}

    for v in violations:
        if v.key in ledger:
            entry = ledger[v.key]
            issue = entry.get("tracking_issue")
            report.pending.append(
                f"LEDGERED  {v.key}  (issue={issue if issue is not None else 'PENDING TRIAGE'}) "
                f"-- {entry.get('reason','')}"
            )
        elif v.key in grandfathered:
            report.pending.append(f"BASELINE  {v.key}")
        else:
            tests = ", ".join(v.test_refs[:3]) + (
                " ..." if len(v.test_refs) > 3 else ""
            )
            what = (
                f"module {v.path} is imported by tests but by no other production module"
                if v.kind == "module"
                else f"{v.path}::{v.symbol} is imported by tests but referenced by no production module"
            )
            report.failures.append(
                f"UNWIRED PRIMITIVE  {v.key}\n"
                f"    {what}\n"
                f"    test callers: {tests}\n"
                f"    Two ways forward, pick one:\n"
                f"      (1) WIRE IT: call it from the production path it was written to\n"
                f"          replace, and delete the path it replaces. This is the fix.\n"
                f"      (2) LEDGER IT: add an entry to {MANIFEST_REL} with a real\n"
                f"          'reason' and a 'tracking_issue' (null = pending triage) if it\n"
                f"          is deliberately staged. It will start failing again the moment\n"
                f"          it becomes reachable, so it cannot be forgotten.\n"
                f"    Do NOT add it to 'grandfathered' -- that list is a frozen snapshot\n"
                f"    of what predates this gate and may only shrink."
            )

    # ratchet: a ledgered entry that got wired must be removed
    for key, entry in sorted(ledger.items()):
        if key in live:
            continue
        if not key_exists(idx, key):
            report.pending.append(
                f"NOT LANDED  {key} -- ledgered but absent from this checkout "
                f"(entry starts enforcing when the code lands)"
            )
            continue
        report.failures.append(
            f"LEDGER ENTRY RESOLVED  {key}\n"
            f"    This primitive now HAS a production caller -- the gap it was ledgered\n"
            f"    for is closed. Delete its entry from {MANIFEST_REL}.\n"
            f"    (reason was: {entry.get('reason','')})"
        )

    for key in sorted(grandfathered):
        if key not in live:
            report.resolved.append(key)
    if report.resolved:
        report.notes.append(
            f"{len(report.resolved)} grandfathered entr"
            f"{'y is' if len(report.resolved) == 1 else 'ies are'} no longer a violation "
            f"(wired or deleted -- good). Non-blocking; run "
            f"`python3 scripts/check_production_reachability.py --write-baseline` to shrink "
            f"the baseline."
        )
    return report, violations, idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(REPO_ROOT), help="repository root to scan")
    ap.add_argument("--manifest", default=None, help=f"default: <root>/{MANIFEST_REL}")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="regenerate 'grandfathered' from the current violation set (may only shrink)",
    )
    ap.add_argument(
        "--allow-growth",
        action="store_true",
        help=(
            "with --write-baseline: permit the baseline to GROW. Intended for the "
            "one-time initial seeding of the snapshot only. Using this to silence a new "
            "unwired primitive defeats the entire gate -- ledger it instead."
        ),
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest) if args.manifest else root / MANIFEST_REL
    if not manifest_path.exists():
        print(f"FAIL: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = load_manifest(manifest_path)

    if args.write_baseline:
        exclude = manifest.get("config", {}).get(
            "exclude_dirs", list(DEFAULT_EXCLUDE_DIRS)
        )
        idx = build_index(root, exclude)
        ledger_keys = {
            e["key"]
            for e in manifest.get("ledger", [])
            if e.get("kind") != "unenforceable" and e.get("key")
        }
        current = {v.key for v in find_violations(idx, root)} - ledger_keys
        old = set(manifest.get("grandfathered", []))
        added = sorted(current - old)
        if added and not args.allow_growth:
            # Refuse to write: silently absorbing a new gap into the baseline is
            # the one way this gate could be neutered without anyone noticing.
            print(
                "REFUSED: the baseline would GROW by "
                f"{len(added)} entr{'y' if len(added) == 1 else 'ies'}: {added[:5]}\n"
                "  The baseline may only shrink. Wire the primitive, or add a reviewed\n"
                f"  'ledger' entry with a reason to {MANIFEST_REL}.\n"
                "  (--allow-growth exists only for the one-time initial seeding.)",
                file=sys.stderr,
            )
            return 1
        manifest["grandfathered"] = sorted(current)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
            fh.write("\n")
        print(f"baseline written: {len(manifest['grandfathered'])} entries")
        if added:
            print(
                "NOTE: --allow-growth added "
                f"{len(added)} entr{'y' if len(added) == 1 else 'ies'} to the baseline. "
                "This flag is for the one-time initial seeding only.",
                file=sys.stderr,
            )
        return 0

    report, violations, _idx = check(root, manifest)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "n_violations": len(violations),
                    "failures": report.failures,
                    "pending": report.pending,
                    "resolved": report.resolved,
                    "notes": report.notes,
                },
                indent=2,
            )
        )
        return 0 if report.ok else 1

    print(f"[production-reachability] {len(violations)} violation(s) in tree")
    print(
        f"[production-reachability] {len(report.pending)} accounted for "
        f"(ledgered or grandfathered)"
    )
    for note in report.notes:
        print(f"  note: {note}")
    if report.failures:
        print("")
        for f in report.failures:
            print(f"FAIL: {f}")
        print(f"\n[production-reachability] FAILED with {len(report.failures)} problem(s)")
        return 1
    print("[production-reachability] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
