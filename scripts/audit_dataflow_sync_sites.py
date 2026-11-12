#!/usr/bin/env python3
"""Static inventory of CPU<->GPU transfer and host-readback sites (issue #279, Phase 0/1).

Pure-stdlib AST scan. **Imports no torch** so it runs anywhere, including the
CPU-only CI image (see issue #536 -- a scanner that imported torch at module
scope would have its tests silently excluded from CI).

What it finds
-------------
Two families of call/attribute sites, per issue #279's Phase 1 taxonomy:

* ``D2H`` -- GPU->CPU synchronization candidates: ``.cpu()``, ``.numpy()``,
  ``.item()``, ``.tolist()``, ``float(t)``/``int(t)`` on a tensor-looking
  expression, ``torch.cuda.synchronize()``.
* ``H2D`` -- CPU->GPU transfer candidates: ``.cuda()``, ``.to(device=...)`` /
  ``.to("cuda")``, ``.pin_memory()``, and any call carrying ``non_blocking=``.

Classification (HOT / AMORTIZED / COLD / UNCLASSIFIED)
-----------------------------------------------------
Deliberately a *computed* rule over AST ancestry, not a curated list, so a
reader can re-derive every label. The rule, in priority order:

1. ``COLD`` -- the site's file is a cold-path module (reporting, eval, export,
   render, metrics, viewers) OR its enclosing function name matches a cold
   prefix/substring (report/eval/export/save/render/metric/plot/summary/
   ``__main__``/argument-validation/one-shot construction).
2. ``AMORTIZED`` -- some enclosing ``if``/``elif`` test contains a cadence
   marker (``% <interval>``, ``*_interval``, ``should_log``, ``testing_iterations``,
   ``saving_iterations``, ``checkpoint_iterations``, ``verbose``, ``debug``,
   ``tb_writer is not None``). The guard source text is recorded verbatim.
3. ``HOT`` -- the site is lexically inside a per-iteration or per-frame loop
   (a ``for``/``while`` whose target or iterable names an iteration/frame/camera)
   or inside a function registered in ``PER_FRAME_FUNCTIONS`` below, with no
   cadence guard above it.
4. ``HOT_REACHABLE`` -- a second pass. Rules 1-3 are purely *lexical*, so a
   host readback sitting in a helper that the hot loop calls every iteration
   would otherwise be missed. Pass 2 therefore propagates hotness along the
   call graph: every call expression that itself classified ``HOT`` marks the
   function it names as hot-reachable, transitively, skipping calls that sit
   under a cadence guard (those propagate no hotness). Sites in a hot-reachable
   function that rules 1-3 left unresolved become ``HOT_REACHABLE``.
   The call graph is resolved **by bare function/method name only** -- this
   scanner does not resolve imports or method receivers, so a name shared by a
   hot and a cold function over-marks. Treat ``HOT_REACHABLE`` as "on a
   per-iteration call path unless a name collision says otherwise", weaker
   evidence than ``HOT``.
5. ``UNCLASSIFIED`` -- everything the above cannot resolve. The count is
   published on purpose: it bounds how much of the split is assertion rather
   than derivation.

Rules 1-3 are intentionally conservative: a site is only ``HOT`` when the AST
proves per-iteration/per-frame reachability *lexically*, or the function is
named in ``PER_FRAME_FUNCTIONS``.

Usage
-----
    python scripts/audit_dataflow_sync_sites.py                # summary to stdout
    python scripts/audit_dataflow_sync_sites.py --json out.json
    python scripts/audit_dataflow_sync_sites.py --class HOT    # filter
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Scan scope
# --------------------------------------------------------------------------

#: Directories scanned. Everything else in the repo (tests, scripts, docs,
#: submodules, vendored ablator) is out of scope -- this audit is about what
#: a real training run executes.
SCAN_ROOTS = (
    "train.py",
    "train_streaming.py",
    "train_ros_node.py",
    "render.py",
    "metrics.py",
    "eligsir",
    "scene",
    "utils",
    "gaussian_renderer",
    "geometry",
    "slam",
    "streaming",
    "contracts",
    "loop_closure",
    "viz",
    "arguments",
)

EXCLUDE_PARTS = ("submodules", "ablator", "tests", "docs", "node_modules", ".git")

# --------------------------------------------------------------------------
# Site taxonomy
# --------------------------------------------------------------------------

D2H_METHODS = {"cpu", "item", "numpy", "tolist"}
H2D_METHODS = {"cuda", "pin_memory"}

#: Files whose entire contents are cold path (post-training reporting, offline
#: evaluation, export, standalone viewers). Paths are repo-relative prefixes.
COLD_FILES = (
    "render.py",
    "metrics.py",
    "utils/comparison_report.py",
    "utils/partition_eval.py",
    "utils/scene_geometry_metrics.py",
    "utils/map_comparison.py",
    "utils/seed_fade_severity.py",
    "utils/seed_fade_checker.py",
    "utils/image_utils.py",
    "utils/loss_utils.py",
    "gaussian_renderer/web_viewer.py",
    "viz/",
    "eligsir/viewers/",
    "eligsir/reporting/",
    "lpipsPyTorch/",
)

#: Substrings in the enclosing function's qualified name that mark cold path.
COLD_FUNC_MARKERS = (
    "report",
    "_eval",
    "eval_",
    "export",
    "save_",
    "_save",
    "render_report",
    "metric",
    "plot",
    "summar",
    "contact_sheet",
    "_dump",
    "diagnos",
    "validate_",
    "_validate",
    "audit",
    "compare_",
    "write_",
    "to_json",
    "from_json",
    "main",
)

#: Substrings in an enclosing ``if`` test that mark a cadence/amortized guard.
CADENCE_MARKERS = (
    "_interval",
    "interval)",
    "should_log",
    "testing_iterations",
    "saving_iterations",
    "checkpoint_iterations",
    "test_iterations",
    "tb_writer is not None",
    "verbose",
    "debug",
    "_every",
    "is_last",
    "final_",
)

#: Functions hand-registered as executing once per ingested frame (not per
#: optimizer iteration). Sites inside these are HOT unless cadence-guarded.
#: Each entry is ``<repo-relative-file>::<function name>``.
PER_FRAME_FUNCTIONS = frozenset(
    {
        "utils/camera_utils.py::_decode_image_tensor",
        "utils/camera_utils.py::loadCam",
        "eligsir/mapping/resolution_cache.py::build_resolution_cache",
        "eligsir/mapping/integration/frame_preparation.py::prepare_frame_tensors",
        "train_streaming.py::record_frame_observation",
        "train_streaming.py::bind_resolution_tier",
        "train_streaming.py::bind_resolution_tier_ex",
    }
)

#: Loop-target / iterable name fragments that mark a per-iteration or
#: per-frame loop body.
HOT_LOOP_MARKERS = ("iteration", "frame", "cam", "view", "step", "batch")


#: Hand-verified corrections applied LAST, over the computed rule, keyed either
#: by ``<file>::<function>`` (whole function) or ``<file>:<line>`` (one site).
#: Every entry was checked by reading the call sites; the reason states what was
#: read. These are counted and reported separately so the computed rule's own
#: accuracy stays visible -- an override is an admission the rule got it wrong.
HAND_VERIFIED_OVERRIDES: dict[str, tuple[str, str]] = {
    # Called exactly once, from train_streaming.py:9242, to write a snapshot
    # PNG grid. The `for cam in cameras` loop tripped the lexical HOT rule.
    "eligsir/mapping/integration/frame_preparation.py::_render_streaming_snapshot": (
        "COLD",
        "single call site (train_streaming.py:9242) is a snapshot render/PNG export",
    ),
    # An alternative whole-run training MODE (coordinator.run_submap_stitch_mode),
    # not a per-iteration step: these .cpu() calls fire once per submap.
    "eligsir/mapping/integration/submap_stitch_application.py::_run_submap_stitch": (
        "COLD",
        "runs once per submap inside an alternative one-shot training mode",
    ),
    # Append-only incremental cache (`self.train_cameras[len(cache):]`): the
    # readback happens once per NEWLY ARRIVED camera, not per iteration. This is
    # the cached-camera-centers optimization #279 cites from the June audit.
    "scene/streaming_scene.py::_ensure_camera_caches_np": (
        "AMORTIZED",
        "append-only cache fill; one readback per newly arrived camera, not per iteration",
    ),
    # Only callers are under scripts/ (local_quality_offline_demo.py,
    # local_region_report.py). No training-run call site exists.
    "utils/local_quality.py::covisibility_snapshot": (
        "COLD",
        "only callers are scripts/ offline tools; no training-run call site",
    ),
    # One-shot pre-loop precompute over all train cameras (train.py:771-772),
    # lexically above `for iteration in ...`.
    # Deliberately-uncached per-view migration; the method's own docstring
    # (utils/local_quality.py:277-289) states it runs once per view per tick and
    # that ticks are 1000 iterations apart, trading ~8 MiB/view of transfer for
    # O(1) rather than O(n_cameras) device residency.
    "utils/local_quality.py::index_on": (
        "AMORTIZED",
        "once per view per ~1000-iteration tick, per the method's own docstring",
    ),
    "train.py:772": (
        "COLD",
        "one-shot taming edge-map precompute before the training loop starts",
    ),
}


@dataclass
class Site:
    file: str
    line: int
    col: int
    direction: str  # "D2H" | "H2D"
    kind: str  # e.g. ".item()", ".to(device)", "torch.cuda.synchronize()"
    snippet: str
    function: str
    classification: str  # HOT | AMORTIZED | COLD | UNCLASSIFIED
    reason: str
    guards: list[str] = field(default_factory=list)
    non_blocking: bool = False
    in_loop: bool = False
    overridden: bool = False


def _iter_python_files(repo: Path):
    for root in SCAN_ROOTS:
        p = repo / root
        if p.is_file() and p.suffix == ".py":
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                if any(part in EXCLUDE_PARTS for part in f.parts):
                    continue
                yield f


def _seg(src_lines: list[str], node: ast.AST) -> str:
    try:
        lo = node.lineno - 1
        hi = getattr(node, "end_lineno", node.lineno)
        text = " ".join(l.strip() for l in src_lines[lo:hi])
    except Exception:
        return ""
    return text[:200]


def _is_cold_file(rel: str) -> bool:
    return any(rel == c or rel.startswith(c) for c in COLD_FILES)


def _classify(rel: str, func: str, guards: list[str], in_loop: bool) -> tuple[str, str]:
    if _is_cold_file(rel):
        return "COLD", f"cold-path module ({rel})"
    low = func.lower()
    for m in COLD_FUNC_MARKERS:
        if m in low:
            return "COLD", f"enclosing function name matches cold marker {m!r}"
    for g in guards:
        gl = g.lower()
        for m in CADENCE_MARKERS:
            if m in gl:
                return "AMORTIZED", f"cadence guard matches {m!r}: {g[:120]}"
        # bare modulo guard, e.g. `iteration % 100 == 0`
        if "%" in gl and ("iteration" in gl or "step" in gl or "frame" in gl):
            return "AMORTIZED", f"modulo cadence guard: {g[:120]}"
    key = f"{rel}::{func.split('.')[-1]}"
    if key in PER_FRAME_FUNCTIONS:
        return "HOT", "registered per-frame function, no cadence guard"
    if in_loop:
        return "HOT", "lexically inside a per-iteration/per-frame loop, no cadence guard"
    return "UNCLASSIFIED", "no cold marker, no cadence guard, not provably per-iteration"


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, src_lines: list[str]):
        self.rel = rel
        self.src = src_lines
        self.sites: list[Site] = []
        #: (caller_key, callee_name, caller_is_hot_context)
        self.edges: list[tuple[str, str, bool]] = []
        self._func: list[str] = []
        self._guards: list[str] = []
        self._loops: int = 0

    # -- scope tracking ----------------------------------------------------
    def visit_FunctionDef(self, node):
        self._func.append(node.name)
        # a new function body resets lexical loop/guard ancestry
        loops, guards = self._loops, self._guards
        self._loops, self._guards = 0, []
        self.generic_visit(node)
        self._loops, self._guards = loops, guards
        self._func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._func.append(node.name)
        self.generic_visit(node)
        self._func.pop()

    def visit_If(self, node):
        test = _seg(self.src, node.test)
        self._guards.append(test)
        for n in node.body:
            self.visit(n)
        self._guards.pop()
        for n in node.orelse:
            self.visit(n)

    def _visit_loop(self, node):
        text = _seg(self.src, node)
        hot = any(m in text.lower() for m in HOT_LOOP_MARKERS)
        self._loops += 1 if hot else 0
        self.generic_visit(node)
        self._loops -= 1 if hot else 0

    visit_For = _visit_loop
    visit_AsyncFor = _visit_loop
    visit_While = _visit_loop

    # -- site detection ----------------------------------------------------
    def _record(self, node, direction, kind, non_blocking=False):
        func = ".".join(self._func) or "<module>"
        guards = list(self._guards)
        in_loop = self._loops > 0
        cls, reason = _classify(self.rel, func, guards, in_loop)
        self.sites.append(
            Site(
                file=self.rel,
                line=node.lineno,
                col=node.col_offset,
                direction=direction,
                kind=kind,
                snippet=_seg(self.src, node),
                function=func,
                classification=cls,
                reason=reason,
                guards=guards[-3:],
                non_blocking=non_blocking,
                in_loop=in_loop,
            )
        )

    def _record_call_edge(self, node):
        """Record a caller->callee name edge, tagged with the caller's own
        hot/cold context, for the pass-2 reachability propagation."""
        f = node.func
        if isinstance(f, ast.Attribute):
            callee = f.attr
        elif isinstance(f, ast.Name):
            callee = f.id
        else:
            return
        caller = ".".join(self._func) or "<module>"
        cls, _ = _classify(self.rel, caller, list(self._guards), self._loops > 0)
        # Only HOT / HOT-context call sites propagate hotness. A call under a
        # cadence guard or in a cold module runs at that cadence, not per-iter.
        hot_ctx = cls == "HOT" or (
            self._loops > 0 and not _is_cold_file(self.rel) and not self._cadence_guarded()
        )
        self.edges.append((f"{self.rel}::{caller.split('.')[-1]}", callee, hot_ctx))

    def _cadence_guarded(self) -> bool:
        for g in self._guards:
            gl = g.lower()
            if any(m in gl for m in CADENCE_MARKERS):
                return True
            if "%" in gl and ("iteration" in gl or "step" in gl or "frame" in gl):
                return True
        return False

    def visit_Call(self, node):
        self._record_call_edge(node)
        f = node.func
        nb = any(
            kw.arg == "non_blocking"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
        if isinstance(f, ast.Attribute):
            name = f.attr
            if name in D2H_METHODS:
                self._record(node, "D2H", f".{name}()")
            elif name in H2D_METHODS:
                self._record(node, "H2D", f".{name}()", non_blocking=nb)
            elif name == "synchronize":
                base = _seg(self.src, f)
                if "cuda" in base:
                    self._record(node, "D2H", "torch.cuda.synchronize()")
            elif name == "to":
                seg = _seg(self.src, node).lower()
                if "cuda" in seg or "device" in seg or nb:
                    self._record(node, "H2D", ".to(device)", non_blocking=nb)
        elif isinstance(f, ast.Name) and f.id in ("float", "int", "bool"):
            # float(tensor)/int(tensor) forces a host read; only flag when the
            # argument is itself a call/attribute (a bare literal cannot sync).
            if node.args and isinstance(node.args[0], (ast.Call, ast.Attribute, ast.Subscript)):
                seg = _seg(self.src, node)
                if any(t in seg for t in (".sum(", ".mean(", ".max(", ".min(", ".item(", ".norm(", ".count_nonzero(")):
                    self._record(node, "D2H", f"{f.id}(tensor)")
        self.generic_visit(node)


def scan(repo: Path) -> list[Site]:
    out: list[Site] = []
    edges: list[tuple[str, str, bool]] = []
    for path in _iter_python_files(repo):
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = str(path.relative_to(repo))
        v = _Visitor(rel, text.splitlines())
        v.visit(tree)
        out.extend(v.sites)
        edges.extend(v.edges)
    _propagate_hot_reachability(out, edges)
    _apply_overrides(out)
    return out


def _apply_overrides(sites: list[Site]) -> None:
    for s in sites:
        for key in (f"{s.file}:{s.line}", f"{s.file}::{s.function.split('.')[-1]}"):
            hit = HAND_VERIFIED_OVERRIDES.get(key)
            if hit is not None:
                s.classification, s.reason = hit[0], f"hand-verified override: {hit[1]}"
                s.overridden = True
                break


def _propagate_hot_reachability(sites: list[Site], edges: list[tuple[str, str, bool]]) -> None:
    """Pass 2: mark UNCLASSIFIED sites whose function is called (transitively)
    from a hot-context call site. Name-resolved only -- see module docstring."""
    # callee-name -> set of caller keys, restricted to hot-context edges
    hot_callees: set[str] = {callee for _, callee, hot in edges if hot}
    # caller-key -> callee names (all edges; used for transitive walk)
    out_edges: dict[str, set[str]] = {}
    for caller, callee, _ in edges:
        out_edges.setdefault(caller, set()).add(callee)
    # short function name -> caller keys defining it
    by_name: dict[str, set[str]] = {}
    for caller in out_edges:
        by_name.setdefault(caller.split("::")[-1], set()).add(caller)
    for s in sites:
        by_name.setdefault(s.function.split(".")[-1], set()).add(
            f"{s.file}::{s.function.split('.')[-1]}"
        )

    seen: set[str] = set()
    frontier = set(hot_callees)
    hot_names: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        hot_names.add(name)
        for key in by_name.get(name, ()):
            for nxt in out_edges.get(key, ()):
                if nxt not in seen:
                    frontier.add(nxt)

    for s in sites:
        if s.classification != "UNCLASSIFIED":
            continue
        short = s.function.split(".")[-1]
        if short in hot_names:
            s.classification = "HOT_REACHABLE"
            s.reason = (
                f"pass 2: enclosing function {short!r} is name-reachable from a "
                "hot-context call site (name-resolved only)"
            )


def summarize(sites: list[Site]) -> dict:
    by_class: dict[str, int] = {}
    by_dir: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    cross: dict[str, dict[str, int]] = {}
    for s in sites:
        by_class[s.classification] = by_class.get(s.classification, 0) + 1
        by_dir[s.direction] = by_dir.get(s.direction, 0) + 1
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        cross.setdefault(s.direction, {})
        cross[s.direction][s.classification] = cross[s.direction].get(s.classification, 0) + 1
    nb = [s for s in sites if s.non_blocking]
    return {
        "total": len(sites),
        "by_classification": dict(sorted(by_class.items())),
        "by_direction": dict(sorted(by_dir.items())),
        "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])),
        "direction_x_classification": cross,
        "non_blocking_sites": len(nb),
        "hand_verified_overrides_applied": sum(1 for s in sites if s.overridden),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=None, help="repo root (default: parent of this script)")
    ap.add_argument("--json", dest="json_out", default=None, help="write full inventory JSON here")
    ap.add_argument("--class", dest="cls", default=None, help="filter by classification")
    ap.add_argument("--direction", default=None, help="filter by D2H/H2D")
    args = ap.parse_args(argv)

    repo = Path(args.repo) if args.repo else Path(__file__).resolve().parent.parent
    sites = scan(repo)
    summary = summarize(sites)

    shown = sites
    if args.cls:
        shown = [s for s in shown if s.classification == args.cls]
    if args.direction:
        shown = [s for s in shown if s.direction == args.direction]

    if args.json_out:
        payload = {"summary": summary, "sites": [asdict(s) for s in sites]}
        Path(args.json_out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"wrote {args.json_out} ({len(sites)} sites)")

    print(json.dumps(summary, indent=2))
    if args.cls or args.direction:
        for s in shown:
            print(f"{s.file}:{s.line}\t{s.direction}\t{s.kind}\t{s.classification}\t{s.function}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
