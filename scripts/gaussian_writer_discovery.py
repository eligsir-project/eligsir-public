#!/usr/bin/env python3
"""AST discovery of every code path that WRITES Gaussian topology, parameters,
lifecycle, support or anchor/reference state.

Issue #325 ("[Single-Owner 0] Build static Gaussian writer/owner registry"),
root of lane P1-E under master #324.

This module is deliberately split from ``scripts/gaussian_writer_registry.py``:
this half *discovers* writers mechanically from the AST and knows nothing about
ownership; the registry half carries the hand-curated ``declared_owner`` /
``semantic_operation`` annotation layer and the conflict checks. The join
between them fails loudly in BOTH directions (an undiscovered annotation and an
unannotated discovery are both errors), which is the drift guard -- the repo has
already been burned by hand-maintained inventories silently rotting (#478/#518,
and the two 2026-08 audit docs this issue exists to re-derive).

Design notes
------------
* **Over-inclusion is safe, under-inclusion is silent.** The scanner casts a
  wide net (raw ``params[...]`` writes, private-tensor writes, in-place tensor
  methods, sanctioned model methods, mutation-transaction calls, strategy
  entrypoints, ``setattr`` of shadow-buffer names, and writes through
  function-local aliases of any of the above). Anything that turns out not to
  be a real Gaussian writer is classified ``not_a_gaussian_writer`` *in the
  annotation layer*, with a reason -- it is never dropped by the scanner.
* **Stdlib only.** No torch, no numpy. This runs inside the lean CPU CI image
  (``Dockerfile.ci-cpu``), so its tests actually execute there (cf. #536: a
  module-scope ``import torch`` silently excludes a test file from CPU CI).
* ``SCAN_ROOTS`` is restricted to paths ``Dockerfile.ci-cpu`` actually COPYs,
  so a dev checkout and the CI image see the same file set.

Blind spots (stated, not hidden) -- see ``KNOWN_BLIND_SPOTS``.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent

# Kept in lockstep with the COPY lines in Dockerfile.ci-cpu. `tests/` is
# deliberately excluded (test doubles legitimately hand-build fake models).
SCAN_ROOTS: tuple[str, ...] = (
    "arguments",
    "contracts",
    "gaussian_renderer",
    "geometry",
    "loop_closure",
    "scene",
    "scripts",
    "slam",
    "eligsir",
    "streaming",
    "utils",
    "viz",
    "train.py",
    "train_streaming.py",
    # `train_ros_node.py` was removed by #1076's ROS hard cut (9df507ea); the
    # ROS entry point lives in eligsir-project/eligsir-ros2 now.
)

KNOWN_BLIND_SPOTS: tuple[str, ...] = (
    "dynamic dispatch: a write reached only through a variable holding a method "
    "(`fn = model.prune_points; fn(mask)`) is not resolved",
    "getattr/setattr with a non-literal name argument is not resolved",
    "writes performed inside C/C++/HIP extensions or third-party packages "
    "(gsplat.strategy.ops, torch fused optimizers) are attributed to their "
    "python call site only",
    "aliasing across function boundaries (a raw tensor handed to a helper that "
    "mutates it) is only caught if the helper itself matches a write shape",
    "`exec`/`eval`, monkeypatching, and __setattr__ overrides are not modelled",
    "the scanner proves a write EXISTS in source, not that it EXECUTES in any "
    "given run -- runtime reachability/cadence is #326's job, not this one",
)

# ---------------------------------------------------------------------------
# Vocabulary of things that count as Gaussian state
# ---------------------------------------------------------------------------

#: gsplat-layout parameter keys (``GsplatGaussianModel.params``).
PARAM_KEYS = frozenset({"means", "sh0", "shN", "opacities", "scales", "quats"})

#: legacy-layout private parameter tensors (``GaussianModel``).
PRIVATE_PARAMS = frozenset(
    {"_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest"}
)

#: per-Gaussian auxiliary ("shadow") buffers attached to the model instance.
SHADOW_BUFFERS = frozenset(
    {
        "provisional",
        "birth_frame",
        "birth_iter",
        "birth_method",
        "support_count",
        "lifecycle_state",
        "visibility_ema",
        "geometry_source",
        "surface_state",
        "seen_count",
        "seed_uid_range",
        "gaussian_uid",
        "submap_id",
        "_current_submap_id",
        # NOTE: the legacy densification accumulator pair is represented by
        # `xyz_gradient_accum` only. Its partner `denom` is deliberately NOT
        # listed: the bare name is far too common (loss/metric helpers all over
        # the repo have a local `.denom`) and every real model-side write of it
        # sits in a function that already writes `xyz_gradient_accum`, so
        # including it adds ~10 false-positive units and zero coverage.
        "xyz_gradient_accum",
        "max_radii2D",
        "anchor_xyz",
        "anchor_normal",
        "anchor_scale_log",
        "anchor_opacity_logit",
        "anchor_iter",
        "anchor_confidence",
        "consensus_support_count",
        "consensus_conflict_count",
        "consensus_last_seen_uid",
        "source_surface_cell",
        "_next_gaussian_uid",
    }
)

#: in-place tensor mutators.
INPLACE_METHODS = frozenset(
    {
        "add_",
        "sub_",
        "mul_",
        "div_",
        "clamp_",
        "clamp_min_",
        "clamp_max_",
        "copy_",
        "index_add_",
        "index_copy_",
        "index_fill_",
        "index_put_",
        "zero_",
        "fill_",
        "masked_fill_",
        "masked_scatter_",
        "scatter_",
        "scatter_add_",
        "normal_",
        "uniform_",
        "put_",
        "resize_",
        "set_",
    }
)

#: sanctioned mutation entrypoints on the two model classes.
MODEL_MUTATION_METHODS = frozenset(
    {
        "prune_points",
        "add_points_as_gaussians",
        "cat_tensors_to_optimizer",
        "replace_tensor_to_optimizer",
        "replace_tensors_to_optimizer",
        "densification_postfix",
        "restore",
        "load_ply",
        "create_from_pcd",
        "reset_opacity",
        "append_running_state",
        "reset_relocated_state",
        "grow_append",
        "relocate_gs",
        "relocate_gs_energy_guided",
        "add_new_gs",
        "add_new_gs_energy_guided",
    }
)

#: mutation-transaction / map-backend API.
TRANSACTION_METHODS = frozenset(
    {
        "begin_mutation",
        "commit_mutation",
        "abort_mutation",
        "rollback_mutation",
        "record_external_mutation",
    }
)

#: densification-strategy entrypoints that can mutate the population.
STRATEGY_METHODS = frozenset(
    {
        "step_pre_backward",
        "step_post_backward",
        "inject_noise",
        # NOTE: `add_new_gs` deliberately belongs to MODEL_MUTATION_METHODS
        # (a real population change), not here -- it is a model method the
        # strategy calls, not a strategy entrypoint.
        "_update_param_with_optimizer",
        "build_mutation_plan",
        "apply_mutation_plan",
    }
)

# ---------------------------------------------------------------------------
# State grouping
# ---------------------------------------------------------------------------

STATE_GROUPS: tuple[str, ...] = (
    "means_xyz",
    "scales",
    "rotation",
    "opacity",
    "sh_features",
    "population",  # count / create / grow / prune
    "lifecycle_state",
    "support_state",
    "anchor_reference",
    "identity_provenance",
    "optimizer_state",
    "topology_transaction",
    "strategy_dispatch",
    "params_bulk",  # whole-params or unresolved-key write
)

_GROUP_BY_NAME: dict[str, str] = {
    "means": "means_xyz",
    "_xyz": "means_xyz",
    "scales": "scales",
    "_scaling": "scales",
    "quats": "rotation",
    "_rotation": "rotation",
    "opacities": "opacity",
    "_opacity": "opacity",
    "sh0": "sh_features",
    "shN": "sh_features",
    "_features_dc": "sh_features",
    "_features_rest": "sh_features",
    "provisional": "lifecycle_state",
    "birth_frame": "lifecycle_state",
    "birth_iter": "lifecycle_state",
    "birth_method": "lifecycle_state",
    "lifecycle_state": "lifecycle_state",
    "surface_state": "lifecycle_state",
    "support_count": "support_state",
    "visibility_ema": "support_state",
    "seen_count": "support_state",
    "consensus_support_count": "support_state",
    "consensus_conflict_count": "support_state",
    "consensus_last_seen_uid": "support_state",
    "max_radii2D": "support_state",
    "anchor_xyz": "anchor_reference",
    "anchor_normal": "anchor_reference",
    "anchor_scale_log": "anchor_reference",
    "anchor_opacity_logit": "anchor_reference",
    "anchor_iter": "anchor_reference",
    "anchor_confidence": "anchor_reference",
    "source_surface_cell": "anchor_reference",
    "geometry_source": "identity_provenance",
    "gaussian_uid": "identity_provenance",
    "seed_uid_range": "identity_provenance",
    "submap_id": "identity_provenance",
    "_current_submap_id": "identity_provenance",
    "_next_gaussian_uid": "identity_provenance",
    "xyz_gradient_accum": "optimizer_state",
}

_GROUP_BY_METHOD: dict[str, str] = {
    **{m: "population" for m in MODEL_MUTATION_METHODS},
    **{m: "topology_transaction" for m in TRANSACTION_METHODS},
    **{m: "strategy_dispatch" for m in STRATEGY_METHODS},
}
# these model methods are really parameter rewrites, not population changes
_GROUP_BY_METHOD["relocate_gs"] = "population"
_GROUP_BY_METHOD["relocate_gs_energy_guided"] = "population"
_GROUP_BY_METHOD["reset_opacity"] = "opacity"


_ATTR_BOUNDARY_RE: dict[str, "re.Pattern[str]"] = {
    name: re.compile(r"\." + re.escape(name) + r"(?![A-Za-z0-9_])")
    for name in _GROUP_BY_NAME
}


@dataclass(frozen=True, order=True)
class WriteSite:
    """One syntactic write to Gaussian state."""

    path: str
    qualname: str
    lineno: int
    kind: str  # assign | augassign | inplace | setattr | model_method | transaction | strategy | alias_assign
    target: str  # rendered dotted/subscript path or call expression
    state_group: str


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _dotted(node.value) + "." + node.attr
    if isinstance(node, ast.Subscript):
        try:
            key = ast.literal_eval(node.slice)
        except Exception:
            return _dotted(node.value) + "[...]"
        return _dotted(node.value) + f"[{key!r}]"
    if isinstance(node, ast.Call):
        return _dotted(node.func) + "()"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(_dotted(e) for e in node.elts) + ")"
    if isinstance(node, ast.Starred):
        return "*" + _dotted(node.value)
    return "<expr>"


def _group_for_path(rendered: str) -> str | None:
    """Map a rendered write target onto a state group, or None if it is not
    recognisable Gaussian state."""
    # gsplat params dict, keyed
    for key in PARAM_KEYS:
        if f".params[{key!r}]" in rendered or f'.params["{key}"]' in rendered:
            return _GROUP_BY_NAME[key]
    if ".params[...]" in rendered or rendered.endswith(".params"):
        return "params_bulk"
    # Attribute names (private params + shadow buffers), longest first so that
    # e.g. `consensus_support_count` wins over `support_count`.
    #
    # Two deliberate restrictions:
    #  * the name must appear as an ATTRIBUTE (`.name`), never as a bare local
    #    variable -- `provisional = gaussians.provisional` is a read binding,
    #    not a write, and treating bare names as writes produced ~20 phantom
    #    units. A genuine write THROUGH such a local is still caught, by the
    #    function-local alias tracking in `_Visitor`.
    #  * the match is identifier-bounded, so `._xyz_grad_ema` (an evidence-EMA
    #    buffer that is not the model's `_xyz`) does not match `._xyz`.
    for name in sorted(_GROUP_BY_NAME, key=len, reverse=True):
        if name not in PRIVATE_PARAMS and name not in SHADOW_BUFFERS:
            continue
        if _ATTR_BOUNDARY_RE[name].search(rendered):
            return _GROUP_BY_NAME[name]
    return None


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self._stack: list[str] = []
        # function-local aliases: name -> state_group, for `x = model.params["means"].data`
        self._alias_stack: list[dict[str, str]] = [{}]
        self.sites: list[WriteSite] = []

    # -- scope bookkeeping -------------------------------------------------
    @property
    def _qualname(self) -> str:
        return ".".join(self._stack) if self._stack else "<module>"

    def _push(self, name: str) -> None:
        self._stack.append(name)
        self._alias_stack.append(dict(self._alias_stack[-1]))

    def _pop(self) -> None:
        self._stack.pop()
        self._alias_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._push(node.name)
        self.generic_visit(node)
        self._pop()

    # -- recording ---------------------------------------------------------
    def _record(self, lineno: int, kind: str, target: str, group: str) -> None:
        self.sites.append(
            WriteSite(
                path=self.path,
                qualname=self._qualname,
                lineno=lineno,
                kind=kind,
                target=target,
                state_group=group,
            )
        )

    def _alias_group(self, node: ast.AST) -> str | None:
        """Resolve a write target that goes through a function-local alias."""
        base = node
        while isinstance(base, (ast.Subscript, ast.Attribute)):
            base = base.value
        if isinstance(base, ast.Name):
            return self._alias_stack[-1].get(base.id)
        return None

    # -- write shapes ------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        # (1) real writes
        for tgt in node.targets:
            rendered = _dotted(tgt)
            group = _group_for_path(rendered)
            if group is not None:
                self._record(node.lineno, "assign", rendered, group)
            elif isinstance(tgt, (ast.Subscript, ast.Attribute)):
                alias = self._alias_group(tgt)
                if alias is not None:
                    self._record(node.lineno, "alias_assign", rendered, alias)
        # (2) alias creation: `_op = gaussians.params["opacities"].data`
        src_group = _group_for_path(_dotted(node.value))
        if src_group is not None and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                self._alias_stack[-1][tgt.id] = src_group
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        rendered = _dotted(node.target)
        group = _group_for_path(rendered)
        if group is None:
            group = self._alias_group(node.target)
            kind = "alias_assign"
        else:
            kind = "augassign"
        if group is not None:
            self._record(node.lineno, kind, rendered, group)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func = node.func
        if isinstance(func, ast.Attribute):
            recv = _dotted(func.value)
            attr = func.attr
            if attr in INPLACE_METHODS:
                group = _group_for_path(recv) or self._alias_group(func.value)
                if group is not None:
                    self._record(node.lineno, "inplace", f"{recv}.{attr}()", group)
            elif attr in _GROUP_BY_METHOD:
                self._record(
                    node.lineno,
                    _kind_for_method(attr),
                    f"{recv}.{attr}()",
                    _GROUP_BY_METHOD[attr],
                )
        elif isinstance(func, ast.Name):
            if func.id == "setattr" and len(node.args) >= 2:
                try:
                    name = ast.literal_eval(node.args[1])
                except Exception:
                    name = None
                if isinstance(name, str) and (
                    name in SHADOW_BUFFERS or name in PRIVATE_PARAMS
                ):
                    self._record(
                        node.lineno,
                        "setattr",
                        f"setattr(..., {name!r}, ...)",
                        _GROUP_BY_NAME.get(name, "params_bulk"),
                    )
            elif func.id in _GROUP_BY_METHOD:
                self._record(
                    node.lineno,
                    _kind_for_method(func.id),
                    f"{func.id}()",
                    _GROUP_BY_METHOD[func.id],
                )
        self.generic_visit(node)


def _kind_for_method(name: str) -> str:
    if name in TRANSACTION_METHODS:
        return "transaction"
    if name in STRATEGY_METHODS:
        return "strategy"
    return "model_method"


def iter_python_files(root: Path = REPO_ROOT, roots: Iterable[str] = SCAN_ROOTS) -> Iterator[Path]:
    for rel in roots:
        target = root / rel
        if target.is_file():
            yield target
        elif target.is_dir():
            yield from sorted(target.rglob("*.py"))


def discover(root: Path = REPO_ROOT, roots: Iterable[str] = SCAN_ROOTS) -> list[WriteSite]:
    """Return every discovered Gaussian write site, sorted deterministically."""
    sites: list[WriteSite] = []
    for file in iter_python_files(root, roots):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = _Visitor(file.relative_to(root).as_posix())
        visitor.visit(tree)
        sites.extend(visitor.sites)
    return sorted(sites)


@dataclass
class WriterUnit:
    """All writes of Gaussian state inside one function, in one file."""

    path: str
    qualname: str
    state_groups: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}"


def group_units(sites: Iterable[WriteSite]) -> dict[str, WriterUnit]:
    units: dict[str, WriterUnit] = {}
    for site in sites:
        key = f"{site.path}::{site.qualname}"
        unit = units.get(key)
        if unit is None:
            unit = WriterUnit(path=site.path, qualname=site.qualname)
            units[key] = unit
        if site.state_group not in unit.state_groups:
            unit.state_groups.append(site.state_group)
        if site.kind not in unit.kinds:
            unit.kinds.append(site.kind)
        unit.lines.append(site.lineno)
        if len(unit.examples) < 4 and site.target not in unit.examples:
            unit.examples.append(site.target)
    for unit in units.values():
        unit.state_groups.sort()
        unit.kinds.sort()
        unit.lines.sort()
    return dict(sorted(units.items()))


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit raw sites as JSON")
    ap.add_argument("--units", action="store_true", help="emit grouped writer units")
    args = ap.parse_args()

    sites = discover()
    if args.json:
        print(json.dumps([asdict(s) for s in sites], indent=2))
        return 0
    units = group_units(sites)
    if args.units:
        print(json.dumps({k: asdict(v) for k, v in units.items()}, indent=2))
        return 0
    print(f"{len(sites)} write sites in {len(units)} writer units")
    for key, unit in units.items():
        print(f"  {key}  {unit.state_groups}  n={len(unit.lines)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
