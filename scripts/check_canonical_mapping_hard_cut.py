#!/usr/bin/env python3
"""Pure-AST dependency-cone guard for the #1224 production hard cut."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_ROOTS = (
    "eligsir/runtime/causal_entry.py",
    "eligsir/runtime/production_application.py",
)
CANONICAL_DEPENDENCIES = (
    "eligsir/mapping/canonical_bootstrap.py",
    "eligsir/mapping/canonical_scene.py",
    "eligsir/mapping/canonical_services.py",
    "eligsir/mapping/admission/keyframe_provider.py",
    "eligsir/mapping/integration/device_integration_planner.py",
    "eligsir/mapping/integration/device_candidate_producer.py",
    "eligsir/mapping/optimization_runtime.py",
    "eligsir/runtime/payload_residency.py",
)
FORBIDDEN_IMPORTS = {
    "scene.streaming_scene",
    "train_streaming",
    "eligsir.mapping.coordinator",
    "eligsir.runtime.streaming_sink",
    "eligsir.mapping.integration.integration_planner",
    "eligsir.mapping.integration.rgbd_integrator_impl",
    "eligsir.mapping.scheduler.spf_resolution",
    "eligsir.mapping.resolution_controller",
}
CANONICAL_EXECUTORS = {
    "eligsir/mapping/integration/device_integration_planner.py": {
        "insert_gaussians_from_frame_device_native"
    },
    "eligsir/mapping/integration/device_candidate_producer.py": {
        "finalize_device_surface_candidates"
    },
}
FORBIDDEN_EXECUTOR_AUTHORITIES = {
    "ControlProposal", "MappingWorkPlan", "WorkBudgetOwner",
    "ResolutionPolicyProvider", "ObservationDisposition",
}
STRUCTURAL_WRITERS = {
    "add_gaussians", "add_gaussians_from_tensors", "densify_and_clone",
    "densify_and_prune", "densify_with_taming_scores", "prune_points",
    "relocate", "relocate_gs",
}
RETIRED_OPTIONS = {"streaming_optimization_engine"}
RAW_CAPABILITY_OPTIONS = {
    "gpu_frame_prep", "gpu_frame_resident", "streaming_surface_cell_dedup",
    "streaming_insert_confidence_opacity",
}


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    line: int
    rule: str
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def _tree(root: Path, rel: str) -> ast.Module | None:
    path = root / rel
    if not path.exists():
        return None
    return ast.parse(path.read_text(encoding="utf-8"), filename=rel)


def _module_path(root: Path, module: str) -> str | None:
    relative = module.replace(".", "/")
    candidates = (f"{relative}.py", f"{relative}/__init__.py")
    return next((item for item in candidates if (root / item).is_file()), None)


def _imports(tree: ast.Module) -> tuple[ast.AST, ...]:
    """Include lazy imports: invoked canonical helpers make them reachable."""

    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )


def _package_for_path(rel: str) -> tuple[str, ...]:
    parts = list(Path(rel).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    elif parts:
        parts.pop()
    return tuple(parts)


def _modules(rel: str, node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level:
            package = _package_for_path(rel)
            parents = node.level - 1
            if parents > len(package):
                return ()
            base_parts = package[: len(package) - parents]
            module_parts = tuple(part for part in module.split(".") if part)
            base = ".".join((*base_parts, *module_parts))
        else:
            base = module
        modules = [base] if base else []
        for alias in node.names:
            modules.append(f"{base}.{alias.name}" if base else alias.name)
        return tuple(dict.fromkeys(modules))
    return ()


def _dependency_cone(root: Path) -> tuple[str, ...]:
    pending = list(dict.fromkeys((*CANONICAL_ROOTS, *CANONICAL_DEPENDENCIES)))
    seen: set[str] = set()
    while pending:
        rel = pending.pop()
        if rel in seen or not (root / rel).is_file():
            continue
        seen.add(rel)
        tree = _tree(root, rel)
        if tree is None:
            continue
        for node in _imports(tree):
            for module in _modules(rel, node):
                imported = _module_path(root, module)
                if imported is not None and imported not in seen:
                    pending.append(imported)
    return tuple(sorted(seen))


def _aliases(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                result[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                result[alias.asname or alias.name.split(".")[0]] = alias.name
    return result


def _call_name(node: ast.Call, aliases: dict[str, str] | None = None) -> str:
    if isinstance(node.func, ast.Name):
        return (aliases or {}).get(node.func.id, node.func.id)
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _subscript_key(node: ast.Subscript) -> str | None:
    value = node.slice
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None


def _check_dependency_cone(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    top_import_cache: dict[str, set[int]] = {}
    for rel in _dependency_cone(root):
        tree = _tree(root, rel)
        if tree is None:
            continue
        aliases = _aliases(tree)
        top_import_cache[rel] = {id(node) for node in _imports(tree)}
        for node in ast.walk(tree):
            if id(node) in top_import_cache[rel]:
                for module in _modules(rel, node):
                    if module in FORBIDDEN_IMPORTS:
                        violations.append(Violation(rel, node.lineno, "legacy-import", module))
            if isinstance(node, ast.Call):
                name = _call_name(node, aliases)
                if name == "streaming_training":
                    violations.append(Violation(rel, node.lineno, "legacy-dispatch", name))
                if name == "ObservationDisposition" and (
                    rel in CANONICAL_ROOTS or rel.endswith("provider.py")
                ):
                    violations.append(Violation(rel, node.lineno, "provider-disposition", "providers may not mint terminal dispositions"))
                if name in {"ingest_next_frame", "ingest_frame"} and (
                    rel in CANONICAL_ROOTS or rel.startswith("eligsir/runtime/")
                ):
                    violations.append(Violation(rel, node.lineno, "direct-source-ingress", name))
                if name == "StreamingScene" and (
                    len(node.args) > 2
                    or any(keyword.arg == "frame_source" for keyword in node.keywords)
                ):
                    violations.append(Violation(rel, node.lineno, "direct-source-ingress", "FrameSource passed to scene"))
                if name == "MappingCoordinator" and rel in CANONICAL_ROOTS:
                    violations.append(Violation(rel, node.lineno, "legacy-coordinator-surface", name))
                for keyword in node.keywords:
                    if keyword.arg in RETIRED_OPTIONS:
                        violations.append(Violation(rel, keyword.value.lineno, "legacy-composition-authority", keyword.arg))
            if isinstance(node, ast.Attribute):
                if node.attr in RETIRED_OPTIONS and (
                    rel in CANONICAL_ROOTS or rel.startswith("eligsir/runtime/")
                ):
                    violations.append(Violation(rel, node.lineno, "legacy-composition-authority", node.attr))
                if (
                    rel.endswith(("production_application.py", "device_integration_planner.py", "device_candidate_producer.py"))
                    and isinstance(node.value, ast.Name)
                    and node.value.id in {"args", "config"}
                    and node.attr in RAW_CAPABILITY_OPTIONS
                ):
                    violations.append(Violation(rel, node.lineno, "canonical-option-read", node.attr))
            if isinstance(node, ast.Subscript):
                key = _subscript_key(node)
                if key in RETIRED_OPTIONS and (
                    rel in CANONICAL_ROOTS or rel.startswith("eligsir/runtime/")
                ):
                    violations.append(Violation(rel, node.lineno, "legacy-composition-authority", key))
                if key in RAW_CAPABILITY_OPTIONS and rel.endswith("production_application.py"):
                    violations.append(Violation(rel, node.lineno, "canonical-option-subscript", key))
        if rel.startswith("eligsir/runtime/"):
            for function in (
                node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                for call in (node for node in ast.walk(function) if isinstance(node, ast.Call)):
                    name = _call_name(call, aliases)
                    if name in STRUCTURAL_WRITERS and function.name != "_commit_mutation_barrier":
                        violations.append(Violation(rel, call.lineno, "structural-write-outside-barrier", f"{function.name} calls {name}"))
    return violations


def _function_nodes(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _check_executors(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    for rel, names in CANONICAL_EXECUTORS.items():
        tree = _tree(root, rel)
        if tree is None:
            continue
        functions = _function_nodes(tree)
        for name in names:
            function = functions.get(name)
            if function is None:
                violations.append(Violation(rel, 1, "missing-executor", name))
                continue
            arguments = {
                argument.arg for argument in
                (*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs)
            }
            if "args" in arguments:
                violations.append(Violation(rel, function.lineno, "executor-args-read", name))
            loaded = {
                node.id for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            for authority in sorted(loaded & FORBIDDEN_EXECUTOR_AUTHORITIES):
                violations.append(Violation(rel, function.lineno, "executor-work-selection", f"{name} reads {authority}"))
            for node in ast.walk(function):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
                    violations.append(Violation(rel, node.lineno, "executor-args-read", f"{name} reads args.{node.attr}"))
    return violations


def _check_mutation_barrier(root: Path) -> list[Violation]:
    rel = "eligsir/runtime/production_application.py"
    tree = _tree(root, rel)
    if tree is None:
        return []
    runtime = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CanonicalMappingRuntime"), None)
    if runtime is None:
        return [Violation(rel, 1, "missing-runtime", "CanonicalMappingRuntime")]
    barrier = next((node for node in runtime.body if isinstance(node, ast.FunctionDef) and node.name == "_commit_mutation_barrier"), None)
    if barrier is None:
        return [Violation(rel, 1, "missing-mutation-barrier", "runtime")]
    calls = {_call_name(node) for node in ast.walk(barrier) if isinstance(node, ast.Call)}
    if "GaussianMutationEpoch" not in calls or "commit" not in calls:
        return [Violation(rel, barrier.lineno, "invalid-mutation-barrier", "barrier must construct and commit GaussianMutationEpoch")]
    return []


def _load_profile(path: Path, chain: tuple[Path, ...] = ()) -> dict:
    resolved = path.resolve()
    if resolved in chain:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: dict = {}
    parents = data.pop("extends", ())
    if isinstance(parents, str):
        parents = (parents,)
    for parent in parents:
        merged.update(_load_profile(path.parent / str(parent), (*chain, resolved)))
    merged.update(data)
    return merged


def _check_historical_boundaries(root: Path) -> list[Violation]:
    violations: list[Violation] = []
    train = _tree(root, "train.py")
    if train is not None:
        aliases = _aliases(train)
        top = {id(node) for node in _imports(train)}
        for node in ast.walk(train):
            if isinstance(node, ast.Call) and _call_name(node, aliases) == "streaming_training":
                violations.append(Violation("train.py", node.lineno, "legacy-dispatch", "streaming_training"))
            if id(node) in top and "train_streaming" in _modules("train.py", node):
                violations.append(Violation("train.py", node.lineno, "legacy-dispatch", "train_streaming"))
    profile_rel = "configs/profiles/live/canonical_mapping.yml"
    profile = root / profile_rel
    if profile.exists():
        merged = _load_profile(profile)
        for option in sorted(RETIRED_OPTIONS & set(merged)):
            violations.append(Violation(profile_rel, 1, "legacy-composition-authority", option))
    return violations


def check(root: Path = REPO_ROOT) -> tuple[Violation, ...]:
    return tuple(
        _check_dependency_cone(root) + _check_executors(root)
        + _check_mutation_barrier(root) + _check_historical_boundaries(root)
    )


def main() -> int:
    violations = check(REPO_ROOT)
    if violations:
        print("canonical mapping hard-cut violations:", file=sys.stderr)
        for violation in violations:
            print("  " + violation.render(), file=sys.stderr)
        return 1
    print("canonical mapping hard-cut: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
