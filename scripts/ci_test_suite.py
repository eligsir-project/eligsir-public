"""One CPU-Torch PR gate; full discovery is nightly/manual, never deleted.

Selection is core + changed tests + direct module/path references. This is a
bounded smoke heuristic, NOT transitive coverage analysis. Unknown code changes,
shared test helpers, dependency changes, and an unavailable diff select full CI.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
# Native import/gsplat coupling: same exclusions as the former CPU-Torch job.
NATIVE_ONLY = tuple(f"tests/{name}.py" for name in (
    "test_sh_clamp_diagnostic_native_parity", "test_sh_native_backward_gradient_audit",
    "test_sh_native_backward_synthetic_parity", "test_w3_12_graph_runtime_fallback_safety",
    "test_w3_12_graph_runtime_parity", "test_sh_impl_axis", "test_sh_impl_clamp_warning",
))
CPU_MARKERS = "not cuda and not slow and not hardware and not scientific"


def core_tests(root: Path) -> list[str]:
    paths = [line.split("#", 1)[0].strip()
             for line in (root / "tests/pr_core.txt").read_text().splitlines()]
    paths = [path for path in paths if path]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("PR core must be nonempty and contain no duplicates")
    for path in paths:
        parsed = PurePosixPath(path)
        if (not path.startswith("tests/") or ".." in parsed.parts
                or not parsed.name.startswith("test_") or parsed.suffix != ".py"
                or not (root / path).is_file() or path in NATIVE_ONLY):
            raise ValueError(f"Missing or ineligible core test: {path}")
    return paths


def changed_paths(base: str, root: Path = ROOT) -> list[str] | None:
    if not base or set(base) == {"0"}:
        return None
    try:
        result = subprocess.run(
            ["git", "diff", "--no-renames", "--name-only", "-z", base, "HEAD", "--"], cwd=root,
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        print(f"Git diff unavailable; selecting full coverage: {(detail or str(error))[:2000]}",
              file=sys.stderr, flush=True)
        return None  # Never turn an unavailable/truncated diff into docs-only CI.
    return [path for path in result.stdout.split("\0") if path]


def select_tests(root: Path, changed: list[str] | None, *, full: bool = False) -> dict:
    core = core_tests(root)
    tests = {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
             for path in sorted((root / "tests").rglob("test_*.py"))}
    selected = set(core)
    reason = "explicit full run" if full else "core + changed tests + direct references"
    if changed is None:
        full, reason = True, "diff unavailable"
    for name in changed or []:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid repository path: {name!r}")
        if name in tests:
            selected.add(name)
            continue
        if name.startswith("tests/") and path.name.startswith("test_") and path.suffix == ".py":
            continue  # Deleted test; Git history, not a nonexistent pytest argument.
        # Gate edits affect every test, even when one test directly references them.
        if name in {
            "scripts/ci_test_suite.py", "tests/pr_core.txt", ".github/workflows/ci-cpu.yml",
            "pyproject.toml", "uv.lock", "Dockerfile.ci-sim", ".dockerignore",
            "pytest.ini", ".pytest.ini", "setup.cfg", "tox.ini",
        } or path.name == "conftest.py" or (name.startswith("tests/") and path.suffix == ".py"):
            full, reason = True, f"shared test/dependency surface: {name}"
            continue
        if path.suffix in {".md", ".rst"}:
            continue
        module = name.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        # Include source-inspection/subprocess tests as well as import users.
        needles = {name, module, path.stem if path.stem != "__init__" else path.parent.name}
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(n) for n in needles) + r")(?!\w)")
        related = {test for test, source in tests.items() if pattern.search(source)}
        if related:
            selected.update(related)
        else:
            full, reason = True, f"no direct test reference: {name}"
    return {
        "mode": "full" if full else "pr", "reason": reason,
        "tests": ["tests"] if full else sorted(selected - set(NATIVE_ONLY)),
        "core_files": len(core), "available_files": len(tests),
        "native_exclusions": list(NATIVE_ONLY), "changed": changed,
    }


class _ExecutedCore:
    def __init__(self):
        self.paths: set[str] = set()

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and not report.skipped:
            self.paths.add(report.nodeid.split("::", 1)[0])


def run_suite(plan: dict, root: Path = ROOT) -> int:
    import pytest
    import torch

    core = set(core_tests(root))
    paths = plan["tests"]
    if plan["mode"] not in {"pr", "full"} or not isinstance(paths, list) or not paths:
        raise ValueError("Invalid or empty CI test plan")
    if plan["mode"] == "full":
        if paths != ["tests"]:
            raise ValueError("Full CI must discover tests/")
    else:
        available = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")}
        if not core.issubset(paths) or not set(paths).issubset(available):
            raise ValueError("PR plan omits core tests or contains missing/non-test paths")
    if torch.cuda.is_available():
        raise RuntimeError("CPU contract gate must not use CUDA/ROCm devices")
    print(json.dumps(plan, indent=2), flush=True)
    print(f"torch={torch.__version__}; CPU threads={torch.get_num_threads()}", flush=True)
    os.chdir(root)
    executed = _ExecutedCore()
    result = int(pytest.main([
        "-q", "-rs", "--durations=15", "--junitxml=/tmp/ci-tests.xml", "-m", CPU_MARKERS,
        *[f"--ignore={path}" for path in NATIVE_ONLY], *paths,
    ], plugins=[executed]))
    missing = sorted(core - executed.paths)
    if missing:
        print("CORE COVERAGE NOT EXECUTED: " + ", ".join(missing), flush=True)
    return result or (1 if missing else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--core", action="store_true", help="Explicit core-only validation")
    parser.add_argument("--base")
    args = parser.parse_args()
    # Explicit commands override an inherited workflow plan; no-argument CI uses it.
    if "CI_TEST_PLAN" in os.environ and not (args.plan or args.full or args.core or args.base is not None):
        plan = json.loads(os.environ["CI_TEST_PLAN"])
    else:
        changed = [] if args.core or args.full else changed_paths(args.base or "")
        plan = select_tests(ROOT, changed, full=args.full)
    if args.plan:
        encoded = json.dumps(plan, separators=(",", ":"))
        print(json.dumps(plan, indent=2))
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a") as stream:
                stream.write(f"plan={encoded}\n")
        return 0
    return run_suite(plan)


if __name__ == "__main__":
    raise SystemExit(main())
