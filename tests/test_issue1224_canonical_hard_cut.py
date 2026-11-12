"""Pure-AST teeth checks for the canonical mapping production boundary."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent
_CHECKER = _REPO / "scripts" / "check_canonical_mapping_hard_cut.py"


def _load_checker():
    name = "check_canonical_mapping_hard_cut"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _CHECKER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_checker_rejects_args_driven_executor_work_selection(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/mapping/integration/device_integration_planner.py",
        "def insert_gaussians_from_frame_device_native(args):\n"
        "    return WorkBudgetOwner().resolve(args.streaming_depth_stride)\n",
    )

    violations = checker.check(tmp_path)
    rules = {violation.rule for violation in violations}
    assert "executor-args-read" in rules
    assert "executor-work-selection" in rules


def test_checker_rejects_legacy_import_direct_source_and_provider_disposition(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/runtime/causal_entry.py",
        "import train_streaming\n"
        "def bad(scene, source):\n"
        "    StreamingScene(None, None, source)\n"
        "    scene.ingest_next_frame()\n"
        "    return ObservationDisposition()\n",
    )

    violations = checker.check(tmp_path)
    rules = {violation.rule for violation in violations}
    assert "legacy-import" in rules
    assert "direct-source-ingress" in rules
    assert "provider-disposition" in rules


def test_checker_rejects_structural_writer_outside_mutation_barrier(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/runtime/production_application.py",
        "class CanonicalMappingRuntime:\n"
        "    def _commit_mutation_barrier(self):\n"
        "        GaussianMutationEpoch().commit(())\n"
        "    def tick(self):\n"
        "        self.gaussians.prune_points(None)\n",
    )

    violations = checker.check(tmp_path)
    assert "structural-write-outside-barrier" in {
        violation.rule for violation in violations
    }


def test_checker_follows_imported_canonical_helpers_and_aliases(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/runtime/causal_entry.py",
        "from eligsir.runtime.helper import compose as build\n"
        "def run():\n"
        "    return build()\n",
    )
    _write(
        tmp_path,
        "eligsir/runtime/helper.py",
        "from train_streaming import streaming_training as retired\n"
        "def compose():\n"
        "    return retired(streaming_optimization_engine=True)\n",
    )

    violations = checker.check(tmp_path)
    rules = {violation.rule for violation in violations}
    assert "legacy-import" in rules
    assert "legacy-dispatch" in rules
    assert "legacy-composition-authority" in rules


def test_checker_resolves_relative_imports_into_the_canonical_cone(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/runtime/causal_entry.py",
        "from .helper import compose\n"
        "def run():\n"
        "    return compose()\n",
    )
    _write(
        tmp_path,
        "eligsir/runtime/helper.py",
        "from scene.streaming_scene import StreamingScene\n"
        "def compose():\n"
        "    return StreamingScene\n",
    )

    violations = checker.check(tmp_path)
    assert any(
        violation.rule == "legacy-import"
        and violation.detail == "scene.streaming_scene"
        for violation in violations
    )


def test_checker_rejects_keyword_frame_source_scene_ingress(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/runtime/causal_entry.py",
        "from scene.streaming_scene import StreamingScene\n"
        "def compose(source):\n"
        "    return StreamingScene(args=None, gaussians=None, frame_source=source)\n",
    )

    violations = checker.check(tmp_path)
    rules = {violation.rule for violation in violations}
    assert "legacy-import" in rules
    assert "direct-source-ingress" in rules


def test_checker_rejects_legacy_coordinator_imported_by_canonical_helper(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/runtime/causal_entry.py",
        "from eligsir.mapping.canonical_services import Services\n",
    )
    _write(
        tmp_path,
        "eligsir/mapping/canonical_services.py",
        "def Services():\n"
        "    from eligsir.mapping.coordinator import MappingCoordinator as Legacy\n"
        "    return Legacy()\n",
    )

    violations = checker.check(tmp_path)
    assert any(
        violation.rule == "legacy-import"
        and violation.detail == "eligsir.mapping.coordinator"
        for violation in violations
    )


def test_checker_rejects_indirect_writer_and_args_subscript(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "eligsir/runtime/production_application.py",
        "from eligsir.runtime.writer import mutate as apply_change\n"
        "class CanonicalMappingRuntime:\n"
        "    def _commit_mutation_barrier(self):\n"
        "        GaussianMutationEpoch().commit(())\n"
        "    def tick(self):\n"
        "        apply_change(self.config['gpu_frame_resident'])\n",
    )
    _write(
        tmp_path,
        "eligsir/runtime/writer.py",
        "def mutate(value):\n"
        "    return gaussians.prune_points(value)\n",
    )

    violations = checker.check(tmp_path)
    rules = {violation.rule for violation in violations}
    assert "structural-write-outside-barrier" in rules
    assert "canonical-option-subscript" in rules


def test_checker_validates_merged_canonical_profile(tmp_path):
    checker = _load_checker()
    _write(
        tmp_path,
        "configs/profiles/default.yml",
        "streaming_optimization_engine: true\n",
    )
    _write(
        tmp_path,
        "configs/profiles/live/canonical_mapping.yml",
        "extends:\n  - ../default.yml\nexecution_trainer: causal_mapping\n",
    )

    violations = checker.check(tmp_path)
    assert "legacy-composition-authority" in {
        violation.rule for violation in violations
    }


def test_repository_satisfies_canonical_mapping_hard_cut():
    checker = _load_checker()
    assert checker.check(_REPO) == ()
