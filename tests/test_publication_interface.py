from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import yaml


ROOT = Path(__file__).resolve().parents[1]

# Explicit publication closure. Optional campaign helpers are quarantined and
# must not silently re-enter a clean export merely because they are present in
# a developer checkout.
RETAINED_RUNTIME_UTILS = frozenset(
    {
        "__init__.py",
        "admission_cadence.py", "admission_fingerprint.py", "artifact_manifest.py", "assimilation_pressure.py",
        "baseline_comparison.py", "camera_residency.py", "camera_utils.py",
        "cfg_args_io.py", "champion_promotion.py", "checkpoint_io.py",
        "comparison_report.py", "compiled_kernels.py", "coverage_access.py",
        "coverage_ledger.py", "depth_alpha_diagnostics.py", "depth_sparsity_schedule.py",
        "differentiable_energy_loss.py", "duty_cycle.py", "energy_mcmc.py",
        "eval_tick.py", "execution_semantics.py", "experiment_declaration.py",
        "experiment_origin.py", "experiment_validity.py", "gaussian_belief.py",
        "gaussian_energy_policy.py", "gaussian_lineage.py", "general_utils.py",
        "geometry_metrics.py", "git_metadata.py", "gpu_arch_guard.py",
        "gpu_coresidency.py", "gpu_memory_telemetry.py", "graphics_utils.py",
        "holdout_rule.py", "image_provenance.py", "image_utils.py",
        "ingest_phase_profiler.py", "live_antialiasing_metrics.py", "local_quality.py",
        "loss_utils.py", "map_comparison.py", "mapping_phase_backfill.py",
        "mcmc_candidate_funnel.py", "mcmc_schedule.py", "metric_registry.py",
        "partition_control.py", "partition_eval.py", "partition_health.py",
        "pointcloud_preprocess.py", "presentation_ssaa.py", "psnr_history.py",
        "psnr_history_io.py", "realized_cadence.py", "reloc_utils.py",
        "resolution_exposure.py", "resource_monitor.py", "rgbd_frames.py",
        "rocm_knn_fallback.py", "rocm_reloc_fallback.py", "run_completion_state.py",
        "run_intent.py", "run_summary.py", "scannet_pose_io.py",
        "scene_geometry_metrics.py", "score_contract.py", "seed_fade_severity.py",
        "seed_lineage_freeze.py", "sh_utils.py", "source_attestation.py",
        "streaming_depth_lock.py", "streaming_frames.py", "streaming_state_sync.py",
        "supervision_work.py", "surface_losses.py", "sync_pair_ledger.py",
        "system_utils.py", "taming_3dgs.py", "taming_stats.py", "tick_torch_profile.py",
        "torch_geometry.py", "trajectory_eval.py", "trajectory_quality.py",
        "trajectory_video.py", "video_encode.py",
        "strategies/__init__.py", "strategies/mcmc/arbiter_owned.py",
        "strategies/mcmc/base.py", "strategies/mcmc/baseline.py",
        "strategies/mcmc/energy.py", "strategies/mcmc/energy_policy.py",
        "strategies/mcmc/legacy.py", "strategies/mcmc/live_wrapped.py",
        "strategies/mcmc/proposal.py", "strategies/mcmc/registry.py",
        "strategies/mcmc_strategy.py", "strategies/surface_mcmc_strategy.py",
    }
)


def test_curated_publication_surface_is_present_and_anonymous():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "EliGSiR" in readme
    assert "Authors" not in readme
    assert "https://github.com/eligsir-project/eligsir-ros2" in readme
    assert "https://github.com/eligsir-project/eligsir-rgbd-sync" in readme
    assert "docs/EVALUATION_PROTOCOL.md" not in readme

    config_dir = ROOT / "configs" / "paper"
    for name in ("tum_fr3.yaml", "tum_fr1_desk.yaml", "replica_room2.yaml", "replica_office0.yaml"):
        path = config_dir / name
        assert path.is_file(), name
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "extends" not in data
        text = path.read_text(encoding="utf-8")
        assert "NOT_YET_PAPER_GRADE" in text
        assert any(key in data for key in ("tum_sequence", "replica_init"))
        assert data["execution_trainer"] == "causal_mapping"
        assert "streaming_replay_buffer" in data
        assert "streaming_global_replay_ratio" in data
        assert "streaming_mapping_resolution_tier" in data
        assert "streaming_mapping_resolution_min_tier" in data
        assert "densification_strategy" in data
        assert "streaming_insert_from_depth" in data
        assert "eval" in data and "streaming_eval_hold" in data
        forbidden_values = (
            "6973737565",       # issue
            "63616d706169676e", # campaign
            "2f686f6d652f",
            "2f6d6e742f",
            "626a6f65726e",
            "73706c61746f6772617068",
        )
        for forbidden in tuple(bytes.fromhex(value).decode() for value in forbidden_values):
            assert forbidden not in text.lower(), (path.name, forbidden)

    for name in ("map_tum_fr3.sh", "map_replica_room2.sh", "evaluate.sh", "render_comparison.sh"):
        path = ROOT / "scripts" / "reproduce" / name
        assert path.is_file(), name
        assert path.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash")


def test_curated_configs_use_real_cli_keys_only():
    defaults = yaml.safe_load((ROOT / "configs" / "profiles" / "defaults.reference.yml").read_text(encoding="utf-8"))
    paper_dir = ROOT / "configs" / "paper"
    for path in paper_dir.glob("*.yaml"):
        if path.name == "README.md":
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for key in data:
            assert key == "extends" or key in defaults or key in {
                "execution_trainer", "model_path", "source_path"
            }, (path.name, key)


def test_public_runtime_utility_boundary_imports():
    """The published package must expose the utilities used by its runtime."""
    from eligsir.utils.run_intent import KNOWN_RUN_INTENTS
    from eligsir.utils.system_utils import mkdir_p

    assert "smoke_dev" in KNOWN_RUN_INTENTS
    assert callable(mkdir_p)


def test_streaming_trainer_imports_bounded_admission_fingerprint_adapter():
    """The retained streaming entry point must resolve its report adapter."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import train_streaming; "
            "from eligsir.utils.admission_fingerprint import AdmissionFingerprint; "
            "assert AdmissionFingerprint.__module__ == 'eligsir.utils.admission_fingerprint'",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_wheel_contains_license_and_third_party_notice_bundle(tmp_path):
    """A built distribution must carry the notices used by its shipped code."""
    wheel_dir = tmp_path / "wheel"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-deps",
            ".",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    wheel = next(wheel_dir.glob("eligsir-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    for notice in ("LICENSE.md", "NOTICE.md", "THIRD_PARTY_NOTICES.md"):
        assert any(name.endswith(f"share/eligsir/{notice}") for name in names), notice


def test_unified_cli_parses_public_subcommands():
    """The single public command accepts each supported stage and --help."""
    for command in ("map", "render", "evaluate"):
        result = subprocess.run(
            [sys.executable, "-m", "eligsir", command, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert command in result.stdout


def test_public_map_render_evaluate_runtime_import_closure():
    """The executable publication stages must import without legacy utils."""
    modules = (
        "train",
        "render",
        "metrics",
        "eligsir.runtime.causal_entry",
        "eligsir.mapping.renderer.gsplat_backend",
    )
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{module}: {result.stderr}"


def test_public_map_configuration_lazy_imports_are_available():
    """Map config and default insertion policy resolve without old root utils."""
    from types import SimpleNamespace

    from eligsir.mapping.integration.integration_planner import (
        insert_gaussians_from_frame,
    )
    from eligsir.runtime.config_boundary import _map_deformation_mode, _tracker_fold_mode
    from eligsir.utils.depth_sparsity_schedule import (
        config_from_values,
        resolve_effective_depth_stride,
    )

    args = SimpleNamespace(
        streaming_map_deformation="off",
        streaming_tracker_discontinuity_fold="off",
    )
    assert _map_deformation_mode(args) == "off"
    assert _tracker_fold_mode(args) == "off"
    schedule = config_from_values("coarse_to_fine", 8, 16, 8, 100)
    assert resolve_effective_depth_stride(schedule, 0, 8) == 16
    assert callable(insert_gaussians_from_frame)


def test_public_runtime_identity_defaults_are_neutral():
    """Public runtime defaults must not identify the private source tree."""
    from eligsir.utils.experiment_origin import DEFAULT_REPOSITORY, ORIGIN_ENV_VAR
    from eligsir.utils.git_metadata import _ENV_BRANCH_VAR, _ENV_COMMIT_VAR

    assert DEFAULT_REPOSITORY == "eligsir-project/eligsir-public"
    assert ORIGIN_ENV_VAR == "ELIGSIR_EXPERIMENT_ORIGIN"
    assert _ENV_BRANCH_VAR == "ELIGSIR_GIT_BRANCH"
    assert _ENV_COMMIT_VAR == "ELIGSIR_GIT_COMMIT"


def test_clean_export_map_render_evaluate_import_closure():
    """A fresh-style export carries the intended public utility closure."""
    intended_untracked = (
        "eligsir/__main__.py",
        "eligsir/cli/main.py",
        "eligsir/mapping/anchored_map_deformation.py",
        "eligsir/mapping/tracker_discontinuity_fold.py",
    )
    utility_files = [
        path.relative_to(ROOT)
        for path in (ROOT / "eligsir/utils").rglob("*.py")
    ]
    actual_utils = {
        path.as_posix().removeprefix("eligsir/utils/")
        for path in utility_files
    }
    assert actual_utils == RETAINED_RUNTIME_UTILS
    assert len(actual_utils) == 98
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
    ).stdout.decode().split("\0")
    with tempfile.TemporaryDirectory(prefix="eligsir-clean-export-") as raw:
        export = Path(raw)
        for name in tracked:
            source = ROOT / name
            if not name or not source.is_file() or name.startswith(".github/workflows/"):
                continue
            target = export / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for name in (*intended_untracked, *utility_files):
            source = ROOT / name
            if not source.is_file():
                continue
            target = export / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        # Exact-refinement/loop-closure checkpoint support is intentionally
        # outside this public map/render/evaluate export. Its utility import is
        # local to an optional resume method, so static module import tests do
        # not need to carry that helper.
        assert not (export / "eligsir/utils/submap_manager.py").exists()

        env = {**__import__("os").environ, "PYTHONPATH": str(export)}
        imports = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import train, render, metrics, "
                "eligsir.runtime.causal_entry, eligsir.mapping.renderer.gsplat_backend; "
                "assert 'eligsir.utils.submap_manager' not in sys.modules",
            ],
            cwd=export,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert imports.returncode == 0, imports.stderr

        for command in ("map", "render", "evaluate"):
            help_result = subprocess.run(
                [sys.executable, "-m", "eligsir", command, "--help"],
                cwd=export,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            assert help_result.returncode == 0, help_result.stderr
            assert command in help_result.stdout

        effective = subprocess.run(
            [sys.executable, "-m", "eligsir", "map", "--print-effective-config"],
            cwd=export,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert effective.returncode == 0, effective.stderr
        assert "effective_config:" in effective.stdout

        parser = subprocess.run(
            [sys.executable, "-m", "eligsir", "map", "-s", "/tmp/no-source", "--run_intent", "smoke_dev"],
            cwd=export,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert parser.returncode != 0
        assert "ModuleNotFoundError: No module named 'eligsir.utils'" not in parser.stderr

        package = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--dry-run", "-e", "."],
            cwd=export,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert package.returncode == 0, package.stderr + package.stdout
