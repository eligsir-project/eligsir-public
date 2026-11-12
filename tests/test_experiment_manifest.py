"""Tests for eligsir.experiments: manifest, hashing, validation, diff, report."""

from __future__ import annotations

import json

import pytest

from eligsir.experiments.frozen_artifact import audit_changed_fields, diff_config_paths
from eligsir.experiments.hashing import content_hash
from eligsir.experiments.manifest import ArmSpec, ExperimentManifest, HardwareInfo, ValidityInfo
from eligsir.experiments.report import write_experiment_report, write_experiment_report_unchecked
from eligsir.experiments.validation import ManifestValidationError, validate_manifest

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64


def _make_manifest(**overrides) -> ExperimentManifest:
    defaults = dict(
        experiment_id="exp-001",
        hypothesis="does X improve PSNR",
        independent_variable="X",
        frozen_level="A",
        repository_commit="deadbeef",
        source_artifact_hash=_SHA_A,
        pose_hash=_SHA_B,
        holdout_hash=_SHA_C,
        config_hash=_SHA_D,
        seed_policy="fixed:1234",
        hardware=HardwareInfo(gpu="R9700", driver="6.2", torch="2.4.0", backend="rocm"),
        validity=ValidityInfo(mapping_result_valid=True, source_frames_dropped=0, map_reinitializations=0),
        arms=[
            ArmSpec(name="baseline", changed_fields=[]),
            ArmSpec(name="treatment", changed_fields=["training.lr"]),
        ],
    )
    defaults.update(overrides)
    return ExperimentManifest(**defaults)


class TestManifestRoundTrip:
    def test_to_dict_from_dict_round_trip(self):
        manifest = _make_manifest()
        data = manifest.to_dict()
        restored = ExperimentManifest.from_dict(data)
        assert restored.to_dict() == data

    def test_to_dict_is_json_serializable(self):
        manifest = _make_manifest()
        json.dumps(manifest.to_dict())  # must not raise

    def test_optional_hashes_default_none(self):
        manifest = _make_manifest()
        assert manifest.keyframe_hash is None
        assert manifest.initial_checkpoint_hash is None
        assert manifest.mutation_trace_hash is None


class TestHashingDeterminism:
    def test_hash_stable_across_dict_key_order(self):
        a = {"z": 1, "a": {"y": 2, "x": 3}}
        b = {"a": {"x": 3, "y": 2}, "z": 1}
        assert content_hash(a) == content_hash(b)

    def test_hash_differs_for_different_content(self):
        assert content_hash({"a": 1}) != content_hash({"a": 2})

    def test_hash_is_sha256_hex(self):
        digest = content_hash({"a": 1})
        assert len(digest) == 64
        int(digest, 16)  # must be valid hex

    def test_hash_stable_for_nested_lists(self):
        assert content_hash({"a": [1, 2, 3]}) == content_hash({"a": [1, 2, 3]})

    def test_hash_dataclass_matches_equivalent_dict(self):
        manifest = _make_manifest()
        via_dataclass = content_hash(manifest)
        via_dict = content_hash(manifest.to_dict())
        assert via_dataclass == via_dict


class TestValidation:
    def test_valid_manifest_passes(self):
        validate_manifest(_make_manifest())  # must not raise

    def test_invalid_frozen_level_rejected(self):
        manifest = _make_manifest(frozen_level="Z")
        with pytest.raises(ManifestValidationError, match="frozen_level"):
            validate_manifest(manifest)

    def test_missing_required_hash_for_level_a(self):
        manifest = _make_manifest(source_artifact_hash="")
        with pytest.raises(ManifestValidationError, match="missing required"):
            validate_manifest(manifest)

    def test_level_b_requires_keyframe_hash(self):
        manifest = _make_manifest(frozen_level="B", keyframe_hash=None)
        with pytest.raises(ManifestValidationError, match="keyframe_hash"):
            validate_manifest(manifest)

    def test_level_b_passes_with_keyframe_hash(self):
        manifest = _make_manifest(frozen_level="B", keyframe_hash=_SHA_E)
        validate_manifest(manifest)  # must not raise

    def test_level_c_requires_checkpoint_hash(self):
        manifest = _make_manifest(
            frozen_level="C", keyframe_hash=_SHA_E, initial_checkpoint_hash=None
        )
        with pytest.raises(ManifestValidationError, match="initial_checkpoint_hash"):
            validate_manifest(manifest)

    def test_level_d_requires_mutation_trace_hash(self):
        manifest = _make_manifest(
            frozen_level="D",
            keyframe_hash=_SHA_E,
            initial_checkpoint_hash=_SHA_E,
            mutation_trace_hash=None,
        )
        with pytest.raises(ManifestValidationError, match="mutation_trace_hash"):
            validate_manifest(manifest)

    def test_missing_baseline_arm_rejected(self):
        manifest = _make_manifest(
            arms=[ArmSpec(name="treatment", changed_fields=["training.lr"])]
        )
        with pytest.raises(ManifestValidationError, match="baseline"):
            validate_manifest(manifest)

    def test_baseline_with_nonempty_changed_fields_rejected(self):
        manifest = _make_manifest(
            arms=[
                ArmSpec(name="baseline", changed_fields=["oops"]),
                ArmSpec(name="treatment", changed_fields=["training.lr"]),
            ]
        )
        with pytest.raises(ManifestValidationError, match="'baseline'"):
            validate_manifest(manifest)

    def test_undeclared_changed_fields_on_nonbaseline_arm_rejected(self):
        manifest = _make_manifest(
            arms=[
                ArmSpec(name="baseline", changed_fields=[]),
                ArmSpec(name="treatment", changed_fields=[]),
            ]
        )
        with pytest.raises(ManifestValidationError, match="changed_fields"):
            validate_manifest(manifest)

    def test_duplicate_changed_fields_across_arms_rejected(self):
        manifest = _make_manifest(
            arms=[
                ArmSpec(name="baseline", changed_fields=[]),
                ArmSpec(name="treatment_a", changed_fields=["training.lr"]),
                ArmSpec(name="treatment_b", changed_fields=["training.lr"]),
            ]
        )
        with pytest.raises(ManifestValidationError, match="identical"):
            validate_manifest(manifest)

    def test_no_arms_rejected(self):
        manifest = _make_manifest(arms=[])
        with pytest.raises(ManifestValidationError, match="baseline"):
            validate_manifest(manifest)

    def test_arm_configs_under_declared_rejected(self):
        manifest = _make_manifest()
        arm_configs = {
            "baseline": {"training": {"lr": 0.01, "batch_size": 4}},
            "treatment": {"training": {"lr": 0.02, "batch_size": 8}},
        }
        with pytest.raises(ManifestValidationError, match="under-declares"):
            validate_manifest(manifest, arm_configs=arm_configs)

    def test_arm_configs_fully_declared_passes(self):
        manifest = _make_manifest()
        arm_configs = {
            "baseline": {"training": {"lr": 0.01}},
            "treatment": {"training": {"lr": 0.02}},
        }
        validate_manifest(manifest, arm_configs=arm_configs)  # must not raise

    def test_arm_configs_missing_arm_skipped(self):
        manifest = _make_manifest()
        # "treatment" absent from arm_configs entirely -- must be skipped,
        # not treated as an empty-config diff.
        arm_configs = {"baseline": {"training": {"lr": 0.01}}}
        validate_manifest(manifest, arm_configs=arm_configs)  # must not raise

    def test_strict_rejects_unknown_repository_commit(self):
        manifest = _make_manifest(repository_commit="unknown")
        with pytest.raises(ManifestValidationError, match="repository_commit"):
            validate_manifest(manifest, strict=True)

    def test_non_strict_allows_unknown_repository_commit(self):
        manifest = _make_manifest(repository_commit="unknown")
        validate_manifest(manifest)  # must not raise without strict=True


class TestFrozenArtifactDiff:
    def test_flat_diff(self):
        baseline = {"a": 1, "b": 2}
        treatment = {"a": 1, "b": 3}
        assert diff_config_paths(baseline, treatment) == {"b"}

    def test_nested_diff(self):
        baseline = {"training": {"lr": 0.01, "batch_size": 4}}
        treatment = {"training": {"lr": 0.02, "batch_size": 4}}
        assert diff_config_paths(baseline, treatment) == {"training.lr"}

    def test_added_and_removed_keys_are_diffs(self):
        baseline = {"a": 1}
        treatment = {"a": 1, "b": 2}
        assert diff_config_paths(baseline, treatment) == {"b"}
        assert diff_config_paths(treatment, baseline) == {"b"}

    def test_no_diff_when_identical(self):
        config = {"training": {"lr": 0.01}, "top": 5}
        assert diff_config_paths(config, config) == set()

    def test_audit_flags_under_declaration(self):
        baseline = {"training": {"lr": 0.01, "batch_size": 4}}
        treatment = {"training": {"lr": 0.02, "batch_size": 8}}
        audit = audit_changed_fields(baseline, treatment, declared_changed_fields=["training.lr"])
        assert audit.under_declared == {"training.batch_size"}
        assert audit.is_under_declared

    def test_audit_no_violation_when_fully_declared(self):
        baseline = {"training": {"lr": 0.01}}
        treatment = {"training": {"lr": 0.02}}
        audit = audit_changed_fields(baseline, treatment, declared_changed_fields=["training.lr"])
        assert not audit.is_under_declared
        assert audit.under_declared == set()

    def test_audit_flags_over_declaration_separately(self):
        baseline = {"a": 1}
        treatment = {"a": 1}
        audit = audit_changed_fields(baseline, treatment, declared_changed_fields=["a"])
        assert audit.over_declared == {"a"}
        assert not audit.is_under_declared


class TestReport:
    def test_write_experiment_report_round_trips(self, tmp_path):
        manifest = _make_manifest()
        results = {"psnr": 27.3, "n_live": 12000}
        out_path = write_experiment_report(manifest, results, tmp_path / "report.json")
        loaded = json.loads(out_path.read_text())
        assert loaded["results"] == results
        assert loaded["manifest"]["experiment_id"] == "exp-001"
        assert loaded["validity"] == manifest.to_dict()["validity"]

    def test_write_experiment_report_validates_by_default(self, tmp_path):
        # An invalid manifest (baseline arm with non-empty changed_fields)
        # must fail loudly and write nothing.
        manifest = _make_manifest(
            arms=[
                ArmSpec(name="baseline", changed_fields=["oops"]),
                ArmSpec(name="treatment", changed_fields=["training.lr"]),
            ]
        )
        out_path = tmp_path / "report.json"
        with pytest.raises(ManifestValidationError):
            write_experiment_report(manifest, {"psnr": 1.0}, out_path)
        assert not out_path.exists()

    def test_write_experiment_report_rejects_under_declared_arm_config(self, tmp_path):
        manifest = _make_manifest()
        arm_configs = {
            "baseline": {"training": {"lr": 0.01, "batch_size": 4}},
            "treatment": {"training": {"lr": 0.02, "batch_size": 8}},
        }
        out_path = tmp_path / "report.json"
        with pytest.raises(ManifestValidationError, match="under-declares"):
            write_experiment_report(manifest, {"psnr": 1.0}, out_path, arm_configs=arm_configs)
        assert not out_path.exists()

    def test_write_experiment_report_passes_with_correctly_declared_arm_config(self, tmp_path):
        manifest = _make_manifest()
        arm_configs = {
            "baseline": {"training": {"lr": 0.01}},
            "treatment": {"training": {"lr": 0.02}},
        }
        out_path = tmp_path / "report.json"
        result_path = write_experiment_report(
            manifest, {"psnr": 27.0}, out_path, arm_configs=arm_configs
        )
        assert result_path == out_path
        assert out_path.exists()

    def test_write_experiment_report_unchecked_skips_validation(self, tmp_path):
        manifest = _make_manifest(
            arms=[
                ArmSpec(name="baseline", changed_fields=["oops"]),
                ArmSpec(name="treatment", changed_fields=["training.lr"]),
            ]
        )
        out_path = tmp_path / "report.json"
        result_path = write_experiment_report_unchecked(manifest, {"psnr": 1.0}, out_path)
        assert result_path == out_path
        assert out_path.exists()
