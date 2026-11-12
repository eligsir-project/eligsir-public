"""Tests for EliGSiR producer-side artifact manifest generator."""
from __future__ import annotations

import json
from pathlib import Path

from eligsir.utils.artifact_manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    build_artifact_manifest,
    classify_artifact,
    load_artifact_manifest,
)


def test_classify_artifact_roles(tmp_path: Path) -> None:
    # Scientific Evidence
    kind, role, policy = classify_artifact(tmp_path / "run_summary.json", "run_summary.json")
    assert role == "scientific_evidence"
    assert policy == "IMMEDIATE"

    kind, role, policy = classify_artifact(tmp_path / "report.json", "comparison/iter_600/report.json")
    assert role == "scientific_evidence"
    assert policy == "IMMEDIATE"

    kind, role, policy = classify_artifact(tmp_path / "point_cloud.ply", "point_cloud/iteration_600/point_cloud.ply", is_final_checkpoint=True)
    assert role == "scientific_evidence"
    assert policy == "IMMEDIATE"

    # Reproducibility
    kind, role, policy = classify_artifact(tmp_path / "train.log", "train.log")
    assert role == "reproducibility_artifact"
    assert policy == "IMMEDIATE"

    # Diagnostic
    kind, role, policy = classify_artifact(tmp_path / "test_contact_sheet.png", "comparison/iter_600/test_contact_sheet.png")
    assert role == "diagnostic"
    assert policy == "IMMEDIATE"

    kind, role, policy = classify_artifact(tmp_path / "trajectory.mp4", "comparison/iter_600/trajectory.mp4")
    assert role == "diagnostic"
    assert policy == "IMMEDIATE"

    # PR #1783 follow-up: default-on per-run system-performance trace must be
    # discoverable from the manifest, same diagnostic tier as its
    # psnr_history/replay_sampling siblings.
    for name in (
        "system_performance.jsonl",
        "system_performance.svg",
        "system_performance_summary.json",
    ):
        kind, role, policy = classify_artifact(tmp_path / name, name)
        assert role == "diagnostic"
        assert policy == "IMMEDIATE"

    # Ephemeral / Deferred
    kind, role, policy = classify_artifact(tmp_path / "chkpnt10000.pth", "chkpnt10000.pth")
    assert role == "ephemeral"
    assert policy == "DEFERRED"

    kind, role, policy = classify_artifact(tmp_path / "000000.png", "holdout_gt/000000.png")
    assert role == "ephemeral"
    assert policy == "DEFERRED"


def test_build_and_load_artifact_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_demo"
    run_dir.mkdir()

    # Create dummy artifacts
    (run_dir / "run_summary.json").write_text(json.dumps({"status": "RUN_COMPLETE"}))
    (run_dir / "report.json").write_text(json.dumps({"eval_valid": True}))
    (run_dir / "train.log").write_text("dummy log content")
    (run_dir / "chkpnt1000.pth").write_bytes(b"dummy large checkpoint tensor binary")

    iter_dir = run_dir / "point_cloud" / "iteration_1000"
    iter_dir.mkdir(parents=True)
    (iter_dir / "point_cloud.ply").write_bytes(b"ply binary")

    manifest = build_artifact_manifest(run_dir)
    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["total_files"] == 5
    assert (run_dir / MANIFEST_FILENAME).exists()

    # Check root hash and groups
    assert "root_manifest_sha256" in manifest
    assert "scientific_evidence_manifest_sha256" in manifest["groups"]
    assert "ephemeral_manifest_sha256" in manifest["groups"]

    # Check items
    by_path = {a["relative_path"]: a for a in manifest["artifacts"]}
    assert by_path["run_summary.json"]["hash_status"] == "VERIFIED"
    assert by_path["run_summary.json"]["sha256"] is not None

    assert by_path["point_cloud/iteration_1000/point_cloud.ply"]["hash_status"] == "VERIFIED"
    assert by_path["point_cloud/iteration_1000/point_cloud.ply"]["scientific_role"] == "scientific_evidence"

    assert by_path["chkpnt1000.pth"]["hash_status"] == "DEFERRED"
    assert by_path["chkpnt1000.pth"]["sha256"] is None

    # Test loader
    loaded = load_artifact_manifest(run_dir)
    assert loaded is not None
    assert loaded["root_manifest_sha256"] == manifest["root_manifest_sha256"]
