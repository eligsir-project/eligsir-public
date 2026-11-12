from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Same gap test_gaussian_op_trace.py's requires_registry documents: the
# `cpu-ci` image COPYs source trees but not `docs/`, so this checked-in
# architecture note is genuinely absent on that surface. Skip there rather
# than failing on a missing-docs environment difference; it runs on
# `host` / `training-nogpu` where docs/ is present.
_COMPOSITION_QUALIFICATION_DOC = (
    "docs/architecture/2026-09-01-ready-path-prepared-composition-qualification.md"
)
requires_composition_qualification_doc = pytest.mark.skipif(
    not os.path.exists(_COMPOSITION_QUALIFICATION_DOC),
    reason=f"{_COMPOSITION_QUALIFICATION_DOC} is not present on this test surface "
           "(the cpu-ci image does not COPY docs/); verified on host / "
           "training-nogpu instead",
)

from eligsir.runtime import ready_path_qualification_launcher as launcher
from eligsir.runtime.ready_path_qualification_launcher import (
    rocm_qualification_command,
    run_cpu_ready_path_qualification,
)


def test_cpu_launcher_composes_prepared_application_and_retains_b4_evidence(tmp_path) -> None:
    artifact = run_cpu_ready_path_qualification(
        output_dir=tmp_path, source_commit="a" * 40
    )

    payload = json.loads(artifact.read_text())
    assert payload["claim"] == "gate:merge"
    assert payload["execution_backend"] == "cpu-fallback"
    assert len(payload["sources"]) == 4
    assert payload["bootstrap"]["selected_source_ids"] == payload["sources"]
    assert payload["b4"]["selected_source_ids"] == payload["sources"]
    assert payload["b4"]["ready_camera_count"] == 4
    assert payload["b4"]["qualified"] is True
    assert payload["b4"]["operations"] == {
        "cpu_materializations": 0,
        "cold_decodes": 0,
        "wait_calls": 0,
        "future_result_calls": 0,
        "h2d_entrypoint_calls": 0,
        "d2h_entrypoint_calls": 0,
        "unattributed_transfer_calls": 0,
        "mixed_transfer_owner_calls": 0,
    }
    assert payload["transfer"]["baseline"]["h2d_ops"] == 4
    assert payload["transfer"]["after"] == payload["transfer"]["baseline"]
    assert payload["lineage"]
    assert payload["provenance"]["commit"] == "a" * 40


@pytest.mark.parametrize("source_commit", (None, "unknown", "abc123"))
def test_cpu_launcher_refuses_missing_or_nonimmutable_source_commit(tmp_path, source_commit) -> None:
    with pytest.raises(ValueError, match="source commit"):
        run_cpu_ready_path_qualification(
            output_dir=tmp_path, source_commit=source_commit
        )


def test_rocm_command_matches_documented_rocm_container_contract() -> None:
    assert rocm_qualification_command(
        source_path="/data/tum/fr3", output_dir="output/ready-path-rocm",
        source_commit="b" * 40,
    ) == (
        "docker compose run --rm -T --privileged --device /dev/kfd --device /dev/dri "
        "--group-add keep-groups --security-opt label=disable "
        "--security-opt seccomp=unconfined --ipc host "
        "-e HSA_XNACK=1 -e HSA_ENABLE_SDMA=0 -e ELIGSIR_SOURCE_COMMIT='" + "b" * 40 + "' train python -m "
        "eligsir.runtime.ready_path_qualification_launcher "
        "--rocm-qualification --source '/data/tum/fr3' "
        "--output-dir 'output/ready-path-rocm'"
    )


@requires_composition_qualification_doc
def test_documented_compose_commands_pass_the_source_sha_as_an_explicit_cli_value() -> None:
    document = Path(_COMPOSITION_QUALIFICATION_DOC).read_text()

    assert document.count('--source-commit "$(git rev-parse HEAD)"') == 2
    assert "-e ELIGSIR_SOURCE_COMMIT" not in document


@pytest.mark.parametrize("failure", ("write", "replace"))
def test_atomic_artifact_writer_never_leaves_partial_final_or_temp(
    tmp_path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    artifact = tmp_path / "ready_path_cpu_integration.json"
    if failure == "write":
        monkeypatch.setattr(
            launcher.json,
            "dump",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("ENOSPC")),
        )
    else:
        monkeypatch.setattr(
            launcher.os,
            "replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("interrupted")),
        )

    with pytest.raises(OSError):
        launcher._write_artifact_atomically(artifact, {"claim": "gate:merge"})

    assert not artifact.exists()
    assert list(tmp_path.iterdir()) == []


def test_cpu_launcher_refuses_to_overwrite_retained_final_artifact(tmp_path) -> None:
    run_cpu_ready_path_qualification(output_dir=tmp_path, source_commit="a" * 40)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_cpu_ready_path_qualification(output_dir=tmp_path, source_commit="a" * 40)
