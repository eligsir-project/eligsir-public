from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import benchmarks.true_camera_scaling as benchmark_module
import scripts.analyze_true_camera_scaling as analyzer_module
from benchmarks.true_camera_scaling import (
    add_scaling_efficiency,
    deterministic_subset_indices,
    load_camera_manifest,
    profiler_evidence_unavailable,
    select_distinct_cameras,
    summarize_timings,
)
from scripts.analyze_true_camera_scaling import (
    analyze,
    choose_winners,
    cross_platform_crossover,
    fit_affine,
    load_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "benchmarks" / "true_camera_scaling.py"


def _camera(index: int) -> dict:
    return {
        "id": index,
        "img_name": f"frame-{index:04d}",
        "width": 64,
        "height": 48,
        "fx": 50.0,
        "fy": 50.0,
        "cx": 31.5,
        "cy": 23.5,
        "position": [float(index), 0.0, 0.0],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def test_list_plan_has_full_672_point_contract_and_never_pads_cameras(tmp_path: Path):
    manifest = tmp_path / "cameras.json"
    checkpoint = tmp_path / "chkpnt1.pth"
    manifest.write_text(json.dumps([_camera(i) for i in range(4)]))
    checkpoint.write_bytes(b"immutable-checkpoint-placeholder")

    result = subprocess.run(
        [
            sys.executable,
            str(BENCHMARK),
            "--checkpoint",
            str(checkpoint),
            "--camera-manifest",
            str(manifest),
            "--list-plan",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "eligsir.true_camera_scaling.plan.v1"
    assert len(payload["points"]) == 672
    assert {point["requested_n"] for point in payload["points"]} == {
        8192,
        30000,
        100000,
        "mature",
    }
    c4 = [point for point in payload["points"] if point["requested_c"] == 4]
    assert c4 and all(point["realized_c"] == 4 for point in c4)
    assert all(point["plan_status"] == "eligible" for point in c4)
    c8 = [point for point in payload["points"] if point["requested_c"] == 8]
    assert c8 and all(point["realized_c"] == 4 for point in c8)
    assert all(point["plan_status"] == "skip" for point in c8)
    assert all(point["skip_reason"] == "insufficient_distinct_manifest_cameras" for point in c8)


def test_direct_benchmark_entrypoint_can_import_repo_runtime_modules():
    probe = (
        "import runpy, sys; "
        f"sys.path = [{str(BENCHMARK.parent)!r}] + [p for p in sys.path if p != '']; "
        f"runpy.run_path({str(BENCHMARK)!r}, run_name='benchmark_import_probe'); "
        "import render_report"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_manifest_rejects_duplicate_source_identity(tmp_path: Path):
    manifest = tmp_path / "cameras.json"
    duplicate = _camera(0)
    duplicate["id"] = 99
    manifest.write_text(json.dumps([_camera(0), duplicate]))

    with pytest.raises(ValueError, match="duplicate source identities"):
        load_camera_manifest(manifest)


def test_startup_partial_eligibility_skips_without_waiting_or_padding():
    entries = [_camera(i) for i in range(4)]
    initially_eligible = {"img_name:frame-0000", "img_name:frame-0002"}

    partial = select_distinct_cameras(
        entries, 4, eligible_identities=initially_eligible
    )
    assert partial.status == "skip"
    assert partial.reason == "insufficient_distinct_manifest_cameras"
    assert partial.realized_c == 2
    assert partial.identities == (
        "img_name:frame-0000",
        "img_name:frame-0002",
    )

    complete = select_distinct_cameras(entries, 4)
    assert complete.status == "eligible"
    assert complete.realized_c == 4
    assert len(set(complete.identities)) == 4


def test_gaussian_subset_is_seeded_sorted_unique_and_mature_is_identity():
    first = deterministic_subset_indices(total_n=20, requested_n=8, seed=17)
    second = deterministic_subset_indices(total_n=20, requested_n=8, seed=17)
    other = deterministic_subset_indices(total_n=20, requested_n=8, seed=18)

    assert first.tolist() == second.tolist()
    assert first.tolist() != other.tolist()
    assert first.tolist() == sorted(first.tolist())
    assert len(set(first.tolist())) == 8
    assert deterministic_subset_indices(5, "mature", seed=99).tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="exceeds mature population"):
        deterministic_subset_indices(total_n=7, requested_n=8, seed=0)


def test_timing_summary_and_efficiency_use_realized_camera_count():
    timing = summarize_timings([1.0, 2.0, 3.0, 4.0], realized_c=2)
    assert timing["p95_transaction_ms"] == pytest.approx(3.85)
    assert {key: value for key, value in timing.items() if key != "p95_transaction_ms"} == {
        "p50_transaction_ms": 2.5,
        "mean_transaction_ms": 2.5,
        "ms_per_camera": 1.25,
        "cameras_per_s": 800.0,
    }

    verified_source = {
        "verification_status": "verified_clean_git_head",
        "comparison_eligible": True,
        "resolved_git_head": "c" * 40,
        "resolved_git_tree": "e" * 40,
        "entrypoint_sha256": "a" * 64,
        "working_tree_state": "clean",
    }
    rows = [
        {
            "requested_c": 1,
            "realized_c": 1,
            "mean_transaction_ms": 10.0,
            "code_sha": "c" * 40,
            "source_provenance": verified_source,
        },
        {
            "requested_c": 4,
            "realized_c": 4,
            "mean_transaction_ms": 24.0,
            "code_sha": "c" * 40,
            "source_provenance": verified_source,
        },
    ]
    add_scaling_efficiency(rows)
    assert rows[0]["scaling_efficiency"] == 1.0
    assert rows[1]["scaling_efficiency"] == pytest.approx(40.0 / 24.0)

    rows[1]["source_provenance"] = {
        "verification_status": "unverified_git_metadata_unavailable",
        "comparison_eligible": False,
    }
    add_scaling_efficiency(rows)
    assert rows[1]["scaling_efficiency"] is None
    assert rows[1]["scaling_efficiency_reason"] == "source_provenance_not_verified_for_comparison"


def test_unavailable_profiler_breakdown_is_null_with_capability_reasons():
    evidence = profiler_evidence_unavailable("no supported profiler configured")
    assert evidence["capabilities"]["stage_timings"]["available"] is False
    assert evidence["capabilities"]["stage_timings"]["reason"]
    assert evidence["stage_timings_ms"] == {
        "projection": None,
        "spherical_harmonics": None,
        "intersection_sort": None,
        "raster_forward": None,
        "raster_backward": None,
    }
    assert evidence["gfx_active_percent"] is None
    assert evidence["memory_bandwidth_bytes_per_s"] is None
    assert evidence["kernel_launch_count"] is None


def _observed_gpu_identity(label: str):
    identity_type = getattr(benchmark_module, "ObservedGPUIdentity", None)
    assert identity_type is not None, "typed GPU identity is missing"
    gib = 1024**3
    values = {
        "8060S": {
            "device_name": "Radeon 8060S Graphics",
            "architecture": "gfx1151",
            "backend": "ROCm",
            "pci_vendor_id": 0x1002,
            "pci_device_id": 0x1586,
            "pci_domain_id": 0,
            "pci_bus_id": 0xC4,
            "pci_bdf": "0000:c4:00.0",
            "total_memory_bytes": 124 * gib,
        },
        "R9700": {
            "device_name": "AMD Radeon Graphics",
            "architecture": "gfx1201",
            "backend": "ROCm",
            "pci_vendor_id": 0x1002,
            "pci_device_id": 0x7551,
            "pci_domain_id": 0,
            "pci_bus_id": 0x41,
            "pci_bdf": "0000:41:00.0",
            "total_memory_bytes": 32 * gib,
        },
        "RTX3090": {
            "device_name": "NVIDIA GeForce RTX 3090",
            "architecture": "sm_86",
            "backend": "CUDA",
            "pci_vendor_id": 0x10DE,
            "pci_device_id": 0x2204,
            "pci_domain_id": 0,
            "pci_bus_id": 0x21,
            "pci_bdf": "0000:21:00.0",
            "total_memory_bytes": 24 * gib,
        },
    }
    return identity_type(**values[label])


def test_declared_gpu_label_accepts_typed_expected_physical_identities():
    validator = getattr(benchmark_module, "validate_gpu_label", None)
    assert validator is not None, "GPU label validator is missing"
    validator("8060S", _observed_gpu_identity("8060S"))
    validator("R9700", _observed_gpu_identity("R9700"))
    validator("RTX3090", _observed_gpu_identity("RTX3090"))


def test_r9700_generic_name_is_valid_only_with_gfx1201_pci_and_vram_identity():
    validator = benchmark_module.validate_gpu_label
    observed = _observed_gpu_identity("R9700")
    assert observed.device_name == "AMD Radeon Graphics"

    validator("R9700", observed)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("architecture", None, "architecture"),
        ("architecture", "gfx1151", "architecture"),
        ("backend", "CUDA", "backend"),
        ("pci_vendor_id", 0x10DE, "PCI vendor"),
        ("pci_device_id", 0x1586, "PCI device"),
        ("pci_bdf", None, "PCI BDF"),
        ("pci_bdf", "0000:42:00.0", "PCI BDF"),
        ("total_memory_bytes", 24 * 1024**3, "memory"),
    ),
)
def test_r9700_identity_rejects_adversarial_mismatches(field, value, reason):
    from dataclasses import replace

    observed = replace(_observed_gpu_identity("R9700"), **{field: value})
    with pytest.raises(ValueError, match=reason):
        benchmark_module.validate_gpu_label("R9700", observed)


def test_gpu_identity_rejects_wrong_label_and_name_only_spoof():
    with pytest.raises(ValueError, match="architecture"):
        benchmark_module.validate_gpu_label("8060S", _observed_gpu_identity("R9700"))
    with pytest.raises(TypeError, match="typed physical identity"):
        benchmark_module.validate_gpu_label("R9700", "AMD Radeon AI PRO R9700")


def _row(c: int, mean: float, p95: float, memory: int, platform: str = "R9700") -> dict:
    return {
        "status": "ok",
        "hardware": {
            "gpu_label": platform,
            "device_name": f"{platform} physical device",
            "gpu_uuid": f"uuid-{platform}",
            "gpu_arch": "gfx1201",
            "pci_bus_id": "0000:01:00.0",
            "total_memory_bytes": 32_000_000_000,
            "host_machine_id": "sha256:" + "9" * 64,
            "host_machine_id_source": "dmi_product_uuid_sha256",
        },
        "software": {
            "torch": "2.10.0",
            "torch_cuda": None,
            "torch_hip": "7.2",
            "gsplat": "1.5.3",
            "container_image_digest": "sha256:" + "d" * 64,
        },
        "source_provenance": {
            "verification_status": "verified_clean_git_head",
            "comparison_eligible": True,
            "resolved_git_head": "c" * 40,
            "resolved_git_tree": "e" * 40,
            "entrypoint_sha256": "a" * 64,
            "working_tree_state": "clean",
        },
        "code_sha": "c" * 40,
        "requested_n": "30000",
        "realized_n": 30000,
        "render_mode": "RGB",
        "sh_mode": "production",
        "pass_mode": "forward",
        "mode": "packed+sparse",
        "requested_c": c,
        "realized_c": c,
        "mean_transaction_ms": mean,
        "p50_transaction_ms": mean,
        "p95_transaction_ms": p95,
        "peak_memory_allocated_bytes": memory,
        "cameras_per_s": 1000.0 * c / mean,
        "ms_per_camera": mean / c,
        "scaling_efficiency": 1.0 if c == 1 else 0.9,
        "useful_realized_cameras": c,
    }


def test_fit_uses_explicit_low_mid_domain_and_reports_residuals():
    rows = [_row(c, 2.0 + 0.5 * c, 3.0, 10) for c in (1, 2, 4, 8, 16)]
    rows.append(_row(32, 100.0, 101.0, 10))

    fit = fit_affine(rows, max_c=16, weighting="unweighted")

    assert fit["domain"] == {"min_c": 1, "max_c": 16, "n_points": 5}
    assert fit["weighting"] == "unweighted"
    assert fit["t0_ms"] == pytest.approx(2.0)
    assert fit["t_camera_ms"] == pytest.approx(0.5)
    assert fit["r2"] == pytest.approx(1.0)
    assert len(fit["residuals_ms"]) == 5
    assert fit["t0_standard_error_ms"] == pytest.approx(0.0, abs=1e-12)


def test_offline_and_live_winners_are_separate_objectives():
    rows = [
        _row(1, 10.0, 11.0, 80),
        _row(2, 12.0, 14.0, 100),
        _row(4, 16.0, 25.0, 200),
    ]

    winners = choose_winners(
        rows, freshness_budget_ms=20.0, peak_memory_budget_bytes=150
    )

    assert winners["offline"]["chosen_c"] == 4
    assert winners["offline"]["objective"] == "max_realized_cameras_per_s"
    assert winners["live"]["chosen_c"] == 2
    assert winners["live"]["objective"] == "max_useful_realized_supervision_per_s"
    assert winners["live"]["constraints"]["p95_strictly_below_freshness_budget"] is True


def test_cross_platform_crossover_is_only_reported_when_meaningful():
    fit_a = {"status": "ok", "t0_ms": 2.0, "t_camera_ms": 1.0}
    fit_b = {"status": "ok", "t0_ms": 5.0, "t_camera_ms": 0.5}
    meaningful = cross_platform_crossover("A", fit_a, "B", fit_b)
    assert meaningful["c_star"] == pytest.approx(6.0)
    assert meaningful["status"] == "ok"

    parallel = cross_platform_crossover(
        "A", fit_a, "B", {"status": "ok", "t0_ms": 5.0, "t_camera_ms": 1.0}
    )
    assert parallel["c_star"] is None
    assert parallel["reason"] == "parallel_or_numerically_indistinguishable_slopes"


def test_crossover_uses_observed_fit_domain_not_configured_ceiling():
    rows_a = [_row(c, 2.0 + c, 4.0, 10) for c in (1, 4)]
    rows_b = [_row(c, 7.0 + 0.5 * c, 4.0, 10) for c in (1, 4)]

    fit_a = fit_affine(rows_a, max_c=16, weighting="unweighted")
    fit_b = fit_affine(rows_b, max_c=16, weighting="unweighted")
    crossover = cross_platform_crossover("A", fit_a, "B", fit_b)

    assert fit_a["domain"] == {"min_c": 1, "max_c": 4, "n_points": 2}
    assert fit_a["configured_domain"] == {"max_c": 16}
    assert crossover["c_star"] == pytest.approx(10.0)
    assert crossover["within_overlapping_fit_domain"] is False


def test_analyzer_never_combines_mismatched_checkpoint_provenance():
    a = _row(1, 10.0, 11.0, 80, platform="8060S")
    b = _row(1, 8.0, 9.0, 80, platform="RTX3090")
    a["checkpoint"] = {"sha256": "a" * 64}
    b["checkpoint"] = {"sha256": "b" * 64}

    result = analyze(
        [a, b],
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 2
    assert result["cross_platform_crossover"] == []


def test_analyzer_splits_same_label_on_physical_device_or_backend_mismatch():
    rows = [_row(1, 10.0, 11.0, 80), _row(2, 12.0, 13.0, 90)]
    rows[1]["hardware"]["device_name"] = "Different physical device"
    rows.append(_row(4, 16.0, 17.0, 100))
    rows[2]["software"]["gsplat"] = "1.6.0"

    result = analyze(
        rows,
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 3
    assert {
        (item["series"]["device_name"], item["series"]["gsplat"])
        for item in result["series"]
    } == {
        ("R9700 physical device", "1.5.3"),
        ("Different physical device", "1.5.3"),
        ("R9700 physical device", "1.6.0"),
    }


def test_analyzer_groups_matching_observed_device_and_backend_provenance():
    result = analyze(
        [_row(1, 10.0, 11.0, 80), _row(4, 16.0, 17.0, 100)],
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 1
    assert [row["requested_c"] for row in result["series"][0]["rows"]] == [1, 4]


def test_series_identity_splits_on_additional_physical_provenance():
    rows = [_row(1, 10.0, 11.0, 80), _row(2, 12.0, 13.0, 90)]
    rows[0]["hardware"]["multiprocessor_count"] = 120
    rows[1]["hardware"]["multiprocessor_count"] = 121

    result = analyze(
        rows,
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 2
    assert {
        item["series"]["hardware"]["multiprocessor_count"]
        for item in result["series"]
    } == {120, 121}


def test_separate_container_hostnames_group_one_physical_seven_c_sweep(tmp_path: Path):
    paths = []
    camera_counts = (1, 2, 4, 8, 16, 32, 64)
    for c in camera_counts:
        path = tmp_path / f"c-{c}.jsonl"
        rows = []
        for requested_n in (8192, 30000):
            for render_mode in ("RGB", "RGB+ED"):
                for sh_mode in ("0", "production"):
                    for pass_mode in ("forward", "forward+backward"):
                        row = _row(c, 2.0 + c, 3.0 + c, 80)
                        row["requested_n"] = requested_n
                        row["realized_n"] = requested_n
                        row["render_mode"] = render_mode
                        row["sh_mode"] = sh_mode
                        row["pass_mode"] = pass_mode
                        row["hardware"]["hostname"] = f"ephemeral-container-{c}"
                        row["row_provenance"] = {
                            "container_hostname": f"ephemeral-container-{c}"
                        }
                        row["scaling_efficiency"] = -999.0
                        rows.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        paths.append(path)

    result = analyze(
        load_rows(paths),
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=100.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 16
    assert all(len(item["rows"]) == 7 for item in result["series"])
    assert all(
        {row["realized_c"] for row in item["rows"]} == set(camera_counts)
        for item in result["series"]
    )
    assert all(item["fit"]["status"] == "ok" for item in result["series"])
    assert all(item["selectors"]["offline"]["chosen_c"] == 64 for item in result["series"])
    for item in result["series"]:
        assert [row["scaling_efficiency"] for row in item["rows"]] == pytest.approx(
            [c * 3.0 / (2.0 + c) for c in camera_counts]
        )
        assert {
            row["scaling_efficiency_source"] for row in item["rows"]
        } == {"analyzer_recomputed_grouped_series"}
    assert {
        row["hardware"]["hostname"]
        for item in result["series"]
        for row in item["rows"]
    } == {f"ephemeral-container-{c}" for c in camera_counts}

    title = analyzer_module._series_title(result["series"][0]["series"])
    assert len(title) < 160
    assert "ephemeral-container" not in title
    assert "source_provenance" not in title


def test_series_identity_uses_stable_host_machine_id_when_available():
    rows = [_row(1, 10.0, 11.0, 80), _row(2, 12.0, 13.0, 90)]
    for index, row in enumerate(rows):
        row["hardware"]["hostname"] = f"ephemeral-container-{index}"
        row["hardware"]["host_machine_id"] = "sha256:" + str(index) * 64

    result = analyze(
        rows,
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 2
    assert {
        item["series"]["hardware"]["host_machine_id"]
        for item in result["series"]
    } == {"sha256:" + "0" * 64, "sha256:" + "1" * 64}


def test_analyzer_rejects_missing_or_undefended_stable_host_identity():
    missing = _row(1, 10.0, 11.0, 80)
    missing["hardware"].pop("host_machine_id")
    missing["hardware"].pop("host_machine_id_source")
    source_missing = _row(2, 12.0, 13.0, 90)
    source_missing["hardware"].pop("host_machine_id_source")
    unknown_source = _row(4, 16.0, 17.0, 100)
    unknown_source["hardware"]["host_machine_id_source"] = "unreviewed_sha256"

    result = analyze(
        [missing, source_missing, unknown_source],
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert result["series"] == []
    assert [row["reason"] for row in result["excluded_rows"]] == [
        "stable_host_identity_unavailable",
        "stable_host_identity_unavailable",
        "stable_host_identity_unavailable",
    ]


def test_series_identity_splits_on_new_stable_software_fields():
    rows = [_row(1, 10.0, 11.0, 80), _row(2, 12.0, 13.0, 90)]
    rows[0]["software"]["rocm_runtime_build"] = "rocm-7.2-build-a"
    rows[1]["software"]["rocm_runtime_build"] = "rocm-7.2-build-b"

    result = analyze(
        rows,
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 2


@pytest.mark.parametrize("digest", (None, "", "unknown", "sha256:not-a-digest"))
def test_analyzer_rejects_missing_or_invalid_container_image_identity(digest):
    row = _row(1, 10.0, 11.0, 80)
    row["software"]["container_image_digest"] = digest

    result = analyze(
        [row],
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert result["series"] == []
    assert result["excluded_rows"][0]["reason"] == "software_identity_unavailable"


def test_host_machine_identity_is_automatic_hashed_and_fail_closed(tmp_path: Path):
    resolver = getattr(benchmark_module, "resolve_host_machine_identity", None)
    assert callable(resolver), "stable host identity resolver is missing"
    dmi_uuid = tmp_path / "product_uuid"
    dmi_uuid.write_text("12345678-1234-5678-9abc-def012345678\n")
    boot_id = tmp_path / "boot_id"
    boot_id.write_text("abcdef01-2345-6789-abcd-ef0123456789\n")
    missing = tmp_path / "missing"

    explicit = resolver(
        environ={"ELIGSIR_HOST_MACHINE_ID": "physical-host-a"},
        dmi_uuid_path=dmi_uuid,
        boot_id_path=boot_id,
    )
    automatic = resolver(environ={}, dmi_uuid_path=dmi_uuid, boot_id_path=boot_id)
    boot_fallback = resolver(
        environ={}, dmi_uuid_path=missing, boot_id_path=boot_id
    )
    unavailable = resolver(
        environ={}, dmi_uuid_path=missing, boot_id_path=missing
    )

    assert explicit["host_machine_id_source"] == "explicit_env_sha256"
    assert automatic["host_machine_id_source"] == "dmi_product_uuid_sha256"
    assert boot_fallback["host_machine_id_source"] == "linux_boot_id_sha256"
    assert explicit["host_machine_id"].startswith("sha256:")
    assert len(explicit["host_machine_id"]) == 71
    assert automatic["host_machine_id"].startswith("sha256:")
    assert boot_fallback["host_machine_id"].startswith("sha256:")
    assert explicit["host_machine_id"] != automatic["host_machine_id"]
    assert unavailable == {
        "available": False,
        "host_machine_id": None,
        "host_machine_id_source": None,
        "reason": "stable physical host identity unavailable",
    }


def test_no_kernel_execution_common_row_uses_returned_hardware_host_provenance():
    builder = getattr(benchmark_module, "build_execution_common_row", None)
    assert callable(builder), "pure execution common-row builder is missing"
    source = {
        "verification_status": "verified_clean_git_head",
        "comparison_eligible": True,
        "code_sha": "c" * 40,
    }
    hardware = {
        "gpu_label": "R9700",
        "host_machine_id": "sha256:" + "9" * 64,
        "host_machine_id_source": "linux_boot_id_sha256",
        "stable_host_identity_available": True,
        "stable_host_identity_reason": None,
    }

    common = builder(
        source_provenance=source,
        checkpoint={"sha256": "1" * 64},
        checkpoint_metadata={"mature_n": 100000},
        camera_manifest={"sha256": "2" * 64},
        hardware=hardware,
        software={"torch": "2.10.0"},
        rng_seed=7,
        warmups=1,
        repetitions=3,
        created_at_utc="2026-08-28T12:00:00+00:00",
        container_hostname="benchmark-container",
    )

    assert common["hardware"] is hardware
    assert common["row_provenance"] == {
        "container_hostname": "benchmark-container",
        "stable_host_identity_available": True,
        "stable_host_identity_reason": None,
    }
    assert common["checkpoint"] == {
        "sha256": "1" * 64,
        "mature_n": 100000,
    }


def test_no_kernel_execute_path_writes_common_row_without_resolving_host_twice(
    tmp_path: Path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pth"
    manifest = tmp_path / "cameras.json"
    output = tmp_path / "rows.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("[]")
    checkpoint_sha = benchmark_module.sha256_file(checkpoint)
    manifest_sha = benchmark_module.sha256_file(manifest)
    source = {
        "verification_status": "unverified_git_metadata_unavailable",
        "comparison_eligible": False,
        "code_sha": None,
    }
    hardware = {
        "gpu_label": "R9700",
        "host_machine_id": "sha256:" + "9" * 64,
        "host_machine_id_source": "linux_boot_id_sha256",
        "stable_host_identity_available": True,
        "stable_host_identity_reason": None,
    }
    monkeypatch.setattr(
        benchmark_module,
        "_load_checkpoint_state",
        lambda _path: ({}, {"mature_n": 100000, "production_sh_degree": 3}),
    )
    monkeypatch.setattr(
        benchmark_module,
        "load_camera_manifest",
        lambda _path: ([], {"sha256": manifest_sha}),
    )
    monkeypatch.setattr(
        benchmark_module,
        "_hardware_software_provenance",
        lambda _label: (hardware, {"torch": "2.10.0"}),
    )
    monkeypatch.setattr(
        benchmark_module,
        "resolve_host_machine_identity",
        lambda: (_ for _ in ()).throw(
            AssertionError("execute must not resolve host identity a second time")
        ),
    )
    monkeypatch.setattr(benchmark_module, "build_plan", lambda *_args, **_kwargs: [])
    import torch

    monkeypatch.setattr(torch, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    args = benchmark_module.argparse.Namespace(
        gpu_label="R9700",
        output_jsonl=output,
        checkpoint=checkpoint,
        camera_manifest=manifest,
        seed=7,
        warmups=1,
        repetitions=3,
        camera_counts="1",
        gaussian_sizes="8192",
        render_modes="RGB",
        sh_modes="0",
        pass_modes="forward",
    )

    rows = benchmark_module._execute(
        args,
        {
            "source_provenance": source,
            "checkpoint": {"sha256": checkpoint_sha},
        },
    )

    summary = json.loads(output.with_suffix(".run.json").read_text())
    assert rows == []
    assert summary["common_provenance"]["row_provenance"][
        "stable_host_identity_available"
    ] is True
    assert summary["common_provenance"]["hardware"] is not None


def _cross_platform_rows(*, incompatible_render_mode: bool = False) -> list[dict]:
    r9700 = [_row(c, 2.0 + c, 4.0, 80, platform="R9700") for c in (1, 4)]
    rtx3090 = [_row(c, 7.0 + 0.5 * c, 4.0, 80, platform="RTX3090") for c in (1, 4)]
    for row in r9700 + rtx3090:
        row["checkpoint"] = {"sha256": "1" * 64}
        row["camera_manifest"] = {"sha256": "2" * 64}
        row["gaussian_subset"] = {"indices_sha256": "3" * 64}
    for row in r9700:
        row["hardware"].update(
            {"gpu_arch": "gfx1201", "pci_bus_id": "0000:01:00.0"}
        )
        row["software"].update(
            {
                "platform": "Linux-ROCm",
                "kernel_release": "7.1.4",
                "torch": "2.10.0+rocm7.2",
                "torch_cuda": None,
                "torch_hip": "7.2",
                "container_image_digest": "sha256:" + "a" * 64,
            }
        )
    for row in rtx3090:
        row["hardware"].update(
            {"gpu_arch": "sm_86", "pci_bus_id": "0000:02:00.0"}
        )
        row["software"].update(
            {
                "platform": "Linux-CUDA",
                "kernel_release": "6.8.0",
                "torch": "2.10.0+cu128",
                "torch_cuda": "12.8",
                "torch_hip": None,
                "container_image_digest": "sha256:" + "b" * 64,
            }
        )
        if incompatible_render_mode:
            row["render_mode"] = "RGB+ED"
    return r9700 + rtx3090


def test_cross_platform_crossover_pairs_rocm_and_cuda_and_preserves_provenance():
    rows = _cross_platform_rows()
    result = analyze(
        rows,
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 2
    assert len(result["cross_platform_crossover"]) == 1
    crossover = result["cross_platform_crossover"][0]
    assert crossover["status"] == "ok"
    assert crossover["c_star"] == pytest.approx(10.0)
    provenance = {
        crossover["platform_a_provenance"]["series"]["platform"]: crossover[
            "platform_a_provenance"
        ],
        crossover["platform_b_provenance"]["series"]["platform"]: crossover[
            "platform_b_provenance"
        ],
    }
    assert set(provenance["R9700"]) == {
        "series",
        "hardware",
        "software",
        "source_provenance",
    }
    assert provenance["R9700"]["hardware"] == rows[0]["hardware"]
    assert provenance["RTX3090"]["hardware"] == rows[2]["hardware"]
    assert provenance["R9700"]["software"]["torch_hip"] == "7.2"
    assert provenance["R9700"]["software"]["torch_cuda"] is None
    assert provenance["RTX3090"]["software"]["torch_cuda"] == "12.8"
    assert provenance["RTX3090"]["software"]["torch_hip"] is None
    assert provenance["R9700"]["software"]["container_image_digest"] != provenance[
        "RTX3090"
    ]["software"][
        "container_image_digest"
    ]
    assert provenance["R9700"]["source_provenance"] == rows[0][
        "source_provenance"
    ]


def test_cross_platform_crossover_rejects_incompatible_workload():
    result = analyze(
        _cross_platform_rows(incompatible_render_mode=True),
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert len(result["series"]) == 2
    assert result["cross_platform_crossover"] == []


def test_analyzer_excludes_unverified_declared_code_sha_from_comparisons():
    row = _row(1, 10.0, 11.0, 80)
    row["source_provenance"] = {
        "verification_status": "unverified_git_metadata_unavailable",
        "comparison_eligible": False,
        "declared_code_sha": "f" * 40,
        "resolved_git_head": None,
        "working_tree_state": "unknown",
    }
    row["code_sha"] = None

    result = analyze(
        [row],
        fit_max_c=16,
        weighting="unweighted",
        freshness_budget_ms=20.0,
        peak_memory_budget_bytes=1000,
    )

    assert result["series"] == []
    assert result["excluded_rows"] == [
        {
            "row_index": 0,
            "reason": "source_provenance_not_verified_for_comparison",
            "verification_status": "unverified_git_metadata_unavailable",
        }
    ]


def _init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "tracked.txt").write_text("frozen\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git", "-C", str(path), "-c", "user.name=Benchmark Test",
            "-c", "user.email=benchmark@example.invalid", "commit", "-qm", "initial",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_source_provenance_verifies_declared_sha_against_clean_git_head(tmp_path: Path):
    expected_sha = _init_git_repo(tmp_path)
    resolver = getattr(benchmark_module, "resolve_source_provenance", None)
    assert callable(resolver), "source provenance resolver is missing"

    provenance = resolver(tmp_path, expected_sha)

    assert provenance["verification_status"] == "verified_clean_git_head"
    assert provenance["comparison_eligible"] is True
    assert provenance["code_sha"] == expected_sha
    assert provenance["resolved_git_head"] == expected_sha
    assert len(provenance["resolved_git_tree"]) == 40
    assert len(provenance["entrypoint_sha256"]) == 64
    assert provenance["working_tree_state"] == "clean"

    with pytest.raises(ValueError, match="declared code SHA .* does not match"):
        resolver(tmp_path, "f" * 40)


def test_source_provenance_marks_dirty_and_unresolvable_trees_ineligible(tmp_path: Path):
    resolver = getattr(benchmark_module, "resolve_source_provenance", None)
    assert callable(resolver), "source provenance resolver is missing"
    repo = tmp_path / "repo"
    repo.mkdir()
    expected_sha = _init_git_repo(repo)
    (repo / "tracked.txt").write_text("dirty\n")

    dirty = resolver(repo, expected_sha)
    unavailable = resolver(tmp_path / "not-a-repo", expected_sha)

    assert dirty["verification_status"] == "dirty_git_tree"
    assert dirty["comparison_eligible"] is False
    assert dirty["code_sha"] is None
    assert dirty["working_tree_state"] == "dirty"
    assert len(dirty["working_tree_fingerprint_sha256"]) == 64
    assert unavailable["verification_status"] == "unverified_git_metadata_unavailable"
    assert unavailable["comparison_eligible"] is False
    assert unavailable["declared_code_sha"] == expected_sha
    assert unavailable["code_sha"] is None


def test_analyzer_cli_writes_summary_and_four_panel_plot(tmp_path: Path):
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "analysis.json"
    plot_dir = tmp_path / "plots"
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in (
            _row(1, 10.0, 11.0, 80),
            _row(2, 12.0, 14.0, 100),
            _row(4, 16.0, 25.0, 200),
        ))
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "analyze_true_camera_scaling.py"),
            str(input_path),
            "--output",
            str(output_path),
            "--plots-dir",
            str(plot_dir),
            "--freshness-budget-ms",
            "20",
            "--peak-memory-budget-bytes",
            "150",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    analysis = json.loads(output_path.read_text())
    assert analysis["schema_version"] == "eligsir.true_camera_scaling.analysis.v1"
    assert len(analysis["series"]) == 1
    assert analysis["series"][0]["selectors"]["offline"]["chosen_c"] == 4
    assert analysis["series"][0]["selectors"]["live"]["chosen_c"] == 2
    assert len(analysis["plots"]) == 1
    assert Path(analysis["plots"][0]).is_file()
