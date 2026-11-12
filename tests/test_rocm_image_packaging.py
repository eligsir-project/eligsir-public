from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hipblaslt_pruning_retains_architecture_neutral_runtime_metadata():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert (
        'find /opt/rocm/lib/hipblaslt/library \\'
        '\n         -type f -name "*gfx*" ! -name "*${ROCM_ARCH}*" -delete'
        in dockerfile
    ), "hipBLASLt pruning must delete only non-target architecture files"
    for required_file in (
        "TensileLiteLibrary_lazy_Mapping.dat",
        "hipblasltExtOpLibrary.dat",
        "hipblasltTransform.hsaco",
    ):
        assert f'test -f "/opt/rocm/lib/hipblaslt/library/{required_file}"' in dockerfile


def test_prebuilt_rocm_base_restores_common_hipblaslt_metadata():
    dockerfile = (ROOT / "Dockerfile").read_text()

    for required_file in (
        "TensileLiteLibrary_lazy_Mapping.dat",
        "hipblasltExtOpLibrary.dat",
        "hipblasltTransform.hsaco",
    ):
        source = f"/opt/rocm/lib/hipblaslt/library/{required_file}"
        assert f"COPY --from=gsplat-builder {source} {source}" in dockerfile


def test_prebuilt_rocm_base_restores_architecture_specific_lazy_tensile_database():
    dockerfile = (ROOT / "Dockerfile").read_text()

    source = "/opt/rocm/lib/hipblaslt/library/TensileLibrary_lazy_${ROCM_ARCH}.dat"
    assert f"COPY --from=gsplat-builder {source} {source}" in dockerfile
