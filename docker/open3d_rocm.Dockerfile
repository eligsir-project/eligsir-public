# syntax=docker/dockerfile:1.7
#
# Reproducible Open3D ROCm/HIP Tensor build for EliGSiR.
#
# Upstream support is currently carried by isl-org/Open3D#7509. Until that PR
# merges, NEVER build a moving branch tip: OPEN3D_ROCM_COMMIT is part of the
# scientific/runtime provenance of the frontend benchmark.

ARG ROCM_VERSION=7.2.2
ARG ROCM_ARCH=gfx1151
ARG PYTORCH_VERSION=2.10.0
ARG OPEN3D_ROCM_COMMIT=50bb2505be991392f7cdfd040802db7bbf6ef2a8
ARG OPEN3D_ROCM_PR=7509
ARG OPEN3D_BUILD_JOBS=2
ARG OPEN3D_SOURCE=https://github.com/isl-org/Open3D.git
ARG ROCM_BUILD_IMAGE=rocm/pytorch:rocm${ROCM_VERSION}_ubuntu24.04_py3.12_pytorch_release_${PYTORCH_VERSION}
# FROM arguments must be declared globally (before the first FROM). Keep this
# caller-supplied base; paper-critical builds must use an immutable digest.
# Keeping this argument unset avoids baking a private registry into the export.
# For the overlay target, pass it explicitly, for example:
#   docker build -f docker/open3d_rocm.Dockerfile --target eligsir-open3d-rocm \
#     --build-arg ELIGSIR_BASE=registry.example/eligsir@sha256:<digest> .
# The `open3d-rocm-wheel-artifact` target does not require ELIGSIR_BASE.
ARG ELIGSIR_BASE

FROM ${ROCM_BUILD_IMAGE} AS open3d-rocm-builder

ARG ROCM_ARCH
ARG OPEN3D_ROCM_COMMIT
ARG OPEN3D_ROCM_PR
ARG OPEN3D_BUILD_JOBS
ARG OPEN3D_SOURCE

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_PARALLEL_LEVEL=${OPEN3D_BUILD_JOBS} \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    CMAKE_PREFIX_PATH=/opt/rocm

# Bootstrap only what is needed to fetch/configure Open3D. The full C++/Python
# dependency set is installed by Open3D's own pinned Ubuntu helper below so our
# HIP wheel builder does not drift from upstream's supported build environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --filter=blob:none "${OPEN3D_SOURCE}" Open3D \
    && cd Open3D \
    && git fetch --no-tags origin "pull/${OPEN3D_ROCM_PR}/head" \
    && test "$(git rev-parse FETCH_HEAD)" = "${OPEN3D_ROCM_COMMIT}" \
    && git checkout --detach "${OPEN3D_ROCM_COMMIT}" \
    && git submodule update --init --recursive

WORKDIR /tmp/Open3D
# EliGSiR patch on top of the pinned upstream SHA -- NOT an upstream change.
#
# Upstream #7509 was validated with BUILD_PYTHON_MODULE=OFF, so this path was
# never exercised there. With USE_HIP=ON, CMake gives the `pybind` module target
# HIP as its linker language, so it is linked by /opt/rocm/llvm/bin/clang++ in
# HIP offload mode. That driver silently discards every host object: the link
# exits 0 with no diagnostic and emits a ~4.6kB extension module exporting no
# symbols at all -- not even PyInit_pybind. `cmake --build --target pip-package`
# then packages the stub into a superficially valid ~451kB wheel that fails at
# `import open3d`.
#
# Measured on gfx1151 against this exact SHA: the same 81 objects and 76
# archives relinked with the CXX host driver produce a 174,720,536-byte module
# that correctly exports PyInit_pybind. LTO (`-flto=auto`), the version script,
# --offload-arch on the link line and the post-build strip were each ruled out
# individually; the linker driver is the sole cause.
RUN printf '%s\n' \
    '' \
    '# --- EliGSiR: force the CXX host driver for the Python module link ---' \
    '# The ROCm clang++ HIP offload driver drops all host objects here and emits' \
    '# a symbol-less stub without failing the build. Device code from libOpen3D.a' \
    '# is embedded as HIP fat binaries and is carried through by the host link.' \
    'if (USE_HIP)' \
    '    set_target_properties(pybind PROPERTIES LINKER_LANGUAGE CXX)' \
    'endif()' \
    >> cpp/pybind/CMakeLists.txt

# The dependency helper itself is pinned by OPEN3D_ROCM_COMMIT. GUI is disabled
# below, so avoid the Filament build-from-source dependency set.
RUN SUDO=env bash util/install_deps_ubuntu.sh assume-yes no-filament-deps \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir -r python/requirements_build.txt

# The wheel-existence checks in this step all pass on a wheel whose extension
# module is an empty stub (see the linker-language patch above), which is how a
# 451kB unimportable wheel previously reached CI as a green 48-minute build. The
# final assertions therefore inspect the compiled payload itself: the installed
# extension must exist, must export PyInit_pybind, and must be far larger than
# any conceivable stub.
#
# #7509 reuses Open3D's BUILD_CUDA_MODULE switch/API spelling, but retags the
# GPU translation units as HIP and swaps CUDA runtime/math dependencies for
# HIP/hipBLAS/hipSOLVER/hipSPARSE. Explicit CMAKE_HIP_ARCHITECTURES makes the
# image build reproducible even on a builder with no AMD GPU attached.
RUN cmake -S . -B build -G Ninja \
    -DUSE_HIP=ON \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES="${ROCM_ARCH}" \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_BUILD_TYPE=Release \
    -DDEVELOPER_BUILD=ON \
    -DBUILD_PYTHON_MODULE=ON \
    -DBUILD_PYTORCH_OPS=OFF \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_OPEN3D_ML=OFF \
    -DBUILD_GUI=OFF \
    -DBUILD_WEBRTC=OFF \
    -DBUILD_JUPYTER_EXTENSION=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_ISPC_MODULE=OFF \
    -DBUILD_LIBREALSENSE=OFF \
    -DBUILD_AZURE_KINECT=OFF \
    -DWITH_IPP=OFF \
    && cmake --build build --target pip-package --parallel "${OPEN3D_BUILD_JOBS}" \
    && mkdir -p /tmp/open3d-wheel /tmp/open3d-wheel-check \
    && set -- build/lib/python_package/pip_package/open3d*.whl \
    && test "$#" -eq 1 \
    && test -f "$1" \
    && cp -v "$1" /tmp/open3d-wheel/ \
    && python3 -m pip install --no-cache-dir --no-deps --target /tmp/open3d-wheel-check "$1" \
    && test "$(find /tmp/open3d-wheel -maxdepth 1 -type f -name 'open3d*.whl' | wc -l)" -eq 1 \
    && ext="$(find /tmp/open3d-wheel-check/open3d -name 'pybind*.so' | head -n 1)" \
    && test -n "$ext" \
    && echo "extension: $ext ($(wc -c < "$ext") bytes)" \
    && nm -D --defined-only "$ext" | grep -q ' PyInit_pybind$' \
    && test "$(wc -c < "$ext")" -gt 10000000

RUN printf '%s\n' \
      "upstream_repo=isl-org/Open3D" \
      "upstream_pr=${OPEN3D_ROCM_PR}" \
      "upstream_sha=${OPEN3D_ROCM_COMMIT}" \
      "build_jobs=${OPEN3D_BUILD_JOBS}" \
      "rocm_arch=${ROCM_ARCH}" \
      > /tmp/open3d-wheel/BUILD_PROVENANCE.txt

# Export only the pinned wheel and its provenance. This makes the PR build gate
# retain the exact artifact that later AMD-hardware qualification must consume.
FROM scratch AS open3d-rocm-wheel-artifact
COPY --from=open3d-rocm-builder /tmp/open3d-wheel /

# Default overlay target. Override ELIGSIR_BASE with the exact image digest
# being benchmarked; do not use a mutable tag for paper-critical runs.
FROM ${ELIGSIR_BASE} AS eligsir-open3d-rocm

ARG ROCM_ARCH
ARG OPEN3D_ROCM_COMMIT
ARG OPEN3D_ROCM_PR

COPY --from=open3d-rocm-builder /tmp/open3d-wheel /tmp/open3d-wheel
COPY scripts/verify_open3d_rocm.py /usr/local/bin/verify-open3d-rocm
RUN python -m pip uninstall -y open3d || true \
    && python -m pip install --no-cache-dir --no-deps /tmp/open3d-wheel/open3d*.whl \
    && mkdir -p /opt/eligsir/open3d-rocm \
    && cp /tmp/open3d-wheel/BUILD_PROVENANCE.txt /opt/eligsir/open3d-rocm/BUILD_PROVENANCE.txt \
    && chmod 0755 /usr/local/bin/verify-open3d-rocm \
    && rm -rf /tmp/open3d-wheel

ENV ELIGSIR_OPEN3D_ACCELERATOR=rocm \
    ELIGSIR_OPEN3D_UPSTREAM_PR=${OPEN3D_ROCM_PR} \
    ELIGSIR_OPEN3D_UPSTREAM_SHA=${OPEN3D_ROCM_COMMIT} \
    ELIGSIR_OPEN3D_ROCM_ARCH=${ROCM_ARCH}

LABEL org.opencontainers.image.open3d.upstream="isl-org/Open3D" \
      org.opencontainers.image.open3d.pr="${OPEN3D_ROCM_PR}" \
      org.opencontainers.image.open3d.revision="${OPEN3D_ROCM_COMMIT}" \
      org.opencontainers.image.open3d.accelerator="rocm" \
      org.opencontainers.image.open3d.rocm_arch="${ROCM_ARCH}"

# A real accelerator smoke test is intentionally NOT run during docker build:
# image builders generally do not expose /dev/kfd. Run `verify-open3d-rocm`
# inside the resulting image on an AMD GPU before qualifying a benchmark run.
