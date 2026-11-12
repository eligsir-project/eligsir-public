# syntax=docker/dockerfile:1.7

ARG ROCM_VERSION=7.2.2
ARG PYTORCH_VERSION=2.10.0
ARG TORCHVISION_VERSION=0.25.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG TRITON_VERSION=3.6.0
ARG ROCM_ARCH=gfx1151
ARG BASE_ROCM=rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0@sha256:73136312e61ea15af04e89a9df3830bd3fa91c14aad755cde0e426e707f493ff

# Pre-built, per-architecture snapshot of the heavy ROCm/Python runtime.
# Rebuild it with a caller-supplied, immutable base image. The former default
# named a private registry image and was not usable from a clean publication
# clone. Pass BASE_ROCM_PYTHON explicitly (prefer a digest-pinned image).
# The ROCm gsplat source defaults to the neutral publication fork and exact
# audited commit below. Both values remain caller-overridable for controlled
# qualification builds:
#   docker build -f Dockerfile --target eligsir \
#     --build-arg BASE_ROCM_PYTHON=registry.example/eligsir-rocm@sha256:<digest> \
#     --build-arg GSPLAT_REPO=https://github.com/eligsir-project/gsplat-rocm.git \
#     --build-arg GSPLAT_COMMIT=c6b363ab9239c958a4ac6619f725119dc8e5706b .
ARG BASE_ROCM_PYTHON

# =============================================================================
# STAGE 1: native extension builder
# =============================================================================
FROM ${BASE_ROCM} AS gsplat-builder

ARG ROCM_ARCH
ARG FUSED_SSIM_COMMIT=a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38
ENV HSA_XNACK=1 \
    HSA_ENABLE_SDMA=0 \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build libglm-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# The neutral publication fork carries the ROCm build delta over the upstream
# v1.5.3 base. Keep both values overridable for controlled qualification while
# making the public-safe publication reproduction contract deterministic by
# default.
ARG GSPLAT_REPO=https://github.com/eligsir-project/gsplat-rocm.git
ARG GSPLAT_COMMIT=c6b363ab9239c958a4ac6619f725119dc8e5706b
RUN --mount=type=secret,id=github_token,required=false \
    set -eu; \
    if test -s /run/secrets/github_token; then \
      GITHUB_TOKEN_VALUE="$(tr -d '\\r\\n' < /run/secrets/github_token)"; \
      export GIT_CONFIG_COUNT=1; \
      export GIT_CONFIG_KEY_0=http.extraheader; \
      export GIT_CONFIG_VALUE_0="Authorization: Basic $(printf 'x-access-token:%s' "${GITHUB_TOKEN_VALUE}" | base64 | tr -d '\\n')"; \
      unset GITHUB_TOKEN_VALUE; \
    fi; \
    git clone --filter=blob:none --no-checkout "${GSPLAT_REPO}" /tmp/gsplat \
    && cd /tmp/gsplat \
    && git fetch --depth 1 origin "${GSPLAT_COMMIT}" \
    && git checkout --detach "${GSPLAT_COMMIT}" \
    && test "$(git rev-parse HEAD)" = "${GSPLAT_COMMIT}" \
    && git submodule update --init --recursive \
    && MAX_JOBS="$(nproc)" python3 setup.py bdist_wheel

RUN git clone --depth 1 https://github.com/rahul-goel/fused-ssim.git /tmp/fused-ssim \
    && cd /tmp/fused-ssim \
    && git fetch --depth 1 origin "${FUSED_SSIM_COMMIT}" \
    && git checkout --detach "${FUSED_SSIM_COMMIT}" \
    && CUDA_HOME=/opt/rocm ROCM_HOME=/opt/rocm MAX_JOBS="$(nproc)" \
       pip wheel . --no-build-isolation --no-deps -w /tmp/fused-ssim/dist

COPY submodules/simple-knn /tmp/simple-knn
RUN cd /tmp/simple-knn \
    && CUDA_HOME=/opt/rocm ROCM_HOME=/opt/rocm MAX_JOBS="$(nproc)" \
       pip wheel . --no-build-isolation --no-deps -w /tmp/simple-knn/dist

# =============================================================================
# STAGE 2: pre-built ROCm/Python runtime
# =============================================================================
FROM ${BASE_ROCM_PYTHON} AS rocm-python-base

ARG ROCM_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TRITON_VERSION
ARG ROCM_ARCH

ENV DEBIAN_FRONTEND=noninteractive \
    HSA_XNACK=1 \
    HSA_ENABLE_SDMA=0 \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/venv \
    RERUN_SERVE_WEB=false

# The per-arch base snapshots predating this fix pruned hipBLASLt's common
# mapping and transform metadata because those filenames contain no gfx token.
# Restore the small, ROCm-version-matched files from the full native-extension
# builder so regular image builds are repaired without waiting for a new base
# snapshot publication.
COPY --from=gsplat-builder /opt/rocm/lib/hipblaslt/library/TensileLiteLibrary_lazy_Mapping.dat /opt/rocm/lib/hipblaslt/library/TensileLiteLibrary_lazy_Mapping.dat
COPY --from=gsplat-builder /opt/rocm/lib/hipblaslt/library/hipblasltExtOpLibrary.dat /opt/rocm/lib/hipblaslt/library/hipblasltExtOpLibrary.dat
COPY --from=gsplat-builder /opt/rocm/lib/hipblaslt/library/hipblasltTransform.hsaco /opt/rocm/lib/hipblaslt/library/hipblasltTransform.hsaco
# The architecture-specific lazy database is consulted by hipBLASLt GEMM
# heuristics before the first mapping frame.  Snapshot pruning can leave an
# otherwise runnable image without it, which fails as HIPBLAS_STATUS_INVALID_VALUE
# on gfx1201.  Copy it from the same ROCm-version builder and make its absence a
# build failure rather than a first-frame runtime failure.
COPY --from=gsplat-builder /opt/rocm/lib/hipblaslt/library/TensileLibrary_lazy_${ROCM_ARCH}.dat /opt/rocm/lib/hipblaslt/library/TensileLibrary_lazy_${ROCM_ARCH}.dat

# =============================================================================
# STAGE 2b: scratch builder for the heavy base snapshot
# =============================================================================
FROM ubuntu:24.04@sha256:224a1869083a311ef3f13648a154ba79832fbef6364d31493642ca03082da254 AS rocm-python-base-scratch

ARG ROCM_VERSION
ARG PYTORCH_VERSION
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG TRITON_VERSION
ARG ROCM_ARCH

ENV DEBIAN_FRONTEND=noninteractive \
    HSA_XNACK=1 \
    HSA_ENABLE_SDMA=0 \
    PYTORCH_ROCM_ARCH=${ROCM_ARCH} \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    VIRTUAL_ENV=/opt/venv \
    RERUN_SERVE_WEB=false

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg software-properties-common \
    python3.12 python3.12-venv python3-pip python-is-python3 \
    ffmpeg mesa-va-drivers libglib2.0-0 libnuma1 libelf1 libzstd1 zlib1g \
    liblzma5 libdrm-amdgpu1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p --mode=0755 /etc/apt/keyrings \
    && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg \
    && printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/%s noble main\n' "${ROCM_VERSION}" > /etc/apt/sources.list.d/rocm.list \
    && printf '%s\n' 'Package: *' 'Pin: release o=repo.radeon.com' 'Pin-Priority: 600' > /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get install -y --no-install-recommends \
    rocm-core comgr hsa-rocr hip-runtime-amd rocprofiler-register rocm-smi-lib \
    amd-smi-lib rocminfo rocrand roctracer

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get install -y --no-install-recommends hipblaslt miopen-hip

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get install -y --no-install-recommends hipblas rocblas rocfft rocsolver rocsparse

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get install -y --no-install-recommends \
    hiprand hipfft hipsolver hipsparse rocalution rccl hipsparselt \
    && find /opt/rocm/lib/hipblaslt/library \
         -type f -name "*gfx*" ! -name "*${ROCM_ARCH}*" -delete 2>/dev/null || true \
    && test -f "/opt/rocm/lib/hipblaslt/library/TensileLiteLibrary_lazy_Mapping.dat" \
    && test -f "/opt/rocm/lib/hipblaslt/library/hipblasltExtOpLibrary.dat" \
    && test -f "/opt/rocm/lib/hipblaslt/library/hipblasltTransform.hsaco" \
    && find /opt/rocm/lib/rocblas/library \
         -type f ! -name "*${ROCM_ARCH}*" -delete 2>/dev/null || true \
    && find /opt/rocm/lib/hipsparselt/library \
         -type f ! -name "*${ROCM_ARCH}*" -delete 2>/dev/null || true \
    && find /opt/rocm/share/miopen/db \
         -type f ! -name "*${ROCM_ARCH}*" -delete 2>/dev/null || true \
    && rm -f /opt/rocm/lib/rocfft/rocfft_kernel_cache.db

RUN python -m venv /opt/venv --system-site-packages \
    && pip install --no-cache-dir uv

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
    --find-links "https://repo.radeon.com/rocm/manylinux/rocm-rel-${ROCM_VERSION}/" \
    --prefer-binary \
    "torch==${PYTORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    "triton==${TRITON_VERSION}"

# =============================================================================
# STAGE 3: shared EliGSiR dependencies
# =============================================================================
FROM rocm-python-base AS eligsir-deps

ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=${GIT_COMMIT}
RUN echo "${GIT_COMMIT}" > /tmp/.build_commit

# git is a core runtime dependency for run provenance; it is not supplied by a
# ROS overlay anymore.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG INSTALL_RERUN=true
ARG INSTALL_VISER=true

RUN python -m venv /opt/venv --system-site-packages \
    && pip install --no-cache-dir uv

# Pin the ROCm torch family globally so later dependency installation cannot
# replace it with CUDA wheels.
RUN python - <<'PY' > /etc/pip-torch-constraint.txt
import importlib.metadata as md
for name in ("torch", "torchvision", "torchaudio"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PY
ENV PIP_CONSTRAINT=/etc/pip-torch-constraint.txt \
    UV_CONSTRAINT=/etc/pip-torch-constraint.txt

RUN uv pip install --no-cache \
    "numpy<2" \
    plyfile \
    tqdm \
    opencv-python-headless \
    rich \
    jaxtyping \
    rosbags \
    tensorboard \
    open3d \
    h5py \
    pyrealsense2 \
    pytest

RUN if [ "${INSTALL_RERUN}" = "true" ]; then \
    uv pip install --no-cache "rerun-sdk>=0.21" "numpy<2"; \
    fi

# viser is the sidecar-process backend for --viewer viser (issue #572/#1200,
# wired into the canonical causal_mapping path by #1385). It is opt-in at
# RUNTIME (the flag), not at build time -- installing it unconditionally by
# default here just means the flag actually works, mirroring INSTALL_RERUN's
# own always-true default above. Never imported by the trainer process
# itself, only by the separate sidecar process in viser_server.py.
RUN if [ "${INSTALL_VISER}" = "true" ]; then \
    uv pip install --no-cache "viser>=0.2" "numpy<2" && \
    python3 -c "import viser; print('viser', viser.__version__)"; \
    fi

RUN --mount=from=gsplat-builder,source=/tmp/gsplat/dist,target=/gsplat-dist \
    pip install --no-cache-dir /gsplat-dist/*.whl

RUN --mount=from=gsplat-builder,source=/tmp/fused-ssim/dist,target=/fused-ssim-dist \
    pip install --no-cache-dir /fused-ssim-dist/*.whl

RUN --mount=from=gsplat-builder,source=/tmp/simple-knn/dist,target=/simple-knn-dist \
    pip install --no-cache-dir /simple-knn-dist/*.whl

# Kornia remains a core, transport-neutral frontend dependency. kornia-rs can
# accelerate decode/preprocessing without introducing a ROS dependency.
RUN pip install --no-cache-dir "kornia>=0.7" \
    && python3 -c "import kornia, torch; assert torch.version.hip, 'ROCm torch clobbered'; print('kornia', kornia.__version__, 'torch', torch.__version__)"

# LPIPS is a scientific endpoint metric. The image carries only the manifest;
# model checkpoints are external assets because their redistribution terms are
# unresolved. Users may materialize them explicitly into this mounted path with
# scripts/materialize_eval_assets.py. Runtime loading remains fail-closed.
ENV ELIGSIR_EVAL_ASSET_DIR=/opt/eligsir/eval-assets
VOLUME ["/opt/eligsir/eval-assets"]

# =============================================================================
# STAGE 4: standalone ROS-free EliGSiR runtime
# =============================================================================
FROM eligsir-deps AS eligsir

WORKDIR /workspace/eligsir

COPY arguments ./arguments
COPY contracts ./contracts
COPY gaussian_renderer ./gaussian_renderer
COPY geometry ./geometry
COPY lpipsPyTorch ./lpipsPyTorch
COPY loop_closure ./loop_closure
COPY scene ./scene
COPY slam ./slam
COPY streaming ./streaming
COPY eligsir ./eligsir
COPY utils ./utils
COPY viz ./viz
COPY configs ./configs
COPY scripts ./scripts
COPY train.py train_streaming.py render.py convert.py metrics.py full_eval.py ./
COPY pyproject.toml environment.yml ./
COPY --from=eligsir-deps /tmp/.build_commit /workspace/eligsir/.build_commit

EXPOSE 9876
CMD ["bash"]
