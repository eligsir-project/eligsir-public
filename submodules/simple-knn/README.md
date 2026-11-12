# simple-knn (vendored, ROCm/HIP port)

Vendored copy of the `simple-knn` spatial-KNN extension (`distCUDA2`), patched to
build on AMD ROCm/HIP. Used by `utils/rocm_knn_fallback.py` for Gaussian scale init.

- **Upstream (CUDA):** Inria GRAPHDECO — <https://gitlab.inria.fr/bkerbl/simple-knn>
  (part of [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting),
  Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023).
- **Standalone ROCm/HIP port (with wheel-build CI):**
  <https://github.com/eligsir-project/simple-knn-rocm>
- **License:** Inria GRAPHDECO research/evaluation license — see `LICENSE.md` (preserved unchanged).

ROCm changes are build-only and guarded with `#ifndef __HIP_PLATFORM_AMD__` (see the
header comments in `simple_knn.cu`); the KNN algorithm is unmodified.
