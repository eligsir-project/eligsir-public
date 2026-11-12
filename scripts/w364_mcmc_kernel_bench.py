"""W3-64 throwaway cross-platform MCMC-kernel-dispatch + microbenchmark script.

Reproduces, verbatim in methodology, the introspection and isolated
microbenchmark from `docs/architecture/2026-08-10-w3-35-mcmc-kernel-audit.md`
("Reproduction" section) on a platform other than the primary machine
(gfx1151). See `docs/architecture/2026-08-10-w3-64-mcmc-kernel-cross-platform.md`
for the results this script produced on A100 and r9700.

Not wired into any production path or CI list -- a one-off measurement
script, kept for reproducibility, matching the standing this repo already
gives W3-35's own reproduction script (documented in-doc, not unit-tested).

Usage: python scripts/w364_mcmc_kernel_bench.py [output_json_path]
(defaults to /tmp/w364_bench_result.json if no path given)
"""
import inspect
import json
import sys

import torch

result: dict = {}

# --- Part 1: native-kernel-dispatch check (mirrors W3-35's introspection) --
import gsplat
result["gsplat_version"] = getattr(gsplat, "__version__", None)
result["gsplat_file"] = getattr(gsplat, "__file__", None)

try:
    from gsplat.cuda._backend import _C
    result["C_has_relocation"] = hasattr(_C, "relocation")
    result["C_ops"] = sorted(n for n in dir(_C) if not n.startswith("_"))
except Exception as e:  # pragma: no cover - diagnostic only
    result["C_backend_error"] = repr(e)

try:
    from gsplat.strategy.ops import inject_noise_to_position
    src = inspect.getsource(inject_noise_to_position)
    result["inject_noise_calls_torch_ops_gsplat"] = "torch.ops.gsplat" in src
    result["inject_noise_source_len"] = len(src)
except Exception as e:  # pragma: no cover
    result["inject_noise_introspect_error"] = repr(e)

try:
    result["torch_ops_gsplat_has_mcmc_perturb_positions"] = (
        hasattr(torch.ops, "gsplat")
        and hasattr(torch.ops.gsplat, "mcmc_perturb_positions")
    )
except Exception as e:  # pragma: no cover
    result["torch_ops_gsplat_check_error"] = repr(e)

try:
    from gsplat.strategy.ops import compute_relocation
    src2 = inspect.getsource(compute_relocation)
    result["compute_relocation_calls_lazy_cuda_func"] = "_make_lazy_cuda_func" in src2
except Exception as e:  # pragma: no cover
    result["compute_relocation_introspect_error"] = repr(e)

result["cuda_available"] = torch.cuda.is_available()
if torch.cuda.is_available():
    result["device_name"] = torch.cuda.get_device_name(0)
    result["torch_version"] = torch.__version__
    result["torch_cuda_version"] = torch.version.cuda
    result["torch_hip_version"] = getattr(torch.version, "hip", None)

# --- Part 2: isolated microbenchmark (verbatim from W3-35's Reproduction) --
bench_results: dict = {}
if torch.cuda.is_available():
    from gsplat.strategy.ops import inject_noise_to_position
    import time

    def bench(N, iters=200):
        params = {
            "means": torch.nn.Parameter(torch.randn(N, 3, device="cuda")),
            "quats": torch.nn.Parameter(torch.randn(N, 4, device="cuda")),
            "scales": torch.nn.Parameter(torch.randn(N, 3, device="cuda") * 0.01 - 3.0),
            "opacities": torch.nn.Parameter(torch.randn(N, device="cuda") * 0.5),
        }
        for _ in range(10):
            inject_noise_to_position(params=params, optimizers={}, state={}, scaler=0.001)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            inject_noise_to_position(params=params, optimizers={}, state={}, scaler=0.001)
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / iters

    for N in (7900, 28400):
        bench_results[str(N)] = bench(N)

result["bench_ms_per_call"] = bench_results

out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/w364_bench_result.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
