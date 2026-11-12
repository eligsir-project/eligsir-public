"""W3-65 Part 2 reproduction: mature-mask quiescence policy overhead benchmark.

Compares the current clone->perturb-all-N->restore pattern in
utils/strategies/mcmc/energy.py's inject_noise against a synthetic
masked/indexed-subset prototype, at N=100k/400k/1M and active_frac=
1.0/0.5/0.25/0.10. Also includes a bare inject_noise_to_position call as a
same-process reference, so bare and current-policy costs are directly
comparable without cross-run GPU clock/thermal drift (see
docs/architecture/2026-08-10-w3-65-noise-highn-mature-mask.md's Part 2
caveat for why that comparison matters).

Run inside the training container:
    docker compose run --rm train python scripts/w3_65_mature_mask_policy_bench.py
"""
import torch, time
from gsplat.strategy.ops import inject_noise_to_position

def make_params(N):
    return {
        "means": torch.nn.Parameter(torch.randn(N, 3, device="cuda")),
        "quats": torch.nn.Parameter(torch.randn(N, 4, device="cuda")),
        "scales": torch.nn.Parameter(torch.randn(N, 3, device="cuda") * 0.01 - 3.0),
        "opacities": torch.nn.Parameter(torch.randn(N, device="cuda") * 0.5),
    }

def bench_current_policy(N, active_frac, iters=100):
    params = make_params(N)
    n_active = int(N * active_frac)
    perm = torch.randperm(N, device="cuda")
    active_idx = perm[:n_active]
    mask = torch.zeros(N, dtype=torch.bool, device="cuda")
    mask[active_idx] = True
    mature_mask = ~mask  # mature = NOT active

    def step():
        if bool(mature_mask.any()):
            mature_means = params["means"].data[mature_mask].clone()
        else:
            mature_means = None
        inject_noise_to_position(params=params, optimizers={}, state={}, scaler=0.001)
        if mature_means is not None:
            params["means"].data[mature_mask] = mature_means

    for _ in range(10):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / iters

def bench_masked_prototype(N, active_frac, iters=100):
    params = make_params(N)
    n_active = int(N * active_frac)
    perm = torch.randperm(N, device="cuda")
    active_idx = perm[:n_active]

    def step():
        sub = {k: torch.nn.Parameter(v.data[active_idx].clone()) for k, v in params.items()}
        inject_noise_to_position(params=sub, optimizers={}, state={}, scaler=0.001)
        params["means"].data[active_idx] = sub["means"].data

    for _ in range(10):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / iters

# sanity: bare call cost for reference
def bench_bare(N, iters=100):
    params = make_params(N)
    for _ in range(10):
        inject_noise_to_position(params=params, optimizers={}, state={}, scaler=0.001)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        inject_noise_to_position(params=params, optimizers={}, state={}, scaler=0.001)
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / iters

for N in [100000, 400000, 1000000]:
    bare = bench_bare(N)
    print(f"N={N} BARE={bare:.4f}ms")
    for af in [1.0, 0.5, 0.25, 0.10]:
        cur = bench_current_policy(N, af)
        msk = bench_masked_prototype(N, af)
        print(f"  N={N} active_frac={af} current={cur:.4f}ms masked={msk:.4f}ms ratio={cur/msk:.2f}x  (current-vs-bare={cur/bare:.2f}x)")
