import torch

# Prefer the compiled simple_knn HIP extension (exact O(N log N) Morton-grid KNN,
# the canonical upstream-3DGS distCUDA2 — mean squared distance to the 3 nearest
# neighbours). It is HIPified for gfx1201/gfx1151; when the wheel is absent (e.g.
# a stripped image or a non-ROCm host) we transparently fall back to the pure-torch
# O(N^2) chunked-cdist path below, so call sites never change.
_SIMPLE_KNN_C = None
_SIMPLE_KNN_TRIED = False


def _simple_knn_c():
    global _SIMPLE_KNN_C, _SIMPLE_KNN_TRIED
    if not _SIMPLE_KNN_TRIED:
        _SIMPLE_KNN_TRIED = True
        try:
            from simple_knn._C import distCUDA2 as _c
            _SIMPLE_KNN_C = _c
        except Exception:
            _SIMPLE_KNN_C = None
    return _SIMPLE_KNN_C


_SIMPLE_KNN_IDX = None
_SIMPLE_KNN_IDX_TRIED = False


def _simple_knn_indices_c():
    global _SIMPLE_KNN_IDX, _SIMPLE_KNN_IDX_TRIED
    if not _SIMPLE_KNN_IDX_TRIED:
        _SIMPLE_KNN_IDX_TRIED = True
        try:
            from simple_knn._C import knnIndices as _ki  # added by the ROCm port
            _SIMPLE_KNN_IDX = _ki
        except Exception:
            _SIMPLE_KNN_IDX = None
    return _SIMPLE_KNN_IDX


def _knn_indices_torch(query, reference, k, exclude_self=False, chunk_size=4096):
    """Pure-torch chunked cross-set k-NN fallback. Returns (dist2 [Nq,k], idx [Nq,k]).

    Chunked over the query to bound the distance-matrix memory (the same reason the
    HIP kernel is preferred: it never materialises the full Nq x Nr matrix).
    """
    q = query.contiguous().float()
    r = reference.contiguous().float()
    Nq = q.shape[0]
    kk = min(int(k), r.shape[0])
    out_d = torch.empty(Nq, kk, device=q.device, dtype=torch.float32)
    out_i = torch.empty(Nq, kk, device=q.device, dtype=torch.int64)
    with torch.no_grad():
        for s in range(0, Nq, chunk_size):
            e = min(s + chunk_size, Nq)
            d = torch.cdist(q[s:e], r) ** 2  # [chunk, Nr]
            if exclude_self:
                rows = torch.arange(e - s, device=q.device)
                d[rows, torch.arange(s, e, device=q.device)] = float("inf")
            dk, ik = torch.topk(d, k=kk, largest=False)
            out_d[s:e] = dk
            out_i[s:e] = ik
    return out_d, out_i


def knn_indices(query: torch.Tensor, reference: torch.Tensor, k: int, exclude_self: bool = False):
    """Cross-set k-NN. Returns (dist2 [Nq, k] float32, idx [Nq, k] int64 into reference).

    Prefers the compiled simple_knn HIP kernel (exact Morton-grid, deterministic,
    never materialises the Nq x Nr matrix); falls back to chunked torch cdist+topk
    when the extension is absent. `exclude_self` skips each query's own index (use
    when query IS reference). NOTE: the neighbour search is non-differentiable —
    gather on the returned (detached) indices to build any differentiable term.
    """
    _ki = _simple_knn_indices_c()
    if (
        _ki is not None
        and query.is_cuda and reference.is_cuda
        and 1 <= int(k) <= 32
        and reference.shape[0] > 0
    ):
        try:
            idx, dist2 = _ki(query.contiguous().float(), reference.contiguous().float(),
                             int(k), bool(exclude_self))
            return dist2, idx
        except Exception:
            pass
    return _knn_indices_torch(query, reference, k, exclude_self=exclude_self)


def knn_self(points: torch.Tensor, k: int):
    """Self k-NN (each point's k nearest *other* points). Returns (dist2, idx)."""
    return knn_indices(points, points, k, exclude_self=True)


def _distCUDA2_torch(points: torch.Tensor, chunk_size: int = 8192, k: int = 4):
    """Pure-PyTorch O(N^2) chunked fallback for simple_knn._C.distCUDA2.

    points: [N, 3] tensor
    returns: squared distance estimate per point (mean over k nearest neighbours)
    """
    points = points.contiguous()
    out = torch.empty(points.shape[0], device=points.device, dtype=points.dtype)

    # k must not exceed N-1 (self is excluded)
    k = min(k, points.shape[0] - 1)
    if k <= 0:
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)

    with torch.no_grad():
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            q = points[start:end]
            d = torch.cdist(q.float(), points.float())

            rows = torch.arange(end - start, device=points.device)
            cols = torch.arange(start, end, device=points.device)
            d[rows, cols] = float("inf")

            knn_dists, _ = torch.topk(d, k=k, largest=False)
            out[start:end] = knn_dists.mean(dim=1).to(points.dtype) ** 2

    return out


def distCUDA2(points: torch.Tensor, chunk_size: int = 8192, k: int = 4):
    """Mean squared distance to the nearest neighbours, per point.

    points: [N, 3] tensor
    returns: [N] squared-distance estimate per point.

    Uses the compiled simple_knn HIP kernel when available (GPU float [N,3], the
    default k=4 request — the kernel itself returns the mean over the 3 nearest
    neighbours, matching upstream 3DGS). Otherwise uses the pure-torch fallback.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")

    N = points.shape[0]
    _c = _simple_knn_c()
    # The kernel needs a CUDA/HIP float tensor and enough points to populate its
    # 3-NN search (tiny N hits a degenerate Morton box); defer those to torch.
    if (
        _c is not None
        and points.is_cuda
        and k == 4
        and N > 8
    ):
        try:
            pts = points.contiguous().to(torch.float32)
            res = _c(pts)  # [N] mean sq-dist to 3 NN, same device
            # Guard the all-identical / degenerate-bbox case (NaN/inf Morton codes):
            # fall back to the robust torch path rather than seed NaN scales.
            if torch.isfinite(res).all():
                return res.to(points.dtype)
        except Exception:
            pass  # any kernel/runtime issue -> torch fallback

    return _distCUDA2_torch(points, chunk_size=chunk_size, k=k)
