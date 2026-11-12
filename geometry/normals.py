"""Normal estimation utilities for surface priors and insertion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NormalEstimationConfig:
    mode: str = "finite_difference"
    window: int = 5
    max_depth_jump: float = 0.05
    min_valid_ratio: float = 0.6
    min_depth: float = 0.1
    max_depth: float = 8.0


def estimate_depth_normals_finite_difference(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    *,
    min_depth: float = 0.1,
    max_depth: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera-space normals with central depth differences."""

    z = np.asarray(depth_m, dtype=np.float32)
    dzdx = np.zeros_like(z)
    dzdy = np.zeros_like(z)
    dzdx[:, 1:-1] = z[:, 2:] - z[:, :-2]
    dzdy[1:-1, :] = z[2:, :] - z[:-2, :]
    nx = -dzdx / max(float(fx) * 2.0, 1e-6)
    ny = -dzdy / max(float(fy) * 2.0, 1e-6)
    nz = np.ones_like(z)
    normals = np.stack([nx, ny, nz], axis=0)
    norms = np.linalg.norm(normals, axis=0, keepdims=True) + 1e-8
    normals = normals / norms
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    confidence = valid.astype(np.float32)
    normals[:, ~valid] = 0.0
    return normals.astype(np.float32), confidence


def estimate_depth_normals_pca(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: NormalEstimationConfig | None = None,
    return_residual: bool = False,
    device: "str | torch.device | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera-space normals by fitting local PCA planes.

    Fully vectorized: per-pixel windowed covariance via sliding windows + a
    single batched ``np.linalg.eigh`` over all pixels. (The previous
    pixel-by-pixel Python double loop with a per-pixel eigh cost several
    seconds per call and was the dominant streaming GPU-idle stall.)

    When ``return_residual`` is True, also returns a ``[H,W]`` map of the local
    plane-fit residual in metres (√ smallest eigenvalue = RMS point-to-plane
    distance) — an intra-frame, pose-free surface-roughness / sensor-noise proxy.
    Pixels without a reliable fit are NaN.

    ``depth_m`` is plain numpy (no device-bearing input tensor to derive a
    device from), so ``device`` is an explicit, opt-in parameter: pass it when
    the caller already has a device context (e.g. a GPU-resident model) it
    wants this computation to run on. Left as ``None`` (the default, and every
    current call site), this auto-selects CUDA when available and otherwise
    CPU -- unchanged from prior behavior, just no longer a bare hardcoded
    literal buried in the function body.
    """

    import torch.nn.functional as F

    cfg = config or NormalEstimationConfig(mode="pca_plane")
    z_np = np.asarray(depth_m, dtype=np.float32)
    h, w = z_np.shape
    win = max(1, int(cfg.window))
    if win % 2 == 0:
        win += 1
    radius = win // 2
    k = win * win
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    max_jump = max(float(cfg.max_depth_jump), 1e-6)

    with torch.no_grad():
        z = torch.from_numpy(z_np).to(device)
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        x3 = (xs - float(cx)) / max(float(fx), 1e-6) * z
        y3 = (ys - float(cy)) / max(float(fy), 1e-6) * z
        pts = torch.stack([x3, y3, z], dim=0).unsqueeze(0)  # [1,3,H,W]
        valid = torch.isfinite(z) & (z > cfg.min_depth) & (z < cfg.max_depth)

        # Gather K-neighborhoods per pixel via unfold (zero-padded borders are
        # masked out by the validity mask, so they never enter the fit).
        p_cols = F.unfold(pts, kernel_size=win, padding=radius)            # [1,3K,HW]
        p_cols = p_cols.view(3, k, h * w).permute(2, 1, 0)                 # [HW,K,3]
        z_cols = F.unfold(z.view(1, 1, h, w), kernel_size=win, padding=radius).view(k, h * w).permute(1, 0)
        v_cols = F.unfold(valid.view(1, 1, h, w).float(), kernel_size=win, padding=radius).view(k, h * w).permute(1, 0) > 0.5

        center_z = z.view(-1, 1)
        local_valid = v_cols & ((z_cols - center_z).abs() <= max_jump)     # [HW,K]
        n_valid = local_valid.sum(dim=1).float()
        valid_ratio = n_valid / float(k)
        enough = valid.view(-1) & (valid_ratio >= cfg.min_valid_ratio) & (n_valid >= 3)

        m = local_valid.unsqueeze(-1).float()                              # [HW,K,1]
        cnt = m.sum(dim=1).clamp_min(1.0)                                  # [HW,1]
        mean = (p_cols * m).sum(dim=1) / cnt                              # [HW,3]
        centered = (p_cols - mean.unsqueeze(1)) * m
        cov = torch.einsum("nki,nkj->nij", centered, centered)
        cov = cov / (cnt.unsqueeze(-1).clamp_min(2.0) - 1.0)              # [HW,3,3]

        # Identity where the fit is unreliable keeps eigh well-posed.
        cov = torch.where(enough.view(-1, 1, 1), cov, torch.eye(3, device=device))
        evals, evecs = torch.linalg.eigh(cov)                             # ascending
        normal = evecs[..., 0]                                            # smallest-eigenvalue dir [HW,3]
        normal = torch.where((normal[:, 2] < 0).unsqueeze(-1), -normal, normal)  # orient to camera

        residual = evals[:, 0].clamp_min(0.0).sqrt()
        conf = (valid_ratio * torch.exp(-residual / max_jump)).clamp(0.0, 1.0)
        normal = torch.where(enough.unsqueeze(-1), normal, torch.zeros_like(normal))
        conf = torch.where(enough, conf, torch.zeros_like(conf))

        normals = normal.permute(1, 0).view(3, h, w).contiguous().cpu().numpy().astype(np.float32)
        confidence = conf.view(h, w).cpu().numpy().astype(np.float32)
        if return_residual:
            res = torch.where(enough, residual, torch.full_like(residual, float("nan")))
            residual_map = res.view(h, w).cpu().numpy().astype(np.float32)
    if return_residual:
        return normals, confidence, residual_map
    return normals, confidence


def estimate_depth_normals(
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: NormalEstimationConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = config or NormalEstimationConfig()
    if cfg.mode == "pca_plane":
        return estimate_depth_normals_pca(depth_m, fx, fy, cx, cy, cfg)
    return estimate_depth_normals_finite_difference(
        depth_m,
        fx,
        fy,
        min_depth=cfg.min_depth,
        max_depth=cfg.max_depth,
    )


def quaternion_from_normal_torch(normals: torch.Tensor) -> torch.Tensor:
    """Return wxyz quaternions rotating +Z to each normal."""

    n = torch.nn.functional.normalize(normals, dim=-1)
    z = torch.zeros_like(n)
    z[..., 2] = 1.0
    cross = torch.cross(z, n, dim=-1)
    w = 1.0 + (z * n).sum(dim=-1, keepdim=True)
    q = torch.cat([w, cross], dim=-1)
    bad = q.norm(dim=-1) < 1e-6
    if bad.any():
        q = q.clone()
        q[bad] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
    return torch.nn.functional.normalize(q, dim=-1)
