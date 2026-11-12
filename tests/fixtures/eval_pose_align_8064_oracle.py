"""Required excerpt of utils/eval_pose_align.py at 8064aca93886dd07ad6d890bccbdcda960a9a2af."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from eligsir.pose_opt import CameraPoseOptModule, apply_pose_delta


def _fit_pose_delta(
    cam,
    gaussians,
    render_fn: Callable,
    pipe,
    background,
    gt: torch.Tensor,
    ssim_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_steps: int,
    lr: float = 1e-3,
    lambda_dssim: float = 0.2,
    init_delta6: Optional[torch.Tensor] = None,
    loss_out: Optional[dict] = None,
) -> torch.Tensor:
    device = cam.world_view_transform.device
    pose_opt = CameraPoseOptModule(1, device=device)
    if init_delta6 is not None:
        with torch.no_grad():
            pose_opt.deltas[0].copy_(
                init_delta6.to(device=device, dtype=pose_opt.deltas.dtype)
            )
    optimizer = torch.optim.Adam(pose_opt.parameters(), lr=lr)
    initial_loss: Optional[float] = None
    step_loss = None
    for _ in range(n_steps):
        aligned_cam = apply_pose_delta(cam, pose_opt.delta_matrix(0))
        out = render_fn(aligned_cam, gaussians, pipe, background)
        img = out["render"] if isinstance(out, dict) else out
        l1 = torch.abs(img - gt).mean()
        dssim = 1.0 - ssim_fn(img, gt)
        step_loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim
        if initial_loss is None:
            initial_loss = float(step_loss.detach().item())
        optimizer.zero_grad(set_to_none=True)
        step_loss.backward()
        optimizer.step()
    if loss_out is not None:
        loss_out["initial"] = initial_loss
        loss_out["final"] = (
            float(step_loss.detach().item()) if step_loss is not None else initial_loss
        )
    with torch.no_grad():
        delta = pose_opt.delta_matrix(0).detach().clone()
    return delta
