"""Per-camera affine exposure primitives; endpoint wiring owns persistence."""
from __future__ import annotations
from typing import Any, Tuple
import torch
from torch import nn

_VALID_MODES = ("gray_affine", "rgb_affine")

def _channels(mode: str) -> int:
    if mode not in _VALID_MODES:
        raise ValueError(f"exposure_opt_mode must be one of {_VALID_MODES}, got {mode!r}")
    return 1 if mode == "gray_affine" else 3

class ExposureOptModule(nn.Module):
    def __init__(self, num_cameras: int, mode: str = "rgb_affine", device: Any = None, dtype: torch.dtype = torch.float32):
        super().__init__(); self.mode = mode; n = _channels(mode)
        self.gains = nn.Parameter(torch.ones(int(num_cameras), n, device=device, dtype=dtype))
        self.biases = nn.Parameter(torch.zeros(int(num_cameras), n, device=device, dtype=dtype))
    def _broadcast(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gain, bias = self.gains[idx], self.biases[idx]
        return (gain.expand(3), bias.expand(3)) if gain.numel() == 1 else (gain, bias)
    def apply(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        gain, bias = self._broadcast(idx); return image * gain.view(3,1,1) + bias.view(3,1,1)
    def regularization(self, idx: int) -> torch.Tensor:
        gain, bias = self._broadcast(idx); return ((gain-1)**2).sum() + (bias**2).sum()

def fit_affine(render: torch.Tensor, gt: torch.Tensor, mode: str = "rgb_affine", eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    _channels(mode); r, g = render.reshape(3,-1).to(torch.float64), gt.reshape(3,-1).to(torch.float64)
    if mode == "gray_affine": r, g = r.reshape(1,-1), g.reshape(1,-1)
    rm, gm = r.mean(1), g.mean(1); rc, gc = r-rm[:,None], g-gm[:,None]
    var, cov = (rc*rc).mean(1), (rc*gc).mean(1); bad = var < eps
    gain = torch.where(bad, torch.ones_like(var), cov / var.clamp_min(eps)); bias = gm - gain*rm
    return gain.to(render.dtype), bias.to(render.dtype)

__all__ = ["ExposureOptModule", "fit_affine"]
