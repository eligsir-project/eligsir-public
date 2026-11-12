#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    """Flatten symmetric lower-diagonal terms on the input tensor's device."""
    uncertainty = L.new_zeros((L.shape[0], 6), dtype=torch.float)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    """Quaternion->rotation on ``r.device``; never implicitly selects cuda:0."""
    norm = torch.sqrt((r * r).sum(dim=1))
    q = r / norm[:, None]
    R = q.new_zeros((q.size(0), 3, 3))
    qw, x, y, z = q.unbind(dim=1)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - qw * z)
    R[:, 0, 2] = 2 * (x * z + qw * y)
    R[:, 1, 0] = 2 * (x * y + qw * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - qw * x)
    R[:, 2, 0] = 2 * (x * z - qw * y)
    R[:, 2, 1] = 2 * (y * z + qw * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    """Build scaling rotation on the source tensor's device/dtype."""
    L = s.new_zeros((s.shape[0], 3, 3))
    R = build_rotation(r.to(device=s.device))
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


def safe_state(silent, seed=1, *, device=None, strict_deterministic=False):
    """Seed reproducible state without silently owning a physical GPU.

    ``strict_deterministic=False`` is the normal research mode: source/order/RNG
    reproducibility while kernel-level numerical variation remains permitted.
    Set it only for qualification/debug because deterministic kernels may be
    slower or unavailable for some operations.
    """
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(datetime.now().strftime("%d/%m %H:%M:%S")),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is not None:
        target = torch.device(device)
        if target.type == "cuda":
            torch.cuda.set_device(target)
    if strict_deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
