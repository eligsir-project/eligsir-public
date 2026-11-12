#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    """Canonical repository PSNR entry point.

    The historical tensor formula is implemented in ``utils.score_contract``;
    this compatibility wrapper ensures old imports in training/reporting code
    receive the same arithmetic while now failing closed on non-scientific
    score domains (integer/serialized artifacts, shape/device mismatches).
    """
    from eligsir.utils.score_contract import scientific_psnr

    return scientific_psnr(img1, img2, context="utils.image_utils.psnr")
