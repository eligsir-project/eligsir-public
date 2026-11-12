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

# Keep backend imports lazy: GPU states resolve to gsplat, while tiny CPU
# correctness/CI scenes resolve to the pure-PyTorch reference renderer.
from eligsir.mapping.renderer.dispatch import render

__all__ = ["render"]
