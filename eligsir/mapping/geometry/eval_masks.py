"""Pixel-level glass/specular evaluation masks (plan section 13.6 Subagent E,
deliverable 5: "Glass/specular evaluation uses pixel masks, not only
trajectory index ranges").

``scripts/analyze_glass_regions.py`` (pre-existing) classifies whole test
*views* as glass/non-glass from a hand-picked ``source_index`` range
(``docs/superpowers/plans/2026-07-23-floor3-glass-region-diagnostic.md``-era
tooling) — coarse, and wrong whenever a glass view also contains ordinary
non-glass geometry (or vice versa) in the same frame. This module instead
derives a *per-pixel* mask directly from image content, so a single view can
be split into its glass-risk and non-glass-risk pixels rather than being
bucketed whole.

Two independent signals feed the mask (either alone is a legitimate glass/
specular indicator; the mask is their union):

- **Specular/extreme-intensity pixels in the ground-truth image** — glass
  and other specular surfaces routinely blow out to near-white highlights
  or transmit background light, producing near-saturated or (for dark
  glass/deep shadow behind glass) near-black pixels that plain matte
  geometry rarely produces at the same spatial frequency.
- **Sensor-invalid-depth pixels, when a depth map is available** — the same
  ``UNKNOWN``-by-default signal used by
  :mod:`eligsir.mapping.geometry.ray_classifier` (glass is the single
  most common real-world cause of a persistent invalid depth return).

This module has no rendering/training dependency — it operates on already
decoded uint8 images (e.g. loaded back from
``comparison/iter_N/test/<name>.png``, which already stores
render|gt|diff panels side by side — see ``utils/comparison_report.py``'s
``_hconcat_with_gap([r_u8, g_u8, d_u8], gap_px=4)``), so it can be applied
post-hoc to any existing run's saved artifacts without a new training pass.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def specular_saturation_mask(
    gt_rgb_u8: np.ndarray,
    *,
    bright_thresh: int = 250,
    dark_thresh: int = 5,
) -> np.ndarray:
    """Boolean ``[H, W]`` mask over near-saturated / near-black pixels in an
    ``[H, W, 3]`` uint8 ground-truth image."""
    intensity = gt_rgb_u8.astype(np.float32).mean(axis=-1)
    return (intensity >= bright_thresh) | (intensity <= dark_thresh)


def invalid_depth_mask(sensor_depth: np.ndarray) -> np.ndarray:
    """Boolean ``[H, W]`` mask over invalid (``<=0``/non-finite) depth pixels."""
    return ~(np.isfinite(sensor_depth) & (sensor_depth > 0))


def glass_specular_pixel_mask(
    gt_rgb_u8: np.ndarray,
    sensor_depth: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Union of :func:`specular_saturation_mask` and (optionally)
    :func:`invalid_depth_mask`, resized to ``gt_rgb_u8``'s resolution if a
    depth map of a different resolution is supplied."""
    mask = specular_saturation_mask(gt_rgb_u8, **kwargs)
    if sensor_depth is not None:
        depth_invalid = invalid_depth_mask(sensor_depth)
        if depth_invalid.shape != mask.shape:
            # Nearest-neighbour resize without an extra image-library
            # dependency: index by rounded ratio.
            h, w = mask.shape
            ys = np.clip((np.arange(h) * depth_invalid.shape[0] / h).astype(np.int64), 0, depth_invalid.shape[0] - 1)
            xs = np.clip((np.arange(w) * depth_invalid.shape[1] / w).astype(np.int64), 0, depth_invalid.shape[1] - 1)
            depth_invalid = depth_invalid[ys][:, xs]
        mask = mask | depth_invalid
    return mask


def split_render_gt_diff_panel(panel: np.ndarray, gap_px: int = 4) -> tuple:
    """Inverse of ``utils.comparison_report._hconcat_with_gap`` for the
    standard 3-panel ``[render | gt | diff]`` layout: split back into three
    equal-width ``[H, W, 3]`` uint8 images."""
    h, total_w, c = panel.shape
    cell_w = (total_w - 2 * gap_px) // 3
    render = panel[:, 0:cell_w]
    gt = panel[:, cell_w + gap_px: 2 * cell_w + gap_px]
    diff = panel[:, 2 * (cell_w + gap_px): 2 * (cell_w + gap_px) + cell_w]
    return render, gt, diff


def masked_psnr(render_u8: np.ndarray, gt_u8: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """PSNR restricted to ``mask`` (True pixels), or ``None`` if the mask
    selects no pixels."""
    if not bool(mask.any()):
        return None
    r = render_u8.astype(np.float64)
    g = gt_u8.astype(np.float64)
    diff = (r[mask] - g[mask]) ** 2
    mse = float(diff.mean())
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))
