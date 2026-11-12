"""Gaussian-model lifecycle-buffer length synchronization.

Moved out of ``train_streaming.py`` (issue #1364 audit, 2026-08-31): this pure
function -- it only touches its ``gaussians`` argument and ``torch`` -- had a
genuine live caller in ``utils/strategies/mcmc/arbiter_owned.py`` (part of the
canonical ``gsplat_mcmc`` strategy, reachable from the production
``causal_entry``/``production_application`` runtime), which imported it
directly from ``train_streaming`` via a lazy in-function import. That import
edge pulled the entire confirmed-dead legacy module graph
(``train_streaming.py``, ``eligsir.mapping.coordinator``,
``scene.streaming_scene``, etc.) into the canonical dependency cone as
computed by ``scripts/check_canonical_mapping_hard_cut.py``, breaking that
gate's own ``test_repository_satisfies_canonical_mapping_hard_cut`` test
(regressed by the PR that added ``arbiter_owned.py``, `ff050f65`; confirmed via
bisection still broken on `origin/main` as of this audit). Giving this
function a module home outside both the legacy and canonical trees removes the
edge entirely rather than special-casing it in the checker.

``train_streaming.py`` keeps re-exporting the same name (``from
utils.streaming_state_sync import _sync_streaming_state_lengths``) for its own
two call sites, so this is a pure relocation, not a behavior change.
"""

from __future__ import annotations

import torch

from eligsir.utils.gaussian_belief import sync_belief_buffer_lengths


def _sync_streaming_state_lengths(gaussians):
    """Keep streaming lifecycle tensors aligned with the current Gaussian count."""
    count = int(gaussians.get_xyz.shape[0])
    for name, fill_value in (
        ("birth_frame", 0),
        ("support_count", 0),
        ("provisional", False),
        ("anchor_iter", 0),
        ("depth_conflict_count", 0),
        ("depth_last_support_uid", -1),
        ("depth_last_conflict_uid", -1),
        ("lifecycle_state", 1),
        ("utility_ema", 0.0),
    ):
        tensor = getattr(gaussians, name, None)
        if tensor is None:
            continue
        if tensor.shape[0] == count:
            continue
        if tensor.shape[0] > count:
            setattr(gaussians, name, tensor[:count].contiguous())
            continue
        pad = torch.full(
            (count - tensor.shape[0],),
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
    # anchor_xyz needs per-point 3-vector; pad with current positions so anchor=birth position
    anchor_xyz = getattr(gaussians, "anchor_xyz", None)
    if anchor_xyz is not None and anchor_xyz.shape[0] != count:
        if anchor_xyz.shape[0] > count:
            gaussians.anchor_xyz = anchor_xyz[:count].contiguous()
        else:
            deficit = count - anchor_xyz.shape[0]
            new_means = gaussians.get_xyz[-deficit:].detach().clone()
            gaussians.anchor_xyz = torch.cat([anchor_xyz, new_means], dim=0)
    # anchor_scale_log / anchor_opacity_logit: pad with current param values
    for attr, param_key, ndim in (
        ("anchor_scale_log", "scales", 3),
        ("anchor_opacity_logit", "opacities", 1),
    ):
        buf = getattr(gaussians, attr, None)
        if buf is None:
            continue
        if buf.shape[0] == count:
            continue
        if buf.shape[0] > count:
            setattr(gaussians, attr, buf[:count].contiguous())
        else:
            deficit = count - buf.shape[0]
            # Was gaussians.params[param_key][...] -- reads the raw
            # (pre-activation) tensor directly out of the model's private
            # params dict. Both model layouts already expose the same
            # values via a sanctioned public property (raw_scales/
            # raw_opacities), so use that instead of reaching into .params.
            if param_key == "scales":
                new_vals = gaussians.raw_scales[-deficit:].detach().clone()
            elif param_key == "opacities":
                new_vals = gaussians.raw_opacities[-deficit:].detach().clone()
            else:
                new_vals = torch.zeros((deficit, ndim) if ndim > 1 else (deficit,), device=buf.device, dtype=buf.dtype)
            setattr(gaussians, attr, torch.cat([buf, new_vals], dim=0))
    # Geometry-prior metadata buffers.
    anchor_normal = getattr(gaussians, "anchor_normal", None)
    if anchor_normal is not None and anchor_normal.shape[0] != count:
        if anchor_normal.shape[0] > count:
            gaussians.anchor_normal = anchor_normal[:count].contiguous()
        else:
            pad = torch.zeros((count - anchor_normal.shape[0], 3), dtype=anchor_normal.dtype, device=anchor_normal.device)
            pad[:, 2] = 1.0
            gaussians.anchor_normal = torch.cat([anchor_normal, pad], dim=0)
    for name, dtype, fill in (
        ("anchor_confidence", torch.float32, 0.0),
        ("geometry_source", torch.int16, 0),
        ("surface_state", torch.int8, 0),
    ):
        tensor = getattr(gaussians, name, None)
        if tensor is None or tensor.shape[0] == count:
            continue
        if tensor.shape[0] > count:
            setattr(gaussians, name, tensor[:count].contiguous())
        else:
            pad = torch.full((count - tensor.shape[0],), fill, dtype=tensor.dtype, device=tensor.device)
            setattr(gaussians, name, torch.cat([tensor, pad], dim=0))
    # GaussianBelief (issue #1365 Phase 1).
    sync_belief_buffer_lengths(gaussians)
