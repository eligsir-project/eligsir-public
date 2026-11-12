"""
Post-training comparison report.

Used by both offline (train.py) and streaming (train_streaming.py) paths to
produce a uniform set of artifacts under ``<model_path>/comparison/iter_{N}/``:

  test/{name}.png            side-by-side render | gt | abs-diff (one per test view)
  test/{name}_normals.png    side-by-side splat-axis | depth-derived | gt (per test view)
  test_contact_sheet.png     grid of all test views (render row + gt row)
  normals_contact_sheet.png  grid of splat-axis normals for test views (normals | gt)
  surf_normals_contact_sheet.png  grid of depth-derived surface normals (normals | gt)
  trajectory.mp4             video over the train cameras in order
  report.json                summary metrics (mean/min/max PSNR, counts)

The reporter prints the mean test PSNR to stdout and logs it to TensorBoard
when a writer is supplied. Test PSNR is always reported when test views exist.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from eligsir.io.report_provenance import infer_report_provenance
from eligsir.runtime.source_ingress import source_ingress_warning
from eligsir.utils.trajectory_video import (
    DEFAULT_MAX_ENCODED_FRAMES,
    DEFAULT_MAX_INTERPOLATION_PER_SLICE,
    DEFAULT_MAX_INTERPOLATION_RENDERS,
    DEFAULT_MAX_VIDEO_WIDTH,
    DEFAULT_MIN_VIDEO_WIDTH,
    cams_have_usable_timestamps,
    compute_interpolation_render_counts,
    compute_playback_speed,
    compute_trajectory_timing,
    distribute_frame_repeats,
    is_upscaled,
    scale_dims_for_video,
    split_repeats_evenly,
)


# ---------------------------------------------------------------------------
# reproduce.sh generation
# ---------------------------------------------------------------------------

def _parse_docker_volumes() -> dict:
    """Discover host→container path mappings for reproduce.sh.

    Combines mountinfo (kernel view) with docker-compose.yml (orchestrator view)
    so network-backed mounts (NFS, CIFS) are also captured.

    Returns {container_path: host_path} for each detected volume/bind-mount.
    Skips standard system paths and filesystem types that are not user data.
    """
    SKIP_FSTYPES = {
        "overlay", "tmpfs", "proc", "sysfs", "devpts", "cgroup", "cgroup2",
        "mqueue", "shm", "nsfs", "hugetlbfs", "fuse", "debugfs", "tracefs",
        "securityfs", "pstorefs", "bpf", "autofs", "rpc_pipefs",
    }
    SKIP_MOUNT_PREFIXES = ("/proc", "/sys", "/dev", "/etc/", "/run")

    volumes = {}

    # 1. Kernel mountinfo
    try:
        with open("/proc/self/mountinfo") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                host_root = parts[3]    # path on host device
                mountpoint = parts[4]   # path inside container

                # Find " - " separator between optional fields and fstype
                sep = None
                for i, p in enumerate(parts):
                    if p == "-":
                        sep = i
                        break
                if sep is None or sep + 1 >= len(parts):
                    continue
                fstype = parts[sep + 1]

                if fstype in SKIP_FSTYPES:
                    continue
                if mountpoint == "/":
                    continue
                if any(mountpoint == p or mountpoint.startswith(p + "/")
                       for p in SKIP_MOUNT_PREFIXES):
                    continue
                # host_root == "/" is typical for network FS (NFS, CIFS);
                # for Docker/Podman bind-mounts longer paths are more common.
                if host_root.startswith("/"):
                    volumes[mountpoint] = host_root
    except Exception:
        pass

    # 2. docker-compose.yml (catches network mounts mountinfo misses)
    for candidate in ("/workspace/3dgs-mcmc/docker-compose.yml",
                      "../docker-compose.yml",
                      "docker-compose.yml"):
        if os.path.isfile(candidate):
            try:
                with open(candidate) as fh:
                    compose = yaml.safe_load(fh)
            except Exception:
                continue
            for vol in (compose.get("services", {}).get("train", {}).get("volumes", []) or []):
                if not isinstance(vol, str) or ":" not in vol:
                    continue
                parts = vol.split(":", 2)
                hpath, cpath = parts[0].strip(), parts[1].strip()
                # Compose short-syntax also allows a mode (e.g. ":ro"),
                # which would be the 3rd component — strip it if present.
                if cpath == "":
                    continue
                if hpath.startswith(("~", "/")):
                    volumes.setdefault(cpath, os.path.expanduser(hpath))
            break  # only parse the first match

    return volumes


def write_reproduce_sh(path: str, argv: list) -> None:
    """Write a reproduce.sh that includes the Docker wrapper when running in a container.

    When /.dockerenv is present (Docker/Podman), the script contains:
      - A `docker compose run --rm ...` section (for re-running from the host)
      - A plain `python ...` section (for re-running inside the container)

    When not in a container, writes a plain `python ...` command.
    """
    import shlex
    import stat

    cmd = " ".join(shlex.quote(a) for a in argv)
    in_docker = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv") or os.environ.get("container") in ("docker", "podman")

    lines = [
        "#!/bin/bash",
        "# Reproduces the training run stored in this output directory.",
        "# Auto-generated at training start.",
        "",
    ]

    if in_docker:
        volumes = _parse_docker_volumes()
        lines += [
            "# ── Run from the HOST machine (project root) ─────────────────────",
            "docker compose run --rm \\",
        ]
        for cpath in sorted(volumes):
            hpath = volumes[cpath]
            lines.append(f'  -v "{hpath}:{cpath}" \\')
        lines += [
            "  train \\",
            f"  python {cmd}",
            "",
            "# ── OR run directly inside the container ─────────────────────────",
            f"# python {cmd}",
        ]
    else:
        lines += [f"python {cmd}"]

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)
    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except OSError as e:
        # Cosmetic (makes the script directly executable) — must never crash
        # training. Confirmed live: a stale reproduce.sh from a prior run,
        # owned by a different UID on the shared NFS mount, made chmod raise
        # PermissionError and killed the whole run at iteration 0.
        print(f"[streaming] reproduce.sh chmod +x skipped: {e}", flush=True)


# Issue #1259/#1284: a real ScanNet++ campaign run on r9700 hit
# "RuntimeWarning: overflow encountered in multiply" immediately followed by
# "RuntimeWarning: invalid value encountered in cast" at the `* 255.0`
# return line below, THEN THE PROCESS WENT SILENT (no traceback, no
# [TERMINATION] marker, dead not hung, exit code 1) -- despite the
# torch-side `nan_to_num(..., posinf=1.0, neginf=0.0)` + `clamp(0, 1)` guard
# already being in place at that commit (verified via `git show
# <pinned_commit>:utils/comparison_report.py` and reproduced locally: every
# NaN/Inf/huge-finite/int64/float16-max synthetic case sanitizes cleanly
# through that guard with zero warnings on this host's torch/numpy). A
# mathematically clamped [0, 1] array cannot overflow on `* 255.0` --  so
# the array reaching the cast on that run was NOT actually clamped, despite
# the guard code being present and correct in isolation. Root cause not
# confirmed (tracked separately, see the module docstring reference to
# issue #1284's follow-up); the leading hypothesis is a ROCm/HIP kernel gap
# in `torch.nan_to_num`/`clamp` for some dtype/shape on that hardware
# (silent no-op rather than raising), which pure-Python/numpy code cannot
# rule out by inspection alone. Given that, NEVER trust the torch-side
# guard as sufficient on its own: re-verify with plain numpy immediately
# before the cast that actually broke, and fail loud (sanitize + an
# unmissable warning naming the view and array stats) rather than let a
# corrupted uint8 array silently propagate into a contact sheet / video
# frame as if nothing happened.
_NAN_SANITIZE_STATS: Dict[str, int] = {"count": 0}
_NAN_SANITIZE_EXAMPLES: List[dict] = []
_NAN_SANITIZE_EXAMPLES_CAP = 20


def reset_nan_sanitize_tracking() -> None:
    """Clear the accumulated NaN/Inf-cast sanitization accounting.

    Call at the start of a report-generation run (long-lived processes /
    test suites can otherwise leak counts from one report into the next).
    """
    _NAN_SANITIZE_STATS["count"] = 0
    _NAN_SANITIZE_EXAMPLES.clear()


def get_nan_sanitize_report() -> Optional[dict]:
    """Return accumulated NaN/Inf-cast sanitization accounting, or None if clean.

    Mirrors the `n_test_metrics_failed` / `test_metrics_render_failures`
    accounting pattern (issue #437): a report whose render/GT arrays needed
    sanitizing before a uint8 cast must never look identical to a clean
    report -- callers should fold this into `report.json` (e.g. under
    `geometry` or top-level `nan_sanitized_casts`) whenever it is non-None.
    """
    if _NAN_SANITIZE_STATS["count"] == 0:
        return None
    return {
        "n_sanitized": _NAN_SANITIZE_STATS["count"],
        "examples": list(_NAN_SANITIZE_EXAMPLES),
        "examples_truncated": _NAN_SANITIZE_STATS["count"] > len(_NAN_SANITIZE_EXAMPLES),
    }


def _to_uint8_hwc(img: torch.Tensor, label: Optional[str] = None) -> np.ndarray:
    """Convert a CHW float tensor in [0, 1] to HWC uint8.

    ``clamp`` does not remove NaN (NaN compares false to both bounds), so a
    NaN pixel in the source render (seen on real runs, e.g. numerically
    unstable Gaussians) survives into the ``* 255`` cast and triggers an
    "invalid value encountered in cast" warning. Sanitize before clamping.

    That torch-side sanitize+clamp is NOT trusted as sufficient on its own
    (see the module-level comment above `_NAN_SANITIZE_STATS`, issue
    #1259/#1284) -- a second, independent numpy-level finite check runs
    immediately before the final cast and sanitizes + logs loudly if the
    torch-side guard somehow didn't hold, instead of letting an
    "overflow"/"invalid value" numpy RuntimeWarning silently produce an
    undefined uint8 array. ``label`` (e.g. a camera/view name) is included
    in the warning and in `get_nan_sanitize_report()` so a corrupted report
    can always be traced back to which view triggered it.
    """
    arr = torch.nan_to_num(img.detach(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    # Not just `not np.isfinite(arr).all()`: a huge-but-finite value (e.g.
    # 1e30) passes an isfinite check yet still overflows on `* 255.0` below
    # (numpy's "overflow encountered in multiply"), which is exactly the
    # shape of the real issue #1284 failure. Treat anything outside the
    # valid [0, 1] image range -- finite or not -- as needing sanitization.
    finite_mask = np.isfinite(arr)
    out_of_range = finite_mask & ((arr < 0.0) | (arr > 1.0))
    bad_mask = ~finite_mask | out_of_range
    if bad_mask.any():
        nan_count = int(np.isnan(arr).sum())
        posinf_count = int(np.isposinf(arr).sum())
        neginf_count = int(np.isneginf(arr).sum())
        finite_vals = arr[finite_mask]
        stats = {
            "label": label or "<unlabeled>",
            "shape": list(arr.shape),
            "nan_count": nan_count,
            "posinf_count": posinf_count,
            "neginf_count": neginf_count,
            "out_of_range_finite_count": int(out_of_range.sum()),
            "finite_min": float(finite_vals.min()) if finite_vals.size else None,
            "finite_max": float(finite_vals.max()) if finite_vals.size else None,
        }
        _NAN_SANITIZE_STATS["count"] += 1
        if len(_NAN_SANITIZE_EXAMPLES) < _NAN_SANITIZE_EXAMPLES_CAP:
            _NAN_SANITIZE_EXAMPLES.append(stats)
        print(
            f"[comparison_report] WARNING: NaN/Inf/out-of-range value detected "
            f"in render for view '{stats['label']}' AFTER the torch-side "
            f"nan_to_num/clamp guard already ran (nan={nan_count} "
            f"posinf={posinf_count} neginf={neginf_count} "
            f"out_of_range_finite={stats['out_of_range_finite_count']} "
            f"finite_range=[{stats['finite_min']}, {stats['finite_max']}], "
            f"shape={stats['shape']}); sanitizing before the uint8 cast "
            f"instead of letting numpy silently produce an undefined value "
            f"(or overflow on the `* 255.0` multiply). See issue #1259/#1284.",
            flush=True,
        )
        arr = np.clip(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _abs_diff(render: torch.Tensor, gt: torch.Tensor, scale: float = 5.0) -> np.ndarray:
    """|render - gt| amplified by `scale` for visibility."""
    if gt.device != render.device:
        gt = gt.to(render.device, non_blocking=True)
    diff = (render - gt).abs().mean(dim=0, keepdim=True) * scale
    diff = diff.clamp(0.0, 1.0).expand(3, -1, -1)
    return _to_uint8_hwc(diff)


def _hconcat_with_gap(images: List[np.ndarray], gap_px: int = 4) -> np.ndarray:
    """Stack uint8 HWC images horizontally with a thin white gap between."""
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    h = max(im.shape[0] for im in images)
    parts = []
    for i, im in enumerate(images):
        if im.shape[0] != h:
            # Pad to common height (top-aligned).
            pad = np.full((h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
            im = np.concatenate([im, pad], axis=0)
        parts.append(im)
        if i < len(images) - 1:
            parts.append(np.full((h, gap_px, 3), 255, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def _label_strip(text: str, width: int, height: int = 18) -> np.ndarray:
    """Render a label as a uint8 HWC strip using cv2 if available, else blank."""
    try:
        import cv2  # type: ignore
        strip = np.full((height, width, 3), 255, dtype=np.uint8)
        cv2.putText(
            strip, text, (4, height - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )
        return strip
    except Exception:
        return np.full((height, width, 3), 255, dtype=np.uint8)


def _save_png(path: str, arr_hwc_uint8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import cv2  # type: ignore
        bgr = arr_hwc_uint8[:, :, ::-1].copy()
        cv2.imwrite(path, bgr)
    except Exception:
        # Fallback: torchvision
        import torchvision.utils as tvu
        t = torch.from_numpy(arr_hwc_uint8).permute(2, 0, 1).float() / 255.0
        tvu.save_image(t, path)


def _render_image(cam, gaussians, render_fn, pipe, background) -> torch.Tensor:
    with torch.no_grad():
        out = render_fn(cam, gaussians, pipe, background)
        img = out["render"] if isinstance(out, dict) else out
    return img


def _render_full_pass(cam, gaussians, render_fn, pipe, background) -> Optional[dict]:
    """Single render call that returns all outputs needed for every visual artifact.

    Requests render_normals=True so RGB, splat normals, surf normals, conics, and
    depth come back in one GPU forward pass instead of three separate calls.
    Falls back to a plain render when render_normals is unsupported.
    """
    try:
        with torch.no_grad():
            out = render_fn(cam, gaussians, pipe, background, render_normals=True)
        if not isinstance(out, dict) or "render" not in out:
            return None
        out["_cam"] = cam
        out["_pipe"] = pipe
        return out
    except TypeError:
        # render_fn doesn't accept render_normals — fall back to plain call.
        try:
            with torch.no_grad():
                out = render_fn(cam, gaussians, pipe, background)
            if not isinstance(out, dict) or "render" not in out:
                return None
            out["_cam"] = cam
            out["_pipe"] = pipe
            return out
        except Exception:
            return None
    except Exception:
        return None


def _render_normals_image(cam, gaussians, render_fn, pipe, background):
    """Render normal maps for a view.

    Returns ``(splat_normals, surf_normals)`` — each [3, H, W] in [0,1], or None if
    unsupported. ``splat_normals`` is the per-Gaussian shortest-axis normal;
    ``surf_normals`` is derived from the rendered depth (may be None on older paths).
    """
    try:
        with torch.no_grad():
            out = render_fn(cam, gaussians, pipe, background, render_normals=True)
            if not isinstance(out, dict):
                return None
            normals = out.get("rendered_normals")
            if normals is None:
                return None
            surf = out.get("rendered_surf_normals")
        return (
            normals.clamp(0.0, 1.0),
            surf.clamp(0.0, 1.0) if surf is not None else None,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SuperSplat-style splat-geometry overlays: Centers (dot per Gaussian center)
# and Rings (projected covariance ellipse outline per Gaussian).
# ---------------------------------------------------------------------------

# Defaults (kept conservative so dense scenes stay legible / fast to draw).
_OVERLAY_MAX_MARKERS = 30000   # cap markers drawn per view (top-N by opacity)
_OVERLAY_DOT_RADIUS = 2        # center dot radius in px
_OVERLAY_RING_K = 2.0          # ring drawn at k-sigma boundary
_CENTER_COLOR = (40, 110, 255)  # RGB blue-ish
_RING_COLOR = (40, 230, 60)     # RGB green
_EPS2D = 0.3                    # matches gsplat rasterization eps2d default


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """[N,4] wxyz (possibly unnormalised) → [N,3,3] rotation matrices."""
    q = torch.nn.functional.normalize(quats, dim=-1, eps=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)


def _cov2d_from_conics(conics: torch.Tensor) -> Optional[np.ndarray]:
    """Invert gsplat conics ([M,3] (a,b,c) or [M,2,2]) → 2D covariances [M,2,2] (numpy)."""
    c = conics.detach().float().cpu()
    if c.ndim == 2 and c.shape[-1] == 3:
        a, b, cc = c[:, 0], c[:, 1], c[:, 2]
    elif c.ndim == 3 and c.shape[-2:] == (2, 2):
        a, b, cc = c[:, 0, 0], c[:, 0, 1], c[:, 1, 1]
    else:
        return None
    det = (a * cc - b * b).clamp_min(1e-12)
    cov = torch.stack([cc / det, -b / det, -b / det, a / det], dim=1).reshape(-1, 2, 2)
    return cov.numpy()


def _cov2d_ewa(gaussians, cam, ids: Optional[torch.Tensor], pipe) -> Optional[np.ndarray]:
    """Self-contained EWA projection of 3D covariances → 2D covariances [M,2,2] (numpy).

    Fallback used only when the renderer ``meta`` does not expose ``conics``.
    """
    try:
        from gaussian_renderer.gsplat_backend import _camera_tensors
    except Exception:
        return None
    device = gaussians.get_xyz.device
    viewmat, K, _ = _camera_tensors(cam, device)
    means = gaussians.get_xyz.detach()
    quats = gaussians.get_rotation.detach()
    scales = gaussians.get_scaling.detach()
    if ids is not None:
        means, quats, scales = means[ids], quats[ids], scales[ids]
    R3 = _quat_to_rotmat(quats)                                   # [M,3,3]
    S2 = (scales * scales)                                        # [M,3]
    Sigma3d = R3 @ torch.diag_embed(S2) @ R3.transpose(1, 2)      # [M,3,3]
    Rcw = viewmat[:3, :3]
    tcw = viewmat[:3, 3]
    Sigma_cam = Rcw @ Sigma3d @ Rcw.T                             # [M,3,3]
    p_cam = means @ Rcw.T + tcw                                   # [M,3]
    z = p_cam[:, 2].clamp_min(1e-6)
    fx, fy = K[0, 0], K[1, 1]
    zeros = torch.zeros_like(z)
    J = torch.stack([
        fx / z, zeros, -fx * p_cam[:, 0] / (z * z),
        zeros, fy / z, -fy * p_cam[:, 1] / (z * z),
    ], dim=1).reshape(-1, 2, 3)                                   # [M,2,3]
    Sigma2d = J @ Sigma_cam @ J.transpose(1, 2)                   # [M,2,2]
    eye = torch.eye(2, device=device) * _EPS2D
    Sigma2d = Sigma2d + eye
    return Sigma2d.detach().float().cpu().numpy()


def _compute_view_markers(out: dict, gaussians, max_markers: int):
    """From a render output dict, return (xy [K,2] float, cov2d [K,2,2] or None).

    Selects up to ``max_markers`` Gaussians (top-N by opacity when over the cap).
    ``cov2d`` is taken from ``meta['conics']`` when present, else computed via EWA.
    """
    meta = out.get("meta") if isinstance(out, dict) else None
    if not meta:
        return None, None
    means2d = meta.get("means2d")
    if means2d is None:
        return None, None
    means2d = means2d.detach()
    if means2d.ndim == 3:           # [1,M,2] → [M,2]
        means2d = means2d[0]
    M = means2d.shape[0]
    if M == 0:
        return None, None
    ids = meta.get("gaussian_ids")
    if ids is not None:
        ids = ids.detach().long().view(-1)
        if ids.shape[0] != M:
            ids = None

    # Opacity for selection / ordering (front-most/strongest first).
    op = None
    try:
        op_all = gaussians.get_opacity.squeeze(-1).detach()
        op = op_all[ids] if ids is not None else (op_all if op_all.shape[0] == M else None)
    except Exception:
        op = None

    sel = None
    if M > max_markers:
        if op is not None:
            sel = torch.topk(op, max_markers).indices
        else:
            sel = torch.arange(0, M, device=means2d.device)[:max_markers]
        means2d = means2d[sel]
        if ids is not None:
            ids = ids[sel]

    cov2d = None
    conics = meta.get("conics")
    if conics is not None:
        conics = conics.detach()
        if conics.ndim == 3 and conics.shape[0] == 1:
            conics = conics[0]
        if sel is not None and conics.shape[0] == M:
            conics = conics[sel]
        cov2d = _cov2d_from_conics(conics)
    if cov2d is None:
        try:
            cov2d = _cov2d_ewa(gaussians, out.get("_cam"), ids, out.get("_pipe"))
        except Exception:
            cov2d = None

    xy = means2d.float().cpu().numpy()
    # Guard: rings need one covariance per drawn centre; drop rings on any mismatch.
    if cov2d is not None and cov2d.shape[0] != xy.shape[0]:
        cov2d = None
    return xy, cov2d


def _draw_centers(base_u8: np.ndarray, xy: np.ndarray, radius: int, color) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception:
        return base_u8
    img = base_u8.copy()
    H, W = img.shape[:2]
    pts = np.round(xy).astype(np.int32)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    for x, y in pts[inside]:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return img


def _draw_rings(base_u8: np.ndarray, xy: np.ndarray, cov2d: np.ndarray,
                k: float, color, thickness: int = 1) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except Exception:
        return base_u8
    img = base_u8.copy()
    H, W = img.shape[:2]
    A = cov2d[:, 0, 0]; B = cov2d[:, 0, 1]; C = cov2d[:, 1, 1]
    tr = A + C
    tmp = np.sqrt(np.maximum(((A - C) * 0.5) ** 2 + B * B, 0.0))
    l1 = np.maximum(tr * 0.5 + tmp, 0.0)
    l2 = np.maximum(tr * 0.5 - tmp, 0.0)
    major = (k * np.sqrt(l1)).astype(np.int32)
    minor = (k * np.sqrt(l2)).astype(np.int32)
    angle = np.degrees(0.5 * np.arctan2(2.0 * B, A - C))
    pts = np.round(xy).astype(np.int32)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H) & (major >= 1)
    for i in np.nonzero(inside)[0]:
        cv2.ellipse(img, (int(pts[i, 0]), int(pts[i, 1])),
                    (int(major[i]), int(max(minor[i], 1))), float(angle[i]),
                    0, 360, color, thickness, lineType=cv2.LINE_AA)
    return img


def _render_overlay_images(cam, gaussians, render_fn, pipe, background,
                           *, max_markers=_OVERLAY_MAX_MARKERS,
                           _cached_out: Optional[dict] = None):
    """Render a view → (centers_u8, rings_u8, base_rgb_u8), or (None, None, None).

    Pass ``_cached_out`` (from ``_render_full_pass``) to skip a redundant GPU call.
    """
    try:
        if _cached_out is not None:
            out = _cached_out
        else:
            with torch.no_grad():
                out = render_fn(cam, gaussians, pipe, background)
            if not isinstance(out, dict) or "render" not in out:
                return None, None, None
            out["_cam"] = cam       # for EWA fallback inside _compute_view_markers
            out["_pipe"] = pipe
        base = _to_uint8_hwc(out["render"].clamp(0.0, 1.0))
        xy, cov2d = _compute_view_markers(out, gaussians, max_markers)
        if xy is None:
            return None, None, base
        centers = _draw_centers(base, xy, _OVERLAY_DOT_RADIUS, _CENTER_COLOR)
        rings = _draw_rings(base, xy, cov2d, _OVERLAY_RING_K, _RING_COLOR) if cov2d is not None else None
        return centers, rings, base
    except Exception as e:
        print(f"[report] overlay render failed: {e}", flush=True)
        return None, None, None


def _make_overview_cameras(base_cams: List, gaussians, n: int = 4,
                           fov_scale: float = 1.0, margin: float = 1.12) -> List:
    """Synthesize overview cameras framed to *just fit the whole splat cloud*.

    For each of ``n`` representative base cameras, the rotation is kept (so the view
    direction is a real scene-facing angle) but the camera is placed looking at the
    cloud centroid at the distance where the **farthest ring boundary** —
    ``max(||xyz - centroid|| + k·max_scale)`` — exactly fills the FoV (times ``margin``).
    This guarantees the largest-extent Gaussian's ring is visible and the scene fills the
    frame, instead of an arbitrary fixed pull-back. Returns ``MiniCam``s.
    """
    if not base_cams:
        return []
    try:
        from scene.cameras import MiniCam
        from eligsir.utils.graphics_utils import getProjectionMatrix
    except Exception:
        return []

    xyz = gaussians.get_xyz.detach()
    if xyz.shape[0] == 0:
        return []
    centroid = xyz.mean(dim=0)
    dist_from_c = (xyz - centroid).norm(dim=1)
    try:
        ring_extent = _OVERLAY_RING_K * gaussians.get_scaling.detach().amax(dim=1)
    except Exception:
        ring_extent = torch.zeros_like(dist_from_c)
    R_fit = float((dist_from_c + ring_extent).max().item())   # farthest ring boundary
    R_fit = max(R_fit, 1e-3)

    idxs = np.linspace(0, len(base_cams) - 1, num=min(n, len(base_cams)), dtype=np.int64)
    seen, picks = set(), []
    for j in idxs.tolist():
        if j not in seen:
            seen.add(j); picks.append(base_cams[j])
    out = []
    for cam in picks:
        try:
            device = cam.world_view_transform.device
            V = cam.world_view_transform.transpose(0, 1).to(torch.float32)  # actual W2V
            Rcw = V[:3, :3]
            forward = Rcw.T @ torch.tensor([0.0, 0.0, 1.0], device=device)  # cam +Z in world
            fovx = 2.0 * math.atan(min(fov_scale * math.tan(cam.FoVx / 2.0), 10.0))
            fovy = 2.0 * math.atan(min(fov_scale * math.tan(cam.FoVy / 2.0), 10.0))
            half_fov = min(fovx, fovy) / 2.0
            dist = margin * R_fit / max(math.tan(half_fov), 1e-3)
            new_center = centroid - forward * dist           # look at centroid
            t_new = -(Rcw @ new_center)
            V_new = torch.eye(4, device=device, dtype=torch.float32)
            V_new[:3, :3] = Rcw; V_new[:3, 3] = t_new
            wvt = V_new.transpose(0, 1).contiguous()
            znear = max(float(getattr(cam, "znear", 0.01)), dist - 2.0 * R_fit, 1e-3)
            zfar = dist + 2.0 * R_fit + 1.0
            proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).to(device)
            full = (wvt.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
            mc = MiniCam(int(cam.image_width), int(cam.image_height),
                         fovy, fovx, znear, zfar, wvt, full)
            out.append(mc)
        except Exception as e:
            print(f"[report] overview camera build failed: {e}", flush=True)
    return out


def _batch_render_images(cams, gaussians, pipe, background,
                         batch_render_fn=None, chunk_size: int = 32) -> list:
    """Render a list of cameras in batches. Returns list of (3,H,W) tensors."""
    if not cams:
        return []
    if batch_render_fn is None:
        return []
    with torch.no_grad():
        return batch_render_fn(cams, gaussians, pipe, background, chunk_size=chunk_size)


def _restore_native_resolution_for_report(cams) -> None:
    """Re-bind any mapping-tiered camera back to its NATIVE view before scoring.

    Issue #1259 follow-up: ``eligsir/mapping/scheduler/work_budget.py``
    unconditionally forces ``ResolutionTier.NATIVE`` for post-mapping
    refinement/reporting -- but only for future mapping-plan decisions, not
    for camera objects already sitting in ``streaming_scene.getTrainCameras()``
    that were last materialized to a non-native mapping tier (e.g.
    ``quarter``) during the mapping phase. Nothing re-derives them before
    report time, and the renderer (``gaussian_renderer``'s ``render``/
    ``render_batch``) draws at whatever ``camera.image_width``/
    ``image_height`` currently are -- it has no notion of "the current
    scheduler tier", so a report render of a stale-tiered camera happens AT
    that stale, downsampled resolution too, not at native. The render and the
    cached GT therefore end up the *same shape* (this is not a plain
    shape mismatch); the failure is that the GT's ``score_provenance`` still
    carries the mapping-time tier's declared resize-transform step, while a
    freshly rendered tensor's provenance has no such step in its own
    history -- exactly the "Transform mismatch: GT and render use different
    declared score-grid transforms" error ``utils/score_contract.py``'s
    ``validate_score_pair`` correctly refuses to score.

    Restoring to native (rather than rendering at the camera's current tier
    and declaring a matching transform there) is also the only choice that
    keeps ``mean_train_psnr`` comparable to the native ``mean_test_psnr`` --
    scoring train at a downsampled tier would silently make any
    train-vs-holdout PSNR comparison apples-to-oranges.

    ``materialize_resolution_tier`` derives every tier solely from the
    camera's own cached ``_canonical_native_resolution_state`` (set once,
    the first time any tier is materialized for that camera) and is a no-op
    when the camera is already bound to the requested tier -- so calling it
    here with ``ResolutionTier.NATIVE`` is safe and idempotent for every
    camera, tiered or not, and does not depend on any state that only the
    live mapping path would have set up.
    """
    from eligsir.contracts.control import ResolutionTier
    from eligsir.mapping.resolution_materialization import materialize_resolution_tier

    for cam in cams or []:
        tier = getattr(cam, "_canonical_resolution_tier", None)
        if tier is None or tier is ResolutionTier.NATIVE:
            continue
        resampling_mode = getattr(cam, "_canonical_resolution_resampling_mode", None) or "legacy"
        try:
            materialize_resolution_tier(cam, ResolutionTier.NATIVE, resampling_mode=resampling_mode)
        except Exception as exc:  # pragma: no cover - defensive; scoring below still fails closed
            print(
                f"[report] WARNING: could not restore native resolution for "
                f"camera {getattr(cam, 'uid', getattr(cam, 'colmap_id', '?'))}: {exc}",
                flush=True,
            )


def _gt_image(cam) -> torch.Tensor:
    from eligsir.utils.score_contract import score_tensor_from_camera

    if getattr(cam, "original_image", None) is None:
        materialize = getattr(cam, "materialize_pending_image", None)
        if callable(materialize):
            materialize()
    if getattr(cam, "original_image", None) is None:
        raise AttributeError("camera has no materializable original_image")
    return score_tensor_from_camera(cam, device="cuda")


def _get_sensor_depth(cam, target_h: int, target_w: int) -> Optional[torch.Tensor]:
    """Return streaming sensor depth as a CPU [1, H, W] float tensor in metres.

    BUG FIX (issue #162, 2026-08-13): this was a long-standing duplicate of
    ``eligsir.mapping.integration.depth_helpers._get_sensor_depth`` that
    PR #185 did not touch (that PR only fixed the training chokepoint that
    actually reproduced the reported HSA_STATUS_ERROR_EXCEPTION). The exact
    same hazard applies here: `cam._sensor_depth_cache` can be populated by
    `bind_resolution_tier` (multires) or by an earlier call to this very
    function with a *different* `target_h/target_w`, at a depth-sensor
    resolution that differs from the color camera's (ScanNet: 640x480
    depth vs 1296x968 color). Returning the cached tensor unchecked handed
    callers a tensor shaped for the WRONG resolution tier. No known call
    site in this file currently feeds the result into a raw pixel-index
    gather (the exact operation that produced #162's GPU hardware fault in
    the training path, `_update_provisional_support`'s `sensor_d[0, vi,
    ui]`) -- both call sites here do elementwise/broadcast ops or resize
    via PIL first -- so this specific duplicate was not observed to
    reproduce the crash, but it is the same bug class and should not be
    trusted to stay silently safe as report code evolves. Mirrors
    depth_helpers.py's fix: resize-and-return WITHOUT writing back (a
    later tier switch would otherwise double-resample an already-stretched
    tensor).
    """
    cached = getattr(cam, "_sensor_depth_cache", None)
    if cached is False:
        return None
    if cached is not None:
        if cached.shape[-2] != target_h or cached.shape[-1] != target_w:
            return torch.nn.functional.interpolate(
                cached[None], (target_h, target_w), mode="nearest"
            )[0]
        return cached
    if not hasattr(cam, "_streaming_depth_path") and not hasattr(cam, "_streaming_frame"):
        return None
    try:
        from eligsir.utils.rgbd_frames import depth_to_meters
        frame = getattr(cam, "_streaming_frame", None)
        # `_depth_np` must be checked alongside `_depth_bytes`: a ros2://live
        # frame carries its depth ONLY as `_depth_np` (see
        # eligsir/io/sources/datasets/ros2_live.py's StreamingRGBDFrame
        # construction) with `depth_path=None` and no `_depth_bytes`, so the
        # old bytes-only guard fell through to the depth_path branch, found no
        # file, and returned None -- SILENTLY, since the caller only logs on an
        # exception. Consequence (issue #358): every live/bag-replay run wrote
        # holdout_gt/<id>_rgb.jpg but zero <id>_depth.png, which is exactly the
        # fixed sensor depth Level A requires, so no corridor/multi-room run on
        # disk can be measured. `load_frame_depth_np` already prefers `_depth_np`
        # itself; only this guard excluded it. Same bug shape as the earlier
        # live-insertion no-op (a guard that missed `_depth_np`).
        if frame is not None and (
            getattr(frame, "_depth_np", None) is not None
            or getattr(frame, "_depth_bytes", None) is not None
        ):
            from eligsir.utils.streaming_frames import load_frame_depth_np
            depth_raw = load_frame_depth_np(frame)
        else:
            depth_path = getattr(cam, "_streaming_depth_path", None)
            if depth_path is None or not os.path.exists(depth_path):
                cam._sensor_depth_cache = False
                return None
            from PIL import Image as _Image
            depth_raw = np.array(_Image.open(depth_path))
        if depth_raw is None:
            cam._sensor_depth_cache = False
            return None
        depth_scale = float(getattr(cam, "_streaming_depth_scale", 1000.0))
        depth_m = depth_to_meters(depth_raw, depth_scale).astype(np.float32)
    except Exception:
        cam._sensor_depth_cache = False
        return None
    d_t = torch.from_numpy(depth_m)[None, None]
    if d_t.shape[2] != target_h or d_t.shape[3] != target_w:
        d_t = torch.nn.functional.interpolate(d_t, (target_h, target_w), mode="nearest")
    cam._sensor_depth_cache = d_t[0]
    return cam._sensor_depth_cache


def _depth_colormap(depth_m: np.ndarray, vmax: float) -> np.ndarray:
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    norm = np.zeros_like(d, dtype=np.float32)
    if valid.any() and vmax > 0:
        norm[valid] = np.clip(d[valid] / vmax, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(norm * 4.0 - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(norm * 4.0 - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(norm * 4.0 - 1.0), 0.0, 1.0)
    out = (np.stack([r, g, b], axis=-1) * 255.0 + 0.5).astype(np.uint8)
    out[~valid] = 0
    return out


def _depth_edge_mask(depth_m: np.ndarray) -> np.ndarray:
    z = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(z) & (z > 0)
    if not valid.any():
        return np.zeros(z.shape[:2], dtype=bool)
    dzx = np.zeros_like(z, dtype=np.float32)
    dzy = np.zeros_like(z, dtype=np.float32)
    px = valid[:, 1:] & valid[:, :-1]
    py = valid[1:, :] & valid[:-1, :]
    dzx[:, 1:][px] = np.abs(z[:, 1:][px] - z[:, :-1][px])
    dzy[1:, :][py] = np.abs(z[1:, :][py] - z[:-1, :][py])
    vals = z[valid]
    thresh = max(0.025, float(np.percentile(vals, 95) - np.percentile(vals, 5)) * 0.015)
    boundary = np.zeros_like(valid, dtype=bool)
    boundary[:, 1:] |= valid[:, 1:] & ~valid[:, :-1]
    boundary[:, :-1] |= valid[:, :-1] & ~valid[:, 1:]
    boundary[1:, :] |= valid[1:, :] & ~valid[:-1, :]
    boundary[:-1, :] |= valid[:-1, :] & ~valid[1:, :]
    return (valid & ((dzx > thresh) | (dzy > thresh))) | boundary


def _depth_compare_frame(
    render_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    rendered_depth: torch.Tensor,
    sensor_depth: torch.Tensor,
    panel_width: int = 320,
) -> np.ndarray:
    r_u8 = _resize_to(_to_uint8_hwc(render_rgb), panel_width)
    g_u8 = _resize_to(_to_uint8_hwc(gt_rgb), panel_width)
    rd = rendered_depth.detach().cpu().numpy().squeeze().astype(np.float32)
    sd = sensor_depth.detach().cpu().numpy().squeeze().astype(np.float32)
    if rd.shape != sd.shape:
        from PIL import Image as _Image
        rd = np.asarray(_Image.fromarray(rd).resize((sd.shape[1], sd.shape[0]), _Image.Resampling.NEAREST))
    valid_vals = np.concatenate([rd[rd > 0], sd[sd > 0]])
    vmax = float(np.percentile(valid_vals, 98)) if valid_vals.size else 5.0
    vmax = max(vmax, 0.5)
    rd_u8 = _resize_to(_depth_colormap(rd, vmax), panel_width)
    sd_u8 = _resize_to(_depth_colormap(sd, vmax), panel_width)
    both = (rd > 0) & (sd > 0)
    diff = np.zeros_like(sd, dtype=np.float32)
    diff[both] = np.abs(rd[both] - sd[both])
    diff_u8 = _resize_to(_depth_colormap(diff, max(0.10, vmax * 0.25)), panel_width)
    edge_rgb = _to_uint8_hwc(gt_rgb)
    edges = _depth_edge_mask(sd)
    if edges.shape[:2] != edge_rgb.shape[:2]:
        from PIL import Image as _Image
        edges = np.asarray(
            _Image.fromarray(edges.astype(np.uint8) * 255).resize(
                (edge_rgb.shape[1], edge_rgb.shape[0]), _Image.Resampling.NEAREST
            )
        ) > 0
    edge_rgb = edge_rgb.copy()
    edge_rgb[edges] = np.array([255, 32, 32], dtype=np.uint8)
    edge_u8 = _resize_to(edge_rgb, panel_width)
    return _hconcat_with_gap([r_u8, g_u8, edge_u8, rd_u8, sd_u8, diff_u8], gap_px=4)


def _resize_to(arr: np.ndarray, target_w: int) -> np.ndarray:
    """Resize HWC uint8 to a target width while preserving aspect; cv2 if available."""
    if arr.shape[1] == target_w:
        return arr
    new_h = max(1, int(round(arr.shape[0] * target_w / arr.shape[1])))
    try:
        import cv2  # type: ignore
        return cv2.resize(arr, (target_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        # Nearest-neighbour fallback
        ys = (np.arange(new_h) * arr.shape[0] / new_h).astype(np.int64)
        xs = (np.arange(target_w) * arr.shape[1] / target_w).astype(np.int64)
        return arr[ys][:, xs]


def _build_contact_sheet(
    pairs: List[Tuple[str, np.ndarray, np.ndarray]],
    cols: int = 4,
    cell_width: int = 320,
) -> np.ndarray:
    """Grid of (label, render, gt). Render on top row of each cell, gt on bottom."""
    if not pairs:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    cols = max(1, min(cols, len(pairs)))
    rows = math.ceil(len(pairs) / cols)
    cell_rows: List[np.ndarray] = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(pairs):
                blank = np.full((1, cell_width, 3), 255, dtype=np.uint8)
                row_cells.append(blank)
                continue
            name, render, gt = pairs[idx]
            render = _resize_to(render, cell_width)
            gt = _resize_to(gt, cell_width)
            label = _label_strip(name, cell_width)
            cell = np.concatenate([label, render, gt], axis=0)
            row_cells.append(cell)
        row = _hconcat_with_gap(row_cells, gap_px=4)
        cell_rows.append(row)
    # Vertical concat with thin gap
    max_w = max(row.shape[1] for row in cell_rows)
    padded = []
    for row in cell_rows:
        if row.shape[1] < max_w:
            pad = np.full((row.shape[0], max_w - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded.append(row)
        padded.append(np.full((4, max_w, 3), 255, dtype=np.uint8))
    return np.concatenate(padded[:-1], axis=0)


def _resize_frame_for_video_canvas(frame: np.ndarray, w2: int, h2: int) -> np.ndarray:
    """Resize a uint8 HWC RGB frame onto a (w2, h2) canvas via PIL, not cv2.

    Used by the ffmpeg-pipe CPU/VAAPI encode paths in _write_mp4_cpu and
    _write_mp4_vaapi when a later frame's resolution differs from frame 0's
    (e.g. a multires tier switch mid-run binds a train camera to a
    different resolution) -- deliberately PIL rather than cv2.resize: found
    live (2026-08-14) that cv2's native extension loader is broken in the
    project's CPU-smoke/review containers ("recursion is detected during
    loading of cv2 binary extensions"), which crashed this resize path and,
    via the (already-fixed) exception propagation this call sits inside,
    took down the CPU fallback that exists specifically to keep video
    rendering working when hardware VAAPI encode isn't available -- the one
    environment where a working fallback matters most. PIL is already a
    direct dependency and has no such loader issue here.
    """
    from eligsir.utils.video_encode import resize_frame_for_video_canvas

    return resize_frame_for_video_canvas(frame, w2, h2)


def _write_mp4(
    path: str,
    frames_iter: Iterable[np.ndarray],
    fps: int = 30,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Optional[str]:
    """Write an MP4 from an iterable of uint8 HWC RGB frames.

    Tries VAAPI hardware encode first (h264_vaapi via /dev/dri/renderD128)
    when ffmpeg and a VAAPI render node are both present, falling back to
    the CPU libx264 path (all threads via -threads 0) on any VAAPI failure
    or when no VAAPI device is available. Falls back further to
    cv2.VideoWriter with mp4v if ffmpeg itself is unavailable. All three
    paths write frames at a canvas derived from the first frame's
    resolution: same-size frames are only cropped by 1 row/column if
    needed to make dimensions even (required by yuv420p's 2x2 chroma
    subsampling); a later frame whose shape differs from frame 0 (e.g. a
    multires tier switch mid-run binds a train camera to a different
    resolution) is resized -- never cropped -- onto that canvas so it
    still shows the full scene.

    ``min_width``/``max_width`` (issue #397): when either is set, the
    canvas is additionally passed through
    ``utils.trajectory_video.scale_dims_for_video`` -- aspect-preserving
    upscale to ``min_width`` if frame 0 is narrower, downscale to fit
    ``max_width`` if wider, no-op otherwise. This is a **post-render CPU
    resize**, applied to every frame (not just frame 0): it does not make
    the Gaussian rasterizer render more detail. Both default to ``None``
    (fully disabled, exact pre-existing behavior) so every caller other
    than the trajectory/depth-timeseries videos is unaffected.
    Returns an error string on failure, or None on success.
    """
    import shutil
    import subprocess

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if shutil.which("ffmpeg"):
        if os.path.exists("/dev/dri/renderD128"):
            # frames_iter is a one-shot generator in every real caller (see
            # tests/test_write_mp4_vaapi_fallback.py's docstring) -- if VAAPI
            # fails we must retry with the CPU path using the SAME frames,
            # not re-render. _write_mp4_vaapi is fed a tee'd iterator that
            # both streams frames to the encoder as they're produced (see
            # _stream_frames_to_proc) AND appends every frame it sees into
            # frames_buffer, so frames_buffer always ends up complete --
            # rendered fully, even if the VAAPI pipe breaks partway through
            # -- and can be reused by the CPU fallback below without
            # re-invoking the (already exhausted) generator.
            frames_buffer: List[np.ndarray] = []
            _src = iter(frames_iter)

            def _tee():
                for f in _src:
                    frames_buffer.append(f)
                    yield f

            err = _write_mp4_vaapi(path, _tee(), fps, min_width=min_width, max_width=max_width)
            if err is None:
                return None
            if err == "no frames written":
                return err
            print(f"[report] VAAPI encode failed ({err}), falling back to CPU libx264", flush=True)
            return _write_mp4_cpu(path, frames_buffer, fps, min_width=min_width, max_width=max_width)

        return _write_mp4_cpu(path, frames_iter, fps, min_width=min_width, max_width=max_width)

    # Fallback: cv2.VideoWriter
    try:
        import cv2  # type: ignore
    except Exception as e:
        return f"ffmpeg unavailable and cv2 unavailable: {e}"

    writer = None
    n = 0
    try:
        for frame in frames_iter:
            if writer is None:
                h, w = frame.shape[:2]
                w2, h2 = scale_dims_for_video(w, h, min_width=min_width, max_width=max_width)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, (w2, h2))
                if not writer.isOpened():
                    return "cv2.VideoWriter failed to open"
                _canvas_h, _canvas_w = h2, w2
            if frame.shape[:2] != (_canvas_h, _canvas_w):
                # Resize (not crop) mismatched frames -- e.g. a multires
                # tier switch mid-run means later train cameras can be
                # bound to a different resolution than the first sampled
                # frame, or the min/max-width policy above scaled the
                # canvas away from frame 0's native size; cropping would
                # silently truncate the visible scene instead of showing
                # the full (rescaled) frame.
                frame = cv2.resize(frame, (_canvas_w, _canvas_h), interpolation=cv2.INTER_AREA)
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
            n += 1
    finally:
        if writer is not None:
            writer.release()
    if n == 0:
        return "no frames written"
    return None


def _stream_frames_to_proc(
    proc,
    first: np.ndarray,
    remaining: Iterable[np.ndarray],
    h2: int,
    w2: int,
) -> Tuple[int, Optional[Exception]]:
    """Feed frames to ``proc.stdin`` from a dedicated writer thread while the
    calling thread keeps pulling from ``remaining`` -- overlaps upstream
    frame production (GPU render) with ffmpeg's CPU encode of frames already
    produced, instead of the old ``frames = list(frames_iter)`` behavior that
    fully materialized every frame before the encoder subprocess ever saw
    the first one (issue: post-mapping report phase render+encode
    serialization).

    Always consumes ``remaining`` to completion, even if a write to
    ``proc.stdin`` raises partway through -- callers that may retry the same
    frames on a different encoder (VAAPI -> CPU fallback, see
    tests/test_write_mp4_vaapi_fallback.py) rely on every frame having been
    pulled from a one-shot generator exactly once, regardless of whether the
    encode itself succeeded.

    Returns ``(n_frames_seen, first_write_error_or_None)``.
    """
    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=8)
    state: Dict[str, Optional[Exception]] = {"error": None}

    def _writer():
        while True:
            item = q.get()
            if item is None:
                return
            if state["error"] is not None:
                # Already broken -- keep draining so put() on the main
                # thread never blocks against a dead pipe.
                continue
            frame = item
            try:
                if frame.shape[:2] != (h2, w2):
                    # Resize (not crop) -- a multires tier switch mid-run can
                    # bind later train cameras to a different resolution
                    # than frame 0.
                    frame = _resize_frame_for_video_canvas(frame, w2, h2)
                else:
                    frame = frame[:h2, :w2]
                proc.stdin.write(frame.tobytes())
            except Exception as e:
                state["error"] = e

    thread = threading.Thread(target=_writer, daemon=True)
    thread.start()
    n = 0
    q.put(first)
    n += 1
    for frame in remaining:
        q.put(frame)
        n += 1
    q.put(None)
    thread.join()
    return n, state["error"]


def _write_mp4_cpu(
    path: str,
    frames: Iterable[np.ndarray],
    fps: int,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Optional[str]:
    """CPU libx264 encode, all threads (-threads 0). Canvas size is frame 0's
    (even-cropped, then optionally min/max-width-scaled -- see _write_mp4's
    docstring); mismatched later frames are resized onto it, matching
    _write_mp4_vaapi.

    ``frames`` may be a one-shot iterator: the first frame is pulled to size
    the canvas/ffmpeg command, and the rest are streamed to the encoder from
    a background writer thread (_stream_frames_to_proc) while this thread
    keeps pulling -- see that function's docstring for why."""
    import subprocess

    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        return "no frames written"
    h, w = first.shape[:2]
    w2, h2 = scale_dims_for_video(w, h, min_width=min_width, max_width=max_width)
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{w2}x{h2}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-threads", "0",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        # The iterator may be a tee (see _write_mp4) whose caller expects it
        # fully drained (into its buffer) regardless of what happens here.
        for _ in it:
            pass
        return f"ffmpeg Popen failed: {e}"
    n, write_err = _stream_frames_to_proc(proc, first, it, h2, w2)
    if write_err is not None:
        # A write failure (e.g. the ffmpeg subprocess already died) commonly
        # leaves stdin in a broken state where close()/wait() themselves
        # raise a second time (BrokenPipeError) -- letting that escape
        # would crash this whole function instead of returning the error
        # string _write_mp4's caller relies on to fall back further.
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
        return f"ffmpeg pipe error: {write_err}"
    proc.stdin.close()
    proc.wait()
    if n == 0:
        return "no frames written"
    if proc.returncode != 0:
        return f"ffmpeg exited with code {proc.returncode}"
    return None


def _write_mp4_vaapi(
    path: str,
    frames: Iterable[np.ndarray],
    fps: int,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
) -> Optional[str]:
    """Hardware-accelerated H.264 encode via VAAPI (/dev/dri/renderD128).

    Same canvas-sizing behavior as _write_mp4_cpu: frame 0's size
    (even-cropped, then optionally min/max-width-scaled) is the canvas;
    mismatched later frames are resized onto it rather than cropped.

    ``frames`` may be a one-shot iterator (see _write_mp4_cpu's matching
    docstring) -- streamed to the encoder via _stream_frames_to_proc.
    """
    import subprocess

    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        return "no frames written"
    h, w = first.shape[:2]
    w2, h2 = scale_dims_for_video(w, h, min_width=min_width, max_width=max_width)
    try:
        proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-nostats", "-loglevel", "error",
                "-vaapi_device", "/dev/dri/renderD128",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{w2}x{h2}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-vf", "format=nv12,hwupload",
                "-c:v", "h264_vaapi",
                "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except Exception as e:
        # The iterator may be a tee (see _write_mp4) whose caller expects it
        # fully drained (into its buffer) regardless of what happens here.
        for _ in it:
            pass
        return f"vaapi Popen failed: {e}"
    n, write_err = _stream_frames_to_proc(proc, first, it, h2, w2)
    if write_err is not None:
        # Same rationale as _write_mp4_cpu's matching handler: a write
        # failure often leaves stdin already broken, so close() itself can
        # raise a second time (BrokenPipeError) -- must not let that escape
        # here, since _write_mp4 depends on this function always returning
        # an error string (never raising) to fall back to the CPU path.
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
        return f"vaapi pipe error: {write_err}"
    # communicate() closes stdin itself (and would try to flush it if we
    # closed it ourselves first, raising "ValueError: flush of closed
    # file") -- do not call proc.stdin.close() here.
    _, stderr = proc.communicate()
    if n == 0:
        return "no frames written"
    if proc.returncode != 0:
        stderr_tail = stderr.decode(errors="replace")[-500:] if stderr else ""
        return f"ffmpeg vaapi exited with code {proc.returncode}: {stderr_tail}"
    return None


def _infer_realtime_fps(cams: List, default_fps: int, max_fps: int = 60) -> int:
    """Infer video FPS from camera timestamps when available.

    Streaming datasets often carry real frame timestamps. Encoding one frame per
    camera at a fixed 30 fps fast-forwards low-FPS sources, so use the median
    timestamp spacing when it is sane.
    """
    timestamps = []
    for cam in cams:
        ts = getattr(cam, "_streaming_timestamp", None)
        if ts is None:
            return int(default_fps)
        try:
            timestamps.append(float(ts))
        except Exception:
            return int(default_fps)
    if len(timestamps) < 2:
        return int(default_fps)
    deltas = np.diff(np.asarray(timestamps, dtype=np.float64))
    deltas = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
    if deltas.size == 0:
        return int(default_fps)
    fps = int(round(1.0 / float(np.median(deltas))))
    return max(1, min(int(max_fps), fps))


def is_degenerate_split(train_cams, test_cams, degenerate_split: bool = False) -> bool:
    """True when the test split is not a real holdout (same cams as train).

    A degenerate split must never be reported as a generalization metric — the
    bootstrap-views debug report deliberately passes ``test_cams is train_cams``.
    """
    if degenerate_split:
        return True
    if test_cams is train_cams:
        return True
    if test_cams and train_cams and len(test_cams) == len(train_cams):
        return all(a is b for a, b in zip(test_cams, train_cams))
    return False


# _camera_set_hash / _realized_holdout_hash now live in
# utils/baseline_comparison.py (torch-free, issue #423) and are re-exported
# here so every existing import site keeps working unchanged.
from eligsir.utils.baseline_comparison import (  # noqa: F401,E402
    _camera_set_hash,
    _realized_holdout_hash,
)

def _scene_fingerprint(cams) -> Optional[str]:
    """Content hash identifying the SCENE a run was evaluated on.

    sha1 over 32x32x3 uint8 downsampled GT images of up to 3 cameras
    (first/middle/last of the given list). Robust to run names and bag
    paths (which never reach cfg_args on the ros2://live replay path);
    two runs on the same bag + holdout produce the same fingerprint,
    two different bags cannot.
    """
    import hashlib

    if not cams:
        return None
    picks = sorted({0, len(cams) // 2, len(cams) - 1})
    h = hashlib.sha1()
    for i in picks:
        cam = cams[i]
        img = cam.original_image
        if img is None:
            continue
        # Device-normalized on purpose (issue #398): this hash is compared
        # ACROSS runs, and `interpolate(mode="area")` is not guaranteed to be
        # bit-identical between the CPU and CUDA kernels. Before #398 test
        # cameras were always GPU-resident here, so the historical fingerprints
        # in every existing report.json are CUDA-computed values; keeping
        # holdout images CPU-resident during mapping would silently have moved
        # this computation onto the CPU kernel and could have changed the
        # fingerprint of an otherwise identical run. Uploading up to 3 images,
        # one at a time, reproduces the historical value exactly and makes the
        # fingerprint independent of any future residency policy.
        #
        # NOT purely conservative on the `train_cams` FALLBACK path (used only
        # when `test_cams` is empty, i.e. the `*_bootstrap_views/` debug
        # reports): there the picks were previously a MIX of devices, because
        # the hot-window eviction has already offloaded the first/middle
        # cameras while the last is still GPU-resident -- so that path's
        # fingerprint may differ from a pre-#398 run. Accepted deliberately:
        # the alternative (leave this alone) moves the PRIMARY, test-camera
        # path instead, which feeds the hard comparability gate in
        # scripts/analyze_issue285_knockouts.py.
        t = img.detach().float()
        if torch.cuda.is_available() and t.device.type == "cpu":
            t = t.to("cuda")
        if t.dim() == 3:  # (C,H,W) -> (1,C,H,W)
            t = t.unsqueeze(0)
        small = torch.nn.functional.interpolate(t, size=(32, 32), mode="area")
        arr = (small.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Partitioned evaluation (per-scene trajectory bins)
# ---------------------------------------------------------------------------
# Helpers are shared with the in-training sentinel telemetry
# (train_streaming.py) and live in utils/partition_eval.py.

from eligsir.utils.partition_eval import (  # noqa: E402
    PARTITIONS,
    camera_source_index,
    compute_partition_stats,
    compute_trajectory_coverage,
    partition_id,
    registry_n_source_frames as _registry_n_source_frames,
)

def render_camera_path(
    cams: List,
    gaussians,
    render_fn,
    pipe,
    background,
    *,
    out_path: str,
    fps: int = 30,
    batch_render_fn=None,
    chunk_size: int = 8,
    min_width: Optional[int] = None,
    max_width: Optional[int] = None,
    frame_repeats: Optional[List[int]] = None,
    supersample_factor: int = 1,
    sharpen_strength: float = 0.0,
    depth_cam_ids: Optional[set] = None,
    render_cache_out: Optional[dict] = None,
) -> Optional[str]:
    """Render ``cams`` in order and encode them to an MP4 at ``out_path``.

    Standalone version of the trajectory-mp4 section of
    ``write_post_training_report`` — extracted so both the inline
    (in-training) report and a standalone post-hoc renderer call the exact
    same chunked-render/encode code (no duplication/drift risk). Renders in
    small chunks via ``batch_render_fn`` when given (falls back to
    per-camera ``render_fn``), converting each chunk to CPU uint8 before the
    next is fetched so the whole trajectory is never GPU-resident at once.

    ``min_width``/``max_width`` (issue #397): forwarded to ``_write_mp4``'s
    post-render CPU upscale/downscale policy (see its docstring). Both
    default to ``None`` (disabled) here so callers that don't pass them
    keep the exact pre-existing raw-render-resolution behavior.

    ``frame_repeats`` (issue #397, optional): one repeat-count per entry of
    ``cams`` (see ``utils.trajectory_video.distribute_frame_repeats``) --
    each rendered frame is written to the encoder that many times in a
    row, stretching a bounded number of unique RENDERS over the real
    capture duration at a conventional ``fps`` without rendering more
    frames than ``len(cams)``. ``None`` (the default) repeats every frame
    exactly once -- the exact pre-existing behavior for every caller other
    than the trajectory-video call site.

    ``supersample_factor``/``sharpen_strength`` (issue #1323, presentation-only SSAA +
    unsharp-mask sharpening -- see ``utils/presentation_ssaa.py``): each camera is rendered
    at ``supersample_factor``x linear resolution, then area-downsampled back to native size
    and mildly sharpened before being written to the video. Defaults (``1``/``0.0``) are
    exact no-ops preserving pre-existing behavior for any caller that doesn't pass them.
    This is purely a post-render/pre-encode transform on the returned frame -- it never
    touches ``cams`` themselves or anything a scored metric reads.

    When the caller supplies both ``depth_cam_ids`` and ``render_cache_out``,
    a non-batched, non-SSAA camera in that exact identity set is rendered once
    with ``render_depth=True``. The resulting RGB/depth package feeds this
    trajectory frame and is retained for the depth montage. The RGB path uses
    the renderer's depth-capable composition, so it preserves displayed
    semantics but is not promised bit-identical to a plain RGB render.
    The cache is local to this one synchronous report call: it never crosses
    a map, pose, exposure, renderer, or camera-state revision.

    Returns the written path, or ``None`` if ``cams`` is empty or the write
    failed (error is printed, not raised).
    """
    if not cams:
        print("[render_camera_path] no cameras given — skipping", flush=True)
        return None
    if frame_repeats is not None and len(frame_repeats) != len(cams):
        print(
            f"[render_camera_path] frame_repeats length {len(frame_repeats)} != "
            f"len(cams) {len(cams)} -- ignoring, repeating each frame once",
            flush=True,
        )
        frame_repeats = None

    from eligsir.utils.presentation_ssaa import make_supersampled_camera, presentation_postprocess
    ssaa = int(supersample_factor) if supersample_factor else 1

    def _frames():
        idx = 0
        for start in range(0, len(cams), chunk_size):
            chunk = cams[start:start + chunk_size]
            render_chunk = [make_supersampled_camera(c, ssaa) for c in chunk] if ssaa > 1 else chunk
            chunk_imgs = []
            if batch_render_fn is not None:
                try:
                    chunk_imgs = batch_render_fn(render_chunk, gaussians, pipe, background,
                                                  chunk_size=chunk_size)
                except Exception as e:
                    print(f"[render_camera_path] batch render failed, falling back: {e}", flush=True)
                    chunk_imgs = []
                if len(chunk_imgs) != len(chunk):
                    chunk_imgs = []
            for i, cam in enumerate(chunk):
                try:
                    if chunk_imgs:
                        img = chunk_imgs[i].clamp(0.0, 1.0)
                    elif (
                        depth_cam_ids is not None
                        and render_cache_out is not None
                        and render_chunk[i] is cam
                        and id(cam) in depth_cam_ids
                    ):
                        with torch.no_grad():
                            out = render_fn(cam, gaussians, pipe, background, render_depth=True)
                        if isinstance(out, dict) and out.get("rendered_depth") is not None:
                            render_cache_out[id(cam)] = out
                            img = out["render"]
                        else:
                            # No depth package means no compatible shared
                            # modality. Keep the trajectory's plain RGB path.
                            img = _render_image(render_chunk[i], gaussians, render_fn, pipe, background)
                    else:
                        img = _render_image(render_chunk[i], gaussians, render_fn, pipe, background)
                except Exception as e:
                    print(f"[render_camera_path] render failed for {getattr(cam, 'image_name', '?')}: {e}",
                          flush=True)
                    idx += 1
                    continue
                frame = _to_uint8_hwc(img)
                del img
                if ssaa > 1 or sharpen_strength:
                    frame = presentation_postprocess(
                        frame, supersample_factor=ssaa, sharpen_strength=sharpen_strength)
                repeats = 1 if frame_repeats is None else max(1, int(frame_repeats[idx]))
                for _ in range(repeats):
                    yield frame
                idx += 1
            del chunk_imgs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_frames = len(cams) if frame_repeats is None else sum(max(1, int(r)) for r in frame_repeats)
    err = _write_mp4(out_path, _frames(), fps=fps, min_width=min_width, max_width=max_width)
    if err is not None:
        print(f"[render_camera_path] {out_path} SKIPPED: {err}", flush=True)
        return None
    print(
        f"[render_camera_path] {out_path} written "
        f"({len(cams)} rendered, {total_frames} encoded frames @ {fps}fps)",
        flush=True,
    )
    return out_path

# ---------------------------------------------------------------------------
# --baseline_run comparison (opt-in; see CLAUDE.md's "Compare PSNR curves,
# not just the final number" run contract)
#
# The implementation lives in utils/baseline_comparison.py -- it is entirely
# torch-free, and keeping it out of this (torch-importing) module is what lets
# the CPU-only smoke image collect tests/test_baseline_comparison.py and
# tests/test_scored_subset_hash.py (issue #423). Re-exported here so every
# existing import site keeps working unchanged.
# ---------------------------------------------------------------------------

from eligsir.utils.baseline_comparison import (  # noqa: F401,E402
    _BASELINE_SMALL_DELTA_DB,
    _BASELINE_LARGE_DELTA_DB,
    _BASELINE_MIN_ALIGNED_POINTS,
    _align_psnr_curves,
    _classify_curve_shape,
    _find_latest_report_json,
    compare_against_baseline,
)

def depth_insertion_report_block(enabled, total_inserted, frames_admitted) -> dict:
    """Build the Phase 2 depth-insertion sanity-gate block (issue #303).

    ``inert`` is True iff depth insertion was ENABLED and frames WERE
    admitted, yet nothing was ever inserted -- the signature of a silently
    dropped depth stream (the compressedDepth transport bug this gate ships
    with, or the earlier ``live-insertion-noop-bug``). Requiring
    ``frames_admitted > 0`` keeps a run that admitted nothing for an
    unrelated reason from being mislabeled a depth-decode failure, and
    requiring ``enabled`` keeps a deliberately depth-insertion-free run
    (``--no-streaming_insert_from_depth``) from tripping it.

    Pure function so the gate's logic is testable without a training run;
    ``write_post_training_report``'s ``depth_insertion`` parameter consumes
    the result.
    """
    enabled = bool(enabled)
    total_inserted = int(total_inserted or 0)
    frames_admitted = int(frames_admitted or 0)
    return {
        "enabled": enabled,
        "total_inserted": total_inserted,
        "frames_admitted": frames_admitted,
        "inert": bool(enabled and frames_admitted > 0 and total_inserted <= 0),
    }


def lineage_preserving_setup_report_block(events) -> "Optional[dict]":
    """Build the ``geometry.lineage_preserving_setup`` block (issue #374 item 3).

    ``events`` is ``getattr(gaussians, "_lineage_preserving_setup_events", [])``
    -- the list ``GsplatGaussianModel._restore_lineage_snapshot`` appends to
    every time ``training_setup(preserve_lineage=True)`` runs (currently only
    ``rolling_seed``/``submap_stitch``'s mid-run optimizer rebuild, see that
    method's docstring). Each event already carries ``iteration``/``site``/
    ``n_gaussians``/``restored``/``skipped``/``next_gaussian_uid``.

    Returns ``None`` for an empty/falsy ``events`` (the overwhelming majority
    of runs, which never call ``preserve_lineage=True`` at all) so the caller
    can use the same ``if lineage_preserving_setup:`` gate every other
    optional ``geometry`` sub-block uses (``depth_insertion``,
    ``mapping_clocks``, ...) -- omitting the key reproduces the exact
    ``report.json`` from before this field existed.

    Pure function so the block is testable without a training run;
    ``write_post_training_report``'s ``lineage_preserving_setup`` parameter
    consumes the result.
    """
    events = list(events or [])
    if not events:
        return None
    return {
        "count": len(events),
        "iterations": [e.get("iteration") for e in events],
        "sites": sorted({e["site"] for e in events if e.get("site")}),
        "events": events,
    }


def seed_cohort_report_block(gaussians) -> dict:
    """Build the ALWAYS-PRESENT ``geometry.seed_cohort`` block (issue #346).

    Records whether the EXACT bootstrap cohort (issue #283's
    ``utils.gaussian_lineage.seed_cohort_mask``) is determinable for this run,
    and if so its counts. Unlike every other optional ``geometry`` sub-block,
    this one is emitted unconditionally, including on failure -- the whole
    point is that a future session can tell "this run is seed-fade-cohort
    analyzable" from "it is not, and here is why" **from report.json alone**,
    without opening a multi-GB ``chkpnt*.pth``.

    Why that matters (the finding this function exists to close): an audit of
    all 174 checkpoint-bearing run directories in ``output/`` on 2026-08-13
    found only 13 carrying per-Gaussian lineage buffers at all (every one
    written after #145 landed on 2026-08-12) and **zero** carrying the
    ``seed_uid_range`` key #283 needs -- i.e. under the shipped
    ``GsplatGaussianModel.restore()`` the exact cohort is UNDETERMINED for
    every historical run in the repo. None of that was discoverable from any
    run artifact; it took a byte-level scan of the checkpoint archives. See
    ``scripts/lineage_coverage_audit.py`` and the manifest it regenerates.

    ``available`` is False (with a human-readable ``reason`` lifted verbatim
    from ``SeedCohortUnavailable``) for a PLY-loaded model, a pre-#283
    checkpoint restore, a mid-run ``training_setup()`` re-bootstrap, or the
    legacy ``GaussianModel`` layout -- the four cases
    ``seed_cohort_mask``'s docstring enumerates. It never guesses.

    Pure apart from reading the model, so it is testable without a training
    run; ``write_post_training_report``'s ``seed_cohort`` parameter consumes
    the result.
    """
    from eligsir.utils.gaussian_lineage import SeedCohortUnavailable, seed_cohort_summary

    if gaussians is None:
        return {"available": False, "reason": "no Gaussian model was supplied to the report"}
    try:
        summary = seed_cohort_summary(gaussians)
    except SeedCohortUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    except Exception as exc:  # never let telemetry break a report
        return {"available": False, "reason": f"unexpected error computing seed cohort: {exc!r}"}
    return {"available": True, "reason": "", **summary}


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> [w, x, y, z] quaternion (Shepperd's method)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / (np.linalg.norm(q) + 1e-12)
    theta0 = math.acos(dot)
    theta = theta0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def _interpolate_camera(cam_a, cam_b, t: float):
    """Synthetic render-only camera at parameter ``t`` in [0, 1] between
    ``cam_a`` (t=0) and ``cam_b`` (t=1) -- slerp rotation, lerp camera
    center. Reuses intrinsics (width/height/FoV) from ``cam_a``, since
    consecutive trajectory cameras share the same capture device. Only
    carries the attributes ``gaussian_renderer...render()`` actually reads
    (``world_view_transform``, ``camera_center``, ``image_width``,
    ``image_height``, ``FoVx``, ``FoVy``) -- not a full ``Camera``.
    """
    import types
    c2w_a = cam_a.world_view_transform.inverse()
    c2w_b = cam_b.world_view_transform.inverse()
    Ra = c2w_a[:3, :3].detach().cpu().numpy()
    Rb = c2w_b[:3, :3].detach().cpu().numpy()
    q = _slerp_quat(_rotmat_to_quat(Ra), _rotmat_to_quat(Rb), t)
    R = _quat_to_rotmat(q)

    center = (1.0 - t) * c2w_a[3, :3] + t * c2w_b[3, :3]
    c2w = c2w_a.clone()
    c2w[:3, :3] = torch.as_tensor(R, dtype=c2w.dtype, device=c2w.device)
    c2w[3, :3] = center
    wvt = c2w.inverse()

    cam = types.SimpleNamespace()
    cam.world_view_transform = wvt
    cam.camera_center = wvt.inverse()[3, :3]
    cam.image_width = cam_a.image_width
    cam.image_height = cam_a.image_height
    cam.FoVx = cam_a.FoVx
    cam.FoVy = cam_a.FoVy
    cam.image_name = f"{getattr(cam_a, 'image_name', 'cam')}_interp{t:.2f}"
    return cam


def _expand_cams_with_interpolation(
    cams: List,
    frame_repeats: Optional[List[int]],
    max_extra_renders: Optional[int] = DEFAULT_MAX_INTERPOLATION_RENDERS,
    max_per_slice: Optional[int] = DEFAULT_MAX_INTERPOLATION_PER_SLICE,
):
    """Turn each real-timestamped slice's repeated-hold frames into a smooth
    interpolated sweep (issue #397 follow-up: "trajectory.mp4 looks
    chonky/stuttery"), WITHOUT changing the total encoded-frame count.

    ``cams``/``frame_repeats`` are the exact inputs ``render_camera_path``
    would otherwise receive (one real camera + its repeat count per
    ``distribute_frame_repeats`` slice). Returns ``(new_cams,
    new_frame_repeats)`` where ``new_cams`` interleaves ``_interpolate_camera``
    poses between consecutive real cameras and ``new_frame_repeats`` is the
    corresponding per-entry repeat count -- ``sum(new_frame_repeats) ==
    sum(frame_repeats)`` always (see ``split_repeats_evenly``), so total
    video duration is exactly preserved; only which pose backs each output
    frame changes.

    A no-op (returns ``cams``/``frame_repeats`` unchanged) when
    ``frame_repeats`` is ``None`` -- callers not on the timestamped-pacing
    path never see interpolated frames from this function. These
    interpolated cameras carry no ``_streaming_timestamp`` (same as
    ``_interpolate_camera`` everywhere else), so they must only ever reach
    the renderer here, at the very end of the pacing decision -- never
    ``_infer_realtime_fps``, PSNR/SSIM/LPIPS scoring, or any other
    timing/metrics code path (all of which already run on the un-expanded
    ``cams``/timestamps before this is called).
    """
    if frame_repeats is None or not cams:
        return list(cams), frame_repeats
    if len(frame_repeats) != len(cams):
        # Same defensive contract as render_camera_path's own frame_repeats
        # length check -- degrade to a no-op rather than indexing out of
        # range below.
        print(
            f"[_expand_cams_with_interpolation] frame_repeats length {len(frame_repeats)} != "
            f"len(cams) {len(cams)} -- skipping interpolation, repeating each frame verbatim",
            flush=True,
        )
        return list(cams), frame_repeats
    interp_counts = compute_interpolation_render_counts(
        frame_repeats, max_extra_renders=max_extra_renders, max_per_slice=max_per_slice)
    new_cams: List = []
    new_repeats: List[int] = []
    n = len(cams)
    for i in range(n):
        m = int(interp_counts[i]) + 1
        sub = split_repeats_evenly(int(frame_repeats[i]), m)
        new_cams.append(cams[i])
        new_repeats.append(sub[0])
        if m > 1 and i + 1 < n:
            for k in range(1, m):
                new_cams.append(_interpolate_camera(cams[i], cams[i + 1], k / m))
                new_repeats.append(sub[k])
    return new_cams, new_repeats


def _pace_cams_for_trajectory(cams: List, target_percentile: float = 50.0, max_steps: int = 12) -> List:
    """Insert interpolated in-between frames so the rendered trajectory
    video plays back at a roughly constant apparent speed, instead of
    jumping fast across the (spatially) large gaps that remain even after
    ``_order_cams_for_trajectory`` -- a real capture has irregular spacing
    between consecutive kept frames (e.g. subsampling, walking-pace
    variation), which at a fixed frames-per-camera-per-video-frame render
    reads as erratic speed-ups.

    ``target_percentile`` sets the "normal walking speed" reference gap
    (median, by default) that every OTHER gap gets paced to match by
    inserting ``round(gap / target) - 1`` interpolated frames.
    ``max_steps`` caps a single pathologically large gap from turning into
    a long slow-motion stall. Real captured cameras are never dropped or
    moved -- every one still appears in the output, exactly at its own
    original pose; only synthetic in-between frames are added around them.
    """
    if len(cams) < 3:
        return list(cams)
    centers = [c.camera_center for c in cams]
    dists = [float(torch.linalg.norm(centers[i + 1] - centers[i])) for i in range(len(cams) - 1)]
    nonzero = [d for d in dists if d > 1e-6]
    if not nonzero:
        return list(cams)
    target = float(np.percentile(nonzero, target_percentile))
    if target <= 1e-6:
        return list(cams)

    out = [cams[0]]
    for i, d in enumerate(dists):
        n_steps = max(1, min(max_steps, int(round(d / target))))
        for k in range(1, n_steps + 1):
            if k < n_steps:
                out.append(_interpolate_camera(cams[i], cams[i + 1], k / n_steps))
            else:
                out.append(cams[i + 1])
    return out


def _order_cams_for_trajectory(cams: List) -> List:
    """Greedy nearest-neighbor reorder by ``camera_center`` so
    ``trajectory.mp4`` sweeps through the scene continuously instead of
    jumping between cameras in whatever order the source dataset happens
    to list them in.

    Found live (2026-08-13, pixel10a RealityScan Mobile capture): source
    images are individual per-shot JPG+XMP pairs named by a random UUID,
    and ``realityscan_importer.project.discover_images`` enumerates them
    via ``sorted(glob("*.jpg"))`` -- i.e. alphabetically by UUID, which
    has no relation to capture order. Neither the JPG (EXIF stripped)
    nor the XMP sidecar (RealityScan pose-prior metadata only) carries a
    usable capture timestamp to sort by instead, so this is a spatial
    heuristic (nearest unvisited camera center), not a recovery of the
    true chronological order -- good enough to make the flythrough look
    continuous, but can still take a suboptimal path where the real
    trajectory doubles back on itself.

    Deliberately scoped to the camera list handed to ``render_camera_path``
    only -- does NOT touch ``train_cams``/``test_cams`` or any train/test
    split, so it cannot change PSNR, ``holdout_hash``, or any other
    training/eval metric.
    """
    if len(cams) <= 2:
        return list(cams)
    remaining = list(cams)
    centers = [torch.as_tensor(c.camera_center, dtype=torch.float32).cpu() for c in remaining]
    ordered = [remaining.pop(0)]
    ordered_centers = [centers.pop(0)]
    while remaining:
        last = ordered_centers[-1]
        dists = torch.stack([torch.linalg.norm(last - c) for c in centers])
        nn_idx = int(torch.argmin(dists))
        ordered.append(remaining.pop(nn_idx))
        ordered_centers.append(centers.pop(nn_idx))
    return ordered


def git_provenance_from_args(args) -> Optional[dict]:
    """Build the ``git_provenance`` kwarg for ``write_post_training_report()``
    from an argparse-style ``args`` namespace that already carries
    ``git_commit``/``git_branch``/``git_dirty`` (set once at process startup
    by ``train.py``'s ``get_git_metadata()``, __main__-level, before any
    dispatch to the streaming or legacy training loop).

    Issue #259: ablator runners execute jobs from a mutable, shared git
    checkout with nothing pinning a job to a commit -- a spec re-run minutes
    apart, or a concurrent agent switching branches in the same checkout,
    can silently execute different code under an identically-worded job.
    `cfg_args`/`report.json` recorded only *arguments* before this existed,
    never the commit that actually ran. This is deliberately a passive
    record, not a gate -- it does not refuse or alter a run, only makes a
    drifted run auditable after the fact (see `scripts/audit_run_drift.py`
    for the complementary ablator-queue-side audit).

    Returns ``None`` (never partial) if ``args`` lacks these attributes
    entirely, so a call site that forgot to import/set them degrades to the
    pre-existing behavior rather than writing a misleading empty dict.

    ``git_dirty`` (as of the 2026-08-14 fix, see
    ``utils.git_metadata.get_git_metadata()``'s docstring) is computed against TRACKED files
    only, so it is no longer pinned to ``True`` by untracked scratch/
    worktree litter. ``git_dirty_paths`` (optional on ``args`` -- older
    call sites / hand-built namespaces may not set it) carries the list of
    dirty tracked paths that produced the boolean, when available, so a
    reader can see *what* changed rather than just *that* something did.

    Issue #784: when git provenance capture failed, ``unavailable_reason``
    records the verbatim error (classified by mechanism: git binary missing,
    safe.directory refusal, linked-worktree gitdir unreachable, etc.) so a
    reader can distinguish a real checkout state change from a container/image
    limitation without re-reading train.log. Follows the ``seed_cohort``
    precedent of including a reason field alongside the unavailability.
    """
    if args is None:
        return None
    commit = getattr(args, "git_commit", None)
    branch = getattr(args, "git_branch", None)
    dirty = getattr(args, "git_dirty", None)
    dirty_paths = getattr(args, "git_dirty_paths", None)
    unavailable_reason = getattr(args, "git_provenance_unavailable_reason", None)
    if commit is None and branch is None and dirty is None:
        return None
    result = {
        "git_commit": commit if commit is not None else "unknown",
        "git_branch": branch if branch is not None else "unknown",
        "git_dirty": dirty if dirty is not None else "unknown",
    }
    if dirty_paths is not None:
        result["git_dirty_paths"] = dirty_paths
    if unavailable_reason:
        result["unavailable_reason"] = unavailable_reason
    return result


def determinism_from_args(args) -> Optional[dict]:
    """Build the ``determinism`` kwarg for ``write_post_training_report()``
    from an argparse-style ``args`` namespace that carries
    ``strict_deterministic`` (the ``--strict_deterministic``/
    ``--no-strict_deterministic`` CLI flag, ``ModelParams.strict_deterministic``
    in ``arguments/__init__.py``, default ``False``).

    Audit §14: ``utils.general_utils.safe_state`` already implemented a real
    two-mode determinism split (``strict_deterministic=True`` forces
    ``torch.use_deterministic_algorithms(True)`` plus cudnn
    benchmark-off/deterministic-on), but before this there was no CLI flag to
    select it and no record of which mode a run actually used, so even
    editing the source default wouldn't have been auditable after the fact.
    This mirrors ``git_provenance_from_args``'s pattern: a passive record of
    what actually ran, not a gate.

    Returns ``None`` (never partial) if ``args`` lacks the attribute
    entirely, so a call site built from a hand-rolled namespace degrades to
    the pre-existing (absent-key) behavior rather than writing a misleading
    default.
    """
    if args is None:
        return None
    strict_deterministic = getattr(args, "strict_deterministic", None)
    if strict_deterministic is None:
        return None
    return {"strict_deterministic": bool(strict_deterministic)}


def write_post_training_report(
    model_path: str,
    iteration: int,
    gaussians,
    train_cams: List,
    test_cams: List,
    render_fn,
    pipe,
    background,
    *,
    tb_writer=None,
    mp4_max_frames: int = 300,
    mp4_fps: int = 30,
    contact_sheet_cols: int = 4,
    contact_sheet_cell_width: int = 320,
    log_prefix: str = "report",
    subdir: Optional[str] = None,
    batch_render_fn=None,
    train_metrics_max: Optional[int] = None,
    test_metrics_max: Optional[int] = None,
    trajectory_max_frames: Optional[int] = None,
    degenerate_split: bool = False,
    eval_metadata: Optional[dict] = None,
    expressiveness: Optional[dict] = None,
    input_quality: Optional[dict] = None,
    pose_provenance: Optional[dict] = None,
    depth_max_m: float = 10.0,
    resumed_from_iteration: Optional[int] = None,
    seed_fade: Optional[dict] = None,
    frame_admission: Optional[str] = None,
    # Issue #600: SoftAllAdmitter's realized behavior plus the insertion
    # invariant counter (`inserted_from_skipped_frame`, which must read 0).
    # None for any run that didn't wire it.
    soft_all_admission: Optional[dict] = None,
    frames_admitted_override: Optional[int] = None,
    coverage_counters: Optional[dict] = None,
    reference_map_path: Optional[str] = None,
    map_comparison_align: bool = True,
    streaming_scene=None,
    offsurface_tau_m: float = 0.05,
    stream_end_reason: Optional[str] = None,
    tracking_reinit_events: Optional[list] = None,
    mapping_clocks: Optional[dict] = None,
    depth_insertion: Optional[dict] = None,
    lineage_preserving_setup: Optional[dict] = None,
    seed_cohort: Optional[dict] = None,
    exposure_opacity_survival: Optional[dict] = None,
    multires_cache: Optional[dict] = None,
    gpu_memory: Optional[dict] = None,
    resource_monitor: Optional[dict] = None,
    duty_cycle: Optional[dict] = None,
    transfer_pool: Optional[dict] = None,
    source_ingress: Optional[dict] = None,
    bag_replay_report: Optional[dict] = None,
    baseline_run: Optional[str] = None,
    ingestion: Optional[dict] = None,
    replica_rgb: Optional[dict] = None,
    last_ingest_iter: Optional[int] = None,
    true_final_iteration: Optional[int] = None,
    mapping_phase_source: Optional[str] = None,
    git_provenance: Optional[dict] = None,
    determinism: Optional[dict] = None,
    source_attestation: Optional[dict] = None,
    experiment_origin: Optional[dict] = None,
    execution_semantics: Optional[dict] = None,
    run_intent: Optional[dict] = None,
    experiment_context: Optional[dict] = None,
    realized_cadence: Optional[dict] = None,
    backpressure: Optional[dict] = None,
    supervision_work: Optional[dict] = None,
    spf_resolution: Optional[dict] = None,
    resolution: Optional[dict] = None,
    experiment_validity_class: Optional[str] = None,
    experiment_validity_is_final: Optional[bool] = None,
    experiment_validity_source_indices: Optional[Iterable[int]] = None,
    experiment_validity_source_index_provenance: Optional[str] = None,
    report_video_min_width: Optional[int] = None,
    report_video_max_width: Optional[int] = None,
    # Task 8 (2026-09-08): CLI-flag equivalent of SPLAT_REPORT_VIDEOS (see
    # this function's `_do_videos` computation below) -- ablator does not
    # reliably pass that env var through to the container for this job type
    # (confirmed live via `docker exec <container> env`, twice), so this
    # kwarg lets --streaming_report_videos/--no-streaming_report_videos
    # (arguments/__init__.py) disable the two post-mapping videos
    # (trajectory.mp4/train_depth_timeseries.mp4) without depending on that
    # broken passthrough. Default True reproduces prior behavior
    # byte-for-byte. The env var, when it DOES reach the container, remains
    # a secondary override on top of this flag -- see `_do_videos` below.
    report_videos: bool = True,
    admission_fingerprint: Optional[dict] = None,
    budgeted_admission: Optional[dict] = None,
    ingress_ownership: Optional[dict] = None,
    ingest_counters: Optional[dict] = None,
    partition_control: Optional[dict] = None,
    bag_source_manifest: Optional[dict] = None,
    admission_cadence: Optional[dict] = None,
    sync_pair_ledger: Optional[dict] = None,
    streaming_phase: Optional[dict] = None,
    evaluation_tick_clock_snapshot: Optional[dict] = None,
    scene_source: Optional[str] = None,
    presentation_supersample_factor: int = 1,
    presentation_sharpen_strength: float = 0.0,
    presentation_trajectory_interpolate: bool = True,
    presentation_trajectory_interpolate_max_extra_renders: Optional[int] = DEFAULT_MAX_INTERPOLATION_RENDERS,
    presentation_trajectory_interpolate_max_per_slice: Optional[int] = DEFAULT_MAX_INTERPOLATION_PER_SLICE,
    allocator_phase_tracker: Optional[Any] = None,
) -> dict:
    """Produce side-by-side PNGs, contact sheet, trajectory MP4, and report.json.

    Returns a summary dict containing:
        n_train, n_test, mean_test_psnr, min_test_psnr, max_test_psnr,
        output_dir, artifacts (paths written)

    ``SPLAT_REPORT_PRESENTATION=0`` retains the normal scalar and per-view
    metric pass but skips presentation-only PNGs, overview renders, and video
    encoding. ``report.json`` records this as deferred; it does not schedule
    a writer against live map state or promise an exact later regeneration.

    ``presentation_supersample_factor``/``presentation_sharpen_strength`` (issue #1323):
    presentation-only SSAA + unsharp-mask sharpening. Full SSAA (re-render at higher
    resolution, then downsample) applies ONLY to ``trajectory.mp4`` and
    ``train_contact_sheet.png`` -- the two artifacts in this function with a render pass
    fully independent of ``_compute_metrics``. Sharpening-only (no re-render) additionally
    applies to the TEST-split contact sheets, on the already-quantized uint8 display copy,
    never on the float tensor ``_compute_metrics`` reads out of ``_visual_cam_render_cache``.
    See ``utils/presentation_ssaa.py``'s module docstring for the exact per-artifact
    breakdown and CLAUDE.md's #1259 campaign-isolation note for why this split exists.
    Defaults (``1``/``0.0``) are exact no-ops -- every existing caller that doesn't pass
    these keeps identical output.

    ``presentation_trajectory_interpolate``/``_max_extra_renders``/``_max_per_slice``
    (issue #397 follow-up, "trajectory.mp4 looks chonky/stuttery"): applies ONLY to
    ``trajectory.mp4``'s real-timestamped pacing path (``distribute_frame_repeats``),
    never to ``train_depth_timeseries.mp4``, any contact sheet, or scored metrics.
    When on (the default -- see ``_expand_cams_with_interpolation``'s docstring for
    why this is safe to default on: bounded, presentation-only cost, and it cannot
    reach ``_infer_realtime_fps``/scoring), each real-timestamped slice's repeated
    identical-copy frames are replaced by SLERP/lerp-interpolated poses swept toward
    the next real camera, spending exactly the same total encoded-frame budget
    (``sum(frame_repeats)`` -- and therefore total video duration -- is unchanged).
    Cost is bounded by ``_max_extra_renders`` (total additional unique renders across
    the whole video) and ``_max_per_slice`` (per real-camera-gap cap); both default to
    ``utils.trajectory_video``'s ``DEFAULT_MAX_INTERPOLATION_RENDERS``/
    ``DEFAULT_MAX_INTERPOLATION_PER_SLICE``. Set ``presentation_trajectory_interpolate=False``
    to reproduce the exact pre-existing repeated-hold behavior.

    ``report_video_min_width``/``report_video_max_width`` (issue #397):
    resolution policy for ``trajectory.mp4`` -- an aspect-preserving
    post-render CPU upscale to at least ``report_video_min_width``
    (default: ``utils.trajectory_video``'s ``DEFAULT_MIN_VIDEO_WIDTH``,
    1920/"FHD-class") when the rendered camera is narrower, downscaled to
    fit ``report_video_max_width`` (default ``DEFAULT_MAX_VIDEO_WIDTH``,
    3840) when wider, no-op otherwise. ``train_depth_timeseries.mp4`` uses
    only the ``max_width`` side (its own min-width upscale is always off --
    it is a 6-panel diagnostic montage, and upscaling it would shrink/blur
    every individual panel for no benefit). ``None`` (the default here)
    falls back first to the ``SPLAT_REPORT_VIDEO_MIN_WIDTH``/
    ``SPLAT_REPORT_VIDEO_MAX_WIDTH`` env vars, then to the module defaults
    -- **only ``train.py``'s legacy-path call site passes ``None`` and
    reaches this fallback today**; ``train_streaming.py`` always passes a
    concrete int resolved from ``--streaming_report_video_min_width``/
    ``max_width`` (StreamingParams' own argparse defaults), since that CLI
    flag pair already IS the streaming-run configuration knob -- the env
    vars are inert for a streaming run unless a future call site there is
    changed to pass ``None`` too. Pass ``0`` explicitly to disable that
    side of the policy entirely (e.g. for a smoke test that wants the old
    raw-render-resolution/faster-encode behavior). This is a
    **presentation-only upscale of the already-rendered frame**, not a
    higher-resolution Gaussian render -- see
    ``utils.trajectory_video``'s module docstring for the cost/quality
    tradeoff this implies, and CLAUDE.md's note on measuring the added
    report-step wall-clock cost before shipping a resolution change.

    ``mapping_clocks`` (Wave 2 Task 8, optional): a plain dict snapshot of
    the caller's ``MappingCoordinator.four_clocks.summary()`` (see
    ``eligsir/mapping/coordinator.py``'s ``FourClocksSummary``) --
    ``source_frame_rate_hz``, ``admitted_keyframe_rate_hz``,
    ``map_commit_rate_hz``, ``map_age_ms``, ``input_backlog_frames``,
    ``realtime_factor``. Follows the same gated/optional pattern as
    ``input_quality``/Tier D's ``geom["offsurface"]``: omitting it (the
    default) reproduces the exact ``report.json`` from before this field
    existed -- it is only present, under ``geom["mapping_clocks"]``, when
    the caller (currently only ``train_streaming.py``'s streaming loop)
    supplies it. The offline (non-streaming) ``train.py`` path has no
    ``MappingCoordinator``/four-clocks instrumentation yet (Wave 2 Task 7
    found it doesn't use ``GaussianOptimizationEngine`` either), so this
    field is streaming-only for now, same scope limit as ``it/s`` vs.
    genuine mapping-rate telemetry described in CLAUDE.md's
    streaming-eval-metric-unreliable memory. **Wall-clock Hz only** -- the
    per-optimizer-step realized cadence is a different clock family and lives
    in ``realized_cadence`` (issue #289, below).

    ``realized_cadence`` (issue #289, optional): the run's realized
    release/admission/insertion/supervision/optimizer/mutation cadence,
    denominated in **optimizer steps**, plus a requested-vs-realized
    reconciliation verdict -- see ``utils/realized_cadence.py``. Written to
    ``report.json["realized_cadence"]`` (top-level, not under ``geometry``)
    with the verdict mirrored into the ``ingestion`` block.

    ``spf_resolution`` (issue #596, optional): the REQUESTED-vs-RESOLVED
    steps-per-frame record -- requested SPF, coverage-required SPF, the
    ``min_spf_floor``, the resolved value and which mechanism owns it, plus
    whether it actually governs frame release. Built by
    ``eligsir.mapping.scheduler.spf_resolution.build_spf_resolution`` and
    formatted into train.log's ``[spf]`` startup block from the same record, so
    console and metadata cannot disagree. Written to
    ``report.json["spf_resolution"]`` (top-level). The REALIZED counterpart is
    ``realized_cadence`` above.

    ``resolution`` (issue #595, optional): how the run's IMAGE resolution was
    resolved -- native source size, the requested ``--streaming_resolution``
    divisor, the hard ``--streaming_auto_downscale_max_pixels`` safety cap and
    whether it fired (``startup_override_reason``), the independent
    camera-level ``--resolution`` downscale that composes with it, and -- nested
    under ``"dynamic"``, deliberately never flattened -- what the
    throughput-driven ``ResolutionController`` did during training (initial and
    current tier, a bounded first-N tier-change history, and the OBSERVED
    trained image size). Built by
    ``eligsir.mapping.resolution_provenance.resolution_report_block`` and
    formatted into train.log's ``[resolution]`` startup block from the same
    record. Written to ``report.json["resolution"]`` (top-level). Unrelated to
    ``spf_resolution`` above despite the shared word: that one is
    steps-per-frame, this one is pixels.

    ``depth_insertion`` (issue #303, optional): ``{"enabled", "total_inserted",
    "frames_admitted", "inert"}`` -- the Phase 2 depth-insertion sanity gate.
    Written to ``geometry.depth_insertion``; ``inert`` is True iff depth
    insertion was enabled and frames were admitted yet nothing was ever
    inserted (a depth-inert run: MCMC-growth-only, never to be read as a
    normal RGB-D result). Supplied only by ``train_streaming.py``'s FINAL
    report -- a mid-run/pre-run report legitimately has
    ``total_inserted == 0`` and must not be flagged. Omitting it (the
    default) reproduces the exact ``report.json`` from before it existed.

    ``lineage_preserving_setup`` (issue #374 item 3, optional): the result of
    ``lineage_preserving_setup_report_block(getattr(gaussians,
    "_lineage_preserving_setup_events", []))`` -- count + per-event
    iteration/site of every mid-run ``training_setup(preserve_lineage=True)``
    call (currently only ``rolling_seed``/``submap_stitch``'s optimizer
    rebuild, see ``GsplatGaussianModel.training_setup``'s docstring). Written
    to ``geometry.lineage_preserving_setup``. Both of today's real call sites
    return before any report is written, so in practice this key is not yet
    populated by a real run -- it exists so a caller reaching this function
    with lineage events already on the model (a future call site, or a
    ``report.json`` regenerated from a checkpoint) surfaces them without a new
    top-level schema. Omitting it (the default, or an empty event list)
    reproduces the exact ``report.json`` from before this field existed.

    ``seed_cohort`` (issue #346, optional-to-pass but ALWAYS-emitted once
    passed): the result of ``seed_cohort_report_block(gaussians)`` -- whether
    the exact #283 bootstrap cohort is determinable for this run, plus its
    counts when it is, or a verbatim ``SeedCohortUnavailable`` reason when it
    is not. Written to ``geometry.seed_cohort``. Unlike the other optional
    blocks this is NOT gated on truthiness at the write site, because
    ``available: False`` is exactly the case a future analysis session needs
    recorded. Omitting the parameter entirely (the default, e.g. the offline
    ``train.py`` path) still reproduces the exact ``report.json`` from before
    this field existed.

    ``bag_replay_report`` (W3-15, optional): a
    ``BagReplayIntegrityReport.as_geometry_fields()`` dict
    (``eligsir/io/policies/bag_replay.py``), threaded straight into
    ``compute_trajectory_coverage``'s ``bag_replay_geometry_fields``
    (see that function's docstring). ``None`` (the default) whenever the
    caller's ingestion path didn't use a ``BagReplayPolicy`` -- true for
    every production call site as of this parameter's addition, since
    that policy has no production instantiation yet; this parameter is
    the tested plumbing a future ingestion-side integration will feed.

    ``ingest_counters`` (issue #865, acceptance item 4, optional): a detached
    dict snapshot of ``eligsir/io/sources/ros2_live.py``'s accept/reject
    counters (``accepted_frames``, ``rgb_received``, ``queued_frames``,
    per-topic buffer sizes, every ``reject_*`` counter), threaded straight
    into ``compute_trajectory_coverage``'s ``ingest_counters`` (see that
    function's docstring). Previously computed on every diagnostics tick
    and published ONLY to a transient ROS2 DiagnosticArray topic, never
    persisted. ``None`` (the default) for every non-``ros2://live`` source.
    Report-only/additive -- does not participate in ``verdict``/
    ``mapping_result_valid`` on its own (see ``bag_source_manifest``
    immediately below, which DOES use it together with this).

    ``bag_source_manifest`` (issue #865, acceptance items 1-2, optional): the
    bag's own real per-topic message counts, normally read from
    ``<model_path>/bag_source_manifest.json`` (written by
    ``human_launch_scripts/05_run_offline_rosbag.sh`` before the trainer
    launches, since the bag file is static and its counts are known up
    front -- see that script and ``utils.partition_eval.
    compute_bag_source_gap``'s docstring for the exact shape). Threaded
    straight into ``compute_trajectory_coverage``'s parameter of the same
    name, which cross-checks it against ``ingest_counters``/
    ``frames_admitted`` and, unlike ``ingest_counters`` alone, DOES fold the
    result into ``mapping_result_valid`` -- a real, positively detected gap
    between the bag's message index and what the trainer actually ingested
    invalidates the run exactly like ``map_reinit_invalidated``/
    ``frame_drop_detected`` already do. ``None`` (the default) for every
    run that isn't a bag replay launched by that script, or predates this
    mechanism -- means "cannot verify", never "clean". This is the direct
    fix for issue #865's finding: a real run lost 33 real frames at the
    tail of a bag while ``stream_end_reason`` reported ``"bag_complete"``
    and every existing signal reported clean.

    ``multires_cache`` (issue #464, optional-to-pass but ALWAYS-emitted once
    passed): the result of ``train_streaming.multires_cache_report_block()`` --
    how many train frames actually got a dual-tier resolution cache, how many
    failed inside ``record_frame_observation``'s deliberately-broad (run-alive)
    ``except``, and the error types. Written TOP-LEVEL as
    ``summary["multires_cache"]`` (like #437's ``n_test_metrics_failed``, and
    deliberately NOT inside the ``geometry`` try/except -- a block that reports a
    swallowed failure must not itself be erasable by a different swallowed
    failure), with a loud ``[report] WARNING:`` whenever ``frames_failed > 0``.
    Absent means the report predates this fix OR came from an offline
    ``train.py`` run (only ``train_streaming`` passes it) -- it never means
    nothing failed.

    ``gpu_memory`` (issue #399, optional-to-pass but ALWAYS-emitted once
    passed): the result of
    ``utils.gpu_memory_telemetry.GPUMemoryResidencyTracker.report_block()`` (or
    ``disabled_report_block()``) -- per-subsystem GPU memory residency, peak +
    final + per-bucket bytes, the growth-vs-frames-ingested slope, and the
    #396/#398/#333 discriminators (``cold_resident_count``,
    ``holdout_resident_count``, ``multires_double_represented``). Also written
    TOP-LEVEL, for the same non-erasability reason as ``multires_cache``.
    Absent means the report predates #399; ``enabled: false`` means telemetry
    was off; ``n_snapshots: 0`` means it was on but never sampled.

    ``duty_cycle`` (issue #562, optional-to-pass but ALWAYS-emitted once
    passed): ``utils.duty_cycle.DutyCycleTimeline.report_block()`` (or
    ``disabled_report_block()``) -- an exact partition of host wall-clock across
    the documented stall taxonomy, plus ``optimizer_region_frac`` (an upper
    bound on the OPTIMIZER's GPU time only -- NOT on total GPU-busy time, and
    not a utilization measurement) and the top stall events.
    Written TOP-LEVEL for the same non-erasability reason as ``gpu_memory``.
    Absent means the report predates #562; ``enabled: false`` means the
    timeline was off; ``n_windows: 0`` means it ran but closed no window.

    ``baseline_run`` (2026-08-10, optional): a previously-completed run's
    ``model_path`` to automatically compare this run against (see
    ``compare_against_baseline``). ``None`` (the default) reproduces the
    exact ``report.json`` from before this parameter existed -- strictly
    opt-in, wired from ``--baseline_run`` on the CLI. When set, adds a
    ``summary["baseline_comparison"]`` dict comparing the FULL
    ``psnr_history`` curve against the baseline's, not just the final PSNR
    (per CLAUDE.md's "Compare PSNR curves, not just the final number" run
    contract) -- never raises; a bad/missing baseline produces a
    ``{"comparable": False, "reason": ...}`` entry instead of interrupting
    this run's own report.

    ``last_ingest_iter`` / ``true_final_iteration`` / ``mapping_phase_source``
    (issue #346 follow-up, 2026-08-13, all optional): threaded straight into
    ``compute_trajectory_coverage`` -- see that function's own docstring for
    full semantics. ``last_ingest_iter`` is the caller's real
    ``_last_ingest_iter`` at report time. ``true_final_iteration`` exists
    specifically to work around ``train_streaming.py``'s FINAL report call
    site passing ``iteration=opt.iterations`` (a known-bogus 2^31 sentinel
    for ``--online_mode`` runs, per CLAUDE.md's "online_mode report.json
    iteration sentinel bug" project memory) -- when supplied, it is used as
    the trajectory-coverage block's ``final_iteration`` INSTEAD of this
    function's own (possibly-bogus) ``iteration`` parameter. As of issue #922,
    it is also used for output directory naming (``iter_{true_final_iteration}``)
    so that final comparison/report artifacts use the real terminal iteration
    instead of a sentinel value. ``summary["iteration"]`` itself still comes
    from the ``iteration`` parameter unchanged. Omitting all three reproduces
    the exact report.json and directory naming from before this parameter set
    existed.
    ``git_provenance`` (2026-08-13, issue #259, optional): a plain dict
    with ``git_commit``/``git_branch``/``git_dirty`` describing the
    checkout this process actually ran from (e.g. ``train.py``'s
    ``get_git_metadata()``, captured once at process startup and stashed
    on ``args``). Ablator runners execute jobs from a mutable shared
    checkout with nothing pinning a job to a commit, so two runs launched
    from an identically-worded spec can silently execute different code
    (see issue #259) -- `cfg_args`/`report.json` recorded only *arguments*
    before this field existed, never the commit that ran them. ``None``
    (the default) reproduces the exact ``report.json`` from before this
    parameter existed. When provided, written verbatim to
    ``summary["git_provenance"]`` -- this is a record, not a gate; it does
    not block or alter training in any way.

    ``experiment_validity_class`` (2026-08-14, issue #203 / lane P0-E of
    #202, optional): the class this run's spec DECLARES it is producing
    evidence for -- ``SMOKE_ONLY`` / ``DISTRIBUTED_TRAJECTORY_PROXY`` /
    ``FULL_CONTINUAL_VALIDATION`` (``--experiment_validity_class``).
    ``None`` records ``declared: "UNDECLARED"``, which is the LEAST
    permissive value, never a pass. Regardless of what is declared, this
    function ALSO derives the class the run actually realized -- from the
    train cameras' source-frame indices against ``eval/scenes.json``'s
    ``n_source_frames`` (coverage fraction, trajectory span, and occupancy
    of ``utils.partition_eval``'s trajectory partitions) -- and writes
    ``declared``/``achieved``/``effective`` plus a ``mismatch`` flag and
    ``negative_conclusions_permitted`` into
    ``summary["experiment_validity"]``. See ``utils/experiment_validity.py``
    for the lattice, the thresholds, and why UNKNOWN is the bottom element.
    Purely a record: it gates nothing, and is deliberately a SIBLING field
    rather than folded into ``eval_valid`` / ``verdict`` /
    ``trajectory_coverage.mapping_result_valid`` (same reasoning as
    ``map_reinit_invalidated``). A MID-RUN checkpoint report legitimately
    grades ``achieved: SMOKE_ONLY`` even for a run that ends full --
    ``experiment_validity_is_final`` records which kind of report this is
    (``True`` only for the end-of-run report; ``None`` when the caller does
    not say).

    ``run_intent`` (2026-08-14, issue #524 acceptance criterion 3, optional):
    the resolved ``report.json["run_intent"]`` block -- built ONCE at
    ``train.py`` ``__main__`` startup by
    ``utils.run_intent.run_intent_report_from_args()`` (mirroring
    ``experiment_origin``/``execution_semantics`` above: a pure accessor over
    an object resolved exactly once, so it cannot diverge across a run's
    reports) and threaded through unchanged. Carries ``declared`` (the
    ``--run_intent`` value, or ``"UNDECLARED"`` -- the same least-permissive
    convention ``experiment_validity_class`` uses), ``resolved_run_profile``,
    ``evidence_grade``/``paper_registered`` (from
    ``docs/inventory/run_intents.yml``), ``profile_stack`` (the ordered list
    of config layer paths this run's config actually merged, parents first),
    ``effective_config`` (the merged dict those layers produced -- NOT the
    full argument set; see that function's docstring for why), and
    ``cli_overridden_keys`` (which profile-set keys an explicit CLI flag then
    overrode). ``None`` (the default) omits the key entirely, reproducing the
    exact ``report.json`` from before this parameter existed. A record only:
    gates nothing, changes no training behavior.
    """
    # Issue #922: use true_final_iteration for directory naming when provided,
    # to avoid sentinel values (2^31 for online_mode) appearing in directory names.
    dir_iteration = true_final_iteration if true_final_iteration is not None else iteration
    out_dir = os.path.join(model_path, "comparison", subdir or f"iter_{dir_iteration}")
    os.makedirs(out_dir, exist_ok=True)
    artifacts: List[str] = []

    # Presentation is intentionally separate from scoring.  A disabled
    # presentation pass still renders the representative test views needed by
    # `_compute_metrics`, but does not encode or write derived visual output.
    # It never schedules a live-state writer: a later full report must start
    # from its own frozen checkpoint and inputs.
    presentation_enabled = os.environ.get("SPLAT_REPORT_PRESENTATION", "1") == "1"

    # Issue #1259/#1284: reset the NaN/Inf-cast sanitization accounting so a
    # long-lived process (multiple report checkpoints in one run, or the test
    # suite) doesn't leak a previous report's counts into this one.
    reset_nan_sanitize_tracking()

    # Issue #1259: restore every camera's NATIVE resolution binding before any
    # rendering/scoring in this report. Done on the FULL train_cams/test_cams
    # lists, once, up front -- not on the smaller subsampled lists individual
    # blocks below draw for metrics/sheets/video. Several of those blocks
    # (`_write_visual_sheet`, the trajectory MP4, the depth timeseries) each
    # pick their own independent subset of the same camera objects, and
    # `render_batch` (`eligsir/mapping/renderer/gsplat_backend.py`)
    # refuses a batch whose cameras don't all share one resolution -- so
    # restoring only the metrics subset would leave the full list a mix of
    # native (just-scored) and still-tiered cameras and move the crash
    # elsewhere instead of fixing it. See `_restore_native_resolution_for_report`.
    _restore_native_resolution_for_report(train_cams)
    _restore_native_resolution_for_report(test_cams)

    # Issue #397: resolve the trajectory-video resolution policy. Explicit
    # kwarg wins; else an env var (ops-level override, same pattern as
    # SPLAT_REPORT_VIDEOS); else the module default (FHD-class). Any of
    # these resolving to <= 0 means "disabled" (scale_dims_for_video
    # already treats <= 0 as off) -- kept as an int, not coerced to None,
    # so a caller can tell "explicitly disabled" apart from "unset".
    def _resolve_video_width(explicit: Optional[int], env_name: str, default: int) -> int:
        if explicit is not None:
            return int(explicit)
        env_val = os.environ.get(env_name)
        if env_val is not None:
            try:
                return int(env_val)
            except ValueError:
                pass
        return default

    report_video_min_width = _resolve_video_width(
        report_video_min_width, "SPLAT_REPORT_VIDEO_MIN_WIDTH", DEFAULT_MIN_VIDEO_WIDTH)
    report_video_max_width = _resolve_video_width(
        report_video_max_width, "SPLAT_REPORT_VIDEO_MAX_WIDTH", DEFAULT_MAX_VIDEO_WIDTH)

    summary = {
        "iteration": iteration,
        # Honest, visible telemetry for a resumed/composite run (never silent
        # about it — see CLAUDE.md's "Run Contracts" convention): None means
        # this run's iteration history is a single unbroken from-scratch
        # attempt; an int means the process (re)started from a checkpoint at
        # that iteration at least once (preemption resume or a manual
        # --start_checkpoint), so this report.json describes a composite run.
        "resumed_from_iteration": resumed_from_iteration,
        "n_train": len(train_cams) if train_cams else 0,
        "n_test": len(test_cams) if test_cams else 0,
        "mean_test_psnr": None,
        "min_test_psnr": None,
        "max_test_psnr": None,
        "mean_test_ssim": None,
        "min_test_ssim": None,
        "max_test_ssim": None,
        "mean_test_lpips": None,
        "min_test_lpips": None,
        "max_test_lpips": None,
        "mean_train_psnr": None,
        "min_train_psnr": None,
        "max_train_psnr": None,
        "mean_train_ssim": None,
        "min_train_ssim": None,
        "max_train_ssim": None,
        "mean_train_lpips": None,
        "min_train_lpips": None,
        "max_train_lpips": None,
        "trajectory_fps": None,
        "n_train_metrics": 0,
        "n_test_metrics": 0,
        # Issue #437: a per-camera render failure inside the metrics loop is
        # caught and skipped (deliberately -- one bad camera must not lose the
        # whole report), which silently SHRINKS n_test_metrics. Nothing in
        # report.json ever recorded that, so "scored 128 of 315 because the cap
        # subsampled it" and "scored 128 of 315 because 187 renders raised" were
        # indistinguishable downstream. These three fields make the difference
        # explicit and machine-readable; the invariant
        #     n_*_metrics + n_*_metrics_failed == n_*_metrics_attempted
        # holds for every report written after this fix. Initialised here so a
        # report that never reaches the metrics loop still carries them --
        # an ABSENT key therefore means "report predates this fix", not "zero".
        "n_train_metrics_attempted": 0,
        "n_train_metrics_failed": 0,
        "n_test_metrics_attempted": 0,
        "n_test_metrics_failed": 0,
        "output_dir": out_dir,
        "artifacts": artifacts,
    }
    if not presentation_enabled:
        summary["presentation_artifacts"] = {
            "status": "deferred",
            "reason": "SPLAT_REPORT_PRESENTATION=0",
            "regeneration": {
                "status": "pending",
                "reason": "no_frozen_presentation_snapshot_or_live_writer",
            },
            "inputs": {
                "report_iteration": int(iteration),
                "n_train": len(train_cams) if train_cams else 0,
                "n_test": len(test_cams) if test_cams else 0,
            },
        }

    summary["train_metric_binding"] = {
        "iteration": int(iteration),
        "mapping_phase_source": mapping_phase_source,
    }

    _allocator_phase_enabled = (
        os.environ.get("SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES", "0") == "1"
    )
    _allocator_frames_ingested = int((gpu_memory or {}).get("final", {}).get(
        "frames_ingested", 0
    ) or 0)
    summary["report_allocator_phases"] = {
        "enabled": _allocator_phase_enabled,
        "available": False,
        "reason": (
            "SPLAT_REPORT_ALLOCATOR_PHASE_SAMPLES=0"
            if not _allocator_phase_enabled
            else "no_existing_gpu_memory_residency_tracker"
            if allocator_phase_tracker is None
            else "no_successful_allocator_phase_sample",
        ),
        "samples": {},
    }

    def _record_allocator_phase(name: str, *, presentation_status: Optional[str] = None) -> None:
        """Record bounded report-boundary telemetry without changing residency."""
        if not _allocator_phase_enabled or allocator_phase_tracker is None:
            return
        try:
            sample = allocator_phase_tracker.phase_snapshot(
                name,
                iteration=int(iteration),
                frames_ingested=_allocator_frames_ingested,
                gaussians=gaussians,
                streaming_scene=streaming_scene,
                observation_db=getattr(streaming_scene, "observation_db", None),
            )
            if presentation_status is not None:
                sample["presentation_status"] = presentation_status
        except Exception as exc:
            sample = {
                "phase": name,
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        phases = summary["report_allocator_phases"]
        phases["samples"][name] = sample
        if sample.get("status") == "measured":
            phases["available"] = True
            if phases["reason"] == "no_successful_allocator_phase_sample":
                phases["reason"] = None
        else:
            reason = str(sample.get("reason") or "allocator phase sample unavailable")
            phases["reason"] = (
                f"partial_allocator_phase_failure: {reason}"
                if phases["available"]
                else reason
            )

    _record_allocator_phase("report_entry")
    if _allocator_phase_enabled and allocator_phase_tracker is not None:
        gpu_memory = allocator_phase_tracker.report_block()

    def _record_report_phase(name: str, operation):
        """Time report-only work without synchronizing the renderer."""
        started = time.perf_counter()
        print(f"[report] phase={name} event=begin monotonic_s={started:.6f}", flush=True)
        try:
            result = operation()
        except Exception as exc:
            ended = time.perf_counter()
            summary.setdefault("report_phase_timings", {})[name] = {
                "status": "failed",
                "begin_monotonic_s": started,
                "end_monotonic_s": ended,
                "duration_s": ended - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(
                f"[report] phase={name} event=error duration_s={ended - started:.6f} "
                f"error={type(exc).__name__}",
                flush=True,
            )
            raise
        ended = time.perf_counter()
        summary.setdefault("report_phase_timings", {})[name] = {
            "status": "completed",
            "begin_monotonic_s": started,
            "end_monotonic_s": ended,
            "duration_s": ended - started,
        }
        print(
            f"[report] phase={name} event=end duration_s={ended - started:.6f}",
            flush=True,
        )
        return result

    # Multires resolution-cache health (issue #464). Built by
    # train_streaming.multires_cache_report_block(); see its docstring for the
    # frames_cached + frames_failed + frames_empty_no_depth == frames_attempted
    # invariant. `all_failed` is the total, near-silent 100%-
    # bind_resolution_tier-failure mode #464 was filed about.
    #
    # Written TOP-LEVEL (like #437's n_test_metrics_failed), deliberately NOT
    # inside the big `geometry` try/except further down: this block exists to
    # report a swallowed failure, so putting it somewhere a *different*
    # swallowed failure could erase it would defeat its entire purpose. An
    # ABSENT key means either the report predates this fix OR it came from an
    # offline `train.py` run -- only train_streaming's _streaming_report_kwargs
    # passes it. It never means "the cache was healthy".
    # Issue #795: admission fingerprint (realized policy vs. nominal config mismatch).
    # Persisted TOP-LEVEL for visibility: a possible mismatch between the nominal
    # admission policy (from config) and what actually happened is a key finding
    # for run comparability and mechanism attribution. An ABSENT key means the
    # report predates this fix (or came from an offline train.py run).
    if admission_fingerprint is not None:
        if isinstance(admission_fingerprint, dict):
            summary["admission_fingerprint"] = admission_fingerprint
        else:
            admission_fingerprint.attach_canonical_funnel(
                admission_cadence=admission_cadence,
                realized_cadence=realized_cadence,
                ingestion=ingestion,
            )
            summary["admission_fingerprint"] = admission_fingerprint.to_dict()
        _af = summary["admission_fingerprint"]
        if _af.get("admission_realized_mismatch"):
            print(
                f"[report] WARNING: admission_fingerprint: "
                f"realized policy contradicts nominal config! "
                f"nominal={_af.get('nominal_policy')} "
                f"mismatch_reason={_af.get('mismatch_reason')}",
                flush=True,
            )

    # Issue #500: the cohort owner is the only authoritative source for the
    # fixed-K selector's candidate/commit/drop counts. Keep it separate from
    # admission_fingerprint, whose per-source-event observation cannot see a
    # deferred candidate that is committed from cache by a later event.
    if budgeted_admission is not None:
        summary["budgeted_admission"] = dict(budgeted_admission)

    # Issue #837: machine-readable ingress ownership contract (sibling of the
    # admission_fingerprint/source_decimation blocks above -- "one owner per
    # ingress responsibility", see eligsir/io/ingress_ownership.py). An
    # ABSENT key means the report predates this fix or came from a caller
    # that didn't build a block (e.g. train.py's legacy, non-streaming path).
    # This is report/provenance only: `valid`/`failure_codes` are a loud,
    # machine-readable structural verdict for a human or a downstream
    # validity gate to act on (mirroring eval_valid/SUSPECTED_TRUNCATED/
    # map_reinit_invalidated) -- write_post_training_report never alters
    # training behavior because of it.
    if ingress_ownership is not None:
        summary["ingress_ownership"] = dict(ingress_ownership)
        if not ingress_ownership.get("valid", True):
            print(
                f"[report] WARNING: ingress_ownership: structural verdict INVALID "
                f"(topology={ingress_ownership.get('topology')}, "
                f"failure_codes={ingress_ownership.get('failure_codes')}) -- "
                f"never quote this run as a canonical single-owner ingress "
                f"topology.",
                flush=True,
            )

    if multires_cache is not None:
        summary["multires_cache"] = dict(multires_cache)
        if multires_cache.get("frames_failed"):
            _mc_f = multires_cache.get("failures") or {}
            print(
                f"[report] WARNING: multires_cache: "
                f"{multires_cache.get('frames_failed')} of "
                f"{multires_cache.get('frames_attempted')} train frames failed to "
                f"build a resolution cache (all_failed="
                f"{multires_cache.get('all_failed')}, device="
                f"{multires_cache.get('device')!r}, error_types="
                f"{_mc_f.get('error_types')}) -- bind_resolution_tier could not "
                f"bind for those frames. Example: {_mc_f.get('example_error')}",
                flush=True,
            )

    # Per-subsystem GPU memory residency telemetry (issue #399). Built by
    # utils.gpu_memory_telemetry.GPUMemoryResidencyTracker.report_block(); see
    # that module's docstring for the storage-dedup / fixed-bucket-priority
    # attribution scheme and the meaning of the `unattributed` residual.
    # Written TOP-LEVEL for the same reason multires_cache is: this block
    # exists to make a memory problem visible, so it must not live somewhere a
    # swallowed geometry failure could erase it. An ABSENT key means the report
    # predates #399 (or came from an offline train.py run); `enabled: false`
    # means telemetry was really off; `n_snapshots: 0` means it was on but
    # never sampled. Those three must stay distinguishable (cf. the seed_fade
    # NO_SAMPLES audit, #282).
    if gpu_memory is not None:
        summary["gpu_memory"] = dict(gpu_memory)
        _gm_final = gpu_memory.get("final") or {}
        if _gm_final:
            _mib = 1024.0 * 1024.0
            _gtt_used = _gm_final.get("gtt_used_bytes")
            _gtt_total = _gm_final.get("gtt_total_bytes")
            _gtt_summary = "gtt_global=unavailable"
            if _gm_final.get("gtt_available") and _gtt_used is not None and _gtt_total is not None:
                _gtt_summary = (
                    f"gtt_global={_gtt_used / _mib:.1f}MiB/{_gtt_total / _mib:.1f}MiB"
                )
            print(
                f"[report] gpu_memory (#399): torch_allocated="
                f"{_gm_final.get('allocated_bytes', 0) / _mib:.1f}MiB "
                f"torch_peak={gpu_memory.get('peak_allocated_bytes', 0) / _mib:.1f}MiB "
                f"torch_unattributed="
                f"{(_gm_final.get('buckets') or {}).get('unattributed', 0) / _mib:.1f}MiB "
                f"{_gtt_summary} gtt_scope=device_global_shared_not_per_process "
                f"cold_resident={_gm_final.get('cold_resident_count')} "
                f"holdout_resident={_gm_final.get('holdout_resident_count')} "
                f"multires_frames={_gm_final.get('multires_cached_frames')} "
                f"multires_double_represented="
                f"{_gm_final.get('multires_double_represented')}",
                flush=True,
            )

    # Host resource monitor (SPLAT_RESOURCE_MONITOR=1, utils/resource_monitor.py).
    # Peak/mean CPU%/RAM%/GPU-util%/VRAM% sampled over the run's whole
    # wall-clock duration by polling the local host's Glances REST API from a
    # background thread. An ABSENT key means the run predates this feature;
    # `resource_monitor["enabled"] == False` means it was off (default) or
    # Glances was unreachable at startup -- both distinguishable via
    # `resource_monitor["reachable"]`.
    if resource_monitor is not None:
        summary["resource_monitor"] = dict(resource_monitor)

    # CPU/GPU duty-cycle timeline (issue #562). Top-level for the same
    # non-erasability reason as gpu_memory. `optimizer_region_frac` bounds the
    # OPTIMIZER's GPU time only, not total GPU-busy time (several other
    # categories also launch device work) -- see utils/duty_cycle.py's docstring.
    # ABSENT = report predates #562; enabled:false = telemetry off;
    # n_windows:0 = on but never closed a window.
    if duty_cycle is not None:
        summary["duty_cycle"] = dict(duty_cycle)
        if duty_cycle.get("enabled"):
            _dc_frac = duty_cycle.get("frac") or {}
            print(
                f"[report] duty_cycle (#562): wall={duty_cycle.get('wall_s')}s "
                f"iters={duty_cycle.get('total_iterations')} "
                f"optimizer_region={duty_cycle.get('optimizer_region_frac')} "
                f"unattributed={duty_cycle.get('unattributed_frac')} "
                f"ingest_decode={_dc_frac.get('ingest_decode')} "
                f"depth_insertion={_dc_frac.get('depth_insertion')} "
                f"n_stalls={duty_cycle.get('n_stalls')}",
                flush=True,
            )

    # Pinned-transfer-pool high-water-mark (issue #1259). ABSENT means no
    # DeviceTransferOwner was ever constructed for this run (e.g. a legacy
    # non-device call path) -- never read that as "zero usage".
    if transfer_pool is not None:
        summary["transfer_pool"] = dict(transfer_pool)

    # Per-frame source-ingress attribution for the causal paced source buffer
    # (issue #1273). ABSENT means the run had no paced transport buffer at all
    # (every non-causal call path) -- never read that as "zero drops". The
    # block is REAL_TIME_DROP-governed and additive: it is threaded into
    # trajectory_coverage below but participates in no verdict and never
    # latches mapping_result_valid.
    if source_ingress is not None:
        summary["source_ingress"] = dict(source_ingress)
        _ingress_warning = source_ingress_warning(source_ingress)
        if _ingress_warning:
            print(_ingress_warning, flush=True)

    # A degenerate split (e.g. the bootstrap-views debug report passes the same
    # camera list as both train and test) must NEVER emit a "test_*" metric — that
    # is how an apparent "test PSNR" leaked from train-on-train views. Detect it and
    # route the numbers to neutral "view_*" keys instead.
    _degenerate = is_degenerate_split(train_cams, test_cams, degenerate_split)
    summary["split_kind"] = "degenerate_train_views" if _degenerate else "holdout"
    # A run is eval-valid only if it has a real (non-degenerate) held-out test set.
    summary["eval_valid"] = bool(test_cams) and not _degenerate
    # eval_metadata (holdout_hash comparability contract -- CLAUDE.md's Run
    # Contracts): explicit callers (currently only train_streaming.py's final
    # report) pass this directly. Every OTHER streaming call site (mid-training
    # ticks, rolling_seed_application.py, submap_stitch_application.py) already
    # passes `streaming_scene=` for other purposes (partition metrics, four-clocks)
    # but historically forgot to also pass `eval_metadata=streaming_scene.
    # eval_metadata()`, so those report.json files silently had no eval_metadata
    # key at all -- discovered 2026-08-13 comparing two real completed runs
    # (issue116_partfocus_ctrl/comparison/iter_7000, a mid-training tick, vs.
    # issue109exp_E1/comparison/iter_104000, a final report). Rather than fix
    # every call site individually (and risk a second, drifting implementation
    # of the same lookup), derive it here from the SAME `streaming_scene.
    # eval_metadata()` the final-report call site already uses, whenever a
    # caller passed `streaming_scene=` but not `eval_metadata=` explicitly.
    # Gated on `not _degenerate` for the same reason `realized_holdout_hash`
    # below is: a degenerate split (e.g. the *_bootstrap_views debug report)
    # must never emit anything that looks like a real holdout fingerprint.
    if eval_metadata is None and streaming_scene is not None and not _degenerate:
        try:
            eval_metadata = streaming_scene.eval_metadata()
        except Exception:
            eval_metadata = None
    if eval_metadata:
        summary["eval_metadata"] = dict(eval_metadata)
    # realized_holdout_hash (issue #199): hash of the test cameras this run
    # ACTUALLY evaluated, as opposed to eval_metadata["holdout_hash"]'s
    # selection-RULE hash, which can match across runs whose realized test
    # sets differ in size (e.g. two runs of the same scene at different
    # --iterations budgets). Computed here, not in eval_metadata() itself,
    # because test_cams is the one list every write_post_training_report
    # call site (streaming and offline alike) already passes as the
    # realized set at report time -- see _realized_holdout_hash's docstring.
    # Gated on `not _degenerate`: a degenerate split (e.g. the
    # *_bootstrap_views debug report, which deliberately passes the same
    # cameras as both train and test) must never emit anything that looks
    # like a real holdout fingerprint, matching the existing
    # eval_valid/split_kind convention just above. Note this deliberately
    # ADDS an `eval_metadata` key (via setdefault) even on mid-run
    # checkpoints that pass no `eval_metadata` at all (e.g. every
    # w3_106_batch_B1_20260812 iter_1000..3000 checkpoint on disk predates
    # holdout_hash entirely) -- a real report.json can end up with
    # eval_metadata containing ONLY realized_holdout_hash, no holdout_hash.
    if not _degenerate:
        try:
            _realized_hash = _realized_holdout_hash(test_cams)
        except Exception:
            _realized_hash = None
        if _realized_hash is not None:
            summary.setdefault("eval_metadata", {})
            summary["eval_metadata"]["realized_holdout_hash"] = _realized_hash
    if soft_all_admission is not None:
        # Issue #600. Surfaced verbatim; report-only, and deliberately NOT part
        # of any validity verdict -- but `inserted_from_skipped_frame` must read
        # 0, and a loud warning fires here if it ever doesn't, because a nonzero
        # value means this run's geometry includes contributions from frames the
        # mapper rejected and cannot be attributed to its admitted supervision
        # set.
        summary["soft_all_admission"] = dict(soft_all_admission)
        _leaked = int(soft_all_admission.get("inserted_from_skipped_frame", 0) or 0)
        if _leaked:
            print(
                f"[report] WARNING: #600 insertion invariant VIOLATED -- "
                f"{_leaked} frame(s) skipped for mapping supervision still "
                f"reached depth insertion. This run's geometry cannot be "
                f"attributed to its admitted supervision set.",
                flush=True,
            )
    from eligsir.utils.psnr_history_io import read_psnr_history_failures

    _psnr_history_failures = read_psnr_history_failures(model_path)
    _psnr_history_exists = os.path.exists(
        os.path.join(model_path, "psnr_history.jsonl")
    )
    summary["psnr_history_curve_valid"] = (
        not _psnr_history_failures if _psnr_history_exists else None
    )
    summary["psnr_history_invalid_ticks"] = [
        {
            "iteration": row.get("iteration"),
            "n_attempted": row.get("n_attempted"),
            "error": row.get("error"),
        }
        for row in _psnr_history_failures
    ]

    if seed_fade is not None:
        # Issue #346: grade the detector's raw block into a severity verdict.
        # The detector alone emits identical booleans for a 2.9dB blip and a
        # 15.6dB/75k-iteration catastrophe -- the only separator was a float
        # nothing gated on. grade_seed_fade_for_run adds `severity`,
        # `severity_reasons`, `confounded` and the psnr_history curve-shape
        # fields (per the "compare curves, not the final number" contract),
        # and never raises: a run with no readable psnr_history.jsonl still
        # gets a magnitude-only grade. REPORTING ONLY -- no training, loss,
        # lambda or controller path reads any of this.
        try:
            from eligsir.utils.seed_fade_severity import grade_seed_fade_for_run

            # Issue #419: hand the grader this call's OWN live
            # `_last_ingest_iter` (the mapping -> refinement boundary) rather
            # than letting it resolve one off disk. At this point the final
            # report.json does not exist yet, so an on-disk resolution would
            # read an earlier checkpoint's value (usually None, at best an
            # earlier/smaller boundary). When the caller passes none -- e.g.
            # render_report.py's post-hoc re-render, or a non-streaming run --
            # the grader falls back to its own on-disk resolution
            # (report.json, then the train.log idle-stop banner) and, failing
            # that, to legacy whole-run plateau grading.
            summary["seed_fade"] = grade_seed_fade_for_run(
                seed_fade, model_path,
                mapping_end_iter=last_ingest_iter,
                mapping_phase_source=("live" if last_ingest_iter is not None else None),
            )
        except Exception:
            summary["seed_fade"] = seed_fade
        # Top-level mirror so downstream tooling can refuse to quote this
        # run's PSNR without reaching into the nested block.
        summary["seed_fade_confounded"] = bool(summary["seed_fade"].get("confounded", False))
        # Issue #290: a THIRD, independent flag, deliberately alongside (never
        # replacing) seed_fade_confounded. It is derived from TWO independent
        # signals OR-ed together:
        #   1. the monitor's own `indeterminate` (it knows whether its state
        #      survived the resume, and also catches a config change across
        #      the resume that no report field could see), and
        #   2. this function's `resumed_from_iteration` argument, which is
        #      plumbed independently of the monitor -- so a call path that
        #      resumes but forgets SeedFadeMonitor.mark_resumed() still cannot
        #      report a clean verdict by omission.
        # A monitor block that positively asserts `resume_state_restored: True`
        # clears signal 2: that run's verdict really does span the whole run.
        _sf = summary["seed_fade"]
        _indet_reasons = list(_sf.get("indeterminate_reasons") or [])
        _indet = bool(_sf.get("indeterminate"))
        if _psnr_history_failures:
            _indet = True
            _indet_reasons.append(
                f"{len(_psnr_history_failures)} held-out PSNR history tick(s) failed; "
                "the curve is incomplete"
            )
        if resumed_from_iteration is not None and _sf.get("resume_state_restored") is not True:
            _indet = True
            if not _indet_reasons:
                _indet_reasons = [
                    f"report.resumed_from_iteration={resumed_from_iteration} but the "
                    "seed_fade block carries no positive resume_state_restored=True "
                    "assertion (telemetry predates issue #290's resume fix, or the "
                    "detector's state was not restored); the seed-fade verdict "
                    "describes only the post-resume segment"
                ]
        _sf["indeterminate"] = _indet
        _sf["indeterminate_reasons"] = _indet_reasons
        summary["seed_fade_indeterminate"] = _indet
        if _indet:
            # The #346 severity above was already graded from `max_dip_db`,
            # which on a resumed run is itself derived from the reset running
            # max. Rather than invent a `severity="indeterminate"` (which would
            # break SEVERITY_ORDER for every consumer), annotate the reasons
            # list every severity consumer already prints, so `severity` is
            # never read as trustworthy in isolation.
            _sf.setdefault("severity_reasons", []).append(
                "INDETERMINATE (#290): training resumed from a checkpoint without "
                "the detector's state, so max_dip_db -- and therefore this "
                "severity grade -- is derived from a running max that restarted "
                "mid-run"
            )
    _seed_lineage_freeze = getattr(gaussians, "_seed_lineage_freeze", None)
    if _seed_lineage_freeze is not None:
        summary["seed_lineage_freeze"] = _seed_lineage_freeze.report()
    if ingestion:
        # Issue #302: which frame-release pacing this run actually ran under,
        # and WHY that mode was chosen (explicit flag / online_mode default /
        # ros2 source default / offline default). Recorded because the corpus
        # could not otherwise be partitioned into iter_based vs dataset_fps
        # runs without forensic per-run log archaeology -- a reverted fix left
        # every --online_mode run between 2026-06-29 and 2026-08-13 on
        # dataset_fps pacing with no self-identifying record. Optional and
        # additive: omitting it reproduces the exact report.json from before
        # this field existed (same gating pattern as input_quality /
        # mapping_clocks).
        summary["ingestion"] = dict(ingestion)
    if backpressure:
        # Issue #601: what the ONE bounded backpressure owner actually did --
        # its resolved hard bounds (pending_max_count / pending_max_age_* /
        # max_admissions_per_window / frontier quota), the decision histogram
        # by action and by machine-readable reason, queue-depth and pending-age
        # quantiles, and the force-all stress flag.
        #
        # READ THIS BEFORE READING COVERAGE: a skipped observation still
        # advanced pose/source/history metadata (observe=True) and still
        # counted as source-released, but did NOT become expensive mapping
        # work. ``n_admitted`` is therefore the expensive-mapping subset, not
        # the coverage numerator -- do not read a high ``skipped_fraction`` as
        # a coverage failure. ``stress_mode: true`` means the owner was
        # deliberately bypassed (#600's force-all) and the run is a diagnostic
        # assimilation-overload condition, not a production result.
        #
        # An ABSENT key means the run predates #601 or never enabled the
        # owner -- deliberately distinguishable from an owner that ran and
        # skipped nothing.
        summary["backpressure"] = dict(backpressure)
        if backpressure.get("stress_mode"):
            print(
                "[report] WARNING: backpressure force-all STRESS MODE was active "
                "-- mapping backlog was intentionally unbounded. Not a production "
                "result.",
                flush=True,
            )
    if replica_rgb:
        # Issue #641: RGB source fidelity for Replica synthetic frames.
        # Omitted for non-Replica runs. Recorded so pre-fix and post-fix runs
        # self-identify their protocol difference in artifacts rather than only
        # in git history.
        summary["replica_rgb"] = dict(replica_rgb)
    if spf_resolution:
        # Issue #596: the REQUESTED-vs-RESOLVED half of the cadence story -- what
        # --streaming_steps_per_frame asked for, what actually governs frame
        # release, and WHICH mechanism owns that value (request / coverage /
        # min_spf_floor). Formatted from the same immutable record as train.log's
        # [spf] startup block, so console and metadata cannot disagree.
        #
        # Complementary to realized_cadence below, not a duplicate of it: this is
        # the REQUESTED/RESOLVED half (decided before the first iteration), that
        # is the REALIZED half (measured from the run's own clocks). W3-38's
        # silent min_spf_floor override is exactly the failure this makes
        # self-identifying rather than forensic.
        #
        # Optional/additive exactly like ingestion / realized_cadence: omitting it
        # reproduces the exact report.json from before this field existed (the
        # offline train.py path has no SPF resolution and passes nothing).
        summary["spf_resolution"] = dict(spf_resolution)
    if resolution:
        # Issue #595: how this run's IMAGE resolution was resolved. Distinct
        # from spf_resolution above in every way except the word "resolution"
        # (that one is steps-per-frame; this one is pixels).
        #
        # Records THREE composing-but-separate mechanisms without conflating
        # them: the hard --streaming_auto_downscale_max_pixels startup safety
        # gate (`hard_*` + `startup_override_reason`), the independent
        # camera-level --resolution downscale (`camera_*`), and the dynamic
        # throughput-driven ResolutionController (nested under "dynamic",
        # deliberately not flattened, so a tier change can never be misread as
        # a memory cap). The motivating run trained at 960x720 from a
        # 1920x1440 source with nothing but a train.log prose line saying so.
        #
        # Optional/additive exactly like ingestion / spf_resolution: omitting it
        # reproduces the exact report.json from before this field existed (the
        # offline train.py path has no streaming resolution gate and passes
        # nothing).
        summary["resolution"] = dict(resolution)
    if streaming_phase is not None:
        summary["streaming_phase"] = dict(streaming_phase)
    if supervision_work:
        # Issue #703/#755: canonical supervision-WORK accounting for this run --
        # how many rasterizer invocations and supervised pixels actually paid for
        # the optimizer steps that were taken, plus the derived per-step ratios.
        #
        # Deliberately distinct from realized_cadence below (which counts EVENTS
        # per optimizer step -- releases, admissions, insertions) and from
        # geom["mapping_clocks"] (which measures wall-clock Hz). This one counts
        # WORK: render calls and supervised pixels. #755's engine-promotion A/B
        # needs the two arms' work to be comparable at identical step counts, and
        # neither of the other two mechanisms can express that.
        #
        # Optional/additive exactly like ingestion / spf_resolution: omitting it
        # reproduces the exact report.json from before this field existed (the
        # offline train.py path has no supervision-work clocks and passes
        # nothing; a streaming run that took zero optimizer steps passes an empty
        # dict, which is falsy and therefore also omitted).
        summary["supervision_work"] = dict(supervision_work)
    if realized_cadence:
        # Issue #289: the run's REALIZED cadence -- source releases, admissions,
        # depth-insertion commits, supervision views, optimizer steps and MCMC
        # mutations, all denominated in OPTIMIZER STEPS (not iterations, which
        # diverge under gradient accumulation and under the placement_only /
        # colors_only diagnostic training modes), plus a requested-vs-realized
        # reconciliation verdict. The seven clock definitions live in
        # utils/realized_cadence.py's module docstring.
        #
        # Deliberately distinct from geom["mapping_clocks"] (four_clocks), which
        # measures WALL-CLOCK Hz -- the right clock for "can this keep up with a
        # live sensor", the wrong one for "did the map get enough optimizer steps
        # per unit of new data". Complementary; neither subsumes the other, and
        # a third mechanism should not be added.
        #
        # Optional/additive exactly like ingestion / mapping_clocks above:
        # omitting it reproduces the exact report.json from before this field
        # existed (the offline train.py path has no streaming clocks and passes
        # nothing).
        summary["realized_cadence"] = dict(realized_cadence)
        # Mirror the verdict into the existing #302 ingestion block rather than
        # standing up a second parallel mechanism: "which pacing ran" and "did
        # that pacing honor the requested cadence" are one question asked twice.
        _rc_recon = realized_cadence.get("reconciliation") or {}
        if _rc_recon and isinstance(summary.get("ingestion"), dict):
            summary["ingestion"]["cadence_verdict"] = _rc_recon.get("verdict")
            # Issue #725: the verdict above is now derived from the realized
            # inter-release GAP, not the budget-wide ratio -- mirror both so a
            # reader of report.json["ingestion"] alone (without following into
            # realized_cadence) sees the field the verdict actually came from,
            # not just the deprecated-for-verdict one.
            summary["ingestion"]["realized_release_gap_steps"] = _rc_recon.get(
                "realized_release_gap_steps")
            summary["ingestion"]["realized_steps_per_release"] = _rc_recon.get(
                "realized_steps_per_release")
            summary["ingestion"]["cadence_message"] = _rc_recon.get("message")
    # Scene fingerprint (unique-scene contract): content hash of the GT test
    # frames. Bag replays record only source_path='ros2://live', so the run
    # name is the ONLY scene identity — which silently drifted once ("floor3"
    # runs actually replaying the handheld bag). The fingerprint hashes the
    # actual pixels the eval sees: sha1 over 32x32 downsampled GT of up to 3
    # test cameras (first/middle/last). Identical scene + holdout => identical
    # fingerprint; different bag => different fingerprint, regardless of name.
    try:
        summary["scene_fingerprint"] = _scene_fingerprint(test_cams or train_cams)
    except Exception:
        summary["scene_fingerprint"] = None
    # Composition roots that already own a concrete source identity pass it
    # directly. The environment remains a compatibility fallback for legacy
    # launch scripts, but is no longer the only route to a gradeable report.
    _scene_src = str(scene_source or os.environ.get("SCENE_SOURCE", ""))
    if _scene_src:
        summary["scene_source"] = _scene_src
    if pose_provenance:
        summary["pose_provenance"] = dict(pose_provenance)
        if pose_provenance.get("tier") == "unknown":
            print(
                f"[report] iter={iteration} !!! pose_provenance=unknown — "
                "quality claims for this run are provenance-unverified.",
                flush=True,
            )
    if not summary["eval_valid"]:
        print(
            f"[report] iter={iteration} EVAL NOT VALID — "
            f"{'degenerate (test==train) split' if _degenerate else 'no held-out test views'}; "
            f"test PSNR will not be reported as a generalization metric.",
            flush=True,
        )

    try:
        from lpipsPyTorch import lpips as _lpips
    except ImportError as exc:
        if test_cams:
            raise RuntimeError("LPIPS is required for exact inline score evidence") from exc
        _lpips = None

    # Issue #437: per-split accounting of cameras the metrics loop attempted and
    # of the ones whose render/GT-fetch raised (and were therefore silently
    # dropped from the PSNR/SSIM/LPIPS average). Kept in an enclosing-scope dict
    # rather than widening _compute_metrics' return arity, so neither call site
    # changes and there is no risk of a missed unpack.
    #
    # SCOPE of the failure count: it covers the render + _gt_image step only --
    # the `_lpips(...)` call below sits OUTSIDE this try/except and has always
    # been allowed to propagate. A camera counted as "failed" here contributed
    # to none of the metrics; a camera NOT counted here contributed to PSNR and
    # (when lpipsPyTorch imported) to LPIPS. Do not read this number as "LPIPS
    # failures", and do not widen the try to make it one -- that would be a
    # behaviour change, not a reporting change.
    _render_failures: Dict[str, List[dict]] = {}
    _render_attempted: Dict[str, int] = {}

    def _compute_metrics(
        cams: List,
        split: str,
        render_cache: Optional[dict] = None,
        *,
        exact_artifact_root: Optional[str] = None,
        artifact_index_by_cam: Optional[dict] = None,
    ):
        from metrics import (
            SCORE_PARITY_TOLERANCES,
            mean_score_values,
            reset_score_artifact_set,
            save_score_pair,
            score_persisted_pairs,
            write_score_set_manifest,
        )
        from eligsir.utils.loss_utils import evaluation_ssim
        from eligsir.utils.score_contract import (
            apply_declared_score_clamp_pair,
            attach_score_provenance,
            camera_provenance,
            canonical_score_batch,
            score_canonical_pair,
        )

        psnr_list = []
        ssim_list = []
        lpips_list = []
        cams_used = []
        exact_entries = []
        metric_rows = []
        exact_metrics = None
        exact_score_device = None
        clamp_changed = 0
        clamp_total = 0
        clamp_nonfinite = 0
        raw_min = None
        raw_max = None
        score_provenance = None
        failures = _render_failures.setdefault(split, [])
        _render_attempted[split] = len(cams)
        if exact_artifact_root is not None:
            reset_score_artifact_set(exact_artifact_root)
        for i, cam in enumerate(cams):
            _cam_name = str(getattr(cam, "image_name", f"idx_{i}"))
            try:
                cached = (render_cache or {}).get(id(cam))
                raw_img = (
                    cached["render"] if cached is not None
                    else _render_image(cam, gaussians, render_fn, pipe, background)
                )
                raw_gt = _gt_image(cam)
                if raw_gt.device != raw_img.device:
                    raw_gt = attach_score_provenance(
                        raw_gt.to(raw_img.device, non_blocking=True),
                        camera_provenance(cam),
                    )
                score_img = raw_img.unsqueeze(0) if raw_img.ndim == 3 else raw_img
                score_gt = canonical_score_batch(
                    raw_gt, context=f"comparison_report:{split}:gt_batch"
                )
                img, gt, score_transform = apply_declared_score_clamp_pair(
                    score_img,
                    score_gt,
                    context=f"comparison_report:{split}",
                )
            except Exception as e:
                print(f"[report] render failed for {_cam_name}: {e}", flush=True)
                failures.append({
                    "camera": _cam_name,
                    "error_type": type(e).__name__,
                    "error": str(e),
                })
                continue

            artifact_name = None
            if exact_artifact_root is not None:
                artifact_index = (
                    artifact_index_by_cam.get(id(cam), i)
                    if artifact_index_by_cam is not None else i
                )
                artifact_name = f"{int(artifact_index):05d}.pt"
                # Intentionally outside the tolerated render/GT-fetch block:
                # scientific evidence persistence failures are fatal.
                exact_entries.append(save_score_pair(
                    raw_img,
                    raw_gt,
                    exact_artifact_root,
                    artifact_name=artifact_name,
                    view_name=_cam_name,
                ))
                exact_score_device = str(raw_img.device)

            render_transform = score_transform["render"]
            pair_provenance = score_transform["score_provenance"]
            if score_provenance is None:
                score_provenance = pair_provenance
            elif score_provenance != pair_provenance:
                raise ValueError(
                    f"score provenance changed across {split} views; refusing mixed-domain metrics"
                )
            clamp_changed += render_transform["samples_changed"]
            clamp_total += render_transform["samples_total"]
            clamp_nonfinite += render_transform["samples_nonfinite"]
            raw_min = (
                render_transform["value_min"]
                if raw_min is None
                else min(raw_min, render_transform["value_min"])
            )
            raw_max = (
                render_transform["value_max"]
                if raw_max is None
                else max(raw_max, render_transform["value_max"])
            )
            if exact_artifact_root is not None:
                cams_used.append(cam)
                del img, gt
                if torch.cuda.is_available() and (i + 1) % 32 == 0:
                    torch.cuda.empty_cache()
                continue
            scores = score_canonical_pair(
                img,
                gt,
                ssim_fn=evaluation_ssim,
                lpips_fn=(
                    (lambda pair_render, pair_gt: _lpips(
                        pair_render, pair_gt, net_type="vgg"
                    ))
                    if _lpips is not None else None
                ),
                context=f"comparison_report:{split}",
            )
            assert scores.ssim is not None
            psnr_value = float(scores.psnr.mean().item())
            ssim_value = float(scores.ssim.item())
            lpips_value = float(scores.lpips.item()) if scores.lpips is not None else None
            psnr_list.append(psnr_value)
            ssim_list.append(ssim_value)
            cams_used.append(cam)
            if lpips_value is not None:
                lpips_list.append(lpips_value)
            if artifact_name is not None:
                metric_rows.append({
                    "artifact": artifact_name,
                    "view_name": _cam_name,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                })
            del img, gt
            if torch.cuda.is_available() and (i + 1) % 32 == 0:
                torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if exact_artifact_root is not None and exact_entries:
            (
                exact_entries,
                score_provenance,
                exact_metrics,
                _,
            ) = score_persisted_pairs(
                exact_artifact_root,
                device=exact_score_device,
                lpips_fn=lambda pair_render, pair_gt: _lpips(
                    pair_render, pair_gt, net_type="vgg"
                ),
            )
            metric_rows = exact_metrics["per_view"]
            psnr_list = [row["psnr"] for row in metric_rows]
            ssim_list = [row["ssim"] for row in metric_rows]
            lpips_list = [row["lpips"] for row in metric_rows]
        summary.setdefault("score_domain_clamp", {})[split] = {
            "applied": True,
            "range": [0.0, 1.0],
            "nonfinite_policy": "rejected before clamp",
            "samples_nonfinite": clamp_nonfinite,
            "samples_changed": clamp_changed,
            "samples_total": clamp_total,
            "fraction_changed": clamp_changed / max(1, clamp_total),
            "value_min": raw_min,
            "value_max": raw_max,
        }
        if score_provenance is not None:
            summary.setdefault("score_provenance", {})[split] = score_provenance
        if exact_artifact_root is not None:
            manifest_metrics = exact_metrics or {
                "per_view": metric_rows,
                "aggregate": {
                    "psnr": mean_score_values(psnr_list),
                    "ssim": mean_score_values(ssim_list),
                    "lpips": mean_score_values(lpips_list),
                },
            }
            manifest = write_score_set_manifest(
                exact_artifact_root,
                entries=exact_entries,
                split=split,
                iteration=iteration,
                role="inline_scientific_metric_inputs",
                score_provenance=score_provenance,
                metrics=manifest_metrics,
                tolerances=SCORE_PARITY_TOLERANCES,
            )
            manifest_path = os.path.join(exact_artifact_root, "manifest.json")
            artifacts.append(manifest_path)
            summary.setdefault("score_artifacts", {})[split] = {
                "path": exact_artifact_root,
                "manifest": manifest_path,
                "n_views": manifest["n_views"],
                "artifact_set_sha256": manifest["artifact_set_sha256"],
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            }
        return psnr_list, ssim_list, lpips_list, cams_used

    _RENDER_FAILURE_CAMS_CAP = 50

    def _record_render_failures(split: str, n_scored: int) -> None:
        """Issue #437: publish the split's render-failure accounting into `summary`.

        Writes, unconditionally (no `_degenerate` gating -- these are counts, not
        holdout fingerprints, so they mirror how n_*_metrics / test_metrics_max /
        test_metrics_subsampled are already set):

          n_<split>_metrics_attempted  cameras handed to the metrics loop
          n_<split>_metrics_failed     of those, how many raised and were skipped
          <split>_metrics_render_failures  detail block, only when >0

        The point of the pair is that
            n_<split>_metrics + n_<split>_metrics_failed == n_<split>_metrics_attempted
        so a consumer can finally tell "scored 128 of 315 because the
        test_metrics_max cap subsampled the pool" (attempted == 128, failed == 0)
        apart from "scored 128 of 315 because 187 renders raised"
        (attempted == 315, failed == 187) -- previously indistinguishable, since
        only the scored count was ever recorded.
        """
        failures = _render_failures.get(split, [])
        attempted = _render_attempted.get(split, n_scored + len(failures))
        summary[f"n_{split}_metrics_attempted"] = int(attempted)
        summary[f"n_{split}_metrics_failed"] = int(len(failures))
        if not failures:
            return
        error_types: Dict[str, int] = {}
        for f in failures:
            error_types[f["error_type"]] = error_types.get(f["error_type"], 0) + 1
        names = [f["camera"] for f in failures]
        summary[f"{split}_metrics_render_failures"] = {
            "n_failed": len(failures),
            "n_attempted": int(attempted),
            "n_scored": int(n_scored),
            # Capped: a whole-split failure (e.g. no working GPU) would otherwise
            # write hundreds of identical entries into report.json.
            "failed_cameras": names[:_RENDER_FAILURE_CAMS_CAP],
            "failed_cameras_truncated": len(names) > _RENDER_FAILURE_CAMS_CAP,
            "error_types": error_types,
            "example_error": failures[0]["error"],
        }
        print(
            f"[report] WARNING: {len(failures)} of {attempted} {split} camera(s) FAILED to "
            f"render and were EXCLUDED from mean_{split}_psnr -- it is a "
            f"{n_scored}-view mean, not an {attempted}-view one. "
            f"error_types={error_types}; first failure: "
            f"{names[0]}: {failures[0]['error']}. "
            f"Do not compare this run's {split} metrics against a run with a different "
            f"n_{split}_metrics_failed. See issue #437.",
            flush=True,
        )

    def _representative_cams(cams: List, max_items: int) -> List:
        if not cams or max_items <= 0:
            return []
        if len(cams) <= max_items:
            return list(cams)
        idxs = np.linspace(0, len(cams) - 1, num=max_items, dtype=np.int64)
        # Preserve order while removing duplicates from integer rounding.
        seen = set()
        out = []
        for idx in idxs.tolist():
            if idx in seen:
                continue
            seen.add(idx)
            out.append(cams[idx])
        return out

    def _write_visual_sheet(cams: List, split: str, max_items: int,
                             ssaa_factor: int = 1, ssaa_sharpen: float = 0.0) -> Optional[str]:
        """``ssaa_factor``/``ssaa_sharpen`` (issue #1323, presentation-only): each visual
        cam is rendered at ``ssaa_factor``x resolution then area-downsampled back to its
        native size before anything else happens, so the GT-diff/contact-sheet code below
        is unaffected -- only the ``render`` cell is additionally sharpened. This sheet
        (``{split}_contact_sheet.png``) does its own independent render pass with no
        connection to ``_compute_metrics``, so this is safe (see
        ``utils/presentation_ssaa.py`` module docstring)."""
        visual_cams = _representative_cams(cams, max_items=max_items)
        if not visual_cams:
            return None
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        from eligsir.utils.presentation_ssaa import make_supersampled_camera, downsample_tensor_chw, unsharp_mask
        ssaa_factor = int(ssaa_factor) if ssaa_factor else 1
        render_cams_v = ([make_supersampled_camera(c, ssaa_factor) for c in visual_cams]
                          if ssaa_factor > 1 else visual_cams)
        if batch_render_fn is not None:
            try:
                batch_imgs_v = _batch_render_images(
                    render_cams_v, gaussians, pipe, background,
                    batch_render_fn=batch_render_fn,
                )
            except Exception as e:
                print(f"[report] batch {split} sheet render failed, falling back: {e}", flush=True)
                batch_imgs_v = []
        else:
            batch_imgs_v = []
        use_batch_v = len(batch_imgs_v) == len(visual_cams)
        pairs_for_visual_sheet: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for i, cam in enumerate(visual_cams):
            try:
                if use_batch_v:
                    img = batch_imgs_v[i].clamp(0.0, 1.0)
                else:
                    img = _render_image(render_cams_v[i], gaussians, render_fn, pipe, background)
                if ssaa_factor > 1:
                    img = downsample_tensor_chw(
                        img, ssaa_factor, target_hw=(cam.image_height, cam.image_width))
                gt = _gt_image(cam)
            except Exception as e:
                print(f"[report] {split} sheet render failed for {cam.image_name}: {e}", flush=True)
                continue
            name = str(getattr(cam, "image_name", f"view_{len(pairs_for_visual_sheet):04d}"))
            r_u8 = _to_uint8_hwc(img, label=f"{split}:{name}:render")
            if ssaa_sharpen:
                r_u8 = unsharp_mask(r_u8, ssaa_sharpen)
            g_u8 = _to_uint8_hwc(gt, label=f"{split}:{name}:gt")
            d_u8 = _abs_diff(img, gt)
            side = _hconcat_with_gap([r_u8, g_u8, d_u8], gap_px=4)
            _save_png(os.path.join(split_dir, f"{name}.png"), side)
            pairs_for_visual_sheet.append((name, r_u8, g_u8))
            del img, gt
        if not pairs_for_visual_sheet:
            return None
        sheet = _build_contact_sheet(
            pairs_for_visual_sheet,
            cols=contact_sheet_cols,
            cell_width=contact_sheet_cell_width,
        )
        sheet_path = os.path.join(out_dir, f"{split}_contact_sheet.png")
        _save_png(sheet_path, sheet)
        artifacts.append(sheet_path)
        return sheet_path

    _train_metric_limit = len(train_cams) if train_metrics_max is None else int(train_metrics_max)
    _test_metric_limit = len(test_cams) if test_metrics_max is None else int(test_metrics_max)

    # Defined here (not just inside "if test_cams:") because the geometry
    # section's _visual_cam_render_cache.clear() below runs unconditionally --
    # an empty test_cams (--trajectory recorded/dense reports) previously hit
    # a NameError on that line, silently swallowed by the geometry section's
    # own try/except (found 2026-08-04 verifying scene_vs_trajectory wiring).
    _visual_cam_render_cache: dict = {}

    # --- Trajectory-coverage / run-validity block ---------------------------
    # UNCONDITIONAL (issue #504, 2026-08-14): this used to live inside the
    # "if test_cams:" body below, so a report with no test cameras got NO
    # coverage block at all -- and with it no map_reinit_invalidated, no
    # stream_end_reason, no frame_drop_detected. Those safety fields have
    # nothing to do with the holdout split (CLAUDE.md gives
    # map_reinit_invalidated the same standing as eval_valid == False, i.e.
    # they matter MOST on a run whose eval is already suspect), and every one
    # of them is computable with zero test cameras. Measured on disk: 3 of 400
    # real reports had no block for exactly this reason. The partition-derived
    # fields are passed as explicitly not-measured (partition_psnr=None ->
    # partition_data_available=False, verdict UNKNOWN_NO_PARTITION_DATA)
    # rather than defaulted to a confident-looking PARTIAL -- see
    # compute_trajectory_coverage's docstring. When a real holdout eval runs,
    # the partitioned-eval block below OVERWRITES this with the measured
    # version (direct assignment, not setdefault).
    #
    # WS-3 / final review (2026-08-06, PR #37): derived from streaming_scene's
    # own frame source rather than threading a new parameter through every
    # write_post_training_report() call site -- see
    # ROS2LiveStreamingFrameSource.frames_dropped_by_inner_queue's
    # docstring for what this counts and why it matters for bag-replay
    # full-coverage validity.
    _frame_source = getattr(streaming_scene, "_frame_source", None)
    _residual_attempts = int(
        getattr(streaming_scene, "_residual_replay_score_attempts", 0) or 0
    )
    if _residual_attempts:
        summary["residual_replay_score_contract"] = {
            "attempts": _residual_attempts,
            "uniform_fallbacks": int(
                getattr(streaming_scene, "_residual_replay_score_fallbacks", 0) or 0
            ),
            "last_failure": getattr(
                streaming_scene, "_residual_replay_last_failure", None
            ),
            "score_domain_clamp": getattr(
                streaming_scene, "_residual_replay_score_clamp", None
            ),
        }
    if streaming_scene is not None:
        _global_psnr_attempts = int(
            getattr(streaming_scene, "_global_psnr_measurement_attempts", 0) or 0
        )
        _global_psnr_successes = int(
            getattr(streaming_scene, "_global_psnr_measurement_successes", 0) or 0
        )
        _global_psnr_failures = int(
            getattr(streaming_scene, "_global_psnr_measurement_failures", 0) or 0
        )
        summary["global_psnr_measurement"] = {
            "status": (
                "VALID"
                if _global_psnr_attempts > 0
                and _global_psnr_successes == _global_psnr_attempts
                and _global_psnr_failures == 0
                else "NOT_MEASURED"
                if _global_psnr_attempts == 0
                else "INVALID"
            ),
            "valid": (
                True
                if _global_psnr_attempts > 0
                and _global_psnr_successes == _global_psnr_attempts
                and _global_psnr_failures == 0
                else None
                if _global_psnr_attempts == 0
                else False
            ),
            "attempts": _global_psnr_attempts,
            "successes": _global_psnr_successes,
            "failures": _global_psnr_failures,
            "last_failure": getattr(
                streaming_scene, "_global_psnr_last_failure", None
            ),
        }
    _frames_dropped_inner = int(
        getattr(_frame_source, "frames_dropped_by_inner_queue", 0) or 0
    )
    # Issue #1135: same getattr-with-default pattern as _frames_dropped_inner
    # above -- read directly off streaming_scene rather than threading a new
    # parameter through every write_post_training_report() call site. See
    # StreamingScene.__init__'s own comment for what this counts.
    _stale_pose_holdout_rejected = int(
        getattr(streaming_scene, "_stale_pose_holdout_rejected", 0) or 0
    )
    # Issue #346 follow-up: true_final_iteration (when supplied) works
    # around the `iteration` parameter's known online_mode sentinel bug
    # for JUST the trajectory_coverage.final_iteration field -- see this
    # function's own docstring above. Never falls back silently to a
    # wrong value: when the caller doesn't supply it, this is simply
    # `iteration` itself (correct for every non-buggy call site).
    _effective_final_iteration = (
        true_final_iteration if true_final_iteration is not None else iteration
    )
    # Issue #602: the admission target the run actually RESOLVED to, as opposed
    # to `frame_admission` (the raw requested flag). Threaded into the coverage
    # verdict so a soft-all run's budgeted shortfall reads as
    # PARTIAL_BY_BUDGET rather than PARTIAL ("lost trajectory"). None whenever
    # the run declared no target -- every pre-#600 run -- which reproduces the
    # previous verdict behavior exactly.
    _admission_target_resolved = None
    if isinstance(admission_cadence, dict):
        _admission_target_resolved = (
            (admission_cadence.get("resolved_semantics") or {})
            .get("admission_target_resolved")
        )
    summary.setdefault("trajectory_coverage", compute_trajectory_coverage(
        scene_source=summary.get("scene_source"),
        n_reg=_registry_n_source_frames(summary.get("scene_source")),
        frames_admitted=(
            frames_admitted_override
            if frames_admitted_override is not None
            else len(train_cams or []) + len(test_cams or [])
        ),
        partition_psnr=None,
        frame_admission=frame_admission,
        stream_end_reason=stream_end_reason,
        tracking_reinit_events=tracking_reinit_events,
        frames_dropped_by_inner_queue=_frames_dropped_inner,
        eval_valid=summary.get("eval_valid"),
        bag_replay_geometry_fields=bag_replay_report,
        source_ingress_geometry_fields=source_ingress,
        last_ingest_iter=last_ingest_iter,
        final_iteration=_effective_final_iteration,
        mapping_phase_source=mapping_phase_source,
        ingest_counters=ingest_counters,
        bag_source_manifest=bag_source_manifest,
        admission_target_resolved=_admission_target_resolved,
        sync_pair_ledger=sync_pair_ledger,
        stale_pose_holdout_rejected=_stale_pose_holdout_rejected,
        coverage_counters=coverage_counters,
    ))

    # --- Test split: side-by-side + contact sheet + metrics -----------------------------
    psnrs: List[float] = []
    lpipss: List[float] = []
    pairs_for_sheet: List[Tuple[str, np.ndarray, np.ndarray]] = []
    if test_cams:
        metric_test_cams = _representative_cams(test_cams, max_items=_test_metric_limit)
        test_dir = os.path.join(out_dir, "test")
        os.makedirs(test_dir, exist_ok=True)
        visual_test_cams = _representative_cams(
            test_cams,
            max_items=max(1, contact_sheet_cols * 2),
        )
        # One render pass per visual test cam feeds RGB sheet, normals sheets,
        # centers/rings overlays, and the geometry TierB/C section.  The combined
        # render_normals=True call replaces the previous 3 separate GPU dispatches.
        normals_pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []
        surf_pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []
        centers_pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []
        rings_pairs: List[Tuple[str, np.ndarray, np.ndarray]] = []
        # Map cam id → full render dict so _compute_metrics can reuse for visual cams.
        _visual_cam_render_cache: dict = {}

        for i, cam in enumerate(visual_test_cams):
            try:
                full_out = _render_full_pass(cam, gaussians, render_fn, pipe, background)
                if full_out is None:
                    continue
                _visual_cam_render_cache[id(cam)] = full_out
                if not presentation_enabled:
                    # Metrics below consume this exact render.  Do not turn a
                    # presentation opt-out into a second scoring pass.
                    continue
                img = full_out["render"].clamp(0.0, 1.0)
                gt = _gt_image(cam)
                name = str(getattr(cam, "image_name", f"view_{i:04d}"))
                r_u8 = _to_uint8_hwc(img, label=f"test:{name}:render")
                # Issue #1323: sharpen ONLY this display copy of the render, AFTER the
                # float `img` tensor has already been cached in _visual_cam_render_cache
                # (line above) and used for `d_u8` below -- `_compute_metrics` reads
                # `full_out["render"]` from that cache directly, never `r_u8`, so this
                # cannot move mean_test_psnr/SSIM/LPIPS. No SSAA re-render here (that
                # WOULD require touching the shared render pass) -- sharpening-only, see
                # utils/presentation_ssaa.py's module docstring for the full reasoning.
                if presentation_sharpen_strength:
                    from eligsir.utils.presentation_ssaa import unsharp_mask as _unsharp_mask
                    r_u8 = _unsharp_mask(r_u8, presentation_sharpen_strength)
                g_u8 = _to_uint8_hwc(gt, label=f"test:{name}:gt")

                # RGB side-by-side + contact sheet
                d_u8 = _abs_diff(img, gt)
                _save_png(os.path.join(test_dir, f"{name}.png"),
                          _hconcat_with_gap([r_u8, g_u8, d_u8], gap_px=4))
                pairs_for_sheet.append((name, r_u8, g_u8))

                # Normals sheets (splat-axis + depth-derived) from same render dict
                n_raw = full_out.get("rendered_normals")
                s_raw = full_out.get("rendered_surf_normals")
                if n_raw is not None:
                    n_u8 = _to_uint8_hwc(n_raw.clamp(0.0, 1.0), label=f"test:{name}:normals")
                    normals_pairs.append((name, n_u8, g_u8))
                    if s_raw is not None:
                        s_u8 = _to_uint8_hwc(s_raw.clamp(0.0, 1.0), label=f"test:{name}:surf_normals")
                        surf_pairs.append((name, s_u8, g_u8))
                        _save_png(os.path.join(test_dir, f"{name}_normals.png"),
                                  _hconcat_with_gap([n_u8, s_u8, g_u8], gap_px=4))
                    else:
                        _save_png(os.path.join(test_dir, f"{name}_normals.png"),
                                  _hconcat_with_gap([n_u8, g_u8], gap_px=4))

                # Centers / rings overlays reuse the already-rendered base
                centers_u8, rings_u8, _base = _render_overlay_images(
                    cam, gaussians, render_fn, pipe, background,
                    _cached_out=full_out)
                if centers_u8 is not None:
                    _save_png(os.path.join(test_dir, f"{name}_centers.png"),
                              _hconcat_with_gap([centers_u8, g_u8], gap_px=4))
                    centers_pairs.append((name, centers_u8, g_u8))
                    if rings_u8 is not None:
                        _save_png(os.path.join(test_dir, f"{name}_rings.png"),
                                  _hconcat_with_gap([rings_u8, g_u8], gap_px=4))
                        rings_pairs.append((name, rings_u8, g_u8))

                del img, gt
                # full_out kept in _visual_cam_render_cache for geometry TierB/C reuse
            except Exception as e:
                print(f"[report] render failed for {getattr(cam, 'image_name', i)}: {e}", flush=True)

        if pairs_for_sheet:
            sheet = _build_contact_sheet(
                pairs_for_sheet,
                cols=contact_sheet_cols,
                cell_width=contact_sheet_cell_width,
            )
            sheet_path = os.path.join(out_dir, "test_contact_sheet.png")
            _save_png(sheet_path, sheet)
            artifacts.append(sheet_path)

        if normals_pairs:
            normals_sheet = _build_contact_sheet(
                normals_pairs, cols=contact_sheet_cols,
                cell_width=contact_sheet_cell_width,
            )
            normals_sheet_path = os.path.join(out_dir, "normals_contact_sheet.png")
            _save_png(normals_sheet_path, normals_sheet)
            artifacts.append(normals_sheet_path)
            summary["normals_contact_sheet"] = normals_sheet_path
        if surf_pairs:
            surf_sheet = _build_contact_sheet(
                surf_pairs, cols=contact_sheet_cols,
                cell_width=contact_sheet_cell_width,
            )
            surf_sheet_path = os.path.join(out_dir, "surf_normals_contact_sheet.png")
            _save_png(surf_sheet_path, surf_sheet)
            artifacts.append(surf_sheet_path)
            summary["surf_normals_contact_sheet"] = surf_sheet_path

        for _pairs, _fname, _key in (
            (centers_pairs, "centers_contact_sheet.png", "centers_contact_sheet"),
            (rings_pairs, "rings_contact_sheet.png", "rings_contact_sheet"),
        ):
            if _pairs:
                _sheet = _build_contact_sheet(_pairs, cols=contact_sheet_cols,
                                              cell_width=contact_sheet_cell_width)
                _p = os.path.join(out_dir, _fname)
                _save_png(_p, _sheet)
                artifacts.append(_p)
                summary[_key] = _p

        # Metrics pass: reuse cached renders for visual cams, re-render the rest.
        _test_artifact_indices = {id(cam): i for i, cam in enumerate(test_cams)}
        psnrs, ssims, lpipss, test_cams_used = _compute_metrics(
            metric_test_cams,
            "test",
            render_cache=_visual_cam_render_cache,
            exact_artifact_root=os.path.join(out_dir, "score_tensors", "test"),
            artifact_index_by_cam=_test_artifact_indices,
        )
        summary["n_test_metrics"] = len(psnrs)
        # Issue #437: record how many of the attempted cameras raised, so a
        # shrunken n_test_metrics is visible rather than silent. Placed here,
        # immediately alongside n_test_metrics, deliberately.
        _record_render_failures("test", len(psnrs))
        # Issue #420: fingerprint the EXACT cameras mean_test_psnr was averaged
        # over. `realized_holdout_hash` above covers the realized holdout POOL
        # (`test_cams`); this covers the subset actually scored -- i.e. after
        # `_representative_cams(..., _test_metric_limit)` capped it and after
        # `_compute_metrics` dropped any camera whose render raised. Two runs
        # can share an identical realized pool hash, n_test AND n_test_metrics
        # while averaging over different cameras (different pool ORDER at the
        # same evenly-spaced positions, or a render failure on one side), and
        # before this field existed compare_against_baseline reported that at
        # its strongest comparability level.
        #
        # Gated on `not _degenerate` for the same reason realized_holdout_hash
        # is: a degenerate split (*_bootstrap_views debug report) must never
        # emit anything that looks like a real holdout fingerprint. Withheld
        # (key absent) rather than null when the cameras carry no stable
        # identifier -- absent degrades to a lower verification level, which
        # is safer than a hash that looks authoritative but isn't.
        #
        # Scope note: `test_cams_used` is the PSNR-scored list. LPIPS is
        # computed over the same cameras when lpipsPyTorch imports, and
        # skipped entirely when it does not -- so this hash describes the
        # PSNR/SSIM-scored set exactly, and the LPIPS-scored set whenever
        # LPIPS was computed at all.
        if not _degenerate:
            try:
                _scored_hash = _camera_set_hash(test_cams_used)
            except Exception:
                _scored_hash = None
            if _scored_hash is not None:
                summary.setdefault("eval_metadata", {})
                summary["eval_metadata"]["scored_holdout_hash"] = _scored_hash
        if not presentation_enabled:
            # These are the actual metric inputs, not a promise that an
            # equivalent presentation snapshot was captured.  A later report
            # must establish its own frozen renderer/camera state.
            summary["presentation_artifacts"]["inputs"].update({
                "scored_source_indices": [camera_source_index(cam) for cam in test_cams_used],
                "scored_holdout_hash": summary.get("eval_metadata", {}).get("scored_holdout_hash"),
            })
        # W3-42: make the test_metrics_max subsampling loud and machine-readable.
        # `n_test` (the full holdout size) and `n_test_metrics` (how many views
        # mean_test_psnr was actually averaged over) have always both been
        # present in report.json, but nothing ever compared them -- a real
        # campaign (kitchen1 G0-G6, n_test=253 > the default cap of 128) quoted
        # mean_test_psnr as if it were the full-holdout mean without anyone
        # noticing only 128 of 253 views were scored (see docs/architecture/
        # 2026-08-09-w3-42-psnr-subsample-audit.md). `test_metrics_subsampled`
        # is now explicit so a future reader/script doesn't have to manually
        # diff the two counts, and a loud one-time print flags it at report
        # time (same "loud, not silent" convention as W3-45's retirement-cap
        # warning) rather than relying on someone remembering to check.
        summary["test_metrics_max"] = None if test_metrics_max is None else int(test_metrics_max)
        summary["test_metrics_subsampled"] = bool(len(test_cams) > _test_metric_limit)
        if summary["test_metrics_subsampled"]:
            print(
                f"[report] WARNING: mean_test_psnr is a {len(psnrs)}-view SUBSAMPLE mean, "
                f"not the full {len(test_cams)}-view holdout -- "
                f"streaming_report_test_metrics_max={_test_metric_limit} capped it. "
                f"Views are evenly-spaced (np.linspace), so cross-arm comparisons at the "
                f"SAME n_test remain apples-to-apples, but do not quote this number as a "
                f"full-holdout PSNR. See docs/architecture/2026-08-09-w3-42-psnr-subsample-audit.md.",
                flush=True,
            )

        # --- Zoomed-out overview views (synthetic pulled-back, wide-FoV cameras) ---
        # Each overview camera is a full extra render + per-Gaussian marker pass.
        # Skippable for fast reports via SPLAT_REPORT_OVERVIEW=0.
        try:
            if presentation_enabled and os.environ.get("SPLAT_REPORT_OVERVIEW", "1") == "1":
                overview_cams = _make_overview_cameras(
                    train_cams if train_cams else test_cams, gaussians, n=4)
            else:
                overview_cams = []
            centers_ov: List[Tuple[str, np.ndarray, np.ndarray]] = []
            rings_ov: List[Tuple[str, np.ndarray, np.ndarray]] = []
            if overview_cams:
                ov_dir = os.path.join(out_dir, "overview")
                os.makedirs(ov_dir, exist_ok=True)
            for i, oc in enumerate(overview_cams):
                centers_u8, rings_u8, base_u8 = _render_overlay_images(
                    oc, gaussians, render_fn, pipe, background)
                if centers_u8 is None or base_u8 is None:
                    continue
                name = f"ov{i:02d}"
                _save_png(os.path.join(ov_dir, f"{name}_rgb.png"), base_u8)
                _save_png(os.path.join(ov_dir, f"{name}_centers.png"), centers_u8)
                centers_ov.append((name, centers_u8, base_u8))
                if rings_u8 is not None:
                    _save_png(os.path.join(ov_dir, f"{name}_rings.png"), rings_u8)
                    rings_ov.append((name, rings_u8, base_u8))
            for _pairs, _fname, _key in (
                (centers_ov, "centers_overview_sheet.png", "centers_overview_sheet"),
                (rings_ov, "rings_overview_sheet.png", "rings_overview_sheet"),
            ):
                if _pairs:
                    _sheet = _build_contact_sheet(_pairs, cols=contact_sheet_cols,
                                                  cell_width=contact_sheet_cell_width)
                    _p = os.path.join(out_dir, _fname)
                    _save_png(_p, _sheet)
                    artifacts.append(_p)
                    summary[_key] = _p
        except Exception as e:
            print(f"[report] overview views failed: {e}", flush=True)

        # Neutral key prefix for a degenerate split so a bogus "test_*" number can
        # never originate from train-on-train views.
        _pfx = "view" if _degenerate else "test"
        _lbl = "view" if _degenerate else "test"
        if psnrs:
            summary[f"mean_{_pfx}_psnr"] = float(np.mean(psnrs))
            summary[f"min_{_pfx}_psnr"] = float(np.min(psnrs))
            summary[f"max_{_pfx}_psnr"] = float(np.max(psnrs))
            msg = (
                f"[report] iter={iteration} {_lbl} PSNR mean={float(np.mean(psnrs)):.2f}dB "
                f"min={float(np.min(psnrs)):.2f} max={float(np.max(psnrs)):.2f} "
                f"({len(psnrs)}/{len(test_cams)} views)"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                tb_writer.add_scalar(f"{log_prefix}/{_pfx}_psnr_mean", float(np.mean(psnrs)), iteration)
                tb_writer.add_scalar(f"{log_prefix}/{_pfx}_psnr_min", float(np.min(psnrs)), iteration)
                tb_writer.add_scalar(f"{log_prefix}/{_pfx}_psnr_max", float(np.max(psnrs)), iteration)
        if ssims:
            summary[f"mean_{_pfx}_ssim"] = float(np.mean(ssims))
            summary[f"min_{_pfx}_ssim"] = float(np.min(ssims))
            summary[f"max_{_pfx}_ssim"] = float(np.max(ssims))
            if tb_writer is not None:
                tb_writer.add_scalar(
                    f"{log_prefix}/{_pfx}_ssim_mean", float(np.mean(ssims)), iteration
                )
        if lpipss:
            summary[f"mean_{_pfx}_lpips"] = float(np.mean(lpipss))
            summary[f"min_{_pfx}_lpips"] = float(np.min(lpipss))
            summary[f"max_{_pfx}_lpips"] = float(np.max(lpipss))
            msg = (
                f"[report] iter={iteration} {_lbl} LPIPS mean={float(np.mean(lpipss)):.4f} "
                f"min={float(np.min(lpipss)):.4f} max={float(np.max(lpipss)):.4f}"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                tb_writer.add_scalar(f"{log_prefix}/{_pfx}_lpips_mean", float(np.mean(lpipss)), iteration)

        # --- Partitioned evaluation (only for a real holdout eval) -----------
        if psnrs and summary.get("eval_valid"):
            try:
                # Denominator contract: a PER-SCENE CONSTANT from eval/scenes.json
                # (n_source_frames), never the max index this run happened to
                # admit — admission varies run-to-run and would flip bin edges.
                _n_reg = _registry_n_source_frames(summary.get("scene_source"))
                if _n_reg and _n_reg > 1:
                    _max_src = _n_reg - 1
                    summary["partition_denominator_source"] = "registry"
                else:
                    _all_idx = [camera_source_index(c)
                                for c in list(train_cams or []) + list(test_cams or [])]
                    _all_idx = [i for i in _all_idx if i is not None]
                    _max_src = max(_all_idx) if _all_idx else 0
                    summary["partition_denominator_source"] = "fallback_max_cams"
                summary["partition_max_source_index"] = int(_max_src)
                per_view = []
                for metric_index, (cam, v) in enumerate(zip(test_cams_used, psnrs)):
                    src = camera_source_index(cam)
                    per_view.append({
                        "name": str(getattr(cam, "image_name", "")),
                        "source_index": src,
                        "partition": (partition_id(src, _max_src)
                                      if src is not None else None),
                        "psnr": float(v),
                        "ssim": float(ssims[metric_index]),
                        "lpips": (
                            float(lpipss[metric_index])
                            if len(lpipss) == len(psnrs) else None
                        ),
                    })
                summary["test_per_view"] = per_view
                summary.update(compute_partition_stats(per_view))
                summary["trajectory_coverage"] = compute_trajectory_coverage(
                    scene_source=summary.get("scene_source"),
                    n_reg=_n_reg,
                    frames_admitted=(
                        frames_admitted_override
                        if frames_admitted_override is not None
                        else len(train_cams or []) + len(test_cams or [])
                    ),
                    partition_psnr=summary.get("partition_psnr", {}),
                    frame_admission=frame_admission,
                                stream_end_reason=stream_end_reason,
                    tracking_reinit_events=tracking_reinit_events,
                    frames_dropped_by_inner_queue=_frames_dropped_inner,
                    eval_valid=summary.get("eval_valid"),
                    bag_replay_geometry_fields=bag_replay_report,
                    source_ingress_geometry_fields=source_ingress,
                    last_ingest_iter=last_ingest_iter,
                    final_iteration=_effective_final_iteration,
                    mapping_phase_source=mapping_phase_source,
                    ingest_counters=ingest_counters,
                    bag_source_manifest=bag_source_manifest,
                    admission_target_resolved=_admission_target_resolved,
                    sync_pair_ledger=sync_pair_ledger,
                    stale_pose_holdout_rejected=_stale_pose_holdout_rejected,
                    coverage_counters=coverage_counters,
                )
                _pp = summary["partition_psnr"]
                print("[report] partition PSNR (" +
                      summary["partition_denominator_source"] + "): " +
                      " ".join(f"{k}={_pp[k]:.2f}" if _pp[k] is not None else f"{k}=--"
                               for k in sorted(_pp)) +
                      f"  min={summary['partition_psnr_min']}"
                      f" std={summary['partition_psnr_std']}", flush=True)
            except Exception as e:
                print(f"[report] partitioned eval failed: {e}", flush=True)

            # PLY-side partition mass: bin final live Gaussians by the
            # partition of their nearest camera (train+test). Cheap, chunked.
            try:
                xyz = gaussians.get_xyz.detach()
                opa = gaussians.get_opacity.detach().reshape(-1)
                live_pts = xyz[opa > 0.05]
                _cams_all = list(train_cams or []) + list(test_cams or [])
                _centers, _cparts = [], []
                _max_src_i = int(summary.get("partition_max_source_index", 0) or 0)
                for c in _cams_all:
                    ctr = getattr(c, "camera_center", None)
                    src = camera_source_index(c)
                    if ctr is None or src is None:
                        continue
                    _centers.append(ctr.detach().reshape(-1)[:3])
                    _cparts.append(partition_id(src, _max_src_i))
                if _centers and live_pts.numel():
                    centers_t = torch.stack(_centers).to(live_pts)
                    cparts_t = torch.tensor(_cparts, device=live_pts.device)
                    counts = torch.zeros(PARTITIONS, dtype=torch.long,
                                         device=live_pts.device)
                    for s in range(0, live_pts.shape[0], 65536):
                        chunk = live_pts[s:s + 65536]
                        nearest = torch.cdist(chunk, centers_t).argmin(dim=1)
                        counts += torch.bincount(cparts_t[nearest],
                                                 minlength=PARTITIONS)
                    summary["partition_live_mass"] = {
                        f"p{i}": int(counts[i].item()) for i in range(PARTITIONS)}
            except Exception:
                pass  # non-essential; skip silently per contract
    else:
        print(f"[report] iter={iteration} no test views — skipping test PSNR/LPIPS/contact sheet", flush=True)

    # --- Train split: metrics ------------------------------------
    if train_cams:
        metric_train_cams = _representative_cams(train_cams, max_items=_train_metric_limit)
        train_psnrs, train_ssims, train_lpipss, train_cams_used = _compute_metrics(
            metric_train_cams, "train"
        )
        summary["n_train_metrics"] = len(train_psnrs)
        _record_render_failures("train", len(train_psnrs))  # issue #437
        train_per_view = []
        for index, camera in enumerate(train_cams_used):
            identity = getattr(camera, "_report_source_identity", None)
            if identity is None:
                identity_record = {
                    "status": "unknown",
                    "reason": "report_source_identity_missing",
                }
            else:
                identity_record = {
                    "status": "known",
                    "fingerprint": identity.fingerprint(),
                    "source_ordinal": int(identity.source_ordinal),
                    "timestamp_ns": int(identity.timestamp_ns),
                }
            train_per_view.append({
                "iteration": int(iteration),
                "view_name": str(getattr(camera, "image_name", f"idx_{index}")),
                "psnr": float(train_psnrs[index]),
                "ssim": float(train_ssims[index]),
                "lpips": (
                    float(train_lpipss[index])
                    if index < len(train_lpipss) else None
                ),
                "identity": identity_record,
                "mapping_phase_source": mapping_phase_source,
            })
        if len(train_per_view) != summary["n_train_metrics"]:
            raise RuntimeError("train_per_view row count does not match n_train_metrics")
        summary["train_per_view"] = train_per_view
        if train_psnrs:
            summary["mean_train_psnr"] = float(np.mean(train_psnrs))
            summary["min_train_psnr"] = float(np.min(train_psnrs))
            summary["max_train_psnr"] = float(np.max(train_psnrs))
            msg = (
                f"[report] iter={iteration} train PSNR mean={summary['mean_train_psnr']:.2f}dB "
                f"min={summary['min_train_psnr']:.2f} max={summary['max_train_psnr']:.2f} "
                f"({len(train_psnrs)}/{len(train_cams)} views)"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                tb_writer.add_scalar(f"{log_prefix}/train_psnr_mean", summary["mean_train_psnr"], iteration)
        if train_lpipss:
            summary["mean_train_lpips"] = float(np.mean(train_lpipss))
            summary["min_train_lpips"] = float(np.min(train_lpipss))
            summary["max_train_lpips"] = float(np.max(train_lpipss))
            msg = (
                f"[report] iter={iteration} train LPIPS mean={summary['mean_train_lpips']:.4f} "
                f"min={summary['min_train_lpips']:.4f} max={summary['max_train_lpips']:.4f}"
            )
            print(msg, flush=True)
            if tb_writer is not None:
                tb_writer.add_scalar(f"{log_prefix}/train_lpips_mean", summary["mean_train_lpips"], iteration)
        if train_ssims:
            summary["mean_train_ssim"] = float(np.mean(train_ssims))
            summary["min_train_ssim"] = float(np.min(train_ssims))
            summary["max_train_ssim"] = float(np.max(train_ssims))
        # The held-out test sheet is useful for metrics, but in streaming runs it
        # often shows future/unobserved cameras. A small representative train
        # sheet gives visually meaningful render/GT pairs for already ingested
        # views without changing the held-out evaluation numbers.
        if presentation_enabled:
            train_sheet_path = _write_visual_sheet(
                train_cams,
                split="train",
                max_items=max(1, contact_sheet_cols * 2),
                ssaa_factor=presentation_supersample_factor,
                ssaa_sharpen=presentation_sharpen_strength,
            )
            if train_sheet_path is not None:
                summary["train_contact_sheet"] = train_sheet_path

    _record_allocator_phase("scored_metrics_complete")

    # --- Trajectory MP4 over train cameras ------------------------------------
    # Video render+encode is heavy and synchronous; skippable via
    # --no-streaming_report_videos (report_videos=False, task 8) or, as a
    # secondary override on whichever mechanism does reach the container,
    # SPLAT_REPORT_VIDEOS=0. presentation_enabled is a separate,
    # pre-existing gate (unrelated to this flag) that must also be true.
    _do_videos = presentation_enabled and os.environ.get("SPLAT_REPORT_VIDEOS", "1" if report_videos else "0") == "1"
    # Compute the exact depth-montage camera identities before trajectory
    # rendering so the same-state subset can reuse one depth-capable render.
    depth_cams: List = []
    if train_cams and _do_videos:
        _depth_limit = mp4_max_frames if trajectory_max_frames is None else int(trajectory_max_frames)
        depth_cams = [
            cam for cam in _representative_cams(train_cams, max_items=_depth_limit)
            if hasattr(cam, "_streaming_depth_path") or hasattr(cam, "_streaming_frame")
        ]
    _depth_cam_ids = {id(cam) for cam in depth_cams}
    _shared_video_render_cache: dict = {}
    if train_cams and _do_videos:
        _traj_limit = mp4_max_frames if trajectory_max_frames is None else int(trajectory_max_frames)
        cams_for_video = _representative_cams(train_cams, max_items=_traj_limit)
        if not cams_for_video:
            print(f"[report] iter={iteration} no trajectory views selected — skipping trajectory MP4", flush=True)
        else:
            frame_repeats = None
            if all(hasattr(c, "camera_center") for c in cams_for_video):
                # Issue #397: only spatially reorder + spatially-pace a
                # camera set that has no trustworthy per-frame timestamps
                # (e.g. RealityScan's UUID-named unordered capture -- see
                # _order_cams_for_trajectory's docstring). A sequential,
                # timestamped capture (TUM/Orbbec/rosbag) instead keeps its
                # real order AND every real camera (never reordered,
                # dropped, or spatially interpolated) and is paced onto the
                # real capture duration by REPEATING each rendered frame on
                # ENCODE for as many output frames as its real time slice
                # covers (utils.trajectory_video.distribute_frame_repeats)
                # -- not by rendering more frames. This is what fixes the
                # motivating issue's "video plays 2-3x too fast" symptom
                # (measured, not just inferred: the real fr3 artifact this
                # issue cites encodes at a flat 30fps regardless of real
                # capture span, root-caused to _pace_cams_for_trajectory's
                # synthetic interpolated frames carrying no
                # _streaming_timestamp, which made the pre-existing
                # _infer_realtime_fps fall back to its default fps the
                # moment any spatial pacing had run at all) WITHOUT
                # multiplying report-step render cost: repeated frames are
                # near-free for libx264 to encode (near-zero residual
                # between identical consecutive frames), unlike unique
                # renders, so the render budget stays exactly what it was
                # before this fix (mp4_max_frames/trajectory_max_frames).
                _raw_ts = [getattr(c, "_streaming_timestamp", None) for c in cams_for_video]
                _ts_floats = [float(t) for t in _raw_ts] if cams_have_usable_timestamps(_raw_ts) else []
                # cams_for_video is already in source order (see
                # _representative_cams); for a real sequential capture that
                # IS chronological order. Require it to actually be
                # non-decreasing before trusting index-for-index alignment
                # between cams and timestamps -- an out-of-order timestamp
                # set (e.g. a loop-closure-reordered replay) falls back to
                # the spatial path instead of silently misaligning them.
                _ts_monotonic = bool(_ts_floats) and all(
                    b >= a for a, b in zip(_ts_floats, _ts_floats[1:]))
                if _ts_monotonic:
                    ordering = "timestamp"
                    video_fps = max(1, int(mp4_fps))  # conventional presentation rate, not inferred
                    timing = compute_trajectory_timing(_ts_floats, default_fps=video_fps)
                    summary["trajectory_timing_source"] = timing["timing_source"]
                    if timing["duration_s"] is not None:
                        _source_duration_s = float(timing["duration_s"])
                        summary["trajectory_source_duration_s"] = _source_duration_s
                        frame_repeats = distribute_frame_repeats(
                            _ts_floats, video_fps, max_encoded_frames=DEFAULT_MAX_ENCODED_FRAMES)
                        _video_duration_s = sum(frame_repeats) / video_fps
                        summary["trajectory_video_duration_s"] = _video_duration_s
                        _playback_speed = compute_playback_speed(_source_duration_s, _video_duration_s)
                        if _playback_speed is not None:
                            summary["trajectory_playback_speed"] = _playback_speed
                        # Issue #397 follow-up ("chonky/stuttery"): turn each
                        # slice's repeated-hold frames into an interpolated
                        # sweep. Purely presentation -- runs strictly AFTER
                        # duration/fps were derived from the ORIGINAL
                        # timestamps above, and its output only ever reaches
                        # render_camera_path below, never
                        # _infer_realtime_fps/scoring. sum(frame_repeats) is
                        # unchanged by this call (see
                        # _expand_cams_with_interpolation's docstring), so
                        # trajectory_video_duration_s/playback_speed already
                        # recorded above remain correct with interpolation on
                        # or off.
                        if presentation_trajectory_interpolate:
                            _n_before = len(cams_for_video)
                            cams_for_video, frame_repeats = _expand_cams_with_interpolation(
                                cams_for_video, frame_repeats,
                                max_extra_renders=presentation_trajectory_interpolate_max_extra_renders,
                                max_per_slice=presentation_trajectory_interpolate_max_per_slice,
                            )
                            summary["trajectory_interpolation"] = {
                                "enabled": True,
                                "added_renders": len(cams_for_video) - _n_before,
                                "max_extra_renders": presentation_trajectory_interpolate_max_extra_renders,
                                "max_per_slice": presentation_trajectory_interpolate_max_per_slice,
                            }
                        else:
                            summary["trajectory_interpolation"] = {"enabled": False}
                else:
                    ordering = "spatial"
                    cams_for_video = _order_cams_for_trajectory(cams_for_video)
                    cams_for_video = _pace_cams_for_trajectory(cams_for_video)
                    video_fps = _infer_realtime_fps(cams_for_video, mp4_fps)
            else:
                ordering = "source"
                video_fps = mp4_fps
            summary["trajectory_ordering"] = ordering
            summary["trajectory_fps"] = video_fps
            mp4_path = os.path.join(out_dir, "trajectory.mp4")
            written = render_camera_path(
                cams_for_video, gaussians, render_fn, pipe, background,
                out_path=mp4_path, fps=video_fps, batch_render_fn=batch_render_fn,
                min_width=report_video_min_width, max_width=report_video_max_width,
                frame_repeats=frame_repeats,
                supersample_factor=presentation_supersample_factor,
                sharpen_strength=presentation_sharpen_strength,
                depth_cam_ids=_depth_cam_ids,
                render_cache_out=_shared_video_render_cache,
            )
            if written is not None:
                artifacts.append(written)
                _w0 = getattr(cams_for_video[0], "image_width", None)
                _h0 = getattr(cams_for_video[0], "image_height", None)
                if _w0 and _h0:
                    _out_w, _out_h = scale_dims_for_video(
                        _w0, _h0, min_width=report_video_min_width, max_width=report_video_max_width)
                    summary["trajectory_video"] = {
                        "width": _out_w,
                        "height": _out_h,
                        "upscaled": is_upscaled(_w0, _h0, _out_w, _out_h),
                        "min_width": report_video_min_width,
                        "max_width": report_video_max_width,
                        "fps": video_fps,
                        "rendered_frame_count": len(cams_for_video),
                        "encoded_frame_count": (
                            len(cams_for_video) if frame_repeats is None else sum(frame_repeats)),
                        "ordering": ordering,
                        "source_duration_s": summary.get("trajectory_source_duration_s"),
                        "video_duration_s": summary.get("trajectory_video_duration_s"),
                        "playback_speed": summary.get("trajectory_playback_speed"),
                    }
    else:
        print(f"[report] iter={iteration} no train views — skipping trajectory MP4", flush=True)

    # --- Depth/RGB time series over train cameras ----------------------------
    # Streaming RGB-D cameras carry a sensor depth image. Export the same ordered
    # train-frame time series as rendered RGB, GT RGB, sensor-depth edges on GT,
    # rendered depth, sensor depth, and absolute depth error so RGB/depth sync and
    # geometry disagreements are inspectable without starting a viewer.
    if depth_cams:
        depth_dir = os.path.join(out_dir, "train_depth_timeseries")
        os.makedirs(depth_dir, exist_ok=True)
        manifest = {
            "frames": [],
            "panel_order": [
                "render_rgb",
                "gt_rgb",
                "sensor_depth_edges_on_gt",
                "rendered_depth",
                "sensor_depth",
                "abs_depth_error",
            ],
        }

        def _depth_frames():
            for i, cam in enumerate(depth_cams):
                try:
                    pkg = _shared_video_render_cache.get(id(cam))
                    if pkg is None:
                        with torch.no_grad():
                            pkg = render_fn(cam, gaussians, pipe, background, render_depth=True)
                    img = pkg["render"] if isinstance(pkg, dict) else pkg
                    rendered_depth = pkg.get("rendered_depth") if isinstance(pkg, dict) else None
                    if rendered_depth is None:
                        continue
                    gt = _gt_image(cam)
                    sensor_depth = _get_sensor_depth(cam, int(img.shape[1]), int(img.shape[2]))
                    if sensor_depth is None:
                        continue
                    panel = _depth_compare_frame(img.clamp(0.0, 1.0), gt, rendered_depth, sensor_depth)
                    name = str(getattr(cam, "image_name", f"{i:06d}"))
                    path = os.path.join(depth_dir, f"{i:06d}_{name}.png")
                    _save_png(path, panel)
                    frame = getattr(cam, "_streaming_frame", None)
                    timestamp = getattr(cam, "_streaming_timestamp", None)
                    rgb_depth_dt_ns = None if frame is None else getattr(frame, "rgb_depth_dt_ns", None)
                    manifest["frames"].append({
                        "index": int(i),
                        "image_name": name,
                        "timestamp": None if timestamp is None else float(timestamp),
                        "rgb_depth_dt_ns": None if rgb_depth_dt_ns is None else int(rgb_depth_dt_ns),
                        "path": path,
                    })
                    yield panel
                    del img, gt, rendered_depth, sensor_depth
                except Exception as e:
                    print(f"[report] depth time series failed for {getattr(cam, 'image_name', i)}: {e}", flush=True)
                if torch.cuda.is_available() and (i + 1) % 16 == 0:
                    torch.cuda.empty_cache()

        depth_mp4_path = os.path.join(out_dir, "train_depth_timeseries.mp4")
        # Unlike trajectory.mp4, depth_cams is never spatially reordered or
        # paced (no _pace_cams_for_trajectory call for this video), so
        # _infer_realtime_fps was never defeated by untimed interpolated
        # frames the way trajectory.mp4's was -- no fps fix needed here.
        video_fps = _infer_realtime_fps(depth_cams, mp4_fps)
        # This is a 6-panel diagnostic montage (render|gt|edges|depth x2|
        # error), not a presentation video -- upscaling the whole composite
        # to FHD width would shrink and blur each individual panel for no
        # benefit while adding real encode cost to every streaming report.
        # Keep the max-width safety cap (a pathologically large source
        # panel shouldn't balloon either), but leave the min-width upscale
        # off by default for this video specifically.
        err = _write_mp4(
            depth_mp4_path, _depth_frames(), fps=video_fps,
            min_width=0, max_width=report_video_max_width,
        )
        if err is None:
            artifacts.append(depth_mp4_path)
            summary["train_depth_timeseries"] = depth_mp4_path
            summary["n_train_depth_timeseries"] = len(manifest["frames"])
            print(
                f"[report] train_depth_timeseries.mp4 written "
                f"({len(manifest['frames'])} frames @ {video_fps}fps)",
                flush=True,
            )
        else:
            print(f"[report] train_depth_timeseries.mp4 SKIPPED: {err}", flush=True)
        manifest_path = os.path.join(depth_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        artifacts.append(manifest_path)

    _record_allocator_phase(
        "presentation_complete",
        presentation_status="completed" if presentation_enabled else "deferred",
    )
    _shared_video_render_cache.clear()
    _record_allocator_phase("presentation_cache_released")

    # --- Geometry metrics (Tier A model-space + Tier B/C render-space) ---------
    # Run-comparison descriptors (see utils/scene_geometry_metrics). Tier A is
    # deterministic from the model; Tier B/C reuse the SAME representative test
    # cameras as the visual sheets so two runs stay comparable.
    try:
        from eligsir.utils.scene_geometry_metrics import (
            shape_stats_from_model, lifecycle_stats_from_model,
            gaussian_belief_summary_from_model,
            angle_between_normal_maps,
            normal_agreement_stats, depth_accuracy_stats, aggregate_view_metrics,
        )
        try:
            from eligsir.utils.energy_mcmc import pop_pruning_stats, get_cumulative_pruning_stats
            _pruning = pop_pruning_stats()
            _pruning_cumulative = get_cumulative_pruning_stats()
        except Exception:
            _pruning = {}
            _pruning_cumulative = {}
        # issue #1365 (formerly #1372 Phase 7, superseded): purely observational MCMC candidate-survival
        # funnel telemetry. Empty/absent unless
        # --streaming_mcmc_candidate_funnel_telemetry was passed at launch.
        try:
            from eligsir.utils.mcmc_candidate_funnel import summarize_history as _funnel_summarize
            _funnel_summary = _funnel_summarize()
        except Exception:
            _funnel_summary = None
        geom = {
            "shape": shape_stats_from_model(gaussians),
            "lifecycle": lifecycle_stats_from_model(gaussians),
            # issue #1365 Phase 1: purely observational, changes no decision.
            "gaussian_belief": gaussian_belief_summary_from_model(gaussians),
            # "pruning" is windowed (since the previous periodic report), NOT a
            # whole-run total -- use "pruning_cumulative" for the true run-total.
            "pruning": _pruning,
            "pruning_cumulative": _pruning_cumulative,
            "mcmc_candidate_funnel": _funnel_summary,
        }
        if _funnel_summary is not None:
            try:
                import json as _json
                from eligsir.utils.mcmc_candidate_funnel import get_funnel_history as _funnel_get_history
                _funnel_path = os.path.join(model_path, "mcmc_candidate_funnel.json")
                with open(_funnel_path, "w") as _f:
                    _json.dump(
                        {"summary": _funnel_summary, "history": _funnel_get_history()},
                        _f, indent=2,
                    )
            except Exception:
                pass
        if test_cams:
            from eligsir.mapping.renderer.gsplat_backend import _normals_from_depth
            geo_cams = _representative_cams(test_cams, max_items=max(1, contact_sheet_cols * 2))
            tierB, tierC = [], []
            for cam in geo_cams:
                try:
                    _vcache = locals().get("_visual_cam_render_cache") or {}
                    cached = _vcache.get(id(cam))
                    if cached is not None:
                        out = cached
                    else:
                        with torch.no_grad():
                            out = render_fn(cam, gaussians, pipe, background, render_normals=True)
                    if not isinstance(out, dict):
                        continue
                    nr = out.get("rendered_normals_raw")
                    ns = out.get("rendered_surf_normals_raw")
                    valid = out.get("rendered_normals_valid")
                    rdepth = out.get("rendered_normals_depth")
                    if nr is None or ns is None or valid is None:
                        continue
                    # Tier B: splat-axis vs rendered-depth normal agreement, edge-masked.
                    ang = angle_between_normal_maps(nr, ns)
                    mask = valid.squeeze(0).clone()
                    if rdepth is not None:
                        edge = _depth_edge_mask(rdepth.squeeze(0).detach().cpu().numpy())
                        mask = mask & ~torch.from_numpy(edge).to(mask.device)
                    tierB.append(normal_agreement_stats(ang, mask))
                    # Tier C: sensor-anchored accuracy (streaming only).
                    sensor = _get_sensor_depth(cam, int(ang.shape[0]), int(ang.shape[1]))
                    if sensor is not None and rdepth is not None:
                        sensor = sensor.to(rdepth.device)
                        c = depth_accuracy_stats(rdepth, sensor, max_depth=depth_max_m)
                        K = out.get("rendered_normals_K")
                        if K is not None:
                            s_norm, s_valid = _normals_from_depth(sensor, K)
                            s_ang = angle_between_normal_maps(nr, s_norm)
                            s_edge = _depth_edge_mask(sensor.squeeze(0).detach().cpu().numpy())
                            s_mask = (valid.squeeze(0) & s_valid.squeeze(0)) & \
                                     ~torch.from_numpy(s_edge).to(valid.device)
                            sn = normal_agreement_stats(s_ang, s_mask)
                            c.update({f"sensor_normal_{k}": v for k, v in sn.items()})
                        tierC.append(c)
                except Exception as e:
                    print(f"[report] geometry view failed: {e}", flush=True)
            if tierB:
                geom["normal_agreement"] = aggregate_view_metrics(tierB)
            if tierC:
                geom["sensor_accuracy"] = aggregate_view_metrics(tierC)

        _visual_cam_render_cache.clear()

        # Scene expressiveness + per-run efficiency. SV (scene constant) comes from the
        # caller; efficiency ratios combine it with this run's live count + test PSNR.
        # gaussians_per_surfel is the size/growth-invariant headline; psnr_per_kgaussian
        # is a within-identical-scene diagnostic only (and only for a valid holdout eval).
        try:
            from eligsir.utils.scene_geometry_metrics import expressiveness_metrics
            _shape = geom.get("shape", {})
            _expr = dict(expressiveness or {})
            _eff = expressiveness_metrics(
                sv=_expr.get("sv"),
                n_live=int(_shape.get("n_live", 0) or 0),
                n_total=_shape.get("n_total"),
                opacity_sum=_shape.get("opacity_sum"),
                mean_test_psnr=summary.get("mean_test_psnr"),
                eval_valid=bool(summary.get("eval_valid", False)),
            )
            # SV scene block + efficiency, without clobbering n_live from _eff.
            _expr.update({k: v for k, v in _eff.items() if k != "sv"})
            geom["expressiveness"] = _expr
            _gps = _expr.get("gaussians_per_surfel")
            if _gps is not None:
                print(f"[report] expressiveness: SV={_expr.get('sv')} "
                      f"gauss/surfel={_gps:.2f} n_eff_frac={_shape.get('n_eff_frac')}",
                      flush=True)
        except Exception as e:
            print(f"[report] expressiveness metrics failed: {e}", flush=True)

        # Input quality (sensor-noise vs tracking-error split), supplied by the caller.
        if input_quality:
            geom["input_quality"] = dict(input_quality)
            print(f"[report] input_quality: verdict={input_quality.get('quality_verdict')} "
                  f"sensor_noise_m={input_quality.get('sensor_noise_m')} "
                  f"tracking_rmse_m={input_quality.get('tracking_inlier_rmse_m')} "
                  f"dim={input_quality.get('complexity_dim')}", flush=True)

        # Phase 2 depth-insertion sanity gate (issue #303) -- the
        # machine-readable form of the `live-insertion-noop-bug` memory's
        # "sanity-gate total_inserted>0" recommendation. A run whose depth
        # insertion was ENABLED but inserted nothing (e.g. every depth frame
        # silently dropped by a transport/decode failure) is MCMC-growth-only
        # and must not be read as a normal RGB-D run: it self-identifies here
        # instead of requiring log forensics after the fact.
        if depth_insertion:
            geom["depth_insertion"] = dict(depth_insertion)
            if depth_insertion.get("inert"):
                print(
                    "[report] *** DEPTH INSERTION INERT *** "
                    f"streaming_insert_from_depth was enabled and "
                    f"{depth_insertion.get('frames_admitted')} frames were admitted, but "
                    f"total_inserted={depth_insertion.get('total_inserted')} -- Phase 2 "
                    "contributed NOTHING to this map (MCMC-growth-only). Check the "
                    "depth transport/decode path (issue #303) before trusting this run.",
                    flush=True,
                )
            else:
                print(
                    f"[report] depth_insertion: enabled={depth_insertion.get('enabled')} "
                    f"total_inserted={depth_insertion.get('total_inserted')} "
                    f"frames_admitted={depth_insertion.get('frames_admitted')}",
                    flush=True,
                )

        # Exact SEED-cohort availability (issue #346 / #283). ALWAYS emitted
        # when supplied -- an unavailable cohort is the informative case, so
        # this is deliberately not gated on truthiness like the blocks around
        # it. See seed_cohort_report_block's docstring.
        # Issue #1262 (#6.2.1): per-Gaussian opacity survival by exposure group.
        # Emitted ONLY when the resolution-exposure feature is on -- a flag-off
        # report is byte-identical to before this existed.
        if exposure_opacity_survival is not None:
            geom["exposure_opacity_survival"] = dict(exposure_opacity_survival)
        if seed_cohort is not None:
            geom["seed_cohort"] = dict(seed_cohort)
            if seed_cohort.get("available"):
                print(
                    f"[report] seed_cohort: available n_seed_born="
                    f"{seed_cohort.get('n_seed_born')} n_seed_alive="
                    f"{seed_cohort.get('n_seed_alive')} survival="
                    f"{seed_cohort.get('seed_survival_fraction')}",
                    flush=True,
                )
            else:
                print(
                    "[report] seed_cohort: UNAVAILABLE -- this run is NOT "
                    "seed-fade-cohort analyzable (issue #346). Reason: "
                    f"{seed_cohort.get('reason')}",
                    flush=True,
                )

        # Mid-run training_setup(preserve_lineage=True) events (issue #374
        # item 3) -- see write_post_training_report's docstring.
        if lineage_preserving_setup:
            geom["lineage_preserving_setup"] = dict(lineage_preserving_setup)
            print(
                f"[report] lineage_preserving_setup: count="
                f"{lineage_preserving_setup.get('count')} "
                f"iterations={lineage_preserving_setup.get('iterations')} "
                f"sites={lineage_preserving_setup.get('sites')}",
                flush=True,
            )

        # Four-clocks telemetry snapshot (Wave 2 Task 8) -- see this
        # function's docstring for the ``mapping_clocks`` param and
        # FourClocksSummary's docstring (coordinator.py) for field meaning.
        if mapping_clocks:
            geom["mapping_clocks"] = dict(mapping_clocks)
            print(f"[report] mapping_clocks: source={mapping_clocks.get('source_frame_rate_hz'):.2f}Hz "
                  f"admitted={mapping_clocks.get('admitted_keyframe_rate_hz'):.2f}Hz "
                  f"commit={mapping_clocks.get('map_commit_rate_hz'):.2f}Hz "
                  f"map_age_ms={mapping_clocks.get('map_age_ms')} "
                  f"backlog={mapping_clocks.get('input_backlog_frames')} "
                  f"realtime_factor={mapping_clocks.get('realtime_factor')}", flush=True)

        # Scene-vs-trajectory extent (plume quantification) -- see
        # utils/scene_geometry_metrics.compare_scene_vs_trajectory for why this
        # stays eval-only (not a live per-iteration hook) for now.
        try:
            from eligsir.utils.scene_geometry_metrics import compare_scene_vs_trajectory
            traj_positions = [c.camera_center for c in train_cams if hasattr(c, "camera_center")]
            if traj_positions:
                geom["scene_vs_trajectory"] = compare_scene_vs_trajectory(
                    gaussians.get_xyz, gaussians.get_opacity, torch.stack(traj_positions))
                frac = geom["scene_vs_trajectory"].get("frac_mass_outside_trajectory_bbox")
                if frac is not None:
                    print(f"[report] scene_vs_trajectory: frac_mass_outside_bbox={frac:.4f} "
                          f"bbox_ratio={geom['scene_vs_trajectory'].get('bbox_volume_ratio')}", flush=True)
        except Exception as e:
            print(f"[report] scene_vs_trajectory metrics failed: {e}", flush=True)

        # Tier D — opacity-weighted off-surface/floater mass vs the model's own
        # accumulated (Gaussian-independent) sensor point cloud. Default-on
        # whenever a streaming_scene with accumulated input-quality voxels is
        # passed in -- no --reference_map needed (see
        # utils/scene_geometry_metrics.py's Tier D docstring for why this is a
        # real, independent-of-the-Gaussians reference, not circular).
        if streaming_scene is not None:
            try:
                from eligsir.utils.scene_geometry_metrics import compute_offsurface_metrics
                sensor_pts_np = streaming_scene.get_accumulated_sensor_points()
                if sensor_pts_np.shape[0] > 0:
                    sensor_pts = torch.from_numpy(sensor_pts_np).to(gaussians.get_xyz.device)
                    geom["offsurface"] = compute_offsurface_metrics(
                        gaussians.get_xyz, gaussians.get_opacity, sensor_pts,
                        tau=offsurface_tau_m,
                    )
                    off = geom["offsurface"]
                    if off.get("valid"):
                        print(f"[report] offsurface (Tier D): tau={off['tau_m']}m "
                              f"mass_frac={off['offsurface_mass_frac']:.4f} "
                              f"count_frac={off['offsurface_count_frac']:.4f} "
                              f"nn_p50={off['nn_dist_p50_m']:.4f}m "
                              f"n_sensor_points={off['n_sensor_points']}", flush=True)
                else:
                    geom["offsurface"] = {"valid": False, "reason": "no accumulated sensor points"}
            except Exception as e:
                print(f"[report] offsurface (Tier D) metrics failed: {e}", flush=True)

        # Gaussian vs. independent point-cloud-only reference map (e.g. an embed_slam
        # TSDF reconstruction from the same pseudo-GT trajectory) -- strictly opt-in,
        # only runs when a reference map was supplied. See CLAUDE.md's "Gaussian vs.
        # point-cloud-only map comparison" section for the cross-repo workflow that
        # produces reference_map_path.
        if reference_map_path:
            try:
                from eligsir.utils.map_comparison import (
                    compare_gaussian_vs_reference_map, load_reference_pointcloud, render_map_overlay,
                )
                reference_pcd = load_reference_pointcloud(reference_map_path)
                geom["map_comparison"] = compare_gaussian_vs_reference_map(
                    gaussians, reference_pcd, align=map_comparison_align)
                mc = geom["map_comparison"]
                if mc.get("valid"):
                    print(f"[report] map_comparison: symmetric_chamfer_m={mc.get('symmetric_chamfer_m'):.4f} "
                          f"gaussian->reference_offsurface_mass="
                          f"{mc['gaussian_to_reference'].get('offsurface_mass_frac'):.4f} "
                          f"reference->gaussian_offsurface_mass="
                          f"{mc['reference_to_gaussian'].get('offsurface_mass_frac'):.4f} "
                          f"alignment_fitness={mc['alignment'].get('fitness')}", flush=True)
                    # Default-on overlay render alongside the JSON metrics -- no
                    # separate --viz flag, matches this report's convention of
                    # always writing its visual artifacts (contact sheets, mp4s).
                    overlay_path = os.path.join(out_dir, "map_comparison.svg")
                    gaussian_xyz_np = gaussians.get_xyz.detach().float().cpu().numpy()
                    reference_xyz_np = np.asarray(reference_pcd.points)
                    render_map_overlay(overlay_path, gaussian_xyz_np, reference_xyz_np)
                    artifacts.append(overlay_path)
                    print(f"[report] map_comparison: wrote {overlay_path}", flush=True)
            except Exception as e:
                print(f"[report] map_comparison failed: {e}", flush=True)

        summary["geometry"] = geom
        geom_path = os.path.join(out_dir, "geometry_metrics.json")
        with open(geom_path, "w") as f:
            json.dump(geom, f, indent=2)
        artifacts.append(geom_path)
        _nf = geom.get("shape", {}).get("needle_frac_lt_0.3")
        if _nf is not None:
            print(f"[report] geometry: needle_frac(s2/s1<0.3)={_nf:.3f} "
                  f"n_live={geom['shape'].get('n_live')}", flush=True)
    except Exception as e:
        print(f"[report] geometry metrics failed: {e}", flush=True)

    if baseline_run:
        try:
            bc = compare_against_baseline(baseline_run, summary, model_path)
            summary["baseline_comparison"] = bc
            if bc.get("comparable"):
                print(
                    f"[report] baseline_comparison vs {baseline_run}: "
                    f"final_delta={bc.get('final_psnr_delta')} "
                    f"curve_shape={bc.get('curve_shape')} "
                    f"n_aligned={bc.get('n_aligned_checkpoints')} "
                    f"holdout_verified={bc.get('holdout_verified')} "
                    f"verification_level={bc.get('verification_level')}",
                    flush=True,
                )
            else:
                print(
                    f"[report] baseline_comparison NOT comparable "
                    f"(verification_level={bc.get('verification_level')}): {bc.get('reason')}",
                    flush=True,
                )
        except Exception as e:
            print(f"[report] baseline_comparison failed: {e}", flush=True)
            summary["baseline_comparison"] = {"comparable": False, "reason": f"unexpected error: {e!r}"}

    # --- GPU co-residency (issue #295 fix item 4) ------------------------------
    # Self-identifying record of whether another process shared this run's GPU
    # at start/end -- see utils/gpu_coresidency.py. Best-effort, never blocks
    # or fails report generation.
    try:
        from eligsir.utils.gpu_coresidency import build_report_block as _gpu_coresidency_block
        summary["gpu_coresidency"] = _gpu_coresidency_block(model_path)
        if summary["gpu_coresidency"].get("contaminated"):
            print(
                "[report] WARNING: gpu_coresidency indicates another process "
                "was resident on the GPU at both run start and end -- "
                "throughput/timing claims from this run may be contaminated. "
                "See report.json['gpu_coresidency'].",
                flush=True,
            )
    except Exception as e:
        print(f"[report] gpu_coresidency snapshot failed: {e}", flush=True)
        summary["gpu_coresidency"] = {"start": None, "end": None, "contaminated": None,
                                       "error": repr(e)}
    if git_provenance is not None:
        summary["git_provenance"] = git_provenance
    if determinism is not None:
        summary["determinism"] = determinism

    # --- source attestation (issue #1302) ---------------------------------
    # A passive, operator-declared record of whether this run's RGB-D source
    # was ever attested through eligsir-rgbd-sync's capture-provenance
    # gate (see utils/source_attestation.py and
    # docs/architecture/2026-08-30-issue1302-legacy-attestation-design.md).
    # Absent entirely == this report predates the field, same convention as
    # git_provenance/execution_semantics above. Status "unknown" (the
    # default once the flag exists but wasn't set to something else) is
    # written explicitly rather than omitted, so a reader can tell "declared
    # unknown" apart from "field doesn't exist yet".
    if source_attestation is not None:
        summary["source_attestation"] = source_attestation
        if source_attestation.get("status") != "validated":
            print(
                "[report] WARNING: source_attestation.status="
                f"{source_attestation.get('status')!r} -- this run's RGB-D "
                "source has not been attested through the capture-provenance "
                "gate. Never quote this run's numbers as canonical/MANAGED "
                "evidence; see report.json['source_attestation'] and "
                "eligsir#1302.",
                flush=True,
            )

    # --- report provenance (issue #630) ----------------------------------
    # Whether this report was produced by the live training loop or regenerated
    # post-hoc from a frozen checkpoint. Always written unconditionally so a
    # reader can distinguish "predates the field" (absent) from "present but
    # unknown" (kind: "unknown") -- the distinction is load-bearing. See
    # eligsir.io.report_provenance.infer_report_provenance for signal
    # documentation and fail-closed strategy.
    _prov = infer_report_provenance(
        log_prefix=log_prefix,
        train_cams=train_cams,
        test_cams=test_cams,
        streaming_scene=streaming_scene,
        tb_writer=tb_writer,
        gaussians=gaussians,
    )
    summary["report_provenance"] = _prov

    # --- experiment origin (issue #511, lane P0-A of #202) -------------------
    # WHY this run exists / which tracked task owns it. Resolved ONCE at
    # train.py startup and also written to <model_path>/experiment_origin.json
    # from that same object -- the two artifacts cannot diverge because there
    # is only one resolution event (see utils/experiment_origin.py).
    # Omitted entirely when not passed, so a caller that predates this kwarg
    # writes the exact same report.json as before.
    if experiment_origin is not None:
        summary["experiment_origin"] = copy.deepcopy(experiment_origin)
    # --- resolved execution semantics (issue #168, lane P0-B of #202) --------
    # The machine-readable record of the resolved value of every represented
    # run-mode axis, so a run self-identifies its semantics instead of leaving
    # a reader to re-derive them from raw flags (which is how a profile named
    # offline/* came to run the streaming trainer). Deliberately the same
    # mechanism as git_provenance / experiment_validity above: a record, never
    # a gate. Written whenever run_args is available; absent otherwise, which
    # means "this report predates the contract", not "the axes were default".
    if execution_semantics is not None:
        summary["execution_semantics"] = execution_semantics
        for _w in (execution_semantics.get("warnings") or []):
            print(f"[report] execution_semantics: {_w}", flush=True)

    # --- admission/cadence invariants + queue telemetry (issue #602) ---------
    # Absent key == "this report predates the contract", never "nothing was
    # measured" -- the same absent != zero rule the block itself applies to
    # its own fields. Violations are RECORDED and printed, never raised:
    # report generation runs after hours of GPU work and must not be aborted
    # to complain about telemetry (utils/admission_cadence.raise_on_errors is
    # the opt-in strict path, used by tests).
    if admission_cadence is not None:
        summary["admission_cadence"] = admission_cadence
        try:
            from eligsir.utils.admission_cadence import format_violations
            _inv = admission_cadence.get("invariants") or {}
            for _line in format_violations(_inv.get("violations") or []):
                print(_line, flush=True)
            if _inv.get("violated"):
                print(
                    "[admission_cadence] one or more HARD invariants were "
                    "violated -- this run's admission/cadence accounting is "
                    "self-inconsistent; see report.json['admission_cadence']",
                    flush=True)
            # Compact, directly diffable side-artifact for #289/#290/#563
            # analyses, mirroring how geometry_metrics.json duplicates
            # summary["geometry"].
            _ac_path = os.path.join(out_dir, "admission_cadence_summary.json")
            with open(_ac_path, "w") as _f:
                json.dump(admission_cadence, _f, indent=2)
            artifacts.append(_ac_path)
        except Exception as _exc:  # never let telemetry break the report
            print(f"[admission_cadence] WARNING: side-artifact/print step "
                  f"failed: {_exc!r}", flush=True)

    # --- compute backend / compile status (issue #597) ------------------------
    # Requested vs RESOLVED torch.compile status, plus the fused/custom kernel
    # choices that also govern throughput -- so a run manifest can tell an
    # intentional `--compile_mode=off` apart from an unsupported or
    # failed-and-fell-back compile, and so `compile_mode=off` is never misread
    # as "all optimized kernels disabled". Deliberately additive and
    # self-sourcing: it reads process state that `configure_torch_compile()`
    # recorded from the SAME args namespace both trainers already pass it, so
    # neither trainer needed a new kwarg. A process that never configured
    # compilation (a bare render/report tool) records
    # `compile_status="not_configured"` rather than a misleading "off".
    try:
        from eligsir.utils.compiled_kernels import (
            build_compute_backend_block as _build_compute_backend_block,
            compute_backend_differences as _compute_backend_differences,
        )

        summary["compute_backend"] = _build_compute_backend_block()
        if summary["compute_backend"].get("compile_requested_but_not_active"):
            print(
                "[report] WARNING: compute_backend.compile_status="
                f"{summary['compute_backend'].get('compile_status')!r} -- torch.compile "
                "was REQUESTED but is not active for this run. Do not compare its "
                "throughput against a compiled run.",
                flush=True,
            )
        _bc = summary.get("baseline_comparison")
        if isinstance(_bc, dict) and baseline_run:
            # Re-read the baseline report here rather than plumbing it out of
            # compare_against_baseline: that module is on the torch-free
            # CPU-smoke list and must not learn about compiled_kernels.
            from eligsir.utils.baseline_comparison import _find_latest_report_json

            _base_backend = None
            _base_path = _find_latest_report_json(baseline_run)
            if _base_path:
                with open(_base_path) as _bf:
                    _base_backend = json.load(_bf).get("compute_backend")
            _bc["compute_backend"] = _compute_backend_differences(
                summary["compute_backend"], _base_backend
            )
            _cb_diffs = (_bc["compute_backend"].get("differences") or {})
            if _cb_diffs:
                print(
                    "[report] WARNING: compute-backend mismatch vs baseline "
                    f"({', '.join(sorted(_cb_diffs))}) -- throughput/timing "
                    "comparisons between these two runs are not like-for-like.",
                    flush=True,
                )
    except Exception as e:
        print(f"[report] compute_backend block failed: {e}", flush=True)
        summary["compute_backend"] = {
            "schema_version": 1, "configured": False,
            "compile_status": "not_configured", "error": repr(e),
        }

    # --- run intent (issue #524 acceptance criterion 3) -----------------------
    # Declared run intent + resolved profile stack + effective config.
    # Deliberately the same mechanism as git_provenance / experiment_origin /
    # execution_semantics above: a record, never a gate. Written whenever the
    # caller passes it (both trainer paths do, unconditionally, as of #524's
    # deferred half); absent otherwise, meaning "this report predates the
    # contract", not "no intent was declared" (that case is recorded INSIDE
    # the block as declared == "UNDECLARED").
    if run_intent is not None:
        summary["run_intent"] = run_intent

    # Issue #951: weak-region control and scored evaluation are separate data
    # roles. Keep split validity (`eval_valid`) independent, but make an
    # explicit scored-holdout oracle intervention impossible to mistake for
    # an unbiased test verdict.
    if partition_control is not None:
        summary["partition_control"] = copy.deepcopy(partition_control)
        if bool(partition_control.get("controller_actuation_used")):
            _control_eval_status = str(
                partition_control.get("scientific_eval_status") or "UNKNOWN"
            )
            summary["scientific_eval_status"] = _control_eval_status
            summary["unbiased_test_verdict_permitted"] = (
                _control_eval_status != "ORACLE_INTERVENTION"
                and int(partition_control.get("control_scored_overlap_n") or 0) == 0
            )
            if not summary["unbiased_test_verdict_permitted"]:
                print(
                    "[report] ORACLE_INTERVENTION: partition-focus control used "
                    "scored holdout observations; unbiased test verdicts are not permitted.",
                    flush=True,
                )

    # --- immutable experiment declaration (issue #705) ----------------------
    # Optional keeps standalone/legacy report callers behavior-neutral. Both
    # real trainer entrypoints pass the startup-validated context, including
    # NON_GRADEABLE_DIRECT for an intentionally unmanaged run.
    if experiment_context is not None:
        summary["experiment_declaration"] = copy.deepcopy(experiment_context)

    # --- experiment validity class (issue #203, lane P0-E of #202) -----------
    # Declared-vs-achieved, mirroring holdout_hash vs realized_holdout_hash vs
    # scored_holdout_hash above. Pure reporting: it gates nothing and is
    # deliberately a SIBLING field, never folded into eval_valid / verdict /
    # mapping_result_valid (same reasoning as map_reinit_invalidated).
    try:
        from eligsir.utils.experiment_validity import build_experiment_validity_block

        # A canonical causal caller owns an authoritative terminal retained
        # catalog (COMMIT+DEFER).  Its bounded resident train-camera set is an
        # execution cache, not the trajectory-validity population.  Legacy
        # streaming callers without that evidence retain the conservative
        # train-camera derivation.
        if experiment_validity_source_indices is not None:
            _src_idx = list(experiment_validity_source_indices)
            _src_idx_provenance = (
                experiment_validity_source_index_provenance
                or "explicit_retained_catalog_from_report_caller"
            )
        else:
            # Holdouts are reserved every Nth frame across the whole trajectory
            # BEFORE admission, so mixing test cameras in would make a pure
            # prefix run look distributed.
            _src_idx = [camera_source_index(c) for c in (train_cams or [])]
            _src_idx = [i for i in _src_idx if i is not None]
            _src_idx_provenance = (
                "train_cameras_resident_at_report_time (utils.partition_eval."
                "camera_source_index over train_cams; NOTE: partition-focus "
                "demotion can evict an admitted camera from this buffer, so "
                "this can undercount vs realized_admission_count)"
            )
        _ev_last_src = None
        _ev_release = None
        if streaming_scene is not None:
            _ev_last_src = getattr(streaming_scene, "_last_admitted_source_idx", None)
            _ev_release = getattr(streaming_scene, "_n_train_frames_ingested", None)

        # Issue #203/#289: wire the realized-cadence tracker's measured
        # supervision_view clock (utils/realized_cadence.py) into the
        # previously-always-null experiment_validity.supervision_views field.
        # Streaming-only by construction -- realized_cadence is None on the
        # offline train.py path, which has no such clock, and the field stays
        # null-with-reason exactly as before this change in that case.
        #
        # A resumed run's tracker is in-memory only (see that module's
        # "Honesty about partial coverage" docstring) and restarts its counter
        # from 0 at the resume point, describing only the post-resume segment
        # -- the same class of understatement issue #290's
        # seed_fade_indeterminate exists to flag for the seed-fade monitor.
        # Rather than write a number that silently undercounts the whole run,
        # withhold it (null-with-reason, carrying the partial count in the
        # reason text) whenever EITHER the tracker itself reports partial
        # coverage OR this report's own resumed_from_iteration is set -- an OR
        # of both signals is the conservative direction if a future caller's
        # tracker and this function's resume bookkeeping ever disagree.
        _rc = realized_cadence or {}
        _rc_sv = (_rc.get("clocks") or {}).get("supervision_view")
        _rc_partial = bool((_rc.get("coverage") or {}).get("partial_coverage"))
        _sv_resumed = _rc_partial or (resumed_from_iteration is not None)
        _ev_supervision_views: Optional[int] = None
        _ev_supervision_views_reason: Optional[str] = None
        if _rc_sv is not None and not _sv_resumed:
            _ev_supervision_views = int(_rc_sv)
        elif _rc_sv is not None and _sv_resumed:
            _ev_supervision_views_reason = (
                f"resumed run: the realized-cadence tracker is in-memory only, "
                f"so its observed count ({int(_rc_sv)} supervision views) "
                "describes only the post-resume segment, not the whole run "
                "(#290's resume-gap class) -- withheld rather than reported "
                "as if it were the full-run total"
            )
        summary["experiment_validity"] = build_experiment_validity_block(
            declared=experiment_validity_class,
            source_indices=_src_idx,
            source_frame_count_total=_registry_n_source_frames(
                summary.get("scene_source")),
            # Second half of the derivation gate. `scene_source` comes from
            # the SCENE_SOURCE env var, which launch scripts/ablator set for
            # streaming and non-streaming jobs alike -- so a registry hit
            # alone does NOT prove this run had causal source-frame
            # ordering. A legacy train.py run passes no streaming_scene and
            # correctly grades UNKNOWN.
            is_streaming_run=streaming_scene is not None,
            source_index_provenance=_src_idx_provenance,
            realized_release_count=(
                int(_ev_release) if _ev_release is not None else None),
            realized_admission_count=frames_admitted_override,
            source_insert_count=(
                (depth_insertion or {}).get("total_inserted")
                if depth_insertion is not None else None),
            optimizer_updates=(
                true_final_iteration if true_final_iteration is not None
                else iteration),
            supervision_views=_ev_supervision_views,
            supervision_views_unavailable_reason=_ev_supervision_views_reason,
            mapping_end_iter=last_ingest_iter,
            final_iteration=(
                true_final_iteration if true_final_iteration is not None
                else iteration),
            last_source_frame_reached=_ev_last_src,
            is_final_report=experiment_validity_is_final,
            # Issue #1045: thread the real-holdout-validity signals through
            # so an unregistered-scene run that also failed the holdout
            # contract gets a loud, machine-readable warning instead of
            # silently completing with no coverage-validity signal at all.
            eval_valid=summary.get("eval_valid"),
            n_test=summary.get("n_test"),
        )
        for _w in summary["experiment_validity"].get("warnings", []):
            print(f"[report] experiment_validity: {_w}", flush=True)
    except Exception as e:
        print(f"[report] experiment_validity block failed: {e}", flush=True)
        summary["experiment_validity"] = {
            "declared": None, "achieved": "UNKNOWN", "effective": "UNKNOWN",
            "negative_conclusions_permitted": False,
            "negative_result_status_ceiling": "needs_better_proxy",
            "error": repr(e),
        }

    # --- report.json ----------------------------------------------------------
    # Issue #698: the loop-time PsnrTickResult path deliberately excludes the
    # final iteration because this report already owns the terminal metrics
    # pass.  Persist that exact per-view pass as the terminal canonical tick,
    # then make report.json name the content-addressed row.  This avoids a
    # second render and gives #358's local-quality sidecar one timeline to
    # join.  A final report with no usable/consistent binding is explicitly
    # ungradeable: do not let its standalone mean_test_psnr look promotable.
    if experiment_validity_is_final is True:
        _final_tick_iteration = (
            int(true_final_iteration)
            if true_final_iteration is not None else int(iteration)
        )
        if not summary.get("eval_valid") or not psnrs or not test_cams_used:
            summary["evaluation_tick_binding"] = {
                "status": "unavailable",
                "gradeable": False,
                "reason": "final_report_has_no_valid_scored_holdout_metrics",
                "evaluation_tick_iteration": _final_tick_iteration,
                "report_iteration": int(iteration),
            }
        else:
            _t_record_final_report_eval_tick = time.perf_counter()
            try:
                from eligsir.utils.eval_tick import record_final_report_eval_tick

                summary["evaluation_tick_binding"] = _record_report_phase(
                    "record_final_report_eval_tick",
                    lambda: record_final_report_eval_tick(
                        model_path=model_path,
                        iteration=_final_tick_iteration,
                        report_iteration=int(iteration),
                        view_psnr_db=psnrs,
                        scored_cams=test_cams_used,
                        test_cams=test_cams,
                        scene_source=summary.get("scene_source"),
                        reported_mean_test_psnr_db=summary.get("mean_test_psnr"),
                        clock_snapshot=evaluation_tick_clock_snapshot,
                    ),
                )
                summary["evaluation_tick_binding"]["gradeable"] = True
                print(
                    "[report] record_final_report_eval_tick took "
                    f"{time.perf_counter() - _t_record_final_report_eval_tick:.1f}s",
                    flush=True,
                )
            except Exception as e:
                summary["evaluation_tick_binding"] = {
                    "status": "mismatch" if isinstance(e, ValueError) else "unavailable",
                    "gradeable": False,
                    "reason": str(e),
                    "evaluation_tick_iteration": _final_tick_iteration,
                    "report_iteration": int(iteration),
                }
                summary["eval_valid"] = False
                print(
                    "[report] ERROR: final evaluation-tick binding failed; "
                    "this report is not gradeable: " + str(e),
                    flush=True,
                )
                print(
                    "[report] record_final_report_eval_tick took "
                    f"{time.perf_counter() - _t_record_final_report_eval_tick:.1f}s "
                    "(failed)",
                    flush=True,
                )

        _t_finalize_eval_tick_history = time.perf_counter()
        try:
            from eligsir.utils.eval_tick import finalize_eval_tick_history

            summary["evaluation_tick_telemetry"] = _record_report_phase(
                "finalize_eval_tick_history",
                lambda: finalize_eval_tick_history(model_path),
            )
            print(
                "[report] finalize_eval_tick_history took "
                f"{time.perf_counter() - _t_finalize_eval_tick_history:.1f}s",
                flush=True,
            )
        except Exception as e:
            summary["evaluation_tick_telemetry"] = {
                "status": "unavailable",
                "reason": str(e),
            }
            print(
                "[report] finalize_eval_tick_history took "
                f"{time.perf_counter() - _t_finalize_eval_tick_history:.1f}s "
                "(failed)",
                flush=True,
            )

    trajectory_path = os.path.join(model_path, "trajectory_full.jsonl")
    if summary.get("test_per_view") and os.path.isfile(trajectory_path):
        _t_write_trajectory_quality_profile = time.perf_counter()
        try:
            from eligsir.utils.trajectory_quality import write_trajectory_quality_profile

            trajectory_plot = os.path.join(
                out_dir, "trajectory_psnr_by_distance.svg"
            )
            summary["trajectory_quality_profile"] = _record_report_phase(
                "write_trajectory_quality_profile",
                lambda: write_trajectory_quality_profile(
                    trajectory_path=trajectory_path,
                    per_view=summary["test_per_view"],
                    output_path=trajectory_plot,
                ),
            )
            artifacts.append(trajectory_plot)
            print(
                "[report] write_trajectory_quality_profile took "
                f"{time.perf_counter() - _t_write_trajectory_quality_profile:.1f}s",
                flush=True,
            )
        except Exception as exc:
            summary["trajectory_quality_profile"] = {
                "status": "unavailable",
                "reason": str(exc),
            }
            print(
                "[report] write_trajectory_quality_profile took "
                f"{time.perf_counter() - _t_write_trajectory_quality_profile:.1f}s "
                "(failed)",
                flush=True,
            )
            print(
                "[report] ERROR: trajectory-distance quality plot failed: "
                + str(exc),
                flush=True,
            )

    # Issue #1259/#1284: surface any NaN/Inf-cast sanitization that happened
    # anywhere during this report's rendering, the same way #437's
    # n_test_metrics_failed makes a partial-render report un-mistakable for a
    # clean one. Top-level and only present when non-empty (mirrors that
    # pattern's "*_metrics_render_failures block only when >0" convention).
    _nan_sanitize_report = get_nan_sanitize_report()
    if _nan_sanitize_report is not None:
        summary["nan_sanitized_casts"] = _nan_sanitize_report
        print(
            f"[report] WARNING: {_nan_sanitize_report['n_sanitized']} render/GT "
            f"array(s) needed NaN/Inf sanitization AFTER the torch-side guard "
            f"already ran before their uint8 cast -- this report's images for "
            f"those views were corrupted-then-repaired, not clean. See "
            f"summary['nan_sanitized_casts'] and issue #1259/#1284.",
            flush=True,
        )

    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    artifacts.append(report_path)

    return summary
