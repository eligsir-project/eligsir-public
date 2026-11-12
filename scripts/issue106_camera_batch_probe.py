#!/usr/bin/env python3
"""Feasibility probe: does gsplat's TRUE camera-batch dimension (C > 1) work on
this ROCm build, and is it cheaper than C sequential renders?

GitHub issue #106 ("ICRA: Implement and benchmark batched multi-keyframe
rendering/loss optimization"), ledger item 2 of issue #614.

WHY THIS EXISTS
---------------
EliGSiR's current multi-view batching ("B" arms) batches OPTIMIZATION
semantics -- it accumulates a loss over several keyframes before stepping --
but it still renders those views SEQUENTIALLY, one ``rasterization()`` call per
view. The open question for #106 is whether the rasterizer can take a real
camera dimension instead, and whether that is actually faster / cheaper.

WHAT IS ALREADY KNOWN (do not re-derive from this probe)
--------------------------------------------------------
C > 1 is EXPRESSIBLE and is already used in this repo, forward-only:

  * ``eligsir/mapping/renderer/gsplat_backend.py::render_batch`` (line 493)
    stacks ``viewmats`` to ``(C, 4, 4)``, ``Ks`` to ``(C, 3, 3)`` and
    per-camera ``colors`` to ``(C, N, 3)``, chunked at ``chunk_size=32``. It is
    used for report/contact-sheet rendering.
  * gsplat 1.5.3's own ``rasterization`` docstring types the argument as
    ``viewmats: [..., C, 4, 4]`` / ``Ks: [..., C, 3, 3]`` /
    ``colors: [..., (C,) N, D]`` / ``backgrounds: [..., C, D]``.

So the probe is NOT asking "is C>1 in the signature". It is asking the three
things ``render_batch`` never exercises:

  1. **BACKWARD at C>1**, which ``render_batch`` never runs (it is
     inference-only). This is the load-bearing unknown for #106, because
     batching the loss requires gradients through the batched call.
  2. **BACKWARD at C>1 with ``sparse_grad=True``**, which is what the real
     training loop uses (``render()`` line 323 passes
     ``pipe.gsplat_sparse_grad``, forced ``True`` by the default
     ``--parallelism_profile safe``; see CLAUDE.md issue #228).
     ``render_batch`` hardcodes ``sparse_grad=False``, so a probe that only
     tests ``False`` answers a question the training loop does not ask.
  3. Whether batched is actually **faster / lower peak-memory** than the two
     realistic sequential shapes (see "Memory is asymmetric" below).

A BACKWARD-PARITY FAILURE IS NOT SELF-INTERPRETING (issue #917)
---------------------------------------------------------------
``backward.all_within_tolerance`` compares the batched gradients against the
sequential ones ONCE. That difference only means something if the same path
compared against ITSELF agrees -- a ``sparse_grad=False`` backward accumulates
per-Gaussian gradients with ``atomicAdd``, whose float32 summation order is not
guaranteed to repeat between two launches of the same kernel, so a nonzero
batched-vs-sequential difference can be pure accumulation-order noise. (This has
already bitten this probe once: see ``compare_tensors``' "HISTORY OF THE OR
RULE", where an absolute-only criterion flagged 31/32 cells including C=1.)

Every config therefore also repeats each path and records
``backward_self_consistency`` -- ``batched`` (batched vs. batched), ``sequential``
(sequential vs. sequential), and ``noise_floor_comparison`` holding each
parameter's cross-path ``max_abs_diff`` against the larger of the two same-path
ones. **Read ``backward_noise_floor_exceeded``, not
``backward.all_within_tolerance``, to decide whether a parity failure is a real
gradient defect.** ``None`` there means undetermined, never "clean".

A REAL GRADIENT DEFECT IS USUALLY LOCALIZED TO SOME PARAMETERS (issue #920)
--------------------------------------------------------------------------
``backward_noise_floor_exceeded`` is per-cell ``any()`` over a per-parameter
conjunction (real parity failure AND above that parameter's own noise floor) --
see ``noise_floor_comparison``. It shipped as ``all()`` over the ratio alone,
which reported 2 of 18 cells on a dataset whose per-parameter data said 18 of 18,
because the two parameters with no disagreement at all dragged the conjunction
down. When reading a summary, ``backward_unexplained_disagreement_params`` names
WHICH parameters carry it -- a defect confined to a subset of parameters is
evidence about where it lives, not a reason to discount it.

KWARGS ARE PINNED TO ``render()``'S REAL VALUES
-----------------------------------------------
In particular ``tile_size=16``, NOT gsplat's own default of 8 --
``gsplat_backend.py`` line 301-303: "tile_size=8 causes NaN gradients on
ROCm/gfx1151 with wave32-patched gsplat". Leaving the gsplat default would make
the backward check report a failure that has nothing to do with C. Likewise
``packed=True``, ``eps2d=0.3``, and the explicitly recorded
``rasterize_mode={classic,antialiased}``,
``near_plane=0.01``, ``far_plane=1e10``, ``radius_clip=0.0``, ``render_mode="RGB"``.

``backgrounds`` is passed as ``None`` on BOTH paths on purpose: the docstring
types it ``[C, D]`` while ``render_batch`` line 522 passes a flat ``[3]``. That
discrepancy is real but is a background-shape question, not a camera-batch
question, and conflating them would make this probe's result unreadable. (No
production code is changed by this file.)

COLOR PATHS
-----------
Two are measured because the batching win differs materially between them:

  * ``sh``  -- shared SH coefficients ``(N, K, 3)`` + ``sh_degree``, so the
    KERNEL does per-camera view-dependent SH. This is the path where batching
    can actually win, and is what ``--sh_impl gsplat`` selects in production.
  * ``precomputed`` -- per-camera post-activation colors ``(C, N, 3)``, i.e.
    the repo-default ``sh_impl="python"`` path
    (``_python_sh_colors``), where C SH evaluations happen OUTSIDE the
    rasterization call and eat much of the batching win.

MEMORY IS ASYMMETRIC BY CONSTRUCTION
------------------------------------
A batched call holds C images live; one sequential render holds one. Reporting
only those two makes "batched uses more memory" a tautology. Three peaks are
recorded per configuration, and the third is the one that matters:

  * ``batched``              -- one ``rasterization`` call over C cameras.
  * ``sequential_streamed``  -- render view i, backward view i, free it, repeat.
                                Memory-light; C separate backward launches.
  * ``sequential_retained``  -- render all C keeping graphs, then ONE backward
                                over the summed loss. This is the shape a
                                multi-keyframe batched LOSS takes today, and is
                                the honest sequential comparand for #106.

LOSS SCALING
------------
Both paths use a plain ``sum()`` reduction over all rendered pixels of all C
views. A ``mean`` over ``C*H*W`` on one side and a per-view ``mean`` on the
other differs by exactly a factor of C, which reads as a kernel bug and is not
one.

TOLERANCES ARE ELEMENTWISE COMBINED
------------------------------------
Correctness verdicts pass the elementwise criterion ``|a_i - b_i| <= atol +
rtol * |b_i|`` for every index i. An absolute-only bound is meaningless on
gradients spanning 1e-7 to 1e4 and produced 31/32 spurious backward "failures"
on the first GPU run using the old global-scale rule -- all of them the
``scales`` gradient at large magnitude, i.e. float32 round-off, including at
C=1 where the two paths are the same call. The combined elementwise rule
subsumes the old fix; see ``compare_tensors``' docstring.

RECORD, DO NOT DIE
------------------
Every configuration is wrapped: if this ROCm gsplat build simply refuses C > 1,
or OOMs, the exact exception type/message is recorded for that configuration
and the probe continues. A clean "not supported, and here is the exception" is
a FULLY VALID outcome of this probe, not a crash. ``probe_results.json`` is
written even when there is no GPU at all (that is recorded as the outcome).

USAGE (inside the project container -- host Python has no ROCm torch/gsplat)
---------------------------------------------------------------------------
    docker compose run --rm train \\
      python scripts/issue106_camera_batch_probe.py \\
      --output-dir output/issue106_camera_batch_probe

Bare invocation is a valid short probe. ``--sizes``/``--cameras`` widen it.
Fully synthetic -- no dataset, no scene, no checkpoint.

``--self-test`` runs a CPU-only mechanical check of every non-kernel helper
(population/camera construction, the sparse-COO-aware comparison, the timing
wrapper, the table renderer) and exits without touching the GPU.

NOTE on an unsupported build: if the batched forward raises, that configuration
returns immediately with the exception recorded and produces NO runtime/memory
numbers -- including no sequential baseline. If the probe comes back
"unsupported", re-run with ``--cameras 1`` to get sequential reference timings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import torch

# Values copied from eligsir/mapping/renderer/gsplat_backend.py::render()
# (lines 301-326). Changing these makes the probe measure something other than
# what production rendering actually does.
RASTER_KWARGS: dict[str, Any] = {
    "near_plane": 0.01,
    "far_plane": 1e10,
    "radius_clip": 0.0,
    "eps2d": 0.3,
    "packed": True,
    "tile_size": 16,  # NOT gsplat's default 8 -- NaN grads on gfx1151.
    "backgrounds": None,
    "render_mode": "RGB",
    "absgrad": False,
    "rasterize_mode": "classic",
}

PARAM_NAMES = ("means", "quats", "scales", "opacities", "colors")


# --------------------------------------------------------------------------- #
# environment / provenance
# --------------------------------------------------------------------------- #
def _git_commit(explicit: str | None) -> dict[str, Any]:
    """Resolve the git commit, tolerantly.

    ``docker compose`` mounts ``./:/workspace/eligsir``. When the repo is
    checked out as a git WORKTREE, ``.git`` is a *file* pointing at
    ``<main repo>/.git/worktrees/<name>``, which is NOT mounted -- so
    ``git rev-parse`` inside the container legitimately fails. Fall back to
    ``--git-commit`` / ``$ELIGSIR_GIT_COMMIT``, else record null + reason.
    """
    if explicit:
        return {"commit": explicit, "source": "--git-commit"}
    env = os.environ.get("ELIGSIR_GIT_COMMIT")
    if env:
        return {"commit": env, "source": "ELIGSIR_GIT_COMMIT"}
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if out.returncode == 0 and out.stdout.strip():
            return {"commit": out.stdout.strip(), "source": "git rev-parse"}
        return {"commit": None, "source": None,
                "reason": f"git rev-parse rc={out.returncode}: {out.stderr.strip()[:200]}"}
    except Exception as exc:  # noqa: BLE001 - provenance must never abort the probe
        return {"commit": None, "source": None,
                "reason": f"{type(exc).__name__}: {exc}"}


def _env_info(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", None),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "git": _git_commit(args.git_commit),
    }
    try:
        info["torch_hip_version"] = getattr(torch.version, "hip", None)
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception as exc:  # noqa: BLE001
        info["torch_version_probe_error"] = f"{type(exc).__name__}: {exc}"

    if torch.cuda.is_available():
        try:
            info["device_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["device_total_memory_bytes"] = int(props.total_memory)
            info["device_gcn_arch"] = getattr(props, "gcnArchName", None)
        except Exception as exc:  # noqa: BLE001
            info["device_probe_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import gsplat  # noqa: PLC0415 - deliberately lazy

        info["gsplat"] = getattr(gsplat, "__version__", None)
        info["gsplat_file"] = getattr(gsplat, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        info["gsplat"] = None
        info["gsplat_import_error"] = f"{type(exc).__name__}: {exc}"

    for var in ("HSA_XNACK", "PYTORCH_ROCM_ARCH", "ROCM_PATH", "HIP_VISIBLE_DEVICES"):
        if var in os.environ:
            info.setdefault("env", {})[var] = os.environ[var]
    return info


# --------------------------------------------------------------------------- #
# synthetic population + cameras
# --------------------------------------------------------------------------- #
def make_population(n: int, sh_degree: int, device: torch.device,
                    seed: int) -> dict[str, torch.Tensor]:
    """A synthetic Gaussian population in the layout gsplat_backend normalizes to.

    Mirrors ``GsplatGaussianModel``'s ``nn.ParameterDict`` keys (means, sh0,
    shN, opacities, scales, quats) but stores the ALREADY-ACTIVATED values the
    renderer passes to ``rasterization`` (``get_scaling`` = exp(scales),
    ``get_opacity`` = sigmoid(opacities)), since the probe measures the
    rasterizer, not the activation functions.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    # A shell of Gaussians in a ~4m box centred at the origin, i.e. roughly the
    # spatial extent of a small indoor streaming scene.
    means = (torch.rand((n, 3), generator=g) * 4.0 - 2.0).to(device)
    quats = torch.randn((n, 4), generator=g).to(device)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    # log-scales around 2cm, matching --streaming_insert_voxel_size 0.02.
    scales = (torch.rand((n, 3), generator=g) * 0.02 + 0.01).to(device)
    opacities = (torch.rand((n,), generator=g) * 0.6 + 0.2).to(device)

    k = (sh_degree + 1) ** 2
    sh = (torch.rand((n, k, 3), generator=g) * 0.4 - 0.2).to(device)
    sh[:, 0, :] = torch.rand((n, 3), generator=g).to(device)  # DC term ~ sh0

    for t in (means, quats, scales, opacities, sh):
        t.requires_grad_(True)
    return {"means": means, "quats": quats, "scales": scales,
            "opacities": opacities, "sh": sh}


def make_cameras(c: int, width: int, height: int,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """C cameras on an arc at radius 5m looking at the origin.

    Returns (viewmats [C,4,4] world-to-camera, Ks [C,3,3] pinhole), matching
    what ``_camera_tensors`` hands ``rasterization`` (the ACTUAL W2C matrix,
    not Inria's transposed ``world_view_transform``).
    """
    viewmats = []
    for i in range(c):
        ang = (i / max(c, 1)) * 2.0 * math.pi * 0.35
        eye = torch.tensor([5.0 * math.sin(ang), 0.6 * i / max(c - 1, 1),
                            5.0 * math.cos(ang)], dtype=torch.float32)
        fwd = torch.nn.functional.normalize(-eye, dim=0)
        up_ref = torch.tensor([0.0, 1.0, 0.0])
        right = torch.nn.functional.normalize(torch.cross(up_ref, fwd, dim=0), dim=0)
        up = torch.cross(fwd, right, dim=0)
        rot = torch.stack([right, up, fwd], dim=0)      # world -> camera
        vm = torch.eye(4, dtype=torch.float32)
        vm[:3, :3] = rot
        vm[:3, 3] = -rot @ eye
        viewmats.append(vm)

    fx = fy = 0.8 * width
    k = torch.tensor([[fx, 0.0, width / 2.0],
                      [0.0, fy, height / 2.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    return (torch.stack(viewmats).to(device).contiguous(),
            k[None].repeat(c, 1, 1).to(device).contiguous())


def precomputed_colors(pop: dict[str, torch.Tensor], c: int) -> torch.Tensor:
    """Per-camera post-activation colors (C, N, 3), the ``sh_impl='python'`` shape.

    Derived from the DC SH term with a small per-camera offset so the C slices
    are genuinely different (a broadcast-identical tensor would hide any
    per-camera indexing bug in the batched path).
    """
    dc = pop["sh"][:, 0, :]
    offs = torch.linspace(0.0, 0.1, c, device=dc.device).view(c, 1, 1)
    return (dc[None].expand(c, -1, -1) + offs).contiguous()


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
def _dense(t: torch.Tensor | None) -> torch.Tensor | None:
    if t is None:
        return None
    if t.is_sparse:            # sparse_grad=True returns sparse COO grads
        return t.coalesce().to_dense()
    return t


def _percentile_py(data: list[float], p: float) -> float:
    """Pure-Python percentile for timing stats (no numpy dependency)."""
    if not data:
        return float("nan")
    s = sorted(data)
    i = (len(s) - 1) * p / 100.0
    i_low, i_high = int(i), int(i) + 1
    frac = i - i_low
    if i_high >= len(s):
        return float(s[-1])
    return s[i_low] * (1 - frac) + s[i_high] * frac


def noise_floor_comparison(
    cross: dict[str, Any],
    self_batched: dict[str, Any],
    self_sequential: dict[str, Any],
) -> tuple[dict[str, Any], bool | None]:
    """Issue #917: is a cross-path gradient disagreement above the noise floor?

    ``cross`` is the batched-vs-sequential ``compare_tensors`` block; the other
    two are the same-path repeat comparisons (batched-vs-batched and
    sequential-vs-sequential). All three are keyed by ``PARAM_NAMES``.

    Per parameter, the cross-path ``max_abs_diff`` is held against the LARGER
    of the two same-path ``max_abs_diff`` values. The same-path number is the
    empirical run-to-run noise of that kernel at that configuration -- for a
    ``sparse_grad=False`` backward it is the float32 ``atomicAdd`` accumulation
    order, which is not guaranteed to repeat between two launches. A cross-path
    difference at or below that floor is therefore fully explained by
    accumulation order and is NOT evidence of a gradient defect, no matter what
    ``backward.all_within_tolerance`` says.

    THE CELL VERDICT IS ``any()`` OVER A CONJUNCTION, NOT ``all()`` OVER THE
    RATIO (issue #920). A parameter's disagreement is *unexplained* iff it BOTH

      (a) fails the cross-path parity tolerance -- there is a real disagreement
          to account for at all; AND
      (b) strictly exceeds its own same-path noise floor -- that disagreement is
          not accumulation order.

    and the cell exceeds the floor iff ANY parameter's disagreement is
    unexplained. Both halves are load-bearing, and the first real dataset
    (``i917selfcons_n500k``) exercises both failure modes:

      * A bare ``all()`` over (b) -- what this function shipped with -- reported
        2 of 18 dense ``C > 1`` cells when the per-parameter data said 18 of 18.
        ``opacities`` and ``colors`` have no cross-path disagreement to explain
        (they pass parity, ratio ~1.0), so conjoining them makes the verdict
        unreachable whenever the defect is LOCALIZED -- the normal shape of a
        real gradient bug, and exactly what #898 found on means/quats/scales.
      * A bare ``any()`` over (b) alone would flip the error: in the
        ``sparse_grad=True, C > 1`` control arm, 5-8 of 18 cells carry some
        parameter at ratio 1.5-2.0, boundary wobble around 1.0 on parameters
        with no real disagreement. Requiring (a) keeps that arm ``False``.

    Returns ``(per_parameter_dict, overall)``. ``overall`` is True when at least
    one parameter's disagreement is unexplained, False when none is, and ``None``
    when any parameter could not be resolved (a missing or non-float
    ``max_abs_diff``, or a non-boolean ``within_tolerance``) -- never a silent
    False.
    """
    per_param: dict[str, Any] = {}
    for name in PARAM_NAMES:
        cross_p = cross.get(name) or {}
        c = cross_p.get("max_abs_diff")
        nb = (self_batched.get(name) or {}).get("max_abs_diff")
        ns = (self_sequential.get(name) or {}).get("max_abs_diff")
        within = cross_p.get("within_tolerance")
        if not all(isinstance(v, float) and v == v for v in (c, nb, ns)):
            per_param[name] = None
            continue
        if not isinstance(within, bool):
            # `within_tolerance` is present on every `compare_tensors` block that
            # carries a `max_abs_diff`, so this is an unexpected shape, not an
            # ordinary absence. Refuse the parameter rather than guessing which
            # way to read it.
            per_param[name] = None
            continue
        noise = max(nb, ns)
        exceeds = bool(c > noise)
        per_param[name] = {
            "cross_path_max_abs_diff": c,
            "self_path_max_abs_diff": noise,
            "ratio": (c / noise) if noise > 0.0 else None,
            "exceeds_noise_floor": exceeds,
            "cross_within_tolerance": within,
            "disagreement_unexplained": bool(exceeds and not within),
        }
    if not per_param or any(v is None for v in per_param.values()):
        return per_param, None
    return per_param, any(
        v["disagreement_unexplained"] for v in per_param.values())


def compare_tensors(a: torch.Tensor | None, b: torch.Tensor | None,
                    tol: float, rel_tol: float = 1e-4) -> dict[str, Any]:
    """Elementwise relative-error criterion |a_i - b_i| <= atol + rtol * |b_i|.

    This function computes per-tensor parity using the elementwise criterion
    equivalent to torch.testing.assert_close semantics. The verdict passes iff
    every element satisfies: |a_i - b_i| <= tol + rel_tol * |b_i|.

    HISTORY OF THE OR RULE (first GPU run, 2026-08-16): an absolute-only verdict
    reported ``backward_matches_sequential: false`` for 31 of 32 configurations
    -- including at **C=1**, where the two paths are literally the same call and
    can only differ by nondeterministic atomic accumulation order. Every one of
    those failures was the ``scales`` gradient at ``max_abs_diff ~ 1.5e-3``
    against ``ref_max_abs ~ 6.6e3``, i.e. a global-scale ratio of ~2.2e-7 --
    pure float32 round-off on a tensor of magnitude 1e4. The old rule computed
    a single ``scale = max(|b|)`` and checked ``max(|diff|) / scale <= rtol``,
    which hides errors on small entries when a large reference entry exists.
    The new elementwise rule subsumes that fix: it applies the tolerance to each
    entry independently, so small-magnitude entries get scrutinized fairly.

    MAGNITUDE MASK: A relative error is meaningless on near-zero entries. This
    function computes a scale-relative magnitude threshold per tensor:
    ``mag_threshold = max(abs_floor, rel_floor * ref_max_abs)`` where
    ``abs_floor`` is a tiny fixed value (1e-7) and ``rel_floor`` is 0.1% of the
    reference max. Entries below this threshold are excluded from the relative
    error statistics (p50/p95/p99, cosine similarity, sign disagreement); the
    absolute error is still reported. An all-zero reference tensor returns
    None for relative quantities and records the reason.

    INFINITY HANDLING: Infinite values (±∞) on either side always fail, even if
    both sides match exactly (e.g., +∞ vs +∞). NaN on either side always fails.

    Per-tensor fields returned:
      - available, reason: whether the comparison ran
      - max_abs_diff, mean_abs_diff: absolute error
      - ref_max_abs, ref_mean_abs: reference magnitude
      - max_rel_diff_nonneg: max elementwise relative error over non-negligible
        entries; None if no entries above magnitude threshold
      - p50_rel_error, p95_rel_error, p99_rel_error: percentiles over non-negligible
        entries; None if insufficient entries
      - cosine_similarity: cosine similarity over non-negligible entries; None if
        reference has zero norm
      - sign_disagreement_frac: fraction of non-negligible entries where a and b
        have opposite signs; None if no non-negligible entries
      - n_nan_a, n_nan_b: NaN counts
      - n_inf_a, n_inf_b: Infinity counts
      - n_nonfinite_a, n_nonfinite_b: NaN + Inf counts
      - magnitude_threshold_abs_floor: absolute floor (1e-7)
      - magnitude_threshold_rel_floor: relative floor as fraction of ref_max_abs
      - magnitude_threshold_resolved: actual resolved threshold used
      - n_nonnegl_entries: count of entries above magnitude threshold
      - rel_l2_error: relative L2 error (||a-b||_2 / ||b||_2); None if b is zero
      - tolerance, rel_tolerance: parameters
      - legacy_maxabs_over_refmax: old global-scale ratio (no longer used for verdict)
      - within_tolerance: boolean verdict (all entries pass the elementwise bound
        AND no non-finite values)
    """
    a, b = _dense(a), _dense(b)
    if a is None or b is None:
        return {"available": False,
                "reason": f"a_is_none={a is None} b_is_none={b is None}"}
    if a.shape != b.shape:
        return {"available": False,
                "reason": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}

    with torch.no_grad():
        a32, b32 = a.detach().float(), b.detach().float()
        diff = (a32 - b32).abs()

        # Basic magnitudes
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        ref_max_abs = float(b32.abs().max().item())
        ref_mean_abs = float(b32.abs().mean().item())

        # Non-finite counts
        nan_a = int(torch.isnan(a32).sum().item())
        nan_b = int(torch.isnan(b32).sum().item())
        inf_a = int(torch.isinf(a32).sum().item())
        inf_b = int(torch.isinf(b32).sum().item())
        has_nonfinite = (nan_a + nan_b + inf_a + inf_b) > 0

        # Magnitude threshold: elements below this are excluded from relative stats
        abs_floor = 1e-7
        rel_floor = 1e-3  # 0.1% of reference max
        mag_threshold = max(abs_floor, rel_floor * ref_max_abs)

        # Mask of non-negligible entries
        b_abs = b32.abs()
        nonnegl_mask = b_abs >= mag_threshold
        n_nonnegl = int(nonnegl_mask.sum().item())

        out = {
            "available": True,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "ref_max_abs": ref_max_abs,
            "ref_mean_abs": ref_mean_abs,
            "tolerance": tol,
            "rel_tolerance": rel_tol,
            "magnitude_threshold_abs_floor": float(abs_floor),
            "magnitude_threshold_rel_floor": float(rel_floor),
            "magnitude_threshold_resolved": float(mag_threshold),
            "n_nonnegl_entries": n_nonnegl,
            "n_nan_a": nan_a,
            "n_nan_b": nan_b,
            "n_inf_a": inf_a,
            "n_inf_b": inf_b,
            "n_nonfinite_a": nan_a + inf_a,
            "n_nonfinite_b": nan_b + inf_b,
        }

        # Elementwise criterion: |a_i - b_i| <= tol + rtol * |b_i|
        bounds = tol + rel_tol * b_abs
        within_bounds = diff <= bounds
        all_within = bool(within_bounds.all().item()) and not has_nonfinite
        out["within_tolerance"] = all_within

        # Relative error statistics, only over non-negligible entries
        if n_nonnegl > 0:
            nonnegl_diff = diff[nonnegl_mask]
            nonnegl_b = b_abs[nonnegl_mask]
            rel_errors = (nonnegl_diff / nonnegl_b).detach().cpu().tolist()

            out["max_rel_diff_nonneg"] = float(max(rel_errors)) if rel_errors else None
            # Percentiles require at least that many entries
            if len(rel_errors) >= 2:
                out["p50_rel_error"] = _percentile_py(rel_errors, 50.0)
            else:
                out["p50_rel_error"] = None
            if len(rel_errors) >= 20:
                out["p95_rel_error"] = _percentile_py(rel_errors, 95.0)
                out["p99_rel_error"] = _percentile_py(rel_errors, 99.0)
            else:
                out["p95_rel_error"] = None
                out["p99_rel_error"] = None

            # Cosine similarity over non-negligible entries
            a_nonnegl = a32[nonnegl_mask]
            b_nonnegl = b32[nonnegl_mask]
            a_norm = torch.norm(a_nonnegl)
            b_norm = torch.norm(b_nonnegl)
            if a_norm > 0 and b_norm > 0:
                cos_sim = float((a_nonnegl * b_nonnegl).sum().item() / (a_norm * b_norm).item())
                out["cosine_similarity"] = cos_sim
            else:
                out["cosine_similarity"] = None

            # Sign disagreement over non-negligible entries
            sign_a = torch.sign(a_nonnegl)
            sign_b = torch.sign(b_nonnegl)
            disagree = (sign_a != sign_b).sum().item()
            out["sign_disagreement_frac"] = float(disagree / max(n_nonnegl, 1))
        else:
            out["max_rel_diff_nonneg"] = None
            out["p50_rel_error"] = None
            out["p95_rel_error"] = None
            out["p99_rel_error"] = None
            out["cosine_similarity"] = None
            out["sign_disagreement_frac"] = None

        # Relative L2 error: ||a-b||_2 / ||b||_2
        b_norm = torch.norm(b32)
        if b_norm > 0:
            diff_norm = torch.norm(diff)
            out["rel_l2_error"] = float((diff_norm / b_norm).item())
        else:
            out["rel_l2_error"] = None

        # For backward compatibility: legacy_maxabs_over_refmax is the old
        # global-scale ratio that is no longer the verdict mechanism
        out["legacy_maxabs_over_refmax"] = (max_abs / ref_max_abs) if ref_max_abs > 0 else None

    return out


def grab_grads(pop: dict[str, torch.Tensor],
               colors: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
    out: dict[str, torch.Tensor | None] = {}
    for key in ("means", "quats", "scales", "opacities"):
        g = pop[key].grad
        out[key] = _dense(g).detach().clone() if g is not None else None
    src = colors if colors is not None else pop["sh"]
    g = src.grad
    out["colors"] = _dense(g).detach().clone() if g is not None else None
    return out


def zero_grads(pop: dict[str, torch.Tensor], colors: torch.Tensor | None) -> None:
    for key in ("means", "quats", "scales", "opacities", "sh"):
        pop[key].grad = None
    if colors is not None:
        colors.grad = None


# --------------------------------------------------------------------------- #
# the two render paths
# --------------------------------------------------------------------------- #
def _call(rasterization, pop, colors_arg, sh_degree_arg, viewmats, ks,
          width, height, sparse_grad):
    return rasterization(
        means=pop["means"],
        quats=pop["quats"],
        scales=pop["scales"],
        opacities=pop["opacities"],
        colors=colors_arg,
        viewmats=viewmats,
        Ks=ks,
        width=width,
        height=height,
        sh_degree=sh_degree_arg,
        sparse_grad=sparse_grad,
        **RASTER_KWARGS,
    )


def render_batched(rasterization, pop, color_mode, colors, sh_degree,
                   viewmats, ks, width, height, sparse_grad):
    """ONE rasterization call over all C cameras -> [C, H, W, 3]."""
    if color_mode == "sh":
        colors_arg, sh_arg = pop["sh"], sh_degree
    else:
        colors_arg, sh_arg = colors, None
    out, _alphas, _meta = _call(rasterization, pop, colors_arg, sh_arg,
                                viewmats, ks, width, height, sparse_grad)
    return out


def render_sequential(rasterization, pop, color_mode, colors, sh_degree,
                      viewmats, ks, width, height, sparse_grad):
    """C rasterization calls at C=1 each; returns the list of [1, H, W, 3]."""
    outs = []
    for i in range(viewmats.shape[0]):
        if color_mode == "sh":
            colors_arg, sh_arg = pop["sh"], sh_degree
        else:
            colors_arg, sh_arg = colors[i:i + 1], None
        out, _alphas, _meta = _call(
            rasterization, pop, colors_arg, sh_arg,
            viewmats[i:i + 1].contiguous(), ks[i:i + 1].contiguous(),
            width, height, sparse_grad,
        )
        outs.append(out)
    return outs


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _get_gpu_telemetry() -> dict[str, Any]:
    """Capture GPU clock/power/temperature if rocm-smi is available (best-effort)."""
    try:
        import subprocess  # noqa: PLC0415 - already imported at module scope
        result = subprocess.run(
            ["rocm-smi", "--json"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            import json  # noqa: PLC0415 - already imported at module scope
            data = json.loads(result.stdout)
            # rocm-smi --json schema: {card0: {sclk: {current: [MHz]}, mclk: {current: [MHz]}, ...}, ...}
            # Extract first GPU's data (card0) if available
            if isinstance(data, dict):
                first_card_key = next(iter(data), None)  # Get first key (e.g., "card0")
                if first_card_key and first_card_key in data:
                    gpu_data = data[first_card_key]
                    return {
                        "available": True,
                        "sclk_mhz": gpu_data.get("sclk", {}).get("current", [None])[0],
                        "mclk_mhz": gpu_data.get("mclk", {}).get("current", [None])[0],
                        "soc_power_w": gpu_data.get("soc_power", [None])[0],
                        "gpu_power_w": gpu_data.get("gpu_power", [None])[0],
                        "junction_temp_c": gpu_data.get("junction_temp", [None])[0],
                    }
            return {"available": False, "reason": "rocm-smi returned data with unexpected schema"}
        return {"available": False, "reason": f"rocm-smi rc={result.returncode}"}
    except FileNotFoundError:
        return {"available": False, "reason": "rocm-smi not found in PATH"}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def time_paired(fn_reference, fn_candidate, warmup: int, iters: int,
                ) -> dict[str, Any]:
    """Paired ABBA timing measurement: reference, candidate, candidate, reference.

    Returns both series and the per-sample speedup ratio distribution so the
    caller can compute uncertainty from the paired samples.
    """
    try:
        # Warmup both sides
        for _ in range(warmup):
            fn_reference()
        for _ in range(warmup):
            fn_candidate()
        _sync()

        ref_times = []
        cand_times = []
        speedups = []

        for _ in range(iters):
            # ABBA pattern: reference, candidate, candidate, reference
            _sync()
            t0 = time.perf_counter()
            fn_reference()
            _sync()
            ref_times.append((time.perf_counter() - t0) * 1000.0)

            _sync()
            t0 = time.perf_counter()
            fn_candidate()
            _sync()
            cand_times.append((time.perf_counter() - t0) * 1000.0)

            _sync()
            t0 = time.perf_counter()
            fn_candidate()
            _sync()
            cand_times.append((time.perf_counter() - t0) * 1000.0)

            _sync()
            t0 = time.perf_counter()
            fn_reference()
            _sync()
            ref_times.append((time.perf_counter() - t0) * 1000.0)

            # Paired ratio (speedup = ref / cand)
            if cand_times[-1] > 0:
                speedups.append(ref_times[-1] / cand_times[-1])

        ref_times.sort()
        cand_times.sort()
        speedups.sort()

        def _stats(times):
            if not times:
                return {}
            mean = sum(times) / len(times)
            median = times[len(times) // 2]
            p10 = _percentile_py(times, 10.0)
            p90 = _percentile_py(times, 90.0)
            variance = sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)
            stdev = variance ** 0.5
            return {
                "mean_ms": mean,
                "median_ms": median,
                "p10_ms": p10,
                "p90_ms": p90,
                "stdev_ms": stdev,
                "min_ms": times[0],
                "max_ms": times[-1],
            }

        return {
            "ok": True,
            "iters": iters,
            "warmup": warmup,
            "reference": {**_stats(ref_times), "n_samples": len(ref_times)},
            "candidate": {**_stats(cand_times), "n_samples": len(cand_times)},
            "speedup_reference_over_candidate": {
                **_stats(speedups),
                "n_samples": len(speedups),
                "definition": "speedup = reference_ms / candidate_ms",
            },
            "gpu_telemetry_reference_start": _get_gpu_telemetry(),
            "gpu_telemetry_candidate_end": _get_gpu_telemetry(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error_type": type(exc).__name__,
                "error": str(exc)[:600]}


def time_it(fn, warmup: int, iters: int) -> dict[str, Any]:
    try:
        for _ in range(warmup):
            fn()
        _sync()
        samples = []
        for _ in range(iters):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / max(len(samples) - 1, 1)
        stdev = variance ** 0.5
        return {
            "ok": True,
            "iters": iters,
            "warmup": warmup,
            "mean_ms": mean,
            "median_ms": samples[len(samples) // 2],
            "p10_ms": _percentile_py(samples, 10.0),
            "p90_ms": _percentile_py(samples, 90.0),
            "stdev_ms": stdev,
            "min_ms": samples[0],
            "max_ms": samples[-1],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error_type": type(exc).__name__,
                "error": str(exc)[:600]}


def _peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _build_comparison(reference: str, candidate: str,
                      ref_ms: float, cand_ms: float) -> dict[str, Any] | None:
    """Build a comparison entry per #898: speedup = reference_ms / candidate_ms.

    This helper is called by both run_config() (shipping path) and self_test()
    (regression pin). An inversion in one path fails the test because both share
    this code under test.
    """
    if cand_ms <= 0:
        return None
    speedup = ref_ms / cand_ms
    return {
        "reference": reference,
        "candidate": candidate,
        "speedup_reference_over_candidate": speedup,
        "definition": "speedup = reference_ms / candidate_ms",
    }


def _build_all_comparisons(runtime: dict) -> dict:
    """Build every `runtime["comparisons"]` entry per #898.

    Extracted verbatim from the loop that used to live inline in run_config()
    so self_test() can call the exact same code the shipping path calls,
    rather than only pinning the _build_comparison() helper in isolation
    (which cannot see which dict each argument came from, and so cannot
    catch a reference/candidate inversion at the call site).
    """
    comparisons: dict[str, Any] = {}
    for pair in (("fwd_bwd_sequential_retained", "fwd_bwd_batched"),
                 ("fwd_bwd_sequential_streamed", "fwd_bwd_batched"),
                 ("fwd_bwd_sequential_retained", "fwd_bwd_sequential_streamed")):
        reference, candidate = pair
        ref_data = runtime.get(reference, {})
        cand_data = runtime.get(candidate, {})
        if (ref_data.get("ok") and cand_data.get("ok") and
                ref_data.get("mean_ms", 0) > 0):
            comp = _build_comparison(reference, candidate,
                                     ref_data["mean_ms"], cand_data["mean_ms"])
            if comp:
                comparisons[f"{reference}_vs_{candidate}"] = comp
    return comparisons


def _measure_peak(fn) -> dict[str, Any]:
    _reset_peak()
    try:
        fn()
        _sync()
        return {"ok": True, "peak_mb": _peak_mb()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error_type": type(exc).__name__,
                "error": str(exc)[:600], "peak_mb": _peak_mb()}


# --------------------------------------------------------------------------- #
# finite-difference gradient reference (issue #917 bisect)
# --------------------------------------------------------------------------- #
# WHY THIS EXISTS
# ---------------
# #918 measured the noise floor and #920 fixed how it is read: the
# batched-vs-sequential BACKWARD disagrees on means/quats/scales exactly when
# `C > 1 AND sparse_grad=False`, at cross-path rel_l2 0.373-1.02 against a
# same-path repeat floor of ~1e-7. The disagreement is REAL. What no comparison
# between those two paths can ever say is WHICH ONE IS RIGHT -- that needs a
# third, independent reference.
#
# This is that reference, and it needs no analytic projection Jacobian and no
# CPU rasterizer. Both existing paths differentiate the SAME scalar objective
# -- a plain sum() over every pixel of every view -- so for one scalar element
# `p` of one parameter tensor:
#
#     g_fd(p) = ( S(p + eps) - S(p - eps) ) / (2 * eps)
#
# evaluated FORWARD-ONLY under `no_grad`. The forward is verified path-parity
# clean on every cell, so an FD reference computed through it cannot inherit
# the bias of whichever backward it is judging -- and this module MEASURES that
# rather than assuming it, by running the whole stencil independently through
# both closures and comparing (`fd_path_independence`).
#
# ACCURACY TARGET IS LOOSE ON PURPOSE. The disagreement to discriminate is
# ~0.4-1.0 RELATIVE. An FD estimate good to ~10-20% settles it. Do not read the
# gates below as an attempt at a high-accuracy derivative; they are an attempt
# at an HONEST one -- every element either clears all of them or is reported
# `unresolved` with the failing gates named, and is never scored.
#
# THREE REJECTION GATES (each can fire independently; all are recorded)
#   (a) eps_agreement    -- the estimate must not move materially with eps.
#                           Judged against the MEASURED repeat noise at each eps
#                           (see gate c), with a relative floor -- a fixed
#                           tolerance would turn this gate into a noise detector
#                           at the small-eps end and reject everything.
#   (b) smoothness       -- |S(+) - 2 S(0) + S(-)| must be small against
#                           |S(+) - S(-)|. The rasterizer is PIECEWISE (tile
#                           assignment, radius_clip, near_plane); an element
#                           sitting on a discontinuity gives a meaningless
#                           difference, and this is what catches it.
#   (c) self_consistency -- `bit_exact` is False on 48/48 cells, so the FORWARD
#                           is nondeterministic too. Per-pixel noise sigma over
#                           K touched pixels puts ~sigma*sqrt(K)/(2*eps) on the
#                           estimate -- AMPLIFIED by 1/eps and, before this,
#                           entirely unmeasured. The whole stencil is evaluated
#                           twice at every eps and the repeat spread recorded.
#                           This is #918's contribution one layer up: a
#                           difference is evidence only once the noise it has to
#                           clear is measured.
#   (+) signal           -- a fourth, bookkeeping gate: an element whose FD
#                           estimate is ~0 cannot discriminate anything.
#   (+) path_independence-- an element whose two closure-FD estimates disagree
#                           is refused (conservative). Broad failure additionally
#                           forces the whole cell to `inconclusive`, because it
#                           falsifies the premise this reference rests on.
#
# CATASTROPHIC CANCELLATION is avoided by never differencing two float32 scalar
# sums: the rendered TENSORS are differenced elementwise (nearby floats, so the
# subtraction is essentially exact) and only then reduced in float64.
#
# The driver is deliberately separable from the kernel -- it takes a callable
# returning a tensor -- because this file's comparison-assembly layer has
# produced confidently-wrong conclusions twice (the OR rule in
# `compare_tensors`, the `all()` in `noise_floor_comparison`/#920) and both
# times the fix was cheap only because the logic was testable without a GPU.
# `--self-test` exercises the happy path AND every rejection gate.

# Parameters the bisect samples. `colors` is excluded on purpose: in `sh` color
# mode it aliases `pop["sh"]`, so "the raw tensor autograd tracked" is
# mode-dependent, and #898 already found zero cross-path disagreement there.
FD_PARAM_NAMES = ("means", "quats", "scales", "opacities")


class FDPerturbationError(RuntimeError):
    """A perturbation did not land, or a restore was not bitwise exact.

    Raised rather than reported, because either one silently turns every
    subsequent FD estimate into garbage that looks clean -- e.g. a
    non-contiguous tensor whose `reshape(-1)` returns a COPY makes the write a
    no-op, `g_fd` comes back ~0 for every element, and the bisect confidently
    answers `neither`.
    """


def _fd_flat_view(tensor: torch.Tensor) -> torch.Tensor:
    """A flat VIEW of ``tensor``'s storage -- never a copy.

    ``view(-1)`` raises on a non-contiguous tensor instead of silently copying
    the way ``reshape(-1)`` would. That raise is the point.
    """
    return tensor.detach().view(-1)


def _fd_write(tensor: torch.Tensor, flat_index: int, value: float) -> None:
    """Write ``value`` into one element and VERIFY it landed."""
    flat = _fd_flat_view(tensor)
    expected = float(torch.tensor([value], dtype=tensor.dtype).item())
    with torch.no_grad():
        flat[flat_index] = value
    got = float(flat[flat_index].item())
    if got != expected:
        raise FDPerturbationError(
            f"perturbation did not land: wrote {value!r} (dtype-rounded "
            f"{expected!r}) at flat index {flat_index}, read back {got!r}")


def _fd_reduce_pair(a: torch.Tensor, b: torch.Tensor) -> float:
    """``(a - b)`` elementwise, summed in float64.

    The elementwise subtraction happens in the tensors' own dtype on two nearby
    values (exact under Sterbenz for float32 operands within a factor of two),
    so the large common part cancels PER PIXEL, before any summation. Reducing
    the already-cancelled residual in float64 then loses nothing.
    """
    return float((a - b).double().sum().item())


def fd_stencil_element(
    render_fn,
    tensor: torch.Tensor,
    flat_index: int,
    eps_values,
    *,
    repeats: int = 2,
) -> dict[str, Any]:
    """Three-point FD stencil at one scalar element, at each eps, twice.

    ``render_fn()`` must return a float tensor (the forward output) and must be
    called under ``no_grad`` by the caller's closure or by this function's
    context -- it is invoked inside ``torch.no_grad()`` here.

    Returns a dict with a ``per_eps`` list carrying, for every eps:
      ``eps``                    absolute eps actually used
      ``estimates``              one FD estimate per repeat
      ``g``                      mean of ``estimates``
      ``noise_abs``              max spread across repeats (the measured noise)
      ``first_diff`` /``second_diff``  float64 reductions at the last repeat
      ``n_pixels_changed``       K -- output elements the perturbation moved
      ``smoothness_ratio``       |second_diff| / |first_diff|

    Every quantity a later retune would need (absolute eps, raw |S(+)-S(-)|,
    K) is recorded, so a run in which every element comes back unresolved is a
    TUNING DATASET rather than a wasted slot.
    """
    out: dict[str, Any] = {"flat_index": int(flat_index), "per_eps": []}
    orig_t = _fd_flat_view(tensor)[flat_index].clone()
    orig = float(orig_t.item())
    out["value"] = orig

    with torch.no_grad():
        base = render_fn().detach().clone()
        try:
            for eps in eps_values:
                eps = float(eps)
                entry: dict[str, Any] = {"eps": eps, "estimates": []}
                for _rep in range(max(int(repeats), 1)):
                    _fd_write(tensor, flat_index, orig + eps)
                    plus = render_fn().detach().clone()
                    _fd_write(tensor, flat_index, orig - eps)
                    minus = render_fn().detach().clone()
                    # d_plus/d_minus rather than a single (plus - minus): the
                    # decomposition yields the first AND second difference from
                    # the same two cancelled residuals, consistently.
                    d_plus = _fd_reduce_pair(plus, base)
                    d_minus = _fd_reduce_pair(base, minus)
                    first = d_plus + d_minus
                    second = d_plus - d_minus
                    entry["estimates"].append(first / (2.0 * eps))
                    entry["first_diff"] = first
                    entry["second_diff"] = second
                    entry["n_pixels_changed"] = int(
                        (plus != minus).sum().item())
                    del plus, minus
                ests = entry["estimates"]
                entry["g"] = sum(ests) / len(ests)
                entry["noise_abs"] = (max(ests) - min(ests)) if len(ests) > 1 else None
                fd_abs = abs(entry["first_diff"])
                entry["smoothness_ratio"] = (
                    abs(entry["second_diff"]) / fd_abs) if fd_abs > 0.0 else None
                out["per_eps"].append(entry)
        finally:
            # Restore BITWISE, from the saved tensor element (not from the
            # python float), and verify. A lossy restore would quietly corrupt
            # every element evaluated after this one.
            with torch.no_grad():
                _fd_flat_view(tensor)[flat_index] = orig_t
            back = _fd_flat_view(tensor)[flat_index]
            if not bool(torch.equal(back, orig_t)):
                raise FDPerturbationError(
                    f"restore was not bitwise exact at flat index {flat_index}: "
                    f"{float(back.item())!r} != {orig!r}")
    return out


def fd_apply_gates(
    stencil: dict[str, Any],
    *,
    eps_rel_tol: float = 0.10,
    eps_noise_slack: float = 3.0,
    smoothness_tol: float = 0.10,
    noise_signal_tol: float = 0.20,
    min_signal: float = 0.0,
) -> dict[str, Any]:
    """Apply the rejection gates to one ``fd_stencil_element`` result.

    Returns ``{g_fd, reference_eps, gates, failed_gates, resolved, ...}``.
    ALL gates are evaluated and recorded, never short-circuited on the first
    failure: injected noise trips (c) but usually also (a), so a first-failure
    report could not demonstrate (c) at all.

    Gate (a) is judged against the noise gate (c) MEASURES, not a fixed
    tolerance: each FD estimate carries ~sigma*sqrt(K)/(2*eps), so the small-eps
    end is intrinsically noisier and a fixed spread bound would reject nearly
    everything. The eps-to-eps spread passes when it is within either the
    measured repeat noise (times a slack) or a relative floor.
    """
    per_eps = stencil.get("per_eps") or []
    result: dict[str, Any] = {
        "flat_index": stencil.get("flat_index"),
        "value": stencil.get("value"),
        "per_eps": per_eps,
    }
    if not per_eps:
        result.update({"g_fd": None, "reference_eps": None, "resolved": False,
                       "gates": {}, "failed_gates": ["no_samples"]})
        return result

    def _noise_rel(e):
        na, g = e.get("noise_abs"), e.get("g")
        if na is None or g is None or abs(g) <= 0.0:
            return float("inf")
        return abs(na) / abs(g)

    # Reference eps: the one with the smallest measured RELATIVE noise, ties
    # broken by order. Chosen independently of the gates so the gate verdicts
    # cannot depend circularly on which eps they are evaluated at.
    ref = min(range(len(per_eps)),
              key=lambda i: (_noise_rel(per_eps[i]), i))
    ref_entry = per_eps[ref]
    g_fd = ref_entry.get("g")

    # (a) eps agreement.
    gs = [e["g"] for e in per_eps if e.get("g") is not None]
    if len(gs) < 2:
        eps_spread = 0.0
        eps_allow = float("inf")   # cannot test with one eps; do not fail it
        eps_ok = True
        eps_untested = True
    else:
        eps_untested = False
        eps_spread = max(gs) - min(gs)
        max_noise = max((e.get("noise_abs") or 0.0) for e in per_eps)
        eps_allow = max(eps_noise_slack * max_noise,
                        eps_rel_tol * max(abs(g) for g in gs))
        eps_ok = bool(eps_spread <= eps_allow)

    # (b) smoothness, at the reference eps.
    sr = ref_entry.get("smoothness_ratio")
    smooth_ok = bool(sr is not None and sr <= smoothness_tol)

    # (c) FD self-consistency, at the reference eps.
    nr = _noise_rel(ref_entry)
    noise_ok = bool(nr <= noise_signal_tol)

    # (+) signal.
    signal_ok = bool(g_fd is not None and abs(g_fd) > min_signal)

    gates = {
        "eps_agreement": {"passed": eps_ok, "spread": eps_spread,
                          "allowance": (None if eps_allow == float("inf")
                                        else eps_allow),
                          "untested_single_eps": eps_untested},
        "smoothness": {"passed": smooth_ok, "ratio": sr,
                       "tolerance": smoothness_tol},
        "self_consistency": {"passed": noise_ok, "noise_rel": (
            None if nr == float("inf") else nr), "tolerance": noise_signal_tol},
        "signal": {"passed": signal_ok, "abs_g_fd": (
            None if g_fd is None else abs(g_fd)), "floor": min_signal},
    }
    failed = [k for k, v in gates.items() if not v["passed"]]
    result.update({
        "g_fd": g_fd,
        "reference_eps": ref_entry.get("eps"),
        "reference_eps_index": ref,
        "gates": gates,
        "failed_gates": failed,
        "resolved": not failed,
    })
    return result


def fd_verdict(elements, *, accept_rel_tol: float = 0.20,
               consistency_frac: float = 0.80,
               path_independence_fail_frac: float = 0.25) -> dict[str, Any]:
    """Assemble the per-parameter verdict from scored elements.

    ``elements`` is a list of dicts with at least ``param``, ``stratum``,
    ``resolved``, ``failed_gates``, and (when resolved) ``g_fd``,
    ``g_batched``, ``g_sequential``.

    Reports RELATIVE ERRORS AS NUMBERS, and reaches an explicit ``neither``:
    both paths being wrong is a real possibility nobody has excluded. The other
    outcomes are ``batched`` / ``sequential`` / ``both`` (FD agrees with both --
    the expected answer in the agreement/control stratum, and the on-substrate
    validity check for this whole driver) / ``inconclusive``.
    """
    per_param: dict[str, Any] = {}
    for name in FD_PARAM_NAMES:
        mine = [e for e in elements if e.get("param") == name]
        resolved = [e for e in mine if e.get("resolved")]
        gate_counts: dict[str, int] = {}
        for e in mine:
            for gate in (e.get("failed_gates") or []):
                gate_counts[gate] = gate_counts.get(gate, 0) + 1

        rel_b, rel_s, winners = [], [], []
        for e in resolved:
            gfd, gb, gs = e.get("g_fd"), e.get("g_batched"), e.get("g_sequential")
            if gfd is None or gb is None or gs is None or abs(gfd) <= 0.0:
                continue
            rb, rs = abs(gb - gfd) / abs(gfd), abs(gs - gfd) / abs(gfd)
            e["rel_err_batched"], e["rel_err_sequential"] = rb, rs
            rel_b.append(rb)
            rel_s.append(rs)
            winners.append("batched" if rb < rs else
                           ("sequential" if rs < rb else "tie"))

        n_scored = len(rel_b)
        n_path_fail = sum(
            1 for e in mine
            if "path_independence" in (e.get("failed_gates") or []))
        path_fail_frac = (n_path_fail / len(mine)) if mine else 0.0

        entry: dict[str, Any] = {
            "n_sampled": len(mine),
            "n_resolved": len(resolved),
            "n_scored": n_scored,
            "unresolved_by_gate": gate_counts,
            "path_independence_failed_frac": path_fail_frac,
            "median_rel_err_batched": _percentile_py(rel_b, 50.0) if rel_b else None,
            "median_rel_err_sequential": _percentile_py(rel_s, 50.0) if rel_s else None,
            "max_rel_err_batched": max(rel_b) if rel_b else None,
            "max_rel_err_sequential": max(rel_s) if rel_s else None,
            "winner_counts": {
                w: winners.count(w) for w in ("batched", "sequential", "tie")},
            "accept_rel_tol": accept_rel_tol,
        }

        if n_scored == 0:
            entry["verdict"] = "inconclusive"
            entry["verdict_reason"] = "no element passed every gate"
            per_param[name] = entry
            continue

        mb = entry["median_rel_err_batched"]
        ms = entry["median_rel_err_sequential"]
        if mb <= accept_rel_tol and ms <= accept_rel_tol:
            verdict, reason = "both", "FD agrees with both paths"
        elif mb <= accept_rel_tol < ms:
            verdict, reason = "batched", "FD agrees with the batched path only"
        elif ms <= accept_rel_tol < mb:
            verdict, reason = "sequential", "FD agrees with the sequential path only"
        else:
            verdict, reason = "neither", "FD agrees with NEITHER path"

        # Element-level agreement must back the median up.
        decided = entry["winner_counts"]["batched"] + entry["winner_counts"]["sequential"]
        if verdict in ("batched", "sequential") and decided:
            frac = entry["winner_counts"][verdict] / decided
            entry["winner_consistency"] = frac
            if frac < consistency_frac:
                verdict = "inconclusive"
                reason = (f"median favours one path but only {frac:.2f} of "
                          f"elements agree (< {consistency_frac})")
        if path_fail_frac > path_independence_fail_frac:
            verdict = "inconclusive"
            reason = (f"{path_fail_frac:.2f} of sampled elements failed FD "
                      f"path-independence -- the reference's own premise "
                      f"(forward parity) does not hold here")
        entry["verdict"] = verdict
        entry["verdict_reason"] = reason
        per_param[name] = entry
    return per_param


def select_fd_samples(g_batched, g_sequential, tensors, n_per_param: int,
                      *, magnitude_floor_frac: float = 1e-3) -> list[dict[str, Any]]:
    """Stratified element sampling: disagreement stratum + agreement control.

    * ``disagreement`` -- for means/quats/scales, the elements where the two
      backwards differ MOST. Best FD relative accuracy, and the population the
      verdict is actually about.
    * ``agreement`` -- a CONTROL: `opacities` (which #898 found 0/90 disagreeing)
      plus the LOWEST-disagreement `means` elements. FD siding with BOTH paths
      here validates the driver against live GPU data rather than only against
      the closed-form self-test, and does so INSIDE the same cell -- so
      restricting the bisect to parity-failing cells does not cost the control.

    Candidates are restricted to elements whose gradient magnitude clears
    ``magnitude_floor_frac`` of the tensor's own max: an FD estimate of a
    near-zero gradient discriminates nothing and just burns forwards.
    """
    samples: list[dict[str, Any]] = []
    if n_per_param <= 0:
        return samples

    def _emit(name, idx_list, stratum):
        gb, gs = g_batched.get(name), g_sequential.get(name)
        for i in idx_list:
            samples.append({
                "param": name,
                "stratum": stratum,
                "flat_index": int(i),
                "g_batched": float(gb.reshape(-1)[i].item()),
                "g_sequential": float(gs.reshape(-1)[i].item()),
            })

    for name in FD_PARAM_NAMES:
        gb, gs = g_batched.get(name), g_sequential.get(name)
        t = tensors.get(name)
        if gb is None or gs is None or t is None or gb.shape != gs.shape:
            continue
        fb, fs = gb.reshape(-1).float(), gs.reshape(-1).float()
        mag = torch.maximum(fb.abs(), fs.abs())
        mmax = float(mag.max().item()) if mag.numel() else 0.0
        if mmax <= 0.0:
            continue
        eligible = mag >= magnitude_floor_frac * mmax
        diff = (fb - fs).abs()
        masked_diff = torch.where(eligible, diff, torch.full_like(diff, -1.0))
        k = min(n_per_param, int(eligible.sum().item()))
        if k <= 0:
            continue
        if name == "opacities":
            # No disagreement to sample: take the best-conditioned elements.
            _emit(name, torch.topk(torch.where(
                eligible, mag, torch.zeros_like(mag)), k).indices.tolist(),
                "agreement")
            continue
        _emit(name, torch.topk(masked_diff, k).indices.tolist(), "disagreement")
        if name == "means":
            # Agreement control on a parameter that DOES carry the defect
            # elsewhere: the lowest-disagreement, well-conditioned elements.
            low = torch.where(eligible, diff,
                              torch.full_like(diff, float("inf")))
            _emit(name, torch.topk(-low, k).indices.tolist(), "agreement")
    return samples


def run_fd_bisect(render_fns: dict[str, Any], tensors, g_batched, g_sequential,
                  *, n_per_param: int, eps_fracs, repeats: int,
                  eps_rel_tol: float, smoothness_tol: float,
                  noise_signal_tol: float, accept_rel_tol: float,
                  path_rel_tol: float, max_seconds: float,
                  eps_absolute=None) -> dict[str, Any]:
    """Full FD bisect for one configuration.

    ``render_fns`` must hold BOTH closures (``batched`` and ``sequential``),
    each returning a single tensor. The entire stencil runs independently
    through each, and the two FD references are compared -- path independence
    is MEASURED here, never assumed from forward parity at some tolerance.

    ``eps_fracs`` are fractions of each parameter tensor's own characteristic
    magnitude (mean |value|), because means (~1m), scales (~0.02) and opacities
    (~0.5) live on completely different scales. ``eps_absolute`` overrides that
    heuristic with literal eps values -- used by the self-test, whose planted
    discontinuity needs every swept eps to straddle the step.
    """
    t_start = time.perf_counter()
    block: dict[str, Any] = {
        "available": True,
        "n_per_param": n_per_param,
        "eps_fracs": list(eps_fracs) if eps_absolute is None else None,
        "eps_absolute": list(eps_absolute) if eps_absolute is not None else None,
        "repeats": repeats,
        "tolerances": {
            "eps_rel_tol": eps_rel_tol, "smoothness_tol": smoothness_tol,
            "noise_signal_tol": noise_signal_tol,
            "accept_rel_tol": accept_rel_tol, "path_rel_tol": path_rel_tol,
        },
        "max_seconds": max_seconds,
    }
    samples = select_fd_samples(g_batched, g_sequential, tensors, n_per_param)
    block["n_sampled"] = len(samples)

    elements: list[dict[str, Any]] = []
    budget_exhausted = False
    for s in samples:
        if time.perf_counter() - t_start > max_seconds:
            budget_exhausted = True
            break
        t = tensors[s["param"]]
        if eps_absolute is not None:
            eps_values = [float(e) for e in eps_absolute]
        else:
            scale = float(t.detach().abs().mean().item())
            if scale <= 0.0:
                scale = 1.0
            eps_values = [float(f) * scale for f in eps_fracs]

        per_closure: dict[str, Any] = {}
        for closure_name, fn in render_fns.items():
            stencil = fd_stencil_element(fn, t, s["flat_index"], eps_values,
                                         repeats=repeats)
            per_closure[closure_name] = fd_apply_gates(
                stencil, eps_rel_tol=eps_rel_tol,
                smoothness_tol=smoothness_tol,
                noise_signal_tol=noise_signal_tol)

        primary = per_closure.get("batched") or next(iter(per_closure.values()))
        other = per_closure.get("sequential")
        failed = list(primary.get("failed_gates") or [])
        if other is not None:
            failed += [f"{g}@sequential_closure"
                       for g in (other.get("failed_gates") or [])
                       if g not in failed]

        path_rel = None
        if (other is not None and primary.get("g_fd") is not None
                and other.get("g_fd") is not None):
            denom = max(abs(primary["g_fd"]), abs(other["g_fd"]))
            if denom > 0.0:
                path_rel = abs(primary["g_fd"] - other["g_fd"]) / denom
                if path_rel > path_rel_tol and "path_independence" not in failed:
                    failed.append("path_independence")

        elements.append({
            **s,
            "eps_values": eps_values,
            "g_fd": primary.get("g_fd"),
            "g_fd_sequential_closure": (
                None if other is None else other.get("g_fd")),
            "fd_path_rel_diff": path_rel,
            "failed_gates": failed,
            "resolved": not failed,
            "closures": per_closure,
        })

    block["budget_exhausted"] = budget_exhausted
    block["n_evaluated"] = len(elements)
    block["elapsed_s"] = time.perf_counter() - t_start
    block["elements"] = elements

    path_rels = [e["fd_path_rel_diff"] for e in elements
                 if e.get("fd_path_rel_diff") is not None]
    n_path_fail = sum(1 for e in elements
                      if "path_independence" in (e.get("failed_gates") or []))
    block["fd_path_independence"] = {
        "n_compared": len(path_rels),
        "n_failed": n_path_fail,
        "failed_frac": (n_path_fail / len(elements)) if elements else None,
        "max_rel_diff": max(path_rels) if path_rels else None,
        "median_rel_diff": _percentile_py(path_rels, 50.0) if path_rels else None,
        "tolerance": path_rel_tol,
        "holds": (bool(n_path_fail == 0) if elements else None),
    }
    block["verdict"] = fd_verdict(elements, accept_rel_tol=accept_rel_tol)
    return block


# --------------------------------------------------------------------------- #
# one configuration
# --------------------------------------------------------------------------- #
def run_config(rasterization, n: int, c: int, color_mode: str, sparse_grad: bool,
               args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "n_gaussians": n,
        "n_cameras": c,
        "color_mode": color_mode,
        "sparse_grad": sparse_grad,
        "width": args.width,
        "height": args.height,
        "sh_degree": args.sh_degree,
        "raster_kwargs": {k: v for k, v in RASTER_KWARGS.items()},
        "status": "pending",
    }

    try:
        pop = make_population(n, args.sh_degree, device, args.seed)
        viewmats, ks = make_cameras(c, args.width, args.height, device)
        colors = None
        if color_mode == "precomputed":
            colors = precomputed_colors(pop, c).detach().requires_grad_(True)
    except Exception as exc:  # noqa: BLE001
        cfg["status"] = "setup_failed"
        cfg["error_type"] = type(exc).__name__
        cfg["error"] = str(exc)[:600]
        cfg["traceback"] = traceback.format_exc()[-2000:]
        return cfg

    def _batched():
        return render_batched(rasterization, pop, color_mode, colors,
                              args.sh_degree, viewmats, ks,
                              args.width, args.height, sparse_grad)

    def _sequential():
        return render_sequential(rasterization, pop, color_mode, colors,
                                 args.sh_degree, viewmats, ks,
                                 args.width, args.height, sparse_grad)

    # ---- 1. forward correctness ------------------------------------------- #
    try:
        with torch.no_grad():
            b_out = _batched()
            s_out = torch.cat(_sequential(), dim=0)
        cfg["batched_output_shape"] = list(b_out.shape)
        cfg["sequential_output_shape"] = list(s_out.shape)
        cfg["forward"] = compare_tensors(b_out, s_out, args.forward_tol,
                                         args.forward_rel_tol)
        cfg["batched_supported"] = True
        del b_out, s_out
    except Exception as exc:  # noqa: BLE001
        # THIS IS A VALID OUTCOME: gsplat on this build refuses C>1 (or OOMs).
        cfg["status"] = "batched_forward_unsupported"
        cfg["batched_supported"] = False
        cfg["error_type"] = type(exc).__name__
        cfg["error"] = str(exc)[:600]
        cfg["oom"] = "out of memory" in str(exc).lower()
        cfg["traceback"] = traceback.format_exc()[-2000:]
        return cfg

    # ---- 2. backward correctness ------------------------------------------ #
    # Both paths use a plain sum() over every pixel of every view, so the two
    # losses are numerically the same scalar and the gradients are directly
    # comparable (a mean over C*H*W vs per-view means differs by exactly C).
    try:
        zero_grads(pop, colors)
        _batched().sum().backward()
        _sync()
        g_batched = grab_grads(pop, colors)

        zero_grads(pop, colors)
        total = None
        for out in _sequential():
            s = out.sum()
            total = s if total is None else total + s
        total.backward()
        _sync()
        g_seq = grab_grads(pop, colors)

        cfg["backward"] = {
            name: compare_tensors(g_batched.get(name), g_seq.get(name),
                                  args.grad_tol, args.grad_rel_tol)
            for name in PARAM_NAMES
        }
        cfg["backward"]["all_within_tolerance"] = all(
            v.get("within_tolerance") is True for v in
            (cfg["backward"][k] for k in PARAM_NAMES)
        )

        # ---- 2b. SELF-CONSISTENCY CONTROL (issue #917) ---------------------- #
        # A "batched != sequential" verdict is only evidence of a defect if the
        # SAME path compared against ITSELF agrees. Without this control the
        # probe measures a difference with no knowledge of the noise floor that
        # difference has to clear -- and the dense (sparse_grad=False) backward
        # accumulates with atomicAdd, whose float32 summation order is not
        # guaranteed to repeat between two launches of the same kernel.
        #
        # Repeat each path once more, under the identical criterion and
        # tolerances, and record both the per-path verdict and whether the
        # cross-path disagreement actually exceeds the same-path disagreement.
        #
        # This runs in its OWN nested try/except: a failure here (e.g. OOM on
        # the extra backward) must not destroy the primary
        # batched-vs-sequential result already computed above.
        try:
            zero_grads(pop, colors)
            _batched().sum().backward()
            _sync()
            g_batched_2 = grab_grads(pop, colors)

            zero_grads(pop, colors)
            total2 = None
            for out in _sequential():
                s = out.sum()
                total2 = s if total2 is None else total2 + s
            total2.backward()
            _sync()
            g_seq_2 = grab_grads(pop, colors)

            self_batched: dict[str, Any] = {
                name: compare_tensors(g_batched.get(name), g_batched_2.get(name),
                                      args.grad_tol, args.grad_rel_tol)
                for name in PARAM_NAMES
            }
            self_batched["all_within_tolerance"] = all(
                self_batched[k].get("within_tolerance") is True
                for k in PARAM_NAMES
            )
            self_batched["bit_exact"] = all(
                self_batched[k].get("max_abs_diff") == 0.0 for k in PARAM_NAMES
            )
            self_sequential: dict[str, Any] = {
                name: compare_tensors(g_seq.get(name), g_seq_2.get(name),
                                      args.grad_tol, args.grad_rel_tol)
                for name in PARAM_NAMES
            }
            self_sequential["all_within_tolerance"] = all(
                self_sequential[k].get("within_tolerance") is True
                for k in PARAM_NAMES
            )
            self_sequential["bit_exact"] = all(
                self_sequential[k].get("max_abs_diff") == 0.0
                for k in PARAM_NAMES
            )

            exceeds, exceeded = noise_floor_comparison(
                cfg["backward"], self_batched, self_sequential)
            cfg["backward_self_consistency"] = {
                "available": True,
                "batched": self_batched,
                "sequential": self_sequential,
                "noise_floor_comparison": exceeds,
                "backward_noise_floor_exceeded": exceeded,
            }

            del g_batched_2, g_seq_2
        except Exception as exc:  # noqa: BLE001
            cfg["backward_self_consistency"] = {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:600],
                "oom": "out of memory" in str(exc).lower(),
                "traceback": traceback.format_exc()[-2000:],
            }

        # ---- 2c. FINITE-DIFFERENCE BISECT (issue #917), opt-in ------------- #
        # Which path is RIGHT cannot come from comparing the two paths to each
        # other. This computes an independent reference through the FORWARD
        # only. Default-off: with the flag unset this whole block is skipped and
        # the configuration's result dict is byte-identical to before it existed.
        #
        # Cost gate: by default the bisect runs only where the question exists
        # -- C > 1 AND the cross-path backward parity actually failed. That also
        # drops the sparse_grad=True control arm for free. The AGREEMENT control
        # survives the gate because it is drawn from `opacities` (and the
        # lowest-disagreement `means`) WITHIN those same cells.
        if args.fd_bisect:
            try:
                parity_failed = cfg["backward"].get(
                    "all_within_tolerance") is not True
                if not (args.fd_all_cells or (c > 1 and parity_failed)):
                    cfg["fd_bisect"] = {
                        "available": False,
                        "skipped": True,
                        "reason": ("cell is not C>1-with-failing-backward-parity; "
                                   "pass --fd-all-cells to force"),
                        "n_cameras": c, "backward_parity_failed": parity_failed,
                    }
                else:
                    fd_tensors = {name: pop[name] for name in FD_PARAM_NAMES}

                    def _fd_batched():
                        return _batched()

                    def _fd_sequential():
                        return torch.cat(_sequential(), dim=0)

                    cfg["fd_bisect"] = run_fd_bisect(
                        {"batched": _fd_batched, "sequential": _fd_sequential},
                        fd_tensors, g_batched, g_seq,
                        n_per_param=args.fd_elements_per_param,
                        eps_fracs=args.fd_eps_fracs,
                        repeats=args.fd_repeats,
                        eps_rel_tol=args.fd_eps_rel_tol,
                        smoothness_tol=args.fd_smoothness_tol,
                        noise_signal_tol=args.fd_noise_signal_tol,
                        accept_rel_tol=args.fd_accept_rel_tol,
                        path_rel_tol=args.fd_path_rel_tol,
                        max_seconds=args.fd_max_seconds,
                    )
            except Exception as exc:  # noqa: BLE001
                cfg["fd_bisect"] = {
                    "available": False, "skipped": False,
                    "error_type": type(exc).__name__, "error": str(exc)[:600],
                    "oom": "out of memory" in str(exc).lower(),
                    "traceback": traceback.format_exc()[-2000:],
                }

        zero_grads(pop, colors)
        del g_batched, g_seq
    except Exception as exc:  # noqa: BLE001
        cfg["backward"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:600],
            "oom": "out of memory" in str(exc).lower(),
            "traceback": traceback.format_exc()[-2000:],
        }

    # ---- 3. memory scaling ------------------------------------------------- #
    # Three peaks, because "batched holds C images, one sequential render holds
    # one" is a tautology unless the retained-graph sequential shape (the one a
    # real batched multi-keyframe LOSS takes) is also measured.
    def _mem_batched():
        zero_grads(pop, colors)
        _batched().sum().backward()
        zero_grads(pop, colors)

    def _mem_seq_retained():
        zero_grads(pop, colors)
        total = None
        for out in _sequential():
            s = out.sum()
            total = s if total is None else total + s
        total.backward()
        zero_grads(pop, colors)

    def _mem_seq_streamed():
        zero_grads(pop, colors)
        for i in range(c):
            if color_mode == "sh":
                colors_arg, sh_arg = pop["sh"], args.sh_degree
            else:
                colors_arg, sh_arg = colors[i:i + 1], None
            out, _a, _m = _call(rasterization, pop, colors_arg, sh_arg,
                                viewmats[i:i + 1].contiguous(),
                                ks[i:i + 1].contiguous(),
                                args.width, args.height, sparse_grad)
            out.sum().backward()
            del out
        zero_grads(pop, colors)

    cfg["memory"] = {
        "batched": _measure_peak(_mem_batched),
        "sequential_retained": _measure_peak(_mem_seq_retained),
        "sequential_streamed": _measure_peak(_mem_seq_streamed),
    }

    # ---- 4. runtime -------------------------------------------------------- #
    def _fwd_batched():
        with torch.no_grad():
            _batched()

    def _fwd_seq():
        with torch.no_grad():
            _sequential()

    runtime = {
        "forward_batched": time_it(_fwd_batched, args.warmup, args.iters),
        "forward_sequential": time_it(_fwd_seq, args.warmup, args.iters),
        "fwd_bwd_batched": time_it(_mem_batched, args.warmup, args.iters),
        "fwd_bwd_sequential_retained": time_it(_mem_seq_retained, args.warmup, args.iters),
        "fwd_bwd_sequential_streamed": time_it(_mem_seq_streamed, args.warmup, args.iters),
    }
    for key, entry in list(runtime.items()):
        if entry.get("ok"):
            entry["ms_per_view"] = entry["mean_ms"] / max(c, 1)

    # Paired timing: fwd_bwd_sequential_streamed (reference, current EliGSiR)
    # vs fwd_bwd_batched (candidate). This is the PRIMARY comparator for #106.
    runtime["fwd_bwd_paired_streamed_vs_batched"] = time_paired(
        _mem_seq_streamed, _mem_batched, args.warmup, args.paired_iters
    )

    # All three speedup ratios with unambiguous naming.
    # Issue #898: sequential_streamed is the REFERENCE (current EliGSiR shape),
    # batched is the CANDIDATE. Use shared helper to ensure direction.
    runtime["comparisons"] = _build_all_comparisons(runtime)

    # Legacy speedup_batched_over for backward compatibility (deprecated key name)
    for pair in (("fwd_bwd_batched", "fwd_bwd_sequential_retained"),
                 ("fwd_bwd_batched", "fwd_bwd_sequential_streamed")):
        a, b = runtime.get(pair[0]), runtime.get(pair[1])
        if a and b and a.get("ok") and b.get("ok") and a["mean_ms"] > 0:
            runtime.setdefault("speedup_batched_over", {})[pair[1]] = b["mean_ms"] / a["mean_ms"]

    cfg["runtime"] = runtime

    cfg["status"] = "ok"
    zero_grads(pop, colors)
    del pop, colors, viewmats, ks
    _reset_peak()
    return cfg


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def print_table(results: list[dict[str, Any]]) -> None:
    hdr = (f"{'N':>8} {'C':>3} {'colors':>11} {'sparse':>6} {'status':>28} "
           f"{'fwd_ok':>6} {'bwd_ok':>6} {'selfok':>6} {'>noise':>6} "
           f"{'strm_ms':>9} {'bat_ms':>9} {'spdup':>8} "
           f"{'strm_MB':>9} {'bat_MB':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    print("PRIMARY COMPARATOR: fwd_bwd_sequential_streamed (current EliGSiR) "
          "vs fwd_bwd_batched")
    print("-" * len(hdr))

    for r in results:
        fwd = r.get("forward", {})
        bwd = r.get("backward", {})
        rt = r.get("runtime", {})
        mem = r.get("memory", {})
        bat = rt.get("fwd_bwd_batched", {})
        strm = rt.get("fwd_bwd_sequential_streamed", {})
        comparisons = rt.get("comparisons", {})
        speed = comparisons.get("fwd_bwd_sequential_streamed_vs_fwd_bwd_batched", {}).get(
            "speedup_reference_over_candidate")

        def _f(v, fmt="{:9.2f}"):
            return fmt.format(v) if isinstance(v, (int, float)) else f"{'-':>9}"

        # Issue #917 self-consistency control columns.
        _sc = r.get("backward_self_consistency") or {}
        if _sc.get("available") is True:
            _self_ok = (
                (_sc.get("batched") or {}).get("all_within_tolerance") is True
                and (_sc.get("sequential") or {}).get("all_within_tolerance") is True
            )
            _exceeds = _sc.get("backward_noise_floor_exceeded")
            if _exceeds is None:
                _exceeds = "-"
        else:
            _self_ok = "-"
            _exceeds = "-"

        print(f"{r['n_gaussians']:>8} {r['n_cameras']:>3} "
              f"{r['color_mode']:>11} {str(r['sparse_grad']):>6} "
              f"{r.get('status', '?')[:28]:>28} "
              f"{str(fwd.get('within_tolerance', '-')):>6} "
              f"{str(bwd.get('all_within_tolerance', '-')):>6} "
              f"{str(_self_ok):>6} {str(_exceeds):>6} "
              f"{_f(strm.get('mean_ms'))} {_f(bat.get('mean_ms'))} "
              f"{(f'{speed:8.2f}x' if isinstance(speed, float) else f'{chr(45):>8}')} "
              f"{_f((mem.get('sequential_streamed') or {}).get('peak_mb'))} "
              f"{_f((mem.get('batched') or {}).get('peak_mb'))}")
    print()

    # Print secondary comparisons
    print("SECONDARY COMPARISONS:")
    print("  - fwd_bwd_batched vs fwd_bwd_sequential_retained (hypothetical one-backward shape)")
    print("  - fwd_bwd_sequential_retained vs fwd_bwd_sequential_streamed (single vs per-frame bwd)")
    print()

    unsupported = [r for r in results if r.get("batched_supported") is False]
    if unsupported:
        print("Configurations where the BATCHED (C>1) call did not run -- this is a "
              "valid probe outcome, not a crash:")
        for r in unsupported:
            print(f"  N={r['n_gaussians']} C={r['n_cameras']} "
                  f"colors={r['color_mode']} sparse_grad={r['sparse_grad']}: "
                  f"{r.get('error_type')}: {str(r.get('error'))[:200]}")
        print()


def fd_bisect_summary(ok: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll the per-cell #917 FD bisect up across configurations.

    Reports the per-parameter verdict SET, not a single collapsed answer: a
    parameter answering `batched` on some cells and `sequential` on others is
    itself the finding, and collapsing it would hide exactly that. An empty
    block (the default, flag unset) carries `ran: False` rather than absent
    keys, so a reader can tell "not run" from "ran and found nothing".
    """
    cells = [r for r in ok if isinstance(r.get("fd_bisect"), dict)]
    ran = [r for r in cells if r["fd_bisect"].get("available") is True]
    out: dict[str, Any] = {
        "fd_bisect_ran": bool(ran),
        "fd_bisect_cells": len(ran),
        "fd_bisect_cells_failed": sorted(
            f"N={r['n_gaussians']},C={r['n_cameras']},"
            f"colors={r['color_mode']},sparse_grad={r['sparse_grad']}"
            for r in cells
            if r["fd_bisect"].get("available") is not True
            and not r["fd_bisect"].get("skipped")),
    }
    if not ran:
        return out
    verdicts: dict[str, list[str]] = {}
    unresolved: dict[str, int] = {}
    path_fail_cells = []
    for r in ran:
        blk = r["fd_bisect"]
        pi = blk.get("fd_path_independence") or {}
        if pi.get("holds") is False:
            path_fail_cells.append(
                f"N={r['n_gaussians']},C={r['n_cameras']},"
                f"colors={r['color_mode']},sparse_grad={r['sparse_grad']}")
        for name, v in (blk.get("verdict") or {}).items():
            if not isinstance(v, dict):
                continue
            verdicts.setdefault(name, []).append(v.get("verdict"))
            for gate, cnt in (v.get("unresolved_by_gate") or {}).items():
                unresolved[gate] = unresolved.get(gate, 0) + cnt
    out["fd_bisect_verdicts"] = {
        name: sorted({v for v in vs if v}) for name, vs in verdicts.items()}
    out["fd_bisect_unresolved_by_gate"] = unresolved
    out["fd_bisect_path_independence_failed_cells"] = sorted(path_fail_cells)
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    multi = [r for r in results if r["n_cameras"] > 1]
    ok = [r for r in multi if r.get("status") == "ok"]

    # Collect all speedup ratios (do not collapse them into a single number)
    speedups_streamed_vs_batched = []
    speedups_retained_vs_batched = []
    for r in ok:
        comparisons = r.get("runtime", {}).get("comparisons", {})
        s1 = comparisons.get("fwd_bwd_sequential_streamed_vs_fwd_bwd_batched", {}).get(
            "speedup_reference_over_candidate")
        s2 = comparisons.get("fwd_bwd_sequential_retained_vs_fwd_bwd_batched", {}).get(
            "speedup_reference_over_candidate")
        if s1:
            speedups_streamed_vs_batched.append(s1)
        if s2:
            speedups_retained_vs_batched.append(s2)

    def _range_str(vals):
        if not vals:
            return None
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals),
                "count": len(vals)}

    return {
        "n_configurations": len(results),
        "n_multi_camera_configurations": len(multi),
        "n_multi_camera_ran": len(ok),
        "batched_c_gt_1_expressible": bool(ok),
        "forward_matches_sequential": all(
            (r.get("forward") or {}).get("within_tolerance") is True for r in ok
        ) if ok else None,
        "backward_matches_sequential": all(
            (r.get("backward") or {}).get("all_within_tolerance") is True for r in ok
        ) if ok else None,
        # Issue #917: a backward-parity FAILURE is only evidence of a defect if
        # the same path compared against itself AGREES. Read these together --
        # `backward_matches_sequential: false` with
        # `backward_self_consistency_clean: false` is an accumulation-order
        # noise floor, not a gradient bug.
        "backward_self_consistency_clean": all(
            ((r.get("backward_self_consistency") or {}).get("batched") or {})
            .get("all_within_tolerance") is True
            and ((r.get("backward_self_consistency") or {}).get("sequential") or {})
            .get("all_within_tolerance") is True
            for r in ok
        ) if ok else None,
        "backward_disagreement_exceeds_noise_floor": sorted(
            f"N={r['n_gaussians']},C={r['n_cameras']},"
            f"colors={r['color_mode']},sparse_grad={r['sparse_grad']}"
            for r in ok
            if (r.get("backward_self_consistency") or {}).get(
                "backward_noise_floor_exceeded") is True
        ),
        # Issue #920: the cell list above says WHERE, this says WHICH parameters
        # carry the unexplained disagreement. A defect localized to a subset of
        # parameters is the normal shape of a real one (#898: means/quats/scales
        # and never opacities/colors), so the subset is the finding, not a
        # reason to discount the cell.
        "backward_unexplained_disagreement_params": sorted({
            name
            for r in ok
            for name, v in (
                ((r.get("backward_self_consistency") or {})
                 .get("noise_floor_comparison") or {}).items()
            )
            if isinstance(v, dict) and v.get("disagreement_unexplained") is True
        }),
        "backward_self_consistency_unavailable": sorted(
            f"N={r['n_gaussians']},C={r['n_cameras']},"
            f"colors={r['color_mode']},sparse_grad={r['sparse_grad']}"
            for r in ok
            if (r.get("backward_self_consistency") or {}).get(
                "available") is not True
        ),
        "unsupported_error_types": sorted({
            r.get("error_type") for r in multi
            if r.get("batched_supported") is False and r.get("error_type")
        }),
        **fd_bisect_summary(ok),
        "speedup_summary": {
            "note": "PRIMARY COMPARATOR: batched / sequential_streamed (current EliGSiR)",
            "batched_over_sequential_streamed": _range_str(speedups_streamed_vs_batched),
            "note2": "SECONDARY: batched / sequential_retained (hypothetical single-backward)",
            "batched_over_sequential_retained": _range_str(speedups_retained_vs_batched),
        },
    }


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="gsplat true camera-batch (C>1) ROCm feasibility probe "
                    "(GitHub issue #106, #614 ledger item 2). Fully synthetic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default="output/issue106_camera_batch_probe",
                   help="Directory for probe_results.json (created if absent).")
    p.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000],
                   help="Gaussian population sizes. Add 500000 for the large arm.")
    p.add_argument("--cameras", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="Camera-batch sizes C to probe.")
    p.add_argument("--color-modes", nargs="+", default=["sh", "precomputed"],
                   choices=["sh", "precomputed"],
                   help="'sh' = shared SH coeffs (kernel does per-camera SH, "
                        "sh_impl=gsplat); 'precomputed' = per-camera (C,N,3) "
                        "colors (repo-default sh_impl=python).")
    p.add_argument("--sparse-grad", nargs="+", default=["false", "true"],
                   choices=["false", "true"],
                   help="gsplat sparse_grad values to probe. 'true' is what the "
                        "real training loop uses under --parallelism_profile safe.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--sh-degree", type=int, default=3,
                   help="SH degree for the 'sh' color mode ((deg+1)^2 coeffs).")
    p.add_argument(
        "--rasterize-mode",
        choices=("classic", "antialiased"),
        default="classic",
        help="gsplat rasterizer mode. Use matched runs for the live AA A/B.",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--paired-iters", type=int, default=30,
                   help="Sample count for paired ABBA timing of fwd_bwd_sequential_streamed "
                        "vs fwd_bwd_batched (the primary shipping-reference comparator).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forward-tol", type=float, default=1e-4,
                   help="ABSOLUTE tolerance for the batched-vs-sequential image "
                        "comparison. Reported, never asserted.")
    p.add_argument("--forward-rel-tol", type=float, default=1e-4,
                   help="RELATIVE tolerance for the image comparison. The "
                        "elementwise verdict passes for |a_i - b_i| <= atol + "
                        "rtol * |b_i|.")
    p.add_argument("--grad-tol", type=float, default=1e-3,
                   help="ABSOLUTE tolerance for the per-parameter gradient "
                        "comparison. Reported, never asserted.")
    p.add_argument("--grad-rel-tol", type=float, default=1e-4,
                   help="RELATIVE tolerance for the gradient comparison. The "
                        "elementwise verdict passes for |a_i - b_i| <= atol + "
                        "rtol * |b_i|, which subsumes the old global-scale "
                        "rule and eliminates 31/32 spurious backward failures "
                        "from the first GPU run (see compare_tensors' docstring).")
    p.add_argument("--git-commit", default=None,
                   help="Override the recorded git commit (git rev-parse fails "
                        "inside the container when the repo is a worktree).")
    # ---- issue #917 finite-difference bisect (default OFF) ----------------- #
    p.add_argument("--fd-bisect", action="store_true",
                   help="Compute an independent FINITE-DIFFERENCE gradient "
                        "reference through the FORWARD only and report, per "
                        "parameter, which backward path it agrees with "
                        "(issue #917). Diagnostics only; changes nothing about "
                        "the existing forward/backward/timing/memory results. "
                        "Off by default -- with it unset the results are "
                        "byte-identical to before this existed.")
    p.add_argument("--fd-all-cells", action="store_true",
                   help="Run the FD bisect on EVERY configuration instead of "
                        "only C>1 cells whose backward parity failed. Much "
                        "more expensive; the default gate is where the #917 "
                        "question actually lives.")
    p.add_argument("--fd-elements-per-param", type=int, default=6,
                   help="Sampled scalar elements per parameter per stratum. A "
                        "full sweep over 500k Gaussians is absurd; 100-300 "
                        "elements total is plenty for a ~0.4-1.0 relative "
                        "discrimination.")
    p.add_argument("--fd-eps-fracs", type=float, nargs="+",
                   default=[3e-3, 1e-2],
                   help="eps values as FRACTIONS of each parameter tensor's own "
                        "mean |value| (means ~1m, scales ~0.02 and opacities "
                        "~0.5 cannot share an absolute eps). Two values is the "
                        "minimum for the eps-agreement gate.")
    p.add_argument("--fd-repeats", type=int, default=2,
                   help="Times the whole stencil is re-evaluated at EACH eps. "
                        "This is what measures the forward's own noise, which "
                        "the FD estimate amplifies by 1/(2*eps). Must be >= 2 "
                        "for the self-consistency gate to have any content.")
    p.add_argument("--fd-eps-rel-tol", type=float, default=0.10,
                   help="Relative floor for the eps-agreement gate. The gate "
                        "passes when the eps-to-eps spread is within EITHER the "
                        "measured repeat noise (times 3) or this fraction of "
                        "the estimate.")
    p.add_argument("--fd-smoothness-tol", type=float, default=0.10,
                   help="Max |S(+)-2S(0)+S(-)| / |S(+)-S(-)|. The rasterizer is "
                        "piecewise; this rejects elements on a discontinuity "
                        "(tile boundary, radius_clip, near_plane).")
    p.add_argument("--fd-noise-signal-tol", type=float, default=0.20,
                   help="Max repeat-to-repeat FD noise as a fraction of the FD "
                        "estimate. Above this the element's estimate is not "
                        "distinguishable from forward nondeterminism.")
    p.add_argument("--fd-accept-rel-tol", type=float, default=0.20,
                   help="A backward path 'agrees with' FD when its median "
                        "relative error is within this. 0.2 is the "
                        "discriminating range against the measured 0.373-1.02 "
                        "cross-path disagreement.")
    p.add_argument("--fd-path-rel-tol", type=float, default=0.10,
                   help="Max relative disagreement between the FD reference "
                        "computed through the batched closure and through the "
                        "sequential one. Path independence is MEASURED, not "
                        "assumed from forward parity.")
    p.add_argument("--fd-max-seconds", type=float, default=180.0,
                   help="Per-configuration wall-clock budget for the FD bisect. "
                        "Sampling stops when exceeded and the block records "
                        "budget_exhausted.")
    p.add_argument("--experiment-origin-issue", "--experiment_origin_issue",
                   dest="experiment_origin_issue", default=None,
                   help="Provenance only: the GitHub issue this probe run was "
                        "queued for. Recorded verbatim in the output JSON's "
                        "args block; changes no behavior. Accepted so a spec "
                        "file can carry the same provenance flag train.py jobs "
                        "do (the probe job type passes extra_args to THIS "
                        "script, so an unknown flag would exit 2 at launch).")
    p.add_argument("--self-test", action="store_true",
                   help="CPU-only harness check: exercises population/camera "
                        "construction, the sparse-COO-aware comparison, the "
                        "timing wrapper and the table renderer, then exits. "
                        "Touches no GPU and calls no gsplat kernel.")
    return p


def self_test() -> int:
    """CPU-only mechanical check of every non-kernel helper in this file.

    Deliberately does NOT call ``rasterization`` -- that needs a GPU. This only
    proves the harness plumbing is sound, never anything about batching.
    """
    dev = torch.device("cpu")
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}")
        if not cond:
            failures.append(name)

    print("[self-test] population / cameras")
    pop = make_population(1000, 3, dev, seed=0)
    check("population keys", set(pop) == {"means", "quats", "scales", "opacities", "sh"})
    check("means shape (N,3)", tuple(pop["means"].shape) == (1000, 3))
    check("sh shape (N,(deg+1)^2,3)", tuple(pop["sh"].shape) == (1000, 16, 3))
    check("all params require grad", all(t.requires_grad for t in pop.values()))
    check("quats normalized", bool((pop["quats"].norm(dim=-1) - 1).abs().max() < 1e-5))

    viewmats, ks = make_cameras(4, 640, 480, dev)
    check("viewmats (C,4,4)", tuple(viewmats.shape) == (4, 4, 4))
    check("Ks (C,3,3)", tuple(ks.shape) == (4, 3, 3))
    dets = torch.linalg.det(viewmats[:, :3, :3])
    check("viewmat rotations are proper (det=+1)", bool((dets - 1).abs().max() < 1e-4))
    check("viewmats differ per camera",
          bool((viewmats[0] - viewmats[1]).abs().max() > 1e-3))

    colors = precomputed_colors(pop, 4)
    check("precomputed colors (C,N,3)", tuple(colors.shape) == (4, 1000, 3))
    check("precomputed color slices differ",
          bool((colors[0] - colors[1]).abs().max() > 0))

    print("[self-test] comparison")
    a = torch.randn(64)
    check("identical tensors within tolerance",
          compare_tensors(a, a.clone(), 1e-6)["within_tolerance"] is True)
    check("shifted tensors outside tolerance",
          compare_tensors(a + 1.0, a, 1e-6)["within_tolerance"] is False)
    check("NaN is never 'within tolerance'",
          compare_tensors(a * float("nan"), a, 1e9)["within_tolerance"] is False)
    check("shape mismatch reported unavailable",
          compare_tensors(a, torch.randn(63), 1e-6)["available"] is False)
    check("None reported unavailable",
          compare_tensors(None, a, 1e-6)["available"] is False)
    sp = torch.sparse_coo_tensor(torch.tensor([[0, 2]]), torch.tensor([1.0, 2.0]), (5,))
    check("sparse COO grad densified before compare",
          compare_tensors(sp, sp.to_dense(), 1e-9)["within_tolerance"] is True)

    # Regression cases for the absolute-only verdict that produced 31/32
    # spurious backward failures on the first GPU run. Numbers are the real
    # observed ones: a 'scales' gradient of magnitude ~6.6e3 differing by
    # ~1.5e-3 is float32 round-off, not a kernel defect.
    big = torch.full((16,), 6590.0)
    noisy = big.clone()
    noisy[0] += 1.465e-3
    r = compare_tensors(noisy, big, tol=1e-3, rel_tol=1e-4)
    check("float32 round-off on a 1e4-magnitude grad passes elementwise bound",
          r["within_tolerance"] is True)
    check("...and the elementwise bound DOES account for reference magnitude",
          r["max_rel_diff_nonneg"] is not None and r["max_rel_diff_nonneg"] < 1e-4)

    # Test small-entry corruption case: the old global-scale rule hid this
    tiny = torch.full((16,), 6590.0)
    tiny[3] = 1e-5
    corrupted = tiny.clone()
    corrupted[3] += 0.5  # corrupt the tiny entry
    r_corrupt = compare_tensors(corrupted, tiny, tol=1e-3, rel_tol=1e-4)
    check("small-entry corruption FAILS elementwise bound (catches old false negative)",
          r_corrupt["within_tolerance"] is False)
    check("...but old global-scale rule PASSED (demonstrating the regression)",
          r_corrupt["legacy_maxabs_over_refmax"] is not None and
          r_corrupt["legacy_maxabs_over_refmax"] <= 1e-4)

    wrong = big * 1.10  # a genuinely wrong gradient, 10% off
    r2 = compare_tensors(wrong, big, tol=1e-3, rel_tol=1e-4)
    check("a 10%-wrong gradient still fails elementwise bound",
          r2["within_tolerance"] is False)

    nan_big = big.clone()
    nan_big[0] = float("nan")
    check("NaN fails even when bounds would pass",
          compare_tensors(nan_big, big, tol=1e9, rel_tol=1e9)["within_tolerance"] is False)

    inf_big = big.clone()
    inf_big[0] = float("inf")
    check("Infinity fails even when bounds would pass",
          compare_tensors(inf_big, big, tol=1e9, rel_tol=1e9)["within_tolerance"] is False)

    zero_ref = torch.zeros((16,))
    check("all-zero reference returns None for relative stats",
          compare_tensors(zero_ref, zero_ref, 1e-3)["rel_l2_error"] is None)

    print("[self-test] grad plumbing")
    zero_grads(pop, colors)
    check("zero_grads clears sh grad", pop["sh"].grad is None)
    (pop["means"].sum() + pop["sh"].sum()).backward()
    grads = grab_grads(pop, None)
    check("grab_grads returns every param name", set(grads) == set(PARAM_NAMES))
    check("grab_grads picks up means grad", grads["means"] is not None)
    check("grab_grads falls back to sh for 'colors'",
          grads["colors"] is not None and tuple(grads["colors"].shape) == (1000, 16, 3))

    print("[self-test] noise-floor control (issue #917)")

    def _blk(within=None, **kw):
        """A comparison block. `within` is the per-parameter cross-path parity
        verdict; the same-path blocks never carry one, so it defaults to None
        there and is simply not read."""
        out: dict[str, Any] = {}
        for n in PARAM_NAMES:
            entry: dict[str, Any] = {"max_abs_diff": kw[n]}
            if within is not None:
                entry["within_tolerance"] = (
                    within[n] if isinstance(within, dict) else within)
            out[n] = entry
        return out

    def _cross(diffs, within):
        return _blk(within=within, **diffs)

    _all = {n: 1.0 for n in PARAM_NAMES}

    # Case 1: cross-path difference is 100x the same-path repeat difference on
    # every parameter, and every parameter really fails parity -> a real
    # disagreement, above the noise floor.
    per, overall = noise_floor_comparison(
        _cross(_all, False),
        _blk(**{n: 0.01 for n in PARAM_NAMES}),
        _blk(**{n: 0.01 for n in PARAM_NAMES}))
    check("noise_floor: cross >> self -> exceeded", overall is True)
    check("noise_floor: ratio computed", abs(per["means"]["ratio"] - 100.0) < 1e-9)

    # Case 2: the same path disagrees with ITSELF by as much as the cross-path
    # comparison -> accumulation-order noise, NOT a defect. This is the case
    # the probe could not previously distinguish at all.
    _, overall = noise_floor_comparison(
        _cross(_all, False),
        _blk(**_all),
        _blk(**{n: 0.5 for n in PARAM_NAMES}))
    check("noise_floor: cross == self -> NOT exceeded (strict >)", overall is False)

    # Case 3: the floor is the LARGER of the two same-path numbers, so a quiet
    # batched repeat must not mask a noisy sequential one.
    _, overall = noise_floor_comparison(
        _cross(_all, False),
        _blk(**{n: 0.0 for n in PARAM_NAMES}),
        _blk(**{n: 2.0 for n in PARAM_NAMES}))
    check("noise_floor: floor is max(batched, sequential)", overall is False)

    # Case 4 (issue #920, THE REGRESSION THIS FUNCTION SHIPPED WITH): a defect
    # confined to ONE parameter, with the other four passing parity outright, IS
    # an exceedance. The old `all()` over the ratio returned False here, which is
    # how a 18-of-18 real finding was reported as 2 of 18.
    mixed = _cross(
        {**{n: 0.001 for n in PARAM_NAMES}, "scales": 99.0},
        {**{n: True for n in PARAM_NAMES}, "scales": False})
    per, overall = noise_floor_comparison(
        mixed, _blk(**_all), _blk(**_all))
    check("noise_floor: localized defect IS an overall exceedance (#920)",
          overall is True)
    check("noise_floor: the localized param is the one flagged",
          per["scales"]["disagreement_unexplained"] is True
          and per["means"]["disagreement_unexplained"] is False)

    # Case 5: an unresolvable parameter yields None, never a silent False.
    missing = _cross(_all, False)
    missing["quats"] = {}
    per, overall = noise_floor_comparison(
        missing,
        _blk(**{n: 0.0 for n in PARAM_NAMES}),
        _blk(**{n: 0.0 for n in PARAM_NAMES}))
    check("noise_floor: unresolvable param -> overall None, not False",
          overall is None and per["quats"] is None)

    # Case 6: a zero floor on both sides leaves ratio undefined but still
    # resolves the verdict.
    per, overall = noise_floor_comparison(
        _cross({n: 1e-9 for n in PARAM_NAMES}, False),
        _blk(**{n: 0.0 for n in PARAM_NAMES}),
        _blk(**{n: 0.0 for n in PARAM_NAMES}))
    check("noise_floor: zero floor -> ratio None but verdict True",
          overall is True and per["means"]["ratio"] is None)

    # Case 7 (issue #920, the OTHER failure mode): the `sparse_grad=True`
    # control arm. Ratios wobble to 2x around the floor, but every parameter
    # PASSES cross-path parity -- there is no disagreement to explain -- so the
    # cell must stay False. A bare `any()` over the ratio would report it True
    # and destroy the control.
    _, overall = noise_floor_comparison(
        _cross({n: 2.0 for n in PARAM_NAMES}, True),
        _blk(**_all),
        _blk(**_all))
    check("noise_floor: ratio wobble on a parity-CLEAN cell stays False (#920)",
          overall is False)

    # Case 8 (issue #920): parity failure alone is not enough either -- a
    # parameter that fails parity but sits AT its own noise floor is explained
    # by accumulation order, which is this whole function's reason to exist.
    _, overall = noise_floor_comparison(
        _cross(_all, False),
        _blk(**_all),
        _blk(**_all))
    check("noise_floor: parity failure at the floor stays False (#920)",
          overall is False)

    # Case 9 (issue #920): a missing/non-boolean cross-path parity verdict is
    # unresolvable, not an assumed False.
    no_verdict = _cross(_all, False)
    no_verdict["colors"].pop("within_tolerance")
    _, overall = noise_floor_comparison(
        no_verdict,
        _blk(**{n: 0.0 for n in PARAM_NAMES}),
        _blk(**{n: 0.0 for n in PARAM_NAMES}))
    check("noise_floor: missing within_tolerance -> None, not False (#920)",
          overall is None)

    print("[self-test] FD bisect driver -- happy path (issue #917)")
    # A smooth closed-form stand-in with KNOWN analytic gradients, plus two
    # deliberately planted pathologies. Absolute eps is injected so the planted
    # discontinuity is straddled by every swept eps -- the per-parameter scale
    # heuristic would put eps somewhere unrelated to the step.
    #   [0] 3*x            -> grad 3, exactly linear
    #   [1] sin(x)         -> grad cos(x), smooth and nonlinear
    #   [2] x + 1e6*x^3    -> grad 1 at x=0, but S''=0 EXACTLY (odd function) so
    #                         it PASSES smoothness while the O(eps^2)
    #                         truncation term explodes: 2.0 at eps=1e-3, 10.0 at
    #                         eps=3e-3. This is the eps-sensitivity gate's case.
    #   [3] 2*x + 1000*(x>0.5) at x=0.5+1e-6 -> a discontinuity inside every
    #                         swept eps. Second difference is the whole step.
    fd_eps = [1e-3, 3e-3]
    fd_x = torch.tensor([0.0, 0.3, 0.0, 0.5 + 1e-6], dtype=torch.float32)

    def _fd_clean():
        return torch.stack([
            3.0 * fd_x[0],
            torch.sin(fd_x[1]),
            fd_x[2] + 1.0e6 * fd_x[2] ** 3,
            2.0 * fd_x[3] + torch.where(fd_x[3] > 0.5,
                                        torch.tensor(1000.0),
                                        torch.tensor(0.0)),
        ])

    g_lin = fd_apply_gates(fd_stencil_element(_fd_clean, fd_x, 0, fd_eps))
    check("FD: linear element resolves", g_lin["resolved"] is True)
    check("FD: linear element recovers the analytic gradient (3.0)",
          abs(g_lin["g_fd"] - 3.0) < 1e-3)
    g_sin = fd_apply_gates(fd_stencil_element(_fd_clean, fd_x, 1, fd_eps))
    check("FD: smooth nonlinear element resolves", g_sin["resolved"] is True)
    check("FD: ...and recovers cos(0.3) = 0.9553",
          abs(g_sin["g_fd"] - math.cos(0.3)) < 1e-3)

    print("[self-test] FD bisect driver -- REJECTION paths")
    g_eps = fd_apply_gates(fd_stencil_element(_fd_clean, fd_x, 2, fd_eps))
    check("FD gate (a): eps-sensitive element is UNRESOLVED",
          g_eps["resolved"] is False)
    check("FD gate (a): ...flagged by eps_agreement SPECIFICALLY (it passes "
          "smoothness -- an odd function has S''=0 exactly)",
          g_eps["failed_gates"] == ["eps_agreement"])
    check("FD gate (a): the estimate really does move with eps (2.0 -> 10.0)",
          abs(g_eps["per_eps"][0]["g"] - 2.0) < 0.1
          and abs(g_eps["per_eps"][1]["g"] - 10.0) < 0.5)

    g_disc = fd_apply_gates(fd_stencil_element(_fd_clean, fd_x, 3, fd_eps))
    check("FD gate (b): element on a discontinuity is UNRESOLVED",
          g_disc["resolved"] is False)
    check("FD gate (b): ...flagged by smoothness",
          "smoothness" in g_disc["failed_gates"])
    check("FD gate (b): the second difference IS the whole step",
          g_disc["gates"]["smoothness"]["ratio"] > 0.9)

    # Gate (c): a NOISY forward, mirroring the real substrate (bit_exact is
    # False on 48/48 cells). The stencil sees a different realization on every
    # call, so the repeat spread -- amplified by 1/(2*eps) -- swamps a
    # gradient of 1.0. Nothing else in this file measures that amplification.
    torch.manual_seed(917)
    fd_y = torch.tensor([0.0], dtype=torch.float32)

    def _fd_noisy():
        return torch.stack([fd_y[0] + torch.randn(()) * 1.0e-2])

    g_noise = fd_apply_gates(fd_stencil_element(_fd_noisy, fd_y, 0, fd_eps))
    check("FD gate (c): noisy forward leaves the element UNRESOLVED",
          g_noise["resolved"] is False)
    check("FD gate (c): ...flagged by self_consistency",
          "self_consistency" in g_noise["failed_gates"])
    check("FD gate (c): the repeat noise is recorded, not merely detected",
          g_noise["gates"]["self_consistency"]["noise_rel"] is None
          or g_noise["gates"]["self_consistency"]["noise_rel"] > 0.2)

    print("[self-test] FD bisect driver -- perturbation safety")
    fd_before = fd_x.clone()
    fd_apply_gates(fd_stencil_element(_fd_clean, fd_x, 1, fd_eps))
    check("FD: the perturbed tensor is restored BITWISE",
          bool(torch.equal(fd_x, fd_before)))
    noncontig = torch.zeros((4, 4), dtype=torch.float32).t()
    raised = False
    try:
        _fd_write(noncontig, 1, 1.0)
    except FDPerturbationError:
        raised = True
    except RuntimeError:
        raised = True   # torch's own view() refusal, which is the guard working
    check("FD: a non-contiguous tensor REFUSES the write (a reshape() copy "
          "would make every estimate ~0 and answer 'neither' confidently)",
          raised)

    print("[self-test] FD verdict assembly")
    def _fd_elem(param, gfd, gb, gs, resolved=True, gates=()):
        return {"param": param, "stratum": "disagreement", "g_fd": gfd,
                "g_batched": gb, "g_sequential": gs, "resolved": resolved,
                "failed_gates": list(gates)}

    v = fd_verdict([_fd_elem("means", 1.0, 1.0, 2.0) for _ in range(4)])
    check("FD verdict: FD sits with batched -> 'batched'",
          v["means"]["verdict"] == "batched")
    check("FD verdict: relative errors are reported as NUMBERS",
          v["means"]["median_rel_err_batched"] == 0.0
          and abs(v["means"]["median_rel_err_sequential"] - 1.0) < 1e-9)
    v = fd_verdict([_fd_elem("means", 1.0, 2.0, 1.0) for _ in range(4)])
    check("FD verdict: FD sits with sequential -> 'sequential'",
          v["means"]["verdict"] == "sequential")
    v = fd_verdict([_fd_elem("means", 1.0, 5.0, 9.0) for _ in range(4)])
    check("FD verdict: BOTH paths wrong is reachable -> 'neither'",
          v["means"]["verdict"] == "neither")
    v = fd_verdict([_fd_elem("opacities", 1.0, 1.0, 1.0) for _ in range(4)])
    check("FD verdict: control stratum where both agree -> 'both'",
          v["opacities"]["verdict"] == "both")
    v = fd_verdict([_fd_elem("means", None, None, None, resolved=False,
                             gates=["smoothness"]) for _ in range(3)])
    check("FD verdict: nothing resolved -> 'inconclusive', never a guess",
          v["means"]["verdict"] == "inconclusive"
          and v["means"]["unresolved_by_gate"] == {"smoothness": 3})
    # The median can favour one path while the ELEMENTS disagree about which is
    # closer. Three elements say batched (0 vs 1.0 relative), two say sequential
    # (0.1 vs 0.05) -- both medians point at batched, but only 3/5 elements do.
    mixed = ([_fd_elem("means", 1.0, 1.0, 2.0) for _ in range(3)]
             + [_fd_elem("means", 1.0, 1.1, 1.05) for _ in range(2)])
    v_mixed = fd_verdict(mixed)["means"]
    check("FD verdict: median favours a path but elements disagree -> "
          "inconclusive, not the median's answer",
          v_mixed["verdict"] == "inconclusive"
          and abs(v_mixed["winner_consistency"] - 0.6) < 1e-9)
    # A fully split population where NEITHER median clears the tolerance is a
    # different outcome and must not be laundered into a path verdict either.
    split = ([_fd_elem("means", 1.0, 1.0, 2.0) for _ in range(2)]
             + [_fd_elem("means", 1.0, 2.0, 1.0) for _ in range(2)])
    check("FD verdict: fully split population -> 'neither', never a path",
          fd_verdict(split)["means"]["verdict"] == "neither")
    pathbad = [_fd_elem("means", 1.0, 1.0, 2.0, resolved=False,
                        gates=["path_independence"]) for _ in range(4)]
    check("FD verdict: broad path-independence failure forces inconclusive",
          fd_verdict(pathbad)["means"]["verdict"] == "inconclusive")

    print("[self-test] FD bisect end-to-end (CPU, no kernel)")
    # A linear stand-in population whose analytic gradients are exact, with a
    # deliberately WRONG 'sequential' gradient on means only. The bisect must
    # side with batched on means and with BOTH on every other parameter --
    # which is precisely the agreement-stratum control's job on real data.
    fd_pop = {
        "means": torch.rand((6, 3)) * 2.0 - 1.0,
        "quats": torch.rand((6, 4)),
        "scales": torch.rand((6, 3)) * 0.02 + 0.01,
        "opacities": torch.rand((6,)) * 0.6 + 0.2,
    }
    fd_coef = {"means": 3.0, "quats": 2.0, "scales": 5.0, "opacities": 0.7}

    def _fd_forward():
        return torch.cat([(fd_coef[k] * fd_pop[k]).reshape(-1)
                          for k in FD_PARAM_NAMES])

    fd_true = {k: torch.full_like(fd_pop[k], fd_coef[k]) for k in FD_PARAM_NAMES}
    fd_wrong = {k: v.clone() for k, v in fd_true.items()}
    fd_wrong["means"] = fd_wrong["means"] * 1.5      # 50% off, like the real gap

    blk = run_fd_bisect({"batched": _fd_forward, "sequential": _fd_forward},
                        fd_pop, fd_true, fd_wrong,
                        n_per_param=3, eps_fracs=[3e-3, 1e-2], repeats=2,
                        eps_rel_tol=0.10, smoothness_tol=0.10,
                        noise_signal_tol=0.20, accept_rel_tol=0.20,
                        path_rel_tol=0.10, max_seconds=60.0,
                        eps_absolute=[1e-3, 3e-3])
    check("FD e2e: every sampled element resolved on a smooth stand-in",
          blk["n_evaluated"] > 0
          and all(e["resolved"] for e in blk["elements"]))
    check("FD e2e: verdict for the corrupted parameter is 'batched'",
          blk["verdict"]["means"]["verdict"] == "batched")
    check("FD e2e: verdict for the uncorrupted parameters is 'both'",
          all(blk["verdict"][k]["verdict"] == "both"
              for k in ("quats", "scales", "opacities")))
    check("FD e2e: path independence MEASURED and holds when both closures "
          "compute the same forward",
          blk["fd_path_independence"]["holds"] is True
          and blk["fd_path_independence"]["n_compared"] > 0)
    check("FD e2e: the agreement-stratum control is actually sampled",
          any(e["stratum"] == "agreement" for e in blk["elements"]))
    check("FD e2e: absolute eps, raw |S(+)-S(-)| and touched-pixel count are "
          "recorded (an all-unresolved run is a TUNING dataset)",
          all("first_diff" in pe and "n_pixels_changed" in pe and "eps" in pe
              for e in blk["elements"]
              for pe in e["closures"]["batched"]["per_eps"]))

    def _fd_forward_biased():
        return _fd_forward() * 1.3

    blk_bad = run_fd_bisect(
        {"batched": _fd_forward, "sequential": _fd_forward_biased},
        fd_pop, fd_true, fd_wrong,
        n_per_param=2, eps_fracs=[3e-3, 1e-2], repeats=2,
        eps_rel_tol=0.10, smoothness_tol=0.10, noise_signal_tol=0.20,
        accept_rel_tol=0.20, path_rel_tol=0.10, max_seconds=60.0,
        eps_absolute=[1e-3, 3e-3])
    check("FD e2e: two closures giving DIFFERENT forwards fail path "
          "independence (it is measured, not asserted from forward parity)",
          blk_bad["fd_path_independence"]["holds"] is False)
    check("FD e2e: ...and that forces every parameter's verdict inconclusive",
          all(v["verdict"] == "inconclusive"
              for v in blk_bad["verdict"].values()))

    fd_sum = fd_bisect_summary([
        {"n_gaussians": 10, "n_cameras": 4, "color_mode": "sh",
         "sparse_grad": False, "status": "ok", "fd_bisect": blk}])
    check("FD summarize: per-parameter verdict set rolls up",
          fd_sum["fd_bisect_ran"] is True
          and fd_sum["fd_bisect_verdicts"]["means"] == ["batched"])
    check("FD summarize: absent block reads as 'not run', not as a finding",
          fd_bisect_summary([])["fd_bisect_ran"] is False)
    check("FD summarize: reaches summarize() itself",
          summarize([])["fd_bisect_ran"] is False)

    print("[self-test] timing / reporting")
    t = time_it(lambda: None, warmup=1, iters=3)
    check("time_it ok", t.get("ok") is True and t.get("iters") == 3)
    bad = time_it(lambda: (_ for _ in ()).throw(RuntimeError("boom")), 0, 1)
    check("time_it records error instead of raising",
          bad.get("ok") is False and bad.get("error_type") == "RuntimeError")
    check("summarize on empty results",
          summarize([])["batched_c_gt_1_expressible"] is False)
    # Test the shared _build_comparison helper with known inputs
    test_comp = _build_comparison("fwd_bwd_sequential_streamed", "fwd_bwd_batched", 1.5, 1.0)
    check("_build_comparison direction: ref=1.5ms, cand=1.0ms -> speedup=1.5x",
          test_comp is not None and abs(test_comp["speedup_reference_over_candidate"] - 1.5) < 1e-6)
    check("_build_comparison includes definition field",
          test_comp is not None and "definition" in test_comp and
          "reference_ms / candidate_ms" in test_comp["definition"])

    # Regression pin for #898: call _build_all_comparisons -- the SAME function
    # run_config() calls to populate runtime["comparisons"] -- on a fixture with
    # mutually DISTINCT timings in every slot (equal timings would make a
    # reference/candidate inversion at the call site invisible). Assert the
    # returned dict directly, not a hand-typed literal that could silently
    # disagree with what the shipping path actually produces.
    fixture_runtime = {
        "fwd_bwd_batched": {"ok": True, "mean_ms": 1.0},
        "fwd_bwd_sequential_streamed": {"ok": True, "mean_ms": 1.5},
        "fwd_bwd_sequential_retained": {"ok": True, "mean_ms": 2.0},
    }
    all_comps = _build_all_comparisons(fixture_runtime)

    c1 = all_comps.get("fwd_bwd_sequential_streamed_vs_fwd_bwd_batched")
    check("_build_all_comparisons: streamed_vs_batched reference is streamed",
          c1 is not None and c1["reference"] == "fwd_bwd_sequential_streamed")
    check("_build_all_comparisons: streamed_vs_batched candidate is batched",
          c1 is not None and c1["candidate"] == "fwd_bwd_batched")
    check("_build_all_comparisons: streamed_vs_batched speedup = 1.5ms / 1.0ms = 1.5x",
          c1 is not None and abs(c1["speedup_reference_over_candidate"] - 1.5) < 1e-6)

    c2 = all_comps.get("fwd_bwd_sequential_retained_vs_fwd_bwd_batched")
    check("_build_all_comparisons: retained_vs_batched reference is retained",
          c2 is not None and c2["reference"] == "fwd_bwd_sequential_retained")
    check("_build_all_comparisons: retained_vs_batched candidate is batched",
          c2 is not None and c2["candidate"] == "fwd_bwd_batched")
    check("_build_all_comparisons: retained_vs_batched speedup = 2.0ms / 1.0ms = 2.0x",
          c2 is not None and abs(c2["speedup_reference_over_candidate"] - 2.0) < 1e-6)

    c3 = all_comps.get("fwd_bwd_sequential_retained_vs_fwd_bwd_sequential_streamed")
    check("_build_all_comparisons: retained_vs_streamed reference is retained",
          c3 is not None and c3["reference"] == "fwd_bwd_sequential_retained")
    check("_build_all_comparisons: retained_vs_streamed candidate is streamed",
          c3 is not None and c3["candidate"] == "fwd_bwd_sequential_streamed")
    check("_build_all_comparisons: retained_vs_streamed speedup = 2.0ms / 1.5ms",
          c3 is not None and
          abs(c3["speedup_reference_over_candidate"] - (2.0 / 1.5)) < 1e-6)

    demo = [
        {"n_gaussians": 10, "n_cameras": 2, "color_mode": "sh", "sparse_grad": True,
         "status": "ok", "forward": {"within_tolerance": True},
         "backward": {"all_within_tolerance": True},
         "runtime": {"fwd_bwd_batched": {"ok": True, "mean_ms": 1.0},
                     "fwd_bwd_sequential_retained": {"ok": True, "mean_ms": 2.0},
                     "fwd_bwd_sequential_streamed": {"ok": True, "mean_ms": 1.5},
                     "comparisons": _build_all_comparisons({
                         "fwd_bwd_batched": {"ok": True, "mean_ms": 1.0},
                         "fwd_bwd_sequential_retained": {"ok": True, "mean_ms": 2.0},
                         "fwd_bwd_sequential_streamed": {"ok": True, "mean_ms": 1.5},
                     }),
                     "speedup_batched_over": {"fwd_bwd_sequential_retained": 2.0}},
         "memory": {"batched": {"peak_mb": 1.5},
                    "sequential_retained": {"peak_mb": 3.0},
                    "sequential_streamed": {"peak_mb": 2.0}}},
        {"n_gaussians": 10, "n_cameras": 4, "color_mode": "precomputed",
         "sparse_grad": False, "status": "batched_forward_unsupported",
         "batched_supported": False, "error_type": "RuntimeError",
         "error": "synthetic unsupported-path example"},
    ]
    print_table(demo)
    s = summarize(demo)
    check("summarize counts multi-camera configs", s["n_multi_camera_configurations"] == 2)
    check("summarize records unsupported error types",
          s["unsupported_error_types"] == ["RuntimeError"])

    # Issue #920: summarize must roll the localized defect up to BOTH the cell
    # list and the parameter list. Reuse the real function, not a restatement of
    # its logic: build the noise_floor_comparison block the same way run_config
    # does, from a cell where only `scales` carries an unexplained disagreement.
    nfc_per, nfc_overall = noise_floor_comparison(
        _cross({**{n: 0.001 for n in PARAM_NAMES}, "scales": 99.0},
               {**{n: True for n in PARAM_NAMES}, "scales": False}),
        _blk(**_all), _blk(**_all))
    localized = [{
        "n_gaussians": 10, "n_cameras": 4, "color_mode": "sh",
        "sparse_grad": False, "status": "ok",
        "backward_self_consistency": {
            "available": True,
            "noise_floor_comparison": nfc_per,
            "backward_noise_floor_exceeded": nfc_overall,
        },
    }]
    s_loc = summarize(localized)
    check("summarize: localized defect reaches the cell list (#920)",
          s_loc["backward_disagreement_exceeds_noise_floor"] ==
          ["N=10,C=4,colors=sh,sparse_grad=False"])
    check("summarize: rollup names only the affected parameter (#920)",
          s_loc["backward_unexplained_disagreement_params"] == ["scales"])
    check("summarize: rollup is empty when nothing is unexplained (#920)",
          summarize(demo)["backward_unexplained_disagreement_params"] == [])

    # ...and specifically: a parameter that clears its noise floor on a cell
    # whose parity is CLEAN must not reach either list. This is the control-arm
    # shape, and it is what separates the rollup from a plain
    # `exceeds_noise_floor` listing.
    wobble_per, wobble_overall = noise_floor_comparison(
        _cross({n: 2.0 for n in PARAM_NAMES}, True), _blk(**_all), _blk(**_all))
    s_wob = summarize([{
        "n_gaussians": 10, "n_cameras": 4, "color_mode": "sh",
        "sparse_grad": True, "status": "ok",
        "backward_self_consistency": {
            "available": True,
            "noise_floor_comparison": wobble_per,
            "backward_noise_floor_exceeded": wobble_overall,
        },
    }])
    check("summarize: parity-clean ratio wobble reaches neither list (#920)",
          s_wob["backward_disagreement_exceeds_noise_floor"] == []
          and s_wob["backward_unexplained_disagreement_params"] == []
          and wobble_per["means"]["exceeds_noise_floor"] is True)

    # An UNDETERMINED cell (verdict None) must land in the unavailable/
    # undetermined accounting, never in the exceedance list -- listing it would
    # read as positive evidence of a defect the run could not actually resolve.
    undet_per, undet_overall = noise_floor_comparison(
        {**_cross(_all, False), "quats": {}}, _blk(**_all), _blk(**_all))
    s_undet = summarize([{
        "n_gaussians": 10, "n_cameras": 4, "color_mode": "sh",
        "sparse_grad": False, "status": "ok",
        "backward_self_consistency": {
            "available": True,
            "noise_floor_comparison": undet_per,
            "backward_noise_floor_exceeded": undet_overall,
        },
    }])
    check("summarize: an undetermined verdict is not an exceedance (#920)",
          undet_overall is None
          and s_undet["backward_disagreement_exceeds_noise_floor"] == [])

    # Verify the demo fixture's comparisons (built via _build_all_comparisons above)
    # are self-consistent with its own recorded mean_ms.
    r = demo[0]
    rt = r["runtime"]
    comp = rt["comparisons"]["fwd_bwd_sequential_streamed_vs_fwd_bwd_batched"]
    ref_ms = rt["fwd_bwd_sequential_streamed"]["mean_ms"]
    cand_ms = rt["fwd_bwd_batched"]["mean_ms"]
    computed_speedup = ref_ms / cand_ms
    check("fixture speedup recomputed from recorded mean_ms matches stored value",
          abs(comp["speedup_reference_over_candidate"] - computed_speedup) < 1e-6)
    check("json-serializable", isinstance(json.dumps(demo, default=str), str))

    if failures:
        print(f"\n[self-test] FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("\n[self-test] all checks passed (CPU-only; proves harness plumbing, "
          "NOT anything about batching)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    RASTER_KWARGS["rasterize_mode"] = args.rasterize_mode
    if args.self_test:
        return self_test()
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "probe_results.json")

    payload: dict[str, Any] = {
        "probe": "issue106_camera_batch_probe",
        "issue": ["#106", "#614 ledger item 2"],
        "args": vars(args),
        "environment": _env_info(args),
        "results": [],
    }

    def _write() -> None:
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=False, default=str)

    if not torch.cuda.is_available():
        # A CPU invocation is data, not a crash: record it and exit cleanly.
        payload["outcome"] = "no_gpu"
        payload["outcome_detail"] = (
            "torch.cuda.is_available() is False. This probe must run inside the "
            "project container with GPU passthrough: "
            "docker compose run --rm train python scripts/issue106_camera_batch_probe.py"
        )
        _write()
        print(f"[probe] NO GPU -- wrote {out_path}")
        return 2

    try:
        from gsplat.rendering import rasterization  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        payload["outcome"] = "gsplat_import_failed"
        payload["outcome_detail"] = f"{type(exc).__name__}: {exc}"
        _write()
        print(f"[probe] gsplat import failed: {exc}")
        return 3

    device = torch.device("cuda")
    sparse_values = [v == "true" for v in args.sparse_grad]

    total = len(args.sizes) * len(args.cameras) * len(args.color_modes) * len(sparse_values)
    print(f"[probe] {total} configurations "
          f"(sizes={args.sizes} cameras={args.cameras} "
          f"color_modes={args.color_modes} sparse_grad={sparse_values})")
    print(f"[probe] device={payload['environment'].get('device_name')} "
          f"torch={payload['environment'].get('torch')} "
          f"gsplat={payload['environment'].get('gsplat')}")

    idx = 0
    for n in args.sizes:
        for c in args.cameras:
            for mode in args.color_modes:
                for sg in sparse_values:
                    idx += 1
                    print(f"[probe] ({idx}/{total}) N={n} C={c} colors={mode} "
                          f"sparse_grad={sg} ...", flush=True)
                    try:
                        res = run_config(rasterization, n, c, mode, sg, args, device)
                    except Exception as exc:  # noqa: BLE001 - never abort the sweep
                        res = {"n_gaussians": n, "n_cameras": c, "color_mode": mode,
                               "sparse_grad": sg, "status": "harness_error",
                               "error_type": type(exc).__name__,
                               "error": str(exc)[:600],
                               "traceback": traceback.format_exc()[-2000:]}
                    payload["results"].append(res)
                    _write()  # checkpoint after every configuration
                    _reset_peak()

    payload["summary"] = summarize(payload["results"])
    payload["outcome"] = "completed"
    _write()

    print_table(payload["results"])
    print(json.dumps(payload["summary"], indent=2))
    print(f"[probe] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
