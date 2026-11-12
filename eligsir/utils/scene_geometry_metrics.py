"""Quantitative scene-geometry metrics for comparing GS runs.

Distinct from ``utils/geometry_metrics.py`` (runtime floater/visibility dashboard):
this module produces *run-comparison* descriptors written into ``report.json`` /
``geometry_metrics.json`` and tabulated across runs.

Tiers (see the project plan):
  A. model-space per-Gaussian shape stats — deterministic, GT-free, from a PLY or model.
  B. masked splat-vs-depth normal agreement — render-space self-consistency.
  C. sensor-anchored accuracy — rendered/splat-normal vs sensor depth (RGB-D streaming).
  D. chamfer / off-surface floater mass — ``compute_offsurface_metrics`` (bottom of this
     file) is a real, complete implementation (opacity-weighted NN-distance vs an
     independent sensor point cloud), not a stub — it was just never wired into the
     *default* per-run report, only the opt-in ``--reference_map`` path
     (``utils/map_comparison.py``). Fixed 2026-08-05:
     ``sensor_points_from_iq_vox`` decodes the mean position of each voxel already
     accumulated by ``StreamingScene``'s existing input-quality tracker
     (``_accumulate_thickness``, on by default via ``--streaming_track_input_quality``)
     into a real, independent-of-the-Gaussians point cloud, letting
     ``write_post_training_report`` populate ``geometry.offsurface`` on every
     streaming run with no new accumulation cost and no external reference map
     needed. The opt-in ``--reference_map`` path is unaffected and still the
     stronger, truly-independent-source check when available.

All functions return plain JSON-serialisable dicts.

Comparability contract (the whole point of these numbers): compare runs only on the
**same test cameras + resolution**, and restrict per-Gaussian stats to **live mass**
(opacity > ``opacity_threshold``) so dead, never-rendered Gaussians don't skew percentiles.

Needle vs disk (locked definition): sort per-Gaussian scales s1>=s2>=s3.
  * needle  <=>  s2/s1 small  (the second axis has collapsed)        -> the spike artifact
  * disk    <=>  s3/s2 small WITH s2/s1 ~ 1  (a healthy flat surfel)  -> desirable
The legacy ``s_max/s_min`` (= s1/s3) conflates disks and needles and is NOT the headline.
"""

from __future__ import annotations

import numpy as np


def _to_np(x) -> np.ndarray:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def _pcts(a: np.ndarray, ps=(10, 50, 90, 99)) -> dict:
    if a.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    q = np.percentile(a, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, q)}


# ---------------------------------------------------------------------------
# Tier A — model-space per-Gaussian shape stats
# ---------------------------------------------------------------------------
def gaussian_shape_stats(scales, opacities, xyz=None,
                         opacity_threshold: float = 0.05,
                         large_scale_abs: float = 0.10,
                         needle_thresholds=(0.3, 0.2),
                         flat_threshold: float = 0.2) -> dict:
    """Per-Gaussian shape distribution over *live* mass (opacity > threshold).

    Args:
        scales: ``[N,3]`` ACTIVATED (metric) scales.
        opacities: ``[N]`` or ``[N,1]`` opacities in [0,1].
        xyz: optional ``[N,3]`` centres (radius / far-count stats).
        opacity_threshold: live-mass cutoff.
        large_scale_abs: metric-scale (m) above which a splat counts as "large".
        needle_thresholds: s2/s1 cutoffs reported as needle fractions.
        flat_threshold: s3/s2 cutoff for the flat-surfel fraction.
    """
    s = _to_np(scales).reshape(-1, 3)
    op = _to_np(opacities).reshape(-1)
    n_total = int(s.shape[0])
    if n_total == 0:
        return {"n_total": 0}

    live = op > float(opacity_threshold)
    n_live = int(live.sum())
    out = {
        "n_total": n_total,
        "n_live": n_live,
        "live_frac": float(n_live / max(n_total, 1)),
        "opacity_mean": float(op.mean()),
        "opacity_sum": float(op.sum()),          # opacity-weighted effective count (N_eff)
        "n_eff_frac": float(op.sum() / max(n_total, 1)),  # N_eff / N_total — redundancy (size-invariant)
        "opacity_near0_frac": float((op < 0.05).mean()),
        "opacity_near1_frac": float((op > 0.95).mean()),
        "opacity_threshold": float(opacity_threshold),
    }

    sl = s[live] if n_live > 0 else s
    opl = op[live] if n_live > 0 else op
    ss = -np.sort(-sl, axis=1)                     # descending: s1>=s2>=s3
    s1 = np.clip(ss[:, 0], 1e-12, None)
    s2 = np.clip(ss[:, 1], 1e-12, None)
    s3 = np.clip(ss[:, 2], 1e-12, None)
    needle = s2 / s1                               # small => needle (headline)
    disk = s3 / s2                                 # small => flat disk (good)
    aniso = s1 / s3                                # legacy s_max/s_min (not headline)
    vol = s1 * s2 * s3

    out["needle_s2_s1_median"] = float(np.median(needle))
    out["needle_s2_s1_p10"] = float(np.percentile(needle, 10))
    for t in needle_thresholds:
        out[f"needle_frac_lt_{t}"] = float((needle < t).mean())
    out["disk_s3_s2_median"] = float(np.median(disk))
    out[f"flat_frac_lt_{flat_threshold}"] = float((disk < flat_threshold).mean())
    out["aniso_s1_s3_median"] = float(np.median(aniso))
    out["aniso_s1_s3_p99"] = float(np.percentile(aniso, 99))

    out.update({f"max_scale_{k}": v for k, v in _pcts(s1).items()})
    out.update({f"volume_{k}": v for k, v in _pcts(vol).items()})
    out["large_scale_abs"] = float(large_scale_abs)
    out["large_splat_frac"] = float((s1 > large_scale_abs).mean())

    # Opacity-weighted (mass-aware) needle fraction — robust to count differences.
    w = opl / max(float(opl.sum()), 1e-12)
    out["needle_massfrac_lt_0.3"] = float(w[needle < 0.3].sum())

    if xyz is not None:
        p = _to_np(xyz).reshape(-1, 3)
        pl = p[live] if n_live > 0 else p
        c = np.median(pl, axis=0)
        r = np.linalg.norm(pl - c, axis=1)
        out.update({f"radius_{k}": v for k, v in _pcts(r, (95, 99)).items()})
        out["radius_max_m"] = float(r.max())
        out["count_gt_5m"] = int((r > 5.0).sum())
        out["count_gt_10m"] = int((r > 10.0).sum())
    return out


def scene_complexity_stats(points_world, normals, *,
                           voxel_size: float,
                           normal_bins: int = 8,
                           scales=(1, 2, 4)) -> dict:
    """Input-side scene expressiveness: surfel demand ``SV`` of the observed surface.

    ``SV`` = number of UNIQUE surface cells (position-voxel x normal-bin) spanned by the
    fused observed depth cloud. A flat wall occupies few normal-bins; a curved/cluttered
    surface spreads across many -> higher SV. This is a reconstruction-independent property
    of the scene+sensor (a per-scene constant), so it normalises Gaussian count for
    efficiency: ``gaussians_per_surfel = n_live / SV`` is size/growth-invariant.

    Multiscale SV over ``scales``*``voxel_size`` yields a box-counting ``complexity_dim``
    (SV ~ delta^-D). Reuses ``geometry.surface_cells.surface_cell_keys``.

    Args:
        points_world: ``[M,3]`` fused surface points in world space (stable/filtered depth).
        normals: ``[M,3]`` per-point normals (world).
        voxel_size: base cell size (m), tie to sensor scale.
        normal_bins: orientation bins per axis (matches SurfaceCellIndex default 8).
        scales: multipliers on ``voxel_size`` for the multiscale / dimension fit.
    """
    from geometry.surface_cells import surface_cell_keys

    pts = _to_np(points_world).reshape(-1, 3)
    nrm = _to_np(normals).reshape(-1, 3)
    n_points = int(pts.shape[0])
    base = {
        "sv": 0,
        "voxel_size": float(voxel_size),
        "normal_bins": int(normal_bins),
        "n_points": n_points,
        "sv_multiscale": {},
        "complexity_dim": None,
    }
    if n_points == 0:
        return base

    deltas, svs, boxes, multiscale = [], [], [], {}
    for m in scales:
        d = float(voxel_size) * float(m)
        # SV (surfel demand): orientation-aware unique cells (voxel x normal-bin).
        sv = int(np.unique(surface_cell_keys(pts, nrm, voxel_size=d, normal_bins=normal_bins)).size)
        # Box count: POSITION-ONLY occupied voxels (clean geometric box-counting; the
        # normal-bin term would otherwise contaminate the fractal dimension).
        vox = np.floor((pts + 1000.0) / max(d, 1e-9)).astype(np.int64)
        nbox = int(np.unique(vox, axis=0).shape[0])
        multiscale[f"x{m}"] = {"voxel_size": d, "sv": sv, "occupied_voxels": nbox}
        deltas.append(d)
        svs.append(sv)
        boxes.append(nbox)

    base["sv"] = multiscale[f"x{scales[0]}"]["sv"]
    base["sv_multiscale"] = multiscale
    # Box-counting dimension from POSITION voxels: N(delta) ~ delta^-D => D = -slope.
    # ~2 for a clean surface; >2 as volumetric clutter/noise fills space.
    if len(deltas) >= 2 and min(boxes) > 0:
        slope = float(np.polyfit(np.log(deltas), np.log(boxes), 1)[0])
        base["complexity_dim"] = -slope
    return base


def expressiveness_metrics(*, sv, n_live, n_total=None, opacity_sum=None,
                           mean_test_psnr=None, eval_valid=True) -> dict:
    """Per-run efficiency from a scene's SV and a run's live count.

    Headline ``gaussians_per_surfel`` is size/growth-invariant (SV scales with the scene).
    ``psnr_per_kgaussian`` is a within-one-fixed-scene diagnostic only (size-confounded) and
    is emitted only for a valid holdout eval.
    """
    out = {"sv": (int(sv) if sv is not None else None), "n_live": int(n_live)}
    if sv and sv > 0:
        out["gaussians_per_surfel"] = float(n_live) / float(sv)
    else:
        out["gaussians_per_surfel"] = None
    if opacity_sum is not None and n_total:
        out["n_eff_frac"] = float(opacity_sum) / float(max(n_total, 1))
    if mean_test_psnr is not None and eval_valid and n_live > 0:
        # Diagnostic ONLY — valid within an identical scene; not cross-scene comparable.
        out["psnr_per_kgaussian"] = float(mean_test_psnr) / (float(n_live) / 1000.0)
        out["psnr_per_kgaussian_note"] = "within-identical-scene diagnostic only (size-confounded)"
    return out


def input_quality_stats(*, sensor_noise_m=None, sensor_noise_std_m=None, sensor_n=0,
                         odom_stats=None, surface_thickness_m=None,
                         complexity_dim=None, pose_source=None) -> dict:
    """Split input degradation into sensor noise (intra-frame) vs tracking error
    (inter-frame). Identity: surface_thickness^2 ~ sensor_noise^2 + tracking_thickness^2.

    - sensor noise: mean per-frame PCA plane residual (m), pose-free.
    - tracking error: aggregated odometry registration stats (rmse/fitness/fail) —
      None for ground-truth/precomputed poses (tracking ~ 0 by construction).
    - combined: surface_thickness (m) + box-counting complexity_dim (~2 clean, >2
      volumetric). Model cross-check tracking_thickness = sqrt(thickness^2 - sensor^2).
    """
    out = {
        "pose_source": pose_source,
        "sensor_noise_m": (None if sensor_noise_m is None else float(sensor_noise_m)),
        "sensor_noise_std_m": (None if sensor_noise_std_m is None else float(sensor_noise_std_m)),
        "sensor_n_px": int(sensor_n),
        "surface_thickness_m": (None if surface_thickness_m is None else float(surface_thickness_m)),
        "complexity_dim": (None if complexity_dim is None else float(complexity_dim)),
        "dim_excess": (None if complexity_dim is None else float(complexity_dim) - 2.0),
    }

    odom = [o for o in (odom_stats or []) if isinstance(o, dict)]
    if odom:
        rmse = [o["inlier_rmse"] for o in odom if o.get("inlier_rmse") is not None]
        fit = [o["fitness"] for o in odom if o.get("fitness") is not None]
        nfail = sum(1 for o in odom if o.get("valid") is False)
        out.update({
            "tracking_available": True,
            "tracking_inlier_rmse_m": (float(np.mean(rmse)) if rmse else None),
            "tracking_fitness": (float(np.mean(fit)) if fit else None),
            "tracking_fail_frac": float(nfail / max(len(odom), 1)),
            "tracking_n_edges": len(odom),
        })
    else:
        out.update({
            "tracking_available": False,
            "tracking_inlier_rmse_m": None, "tracking_fitness": None,
            "tracking_fail_frac": None, "tracking_n_edges": 0,
            "tracking_note": "no odometry stats — ground-truth/precomputed poses (tracking ~ 0)",
        })

    # Non-sensor thickening = total thickness with the sensor-noise floor removed. This
    # is an UPPER BOUND on tracking error: it also absorbs genuine surface curvature and
    # depth-edge / voxel-mixing, so it is not pure tracking on its own. Clean tracking
    # isolation needs the controlled GT-vs-odometry thickness diff (same scene).
    nst = None
    if surface_thickness_m is not None and sensor_noise_m is not None:
        nst = float(np.sqrt(max(0.0, float(surface_thickness_m) ** 2 - float(sensor_noise_m) ** 2)))
    out["nonsensor_thickness_m"] = nst
    out["tracking_thickness_m"] = nst  # kept for back-compat; == nonsensor (upper bound)
    out["tracking_thickness_note"] = (
        "upper bound on tracking — also includes curvature & depth-edge mixing; "
        "isolate true tracking via GT-vs-odometry thickness diff")

    # Verdict (robust): dim_excess is the trustworthy single-run thickening signal.
    # When poses are ground-truth/precomputed (no odometry), tracking error is ~0 by
    # construction, so the run can only be 'clean' or 'sensor-limited' — never
    # 'tracking-limited' (the non-sensor thickening there is curvature/edges, not drift).
    dim_excess = out["dim_excess"]
    sensor = sensor_noise_m or 0.0
    if dim_excess is not None and dim_excess < 0.15:
        verdict = "clean"
    elif not out["tracking_available"]:
        verdict = "sensor-limited" if sensor > 0 else "clean"
    else:
        track = out["tracking_inlier_rmse_m"] or 0.0
        nst_v = nst or 0.0
        if nst_v > 1.3 * sensor and nst_v > 0:
            verdict = "tracking-limited"
        elif sensor > 1.3 * nst_v and sensor > 0:
            verdict = "sensor-limited"
        elif sensor > 0 or nst_v > 0:
            verdict = "mixed"
        else:
            verdict = "unknown"
    out["quality_verdict"] = verdict
    return out


def shape_stats_from_model(gaussians, **kw) -> dict:
    """Tier A from a live Gaussian model (activated scales/opacity)."""
    return gaussian_shape_stats(gaussians.get_scaling, gaussians.get_opacity,
                                xyz=gaussians.get_xyz, **kw)


def lifecycle_stats_from_model(gaussians, opacity_threshold: float = 0.05) -> dict:
    """Count Gaussians per lifecycle state over the live mass (opacity > threshold).

    States: 0=PROVISIONAL, 1=YOUNG, 2=MATURE, 3=FROZEN.
    Returns counts and fractions relative to n_live.
    """
    lc = getattr(gaussians, "lifecycle_state", None)
    if lc is None:
        return {"lifecycle_available": False}
    import torch
    with torch.no_grad():
        op = gaussians.get_opacity
        if hasattr(op, "squeeze"):
            op = op.squeeze(-1)
        live = op > opacity_threshold
        n_live = int(live.sum().item())
        lc_live = lc[live].to(torch.int32)
        names = {0: "provisional", 1: "young", 2: "mature", 3: "frozen"}
        counts = {name: int((lc_live == idx).sum().item()) for idx, name in names.items()}
        fracs = {f"{name}_frac": (counts[name] / n_live if n_live > 0 else 0.0)
                 for name in names.values()}
    return {"lifecycle_available": True, "n_live": n_live, **counts, **fracs}



def gaussian_belief_summary_from_model(gaussians) -> dict:
    """Observational summary of ``GaussianBelief`` (issue #1365 Phase 1).

    Purely additive telemetry -- no decision anywhere reads this. Reports basic
    live-mass aggregates over the belief buffers plus each field's provenance
    (``alias`` / ``new_wired`` / ``new_structure_only``) so a reader of
    ``report.json`` can tell a real measurement apart from a not-yet-computed
    placeholder without cross-referencing source. Defensive by construction: a
    missing buffer (e.g. a pre-#1372 checkpoint whose restore skipped a
    size-mismatched buffer) degrades to ``belief_available: False`` rather than
    raising, matching ``lifecycle_stats_from_model``'s own precedent above.
    """
    try:
        from eligsir.utils.gaussian_belief import (
            ExistenceBelief, FIELD_PROVENANCE, GaussianBelief, GeometryBelief,
            ObservabilityBelief, ProvenanceBelief, RepresentationBelief,
            TemporalBelief, compute_p_exist,
        )
        import torch

        belief = GaussianBelief.from_gaussians(gaussians)
        # Defensive schema check: catches a future refactor of GaussianBelief's
        # group dataclasses silently drifting out of sync with this reader.
        assert isinstance(belief.provenance, ProvenanceBelief)
        assert isinstance(belief.observability, ObservabilityBelief)
        assert isinstance(belief.existence, ExistenceBelief)
        assert isinstance(belief.geometry, GeometryBelief)
        assert isinstance(belief.representation, RepresentationBelief)
        assert isinstance(belief.temporal, TemporalBelief)
        with torch.no_grad():
            op = gaussians.get_opacity
            if hasattr(op, "squeeze"):
                op = op.squeeze(-1)
            live = op > 0.05
            n_live = int(live.sum().item())
            p_exist = belief.existence.p_exist
            mean_p_exist_live = (
                float(p_exist[live].to(torch.float32).mean().item()) if n_live > 0 else None
            )
            # Consistency check against the stored default: real alpha/beta
            # accumulation is not wired yet (both are 0 for every Gaussian this
            # phase), so this should equal mean_p_exist_live exactly today --
            # once a later stage starts accumulating real evidence, this becomes
            # a genuine independent re-derivation to compare the stored value against.
            recomputed = compute_p_exist(
                belief.existence.support_evidence.to(torch.float32),
                belief.existence.contradiction_evidence.to(torch.float32),
            )
            mean_computed_p_exist_live = (
                float(recomputed[live].mean().item()) if n_live > 0 else None
            )
        return {
            "belief_available": True,
            "n": belief.n,
            "n_live": n_live,
            "mean_p_exist_live": mean_p_exist_live,
            "mean_computed_p_exist_live": mean_computed_p_exist_live,
            "field_provenance": FIELD_PROVENANCE,
        }
    except Exception as exc:
        return {"belief_available": False, "error": str(exc)}


def shape_stats_from_ply(path: str, **kw) -> dict:
    """Tier A from a saved ``point_cloud.ply`` (raw log-scales + opacity logits)."""
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    scales = np.stack([np.exp(np.asarray(v[f"scale_{i}"], dtype=np.float32)) for i in range(3)], axis=1)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    xyz = np.stack([np.asarray(v[a], dtype=np.float32) for a in ("x", "y", "z")], axis=1)
    return gaussian_shape_stats(scales, op, xyz=xyz, **kw)


# ---------------------------------------------------------------------------
# Tier A extension — scale/quaternion-aware rendered-extent estimate (#137)
# ---------------------------------------------------------------------------
def compute_rendered_extent_stats(means, scales, quats, opacities, *,
                                   opacity_threshold: float = 0.05,
                                   sigma: float = 3.0,
                                   physical_scale_threshold_abs_m: float = 1.0,
                                   physical_scale_threshold_rel: float = 5.0) -> dict:
    """Scale/quaternion-aware rendered-extent estimate (#137).

    ``compare_scene_vs_trajectory``/``compute_trajectory_extent`` above bound
    the scene using Gaussian **centres** only. That misses catastrophically
    oversized, fully-opaque splats entirely: a handful of ~40m-scale
    Gaussians are invisible to a centre-only bbox even though they dominate
    what actually gets rendered (real ScanNet++ finding: centre bbox ~30m,
    rendered 3-sigma extent hundreds of metres -- see issue #137).

    This restricts to live mass (``opacity > opacity_threshold``, matching
    this module's comparability contract) and, for each live Gaussian,
    builds its ``sigma``-sigma ellipsoid's *world-space* axis-aligned
    half-extent from its quaternion and scale (not just an isotropic radius):

        half_extent[a] = sigma * sqrt(sum_i (R[a, i] * scale[i])^2)

    (``R`` = wxyz-quaternion rotation matrix, world = R @ local -- the same
    convention as ``_compute_splat_normals_camera`` in
    ``eligsir/mapping/renderer/gsplat_backend.py``.)

    Returns BOTH:
      * a union world AABB over all live Gaussians' sigma-ellipsoids
        (``union_aabb_*``) -- the scale/quat-aware analogue of the
        centre-only bbox above, and
      * robust per-Gaussian statistics of each Gaussian's own 3-sigma extent
        along its longest axis (``extent_p50/p90/p99/p99.9``, ``extent_max``,
        ``extent_mass_weighted_m``, ``count_above_physical_scale_threshold``)
        so a single absurd splat is visible (max/p99.9/threshold count)
        without erasing the shape of the typical population (median/p90/
        mass-weighted mean) -- the union AABB alone would be dominated by
        exactly one outlier, which is the failure mode being fixed here.

    Args:
        means: ``[N,3]`` Gaussian centres (world space).
        scales: ``[N,3]`` ACTIVATED (metric) scales, same convention as
            ``gaussian_shape_stats``.
        quats: ``[N,4]`` quaternions, wxyz order, need not be pre-normalised.
        opacities: ``[N]`` or ``[N,1]`` opacities in [0,1].
        opacity_threshold: live-mass cutoff (see module comparability contract).
        sigma: ellipsoid confidence radius in standard deviations (default 3).
        physical_scale_threshold_abs_m: absolute floor (metres) for the
            "oversized splat" threshold on a Gaussian's raw longest semi-axis.
        physical_scale_threshold_rel: multiplier on the scene's own live
            median longest semi-axis for a scale-relative threshold.
            ``threshold_used = max(abs, rel * median_scale)`` so tiny scenes
            still need an absolute minimum size to count as "oversized" and
            large scenes don't get spuriously flagged just for being large.
    """
    means_np = _to_np(means).reshape(-1, 3)
    scales_np = _to_np(scales).reshape(-1, 3)
    quats_np = _to_np(quats).reshape(-1, 4)
    op = _to_np(opacities).reshape(-1)

    n_total = int(means_np.shape[0])
    out = {"valid": False, "n_total": n_total, "opacity_threshold": float(opacity_threshold),
           "sigma": float(sigma)}
    if n_total == 0:
        out["reason"] = "empty input"
        return out

    live = op > float(opacity_threshold)
    n_live = int(live.sum())
    out["n_live"] = n_live
    if n_live == 0:
        out["reason"] = "no live mass above opacity_threshold"
        return out

    m = means_np[live]
    s = np.clip(scales_np[live].astype(np.float64), 1e-12, None)
    q = quats_np[live].astype(np.float64)
    opl = op[live].astype(np.float64)

    qn = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    # wxyz quaternion -> rotation matrix [n_live, 3, 3]; world = R @ local.
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=1).reshape(-1, 3, 3)

    # Per-axis world-space half-extent of the sigma-ellipsoid:
    # half_extent[n,a] = sigma * sqrt(sum_i R[n,a,i]^2 * s[n,i]^2)
    half_extent = float(sigma) * np.sqrt(np.einsum('nai,ni->na', R ** 2, s ** 2))

    mins = m - half_extent
    maxs = m + half_extent
    union_min = mins.min(axis=0)
    union_max = maxs.max(axis=0)
    union_size = union_max - union_min

    out["union_aabb_min"] = union_min.tolist()
    out["union_aabb_max"] = union_max.tolist()
    out["union_aabb_size_m"] = union_size.tolist()
    out["union_aabb_max_dim_m"] = float(union_size.max())
    out["union_aabb_volume_m3"] = float(np.prod(np.maximum(union_size, 0.0)))

    # Per-Gaussian rotation-invariant "physical size": raw longest semi-axis,
    # and its sigma-sigma extent -- robust stats over the live population.
    s1 = s.max(axis=1)                       # [n_live] raw metric scale (m)
    extent = float(sigma) * s1               # [n_live] 3-sigma extent along own longest axis (m)

    pcts = _pcts(extent, ps=(50, 90, 99, 99.9))
    out["extent_p50_m"] = pcts["p50"]
    out["extent_p90_m"] = pcts["p90"]
    out["extent_p99_m"] = pcts["p99"]
    out["extent_p99.9_m"] = pcts["p99.9"]
    out["extent_max_m"] = float(extent.max())
    out["extent_min_m"] = float(extent.min())

    w_mass = opl / max(float(opl.sum()), 1e-12)
    out["extent_mass_weighted_m"] = float((extent * w_mass).sum())

    median_scale_m = float(np.median(s1))
    out["median_scale_m"] = median_scale_m
    threshold_abs = float(physical_scale_threshold_abs_m)
    threshold_rel = float(physical_scale_threshold_rel) * median_scale_m
    threshold_used = max(threshold_abs, threshold_rel)
    out["physical_scale_threshold_abs_m"] = threshold_abs
    out["physical_scale_threshold_rel_multiplier"] = float(physical_scale_threshold_rel)
    out["physical_scale_threshold_used_m"] = threshold_used

    above = s1 > threshold_used
    out["count_above_physical_scale_threshold"] = int(above.sum())
    out["frac_count_above_physical_scale_threshold"] = float(above.mean())
    out["mass_above_physical_scale_threshold"] = float(opl[above].sum() / max(float(opl.sum()), 1e-12))

    out["valid"] = True
    return out


def rendered_extent_stats_from_ply(path: str, **kw) -> dict:
    """``compute_rendered_extent_stats`` from a saved ``point_cloud.ply``
    (raw log-scales, raw quaternion, raw opacity logits) -- mirrors
    ``shape_stats_from_ply``'s standalone-PLY convention so this can run
    against a saved checkpoint with no live training run/model object.
    """
    from plyfile import PlyData
    v = PlyData.read(path)["vertex"]
    scales = np.stack([np.exp(np.asarray(v[f"scale_{i}"], dtype=np.float32)) for i in range(3)], axis=1)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], dtype=np.float32)))
    xyz = np.stack([np.asarray(v[a], dtype=np.float32) for a in ("x", "y", "z")], axis=1)
    rot_names = sorted([p.name for p in v.properties if p.name.startswith("rot_")],
                        key=lambda n: int(n.split("_")[-1]))
    quats = np.stack([np.asarray(v[n], dtype=np.float32) for n in rot_names], axis=1)
    return compute_rendered_extent_stats(xyz, scales, quats, op, **kw)


# ---------------------------------------------------------------------------
# Tier B/C — render-space normal agreement & sensor-anchored accuracy
# ---------------------------------------------------------------------------
def angle_between_normal_maps(n1, n2, eps: float = 1e-8):
    """``[3,H,W]`` (raw, signed) normal maps -> ``[H,W]`` angle in degrees (torch)."""
    import torch
    a = torch.nn.functional.normalize(n1, dim=0, eps=eps)
    b = torch.nn.functional.normalize(n2, dim=0, eps=eps)
    dot = (a * b).sum(dim=0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(dot))


def normal_agreement_stats(angle_deg, mask) -> dict:
    """Aggregate a per-pixel normal-angle map under a validity mask (Tier B/C).

    ``mask`` should already exclude uncovered pixels and depth discontinuities.
    """
    a = _to_np(angle_deg).reshape(-1)
    m = _to_np(mask).reshape(-1) > 0.5
    a = a[m & np.isfinite(a)]
    if a.size == 0:
        return {"n_px": 0}
    return {
        "n_px": int(a.size),
        "normal_angle_mean_deg": float(a.mean()),
        "normal_angle_median_deg": float(np.median(a)),
        "normal_within_10deg": float((a < 10).mean()),
        "normal_within_20deg": float((a < 20).mean()),
        "normal_within_30deg": float((a < 30).mean()),
    }


def depth_accuracy_stats(rendered_depth, sensor_depth, max_depth: float = 10.0) -> dict:
    """Rendered-vs-sensor depth accuracy over jointly-valid pixels (Tier C)."""
    r = _to_np(rendered_depth).reshape(-1)
    s = _to_np(sensor_depth).reshape(-1)
    valid = (s > 1e-3) & (s < max_depth) & (r > 1e-3) & np.isfinite(r) & np.isfinite(s)
    r = r[valid]; s = s[valid]
    if r.size == 0:
        return {"n_px": 0}
    err = np.abs(r - s)
    ratio = np.maximum(r / s, s / r)
    return {
        "n_px": int(r.size),
        "depth_mae_m": float(err.mean()),
        "depth_rmse_m": float(np.sqrt((err ** 2).mean())),
        "depth_absrel": float((err / s).mean()),
        "delta_1.25": float((ratio < 1.25).mean()),
        "delta_1.25_2": float((ratio < 1.25 ** 2).mean()),
        "delta_1.25_3": float((ratio < 1.25 ** 3).mean()),
    }


def aggregate_view_metrics(per_view: list) -> dict:
    """Mean-aggregate a list of per-view metric dicts (numeric keys only)."""
    if not per_view:
        return {}
    keys = set().union(*[d.keys() for d in per_view])
    out = {}
    for k in keys:
        vals = [d[k] for d in per_view
                if isinstance(d.get(k), (int, float)) and np.isfinite(d[k])]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_views"] = len(per_view)
    return out


# ---------------------------------------------------------------------------
# Scene-vs-trajectory extent — "how much of the reconstruction is outside
# where the camera actually went" (the ad-hoc floater/plume diagnostic this
# session built by hand for ScanNet++/floor3, now a standing metric).
# ---------------------------------------------------------------------------
def compute_voxel_occupancy(points, voxel_size: float) -> dict:
    """Real occupied-volume figure via open3d voxelization (not an AABB proxy).

    Kept alongside the coarse AABB bbox_volume_m3 (not a replacement for it) --
    see the earlier ``scene_metadata.json`` COARSE_PROXY figure this supersedes
    for occupied-volume purposes only.
    """
    import open3d as o3d

    pts = _to_np(points).reshape(-1, 3)
    n_points = int(pts.shape[0])
    out = {"voxel_size": float(voxel_size), "n_points": n_points, "n_voxels_occupied": 0,
           "occupied_volume_m3": 0.0, "bbox_min": None, "bbox_max": None,
           "bbox_size": None, "bbox_volume_m3": 0.0}
    if n_points == 0:
        return out

    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    out["bbox_min"] = bbox_min.tolist()
    out["bbox_max"] = bbox_max.tolist()
    out["bbox_size"] = bbox_size.tolist()
    out["bbox_volume_m3"] = float(np.prod(np.maximum(bbox_size, 0.0)))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=float(voxel_size))
    n_vox = len(vg.get_voxels())
    out["n_voxels_occupied"] = int(n_vox)
    out["occupied_volume_m3"] = float(n_vox) * (float(voxel_size) ** 3)
    return out


def compute_trajectory_extent(camera_positions) -> dict:
    """Bbox/centroid stats from camera positions (``entry["position"]`` in
    cameras.json / trajectory_full.jsonl's shared camera_to_JSON schema)."""
    pos = _to_np(camera_positions).reshape(-1, 3)
    n = int(pos.shape[0])
    out = {"n_positions": n, "bbox_min": None, "bbox_max": None, "bbox_size": None,
           "bbox_volume_m3": 0.0, "centroid": None}
    if n == 0:
        return out
    bbox_min = pos.min(axis=0)
    bbox_max = pos.max(axis=0)
    bbox_size = bbox_max - bbox_min
    out["bbox_min"] = bbox_min.tolist()
    out["bbox_max"] = bbox_max.tolist()
    out["bbox_size"] = bbox_size.tolist()
    out["bbox_volume_m3"] = float(np.prod(np.maximum(bbox_size, 0.0)))
    out["centroid"] = pos.mean(axis=0).tolist()
    return out


def compare_scene_vs_trajectory(gaussian_points, opacities, trajectory_positions,
                                margin_m: float = 2.0, opacity_threshold: float = 0.05) -> dict:
    """Plume quantification: how much reconstructed mass sits outside the space
    the camera actually visited. Fills the gap Tier D's sensor-vs-Gaussian
    comparison doesn't cover (that's floater-vs-sensor-surface; this is
    reconstruction-extent-vs-camera-trajectory).

    Restricted to live mass (opacity > opacity_threshold) before any bbox/NN
    work, per this module's comparability contract (see module docstring) --
    a single near-dead outlier Gaussian would otherwise dominate
    max_distance_from_trajectory_m/bbox_volume_ratio even though it contributes
    negligible rendered mass.

    NOTE (why this is eval-only, not wired into live per-iteration training
    metrics): the nearest-neighbor query below is a real O(N log M) cost
    against the live Gaussian set, not free the way report.json's existing
    periodic PSNR tick is (that tick already pays for a render it needs
    anyway). This metric is brand new and hasn't yet been validated across
    enough real runs to justify that recurring live cost -- same standing
    discipline this session applied to `@playcanvas/splat-transform`'s
    filters (proven harmful before being ruled out as automatic). If a
    future session's real eval-time numbers show frac_mass_outside_trajectory_bbox
    correlating with known-bad runs (e.g. the ScanNet++ plateau) across
    several scenes, that's the validation bar for adding a live, periodic
    (not per-iteration) hook -- see the plan doc for how.
    """
    from scipy.spatial import cKDTree

    pts_all = _to_np(gaussian_points).reshape(-1, 3)
    op_all = _to_np(opacities).reshape(-1)
    traj = _to_np(trajectory_positions).reshape(-1, 3)

    n_total = int(pts_all.shape[0])
    out = {"valid": False, "margin_m": float(margin_m), "opacity_threshold": float(opacity_threshold),
           "n_total": n_total}
    if n_total == 0 or traj.shape[0] == 0:
        out["reason"] = "empty inputs"
        return out

    live = op_all > float(opacity_threshold)
    pts = pts_all[live]
    op = op_all[live]
    out["n_live"] = int(pts.shape[0])
    if pts.shape[0] == 0:
        out["reason"] = "no live mass above opacity_threshold"
        return out

    scene_bbox_min = pts.min(axis=0)
    scene_bbox_max = pts.max(axis=0)
    scene_vol = float(np.prod(np.maximum(scene_bbox_max - scene_bbox_min, 0.0)))

    traj_bbox_min = traj.min(axis=0)
    traj_bbox_max = traj.max(axis=0)
    traj_vol = float(np.prod(np.maximum(traj_bbox_max - traj_bbox_min, 0.0)))

    out["scene_bbox_volume_m3"] = scene_vol
    out["trajectory_bbox_volume_m3"] = traj_vol
    out["bbox_volume_ratio"] = float(scene_vol / traj_vol) if traj_vol > 1e-9 else float("inf")

    margin_min = traj_bbox_min - float(margin_m)
    margin_max = traj_bbox_max + float(margin_m)
    inside = np.all((pts >= margin_min) & (pts <= margin_max), axis=1)
    outside = ~inside
    denom = float(op.sum()) + 1e-8
    out["frac_mass_outside_trajectory_bbox"] = float(op[outside].sum() / denom)
    out["frac_count_outside_trajectory_bbox"] = float(outside.mean())

    kdtree = cKDTree(traj)
    dists, _ = kdtree.query(pts, k=1)

    out["max_distance_from_trajectory_m"] = float(np.nanmax(dists))
    out["distance_percentiles"] = _pcts(dists, ps=(50, 90, 99))
    out["valid"] = True
    return out


def sensor_points_from_iq_vox(iq_vox: dict):
    """Decode ``ObservationStore.iq_vox``'s per-voxel Welford accumulators
    (``{voxel_key: [n, sx, sy, sz, sxx, syy, szz, sxy, sxz, syz]}``, built by
    ``StreamingScene._accumulate_thickness`` on every ingested frame when
    ``--streaming_track_input_quality`` is on, the default) into an ``[M, 3]``
    array of per-voxel mean sensor positions.

    This is a REAL, independent-of-the-Gaussians point cloud (it comes from
    backprojected sensor depth, accumulated regardless of what the Gaussian
    model does with that geometry) suitable as Tier D's ``sensor_points``
    without needing a separate ``--reference_map``. Returns an empty
    ``(0, 3)`` array if ``iq_vox`` is empty (e.g. input-quality tracking was
    disabled, or no frames have been ingested yet).
    """
    import numpy as np
    if not iq_vox:
        return np.zeros((0, 3), dtype=np.float32)
    sums = np.asarray(list(iq_vox.values()), dtype=np.float64)
    n = sums[:, 0]
    return (sums[:, 1:4] / n[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Tier D — chamfer / off-surface floater mass
# ---------------------------------------------------------------------------
def compute_offsurface_metrics(
    gaussian_xyz,
    opacity,
    sensor_points,
    tau: float = 0.05,
    max_gaussians: int = 400000,
    max_sensor: int = 400000,
    return_distances: bool = False,
) -> dict:
    """Tier D — opacity-weighted off-surface / floater mass vs a sensor point cloud.

    For each Gaussian centre, the nearest sensor point is found (exact cross-set 1-NN
    via the simple_knn ``knnIndices`` kernel — no Gaussians x sensor matrix; falls
    back to chunked cdist when the extension is absent). The headline metric is the
    opacity-weighted fraction of Gaussian mass whose nearest sensor point is farther
    than ``tau`` metres (off-surface = floaters), plus NN-distance percentiles.

    Args:
      gaussian_xyz : [N,3] Gaussian centres (CUDA float).
      opacity      : [N] activated opacities (CUDA float).
      sensor_points: [M,3] world-space sensor points (CUDA float); e.g. backprojected
                     sensor depth across observed frames, voxel-downsampled.
      tau          : off-surface distance threshold in metres.
      return_distances: if True, also return the (possibly subsampled) ``xyz``/
                     ``opacity``/per-point NN distance tensors used, under the
                     ``_xyz``/``_alpha``/``_nn_dist`` keys, for callers that need to
                     derive further metrics (e.g. F-score at other thresholds, or
                     partitioning points by distance) without repeating the KNN
                     query. Default False keeps the return dict identical to before
                     this flag existed.
    Returns a dict (empty/zeroed if inputs are degenerate).
    """
    import torch
    if gaussian_xyz is None or sensor_points is None:
        return {"valid": False, "reason": "missing inputs"}
    if gaussian_xyz.shape[0] == 0 or sensor_points.shape[0] == 0:
        return {"valid": False, "reason": "empty inputs"}

    xyz = gaussian_xyz
    alpha = opacity.reshape(-1)
    # Deterministic subsample to bound work at very large N (linspace = reproducible).
    if xyz.shape[0] > max_gaussians:
        idx = torch.linspace(0, xyz.shape[0] - 1, steps=max_gaussians, device=xyz.device).long()
        xyz = xyz[idx]; alpha = alpha[idx]
    sp = sensor_points
    if sp.shape[0] > max_sensor:
        idx = torch.linspace(0, sp.shape[0] - 1, steps=max_sensor, device=sp.device).long()
        sp = sp[idx]

    from eligsir.utils.rocm_knn_fallback import knn_indices
    dist2, _ = knn_indices(xyz.float(), sp.float(), k=1)  # cross-set 1-NN
    nn = dist2[:, 0].clamp_min(0).sqrt()

    off = nn > float(tau)
    denom = alpha.sum() + 1e-8
    offsurface_mass = float(alpha[off].sum() / denom)
    q = torch.quantile(nn, torch.tensor([0.5, 0.9, 0.99], device=nn.device))
    out = {
        "valid": True,
        "tau_m": float(tau),
        "offsurface_mass_frac": offsurface_mass,
        "offsurface_count_frac": float(off.float().mean()),
        "nn_dist_p50_m": float(q[0]),
        "nn_dist_p90_m": float(q[1]),
        "nn_dist_p99_m": float(q[2]),
        "nn_dist_mean_m": float(nn.mean()),
        "n_gaussians": int(xyz.shape[0]),
        "n_sensor_points": int(sp.shape[0]),
    }
    if return_distances:
        out["_xyz"] = xyz
        out["_alpha"] = alpha
        out["_nn_dist"] = nn
    return out
