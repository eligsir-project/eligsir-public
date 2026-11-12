"""Compare a Gaussian-splat map against an independent, non-neural
point-cloud-only map (e.g. an Open3D TSDF reconstruction from embed_slam).

Reuses ``utils/scene_geometry_metrics.py``'s existing Tier D off-surface
metric (``compute_offsurface_metrics``) and voxel-occupancy metric
(``compute_voxel_occupancy``) unmodified -- this module only adds the
reference-map loading, optional ICP registration, and symmetric-direction
orchestration those functions don't do themselves.

See CLAUDE.md's "Gaussian vs. point-cloud-only map comparison" section
for the full cross-repo workflow (rosbag-colmap-pipeline -> embed_slam ->
eligsir -> this module) that makes a comparison here apples-to-apples
(same pseudo-GT trajectory feeding both maps).
"""
from __future__ import annotations

import numpy as np

from eligsir.plot_style import svg_output_path
from eligsir.utils.scene_geometry_metrics import compute_offsurface_metrics, compute_voxel_occupancy


def render_map_overlay(out_path: str, gaussian_xyz: np.ndarray, reference_xyz: np.ndarray,
                        dpi: int = 400, point_size: float = 0.15) -> None:
    """Three-panel (top-down + 2 side views) scatter overlay of the Gaussian
    map vs. the reference map, for visual inspection alongside the JSON
    metrics. The points remain vector artists in the SVG so zooming preserves
    point-level structure without embedding a raster image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    views = [(0, 1, "top-down (X-Y)"), (0, 2, "side (X-Z)"), (1, 2, "side (Y-Z)")]
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    for ax, (i, j, label) in zip(axes, views):
        ax.scatter(reference_xyz[:, i], reference_xyz[:, j], s=point_size, c="tab:blue",
                   alpha=0.3, label="reference map")
        ax.scatter(gaussian_xyz[:, i], gaussian_xyz[:, j], s=point_size, c="tab:red",
                   alpha=0.3, label="Gaussian map")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.legend(markerscale=40, fontsize=8)
    fig.tight_layout()
    fig.savefig(svg_output_path(out_path), dpi=dpi)
    plt.close(fig)


def load_reference_pointcloud(path: str, mesh_sample_points: int = 2_000_000):
    """Load a reference map (.ply mesh or point cloud) as an open3d PointCloud.

    Mesh vs. point-cloud is auto-detected by triangle count. A mesh is
    densified via Poisson-disk sampling so NN-distance metrics against it
    aren't biased by uneven vertex density (e.g. large flat wall triangles
    vs. densely-triangulated clutter).
    """
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        n_sample = min(int(mesh_sample_points), max(len(mesh.vertices) * 4, 1))
        return mesh.sample_points_poisson_disk(n_sample)

    return o3d.io.read_point_cloud(path)


def align_point_clouds(source_xyz: np.ndarray, target_xyz: np.ndarray, voxel_size: float = 0.05):
    """Register source onto target via centroid init + point-to-point ICP.

    No scale correction is applied -- both maps are nominally metric, so a
    large residual scale mismatch after ICP is itself a diagnostic (the two
    maps likely weren't built from the same pose source), not something to
    silently paper over.

    Returns (transform_4x4, fitness, inlier_rmse). A low fitness / high rmse
    means the alignment is not trustworthy -- callers must check this rather
    than assume registration succeeded.
    """
    import open3d as o3d

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_xyz.astype(np.float64))
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_xyz.astype(np.float64))

    init = np.eye(4)
    init[:3, 3] = target_xyz.mean(axis=0) - source_xyz.mean(axis=0)

    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=float(voxel_size) * 15.0,
        init=init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return np.asarray(result.transformation), float(result.fitness), float(result.inlier_rmse)


def _apply_transform(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
    return (homo @ transform.T)[:, :3]


F_SCORE_THRESHOLDS_M = (0.02, 0.05, 0.1, 0.2)


def _f_score_curve(gaussian_nn_dist, reference_nn_dist, gaussian_alpha, thresholds=F_SCORE_THRESHOLDS_M) -> dict:
    """Precision/recall/F-score at several distance thresholds (Tanks & Temples convention).

    precision(tau) = opacity-weighted fraction of Gaussian mass within tau of the
    reference (== 1 - gaussian_to_reference offsurface_mass_frac at that tau).
    recall(tau) = fraction of reference points within tau of a Gaussian.
    Reuses the NN-distance arrays already computed by compute_offsurface_metrics
    (return_distances=True) instead of re-running KNN per threshold.
    """
    import torch

    denom = gaussian_alpha.sum() + 1e-8
    curve = {}
    for tau in thresholds:
        precision = float(gaussian_alpha[gaussian_nn_dist <= tau].sum() / denom)
        recall = float((reference_nn_dist <= tau).float().mean())
        f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        curve[f"{tau:g}"] = {"precision": precision, "recall": recall, "f_score": f_score}
    return curve


def _floater_volume(gaussian_xyz_np, gaussian_nn_dist, tau_m: float, voxel_size: float) -> dict:
    """Voxel-occupancy of the Gaussian subset whose nearest reference point is
    farther than tau_m -- a genuine floater VOLUME figure, distinct from
    bbox_volume_ratio (conflates big-real-room with diffuse floater cloud)."""
    floater_mask = (gaussian_nn_dist.detach().cpu().numpy() > tau_m)
    floater_xyz = gaussian_xyz_np[floater_mask]
    floater_occupancy = compute_voxel_occupancy(floater_xyz, voxel_size)
    return {
        "n_floater_gaussians": int(floater_xyz.shape[0]),
        "floater_frac_count": float(floater_mask.mean()) if floater_mask.size else 0.0,
        "floater_volume_m3": floater_occupancy["occupied_volume_m3"],
        "floater_occupancy": floater_occupancy,
    }


def compare_gaussian_vs_reference_map(
    gaussians,
    reference_pcd,
    min_opacity: float = 0.05,
    tau_m: float = 0.05,
    align: bool = True,
    voxel_size: float = 0.05,
) -> dict:
    """Full comparison report: Gaussian map vs. an independent reference map.

    Args:
        gaussians: a loaded GaussianModel/GsplatGaussianModel (get_xyz/get_opacity).
        reference_pcd: an open3d.geometry.PointCloud (see load_reference_pointcloud).
        min_opacity: Gaussians below this activated opacity are dropped before comparison.
        tau_m: off-surface distance threshold in metres (passed through to
            compute_offsurface_metrics, and used for the floater-volume split).
        align: run ICP registration of the reference map onto the Gaussian map before
            comparing. Set False only when both maps are already known to share the
            same pose source (e.g. both built from the same /gt/colmap_pose bag).
        voxel_size: used for both ICP correspondence distance and voxel-occupancy.
    """
    import torch

    with torch.no_grad():
        xyz = gaussians.get_xyz
        opacity = gaussians.get_opacity
        if opacity.dim() > 1:
            opacity = opacity.squeeze(-1)
        keep = opacity >= float(min_opacity)
        xyz = xyz[keep]
        opacity = opacity[keep]

    gaussian_xyz_np = xyz.detach().float().cpu().numpy()
    reference_xyz_np = np.asarray(reference_pcd.points, dtype=np.float32)

    out = {
        "min_opacity": float(min_opacity),
        "tau_m": float(tau_m),
        "n_gaussians_kept": int(gaussian_xyz_np.shape[0]),
        "n_reference_points": int(reference_xyz_np.shape[0]),
    }

    if gaussian_xyz_np.shape[0] == 0 or reference_xyz_np.shape[0] == 0:
        out["valid"] = False
        out["reason"] = "empty inputs"
        return out

    if align:
        transform, fitness, rmse = align_point_clouds(reference_xyz_np, gaussian_xyz_np, voxel_size=voxel_size)
        reference_xyz_np = _apply_transform(reference_xyz_np, transform)
        out["alignment"] = {
            "applied": True,
            "fitness": fitness,
            "inlier_rmse": rmse,
            "transform": transform.tolist(),
        }
    else:
        out["alignment"] = {"applied": False, "reason": "align=False"}

    device = xyz.device
    reference_xyz_t = torch.from_numpy(reference_xyz_np).float().to(device)
    reference_opacity_t = torch.ones(reference_xyz_t.shape[0], dtype=torch.float32, device=device)

    gaussian_to_reference = compute_offsurface_metrics(
        xyz, opacity, reference_xyz_t, tau=tau_m, return_distances=True)
    reference_to_gaussian = compute_offsurface_metrics(
        reference_xyz_t, reference_opacity_t, xyz, tau=tau_m, return_distances=True)

    gaussian_nn_dist = gaussian_to_reference.pop("_nn_dist", None)
    gaussian_nn_alpha = gaussian_to_reference.pop("_alpha", None)
    gaussian_nn_xyz = gaussian_to_reference.pop("_xyz", None)
    reference_nn_dist = reference_to_gaussian.pop("_nn_dist", None)
    reference_to_gaussian.pop("_alpha", None)
    reference_to_gaussian.pop("_xyz", None)

    out["gaussian_to_reference"] = gaussian_to_reference
    out["reference_to_gaussian"] = reference_to_gaussian

    if gaussian_to_reference.get("valid") and reference_to_gaussian.get("valid"):
        out["symmetric_chamfer_m"] = 0.5 * (
            gaussian_to_reference["nn_dist_mean_m"] + reference_to_gaussian["nn_dist_mean_m"]
        )
        out["f_score_curve"] = _f_score_curve(gaussian_nn_dist, reference_nn_dist, gaussian_nn_alpha)
        # _xyz mirrors whatever (possibly deterministic-subsampled) set nn_dist was
        # computed against -- must use it, not the full gaussian_xyz_np, to stay
        # index-aligned with gaussian_nn_dist.
        gaussian_nn_xyz_np = gaussian_nn_xyz.detach().cpu().numpy()
        out["floater_volume"] = _floater_volume(gaussian_nn_xyz_np, gaussian_nn_dist, tau_m, voxel_size)

    gaussian_occupancy = compute_voxel_occupancy(gaussian_xyz_np, voxel_size)
    reference_occupancy = compute_voxel_occupancy(reference_xyz_np, voxel_size)
    out["gaussian_occupancy"] = gaussian_occupancy
    out["reference_occupancy"] = reference_occupancy
    if reference_occupancy["bbox_volume_m3"] > 1e-9:
        out["bbox_volume_ratio"] = gaussian_occupancy["bbox_volume_m3"] / reference_occupancy["bbox_volume_m3"]
    if reference_occupancy["occupied_volume_m3"] > 1e-9:
        out["occupied_volume_ratio"] = (
            gaussian_occupancy["occupied_volume_m3"] / reference_occupancy["occupied_volume_m3"]
        )

    out["valid"] = True
    return out
