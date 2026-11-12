"""Trajectory evaluation utilities.

Saves estimated camera trajectories in TUM format, produces PNG plots,
and computes ATE/RPE metrics (with optional GT alignment via Umeyama SE(3)).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Sequence

import numpy as np

from eligsir.plot_style import svg_output_path


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    m = R
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to 3×3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# GT trajectory auto-discovery
# ---------------------------------------------------------------------------

def _materialize_scannet_gt(source_path: str) -> Optional[str]:
    """Convert ScanNet per-frame pose/*.txt into a TUM-format GT file."""
    pose_dir = os.path.join(source_path, "pose")
    color_dir = os.path.join(source_path, "color")
    if not (os.path.isdir(pose_dir) and os.path.isdir(color_dir)):
        return None

    cache_dir = os.path.join(source_path, ".cache")
    out_path = os.path.join(cache_dir, "groundtruth_tum.txt")
    if os.path.exists(out_path):
        return out_path

    # Import the torch-free reader (utils/scannet_pose_io.py), NOT
    # scene.readers.scannet -- that module imports torch transitively, and the
    # bare `except` below would turn a missing torch into a silent "no ScanNet
    # GT here" instead of an error (issue #449, same class as #434/#435).
    try:
        from eligsir.utils.scannet_pose_io import _read_scannet_pose
    except Exception:
        return None

    pose_files = sorted(
        f for f in os.listdir(pose_dir)
        if f.endswith(".txt") and f.removesuffix(".txt").isdigit()
    )
    if not pose_files:
        return None

    ts_list, c2w_list = [], []
    for pf in pose_files:
        fid = pf.removesuffix(".txt")
        c2w = _read_scannet_pose(os.path.join(pose_dir, pf))
        if c2w is None:
            continue
        ts_list.append(float(fid))
        c2w_list.append(np.asarray(c2w, dtype=np.float64))
    if not c2w_list:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    save_tum_trajectory(out_path, np.asarray(ts_list), np.stack(c2w_list))
    return out_path


def _materialize_replica_gt(source_path: str) -> Optional[str]:
    """Convert Replica traj.txt (one 4×4 per line, 16 floats) into TUM-format GT."""
    traj = os.path.join(source_path, "traj.txt")
    if not os.path.exists(traj):
        return None

    cache_dir = os.path.join(source_path, ".cache")
    out_path = os.path.join(cache_dir, "groundtruth_tum.txt")
    if os.path.exists(out_path):
        return out_path

    ts_list, c2w_list = [], []
    with open(traj) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 16:
                continue
            T = np.asarray([float(x) for x in parts], dtype=np.float64).reshape(4, 4)
            ts_list.append(float(i))
            c2w_list.append(T)
    if not c2w_list:
        return None

    os.makedirs(cache_dir, exist_ok=True)
    save_tum_trajectory(out_path, np.asarray(ts_list), np.stack(c2w_list))
    return out_path


def discover_gt_trajectory(source_path: str) -> Optional[str]:
    """Return a TUM-format GT file path for the input source, or None.

    Materializes a cached `.cache/groundtruth_tum.txt` for ScanNet / Replica
    sources on first call. RealSense / Orbbec bag inputs have no GT and
    return None.
    """
    if not source_path:
        return None
    source_path = os.path.abspath(source_path)

    # Non-directory inputs (e.g., .bag files) have no GT.
    if not os.path.isdir(source_path):
        return None

    # TUM RGB-D: groundtruth.txt in source, or one level up.
    for cand_dir in (source_path, os.path.dirname(source_path)):
        cand = os.path.join(cand_dir, "groundtruth.txt")
        if os.path.exists(cand):
            return cand

    # ScanNet
    scannet = _materialize_scannet_gt(source_path)
    if scannet:
        return scannet

    # Replica
    replica = _materialize_replica_gt(source_path)
    if replica:
        return replica

    return None


# ---------------------------------------------------------------------------
# TUM I/O
# ---------------------------------------------------------------------------

def save_tum_trajectory(out_path: str, timestamps_sec: np.ndarray, c2w_arr: np.ndarray) -> None:
    """Save N poses to TUM format: t tx ty tz qx qy qz qw per line."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# TUM trajectory — t tx ty tz qx qy qz qw\n")
        for t, T in zip(timestamps_sec, c2w_arr):
            xyz = T[:3, 3]
            q = _rot_to_quat(T[:3, :3])
            f.write(f"{t:.9f} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")


def load_tum_trajectory(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load TUM file, return (timestamps_sec [N], c2w_arr [N,4,4])."""
    ts, mats = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            t = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = _quat_to_rot(np.array([qx, qy, qz, qw]))
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]
            ts.append(t)
            mats.append(T)
    return np.array(ts), np.stack(mats)


def associate_by_timestamp(
    ts_est: np.ndarray,
    c2w_est: np.ndarray,
    ts_ref: np.ndarray,
    c2w_ref: np.ndarray,
    *,
    max_dt: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Nearest-neighbor timestamp association for trajectory metrics."""
    if len(ts_est) == 0 or len(ts_ref) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            {"n_matches": 0, "max_dt": float(max_dt)},
        )
    ref_order = np.argsort(ts_ref)
    ts_ref_sorted = ts_ref[ref_order]
    c2w_ref_sorted = c2w_ref[ref_order]
    est_out, ref_out, ts_out, dts = [], [], [], []
    for t, pose in zip(ts_est, c2w_est):
        idx = int(np.searchsorted(ts_ref_sorted, t))
        candidates = []
        if idx < len(ts_ref_sorted):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            continue
        best = min(candidates, key=lambda i: abs(float(ts_ref_sorted[i]) - float(t)))
        dt = float(ts_ref_sorted[best]) - float(t)
        if abs(dt) > max_dt:
            continue
        ts_out.append(float(t))
        est_out.append(pose)
        ref_out.append(c2w_ref_sorted[best])
        dts.append(dt)
    if not est_out:
        return (
            np.asarray([], dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0, 4, 4), dtype=np.float64),
            {"n_matches": 0, "max_dt": float(max_dt)},
        )
    dts_np = np.asarray(dts, dtype=np.float64)
    info = {
        "n_matches": len(est_out),
        "max_dt": float(max_dt),
        "dt_mean": float(dts_np.mean()),
        "dt_abs_p95": float(np.percentile(np.abs(dts_np), 95)),
        "dt_abs_max": float(np.max(np.abs(dts_np))),
    }
    return (
        np.asarray(ts_out, dtype=np.float64),
        np.stack(est_out).astype(np.float64),
        np.stack(ref_out).astype(np.float64),
        info,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trajectory(
    out_path: str,
    c2w_arr: np.ndarray,
    *,
    title: Optional[str] = None,
    gt: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Top-down + side view of one trajectory, optional GT overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Align estimated → GT via Umeyama SE(3) so the plot reflects the metric
    # (raw trajectories live in different conventions: Open3D-first-camera vs.
    # TUM world; without alignment they appear offset and rotation-flipped).
    plot_c2w = c2w_arr
    if gt is not None and len(c2w_arr) >= 2 and len(gt) >= 2:
        n = min(len(c2w_arr), len(gt))
        T_align = _umeyama_align(c2w_arr[:n, :3, 3], gt[:n, :3, 3])
        plot_c2w = c2w_arr.copy()
        plot_c2w[:, :3, 3] = (T_align[:3, :3] @ c2w_arr[:, :3, 3].T).T + T_align[:3, 3]

    xyz = plot_c2w[:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _plot_2d(axes[0], xyz[:, 0], xyz[:, 2], "X", "Z (forward)", "Top-down (XZ, aligned)", gt=gt)
    _plot_2d(axes[1], xyz[:, 0], xyz[:, 1], "X", "Y (up)", "Side view (XY, aligned)", gt=gt, invert_y=True)
    if title:
        fig.suptitle(title, fontsize=12)
    if metrics:
        _add_metrics_box(fig, metrics)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(svg_output_path(out_path), dpi=120)
    plt.close(fig)


def mapped_camera_trajectory_selection(
    trajectory_rows: Sequence[dict],
    decision_rows: Sequence[dict],
    *,
    successful_supervision_uses: Optional[dict[int, int]] = None,
) -> dict:
    """Summarize trace-proven mapping use against the source trajectory."""
    trajectory_ordinals = [int(row["id"]) for row in trajectory_rows]
    known = set(trajectory_ordinals)
    candidates: set[int] = set()
    supervision_uses: dict[int, int] = (
        {}
        if successful_supervision_uses is None
        else {int(ordinal): int(uses) for ordinal, uses in successful_supervision_uses.items()}
    )
    for row in decision_rows:
        candidates.update(
            int(disposition["source_ordinal"])
            for disposition in row.get("dispositions", ())
            if disposition.get("mapping_candidate")
        )
        if (
            successful_supervision_uses is None
            and bool(row.get("executed"))
            and int(row.get("optimizer_steps", 0)) > 0
        ):
            for source in row.get("supervision_source_records", ()):
                ordinal = int(source["source_ordinal"])
                supervision_uses[ordinal] = supervision_uses.get(ordinal, 0) + 1
    supervised = set(supervision_uses)
    return {
        "trajectory_ordinals": trajectory_ordinals,
        "mapping_candidate_ordinals": sorted(candidates & known),
        "successfully_supervised_ordinals": sorted(supervised & known),
        "successful_supervision_uses": {
            str(ordinal): supervision_uses[ordinal]
            for ordinal in sorted(supervised & known)
        },
        "unknown_mapping_candidate_ordinals": sorted(candidates - known),
        "unknown_successfully_supervised_ordinals": sorted(supervised - known),
    }


def mapped_camera_trajectory_display_coordinates(
    positions: np.ndarray,
) -> np.ndarray:
    """Return the world-horizontal X/Y coordinates for the mapped-path plot."""
    return np.asarray(positions, dtype=np.float64)[:, :2]


def plot_mapped_camera_trajectory(
    out_path: str,
    trajectory_rows: Sequence[dict],
    decision_rows: Sequence[dict],
    *,
    successful_supervision_uses: Optional[dict[int, int]] = None,
) -> dict:
    """Plot the submitted path with trace-proven candidate and used cameras."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    selection = mapped_camera_trajectory_selection(
        trajectory_rows,
        decision_rows,
        successful_supervision_uses=successful_supervision_uses,
    )
    positions = {
        int(row["id"]): np.asarray(row["position"], dtype=np.float64)
        for row in trajectory_rows
    }
    trajectory = np.stack(
        [positions[ordinal] for ordinal in selection["trajectory_ordinals"]]
    )
    trajectory_xy = mapped_camera_trajectory_display_coordinates(trajectory)
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.plot(
        trajectory_xy[:, 0], trajectory_xy[:, 1], color="0.65", linewidth=0.8,
        alpha=0.8, label=f"source trajectory ({len(trajectory)})",
    )
    candidates = selection["mapping_candidate_ordinals"]
    if candidates:
        points = mapped_camera_trajectory_display_coordinates(
            np.stack([positions[ordinal] for ordinal in candidates])
        )
        axis.scatter(
            points[:, 0], points[:, 1], s=9, color="#f59e0b", alpha=0.55,
            label=f"mapping candidate ({len(candidates)})", zorder=2,
        )
    supervised = selection["successfully_supervised_ordinals"]
    if supervised:
        points = mapped_camera_trajectory_display_coordinates(
            np.stack([positions[ordinal] for ordinal in supervised])
        )
        uses = np.asarray(
            [selection["successful_supervision_uses"][str(ordinal)] for ordinal in supervised]
        )
        if uses.min() == uses.max():
            axis.scatter(
                points[:, 0], points[:, 1], s=42, color="#059669",
                edgecolors="white", linewidths=0.35,
                label=f"supervised cameras ({len(supervised)}; {uses[0]} use each)",
                zorder=3,
            )
        else:
            sizes = 10.0 + 80.0 * np.log(uses) / np.log(uses.max())
            scatter = axis.scatter(
                points[:, 0], points[:, 1], c=uses, s=sizes, cmap="viridis",
                norm=LogNorm(vmin=uses.min(), vmax=uses.max()), alpha=0.9,
                edgecolors="white", linewidths=0.3,
                label=f"supervised cameras ({len(supervised)})", zorder=3,
            )
            colorbar = fig.colorbar(scatter, ax=axis, fraction=0.045, pad=0.02)
            colorbar.set_label("successful supervision uses", fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
    axis.set(title="Mapped camera trajectory (top-down)", xlabel="X (m)", ylabel="Y (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best", fontsize=7, framealpha=0.9, markerscale=0.8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(svg_output_path(out_path), dpi=150)
    plt.close(fig)
    return selection


def plot_trajectory_pair(
    out_path: str,
    c2w_a: np.ndarray,
    c2w_b: np.ndarray,
    *,
    labels: tuple[str, str] = ("A", "B"),
    title: Optional[str] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Overlay two trajectories for comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz_a = c2w_a[:, :3, 3]
    xyz_b = c2w_b[:, :3, 3]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, xkey, ykey, xlabel, ylabel, view_title in [
        (axes[0], 0, 2, "X", "Z", "Top-down (XZ)"),
        (axes[1], 0, 1, "X", "Y", "Side view (XY)"),
    ]:
        ax.plot(xyz_a[:, xkey], xyz_a[:, ykey], lw=1.5, label=labels[0], color="steelblue")
        ax.plot(xyz_b[:, xkey], xyz_b[:, ykey], lw=1.5, label=labels[1], color="tomato", linestyle="--")
        ax.scatter(xyz_a[0, xkey], xyz_a[0, ykey], marker="o", s=60, color="steelblue", zorder=5)
        ax.scatter(xyz_b[0, xkey], xyz_b[0, ykey], marker="o", s=60, color="tomato", zorder=5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(view_title)
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        if ykey == 1:
            ax.invert_yaxis()
    if title:
        fig.suptitle(title, fontsize=12)
    if metrics:
        _add_metrics_box(fig, metrics)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(svg_output_path(out_path), dpi=120)
    plt.close(fig)


def _plot_2d(ax, xs, ys, xlabel, ylabel, title, *, gt=None, invert_y=False):
    ax.plot(xs, ys, lw=1.5, label="estimated", color="steelblue")
    ax.scatter(xs[0], ys[0], marker="o", s=60, color="steelblue", zorder=5)
    if gt is not None:
        gx, gy = gt[:, :3, 3][:, 0], gt[:, :3, 3][:, 2 if ylabel.startswith("Z") else 1]
        ax.plot(gx, gy, lw=1.5, linestyle="--", label="GT", color="tomato")
        ax.scatter(gx[0], gy[0], marker="o", s=60, color="tomato", zorder=5)
        ax.legend(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal")
    if invert_y:
        ax.invert_yaxis()


def _trajectory_distance(c2w_arr: np.ndarray) -> float:
    xyz = c2w_arr[:, :3, 3]
    if len(xyz) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


# A real out-and-back (or single-loop) capture's summed consecutive-frame
# distance should land within a small multiple of its bounding-box diagonal
# (~2x for a clean there-and-back walk). A much larger ratio means the
# "distance" figure is not measuring physical path length -- either the
# per-frame order fed in isn't chronological (so consecutive array entries
# are not consecutive in space/time) or the underlying poses themselves are
# unreliable (e.g. a degenerate/rank-deficient SfM reconstruction). Found via
# a real bug (2026-08-12): a RealityScan Mobile hallway capture reported
# "distance: 2213 m" for what was physically a ~30-60m walk -- root-caused to
# BOTH mechanisms at once (see docs/architecture/2026-08-12-realityscan-
# trajectory-distance-bug.md): `sfm.txt` registration-order fallback turned
# out uncorrelated with true capture order (Spearman rho ~0.03), AND the
# camera positions themselves were 99.97%-collinear (rank-1, via SVD) with
# large ~random magnitude along that one axis -- the classic monocular-SfM
# "forward motion" / bas-relief degeneracy for a mostly-straight walk with
# little lateral parallax. Neither mechanism alone is fixable by relabeling
# the metric; both are real, and this detector is deliberately mechanism-
# agnostic (it flags "this number is not a path length" regardless of which
# caused it), plus a rank/collinearity check to name the SfM-degeneracy case
# specifically when it applies.
_IMPLAUSIBLE_DISTANCE_RATIO = 8.0
# A well-conditioned trajectory's dominant SVD singular value should not
# explain nearly all position variance -- a real 2D/3D path (a room, a
# hallway with any turn or width) has some extent off its dominant axis.
_DEGENERATE_COLLINEARITY_FRAC = 0.999


def _trajectory_plausibility(c2w_arr: np.ndarray) -> dict:
    """Order-independent sanity metrics for a trajectory's reported distance.

    Returns bbox diagonal, the summed consecutive-frame distance / bbox
    diagonal ratio, max/median consecutive step, and a collinearity fraction
    (top singular value's share of total position variance, via SVD) that
    flags a rank-deficient/degenerate pose set. See `_trajectory_distance`'s
    docstring above for the real incident this was built to catch.
    """
    xyz = c2w_arr[:, :3, 3]
    n = len(xyz)
    out = {
        "bbox_diagonal_m": None,
        "distance_over_bbox_diagonal": None,
        "max_step_m": None,
        "median_step_m": None,
        "collinearity_frac": None,
        "implausible_distance": False,
        "degenerate_poses": False,
    }
    if n < 2:
        return out

    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    distance = float(steps.sum())

    out["bbox_diagonal_m"] = bbox_diag
    out["max_step_m"] = float(steps.max())
    out["median_step_m"] = float(np.median(steps))

    if bbox_diag > 1e-9:
        ratio = distance / bbox_diag
        out["distance_over_bbox_diagonal"] = ratio
        out["implausible_distance"] = bool(ratio > _IMPLAUSIBLE_DISTANCE_RATIO)

    if n >= 3:
        try:
            centered = xyz - xyz.mean(axis=0)
            singular_values = np.linalg.svd(centered, compute_uv=False)
            total_var = float((singular_values ** 2).sum())
            if total_var > 1e-12:
                collinearity_frac = float(singular_values[0] ** 2 / total_var)
                out["collinearity_frac"] = collinearity_frac
                out["degenerate_poses"] = bool(collinearity_frac > _DEGENERATE_COLLINEARITY_FRAC)
        except np.linalg.LinAlgError:
            pass

    return out


def _fmt_metric(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        if not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.4g}{suffix}"
    except Exception:
        return "n/a"


def _add_metrics_box(fig, metrics: dict) -> None:
    lines = [
        f"distance: {_fmt_metric(metrics.get('distance_m'), ' m')}",
        f"ATE RMSE: {_fmt_metric(metrics.get('ate_rmse'), ' m')}",
        f"ATE mean: {_fmt_metric(metrics.get('ate_mean'), ' m')}",
        f"RPE trans: {_fmt_metric(metrics.get('rpe_trans_rmse'), ' m')}",
        f"RPE rot: {_fmt_metric(metrics.get('rpe_rot_deg_rmse'), ' deg')}",
    ]
    if metrics.get("gt_distance_m") is not None:
        lines.insert(1, f"GT distance: {_fmt_metric(metrics.get('gt_distance_m'), ' m')}")
    implausible = metrics.get("implausible_distance")
    degenerate = metrics.get("degenerate_poses")
    if implausible or degenerate:
        reasons = []
        if implausible:
            reasons.append("distance/bbox=" + _fmt_metric(metrics.get("distance_over_bbox_diagonal")))
        if degenerate:
            reasons.append("collinearity=" + _fmt_metric(metrics.get("collinearity_frac")))
        lines.append("⚠ distance_m UNRELIABLE (" + ", ".join(reasons) + ")")
        box_kwargs = {"boxstyle": "round,pad=0.35", "facecolor": "#ffe8e0", "alpha": 0.9, "edgecolor": "#cc4400"}
    else:
        box_kwargs = {"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"}
    fig.text(
        0.01,
        0.01,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=9,
        family="monospace",
        bbox=box_kwargs,
    )


# ---------------------------------------------------------------------------
# Umeyama SE(3) alignment (no scale)
# ---------------------------------------------------------------------------

def _umeyama_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align src positions [N,3] to dst [N,3] via SE(3) least-squares.

    Returns 4×4 transform T such that T @ src ≈ dst.
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.T @ dst_c / n
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = mu_dst - R @ mu_src
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# ATE / RPE metrics
# ---------------------------------------------------------------------------

def trajectory_metrics(c2w_est: np.ndarray, c2w_ref: np.ndarray) -> dict:
    """Compute ATE and RPE between estimated and reference trajectories.

    Both arrays are [N,4,4] world-from-camera matrices.
    Aligns est to ref via Umeyama SE(3) before computing ATE.
    Returns dict with ate_rmse, ate_mean, rpe_trans_rmse, rpe_rot_deg_rmse, n_frames.
    """
    n = min(len(c2w_est), len(c2w_ref))
    if n < 2:
        return {"ate_rmse": None, "ate_mean": None,
                "rpe_trans_rmse": None, "rpe_rot_deg_rmse": None, "n_frames": n,
                "distance_m": _trajectory_distance(c2w_est[:n]),
                "gt_distance_m": _trajectory_distance(c2w_ref[:n])}

    pos_est = c2w_est[:n, :3, 3]
    pos_ref = c2w_ref[:n, :3, 3]

    T_align = _umeyama_align(pos_est, pos_ref)
    pos_est_aligned = (T_align[:3, :3] @ pos_est.T).T + T_align[:3, 3]

    ate_err = np.linalg.norm(pos_est_aligned - pos_ref, axis=1)
    ate_rmse = float(np.sqrt(np.mean(ate_err ** 2)))
    ate_mean = float(np.mean(ate_err))

    # RPE: relative pose error between consecutive frames
    trans_errs, rot_errs = [], []
    R_align = T_align[:3, :3]
    for i in range(n - 1):
        # relative pose in reference
        dT_ref = np.linalg.inv(c2w_ref[i]) @ c2w_ref[i + 1]
        # relative pose in aligned estimate
        c2w_est_i = c2w_est[i].copy()
        c2w_est_i[:3, 3] = pos_est_aligned[i]
        c2w_est_i[:3, :3] = R_align @ c2w_est[i, :3, :3]
        c2w_est_i1 = c2w_est[i + 1].copy()
        c2w_est_i1[:3, 3] = pos_est_aligned[i + 1]
        c2w_est_i1[:3, :3] = R_align @ c2w_est[i + 1, :3, :3]
        dT_est = np.linalg.inv(c2w_est_i) @ c2w_est_i1

        err_T = np.linalg.inv(dT_ref) @ dT_est
        trans_errs.append(np.linalg.norm(err_T[:3, 3]))
        # rotation angle from trace
        cos_angle = np.clip((np.trace(err_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot_errs.append(np.degrees(np.arccos(cos_angle)))

    rpe_trans = float(np.sqrt(np.mean(np.array(trans_errs) ** 2)))
    rpe_rot = float(np.sqrt(np.mean(np.array(rot_errs) ** 2)))

    return {
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "rpe_trans_rmse": rpe_trans,
        "rpe_rot_deg_rmse": rpe_rot,
        "n_frames": n,
        "distance_m": _trajectory_distance(c2w_est[:n]),
        "gt_distance_m": _trajectory_distance(c2w_ref[:n]),
    }


# ---------------------------------------------------------------------------
# Camera list → numpy arrays
# ---------------------------------------------------------------------------

def _camera_frame_id(cam, position: int) -> float:
    """Best-available per-camera ordinal/timestamp for trajectory export.

    CRITICAL: never fall back to ``position`` (the camera's index within
    whatever list was handed to ``_cams_to_c2w``) when any real identity is
    available. ``Scene.__init__`` shuffles ``train_cameras``/``test_cameras``
    in place (``random.shuffle``, for randomized minibatch sampling) and, for
    an ``--eval`` run, that list is also a holdout-COMPACTED subset of the
    real per-dataset-frame sequence (every Nth frame removed, the rest
    re-indexed with no gaps). Using list position as a stand-in "timestamp"
    silently produces a random permutation of real poses paired with
    monotonically-increasing fake timestamps -- not a trajectory at all.

    This was root-caused on a real eligsir/gs-slam-bench Replica room0
    run (2026-08-22): Sim3-aligning the exported ``trajectory_tum.txt``
    against the dataset's real ``traj.txt`` GT collapsed to scale=0.0203,
    ate_mean=1.44m, and ``utils/trajectory_eval.py``'s own plausibility
    check flagged the path as a self-crossing scribble
    (distance/bbox_diagonal=656). Re-deriving each camera's REAL dataset
    frame id (from ``cameras.json``, written by ``Scene.__init__`` *before*
    the shuffle) and re-aligning gave scale=0.99999999..., ate_rmse=1.1e-7 --
    i.e. training itself and pose scale were always correct; only the
    identity used to label/order the exported trajectory was wrong.

    A first version of this fix used ``cam.uid`` for priority 3 below and
    still reproduced the exact original bug end-to-end -- because
    ``scene.cameras.Camera.uid`` is NOT ``CameraInfo.uid``:
    ``utils/camera_utils.py``'s ``cameraList_from_camInfos`` rebinds
    ``uid=id`` from ``enumerate(cam_infos)`` -- the camera's position in
    whatever (already-shuffled, by the time this runs) list is passed in --
    while the *original*, pre-shuffle ``CameraInfo.uid`` survives untouched
    as ``Camera.colmap_id`` (``loadCam``: ``colmap_id=cam_info.uid``). So
    ``cam.uid`` on a real training-loop Camera object is exactly the same
    shuffled-list-position value this function exists to avoid; ``colmap_id``
    is the one that is actually stable across the shuffle.

    Priority:
      1. ``cam._streaming_timestamp`` -- real wall/dataset time, set by the
         streaming path (`train_streaming.py`); left untouched.
      2. ``cam.image_name`` parsed as a TUM-style ``sec_usec`` timestamp
         (``"1305031102_175304"`` -> ``1305031102.175304``, see
         ``scene/readers/tum.py``) -- the exact real GT-comparable
         timestamp for TUM sequences.
      3. ``cam.colmap_id`` -- every scene reader (tum/rgbd_sequence/scannet/
         replica/blender/colmap, see ``scene/readers/*.py``) assigns
         ``CameraInfo.uid`` as the camera's position in the FULL, unsplit,
         dataset-ordered sequence at load time, and ``loadCam`` carries that
         value into ``Camera.colmap_id`` verbatim -- i.e. the real
         per-dataset-frame index, unaffected by ``--eval`` holdout
         compaction or ``Scene``'s post-load shuffle.
      4. ``cam.uid`` -- fallback for camera-like objects that are not a real
         ``scene.cameras.Camera`` (e.g. lightweight test doubles) and only
         set ``uid``; NOT safe on a real Camera (see above).
      5. ``position`` -- only reachable for a camera object with none of the
         above, which no current reader/path produces.
    """
    streaming_ts = getattr(cam, "_streaming_timestamp", None)
    if streaming_ts is not None:
        return float(streaming_ts)

    image_name = getattr(cam, "image_name", None)
    if image_name:
        m = re.fullmatch(r"(\d+)_(\d+)", str(image_name))
        if m:
            try:
                return float(f"{m.group(1)}.{m.group(2)}")
            except ValueError:
                pass

    colmap_id = getattr(cam, "colmap_id", None)
    if colmap_id is not None:
        try:
            return float(colmap_id)
        except (TypeError, ValueError):
            pass

    uid = getattr(cam, "uid", None)
    if uid is not None:
        try:
            return float(uid)
        except (TypeError, ValueError):
            pass

    return float(position)


def _cams_to_c2w(cams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract timestamps (float seconds), c2w [N,4,4], and the temporal sort
    order (indices into the input ``cams`` sequence) from a camera list.

    The returned ``order`` lets a caller rebuild any per-camera side table
    (e.g. ``trajectory_test_index.json``) in the SAME order as the ts/c2w
    arrays -- required because this function sorts its output into temporal
    order while ``cams`` itself may be in Scene.__init__'s shuffled order.
    """
    import torch
    ts, mats = [], []
    for cam in cams:
        # 3DGS convention: cam.R stores R_c2w (camera-to-world rotation, i.e. the
        # camera basis vectors as columns), cam.T stores t_w2c (the translation in
        # the world-to-camera matrix).  getWorld2View2 builds: Rt[:3,:3] = R.T,
        # so Rt_rotation = R_c2w.T = R_w2c.
        R_c2w = np.array(cam.R, dtype=np.float64)   # 3×3 (cam.R = R_c2w)
        t_w2c = np.array(cam.T, dtype=np.float64)   # 3,
        t_c2w = -R_c2w @ t_w2c
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_c2w
        T[:3, 3] = t_c2w
        mats.append(T)
        ts.append(_camera_frame_id(cam, len(ts)))
    ts_arr = np.array(ts, dtype=np.float64)
    mats_arr = np.stack(mats)
    # Camera lists handed in here (scene.getTrainCameras()/getTestCameras())
    # are shuffled in place by Scene.__init__ for randomized minibatch
    # sampling -- sort back into temporal order so the exported TUM file,
    # the distance/plausibility metrics, and the plot's polyline all reflect
    # a real path (consecutive rows = consecutive real frames), not an
    # arbitrary permutation. Association-by-timestamp is order-independent
    # so this is purely for the non-GT-alignment consumers.
    order = np.argsort(ts_arr, kind="stable")
    return ts_arr[order], mats_arr[order], order


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_trajectory_eval(
    output_dir: str,
    train_cams: Sequence,
    *,
    test_cams: Sequence = (),
    gt_cams: Optional[Sequence] = None,
    gt_tum_path: Optional[str] = None,
    association_max_dt: float = 0.03,
    method_label: str = "estimated",
) -> None:
    """Save TUM trajectory, PNG plot, and metrics.json for one run.

    When ``test_cams`` is non-empty, also writes ``trajectory_test_tum.txt`` (same
    TUM format, held-out test cameras) and ``trajectory_test_index.json`` (name/
    source_index per row, for correlating back to report.json's test_per_view).
    """
    os.makedirs(output_dir, exist_ok=True)

    if not train_cams:
        return

    ts, c2w, _train_order = _cams_to_c2w(train_cams)

    tum_path = os.path.join(output_dir, "trajectory_tum.txt")
    save_tum_trajectory(tum_path, ts, c2w)

    if test_cams:
        test_ts, test_c2w, test_order = _cams_to_c2w(test_cams)
        test_tum_path = os.path.join(output_dir, "trajectory_test_tum.txt")
        save_tum_trajectory(test_tum_path, test_ts, test_c2w)

        # Reorder test_cams the SAME way _cams_to_c2w reordered its output
        # (temporal sort), so row i here still describes row i of
        # trajectory_test_tum.txt -- test_cams itself is in Scene.__init__'s
        # shuffled order, not the order test_ts/test_c2w now come back in.
        ordered_test_cams = [test_cams[i] for i in test_order]
        index = [
            {"name": str(cam.image_name), "source_index": int(cam.image_name)}
            for cam in ordered_test_cams
        ]
        index_path = os.path.join(output_dir, "trajectory_test_index.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    gt_c2w = None
    association_info = None
    if gt_cams:
        _, gt_c2w, _ = _cams_to_c2w(gt_cams)
    elif gt_tum_path:
        if os.path.exists(gt_tum_path):
            gt_ts, gt_all = load_tum_trajectory(gt_tum_path)
            matched_ts, c2w_matched, gt_c2w, association_info = associate_by_timestamp(
                ts,
                c2w,
                gt_ts,
                gt_all,
                max_dt=association_max_dt,
            )
            if len(matched_ts) > 0:
                ts = matched_ts
                c2w = c2w_matched
                save_tum_trajectory(os.path.join(output_dir, "trajectory_tum_associated.txt"), ts, c2w)
                save_tum_trajectory(os.path.join(output_dir, "groundtruth_tum_associated.txt"), ts, gt_c2w)
            else:
                gt_c2w = None
        else:
            print(f"[trajectory_eval] GT path not found: {gt_tum_path}", flush=True)

    if gt_c2w is not None:
        metrics = trajectory_metrics(c2w, gt_c2w)
    else:
        metrics = {"ate_rmse": None, "ate_mean": None,
                   "rpe_trans_rmse": None, "rpe_rot_deg_rmse": None,
                   "n_frames": len(train_cams),
                   "distance_m": _trajectory_distance(c2w),
                   "gt_distance_m": None}
    metrics["method"] = method_label
    metrics["n_train_cams"] = len(train_cams)
    metrics.update(_trajectory_plausibility(c2w))
    if metrics.get("implausible_distance") or metrics.get("degenerate_poses"):
        print(
            "[trajectory_eval] WARNING: distance_m does not look like a real path "
            f"length (distance/bbox_diagonal={_fmt_metric(metrics.get('distance_over_bbox_diagonal'))}, "
            f"collinearity_frac={_fmt_metric(metrics.get('collinearity_frac'))}). "
            "Do not quote distance_m for this run -- see trajectory.svg's warning "
            "and utils/trajectory_eval.py's _trajectory_distance docstring.",
            flush=True,
        )
    if gt_tum_path:
        metrics["gt_tum_path"] = gt_tum_path
    if association_info:
        metrics["association"] = association_info
        n_matched = association_info.get("n_matches", 0)
        n_est = len(train_cams)
        metrics["gt_coverage_frac"] = round(n_matched / n_est, 4) if n_est > 0 else None
        if n_matched < n_est:
            print(
                f"[trajectory_eval] WARNING: GT coverage {n_matched}/{n_est} "
                f"({100*n_matched/n_est:.1f}%) — ATE computed only on covered frames. "
                f"The remaining {n_est - n_matched} estimated poses have no GT within "
                f"{association_max_dt*1000:.0f}ms (e.g. mocap volume gap or tracking loss).",
                flush=True,
            )

    svg_path = os.path.join(output_dir, "trajectory.svg")
    plot_trajectory(svg_path, c2w, title=f"Trajectory — {method_label}", gt=gt_c2w, metrics=metrics)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[trajectory_eval] → {output_dir}  "
        f"n={len(train_cams)} ATE={metrics.get('ate_rmse')}",
        flush=True,
    )
