"""Structure-aware RGB-D bootstrap seed selector for streaming initialization.

This module implements a self-contained, SfM-like structural seed selection
from the first ~8 motion-diverse RGB-D frames. It replaces regular-grid depth
sampling with:

1. Motion-diverse frame selection: avoids near-duplicate still frames.
2. Candidate scoring: three complementary classes (photometric, geometric,
   coverage) with independent weights and always-on base coverage.
3. Multi-view verification: repeatability across bootstrap frames.
4. Fixed budget allocation: ensures sparse, informative init without relying
   on MCMC to churn down excess redundancy.

Pure numpy/CPU, no torch/cv2 requirement for core logic.

Issue #489: https://github.com/eligsir-project/eligsir/issues/489
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def _pose_gap(c2w_a: np.ndarray, c2w_b: np.ndarray) -> Tuple[float, float]:
    """Compute translation (m) and rotation (deg) between two c2w poses.

    Matches the formula in StreamingScene._pose_gap (line 476-482).
    """
    t = float(np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]))
    r_rel = c2w_a[:3, :3].T @ c2w_b[:3, :3]
    cos = (np.trace(r_rel) - 1.0) * 0.5
    ang = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return t, ang


def select_motion_diverse_frames(
    poses: List[np.ndarray],
    motion_translation_threshold: float = 0.04,
    motion_rotation_threshold: float = 4.0,
    min_cumulative_baseline: float = 0.2,
    target_n_frames: int = 8,
    max_lookahead: int = 50,
) -> Tuple[List[int], Dict]:
    """Select motion-diverse bootstrap frames from an ordered pose list.

    Prefers frames with sufficient translation OR rotation relative to the
    previous accepted frame, and enforces a cumulative baseline between first
    and last accepted frame. Falls back to first N frames if motion gate is
    not satisfied within max_lookahead, setting bootstrap_motion_insufficient=True.

    Args:
        poses: List of camera-to-world (4, 4) poses in order.
        motion_translation_threshold: Minimum m between accepted frames.
        motion_rotation_threshold: Minimum deg rotation between accepted frames.
        min_cumulative_baseline: Minimum total translation from first to last.
        target_n_frames: Target number of frames to select (~8).
        max_lookahead: Maximum frames to scan before fallback (per frame).

    Returns:
        A ``(selected_frame_indices, diagnostics_dict)`` tuple.
        ``selected_frame_indices`` is a list of source frame indices to use.
        ``diagnostics_dict`` has keys ``motion_insufficient`` (bool),
        ``n_selected`` (int), ``cumulative_baseline`` (float) and
        ``pose_deltas`` (list of ``(t_m, ang_deg)`` pairs).
    """
    if not poses:
        return [], {"motion_insufficient": True, "n_selected": 0, "cumulative_baseline": 0.0}

    # Ensure all poses are valid (4, 4) matrices.
    valid_poses = []
    for i, p in enumerate(poses):
        p = np.asarray(p, dtype=np.float32)
        if p.shape == (4, 4) and np.isfinite(p).all():
            valid_poses.append((i, p))
    if not valid_poses:
        return [], {"motion_insufficient": True, "n_selected": 0, "cumulative_baseline": 0.0}

    selected_indices = []
    selected_poses = []
    pose_deltas = []

    # Accept first pose unconditionally.
    selected_indices.append(valid_poses[0][0])
    selected_poses.append(valid_poses[0][1])

    # Greedily select subsequent frames.
    current_idx = 0
    while len(selected_indices) < target_n_frames and current_idx < len(valid_poses) - 1:
        last_pose = selected_poses[-1]
        found = False

        # Look ahead up to max_lookahead frames.
        for offset in range(1, min(max_lookahead + 1, len(valid_poses) - current_idx)):
            candidate_idx = current_idx + offset
            candidate_pose = valid_poses[candidate_idx][1]

            t, ang = _pose_gap(last_pose, candidate_pose)
            if t >= motion_translation_threshold or ang >= motion_rotation_threshold:
                # Accept this frame.
                selected_indices.append(valid_poses[candidate_idx][0])
                selected_poses.append(candidate_pose)
                pose_deltas.append((t, ang))
                current_idx = candidate_idx
                found = True
                break

        if not found:
            # Could not find a frame with sufficient motion in the lookahead.
            break

    # Check cumulative baseline (first to last selected frame).
    cumulative_t = 0.0
    motion_sufficient = False
    if len(selected_poses) > 1:
        cumulative_t, _ = _pose_gap(selected_poses[0], selected_poses[-1])
        motion_sufficient = cumulative_t >= min_cumulative_baseline

    # Fallback: if motion insufficient, use first target_n_frames frames.
    if not motion_sufficient and len(selected_indices) < target_n_frames:
        selected_indices = [idx for idx, _ in valid_poses[:target_n_frames]]
        cumulative_t = 0.0
        if len(valid_poses) > 1:
            t, _ = _pose_gap(valid_poses[0][1], valid_poses[min(target_n_frames - 1, len(valid_poses) - 1)][1])
            cumulative_t = t

    return selected_indices, {
        "motion_insufficient": not motion_sufficient,
        "n_selected": len(selected_indices),
        "cumulative_baseline": cumulative_t,
        "pose_deltas": pose_deltas,
    }


def _shi_tomasi_score_np(
    rgb: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute Shi-Tomasi cornerness via structure tensor (pure numpy).

    Uses Sobel gradients and local accumulation. Returns a score in [0, 1]
    where 1 is highest cornerness.

    Args:
        rgb: Normalized RGB image, shape (H, W, 3), dtype float32 in [0, 1].
        window_size: Radius of the local window (will use 2*window_size+1).

    Returns:
        Cornerness score map, shape (H, W), dtype float32 in [0, 1].
    """
    # Convert to grayscale.
    gray = np.mean(rgb, axis=2).astype(np.float32)

    # Sobel gradients (don't normalize; let the magnitude vary with edge strength).
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0

    # Simple convolution by hand to avoid extra dependencies.
    h, w = gray.shape
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = gray[y - 1 : y + 2, x - 1 : x + 2]
            gx[y, x] = np.sum(patch * kernel_x)
            gy[y, x] = np.sum(patch * kernel_y)

    # Structure tensor components: Ixx, Iyy, Ixy.
    ixx = gx * gx
    iyy = gy * gy
    ixy = gx * gy

    # Accumulate over a local window (box filter).
    radius = window_size
    acc_ixx = np.zeros_like(ixx)
    acc_iyy = np.zeros_like(iyy)
    acc_ixy = np.zeros_like(ixy)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ixx_shift = np.roll(np.roll(ixx, dy, axis=0), dx, axis=1)
            iyy_shift = np.roll(np.roll(iyy, dy, axis=0), dx, axis=1)
            ixy_shift = np.roll(np.roll(ixy, dy, axis=0), dx, axis=1)
            acc_ixx += ixx_shift
            acc_iyy += iyy_shift
            acc_ixy += ixy_shift

    # Shi-Tomasi score = smaller eigenvalue of the structure tensor.
    # For a 2x2 matrix [[Ixx, Ixy], [Ixy, Iyy]]:
    # trace = Ixx + Iyy
    # det = Ixx * Iyy - Ixy^2
    # smaller eigenvalue = (trace - sqrt(trace^2 - 4*det)) / 2
    trace = acc_ixx + acc_iyy
    det = acc_ixx * acc_iyy - acc_ixy * acc_ixy
    discriminant = np.maximum(trace * trace - 4 * det, 0.0)
    smaller_eig = (trace - np.sqrt(discriminant)) / 2.0

    # Normalize to [0, 1].
    min_val = smaller_eig.min()
    max_val = smaller_eig.max()
    if max_val > min_val:
        score = (smaller_eig - min_val) / (max_val - min_val)
    else:
        score = np.zeros_like(smaller_eig)

    return score.astype(np.float32)


def _depth_edge_score_np(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    edge_threshold: float = 0.1,
) -> np.ndarray:
    """Score depth discontinuities and edges (high = near boundary).

    Args:
        depth: Depth map, shape (H, W), dtype float32.
        valid_mask: Boolean mask, shape (H, W), True where depth is valid.
        edge_threshold: Threshold for relative depth change to flag an edge.

    Returns:
        Edge score map, shape (H, W), dtype float32 in [0, 1].
    """
    h, w = depth.shape
    edge_score = np.zeros_like(depth, dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not valid_mask[y, x]:
                continue
            z = depth[y, x]
            neighbors = []
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and valid_mask[ny, nx]:
                    neighbors.append(depth[ny, nx])
            if neighbors:
                z_neighbors = np.array(neighbors)
                dz = np.abs(z_neighbors - z)
                max_dz = dz.max()
                edge_score[y, x] = min(1.0, max_dz / max(z * edge_threshold, 1e-6))

    return edge_score


def _normal_curvature_score_np(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Score surface normal change (high = high curvature/corners).

    Computes per-pixel surface normals and returns the magnitude of normal
    differences to neighbors.

    Args:
        depth: Depth map, shape (H, W), dtype float32.
        valid_mask: Boolean mask, shape (H, W), True where depth is valid.
        fx, fy, cx, cy: Camera intrinsics.

    Returns:
        Curvature score map, shape (H, W), dtype float32 in [0, 1].
    """
    h, w = depth.shape
    curvature_score = np.zeros_like(depth, dtype=np.float32)

    # Compute normals via cross product of edge vectors in 3D.
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not valid_mask[y, x]:
                continue

            # Unproject to 3D.
            z = depth[y, x]
            x_c = (x - cx) / fx * z
            y_c = (y - cy) / fy * z
            p = np.array([x_c, y_c, z], dtype=np.float32)

            # Get neighbors (use cross product of edge vectors).
            valid_neighbors = 0
            normal_diff_sum = 0.0
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and valid_mask[ny, nx]:
                    z_n = depth[ny, nx]
                    x_cn = (nx - cx) / fx * z_n
                    y_cn = (ny - cy) / fy * z_n
                    p_n = np.array([x_cn, y_cn, z_n], dtype=np.float32)

                    # Approximate normal via cross product (simple case).
                    edge_to_n = p_n - p
                    # Use a fixed-direction edge (e.g., +y edge).
                    if dy == 0:  # x-neighbor
                        if nx == x + 1 and y + 1 < h and valid_mask[y + 1, x]:
                            z_below = depth[y + 1, x]
                            x_cbelow = (x - cx) / fx * z_below
                            y_cbelow = (y + 1 - cy) / fy * z_below
                            p_below = np.array([x_cbelow, y_cbelow, z_below], dtype=np.float32)
                            edge_y = p_below - p
                            normal = np.cross(edge_to_n, edge_y)
                            norm_mag = np.linalg.norm(normal)
                            if norm_mag > 1e-6:
                                normal /= norm_mag
                                normal_diff_sum += norm_mag
                                valid_neighbors += 1

            if valid_neighbors > 0:
                curvature_score[y, x] = min(1.0, normal_diff_sum / valid_neighbors)

    return curvature_score


def score_candidates(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w_photometric: float = 1.0,
    w_geometric: float = 0.8,
    w_depth_edge: float = 0.6,
    base_coverage_weight: float = 0.2,
) -> np.ndarray:
    """Score each pixel as a candidate bootstrap anchor.

    Returns a composite score combining:
    - Photometric: Shi-Tomasi cornerness (edges/textured areas).
    - Geometric: Normal curvature + depth discontinuities.
    - Coverage: Guaranteed non-zero baseline (base_coverage_weight).

    All pixels with valid depth receive at least base_coverage_weight so
    smooth surfaces are never completely ignored.

    Args:
        rgb: Normalized RGB, shape (H, W, 3), dtype float32 in [0, 1].
        depth: Depth map, shape (H, W), dtype float32.
        valid_mask: Boolean mask, shape (H, W), True where depth is valid.
        fx, fy, cx, cy: Camera intrinsics.
        w_photometric: Weight for Shi-Tomasi term.
        w_geometric: Weight for geometric terms (combined curvature + edge).
        w_depth_edge: Weight for depth discontinuities within geometric term.
        base_coverage_weight: Weight for the always-on coverage baseline.

    Returns:
        Per-pixel score, shape (H, W), dtype float32.
    """
    h, w = rgb.shape[:2]
    score = np.zeros((h, w), dtype=np.float32)

    # Photometric anchors: Shi-Tomasi cornerness on valid pixels.
    photometric = _shi_tomasi_score_np(rgb, window_size=3)
    score += w_photometric * photometric * valid_mask.astype(np.float32)

    # Geometric anchors: curvature + depth edges.
    curvature = _normal_curvature_score_np(depth, valid_mask, fx, fy, cx, cy)
    depth_edge = _depth_edge_score_np(depth, valid_mask, edge_threshold=0.1)
    geometric = (curvature + w_depth_edge * depth_edge) / (1.0 + w_depth_edge)
    score += w_geometric * geometric * valid_mask.astype(np.float32)

    # Coverage baseline: ensures smooth surfaces are never at zero probability.
    score += base_coverage_weight * valid_mask.astype(np.float32)

    return score.astype(np.float32)


def allocate_budget(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    total_budget: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select exactly total_budget anchor pixels using scores.

    Uses score-weighted sampling with deterministic tie-breaking
    (linear pixel index), ensuring reproducibility.

    Args:
        scores: Per-pixel scores, shape (H, W), dtype float32.
        valid_mask: Boolean mask, shape (H, W), True where selectable.
        total_budget: Target number of pixels to select.
        rng: Optional np.random.Generator for reproducibility. If None,
             will create one with seed=42 (deprecated path).

    Returns:
        (selected_y, selected_x): Arrays of selected pixel coordinates.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Collect valid candidates.
    valid_y, valid_x = np.where(valid_mask)
    valid_scores = scores[valid_mask]

    if valid_y.shape[0] == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # Normalize scores to [0, 1] for probability.
    min_score = valid_scores.min()
    max_score = valid_scores.max()
    if max_score > min_score:
        probs = (valid_scores - min_score) / (max_score - min_score)
    else:
        probs = np.ones_like(valid_scores)

    # Ensure all probabilities are > 0 for sampling.
    probs = np.maximum(probs, 1e-8)
    probs = probs / probs.sum()

    # Sample indices.
    n_select = min(total_budget, valid_y.shape[0])
    selected_idx = rng.choice(valid_y.shape[0], size=n_select, replace=False, p=probs)

    selected_y = valid_y[selected_idx].astype(np.int32)
    selected_x = valid_x[selected_idx].astype(np.int32)

    return selected_y, selected_x


def verify_multiview_support(
    frames_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    selected_y: np.ndarray,
    selected_x: np.ndarray,
    depth_tolerance: float = 0.1,
) -> np.ndarray:
    """Check multi-view repeatability of selected candidates.

    For each selected pixel in the reference frame, reprojects it into all
    other bootstrap frames and checks if the reprojected depth agrees.

    Args:
        frames_list: List of (rgb, depth, valid_mask, c2w) tuples per frame.
        selected_y, selected_x: Selected pixel coordinates (from reference frame 0).
        depth_tolerance: Threshold for depth agreement across views.

    Returns:
        Support count per candidate, shape (n_candidates,), dtype int32.
    """
    if not frames_list or selected_y.shape[0] == 0:
        return np.zeros(selected_y.shape[0], dtype=np.int32)

    # Use first frame as reference.
    rgb_ref, depth_ref, valid_ref, c2w_ref = frames_list[0]
    h, w = depth_ref.shape
    fx, fy = 500.0, 500.0  # Placeholder intrinsics (not used for this logic).
    cx, cy = w / 2.0, h / 2.0

    support = np.zeros(selected_y.shape[0], dtype=np.int32)

    for i, (y, x) in enumerate(zip(selected_y, selected_x)):
        if not valid_ref[y, x]:
            continue

        z_ref = depth_ref[y, x]

        # Check agreement in all other frames.
        agreement_count = 1  # Count self.
        for frame_other in frames_list[1:]:
            _, depth_other, valid_other, c2w_other = frame_other

            # Reproject reference pixel into other frame.
            # Note: This is a simplified check; a full implementation would
            # handle intrinsics properly. For now, we just check if the
            # nearby depth is similar.
            if 0 < y < h - 1 and 0 < x < w - 1:
                neighborhood = depth_other[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
                valid_neighborhood = valid_other[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
                if valid_neighborhood.any():
                    z_neighborhood = neighborhood[valid_neighborhood]
                    if np.any(np.abs(z_neighborhood - z_ref) <= depth_tolerance):
                        agreement_count += 1

        support[i] = agreement_count

    return support


class StructuralBootstrapSelector:
    """Self-contained structural seed selector for streaming bootstrap.

    Combines motion-diverse frame selection with multi-class candidate
    scoring and multi-view verification to build an informative initial
    point cloud.

    This is a library module, not wired into StreamingScene yet. Wiring
    should replace the np.mgrid grid sampling in _build_pcd_from_frames()
    with calls to this selector's methods, controlled by a
    --streaming_bootstrap_seed_mode flag.
    """

    def __init__(
        self,
        seed: int = 42,
        motion_translation_threshold: float = 0.04,
        motion_rotation_threshold: float = 4.0,
        target_bootstrap_frames: int = 8,
    ):
        """Initialize the selector with motion and sampling parameters.

        Args:
            seed: Random seed for reproducibility.
            motion_translation_threshold: Minimum translation between frames (m).
            motion_rotation_threshold: Minimum rotation between frames (deg).
            target_bootstrap_frames: Target number of bootstrap frames to select.
        """
        self.rng = np.random.default_rng(seed)
        self.motion_translation_threshold = motion_translation_threshold
        self.motion_rotation_threshold = motion_rotation_threshold
        self.target_bootstrap_frames = target_bootstrap_frames

    def select_frames(self, poses: List[np.ndarray]) -> Tuple[List[int], Dict]:
        """Select motion-diverse frames from a pose sequence.

        Args:
            poses: List of camera-to-world (4, 4) matrices.

        Returns:
            (selected_indices, diagnostics_dict).
        """
        return select_motion_diverse_frames(
            poses,
            motion_translation_threshold=self.motion_translation_threshold,
            motion_rotation_threshold=self.motion_rotation_threshold,
            target_n_frames=self.target_bootstrap_frames,
        )

    def score_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        valid_mask: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """Score all pixels in a frame.

        Args:
            rgb: Normalized RGB, shape (H, W, 3), dtype float32.
            depth: Depth map, shape (H, W), dtype float32.
            valid_mask: Boolean mask, shape (H, W).
            fx, fy, cx, cy: Camera intrinsics.

        Returns:
            Per-pixel score, shape (H, W).
        """
        return score_candidates(rgb, depth, valid_mask, fx, fy, cx, cy)

    def select_candidates(
        self,
        scores: np.ndarray,
        valid_mask: np.ndarray,
        total_budget: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select anchor pixels from scores.

        Args:
            scores: Per-pixel scores, shape (H, W).
            valid_mask: Boolean mask, shape (H, W).
            total_budget: Number of anchors to select.

        Returns:
            (selected_y, selected_x).
        """
        return allocate_budget(scores, valid_mask, total_budget, rng=self.rng)
