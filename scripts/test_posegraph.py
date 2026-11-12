#!/usr/bin/env python3
"""Isolated unit test of the loop-closure pose-graph optimization.

Reproduces the LC node's exact edge convention on a synthetic trajectory with
known odometry drift + one loop, and checks whether global_optimization actually
moves the nodes. The full-pipeline run shows "max node moved 0mm" despite large
(773mm) loop residuals — this isolates whether the Open3D optimization / edge
convention is the culprit, with no GPU or bag needed.

Convention (must match slam_loop_closure_node.py):
  node.pose = T_world_cam (Twc)
  edge(source, target).transformation = inv(Twc_target) @ Twc_source
"""
import numpy as np
import open3d as o3d

_reg = o3d.pipelines.registration


def T(tx, ty=0.0, tz=0.0, yaw=0.0):
    M = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    M[:3, 3] = [tx, ty, tz]
    return M


def build(loop_info_scale, pref_loop, edge_prune, odom_info, loop_uncertain=True):
    # Ground truth: 8 cameras on a closed loop (square), returning near start.
    N = 8
    gt = [T(np.cos(a), np.sin(a), 0, a + np.pi / 2)
          for a in np.linspace(0, 2 * np.pi, N, endpoint=False)]
    # Odometry: accumulate the GT relative transforms but inject drift so the
    # trajectory does NOT close (last node ends ~0.8m from where GT says).
    drift = T(0, 0, 0, 0.10)                      # small yaw error per step
    odo = [gt[0].copy()]
    for i in range(1, N):
        rel = np.linalg.inv(gt[i - 1]) @ gt[i]    # true relative (source=i-1->i)
        odo.append(odo[i - 1] @ rel @ drift)      # drifted accumulation

    pg = _reg.PoseGraph()
    for i in range(N):
        pg.nodes.append(_reg.PoseGraphNode(odo[i].copy()))
    # Odometry edges (uncertain=False), convention inv(Twc_t)@Twc_s.
    for i in range(1, N):
        X = np.linalg.inv(odo[i]) @ odo[i - 1]
        pg.edges.append(_reg.PoseGraphEdge(i - 1, i, X, np.eye(6) * odom_info, uncertain=False))
    # Loop edge closing node 0 <-> node N-1, using the TRUE relative transform
    # (what RANSAC on real keypoints would recover): source=0, target=N-1.
    X_loop = np.linalg.inv(gt[N - 1]) @ gt[0]
    fit = 0.35
    pg.edges.append(_reg.PoseGraphEdge(
        0, N - 1, X_loop, np.eye(6) * fit * loop_info_scale, uncertain=loop_uncertain))

    odom_resid = np.linalg.norm((np.linalg.inv(odo[N - 1]) @ odo[0])[:3, 3]
                                - X_loop[:3, 3]) * 1000
    before = np.stack([n.pose[:3, 3] for n in pg.nodes])
    opt = _reg.GlobalOptimizationLevenbergMarquardt()
    crit = _reg.GlobalOptimizationConvergenceCriteria()
    option = _reg.GlobalOptimizationOption(
        max_correspondence_distance=0.05, edge_prune_threshold=edge_prune,
        preference_loop_closure=pref_loop, reference_node=0)
    _reg.global_optimization(pg, opt, crit, option)
    after = np.stack([np.asarray(n.pose)[:3, 3] for n in pg.nodes])
    moved = np.linalg.norm(after - before, axis=1).max() * 1000
    return odom_resid, moved


if __name__ == "__main__":
    print("loop residual vs node movement across settings:\n")
    print(f"{'loop_info':>10} {'pref_loop':>10} {'prune':>7} {'odom_info':>10} "
          f"{'resid_mm':>9} {'moved_mm':>9}")
    for lis in (1.0, 30.0, 1000.0):
        for pref in (1.0, 2.0, 5.0):
            for prune in (0.25,):
                for oi in (1.0,):
                    r, m = build(lis, pref, prune, oi)
                    print(f"{lis:>10.0f} {pref:>10.1f} {prune:>7.2f} {oi:>10.1f} "
                          f"{r:>9.0f} {m:>9.1f}")
    # also: weaken odometry so the chain bends easily
    print("\nweaken odometry (odom_info=0.01), loop_info=1000:")
    r, m = build(1000.0, 5.0, 0.25, 0.01)
    print(f"  resid={r:.0f}mm  moved={m:.1f}mm")

    # KEY TEST: verified loops as HARD constraints (uncertain=False) so the line
    # process cannot switch them off. Sweep odom_info to find a robust regime.
    print("\nloop uncertain=FALSE (hard constraint), loop_info=1.0:")
    print(f"{'odom_info':>10} {'resid_mm':>9} {'moved_mm':>9}")
    for oi in (1.0, 0.5, 0.1, 0.01):
        r, m = build(1.0, 1.0, 0.25, oi, loop_uncertain=False)
        print(f"{oi:>10.2f} {r:>9.0f} {m:>9.1f}")

    print("\nloop uncertain=FALSE, odom_info=1.0, vary loop_info:")
    for lis in (1.0, 5.0, 30.0):
        r, m = build(lis, 1.0, 0.25, 1.0, loop_uncertain=False)
        print(f"  loop_info={lis:>5.0f}  resid={r:.0f}mm  moved={m:.1f}mm")
