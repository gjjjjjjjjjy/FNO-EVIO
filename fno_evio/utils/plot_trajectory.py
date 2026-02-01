"""
Trajectory plotting utilities for FNO-EVIO.

Provides functions to visualize predicted vs ground truth trajectories,
similar to DEIO's plot_utils.py.

Author: gjjjjjjjjjy
Created: 2026-02-01
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from evo.core import sync
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import plot
    from evo.core.geometry import GeometryException
    HAS_EVO = True
except ImportError:
    HAS_EVO = False


def make_traj(
    positions: np.ndarray,
    orientations: np.ndarray,
    timestamps: np.ndarray,
) -> "PoseTrajectory3D":
    """
    Create a PoseTrajectory3D from position, orientation, and timestamp arrays.

    Args:
        positions: (N, 3) array of xyz positions
        orientations: (N, 4) array of quaternions in xyzw format
        timestamps: (N,) array of timestamps

    Returns:
        PoseTrajectory3D object
    """
    if not HAS_EVO:
        raise ImportError("evo library is required for trajectory plotting. Install with: pip install evo")

    # Convert xyzw to wxyz for evo
    quat_wxyz = orientations[:, [3, 0, 1, 2]]
    return PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quat_wxyz,
        timestamps=timestamps,
    )


def best_plotmode(traj: "PoseTrajectory3D"):
    """
    Determine the best 2D projection plane for plotting based on trajectory variance.

    Returns the PlotMode that shows the most variance in the trajectory.
    """
    if not HAS_EVO:
        return None

    _, i1, i2 = np.argsort(np.var(traj.positions_xyz, axis=0))
    plot_axes = "xyz"[i2] + "xyz"[i1]
    return getattr(plot.PlotMode, plot_axes)


def plot_trajectory(
    pred_pos: np.ndarray,
    pred_quat: np.ndarray,
    pred_t: np.ndarray,
    gt_pos: Optional[np.ndarray] = None,
    gt_quat: Optional[np.ndarray] = None,
    gt_t: Optional[np.ndarray] = None,
    title: str = "",
    filename: str = "",
    align: bool = True,
    correct_scale: bool = True,
    max_diff_sec: float = 0.01,
) -> None:
    """
    Plot predicted trajectory against ground truth and save to file.

    Args:
        pred_pos: (N, 3) predicted positions
        pred_quat: (N, 4) predicted quaternions (xyzw)
        pred_t: (N,) predicted timestamps
        gt_pos: (M, 3) ground truth positions (optional)
        gt_quat: (M, 4) ground truth quaternions (xyzw, optional)
        gt_t: (M,) ground truth timestamps (optional)
        title: Plot title
        filename: Output file path (supports png, pdf, svg)
        align: Whether to align trajectories using SE(3) or Sim(3)
        correct_scale: Whether to correct scale during alignment
        max_diff_sec: Maximum time difference for trajectory association
    """
    if not HAS_EVO or not HAS_MATPLOTLIB:
        print("[PLOT] Skipping plot - evo or matplotlib not installed")
        return

    pred_traj = make_traj(pred_pos, pred_quat, pred_t)

    if gt_pos is not None and gt_quat is not None and gt_t is not None:
        gt_traj = make_traj(gt_pos, gt_quat, gt_t)
        gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj, max_diff=max_diff_sec)

        if align:
            try:
                pred_traj.align(gt_traj, correct_scale=correct_scale)
            except GeometryException as e:
                print(f"[PLOT] Alignment error: {e}")
    else:
        gt_traj = None

    plot_collection = plot.PlotCollection("PlotCol")
    fig = plt.figure(figsize=(8, 8))
    plot_mode = best_plotmode(gt_traj if gt_traj is not None else pred_traj)
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(title)

    if gt_traj is not None:
        plot.traj(ax, plot_mode, gt_traj, '--', 'gray', "Ground Truth")
    plot.traj(ax, plot_mode, pred_traj, '-', 'blue', "Predicted")

    ax.legend()

    if filename:
        plot_collection.add_figure("traj", fig)
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        plot_collection.export(filename, confirm_overwrite=False)
        print(f"[PLOT] Saved trajectory plot to {filename}")

    plt.close(fig=fig)


def plot_trajectory_3d(
    pred_pos: np.ndarray,
    gt_pos: Optional[np.ndarray] = None,
    title: str = "",
    filename: str = "",
    align: bool = True,
) -> None:
    """
    Plot 3D trajectory visualization.

    Args:
        pred_pos: (N, 3) predicted positions
        gt_pos: (M, 3) ground truth positions (optional)
        title: Plot title
        filename: Output file path
        align: Whether to align trajectories
    """
    if not HAS_MATPLOTLIB:
        print("[PLOT] Skipping 3D plot - matplotlib not installed")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if gt_pos is not None:
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], '--', color='gray', label='Ground Truth', alpha=0.7)

    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], '-', color='blue', label='Predicted')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title or "3D Trajectory")
    ax.legend()

    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved 3D trajectory plot to {filename}")

    plt.close(fig=fig)


def plot_trajectory_simple(
    pred_pos: np.ndarray,
    gt_pos: Optional[np.ndarray] = None,
    title: str = "",
    filename: str = "",
) -> None:
    """
    Simple 2D trajectory plot without evo dependency.

    Automatically selects the best 2D projection plane based on variance.

    Args:
        pred_pos: (N, 3) predicted positions
        gt_pos: (M, 3) ground truth positions (optional)
        title: Plot title
        filename: Output file path
    """
    if not HAS_MATPLOTLIB:
        print("[PLOT] Skipping plot - matplotlib not installed")
        return

    # Find best projection plane based on variance
    ref_pos = gt_pos if gt_pos is not None else pred_pos
    variances = np.var(ref_pos, axis=0)
    sorted_axes = np.argsort(variances)[::-1]  # Descending order
    ax1, ax2 = sorted_axes[0], sorted_axes[1]
    axis_labels = ['X', 'Y', 'Z']

    fig, ax = plt.subplots(figsize=(8, 8))

    if gt_pos is not None:
        ax.plot(gt_pos[:, ax1], gt_pos[:, ax2], '--', color='gray', label='Ground Truth', alpha=0.7, linewidth=1.5)
        ax.scatter(gt_pos[0, ax1], gt_pos[0, ax2], c='green', s=100, marker='o', zorder=5, label='Start (GT)')
        ax.scatter(gt_pos[-1, ax1], gt_pos[-1, ax2], c='red', s=100, marker='x', zorder=5, label='End (GT)')

    ax.plot(pred_pos[:, ax1], pred_pos[:, ax2], '-', color='blue', label='Predicted', linewidth=1.5)
    ax.scatter(pred_pos[0, ax1], pred_pos[0, ax2], c='green', s=100, marker='o', zorder=5)
    ax.scatter(pred_pos[-1, ax1], pred_pos[-1, ax2], c='red', s=100, marker='x', zorder=5)

    ax.set_xlabel(f'{axis_labels[ax1]} (m)')
    ax.set_ylabel(f'{axis_labels[ax2]} (m)')
    ax.set_title(title or "Trajectory")
    ax.legend()
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved trajectory plot to {filename}")

    plt.close(fig=fig)


def save_trajectory_tum(
    positions: np.ndarray,
    orientations: np.ndarray,
    timestamps: np.ndarray,
    filename: str,
) -> None:
    """
    Save trajectory in TUM format.

    TUM format: timestamp tx ty tz qx qy qz qw

    Args:
        positions: (N, 3) xyz positions
        orientations: (N, 4) quaternions in xyzw format
        timestamps: (N,) timestamps
        filename: Output file path
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, 'w') as f:
        for i in range(len(timestamps)):
            t = timestamps[i]
            p = positions[i]
            q = orientations[i]  # xyzw
            f.write(f"{t:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")

    print(f"[SAVE] Saved trajectory to {filename}")


def save_trajectory_kitti(
    positions: np.ndarray,
    orientations: np.ndarray,
    filename: str,
) -> None:
    """
    Save trajectory in KITTI format.

    KITTI format: 12 values per line (flattened 3x4 transformation matrix)

    Args:
        positions: (N, 3) xyz positions
        orientations: (N, 4) quaternions in xyzw format
        filename: Output file path
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, 'w') as f:
        for i in range(len(positions)):
            p = positions[i]
            q = orientations[i]  # xyzw

            # Convert quaternion to rotation matrix
            x, y, z, w = q
            R = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ])

            # Create 3x4 transformation matrix [R|t]
            T = np.hstack([R, p.reshape(3, 1)])

            # Write flattened matrix
            f.write(' '.join(f'{v:.6e}' for v in T.flatten()) + '\n')

    print(f"[SAVE] Saved trajectory to {filename}")
