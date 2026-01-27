"""
Compatibility shim for FNO-EVIO: re-export commonly used utilities.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

This module intentionally provides a flat namespace to ease gradual refactors and to keep
import sites readable (`from utils import ...`) inside the FNO-EVIO codebase.
"""

from __future__ import annotations

from fno_evio.legacy.utils import (
    Logger,
    QuaternionUtils,
    align_trajectory,
    align_trajectory_with_timestamps,
    align_trajectory_with_timestamps_sim3,
    associate_by_timestamp,
    compose_se3,
    compute_ols_scale_stats,
    compute_rpe_loss,
    ensure_dir,
    is_low_shm,
    kb4_project,
    kb4_project_torch,
    kb4_unproject,
    kb4_unproject_torch,
    matrix_to_rotation_6d,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    read_txt_skip_first_line,
    rescale_intrinsics_kb4,
    rescale_intrinsics_pinhole,
    rotation_6d_to_matrix,
    safe_geodesic_loss,
    warp_events_flow,
    warp_events_flow_torch,
    warp_events_flow_torch_kb4,
 )

__all__ = [
    "Logger",
    "QuaternionUtils",
    "align_trajectory",
    "warp_events_flow",
    "warp_events_flow_torch",
    "warp_events_flow_torch_kb4",
    "align_trajectory_with_timestamps",
    "align_trajectory_with_timestamps_sim3",
    "associate_by_timestamp",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
    "rescale_intrinsics_pinhole",
    "rescale_intrinsics_kb4",
    "kb4_project",
    "kb4_unproject",
    "kb4_project_torch",
    "kb4_unproject_torch",
    "compute_rpe_loss",
    "compute_ols_scale_stats",
    "quat_normalize",
    "quat_multiply",
    "quat_inverse",
    "quat_to_rotmat",
    "compose_se3",
    "safe_geodesic_loss",
    "is_low_shm",
    "ensure_dir",
    "read_txt_skip_first_line",
]
