"""
Math and geometry utilities for FNO-EVIO.

This module provides a clean interface to utility functions. Implementation
details are in fno_evio.legacy.utils, but this module provides stable public API.

Usage:
    from fno_evio.utils import QuaternionUtils
    from fno_evio.utils import rotation_6d_to_matrix
    from fno_evio.utils import align_trajectory_with_timestamps
"""

from __future__ import annotations

# Quaternion utilities
from fno_evio.utils.quaternion_np import QuaternionUtils

# Rotation representation utilities
from fno_evio.utils.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix

# Trajectory alignment utilities
from fno_evio.utils.trajectory import (
    align_trajectory_with_timestamps,
    align_trajectory_with_timestamps_sim3,
    associate_by_timestamp,
    compute_ols_scale_stats,
)

# Event warping utilities
from fno_evio.utils.events_warp import (
    warp_events_flow_torch,
    warp_events_flow_torch_kb4,
)

# Metric computation utilities
from fno_evio.utils.metrics import compute_rpe_loss

__all__ = [
    # Quaternion
    "QuaternionUtils",
    # Rotation
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
    # Trajectory
    "align_trajectory_with_timestamps",
    "align_trajectory_with_timestamps_sim3",
    "associate_by_timestamp",
    "compute_ols_scale_stats",
    # Events
    "warp_events_flow_torch",
    "warp_events_flow_torch_kb4",
    # Metrics
    "compute_rpe_loss",
]
