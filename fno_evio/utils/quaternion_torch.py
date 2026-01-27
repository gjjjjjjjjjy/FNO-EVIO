"""
Torch quaternion utilities.
"""

from __future__ import annotations

from fno_evio.legacy.utils import (
    compose_se3,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    safe_geodesic_loss,
)

__all__ = ["quat_normalize", "quat_multiply", "quat_inverse", "quat_to_rotmat", "compose_se3", "safe_geodesic_loss"]
