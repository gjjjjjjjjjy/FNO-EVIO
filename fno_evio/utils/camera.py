"""
Camera model utilities (pinhole and KB4 fisheye) for FNO-EVIO.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

References:
  - J. Kannala and S. Brandt, “A Generic Camera Model and Calibration Method for Conventional,
    Wide-Angle, and Fish-Eye Lenses”, IEEE TPAMI, 2006.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def rescale_intrinsics_pinhole(
    K: Dict[str, float],
    src_resolution: Tuple[int, int],
    dst_resolution: Tuple[int, int],
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[Dict[str, float], Tuple[float, float]]:
    """
    Rescale pinhole intrinsics when resizing/cropping images.

    Args:
        K: Dict with fx, fy, cx, cy.
        src_resolution: (H, W) source resolution.
        dst_resolution: (H, W) destination resolution.
        crop: Optional (x0, y0, crop_w, crop_h) applied before resizing.

    Returns:
        K_new: Rescaled intrinsics.
        (scale_x, scale_y): Resizing factors.
    """
    src_h, src_w = src_resolution
    dst_h, dst_w = dst_resolution
    fx = float(K.get("fx", 1.0))
    fy = float(K.get("fy", 1.0))
    cx = float(K.get("cx", src_w * 0.5))
    cy = float(K.get("cy", src_h * 0.5))

    if crop is not None:
        x0, y0, crop_w, crop_h = crop
        cx = cx - float(x0)
        cy = cy - float(y0)
        src_w = int(crop_w)
        src_h = int(crop_h)

    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)

    K_new = {
        "fx": fx * scale_x,
        "fy": fy * scale_y,
        "cx": cx * scale_x,
        "cy": cy * scale_y,
    }
    return K_new, (scale_x, scale_y)


def rescale_intrinsics_kb4(
    K: Dict[str, float],
    distortion: Dict[str, float],
    src_resolution: Tuple[int, int],
    dst_resolution: Tuple[int, int],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Rescale KB4 intrinsics for a resolution change.

    Notes:
        KB4 distortion coefficients are defined in normalized coordinates, so the distortion
        parameters do not change with image resizing; only fx/fy/cx/cy scale.

    Args:
        K: Dict with fx, fy, cx, cy.
        distortion: Dict with k1..k4.
        src_resolution: (H, W) source resolution.
        dst_resolution: (H, W) destination resolution.

    Returns:
        K_new: Rescaled intrinsics.
        distortion_new: Copied distortion parameters.
    """
    src_h, src_w = src_resolution
    dst_h, dst_w = dst_resolution
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    K_new = {
        "fx": float(K.get("fx", 1.0)) * scale_x,
        "fy": float(K.get("fy", 1.0)) * scale_y,
        "cx": float(K.get("cx", src_w * 0.5)) * scale_x,
        "cy": float(K.get("cy", src_h * 0.5)) * scale_y,
    }
    return K_new, distortion.copy()


def kb4_unproject(
    u: np.ndarray,
    v: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    max_iter: int = 10,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KB4 unprojection from pixel coordinates to a unit ray (Newton-Raphson).

    Args:
        u, v: Pixel coordinates.
        fx, fy, cx, cy: Intrinsics.
        k1..k4: KB4 distortion coefficients.
        max_iter: Newton iterations.
        tol: Convergence tolerance on delta(theta).

    Returns:
        X, Y, Z: Unit direction in camera frame.
    """
    x = (u - cx) / fx
    y = (v - cy) / fy
    r = np.sqrt(x**2 + y**2)
    small_r_mask = r < 1e-6

    theta = r.copy()
    for _ in range(int(max_iter)):
        theta = np.clip(theta, 0.0, np.pi)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        f = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8) - r
        df = 1.0 + 3 * k1 * theta2 + 5 * k2 * theta4 + 7 * k3 * theta6 + 9 * k4 * theta8
        df = np.maximum(df, 1e-8)
        delta = f / df
        theta = theta - delta
        if np.max(np.abs(delta)) < float(tol):
            break

    theta = np.clip(theta, 0.0, np.pi)
    theta = np.where(small_r_mask, r, theta)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    scale = np.where(r > 1e-12, sin_theta / r, 1.0)
    X = x * scale
    Y = y * scale
    Z = cos_theta
    return X, Y, Z

