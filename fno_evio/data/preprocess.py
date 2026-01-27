"""
Dataset preprocessing utilities for FNO-EVIO.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

This module factors the high-entropy data preparation routine into three stages:
  1) Event stream preprocessing (HDF5 discovery, key resolution, timestamp scale detection)
  2) IMU preprocessing (unit normalization, timestamp correction)
  3) Ground-truth preprocessing (timestamp correction, pose parsing)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from fno_evio.utils.camera import rescale_intrinsics_kb4, rescale_intrinsics_pinhole


@dataclass(frozen=True)
class IMUPreprocessResult:
    imu_t: np.ndarray
    imu_vals: np.ndarray


@dataclass(frozen=True)
class GTPreprocessResult:
    gt_t: np.ndarray
    gt_pos: np.ndarray
    gt_quat: np.ndarray


@dataclass(frozen=True)
class EventsPreprocessResult:
    h5_path: str
    x_key: str
    y_key: str
    t_key: str
    p_key: Optional[str]
    unit_scale: float
    t_coarse: np.ndarray
    sensor_resolution: Tuple[int, int]
    fx_scaled: float
    fy_scaled: float
    cx_scaled: float
    cy_scaled: float
    camera_type: str
    kb4_distortion: Optional[Dict[str, float]]


def _norm_unit_str(u: Any) -> str:
    if u is None:
        return ""
    try:
        s = str(u).strip().lower()
    except Exception:
        return ""
    return s.replace(" ", "")


def preprocess_imu(
    dataset: Any,
    *,
    imu_path: Path,
    imu_data: np.ndarray,
) -> IMUPreprocessResult:
    """
    Preprocess IMU signals: timestamp correction, column parsing, unit conversion, normalization.

    Args:
        dataset: Dataset object providing _correct_timestamps/_parse_imu_columns/_normalize_imu_data and calib.
        imu_path: Path to IMU file (for header-based parsing).
        imu_data: Loaded numeric table (skip header already applied).

    Returns:
        IMUPreprocessResult with corrected timestamps and normalized imu values.
    """
    calib = getattr(dataset, "calib", None)
    imu_time_unit = calib.get("imu_time_unit") if isinstance(calib, dict) else None

    imu_t = dataset._correct_timestamps(imu_data[:, 0], unit=imu_time_unit)
    imu_acc, imu_gyro = dataset._parse_imu_columns(str(imu_path), imu_data)

    acc_mag = float(np.mean(np.linalg.norm(imu_acc, axis=1)))
    gyro_mag = float(np.mean(np.linalg.norm(imu_gyro, axis=1)))
    print(f"[IMU DEBUG] Raw Acc Mean Norm: {acc_mag:.4f} (Expect ~9.81 for m/s^2)")
    print(f"[IMU DEBUG] Raw Gyro Mean Norm: {gyro_mag:.4f} (Expect <3.0 for rad/s)")
    n_cols = int(imu_data.shape[1])
    print(f"[IMU DEBUG] Columns={n_cols}")
    if n_cols >= 10:
        imu_mag = imu_data[:, 7:10]
        mag_mag = float(np.mean(np.linalg.norm(imu_mag, axis=1)))
        print(f"[IMU DEBUG] Raw Mag Mean Norm: {mag_mag:.4f} (Expect tens of uT; check units uT/mT/gauss)")

    acc_unit = _norm_unit_str(calib.get("imu_acc_unit")) if isinstance(calib, dict) else ""
    gyro_unit = _norm_unit_str(calib.get("imu_gyro_unit")) if isinstance(calib, dict) else ""

    acc_unit_known = acc_unit in ("mps2", "m/s^2", "mps^2", "ms2", "m/s2", "g")
    gyro_unit_known = gyro_unit in ("rad", "rad/s", "rads", "deg", "deg/s", "dps")

    if acc_unit == "g":
        print("[IMU UNIT] Acc unit is 'g' from YAML. Converting to m/s^2 (* 9.81).")
        imu_acc = imu_acc * 9.81
    elif not acc_unit_known and (0.5 < acc_mag < 2.0):
        print(f"[IMU AUTO-FIX] Acc seems to be in 'g' (mean={acc_mag:.2f}). Converting to m/s^2 (* 9.81).")
        imu_acc = imu_acc * 9.81

    if gyro_unit in ("deg", "deg/s", "dps"):
        print(f"[IMU UNIT] Gyro unit is '{gyro_unit}' from YAML. Converting to rad/s (* pi/180).")
        imu_gyro = imu_gyro * (np.pi / 180.0)
    elif not gyro_unit_known and gyro_mag > 4.0:
        print(f"[IMU AUTO-FIX] Gyro seems to be in deg/s (mean={gyro_mag:.2f}). Converting to rad/s (* pi/180).")
        imu_gyro = imu_gyro * (np.pi / 180.0)

    imu_ordered = np.concatenate([imu_acc, imu_gyro], axis=1)
    imu_vals = dataset._normalize_imu_data(imu_ordered.astype(np.float32))
    return IMUPreprocessResult(imu_t=imu_t, imu_vals=imu_vals)


def preprocess_gt(
    dataset: Any,
    *,
    gt_path: Path,
    gt_data: np.ndarray,
) -> GTPreprocessResult:
    """
    Preprocess ground-truth poses: timestamp correction and column parsing.
    """
    calib = getattr(dataset, "calib", None)
    gt_time_unit = None
    if isinstance(calib, dict):
        gt_time_unit = calib.get("mocap_time_unit") or calib.get("gt_time_unit") or calib.get("groundtruth_time_unit")
    gt_t = dataset._correct_timestamps(gt_data[:, 0], unit=gt_time_unit)
    gt_pos, gt_quat = dataset._parse_gt_columns(str(gt_path), gt_data)
    return GTPreprocessResult(gt_t=gt_t, gt_pos=gt_pos, gt_quat=gt_quat)


def preprocess_events(
    dataset: Any,
    *,
    root_dir: Path,
    h5_path: Path,
    gt_t: np.ndarray,
    imu_t: Optional[np.ndarray],
    sensor_resolution: Optional[Tuple[int, int]],
    resolution: Tuple[int, int],
) -> EventsPreprocessResult:
    """
    Preprocess event HDF5: resolve dataset keys and infer timestamp units.

    Args:
        dataset: Dataset object providing calib and event_file_candidates.
        root_dir: Dataset root directory.
        h5_path: Path to the event HDF5 file.
        gt_t: Ground-truth timestamps (seconds).
        imu_t: IMU timestamps (seconds), optional.
        sensor_resolution: Sensor (H,W) if already known.
        resolution: Network processing resolution (H,W).

    Returns:
        EventsPreprocessResult with HDF5 key paths and timestamp scaling.
    """
    with h5py.File(str(h5_path), "r") as f:
        if all(k in f for k in ("x", "y")) and ("t" in f or "ts" in f):
            x_key, y_key = "x", "y"
            t_key = "t" if "t" in f else "ts"
            p_key = "p" if "p" in f else ("polarity" if "polarity" in f else None)
            ev_t_ds = f[t_key]
        elif "events" in f:
            g = f["events"]
            x_key, y_key = "events/x", "events/y"
            t_key = "events/t" if "t" in g else "events/ts"
            p_key = "events/p" if "p" in g else ("events/polarity" if "polarity" in g else None)
            ev_t_ds = g["t"] if "t" in g else g["ts"]
        else:
            raise KeyError(f"Unrecognized H5 structure: {list(f.keys())}")

        N = int(ev_t_ds.shape[0])
        k = max(int(1000), 1)
        t_coarse = ev_t_ds[0:N:k]

        gt_duration = float(gt_t[-1] - gt_t[0]) if gt_t.size > 1 else 0.0
        imu_duration = float("nan")
        if imu_t is not None and getattr(imu_t, "size", 0) > 1:
            imu_duration = float(imu_t[-1] - imu_t[0])
        ref_duration = imu_duration if (np.isfinite(imu_duration) and imu_duration > 0.01) else float(gt_duration)
        t_coarse_duration = float(t_coarse[-1] - t_coarse[0]) if t_coarse.size > 1 else 0.0

        unit_scale = 1.0
        if t_coarse.size > 1 and ref_duration > 0.01:
            ratio = t_coarse_duration / ref_duration
            if ratio > 5e7:
                unit_scale = 1e-9
                print(f"[DATASET] Detected nanoseconds (ratio {ratio:.1e}). Scaling by 1e-9.")
            elif ratio > 5e4:
                unit_scale = 1e-6
                print(f"[DATASET] Detected microseconds (ratio {ratio:.1e}). Scaling by 1e-6.")
            else:
                unit_scale = 1.0
                print(f"[DATASET] Detected seconds (ratio {ratio:.2f}). No scaling.")
        else:
            if t_coarse.size > 0 and float(t_coarse[-1]) > 1e14:
                unit_scale = 1e-6
                print("[DATASET] Fallback: detected nanoseconds from large timestamp. Scaling by 1e-6.")

        if unit_scale != 1.0:
            t_coarse = t_coarse * unit_scale

        if sensor_resolution is None:
            h_attr = f.attrs.get("height") or f.attrs.get("sensor_height")
            w_attr = f.attrs.get("width") or f.attrs.get("sensor_width")
            if h_attr is None or w_attr is None:
                try:
                    if "/" in x_key and "/" in y_key:
                        gk_x, dk_x = x_key.split("/", 1)
                        gk_y, dk_y = y_key.split("/", 1)
                        dx = f[gk_x][dk_x]
                        dy = f[gk_y][dk_y]
                    else:
                        dx = f[x_key]
                        dy = f[y_key]
                    n_probe = min(int(dx.shape[0]), 100000)
                    xs = dx[:n_probe]
                    ys = dy[:n_probe]
                    h_attr = int(np.max(ys) + 1) if ys.size > 0 else None
                    w_attr = int(np.max(xs) + 1) if xs.size > 0 else None
                except Exception:
                    h_attr, w_attr = None, None
            if h_attr is not None and w_attr is not None:
                sensor_resolution = (int(h_attr), int(w_attr))
        if sensor_resolution is None:
            raise ValueError("Sensor resolution could not be determined from H5; please provide sensor_resolution.")

    fx_scaled = 1.0
    fy_scaled = 1.0
    cx_scaled = float(resolution[1]) * 0.5
    cy_scaled = float(resolution[0]) * 0.5
    camera_type = "pinhole"
    kb4_distortion = None

    calib = getattr(dataset, "calib", None)
    if isinstance(calib, dict):
        try:
            K_src = calib["K"] if "K" in calib else calib.get("camera", {})
            K_raw = {k: float(K_src.get(k, 1.0 if k in ("fx", "fy") else 0.0)) for k in ("fx", "fy", "cx", "cy")}
            if "resolution" in calib:
                sensor_w, sensor_h = calib["resolution"]
                src_h, src_w = int(sensor_h), int(sensor_w)
            else:
                src_h, src_w = sensor_resolution
            net_h, net_w = tuple(resolution)
            cam_type = str(K_src.get("camera_type", "pinhole")).lower()
            camera_type = cam_type
            if cam_type == "kb4":
                distortion = {
                    "k1": float(K_src.get("k1", 0.0)),
                    "k2": float(K_src.get("k2", 0.0)),
                    "k3": float(K_src.get("k3", 0.0)),
                    "k4": float(K_src.get("k4", 0.0)),
                }
                K_scaled, dist_scaled = rescale_intrinsics_kb4(K_raw, distortion, (src_h, src_w), (net_h, net_w))
                kb4_distortion = dist_scaled
                print(
                    f"[DATASET] Using KB4 fisheye model: "
                    f"k1={distortion['k1']:.6f}, k2={distortion['k2']:.6f}, k3={distortion['k3']:.6f}, k4={distortion['k4']:.6f}"
                )
            else:
                K_scaled, _ = rescale_intrinsics_pinhole(K_raw, (src_h, src_w), (net_h, net_w))
                print("[DATASET] Using Pinhole camera model")

            fx_scaled = float(K_scaled.get("fx", fx_scaled))
            fy_scaled = float(K_scaled.get("fy", fy_scaled))
            cx_scaled = float(K_scaled.get("cx", src_w * 0.5))
            cy_scaled = float(K_scaled.get("cy", src_h * 0.5))
        except Exception as e:
            print(f"[DATASET] Warning: Failed to parse camera intrinsics: {e}")

    return EventsPreprocessResult(
        h5_path=str(h5_path),
        x_key=str(x_key),
        y_key=str(y_key),
        t_key=str(t_key),
        p_key=p_key,
        unit_scale=float(unit_scale),
        t_coarse=np.asarray(t_coarse, dtype=np.float64),
        sensor_resolution=(int(sensor_resolution[0]), int(sensor_resolution[1])),
        fx_scaled=float(fx_scaled),
        fy_scaled=float(fy_scaled),
        cx_scaled=float(cx_scaled),
        cy_scaled=float(cy_scaled),
        camera_type=str(camera_type),
        kb4_distortion=kb4_distortion,
    )
