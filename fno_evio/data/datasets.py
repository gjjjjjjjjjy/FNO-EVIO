"""Datasets for FNO-EVIO (refactored from baseline training script).

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

Notes:
  This file preserves the baseline dataset behavior while factoring out the highest-entropy
  preprocessing routine into three dedicated helpers (events / IMU / GT).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fno_evio.data.events import AdaptiveEventProcessor
from fno_evio.data.preprocess import preprocess_events, preprocess_gt, preprocess_imu
from fno_evio.utils.camera import kb4_unproject
from fno_evio.utils.events_warp import warp_events_flow_torch, warp_events_flow_torch_kb4
from fno_evio.utils.quaternion_np import QuaternionUtils

try:
    import hdf5plugin  # noqa: F401
except Exception:
    hdf5plugin = None


def _norm_unit_str(u: Any) -> str:
    if u is None:
        return ""
    try:
        s = str(u).strip().lower()
    except Exception:
        return ""
    return s.replace(" ", "")


def _resolve_existing_path(
    p: Any,
    bases: List[Path],
    *,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Optional[Path]:
    if p is None:
        return None
    try:
        pp = Path(str(p)).expanduser()
    except Exception:
        return None

    candidates: List[Path] = []
    if pp.is_absolute():
        candidates.append(pp)
    else:
        for b in bases:
            candidates.append(b / pp)

    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            pass
        if not c.exists():
            continue
        if must_be_file and not c.is_file():
            continue
        if must_be_dir and not c.is_dir():
            continue
        return c

    return None


def _pick_shortest_path(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: (len(p.as_posix()), p.as_posix()))[0]


def _find_first_matching_file(
    search_dirs: List[Path],
    patterns: Tuple[str, ...],
    *,
    recursive_root: Optional[Path] = None,
) -> Optional[Path]:
    hits: List[Path] = []
    for d in search_dirs:
        for pat in patterns:
            try:
                hits.extend([p for p in d.glob(pat) if p.is_file()])
            except Exception:
                pass

    if (not hits) and (recursive_root is not None):
        for pat in patterns:
            try:
                hits.extend([p for p in recursive_root.rglob(pat) if p.is_file()])
            except Exception:
                pass

    return _pick_shortest_path(hits)

@dataclass
class OptimizedTUMDataset(Dataset):
    """
    Optimized dataset for event-based VIO training.

    This dataset loads:
      - Events from an HDF5 file (x, y, t, polarity)
      - IMU readings from a text file
      - Ground-truth poses from a text file

    It supports two windowing modes:
      - imu: windowing by IMU time and interpolated GT at window endpoints
      - gt: windowing by GT sample indices (with sample_stride)

    Args:
        root: Dataset directory containing event/imu/gt files.
        dt: Window size in seconds for imu windowing.
        resolution: Voxel resolution (H, W).
        sequence_length: Number of time steps per training sample.
        events_h5: Optional explicit path to events HDF5.
        sample_stride: Stride for gt windowing mode.
        windowing_mode: Either 'imu' or 'gt'.
        window_dt: Optional explicit dt for imu windowing (defaults to dt).
        calib: Optional calibration dictionary (camera intrinsics, time units/offsets).
    """

    root: str
    dt: float
    resolution: Tuple[int, int]
    sequence_length: int = 50
    events_h5: Optional[str] = None
    sample_stride: int = 1
    windowing_mode: str = "imu"
    window_dt: Optional[float] = None
    event_offset_scan: bool = False
    event_offset_scan_range_s: float = 0.5
    event_offset_scan_step_s: float = 0.01
    voxelize_in_dataset: bool = True
    derotate: bool = False
    calib: Optional[Dict[str, Any]] = None
    sensor_resolution: Optional[Tuple[int, int]] = None
    event_file_candidates: Optional[Tuple[str, ...]] = None
    proc_device: Optional[torch.device] = None
    std_norm: bool = False
    log_norm: bool = True
    augment: bool = False
    adaptive_voxel: bool = True
    event_noise_scale: float = 0.01
    event_scale_jitter: float = 0.1
    imu_bias_scale: float = 0.02
    imu_mask_prob: float = 0.0
    adaptive_base_div: int = 60
    adaptive_max_events_div: int = 12
    adaptive_density_cap: float = 2.0

    def __post_init__(self):
        """Single-pass data loading with vectorized operations."""
        self.H, self.W = self.resolution
        stride_factor = max(int(self.sample_stride), 1)
        mode = str(getattr(self, "windowing_mode", "imu")).strip().lower()
        if mode not in ("imu", "gt"):
            mode = "imu"
        self.windowing_mode = mode
        if self.windowing_mode == "imu":
            wd = getattr(self, "window_dt", None)
            try:
                wd_f = float(wd) if wd is not None else float("nan")
            except Exception:
                wd_f = float("nan")
            if not np.isfinite(wd_f) or wd_f <= 0.0:
                wd_f = float(self.dt)
            self.window_dt = float(wd_f)
            self.sample_stride = 1
        else:
            self.window_dt = None
            self.sample_stride = stride_factor
        self.window_t_prev = None
        self.window_t_curr = None
        if self.event_file_candidates is None:
            self.event_file_candidates = ("events-left.h5", "events_left.h5", "mocap-6dof-events_left.h5")
        self.voxelizer = AdaptiveEventProcessor(
            resolution=self.resolution,
            device=self.proc_device or torch.device('cpu'),
            std_norm=self.std_norm,
            log_norm=self.log_norm
        )
        self._preprocess_data()
        self._precompute_indices()
        self.h5 = None

    def _preprocess_data(self):
        """Unified data preprocessing - loads and aligns all sensor data."""
        root_p = Path(self.root)

        search_dirs = [root_p]
        gt_sub = root_p / "mocap-6dof-vi_gt_data"
        if gt_sub.exists():
            search_dirs.append(gt_sub)

        imu_hint = None
        gt_hint = None
        if isinstance(self.calib, dict):
            imu_hint = self.calib.get("imu_path") or self.calib.get("imu_file")
            gt_hint = self.calib.get("gt_path") or self.calib.get("gt_file") or self.calib.get("mocap_path") or self.calib.get("mocap_file")

        imu_path = _resolve_existing_path(imu_hint, bases=search_dirs, must_be_file=True)
        gt_path = _resolve_existing_path(gt_hint, bases=search_dirs, must_be_file=True)

        if imu_path is None:
            imu_path = _resolve_existing_path("imu_data.txt", bases=search_dirs, must_be_file=True)
        if gt_path is None:
            gt_path = _resolve_existing_path("mocap_data.txt", bases=search_dirs, must_be_file=True)

        if imu_path is None:
            imu_path = _find_first_matching_file(search_dirs, ("imu*.txt", "*imu*.txt", "imu*.csv", "*imu*.csv"), recursive_root=root_p)
        if gt_path is None:
            gt_path = _find_first_matching_file(
                search_dirs,
                ("mocap*.txt", "*mocap*.txt", "gt*.txt", "*gt*.txt", "mocap*.csv", "*mocap*.csv", "gt*.csv", "*gt*.csv"),
                recursive_root=root_p,
            )

        if imu_path is None or gt_path is None:
            raise FileNotFoundError(
                f"IMU/GT txt not found under root={str(root_p)} | imu={imu_path} gt={gt_path}. "
                f"Expected imu_data.txt & mocap_data.txt, or configure calib keys imu_path/gt_path."
            )

        imu_data = np.loadtxt(str(imu_path), dtype=np.float64, skiprows=1)
        gt_data = np.loadtxt(str(gt_path), dtype=np.float64, skiprows=1)

        imu_res = preprocess_imu(self, imu_path=Path(str(imu_path)), imu_data=imu_data)
        self.imu_t = imu_res.imu_t
        self.imu_vals = imu_res.imu_vals

        gt_res = preprocess_gt(self, gt_path=Path(str(gt_path)), gt_data=gt_data)
        self.gt_t = gt_res.gt_t
        self.gt_pos = gt_res.gt_pos
        self.gt_quat = gt_res.gt_quat

        if self.events_h5 and Path(self.events_h5).exists():
            self.h5_path = self.events_h5
        else:
            candidates = [root_p / name for name in self.event_file_candidates] + [root_p.parent / name for name in self.event_file_candidates]
            self.h5_path = next((str(p) for p in candidates if p.exists()), None)

            if self.h5_path is None:
                hits_root = [p for p in root_p.glob("*.h5") if p.is_file()] + [p for p in root_p.glob("*.hdf5") if p.is_file()]
                hits_parent = [p for p in root_p.parent.glob("*.h5") if p.is_file()] + [p for p in root_p.parent.glob("*.hdf5") if p.is_file()]
                hits = hits_root + hits_parent
                if hits:
                    preferred = [p for p in hits if "event" in p.name.lower()]
                    picks = preferred if preferred else hits
                    picks = sorted(picks, key=lambda p: (p.parent != root_p, len(p.name), p.name))
                    self.h5_path = str(picks[0])

        if self.h5_path is None:
            raise FileNotFoundError("Events H5 not found. Provide --events_h5 or place file in root/parent.")

        ev_res = preprocess_events(
            self,
            root_dir=root_p,
            h5_path=Path(str(self.h5_path)),
            gt_t=self.gt_t,
            imu_t=self.imu_t,
            sensor_resolution=self.sensor_resolution,
            resolution=tuple(self.resolution),
        )
        self._x_key = ev_res.x_key
        self._y_key = ev_res.y_key
        self._t_key = ev_res.t_key
        self._p_key = ev_res.p_key
        self.unit_scale = ev_res.unit_scale
        self.t_coarse = ev_res.t_coarse
        self.sensor_resolution = ev_res.sensor_resolution
        self.fx_scaled = ev_res.fx_scaled
        self.fy_scaled = ev_res.fy_scaled
        self.cx_scaled = ev_res.cx_scaled
        self.cy_scaled = ev_res.cy_scaled
        self.camera_type = ev_res.camera_type
        self.kb4_distortion = ev_res.kb4_distortion

        if self.t_coarse.size > 0:
            t_ev_start = float(self.t_coarse[0])
            t_gt_start = float(self.gt_t[0])
            print(f"[DATASET] Time Alignment Check: Events Start={t_ev_start:.4f}, GT Start={t_gt_start:.4f}")
            misalign_thresh = float(self.calib.get("auto_align_threshold_sec", 0.5)) if isinstance(self.calib, dict) else 0.5
            auto_align = bool(self.calib.get("auto_align_enable", False)) if isinstance(self.calib, dict) else False
            if auto_align and abs(t_gt_start - t_ev_start) > misalign_thresh:
                offset = t_ev_start - t_gt_start
                self.gt_t = self.gt_t + offset
                self.imu_t = self.imu_t + offset
                print(f"[DATASET] Applied auto-align offset: {offset:.4f}s (threshold={misalign_thresh:.2f}s)")
            diff = abs(self.gt_t[0] - t_ev_start)
            if diff > misalign_thresh and not auto_align:
                print(f"[DATASET] WARNING: Start times differ by {diff:.2f}s. Configure 'mocap_time_offset_ns'/'mocap_to_imu_offset_ns' or enable 'auto_align_enable' in calib.")
            else:
                print(f"[DATASET] Alignment status: Diff={diff:.4f}s (threshold={misalign_thresh:.2f}s)")
        self._validate_data_integrity()

    def _validate_data_integrity(self):
        """
        Validate critical arrays for numeric integrity and minimal invariants.

        Raises:
            ValueError: If NaN/Inf values are detected in core arrays, or if GT timestamps are not
                strictly increasing.
        """
        for name, arr in [("gt_t", self.gt_t), ("gt_pos", self.gt_pos),
                         ("gt_quat", self.gt_quat), ("imu_t", self.imu_t),
                         ("imu_vals", self.imu_vals)]:
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN detected in {name}")
            if np.any(np.isinf(arr)):
                raise ValueError(f"Inf detected in {name}")
        quat_norms = np.linalg.norm(self.gt_quat, axis=1)
        if not np.allclose(quat_norms, 1.0, atol=1e-6):
            print(f"Warning: Some quaternions not normalized, renormalizing...")
            self.gt_quat = QuaternionUtils.normalize(self.gt_quat)
        if not np.all(np.diff(self.gt_t) > 0):
            raise ValueError("Ground truth timestamps are not monotonically increasing")
        if not np.all(np.diff(self.imu_t) > 0):
            print("Warning: IMU timestamps are not perfectly monotonic")
        if np.any(self.gt_t < 0) or np.any(self.imu_t < 0):
            print("Warning: Negative timestamps detected")
        acc_ranges = np.max(np.abs(self.imu_vals[:, :3]), axis=0)
        gyro_ranges = np.max(np.abs(self.imu_vals[:, 3:6]), axis=0)
        if np.any(acc_ranges > 50):  # > 5g
            print(f"Warning: Large accelerometer values detected: {acc_ranges}")
        if np.any(gyro_ranges > 20):  # > 20 rad/s
            print(f"Warning: Large gyroscope values detected: {gyro_ranges}")

    def _validate_events_data(self, xw, yw, tw, pw):
        """
        Sanity-check a raw event packet.

        Args:
            xw, yw: Event pixel coordinates.
            tw: Event timestamps.
            pw: Event polarities, or None.

        Raises:
            ValueError: If NaN is found in event coordinates.
        """
        if xw.size == 0:
            return
        if np.any(np.isnan(xw)) or np.any(np.isnan(yw)) or np.any(np.isnan(tw)):
            raise ValueError("NaN detected in event coordinates")
        if np.any(np.abs(xw) > 1e6) or np.any(np.abs(yw) > 1e6):
            print(f"Warning: Extreme event coordinates detected")
        if tw.size > 0:
            t_min, t_max = tw.min(), tw.max()
            if t_max - t_min > 1000:  # More than 1000 seconds window
                print(f"Warning: Large event time window: {t_max - t_min:.2f}s")
        if pw is not None and pw.size > 0:
            unique_pol = np.unique(pw)
            if not np.all(np.isin(unique_pol, [-1, 0, 1])):
                print(f"Warning: Unexpected polarity values: {unique_pol}")

    def interpolate_gt_data(self, query_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate GT pose at an arbitrary timestamp.

        Args:
            query_time: Target timestamp in seconds.

        Returns:
            (pos, quat): Interpolated position (3,) and quaternion (4,) in [x,y,z,w] order.
        """
        idx = np.searchsorted(self.gt_t, query_time)
        if idx == 0:
            return self.gt_pos[0], self.gt_quat[0]
        if idx >= len(self.gt_t):
            return self.gt_pos[-1], self.gt_quat[-1]
        t1, t2 = self.gt_t[idx-1], self.gt_t[idx]
        pos1, pos2 = self.gt_pos[idx-1], self.gt_pos[idx]
        quat1, quat2 = self.gt_quat[idx-1], self.gt_quat[idx]
        if abs(t2 - t1) < 1e-9:
            return pos1, quat1
        alpha = (query_time - t1) / (t2 - t1)
        alpha = np.clip(alpha, 0.0, 1.0)
        pos_interp = (1 - alpha) * pos1 + alpha * pos2
        quat_interp = self._slerp(quat1, quat2, alpha)
        return pos_interp, quat_interp

    def _slerp(self, q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
        """
        Spherical linear interpolation (SLERP) for unit quaternions.

        Args:
            q1: Start quaternion [x,y,z,w].
            q2: End quaternion [x,y,z,w].
            alpha: Interpolation coefficient in [0, 1].

        Returns:
            Interpolated unit quaternion [x,y,z,w].
        """
        q1 = QuaternionUtils.normalize(q1)
        q2 = QuaternionUtils.normalize(q2)
        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        if abs(dot) > 0.9995:
            result = (1 - alpha) * q1 + alpha * q2
            return QuaternionUtils.normalize(result)
        omega = np.arccos(dot)
        sin_omega = np.sin(omega)
        factor1 = np.sin((1 - alpha) * omega) / sin_omega
        factor2 = np.sin(alpha * omega) / sin_omega

        return factor1 * q1 + factor2 * q2

    def interpolate_gt_velocity(self, query_time: float, dt: float) -> np.ndarray:
        """
        Approximate GT velocity by symmetric finite differences on interpolated positions.

        Args:
            query_time: Target timestamp.
            dt: Small delta time (seconds).

        Returns:
            Velocity vector (3,).
        """
        p_plus, _ = self.interpolate_gt_data(query_time + dt)
        p_minus, _ = self.interpolate_gt_data(query_time - dt)
        return (p_plus - p_minus) / (2 * dt)

    def interpolate_imu_data(self, query_time: float, window_size: float = 0.1) -> np.ndarray:
        """
        Aggregate IMU signals in a time window around query_time.

        Args:
            query_time: Center timestamp.
            window_size: Window width in seconds.

        Returns:
            A single 6D IMU vector (acc, gyro) normalized as in preprocessing.
        """
        mask = np.abs(self.imu_t - query_time) <= window_size / 2
        imu_in_window = self.imu_vals[mask]

        if len(imu_in_window) == 0:
            idx = np.searchsorted(self.imu_t, query_time)
            if idx == 0:
                return self.imu_vals[0]
            if idx >= len(self.imu_t):
                return self.imu_vals[-1]
            t1, t2 = self.imu_t[idx-1], self.imu_t[idx]
            imu1, imu2 = self.imu_vals[idx-1], self.imu_vals[idx]

            if abs(t2 - t1) < 1e-9:
                return imu1

            alpha = (query_time - t1) / (t2 - t1)
            alpha = np.clip(alpha, 0.0, 1.0)
            return (1 - alpha) * imu1 + alpha * imu2

        if len(imu_in_window) == 1:
            return imu_in_window[0]

        times_in_window = self.imu_t[mask]
        weights = 1.0 / (np.abs(times_in_window - query_time) + 1e-6)
        weights = weights / weights.sum()

        return np.sum(imu_in_window * weights[:, np.newaxis], axis=0)

    def _parse_gt_columns(self, gt_path: str, gt_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_cols = int(gt_data.shape[1])
        header_cols = None
        try:
            with open(gt_path, "r") as f:
                first_line = f.readline().strip().lower()
                if first_line and (first_line.startswith("#") or not first_line[0].isdigit()):
                    header_cols = [c.strip() for c in first_line.replace("#", "").split()]
        except Exception:
            header_cols = None

        def _norm_cols(v):
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                out = []
                for x in v:
                    try:
                        out.append(int(x))
                    except Exception:
                        pass
                return out
            if isinstance(v, str):
                parts = [p.strip() for p in v.replace(",", " ").split() if p.strip()]
                out = []
                for p in parts:
                    try:
                        out.append(int(p))
                    except Exception:
                        pass
                return out
            try:
                return [int(v)]
            except Exception:
                return []

        pos_cols = []
        quat_cols = []
        if isinstance(self.calib, dict):
            pos_cols = _norm_cols(self.calib.get("gt_pos_cols") or self.calib.get("mocap_pos_cols"))
            quat_cols = _norm_cols(self.calib.get("gt_quat_cols") or self.calib.get("mocap_quat_cols"))

        def _fix_1_based(cols):
            if not cols:
                return cols
            if all(1 <= c <= n_cols for c in cols):
                cols0 = [c - 1 for c in cols]
                if all(0 <= c < n_cols for c in cols0):
                    return cols0
            return cols

        pos_cols = _fix_1_based(pos_cols)
        quat_cols = _fix_1_based(quat_cols)

        if len(pos_cols) == 3 and len(quat_cols) == 4 and all(0 <= c < n_cols for c in pos_cols + quat_cols):
            pos = gt_data[:, pos_cols].astype(np.float32)
            quat_raw = gt_data[:, quat_cols].astype(np.float64)
            w_idx = int(np.argmax(np.mean(np.abs(quat_raw), axis=0))) if quat_raw.shape[0] > 0 else 3
            if w_idx != 3:
                order = [i for i in range(4) if i != w_idx] + [w_idx]
                quat_raw = quat_raw[:, order]
            return pos, QuaternionUtils.normalize(quat_raw)

        if header_cols and len(header_cols) == n_cols:
            pos_names = {
                "x", "y", "z", "px", "py", "pz", "tx", "ty", "tz",
                "pos_x", "pos_y", "pos_z", "p_x", "p_y", "p_z",
                "position_x", "position_y", "position_z",
            }
            quat_names = {
                "qx", "qy", "qz", "qw",
                "quat_x", "quat_y", "quat_z", "quat_w",
                "q_x", "q_y", "q_z", "q_w",
                "orientation_x", "orientation_y", "orientation_z", "orientation_w",
            }

            idx_q = {}
            idx_p = {}
            for i, col in enumerate(header_cols):
                c = col.lower()
                if c in ("t", "ts", "timestamp", "time"):
                    continue
                if c in pos_names:
                    idx_p[c] = i
                if c in quat_names:
                    idx_q[c] = i

            have_p = all(k in idx_p for k in ("x", "y", "z")) or all(k in idx_p for k in ("px", "py", "pz")) or all(k in idx_p for k in ("tx", "ty", "tz"))
            have_q = all(k in idx_q for k in ("qx", "qy", "qz", "qw"))

            if have_p and have_q:
                if all(k in idx_p for k in ("tx", "ty", "tz")):
                    pos_cols = [idx_p["tx"], idx_p["ty"], idx_p["tz"]]
                elif all(k in idx_p for k in ("px", "py", "pz")):
                    pos_cols = [idx_p["px"], idx_p["py"], idx_p["pz"]]
                else:
                    pos_cols = [idx_p["x"], idx_p["y"], idx_p["z"]]

                quat_cols = [idx_q["qx"], idx_q["qy"], idx_q["qz"], idx_q["qw"]]
                pos = gt_data[:, pos_cols].astype(np.float32)
                quat_raw = gt_data[:, quat_cols].astype(np.float64)
                return pos, QuaternionUtils.normalize(quat_raw)

        if n_cols >= 8:
            q1 = gt_data[:, 1:5].astype(np.float64)
            q2 = gt_data[:, 4:8].astype(np.float64)

            def _score(q):
                n = np.linalg.norm(q, axis=1)
                n = n[np.isfinite(n)]
                if n.size == 0:
                    return 1e9
                return float(np.mean(np.abs(n - 1.0)))

            s1 = _score(q1)
            s2 = _score(q2)

            if s1 <= s2:
                quat_raw = q1
                pos = gt_data[:, 5:8].astype(np.float32)
            else:
                quat_raw = q2
                pos = gt_data[:, 1:4].astype(np.float32)

            w_idx = int(np.argmax(np.mean(np.abs(quat_raw), axis=0))) if quat_raw.shape[0] > 0 else 3
            if w_idx != 3:
                order = [i for i in range(4) if i != w_idx] + [w_idx]
                quat_raw = quat_raw[:, order]

            print(f"[GT PARSE] Heuristic: score(q@1:5)={s1:.3e}, score(q@4:8)={s2:.3e}, w_idx={w_idx}")
            return pos, QuaternionUtils.normalize(quat_raw)

        pos = gt_data[:, 1:4].astype(np.float32) if n_cols >= 4 else np.zeros((gt_data.shape[0], 3), dtype=np.float32)
        quat_raw = gt_data[:, 4:8].astype(np.float64) if n_cols >= 8 else np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (gt_data.shape[0], 1))
        return pos, QuaternionUtils.normalize(quat_raw)

    def _parse_imu_columns(self, imu_path: str, imu_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        智能解析 IMU 数据列顺序，支持多种格式:
        - TUM-VIE: timestamp gx gy gz ax ay az ...
        - MVSEC:   timestamp ax ay az gx gy gz
        - EuRoC:   timestamp wx wy wz ax ay az

        Returns:
            imu_acc: [N, 3] 加速度数据
            imu_gyro: [N, 3] 陀螺仪数据
        """
        # 尝试读取文件头
        header_cols = None
        try:
            with open(imu_path, 'r') as f:
                first_line = f.readline().strip().lower()
                # 检查是否是注释或标题行
                if first_line.startswith('#') or not first_line[0].isdigit():
                    header_cols = [c.strip() for c in first_line.replace('#', '').split()]
        except Exception:
            pass

        # 定义列名映射
        acc_names = {'ax', 'ay', 'az', 'accel_x', 'accel_y', 'accel_z', 'a_x', 'a_y', 'a_z'}
        gyro_names = {'gx', 'gy', 'gz', 'gyro_x', 'gyro_y', 'gyro_z', 'wx', 'wy', 'wz',
                      'omega_x', 'omega_y', 'omega_z', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'}

        acc_indices = []
        gyro_indices = []

        if header_cols and len(header_cols) >= 7:
            for i, col in enumerate(header_cols):
                col_lower = col.lower()
                if col_lower in acc_names:
                    acc_indices.append(i)
                elif col_lower in gyro_names:
                    gyro_indices.append(i)

            if len(acc_indices) == 3 and len(gyro_indices) == 3:
                print(f"[IMU PARSE] Detected columns from header: acc={acc_indices}, gyro={gyro_indices}")
                imu_acc = imu_data[:, acc_indices]
                imu_gyro = imu_data[:, gyro_indices]
                return imu_acc, imu_gyro

        # 回退：使用启发式方法检测
        # TUM-VIE 格式: timestamp gx gy gz ax ay az (gyro在前)
        # 检查列1-3和列4-6的量级
        if imu_data.shape[1] >= 7:
            col_1_3 = imu_data[:, 1:4]
            col_4_6 = imu_data[:, 4:7]

            mag_1_3 = np.mean(np.linalg.norm(col_1_3, axis=1))
            mag_4_6 = np.mean(np.linalg.norm(col_4_6, axis=1))

            # 加速度量级 ~9.81 (m/s^2) 或 ~1 (g)
            # 陀螺仪量级 ~0.1-3.0 (rad/s) 或 ~10-100 (deg/s)

            # 如果 col_4_6 的量级接近重力加速度，则是 TUM-VIE 格式 (gyro, acc)
            if 5.0 < mag_4_6 < 15.0 and mag_1_3 < 5.0:
                print(f"[IMU PARSE] Heuristic: TUM-VIE format (gyro@1:4, acc@4:7), mag_gyro={mag_1_3:.2f}, mag_acc={mag_4_6:.2f}")
                return col_4_6, col_1_3
            # 如果 col_1_3 的量级接近重力加速度，则是 MVSEC/EuRoC 格式 (acc, gyro)
            elif 5.0 < mag_1_3 < 15.0 and mag_4_6 < 5.0:
                print(f"[IMU PARSE] Heuristic: MVSEC/EuRoC format (acc@1:4, gyro@4:7), mag_acc={mag_1_3:.2f}, mag_gyro={mag_4_6:.2f}")
                return col_1_3, col_4_6
            # 如果 col_1_3 接近 1g (单位是 g)
            elif 0.5 < mag_1_3 < 2.0:
                print(f"[IMU PARSE] Heuristic: acc in 'g' units (acc@1:4, gyro@4:7), mag_acc={mag_1_3:.2f}")
                return col_1_3, col_4_6
            elif 0.5 < mag_4_6 < 2.0:
                print(f"[IMU PARSE] Heuristic: acc in 'g' units (gyro@1:4, acc@4:7), mag_acc={mag_4_6:.2f}")
                return col_4_6, col_1_3

        # 最终回退：假设 TUM-VIE 格式 (gyro@1:4, acc@4:7)
        print(f"[IMU PARSE] Fallback: Assuming TUM-VIE format (gyro@1:4, acc@4:7)")
        return imu_data[:, 4:7], imu_data[:, 1:4]

    def _normalize_imu_data(self, imu_vals: np.ndarray) -> np.ndarray:
        accel_normalized = imu_vals[:, :3] / 9.81
        gyro_normalized = imu_vals[:, 3:6] / np.pi
        normalized = np.concatenate([accel_normalized, gyro_normalized], axis=1)
        normalized = np.clip(normalized, -10.0, 10.0)

        return normalized.astype(np.float32)

    def _correct_timestamps(self, timestamps: np.ndarray, unit: Optional[str] = None) -> np.ndarray:

        result = timestamps.astype(np.float64)
        if result.size == 0:
            return result
        filtered_result = result[np.isfinite(result)]
        if filtered_result.size == 0:
            print("Warning: All timestamps are invalid (NaN/Inf)")
            return result

        unit_n = _norm_unit_str(unit)
        if unit_n:
            if unit_n in ("s", "sec", "secs", "second", "seconds"):
                pass
            elif unit_n in ("ms", "msec", "millisecond", "milliseconds"):
                result = result / 1e3
                print("Converted milliseconds to seconds (from YAML unit)")
            elif unit_n in ("us", "usec", "microsecond", "microseconds", "µs"):
                result = result / 1e6
                print("Converted microseconds to seconds (from YAML unit)")
            elif unit_n in ("ns", "nsec", "nanosecond", "nanoseconds"):
                result = result / 1e9
                print("Converted nanoseconds to seconds (from YAML unit)")
            else:
                unit_n = ""

        if not unit_n:
            mean_val = np.mean(filtered_result)
            std_val = np.std(filtered_result)

            if mean_val > 1e16 and std_val > 1e12:  # Nanoseconds (e.g., 1.7e18 for recent Unix timestamps)
                result = result / 1e9
                print(f"Converted nanoseconds to seconds (mean={mean_val:.0f})")
            elif mean_val > 1e12 and std_val > 1e9:  # Microseconds
                result = result / 1e6
                print(f"Converted microseconds to seconds (mean={mean_val:.0f})")
            elif 1e9 < mean_val < 3e9:  # Seconds (valid modern Unix timestamps ~2001-2065)
                pass
            elif mean_val > 1e4:
                result = result / 1e6
                print(f"Converted small-range microseconds to seconds (mean={mean_val:.0f})")
            else:
                pass

        if np.any(result < 0):
            print(f"Warning: {np.sum(result < 0)} negative timestamps detected")

        return result

    def _resolve_h5_keys(self, f: h5py.File):
        try:
            return f[self._t_key]
        except Exception:
            if "/" in self._t_key:
                g_key, d_key = self._t_key.split("/", 1)
                return f[g_key][d_key]
            raise

    def _events_time_offset_sec(self) -> float:
        if isinstance(self.calib, dict):
            off_ev_ns = float(self.calib.get("events_time_offset_ns", 0.0))
        else:
            off_ev_ns = 0.0
        return off_ev_ns * 1e-9

    def _h5_searchsorted(self, ev_t_ds, target_t: float, side: str, unit_scale: float, off_sec: float) -> int:
        n = int(ev_t_ds.shape[0])
        lo, hi = 0, n
        if side == "left":
            while lo < hi:
                mid = (lo + hi) // 2
                v = float(ev_t_ds[mid]) * unit_scale + off_sec
                if v < target_t:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        while lo < hi:
            mid = (lo + hi) // 2
            v = float(ev_t_ds[mid]) * unit_scale + off_sec
            if v <= target_t:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def _precompute_event_window_indices(self, t_prev_all: np.ndarray, t_curr_all: np.ndarray):
        with h5py.File(self.h5_path, "r") as f:
            ev_t_ds = self._resolve_h5_keys(f)
            n_events = int(ev_t_ds.shape[0])
            n_win = int(t_prev_all.shape[0])
            if n_events <= 0 or n_win <= 0:
                z = np.zeros((n_win,), dtype=np.int64)
                return z, z

            unit_scale = float(self.unit_scale)
            off_sec = float(self._events_time_offset_sec())
            chunk_size = 1_000_000

            start_idx = np.empty((n_win,), dtype=np.int64)
            end_idx = np.empty((n_win,), dtype=np.int64)

            chunk_start = 0
            chunk_end = 0
            t_chunk = None

            def load_chunk(s: int):
                nonlocal chunk_start, chunk_end, t_chunk
                chunk_start = int(s)
                chunk_end = int(min(chunk_start + chunk_size, n_events))
                t_chunk = ev_t_ds[chunk_start:chunk_end].astype(np.float64)
                if unit_scale != 1.0:
                    t_chunk = t_chunk * unit_scale
                if off_sec != 0.0:
                    t_chunk = t_chunk + off_sec

            def advance(ptr: int, target: float, side0: str) -> int:
                nonlocal chunk_start, chunk_end, t_chunk
                if ptr < 0:
                    ptr = 0
                if ptr > n_events:
                    ptr = n_events
                while True:
                    if ptr >= n_events:
                        return n_events
                    if t_chunk is None or not (chunk_start <= ptr < chunk_end):
                        load_chunk((ptr // chunk_size) * chunk_size)
                    sub = int(ptr - chunk_start)
                    if side0 == "left":
                        pos = sub + int(np.searchsorted(t_chunk[sub:], target, side="left"))
                    else:
                        pos = sub + int(np.searchsorted(t_chunk[sub:], target, side="right"))
                    ptr_new = int(chunk_start + pos)
                    if ptr_new < chunk_end or chunk_end >= n_events:
                        return ptr_new
                    ptr = chunk_end

            ptr_s = 0
            ptr_e = 0
            for i in range(n_win):
                t0 = float(t_prev_all[i])
                t1 = float(t_curr_all[i])
                ptr_s = advance(ptr_s, t0, "left")
                if ptr_e < ptr_s:
                    ptr_e = ptr_s
                ptr_e = advance(ptr_e, t1, "right")
                start_idx[i] = ptr_s
                end_idx[i] = ptr_e

            return start_idx, end_idx

    def _detect_segment_boundaries(self) -> np.ndarray:
        n = int(getattr(self, "curr_indices", np.zeros((0,), dtype=np.int64)).shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.int32)

        curr = np.asarray(self.curr_indices, dtype=np.int64)
        seg = np.zeros((n,), dtype=np.int32)

        if n == 1:
            print("[DATASET] Detected 1 segments across 1 samples")
            return seg

        expected_step = int(getattr(self, "sample_stride", 1) or 1)
        if expected_step <= 0:
            diffs_pos = curr[1:] - curr[:-1]
            diffs_pos = diffs_pos[diffs_pos > 0]
            expected_step = int(np.median(diffs_pos)) if diffs_pos.size else 1

        gap_thr = int(max(expected_step * 2, 10))
        try:
            user_thr = int(getattr(self, "segment_gap_threshold"))
            if user_thr > 0:
                gap_thr = user_thr
        except Exception:
            pass

        diffs = curr[1:] - curr[:-1]
        is_gap = diffs > gap_thr

        wtc = getattr(self, "window_t_curr", None)
        wdt = getattr(self, "window_dt", None)
        try:
            wdt_f = float(wdt) if wdt is not None else float("nan")
        except Exception:
            wdt_f = float("nan")

        if wtc is not None and np.isfinite(wdt_f) and wdt_f > 0.0:
            try:
                wtc = np.asarray(wtc, dtype=np.float64)
                if wtc.size == n:
                    dt_jump = wtc[1:] - wtc[:-1]
                    tol = max(abs(wdt_f) * 0.5, 1e-6)
                    is_time_gap = (dt_jump <= 0.0) | (np.abs(dt_jump - wdt_f) > tol)
                    is_gap = is_gap | is_time_gap
            except Exception:
                pass

        seg[1:] = np.cumsum(is_gap.astype(np.int32))
        n_segments = int(seg.max()) + 1 if seg.size else 0

        gap_idx = np.nonzero(is_gap)[0]
        if gap_idx.size:
            show_n = int(min(20, gap_idx.size))
            for k in range(show_n):
                i = int(gap_idx[k])
                print(f"[SEGMENT DETECT] Gap at idx={i}: curr={int(curr[i])} -> next={int(curr[i + 1])}, gap={int(diffs[i])} frames")
            if gap_idx.size > show_n:
                print(f"[SEGMENT DETECT] ... {int(gap_idx.size - show_n)} more gaps omitted")

        print(f"[DATASET] Detected {n_segments} segments across {int(n)} samples")
        return seg

    def _precompute_indices(self):
        mode = str(getattr(self, "windowing_mode", "imu")).strip().lower()
        if mode == "gt":
            self.window_t_prev = None
            self.window_t_curr = None
            self.dt_nominal = float("nan")

            self.sample_indices = np.arange(self.sample_stride, len(self.gt_t))
            self.prev_indices = self.sample_indices - self.sample_stride
            self.curr_indices = self.sample_indices

            dt_pairs = (self.gt_t[self.curr_indices] - self.gt_t[self.prev_indices]).astype(np.float64)
            valid = np.isfinite(dt_pairs) & (dt_pairs > 0)
            if np.any(valid):
                dt_nom = float(np.median(dt_pairs[valid]))
            else:
                dt_nom = float(self.dt)
            self.dt_nominal = dt_nom

            tol = 0.1
            keep = valid & (np.abs(dt_pairs - dt_nom) <= (abs(dt_nom) * tol))
            if np.any(~keep):
                n_drop = int(np.sum(~keep))
                n_tot = int(dt_pairs.shape[0])
                print(f"[DATASET] Dropping {n_drop}/{n_tot} samples due to irregular dt (nominal={dt_nom:.6e}).")
            self.sample_indices = self.sample_indices[keep]
            self.prev_indices = self.prev_indices[keep]
            self.curr_indices = self.curr_indices[keep]

            t_prev_all = self.gt_t[self.prev_indices]
            t_curr_all = self.gt_t[self.curr_indices]
        else:
            win_dt = getattr(self, "window_dt", None)
            try:
                win_dt_f = float(win_dt) if win_dt is not None else float("nan")
            except Exception:
                win_dt_f = float("nan")
            if not np.isfinite(win_dt_f) or win_dt_f <= 0.0:
                win_dt_f = float(self.dt)
            self.window_dt = float(win_dt_f)
            self.dt_nominal = float(self.window_dt)

            t0 = float(self.imu_t[0])
            tN = float(self.imu_t[-1])
            start = t0 + float(self.window_dt)
            if not (np.isfinite(start) and np.isfinite(tN) and (tN - start) > 0.0):
                self.window_t_curr = np.zeros((0,), dtype=np.float64)
                self.window_t_prev = np.zeros((0,), dtype=np.float64)
                self.sample_indices = np.zeros((0,), dtype=np.int64)
                self.prev_indices = np.zeros((0,), dtype=np.int64)
                self.curr_indices = np.zeros((0,), dtype=np.int64)
                self.segment_ids = np.zeros((0,), dtype=np.int32)
                self.h5_start_indices = None
                self.h5_end_indices = None
                return

            n_win = int(np.floor((tN - start) / float(self.window_dt))) + 1
            window_t_curr = (start + np.arange(n_win, dtype=np.float64) * float(self.window_dt)).astype(np.float64)
            window_t_prev = (window_t_curr - float(self.window_dt)).astype(np.float64)

            gt_t = self.gt_t.astype(np.float64)
            prev_gt = np.searchsorted(gt_t, window_t_prev, side="left")
            curr_gt = np.searchsorted(gt_t, window_t_curr, side="left")
            prev_gt = np.clip(prev_gt, 0, len(gt_t) - 1)
            curr_gt = np.clip(curr_gt, 0, len(gt_t) - 1)

            gt_t0 = float(gt_t[0]) if gt_t.size else float("nan")
            gt_tN = float(gt_t[-1]) if gt_t.size else float("nan")
            keep = (
                np.isfinite(window_t_prev)
                & np.isfinite(window_t_curr)
                & (window_t_curr > window_t_prev)
                & np.isfinite(gt_t0)
                & np.isfinite(gt_tN)
                & (window_t_prev >= gt_t0)
                & (window_t_curr <= gt_tN)
            )

            if np.any(~keep):
                n_drop = int(np.sum(~keep))
                n_tot = int(keep.shape[0])
                print(f"[DATASET] Dropping {n_drop}/{n_tot} windows outside GT time range (gt=[{gt_t0:.4f},{gt_tN:.4f}]s, dt={float(self.window_dt):.6e}).")

            window_t_prev = window_t_prev[keep]
            window_t_curr = window_t_curr[keep]
            prev_gt = prev_gt[keep].astype(np.int64)
            curr_gt = curr_gt[keep].astype(np.int64)

            self.window_t_prev = window_t_prev
            self.window_t_curr = window_t_curr
            self.prev_indices = prev_gt
            self.curr_indices = curr_gt
            self.sample_indices = self.curr_indices.copy()

            t_prev_all = self.window_t_prev
            t_curr_all = self.window_t_curr

        if bool(getattr(self, "event_offset_scan", False)):
            try:
                scan_range = float(getattr(self, "event_offset_scan_range_s", 0.5))
                scan_step = float(getattr(self, "event_offset_scan_step_s", 0.01))
            except Exception:
                scan_range = 0.5
                scan_step = 0.01

            if hasattr(self, "t_coarse") and getattr(self, "t_coarse") is not None:
                try:
                    ev0 = float(self.t_coarse[0]) + float(self._events_time_offset_sec())
                    ev1 = float(self.t_coarse[-1]) + float(self._events_time_offset_sec())
                    if np.isfinite(ev0) and np.isfinite(ev1) and ev1 > ev0 and t_prev_all.size > 0:
                        deltas = np.arange(-scan_range, scan_range + 1e-12, scan_step, dtype=np.float64)
                        t0 = t_prev_all.astype(np.float64)
                        t1 = t_curr_all.astype(np.float64)
                        cov = []
                        for d in deltas.tolist():
                            inside = (t0 >= (ev0 + d)) & (t1 <= (ev1 + d))
                            cov.append(float(np.mean(inside)) if inside.size else 0.0)
                        cov_arr = np.asarray(cov, dtype=np.float64)
                        best = int(np.argmax(cov_arr)) if cov_arr.size else 0
                        topk = np.argsort(-cov_arr)[:5]
                        top_str = ", ".join([f"{float(deltas[i]):+.3f}s:{float(cov_arr[i]):.3f}" for i in topk])
                        d_start = float(np.min(t0) - ev0)
                        print(f"[TIME OFFSET SCAN] events_range=[{ev0:.3f},{ev1:.3f}] | suggested_start_delta={d_start:+.3f}s")
                        print(f"[TIME OFFSET SCAN] coverage best_delta={float(deltas[best]):+.3f}s ({float(cov_arr[best]):.3f}) | top5 {top_str}")
                except Exception:
                    pass

        try:
            self.h5_start_indices, self.h5_end_indices = self._precompute_event_window_indices(t_prev_all, t_curr_all)

            counts = self.h5_end_indices - self.h5_start_indices
            mean_count = np.mean(counts) if len(counts) > 0 else 0
            if mean_count < 10:
                print(f"[DATASET] WARNING: Low event density (mean={mean_count:.1f} ev/window). Check time alignment/scaling!")
            elif mean_count > 500000:
                print(f"[DATASET] WARNING: Extremely high event density (mean={mean_count:.1f} ev/window).")

            if np.all(counts == 0):
                print("Warning: No events found in any window! Check timestamp units and alignment.")
        except Exception:
            self.h5_start_indices = None
            self.h5_end_indices = None

        try:
            self.segment_ids = self._detect_segment_boundaries()
        except Exception as e:
            try:
                n_seg = int(len(self.sample_indices))
            except Exception:
                n_seg = 0
            print(f"[DATASET] Warning: Segment detection failed: {e}")
            self.segment_ids = np.zeros((n_seg,), dtype=np.int32)


    def __len__(self) -> int:
        return len(self.sample_indices)

    def __del__(self):
        try:
            if getattr(self, 'h5', None) is not None:
                self.h5.close()
        except Exception:
            pass

    def __getstate__(self):
        """Ensure h5 handle is not pickled when spawning workers."""
        state = self.__dict__.copy()
        state['h5'] = None
        return state

    def _get_h5_file(self):
        """
        Lazy load HDF5 file handle in the worker process with optimized cache settings.

        优化说明:
        - rdcc_nbytes: 原始数据块缓存大小 (1GB)，缓存频繁访问的数据块
        - rdcc_nslots: 哈希表槽位数，使用质数以减少冲突
        - rdcc_w0: 缓存替换策略权重 (0.75 = 倾向保留常用块)
        - libver='latest': 使用最新的 HDF5 格式特性以获得更好性能
        """
        if self.h5 is None:
            self.h5 = h5py.File(
                self.h5_path,
                "r",
                rdcc_nbytes=1024*1024*1024,  # 1GB chunk cache
                rdcc_nslots=10_000_019,      # ~10M slots (prime number)
                rdcc_w0=0.75,                # Favor frequently used chunks
                libver='latest'
            )
        return self.h5

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        i_prev, i_curr = self.prev_indices[idx], self.curr_indices[idx]
        if getattr(self, "window_t_prev", None) is not None and getattr(self, "window_t_curr", None) is not None:
            t_prev = float(self.window_t_prev[idx])
            t_curr = float(self.window_t_curr[idx])
        else:
            t_prev, t_curr = self.gt_t[i_prev], self.gt_t[i_curr]

        if self.h5_start_indices is not None and self.h5_end_indices is not None:
            a, b = int(self.h5_start_indices[idx]), int(self.h5_end_indices[idx])
        else:
            h5 = self._get_h5_file()
            ev_t_ds = self._resolve_h5_keys(h5)
            unit_scale = float(self.unit_scale)
            off_sec = float(self._events_time_offset_sec())
            a = int(self._h5_searchsorted(ev_t_ds, float(t_prev), "left", unit_scale, off_sec))
            b = int(self._h5_searchsorted(ev_t_ds, float(t_curr), "right", unit_scale, off_sec))

        h5 = self._get_h5_file()
        xw = h5[self._x_key][a:b].astype(np.float32)
        yw = h5[self._y_key][a:b].astype(np.float32)
        tw64 = h5[self._t_key][a:b].astype(np.float64)
        if self.unit_scale != 1.0:
            tw64 = tw64 * self.unit_scale
        off_ev_ns = float(self.calib.get("events_time_offset_ns", 0.0)) if isinstance(self.calib, dict) else 0.0
        if off_ev_ns != 0.0:
            tw64 = tw64 + float(off_ev_ns) * 1e-9
        tw = (tw64 - float(t_prev)).astype(np.float32)
        pw = h5[self._p_key][a:b].astype(np.int64) if self._p_key else None


        self._validate_events_data(xw, yw, tw, pw)

        # KB4 去畸变处理 - 将鱼眼坐标转换为归一化针孔坐标
        # 注意：仅在 derotate=False 时执行，因为 derotate=True 时
        # warp_events_flow_torch_kb4 会在内部完整处理 KB4 几何（unproject→rotate→project）
        # 避免同一批坐标被重复当作 KB4 像素处理
        if not self.derotate and self.camera_type == "kb4" and self.kb4_distortion is not None and len(xw) > 0:
            src_h, src_w = self.sensor_resolution or self.resolution
            fx = float(getattr(self, "fx_scaled", 1.0))
            fy = float(getattr(self, "fy_scaled", 1.0))
            cx = float(getattr(self, "cx_scaled", src_w * 0.5))
            cy = float(getattr(self, "cy_scaled", src_h * 0.5))
            k1 = self.kb4_distortion.get("k1", 0.0)
            k2 = self.kb4_distortion.get("k2", 0.0)
            k3 = self.kb4_distortion.get("k3", 0.0)
            k4 = self.kb4_distortion.get("k4", 0.0)

            # 反投影到单位球面，然后重新投影为针孔模型坐标
            X, Y, Z = kb4_unproject(xw, yw, fx, fy, cx, cy, k1, k2, k3, k4)
            # 针孔投影: u = fx * X/Z + cx, v = fy * Y/Z + cy
            Z_safe = np.where(np.abs(Z) > 1e-6, Z, 1e-6)
            xw = (fx * X / Z_safe + cx).astype(np.float32)
            yw = (fy * Y / Z_safe + cy).astype(np.float32)
            # 裁剪到图像范围内
            xw = np.clip(xw, 0, src_w - 1)
            yw = np.clip(yw, 0, src_h - 1)

        imu_mask = (self.imu_t >= t_prev) & (self.imu_t <= t_curr)
        imu_seg = self.imu_vals[imu_mask]
        if imu_seg.size == 0:
            imu_feat_t = torch.zeros((self.sequence_length, 6), dtype=torch.float32)
        else:
            imu_arr = torch.from_numpy(imu_seg.astype(np.float32))  # [N, 6]
            n = int(imu_arr.shape[0])
            if n == self.sequence_length:
                imu_feat_t = imu_arr
            elif n < self.sequence_length:
                pad = self.sequence_length - n
                imu_feat_t = F.pad(imu_arr, (0, 0, 0, pad))
            else:
                k = max(n // self.sequence_length, 1)
                pooled = F.avg_pool1d(imu_arr.transpose(0, 1).unsqueeze(0), kernel_size=k, stride=k, ceil_mode=True)
                imu_feat_t = pooled.squeeze(0).transpose(0, 1)
                if imu_feat_t.shape[0] > self.sequence_length:
                    imu_feat_t = imu_feat_t[:self.sequence_length, :]
                elif imu_feat_t.shape[0] < self.sequence_length:
                    pad = self.sequence_length - imu_feat_t.shape[0]
                    imu_feat_t = F.pad(imu_feat_t, (0, 0, 0, pad))

        if self.calib is not None:
            R_BI = getattr(self, "_R_BI", None)
            if R_BI is None:
                try:
                    if "R_BI" in self.calib:
                        R_BI_np = np.asarray(self.calib["R_BI"], dtype=np.float32)
                    elif "T_imu_marker" in self.calib:
                        T_im = self.calib["T_imu_marker"]
                        if isinstance(T_im, dict) and all(k in T_im for k in ("qx", "qy", "qz", "qw")):
                            q = np.array([T_im["qx"], T_im["qy"], T_im["qz"], T_im["qw"]], dtype=np.float64)
                            R_IM = QuaternionUtils.to_rotation_matrix(q).astype(np.float32)
                            R_BI_np = R_IM.T
                        else:
                            M = np.asarray(T_im, dtype=np.float32)
                            if M.shape == (4, 4):
                                R_BI_np = M[:3, :3].T
                            elif M.shape == (3, 3):
                                R_BI_np = M.T
                            else:
                                R_BI_np = np.eye(3, dtype=np.float32)
                    else:
                        R_BI_np = np.eye(3, dtype=np.float32)
                except Exception:
                    R_BI_np = np.eye(3, dtype=np.float32)
                R_BI = torch.from_numpy(R_BI_np)
                self._R_BI = R_BI
            R_BI_t = R_BI.to(dtype=imu_feat_t.dtype)
            imu_feat_t = imu_feat_t.clone()
            imu_feat_t[:, 0:3] = imu_feat_t[:, 0:3] @ R_BI_t.transpose(0, 1)
            imu_feat_t[:, 3:6] = imu_feat_t[:, 3:6] @ R_BI_t.transpose(0, 1)

        if self.derotate and self.calib is not None and imu_seg.size > 0:
            src_h, src_w = self.sensor_resolution or self.resolution
            K_src = self.calib["K"] if "K" in self.calib else self.calib.get("camera", {})
            K = {k: float(K_src.get(k, 1.0 if k in ("fx", "fy") else 0.0)) for k in ["fx", "fy", "cx", "cy"]}
            if "R_IC" in self.calib:
                R_CI = np.asarray(self.calib["R_IC"], dtype=np.float64)
            elif "T_imu_cam" in self.calib:
                T_ic = self.calib["T_imu_cam"]
                if isinstance(T_ic, dict) and all(k in T_ic for k in ("qx", "qy", "qz", "qw")):
                    q = np.array([T_ic["qx"], T_ic["qy"], T_ic["qz"], T_ic["qw"]], dtype=np.float64)
                    R_CI = QuaternionUtils.to_rotation_matrix(q)
                else:
                    M = np.asarray(T_ic, dtype=np.float64)
                    if M.shape == (4, 4):
                        R_CI = M[:3, :3]
                    elif M.shape == (3, 3):
                        R_CI = M
                    else:
                        R_CI = np.eye(3)
            elif "R_imu_cam" in self.calib:
                R_CI = np.asarray(self.calib["R_imu_cam"], dtype=np.float64)
            else:
                R_CI = np.eye(3)
            omega_cam = R_CI @ np.mean(imu_seg[:, 3:6], axis=0)
            dev = self.proc_device or torch.device('cpu')
            xw_t = torch.from_numpy(xw.astype(np.float32)).to(dev)
            yw_t = torch.from_numpy(yw.astype(np.float32)).to(dev)
            tw_t = torch.from_numpy(tw.astype(np.float32)).to(dev)
            omega_t = torch.from_numpy(np.asarray(omega_cam, dtype=np.float32)).to(dev)

            # 根据相机类型选择warp函数
            if self.camera_type == "kb4" and self.kb4_distortion is not None:
                # KB4 鱼眼相机使用球面旋转补偿
                xw_t, yw_t, valid_mask = warp_events_flow_torch_kb4(
                    xw_t, yw_t, tw_t, omega_t, K, self.kb4_distortion, (src_h, src_w), 0.0
                )
                # A1.5: 使用mask过滤出界事件，避免伪密度堆积
                if valid_mask.any():
                    xw_t = xw_t[valid_mask]
                    yw_t = yw_t[valid_mask]
                    tw_t = tw_t[valid_mask]
                    pw = pw[valid_mask.cpu().numpy()] if isinstance(pw, np.ndarray) else pw[valid_mask]
            else:
                # 针孔相机使用光流补偿
                xw_t, yw_t = warp_events_flow_torch(xw_t, yw_t, tw_t, omega_t, K, (src_h, src_w), 0.0)

            xw = xw_t
            yw = yw_t
            tw = tw_t

        dt_true = float(t_curr - t_prev)
        dt_win = max(dt_true, 1e-6)
        t_prev_pos, q_prev = self.interpolate_gt_data(float(t_prev))
        t_curr_pos, q_curr = self.interpolate_gt_data(float(t_curr))
        v_prev = self.interpolate_gt_velocity(float(t_prev), dt=dt_win * 0.5)
        v_curr = self.interpolate_gt_velocity(float(t_curr), dt=dt_win * 0.5)
        q_delta = QuaternionUtils.multiply(QuaternionUtils.inverse(q_prev), q_curr)
        t_delta = QuaternionUtils.to_rotation_matrix(q_prev).T @ (t_curr_pos - t_prev_pos)
        y = np.concatenate([t_delta.astype(np.float64), q_delta.astype(np.float64), 
                          q_prev.astype(np.float64), v_prev.astype(np.float64), v_curr.astype(np.float64)], axis=0)
        src_h, src_w = self.sensor_resolution or self.resolution
        vox = self.voxelizer.voxelize_events_adaptive(xw, yw, tw, pw, src_w, src_h, 0.0, dt_win)
        if self.augment:
            # Polarity Flip Augmentation
            if torch.rand(1).item() < 0.5:
                vox_clone = vox.clone()
                vox[1] = vox_clone[2]
                vox[2] = vox_clone[1]
                vox[3] = vox_clone[4]
                vox[4] = vox_clone[3]
                
            noise = torch.randn_like(vox) * self.event_noise_scale
            scale_mag = self.event_scale_jitter
            scale_counts = 1.0 + (torch.rand(3, 1, 1, device=vox.device) - 0.5) * scale_mag
            vox[0:3] = torch.clamp(vox[0:3] * scale_counts + noise[0:3], min=0.0)
            # For time channels, apply light noise and clamp to [0, 1]
            time_noise_scale = self.event_noise_scale * 0.25
            vox[3:5] = torch.clamp(vox[3:5] + noise[3:5] * time_noise_scale, min=0.0, max=1.0)
            imu_bias = torch.zeros_like(imu_feat_t)
            bscale = self.imu_bias_scale
            imu_bias[:, 0:3] = (torch.rand(1, 3, device=imu_feat_t.device) - 0.5) * bscale
            imu_bias[:, 3:6] = (torch.rand(1, 3, device=imu_feat_t.device) - 0.5) * bscale
            imu_feat_t = imu_feat_t + imu_bias
            mask_prob = self.imu_mask_prob
            if mask_prob > 0.0:
                keep = (torch.rand(imu_feat_t.size(0), 1, device=imu_feat_t.device) > mask_prob).to(imu_feat_t.dtype)
                imu_feat_t = imu_feat_t * keep
        intr = torch.tensor([float(getattr(self, "fx_scaled", 1.0)), float(getattr(self, "fy_scaled", 1.0))], dtype=torch.float32)
        return vox, imu_feat_t, torch.from_numpy(y.astype(np.float32)), torch.tensor([dt_win], dtype=torch.float32), intr
