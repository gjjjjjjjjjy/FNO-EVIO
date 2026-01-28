#!/usr/bin/env python3
"""
简化的 MVSEC 测试脚本
重用训练脚本中的组件，避免代码重复
"""

from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    import h5py  # type: ignore
except Exception:
    h5py = None

from fno_evio.legacy.train_fno_vio import (
    ModelConfig, HybridVIONet, QuaternionUtils,
    build_device, compute_adaptive_sequence_length, load_calibration,
    AdaptiveEventProcessor,
    SequenceDataset, CollateSequence
)

from fno_evio.legacy.utils import rescale_intrinsics_pinhole


def _rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.ndim == 2:
        R = R[None, ...]
    out = np.empty((R.shape[0], 4), dtype=np.float64)
    for i in range(R.shape[0]):
        m = R[i]
        t = float(m[0, 0] + m[1, 1] + m[2, 2])
        if t > 0.0:
            s = (t + 1.0) ** 0.5 * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = (1.0 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5 * 2.0
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = (1.0 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5 * 2.0
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = (1.0 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5 * 2.0
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        out[i] = np.array([x, y, z, w], dtype=np.float64)
    return out


class MVSECDataset(Dataset):
    """
    简化的 MVSEC 数据集，重用训练脚本中的组件
    """
    def __init__(
        self,
        root: str,
        dt: float,
        resolution: Tuple[int, int],
        sensor_resolution: Optional[Tuple[int, int]] = None,
        calib: Optional[dict] = None,
        sequence_length: int = 50,
        sequence: Optional[str] = None
    ):
        self.root = Path(root)
        self.dt = dt
        self.resolution = resolution
        self.sensor_resolution = sensor_resolution
        self.calib = calib
        self.sequence_length = sequence_length
        self.sequence = sequence

        self.adaptive_processor = AdaptiveEventProcessor(
            resolution=resolution,
            device=torch.device('cpu'),
            std_norm=False,
            log_norm=False
        )

        self._load_data()
        self._precompute_indices()

    def _normalize_event_times(self, t_raw: np.ndarray) -> np.ndarray:
        t = np.asarray(t_raw, dtype=np.float64) * float(getattr(self, "ev_time_scale", 1.0))
        off_ev_ns = float(self.calib.get("events_time_offset_ns", 0.0)) if isinstance(self.calib, dict) else 0.0
        if off_ev_ns != 0.0:
            t = t + off_ev_ns * 1e-9
        base_off = float(getattr(self, "ev_base_offset_sec", 0.0))
        if base_off != 0.0:
            t = t + base_off
        return t

    def _load_data(self):
        import h5py

        if not self.root.exists():
            raise FileNotFoundError(f"MVSEC dataset_root 不存在: {self.root}")

        def _find_files(patterns):
            files = []
            for pat in patterns:
                files.extend(self.root.glob(pat))
            if len(files) == 0:
                for pat in patterns:
                    files.extend(self.root.rglob(pat))
            uniq = sorted({str(p) for p in files})
            return [Path(p) for p in uniq]

        if self.sequence:
            seq = str(self.sequence)
            seq_data = _find_files([f"{seq}_data.hdf5", f"*{seq}*data*.hdf5", f"{seq}*data*.hdf5"])
            seq_gt = _find_files([f"{seq}_gt.hdf5", f"*{seq}*gt*.hdf5", f"*{seq}*groundtruth*.hdf5"])
            if len(seq_data) > 0 and len(seq_gt) > 0:
                self.data_path = str(seq_data[0])
                self.gt_path = str(seq_gt[0])
                print(f"[数据集] 选择序列: {seq}")
                print(f"[数据集] data: {self.data_path}")
                print(f"[数据集] gt:   {self.gt_path}")
            else:
                found = sorted({p.name for p in self.root.rglob("*.hdf5")})
                raise FileNotFoundError(
                    f"在 {self.root} 中未找到序列 '{seq}' 对应的 data/gt hdf5。"
                    f" 期望文件名: {seq}_data.hdf5 / {seq}_gt.hdf5 (或包含该前缀)。"
                    f" 已发现 .hdf5: {found[:20]}{' ...' if len(found) > 20 else ''}"
                )
        else:
            data_files = _find_files(["*_data.hdf5", "*data*.hdf5", "data.hdf5", "*events*.hdf5"])
            gt_files = _find_files(["*_gt.hdf5", "*gt*.hdf5", "gt.hdf5", "*groundtruth*.hdf5", "*truth*.hdf5"])

            if len(data_files) == 0 or len(gt_files) == 0:
                found = sorted({p.name for p in self.root.rglob("*.hdf5")})
                found_preview = found[:20]
                raise FileNotFoundError(
                    f"在 {self.root} 中未找到可用的 data/gt hdf5 文件。"
                    f" 期望 data 命名匹配: *_data.hdf5 / *data*.hdf5 / data.hdf5；"
                    f" gt 命名匹配: *_gt.hdf5 / *gt*.hdf5 / *groundtruth*.hdf5。"
                    f" 已发现 .hdf5: {found_preview}{' ...' if len(found) > 20 else ''}"
                )

            self.data_path = str(data_files[0])
            self.gt_path = str(gt_files[0])

        with h5py.File(self.data_path, "r") as f:
            left = f["davis"]["left"] if "davis" in f else f

            imu_obj = left["imu"]
            if isinstance(imu_obj, h5py.Group):
                if "data" in imu_obj:
                    imu_ds = imu_obj["data"]
                elif "values" in imu_obj:
                    imu_ds = imu_obj["values"]
                elif all(k in imu_obj for k in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]):
                    accel = np.stack([
                        imu_obj["accel_x"][:], imu_obj["accel_y"][:], imu_obj["accel_z"][:]
                    ], axis=1)
                    gyro = np.stack([
                        imu_obj["gyro_x"][:], imu_obj["gyro_y"][:], imu_obj["gyro_z"][:]
                    ], axis=1)
                    imu_data = np.concatenate([accel, gyro], axis=1)
                else:
                    raise KeyError("IMU group missing expected datasets")
                imu_ts = (imu_obj["ts"][:] if "ts" in imu_obj else imu_obj["t"][:]).astype(np.float64)
            else:
                imu_ds = imu_obj
                if getattr(imu_ds.dtype, "names", None):
                    arr = imu_ds[:]
                    names = list(imu_ds.dtype.names)
                    ts_field = "ts" if "ts" in names else ("t" if "t" in names else None)
                    imu_ts = (arr[ts_field].astype(np.float64) if ts_field is not None else arr[:, 0].astype(np.float64))
                    candidates = ["ax","ay","az","gx","gy","gz","accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]
                    cols = []
                    for key in ["ax","ay","az","gx","gy","gz"]:
                        if key in names:
                            cols.append(arr[key].astype(np.float32))
                    if len(cols) != 6:
                        cols = [arr[k].astype(np.float32) for k in candidates if k in names][:6]
                    if len(cols) == 6:
                        imu_data = np.stack(cols, axis=1)
                    else:
                        stacked = np.stack([arr[n].astype(np.float32) for n in names], axis=1)
                        imu_data = stacked[:, 1:7] if stacked.shape[1] >= 7 else stacked[:, :6]
                else:
                    data_arr = imu_ds[:].astype(np.float32)
                    if data_arr.ndim == 2 and data_arr.shape[1] >= 6:
                        imu_data = data_arr[:, :6]
                        if "imu_ts" in left:
                            print(f"[数据集] 找到单独的 IMU 时间戳: {left.name}/imu_ts")
                            imu_ts = left["imu_ts"][:].astype(np.float64)
                        elif "time" in left:
                            imu_ts = left["time"][:].astype(np.float64)
                        else:
                            print("[数据集] 警告: 未找到 IMU 时间戳，假设 200Hz 采样率")
                            imu_ts = np.linspace(0.0, float(len(imu_data)) * 0.005, num=len(imu_data))
                    else:
                        raise KeyError("不支持的 IMU 数据集形状")

        self.imu_time_scale = self._infer_time_scale(imu_ts)
        self.imu_t = imu_ts.astype(np.float64) * float(self.imu_time_scale)
        self.imu_vals = self._normalize_imu_data(imu_data.astype(np.float32))

        with h5py.File(self.gt_path, "r") as f:
            left = f["davis"]["left"] if "davis" in f else f

            if "pose" in left and "pose_ts" in left:
                pose_matrices = left["pose"][:]
                self.gt_t = left["pose_ts"][:].astype(np.float64)
                self.gt_pos = pose_matrices[:, :3, 3].astype(np.float32)

                rotations = pose_matrices[:, :3, :3]
                self.gt_quat = _rotmat_to_quat_xyzw(rotations).astype(np.float64)
            else:
                pose = None
                for k in left.keys():
                    obj = left[k]
                    if isinstance(obj, h5py.Dataset) and obj.shape[1] >= 7:
                        pose = obj[:]
                        break
                if pose is None:
                    raise KeyError("GT 姿态数据未找到")

                self.gt_t = pose[:, 0].astype(np.float64)
                self.gt_pos = pose[:, 1:4].astype(np.float32)
                q = pose[:, 4:8]
                self.gt_quat = q.astype(np.float64)

        self.gt_time_scale = self._infer_time_scale(self.gt_t)
        self.gt_t = self.gt_t.astype(np.float64) * float(self.gt_time_scale)

        self.gt_quat = self.gt_quat / (np.linalg.norm(self.gt_quat, axis=1, keepdims=True) + 1e-12)

        self.time_offset_sec = 0.0
        if len(self.gt_t) > 0 and len(self.imu_t) > 0:
            offset = float(self.imu_t[0]) - float(self.gt_t[0])
            if abs(offset) > 10.0:
                self.gt_t = self.gt_t + offset
                self.time_offset_sec = offset
                print(f"[数据集] 应用时间偏移到 GT: {offset:.4f}s")

        self.ev_base_offset_sec = 0.0

        with h5py.File(self.data_path, "r") as f:
            self._detect_event_structure(f)

    def _detect_event_structure(self, f):
        parent = None
        t_key = None

        def visit_fn(name, obj):
            nonlocal parent, t_key
            if isinstance(obj, h5py.Dataset):
                parent_abs = name.rsplit('/', 1)[0]
                if "events" in name or name.endswith("events"):
                    parent = parent_abs
                    if "ts" in obj.name or "t" in obj.name:
                        t_key = obj.name.split('/')[-1]

        f.visititems(visit_fn)

        if parent is None or t_key is None:
            if "davis" in f:
                davis = f["davis"]
                if "left" in davis:
                    left = davis["left"]
                    for key in left.keys():
                        if "ts" in key or "t" in key or "time" in key:
                            parent = "davis/left"
                            t_key = key
                            break

        if parent is None or t_key is None:
            raise KeyError("无法找到事件时间戳数据")

        if "davis" in f and "left" in f["davis"] and "events" in f["davis"]["left"]:
            self._ev_parent = "davis/left"
            self._ev_t_key = "events"
            self._ev_x_key = None
            self._ev_y_key = None
            self._ev_p_key = None
        else:
            self._ev_parent = parent
            self._ev_t_key = t_key
            self._ev_x_key = "x" if "x" in f[parent] else None
            self._ev_y_key = "y" if "y" in f[parent] else None
            self._ev_p_key = "p" if "p" in f[parent] else ("polarity" if "polarity" in f[parent] else None)

    def _normalize_imu_data(self, imu_vals: np.ndarray) -> np.ndarray:
        accel = imu_vals[:, 0:3] / 9.81
        gyro = imu_vals[:, 3:6] / np.pi

        normalized = np.concatenate([accel, gyro], axis=1)
        return np.clip(normalized, -10.0, 10.0).astype(np.float32)

    def _infer_time_scale(self, ts: np.ndarray) -> float:
        ts = np.asarray(ts, dtype=np.float64)
        if ts.size < 2:
            return 1.0
        d = np.diff(ts[:min(int(ts.size), 10000)])
        d = d[np.isfinite(d)]
        if d.size == 0:
            return 1.0
        dt = float(np.median(np.abs(d)))
        if 1e2 < dt < 1e5:
            return 1e-6
        if 1e5 < dt < 1e8:
            return 1e-9
        return 1.0

    def _precompute_indices(self):
        imu_t = self.imu_t
        dt_window = float(self.dt)

        t_start = float(imu_t[0])
        t_end = float(imu_t[-1])

        window_times_prev = []
        window_times_curr = []
        t = t_start
        while t + dt_window <= t_end:
            window_times_prev.append(t)
            window_times_curr.append(t + dt_window)
            t += dt_window

        if len(window_times_prev) == 0:
            print(f"[数据集] 警告: 未找到有效窗口。IMU 时长={t_end - t_start:.3f}s, dt={dt_window:.3f}s")
            self.sample_indices = np.array([], dtype=np.int64)
            self.prev_indices = np.array([], dtype=np.int64)
            self.curr_indices = np.array([], dtype=np.int64)
            self.window_t_prev = None
            self.window_t_curr = None
            self.h5_start_indices = None
            self.h5_end_indices = None
            return

        t_prev_all = np.array(window_times_prev, dtype=np.float64)
        t_curr_all = np.array(window_times_curr, dtype=np.float64)

        self.window_t_prev = t_prev_all
        self.window_t_curr = t_curr_all

        gt_t = self.gt_t.astype(np.float64)
        prev_gt_indices = np.searchsorted(gt_t, t_prev_all, side='left')
        curr_gt_indices = np.searchsorted(gt_t, t_curr_all, side='left')

        prev_gt_indices = np.clip(prev_gt_indices, 0, len(gt_t) - 1)
        curr_gt_indices = np.clip(curr_gt_indices, 0, len(gt_t) - 1)

        max_gt_offset = dt_window * 0.5
        prev_offset = np.abs(gt_t[prev_gt_indices] - t_prev_all)
        curr_offset = np.abs(gt_t[curr_gt_indices] - t_curr_all)
        valid = (prev_offset <= max_gt_offset) & (curr_offset <= max_gt_offset) & (curr_gt_indices > prev_gt_indices)

        if np.sum(valid) < len(valid):
            n_drop = int(np.sum(~valid))
            print(f"[数据集] 因 GT 对齐问题丢弃 {n_drop}/{len(valid)} 个窗口")

        t_prev_all = t_prev_all[valid]
        t_curr_all = t_curr_all[valid]
        prev_gt_indices = prev_gt_indices[valid]
        curr_gt_indices = curr_gt_indices[valid]

        self.sample_indices = curr_gt_indices
        self.prev_indices = prev_gt_indices
        self.curr_indices = curr_gt_indices
        self.window_t_prev = t_prev_all
        self.window_t_curr = t_curr_all

        self.sample_stride = 1

        print(f"[数据集] IMU 时间轴切窗: {len(self.sample_indices)} 个窗口, dt={dt_window:.4f}s, sample_stride=1")

        self.ev_time_scale = 1.0
        self.ev_base_offset_sec = 0.0
        try:
            import h5py
            with h5py.File(self.data_path, "r") as f:
                grp = f[self._ev_parent]
                ds = grp[self._ev_t_key]
                if self._ev_x_key is None and self._ev_y_key is None:
                    ev_t_raw = ds[:, 2].astype(np.float64)
                else:
                    ev_t_raw = ds[:].astype(np.float64)
                if ev_t_raw.size > 1000:
                    denom = (imu_t[-1] - imu_t[0]) + 1e-12
                    ratio = (ev_t_raw[-1] - ev_t_raw[0]) / denom
                    if 0.8e6 < ratio < 1.2e6:
                        self.ev_time_scale = 1e-6
                    elif 0.8e9 < ratio < 1.2e9:
                        self.ev_time_scale = 1e-9

                ev_t = self._normalize_event_times(ev_t_raw)
                time_offset = float(getattr(self, "time_offset_sec", 0.0))
                if time_offset != 0.0 and ev_t.size > 0 and imu_t.size > 0:
                    ev0 = float(ev_t[0])
                    imu0 = float(imu_t[0])
                    if abs(ev0 - imu0) > 10.0 and abs((ev0 + time_offset) - imu0) < 10.0:
                        self.ev_base_offset_sec = time_offset
                        ev_t = ev_t + time_offset

                self.h5_start_indices = np.searchsorted(ev_t, self.window_t_prev, side="left")
                self.h5_end_indices = np.searchsorted(ev_t, self.window_t_curr, side="right")
        except Exception:
            self.h5_start_indices = None
            self.h5_end_indices = None
            self.ev_time_scale = 1.0
            self.ev_base_offset_sec = 0.0

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        i_prev, i_curr = self.prev_indices[idx], self.curr_indices[idx]
        if hasattr(self, 'window_t_prev') and self.window_t_prev is not None:
            t_prev, t_curr = float(self.window_t_prev[idx]), float(self.window_t_curr[idx])
        else:
            t_prev, t_curr = float(self.gt_t[i_prev]), float(self.gt_t[i_curr])

        with h5py.File(self.data_path, "r") as f:
            grp = f[self._ev_parent]

            if hasattr(self, 'h5_start_indices') and self.h5_start_indices is not None:
                a, b = int(self.h5_start_indices[idx]), int(self.h5_end_indices[idx])
            else:
                events_data = grp[self._ev_t_key]
                if self._ev_x_key is None and self._ev_y_key is None:
                    ev_t_raw = events_data[:, 2].astype(np.float64)
                else:
                    ev_t_raw = grp[self._ev_t_key][:].astype(np.float64)

                ev_t = self._normalize_event_times(ev_t_raw)
                a = np.searchsorted(ev_t, t_prev, side="left")
                b = np.searchsorted(ev_t, t_curr, side="right")

            if self._ev_x_key is None and self._ev_y_key is None:
                events_data = grp[self._ev_t_key][a:b]
                xw = events_data[:, 0].astype(np.float32)
                yw = events_data[:, 1].astype(np.float32)
                tw = events_data[:, 2].astype(np.float64)
                pw = events_data[:, 3].astype(np.int64)
            else:
                xw = grp[self._ev_x_key][a:b].astype(np.float32)
                yw = grp[self._ev_y_key][a:b].astype(np.float32)
                tw = grp[self._ev_t_key][a:b].astype(np.float64)
                pw = grp[self._ev_p_key][a:b].astype(np.int64) if self._ev_p_key else None
            tw = self._normalize_event_times(tw)
            if pw is not None and pw.min() >= 0:
                pw = np.where(pw == 0, -1, 1).astype(np.int64)

        imu_mask = (self.imu_t >= t_prev) & (self.imu_t <= t_curr)
        imu_seg = self.imu_vals[imu_mask]

        if imu_seg.size == 0:
            imu_feat = torch.zeros((self.sequence_length, 6), dtype=torch.float32)
        else:
            imu_arr = torch.from_numpy(imu_seg)
            n = imu_arr.shape[0]
            if n == self.sequence_length:
                imu_feat = imu_arr
            elif n < self.sequence_length:
                pad = self.sequence_length - n
                imu_feat = F.pad(imu_arr, (0, 0, 0, pad))
            else:
                k = max(n // self.sequence_length, 1)
                pooled = F.avg_pool1d(imu_arr.transpose(0, 1).unsqueeze(0), kernel_size=k, stride=k, ceil_mode=True)
                imu_feat = pooled.squeeze(0).transpose(0, 1)
                if imu_feat.shape[0] > self.sequence_length:
                    imu_feat = imu_feat[:self.sequence_length, :]
                elif imu_feat.shape[0] < self.sequence_length:
                    pad = self.sequence_length - imu_feat.shape[0]
                    imu_feat = F.pad(imu_feat, (0, 0, 0, pad))

        t_prev_pos, t_curr_pos = self.gt_pos[i_prev], self.gt_pos[i_curr]
        q_prev, q_curr = self.gt_quat[i_prev], self.gt_quat[i_curr]

        q_delta = QuaternionUtils.multiply(QuaternionUtils.inverse(q_prev), q_curr)
        t_delta = QuaternionUtils.to_rotation_matrix(q_prev).T @ (t_curr_pos - t_prev_pos)

        v_prev = (t_curr_pos - t_prev_pos) / max(t_curr - t_prev, 1e-6)
        y = np.concatenate([t_delta, q_delta, q_prev, v_prev], axis=0)

        src_h, src_w = self.sensor_resolution or self.resolution
        dt_win = max(t_curr - t_prev, 1e-6)

        vox = self.adaptive_processor.voxelize_events_adaptive(
            xw, yw, tw, pw, src_w, src_h, t_prev, t_curr
        ).squeeze(0)

        return vox, imu_feat, torch.from_numpy(y.astype(np.float32)), torch.tensor([dt_win])


def main():
    parser = argparse.ArgumentParser(
        description="测试 FNO-FAST VIO 模型在 MVSEC 数据集上的性能",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset_root", type=str,
                       default="/Users/gjy/eventlearning/code/dataset/MVSEC",
                       help="MVSEC 数据集根目录路径")
    parser.add_argument("--sequence", type=str, default="",
                       help="序列名称前缀，用于选择 <sequence>_data.hdf5 / <sequence>_gt.hdf5")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型检查点文件路径")
    parser.add_argument("--resolution", type=int, nargs=2, default=[320, 320],
                       help="事件相机处理分辨率")
    parser.add_argument("--sensor_resolution", type=int, nargs=2, default=[260, 346],
                       help="原生传感器分辨率")
    parser.add_argument("--calib_yaml", type=str, default="",
                       help="相机内参标定 YAML 文件")
    parser.add_argument("--dt", type=float, default=0.2,
                       help="序列处理时间间隔")
    parser.add_argument("--rpe_dt", type=float, default=0.5,
                       help="相对位姿误差计算时间间隔")
    parser.add_argument("--no_align_scale", action="store_true",
                       help="在 ATE 计算中禁用尺度对齐 (s=1.0)")
    parser.add_argument("--sequence_len", type=int, default=50,
                       help="处理序列长度")
    parser.add_argument("--sequence_stride", type=int, default=50,
                       help="序列之间的步长")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="评估批次大小")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="数据加载器工作进程数")
    parser.add_argument("--pin_memory", action="store_true",
                       help="数据加载时使用 pin memory")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="运行设备")
    parser.add_argument("--visual_only_mode", action="store_true",
                       help="Visual-Only Baseline: 禁用 IMU 先验门控，仅使用视觉输出")
    parser.add_argument("--auto_scale_modes", action="store_true",
                       help="根据分辨率按比例自动调整FNO频域modes")
    parser.add_argument("--train_resolution", type=int, nargs=2, default=None,
                       help="训练时使用的网络分辨率，用于按比例调整modes")
    parser.add_argument("--output_dir", type=str, default="",
                       help="输出目录路径（可选）")

    args = parser.parse_args()
    if h5py is None:
        raise RuntimeError("Missing dependency: h5py (required for MVSEC evaluation)")

    if args.dt <= 0:
        raise ValueError("--dt 必须为正数")
    if args.rpe_dt <= 0:
        raise ValueError("--rpe_dt 必须为正数")

    device = build_device() if args.device == "auto" else torch.device(args.device)
    print(f"使用设备: {device}")

    sensor_res = tuple(args.sensor_resolution)

    calib = load_calibration(args.calib_yaml)
    if calib is not None and "K" in calib:
        fx_raw = float(calib["K"].get("fx", 1.0))
        fy_raw = float(calib["K"].get("fy", 1.0))
        net_h, net_w = tuple(args.resolution)
        if "resolution" in calib:
            sensor_w, sensor_h = calib["resolution"]
            print(f"[INFO] Using camera resolution from YAML: {sensor_w}x{sensor_h}")
        else:
            sensor_h, sensor_w = sensor_res
        sensor_res = (sensor_h, sensor_w)
        K_scaled, scales = rescale_intrinsics_pinhole(
            calib["K"],
            (sensor_h, sensor_w),
            (net_h, net_w)
        )
        scale_x, scale_y = scales
        fx_scaled = float(K_scaled.get("fx", fx_raw))
        fy_scaled = float(K_scaled.get("fy", fy_raw))
        calib["K_scaled"] = K_scaled
        if args.train_resolution:
            tr_h, tr_w = tuple(args.train_resolution)
            modes_scale = max(float(net_w) / float(tr_w), float(net_h) / float(tr_h))
        else:
            modes_scale = max(scale_x, scale_y)
        print(f"[信息] 内参缩放: fx={fx_scaled:.2f}, fy={fy_scaled:.2f} "
              f"(原始: {fx_raw:.2f}, {fy_raw:.2f}, 缩放: {scale_x:.2f}, {scale_y:.2f})")
    else:
        print("[警告] 未找到标定！物理损失将使用 fx=1.0 (无量纲)")

    imu_seq_len = compute_adaptive_sequence_length(args.dt)

    dataset = MVSECDataset(
        root=args.dataset_root,
        dt=args.dt,
        resolution=tuple(args.resolution),
        sensor_resolution=sensor_res,
        calib=calib,
        sequence_length=imu_seq_len,
        sequence=(args.sequence if args.sequence else None)
    )

    seq_dataset = SequenceDataset(dataset, sequence_len=imu_seq_len, stride=args.sequence_stride)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件未找到: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    if device.type != "cuda":
        new_sd = {}
        lstm_converted = False
        for k, v in sd.items():
            if "temporal_lstm" in k and ("weight_" in k or "bias_" in k) and ".cells." not in k:
                try:
                    suffix = k.split("_l")[-1]
                    layer_idx = int(suffix)
                    base_name = k.split("_l")[0].split(".")[-1]
                    prefix = k.rsplit(".", 1)[0]

                    new_key = f"{prefix}.cells.{layer_idx}.{base_name}"
                    new_sd[new_key] = v
                    lstm_converted = True
                except (IndexError, ValueError):
                    new_sd[k] = v
            else:
                new_sd[k] = v

        if lstm_converted:
            print("[INFO] Applied LSTM weight conversion (nn.LSTM -> NativeLSTM) for non-CUDA device")
            sd = new_sd

    k_channels = int(sd["stem.0.weight"].shape[1]) if "stem.0.weight" in sd else 5
    winK = max(k_channels // 5, 1)

    stem_channels = int(sd["stem.0.weight"].shape[0]) if "stem.0.weight" in sd else 64

    modes = 10
    if "fno_block.unit.spec1.weight" in sd:
        modes = int(sd["fno_block.unit.spec1.weight"].shape[2])
    elif "fno_block.unit1.weights1" in sd:
        modes = int(sd["fno_block.unit1.weights1"].shape[2])

    lstm_hidden = 128
    if "imu_encoder.lstm.weight_ih_l0" in sd:
        lstm_hidden = int(sd["imu_encoder.lstm.weight_ih_l0"].shape[0]) // 4

    lstm_layers = 1
    if "imu_encoder.lstm.weight_ih_l1" in sd:
        lstm_layers = 2

    use_cross = any((k.startswith("v_proj") or k.startswith("cross.")) for k in sd.keys())
    fusion_dim = int(sd["v_proj.weight"].shape[0]) if use_cross and "v_proj.weight" in sd else 128
    use_mr = any(k.startswith("fno_block.unit_low") for k in sd.keys())

    has_imu_gate = any("imu_gate" in k for k in sd.keys())
    imu_gate_soft = has_imu_gate

    mlow = 16
    mhigh = 32
    if use_mr:
        for k, v in sd.items():
            if "unit_low.weights1" in k:
                mlow = int(v.shape[2])
                break
        for k, v in sd.items():
            if "unit_high.weights1" in k:
                mhigh = int(v.shape[2])
                break
        mlow_t = max(4, int(round(mlow * (modes_scale if 'modes_scale' in locals() else 1.0))))
        mhigh_t = max(4, int(round(mhigh * (modes_scale if 'modes_scale' in locals() else 1.0))))
    else:
        modes_t = max(4, int(round(modes * (modes_scale if 'modes_scale' in locals() else 1.0))))
    config = ModelConfig(
        modes=(modes_t if not use_mr else modes),
        stem_channels=stem_channels,
        imu_embed_dim=64,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        imu_channels=6,
        sequence_length=imu_seq_len,
        fast_fft=False,
        state_aug=False,
        imu_gate_soft=imu_gate_soft,
        use_cudnn_lstm=(device.type == "cuda"),
        window_stack_K=winK,
        use_cross_attn=use_cross,
        fusion_dim=fusion_dim,
        use_mr_fno=use_mr,
        modes_low=(mlow_t if use_mr else 16),
        modes_high=(mhigh_t if use_mr else 32),
        use_dual_attention=True,
        attn_groups=8
    )

    model = HybridVIONet(config=config).to(device).to(memory_format=torch.channels_last)
    msd = model.state_dict()
    mapped = {}
    for k, v in sd.items():
        if k in msd:
            t = msd[k]
            if v.ndim == 4 and t.ndim == 4 and v.shape[2] == v.shape[3] and t.shape[2] == t.shape[3] and torch.is_complex(v):
                ti_in, ti_out, tm = t.shape[0], t.shape[1], t.shape[2]
                vi_in, vi_out, vm = v.shape[0], v.shape[1], v.shape[2]
                cin = min(ti_in, vi_in)
                cout = min(ti_out, vi_out)
                mm = min(tm, vm)
                w_new = torch.zeros(t.shape, dtype=t.dtype, device=t.device)
                w_new[:cin, :cout, :mm, :mm] = v[:cin, :cout, :mm, :mm]
                mapped[k] = w_new
            elif msd[k].shape == v.shape:
                mapped[k] = v
    model.load_state_dict(mapped, strict=False)

    if args.visual_only_mode:
        try:
            model.imu_encoder.set_dropout_p(1.0)
            print("[信息] Visual-Only Baseline: IMU 编码器 dropout 设为 1.0")
        except Exception:
            print("[警告] Visual-Only Baseline 设置失败")
        model.imu_gate_soft = False
        print("[信息] Visual-Only Baseline: 强制 imu_gate_soft=False (禁用 IMU 先验门控)")

    print(f"[信息] 配置: dt={args.dt}s, 分辨率={args.resolution}, 序列长度={imu_seq_len}")
    val_loader = DataLoader(
        seq_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=CollateSequence(window_stack_k=winK)
    )

    print("开始评估...")
    from fno_evio.legacy.train_fno_vio import evaluate

    ate, rpe_t, rpe_r = evaluate(model, val_loader, device, args.rpe_dt, args.dt)

    print(f"评估结果:")
    print(f"  ATE: {ate:.6f} m")
    print(f"  RPE_t: {rpe_t:.6f} m")
    print(f"  RPE_r: {rpe_r:.6f} deg")

    if getattr(args, "output_dir", ""):
        out_dir = Path(str(args.output_dir)).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "test_results.txt").write_text(
            "\n".join([f"ATE: {ate:.6f}", f"RPE_t: {rpe_t:.6f}", f"RPE_r: {rpe_r:.6f}"]) + "\n", encoding="utf-8"
        )
        result = {"ate": float(ate), "rpe_t": float(rpe_t), "rpe_r": float(rpe_r)}
        (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ate", "rpe_t", "rpe_r"])
            w.writeheader()
            w.writerow(result)


if __name__ == "__main__":
    main()
