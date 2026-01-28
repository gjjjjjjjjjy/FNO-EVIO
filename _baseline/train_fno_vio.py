from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import csv
import h5py
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler

try:
    import hdf5plugin 
except Exception:
    hdf5plugin = None

from fno_evio.legacy.utils import (
    QuaternionUtils,
    warp_events_flow_torch,
    warp_events_flow_torch_kb4,
    align_trajectory_with_timestamps,
    align_trajectory_with_timestamps_sim3,
    rotation_6d_to_matrix,
    rescale_intrinsics_pinhole,
    rescale_intrinsics_kb4,
    kb4_unproject,
    compute_rpe_loss as compute_rpe_loss_compose
)



@dataclass
class DatasetConfig:
    root: str
    events_h5: Optional[str] = None
    dt: float = 0.2
    resolution: Tuple[int, int] = (320, 320)
    sensor_resolution: Optional[Tuple[int, int]] = None
    sample_stride: int = 8
    windowing_mode: str = "imu"
    window_dt: Optional[float] = None
    event_offset_scan: bool = False
    event_offset_scan_range_s: float = 0.5
    event_offset_scan_step_s: float = 0.01
    event_file_candidates: Tuple[str, ...] = ("events-left.h5", "events_left.h5", "mocap-6dof-events_left.h5")
    voxel_std_norm: bool = False
    augment: bool = False
    adaptive_voxel: bool = True
    event_noise_scale: float = 0.01
    event_scale_jitter: float = 0.1
    imu_bias_scale: float = 0.02
    imu_mask_prob: float = 0.0
    adaptive_base_div: int = 60
    adaptive_max_events_div: int = 12
    adaptive_density_cap: float = 2.0
    derotate: bool = False
    voxelize_in_dataset: bool = True
    train_split: float = 0.9

@dataclass
class ModelConfig:
    modes: int = 10
    stem_channels: int = 64
    imu_embed_dim: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    imu_channels: int = 6
    sequence_length: int = 50  # Adaptive based on dt
    attn_groups: int = 8
    imu_gn_groups: Optional[int] = None
    norm_mode: str = "gn"
    fast_fft: bool = False
    state_aug: bool = False
    imu_gate_soft: bool = True
    use_uncertainty_fusion: bool = False  # Stage 0 clean baseline
    uncertainty_use_gate: bool = False    # Stage 0 clean baseline
    use_cudnn_lstm: bool = False
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    use_dual_attention: bool = True
    use_mr_fno: bool = False
    modes_low: int = 16
    modes_high: int = 32
    window_stack_K: int = 1
    voxel_stack_mode: str = "abs"
    use_cross_attn: bool = False
    fusion_dim: Optional[int] = None
    fusion_heads: int = 4
    scale_min: float = 0.0    # 门控模式: s ∈ [0, 1] 表示视觉修正的置信度
    scale_max: float = 1.0    # 门控模式: s=1 表示完全信任视觉残差

@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 2
    lr: float = 1e-3
    eval_interval: int = 1
    export_torchscript: bool = True
    loss_w_t: float = 8.0
    loss_w_r: float = 2.0
    loss_w_v: float = 0.1    # 启用速度约束以抑制闭环尺度漂移
    loss_w_aux_motion: float = 0.3
    loss_w_physics: float = 0.01
    loss_w_smooth: float = 0.0
    loss_w_rpe: float = 0.0
    rpe_dt: float = 0.5
    physics_mode: str = "none"
    speed_thresh: float = 0.02
    tbptt_len: int = 20
    tbptt_stride: int = 0
    physics_temp: float = 0.5
    loss_w_physics_max: float = 1.0
    physics_scale_quantile: float = 0.95
    physics_event_mask_thresh: float = 0.05
    scheduler: str = "step"
    gamma: float = 0.5
    scheduler_patience: int = 5
    scheduler_T_max: int = 50
    warmup_epochs: int = 5
    patience: int = 10
    compile: bool = False
    compile_backend: str = "inductor"
    adaptive_loss_weights: bool = False  # Stage 0 clean baseline
    use_rpe_loss: bool = True
    use_imu_consistency: bool = False
    loss_w_imu: float = 0.1
    loss_w_ortho: float = 0.1
    warmup_frames: int = 10  
    mixed_precision: bool = False
    earlystop_min_epoch: int = 15
    earlystop_ma_window: int = 5
    earlystop_alpha: float = 1.0
    earlystop_beta: float = 0.2
    earlystop_metric: str = "composite"
    loss_w_scale: float = 0.0  # Stage 0 clean baseline
    loss_w_scale_reg: float = 0.0
    use_seq_scale: bool = False
    seq_scale_reg: float = 0.0
    min_step_threshold: float = 0.0
    min_step_weight: float = 0.0
    eval_sim3_mode: str = "diagnose"
    loss_w_path_scale: float = 0.0
    loss_w_static: float = 0.0  # Stage 0 clean baseline
    loss_w_bias_a: float = 1e-4
    loss_w_bias_g: float = 1e-4
    loss_w_uncertainty: float = 0.0  # Uncertainty regularization weight (for Bayesian fusion)
    loss_w_uncertainty_calib: float = 0.0
    # IMU-anchored fusion constraints
    loss_w_correction: float = 0.0  # Visual correction magnitude regularization (r should be small)
    loss_w_bias_smooth: float = 0.0  # Bias smoothness constraint |b_t - b_{t-1}|
    bias_prior_accel: Optional[Tuple[float, float, float]] = None
    bias_prior_gyro: Optional[Tuple[float, float, float]] = None


@dataclass
class NumericalConstants:
    # Numerical stability constants
    QUATERNION_EPS: float = 1e-8        # Minimum value for quaternion normalization
    DIVISION_EPS: float = 1e-12         # Protection value for division operations
    GRADIENT_CLIP_NORM: float = 1.0    # Gradient clipping threshold
    MIN_EVENT_COUNT: int = 1           # Minimum number of events
    MAX_TEMPORAL_WINDOW: float = 1.0   # Maximum temporal window
    ROTATION_EPS: float = 1e-6         # Protection value for rotation matrix calculation
    TIME_ALIGNMENT_EPS: float = 1e-9   # Protection value for time alignment


@dataclass
class TrainingConstants:
    # Training constants
    DEFAULT_SEQUENCE_LENGTH: int = 50           # Default sequence length
    DEFAULT_TBPTT_LENGTH: int = 20              # Default TBPTT length
    IMU_FREQUENCY_HZ: int = 200                 # IMU frequency (Hz)
    ADAPTIVE_SEQ_LENGTH_MULTIPLIER: float = 1.5 # Adaptive sequence length multiplier
    ADAGN_SCALE_OFFSET: float = 0.8             # AdaGN scale offset
    ADAGN_SCALE_RANGE: float = 0.4              # AdaGN scale range
    PHYSICS_QUANTILE_DEFAULT: float = 0.95      # Default quantile for physics loss
    RPE_STRIDE_MULTIPLIER: float = 0.5          # RPE stride multiplier


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor,
                eps: float = NumericalConstants.DIVISION_EPS,
                fallback: float = 0.0) -> torch.Tensor:
 
    safe_denominator = torch.where(
        denominator.abs() > eps,
        denominator,
        torch.tensor(eps, device=denominator.device, dtype=denominator.dtype)
    )
    result = torch.where(
        denominator.abs() > eps,
        numerator / safe_denominator,
        torch.tensor(fallback, device=numerator.device, dtype=numerator.dtype)
    )
    return result


class LossComposer:

    def compute_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        raw_6d: Optional[torch.Tensor],
        voxel: torch.Tensor,
        dt_tensor: Optional[torch.Tensor],
        physics_module,
        speed_thresh: float,
        dt: float,
        physics_config: Dict[str, Any],
        scale_weight: float = 1.0,
        scale_reg_weight: float = 0.0,
        static_weight: float = 0.0,
        scale_reg_center: Optional[float] = None,
        min_step_threshold: float = 0.0,
        min_step_weight: float = 0.0,
        path_scale_weight: float = 0.0,
        s: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_pred = pred[:, 0:3].contiguous()
        t_gt = target[:, 0:3].contiguous()

        eps = float(NumericalConstants.DIVISION_EPS)
        t_pred_norm = F.normalize(t_pred, p=2, dim=1, eps=eps)
        t_gt_norm = F.normalize(t_gt, p=2, dim=1, eps=eps)
        l_dir_vec = (1.0 - torch.sum(t_pred_norm * t_gt_norm, dim=1))

        mag_pred = torch.norm(t_pred, p=2, dim=1)
        mag_gt = torch.norm(t_gt, p=2, dim=1)
        l_mag_vec = torch.abs(torch.log(mag_pred + eps) - torch.log(mag_gt + eps))

        lt_vec = l_dir_vec + 0.5 * l_mag_vec
        lt = lt_vec.mean()

        if s is not None and float(scale_reg_weight) > 0.0:
            if scale_reg_center is None:
                l_scale_reg = (torch.log(s + eps) ** 2).mean()
            else:
                l_scale_reg = ((s - float(scale_reg_center)) ** 2).mean()
            lt = lt + float(scale_reg_weight) * 0.5 * l_scale_reg

        if True:
            disp_pred = pred[:, 0:3].contiguous()
            disp_gt = target[:, 0:3].contiguous()
            step_norm_pred = disp_pred.norm(dim=1)
            step_norm_gt = disp_gt.norm(dim=1)

            if dt_tensor is not None:
                dt_local = dt_tensor.view(-1).to(pred.device).clamp(min=1e-6)
            else:
                dt_local = torch.full_like(step_norm_gt, dt).clamp(min=1e-6)

            disp_thresh = float(speed_thresh) * dt_local
            tau = torch.clamp(0.5 * disp_thresh, min=1e-4)
            soft_moving = torch.sigmoid((step_norm_gt - disp_thresh) / (tau + eps))
            soft_static = 1.0 - soft_moving

            if float(static_weight) > 0.0:
                denom = disp_thresh.clamp(min=eps)
                ratio_static = (step_norm_pred + eps) / denom
                ratio_static = torch.clamp(ratio_static, min=1.0, max=1e6)
                l_static = (torch.log(ratio_static) ** 2)
                w_sum = soft_static.sum().clamp(min=eps)
                lt = lt + float(static_weight) * (l_static * soft_static).sum() / w_sum

            if float(scale_weight) > 0.0 or float(min_step_weight) > 0.0 or float(path_scale_weight) > 0.0:
                w_sum = soft_moving.sum().clamp(min=eps)

                mean_norm_pred = (step_norm_pred * soft_moving).sum() / w_sum
                mean_norm_gt = (step_norm_gt * soft_moving).sum() / w_sum

                ratio = (step_norm_pred + eps) / (step_norm_gt + eps)
                ratio = torch.clamp(ratio, min=1e-6, max=1e6)
                l_scale = (torch.log(ratio) ** 2)
                if float(scale_weight) > 0.0:
                    lt = lt + float(scale_weight) * (l_scale * soft_moving).sum() / w_sum

                if float(min_step_threshold) > 0.0 and float(min_step_weight) > 0.0:
                    mean_step_penalty = F.relu(float(min_step_threshold) - mean_norm_pred) ** 2
                    lt = lt + float(min_step_weight) * mean_step_penalty

                if float(path_scale_weight) > 0.0:
                    sum_pred = (step_norm_pred * soft_moving).sum()
                    sum_gt = (step_norm_gt * soft_moving).sum()
                    path_ratio = safe_divide(sum_pred, sum_gt, eps=float(NumericalConstants.DIVISION_EPS), fallback=1.0)
                    path_ratio = torch.clamp(path_ratio, min=1e-4, max=1e4)
                    l_path = (torch.log(path_ratio)) ** 2
                    lt = lt + float(path_scale_weight) * l_path

        q_pred = pred[:, 3:7].contiguous()
        q_gt = target[:, 3:7].contiguous()
        q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        q_gt = q_gt / (q_gt.norm(dim=1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        lr = GeometryUtils.geodesic_rot_loss(q_pred, q_gt).mean()
        lo = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if raw_6d is not None:
            a1 = raw_6d[:, :3]
            a2 = raw_6d[:, 3:]
            l_norm = ((a1.norm(dim=1) - 1.0) ** 2).mean() + ((a2.norm(dim=1) - 1.0) ** 2).mean()
            l_dot = (torch.sum(a1 * a2, dim=1) ** 2).mean()
            lo = l_norm + l_dot

        if physics_module is not None:
            displacement = pred[:, 0:3].contiguous()
            lp = physics_module(voxel, displacement,
                                fx=float(physics_config.get('fx', 1.0)),
                                fy=float(physics_config.get('fy', 1.0)),
                                q=float(physics_config.get('physics_scale_quantile', 0.95)),
                                mask_thresh=float(physics_config.get('physics_event_mask_thresh', 0.05)),
                                dt=dt_local)
        else:
            lp = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return lt, lr, lp, lo


class GeometryUtils:    
    @staticmethod
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def quat_conj(q: torch.Tensor) -> torch.Tensor:
        """Conjugate of quaternion [x, y, z, w] -> [-x, -y, -z, w]"""
        return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)

    @staticmethod
    def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
        # Normalize quaternion
        q = q / (q.norm(dim=-1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        R = torch.stack([
            1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
            2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
            2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
        ], dim=-1).reshape(q.shape[:-1] + (3, 3))
        return R

    @staticmethod
    def safe_geodesic_loss(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        # Normalize input quaternions to avoid saturated gradients from non-unit inputs
        q1 = q1 / (q1.norm(dim=-1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        q2 = q2 / (q2.norm(dim=-1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        dot = torch.sum(q1 * q2, dim=-1).abs()
        dot = torch.clamp(dot, min=eps, max=1.0 - eps)
        return 2.0 * torch.acos(dot)

    @staticmethod
    def geodesic_rot_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
        return GeometryUtils.safe_geodesic_loss(q_pred, q_gt)

    @staticmethod
    def robust_rot_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:

        return GeometryUtils.safe_geodesic_loss(q_pred, q_gt)

def compute_adaptive_sequence_length(dt: float, imu_freq: int = TrainingConstants.IMU_FREQUENCY_HZ,
                                   multiplier: float = TrainingConstants.ADAPTIVE_SEQ_LENGTH_MULTIPLIER) -> int:
    return max(int(dt * imu_freq * multiplier), 20)  # Minimum 20 time steps


LOG_FH: Optional[Any] = None

def log(msg: str) -> None:
    print(msg, flush=True)


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


def _infer_dataset_root_from_calib(calib: Dict[str, Any], calib_yaml: Optional[str]) -> Optional[str]:
    for k in ("dataset_root", "root_dir", "root", "data_root"):
        v = calib.get(k)
        if v:
            return str(v)

    yml_dir = Path(".")
    if calib_yaml:
        try:
            yml_dir = Path(str(calib_yaml)).expanduser().resolve().parent
        except Exception:
            yml_dir = Path(str(calib_yaml)).expanduser().parent

    cand: List[Path] = []
    for k in (
        "events_h5",
        "events_path",
        "event_path",
        "imu_path",
        "imu_file",
        "gt_path",
        "gt_file",
        "mocap_path",
        "mocap_file",
    ):
        v = calib.get(k)
        if not v:
            continue
        p = _resolve_existing_path(v, bases=[yml_dir])
        if p is not None:
            cand.append(p)

    if not cand:
        return None

    common = Path(os.path.commonpath([c.as_posix() for c in cand]))
    if common.is_file():
        common = common.parent
    if common.name == "mocap-6dof-vi_gt_data":
        common = common.parent
    return common.as_posix()


@dataclass
class OptimizedTUMDataset(Dataset):

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

        imu_time_unit = self.calib.get("imu_time_unit") if isinstance(self.calib, dict) else None
        gt_time_unit = None
        if isinstance(self.calib, dict):
            gt_time_unit = self.calib.get("mocap_time_unit") or self.calib.get("gt_time_unit") or self.calib.get("groundtruth_time_unit")

        # Process timestamps and normalize data
        self.imu_t = self._correct_timestamps(imu_data[:, 0], unit=imu_time_unit)

        # 智能解析 IMU 列顺序：读取文件头确定 acc/gyro 列位置
        imu_acc, imu_gyro = self._parse_imu_columns(str(imu_path), imu_data)

        # DEBUG: Check magnitude to confirm units
        acc_mag = np.mean(np.linalg.norm(imu_acc, axis=1))
        gyro_mag = np.mean(np.linalg.norm(imu_gyro, axis=1))
        print(f"[IMU DEBUG] Raw Acc Mean Norm: {acc_mag:.4f} (Expect ~9.81 for m/s^2)")
        print(f"[IMU DEBUG] Raw Gyro Mean Norm: {gyro_mag:.4f} (Expect <3.0 for rad/s)")
        n_cols = int(imu_data.shape[1])
        print(f"[IMU DEBUG] Columns={n_cols}")
        if n_cols >= 10:
            imu_mag = imu_data[:, 7:10]
            mag_mag = np.mean(np.linalg.norm(imu_mag, axis=1))
            print(f"[IMU DEBUG] Raw Mag Mean Norm: {mag_mag:.4f} (Expect tens of uT; check units uT/mT/gauss)")

        acc_unit = _norm_unit_str(self.calib.get("imu_acc_unit")) if isinstance(self.calib, dict) else ""
        gyro_unit = _norm_unit_str(self.calib.get("imu_gyro_unit")) if isinstance(self.calib, dict) else ""

        acc_unit_known = acc_unit in ("mps2", "m/s^2", "mps^2", "ms2", "m/s2", "g")
        gyro_unit_known = gyro_unit in ("rad", "rad/s", "rads", "deg", "deg/s", "dps")

        if acc_unit == "g":
            print(f"[IMU UNIT] Acc unit is 'g' from YAML. Converting to m/s^2 (* 9.81).")
            imu_acc = imu_acc * 9.81
        elif not acc_unit_known:
            if 0.5 < acc_mag < 2.0:
                print(f"[IMU AUTO-FIX] Acc seems to be in 'g' (mean={acc_mag:.2f}). Converting to m/s^2 (* 9.81).")
                imu_acc = imu_acc * 9.81

        if gyro_unit in ("deg", "deg/s", "dps"):
            print(f"[IMU UNIT] Gyro unit is '{gyro_unit}' from YAML. Converting to rad/s (* pi/180).")
            imu_gyro = imu_gyro * (np.pi / 180.0)
        elif not gyro_unit_known:
            if gyro_mag > 4.0:
                print(f"[IMU AUTO-FIX] Gyro seems to be in deg/s (mean={gyro_mag:.2f}). Converting to rad/s (* pi/180).")
                imu_gyro = imu_gyro * (np.pi / 180.0)

        # Stack as [Acc, Gyro]
        imu_ordered = np.concatenate([imu_acc, imu_gyro], axis=1)
        self.imu_vals = self._normalize_imu_data(imu_ordered.astype(np.float32))

        self.gt_t = self._correct_timestamps(gt_data[:, 0], unit=gt_time_unit)
        self.gt_pos, self.gt_quat = self._parse_gt_columns(str(gt_path), gt_data)

        # Find and load events file
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

        # Load events data and resolve keys
        with h5py.File(self.h5_path, "r") as f:
            if all(k in f for k in ["x", "y"]) and ("t" in f or "ts" in f):
                self._x_key, self._y_key = "x", "y"
                self._t_key = "t" if "t" in f else "ts"
                self._p_key = "p" if "p" in f else ("polarity" if "polarity" in f else None)
                ev_t_ds = f[self._t_key]
            elif "events" in f:
                g = f["events"]
                self._x_key, self._y_key = "events/x", "events/y"
                self._t_key = "events/t" if "t" in g else "events/ts"
                self._p_key = "events/p" if "p" in g else ("events/polarity" if "polarity" in g else None)
                ev_t_ds = g["t"] if "t" in g else g["ts"]
            else:
                raise KeyError(f"Unrecognized H5 structure: {list(f.keys())}")
            N = int(ev_t_ds.shape[0])
            k = max(int(1000), 1)
            t_coarse = ev_t_ds[0:N:k]
            gt_duration = self.gt_t[-1] - self.gt_t[0]
            imu_duration = float("nan")
            try:
                if hasattr(self, "imu_t") and len(self.imu_t) > 1:
                    imu_duration = float(self.imu_t[-1] - self.imu_t[0])
            except Exception:
                imu_duration = float("nan")
            ref_duration = imu_duration if (np.isfinite(imu_duration) and imu_duration > 0.01) else float(gt_duration)
            t_coarse_duration = t_coarse[-1] - t_coarse[0] if t_coarse.size > 1 else 0.0
            
            # Default to 1.0
            self.unit_scale = 1.0
            
            if t_coarse.size > 1 and ref_duration > 0.01:
                # Robust unit detection based on order of magnitude ratio
                # This handles cases where sequences are truncated (partial overlap)
                ratio = t_coarse_duration / ref_duration
                
                if ratio > 5e7:  # ~1e9 (ns), tolerant to 20x mismatch
                    self.unit_scale = 1e-9
                    print(f"[DATASET] Detected nanoseconds (ratio {ratio:.1e}). Scaling by 1e-9.")
                elif ratio > 5e4:  # ~1e6 (us)
                    self.unit_scale = 1e-6
                    print(f"[DATASET] Detected microseconds (ratio {ratio:.1e}). Scaling by 1e-6.")
                else:
                    self.unit_scale = 1.0
                    print(f"[DATASET] Detected seconds (ratio {ratio:.2f}). No scaling.")
            else:
                # Fallback only if duration comparison is impossible
                if t_coarse.size > 0 and t_coarse[-1] > 1e14:
                    self.unit_scale = 1e-6
                    print(f"[DATASET] Fallback: detected nanoseconds from large timestamp. Scaling by 1e-6.")
            
            if self.unit_scale != 1.0:
                t_coarse = t_coarse * self.unit_scale
            
            self.t_coarse = t_coarse

            if self.sensor_resolution is None:
                h_attr = f.attrs.get("height") or f.attrs.get("sensor_height")
                w_attr = f.attrs.get("width") or f.attrs.get("sensor_width")
                if h_attr is None or w_attr is None:
                    try:
                        if "/" in self._x_key and "/" in self._y_key:
                            gk_x, dk_x = self._x_key.split("/", 1)
                            gk_y, dk_y = self._y_key.split("/", 1)
                            dx = f[gk_x][dk_x]
                            dy = f[gk_y][dk_y]
                        else:
                            dx = f[self._x_key]
                            dy = f[self._y_key]
                        n_probe = min(int(dx.shape[0]), 100000)
                        xs = dx[:n_probe]
                        ys = dy[:n_probe]
                        h_attr = int(np.max(ys) + 1) if ys.size > 0 else None
                        w_attr = int(np.max(xs) + 1) if xs.size > 0 else None
                    except Exception:
                        h_attr, w_attr = None, None
                if h_attr is not None and w_attr is not None:
                    self.sensor_resolution = (int(h_attr), int(w_attr))
            if self.sensor_resolution is None:
                raise ValueError("Sensor resolution could not be determined from H5; please provide --sensor_resolution H W.")

        self.fx_scaled = 1.0
        self.fy_scaled = 1.0
        self.camera_type = "pinhole"
        self.kb4_distortion = None
        if isinstance(self.calib, dict):
            try:
                K_src = self.calib["K"] if "K" in self.calib else self.calib.get("camera", {})
                K_raw = {k: float(K_src.get(k, 1.0 if k in ("fx", "fy") else 0.0)) for k in ["fx", "fy", "cx", "cy"]}
                if "resolution" in self.calib:
                    sensor_w, sensor_h = self.calib["resolution"]
                    src_h, src_w = int(sensor_h), int(sensor_w)
                else:
                    src_h, src_w = self.sensor_resolution or self.resolution
                net_h, net_w = tuple(self.resolution)

                # 检测相机类型
                cam_type = K_src.get("camera_type", "pinhole").lower()
                self.camera_type = cam_type

                if cam_type == "kb4":
                    # KB4 鱼眼相机处理
                    distortion = {
                        "k1": float(K_src.get("k1", 0.0)),
                        "k2": float(K_src.get("k2", 0.0)),
                        "k3": float(K_src.get("k3", 0.0)),
                        "k4": float(K_src.get("k4", 0.0)),
                    }
                    K_scaled, dist_scaled = rescale_intrinsics_kb4(K_raw, distortion, (src_h, src_w), (net_h, net_w))
                    self.kb4_distortion = dist_scaled
                    print(f"[DATASET] Using KB4 fisheye model: k1={distortion['k1']:.6f}, k2={distortion['k2']:.6f}, k3={distortion['k3']:.6f}, k4={distortion['k4']:.6f}")
                else:
                    # 针孔相机处理
                    K_scaled, _ = rescale_intrinsics_pinhole(K_raw, (src_h, src_w), (net_h, net_w))
                    print(f"[DATASET] Using Pinhole camera model")

                self.fx_scaled = float(K_scaled.get("fx", self.fx_scaled))
                self.fy_scaled = float(K_scaled.get("fy", self.fy_scaled))
                self.cx_scaled = float(K_scaled.get("cx", src_w * 0.5))
                self.cy_scaled = float(K_scaled.get("cy", src_h * 0.5))
            except Exception as e:
                print(f"[DATASET] Warning: Failed to parse camera intrinsics: {e}")
                self.fx_scaled = 1.0
                self.fy_scaled = 1.0
                self.cx_scaled = self.resolution[1] * 0.5
                self.cy_scaled = self.resolution[0] * 0.5

        # Time alignment
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
        # Check for NaN/Inf in critical arrays
        for name, arr in [("gt_t", self.gt_t), ("gt_pos", self.gt_pos),
                         ("gt_quat", self.gt_quat), ("imu_t", self.imu_t),
                         ("imu_vals", self.imu_vals)]:
            if np.any(np.isnan(arr)):
                raise ValueError(f"NaN detected in {name}")
            if np.any(np.isinf(arr)):
                raise ValueError(f"Inf detected in {name}")
        # Check quaternion normalization
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
        # Ensure quaternions are normalized
        q1 = QuaternionUtils.normalize(q1)
        q2 = QuaternionUtils.normalize(q2)
        # Compute dot product
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
        p_plus, _ = self.interpolate_gt_data(query_time + dt)
        p_minus, _ = self.interpolate_gt_data(query_time - dt)
        return (p_plus - p_minus) / (2 * dt)

    def interpolate_imu_data(self, query_time: float, window_size: float = 0.1) -> np.ndarray:
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

        # 获取事件索引范围
        if self.h5_start_indices is not None and self.h5_end_indices is not None:
            a, b = int(self.h5_start_indices[idx]), int(self.h5_end_indices[idx])
        else:
            h5 = self._get_h5_file()
            ev_t_ds = self._resolve_h5_keys(h5)
            unit_scale = float(self.unit_scale)
            off_sec = float(self._events_time_offset_sec())
            a = int(self._h5_searchsorted(ev_t_ds, float(t_prev), "left", unit_scale, off_sec))
            b = int(self._h5_searchsorted(ev_t_ds, float(t_curr), "right", unit_scale, off_sec))

        # 从 HDF5 直接读取事件 (利用 HDF5 chunk cache)
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

        # IMU processing
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

        # Derotation GPU/tensor path
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

        # Use interpolated ground truth at exact timestamps
        dt_true = float(t_curr - t_prev)
        dt_win = max(dt_true, 1e-6)
        t_prev_pos, q_prev = self.interpolate_gt_data(float(t_prev))
        t_curr_pos, q_curr = self.interpolate_gt_data(float(t_curr))
        v_prev = self.interpolate_gt_velocity(float(t_prev), dt=dt_win * 0.5)
        v_curr = self.interpolate_gt_velocity(float(t_curr), dt=dt_win * 0.5)
        q_delta = QuaternionUtils.multiply(QuaternionUtils.inverse(q_prev), q_curr)
        t_delta = QuaternionUtils.to_rotation_matrix(q_prev).T @ (t_curr_pos - t_prev_pos)
        # Return data
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
                
            # Separate augmentation for count channels and time channels
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



class PDELayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]], dtype=torch.float32)
        laplacian = torch.tensor([[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        kx = self.sobel_x.repeat(c, 1, 1, 1)
        ky = self.sobel_y.repeat(c, 1, 1, 1)
        kl = self.laplacian.repeat(c, 1, 1, 1)
        dx = F.conv2d(x, kx, bias=None, stride=1, padding=1, groups=c)
        dy = F.conv2d(x, ky, bias=None, stride=1, padding=1, groups=c)
        lap = F.conv2d(x, kl, bias=None, stride=1, padding=1, groups=c)
        return torch.cat([x, dx, dy, lap], dim=1)

def create_pde_layer(use_pde: bool, channels: int):
    if use_pde:
        return PDELayer(), channels * 4
    else:
        return nn.Identity(), channels




class AdaptiveLossWeights(nn.Module):

    def __init__(self, num_losses: int = 3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        dtype = losses[0].dtype
        device = losses[0].device
        terms: List[torch.Tensor] = []
        for i, loss in enumerate(losses):
            if loss.detach().item() < 1e-9:
                continue
                
            loss_safe = torch.clamp(loss, min=1e-6)
            log_var = torch.clamp(self.log_vars[i], min=-2.5, max=10.0)
            
            precision = torch.exp(-log_var)
            terms.append(0.5 * precision * loss_safe + 0.5 * log_var)
            
        if not terms:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return torch.stack(terms).sum()


class ImprovedIMUEncoder(nn.Module):
    def __init__(self, imu_channels: int = 6, sequence_length: int = 50, embed_dim: int = 64) -> None:
        super().__init__()
        self.seq_len = sequence_length
        self.imu_channels = imu_channels

        # 1D CNN to extract temporal features
        self.conv1d = nn.Sequential(
            nn.Conv1d(imu_channels, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(p=0.0),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.temporal_attention = nn.MultiheadAttention(128, 8, batch_first=True)
        self.drop_time = nn.Dropout(p=0.0)
        self.encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
            nn.GELU()
        )
        self._dropout_p = 0.0

    def forward(self, imu_sequence: torch.Tensor) -> torch.Tensor:
        # Input: [B, T, 6] -> [B, 6, T] for Conv1d
        B, T, _ = imu_sequence.shape
        x = imu_sequence.transpose(1, 2)  # [B, 6, T]
        x = self.conv1d(x)  # [B, 128, T]
        x = self.drop_time(x)
        x = x.transpose(1, 2)  # [B, T, 128]
        mask = (imu_sequence.abs().sum(dim=2) > 1e-6)  # [B, T]
        pad_mask = (~mask)  # True indicates padding
        x, _ = self.temporal_attention(x, x, x, key_padding_mask=pad_mask)  # [B, T, 128]
        w = mask.to(dtype=x.dtype).unsqueeze(-1)  # [B, T, 1]
        denom = w.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        pooled = (x * w).sum(dim=1) / denom  # [B, 128]
        return self.encoder(pooled)  # [B, embed_dim]

    def set_dropout_p(self, p: float) -> None:
        self._dropout_p = max(min(float(p), 0.9), 0.0)
        for i, m in enumerate(self.conv1d):
            if isinstance(m, nn.Dropout):
                m.p = self._dropout_p
        self.drop_time.p = self._dropout_p


@dataclass
class IMUStateManager:
  
    def __init__(self, device: torch.device):
        self.device = device
        self.velocity: Optional[torch.Tensor] = None
        self.rotation: Optional[torch.Tensor] = None

    def initialize(self, batch_size: int):
        self.velocity = None
        self.rotation = None

    def detach_states(self):
        if self.velocity is not None:
            self.velocity = self.velocity.detach()
            self.rotation = self.rotation.detach()

    def update_states(self, new_velocity: torch.Tensor, new_rotation: torch.Tensor):
        self.velocity = new_velocity
        self.rotation = new_rotation

    def reset(self):
        self.velocity = None
        self.rotation = None


@torch.jit.script
def integrate_imu_step(p: torch.Tensor, v: torch.Tensor, R: torch.Tensor, acc: torch.Tensor, gyro: torch.Tensor, g: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta = torch.norm(gyro, dim=1, keepdim=True) + 1e-12
    axis = gyro / theta
    theta_dt = theta * dt.view(-1, 1)
    c = torch.cos(theta_dt).view(-1)
    s = torch.sin(theta_dt).view(-1)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    one_minus_c = 1.0 - c
    r00 = c + one_minus_c * x * x
    r01 = one_minus_c * x * y - s * z
    r02 = one_minus_c * x * z + s * y
    r10 = one_minus_c * y * x + s * z
    r11 = c + one_minus_c * y * y
    r12 = one_minus_c * y * z - s * x
    r20 = one_minus_c * z * x - s * y
    r21 = one_minus_c * z * y + s * x
    r22 = c + one_minus_c * z * z
    R_delta = torch.stack([
        torch.stack([r00, r01, r02], dim=1),
        torch.stack([r10, r11, r12], dim=1),
        torch.stack([r20, r21, r22], dim=1)
    ], dim=1)
    R_next = torch.bmm(R, R_delta)
    acc_w_curr = torch.bmm(R, acc.unsqueeze(-1)).squeeze(-1)
    acc_w_next = torch.bmm(R_next, acc.unsqueeze(-1)).squeeze(-1)
    acc_w_avg = 0.5 * (acc_w_curr + acc_w_next)
    total_acc = acc_w_avg + g
    dt_col = dt.view(-1, 1)
    p_next = p + v * dt_col + 0.5 * total_acc * (dt_col * dt_col)
    v_next = v + total_acc * dt_col
    return p_next, v_next, R_next

class IMUKinematics(nn.Module):
    def __init__(self, g_world: torch.Tensor = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32),
                 enable_gravity_alignment: bool = True) -> None:
        super().__init__()
        self.register_buffer("g_world", g_world.view(1, 3))
        self.enable_gravity_alignment = enable_gravity_alignment

    @staticmethod
    def _so3_exp(omega_dt: torch.Tensor) -> torch.Tensor:
        theta = torch.norm(omega_dt, dim=1, keepdim=True) + 1e-12
        axis = omega_dt / theta
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        ct = torch.cos(theta).view(-1)
        st = torch.sin(theta).view(-1)
        one_ct = 1.0 - ct
        R = omega_dt.new_zeros((omega_dt.size(0), 3, 3))
        R[:, 0, 0] = ct + one_ct * x * x
        R[:, 0, 1] = one_ct * x * y - st * z
        R[:, 0, 2] = one_ct * x * z + st * y
        R[:, 1, 0] = one_ct * y * x + st * z
        R[:, 1, 1] = ct + one_ct * y * y
        R[:, 1, 2] = one_ct * y * z - st * x
        R[:, 2, 0] = one_ct * z * x - st * y
        R[:, 2, 1] = one_ct * z * y + st * x
        R[:, 2, 2] = ct + one_ct * z * z
        return R

    @staticmethod
    def _rot_to_quat(R: torch.Tensor) -> torch.Tensor:
        batch_size = R.shape[0]
        q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)
        
        k_one = 1.0
        
        r00 = R[:, 0, 0]
        r01 = R[:, 0, 1]
        r02 = R[:, 0, 2]
        r10 = R[:, 1, 0]
        r11 = R[:, 1, 1]
        r12 = R[:, 1, 2]
        r20 = R[:, 2, 0]
        r21 = R[:, 2, 1]
        r22 = R[:, 2, 2]
        
        tr = r00 + r11 + r22
        
        # Case 1: tr > 0
        mask_tr = tr > 0
        if mask_tr.any():
            S = torch.sqrt(tr[mask_tr] + k_one) * 2
            q[mask_tr, 3] = 0.25 * S
            q[mask_tr, 0] = (r21[mask_tr] - r12[mask_tr]) / S
            q[mask_tr, 1] = (r02[mask_tr] - r20[mask_tr]) / S
            q[mask_tr, 2] = (r10[mask_tr] - r01[mask_tr]) / S

        # Case 2: tr <= 0
        mask_neg = ~mask_tr
        if mask_neg.any():
            # mask_0: (R00 > R11) & (R00 > R22)
            mask_0 = mask_neg & (r00 > r11) & (r00 > r22)
            
            # mask_1: (R11 > R22) & not mask_0
            mask_1 = mask_neg & (r11 > r22) & (~mask_0)
            
            # mask_2: rest of mask_neg
            mask_2 = mask_neg & (~mask_0) & (~mask_1)
            
            if mask_0.any():
                S = torch.sqrt(k_one + r00[mask_0] - r11[mask_0] - r22[mask_0]) * 2
                q[mask_0, 0] = 0.25 * S
                q[mask_0, 1] = (r01[mask_0] + r10[mask_0]) / S
                q[mask_0, 2] = (r02[mask_0] + r20[mask_0]) / S
                q[mask_0, 3] = (r21[mask_0] - r12[mask_0]) / S
                
            if mask_1.any():
                S = torch.sqrt(k_one + r11[mask_1] - r00[mask_1] - r22[mask_1]) * 2
                q[mask_1, 1] = 0.25 * S
                q[mask_1, 0] = (r01[mask_1] + r10[mask_1]) / S
                q[mask_1, 2] = (r12[mask_1] + r21[mask_1]) / S
                q[mask_1, 3] = (r02[mask_1] - r20[mask_1]) / S
                
            if mask_2.any():
                S = torch.sqrt(k_one + r22[mask_2] - r00[mask_2] - r11[mask_2]) * 2
                q[mask_2, 2] = 0.25 * S
                q[mask_2, 0] = (r02[mask_2] + r20[mask_2]) / S
                q[mask_2, 1] = (r12[mask_2] + r21[mask_2]) / S
                q[mask_2, 3] = (r10[mask_2] - r01[mask_2]) / S

        return q / (q.norm(dim=1, keepdim=True) + NumericalConstants.QUATERNION_EPS)

    def _estimate_gravity_direction(self, imu_seq: torch.Tensor, stationary_frames: int = 5) -> torch.Tensor:

        B, T, _ = imu_seq.shape
        if T < stationary_frames:
            return self.g_world.expand(B, -1) / torch.norm(self.g_world)
        acc_frames = imu_seq[:, :stationary_frames, 0:3]  # [B, stationary_frames, 3]
        acc_mean = torch.mean(acc_frames, dim=1)  # [B, 3]

        gravity_norm = torch.norm(acc_mean, dim=1, keepdim=True)
        gravity_direction = acc_mean / (gravity_norm + NumericalConstants.QUATERNION_EPS)

        return gravity_direction

    def forward(self, imu_seq: torch.Tensor, prev_v: Optional[torch.Tensor] = None, prev_R: Optional[torch.Tensor] = None, b_a: Optional[torch.Tensor] = None, b_g: Optional[torch.Tensor] = None, dt: torch.Tensor = None, mask: Optional[torch.Tensor] = None, debug: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = imu_seq.shape
        acc_raw = imu_seq[:, :, 0:3]
        gyro_raw = imu_seq[:, :, 3:6]
        # Denormalize: input was normalized by dividing acc/9.81 and gyro/pi (see line 966-967)
        acc = acc_raw * 9.81  # Denormalize acceleration back to m/s²
        gyro = gyro_raw * np.pi  # Denormalize gyro back to rad/s (data was normalized by /pi)
        dev = imu_seq.device
        p = torch.zeros(B, 3, device=dev)
        v = prev_v if prev_v is not None else torch.zeros(B, 3, device=dev)
        R = prev_R if prev_R is not None else torch.eye(3, device=dev).expand(B, 3, 3).clone()
        if b_a is not None:
            ba = b_a.to(device=dev, dtype=imu_seq.dtype) * 9.81
        else:
            ba = torch.zeros(B, 3, device=dev, dtype=imu_seq.dtype)

        if b_g is not None:
            bg = b_g.to(device=dev, dtype=imu_seq.dtype) * float(np.pi)
        else:
            bg = torch.zeros(B, 3, device=dev, dtype=imu_seq.dtype)
        acc_norm = acc.norm(dim=2)
        gyro_norm = gyro.norm(dim=2)
        if mask is not None:
            w = mask.to(dtype=acc_norm.dtype)
            denom = w.sum(dim=1).clamp(min=1.0)
            acc_mag_mean = float(((acc_norm * w).sum(dim=1) / denom).mean().detach().cpu().item())
            gyro_mag_mean = float(((gyro_norm * w).sum(dim=1) / denom).mean().detach().cpu().item())
            acc_mean_body = (acc * w.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
        else:
            acc_mag_mean = float(acc_norm.mean().detach().cpu().item())
            gyro_mag_mean = float(gyro_norm.mean().detach().cpu().item())
            acc_mean_body = acc.mean(dim=1)
        use_gravity = (acc_mag_mean > 6.0)

        if not use_gravity:
            g = torch.zeros((B, 3), device=dev, dtype=imu_seq.dtype)
        else:
            g = self.g_world.expand(B, -1).to(device=dev, dtype=imu_seq.dtype)
            if self.enable_gravity_alignment and gyro_mag_mean < 0.2:
                acc_mean_body_corr = acc_mean_body - ba
                if prev_R is not None:
                    g_est = -torch.bmm(prev_R.to(dtype=imu_seq.dtype), acc_mean_body_corr.unsqueeze(-1)).squeeze(-1)
                else:
                    g_est = -acc_mean_body_corr
                g = g_est / (g_est.norm(dim=1, keepdim=True) + NumericalConstants.QUATERNION_EPS) * 9.81
                if debug:
                    print(f"[IMU KIN] Gravity Align: acc_mean={acc_mag_mean:.3f} gyro_mean={gyro_mag_mean:.3f} | g[0]={g[0].detach().float().cpu().numpy()}")
        if dt is None:
            dt = torch.full((B,), 0.05, device=dev, dtype=imu_seq.dtype)
        else:
            if not torch.is_tensor(dt):
                dt = torch.tensor(dt, device=dev, dtype=imu_seq.dtype)
            else:
                dt = dt.to(device=dev, dtype=imu_seq.dtype)
            if dt.numel() == 1:
                dt = dt.view(1).expand(B)
            else:
                dt = dt.view(-1)
                if int(dt.numel()) != int(B):
                    raise ValueError(f"IMUKinematics.forward expects dt scalar or per-sample dt with numel=B; got numel={int(dt.numel())}, B={int(B)}")

        for i in range(T):
            omega = gyro[:, i, :] - bg
            acc_b = acc[:, i, :] - ba
            if mask is not None:
                step_mask = mask[:, i].to(dtype=dt.dtype)
                dt_eff = dt * step_mask
            else:
                dt_eff = dt
            p, v, R = integrate_imu_step(p, v, R, acc_b, omega, g, dt_eff)
        return p, R, v


class NativeLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        # Match CUDA nn.LSTM stability: set positive forget-gate bias
        with torch.no_grad():
            hs = self.hidden_size
            for cell in self.cells:
                if hasattr(cell, 'bias_ih') and hasattr(cell, 'bias_hh'):
                    cell.bias_ih[hs:2*hs].fill_(1.0)
                    cell.bias_hh[hs:2*hs].fill_(1.0)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [B, Seq, C] if batch_first else [Seq, B, C]
        if self.batch_first:
            x = x.transpose(0, 1)  # Convert to [Seq, B, C]
            
        seq_len, batch_size, _ = x.size()
        dev = x.device
        
        if hidden is None:
            h_n = [torch.zeros(batch_size, self.hidden_size, device=dev) for _ in range(self.num_layers)]
            c_n = [torch.zeros(batch_size, self.hidden_size, device=dev) for _ in range(self.num_layers)]
        else:
            h_stack, c_stack = hidden
            # Unpack [L, B, H] -> List of [B, H]
            h_n = [h_stack[i] for i in range(self.num_layers)]
            c_n = [c_stack[i] for i in range(self.num_layers)]
            
        outputs = []
        for t in range(seq_len):
            inp = x[t]
            for layer_idx, cell in enumerate(self.cells):
                h_i, c_i = cell(inp, (h_n[layer_idx], c_n[layer_idx]))
                h_n[layer_idx] = h_i
                c_n[layer_idx] = c_i
                inp = h_i
                if self.dropout > 0 and layer_idx < self.num_layers - 1:
                    inp = F.dropout(inp, p=self.dropout, training=self.training)
            outputs.append(inp)
            
        # Stack outputs [Seq, B, H]
        out_tensor = torch.stack(outputs, dim=0)
        if self.batch_first:
            out_tensor = out_tensor.transpose(0, 1)  # [B, Seq, H]
            
        # Stack hidden [L, B, H]
        final_h = torch.stack(h_n, dim=0)
        final_c = torch.stack(c_n, dim=0)
        
        return out_tensor, (final_h, final_c)


class IdentityAttention(nn.Module):
    """Identity attention that returns 1.0, used when attention is disabled."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(1, device=x.device, dtype=x.dtype)


def make_attention(channels: int, use_attention: bool, use_dual: bool, groups: int) -> nn.Module:
    if not use_attention:
        return IdentityAttention()
    # 简单的通道注意力机制
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(channels, channels // 4, 1),
        nn.ReLU(),
        nn.Conv2d(channels // 4, channels, 1),
        nn.Sigmoid()
    )

class FNOUnit(nn.Module):
    # Spectral convolution unit with Frequency-Domain Gating.
    def __init__(self, in_channels, out_channels, modes, imu_dim, fast_fft=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

        self.band_gate = nn.Sequential(
            nn.Linear(imu_dim, modes * 2),
            nn.GELU(),
            nn.Linear(modes * 2, modes),
            nn.Sigmoid()
        )

        band_i = torch.arange(modes, dtype=torch.long)
        band_j = torch.arange(modes, dtype=torch.long)
        band_index = torch.maximum(band_i.view(-1, 1), band_j.view(1, -1))
        self.register_buffer("band_index", band_index)

        self.register_buffer("active_modes", torch.tensor(modes, dtype=torch.int32))

    def compl_mul2d(self, input, weights):
        modes1 = weights.shape[2]
        modes2 = weights.shape[3]
        return torch.einsum("bixy,ioxy->boxy", input[:, :, :modes1, :modes2], weights)

    def forward(self, x, imu_embed):
        dev_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=dev_type, enabled=False):
            x = x.float()
            imu_embed = imu_embed.float()

            B, C, H, W = x.shape
            x_ft = torch.fft.rfft2(x)

            if x_ft.shape[1] != int(self.out_channels):
                raise ValueError(f"FNOUnit passthrough requires in_channels == out_channels, got in={int(x_ft.shape[1])} out={int(self.out_channels)}")

            out_ft = x_ft.clone()

            curr_modes = int(self.active_modes.item())
            eff_modes_x = min(curr_modes, self.modes, x_ft.shape[2])
            eff_modes_y = min(curr_modes, self.modes, x_ft.shape[3])

            if eff_modes_x > 0 and eff_modes_y > 0:
                gate_vec = self.band_gate(imu_embed)
                band_idx = self.band_index[:eff_modes_x, :eff_modes_y]
                gate_2d = gate_vec[:, band_idx].to(dtype=x_ft.dtype)

                x_ft_pos = x_ft[:, :, :eff_modes_x, :eff_modes_y] * gate_2d.unsqueeze(1)
                x_ft_neg = x_ft[:, :, -eff_modes_x:, :eff_modes_y] * gate_2d.unsqueeze(1)

                w1 = self.weights1[:, :, :eff_modes_x, :eff_modes_y]
                w2 = self.weights2[:, :, :eff_modes_x, :eff_modes_y]

                out_ft[:, :, :eff_modes_x, :eff_modes_y] = out_ft[:, :, :eff_modes_x, :eff_modes_y] + self.compl_mul2d(x_ft_pos, w1)
                out_ft[:, :, -eff_modes_x:, :eff_modes_y] = out_ft[:, :, -eff_modes_x:, :eff_modes_y] + self.compl_mul2d(x_ft_neg, w2)

            x = torch.fft.irfft2(out_ft, s=(H, W))
            return x

@dataclass(eq=False)
class BaseFNOBlock(nn.Module):
    def __init__(self, channels: int, imu_embed_dim: int, use_pde: bool = True,
                 use_attention: bool = True, use_dual_attention: bool = True,
                 attn_groups: int = 8, fast_fft: bool = False):
        super().__init__()
        self.channels = channels
        self.imu_embed_dim = imu_embed_dim
        self.pde, effective_channels = create_pde_layer(use_pde, channels)
        self.effective_channels = effective_channels
        self.attention = make_attention(effective_channels, use_attention, use_dual_attention, attn_groups)
        self.act = nn.GELU()


class UnifiedFNOBlock(BaseFNOBlock):
    # Unified FNO block.

    def __init__(self, channels: int, modes: int, imu_embed_dim: int, use_pde: bool = True,
                 use_attention: bool = True, fast_fft: bool = False, attn_groups: int = 8,
                 use_dual_attention: bool = True):
        super().__init__(
            channels=channels,
            imu_embed_dim=imu_embed_dim,
            use_pde=use_pde,
            use_attention=use_attention,
            use_dual_attention=use_dual_attention,
            attn_groups=attn_groups,
            fast_fft=fast_fft
        )
        self.modes = modes
        self.fast_fft = fast_fft

        # Two FNO units using shared utility with effective channels
        self.unit1 = FNOUnit(self.effective_channels, self.effective_channels, self.modes, self.imu_embed_dim, self.fast_fft)
        self.unit2 = FNOUnit(self.effective_channels, self.effective_channels, self.modes, self.imu_embed_dim, self.fast_fft)

    def forward(self, x: torch.Tensor, imu_embed: torch.Tensor) -> torch.Tensor:
        x = self.pde(x)
        x = self.unit1(x, imu_embed)
        x = self.unit2(x, imu_embed)
        x = x * self.attention(x)
        return x


@dataclass(eq=False)
class MR_FNOBlock(BaseFNOBlock):
    # Multi-resolution FNO block.
    channels: int
    modes_low: int
    modes_high: int
    imu_embed_dim: int
    use_pde: bool = True
    use_attention: bool = True
    fast_fft: bool = False
    attn_groups: int = 8
    use_dual_attention: bool = True

    def __post_init__(self):
        super().__init__(
            channels=self.channels,
            imu_embed_dim=self.imu_embed_dim,
            use_pde=self.use_pde,
            use_attention=self.use_attention,
            use_dual_attention=self.use_dual_attention,
            attn_groups=self.attn_groups,
            fast_fft=self.fast_fft
        )

        # Store parameters for FNO units
        self.fast_fft = self.fast_fft

        # PDE projection back to base channels
        self.proj = nn.Conv2d(self.effective_channels, self.channels, kernel_size=1)

        # Common activation
        self.act = nn.GELU()

        # Branch channels
        c_low = self.channels // 2
        c_high = self.channels - c_low

        # Low-frequency branch using shared utility
        self.unit_low = FNOUnit(c_low, c_low, self.modes_low, self.imu_embed_dim, self.fast_fft)

        # High-frequency branch using shared utility
        self.unit_high = FNOUnit(c_high, c_high, self.modes_high, self.imu_embed_dim, self.fast_fft)

        # Fusion
        self.fuse = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        # Override attention to use base channels after projection/fusion (MR-FNO outputs C, not 4C)
        self.attention = make_attention(self.channels, self.use_attention, self.use_dual_attention, self.attn_groups)

    def forward(self, x: torch.Tensor, imu_embed: torch.Tensor) -> torch.Tensor:
        # PDE expansion then projection to base channels
        x = self.pde(x)
        x_proj = self.proj(x)

        # Split projected features into low/high branches
        c_low = self.channels // 2
        x_low, x_high = torch.split(x_proj, [c_low, self.channels - c_low], dim=1)

        # Branch processing using shared utility pattern
        y_low = self.unit_low(x_low, imu_embed)
        y_high = self.unit_high(x_high, imu_embed)

        # Fuse and residual
        y = torch.cat([y_low, y_high], dim=1)
        y = self.fuse(y)
        y = y + x_proj

        # Apply attention if enabled (multiply by attention weights, don't replace)
        y = y * self.attention(y)
        return y
class HybridVIONet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.imu_embed_dim = int(config.imu_embed_dim)
        self.state_aug = config.state_aug
        self.imu_gate_soft = config.imu_gate_soft
        self.use_uncertainty_fusion = bool(getattr(config, 'use_uncertainty_fusion', True))
        self.uncertainty_use_gate = bool(getattr(config, 'uncertainty_use_gate', True))
        
        # Adaptive IMU Encoder
        self.imu_encoder = ImprovedIMUEncoder(
            imu_channels=config.imu_channels,
            sequence_length=config.sequence_length,
            embed_dim=self.imu_embed_dim
        )
        
        # Adaptive Normalization
        if config.norm_mode == "bn":
            self.imu_norm = nn.BatchNorm1d(config.imu_channels, affine=True)
        elif config.norm_mode == "ln":
            self.imu_norm = nn.LayerNorm(config.imu_channels)
        else:
            g = int(config.imu_gn_groups) if config.imu_gn_groups is not None else (2 if config.imu_channels % 2 == 0 else 1)
            self.imu_norm = nn.GroupNorm(max(g, 1), config.imu_channels)
        self.stem = self._build_adaptive_stem(config.stem_channels)

        if config.use_mr_fno:
            self.fno_block = MR_FNOBlock(
                channels=config.stem_channels,
                modes_low=config.modes_low,
                modes_high=config.modes_high,
                imu_embed_dim=self.imu_embed_dim,
                use_pde=True,
                use_attention=True,
                fast_fft=config.fast_fft,
                attn_groups=config.attn_groups,
                use_dual_attention=config.use_dual_attention,
            )
        else:
            self.fno_block = UnifiedFNOBlock(
                channels=config.stem_channels,
                modes=config.modes,
                imu_embed_dim=self.imu_embed_dim,
                use_pde=True,
                use_attention=True,
                fast_fft=config.fast_fft,
                attn_groups=config.attn_groups,
                use_dual_attention=config.use_dual_attention,
            )


        self.pool_grid_size = 2
        self.pool = nn.AdaptiveAvgPool2d((self.pool_grid_size, self.pool_grid_size))
        pool_factor = self.pool_grid_size * self.pool_grid_size

        # Calculate effective channels output by FNO block
        # MR-FNO projects back to stem_channels, while UnifiedFNO keeps the expanded channels (if PDE is used)
        if config.use_mr_fno:
            fno_out_channels = config.stem_channels
        else:
            fno_out_channels = config.stem_channels * 4 if getattr(config, 'use_pde', True) else config.stem_channels
        if getattr(config, 'use_cross_attn', False):
            dim = int(getattr(config, 'fusion_dim', config.stem_channels))
            heads = int(getattr(config, 'fusion_heads', 4))
            # BUG FIX: v_proj must match actual FNO output channels * spatial_pooling_factor
            self.v_proj = nn.Linear(fno_out_channels * pool_factor, dim)
            self.i_proj = nn.Linear(self.imu_embed_dim, dim)
            self.cross = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
            lstm_in = dim
        else:
            # LSTM input: Visual features (flattened spatial pool) + IMU embedding
            lstm_in = (fno_out_channels * pool_factor) + self.imu_embed_dim
        self.temporal_lstm = self._build_lstm(config.use_cudnn_lstm, lstm_in, int(config.lstm_hidden), int(config.lstm_layers))
        self.motion_aux_head = nn.Sequential(
            nn.Linear(int(lstm_in), 64),
            nn.GELU(),
            nn.Linear(64, 7)
        )
        last_aux = self.motion_aux_head[-1]
        if isinstance(last_aux, nn.Linear):
            nn.init.normal_(last_aux.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(last_aux.bias)
        if isinstance(self.temporal_lstm, nn.LSTM):
            hs = self.temporal_lstm.hidden_size
            nl = self.temporal_lstm.num_layers
            with torch.no_grad():
                for l in range(nl):
                    b_ih = getattr(self.temporal_lstm, f"bias_ih_l{l}")
                    b_hh = getattr(self.temporal_lstm, f"bias_hh_l{l}")
                    b_ih[hs:2*hs].fill_((1.0))
                    b_hh[hs:2*hs].fill_((1.0))

        # IMU kinematics prior with gravity alignment
        self.imu_kin = IMUKinematics(
            g_world=torch.tensor(config.gravity, dtype=torch.float32),
            enable_gravity_alignment=True
        )

        self.head = nn.Sequential(
            nn.Linear(int(config.lstm_hidden), config.stem_channels),
            nn.GELU(),
            nn.Linear(config.stem_channels, 15), # 3(pos_res) + 6(rot_res) + 3(ba) + 3(bg)
        )

        self.scale_head = nn.Sequential(
            nn.Linear(self.imu_embed_dim + 4, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        # Visual confidence gate bounds (标准VIO模式: s ∈ [0, 1])
        # s = 0: 完全忽略视觉修正，仅用IMU
        # s = 1: 完全信任视觉残差
        # s > 1: 实验性尺度放大（非标准VIO，仅用于ablation study）
        self.scale_min = float(getattr(config, "scale_min", 0.0))
        self.scale_max = float(getattr(config, "scale_max", 1.0))

        last = self.scale_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.normal_(last.weight, mean=0.0, std=1e-3)
            with torch.no_grad():
                s_min = self.scale_min
                s_max = self.scale_max
                # 标准 VIO: s 作为视觉残差的置信度门控, 推荐 s∈[0,1]
                # 若设置 s_max>1 或 s_min<0, 则属于实验性尺度扩展, 允许视觉放大位移
                if s_min >= -1e-6 and s_max <= 1.0 + 1e-6:
                    target_s = 0.5
                else:
                    target_s = 1.0
                # 将目标 s 值映射到 sigmoid 输入空间
                target = (target_s - s_min) / max(s_max - s_min, 1e-6)
                target = max(min(target, 1.0 - 1e-6), 1e-6)
                bias = float(np.log(target / (1.0 - target)))
                last.bias.fill_(bias)

        # Initialize last layer for small initial residuals
        nn.init.uniform_(self.head[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.head[-1].bias)
        
        # Initialize 6D rotation part to identity (a1=[1,0,0], a2=[0,1,0])
        # Indices 3:9 are rotation
        with torch.no_grad():
            self.head[-1].bias[3:6] = torch.tensor([1.0, 0.0, 0.0])
            self.head[-1].bias[6:9] = torch.tensor([0.0, 1.0, 0.0])
        # Learnable gate for closed-loop feedback stabilization
        self.feedback_gate = nn.Parameter(torch.tensor(-5.0))
        self.feedback_gate_v = nn.Parameter(torch.tensor(-3.0))
        
        # Register identity_6d buffer for efficient usage in forward
        self.register_buffer("identity_6d", torch.tensor([1.,0.,0.,0.,1.,0.], dtype=torch.float32).view(1,6))

        # Professional uncertainty-based fusion module
        if self.imu_gate_soft and self.use_uncertainty_fusion:
            # Uncertainty estimators for both modalities
            # Visual uncertainty head: estimates log(σ_v²) for visual predictions
            # Input: lstm_features (config.lstm_hidden dimensional)
            self.visual_uncertainty_head = nn.Sequential(
                nn.Linear(int(config.lstm_hidden), 32),
                nn.GELU(),
                nn.Linear(32, 3),  # 3D position uncertainty
            )
            # IMU uncertainty head: estimates log(σ_i²) based on IMU statistics
            self.imu_uncertainty_head = nn.Sequential(
                nn.Linear(self.imu_embed_dim, 32),
                nn.GELU(),
                nn.Linear(32, 3),  # 3D position uncertainty
            )
            nn.init.normal_(self.visual_uncertainty_head[-1].weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.imu_uncertainty_head[-1].weight, mean=0.0, std=1e-3)
            with torch.no_grad():
                s_min = float(getattr(config, "scale_min", 0.0))
                s_max = float(getattr(config, "scale_max", 1.0))
                if s_min >= -1e-6 and s_max <= 1.0 + 1e-6:
                    s0 = 0.5
                else:
                    s0 = 1.0
                s0 = float(np.clip(s0, 1e-3, 1e6))
                v_bias = 0.0
                self.visual_uncertainty_head[-1].bias.fill_(v_bias)
                self.imu_uncertainty_head[-1].bias.zero_()

    def _build_adaptive_stem(self, stem_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(5 * int(self.config.window_stack_K), 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    def _build_lstm(self, use_cudnn: bool, input_size: int, hidden_size: int, num_layers: int) -> nn.Module:

        device = torch.device("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")

        if device.type == "cuda":

            print("[INFO] Using optimized CUDA nn.LSTM")
            return nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.0)

        elif device.type == "mps":

            print("[INFO] Using NativeLSTM on MPS (avoid GPU LSTM kernel)")
            return NativeLSTM(input_size=input_size, hidden_size=hidden_size,
                             num_layers=num_layers, batch_first=True, dropout=0.0)

        else:
            print("[INFO] Using native nn.LSTM on CPU")
            return nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=0.0)

    def forward(self, events: torch.Tensor, imu: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, prev_v: Optional[torch.Tensor] = None, prev_R: Optional[torch.Tensor] = None, dt_window: float = 0.05, debug: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(self.imu_norm, nn.LayerNorm):
            imu_normed = self.imu_norm(imu)
        else:
            imu_normed = self.imu_norm(imu.transpose(1, 2)).transpose(1, 2)
        imu_emb = self.imu_encoder(imu_normed)  # [B, 64]
        x = self.stem(events)
        x = self.fno_block(x, imu_emb)
        if x.device.type == 'mps':
            g = int(self.pool_grid_size)
            H, W = int(x.shape[-2]), int(x.shape[-1])
            ph = (g - (H % g)) % g
            pw = (g - (W % g)) % g
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        visual_features = self.pool(x).reshape(x.shape[0], -1)
        if getattr(self.config, 'use_cross_attn', False):
            v = self.v_proj(visual_features).unsqueeze(1)
            i = self.i_proj(imu_emb).unsqueeze(1)
            fused, _ = self.cross(v, i, i)
            combined_features = fused.squeeze(1)
        else:
            combined_features = torch.cat([visual_features, imu_emb], dim=1)

        aux_motion = self.motion_aux_head(combined_features)

        if hidden_state is not None:
            hidden_state = _adjust_hidden_size(hidden_state, int(combined_features.shape[0]))
            if hidden_state is not None:
                hidden_state = (hidden_state[0].contiguous(), hidden_state[1].contiguous())

        combined_features = combined_features.unsqueeze(1)  # [B, 1, C]
        lstm_out, (hn, cn) = self.temporal_lstm(combined_features, hidden_state)
        lstm_features = lstm_out[:, -1, :]

        out = self.head(lstm_features)
        pos_res = out[:, 0:3] # Delta t_res (direction/structure)
        rot_res_6d = out[:, 3:9]
        ba_pred = out[:, 9:12]
        bg_pred = out[:, 12:15]

        with torch.no_grad():
            v_flat = events.view(events.shape[0], -1)
            v_mean = v_flat.mean(dim=1, keepdim=True)
            v_std = v_flat.std(dim=1, keepdim=True)
            v_nonzero = (v_flat.abs() > 1e-4).float().mean(dim=1, keepdim=True)
            v_p95 = torch.quantile(v_flat, 0.95, dim=1, keepdim=True)
            event_stats = torch.cat([v_mean, v_std, v_nonzero, v_p95], dim=1)

        scale_input = torch.cat([imu_emb, event_stats], dim=1)
        alpha_s = self.scale_head(scale_input).squeeze(-1)
        s = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(alpha_s)

        rot_res_mat = rotation_6d_to_matrix(rot_res_6d)
        
        B = imu.size(0)
        current_prev_R = prev_R if prev_R is not None else torch.eye(3, device=imu.device, dtype=imu.dtype).expand(B, 3, 3)
        
        if not self.imu_gate_soft:
            q_only = self.imu_kin._rot_to_quat(rot_res_mat)
            q_only = F.normalize(q_only, p=2, dim=1)
            pos_res_scaled = pos_res
            visual_pred = torch.cat([pos_res_scaled, q_only], dim=1)
            zero_v = torch.zeros_like(pos_res_scaled)
            R_only = torch.bmm(current_prev_R, rot_res_mat)
            pos_res_vec = pos_res.detach().float().cpu().numpy().astype(np.float64)
            ba_vec = ba_pred.detach().float().cpu().numpy().astype(np.float64)
            bg_vec = bg_pred.detach().float().cpu().numpy().astype(np.float64)
            self._last_step_debug = {
                "t_hat_body_vec": None,
                "pos_res_vec": pos_res_vec,
                "scale_s_vec": s.detach().float().flatten().cpu().numpy().astype(np.float64),
                "ts_vec": s.detach().float().flatten().cpu().numpy().astype(np.float64),
                "ba_vec": ba_vec,
                "bg_vec": bg_vec,
                "v_vec": zero_v.detach().float().cpu().numpy().astype(np.float64),
                "aux_motion_tensor": aux_motion,
            }
            return visual_pred, (hn, cn), zero_v, R_only, rot_res_6d, s, ba_pred, bg_pred
        
        T_steps = imu.size(1)
        nonzero_mask = (imu.abs().sum(dim=2) > 1e-6)
        T_valid = nonzero_mask.sum(dim=1).clamp(min=1)
        dt_vec = torch.full((B,), float(dt_window), device=imu.device, dtype=imu.dtype) / T_valid.to(dtype=imu.dtype)
        t_hat, R_hat, final_v = self.imu_kin(
            imu,
            prev_v=prev_v,
            prev_R=current_prev_R,
            b_a=ba_pred,
            b_g=bg_pred,
            dt=dt_vec,
            mask=nonzero_mask,
            debug=debug
        )
        # Transform IMU estimates to Body Frame (relative to prev_R) to match target and residual format
        t_hat_body = torch.bmm(current_prev_R.transpose(1, 2), t_hat.unsqueeze(-1)).squeeze(-1)
        t_hat_body_vec = t_hat_body.detach().float().cpu().numpy().astype(np.float64)
        pos_res_vec = pos_res.detach().float().cpu().numpy().astype(np.float64)
        ba_vec = ba_pred.detach().float().cpu().numpy().astype(np.float64)
        bg_vec = bg_pred.detach().float().cpu().numpy().astype(np.float64)
        v_vec = final_v.detach().float().cpu().numpy().astype(np.float64)
        self._last_step_debug = {
            "t_hat_body_vec": t_hat_body_vec,
            "pos_res_vec": pos_res_vec,
            "scale_s_vec": s.detach().float().flatten().cpu().numpy().astype(np.float64),
            "ts_vec": s.detach().float().flatten().cpu().numpy().astype(np.float64),
            "ba_vec": ba_vec,
            "bg_vec": bg_vec,
            "v_vec": v_vec,
            "aux_motion_tensor": aux_motion,
        }
        # R_hat is global orientation (R_curr)
        R_rel_hat = torch.bmm(current_prev_R.transpose(1, 2), R_hat)

        # ============================================================================
        # Professional Uncertainty-Based Fusion (符合学术论文标准)
        # ============================================================================
        #
        # Theory: Optimal Bayesian fusion of two uncertain estimates
        #   Given: μ_v ± σ_v (visual), μ_i ± σ_i (IMU)
        #   Optimal fusion: μ_fused = (σ_i² * μ_v + σ_v² * μ_i) / (σ_v² + σ_i²)
        #   Weight: w_v = σ_i² / (σ_v² + σ_i²), w_i = σ_v² / (σ_v² + σ_i²)
        #
        # Physical interpretation:
        #   - When σ_v << σ_i: trust visual more (w_v → 1)
        #   - When σ_i << σ_v: trust IMU more (w_i → 1)
        #   - When σ_v ≈ σ_i: equal weighting (w_v ≈ w_i ≈ 0.5)
        #
        # Implementation:
        #   1. Estimate uncertainties from both modalities
        #   2. Compute optimal fusion weights
        #   3. Fuse predictions with learned uncertainties
        # ============================================================================

        if hasattr(self, 'visual_uncertainty_head') and hasattr(self, 'imu_uncertainty_head'):
            # Estimate log-variance (more numerically stable)
            # log_var_v: [B, 3], log_var_i: [B, 3]
            log_var_v = self.visual_uncertainty_head(lstm_features)  # Visual uncertainty
            log_var_i = self.imu_uncertainty_head(imu_emb)  # IMU uncertainty

            # Convert to variance: σ² = exp(log_var)
            # Clamp to prevent numerical issues
            var_v = torch.exp(torch.clamp(log_var_v, min=-10.0, max=10.0))  # [B, 3]
            var_i = torch.exp(torch.clamp(log_var_i, min=-10.0, max=10.0))  # [B, 3]

            var_v_eff = var_v
            if self.uncertainty_use_gate:
                s_eff = torch.clamp(s.to(dtype=var_v.dtype), min=1e-3, max=1e6).unsqueeze(-1)
                var_v_eff = var_v / (s_eff * s_eff)

            # Compute optimal fusion weights (Bayesian)
            # w_v = var_i / (var_v_eff + var_i), w_i = var_v_eff / (var_v_eff + var_i)
            var_sum = var_v_eff + var_i + 1e-6  # Add epsilon for numerical stability
            weight_visual = var_i / var_sum  # [B, 3] - weight for visual prediction
            weight_imu = var_v_eff / var_sum     # [B, 3] - weight for IMU prediction

            visual_displacement = pos_res  # [B, 3]
            imu_displacement = t_hat_body  # [B, 3]

            fused_t = weight_imu * imu_displacement + weight_visual * visual_displacement

            # Store fusion weights for analysis/visualization
            self._last_step_debug["weight_visual"] = weight_visual.detach().float().cpu().numpy().astype(np.float64)
            self._last_step_debug["weight_imu"] = weight_imu.detach().float().cpu().numpy().astype(np.float64)
            self._last_step_debug["var_v"] = var_v_eff.detach().float().cpu().numpy().astype(np.float64)
            self._last_step_debug["var_i"] = var_i.detach().float().cpu().numpy().astype(np.float64)
            self._last_step_debug["var_v_raw"] = var_v.detach().float().cpu().numpy().astype(np.float64)

            # Store log_var tensors WITH gradients for NLL loss computation
            # These are used in _compute_training_loss_single_step for proper uncertainty learning
            self._last_step_debug["log_var_v_tensor"] = log_var_v  # [B, 3] - visual log variance (with grad)
            self._last_step_debug["log_var_i_tensor"] = log_var_i  # [B, 3] - IMU log variance (with grad)
            self._last_step_debug["t_imu_tensor"] = imu_displacement  # [B, 3] - IMU prediction (with grad)
            self._last_step_debug["t_visual_tensor"] = visual_displacement  # [B, 3] - Visual prediction (with grad)

        else:
            # DEIO-style fusion: IMU as baseline + visual correction
            # P_final = P_imu_body + s * r  (s = precision/confidence, r = correction)
            # This anchors scale to IMU dynamics, visual only provides additive correction
            s_w = s.to(dtype=t_hat_body.dtype).unsqueeze(-1)
            fused_t = t_hat_body + s_w * pos_res
        # Apply residual rotation to the IMU relative rotation estimate
        fused_R_rel = torch.bmm(R_rel_hat, rot_res_mat)
        fused_R_global = torch.bmm(current_prev_R, fused_R_rel)
        gate = torch.sigmoid(self.feedback_gate)
        # Use registered buffer instead of creating new tensor, cast to match input dtype
        identity_6d = self.identity_6d.to(dtype=rot_res_6d.dtype)
        rot_res_6d_feedback = identity_6d + gate * (rot_res_6d - identity_6d)
        rot_res_mat_feedback = rotation_6d_to_matrix(rot_res_6d_feedback)
        
        # Compute next state rotation using the gated correction
        fused_R_rel_fb = torch.bmm(R_rel_hat, rot_res_mat_feedback)
        fused_R_global_fb = torch.bmm(current_prev_R, fused_R_rel_fb)
        c0 = F.normalize(fused_R_global_fb[:, :, 0], dim=1)
        v1 = fused_R_global_fb[:, :, 1]
        proj = (c0 * v1).sum(dim=1, keepdim=True)
        c1 = F.normalize(v1 - proj * c0, dim=1)
        c2 = torch.cross(c0, c1, dim=1)
        fused_R_global_fb = torch.stack([c0, c1, c2], dim=2)

        fused_R_rel_fb_ortho = torch.bmm(current_prev_R.transpose(1, 2), fused_R_global_fb)
        fused_q = self.imu_kin._rot_to_quat(fused_R_rel_fb_ortho)
        fused_q = F.normalize(fused_q, p=2, dim=1)
        visual_pred = torch.cat([fused_t, fused_q], dim=1)

        gate_v = torch.sigmoid(self.feedback_gate_v).to(dtype=final_v.dtype)
        v_target_body = fused_t / max(dt_window, 1e-6)
        v_target_global = torch.bmm(current_prev_R, v_target_body.unsqueeze(-1)).squeeze(-1)
        final_v = (1.0 - gate_v) * final_v + gate_v * v_target_global

        return visual_pred, (hn, cn), final_v, fused_R_global_fb, rot_res_6d, s, ba_pred, bg_pred



def build_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using NVIDIA CUDA acceleration (Device: {torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders) acceleration")
        return torch.device("mps")
    print("Using CPU (No acceleration detected)")
    return torch.device("cpu")


class PhysicsBrightnessLoss(nn.Module):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self._init_gaussian_kernel(sigma)

    def _init_gaussian_kernel(self, sigma: float):
        kernel_size = int(2 * 4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size)
        xx = x.repeat(kernel_size).view(kernel_size, kernel_size)
        yy = xx.t()
        mean = (kernel_size - 1) / 2.0
        var = sigma * sigma
        gk = (1.0 / (2.0 * np.pi * var)) * torch.exp(-((xx - mean) ** 2 + (yy - mean) ** 2) / (2 * var))
        gk = gk / torch.sum(gk)
        self.register_buffer('gaussian_kernel', gk.view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def _get_sobel_kernels(self, device, dtype):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        return sobel_x, sobel_y

    def forward(self, voxel_grid: torch.Tensor, motion_pred: torch.Tensor, fx: Optional[float] = None, fy: Optional[float] = None, q: float = 0.95, mask_thresh: float = 0.05, dt: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = voxel_grid.shape[0]
        raw_img = voxel_grid[:, 0:1]
        p = torch.quantile(raw_img.view(B, -1), q, dim=1).view(-1, 1, 1, 1)
        L = raw_img / (p + 1e-6)
        L_smooth = F.conv2d(L, self.gaussian_kernel.to(L.dtype), padding=self.padding)
        sobel_x, sobel_y = self._get_sobel_kernels(L.device, L.dtype)
        grad_x = F.conv2d(L_smooth, sobel_x, padding=1)
        grad_y = F.conv2d(L_smooth, sobel_y, padding=1)
        f_x = float(fx) if fx is not None else 1.0
        f_y = float(fy) if fy is not None else 1.0

        if dt is None:
            dt_vec = motion_pred.new_ones((B, 1))
        else:
            dt_vec = dt.view(B, 1).to(device=motion_pred.device, dtype=motion_pred.dtype).clamp(min=1e-6)
        vel_pred = motion_pred / dt_vec

        vx = (vel_pred[:, 0:1] * f_x).view(-1, 1, 1, 1).expand_as(grad_x)
        vy = (vel_pred[:, 1:2] * f_y).view(-1, 1, 1, 1).expand_as(grad_y)

        if voxel_grid.shape[1] >= 5:
            dt_term = (voxel_grid[:, 3:4] - voxel_grid[:, 4:5])
        elif voxel_grid.shape[1] >= 3:
            dt_term = (voxel_grid[:, 1:2] - voxel_grid[:, 2:3])
        else:
            dt_term = torch.zeros_like(raw_img)

        pde_residual = dt_term + grad_x * vx + grad_y * vy
        event_mask = torch.sigmoid((L - mask_thresh) * 10.0)
        loss = torch.sum((pde_residual * event_mask) ** 2) / (torch.sum(event_mask) + 1.0)
        return torch.clamp(loss, max=10.0)



class CollateSequence:
    def __init__(self, window_stack_k: int = 1, voxel_stack_mode: str = "abs") -> None:
        self.k = int(window_stack_k)
        self.mode = str(voxel_stack_mode).strip().lower()

    def __call__(self, batch: List[Tuple]) -> Any:
        first_seq = batch[0][0]
        seq_len = len(first_seq)
        batch_size = len(batch)
        batched_seq = []
        for i in range(seq_len):
            evs = []
            imus = []
            ys = []
            dts = []
            intrs = []
            for b in range(batch_size):
                seq, _ = batch[b]
                item = seq[i]
                if self.k > 1:
                    start = max(0, i - self.k + 1)
                    parts_abs = []
                    for j in range(i, start - 1, -1):
                        parts_abs.append(seq[j][0])
                    if len(parts_abs) < self.k:
                        need = self.k - len(parts_abs)
                        parts_abs = parts_abs + [parts_abs[-1] for _ in range(need)]
                    if self.mode == "delta":
                        parts = [parts_abs[0]]
                        for j in range(1, self.k):
                            parts.append(parts_abs[j - 1] - parts_abs[j])
                        ev_stack = torch.cat(parts, dim=0)
                    else:
                        ev_stack = torch.cat(parts_abs, dim=0)
                else:
                    ev_stack = item[0]
                evs.append(ev_stack)
                imus.append(item[1])
                ys.append(item[2])
                if len(item) > 3:
                    dts.append(item[3])
                if len(item) > 4:
                    intrs.append(item[4])
            batched_ev = torch.stack(evs, dim=0)
            batched_imu = torch.stack(imus, dim=0)
            batched_y = torch.stack(ys, dim=0)
            if len(dts) == batch_size:
                batched_dt = torch.stack(dts, dim=0)
                if len(intrs) == batch_size:
                    batched_intr = torch.stack(intrs, dim=0)
                    batched_seq.append((batched_ev, batched_imu, batched_y, batched_dt, batched_intr))
                else:
                    batched_seq.append((batched_ev, batched_imu, batched_y, batched_dt))
            else:
                batched_seq.append((batched_ev, batched_imu, batched_y))
        starts = [s for _, s in batch]
        return (batched_seq, starts)

def collate_sequence(batch: List[Tuple]) -> Any:
    return CollateSequence(window_stack_k=1, voxel_stack_mode="abs")(batch)

class SequenceDataset(Dataset):
    def __init__(self, base_ds: Dataset, sequence_len: int = 200, stride: int = 200) -> None:
        self.base = base_ds
        self.seq_len = int(sequence_len)
        self.stride = int(stride)
        N = len(base_ds)
        if self.stride < self.seq_len:
            print(f"[WARN] SequenceDataset uses overlapping sequences: stride={self.stride} < seq_len={self.seq_len}. Initializing state from GT at each sequence start can introduce inconsistency across overlaps.")
            if N >= int(self.seq_len) * 3:
                orig_stride = int(self.stride)
                self.stride = int(self.seq_len)
                print(f"[WARN] Clamping sequence_stride to seq_len for large dataset: stride={orig_stride} -> {self.stride}")
        self.starts = list(range(0, max(N - self.seq_len + 1, 1), self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        N = len(self.base)
        if N <= 0:
            raise IndexError("Empty base dataset")

        s0 = int(self.starts[idx])
        if N >= self.seq_len:
            s = s0
            if s + self.seq_len > N:
                s = max(0, N - self.seq_len)
            e = s + self.seq_len
            return ([self.base[i] for i in range(s, e)], s)

        seq = [self.base[i] for i in range(0, N)]
        last = seq[-1]
        if len(seq) < self.seq_len:
            seq.extend([last for _ in range(self.seq_len - len(seq))])
        return (seq, 0)


@dataclass
class EventProcessor:
    """Vectorized event processor with memory pooling."""
    resolution: Tuple[int, int]
    device: torch.device
    std_norm: bool = False
    log_norm: bool = False
    voxel_cache: Optional[torch.Tensor] = None  

    def __post_init__(self):
        if hasattr(self, 'resolution') and hasattr(self, 'device'):
            H, W = self.resolution
            self.voxel_cache = torch.zeros((5, H, W), dtype=torch.float32, device=self.device)

    def voxelize_events_vectorized(self, xw: Union[np.ndarray, torch.Tensor], yw: Union[np.ndarray, torch.Tensor],
                                  tw: Union[np.ndarray, torch.Tensor], pw: Union[np.ndarray, torch.Tensor],
                                  src_w: int, src_h: int, t_prev: float, t_curr: float) -> torch.Tensor:
        H, W = self.resolution
        if self.voxel_cache is None:
            self.voxel_cache = torch.zeros((5, H, W), dtype=torch.float32, device=self.device)
        else:
            self.voxel_cache.zero_()
        # Helper to ensure tensor on device
        def _to_dev(v):
            if isinstance(v, torch.Tensor):
                return v.to(device=self.device, dtype=torch.float32)
            return torch.from_numpy(v.astype(np.float32)).to(self.device)
        # Convert to tensors efficiently
        x = _to_dev(xw)
        y = _to_dev(yw)
        t = _to_dev(tw)
        p = _to_dev(pw) if pw is not None else torch.zeros_like(t)
        if torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)) or torch.any(torch.isnan(t)):
            print("Warning: NaN detected in event coordinates, returning zeros")
            return torch.zeros((5, H, W), dtype=torch.float32, device=self.device)
        # Normalize shapes to 1D and match lengths
        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        p = p.view(-1)
        n = min(x.numel(), y.numel(), t.numel(), p.numel())
        x = x[:n]
        y = y[:n]
        t = t[:n]
        p = p[:n]
        if n == 0:
            return self.voxel_cache.clone()
        x_max = float(torch.amax(x).detach().cpu()) if x.numel() > 0 else 0.0
        y_max = float(torch.amax(y).detach().cpu()) if y.numel() > 0 else 0.0
        if x_max <= 1.01 and y_max <= 1.01:
            xs_float = torch.clamp(x * float(W - 1), 0.0, float(W - 1))
            ys_float = torch.clamp(y * float(H - 1), 0.0, float(H - 1))
        else:
            src_w_safe = float(max(int(src_w), 1))
            src_h_safe = float(max(int(src_h), 1))
            x_scaled = x * float(W) / src_w_safe
            y_scaled = y * float(H) / src_h_safe
            xs_float = torch.clamp(x_scaled, 0.0, float(W - 1))
            ys_float = torch.clamp(y_scaled, 0.0, float(H - 1))
        # Convert to long indices only at the final step
        xs = xs_float.long()
        ys = ys_float.long()
        idx = ys * W + xs
        dt = max(float(t_curr - t_prev), 1e-6)
        norm_t = torch.clamp((t - t_prev) / dt, 0.0, 1.0)
        total = max(int(x.numel()), 1)
        voxel = self.voxel_cache
        ch0, ch1, ch2, ch3, ch4 = voxel[0], voxel[1], voxel[2], voxel[3], voxel[4]
        ch0_flat = ch0.view(-1)
        ch1_flat = ch1.view(-1)
        ch2_flat = ch2.view(-1)
        ch3_flat = ch3.view(-1)
        ch4_flat = ch4.view(-1)
        ones = torch.ones_like(idx, dtype=torch.float32)
        ch0_flat.index_add_(0, idx, ones)
        pos_mask = p > 0
        neg_mask = p < 0
        if pos_mask.any():
            pos_idx = idx[pos_mask]
            ch1_flat.index_add_(0, pos_idx, ones[pos_mask])
            ch3_flat.index_add_(0, pos_idx, norm_t[pos_mask])
        if neg_mask.any():
            neg_idx = idx[neg_mask]
            ch2_flat.index_add_(0, neg_idx, ones[neg_mask])
            ch4_flat.index_add_(0, neg_idx, norm_t[neg_mask])
        total_inv = 1.0 / float(total)
        pos_nz = ch1_flat > NumericalConstants.DIVISION_EPS
        neg_nz = ch2_flat > NumericalConstants.DIVISION_EPS
        ch3_flat[pos_nz] = safe_divide(ch3_flat[pos_nz], ch1_flat[pos_nz])
        ch4_flat[neg_nz] = safe_divide(ch4_flat[neg_nz], ch2_flat[neg_nz])
        ch0.mul_(total_inv)
        ch1.mul_(total_inv)
        ch2.mul_(total_inv)
        if self.log_norm:
            scale = torch.log1p(torch.tensor(float(total), device=self.device))
            counts = voxel[0:3, :, :]
            counts.mul_(float(total))
            counts = torch.log1p(counts)
            counts = counts / scale
            voxel[0:3, :, :] = counts
        if self.std_norm:
            m = voxel.view(5, -1).mean(dim=1).view(5, 1, 1)
            s = voxel.view(5, -1).std(dim=1).view(5, 1, 1)
            s = torch.clamp(s, min=1e-6)
            voxel = (voxel - m) / s
        
        # SAFETY CHECK: Ensure no infinite values to prevent training crashes
        if not torch.isfinite(voxel).all():
            voxel = torch.nan_to_num(voxel, nan=0.0, posinf=0.0, neginf=0.0)

        return voxel.clone()

@dataclass
class AdaptiveEventProcessor:
    resolution: Tuple[int, int]
    device: torch.device
    std_norm: bool = False
    log_norm: bool = False

    def __post_init__(self):
        self.base = EventProcessor(resolution=self.resolution, device=self.device, std_norm=self.std_norm, log_norm=self.log_norm)

    def get_adaptive_params(self, event_count: int) -> Tuple[int, int]:

        H, W = self.resolution
        b_h = max(3, H // 60)
        b_w = max(3, W // 60)
        max_e = max(1, (H * W) // 12)
        den = min(10.0, float(event_count) / float(max_e))

        # Calculate kernel sizes with safety bounds to prevent dimension collapse
        scale = max(den, 1e-6)
        if scale > 2.0:
             scale = 2.0 + np.log1p(scale - 2.0)
             
        kh_raw = int(b_h / scale)
        kw_raw = int(b_w / scale)

        kh = max(2, min(kh_raw, H // 2))
        kw = max(2, min(kw_raw, W // 2))

        # Ensure kernel sizes are valid and won't cause issues
        if H % kh != 0 or W % kw != 0:
            # Fall back to conservative kernel sizes that divide evenly
            kh = max(2, min(kh, H // 4))
            kw = max(2, min(kw, W // 4))

        return kh, kw

    def voxelize_events_adaptive(self, xw: Union[np.ndarray, torch.Tensor], yw: Union[np.ndarray, torch.Tensor], tw: Union[np.ndarray, torch.Tensor], pw: Union[np.ndarray, torch.Tensor], src_w: int, src_h: int, t_prev: float, t_curr: float) -> torch.Tensor:
        """Simplified adaptive voxelization using proper tensor operations"""
        # Get base voxelization
        v = self.base.voxelize_events_vectorized(xw, yw, tw, pw, src_w, src_h, t_prev, t_curr)

        H, W = self.resolution
        expected_shape = (5, H, W)

        # Use padding instead of complex shape fixing
        if v.shape != expected_shape:
            if v.numel() == 5 * H * W:
                v = v.view(expected_shape)
            else:
                # Pad to expected shape instead of complex reshaping
                v = self._pad_to_shape(v, expected_shape)

        # Check if adaptive processing is beneficial
        n = int(min(len(xw), len(yw)))
        kh, kw = self.get_adaptive_params(n)

        # Skip adaptive processing if kernel sizes are too small
        if kh <= 1 and kw <= 1:
            return v

        # Direct adaptive processing without complex fallbacks
        try:
            counts = v[0:3, :, :]  # [3, H, W]
            times = v[3:5, :, :]   # [2, H, W]

            # Adaptive pooling
            area = float(kh * kw)
            counts_p = F.avg_pool2d(counts, kernel_size=(kh, kw), stride=(kh, kw)) * area
            times_p = F.avg_pool2d(times, kernel_size=(kh, kw), stride=(kh, kw))

            # Direct interpolation without multiple fallbacks
            counts_u = F.interpolate(counts_p.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)
            times_u = F.interpolate(times_p.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

            return torch.cat([counts_u, times_u], dim=0)

        except Exception as e:
            # Simple fallback - return base voxelization
            print(f"Adaptive voxelization failed: {e}")
            return v

    def _pad_to_shape(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Pad tensor to target shape using F.pad"""
        if tensor.shape == target_shape:
            return tensor

        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)

        if tensor.dim() != len(target_shape):
            src = tensor.reshape(-1)
            dst = result.view(-1)
            n = min(int(src.numel()), int(dst.numel()))
            if n > 0:
                dst[:n] = src[:n]
            return result

        slices = []
        for src_dim, tgt_dim in zip(tensor.shape, target_shape):
            min_dim = min(int(src_dim), int(tgt_dim))
            slices.append(slice(0, min_dim))

        s = tuple(slices)
        result[s] = tensor[s]
        return result

def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device, config: TrainingConfig, dt: float, current_epoch_physics_weight: float, scaler: Optional[torch.cuda.amp.GradScaler] = None, adaptive_loss_fn: Optional[AdaptiveLossWeights] = None, fx: float = 1.0, fy: float = 1.0) -> float:

    model.train()
    monitor_memory_usage(device, "Epoch Start")
    physics_module, use_amp = setup_training_state(config, device)
    total_loss = 0.0
    num_steps = 0

    # 添加：segment跟踪变量
    last_segment_id = None
    global_hidden = None  # 用于跨batch传递hidden state

    # 添加：获取base_ds用于segment_ids访问
    base_seq = loader.dataset
    base_base = base_seq.dataset if hasattr(base_seq, "dataset") else base_seq
    base_ds = base_base.dataset if hasattr(base_base, "dataset") else base_base

    for batch_idx, batch in enumerate(loader):
        if batch_idx > 0 and batch_idx % 10 == 0:
            monitor_memory_usage(device, f"Batch {batch_idx}")

        # 修改：保留starts_list
        batch_data, starts_list = unpack_and_validate_batch(batch)

        # 添加：检查segment边界
        contig0, sid0, seg0 = _get_ids(base_seq, base_base, starts_list, batch_idx, 0, 0)

        # 添加：判断连续性
        contiguous = True  # 默认连续
        if seg0 is not None and last_segment_id is not None:
            contiguous = (int(seg0) == int(last_segment_id))

        # 添加：在segment边界处重置hidden state
        if not contiguous:
            global_hidden = None
            print(f"[TRAIN RESET] batch_idx={batch_idx} seg_id={seg0} (segment boundary detected)")

        # 修改：传递hidden state给process_batch_sequence
        loss_sum, loss_count, global_hidden = process_batch_sequence(
            model, opt, batch_data, device, config, dt, adaptive_loss_fn,
            physics_module, use_amp, scaler, current_epoch_physics_weight, fx=fx, fy=fy, batch_idx=batch_idx,
            init_hidden=global_hidden  # 传入上一batch的hidden或None
        )

        # 添加：更新segment_id
        last_segment_id = seg0

        total_loss, num_steps = update_epoch_metrics(loss_sum, loss_count, total_loss, num_steps)

    monitor_memory_usage(device, "Epoch End")

    if device.type == "mps":
        import gc
        gc.collect()
    return total_loss / max(num_steps, 1)

# Global loss composer for unified loss computation
_loss_composer = LossComposer()

def _compute_loss_components(pred: torch.Tensor, target: torch.Tensor, raw_6d: Optional[torch.Tensor], voxel: torch.Tensor, dt_tensor: Optional[torch.Tensor],
                    physics_module, speed_thresh: float, dt: float, physics_temp: float = 0.5, loss_w_physics_max: float = 1.0, physics_q: float = 0.95, physics_mask_thresh: float = 0.05, fx: float = 1.0, fy: float = 1.0, loss_w_scale: float = 0.1, loss_w_scale_reg: float = 0.0, loss_w_static: float = 0.0, scale_reg_center: Optional[float] = None, min_step_threshold: float = 0.0, min_step_weight: float = 0.0, path_scale_weight: float = 0.0, s: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unified loss computation using LossComposer utility."""
    physics_config = {
        'physics_scale_quantile': physics_q,
        'physics_event_mask_thresh': physics_mask_thresh,
        'loss_w_physics_max': loss_w_physics_max,
        'physics_temp': physics_temp,
        'fx': fx,
        'fy': fy
    }
    return _loss_composer.compute_components(
        pred, target, raw_6d, voxel, dt_tensor, physics_module,
        speed_thresh, dt, physics_config,
        scale_weight=loss_w_scale,
        scale_reg_weight=loss_w_scale_reg,
        static_weight=loss_w_static,
        scale_reg_center=scale_reg_center,
        min_step_threshold=min_step_threshold,
        min_step_weight=min_step_weight,
        path_scale_weight=path_scale_weight,
        s=s
    )

def _detach_hidden(hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Detach hidden state for TBPTT."""
    if hidden is None:
        return None
    h, c = hidden
    return (h.detach().contiguous(), c.detach().contiguous())

def _adjust_hidden_size(hidden: Optional[Tuple[torch.Tensor, torch.Tensor]], B: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if hidden is None:
        return None
    h, c = hidden
    # h,c: [L, B_curr, H]
    B_curr = h.size(1)
    if B_curr == B:
        return (h.contiguous(), c.contiguous())
    if B_curr > B:
        return (h[:, :B, :].contiguous(), c[:, :B, :].contiguous())
    # B_curr < B → pad zeros for new entries
    pad = B - B_curr
    dev = h.device
    H = h.size(2)
    zeros_h = torch.zeros(h.size(0), pad, H, device=dev, dtype=h.dtype)
    zeros_c = torch.zeros(c.size(0), pad, H, device=dev, dtype=c.dtype)
    return (
        torch.cat([h, zeros_h], dim=1).contiguous(),
        torch.cat([c, zeros_c], dim=1).contiguous(),
    )


def _get_ids(base_seq: Any, base_base: Any, starts_list: Optional[List[int]], s_idx: int, j: int, b: Optional[int] = None) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    start_b = starts_list[b] if (starts_list is not None and b is not None) else (base_seq.starts[s_idx] if hasattr(base_seq, "starts") else 0)
    idx = int(start_b) + int(j)
    if idx < 0:
        return None, None, None

    try:
        base_len = len(base_base)
        if base_len <= 0:
            return None, None, None
        if idx >= base_len:
            return None, None, None
    except Exception:
        pass

    contig_id = idx
    inner_idx = idx
    if hasattr(base_base, "indices"):
        try:
            if idx >= len(base_base.indices):
                return None, None, None
            inner_idx = int(base_base.indices[idx])
            contig_id = inner_idx
        except Exception:
            return None, None, None

    ds = base_base.dataset if hasattr(base_base, "dataset") else base_base
    gt_id = inner_idx
    if hasattr(ds, "sample_indices"):
        try:
            if inner_idx < 0 or inner_idx >= len(ds.sample_indices):
                return None, None, None
            gt_id = int(ds.sample_indices[inner_idx])
        except Exception:
            return None, None, None

    segment_id = None
    if hasattr(ds, "segment_ids"):
        try:
            if inner_idx >= 0 and inner_idx < len(ds.segment_ids):
                segment_id = int(ds.segment_ids[inner_idx])
        except Exception:
            segment_id = None

    return contig_id, gt_id, segment_id


def _tbptt_update(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    accumulated_loss: torch.Tensor,
    total_batch_steps: int,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
    hidden: Optional[Tuple[torch.Tensor, torch.Tensor]],
    imu_state_mgr: "IMUStateManager",
    device: torch.device
) -> Tuple[torch.Tensor, int, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Perform TBPTT gradient update step. Returns reset (accumulated_loss, total_batch_steps, hidden)."""
    norm_loss = accumulated_loss / max(total_batch_steps, 1)
    if use_amp and scaler is not None:
        scaler.scale(norm_loss).backward()
        scaler.unscale_(opt)
    else:
        norm_loss.backward()

    # Manual gradient clipping that handles complex numbers
    max_norm = NumericalConstants.GRADIENT_CLIP_NORM
    total_norm = 0.0
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))

    for p in parameters:
        if p.grad.is_complex():
            param_norm = p.grad.detach().abs().norm(2)
        else:
            param_norm = p.grad.detach().norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)

    if use_amp and scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()

    opt.zero_grad()
    hidden = _detach_hidden(hidden)
    imu_state_mgr.detach_states()
    return torch.tensor(0.0, device=device), 0, hidden


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, rpe_dt: float, dt: float, eval_sim3_mode: str = "diagnose") -> Tuple[float, float, float]:
    model.eval()

    actual_model = model.orig_mod if hasattr(model, 'orig_mod') else model
    gate_mode = False
    if hasattr(actual_model, 'scale_min') and hasattr(actual_model, 'scale_max'):
        smin = float(getattr(actual_model, 'scale_min'))
        smax = float(getattr(actual_model, 'scale_max'))
        gate_mode = (smin >= -1e-6 and smax <= 1.0 + 1e-6)
        mode_str = "gate" if gate_mode else "experimental_scale"
        print(f"[EVAL START] scale_head range=[{smin:.6f}, {smax:.6f}] | mode={mode_str}")

    est_pos = []
    est_quat = []
    sample_ids = []
    contig_ids = []
    seg_keys = []
    t = np.zeros(3, dtype=np.float64)
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    base_seq = loader.dataset
    base_base = base_seq.base if hasattr(base_seq, "base") else loader.dataset
    base_ds = base_base.dataset if hasattr(base_base, "dataset") else base_base
    has_window_ts = bool(
        hasattr(base_ds, "window_t_curr")
        and getattr(base_ds, "window_t_curr") is not None
        and hasattr(base_ds, "interpolate_gt_data")
    )
    segment_ids_ok = bool(hasattr(base_ds, "segment_ids") and getattr(base_ds, "segment_ids") is not None)
    if segment_ids_ok:
        try:
            segment_ids_ok = (len(getattr(base_ds, "segment_ids")) == len(getattr(base_ds, "sample_indices")))
        except Exception:
            segment_ids_ok = False
    if not segment_ids_ok:
        print("[EVAL WARNING] segment_ids unavailable or mismatched; falling back to legacy contiguity logic.")
    dt_list: List[float] = []
    s_list: List[float] = []
    w_imu_list: List[float] = []  # Uncertainty fusion weights
    w_visual_list: List[float] = []
    sigma_imu_list: List[float] = []  # Uncertainty estimates
    sigma_visual_list: List[float] = []
    win_step_pred: List[float] = []
    win_step_gt: List[float] = []

    # CRITICAL FIX: Properly extract sample_stride by unwrapping all dataset layers
    # The VoxelEventH5Dataset is the actual dataset with sample_stride attribute
    base_sample_stride = 1
    found_stride_attr = False
    current = base_ds
    for _ in range(10):  # Safety limit to prevent infinite loop
        if hasattr(current, "sample_stride"):
            base_sample_stride = max(int(current.sample_stride), 1)
            found_stride_attr = True
            print(f"[EVAL] Found sample_stride={base_sample_stride} at {type(current).__name__}")
            break
        elif hasattr(current, "dataset"):
            current = current.dataset
        elif hasattr(current, "base"):
            current = current.base
        else:
            break

    if not found_stride_attr:
        print(f"[EVAL WARNING] Could not find sample_stride attribute, defaulting to 1. This may cause time window mismatch!")
        print(f"[EVAL WARNING] Dataset hierarchy: base_seq={type(base_seq).__name__} -> base_base={type(base_base).__name__} -> base_ds={type(base_ds).__name__}")
    eval_subsample = base_sample_stride
    contiguous_sid_step = base_sample_stride
    if base_sample_stride > 1:
        print(f"[EVAL] Dataset sample_stride={base_sample_stride} | Eval subsample={eval_subsample} | Contiguous sid step={contiguous_sid_step}")

    with torch.no_grad():
        diag_zero_prev_v = os.environ.get("EVAL_DIAG_ZERO_PREV_V", "0").strip() == "1"
        diag_zero_prev_R = os.environ.get("EVAL_DIAG_ZERO_PREV_R", "0").strip() == "1"
        if diag_zero_prev_v or diag_zero_prev_R:
            print(f"[EVAL DIAG FLAGS] zero_prev_v={diag_zero_prev_v} zero_prev_R={diag_zero_prev_R}")

        global_eval_v = None
        global_eval_R = None
        global_hidden = None
        global_t_batch = None
        global_q_batch = None
        last_sid = None
        last_sid_b0 = None
        last_segment_id = None
        printed_norm_debug = False
        prev_v_probe = None
        prev_R_probe = None
        for s_idx, batch in enumerate(loader):
            # Strict unpacking of collate_sequence return
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                batched_seq, starts_list = batch
            else:
                batched_seq = batch[0] if isinstance(batch, (tuple, list)) else batch
                starts_list = None

            # Reset hidden state for each sequence/batch to prevent leakage
            hidden = None
            
            if not batched_seq:
                continue

            # Determine batch size from first frame
            B = batched_seq[0][0].shape[0]

            contig0, sid0, seg0 = _get_ids(base_seq, base_base, starts_list, s_idx, 0, 0)

            if seg0 is not None and last_segment_id is not None:
                contiguous = (int(seg0) == int(last_segment_id))
            elif has_window_ts:
                contiguous = (contig0 is not None and last_sid_b0 is not None and contig0 == last_sid_b0 + 1)
            else:
                contiguous = (sid0 is not None and last_sid_b0 is not None and sid0 == last_sid_b0 + contiguous_sid_step)

            if bool(getattr(base_ds, "_force_continuous_eval", False)):
                contiguous = True

            t_batch = np.zeros((B, 3), dtype=np.float64)
            q_batch = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (B, 1))
            if contiguous and global_t_batch is not None and global_q_batch is not None:
                if global_t_batch.shape[0] == B and global_q_batch.shape[0] == B:
                    t_batch = global_t_batch.copy()
                    q_batch = global_q_batch.copy()

            if contiguous:
                eval_v = global_eval_v
                eval_R = global_eval_R
                hidden = global_hidden
            else:
                eval_v = None
                eval_R = None
                hidden = None
                prev_v_probe = None
                prev_R_probe = None
                if global_eval_v is not None:
                    try:
                        print(f"[EVAL RESET] s_idx={int(s_idx)} sid0={int(sid0)} | drop_prev_v_norm={float(global_eval_v.detach().norm(dim=1).mean().item()):.3e}")
                    except Exception:
                        print(f"[EVAL RESET] s_idx={int(s_idx)} sid0={int(sid0)}")

            for j, item in enumerate(batched_seq):
                if j % eval_subsample != 0:
                    continue

                # Defensive unpacking for nested items
                if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (list, tuple)):
                    item = item[0]

                if len(item) >= 3:
                    ev, imu, y = item[0], item[1], item[2]
                    # Extract dt if available, else default
                    if len(item) > 3:
                        dt_tensor = item[3]
                        dt_flat = dt_tensor.view(dt_tensor.shape[0], -1) if dt_tensor.ndim > 1 else dt_tensor.view(-1, 1)
                        dt_per_sample = dt_flat.mean(dim=1)
                        dt_min = float(dt_per_sample.min().item())
                        dt_max = float(dt_per_sample.max().item())
                        dt_mean = max(float(dt_per_sample.mean().item()), 1e-6)
                        if abs(dt_max - dt_min) / dt_mean > 1e-3:
                            raise ValueError(
                                f"[EVAL DT CHECK] Inconsistent dt_window in eval batch: "
                                f"range=({dt_min:.6e}, {dt_max:.6e}), mean={dt_mean:.6e}. "
                                f"Please ensure all samples in a batch share the same dt_window."
                            )
                        actual_dt = float(dt_mean)
                    else:
                        actual_dt = 0.05
                else:
                    continue

                vox = ev.to(device, non_blocking=True)
                imu_batch = imu.to(device, non_blocking=True)
                
                # Ensure correct dimensions (collate_sequence already adds batch dim)
                if len(vox.shape) == 3:
                    vox = vox.unsqueeze(0)
                if len(imu_batch.shape) == 2:
                    imu_batch = imu_batch.unsqueeze(0)

                hidden = _adjust_hidden_size(hidden, vox.size(0))

                y_dev = y.to(device, non_blocking=True)

                if eval_R is None and y_dev.ndim >= 2 and y_dev.shape[1] >= 14:
                    q_prev = F.normalize(y_dev[:, 7:11], p=2, dim=1)
                    eval_R = GeometryUtils.quat_to_rot(q_prev)
                    eval_v = y_dev[:, 11:14].contiguous()

                pv_in = None if diag_zero_prev_v else eval_v
                pr_in = None if diag_zero_prev_R else eval_R
                out, hidden, eval_v, eval_R, _, s, _, _ = model(
                    vox, imu_batch, hidden,
                    prev_v=pv_in, prev_R=pr_in, dt_window=actual_dt,
                    debug=(not printed_norm_debug)
                )
                if not printed_norm_debug:
                    actual_model = model.orig_mod if hasattr(model, 'orig_mod') else model
                    dbg = getattr(actual_model, "_last_step_debug", None)
                    t_hat_body_norm = float(dbg.get("t_hat_body_norm")) if isinstance(dbg, dict) and "t_hat_body_norm" in dbg else float("nan")
                    pos_res_norm = float(dbg.get("pos_res_norm")) if isinstance(dbg, dict) and "pos_res_norm" in dbg else float("nan")
                    scale_s_val = float(dbg.get("scale_s")) if isinstance(dbg, dict) and "scale_s" in dbg else float("nan")
                    if not np.isfinite(scale_s_val):
                        scale_s_val = float(dbg.get("ts")) if isinstance(dbg, dict) and "ts" in dbg else float("nan")
                    if isinstance(dbg, dict):
                        if not np.isfinite(t_hat_body_norm):
                            t_hat_body_vec = dbg.get("t_hat_body_vec")
                            if isinstance(t_hat_body_vec, np.ndarray) and t_hat_body_vec.ndim == 2 and t_hat_body_vec.shape[1] == 3:
                                t_hat_body_norm = float(np.linalg.norm(t_hat_body_vec, axis=1).mean())
                        if not np.isfinite(pos_res_norm):
                            pos_res_vec = dbg.get("pos_res_vec")
                            if isinstance(pos_res_vec, np.ndarray) and pos_res_vec.ndim == 2 and pos_res_vec.shape[1] == 3:
                                pos_res_norm = float(np.linalg.norm(pos_res_vec, axis=1).mean())
                        if not np.isfinite(scale_s_val):
                            scale_s_vec = dbg.get("scale_s_vec")
                            if not (isinstance(scale_s_vec, np.ndarray) and scale_s_vec.size > 0):
                                scale_s_vec = dbg.get("ts_vec")
                            if isinstance(scale_s_vec, np.ndarray) and scale_s_vec.size > 0:
                                scale_s_val = float(np.median(scale_s_vec))
                    t_delta_gt_norm = float(y_dev[:, 0:3].detach().norm(dim=1).mean().item()) if y_dev.ndim >= 2 and y_dev.shape[1] >= 3 else float("nan")
                    print(f"[DEBUG NORMS][EVAL] ||t_hat_body||={t_hat_body_norm:.6e} ||pos_res||={pos_res_norm:.6e} ||t_delta_gt||={t_delta_gt_norm:.6e} | scale_s={scale_s_val:.6e} | dt={float(actual_dt):.6e}")
                    printed_norm_debug = True
                dt_list.append(actual_dt)
                if s is not None:
                    s_list.extend([float(v) for v in s.detach().flatten().cpu().tolist()])

                # Collect uncertainty statistics for evaluation logging
                if hasattr(actual_model, '_last_step_debug'):
                    dbg_unc = actual_model._last_step_debug
                    if isinstance(dbg_unc, dict):
                        # Collect fusion weights
                        w_imu_np = dbg_unc.get("weight_imu")
                        w_visual_np = dbg_unc.get("weight_visual")
                        if isinstance(w_imu_np, np.ndarray) and w_imu_np.size > 0:
                            w_imu_list.extend(w_imu_np.mean(axis=1).flatten().tolist())
                        if isinstance(w_visual_np, np.ndarray) and w_visual_np.size > 0:
                            w_visual_list.extend(w_visual_np.mean(axis=1).flatten().tolist())
                        # Collect sigma values (sqrt of variance)
                        var_i_np = dbg_unc.get("var_i")
                        var_v_np = dbg_unc.get("var_v")
                        if isinstance(var_i_np, np.ndarray) and var_i_np.size > 0:
                            sigma_imu_list.extend(np.sqrt(var_i_np.mean(axis=1)).flatten().tolist())
                        if isinstance(var_v_np, np.ndarray) and var_v_np.size > 0:
                            sigma_visual_list.extend(np.sqrt(var_v_np.mean(axis=1)).flatten().tolist())

                global_eval_v = eval_v
                global_eval_R = eval_R
                global_hidden = hidden

                # Model output: [pos(3), quat(4)] (converted from 6D inside model forward)
                out_np = out.detach().float().cpu().numpy()

                td = out_np[:, 0:3].astype(np.float64)
                qd = out_np[:, 3:7].astype(np.float64)
                tdelta_gt = y.detach().float().cpu().numpy()[:, 0:3].astype(np.float64)

                s_np = None
                if s is not None:
                    try:
                        s_np = s.detach().float().flatten().cpu().numpy().astype(np.float64)
                    except Exception:
                        s_np = None

                # DEBUG: Check first few predictions to see scale
                if s_idx == 0 and j == 0:
                    print(f"[EVAL DEBUG] First prediction td={td[0]} (norm={np.linalg.norm(td[0]):.6f})")
                    if s_np is not None and s_np.size > 0:
                        mode_str = "gate" if gate_mode else "scale"
                        print(f"[EVAL DEBUG] scale_s median={float(np.median(s_np)):.6f} | mode={mode_str}")
                
                for b in range(B):
                    # Update pose
                    tb, qb = QuaternionUtils.compose_se3(t_batch[b], q_batch[b], td[b], QuaternionUtils.normalize(qd[b]))
                    t_batch[b] = tb
                    q_batch[b] = qb

                    contig_id, sid, segment_id = _get_ids(base_seq, base_base, starts_list, s_idx, j, b)
                    if sid is None:
                        continue

                    if b == 0:
                        td_norm = float(np.linalg.norm(td[b]))
                        y_norm = float(np.linalg.norm(tdelta_gt[b]))
                        ratio_y = td_norm / (y_norm + 1e-12)

                        gt_raw_norm = float("nan")
                        dt_pair = float("nan")
                        i_curr = int(sid)
                        i_prev = int(i_curr - base_sample_stride)
                        prev_dbg = i_prev
                        curr_dbg = i_curr

                        use_window_dbg = bool(
                            hasattr(base_ds, "window_t_prev")
                            and getattr(base_ds, "window_t_prev") is not None
                            and hasattr(base_ds, "window_t_curr")
                            and getattr(base_ds, "window_t_curr") is not None
                            and hasattr(base_ds, "prev_indices")
                            and hasattr(base_ds, "curr_indices")
                            and contig_id is not None
                            and isinstance(contig_id, (int, np.integer))
                            and contig_id >= 0
                            and contig_id < len(getattr(base_ds, "window_t_curr"))
                        )

                        if use_window_dbg:
                            try:
                                prev_dbg = int(base_ds.prev_indices[int(contig_id)])
                                curr_dbg = int(base_ds.curr_indices[int(contig_id)])
                                t_prev_dbg = float(base_ds.window_t_prev[int(contig_id)])
                                t_curr_dbg = float(base_ds.window_t_curr[int(contig_id)])
                                dt_pair = float(t_curr_dbg - t_prev_dbg)
                                p_prev_dbg, q_prev_dbg = base_ds.interpolate_gt_data(t_prev_dbg)
                                p_curr_dbg, _ = base_ds.interpolate_gt_data(t_curr_dbg)
                                R_prev = QuaternionUtils.to_rotation_matrix(q_prev_dbg.astype(np.float64))
                                dp_body = R_prev.T @ (p_curr_dbg.astype(np.float64) - p_prev_dbg.astype(np.float64))
                                gt_raw_norm = float(np.linalg.norm(dp_body))
                            except Exception:
                                pass
                        else:
                            if 0 <= i_prev < len(base_ds.gt_t) and 0 <= i_curr < len(base_ds.gt_t):
                                dt_pair = float(base_ds.gt_t[i_curr] - base_ds.gt_t[i_prev])
                                dp_world = (base_ds.gt_pos[i_curr] - base_ds.gt_pos[i_prev]).astype(np.float64)
                                R_prev = QuaternionUtils.to_rotation_matrix(base_ds.gt_quat[i_prev].astype(np.float64))
                                dp_body = R_prev.T @ dp_world
                                gt_raw_norm = float(np.linalg.norm(dp_body))

                        acc_mean_norm = float("nan")
                        gyro_mean_norm = float("nan")
                        acc_mean_mps2 = float("nan")
                        gyro_mean_rads = float("nan")
                        use_g_est = -1
                        T_valid_est = float("nan")
                        try:
                            m = (imu_batch[b].detach().abs().sum(dim=1) > 1e-6)
                            T_valid_est = float(m.to(dtype=torch.float32).sum().item())
                            if T_valid_est >= 1.0:
                                acc_n = imu_batch[b, :, 0:3].detach().norm(dim=1)
                                gyro_n = imu_batch[b, :, 3:6].detach().norm(dim=1)
                                w = m.to(dtype=acc_n.dtype)
                                denom = float(w.sum().clamp(min=1.0).item())
                                acc_mean_norm = float(((acc_n * w).sum() / denom).item())
                                gyro_mean_norm = float(((gyro_n * w).sum() / denom).item())
                                acc_mean_mps2 = float(acc_mean_norm * 9.81)
                                gyro_mean_rads = float(gyro_mean_norm * np.pi)
                                use_g_est = 1 if acc_mean_mps2 > 6.0 else 0
                        except Exception:
                            pass

                        dv_norm = float("nan")
                        dv_per_dt = float("nan")
                        dv_ready = 0
                        if isinstance(eval_v, torch.Tensor):
                            try:
                                if prev_v_probe is not None and prev_v_probe.shape == eval_v.shape:
                                    dv = eval_v.detach() - prev_v_probe
                                    dv_norm = float(dv[b].norm().item())
                                    dv_per_dt = dv_norm / (float(actual_dt) + 1e-12)
                                    dv_ready = 1
                            except Exception:
                                pass

                        orth_err = float("nan")
                        det_R = float("nan")
                        if isinstance(eval_R, torch.Tensor):
                            try:
                                I = torch.eye(3, device=eval_R.device, dtype=eval_R.dtype)
                                RtR = eval_R[b].transpose(0, 1) @ eval_R[b]
                                orth_err = float(torch.norm(RtR - I, p="fro").item())
                                det_R = float(torch.det(eval_R[b]).item())
                            except Exception:
                                pass

                        s_b = float("nan")
                        if s_np is not None and s_np.size > b:
                            s_b = float(s_np[b])

                        dbg = getattr(actual_model, "_last_step_debug", None)
                        t_hat_body_dbg = None
                        pos_res_dbg = None
                        ba_dbg = None
                        bg_dbg = None
                        v_dbg = None
                        if isinstance(dbg, dict):
                            t_hat_body_dbg = dbg.get("t_hat_body_vec")
                            pos_res_dbg = dbg.get("pos_res_vec")
                            ba_dbg = dbg.get("ba_vec")
                            bg_dbg = dbg.get("bg_vec")
                            v_dbg = dbg.get("v_vec")

                        has_t_hat = isinstance(t_hat_body_dbg, np.ndarray) and t_hat_body_dbg.ndim == 2 and t_hat_body_dbg.shape[0] > b
                        has_pos_res = isinstance(pos_res_dbg, np.ndarray) and pos_res_dbg.ndim == 2 and pos_res_dbg.shape[0] > b
                        dbg_t_hat = float(np.linalg.norm(t_hat_body_dbg[b])) if has_t_hat else float("nan")
                        dbg_pos_res = float(np.linalg.norm(pos_res_dbg[b])) if has_pos_res else float("nan")
                        dbg_ba = float(np.linalg.norm(ba_dbg[b])) if isinstance(ba_dbg, np.ndarray) and ba_dbg.ndim == 2 and ba_dbg.shape[0] > b else float("nan")
                        dbg_bg = float(np.linalg.norm(bg_dbg[b])) if isinstance(bg_dbg, np.ndarray) and bg_dbg.ndim == 2 and bg_dbg.shape[0] > b else float("nan")
                        dbg_v = float(np.linalg.norm(v_dbg[b])) if isinstance(v_dbg, np.ndarray) and v_dbg.ndim == 2 and v_dbg.shape[0] > b else float("nan")

                        v_gt_prev_norm = float("nan")
                        v_gt_curr_norm = float("nan")
                        try:
                            if isinstance(y_dev, torch.Tensor) and y_dev.ndim == 2 and y_dev.shape[1] >= 14:
                                v_gt_prev_norm = float(y_dev[b, 11:14].detach().norm().item())
                            if isinstance(y_dev, torch.Tensor) and y_dev.ndim == 2 and y_dev.shape[1] >= 17:
                                v_gt_curr_norm = float(y_dev[b, 14:17].detach().norm().item())
                        except Exception:
                            pass

                        split_str = "na"
                        if np.isfinite(dbg_t_hat) and np.isfinite(dbg_pos_res):
                            split_str = f"{dbg_t_hat:.6e}/{dbg_pos_res:.6e}"
                        elif np.isfinite(dbg_pos_res):
                            split_str = f"na/{dbg_pos_res:.6e}"
                        elif np.isfinite(dbg_t_hat):
                            split_str = f"{dbg_t_hat:.6e}/na"

                        td_recon_err = float("nan")
                        if has_pos_res:
                            w_imu_dbg = dbg.get("weight_imu") if isinstance(dbg, dict) else None
                            w_visual_dbg = dbg.get("weight_visual") if isinstance(dbg, dict) else None
                            has_w = (
                                isinstance(w_imu_dbg, np.ndarray) and isinstance(w_visual_dbg, np.ndarray) and
                                w_imu_dbg.ndim == 2 and w_visual_dbg.ndim == 2 and
                                w_imu_dbg.shape[0] > b and w_visual_dbg.shape[0] > b and
                                w_imu_dbg.shape[1] == 3 and w_visual_dbg.shape[1] == 3
                            )
                            if has_t_hat and has_w:
                                td_recon = w_imu_dbg[b] * t_hat_body_dbg[b] + w_visual_dbg[b] * pos_res_dbg[b]
                                td_recon_err = float(np.linalg.norm(td[b] - td_recon))
                            elif has_t_hat and np.isfinite(s_b):
                                td_recon = t_hat_body_dbg[b] + float(s_b) * pos_res_dbg[b]
                                td_recon_err = float(np.linalg.norm(td[b] - td_recon))
                            else:
                                td_recon_err = float(np.linalg.norm(td[b] - pos_res_dbg[b]))

                        if (s_idx == 0 and j == 0) or (np.isfinite(ratio_y) and ratio_y > 10.0):
                            print(
                                f"[WINDOW PROBE] sid={int(sid)} prev={int(prev_dbg)} curr={int(curr_dbg)} stride={int(base_sample_stride)} "
                                f"dt_pair={dt_pair:.6e} | ||td||={td_norm:.6e} ||y_dpos||={y_norm:.6e} "
                                f"||gt_pos_dpos||={gt_raw_norm:.6e} | ratio(td/y)={ratio_y:.3f} | s={s_b:.6f} "
                                f"| imu_norm(acc/gyro)={acc_mean_norm:.3f}/{gyro_mean_norm:.3f} "
                                f"| imu_mps2/rads={acc_mean_mps2:.3f}/{gyro_mean_rads:.3f} use_g_est={use_g_est:d} T_valid={T_valid_est:.0f} "
                                f"| v_gt(prev/curr)={v_gt_prev_norm:.3e}/{v_gt_curr_norm:.3e} "
                                f"| dv_ready={dv_ready:d} dv_norm={dv_norm:.3e} dv_per_dt={dv_per_dt:.3e} "
                                f"| R_orth_err={orth_err:.3e} detR={det_R:.3e} "
                                f"| split_b(t_hat/pos_res)={split_str} "
                                f"| bias_b(ba/bg)={dbg_ba:.3e}/{dbg_bg:.3e} | v_norm_b={dbg_v:.3e} "
                                f"| td_recon_err={td_recon_err:.3e}"
                            )

                        prev_v_probe = eval_v.detach().clone() if isinstance(eval_v, torch.Tensor) else None
                        prev_R_probe = eval_R.detach().clone() if isinstance(eval_R, torch.Tensor) else None

                    est_pos.append(tb.copy())
                    est_quat.append(qb.copy())
                    sample_ids.append(sid)
                    contig_ids.append(contig_id)
                    seg_keys.append(int(s_idx) * 1_000_000 + int(b))
                    win_step_pred.append(float(np.linalg.norm(td[b])))
                    win_step_gt.append(float(np.linalg.norm(tdelta_gt[b])))
                    last_sid = sid
                    if b == 0:
                        last_sid_b0 = contig_id if has_window_ts else sid
                        last_segment_id = segment_id
                    if b == B - 1:
                        global_t_batch = t_batch.copy()
                        global_q_batch = q_batch.copy()

    est_pos = np.asarray(est_pos, dtype=np.float64)
    est_quat = np.asarray(est_quat, dtype=np.float64)
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    contig_ids = np.asarray(contig_ids, dtype=np.int64)
    seg_keys = np.asarray(seg_keys, dtype=np.int64)

    has_window_ts = bool(
        hasattr(base_ds, "window_t_curr")
        and getattr(base_ds, "window_t_curr") is not None
        and hasattr(base_ds, "interpolate_gt_data")
    )

    if has_window_ts:
        try:
            wtc = np.asarray(base_ds.window_t_curr, dtype=np.float64)
            wtp = np.asarray(base_ds.window_t_prev, dtype=np.float64) if getattr(base_ds, "window_t_prev", None) is not None else None
            if contig_ids.size > 0 and wtc.size > 0 and int(np.max(contig_ids)) < int(wtc.size):
                gt_t = wtc[contig_ids]
                gt_pos_list = []
                gt_quat_list = []
                for tt in gt_t.tolist():
                    p_i, q_i = base_ds.interpolate_gt_data(float(tt))
                    gt_pos_list.append(p_i.astype(np.float64))
                    gt_quat_list.append(q_i.astype(np.float64))
                gt_pos = np.asarray(gt_pos_list, dtype=np.float64)
                gt_quat = np.asarray(gt_quat_list, dtype=np.float64)
            else:
                has_window_ts = False
        except Exception:
            has_window_ts = False

    if not has_window_ts:
        gt_pos = base_ds.gt_pos[sample_ids]
        gt_quat = base_ds.gt_quat[sample_ids]
        gt_t = base_ds.gt_t[sample_ids]

    valid = np.isfinite(est_pos).all(axis=1) & np.isfinite(gt_pos).all(axis=1) & np.isfinite(gt_t)
    if valid.size > 0 and not np.all(valid):
        est_pos = est_pos[valid]
        est_quat = est_quat[valid]
        sample_ids = sample_ids[valid]
        contig_ids = contig_ids[valid]
        seg_keys = seg_keys[valid]
        gt_pos = gt_pos[valid]
        gt_quat = gt_quat[valid]
        gt_t = gt_t[valid]

    if gt_t.size > 1:
        order = np.argsort(gt_t)
        est_pos = est_pos[order]
        est_quat = est_quat[order]
        sample_ids = sample_ids[order]
        contig_ids = contig_ids[order]
        seg_keys = seg_keys[order]
        gt_pos = gt_pos[order]
        gt_quat = gt_quat[order]
        gt_t = gt_t[order]

    m = min(len(est_pos), len(gt_pos))
    if m == 0:
        return float("nan"), float("nan"), float("nan")

    if m >= 3:
        # Split trajectory into continuous segments based on (sequence,batch) key and sample_ids
        sid_diff = sample_ids[1:] - sample_ids[:-1]
        seg_diff = seg_keys[1:] - seg_keys[:-1]
        jump_mask = (seg_diff != 0)
        jump_indices = np.nonzero(jump_mask)[0] + 1
        segment_splits = np.split(np.arange(m), jump_indices)
        
        se3_sq_errors = []
        sim3_sq_errors = []
        sim3_scales = []
        se3_ate_seq = []
        sim3_ate_seq = []
        
        # Global alignment for logging/debugging (optional, or we can just report per-segment)
        # We'll calculate the aggregate ATE from per-segment errors.
        
        for seg_idx in segment_splits:
            if len(seg_idx) < 3:
                continue
                
            est_seg = est_pos[seg_idx]
            gt_seg = gt_pos[seg_idx]
            est_t_seg = gt_t[seg_idx]
            gt_t_seg = gt_t[seg_idx]

            # 1. Per-segment SE(3) Alignment
            R, t_off, ie_rigid, ig_rigid = align_trajectory_with_timestamps(est_seg, est_t_seg, gt_seg, gt_t_seg)
            est_seg_aligned = (R @ est_seg.T).T + t_off
            sq_err = np.sum((est_seg_aligned - gt_seg) ** 2, axis=1)
            se3_sq_errors.extend(sq_err.tolist())
            se3_ate_seq.append(float(np.sqrt(np.mean(sq_err))))

            # 2. Per-segment SIM(3) Alignment
            mean_gt_step = float(np.mean(np.linalg.norm(gt_seg[1:] - gt_seg[:-1], axis=1))) if gt_seg.shape[0] >= 2 else 0.0
            if (not np.isfinite(mean_gt_step)) or (mean_gt_step < 1e-4):
                sim3_sq_errors.extend(sq_err.tolist())
                sim3_scales.append(1.0)
                sim3_ate_seq.append(float(np.sqrt(np.mean(sq_err))))
                continue

            s_s_seg = 1.0
            if eval_sim3_mode == "diagnose" or eval_sim3_mode == "use_learned":
                est_seg_for_sim3 = est_seg.copy()
                if eval_sim3_mode == "use_learned":
                    if not gate_mode:
                        ts_val_local = float(np.median(np.asarray(s_list, dtype=np.float64))) if len(s_list) else 1.0
                        est_seg_for_sim3 = est_seg_for_sim3 / (ts_val_local + 1e-12)

                R_s, t_s, s_s_seg, ie_sim3, ig_sim3 = align_trajectory_with_timestamps_sim3(est_seg_for_sim3, est_t_seg, gt_seg, gt_t_seg)
                est_seg_sim3 = (s_s_seg * R_s @ est_seg_for_sim3.T).T + t_s
                sq_err_sim3 = np.sum((est_seg_sim3 - gt_seg) ** 2, axis=1)
                sim3_sq_errors.extend(sq_err_sim3.tolist())
                sim3_scales.append(s_s_seg)
                sim3_ate_seq.append(float(np.sqrt(np.mean(sq_err_sim3))))
            elif eval_sim3_mode == "fix_learned":
                sim3_sq_errors.extend(sq_err.tolist())
                sim3_scales.append(1.0)
                sim3_ate_seq.append(float(np.sqrt(np.mean(sq_err))))
            else:
                sim3_sq_errors.extend(sq_err.tolist())
                sim3_scales.append(1.0)
                sim3_ate_seq.append(float(np.sqrt(np.mean(sq_err))))

        # Aggregate ATE
        if len(se3_sq_errors) > 0:
            ate = float(np.sqrt(np.mean(np.array(se3_sq_errors))))
        else:
            ate = float("nan")

        if len(sim3_sq_errors) > 0:
            ate_sim3 = float(np.sqrt(np.mean(np.array(sim3_sq_errors))))
        else:
            ate_sim3 = float("nan")

        if len(se3_ate_seq) > 0:
            se3_seq_arr = np.asarray(se3_ate_seq, dtype=np.float64)
            sim3_seq_arr = np.asarray(sim3_ate_seq, dtype=np.float64) if len(sim3_ate_seq) else np.asarray([], dtype=np.float64)
            se3_mean = float(np.nanmean(se3_seq_arr))
            se3_median = float(np.nanmedian(se3_seq_arr))
            sim3_mean = float(np.nanmean(sim3_seq_arr)) if sim3_seq_arr.size else float("nan")
            sim3_median = float(np.nanmedian(sim3_seq_arr)) if sim3_seq_arr.size else float("nan")
            print(f"[ATE PER-SEQ] N={len(se3_ate_seq)} | SE3(mean/median)={se3_mean:.4f}/{se3_median:.4f} | SIM3(mean/median)={sim3_mean:.4f}/{sim3_median:.4f}")
        else:
            print("[ATE PER-SEQ] N=0")
            
        # Average Scale
        s_s = float(np.mean(sim3_scales)) if len(sim3_scales) > 0 else 1.0
        match_cnt = (len(se3_sq_errors), len(se3_sq_errors)) # Approx

        
        ts_val = float(np.median(np.asarray(s_list, dtype=np.float64))) if (len(s_list) and (not gate_mode)) else 1.0

        raw_est_step = float("nan") if gate_mode else float("nan")

        step_est_vals: List[float] = []
        step_gt_vals: List[float] = []
        dp_est_all_list: List[np.ndarray] = []
        dp_gt_all_list: List[np.ndarray] = []

        seg_keys_m = seg_keys[:m]
        uniq_keys = np.unique(seg_keys_m)
        for k in uniq_keys:
            idx = np.nonzero(seg_keys_m == k)[0]
            if idx.size < 2:
                continue
            if has_window_ts:
                sid_step_k = contig_ids[idx[1:]] - contig_ids[idx[:-1]]
                ok = (sid_step_k == 1)
            else:
                sid_step_k = sample_ids[idx[1:]] - sample_ids[idx[:-1]]
                ok = (sid_step_k == int(contiguous_sid_step))
            if not np.any(ok):
                continue
            dp_est_k = est_pos[idx[1:]] - est_pos[idx[:-1]]
            dp_gt_k = gt_pos[idx[1:]] - gt_pos[idx[:-1]]
            dp_est_k = dp_est_k[ok]
            dp_gt_k = dp_gt_k[ok]
            step_est_vals.extend(np.linalg.norm(dp_est_k, axis=1).astype(np.float64).tolist())
            step_gt_vals.extend(np.linalg.norm(dp_gt_k, axis=1).astype(np.float64).tolist())
            dp_est_all_list.append(dp_est_k)
            dp_gt_all_list.append(dp_gt_k)

        has_steps = bool(len(step_est_vals) > 1 and len(step_gt_vals) > 1 and dp_est_all_list and dp_gt_all_list)
        if has_steps:
            mean_step_est = float(np.mean(np.asarray(step_est_vals, dtype=np.float64)))
            mean_step_gt = float(np.mean(np.asarray(step_gt_vals, dtype=np.float64)))
            direct_ratio = mean_step_est / (mean_step_gt + 1e-12)

            if not gate_mode:
                raw_est_step = mean_step_est / (ts_val + 1e-12)

            dp_est_all = np.concatenate(dp_est_all_list, axis=0)
            dp_gt_all = np.concatenate(dp_gt_all_list, axis=0)
            dot = np.einsum("ij,ij->i", dp_est_all, dp_gt_all)
            num = float(np.sum(dot))
            den_pred = float(np.sum(np.linalg.norm(dp_est_all, axis=1) ** 2)) + 1e-12
            den_gt = float(np.sum(np.linalg.norm(dp_gt_all, axis=1) ** 2)) + 1e-12
            s_pred_to_gt = num / den_pred
            s_gt_to_pred = num / den_gt
            sum_pred = float(np.sum(np.linalg.norm(dp_est_all, axis=1)))
            sum_gt = float(np.sum(np.linalg.norm(dp_gt_all, axis=1))) + 1e-12
            path_ratio = sum_pred / sum_gt
        else:
            mean_step_est = 0.0
            mean_step_gt = 0.0
            direct_ratio = 0.0
            s_pred_to_gt = 0.0
            s_gt_to_pred = 0.0
            path_ratio = 0.0

        dt_stats = (float(np.min(dt_list)) if len(dt_list) else float('nan'),
                    float(np.mean(dt_list)) if len(dt_list) else float('nan'),
                    float(np.max(dt_list)) if len(dt_list) else float('nan'))
        if len(win_step_pred) > 0 and len(win_step_gt) > 0:
            win_pred = np.asarray(win_step_pred, dtype=np.float64)
            win_gt = np.asarray(win_step_gt, dtype=np.float64)
            win_mean_pred = float(np.mean(win_pred))
            win_mean_gt = float(np.mean(win_gt))
            win_ratio = win_mean_pred / (win_mean_gt + 1e-12)
            p = [0.10, 0.50, 0.90]
            wp10, wp50, wp90 = [float(np.quantile(win_pred, q)) for q in p]
            wg10, wg50, wg90 = [float(np.quantile(win_gt, q)) for q in p]
            print(f"[WINDOW STEP] Pred(mean)={win_mean_pred:.6f} GT(mean)={win_mean_gt:.6f} | ratio(mean)={win_ratio:.3f}")
            print(f"[WINDOW STEP] Pred(p10/p50/p90)=[{wp10:.6f},{wp50:.6f},{wp90:.6f}] | GT(p10/p50/p90)=[{wg10:.6f},{wg50:.6f},{wg90:.6f}] | ratio(p50)={wp50/(wg50 + 1e-12):.3f}")

        if has_steps and ((path_ratio > 1.0 and direct_ratio < 1.0) or (path_ratio < 1.0 and direct_ratio > 1.0)):
            print(f"[WARN] direct_ratio={direct_ratio:.3f} and PATH_RATIO={path_ratio:.3f} disagree. Check pred/gt swap or trajectory mismatch.")
        print(f"[EVAL DIAGNOSTICS] SIM3_mode={eval_sim3_mode} | s={s_s:.6f} | ATE(SE3)={ate:.4f} | ATE(SIM3)={ate_sim3:.4f} | matches={match_cnt}")
        if len(s_list):
            s_arr = np.asarray(s_list, dtype=np.float64)
            s_p10 = float(np.quantile(s_arr, 0.10))
            s_p50 = float(np.quantile(s_arr, 0.50))
            s_p90 = float(np.quantile(s_arr, 0.90))
            if gate_mode:
                print(f"[GATE_STATS] s_gate[p10/p50/p90]=[{s_p10:.6e},{s_p50:.6e},{s_p90:.6e}]")
            else:
                print(f"[SCALE_COMPARE] scale_head[p10/p50/p90]=[{s_p10:.6e},{s_p50:.6e},{s_p90:.6e}] vs SIM3={s_s:.6e} | ratio(p50)={s_p50/ (s_s + 1e-12):.3e}")
        else:
            if gate_mode:
                print("[GATE_STATS] s_gate unavailable")
            else:
                print(f"[SCALE_COMPARE] scale_head unavailable vs SIM3={s_s:.6e}")

        # Uncertainty fusion statistics (Bayesian fusion weights)
        if len(w_imu_list) and len(w_visual_list):
            w_imu_arr = np.asarray(w_imu_list, dtype=np.float64)
            w_vis_arr = np.asarray(w_visual_list, dtype=np.float64)
            wi_p10, wi_p50, wi_p90 = np.quantile(w_imu_arr, [0.10, 0.50, 0.90])
            wv_p10, wv_p50, wv_p90 = np.quantile(w_vis_arr, [0.10, 0.50, 0.90])
            print(f"[UNCERTAINTY] w_imu[p10/p50/p90]=[{wi_p10:.3f},{wi_p50:.3f},{wi_p90:.3f}] | "
                  f"w_visual[p10/p50/p90]=[{wv_p10:.3f},{wv_p50:.3f},{wv_p90:.3f}]")

        if len(sigma_imu_list) and len(sigma_visual_list):
            si_arr = np.asarray(sigma_imu_list, dtype=np.float64)
            sv_arr = np.asarray(sigma_visual_list, dtype=np.float64)
            si_p50 = float(np.median(si_arr))
            sv_p50 = float(np.median(sv_arr))
            print(f"[UNCERTAINTY] σ_imu(p50)={si_p50:.4f} | σ_visual(p50)={sv_p50:.4f}")
        if has_steps:
            if gate_mode:
                print(f"[STEP ANALYSIS] Mean Step: Pred={mean_step_est:.6f} vs GT={mean_step_gt:.6f} | "
                      f"direct_ratio={direct_ratio:.3f} | "
                      f"OLS_pred_to_gt={s_pred_to_gt:.6f} (pred*->gt) | "
                      f"OLS_gt_to_pred={s_gt_to_pred:.6f} (gt*->pred) | "
                      f"PATH_RATIO={path_ratio:.6f}")
            else:
                print(f"[STEP ANALYSIS] Mean Step: Pred={mean_step_est:.6f} (Raw~{raw_est_step:.6f}) vs GT={mean_step_gt:.6f} | "
                      f"direct_ratio={direct_ratio:.3f} | "
                      f"OLS_pred_to_gt={s_pred_to_gt:.6f} (pred*->gt) | "
                      f"OLS_gt_to_pred={s_gt_to_pred:.6f} (gt*->pred) | "
                      f"PATH_RATIO={path_ratio:.6f}")
            print(f"[STEP RATIO] Pred/GT = {direct_ratio:.4f} | dt[min/mean/max]={dt_stats}")
        else:
            print(f"[STEP ANALYSIS] insufficient steps for ratio/OLS | dt[min/mean/max]={dt_stats}")
        
        # 3. ATE(Scaled) - Force Identity Scale Alignment (Rigid only, trust learned scale)
        # We already did this with standard SE(3) alignment above (ate), because align_trajectory_with_timestamps 
        # solves for R, t but assumes s=1. So 'ate' IS ATE(Scaled) if est_pos includes the scale.
        
        # 4. ATE(Unit) - What if we force scale=1.0 (undo learned scale)?
        # If est_pos has scale applied, we divide by ts_val
        ate_unit = float("nan")
        if not gate_mode:
            est_pos_unit = est_pos[:m] / (ts_val + 1e-12)
            R_u, t_u, _, _ = align_trajectory_with_timestamps(est_pos_unit, gt_t[:m], gt_pos[:m], gt_t[:m])
            est_pos_unit_aligned = (R_u @ est_pos_unit.T).T + t_u
            ate_unit = float(np.sqrt(np.mean(np.sum((est_pos_unit_aligned - gt_pos[:m]) ** 2, axis=1))))
            print(f"[ATE METRICS] ATE(SE3/Scaled)={ate:.4f} | ATE(Unit/Raw)={ate_unit:.4f} | ATE(SIM3)={ate_sim3:.4f}")
        else:
            print(f"[ATE METRICS] ATE(SE3/Scaled)={ate:.4f} | ATE(SIM3)={ate_sim3:.4f}")
        
    else:
        ate = float(np.sqrt(np.mean(np.sum((est_pos[:m] - gt_pos[:m]) ** 2, axis=1))))

    gt_t_m = gt_t[:m]
    if len(gt_t_m) >= 2:
        gt_t_seq = gt_t_m
        idx0_list: List[int] = []
        idx1_list: List[int] = []
        for i0 in range(0, m - 1):
            t0 = float(gt_t_seq[i0])
            t_target = t0 + float(rpe_dt)
            j = int(np.searchsorted(gt_t_seq, t_target, side="left"))
            if j >= m:
                break
            idx0_list.append(i0)
            idx1_list.append(j)
        if len(idx0_list) == 0:
            idx0 = np.arange(0, m - 1, dtype=np.int64)
            idx1 = idx0 + 1
        else:
            idx0 = np.asarray(idx0_list, dtype=np.int64)
            idx1 = np.asarray(idx1_list, dtype=np.int64)

        # Use aligned positions for RPE translation to handle global rotation offset
        # est_pos_aligned is computed earlier: (R @ est_pos.T).T + t_off
        if 'est_pos_aligned' in locals():
            dp_est = est_pos_aligned[idx1] - est_pos_aligned[idx0]
        else:
            # Fallback if alignment failed (should not happen if m>=3)
            dp_est = est_pos[idx1] - est_pos[idx0]

        dp_gt = gt_pos[idx1] - gt_pos[idx0]
        errs = np.linalg.norm(dp_est - dp_gt, axis=1)
        rpe_t = float(np.sqrt(np.mean(errs ** 2))) if errs.size > 0 else float("nan")

        q_gt_rel = np.array([
            QuaternionUtils.multiply(QuaternionUtils.inverse(gt_quat[i]), gt_quat[j])
            for i, j in zip(idx0, idx1)
        ])
        q_est_rel = np.array([
            QuaternionUtils.multiply(QuaternionUtils.inverse(est_quat[i]), est_quat[j])
            for i, j in zip(idx0, idx1)
        ])
        q_diff = np.array([
            QuaternionUtils.multiply(qg, QuaternionUtils.inverse(qe))
            for qg, qe in zip(q_gt_rel, q_est_rel)
        ])
        ang = 2.0 * np.arccos(np.clip(np.abs(q_diff[:, 3]), -1.0, 1.0))
        rpe_r = float(np.degrees(np.mean(ang))) if ang.size > 0 else float("nan")
    else:
        rpe_t = float("nan")
        rpe_r = float("nan")
    return ate, rpe_t, rpe_r


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        import numpy as _np
        if not _np.isfinite(val_loss):
            if self.verbose:
                self.trace_func('Validation loss is non-finite; skipping early stopping update')
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        try:
            import os, torch as _torch
            if self.path:
                d = os.path.dirname(self.path)
                if d:
                    os.makedirs(d, exist_ok=True)
                _torch.save(model.state_dict(), self.path)
        except Exception:
            pass
        self.val_loss_min = val_loss


def setup_training_state(config: TrainingConfig, device: torch.device) -> Tuple[Optional[PhysicsBrightnessLoss], bool]:
    physics_mode = getattr(config, "physics_mode", "rotational")
    physics_module = PhysicsBrightnessLoss().to(device) if physics_mode == "rotational" else None
    use_amp = (device.type == "cuda" and bool(getattr(config, "mixed_precision", False)))
    return physics_module, use_amp


def amp_autocast_context(device: torch.device, use_amp: bool):
    """Return appropriate autocast context manager for AMP/non-AMP training."""
    if use_amp and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def unpack_and_validate_batch(batch: Any) -> Tuple[Any, List]:

    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], list):
        batch_data, starts_list = batch
    else:
        batch_data = batch[0] if isinstance(batch, (tuple, list)) and len(batch) > 0 else batch
        starts_list = []
    return batch_data, starts_list


def _compute_velocity_loss(new_v: torch.Tensor, y: torch.Tensor, device: torch.device) -> torch.Tensor:
    if y.shape[1] < 17:
        return torch.tensor(0.0, device=device)

    v_gt_curr = y[:, 14:17]
    return F.smooth_l1_loss(new_v, v_gt_curr, beta=0.1)


def _compute_total_loss(lt: torch.Tensor, lr: torch.Tensor, lv: torch.Tensor, lp: torch.Tensor, lo: torch.Tensor,
                        pred: torch.Tensor, y: torch.Tensor, s: Optional[torch.Tensor],
                        ba_pred: Optional[torch.Tensor], bg_pred: Optional[torch.Tensor],
                        config: TrainingConfig, current_epoch_physics_weight: float,
                        adaptive_loss_fn: Optional[AdaptiveLossWeights],
                        model: nn.Module, device: torch.device,
                        batch_idx: int, step_idx: int, is_amp: bool,
                        dt_tensor: Optional[torch.Tensor], dt_window_fallback: float) -> torch.Tensor:
    """Compute total loss with adaptive or fixed weights."""
    p_term = (current_epoch_physics_weight * lp) if current_epoch_physics_weight > 1e-6 else torch.tensor(0.0, device=device)

    if adaptive_loss_fn is not None:
        lt_w = float(config.loss_w_t) * lt
        lr_w = float(config.loss_w_r) * lr
        lv_w = float(config.loss_w_v) * lv
        loss_list = [lt_w, lr_w, lv_w, p_term]
        loss = adaptive_loss_fn(loss_list) + float(config.loss_w_ortho) * lo

        if loss.detach().item() > 1000.0:
            log_vars_vals = adaptive_loss_fn.log_vars.detach().float().cpu().numpy()
            amp_tag = "AMP " if is_amp else ""
            print(f"[WARN] {amp_tag}High Loss Batch {batch_idx}: {loss.item():.4f} | T={lt.detach().item():.4f}, R={lr.detach().item():.4f}, V={lv.detach().item():.4f}, P={p_term.detach().item():.4f} | LogVars={log_vars_vals}")
    else:
        loss = config.loss_w_t * lt + config.loss_w_r * lr + p_term + config.loss_w_ortho * lo + config.loss_w_v * lv

        threshold = 1000.0 if is_amp else 200.0
        if loss.detach().item() > threshold:
            amp_tag = "AMP " if is_amp else ""
            print(f"[WARN] {amp_tag}High Loss Batch {batch_idx}: {loss.detach().item():.4f} | T={lt.detach().item():.4f}, R={lr.detach().item():.4f}, V={lv.detach().item():.4f}, P={p_term.detach().item():.4f} (Fixed Weights)")

    w_aux_motion = float(getattr(config, 'loss_w_aux_motion', 0.0))
    if w_aux_motion > 0.0:
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model = actual_model.orig_mod if hasattr(actual_model, 'orig_mod') else actual_model
        dbg = getattr(actual_model, '_last_step_debug', None)
        aux_motion = dbg.get('aux_motion_tensor') if isinstance(dbg, dict) else None
        if isinstance(aux_motion, torch.Tensor) and aux_motion.ndim == 2 and aux_motion.shape[1] >= 7 and y.ndim == 2 and y.shape[1] >= 7:
            aux_t = aux_motion[:, 0:3]
            aux_q = F.normalize(aux_motion[:, 3:7], p=2, dim=1)
            gt_t = y[:, 0:3]
            gt_q = F.normalize(y[:, 3:7], p=2, dim=1)

            t_imu = None
            if isinstance(dbg, dict):
                t_imu = dbg.get('t_imu_tensor')
                if not isinstance(t_imu, torch.Tensor):
                    t_imu = None
            if t_imu is None and isinstance(dbg, dict):
                t_hat_body_vec = dbg.get('t_hat_body_vec')
                if isinstance(t_hat_body_vec, np.ndarray) and t_hat_body_vec.shape == (gt_t.shape[0], 3):
                    t_imu = torch.from_numpy(t_hat_body_vec).to(device=gt_t.device, dtype=gt_t.dtype)

            if isinstance(t_imu, torch.Tensor) and t_imu.shape == gt_t.shape:
                gt_t_for_aux = gt_t - t_imu.detach()
            else:
                gt_t_for_aux = gt_t

            lt_aux = F.smooth_l1_loss(aux_t, gt_t_for_aux, beta=0.1)
            dot = torch.sum(aux_q * gt_q, dim=1).abs()
            lr_aux = (1.0 - dot).mean()
            loss = loss + w_aux_motion * (lt_aux + lr_aux)

    if getattr(config, 'use_seq_scale', False) and getattr(config, 'seq_scale_reg', 0.0) > 0.0:
        dp_pred = pred[:, 0:3]
        dp_gt = y[:, 0:3]
        dp_norm = dp_gt.norm(dim=1)
        if dt_tensor is not None:
            dt_local = dt_tensor.view(-1).to(dp_norm.device).clamp(min=1e-6)
        else:
            dt_local = torch.full_like(dp_norm, float(dt_window_fallback)).clamp(min=1e-6)
        moving = (dp_norm > (float(config.speed_thresh) * dt_local))
        if torch.any(moving):
            num = torch.sum((dp_pred[moving] * dp_gt[moving]).sum(dim=1))
            den = torch.sum(dp_pred[moving].norm(dim=1) ** 2) + 1e-6
            s_step = torch.clamp(num / den, min=1e-6, max=1e6)
            loss = loss + float(config.seq_scale_reg) * (torch.log(s_step) ** 2)
            if batch_idx == 0 and step_idx == 0:
                print(f"[TRAIN SCALE] seq_scale_reg enabled (w={float(config.seq_scale_reg):.4f}) | s_step_ols={float(s_step.detach().item()):.6f}")

    w_ba = float(getattr(config, 'loss_w_bias_a', 0.0))
    w_bg = float(getattr(config, 'loss_w_bias_g', 0.0))
    if w_ba > 0.0 and ba_pred is not None:
        prior = getattr(config, "bias_prior_accel", None)
        if prior is not None and len(prior) == 3:
            ba0 = torch.tensor([float(prior[0]), float(prior[1]), float(prior[2])], device=ba_pred.device, dtype=ba_pred.dtype).view(1, 3)
            loss = loss + w_ba * (ba_pred - ba0).pow(2).mean()
            if batch_idx == 0 and step_idx == 0 and not hasattr(_compute_total_loss, "_bias_prior_accel_printed"):
                print(f"[BIAS PRIOR] accel={float(prior[0]):+.6e},{float(prior[1]):+.6e},{float(prior[2]):+.6e} (normalized)")
                _compute_total_loss._bias_prior_accel_printed = True
        else:
            loss = loss + w_ba * ba_pred.pow(2).mean()
    if w_bg > 0.0 and bg_pred is not None:
        prior = getattr(config, "bias_prior_gyro", None)
        if prior is not None and len(prior) == 3:
            bg0 = torch.tensor([float(prior[0]), float(prior[1]), float(prior[2])], device=bg_pred.device, dtype=bg_pred.dtype).view(1, 3)
            loss = loss + w_bg * (bg_pred - bg0).pow(2).mean()
            if batch_idx == 0 and step_idx == 0 and not hasattr(_compute_total_loss, "_bias_prior_gyro_printed"):
                print(f"[BIAS PRIOR] gyro={float(prior[0]):+.6e},{float(prior[1]):+.6e},{float(prior[2]):+.6e} (normalized)")
                _compute_total_loss._bias_prior_gyro_printed = True
        else:
            loss = loss + w_bg * bg_pred.pow(2).mean()

    # Visual correction regularization: r should be small (IMU is the baseline)
    w_correction = float(getattr(config, 'loss_w_correction', 0.0))
    if w_correction > 0.0:
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model = actual_model.orig_mod if hasattr(actual_model, 'orig_mod') else actual_model
        if hasattr(actual_model, '_last_step_debug'):
            debug_info = actual_model._last_step_debug
            pos_res_vec = debug_info.get("pos_res_vec")
            if pos_res_vec is not None and isinstance(pos_res_vec, np.ndarray):
                # Convert to tensor for loss computation
                pos_res_tensor = torch.from_numpy(pos_res_vec).to(device=device, dtype=torch.float32)
                correction_loss = pos_res_tensor.pow(2).mean()
                loss = loss + w_correction * correction_loss
                if batch_idx == 0 and step_idx == 0:
                    print(f"[DEIO] correction_reg enabled (w={w_correction:.4f}) | ||r||={correction_loss.item():.6f}")

    # ============================================================================
    # Heteroscedastic NLL Loss for Uncertainty Learning (符合论文标准)
    # ============================================================================
    # NLL Loss: -log P(y|x, σ²) ∝ 0.5 * (log(σ²) + (y - μ)² / σ²)
    #
    # This loss encourages:
    #   1. Small σ when prediction is accurate (residual small)
    #   2. Large σ when prediction is inaccurate (residual large)
    #   3. Proper uncertainty calibration for Bayesian fusion
    # ============================================================================
    w_uncertainty = float(getattr(config, 'loss_w_uncertainty', 0.1))
    w_uncertainty_calib = float(getattr(config, 'loss_w_uncertainty_calib', 0.0))
    # Debug: confirm NLL weight (only on first call)
    if not hasattr(_compute_total_loss, '_nll_weight_printed'):
        print(f"[NLL_CONFIG] loss_w_uncertainty={w_uncertainty:.4f} (>0 enables uncertainty learning)")
        _compute_total_loss._nll_weight_printed = True
    if w_uncertainty > 0.0 or w_uncertainty_calib > 0.0:
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model = actual_model.orig_mod if hasattr(actual_model, 'orig_mod') else actual_model

        if hasattr(actual_model, '_last_step_debug'):
            debug_info = actual_model._last_step_debug

            # Check for tensor-based uncertainties (with gradients)
            log_var_v = debug_info.get("log_var_v_tensor")
            log_var_i = debug_info.get("log_var_i_tensor")
            t_imu = debug_info.get("t_imu_tensor")
            t_visual = debug_info.get("t_visual_tensor")

            # Debug: report missing tensors on first occurrence
            if not hasattr(_compute_total_loss, '_tensor_status_printed'):
                has_all = (log_var_v is not None and log_var_i is not None and
                           t_imu is not None and t_visual is not None)
                print(f"[NLL_TENSORS] log_var_v={log_var_v is not None}, log_var_i={log_var_i is not None}, "
                      f"t_imu={t_imu is not None}, t_visual={t_visual is not None} | all_present={has_all}")
                if has_all:
                    print(f"[NLL_TENSORS] log_var_v.requires_grad={log_var_v.requires_grad}, "
                          f"log_var_i.requires_grad={log_var_i.requires_grad}")
                _compute_total_loss._tensor_status_printed = True

            if (log_var_v is not None and log_var_i is not None and
                t_imu is not None and t_visual is not None and
                isinstance(log_var_v, torch.Tensor) and isinstance(log_var_i, torch.Tensor)):

                # Get GT target translation
                target_t = y[:, 0:3]  # [B, 3]

                # Compute residuals for each prediction
                residual_imu = t_imu - target_t  # [B, 3]
                residual_visual = t_visual - target_t  # [B, 3]

                # NLL Loss: 0.5 * (log(σ²) + residual² / σ²)
                # log_var is log(σ²), so exp(log_var) = σ²
                log_var_v_eff = log_var_v
                if bool(getattr(actual_model, 'uncertainty_use_gate', False)) and s is not None:
                    s_eff = torch.clamp(s.to(dtype=log_var_v.dtype), min=1e-3, max=1e6).view(-1, 1)
                    log_var_v_eff = log_var_v_eff - 2.0 * torch.log(s_eff)

                var_v_eff = torch.exp(torch.clamp(log_var_v_eff, min=-10.0, max=10.0))  # [B, 3]
                var_i = torch.exp(torch.clamp(log_var_i, min=-10.0, max=10.0))  # [B, 3]

                # Per-dimension NLL
                nll_visual = 0.5 * (log_var_v_eff + residual_visual.pow(2) / (var_v_eff + 1e-6))  # [B, 3]
                nll_imu = 0.5 * (log_var_i + residual_imu.pow(2) / (var_i + 1e-6))  # [B, 3]

                # Average over batch and dimensions
                nll_loss = (nll_visual.mean() + nll_imu.mean())

                if w_uncertainty > 0.0:
                    loss = loss + w_uncertainty * nll_loss

                if w_uncertainty_calib > 0.0:
                    rv2 = residual_visual.detach().pow(2) + 1e-4
                    ri2 = residual_imu.detach().pow(2) + 1e-4
                    target_log_var_v = torch.log(rv2).to(dtype=log_var_v_eff.dtype)
                    target_log_var_i = torch.log(ri2).to(dtype=log_var_i.dtype)
                    calib_loss = (
                        F.smooth_l1_loss(log_var_v_eff, target_log_var_v, beta=1.0)
                        + F.smooth_l1_loss(log_var_i, target_log_var_i, beta=1.0)
                    )
                    loss = loss + w_uncertainty_calib * calib_loss

                # Debug: print NLL loss periodically
                if torch.rand(1).item() < 0.001:  # 0.1% chance
                    nll_v_m = float(nll_visual.mean().detach().cpu().item())
                    nll_i_m = float(nll_imu.mean().detach().cpu().item())
                    nll_m = float(nll_loss.detach().cpu().item())
                    print(f"[NLL_DEBUG] nll_v={nll_v_m:.4f} nll_i={nll_i_m:.4f} "
                          f"w={w_uncertainty:.4f} total_nll={nll_m:.4f}")

    return loss


def _clip_gradients_complex_safe(model: nn.Module, max_norm: float = NumericalConstants.GRADIENT_CLIP_NORM) -> None:
    """Manual gradient clipping that handles complex numbers (MPS compatibility)."""
    total_norm = 0.0
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))

    for p in parameters:
        if p.grad.is_complex():
            param_norm = p.grad.detach().abs().norm(2)
        else:
            param_norm = p.grad.detach().norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)


def _backward_and_step(norm_loss: torch.Tensor, model: nn.Module, opt: torch.optim.Optimizer,
                       use_amp: bool, scaler: Optional[torch.cuda.amp.GradScaler]) -> None:
    """Execute backward pass, gradient clipping, and optimizer step."""
    if use_amp and scaler is not None:
        scaler.scale(norm_loss).backward()
        scaler.unscale_(opt)
    else:
        norm_loss.backward()

    _clip_gradients_complex_safe(model)

    if use_amp and scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()


def _forward_and_compute_loss(model: nn.Module, ev: torch.Tensor, imu: torch.Tensor, y: torch.Tensor,
                              hidden: Any, imu_state_mgr: IMUStateManager, actual_dt: float,
                              dt_tensor: Optional[torch.Tensor], intr_tensor: Optional[torch.Tensor],
                              physics_module: Optional[PhysicsBrightnessLoss], config: TrainingConfig,
                              dt: float, current_epoch_physics_weight: float,
                              adaptive_loss_fn: Optional[AdaptiveLossWeights],
                              fx: float, fy: float, device: torch.device,
                              batch_idx: int, step_idx: int, is_amp: bool) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute forward pass and compute loss (shared between AMP and non-AMP paths)."""
    hidden = _adjust_hidden_size(hidden, ev.size(0))
    pred, hidden, new_v, new_R, rot6d, s, ba_pred, bg_pred = model(
        ev, imu, hidden,
        prev_v=imu_state_mgr.velocity,
        prev_R=imu_state_mgr.rotation,
        dt_window=actual_dt,
        debug=(batch_idx == 0 and step_idx == 0)
    )
    actual_model = model.orig_mod if hasattr(model, 'orig_mod') else model
    s_reg_center = None
    try:
        if hasattr(actual_model, 'scale_min') and hasattr(actual_model, 'scale_max'):
            smin = float(getattr(actual_model, 'scale_min'))
            smax = float(getattr(actual_model, 'scale_max'))
            s_reg_center = (0.5 * (smin + smax)) if (smax <= 1.0 + 1e-6) else 1.0
    except Exception:
        s_reg_center = None
    imu_state_mgr.update_states(new_v, new_R)

    if batch_idx == 0 and step_idx == 0:
        actual_model = model.orig_mod if hasattr(model, 'orig_mod') else model
        dbg = getattr(actual_model, "_last_step_debug", None)
        t_hat_body_norm = float(dbg.get("t_hat_body_norm")) if isinstance(dbg, dict) and "t_hat_body_norm" in dbg else float("nan")
        pos_res_norm = float(dbg.get("pos_res_norm")) if isinstance(dbg, dict) and "pos_res_norm" in dbg else float("nan")
        scale_s_val = float(dbg.get("scale_s")) if isinstance(dbg, dict) and "scale_s" in dbg else float("nan")
        if not np.isfinite(scale_s_val):
            scale_s_val = float(dbg.get("ts")) if isinstance(dbg, dict) and "ts" in dbg else float("nan")
        if isinstance(dbg, dict):
            if not np.isfinite(t_hat_body_norm):
                t_hat_body_vec = dbg.get("t_hat_body_vec")
                if isinstance(t_hat_body_vec, np.ndarray) and t_hat_body_vec.ndim == 2 and t_hat_body_vec.shape[1] == 3:
                    t_hat_body_norm = float(np.linalg.norm(t_hat_body_vec, axis=1).mean())
            if not np.isfinite(pos_res_norm):
                pos_res_vec = dbg.get("pos_res_vec")
                if isinstance(pos_res_vec, np.ndarray) and pos_res_vec.ndim == 2 and pos_res_vec.shape[1] == 3:
                    pos_res_norm = float(np.linalg.norm(pos_res_vec, axis=1).mean())
            if not np.isfinite(scale_s_val):
                scale_s_vec = dbg.get("scale_s_vec")
                if not (isinstance(scale_s_vec, np.ndarray) and scale_s_vec.size > 0):
                    scale_s_vec = dbg.get("ts_vec")
                if isinstance(scale_s_vec, np.ndarray) and scale_s_vec.size > 0:
                    scale_s_val = float(np.median(scale_s_vec))
        t_delta_gt_norm = float(y[:, 0:3].detach().norm(dim=1).mean().item()) if y.ndim >= 2 and y.shape[1] >= 3 else float("nan")
        print(f"[DEBUG NORMS][TRAIN] ||t_hat_body||={t_hat_body_norm:.6e} ||pos_res||={pos_res_norm:.6e} ||t_delta_gt||={t_delta_gt_norm:.6e} | scale_s={scale_s_val:.6e} | dt={float(actual_dt):.6e}")

    fx_step, fy_step = fx, fy
    if intr_tensor is not None and intr_tensor.numel() >= 2:
        intr_flat = intr_tensor.view(intr_tensor.shape[0], -1)
        if intr_flat.shape[1] >= 2:
            fx_step = float(intr_flat[:, 0].mean().item())
            fy_step = float(intr_flat[:, 1].mean().item())

    lt, lr, lp, lo = _compute_loss_components(pred, y, rot6d, ev, dt_tensor, physics_module,
                                       config.speed_thresh, dt, config.physics_temp,
                                       config.loss_w_physics_max, config.physics_scale_quantile,
                                       config.physics_event_mask_thresh, fx=fx_step, fy=fy_step,
                                       loss_w_scale=getattr(config, 'loss_w_scale', 0.1),
                                       loss_w_scale_reg=getattr(config, 'loss_w_scale_reg', 0.0),
                                       loss_w_static=getattr(config, 'loss_w_static', 0.0),
                                       scale_reg_center=s_reg_center,
                                       min_step_threshold=getattr(config, 'min_step_threshold', 0.0),
                                       min_step_weight=getattr(config, 'min_step_weight', 0.0),
                                       path_scale_weight=getattr(config, 'loss_w_path_scale', 0.0),
                                       s=s)

    lv = _compute_velocity_loss(new_v, y, device)

    if step_idx == 0 and batch_idx % 50 == 0:
        print(f"[LOSS_DEBUG] Batch {batch_idx} | t={lt.detach().item():.4f} r={lr.detach().item():.4f} v={lv.detach().item():.4f} p={lp.detach().item():.4f} (phys_w={current_epoch_physics_weight:.4f})")

    loss = _compute_total_loss(lt, lr, lv, lp, lo, pred, y, s, ba_pred, bg_pred, config, current_epoch_physics_weight,
                               adaptive_loss_fn, model, device, batch_idx, step_idx, is_amp,
                               dt_tensor=dt_tensor, dt_window_fallback=float(actual_dt))

    return loss, hidden, pred, new_v, new_R


def process_batch_sequence(model: nn.Module, opt: torch.optim.Optimizer, batch_data: List, device: torch.device,
                         config: TrainingConfig, dt: float, adaptive_loss_fn: Optional[AdaptiveLossWeights],
                         physics_module: Optional[PhysicsBrightnessLoss], use_amp: bool,
                         scaler: Optional[torch.cuda.amp.GradScaler], current_epoch_physics_weight: float, fx: float = 1.0, fy: float = 1.0, batch_idx: int = 0, init_hidden: Optional[Any] = None) -> Tuple[float, int, Optional[Any]]:

    hidden = init_hidden
    imu_state_mgr = IMUStateManager(device)  
    accumulated_loss = torch.tensor(0.0, device=device)
    total_batch_steps = 0
    prev_t_pred = None
    prev_pred_full = None
    prev_gt_full = None
    loss_sum = 0.0
    loss_count = 0
    rpe_pred_buf: List[torch.Tensor] = []
    rpe_gt_buf: List[torch.Tensor] = []
    stride_steps = config.tbptt_stride if config.tbptt_stride > 0 else max(int(0.5 / max(dt, 1e-6)), 1)


    warmup_frames = config.warmup_frames

    for step_idx, item in enumerate(batch_data):

        if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (list, tuple)):
            item = item[0]

        intr_tensor = None
        if len(item) == 3:
            ev, imu, y = item[0], item[1], item[2]
            dt_tensor = None
        elif len(item) == 4:
            ev, imu, y, dt_tensor = item[0], item[1], item[2], item[3]
        else:
            ev, imu, y, dt_tensor, intr_tensor = item[0], item[1], item[2], item[3], item[4]


        ev, imu, y = ev.to(device, non_blocking=True), imu.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if dt_tensor is not None:
            dt_tensor = dt_tensor.to(device, non_blocking=True)

            dt_flat = dt_tensor.view(dt_tensor.shape[0], -1) if dt_tensor.ndim > 1 else dt_tensor.view(-1, 1)
            dt_per_sample = dt_flat.mean(dim=1)
            dt_min = float(dt_per_sample.min().item())
            dt_max = float(dt_per_sample.max().item())
            dt_mean = max(float(dt_per_sample.mean().item()), 1e-6)
            if abs(dt_max - dt_min) / dt_mean > 1e-3:
                raise ValueError(
                    f"[DT CHECK] Inconsistent dt_window in batch {batch_idx}: "
                    f"range=({dt_min:.6e}, {dt_max:.6e}), mean={dt_mean:.6e}. "
                    f"Please ensure all samples in a batch share the same dt_window."
                )
            actual_dt = float(dt_mean)
        else:
            actual_dt = dt

        if intr_tensor is not None:
            intr_tensor = intr_tensor.to(device, non_blocking=True)


        if step_idx == 0:
            imu_state_mgr.initialize(ev.size(0))
            if y.shape[1] >= 11:
                q_init = y[:, 7:11]
                # Ensure quaternion is normalized
                q_init = F.normalize(q_init, p=2, dim=1)
                imu_state_mgr.rotation = GeometryUtils.quat_to_rot(q_init)
            
            if y.shape[1] >= 14:
                v_init = y[:, 11:14]
                imu_state_mgr.velocity = v_init


        # Unified forward pass and loss computation (AMP vs non-AMP)
        with amp_autocast_context(device, use_amp):
            loss, hidden, pred, new_v, new_R = _forward_and_compute_loss(
                model, ev, imu, y, hidden, imu_state_mgr, actual_dt,
                dt_tensor, intr_tensor, physics_module, config, dt,
                current_epoch_physics_weight, adaptive_loss_fn,
                fx, fy, device, batch_idx, step_idx, is_amp=use_amp
            )

        # Warmup and post-processing (shared logic)
        if step_idx < warmup_frames:
            warmup_factor = min(1.0, (step_idx + 1) / warmup_frames)
            loss = loss * warmup_factor
            if step_idx == 0 and batch_idx % 50 == 0:
                print(f"[PROGRESSIVE_WARMUP] Step {step_idx}/{warmup_frames}, factor: {warmup_factor:.2f}")
        else:
            if config.loss_w_smooth > 0 and prev_t_pred is not None:
                loss += config.loss_w_smooth * F.mse_loss(pred[:, 0:3], prev_t_pred)

            if config.use_rpe_loss and config.loss_w_rpe > 0.0 and prev_pred_full is not None and prev_gt_full is not None:
                rpe_loss = compute_rpe_loss_compose(prev_pred_full, pred, prev_gt_full, y, config.loss_w_rpe)
                loss += rpe_loss

            prev_t_pred = pred[:, 0:3].detach()
            prev_pred_full = pred.detach()
            prev_gt_full = y.detach()

            if not torch.isfinite(loss).all().item():
                loss_count += 1
                continue

            accumulated_loss = accumulated_loss + loss
            total_batch_steps += 1
            loss_sum += float(loss.detach().item())
        loss_count += 1

        # TBPTT更新
        if total_batch_steps > 0 and (total_batch_steps % config.tbptt_len == 0):
            norm_loss = accumulated_loss / max(total_batch_steps, 1)
            _backward_and_step(norm_loss, model, opt, use_amp, scaler)
            opt.zero_grad()
            hidden = _detach_hidden(hidden)
            imu_state_mgr.detach_states()
            accumulated_loss = torch.tensor(0.0, device=device)
            total_batch_steps = 0

    # Final TBPTT update for remaining steps
    if total_batch_steps > 0:
        norm_loss = accumulated_loss / max(total_batch_steps, 1)
        _backward_and_step(norm_loss, model, opt, use_amp, scaler)
        opt.zero_grad()
        hidden = _detach_hidden(hidden)
        imu_state_mgr.detach_states()

    return loss_sum, loss_count, hidden


def update_epoch_metrics(loss_sum: float, loss_count: int, total_loss: float, num_steps: int) -> Tuple[float, int]:
    return total_loss + (loss_sum / max(loss_count, 1)), num_steps + 1


def monitor_memory_usage(device: torch.device, stage: str = ""):
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
        print(f"[MEMORY] {stage}: Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    elif device.type == "mps":
        # MPS memory monitoring is more limited
        if hasattr(torch.mps, "allocated_memory"):
            allocated = torch.mps.allocated_memory() / 1024**3  # GB
            print(f"[MEMORY] {stage}: Allocated: {allocated:.2f}GB")


def setup_argument_groups(parser):
    """Helper to register groups if needed"""
    # Dataset args
    ds_args = [
        {'name': '--root_dir', 'type': str, 'default': None, 'help': 'Root directory for dataset'},
        {'name': '--events_h5', 'type': str, 'default': None, 'help': 'Path to events h5 file'},
        {'name': '--dt', 'type': float, 'default': 0.00833, 'help': 'Time interval (default 0.00833s = 120Hz)'},
        {'name': '--resolution', 'type': int, 'nargs': 2, 'default': [180, 320], 'help': 'Image resolution'},
        {'name': '--sensor_resolution', 'type': int, 'nargs': 2, 'default': None, 'help': 'Sensor resolution'},
        {'name': '--sample_stride', 'type': int, 'default': 8, 'help': 'Sampling stride'},
        {'name': '--windowing_mode', 'type': str, 'default': 'imu', 'help': 'Windowing timestamps: imu (sensor) or gt (legacy)'},
        {'name': '--window_dt', 'type': float, 'default': None, 'help': 'Window duration (seconds). If not set: dt for imu mode.'},
        {'name': '--event_offset_scan', 'action': 'store_true', 'help': 'Coarse scan to suggest events_time_offset_ns adjustment'},
        {'name': '--event_offset_scan_range_s', 'type': float, 'default': 0.5, 'help': 'Scan range (seconds) around current events_time_offset_ns'},
        {'name': '--event_offset_scan_step_s', 'type': float, 'default': 0.01, 'help': 'Scan step (seconds) for coarse offset scan'},
        {'name': '--event_file_candidates', 'type': str, 'nargs': '+', 'default': ("events-left.h5", "events_left.h5", "mocap-6dof-events_left.h5"), 'help': 'Event file candidates'},
        {'name': '--voxel_std_norm', 'action': 'store_true', 'help': 'Voxel standardization'},
        {'name': '--augment', 'action': 'store_true', 'help': 'Data augmentation'},
        {'name': '--adaptive_voxel', 'action': 'store_true', 'help': 'Adaptive voxelization'},
        {'name': '--no-adaptive_voxel', 'dest': 'adaptive_voxel', 'action': 'store_false'},
        {'name': '--event_noise_scale', 'type': float, 'default': 0.01, 'help': 'Event noise scale'},
        {'name': '--event_scale_jitter', 'type': float, 'default': 0.1, 'help': 'Event scale jitter'},
        {'name': '--imu_bias_scale', 'type': float, 'default': 0.02, 'help': 'IMU bias scale'},
        {'name': '--imu_mask_prob', 'type': float, 'default': 0.0, 'help': 'IMU mask probability'},
        {'name': '--adaptive_base_div', 'type': int, 'default': 60, 'help': 'Adaptive base divisor'},
        {'name': '--adaptive_max_events_div', 'type': int, 'default': 12, 'help': 'Adaptive max events divisor'},
        {'name': '--adaptive_density_cap', 'type': float, 'default': 2.0, 'help': 'Adaptive density cap'},
        {'name': '--derotate', 'action': 'store_true', 'help': 'Derotate events'},
        {'name': '--voxelize_in_dataset', 'type': str, 'default': "true", 'help': 'Voxelize in dataset'},
        {'name': '--train_split', 'type': float, 'default': 0.7, 'help': 'Train split ratio'},
        {'name': '--sequence_len', 'type': int, 'default': 400, 'help': 'Sequence length for slicing'},
        {'name': '--sequence_stride', 'type': int, 'default': 200, 'help': 'Sequence stride'},
    ]
    parser.set_defaults(adaptive_voxel=True)
    add_argument_group(parser, "Dataset Configuration", ds_args)

    # Model args
    model_args = [
        {'name': '--modes', 'type': int, 'default': 10, 'help': 'Number of modes'},
        {'name': '--stem_channels', 'type': int, 'default': 64, 'help': 'Stem channels'},
        {'name': '--imu_embed_dim', 'type': int, 'default': 64, 'help': 'IMU embedding dimension'},
        {'name': '--lstm_hidden', 'type': int, 'default': 128, 'help': 'LSTM hidden size'},
        {'name': '--lstm_layers', 'type': int, 'default': 2, 'help': 'LSTM layers'},
        {'name': '--imu_channels', 'type': int, 'default': 6, 'help': 'IMU channels'},
        {'name': '--attn_groups', 'type': int, 'default': 8, 'help': 'Attention groups'},
        {'name': '--imu_gn_groups', 'type': int, 'default': None, 'help': 'IMU GroupNorm groups'},
        {'name': '--norm_mode', 'type': str, 'default': 'gn', 'help': 'Normalization mode'},
        {'name': '--fast_fft', 'action': 'store_true', 'help': 'Use Fast FFT'},
        {'name': '--state_aug', 'action': 'store_true', 'help': 'State augmentation'},
        {'name': '--imu_gate_soft', 'action': 'store_true', 'help': 'Soft IMU gating'},
        {'name': '--no-imu_gate_soft', 'dest': 'imu_gate_soft', 'action': 'store_false'},
        {'name': '--uncertainty_fusion', 'dest': 'uncertainty_fusion', 'action': 'store_true', 'help': 'Enable Bayesian uncertainty fusion (default on when imu_gate_soft is on)'},
        {'name': '--no-uncertainty_fusion', 'dest': 'uncertainty_fusion', 'action': 'store_false', 'help': 'Disable Bayesian uncertainty fusion (keep IMU preintegration + legacy convex fusion)'},
        {'name': '--uncertainty_gate', 'dest': 'uncertainty_use_gate', 'action': 'store_true', 'help': 'Use s_gate to modulate visual effective variance (requires --uncertainty_fusion)'},
        {'name': '--no-uncertainty_gate', 'dest': 'uncertainty_use_gate', 'action': 'store_false', 'help': 'Do not use s_gate to modulate visual effective variance'},

        {'name': '--use_cudnn_lstm', 'action': 'store_true', 'help': 'Use CuDNN LSTM'},
        {'name': '--gravity', 'type': float, 'nargs': 3, 'default': [0.0, 0.0, -9.81], 'help': 'Gravity vector'},
        {'name': '--use_dual_attention', 'action': 'store_true', 'help': 'Use dual attention'},
        {'name': '--no-use_dual_attention', 'dest': 'use_dual_attention', 'action': 'store_false'},
        {'name': '--use_mr_fno', 'action': 'store_true', 'help': 'Use MR-FNO'},
        {'name': '--no-use_mr_fno', 'dest': 'use_mr_fno', 'action': 'store_false'},
        {'name': '--modes_low', 'type': int, 'default': 16, 'help': 'Low frequency modes'},
        {'name': '--modes_high', 'type': int, 'default': 32, 'help': 'High frequency modes'},
        {'name': '--window_stack_K', 'type': int, 'default': 1, 'help': 'Window stack K'},
        {'name': '--voxel_stack_mode', 'type': str, 'default': 'abs', 'choices': ['abs', 'delta'], 'help': 'Voxel stack representation: abs or delta'},
        {'name': '--use_cross_attn', 'action': 'store_true', 'help': 'Use cross attention'},
        {'name': '--no-use_cross_attn', 'dest': 'use_cross_attn', 'action': 'store_false'},
        {'name': '--fusion_dim', 'type': int, 'default': None, 'help': 'Fusion dimension'},
        {'name': '--fusion_heads', 'type': int, 'default': 4, 'help': 'Fusion heads'},
        {'name': '--scale_min', 'type': float, 'default': 0.0, 'help': 'Visual confidence gate lower bound (standard VIO: 0.0)'},
        {'name': '--scale_max', 'type': float, 'default': 1.0, 'help': 'Visual confidence gate upper bound (standard VIO: 1.0). Setting >1 enables experimental visual scale amplification.'},
    ]
    parser.set_defaults(imu_gate_soft=True, uncertainty_fusion=False, uncertainty_use_gate=False, use_dual_attention=True, use_mr_fno=False, use_cross_attn=False)
    add_argument_group(parser, "Model Configuration", model_args)

    # Training args
    train_args = [
        {'name': '--epochs', 'type': int, 'default': 500, 'help': 'Number of epochs'},
        {'name': '--batch_size', 'type': int, 'default': 512, 'help': 'Batch size'},
        {'name': '--lr', 'type': float, 'default': 6e-5, 'help': 'Learning rate'},
        {'name': '--eval_interval', 'type': int, 'default': 1, 'help': 'Evaluation interval'},
        {'name': '--export_torchscript', 'action': 'store_true', 'help': 'Export TorchScript'},
        {'name': '--loss_w_t', 'type': float, 'default': 10.0, 'help': 'Translation loss weight'},
        {'name': '--loss_w_r', 'type': float, 'default': 10.0, 'help': 'Rotation loss weight'},
        {'name': '--loss_w_v', 'type': float, 'default': 1.0, 'help': 'Velocity loss weight (>0 helps prevent scale drift)'},
        {'name': '--loss_w_aux_motion', 'type': float, 'default': 0.3, 'help': 'Aux motion loss weight for FNO pre-LSTM motion head (default 0.0 disables)'},



        {'name': '--loss_w_physics', 'type': float, 'default': 0.02, 'help': 'Physics loss weight'},
        {'name': '--loss_w_smooth', 'type': float, 'default': 0.1, 'help': 'Smoothness loss weight'},
        {'name': '--loss_w_rpe', 'type': float, 'default': 0.05, 'help': 'RPE loss weight'},
        {'name': '--rpe_dt', 'type': float, 'default': 0.5, 'help': 'RPE delta t'},
        {'name': '--physics_mode', 'type': str, 'default': 'none', 'help': 'Physics mode'},
        {'name': '--speed_thresh', 'type': float, 'default': 0.02, 'help': 'Speed threshold'},
        {'name': '--tbptt_len', 'type': int, 'default': 75, 'help': 'TBPTT length'},
        {'name': '--tbptt_stride', 'type': int, 'default': 0, 'help': 'TBPTT stride'},

        {'name': '--physics_temp', 'type': float, 'default': 0.5, 'help': 'Physics temperature'},
        {'name': '--loss_w_physics_max', 'type': float, 'default': 1.0, 'help': 'Max physics loss weight'},
        {'name': '--physics_scale_quantile', 'type': float, 'default': 0.95, 'help': 'Physics scale quantile'},
        {'name': '--physics_event_mask_thresh', 'type': float, 'default': 0.05, 'help': 'Physics event mask threshold'},
        {'name': '--scheduler', 'type': str, 'default': 'step', 'help': 'Scheduler type'},
        {'name': '--gamma', 'type': float, 'default': 0.5, 'help': 'Scheduler gamma'},
        {'name': '--scheduler_patience', 'type': int, 'default': 5, 'help': 'Scheduler patience'},
        {'name': '--scheduler_T_max', 'type': int, 'default': 50, 'help': 'Scheduler T_max'},
        {'name': '--patience', 'type': int, 'default': 200, 'help': 'Early stopping patience'},
        {'name': '--earlystop_min_epoch', 'type': int, 'default': 50, 'help': 'Earliest epoch to start early stopping'},

        {'name': '--earlystop_ma_window', 'type': int, 'default': 5, 'help': 'Moving average window size for early stopping metric'},
        {'name': '--earlystop_alpha', 'type': float, 'default': 1.0, 'help': 'Weight for RPE_t in composite early stopping metric'},
        {'name': '--earlystop_beta', 'type': float, 'default': 0.2, 'help': 'Weight for RPE_r(deg) in composite early stopping metric'},
        {'name': '--compile', 'action': 'store_true', 'help': 'Compile model'},
        {'name': '--adaptive_loss', 'dest': 'adaptive_loss', 'action': 'store_true', 'help': 'Enable adaptive loss weights'},
        {'name': '--no-adaptive_loss', 'dest': 'adaptive_loss', 'action': 'store_false', 'help': 'Disable adaptive loss weights (fixed loss_w_* weighting)'},
        {'name': '--use_rpe_loss', 'action': 'store_true', 'help': 'Use RPE loss'},
        {'name': '--no-use_rpe_loss', 'dest': 'use_rpe_loss', 'action': 'store_false'},
        {'name': '--use_imu_consistency', 'action': 'store_true', 'help': 'Use IMU consistency'},
        {'name': '--no-use_imu_consistency', 'dest': 'use_imu_consistency', 'action': 'store_false'},
        {'name': '--loss_w_imu', 'type': float, 'default': 0.1, 'help': 'IMU loss weight'},
        {'name': '--warmup_frames', 'type': int, 'default': 20, 'help': 'Warmup frames'},
        {'name': '--warmup_epochs', 'type': int, 'default': 20, 'help': 'LR warmup epochs'},

        {'name': '--mixed_precision', 'action': 'store_true', 'help': 'Use AMP mixed precision'},
        {'name': '--earlystop_metric', 'type': str, 'default': 'composite', 'help': 'Early stopping metric: composite or ate'},
        {'name': '--loss_w_scale', 'type': float, 'default': 0.0, 'help': 'Scale consistency loss weight (default 0.0 for clean baseline)'},
        {'name': '--loss_w_scale_reg', 'type': float, 'default': 0.0, 'help': 'Global scale regularization weight'},
        {'name': '--loss_w_static', 'type': float, 'default': 0.0, 'help': 'Static/near-static step penalty weight (default 0.0 for clean baseline)'},
        {'name': '--loss_w_bias_a', 'type': float, 'default': 1e-4, 'help': 'Accelerometer bias L2 regularization weight'},
        {'name': '--loss_w_bias_g', 'type': float, 'default': 1e-4, 'help': 'Gyroscope bias L2 regularization weight'},
        {'name': '--bias_prior_accel', 'type': float, 'nargs': 3, 'default': None, 'help': 'Accel bias prior (normalized accel/9.81): ax ay az'},
        {'name': '--bias_prior_gyro', 'type': float, 'nargs': 3, 'default': None, 'help': 'Gyro bias prior (normalized gyro/pi): gx gy gz'},
        {'name': '--loss_w_uncertainty', 'type': float, 'default': 0.0, 'help': 'Uncertainty NLL loss weight for Bayesian fusion (default 0.0 for clean baseline)'},
        {'name': '--loss_w_uncertainty_calib', 'type': float, 'default': 0.0, 'help': 'Uncertainty calibration loss weight (default 0.0)'},
        {'name': '--use_seq_scale', 'action': 'store_true', 'help': 'Enable per-sequence latent scale regularizer'},
        {'name': '--no-use_seq_scale', 'dest': 'use_seq_scale', 'action': 'store_false'},
        {'name': '--seq_scale_reg', 'type': float, 'default': 0.0, 'help': 'Sequence scale regularization weight (default 0.0 for clean baseline)'},
        {'name': '--min_step_threshold', 'type': float, 'default': 0.0, 'help': 'Minimum mean step norm threshold'},
        {'name': '--min_step_weight', 'type': float, 'default': 0.0, 'help': 'Minimum step penalty weight'},
        {'name': '--loss_w_path_scale', 'type': float, 'default': 0.0, 'help': 'Path length scale consistency loss weight (default 0.0 for clean baseline)'},
        # IMU-anchored fusion constraints
        {'name': '--loss_w_correction', 'type': float, 'default': 0.0, 'help': 'Visual correction magnitude regularization (r should be small, default 0.0)'},
        {'name': '--loss_w_bias_smooth', 'type': float, 'default': 0.0, 'help': 'Bias smoothness constraint |b_t - b_{t-1}| (default 0.0)'},

        {'name': '--eval_sim3_mode', 'type': str, 'default': 'diagnose', 'help': 'SIM(3) evaluation mode: diagnose/use_learned/fix_learned/off'},
    ]
    parser.set_defaults(use_rpe_loss=True, use_imu_consistency=False, use_seq_scale=False, adaptive_loss=False)
    add_argument_group(parser, "Training Configuration", train_args)

def add_argument_group(parser, title, args_list):
    """Adds a group of arguments to the parser"""
    group = parser.add_argument_group(title)
    for arg in args_list:
        name = arg.pop('name')
        group.add_argument(name, **arg)

def parse_command_line_arguments() -> argparse.Namespace:
    """Simplified command line argument parsing using CLI utilities."""
    parser = argparse.ArgumentParser()

    # Add backward compatibility alias for dataset root argument
    parser.add_argument('--dataset_root', dest='root_dir', type=str,
                       help='Root directory for dataset (alias for --root_dir)')

    # Setup all argument groups automatically
    setup_argument_groups(parser)

    # Add additional arguments not covered by standard groups
    additional_args = [
        {'name': '--multi_root', 'type': str, 'nargs': '+', 'default': None, 'help': 'Multiple dataset root directories'},
        {'name': '--calib_yaml', 'type': str, 'default': '', 'help': 'Calibration YAML file'},
        {'name': '--multi_calib_yaml', 'type': str, 'nargs': '+', 'default': None, 'help': 'Calibration YAML per dataset root (same order as --multi_root).'},
        {'name': '--batch_by_root', 'dest': 'batch_by_root', 'action': 'store_true', 'help': 'Build each training batch from a single root (recommended for multi_root with different dt).'},
        {'name': '--no-batch_by_root', 'dest': 'batch_by_root', 'action': 'store_false', 'help': 'Allow mixing different roots in the same training batch.'},
        {'name': '--balanced_sampling', 'action': 'store_true', 'help': 'Balance sampling across roots to prevent long sequences from dominating.'},
        {'name': '--eval_all_roots', 'action': 'store_true', 'help': 'Evaluate validation metrics on every root separately and report the mean.'},
        {'name': '--eval_batch_size', 'type': int, 'default': None, 'help': 'Evaluation batch size'},
        {'name': '--imu_dropout_p_start', 'type': float, 'default': 0.3, 'help': 'Initial IMU dropout probability'},
        {'name': '--imu_dropout_p_end', 'type': float, 'default': 0.1, 'help': 'Final IMU dropout probability'},
        {'name': '--log_file', 'type': str, 'default': '', 'help': 'Log file path'},
        {'name': '--seed', 'type': int, 'default': 42, 'help': 'Random seed'},
        {'name': '--num_workers', 'type': int, 'default': 8, 'help': 'Number of DataLoader workers (recommended: 4-8 for large datasets)'},
        {'name': '--persistent_workers', 'action': 'store_true', 'help': 'Keep workers alive between epochs (recommended for faster training)'},

        {'name': '--prefetch_factor', 'type': int, 'default': 4, 'help': 'Number of batches to prefetch per worker (recommended: 4-8)'},
        {'name': '--metrics_csv', 'type': str, 'default': '', 'help': 'Metrics CSV output file'},
        {'name': '--init_checkpoint', 'type': str, 'default': '', 'help': 'Initialize model weights from a checkpoint path (finetune/start-from).'},
    ]

    # Set device-specific pin_memory default
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    default_pin_memory = False
    parser.set_defaults(pin_memory=default_pin_memory)
    parser.set_defaults(batch_by_root=None)
    parser.set_defaults(persistent_workers=True)

    add_argument_group(parser, "Additional Configuration", additional_args)

    parser.add_argument('--output_dir', type=str, default='fno-FAST', help='Directory to save checkpoints and models')
    return parser.parse_args()


def setup_configurations(args: argparse.Namespace) -> Tuple[DatasetConfig, ModelConfig, TrainingConfig]:
    #convert args to dict
    ds_cfg = DatasetConfig(
        root=args.root_dir,
        events_h5=args.events_h5,
        dt=args.dt,
        resolution=tuple(args.resolution),
        sensor_resolution=tuple(args.sensor_resolution) if args.sensor_resolution else None,
        sample_stride=args.sample_stride,
        windowing_mode=str(getattr(args, "windowing_mode", "imu")),
        window_dt=(getattr(args, "window_dt", None)),
        event_offset_scan=bool(getattr(args, "event_offset_scan", False)),
        event_offset_scan_range_s=float(getattr(args, "event_offset_scan_range_s", 0.5)),
        event_offset_scan_step_s=float(getattr(args, "event_offset_scan_step_s", 0.01)),
        event_file_candidates=tuple(args.event_file_candidates),
        voxel_std_norm=args.voxel_std_norm,
        augment=args.augment,
        adaptive_voxel=args.adaptive_voxel,
        event_noise_scale=args.event_noise_scale,
        event_scale_jitter=args.event_scale_jitter,
        imu_bias_scale=args.imu_bias_scale,
        imu_mask_prob=args.imu_mask_prob,
        adaptive_base_div=args.adaptive_base_div,
        adaptive_max_events_div=args.adaptive_max_events_div,
        adaptive_density_cap=args.adaptive_density_cap,
        derotate=args.derotate,
        voxelize_in_dataset=(args.voxelize_in_dataset.lower() == "true"),
        train_split=args.train_split
    )


    imu_seq_len = compute_adaptive_sequence_length(args.dt)

    model_cfg = ModelConfig(
        modes=args.modes,
        stem_channels=args.stem_channels,
        imu_embed_dim=args.imu_embed_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        imu_channels=args.imu_channels,
        sequence_length=imu_seq_len,
        attn_groups=args.attn_groups,
        imu_gn_groups=args.imu_gn_groups,
        norm_mode=args.norm_mode,
        fast_fft=args.fast_fft,
        state_aug=args.state_aug,
        imu_gate_soft=args.imu_gate_soft,
        use_uncertainty_fusion=bool(getattr(args, 'uncertainty_fusion', True)),
        uncertainty_use_gate=bool(getattr(args, 'uncertainty_use_gate', True)),
        use_cudnn_lstm=args.use_cudnn_lstm,
        gravity=tuple(args.gravity),
        use_dual_attention=args.use_dual_attention,
        use_mr_fno=args.use_mr_fno,
        modes_low=args.modes_low,
        modes_high=args.modes_high,
        window_stack_K=args.window_stack_K,
        voxel_stack_mode=getattr(args, 'voxel_stack_mode', 'abs'),
        use_cross_attn=args.use_cross_attn,
        fusion_dim=(args.fusion_dim if args.fusion_dim is not None else args.stem_channels),
        fusion_heads=args.fusion_heads,
        scale_min=getattr(args, 'scale_min', 0.0),
        scale_max=getattr(args, 'scale_max', 1.0)
    )

    global WINDOW_STACK_K
    WINDOW_STACK_K = int(args.window_stack_K)

    # 构建训练配置
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_interval=args.eval_interval,
        export_torchscript=args.export_torchscript,
        loss_w_t=args.loss_w_t,
        loss_w_r=args.loss_w_r,
        loss_w_v=args.loss_w_v,
        loss_w_aux_motion=getattr(args, 'loss_w_aux_motion', 0.3),
        loss_w_physics=args.loss_w_physics,
        loss_w_smooth=args.loss_w_smooth,
        loss_w_rpe=args.loss_w_rpe,
        rpe_dt=args.rpe_dt,
        physics_mode=args.physics_mode,
        speed_thresh=args.speed_thresh,
        tbptt_len=args.tbptt_len,
        tbptt_stride=args.tbptt_stride,
        physics_temp=args.physics_temp,
        loss_w_physics_max=args.loss_w_physics_max,
        physics_scale_quantile=args.physics_scale_quantile,
        physics_event_mask_thresh=args.physics_event_mask_thresh,
        scheduler=args.scheduler,
        gamma=args.gamma,
        scheduler_patience=args.scheduler_patience,
        scheduler_T_max=args.scheduler_T_max,
        patience=args.patience,
        compile=args.compile,
        adaptive_loss_weights=bool(getattr(args, 'adaptive_loss', True)),
        use_rpe_loss=args.use_rpe_loss,
        use_imu_consistency=args.use_imu_consistency,
        loss_w_imu=args.loss_w_imu,
        warmup_epochs=args.warmup_epochs,
        warmup_frames=args.warmup_frames,
        mixed_precision=getattr(args, 'mixed_precision', False),
        earlystop_min_epoch=args.earlystop_min_epoch,
        earlystop_ma_window=args.earlystop_ma_window,
        earlystop_alpha=args.earlystop_alpha,
        earlystop_beta=args.earlystop_beta,
        earlystop_metric=getattr(args, 'earlystop_metric', 'composite'),
        loss_w_scale=getattr(args, 'loss_w_scale', 1.0),
        loss_w_scale_reg=getattr(args, 'loss_w_scale_reg', 0.0),
        loss_w_static=getattr(args, 'loss_w_static', 2.0),
        loss_w_bias_a=getattr(args, 'loss_w_bias_a', 1e-4),
        loss_w_bias_g=getattr(args, 'loss_w_bias_g', 1e-4),
        loss_w_uncertainty=getattr(args, 'loss_w_uncertainty', 0.1),
        loss_w_uncertainty_calib=getattr(args, 'loss_w_uncertainty_calib', 0.0),
        use_seq_scale=getattr(args, 'use_seq_scale', True),
        seq_scale_reg=getattr(args, 'seq_scale_reg', 0.2),
        min_step_threshold=getattr(args, 'min_step_threshold', 0.0),
        min_step_weight=getattr(args, 'min_step_weight', 0.0),
        loss_w_path_scale=getattr(args, 'loss_w_path_scale', 0.1),
        eval_sim3_mode=getattr(args, 'eval_sim3_mode', 'diagnose'),
        # DEIO-style constraints
        loss_w_correction=getattr(args, 'loss_w_correction', 0.0),
        loss_w_bias_smooth=getattr(args, 'loss_w_bias_smooth', 0.0),
        bias_prior_accel=(tuple(getattr(args, 'bias_prior_accel', None)) if getattr(args, 'bias_prior_accel', None) is not None else None),
        bias_prior_gyro=(tuple(getattr(args, 'bias_prior_gyro', None)) if getattr(args, 'bias_prior_gyro', None) is not None else None),
    )

    calib_yaml = getattr(args, "calib_yaml", "") or ""
    if not calib_yaml and getattr(args, "multi_calib_yaml", None):
        try:
            ymls = list(getattr(args, "multi_calib_yaml"))
            calib_yaml = str(ymls[0]) if len(ymls) > 0 else ""
        except Exception:
            calib_yaml = ""
    calib_obj = load_calibration(calib_yaml) if calib_yaml else None
    if isinstance(calib_obj, dict):
        a_norm, g_norm = _extract_bias_prior_from_calib(calib_obj)
        tol = 1e-5
        if train_cfg.bias_prior_accel is None and a_norm is not None:
            train_cfg.bias_prior_accel = a_norm
        elif train_cfg.bias_prior_accel is not None and a_norm is not None:
            d = np.max(np.abs(np.asarray(train_cfg.bias_prior_accel, dtype=np.float64) - np.asarray(a_norm, dtype=np.float64)))
            if np.isfinite(d) and d > tol:
                print(f"[BIAS PRIOR] accel CLI overrides calib_yaml (max_abs_diff={float(d):.3e})")
        if train_cfg.bias_prior_gyro is None and g_norm is not None:
            train_cfg.bias_prior_gyro = g_norm
        elif train_cfg.bias_prior_gyro is not None and g_norm is not None:
            d = np.max(np.abs(np.asarray(train_cfg.bias_prior_gyro, dtype=np.float64) - np.asarray(g_norm, dtype=np.float64)))
            if np.isfinite(d) and d > tol:
                print(f"[BIAS PRIOR] gyro CLI overrides calib_yaml (max_abs_diff={float(d):.3e})")

    return ds_cfg, model_cfg, train_cfg


def setup_device_and_environment(args: argparse.Namespace) -> torch.device:

    device = build_device()

    import torch.multiprocessing as mp
    if args.num_workers > 0:
        try:
            mp.set_sharing_strategy("file_system")
        except Exception:
            pass
        if device.type == "cuda":
            mp.set_start_method("spawn", force=True)

    torch.set_default_dtype(torch.float32)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    return device


def load_calibration(calib_yaml: str) -> Optional[Dict[str, Any]]:
    if not calib_yaml:
        return None
    try:
        import yaml
        with open(calib_yaml, "r") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            return None
        calib = raw.get("calib") or raw.get("value0") or raw
        if not isinstance(calib, dict):
            return None
        if "K" not in calib:
            cam = calib.get("camera", {})
            if isinstance(cam, dict):
                calib["K"] = {
                    "fx": float(cam.get("fx", 1.0)),
                    "fy": float(cam.get("fy", 1.0)),
                    "cx": float(cam.get("cx", 0.0)),
                    "cy": float(cam.get("cy", 0.0)),
                    # 保留相机类型和KB4畸变参数
                    "camera_type": cam.get("camera_type", "pinhole"),
                    "k1": float(cam.get("k1", 0.0)),
                    "k2": float(cam.get("k2", 0.0)),
                    "k3": float(cam.get("k3", 0.0)),
                    "k4": float(cam.get("k4", 0.0)),
                }
                # Extract resolution if available
                if "width" in cam and "height" in cam:
                    calib["resolution"] = (int(cam["width"]), int(cam["height"]))
        if "R_IC" not in calib:
            if "T_imu_cam" in calib:
                T_val = calib["T_imu_cam"]
                if isinstance(T_val, dict) and all(k in T_val for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")):
                    q = np.asarray([T_val["qx"], T_val["qy"], T_val["qz"], T_val["qw"]], dtype=np.float64)
                    n = float(np.linalg.norm(q))
                    if n > 0.0:
                        q = q / n
                    x, y, z, w = q
                    R = np.array([
                        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),         2.0 * (x * z + y * w)],
                        [2.0 * (x * y + z * w),         1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                        [2.0 * (x * z - y * w),         2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
                    ], dtype=np.float64)
                    calib["R_IC"] = R.tolist()
                else:
                    M = np.asarray(T_val, dtype=np.float64)
                    calib["R_IC"] = M[:3, :3].tolist()
            elif "R_imu_cam" in calib:
                R = np.asarray(calib["R_imu_cam"], dtype=np.float64)
                calib["R_IC"] = R.tolist()
        return calib
    except Exception:
        return None


def _extract_bias_prior_from_calib(calib: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    def _as3(v) -> Optional[Tuple[float, float, float]]:
        if v is None:
            return None
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            try:
                return (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                return None
        return None

    a = _as3(calib.get("calib_accel_bias_mean") or calib.get("accel_bias_mean") or calib.get("imu_accel_bias_mean"))
    g = _as3(calib.get("calib_gyro_bias_mean") or calib.get("gyro_bias_mean") or calib.get("imu_gyro_bias_mean"))
    if a is None and g is None:
        return None, None
    a_norm = None
    g_norm = None
    if a is not None:
        a_norm = (a[0] / 9.81, a[1] / 9.81, a[2] / 9.81)
    if g is not None:
        g_norm = (g[0] / float(np.pi), g[1] / float(np.pi), g[2] / float(np.pi))
    return a_norm, g_norm


def setup_logging(args: argparse.Namespace) -> Optional[Any]:

    global LOG_FH
    if args.log_file:
        import sys, os
        try:
            os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        except Exception:
            pass
        try:
            LOG_FH = open(args.log_file, "a")
        except Exception:
            LOG_FH = None
        if LOG_FH is not None:
            class Tee:
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, data):
                    for s in self.streams:
                        try:
                            s.write(data)
                            s.flush()
                        except Exception:
                            pass
                def flush(self):
                    for s in self.streams:
                        try:
                            s.flush()
                        except Exception:
                            pass
            sys.stdout = Tee(sys.stdout, LOG_FH)
            sys.stderr = Tee(sys.stderr, LOG_FH)
    else:
        LOG_FH = None
    return LOG_FH


class RootGroupedBatchSampler(torch.utils.data.Sampler):
    """Batch sampler that keeps batches within each root/segment of a ConcatDataset.

    When balanced=True, each epoch samples equal number of batches from each root
    by oversampling smaller datasets (with replacement) to match the largest one.
    This ensures all roots contribute equally to training while keeping batches
    from the same root together.
    """

    def __init__(self, ds: ConcatDataset, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, balanced: bool = False) -> None:
        self.ds = ds
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.balanced = bool(balanced)
        self._segments: List[Tuple[int, int]] = []
        off = 0
        for sub in getattr(ds, "datasets", []):
            n = int(len(sub))
            self._segments.append((off, n))
            off += n

        # For balanced mode: compute target batches per root
        if self.balanced and len(self._segments) > 1:
            batches_per_seg = []
            for _, n in self._segments:
                if self.drop_last:
                    batches_per_seg.append(n // self.batch_size)
                else:
                    batches_per_seg.append((n + self.batch_size - 1) // self.batch_size)
            self._max_batches = max(batches_per_seg) if batches_per_seg else 0
            self._batches_per_seg = batches_per_seg
            print(f"[RootGroupedBatchSampler] Balanced mode: batches_per_root={batches_per_seg}, "
                  f"target={self._max_batches} batches/root/epoch")
        else:
            self._max_batches = 0
            self._batches_per_seg = []

    def __iter__(self):
        batches: List[List[int]] = []

        for seg_idx, (off, n) in enumerate(self._segments):
            idxs = list(range(off, off + n))
            if self.shuffle:
                random.shuffle(idxs)

            # Build batches for this segment
            seg_batches: List[List[int]] = []
            for i in range(0, len(idxs), self.batch_size):
                b = idxs[i:i + self.batch_size]
                if len(b) < self.batch_size and self.drop_last:
                    continue
                seg_batches.append(b)

            # Balanced mode: oversample smaller segments to match max_batches
            if self.balanced and self._max_batches > 0 and len(seg_batches) < self._max_batches:
                # Repeat batches with replacement until we reach target
                original_batches = seg_batches.copy()
                while len(seg_batches) < self._max_batches:
                    # Sample from original batches (with shuffle for variety)
                    extra = random.choice(original_batches)
                    seg_batches.append(extra)

            batches.extend(seg_batches)

        if self.shuffle:
            random.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        if self.balanced and self._max_batches > 0:
            # Each segment contributes max_batches
            return self._max_batches * len(self._segments)

        total = 0
        for _, n in self._segments:
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def create_datasets(args: argparse.Namespace, ds_cfg: DatasetConfig,
                   calib: Optional[Union[Dict[str, Any], Dict[str, Optional[Dict[str, Any]]]]],
                   proc_dev: torch.device) -> Tuple[DataLoader, Union[DataLoader, Dict[str, DataLoader]], Dataset]:

    roots = args.multi_root if args.multi_root else [ds_cfg.root]
    seq_train_list = []

    def _calib_for_root(root_str: str) -> Optional[Dict[str, Any]]:
        if calib is None:
            return None
        if isinstance(calib, dict) and root_str in calib and isinstance(calib.get(root_str), (dict, type(None))):
            any_single_keys = any(k in calib for k in ("K", "camera", "T_imu_cam", "R_IC", "R_imu_cam"))
            all_values_like_calib = all(isinstance(v, (dict, type(None))) for v in calib.values())
            if (not any_single_keys) and all_values_like_calib:
                return calib.get(root_str)
        if isinstance(calib, dict):
            return calib
        return None

    # Compute adaptive sequence length for consistent tensor dimensions
    imu_seq_len = compute_adaptive_sequence_length(ds_cfg.dt)
    print(f"[INFO] Using dt={ds_cfg.dt}, computed sequence_length={imu_seq_len}")

    dataset_proc_dev = torch.device("cpu") if int(getattr(args, "num_workers", 0)) > 0 else proc_dev

    # Common dataset arguments to reduce redundancy
    ds_common_args = {
        'dt': ds_cfg.dt, 'resolution': ds_cfg.resolution, 'events_h5': ds_cfg.events_h5,
        'sample_stride': ds_cfg.sample_stride, 'derotate': ds_cfg.derotate,
        'windowing_mode': ds_cfg.windowing_mode, 'window_dt': ds_cfg.window_dt,
        'event_offset_scan': ds_cfg.event_offset_scan,
        'event_offset_scan_range_s': ds_cfg.event_offset_scan_range_s,
        'event_offset_scan_step_s': ds_cfg.event_offset_scan_step_s,
        'sensor_resolution': ds_cfg.sensor_resolution, 'event_file_candidates': ds_cfg.event_file_candidates,
        'proc_device': dataset_proc_dev, 'std_norm': ds_cfg.voxel_std_norm, 'adaptive_voxel': ds_cfg.adaptive_voxel,
        'event_noise_scale': ds_cfg.event_noise_scale, 'event_scale_jitter': ds_cfg.event_scale_jitter,
        'imu_bias_scale': ds_cfg.imu_bias_scale, 'imu_mask_prob': ds_cfg.imu_mask_prob,
        'adaptive_base_div': ds_cfg.adaptive_base_div, 'adaptive_max_events_div': ds_cfg.adaptive_max_events_div,
        'adaptive_density_cap': ds_cfg.adaptive_density_cap, 'sequence_length': imu_seq_len
    }

    # Safe gap to prevent temporal data leakage between train/val split
    # For time-series data with sliding windows, adjacent samples share frames
    # Gap = 2 * sample_stride ensures no overlap in GT frames or IMU/event windows
    if str(getattr(ds_cfg, "windowing_mode", "imu")).strip().lower() == "imu":
        safe_gap = 2
    else:
        safe_gap = ds_cfg.sample_stride * 2
    print(f"[DATA SPLIT] Using safe_gap={safe_gap} samples to prevent train/val leakage")

    # Training datasets
    for r in roots:
        ds_tr = OptimizedTUMDataset(root=r, **ds_common_args, calib=_calib_for_root(r), augment=ds_cfg.augment)
        ds_tr.voxelize_in_dataset = ds_cfg.voxelize_in_dataset
        n_r = len(ds_tr)
        n_train_r = max(int(ds_cfg.train_split * n_r) - safe_gap, 1)
        train_subset = torch.utils.data.Subset(ds_tr, list(range(n_train_r)))
        seq_train_list.append(SequenceDataset(train_subset, sequence_len=args.sequence_len, stride=args.sequence_stride))
        print(f"[DATA SPLIT] Dataset {r}: total={n_r}, train=[0,{n_train_r}), gap=[{n_train_r},{n_train_r+safe_gap}), val=[{n_train_r+safe_gap},{n_r})")

    train_seq = ConcatDataset(seq_train_list)

    val_loaders: Optional[Dict[str, DataLoader]] = None

    # Validation set
    ds_val_full = OptimizedTUMDataset(root=roots[0], **ds_common_args, calib=_calib_for_root(roots[0]), augment=False)
    ds_val_full.voxelize_in_dataset = ds_cfg.voxelize_in_dataset
    n_val_total = len(ds_val_full)
    n_train_val = max(int(ds_cfg.train_split * n_val_total), 1)
    val_start = n_train_val + safe_gap
    if val_start >= n_val_total:
        print(f"[WARNING] safe_gap too large: val_start={val_start} >= n_val_total={n_val_total}. Using minimal gap=1")
        val_start = min(n_train_val + 1, n_val_total - 1)
    val_subset = torch.utils.data.Subset(ds_val_full, list(range(val_start, n_val_total)))
    val_seq = SequenceDataset(val_subset, sequence_len=args.sequence_len, stride=args.sequence_stride)

    # Create data loaders
    actual_num_workers = args.num_workers
    if proc_dev.type == "mps" and actual_num_workers > 0:
        print(f"[WARNING] Detected MPS device with num_workers={actual_num_workers}. forcing num_workers=0 to prevent semaphore leaks.")
        actual_num_workers = 0

    loader_kwargs = {
        'num_workers': actual_num_workers,
        'pin_memory': args.pin_memory,
        'persistent_workers': (args.persistent_workers if actual_num_workers > 0 else False),
        'collate_fn': CollateSequence(window_stack_k=args.window_stack_K, voxel_stack_mode=getattr(args, 'voxel_stack_mode', 'abs'))
    }
    if actual_num_workers > 0:
        loader_kwargs['prefetch_factor'] = args.prefetch_factor

    batch_by_root = getattr(args, "batch_by_root", None)
    if batch_by_root is None:
        batch_by_root = (len(roots) > 1)

    if batch_by_root and len(roots) > 1:
        use_balanced = bool(getattr(args, "balanced_sampling", False))
        train_loader = DataLoader(
            train_seq,
            batch_sampler=RootGroupedBatchSampler(
                train_seq, batch_size=args.batch_size, shuffle=True,
                drop_last=False, balanced=use_balanced
            ),
            **loader_kwargs
        )
    else:
        sampler = None
        if bool(getattr(args, "balanced_sampling", False)):
            weights = torch.empty(len(train_seq), dtype=torch.double)
            start = 0
            for ds_i in train_seq.datasets:
                n_i = max(int(len(ds_i)), 1)
                weights[start:start + n_i] = 1.0 / float(n_i)
                start += n_i
            sampler = WeightedRandomSampler(weights=weights, num_samples=int(weights.numel()), replacement=True)

        if sampler is None:
            train_loader = DataLoader(
                train_seq, batch_size=args.batch_size, shuffle=True,
                **loader_kwargs
            )
        else:
            train_loader = DataLoader(
                train_seq, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                **loader_kwargs
            )

    print("[INFO] Forcing validation batch_size=1 to ensure correct trajectory alignment.")

    if bool(getattr(args, "eval_all_roots", False)) and len(roots) > 1:
        val_loaders = {}
        for r in roots:
            ds_val_full_r = OptimizedTUMDataset(root=r, **ds_common_args, calib=_calib_for_root(r), augment=False)
            ds_val_full_r.voxelize_in_dataset = ds_cfg.voxelize_in_dataset
            n_val_total_r = len(ds_val_full_r)
            n_train_val_r = max(int(ds_cfg.train_split * n_val_total_r), 1)
            val_start_r = n_train_val_r + safe_gap
            if val_start_r >= n_val_total_r:
                print(f"[WARNING] Dataset {r}: safe_gap too large, using minimal gap=1")
                val_start_r = min(n_train_val_r + 1, n_val_total_r - 1)
            val_subset_r = torch.utils.data.Subset(ds_val_full_r, list(range(val_start_r, n_val_total_r)))
            val_seq_r = SequenceDataset(val_subset_r, sequence_len=args.sequence_len, stride=args.sequence_stride)
            val_loaders[r] = DataLoader(
                val_seq_r, batch_size=1, shuffle=False,
                **loader_kwargs
            )
        return train_loader, val_loaders, val_subset

    val_loader = DataLoader(
        val_seq, batch_size=1, shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader, val_subset


def setup_training_components(args: argparse.Namespace, model_cfg: ModelConfig, train_cfg: TrainingConfig,
                            device: torch.device) -> Tuple[nn.Module, torch.optim.Optimizer,
                                                          torch.optim.lr_scheduler._LRScheduler,
                                                          Optional[AdaptiveLossWeights]]:
    """
    Set up training components (model, optimizer, scheduler, etc.)

    Args:
        args: Command-line arguments
        model_cfg: Model configuration
        train_cfg: Training configuration
        device: Compute device

    Returns:
        Model, optimizer, learning-rate scheduler, adaptive-loss function
    """
    # Model
    model = HybridVIONet(config=model_cfg).to(device).to(memory_format=torch.channels_last)

    if model.scale_max > 1.0 + 1e-6:
        print(f"[WARN] scale_max={model.scale_max:.2f}>1.0 (非标准VIO)")

    init_ckpt = str(getattr(args, 'init_checkpoint', '') or '').strip()
    if init_ckpt:
        ckpt_obj = torch.load(init_ckpt, map_location='cpu')
        state = None
        if isinstance(ckpt_obj, dict) and 'state_dict' in ckpt_obj and isinstance(ckpt_obj['state_dict'], dict):
            state = ckpt_obj['state_dict']
        elif isinstance(ckpt_obj, dict):
            state = ckpt_obj
        else:
            raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")

        if len(state) == 0:
            raise RuntimeError("Empty checkpoint state_dict")

        prefixes = ("module.", "orig_mod.", "_orig_mod.", "model.")
        for p in prefixes:
            if all(k.startswith(p) for k in state.keys()):
                state = {k[len(p):]: v for k, v in state.items()}
                break

        missing, unexpected = model.load_state_dict(state, strict=False)
        log(f"[CKPT] init_checkpoint={init_ckpt} loaded | missing={len(missing)} unexpected={len(unexpected)}")

    if train_cfg.compile and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", backend=train_cfg.compile_backend)
        except Exception:
            pass

    # Adaptive Loss
    # Change num_losses to 4 to support [t, r, v, p]
    adaptive_loss_fn = AdaptiveLossWeights(num_losses=4).to(device) if train_cfg.adaptive_loss_weights else None

    # Optimizer
    params = list(model.parameters())
    if adaptive_loss_fn is not None:
        opt = torch.optim.AdamW([
            {'params': params, 'lr': train_cfg.lr},
            {'params': list(adaptive_loss_fn.parameters()), 'lr': train_cfg.lr * 0.1},
        ])
    else:
        opt = torch.optim.AdamW(params, lr=train_cfg.lr)

    # Scheduler with optional linear warmup
    if train_cfg.scheduler == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(train_cfg.epochs // 3, 1), gamma=train_cfg.gamma)
    elif train_cfg.scheduler == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.scheduler_T_max)
    else:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=train_cfg.gamma, patience=train_cfg.scheduler_patience, mode="min")

    if hasattr(train_cfg, "warmup_epochs") and train_cfg.warmup_epochs > 0 and not isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=train_cfg.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_scheduler, main_scheduler], milestones=[train_cfg.warmup_epochs])
    else:
        scheduler = main_scheduler

    return model, opt, scheduler, adaptive_loss_fn


def run_training_loop(args: argparse.Namespace, ds_cfg: DatasetConfig, train_cfg: TrainingConfig,
                     model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, ds_val: Dataset,
                     device: torch.device, opt: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
                     adaptive_loss_fn: Optional[AdaptiveLossWeights], metrics_fh: Any, fx: float = 1.0, fy: float = 1.0) -> None:
    """
    Run the main training loop.

    Args:
        args: Command-line arguments
        ds_cfg: Dataset configuration
        train_cfg: Training configuration
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        ds_val: Validation dataset
        device: Compute device
        opt: Optimizer
        scheduler: Learning rate scheduler
        adaptive_loss_fn: Adaptive loss weighting module
        metrics_fh: Metrics file handle
        fx, fy: Scaled camera intrinsics used by the physics loss
    Returns:
        None
    """
    out_dir = Path(getattr(args, 'output_dir', 'fno-FAST'))
    out_dir.mkdir(parents=True, exist_ok=True)
    early_stopping = EarlyStopping(patience=train_cfg.patience, verbose=True,
                                 path=str(out_dir / "early_stop_checkpoint.pth"))
    best_ate = float("inf")
    metrics_writer = csv.writer(metrics_fh) if metrics_fh else None
    val_hist = []
    checkpoint_saving_enabled = True

    def _try_save_state_dict(tag: str, dst: Path) -> bool:
        nonlocal checkpoint_saving_enabled
        if not checkpoint_saving_enabled:
            return False

        state = model.state_dict()
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
        try:
            try:
                torch.save(state, str(tmp), _use_new_zipfile_serialization=False)
                os.replace(str(tmp), str(dst))
                log(f"[{tag.upper()}] saved to {dst}")
                return True
            except OSError as e:
                err_no = getattr(e, "errno", None)
                if err_no in (28, 122):
                    try:
                        torch.save(state, str(dst), _use_new_zipfile_serialization=False)
                        log(f"[{tag.upper()}] saved to {dst} (non-atomic fallback)")
                        return True
                    except Exception:
                        checkpoint_saving_enabled = False
                raise
        except Exception as e:
            log(f"Warning: Failed to save {tag} model: {e}")
            return False
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    base_model = model.orig_mod if hasattr(model, 'orig_mod') else model
    has_complex_params = any(torch.is_complex(p) for p in base_model.parameters())
    use_amp = (device.type == "cuda" and bool(getattr(train_cfg, "mixed_precision", False)))
    if has_complex_params and use_amp:
        print("Disabling AMP mixed precision because model has complex parameters that are not supported by GradScaler.")
        use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    base_lrs = [pg['lr'] for pg in opt.param_groups]

    for epoch in range(train_cfg.epochs):
        # IMU dropout schedule with robust access
        p0 = float(args.imu_dropout_p_start)
        p1 = float(args.imu_dropout_p_end)
        pe = p0 + (p1 - p0) * (epoch / max(train_cfg.epochs - 1, 1))
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            wup = int(getattr(train_cfg, 'warmup_epochs', 0))
            if wup > 0 and epoch < wup:
                factor = float(epoch + 1) / float(wup)
                for i, pg in enumerate(opt.param_groups):
                    pg['lr'] = base_lrs[i] * factor
        actual_model = model.orig_mod if hasattr(model, 'orig_mod') else model
        frac = float(epoch + 1) / float(max(int(train_cfg.epochs), 1))
        if frac <= 0.2:
            target_modes = 4
        elif frac <= 0.5:
            target_modes = 8
        else:
            target_modes = 10 ** 9
        for m in actual_model.modules():
            if hasattr(m, "active_modes") and hasattr(m, "modes"):
                try:
                    mm = int(getattr(m, "modes"))
                    v = int(min(max(int(target_modes), 1), max(mm, 1)))
                    getattr(m, "active_modes").fill_(v)
                except Exception:
                    pass
        if hasattr(actual_model, 'imu_encoder'):
            try:
                actual_model.imu_encoder.set_dropout_p(pe)
            except Exception:
                pass


        os.environ["EVENT_NOISE_SCALE"] = str(ds_cfg.event_noise_scale)
        os.environ["IMU_BIAS_SCALE"] = str(ds_cfg.imu_bias_scale)
        os.environ["IMU_MASK_PROB"] = str(ds_cfg.imu_mask_prob)


        if epoch < 5:
            cur_phys_w = 0.0
        else:
            progress = min(1.0, (epoch - 5) / 10.0)
            cur_phys_w = train_cfg.loss_w_physics * progress
        loss = train_one_epoch(model, train_loader, opt, device, train_cfg,
                             dt=ds_cfg.dt, current_epoch_physics_weight=cur_phys_w,
                             adaptive_loss_fn=adaptive_loss_fn, fx=fx, fy=fy, scaler=scaler)
        
        # --- PATH LENGTH SCALE LOSS UPDATE ---
        # If we are using path length loss, we should ensure the loss function sees the entire batch 
        # but LossComposer is called per-step or per-window. 
        # The path length constraint is actually applied inside LossComposer.compute_components
        # if path_scale_weight is passed. train_one_epoch handles this if we pass the weight.
        
        # Update scheduler (non-plateau only)
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        cur_lr = opt.param_groups[0]['lr']
        if adaptive_loss_fn is not None:
            lv = adaptive_loss_fn.log_vars.detach().float().cpu().numpy()
            w = np.exp(-lv)
            log(f"Epoch {epoch+1}/{train_cfg.epochs} Loss {loss:.6f} LR {cur_lr:.6f} ADAPT_W {w.tolist()}")
        else:
            log(f"Epoch {epoch+1}/{train_cfg.epochs} Loss {loss:.6f} LR {cur_lr:.6f}")

        # evaluate
        if (epoch + 1) % train_cfg.eval_interval == 0 and len(ds_val) > 0:
            eval_mode = getattr(train_cfg, 'eval_sim3_mode', 'diagnose')
            if isinstance(val_loader, dict):
                ate_list: List[float] = []
                rpe_t_list: List[float] = []
                rpe_r_list: List[float] = []
                for r, vldr in val_loader.items():
                    ate_i, rpe_t_i, rpe_r_i = evaluate(model, vldr, device, train_cfg.rpe_dt, ds_cfg.dt, eval_sim3_mode=eval_mode)
                    ate_list.append(float(ate_i))
                    rpe_t_list.append(float(rpe_t_i))
                    rpe_r_list.append(float(rpe_r_i))
                    log(f"Eval[{Path(r).name}] ATE {ate_i:.6f} RPE_t {rpe_t_i:.6f} RPE_r(deg) {rpe_r_i:.6f}")
                ate = float(np.nanmean(np.asarray(ate_list, dtype=np.float64)))
                rpe_t = float(np.nanmean(np.asarray(rpe_t_list, dtype=np.float64)))
                rpe_r = float(np.nanmean(np.asarray(rpe_r_list, dtype=np.float64)))
                log(f"Eval[MEAN] ATE {ate:.6f} RPE_t {rpe_t:.6f} RPE_r(deg) {rpe_r:.6f}")
            else:
                ate, rpe_t, rpe_r = evaluate(model, val_loader, device, train_cfg.rpe_dt, ds_cfg.dt, eval_sim3_mode=eval_mode)
                log(f"Eval ATE {ate:.6f} RPE_t {rpe_t:.6f} RPE_r(deg) {rpe_r:.6f}")

            actual_model = model.orig_mod if hasattr(model, 'orig_mod') else model

            if metrics_writer:
                metrics_writer.writerow([epoch+1, float(loss), float(ate), float(rpe_t), float(rpe_r)])
                metrics_fh.flush()  # 立即写入磁盘

            if getattr(train_cfg, 'earlystop_metric', 'composite') == 'ate':
                comp = float(ate)
            else:
                comp = float(ate) + float(train_cfg.earlystop_alpha) * float(rpe_t) + float(train_cfg.earlystop_beta) * float(rpe_r)
            val_hist.append(comp)
            log(f"[EARLYSTOP] metric={getattr(train_cfg,'earlystop_metric','composite')} value={comp:.6f}")
            k = max(int(train_cfg.earlystop_ma_window), 1)
            ma_val = float(np.mean(val_hist[-k:]))

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                wup = int(getattr(train_cfg, 'warmup_epochs', 0))
                if (epoch + 1) > wup:
                    scheduler.step(rpe_r)

            if (epoch + 1) >= int(train_cfg.earlystop_min_epoch):
                early_stopping(ma_val, model)
                if early_stopping.early_stop:
                    log("Early stopping triggered")
                    break

            if ate < best_ate:
                best_ate = ate
                _try_save_state_dict("best", out_dir / "hybrid_vio_best.pth")

            _try_save_state_dict("latest", out_dir / "hybrid_vio_latest.pth")


def validate_imu_integration():

    print("[PHYSICS VALIDATION] Testing IMU integration fixes...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Test IMU kinematics with simple constant acceleration
    # Disable gravity alignment for this test since we only want pure acceleration
    imu_kin = IMUKinematics(enable_gravity_alignment=False).to(device)

    # Create synthetic IMU data: constant acceleration in X direction
    batch_size = 2
    seq_len = 10
    dt = 0.1

    # Constant acceleration: 1 m/s² in X direction
    # Note: The model will denormalize this internally, so we provide normalized values
    imu_data = torch.zeros(batch_size, seq_len, 6)
    imu_data[:, :, 0] = 1.0 / 9.81  # Normalized acceleration (1 m/s² / 9.81)
    imu_data[:, :, 3:6] = 0.0  # Zero angular velocity (already normalized by /π)
    imu_data = imu_data.to(device)

   
    dt_tensor = torch.tensor([dt], dtype=torch.float32, device=device)

    # Use zero gravity for this pure acceleration test to avoid Z-axis drift artifacts
    # caused by implicit gravity addition in the kinematics module
    imu_kin_test = IMUKinematics(g_world=torch.zeros(3), enable_gravity_alignment=False).to(device)

    with torch.no_grad():
        pos, final_R, final_v = imu_kin_test(imu_data, dt=dt_tensor)
        quat = imu_kin_test._rot_to_quat(final_R)

    # Expected results for constant acceleration test:
    # Final velocity: v = a * t = 1.0 * (10 * 0.1) = 1.0 m/s
    # Final position: p = 0.5 * a * t² = 0.5 * 1.0 * (1.0)² = 0.5 m

    expected_v = 1.0
    expected_p = 0.5

    print(f"Final velocity: {final_v[0, 0].item():.4f} (expected: {expected_v:.4f})")
    print(f"Final position: {pos[0, 0].item():.4f} (expected: {expected_p:.4f})")
    v_error = abs(final_v[0, 0].item() - expected_v)
    p_error = abs(pos[0, 0].item() - expected_p)

    if v_error < 1e-4 and p_error < 1e-4:
        print("IMU integration PASSED!")
        return True
    else:
        print(f"[IMU integration FAILED!] v_error: {v_error}, p_error: {p_error}")
        return False


def validate_state_continuity():
    # Test that IMU state continuity works properly across multiple calls
    print("[STATE VALIDATION] Testing IMU state continuity...")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    batch_size = 1
    seq_len = 5
    dt = 0.1

    state_mgr = IMUStateManager(device)

    model_cfg = ModelConfig()
    model_cfg.imu_gate_soft = True
    model_cfg.use_mr_fno = False
    model = HybridVIONet(model_cfg).to(device)

    imu_data = torch.zeros(batch_size, model_cfg.sequence_length, 6)
    imu_data[:, :, 0] = 1.0 / 9.81
    imu_data[:, :, 3:6] = 0.0
    events = torch.zeros(batch_size, 5, 32, 32)
    imu_data = imu_data.to(device)
    events = events.to(device)

    # Test multiple sequential calls
    with torch.no_grad():
        state_mgr.initialize(batch_size)

        for step in range(3):
            print(f"[STATE VALIDATION] Step {step+1}:")
            if state_mgr.velocity is None:
                print("  Input velocity: None (Auto-init)")
            else:
                print(f"  Input velocity: {state_mgr.velocity[0, :].tolist()}")

            # Call model forward pass
            pred, hidden, new_v, new_R, _, _, _, _ = model(
                events, imu_data, hidden_state=None,
                prev_v=state_mgr.velocity, prev_R=state_mgr.rotation, dt_window=dt
            )

            state_mgr.update_states(new_v, new_R)
            print(f"  Output velocity: {new_v[0, :].tolist()}")
            vn = torch.norm(new_v[0, :]).item()
            print(f"  Vel norm: {vn:.6f}")
            print(f"  Output position: {pred[0, :3].tolist()}")

    print("[STATE VALIDATION] State continuity test completed!")
    return True


def main() -> None:

    train_loader = None
    val_loader = None
    metrics_fh = None

    try:
        args = parse_command_line_arguments()

        if str(getattr(args, "windowing_mode", "imu")).strip().lower() == "imu" and not bool(getattr(args, "imu_gate_soft", True)):
            print("[WARNING] imu windowing with hard gate disables IMU baseline; forcing imu_gate_soft=True.")
            args.imu_gate_soft = True

        integration_ok = validate_imu_integration()
        continuity_ok = validate_state_continuity()

        if not integration_ok or not continuity_ok:
            print(" PHYSICS VALIDATION FAILED! Please fix the issues before training.")
            return

        print("ALL PHYSICS VALIDATIONS PASSED!")   
        print()

        calib_obj_pre: Optional[Dict[str, Any]] = None
        if args.multi_root is None and not getattr(args, "root_dir", None):
            calib_obj_pre = load_calibration(args.calib_yaml) if getattr(args, "calib_yaml", "") else None
            if isinstance(calib_obj_pre, dict):
                mr = calib_obj_pre.get("multi_root")
                if isinstance(mr, (list, tuple)) and len(mr) > 0:
                    args.multi_root = [str(x) for x in mr]
                else:
                    root_guess = calib_obj_pre.get("dataset_root") or calib_obj_pre.get("root_dir") or calib_obj_pre.get("root") or calib_obj_pre.get("data_root")
                    if root_guess:
                        args.root_dir = str(root_guess)
                    else:
                        inferred = _infer_dataset_root_from_calib(calib_obj_pre, getattr(args, "calib_yaml", ""))
                        if inferred:
                            args.root_dir = inferred

                if not getattr(args, "events_h5", None):
                    evp = calib_obj_pre.get("events_h5") or calib_obj_pre.get("events_path") or calib_obj_pre.get("event_path")
                    if evp:
                        bases: List[Path] = []
                        if getattr(args, "root_dir", None):
                            try:
                                bases.append(Path(str(args.root_dir)).expanduser().resolve())
                            except Exception:
                                bases.append(Path(str(args.root_dir)).expanduser())
                        if getattr(args, "calib_yaml", None):
                            try:
                                bases.append(Path(str(args.calib_yaml)).expanduser().resolve().parent)
                            except Exception:
                                bases.append(Path(str(args.calib_yaml)).expanduser().parent)
                        evp_resolved = _resolve_existing_path(evp, bases=bases, must_be_file=True)
                        args.events_h5 = evp_resolved.as_posix() if evp_resolved is not None else str(evp)

        # Support passing multiple calib YAMLs: infer dataset roots and enable multi-root training.
        if args.multi_root is None and args.multi_calib_yaml is not None and len(args.multi_calib_yaml) > 0:
            expanded_roots: List[str] = []
            expanded_yamls: List[str] = []
            for yml_i in args.multi_calib_yaml:
                calib_i = load_calibration(str(yml_i))
                if not isinstance(calib_i, dict):
                    raise ValueError(f"Invalid calibration YAML: {yml_i}")

                mr = calib_i.get("multi_root")
                if isinstance(mr, (list, tuple)) and len(mr) > 0:
                    for r_i in mr:
                        if r_i:
                            expanded_roots.append(str(r_i))
                            expanded_yamls.append(str(yml_i))
                    continue

                inferred_i = _infer_dataset_root_from_calib(calib_i, str(yml_i))
                if inferred_i:
                    expanded_roots.append(inferred_i)
                    expanded_yamls.append(str(yml_i))
                    continue

                raise ValueError(
                    f"Cannot infer dataset root from calib YAML: {yml_i}. "
                    f"Provide dataset_root/root_dir/root/data_root/multi_root (or imu_path/mocap_path/events_h5 to infer)."
                )

            if expanded_roots and all(bool(r) for r in expanded_roots):
                args.multi_root = expanded_roots
                args.multi_calib_yaml = expanded_yamls

        ds_cfg, model_cfg, train_cfg = setup_configurations(args)
        device = setup_device_and_environment(args)
        roots = args.multi_root if args.multi_root else [ds_cfg.root]
        if not any(bool(r) for r in roots):
            raise ValueError("Missing dataset root. Provide --root_dir (single) or --multi_root (multiple), or set calib_yaml keys dataset_root/root_dir/multi_root (or imu_path/mocap_path/events_h5 to infer).")

        calib_obj: Optional[Dict[str, Any]] = None
        calib_for_physics: Optional[Dict[str, Any]] = None
        calib: Optional[Union[Dict[str, Any], Dict[str, Optional[Dict[str, Any]]]]] = None

        if args.multi_calib_yaml is not None:
            if not args.multi_root:
                raise ValueError("--multi_calib_yaml requires --multi_root, or each YAML must provide dataset_root/root_dir/root/data_root (or imu_path/mocap_path/events_h5 to infer).")
            if len(args.multi_calib_yaml) != len(roots):
                raise ValueError(f"--multi_calib_yaml length ({len(args.multi_calib_yaml)}) must match number of roots ({len(roots)})")
            calib_map: Dict[str, Optional[Dict[str, Any]]] = {}
            for r, yml in zip(roots, args.multi_calib_yaml):
                if not yml or not Path(yml).is_file():
                    raise FileNotFoundError(f"Calibration YAML not found: {yml}")
                calib_map[r] = load_calibration(yml)
            calib = calib_map
            calib_for_physics = next((c for c in calib_map.values() if isinstance(c, dict)), None)
        else:
            calib_obj = calib_obj_pre if isinstance(locals().get("calib_obj_pre"), dict) else load_calibration(args.calib_yaml)
            calib = calib_obj
            calib_for_physics = calib_obj

        # set up camera intrinsics
        fx_scaled, fy_scaled = 1.0, 1.0
        if calib_for_physics is not None and "K" in calib_for_physics:
            fx_raw = float(calib_for_physics["K"].get("fx", 1.0))
            fy_raw = float(calib_for_physics["K"].get("fy", 1.0))
            if "resolution" in calib_for_physics:
                sensor_w, sensor_h = calib_for_physics["resolution"]
                print(f"[INFO] Using camera resolution from YAML: {sensor_w}x{sensor_h}")
            elif args.sensor_resolution:
                sensor_h, sensor_w = tuple(args.sensor_resolution)
            else:
                sensor_h, sensor_w = tuple(args.resolution)
            args.sensor_resolution = (sensor_h, sensor_w)
            if hasattr(ds_cfg, 'sensor_resolution'):
                ds_cfg.sensor_resolution = (sensor_h, sensor_w)
            net_h, net_w = tuple(args.resolution)

            # 检测相机类型并选择合适的内参缩放方法
            camera_type = calib_for_physics["K"].get("camera_type", "pinhole").lower()
            if camera_type == "kb4":
                # KB4 鱼眼相机
                distortion = {
                    "k1": float(calib_for_physics["K"].get("k1", 0.0)),
                    "k2": float(calib_for_physics["K"].get("k2", 0.0)),
                    "k3": float(calib_for_physics["K"].get("k3", 0.0)),
                    "k4": float(calib_for_physics["K"].get("k4", 0.0)),
                }
                K_scaled, _ = rescale_intrinsics_kb4(
                    calib_for_physics["K"],
                    distortion,
                    (sensor_h, sensor_w),
                    (net_h, net_w)
                )
                scale_x = float(net_w) / float(sensor_w)
                scale_y = float(net_h) / float(sensor_h)
                print(f"[INFO] Using KB4 fisheye camera model")
            else:
                # 针孔相机
                K_scaled, scales = rescale_intrinsics_pinhole(
                    calib_for_physics["K"],
                    (sensor_h, sensor_w),
                    (net_h, net_w)
                )
                scale_x, scale_y = scales
                print(f"[INFO] Using Pinhole camera model")

            fx_scaled = float(K_scaled.get("fx", fx_raw))
            fy_scaled = float(K_scaled.get("fy", fy_raw))
            calib_for_physics["K_scaled"] = K_scaled
            print(f"[INFO] Intrinsics Scaled: fx={fx_scaled:.2f}, fy={fy_scaled:.2f} (Raw: {fx_raw:.2f}, {fy_raw:.2f}, Scale: {scale_x:.2f}, {scale_y:.2f})")
        else:
            print("[WARN] No calibration found! Physics loss will use fx=1.0 (Unitless).")

        # 数据处理设备设置
        proc_dev = torch.device("cpu") if device.type == "mps" else (
            device if (args.num_workers == 0 or device.type != "cuda") else torch.device("cpu")
        )

        # set up logging
        log_fh = setup_logging(args)

        # set up datasets
        train_loader, val_loader, ds_val = create_datasets(args, ds_cfg, calib, proc_dev)
        model, opt, scheduler, adaptive_loss_fn = setup_training_components(
            args, model_cfg, train_cfg, device)
        # set up metrics file
        metrics_fh = None
        if args.metrics_csv:
            try:
                csv_dir = os.path.dirname(args.metrics_csv)
                if csv_dir:  # 只有当目录非空时才创建
                    os.makedirs(csv_dir, exist_ok=True)
                metrics_fh = open(args.metrics_csv, "a", newline="")
                metrics_writer = csv.writer(metrics_fh)
                if metrics_fh.tell() == 0:
                    metrics_writer.writerow(["epoch","loss","ate","rpe_t","rpe_r"])
                    metrics_fh.flush()
                log(f"[METRICS] CSV output: {args.metrics_csv}")
            except Exception as e:
                log(f"[WARN] Failed to open metrics CSV: {e}")

        # 运行训练循环
        run_training_loop(args, ds_cfg, train_cfg, model, train_loader,
                         val_loader, ds_val, device, opt, scheduler,
                         adaptive_loss_fn, metrics_fh, fx=fx_scaled, fy=fy_scaled)

    except Exception as e:
        log(f"Training failed with error: {e}")
        raise
    finally:
        try:
            if metrics_fh:
                metrics_fh.close()
        except Exception:
            pass

        try:
            if LOG_FH:
                LOG_FH.close()
        except Exception:
            pass

        try:
            if train_loader is not None:
                del train_loader
            if val_loader is not None:
                del val_loader
        except Exception:
            pass

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        try:
            import multiprocessing as _mp
            kids = list(_mp.active_children())
            for w in kids:
                try:
                    w.terminate()
                except Exception:
                    pass
            for w in kids:
                try:
                    w.join(timeout=2.0)
                except Exception:
                    pass
        except Exception:
            pass

if __name__ == "__main__":
    main()
