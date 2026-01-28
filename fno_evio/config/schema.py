"""
Dataclass-based configuration schema for FNO-EVIO.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DatasetConfig:
    """Dataset-related configuration."""

    root: str
    dataset_kind: str = "tum"
    multi_root: Optional[List[str]] = None
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
    calib: Optional[Dict[str, Any]] = None


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    modes: int = 10
    stem_channels: int = 64
    imu_embed_dim: int = 64
    lstm_hidden: int = 128
    lstm_layers: int = 2
    imu_channels: int = 6
    sequence_length: int = 50
    attn_groups: int = 8
    imu_gn_groups: Optional[int] = None
    norm_mode: str = "gn"
    fast_fft: bool = False
    state_aug: bool = False
    imu_gate_soft: bool = True
    use_uncertainty_fusion: bool = False
    uncertainty_use_gate: bool = False
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
    scale_min: float = 0.0
    scale_max: float = 1.0


@dataclass
class TrainingConfig:
    """Training hyper-parameters and evaluation configuration."""

    epochs: int = 10
    batch_size: int = 2
    lr: float = 1e-3
    eval_interval: int = 1
    eval_batch_size: Optional[int] = None
    metrics_csv: Optional[str] = None
    export_torchscript: bool = True
    init_checkpoint: Optional[str] = None
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    sequence_len: int = 200
    sequence_stride: int = 200
    batch_by_root: bool = True
    balanced_sampling: bool = False
    eval_all_roots: bool = False
    loss_w_t: float = 8.0
    loss_w_r: float = 2.0
    loss_w_v: float = 0.1
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
    adaptive_loss_weights: bool = False
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
    loss_w_scale: float = 0.0
    loss_w_scale_reg: float = 0.0
    use_seq_scale: bool = False
    seq_scale_reg: float = 0.0
    min_step_threshold: float = 0.0
    min_step_weight: float = 0.0
    eval_sim3_mode: str = "diagnose"
    loss_w_path_scale: float = 0.0
    loss_w_static: float = 0.0
    loss_w_bias_a: float = 1e-4
    loss_w_bias_g: float = 1e-4
    loss_w_uncertainty: float = 0.0
    loss_w_uncertainty_calib: float = 0.0
    loss_w_correction: float = 0.0
    loss_w_bias_smooth: float = 0.0
    bias_prior_accel: Optional[Tuple[float, float, float]] = None
    bias_prior_gyro: Optional[Tuple[float, float, float]] = None


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 0
    device: str = "cuda"
    output_dir: Optional[str] = None
