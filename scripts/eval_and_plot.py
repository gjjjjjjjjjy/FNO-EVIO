#!/usr/bin/env python3
"""
评估已训练的模型并生成轨迹图。

用法:
    python scripts/eval_and_plot.py --checkpoint outputs/xxx/hybrid_vio_best.pth \
        --calib_yaml yaml/mocap-6dof_calib.yaml --output_dir outputs/xxx/eval_plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def build_full_dataset_loader(
    cfg,  # ExperimentConfig
    device,
):
    """构建包含全部数据的评估 DataLoader"""
    import torch
    from torch.utils.data import DataLoader

    from fno_evio.data.datasets import OptimizedTUMDataset
    from fno_evio.data.sequence import CollateSequence, SequenceDataset

    ds_kwargs: Dict[str, Any] = {
        "root": str(cfg.dataset.root),
        "events_h5": cfg.dataset.events_h5,
        "dt": float(cfg.dataset.dt),
        "resolution": tuple(cfg.dataset.resolution),
        "sensor_resolution": cfg.dataset.sensor_resolution,
        "sample_stride": int(cfg.dataset.sample_stride),
        "windowing_mode": str(cfg.dataset.windowing_mode),
        "window_dt": cfg.dataset.window_dt,
        "event_offset_scan": bool(cfg.dataset.event_offset_scan),
        "event_offset_scan_range_s": float(cfg.dataset.event_offset_scan_range_s),
        "event_offset_scan_step_s": float(cfg.dataset.event_offset_scan_step_s),
        "voxelize_in_dataset": bool(cfg.dataset.voxelize_in_dataset),
        "derotate": bool(cfg.dataset.derotate),
        "calib": cfg.dataset.calib,
        "event_file_candidates": tuple(cfg.dataset.event_file_candidates),
        "proc_device": torch.device("cpu"),  # 评估时用 CPU 预处理
        "std_norm": bool(cfg.dataset.voxel_std_norm),
        "log_norm": True,
        "augment": False,  # 评估时关闭增强
        "adaptive_voxel": bool(cfg.dataset.adaptive_voxel),
        "event_noise_scale": 0.0,  # 评估时关闭噪声
        "event_scale_jitter": 0.0,
        "imu_bias_scale": 0.0,
        "imu_mask_prob": 0.0,
        "adaptive_base_div": int(cfg.dataset.adaptive_base_div),
        "adaptive_max_events_div": int(cfg.dataset.adaptive_max_events_div),
        "adaptive_density_cap": float(cfg.dataset.adaptive_density_cap),
        "sequence_length": int(cfg.model.sequence_length),
    }

    # 创建完整数据集（不做 train/val 分割）
    ds_full = OptimizedTUMDataset(**ds_kwargs)
    print(f"[EVAL] Full dataset has {len(ds_full)} samples")

    # 包装成 SequenceDataset
    seq_ds = SequenceDataset(
        ds_full,
        sequence_len=int(cfg.training.sequence_len),
        stride=int(cfg.training.sequence_stride),
    )
    print(f"[EVAL] SequenceDataset has {len(seq_ds)} sequences")

    # 创建 DataLoader
    collate_fn = CollateSequence(
        window_stack_k=int(cfg.model.window_stack_K),
        voxel_stack_mode=str(cfg.model.voxel_stack_mode),
    )

    eval_bs = int(cfg.training.eval_batch_size) if cfg.training.eval_batch_size is not None else 1
    loader = DataLoader(
        seq_ds,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # 评估时单线程，避免问题
        pin_memory=False,
        collate_fn=collate_fn,
    )

    return loader


def main():
    parser = argparse.ArgumentParser(description="评估模型并生成轨迹图")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--calib_yaml", type=str, required=True, help="标定文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--full_dataset", action="store_true", help="使用完整数据集（不分割）")
    parser.add_argument("--train_split", type=float, default=None, help="覆盖 train_split")
    args = parser.parse_args()

    # 设置环境变量启用绘图
    os.environ["FNO_EVIO_EVAL_OUTDIR"] = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    import numpy as np

    from fno_evio.config.calib import infer_dataset_root_from_calib, load_calibration
    from fno_evio.config.loader import load_experiment_config, to_legacy_model_config
    from fno_evio.eval.evaluate import evaluate, plot_eval

    # 加载配置
    config_path = args.config or str(project_root / "configs" / "base.yaml")
    cfg = load_experiment_config(config_path)

    # 覆盖 train_split（如果指定）
    if args.train_split is not None:
        cfg.dataset.train_split = args.train_split

    # 加载标定
    calib = load_calibration(args.calib_yaml)
    if calib is None:
        raise RuntimeError(f"Failed to load calibration: {args.calib_yaml}")

    # 推断数据集根目录
    root = infer_dataset_root_from_calib(calib, args.calib_yaml)
    if not root:
        raise ValueError("Cannot infer dataset root from calibration file")

    cfg.dataset.root = str(Path(root).expanduser().resolve())
    cfg.dataset.calib = calib

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using device: {device}")
    print(f"[EVAL] Dataset root: {cfg.dataset.root}")
    print(f"[EVAL] Checkpoint: {args.checkpoint}")
    print(f"[EVAL] Output dir: {args.output_dir}")

    # 构建数据加载器
    if args.full_dataset:
        print(f"[EVAL] Using FULL dataset (no train/val split)")
        eval_loader = build_full_dataset_loader(cfg, device)
    else:
        from fno_evio.app import build_dataloaders
        print(f"[EVAL] train_split: {cfg.dataset.train_split}")
        train_loader, val_loader = build_dataloaders(
            cfg.dataset,
            training=cfg.training,
            model=cfg.model,
            device=device,
        )
        eval_loader = val_loader
        print(f"[EVAL] Using validation split")

        if isinstance(eval_loader, dict):
            # 多数据集情况，取第一个
            eval_loader = list(eval_loader.values())[0]

    # 构建 legacy ModelConfig（HybridVIONet 需要）
    model_cfg = to_legacy_model_config(cfg.model)

    # 加载模型
    from fno_evio.legacy.train_fno_vio import HybridVIONet

    model = HybridVIONet(model_cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[EVAL] Loaded checkpoint | missing={len(missing)} unexpected={len(unexpected)}")

    model = model.to(device)
    model.eval()

    # 获取评估参数
    dt_window = float(cfg.dataset.dt)
    rpe_dt = float(getattr(cfg.training, "rpe_dt", 0.5))
    eval_sim3_mode = str(getattr(cfg.training, "eval_sim3_mode", "diagnose"))
    speed_thresh = float(getattr(cfg.training, "speed_thresh", 0.0))

    # 评估
    print("[EVAL] Running evaluation...")
    ate, rpe_t, rpe_r, state = evaluate(
        model=model,
        loader=eval_loader,
        device=device,
        rpe_dt=rpe_dt,
        dt=dt_window,
        eval_sim3_mode=eval_sim3_mode,
        speed_thresh=speed_thresh,
    )

    print(f"\n{'='*50}")
    print(f"[RESULT] ATE:   {ate:.4f} m")
    print(f"[RESULT] RPE_t: {rpe_t:.4f}")
    print(f"[RESULT] RPE_r: {rpe_r:.4f}")
    print(f"{'='*50}")

    # 保存指标
    metrics_path = os.path.join(args.output_dir, "eval_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"ATE: {ate:.6f}\n")
        f.write(f"RPE_t: {rpe_t:.6f}\n")
        f.write(f"RPE_r: {rpe_r:.6f}\n")
        for k, v in state.items():
            f.write(f"{k}: {v}\n")
    print(f"[EVAL] Metrics saved to {metrics_path}")

    # 轨迹文件会通过 FNO_EVIO_EVAL_OUTDIR 自动保存
    print(f"[EVAL] Plots and trajectories saved to {args.output_dir}")


if __name__ == "__main__":
    main()
