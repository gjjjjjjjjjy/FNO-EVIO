#!/usr/bin/env python3
"""
几何/归一化/网格一致性诊断脚本

目标：定位事件体素/去畸变/坐标映射中的根因问题
输出：诊断报告 + 可视化对比图

Author: Claude Code
Created: 2026-02-02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def diagnose_time_normalization(
    events: np.ndarray,  # (N, 4) x, y, t, p
    t_start: float,
    t_end: float,
    window_dt: float,
) -> Dict[str, Any]:
    """检查时间归一化是否正确覆盖 [0, 1]"""
    t = events[:, 2]

    # 原始时间范围
    t_min, t_max = t.min(), t.max()
    t_range = t_max - t_min

    # 归一化后的时间
    if t_range > 1e-9:
        t_norm = (t - t_min) / t_range
    else:
        t_norm = np.zeros_like(t)

    # 检查边界饱和
    eps = 0.01
    n_at_zero = np.sum(t_norm < eps)
    n_at_one = np.sum(t_norm > 1 - eps)
    saturation_ratio = (n_at_zero + n_at_one) / len(t_norm) if len(t_norm) > 0 else 0

    return {
        "t_min_raw": float(t_min),
        "t_max_raw": float(t_max),
        "t_range": float(t_range),
        "window_dt": float(window_dt),
        "t_range_vs_window_dt": float(t_range / window_dt) if window_dt > 0 else float("nan"),
        "t_norm_min": float(t_norm.min()) if len(t_norm) > 0 else float("nan"),
        "t_norm_max": float(t_norm.max()) if len(t_norm) > 0 else float("nan"),
        "n_events": len(t),
        "n_at_zero": int(n_at_zero),
        "n_at_one": int(n_at_one),
        "saturation_ratio": float(saturation_ratio),
        "status": "OK" if saturation_ratio < 0.1 else "WARNING: high boundary saturation",
    }


def diagnose_imu_normalization(
    imu_raw: np.ndarray,  # (T, 6) or (T, 7) with timestamp
    expected_acc_norm: float = 9.81,
    expected_gyro_norm: float = 0.5,  # typical for slow motion
) -> Dict[str, Any]:
    """检查 IMU 归一化前后的量级"""
    if imu_raw.shape[1] >= 6:
        # Assume format: [gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z] or with timestamp
        if imu_raw.shape[1] == 7:
            gyro = imu_raw[:, 1:4]
            acc = imu_raw[:, 4:7]
        else:
            gyro = imu_raw[:, 0:3]
            acc = imu_raw[:, 3:6]
    else:
        return {"error": f"Unexpected IMU shape: {imu_raw.shape}"}

    acc_norms = np.linalg.norm(acc, axis=1)
    gyro_norms = np.linalg.norm(gyro, axis=1)

    acc_mean_norm = float(np.mean(acc_norms))
    gyro_mean_norm = float(np.mean(gyro_norms))

    # 检查是否在预期范围
    acc_ratio = acc_mean_norm / expected_acc_norm

    return {
        "acc_mean_norm": acc_mean_norm,
        "acc_std_norm": float(np.std(acc_norms)),
        "acc_min_norm": float(np.min(acc_norms)),
        "acc_max_norm": float(np.max(acc_norms)),
        "gyro_mean_norm": gyro_mean_norm,
        "gyro_std_norm": float(np.std(gyro_norms)),
        "gyro_min_norm": float(np.min(gyro_norms)),
        "gyro_max_norm": float(np.max(gyro_norms)),
        "acc_ratio_to_gravity": acc_ratio,
        "status": "OK" if 0.8 < acc_ratio < 1.2 else f"WARNING: acc_ratio={acc_ratio:.2f}",
    }


def diagnose_kb4_geometry(
    events_xy: np.ndarray,  # (N, 2) sensor pixel coordinates
    K_sensor: np.ndarray,  # 3x3 intrinsic matrix (sensor resolution)
    K_scaled: np.ndarray,  # 3x3 intrinsic matrix (network resolution)
    kb4_params: Tuple[float, float, float, float],  # k1, k2, k3, k4
    sensor_resolution: Tuple[int, int],  # (H, W)
    network_resolution: Tuple[int, int],  # (H, W)
) -> Dict[str, Any]:
    """检查 KB4 去畸变的几何一致性"""
    H_s, W_s = sensor_resolution
    H_n, W_n = network_resolution

    fx_s, fy_s = K_sensor[0, 0], K_sensor[1, 1]
    cx_s, cy_s = K_sensor[0, 2], K_sensor[1, 2]

    fx_n, fy_n = K_scaled[0, 0], K_scaled[1, 1]
    cx_n, cy_n = K_scaled[0, 2], K_scaled[1, 2]

    # 检查缩放比例一致性
    scale_x = W_n / W_s
    scale_y = H_n / H_s

    fx_ratio = fx_n / fx_s if fx_s > 0 else float("nan")
    fy_ratio = fy_n / fy_s if fy_s > 0 else float("nan")
    cx_ratio = cx_n / cx_s if cx_s > 0 else float("nan")
    cy_ratio = cy_n / cy_s if cy_s > 0 else float("nan")

    # 边界事件检查
    x, y = events_xy[:, 0], events_xy[:, 1]
    n_at_edge = np.sum((x < 5) | (x > W_s - 5) | (y < 5) | (y > H_s - 5))
    edge_ratio = n_at_edge / len(x) if len(x) > 0 else 0

    # 检查内参缩放是否与分辨率缩放一致
    fx_scale_error = abs(fx_ratio - scale_x) / scale_x if scale_x > 0 else float("nan")
    fy_scale_error = abs(fy_ratio - scale_y) / scale_y if scale_y > 0 else float("nan")
    cx_scale_error = abs(cx_ratio - scale_x) / scale_x if scale_x > 0 else float("nan")
    cy_scale_error = abs(cy_ratio - scale_y) / scale_y if scale_y > 0 else float("nan")

    max_scale_error = max(fx_scale_error, fy_scale_error, cx_scale_error, cy_scale_error)

    return {
        "sensor_resolution": list(sensor_resolution),
        "network_resolution": list(network_resolution),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "fx_sensor": float(fx_s),
        "fy_sensor": float(fy_s),
        "cx_sensor": float(cx_s),
        "cy_sensor": float(cy_s),
        "fx_scaled": float(fx_n),
        "fy_scaled": float(fy_n),
        "cx_scaled": float(cx_n),
        "cy_scaled": float(cy_n),
        "fx_ratio": float(fx_ratio),
        "fy_ratio": float(fy_ratio),
        "cx_ratio": float(cx_ratio),
        "cy_ratio": float(cy_ratio),
        "fx_scale_error": float(fx_scale_error),
        "fy_scale_error": float(fy_scale_error),
        "cx_scale_error": float(cx_scale_error),
        "cy_scale_error": float(cy_scale_error),
        "max_scale_error": float(max_scale_error),
        "kb4_params": list(kb4_params),
        "n_events": len(x),
        "n_at_edge": int(n_at_edge),
        "edge_ratio": float(edge_ratio),
        "status": "OK" if max_scale_error < 0.01 else f"WARNING: intrinsic scale mismatch {max_scale_error:.4f}",
    }


def diagnose_grid_mapping(
    coords: np.ndarray,  # (N, 2) continuous coordinates
    src_size: Tuple[int, int],  # (H, W) source
    dst_size: Tuple[int, int],  # (H, W) destination
    method: str = "truncate",  # "truncate" or "round"
) -> Dict[str, Any]:
    """检查网格映射的端点对齐和边界处理"""
    H_s, W_s = src_size
    H_d, W_d = dst_size

    x, y = coords[:, 0], coords[:, 1]

    # 当前映射方式 (x * W_d / W_s)
    x_mapped_current = x * W_d / W_s
    y_mapped_current = y * H_d / H_s

    # 端点对齐映射 ((W_d - 1) / (W_s - 1))
    x_mapped_aligned = x * (W_d - 1) / (W_s - 1) if W_s > 1 else x
    y_mapped_aligned = y * (H_d - 1) / (H_s - 1) if H_s > 1 else y

    # 计算差异
    x_diff = np.abs(x_mapped_current - x_mapped_aligned)
    y_diff = np.abs(y_mapped_current - y_mapped_aligned)

    # 边界检查
    if method == "truncate":
        x_idx = np.floor(x_mapped_current).astype(int)
        y_idx = np.floor(y_mapped_current).astype(int)
    else:
        x_idx = np.round(x_mapped_current).astype(int)
        y_idx = np.round(y_mapped_current).astype(int)

    n_out_of_bounds = np.sum((x_idx < 0) | (x_idx >= W_d) | (y_idx < 0) | (y_idx >= H_d))

    # 最大端点误差（在边界处最明显）
    edge_mask = (x > W_s - 2) | (y > H_s - 2)
    edge_x_diff = x_diff[edge_mask].mean() if edge_mask.any() else 0
    edge_y_diff = y_diff[edge_mask].mean() if edge_mask.any() else 0

    return {
        "src_size": list(src_size),
        "dst_size": list(dst_size),
        "method": method,
        "x_diff_mean": float(x_diff.mean()),
        "x_diff_max": float(x_diff.max()),
        "y_diff_mean": float(y_diff.mean()),
        "y_diff_max": float(y_diff.max()),
        "edge_x_diff_mean": float(edge_x_diff),
        "edge_y_diff_mean": float(edge_y_diff),
        "n_out_of_bounds": int(n_out_of_bounds),
        "out_of_bounds_ratio": float(n_out_of_bounds / len(x)) if len(x) > 0 else 0,
        "status": "OK" if x_diff.max() < 1.0 else f"WARNING: max endpoint diff = {x_diff.max():.2f}",
    }


def diagnose_voxel_normalization(
    voxel: np.ndarray,  # (C, H, W) voxel grid
    log_norm: bool = True,
    std_norm: bool = False,
) -> Dict[str, Any]:
    """检查体素归一化状态"""
    C, H, W = voxel.shape

    # 分通道统计
    channel_stats = []
    for c in range(C):
        ch = voxel[c]
        channel_stats.append({
            "channel": c,
            "min": float(ch.min()),
            "max": float(ch.max()),
            "mean": float(ch.mean()),
            "std": float(ch.std()),
            "n_nonzero": int(np.sum(ch != 0)),
            "sparsity": float(1 - np.sum(ch != 0) / ch.size),
        })

    # 整体统计
    overall_min = float(voxel.min())
    overall_max = float(voxel.max())
    overall_mean = float(voxel.mean())
    overall_std = float(voxel.std())

    # 检查归一化状态
    if log_norm:
        # log_norm 后值应该在 [0, ~log(max_count)] 范围
        expected_range = "log_norm: values should be in [0, ~5] for typical event counts"
        status = "OK" if overall_max < 10 else f"WARNING: max={overall_max:.2f} suggests log_norm not applied"
    elif std_norm:
        # std_norm 后值应该接近标准正态分布
        expected_range = "std_norm: values should be roughly N(0,1)"
        status = "OK" if -5 < overall_mean < 5 and 0.5 < overall_std < 2 else f"WARNING: mean={overall_mean:.2f}, std={overall_std:.2f}"
    else:
        expected_range = "raw counts"
        status = "OK"

    return {
        "shape": list(voxel.shape),
        "overall_min": overall_min,
        "overall_max": overall_max,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "channel_stats": channel_stats,
        "log_norm": log_norm,
        "std_norm": std_norm,
        "expected_range": expected_range,
        "status": status,
    }


def run_diagnostics(
    dataset,
    sample_indices: List[int],
    output_dir: str,
) -> Dict[str, Any]:
    """运行完整诊断"""
    import torch

    os.makedirs(output_dir, exist_ok=True)

    results = {
        "n_samples": len(sample_indices),
        "samples": [],
    }

    for idx in sample_indices:
        print(f"[DIAG] Processing sample {idx}...")

        sample = dataset[idx]
        if len(sample) >= 3:
            voxel, imu, y = sample[0], sample[1], sample[2]
        else:
            print(f"[DIAG] Sample {idx} has unexpected format, skipping")
            continue

        sample_result = {
            "index": idx,
        }

        # Voxel diagnostics
        if isinstance(voxel, torch.Tensor):
            voxel_np = voxel.numpy()
        else:
            voxel_np = np.array(voxel)

        sample_result["voxel"] = diagnose_voxel_normalization(
            voxel_np,
            log_norm=getattr(dataset, "log_norm", True),
            std_norm=getattr(dataset, "std_norm", False),
        )

        # IMU diagnostics
        if isinstance(imu, torch.Tensor):
            imu_np = imu.numpy()
        else:
            imu_np = np.array(imu)

        if imu_np.ndim == 2:
            sample_result["imu"] = diagnose_imu_normalization(imu_np)

        results["samples"].append(sample_result)

    # Summary
    voxel_issues = sum(1 for s in results["samples"] if "WARNING" in s.get("voxel", {}).get("status", ""))
    imu_issues = sum(1 for s in results["samples"] if "WARNING" in s.get("imu", {}).get("status", ""))

    results["summary"] = {
        "voxel_issues": voxel_issues,
        "imu_issues": imu_issues,
        "total_issues": voxel_issues + imu_issues,
    }

    # Save results
    output_path = os.path.join(output_dir, "diagnostics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[DIAG] Results saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="几何/归一化诊断")
    parser.add_argument("--calib_yaml", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="diagnostics_output")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from fno_evio.config.calib import infer_dataset_root_from_calib, load_calibration
    from fno_evio.config.loader import load_experiment_config
    from fno_evio.data.datasets import OptimizedTUMDataset

    # Load config
    config_path = args.config or str(project_root / "configs" / "base.yaml")
    cfg = load_experiment_config(config_path)

    # Load calibration
    calib = load_calibration(args.calib_yaml)
    root = infer_dataset_root_from_calib(calib, args.calib_yaml)

    cfg.dataset.root = str(Path(root).expanduser().resolve())
    cfg.dataset.calib = calib

    print(f"[DIAG] Dataset root: {cfg.dataset.root}")

    # Create dataset
    dataset = OptimizedTUMDataset(
        root=cfg.dataset.root,
        dt=float(cfg.dataset.dt),
        resolution=tuple(cfg.dataset.resolution),
        sensor_resolution=cfg.dataset.sensor_resolution,
        sample_stride=int(cfg.dataset.sample_stride),
        windowing_mode=str(cfg.dataset.windowing_mode),
        window_dt=cfg.dataset.window_dt,
        calib=calib,
        voxelize_in_dataset=True,
        augment=False,
        std_norm=bool(cfg.dataset.voxel_std_norm),
        log_norm=True,
    )

    print(f"[DIAG] Dataset size: {len(dataset)}")

    # Select samples (uniform + edge cases)
    n = len(dataset)
    n_samples = min(args.n_samples, n)

    # Include first, last, and random samples
    indices = [0, n - 1] + list(np.random.choice(range(1, n - 1), size=max(0, n_samples - 2), replace=False))
    indices = sorted(set(indices))[:n_samples]

    print(f"[DIAG] Selected samples: {indices}")

    # Run diagnostics
    results = run_diagnostics(dataset, indices, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Samples analyzed: {results['n_samples']}")
    print(f"Voxel issues: {results['summary']['voxel_issues']}")
    print(f"IMU issues: {results['summary']['imu_issues']}")
    print(f"Total issues: {results['summary']['total_issues']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
