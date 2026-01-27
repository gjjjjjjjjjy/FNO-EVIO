"""
Top-level training script for the refactored FNO-EVIO codebase.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from fno_evio.app import main
from fno_evio.config.calib import infer_dataset_root_from_calib, load_calibration
from fno_evio.config.schema import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig
from fno_evio.config.loader import load_experiment_config, merge_cli_overrides


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("FNO-EVIO training (refactored)")
    sup = argparse.SUPPRESS

    p.add_argument("--config", type=str, default=None, help="YAML config path")

    p.add_argument("--root", type=str, default=sup, help="Dataset root directory")
    p.add_argument("--multi_root", type=str, nargs="+", default=sup, help="Multiple dataset roots")
    p.add_argument("--events_h5", type=str, default=sup, help="Explicit events HDF5 path (optional)")
    p.add_argument("--dt", type=float, default=sup, help="Temporal window (seconds)")
    p.add_argument("--resolution", type=int, nargs=2, default=sup, help="Network resolution H W")
    p.add_argument("--sensor_resolution", type=int, nargs=2, default=sup, help="Sensor resolution H W (optional)")
    p.add_argument("--sample_stride", type=int, default=sup, help="Sample stride (GT-windowing mode)")
    p.add_argument("--windowing_mode", type=str, choices=["imu", "gt"], default=sup)
    p.add_argument("--window_dt", type=float, default=sup, help="Explicit window dt for imu windowing (optional)")
    p.add_argument("--train_split", type=float, default=sup)
    p.add_argument("--augment", type=int, choices=[0, 1], default=sup)
    p.add_argument("--derotate", type=int, choices=[0, 1], default=sup)
    p.add_argument("--voxelize_in_dataset", type=int, choices=[0, 1], default=sup)

    p.add_argument("--epochs", type=int, default=sup)
    p.add_argument("--batch_size", type=int, default=sup)
    p.add_argument("--eval_batch_size", type=int, default=sup)
    p.add_argument("--lr", type=float, default=sup)
    p.add_argument("--eval_interval", type=int, default=sup)
    p.add_argument("--device", type=str, default=sup)
    p.add_argument("--seed", type=int, default=sup)
    p.add_argument("--output_dir", type=str, default=sup)
    p.add_argument("--init_checkpoint", type=str, default=sup)
    p.add_argument("--metrics_csv", type=str, default=sup)
    p.add_argument("--export_torchscript", type=int, nargs="?", const=1, choices=[0, 1], default=sup)

    p.add_argument("--num_workers", type=int, default=sup)
    p.add_argument("--pin_memory", type=int, nargs="?", const=1, choices=[0, 1], default=sup)
    p.add_argument("--persistent_workers", type=int, nargs="?", const=1, choices=[0, 1], default=sup)
    p.add_argument("--prefetch_factor", type=int, default=sup)
    p.add_argument("--sequence_len", type=int, default=sup)
    p.add_argument("--sequence_stride", type=int, default=sup)
    p.add_argument("--batch_by_root", type=int, nargs="?", const=1, choices=[0, 1], default=sup)
    p.add_argument("--balanced_sampling", type=int, nargs="?", const=1, choices=[0, 1], default=sup)
    p.add_argument("--eval_all_roots", type=int, nargs="?", const=1, choices=[0, 1], default=sup)

    p.add_argument("--optimizer", type=str, default=sup)
    p.add_argument("--weight_decay", type=float, default=sup)
    p.add_argument("--scheduler", type=str, default=sup)
    p.add_argument("--warmup_epochs", type=int, default=sup)
    p.add_argument("--patience", type=int, default=sup)
    p.add_argument("--earlystop_metric", type=str, default=sup)
    p.add_argument("--mixed_precision", type=int, nargs="?", const=1, choices=[0, 1], default=sup)
    p.add_argument("--compile", type=int, nargs="?", const=1, choices=[0, 1], default=sup)
    p.add_argument("--compile_backend", type=str, default=sup)

    p.add_argument("--tbptt_len", type=int, default=sup)
    p.add_argument("--tbptt_stride", type=int, default=sup)
    p.add_argument("--grad_clip_norm", type=float, default=sup)

    p.add_argument("--window_stack_K", type=int, default=sup)
    p.add_argument("--voxel_stack_mode", type=str, choices=["abs", "delta"], default=sup)
    p.add_argument("--scale_min", type=float, default=sup)
    p.add_argument("--scale_max", type=float, default=sup)
    p.add_argument("--imu_gate_soft", dest="imu_gate_soft", action="store_true", default=sup)
    p.add_argument("--no-imu_gate_soft", dest="imu_gate_soft", action="store_false", default=sup)
    p.add_argument("--uncertainty_fusion", dest="use_uncertainty_fusion", action="store_true", default=sup)
    p.add_argument("--no-uncertainty_fusion", dest="use_uncertainty_fusion", action="store_false", default=sup)
    p.add_argument("--uncertainty_gate", dest="uncertainty_use_gate", action="store_true", default=sup)
    p.add_argument("--no-uncertainty_gate", dest="uncertainty_use_gate", action="store_false", default=sup)

    p.add_argument("--loss_w_static", type=float, default=sup)
    p.add_argument("--loss_w_scale", type=float, default=sup)
    p.add_argument("--loss_w_scale_reg", type=float, default=sup)
    p.add_argument("--loss_w_path_scale", type=float, default=sup)
    p.add_argument("--use_seq_scale", dest="use_seq_scale", action="store_true", default=sup)
    p.add_argument("--no-use_seq_scale", dest="use_seq_scale", action="store_false", default=sup)
    p.add_argument("--seq_scale_reg", type=float, default=sup)
    p.add_argument("--min_step_threshold", type=float, default=sup)
    p.add_argument("--min_step_weight", type=float, default=sup)
    p.add_argument("--speed_thresh", type=float, default=sup)
    p.add_argument("--loss_w_bias_a", type=float, default=sup)
    p.add_argument("--loss_w_bias_g", type=float, default=sup)
    p.add_argument("--loss_w_uncertainty", type=float, default=sup)
    p.add_argument("--loss_w_uncertainty_calib", type=float, default=sup)
    p.add_argument("--loss_w_correction", type=float, default=sup)
    p.add_argument("--adaptive_loss", dest="adaptive_loss_weights", action="store_true", default=sup)
    p.add_argument("--no-adaptive_loss", dest="adaptive_loss_weights", action="store_false", default=sup)

    p.add_argument("--calib_yaml", type=str, default=sup, help="Calibration YAML file (baseline-compatible)")
    p.add_argument("--multi_calib_yaml", type=str, nargs="+", default=sup, help="Calibration YAML per dataset root")
    return p


def build_cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        cfg = load_experiment_config(str(args.config))
    else:
        root = str(getattr(args, "root", "") or "")
        multi_root = getattr(args, "multi_root", None)
        if (not root) and multi_root:
            try:
                roots = list(multi_root)
                root = str(roots[0]) if roots else ""
            except Exception:
                root = ""

        ds = DatasetConfig(
            root=root,
            multi_root=list(multi_root) if isinstance(multi_root, list) and multi_root else None,
            events_h5=str(getattr(args, "events_h5", "")) if hasattr(args, "events_h5") else None,
            dt=float(getattr(args, "dt", 0.2)),
            resolution=tuple(getattr(args, "resolution", (320, 320))),
            sensor_resolution=tuple(getattr(args, "sensor_resolution")) if hasattr(args, "sensor_resolution") else None,
            sample_stride=int(getattr(args, "sample_stride", 8)),
            windowing_mode=str(getattr(args, "windowing_mode", "imu")),
            window_dt=float(getattr(args, "window_dt")) if hasattr(args, "window_dt") else None,
            train_split=float(getattr(args, "train_split", 0.9)),
            augment=bool(int(getattr(args, "augment", 0))) if hasattr(args, "augment") else False,
            derotate=bool(int(getattr(args, "derotate", 0))) if hasattr(args, "derotate") else False,
            voxelize_in_dataset=bool(int(getattr(args, "voxelize_in_dataset", 1))) if hasattr(args, "voxelize_in_dataset") else True,
        )
        model = ModelConfig()
        train = TrainingConfig(
            epochs=int(getattr(args, "epochs", 10)),
            batch_size=int(getattr(args, "batch_size", 2)),
            lr=float(getattr(args, "lr", 1e-3)),
        )
        cfg = ExperimentConfig(
            dataset=ds,
            model=model,
            training=train,
            seed=int(getattr(args, "seed", 0)),
            device=str(getattr(args, "device", "cuda")),
            output_dir=str(getattr(args, "output_dir", "")) if hasattr(args, "output_dir") else None,
        )

    overrides: Dict[str, Any] = {}
    for k in (
        "root",
        "multi_root",
        "events_h5",
        "dt",
        "resolution",
        "sensor_resolution",
        "sample_stride",
        "windowing_mode",
        "window_dt",
        "train_split",
        "augment",
        "derotate",
        "voxelize_in_dataset",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "lr",
        "eval_interval",
        "device",
        "seed",
        "output_dir",
        "init_checkpoint",
        "metrics_csv",
        "export_torchscript",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "sequence_len",
        "sequence_stride",
        "batch_by_root",
        "balanced_sampling",
        "eval_all_roots",
        "optimizer",
        "weight_decay",
        "scheduler",
        "warmup_epochs",
        "patience",
        "earlystop_metric",
        "mixed_precision",
        "compile",
        "compile_backend",
        "tbptt_len",
        "tbptt_stride",
        "grad_clip_norm",
        "window_stack_K",
        "voxel_stack_mode",
        "scale_min",
        "scale_max",
        "imu_gate_soft",
        "use_uncertainty_fusion",
        "uncertainty_use_gate",
        "loss_w_static",
        "loss_w_scale",
        "loss_w_scale_reg",
        "loss_w_path_scale",
        "use_seq_scale",
        "seq_scale_reg",
        "min_step_threshold",
        "min_step_weight",
        "speed_thresh",
        "loss_w_bias_a",
        "loss_w_bias_g",
        "loss_w_uncertainty",
        "loss_w_uncertainty_calib",
        "loss_w_correction",
        "adaptive_loss_weights",
    ):
        if hasattr(args, k):
            overrides[k] = getattr(args, k)

    for k in (
        "augment",
        "derotate",
        "voxelize_in_dataset",
        "export_torchscript",
        "pin_memory",
        "persistent_workers",
        "batch_by_root",
        "balanced_sampling",
        "eval_all_roots",
        "mixed_precision",
        "compile",
    ):
        if k in overrides and isinstance(overrides[k], int):
            overrides[k] = bool(int(overrides[k]))

    cfg = merge_cli_overrides(cfg, overrides=overrides)

    calib_yaml = str(getattr(args, "calib_yaml", "") or "")
    if not calib_yaml and getattr(args, "multi_calib_yaml", None):
        try:
            ymls = list(getattr(args, "multi_calib_yaml"))
            calib_yaml = str(ymls[0]) if len(ymls) > 0 else ""
        except Exception:
            calib_yaml = ""

    calib_obj = None
    if calib_yaml:
        p = Path(calib_yaml).expanduser()
        if not p.exists():
            raise FileNotFoundError(str(p))
        calib_obj = load_calibration(str(p))
        if calib_obj is None:
            raise ValueError(f"Failed to parse calib_yaml: {str(p)}")

    multi_calib_yaml = getattr(args, "multi_calib_yaml", None)
    if multi_calib_yaml:
        ymls = list(multi_calib_yaml)
        calibs = []
        for y in ymls:
            yp = Path(str(y)).expanduser()
            if not yp.exists():
                raise FileNotFoundError(str(yp))
            c = load_calibration(str(yp))
            if c is None:
                raise ValueError(f"Failed to parse calib_yaml: {str(yp)}")
            calibs.append((str(yp), c))

        roots = list(getattr(cfg.dataset, "multi_root", None) or [])
        if not roots and isinstance(calib_obj, dict):
            mr = calib_obj.get("multi_root")
            if isinstance(mr, list) and mr:
                roots = [str(r) for r in mr]

        if not roots:
            inferred = []
            for y, c in calibs:
                r = infer_dataset_root_from_calib(c, y)
                if r:
                    inferred.append(str(r))
            roots = inferred

        if roots and len(roots) == len(calibs):
            cfg.dataset.multi_root = [str(r) for r in roots]
            cfg.dataset.root = str(roots[0])
            cfg.dataset.calib = {str(r): c for r, (_, c) in zip(roots, calibs)}
        else:
            cfg.dataset.calib = calibs[0][1] if calibs else cfg.dataset.calib

    if isinstance(calib_obj, dict) and cfg.dataset.calib is None:
        cfg.dataset.calib = calib_obj

    if isinstance(cfg.dataset.calib, dict) and not str(cfg.dataset.root or "").strip():
        inferred = infer_dataset_root_from_calib(cfg.dataset.calib, calib_yaml)
        if inferred:
            cfg.dataset.root = str(inferred)
        else:
            mr = cfg.dataset.calib.get("multi_root") if isinstance(cfg.dataset.calib, dict) else None
            if isinstance(mr, list) and len(mr) > 0:
                cfg.dataset.root = str(mr[0])

    if (not getattr(cfg.dataset, "multi_root", None)) and isinstance(cfg.dataset.calib, dict):
        mr = cfg.dataset.calib.get("multi_root")
        if isinstance(mr, list) and mr:
            cfg.dataset.multi_root = [str(r) for r in mr]

    if not str(cfg.dataset.root or "").strip():
        raise ValueError("Missing dataset root. Provide --root/--multi_root or set calib_yaml keys dataset_root/root_dir/root/data_root (or imu_path/mocap_path/events_h5 to infer).")

    return cfg


def main_cli(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = build_cfg_from_args(args)
    main(cfg)


if __name__ == "__main__":
    main_cli()
