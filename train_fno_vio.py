from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from fno_evio.app import main as run_app
from fno_evio.config.calib import infer_dataset_root_from_calib, load_calibration
from fno_evio.config.loader import load_experiment_config, merge_cli_overrides
from fno_evio.config.schema import ExperimentConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_fno_vio")
    p.add_argument("--config", type=str, required=True)

    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--resolution", type=int, nargs=2, default=None)
    p.add_argument("--sensor_resolution", type=int, nargs=2, default=None)
    p.add_argument("--sample_stride", type=int, default=None)
    p.add_argument("--dataset_kind", type=str, default=None)
    p.add_argument("--windowing_mode", type=str, default=None)
    p.add_argument("--window_dt", type=float, default=None)
    p.add_argument("--train_split", type=float, default=None)

    p.add_argument("--calib_yaml", type=str, default="")
    p.add_argument("--multi_calib_yaml", type=str, nargs="+", default=None)

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tbptt_len", type=int, default=None)
    p.add_argument("--tbptt_stride", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--metrics_csv", type=str, default=None)
    p.add_argument("--init_checkpoint", type=str, default=None)

    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=None)
    p.add_argument("--no-pin_memory", dest="pin_memory", action="store_false", default=None)
    p.add_argument("--persistent_workers", dest="persistent_workers", action="store_true", default=None)
    p.add_argument("--no-persistent_workers", dest="persistent_workers", action="store_false", default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)

    p.add_argument("--sequence_len", type=int, default=None)
    p.add_argument("--sequence_stride", type=int, default=None)
    p.add_argument("--batch_by_root", dest="batch_by_root", action="store_true", default=None)
    p.add_argument("--no-batch_by_root", dest="batch_by_root", action="store_false", default=None)
    p.add_argument("--balanced_sampling", dest="balanced_sampling", action="store_true", default=None)
    p.add_argument("--no-balanced_sampling", dest="balanced_sampling", action="store_false", default=None)
    p.add_argument("--eval_all_roots", dest="eval_all_roots", action="store_true", default=None)
    p.add_argument("--no-eval_all_roots", dest="eval_all_roots", action="store_false", default=None)

    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--scheduler_patience", type=int, default=None)
    p.add_argument("--scheduler_T_max", type=int, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--earlystop_metric", type=str, default=None)
    p.add_argument("--earlystop_min_epoch", type=int, default=None)
    p.add_argument("--earlystop_ma_window", type=int, default=None)

    p.add_argument("--mixed_precision", dest="mixed_precision", action="store_true", default=None)
    p.add_argument("--no-mixed_precision", dest="mixed_precision", action="store_false", default=None)
    p.add_argument("--adaptive_loss", dest="adaptive_loss_weights", action="store_true", default=None)
    p.add_argument("--no-adaptive_loss", dest="adaptive_loss_weights", action="store_false", default=None)

    p.add_argument("--loss_w_t", type=float, default=None)
    p.add_argument("--loss_w_r", type=float, default=None)
    p.add_argument("--loss_w_v", type=float, default=None)
    p.add_argument("--loss_w_aux_motion", type=float, default=None)
    p.add_argument("--loss_w_physics", type=float, default=None)
    p.add_argument("--loss_w_smooth", type=float, default=None)
    p.add_argument("--loss_w_rpe", type=float, default=None)
    p.add_argument("--rpe_dt", type=float, default=None)
    p.add_argument("--speed_thresh", type=float, default=None)
    p.add_argument("--loss_w_scale", type=float, default=None)
    p.add_argument("--loss_w_scale_reg", type=float, default=None)
    p.add_argument("--loss_w_path_scale", type=float, default=None)
    p.add_argument("--loss_w_static", type=float, default=None)
    p.add_argument("--loss_w_uncertainty", type=float, default=None)
    p.add_argument("--loss_w_uncertainty_calib", type=float, default=None)
    p.add_argument("--loss_w_correction", type=float, default=None)
    p.add_argument("--loss_w_bias_a", type=float, default=None)
    p.add_argument("--loss_w_bias_g", type=float, default=None)
    p.add_argument("--seq_scale_reg", type=float, default=None)
    p.add_argument("--min_step_threshold", type=float, default=None)
    p.add_argument("--min_step_weight", type=float, default=None)
    p.add_argument("--use_seq_scale", dest="use_seq_scale", action="store_true", default=None)
    p.add_argument("--no-use_seq_scale", dest="use_seq_scale", action="store_false", default=None)

    p.add_argument("--window_stack_K", type=int, default=None)
    p.add_argument("--voxel_stack_mode", type=str, default=None)
    p.add_argument("--scale_min", type=float, default=None)
    p.add_argument("--scale_max", type=float, default=None)
    p.add_argument("--imu_gate_soft", dest="imu_gate_soft", action="store_true", default=None)
    p.add_argument("--no-imu_gate_soft", dest="imu_gate_soft", action="store_false", default=None)
    p.add_argument("--uncertainty_fusion", dest="use_uncertainty_fusion", action="store_true", default=None)
    p.add_argument("--no-uncertainty_fusion", dest="use_uncertainty_fusion", action="store_false", default=None)
    p.add_argument("--uncertainty_gate", dest="uncertainty_use_gate", action="store_true", default=None)
    p.add_argument("--no-uncertainty_gate", dest="uncertainty_use_gate", action="store_false", default=None)

    p.add_argument("--imu_mask_prob", type=float, default=None)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p


def build_cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_experiment_config(str(args.config))
    overrides: Dict[str, Any] = {
        "root": getattr(args, "root", None),
        "dt": getattr(args, "dt", None),
        "resolution": tuple(getattr(args, "resolution")) if getattr(args, "resolution", None) is not None else None,
        "sensor_resolution": tuple(getattr(args, "sensor_resolution")) if getattr(args, "sensor_resolution", None) is not None else None,
        "sample_stride": getattr(args, "sample_stride", None),
        "dataset_kind": getattr(args, "dataset_kind", None),
        "windowing_mode": getattr(args, "windowing_mode", None),
        "window_dt": getattr(args, "window_dt", None),
        "train_split": getattr(args, "train_split", None),
        "imu_mask_prob": getattr(args, "imu_mask_prob", None),
        "epochs": getattr(args, "epochs", None),
        "batch_size": getattr(args, "batch_size", None),
        "lr": getattr(args, "lr", None),
        "tbptt_len": getattr(args, "tbptt_len", None),
        "tbptt_stride": getattr(args, "tbptt_stride", None),
        "eval_interval": getattr(args, "eval_interval", None),
        "eval_batch_size": getattr(args, "eval_batch_size", None),
        "metrics_csv": getattr(args, "metrics_csv", None),
        "init_checkpoint": getattr(args, "init_checkpoint", None),
        "num_workers": getattr(args, "num_workers", None),
        "pin_memory": getattr(args, "pin_memory", None),
        "persistent_workers": getattr(args, "persistent_workers", None),
        "prefetch_factor": getattr(args, "prefetch_factor", None),
        "sequence_len": getattr(args, "sequence_len", None),
        "sequence_stride": getattr(args, "sequence_stride", None),
        "batch_by_root": getattr(args, "batch_by_root", None),
        "balanced_sampling": getattr(args, "balanced_sampling", None),
        "eval_all_roots": getattr(args, "eval_all_roots", None),
        "scheduler": getattr(args, "scheduler", None),
        "gamma": getattr(args, "gamma", None),
        "scheduler_patience": getattr(args, "scheduler_patience", None),
        "scheduler_T_max": getattr(args, "scheduler_T_max", None),
        "warmup_epochs": getattr(args, "warmup_epochs", None),
        "patience": getattr(args, "patience", None),
        "earlystop_metric": getattr(args, "earlystop_metric", None),
        "earlystop_min_epoch": getattr(args, "earlystop_min_epoch", None),
        "earlystop_ma_window": getattr(args, "earlystop_ma_window", None),
        "mixed_precision": getattr(args, "mixed_precision", None),
        "adaptive_loss_weights": getattr(args, "adaptive_loss_weights", None),
        "loss_w_t": getattr(args, "loss_w_t", None),
        "loss_w_r": getattr(args, "loss_w_r", None),
        "loss_w_v": getattr(args, "loss_w_v", None),
        "loss_w_aux_motion": getattr(args, "loss_w_aux_motion", None),
        "loss_w_physics": getattr(args, "loss_w_physics", None),
        "loss_w_smooth": getattr(args, "loss_w_smooth", None),
        "loss_w_rpe": getattr(args, "loss_w_rpe", None),
        "rpe_dt": getattr(args, "rpe_dt", None),
        "speed_thresh": getattr(args, "speed_thresh", None),
        "loss_w_scale": getattr(args, "loss_w_scale", None),
        "loss_w_scale_reg": getattr(args, "loss_w_scale_reg", None),
        "loss_w_path_scale": getattr(args, "loss_w_path_scale", None),
        "loss_w_static": getattr(args, "loss_w_static", None),
        "loss_w_uncertainty": getattr(args, "loss_w_uncertainty", None),
        "loss_w_uncertainty_calib": getattr(args, "loss_w_uncertainty_calib", None),
        "loss_w_correction": getattr(args, "loss_w_correction", None),
        "loss_w_bias_a": getattr(args, "loss_w_bias_a", None),
        "loss_w_bias_g": getattr(args, "loss_w_bias_g", None),
        "seq_scale_reg": getattr(args, "seq_scale_reg", None),
        "min_step_threshold": getattr(args, "min_step_threshold", None),
        "min_step_weight": getattr(args, "min_step_weight", None),
        "use_seq_scale": getattr(args, "use_seq_scale", None),
        "window_stack_K": getattr(args, "window_stack_K", None),
        "voxel_stack_mode": getattr(args, "voxel_stack_mode", None),
        "scale_min": getattr(args, "scale_min", None),
        "scale_max": getattr(args, "scale_max", None),
        "imu_gate_soft": getattr(args, "imu_gate_soft", None),
        "use_uncertainty_fusion": getattr(args, "use_uncertainty_fusion", None),
        "uncertainty_use_gate": getattr(args, "uncertainty_use_gate", None),
        "device": getattr(args, "device", None),
        "seed": getattr(args, "seed", None),
        "output_dir": getattr(args, "output_dir", None),
    }
    cfg = merge_cli_overrides(cfg, overrides=overrides)

    multi_calib_yaml = getattr(args, "multi_calib_yaml", None)
    calib_yaml = str(getattr(args, "calib_yaml", "") or "").strip()

    if isinstance(multi_calib_yaml, list) and multi_calib_yaml:
        calibs_by_root: Dict[str, Any] = {}
        roots = []
        for y in multi_calib_yaml:
            y_p = Path(str(y)).expanduser().resolve()
            if not y_p.exists():
                raise FileNotFoundError(str(y_p))
            calib_obj = load_calibration(str(y_p))
            if calib_obj is None:
                raise RuntimeError(f"Failed to parse calib_yaml: {str(y_p)}")
            inferred = infer_dataset_root_from_calib(calib_obj, str(y_p))
            if not inferred:
                raise ValueError(f"Missing dataset root for calib_yaml: {str(y_p)}")
            root = str(Path(inferred).expanduser().resolve())
            roots.append(root)
            calibs_by_root[root] = calib_obj
        cfg.dataset.multi_root = roots
        cfg.dataset.root = roots[0] if roots else cfg.dataset.root
        cfg.dataset.calib = calibs_by_root
        return cfg

    if calib_yaml:
        y_p = Path(calib_yaml).expanduser().resolve()
        if not y_p.exists():
            raise FileNotFoundError(str(y_p))
        calib_obj = load_calibration(str(y_p))
        if calib_obj is None:
            raise RuntimeError(f"Failed to parse calib_yaml: {str(y_p)}")
        cfg.dataset.calib = calib_obj
        if not str(cfg.dataset.root or "").strip():
            inferred = infer_dataset_root_from_calib(calib_obj, str(y_p))
            if inferred:
                cfg.dataset.root = str(Path(inferred).expanduser().resolve())

    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = build_cfg_from_args(args)
    _ = asdict(cfg)
    run_app(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
