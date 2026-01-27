from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fno_evio.config.loader import load_experiment_config


@dataclass(frozen=True)
class DatasetPreset:
    name: str
    calib_yamls: List[str]
    data_args: List[str]
    num_workers: int
    seq_len: int
    seq_stride: int


def _repo_paths(repo_root: str) -> Tuple[Path, Path]:
    repo = Path(repo_root).expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(str(repo))
    return repo, repo / "yaml"


def _dataset_preset(repo_root: str, dataset: str) -> DatasetPreset:
    repo, yaml_dir = _repo_paths(repo_root)
    dataset = str(dataset).strip().lower()

    calib_single = yaml_dir / "mocap-6dof_calib.yaml"
    calib_a = yaml_dir / "calib-A.yaml"
    calib_b = yaml_dir / "calib-B.yaml"

    if dataset in ("multi", "ab", "a+b"):
        return DatasetPreset(
            name="multi",
            calib_yamls=[str(calib_a), str(calib_b)],
            data_args=["--multi_calib_yaml", str(calib_a), str(calib_b), "--eval_all_roots", "--balanced_sampling"],
            num_workers=2,
            seq_len=400,
            seq_stride=400,
        )
    if dataset in ("a", "calib-a"):
        return DatasetPreset(
            name="A",
            calib_yamls=[str(calib_a)],
            data_args=["--calib_yaml", str(calib_a), "--eval_all_roots"],
            num_workers=2,
            seq_len=400,
            seq_stride=400,
        )
    if dataset in ("b", "calib-b"):
        return DatasetPreset(
            name="B",
            calib_yamls=[str(calib_b)],
            data_args=["--calib_yaml", str(calib_b), "--eval_all_roots"],
            num_workers=2,
            seq_len=400,
            seq_stride=400,
        )

    return DatasetPreset(
        name="single",
        calib_yamls=[str(calib_single)],
        data_args=["--calib_yaml", str(calib_single)],
        num_workers=4,
        seq_len=200,
        seq_stride=200,
    )


def _windowing_args(windowing_mode: str, dt: float) -> List[str]:
    m = str(windowing_mode).strip().lower()
    if m == "gt":
        return ["--windowing_mode", "gt"]
    return ["--windowing_mode", "imu", "--window_dt", f"{float(dt):.6f}"]


def _stage_args(stage: str) -> Tuple[str, List[str]]:
    s = str(stage).strip().lower()
    if s == "0":
        return "stage0_baseline", []
    if s == "1":
        return "stage1_fixed_s", ["--scale_min", "0.5", "--scale_max", "0.5"]
    if s == "2":
        return "stage2_mle_s", ["--scale_min", "0.0", "--scale_max", "1.0"]
    if s == "3a":
        return "stage3a_static", ["--scale_min", "0.0", "--scale_max", "1.0", "--loss_w_static", "0.5", "--speed_thresh", "0.05"]
    if s == "3b":
        return "stage3b_static_scale", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "1.0",
            "--loss_w_static",
            "0.5",
            "--loss_w_scale",
            "0.5",
            "--speed_thresh",
            "0.05",
        ]
    if s == "3c":
        return "stage3c_full_constraint", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "1.0",
            "--loss_w_static",
            "0.5",
            "--loss_w_scale",
            "0.5",
            "--loss_w_path_scale",
            "0.1",
            "--speed_thresh",
            "0.05",
        ]
    if s == "4":
        return "stage4_bayesian", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "1.0",
            "--uncertainty_fusion",
            "--loss_w_uncertainty",
            "0.1",
            "--loss_w_uncertainty_calib",
            "0.05",
        ]
    if s in ("4b", "4br"):
        return ("stage4b_bayes_full_ref" if s == "4br" else "stage4b_bayes_full"), [
            "--scale_min",
            "0.0",
            "--scale_max",
            "1.0",
            "--uncertainty_fusion",
            "--loss_w_uncertainty",
            "0.1",
            "--loss_w_uncertainty_calib",
            "0.05",
            "--loss_w_static",
            "0.5",
            "--loss_w_scale",
            "0.5",
            "--loss_w_path_scale",
            "0.1",
            "--speed_thresh",
            "0.05",
            "--use_seq_scale",
            "--seq_scale_reg",
            "0.08",
            "--min_step_threshold",
            "0.001",
            "--min_step_weight",
            "0.1",
        ]
    if s == "4bn":
        return "stage4bn_bayes_bias_prior", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "1.0",
            "--uncertainty_fusion",
            "--loss_w_uncertainty",
            "0.1",
            "--loss_w_uncertainty_calib",
            "0.05",
            "--loss_w_static",
            "0.5",
            "--loss_w_scale",
            "0.5",
            "--loss_w_path_scale",
            "0.1",
            "--speed_thresh",
            "0.05",
            "--use_seq_scale",
            "--seq_scale_reg",
            "0.08",
            "--min_step_threshold",
            "0.001",
            "--min_step_weight",
            "0.1",
            "--loss_w_bias_a",
            "1e-3",
            "--loss_w_bias_g",
            "1e-3",
        ]
    if s == "5":
        return "stage5_imu_anchored", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "1.0",
            "--loss_w_correction",
            "0.1",
            "--loss_w_static",
            "0.5",
            "--loss_w_scale",
            "0.3",
            "--loss_w_bias_a",
            "1e-3",
            "--loss_w_bias_g",
            "1e-3",
            "--speed_thresh",
            "0.05",
        ]
    if s == "6":
        return "stage6_final_baseline", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "1.0",
            "--loss_w_static",
            "0.5",
            "--loss_w_scale",
            "0.5",
            "--loss_w_path_scale",
            "0.1",
            "--speed_thresh",
            "0.05",
            "--use_seq_scale",
            "--seq_scale_reg",
            "0.08",
            "--min_step_threshold",
            "0.001",
            "--min_step_weight",
            "0.1",
        ]
    raise ValueError(f"Unknown stage: {stage}")


def _ablation_args(abl: str) -> Tuple[str, List[str]]:
    a = str(abl).strip().lower()
    if a in ("none", "0", ""):
        return "none", []
    if a in ("imu_only", "imuonly", "1"):
        return "imuOnly", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "0.0",
            "--no-uncertainty_fusion",
            "--no-uncertainty_gate",
            "--loss_w_uncertainty",
            "0",
            "--loss_w_uncertainty_calib",
            "0",
        ]
    if a in ("visual_only", "vis_only", "visualonly", "2"):
        return "visOnly", ["--no-imu_gate_soft", "--imu_mask_prob", "1.0"]
    if a in ("both", "both_off", "3"):
        return "bothOff", [
            "--scale_min",
            "0.0",
            "--scale_max",
            "0.0",
            "--no-imu_gate_soft",
            "--imu_mask_prob",
            "1.0",
            "--no-uncertainty_fusion",
            "--no-uncertainty_gate",
            "--loss_w_uncertainty",
            "0",
            "--loss_w_uncertainty_calib",
            "0",
        ]
    raise ValueError(f"Unknown ablation: {abl}")


def _get_nested(d: Any, keys: List[str], default: Any) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_base_yaml(repo_root: str, base_yaml: str) -> dict:
    repo = Path(repo_root).expanduser().resolve()
    p = Path(base_yaml).expanduser()
    if not p.is_absolute():
        p = (repo / p).resolve()
    cfg = load_experiment_config(str(p))
    return {
        "seed": int(getattr(cfg, "seed", 42)),
        "device": str(getattr(cfg, "device", "cuda")),
        "dataset": {
            "dt": float(getattr(cfg.dataset, "dt", 0.00833)),
            "sample_stride": int(getattr(cfg.dataset, "sample_stride", 4)),
            "resolution": tuple(getattr(cfg.dataset, "resolution", (180, 320))),
            "windowing_mode": str(getattr(cfg.dataset, "windowing_mode", "imu")),
            "window_dt": float(getattr(cfg.dataset, "window_dt", float(getattr(cfg.dataset, "dt", 0.00833)))),
        },
        "model": {
            "window_stack_K": int(getattr(cfg.model, "window_stack_K", 3)),
            "voxel_stack_mode": str(getattr(cfg.model, "voxel_stack_mode", "delta")),
        },
        "training": {
            "epochs": int(getattr(cfg.training, "epochs", 500)),
            "batch_size": int(getattr(cfg.training, "batch_size", 512)),
            "lr": float(getattr(cfg.training, "lr", 1e-3)),
            "eval_interval": int(getattr(cfg.training, "eval_interval", 1)),
            "eval_batch_size": int(getattr(cfg.training, "eval_batch_size", 1)),
            "tbptt_len": int(getattr(cfg.training, "tbptt_len", 75)),
            "num_workers": int(getattr(cfg.training, "num_workers", 2)),
            "sequence_len": int(getattr(cfg.training, "sequence_len", 400)),
            "sequence_stride": int(getattr(cfg.training, "sequence_stride", 400)),
            "persistent_workers": bool(getattr(cfg.training, "persistent_workers", True)),
            "prefetch_factor": int(getattr(cfg.training, "prefetch_factor", 1)),
            "mixed_precision": bool(getattr(cfg.training, "mixed_precision", True)),
            "scheduler": str(getattr(cfg.training, "scheduler", "cosine")),
            "warmup_epochs": int(getattr(cfg.training, "warmup_epochs", 10)),
            "patience": int(getattr(cfg.training, "patience", 50)),
            "earlystop_metric": str(getattr(cfg.training, "earlystop_metric", "ate")),
            "optimizer": str(getattr(cfg.training, "optimizer", "adamw")),
            "weight_decay": float(getattr(cfg.training, "weight_decay", 0.01)),
        },
    }


def build_train_command(args: argparse.Namespace) -> List[str]:
    repo_root, _ = _repo_paths(args.repo_root)
    preset = _dataset_preset(str(repo_root), str(args.dataset))
    stage_name, stage_args = _stage_args(str(args.stage))
    abl_tag, abl_args = _ablation_args(str(args.ablation))
    base_cfg = _load_base_yaml(str(repo_root), str(args.base_yaml))

    dt = float(args.dt) if args.dt is not None else float(_get_nested(base_cfg, ["dataset", "dt"], 0.00833))
    sample_stride = int(args.sample_stride) if args.sample_stride is not None else int(_get_nested(base_cfg, ["dataset", "sample_stride"], 4))
    window_stack_k = int(args.window_stack_k) if args.window_stack_k is not None else int(_get_nested(base_cfg, ["model", "window_stack_K"], 3))
    voxel_stack_mode = str(args.voxel_stack_mode) if args.voxel_stack_mode is not None else str(_get_nested(base_cfg, ["model", "voxel_stack_mode"], "delta"))
    epochs = int(args.epochs) if args.epochs is not None else int(_get_nested(base_cfg, ["training", "epochs"], 500))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(_get_nested(base_cfg, ["training", "batch_size"], 512))
    tbptt_len = int(args.tbptt_len) if args.tbptt_len is not None else int(_get_nested(base_cfg, ["training", "tbptt_len"], 75))
    eval_interval = int(args.eval_interval) if args.eval_interval is not None else int(_get_nested(base_cfg, ["training", "eval_interval"], 1))
    eval_batch_size = int(args.eval_batch_size) if args.eval_batch_size is not None else int(_get_nested(base_cfg, ["training", "eval_batch_size"], 1))
    prefetch_factor = int(args.prefetch_factor) if args.prefetch_factor is not None else int(_get_nested(base_cfg, ["training", "prefetch_factor"], 1))
    seed = int(args.seed) if args.seed is not None else int(_get_nested(base_cfg, ["seed"], 42))
    scheduler = str(args.scheduler) if args.scheduler is not None else str(_get_nested(base_cfg, ["training", "scheduler"], "cosine"))
    warmup_epochs = int(args.warmup_epochs) if args.warmup_epochs is not None else int(_get_nested(base_cfg, ["training", "warmup_epochs"], 10))
    patience = int(args.patience) if args.patience is not None else int(_get_nested(base_cfg, ["training", "patience"], 50))
    earlystop_metric = str(args.earlystop_metric) if args.earlystop_metric is not None else str(_get_nested(base_cfg, ["training", "earlystop_metric"], "ate"))
    win_args = _windowing_args(str(args.windowing_mode), float(dt))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "metrics.csv"

    base_args = [
        "--batch_by_root",
        "--dt",
        str(float(dt)),
        "--sample_stride",
        str(int(sample_stride)),
        "--sequence_len",
        str(int(preset.seq_len)),
        "--sequence_stride",
        str(int(preset.seq_stride)),
        "--window_stack_K",
        str(int(window_stack_k)),
        "--voxel_stack_mode",
        str(voxel_stack_mode),
        "--epochs",
        str(int(epochs)),
        "--batch_size",
        str(int(batch_size)),
        "--tbptt_len",
        str(int(tbptt_len)),
        "--eval_interval",
        str(int(eval_interval)),
        "--eval_batch_size",
        str(int(eval_batch_size)),
        "--num_workers",
        str(int(preset.num_workers)),
        "--prefetch_factor",
        str(int(prefetch_factor)),
        "--seed",
        str(int(seed)),
        "--scheduler",
        str(scheduler),
        "--warmup_epochs",
        str(int(warmup_epochs)),
        "--patience",
        str(int(patience)),
        "--earlystop_metric",
        str(earlystop_metric),
        "--metrics_csv",
        str(metrics_csv),
        "--output_dir",
        str(out_dir),
        "--device",
        str(args.device),
        "--no-uncertainty_fusion",
        "--no-uncertainty_gate",
        "--loss_w_uncertainty",
        "0",
        "--loss_w_uncertainty_calib",
        "0",
        "--loss_w_scale",
        "0",
        "--loss_w_static",
        "0",
        "--loss_w_path_scale",
        "0",
        "--loss_w_scale_reg",
        "0",
        "--no-use_seq_scale",
        "--seq_scale_reg",
        "0",
        "--no-adaptive_loss",
    ]
    if args.mixed_precision is None:
        mixed_precision = bool(_get_nested(base_cfg, ["training", "mixed_precision"], True))
    else:
        mixed_precision = bool(args.mixed_precision)
    if mixed_precision:
        base_args.append("--mixed_precision")
    if args.persistent_workers is None:
        persistent_workers = bool(_get_nested(base_cfg, ["training", "persistent_workers"], True))
    else:
        persistent_workers = bool(args.persistent_workers)
    if persistent_workers:
        base_args.append("--persistent_workers")
    if args.imu_gate_soft is None:
        imu_gate_soft = True
    else:
        imu_gate_soft = bool(args.imu_gate_soft)
    if imu_gate_soft:
        base_args.append("--imu_gate_soft")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fno_evio.legacy.train_fno_vio",
        *preset.data_args,
        *base_args,
        *win_args,
        *stage_args,
        *abl_args,
    ]
    run_name = f"{preset.name}_{stage_name}_{abl_tag}"
    _ = run_name
    return cmd


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser("FNO-EVIO: interactive training launcher (ported from fno-FAST/run_train.sh)")
    p.add_argument("--repo_root", type=str, required=True)
    p.add_argument("--base_yaml", type=str, default="configs/base.yaml")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dataset", type=str, default="single", choices=["single", "A", "B", "multi", "a", "b", "ab"])
    p.add_argument("--windowing_mode", type=str, default="imu", choices=["imu", "gt"])
    p.add_argument("--stage", type=str, default="5")
    p.add_argument("--ablation", type=str, default="none")

    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--sample_stride", type=int, default=None)
    p.add_argument("--window_stack_k", type=int, default=None)
    p.add_argument("--voxel_stack_mode", type=str, default=None, choices=["abs", "delta"])

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--tbptt_len", type=int, default=None)
    p.add_argument("--eval_interval", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=None)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mixed_precision", type=int, default=None, choices=[0, 1])
    p.add_argument("--persistent_workers", type=int, default=None, choices=[0, 1])
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--imu_gate_soft", type=int, default=None, choices=[0, 1])

    p.add_argument("--scheduler", type=str, default=None)
    p.add_argument("--warmup_epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--earlystop_metric", type=str, default=None)

    args = p.parse_args(argv)

    cmd = build_train_command(args)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path(args.repo_root).expanduser().resolve()))
    proc = subprocess.run(cmd, cwd=str(Path(args.repo_root).expanduser().resolve()), env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
