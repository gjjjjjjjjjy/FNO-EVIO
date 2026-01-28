"""
Application entrypoints for training and evaluation (config-driven).

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from fno_evio.common.constants import compute_adaptive_sequence_length
from fno_evio.config.schema import DatasetConfig, ExperimentConfig, ModelConfig, TrainingConfig
from fno_evio.data.datasets import Davis240Dataset, OptimizedTUMDataset, UZHFPVDataset
from fno_evio.data.sequence import CollateSequence, SequenceDataset


def build_dataloaders(
    cfg: DatasetConfig,
    *,
    training: TrainingConfig,
    model: ModelConfig,
    device: torch.device,
) -> Tuple[DataLoader, Union[DataLoader, Dict[str, DataLoader]]]:
    """
    Build train/val dataloaders.

    Notes:
        This preserves baseline-style contiguous splits and adds a safe temporal gap to reduce
        leakage across train/val when using sliding windows.
    """
    roots = list(cfg.multi_root) if isinstance(getattr(cfg, "multi_root", None), list) and cfg.multi_root else [cfg.root]

    def _calib_for_root(root_str: str) -> Optional[Dict[str, Any]]:
        calib = cfg.calib
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

    mode = str(getattr(cfg, "windowing_mode", "imu")).strip().lower()
    safe_gap = 2 if mode == "imu" else max(int(cfg.sample_stride), 1) * 2

    kind = str(getattr(cfg, "dataset_kind", "tum")).strip().lower()
    if kind in ("davis240", "davis240c", "davis"):
        ds_cls = Davis240Dataset
    elif kind in ("uzhfpv", "uzh-fpv", "fpv"):
        ds_cls = UZHFPVDataset
    else:
        ds_cls = OptimizedTUMDataset

    seq_train_list = []
    for r in roots:
        ds_kwargs: Dict[str, Any] = {
            "root": str(r),
            "events_h5": cfg.events_h5,
            "dt": float(cfg.dt),
            "resolution": tuple(cfg.resolution),
            "sensor_resolution": cfg.sensor_resolution,
            "sample_stride": int(cfg.sample_stride),
            "windowing_mode": str(cfg.windowing_mode),
            "window_dt": cfg.window_dt,
            "event_offset_scan": bool(cfg.event_offset_scan),
            "event_offset_scan_range_s": float(cfg.event_offset_scan_range_s),
            "event_offset_scan_step_s": float(cfg.event_offset_scan_step_s),
            "voxelize_in_dataset": bool(cfg.voxelize_in_dataset),
            "derotate": bool(cfg.derotate),
            "calib": _calib_for_root(str(r)),
            "event_file_candidates": tuple(cfg.event_file_candidates),
            "proc_device": (torch.device("cpu") if int(training.num_workers) > 0 else device),
            "std_norm": bool(cfg.voxel_std_norm),
            "log_norm": True,
            "augment": bool(cfg.augment),
            "adaptive_voxel": bool(cfg.adaptive_voxel),
            "event_noise_scale": float(cfg.event_noise_scale),
            "event_scale_jitter": float(cfg.event_scale_jitter),
            "imu_bias_scale": float(cfg.imu_bias_scale),
            "imu_mask_prob": float(cfg.imu_mask_prob),
            "adaptive_base_div": int(cfg.adaptive_base_div),
            "adaptive_max_events_div": int(cfg.adaptive_max_events_div),
            "adaptive_density_cap": float(cfg.adaptive_density_cap),
            "sequence_length": int(model.sequence_length),
        }
        if ds_cls is Davis240Dataset:
            ds_kwargs["side"] = "left"
        ds_tr = ds_cls(**ds_kwargs)
        n_r = len(ds_tr)
        n_train_r = max(int(float(cfg.train_split) * n_r) - int(safe_gap), 1)
        train_subset = torch.utils.data.Subset(ds_tr, list(range(n_train_r)))
        seq_train_list.append(SequenceDataset(train_subset, sequence_len=int(training.sequence_len), stride=int(training.sequence_stride)))

    train_seq = seq_train_list[0] if len(seq_train_list) == 1 else ConcatDataset(seq_train_list)

    actual_num_workers = int(training.num_workers)
    if device.type == "mps" and actual_num_workers > 0:
        actual_num_workers = 0

    loader_kwargs: Dict[str, Any] = {
        "num_workers": actual_num_workers,
        "pin_memory": bool(training.pin_memory),
        "persistent_workers": (bool(training.persistent_workers) if actual_num_workers > 0 else False),
        "collate_fn": CollateSequence(window_stack_k=int(model.window_stack_K), voxel_stack_mode=str(model.voxel_stack_mode)),
    }
    if actual_num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(training.prefetch_factor)

    def _build_val_loader_for_root(r: str) -> DataLoader:
        ds_kwargs: Dict[str, Any] = {
            "root": str(r),
            "events_h5": cfg.events_h5,
            "dt": float(cfg.dt),
            "resolution": tuple(cfg.resolution),
            "sensor_resolution": cfg.sensor_resolution,
            "sample_stride": int(cfg.sample_stride),
            "windowing_mode": str(cfg.windowing_mode),
            "window_dt": cfg.window_dt,
            "event_offset_scan": bool(cfg.event_offset_scan),
            "event_offset_scan_range_s": float(cfg.event_offset_scan_range_s),
            "event_offset_scan_step_s": float(cfg.event_offset_scan_step_s),
            "voxelize_in_dataset": bool(cfg.voxelize_in_dataset),
            "derotate": bool(cfg.derotate),
            "calib": _calib_for_root(str(r)),
            "event_file_candidates": tuple(cfg.event_file_candidates),
            "proc_device": (torch.device("cpu") if int(training.num_workers) > 0 else device),
            "std_norm": bool(cfg.voxel_std_norm),
            "log_norm": True,
            "augment": False,
            "adaptive_voxel": bool(cfg.adaptive_voxel),
            "event_noise_scale": float(cfg.event_noise_scale),
            "event_scale_jitter": float(cfg.event_scale_jitter),
            "imu_bias_scale": float(cfg.imu_bias_scale),
            "imu_mask_prob": float(cfg.imu_mask_prob),
            "adaptive_base_div": int(cfg.adaptive_base_div),
            "adaptive_max_events_div": int(cfg.adaptive_max_events_div),
            "adaptive_density_cap": float(cfg.adaptive_density_cap),
            "sequence_length": int(model.sequence_length),
        }
        if ds_cls is Davis240Dataset:
            ds_kwargs["side"] = "left"
        ds_val = ds_cls(**ds_kwargs)

        n = len(ds_val)
        split = max(int(float(cfg.train_split) * n), 1)
        val_start = split + int(safe_gap)
        if val_start >= n:
            val_start = min(split + 1, max(n - 1, 0))
        val_subset = torch.utils.data.Subset(ds_val, list(range(val_start, n)))
        val_seq = SequenceDataset(val_subset, sequence_len=int(training.sequence_len), stride=int(training.sequence_stride))
        eval_bs = int(training.eval_batch_size) if training.eval_batch_size is not None else 1
        return DataLoader(val_seq, batch_size=eval_bs, shuffle=False, drop_last=False, **loader_kwargs)

    batch_by_root = bool(getattr(training, "batch_by_root", False))
    if len(roots) > 1 and batch_by_root:
        from fno_evio.legacy.train_fno_vio import RootGroupedBatchSampler

        train_loader = DataLoader(
            train_seq,
            batch_sampler=RootGroupedBatchSampler(
                train_seq, batch_size=int(training.batch_size), shuffle=True, drop_last=False, balanced=bool(getattr(training, "balanced_sampling", False))
            ),
            **loader_kwargs,
        )
    else:
        sampler = None
        if len(roots) > 1 and bool(getattr(training, "balanced_sampling", False)) and isinstance(train_seq, ConcatDataset):
            weights = torch.empty(len(train_seq), dtype=torch.double)
            start = 0
            for ds_i in train_seq.datasets:
                n_i = max(int(len(ds_i)), 1)
                weights[start : start + n_i] = 1.0 / float(n_i)
                start += n_i
            sampler = WeightedRandomSampler(weights=weights, num_samples=int(weights.numel()), replacement=True)

        if sampler is None:
            train_loader = DataLoader(train_seq, batch_size=int(training.batch_size), shuffle=True, drop_last=False, **loader_kwargs)
        else:
            train_loader = DataLoader(train_seq, batch_size=int(training.batch_size), sampler=sampler, shuffle=False, drop_last=False, **loader_kwargs)

    if bool(getattr(training, "eval_all_roots", False)) and len(roots) > 1:
        val_loaders: Dict[str, DataLoader] = {}
        for r in roots:
            val_loaders[str(r)] = _build_val_loader_for_root(str(r))
        return train_loader, val_loaders

    val_loader = _build_val_loader_for_root(str(roots[0]))
    return train_loader, val_loader


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    """
    Build the VIO model.

    This function imports the actual model definition lazily to keep the app module light.
    """
    from fno_evio.models.vio import HybridVIONet  # local import to avoid import cycles

    return HybridVIONet(cfg)


def build_trainer(cfg: TrainingConfig, *, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """
    Build optimizer/scheduler/AMP state for training.
    """
    opt_name = str(getattr(cfg, "optimizer", "adamw")).strip().lower()
    betas = tuple(getattr(cfg, "adam_betas", (0.9, 0.999)))
    eps = float(getattr(cfg, "adam_eps", 1e-8))
    weight_decay = float(getattr(cfg, "weight_decay", 0.01))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.lr), betas=betas, eps=eps, weight_decay=weight_decay)

    sched_name = str(cfg.scheduler).strip().lower()
    if sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(int(cfg.epochs) // 3, 1), gamma=float(cfg.gamma))
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg.scheduler_T_max))
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=float(cfg.gamma), patience=int(cfg.scheduler_patience), mode="min")

    return {"optimizer": optimizer, "scheduler": scheduler}


def run_train(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Any,
    cfg: TrainingConfig,
    device: torch.device,
    dt: float,
) -> None:
    """
    Run the training loop.
    """
    from fno_evio.training.loop import train  # local import

    train(model=model, train_loader=train_loader, val_loader=val_loader, cfg=cfg, device=device, dt_window_fallback=float(dt))


def run_eval(
    *,
    model: torch.nn.Module,
    val_loader: Any,
    cfg: TrainingConfig,
    device: torch.device,
    dt: float,
) -> Tuple[float, float, float]:
    """
    Run evaluation loop.
    """
    from fno_evio.eval.evaluate import evaluate  # local import

    if isinstance(val_loader, dict):
        ates = []
        rpes_t = []
        rpes_r = []
        for ld in val_loader.values():
            a, t, r = evaluate(model=model, loader=ld, device=device, rpe_dt=float(cfg.rpe_dt), dt=float(dt), eval_sim3_mode=str(cfg.eval_sim3_mode))
            ates.append(float(a))
            rpes_t.append(float(t))
            rpes_r.append(float(r))
        if not ates:
            return (float("nan"), float("nan"), float("nan"))
        return (sum(ates) / float(len(ates)), sum(rpes_t) / float(len(rpes_t)), sum(rpes_r) / float(len(rpes_r)))
    return evaluate(model=model, loader=val_loader, device=device, rpe_dt=float(cfg.rpe_dt), dt=float(dt), eval_sim3_mode=str(cfg.eval_sim3_mode))


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def main(cfg: ExperimentConfig) -> None:
    """
    Main entrypoint for config-driven training.
    """
    set_global_seed(int(cfg.seed))
    device = torch.device(str(cfg.device))

    cfg.model.sequence_length = compute_adaptive_sequence_length(float(cfg.dataset.dt))
    if cfg.model.fusion_dim is None:
        cfg.model.fusion_dim = int(cfg.model.stem_channels)

    train_loader, val_loader = build_dataloaders(cfg.dataset, training=cfg.training, model=cfg.model, device=device)
    model = build_model(cfg.model).to(device)

    init_ckpt = str(getattr(cfg.training, "init_checkpoint", "") or "").strip()
    if init_ckpt:
        ckpt_obj = torch.load(init_ckpt, map_location="cpu")
        state = None
        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state = ckpt_obj["state_dict"]
        elif isinstance(ckpt_obj, dict):
            state = ckpt_obj
        if isinstance(state, dict):
            prefixes = ("module.", "model.", "orig_mod.")
            for pfx in prefixes:
                if state and all(str(k).startswith(pfx) for k in state.keys()):
                    state = {str(k)[len(pfx) :]: v for k, v in state.items()}
                    break
            model.load_state_dict(state, strict=False)

    trainer = build_trainer(cfg.training, model=model, device=device)
    _ = trainer

    run_train(model=model, train_loader=train_loader, val_loader=val_loader, cfg=cfg.training, device=device, dt=float(cfg.dataset.dt))

    if int(cfg.training.eval_interval) > 0:
        run_eval(model=model, val_loader=val_loader, cfg=cfg.training, device=device, dt=float(cfg.dataset.dt))

    if bool(cfg.training.export_torchscript):
        try:
            out_dir = str(cfg.output_dir) if cfg.output_dir else "outputs"
            import os

            os.makedirs(out_dir, exist_ok=True)
            ts = torch.jit.script(model)
            ts.save(os.path.join(out_dir, "fno_evio_model.pt"))
        except Exception:
            pass

    print("[FNO-EVIO] finished. config=", asdict(cfg))
