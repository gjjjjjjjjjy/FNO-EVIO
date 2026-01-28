"""
Training loop for FNO-EVIO.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

Notes:
  This module exposes a clean training API. The exact TBPTT/segment-contiguity behavior can be
  aligned with the baseline incrementally without changing the loss/model core logic.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from fno_evio.common.constants import NumericalConstants
from fno_evio.config.schema import TrainingConfig
from fno_evio.training.loss_components import LossComposer
from fno_evio.training.physics import PhysicsBrightnessLoss
from fno_evio.training.step import StepResult, train_one_step, unpack_and_validate_batch, unpack_step_item
from fno_evio.utils.metrics import compute_rpe_loss


class AdaptiveLossWeights(nn.Module):
    def __init__(self, num_losses: int = 3) -> None:
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(int(num_losses)))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        dtype = losses[0].dtype
        device = losses[0].device
        terms: List[torch.Tensor] = []
        for i, loss in enumerate(losses):
            if float(loss.detach().item()) < 1e-9:
                continue
            loss_safe = torch.clamp(loss, min=1e-6)
            log_var = torch.clamp(self.log_vars[i], min=-2.5, max=10.0)
            precision = torch.exp(-log_var)
            terms.append(0.5 * precision * loss_safe + 0.5 * log_var)
        if not terms:
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        return torch.stack(terms).sum()


def _build_optimizer_and_scheduler(
    *, model: nn.Module, cfg: TrainingConfig, device: torch.device
) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, Optional[AdaptiveLossWeights], Optional[torch.cuda.amp.GradScaler]]:
    if bool(cfg.compile) and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead", backend=str(cfg.compile_backend))
        except Exception:
            pass

    adaptive_loss_fn = AdaptiveLossWeights(num_losses=4).to(device) if bool(cfg.adaptive_loss_weights) else None

    opt_name = str(getattr(cfg, "optimizer", "adamw")).strip().lower()
    params = list(model.parameters())
    has_complex_params = any(p.is_complex() for p in params)
    betas = tuple(getattr(cfg, "adam_betas", (0.9, 0.999)))
    eps = float(getattr(cfg, "adam_eps", 1e-8))
    weight_decay = float(getattr(cfg, "weight_decay", 0.01))
    if opt_name == "adamw":
        if adaptive_loss_fn is not None:
            opt = torch.optim.AdamW(
                [
                    {"params": params, "lr": float(cfg.lr)},
                    {"params": list(adaptive_loss_fn.parameters()), "lr": float(cfg.lr) * 0.1},
                ],
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
            )
        else:
            opt = torch.optim.AdamW(params, lr=float(cfg.lr), betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(params, lr=float(cfg.lr), betas=betas, eps=eps, weight_decay=weight_decay)

    sched_name = str(cfg.scheduler).strip().lower()
    if sched_name == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=max(int(cfg.epochs) // 3, 1), gamma=float(cfg.gamma))
    elif sched_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.scheduler_T_max))
    else:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=float(cfg.gamma), patience=int(cfg.scheduler_patience), mode="min")

    if int(cfg.warmup_epochs) > 0 and not isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=int(cfg.warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup_scheduler, main_scheduler], milestones=[int(cfg.warmup_epochs)])
    else:
        scheduler = main_scheduler

    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if bool(cfg.mixed_precision) and device.type == "cuda" and not has_complex_params:
        scaler = torch.cuda.amp.GradScaler()
    return model, opt, scheduler, adaptive_loss_fn, scaler


def _clip_gradients(model: nn.Module, max_norm: float) -> None:
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    except Exception:
        pass


def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[Any],
    cfg: TrainingConfig,
    device: torch.device,
    dt_window_fallback: float,
) -> None:
    """
    Train the model for `cfg.epochs` epochs.

    This is a minimal implementation to provide a stable modular API. It performs an optimizer
    step per temporal step; TBPTT refinement can be added while preserving this API.
    """
    model, optimizer, scheduler, adaptive_loss_fn, scaler = _build_optimizer_and_scheduler(model=model, cfg=cfg, device=device)
    loss_composer = LossComposer()
    model.train()

    physics_module = PhysicsBrightnessLoss().to(device) if str(getattr(cfg, "physics_mode", "none")).strip().lower() == "rotational" else None
    physics_config: dict = {
        "physics_scale_quantile": float(cfg.physics_scale_quantile),
        "physics_event_mask_thresh": float(cfg.physics_event_mask_thresh),
        "loss_w_physics_max": float(cfg.loss_w_physics_max),
        "physics_temp": float(cfg.physics_temp),
        "fx": 1.0,
        "fy": 1.0,
    }

    tbptt_len = max(int(cfg.tbptt_len), 1)
    tbptt_stride = int(cfg.tbptt_stride)
    if tbptt_stride <= 0:
        tbptt_stride = tbptt_len
    grad_clip = float(getattr(cfg, "grad_clip_norm", NumericalConstants.GRADIENT_CLIP_NORM))

    for epoch in range(int(cfg.epochs)):
        if epoch == 0:
            best_metric = float("inf")
            bad_epochs = 0
            metric_hist: List[float] = []
        total = 0.0
        count = 0
        hidden: Any = None
        prev_pred_full: Optional[torch.Tensor] = None
        prev_gt_full: Optional[torch.Tensor] = None
        prev_dp_pred: Optional[torch.Tensor] = None
        prev_dp_gt: Optional[torch.Tensor] = None
        for batch_idx, batch in enumerate(train_loader):
            batch_data, _ = unpack_and_validate_batch(batch)
            if not isinstance(batch_data, list):
                continue
            step_idx = 0
            while step_idx < len(batch_data):
                optimizer.zero_grad(set_to_none=True)
                accum_loss: Optional[torch.Tensor] = None
                local_steps = 0
                seg_end = min(step_idx + tbptt_len, len(batch_data))
                for j in range(step_idx, seg_end):
                    ev, imu, y, dt_tensor = unpack_step_item(batch_data[j])
                    is_amp = bool(cfg.mixed_precision) and scaler is not None
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_amp):
                        res: StepResult = train_one_step(
                            model=model,
                            ev=ev,
                            imu=imu,
                            y=y,
                            hidden=hidden,
                            device=device,
                            config=cfg,
                            loss_composer=loss_composer,
                            physics_module=physics_module,
                            physics_config=physics_config,
                            current_epoch_physics_weight=float(cfg.loss_w_physics),
                            adaptive_loss_fn=adaptive_loss_fn,
                            batch_idx=int(batch_idx),
                            step_idx=int(j),
                            dt_window_fallback=float(dt_window_fallback),
                            is_amp=is_amp,
                            dt_tensor=dt_tensor,
                        )
                    hidden = res.hidden
                    warmup_frames = int(getattr(cfg, "warmup_frames", 0))
                    if warmup_frames > 0 and j < warmup_frames:
                        prev_pred_full = None
                        prev_gt_full = None
                        prev_dp_pred = None
                        prev_dp_gt = None
                        continue

                    loss_j = res.loss

                    if bool(getattr(cfg, "use_rpe_loss", False)) and float(getattr(cfg, "loss_w_rpe", 0.0)) > 0.0:
                        try:
                            pred_full = res.pred[:, 0:7]
                            gt_full = y.to(device=device, dtype=pred_full.dtype)[:, 0:7]
                            if prev_pred_full is not None and prev_gt_full is not None:
                                loss_j = loss_j + compute_rpe_loss(prev_pred_full, pred_full, prev_gt_full, gt_full, float(getattr(cfg, "loss_w_rpe", 0.0)))
                            prev_pred_full = pred_full.detach()
                            prev_gt_full = gt_full.detach()
                        except Exception:
                            prev_pred_full = None
                            prev_gt_full = None

                    if float(getattr(cfg, "loss_w_smooth", 0.0)) > 0.0:
                        try:
                            dp_pred = res.pred[:, 0:3]
                            dp_gt = y.to(device=device, dtype=dp_pred.dtype)[:, 0:3]
                            if prev_dp_pred is not None and prev_dp_gt is not None:
                                smooth = torch.nn.functional.smooth_l1_loss(dp_pred - prev_dp_pred, dp_gt - prev_dp_gt, beta=0.1)
                                loss_j = loss_j + float(getattr(cfg, "loss_w_smooth", 0.0)) * smooth
                            prev_dp_pred = dp_pred.detach()
                            prev_dp_gt = dp_gt.detach()
                        except Exception:
                            prev_dp_pred = None
                            prev_dp_gt = None

                    if bool(getattr(cfg, "use_imu_consistency", False)) and float(getattr(cfg, "loss_w_imu", 0.0)) > 0.0:
                        try:
                            actual_model = model.orig_mod if hasattr(model, "orig_mod") else model
                            dbg = getattr(actual_model, "_last_step_debug", None)
                            t_imu = dbg.get("t_imu_tensor") if isinstance(dbg, dict) else None
                            if isinstance(t_imu, torch.Tensor) and t_imu.shape[0] == y.shape[0] and t_imu.shape[1] >= 3:
                                imu_loss = torch.nn.functional.smooth_l1_loss(t_imu[:, 0:3], y.to(device=t_imu.device, dtype=t_imu.dtype)[:, 0:3], beta=0.1)
                                loss_j = loss_j + float(getattr(cfg, "loss_w_imu", 0.0)) * imu_loss
                        except Exception:
                            pass

                    accum_loss = loss_j if accum_loss is None else (accum_loss + loss_j)
                    total += float(loss_j.detach().float().cpu().item())
                    count += 1
                    local_steps += 1

                if accum_loss is None or local_steps <= 0:
                    step_idx += tbptt_stride
                    continue

                norm_loss = accum_loss / float(max(local_steps, 1))
                if scaler is not None and bool(cfg.mixed_precision) and device.type == "cuda":
                    scaler.scale(norm_loss).backward()
                    scaler.unscale_(optimizer)
                    _clip_gradients(model, max_norm=grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    norm_loss.backward()
                    _clip_gradients(model, max_norm=grad_clip)
                    optimizer.step()

                try:
                    if hasattr(hidden, "detach"):
                        hidden = hidden.detach()
                except Exception:
                    pass

                step_idx += tbptt_stride

        mean_loss = total / max(count, 1)
        print(f"[TRAIN] epoch={epoch} mean_loss={mean_loss:.6f}")

        eval_ate = None
        eval_rpe_t = None
        eval_rpe_r = None
        if val_loader is not None and int(cfg.eval_interval) > 0 and ((epoch + 1) % int(cfg.eval_interval) == 0):
            try:
                from fno_evio.eval.evaluate import evaluate

                model.eval()
                if isinstance(val_loader, dict):
                    ates: List[float] = []
                    rpes_t: List[float] = []
                    rpes_r: List[float] = []
                    for ld in val_loader.values():
                        ate_i, rpe_t_i, rpe_r_i = evaluate(
                            model=model,
                            loader=ld,
                            device=device,
                            rpe_dt=float(cfg.rpe_dt),
                            dt=float(dt_window_fallback),
                            eval_sim3_mode=str(cfg.eval_sim3_mode),
                        )
                        ates.append(float(ate_i))
                        rpes_t.append(float(rpe_t_i))
                        rpes_r.append(float(rpe_r_i))
                    if ates:
                        eval_ate = float(sum(ates) / float(len(ates)))
                        eval_rpe_t = float(sum(rpes_t) / float(len(rpes_t)))
                        eval_rpe_r = float(sum(rpes_r) / float(len(rpes_r)))
                else:
                    eval_ate, eval_rpe_t, eval_rpe_r = evaluate(
                        model=model,
                        loader=val_loader,
                        device=device,
                        rpe_dt=float(cfg.rpe_dt),
                        dt=float(dt_window_fallback),
                        eval_sim3_mode=str(cfg.eval_sim3_mode),
                    )
            except Exception:
                eval_ate = None
                eval_rpe_t = None
                eval_rpe_r = None
            finally:
                model.train()

        metrics_csv_path = str(getattr(cfg, "metrics_csv", "") or "").strip()
        if metrics_csv_path and eval_ate is not None:
            p = Path(metrics_csv_path).expanduser().resolve()
            os.makedirs(str(p.parent), exist_ok=True)
            file_exists = p.exists()
            need_header = (not file_exists) or p.stat().st_size == 0
            with p.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if need_header:
                    w.writerow(["epoch", "loss", "ate", "rpe_t", "rpe_r"])
                w.writerow(
                    [
                        int(epoch + 1),
                        float(mean_loss),
                        float(eval_ate),
                        float(eval_rpe_t) if eval_rpe_t is not None else float("nan"),
                        float(eval_rpe_r) if eval_rpe_r is not None else float("nan"),
                    ]
                )
                f.flush()

        metric_mode = str(getattr(cfg, "earlystop_metric", "composite")).strip().lower()
        alpha = float(getattr(cfg, "earlystop_alpha", 1.0))
        beta = float(getattr(cfg, "earlystop_beta", 0.2))
        if metric_mode == "train_loss":
            metric_val = float(mean_loss)
        elif metric_mode == "eval_loss":
            metric_val = float(eval_ate) if eval_ate is not None else float(mean_loss)
        else:
            base_val = float(eval_ate) if eval_ate is not None else float(mean_loss)
            metric_val = alpha * base_val + beta * float(mean_loss)
        metric_hist.append(metric_val)
        ma_w = max(int(getattr(cfg, "earlystop_ma_window", 1)), 1)
        ma_val = float(sum(metric_hist[-ma_w:]) / float(min(len(metric_hist), ma_w)))
        if epoch + 1 >= int(getattr(cfg, "earlystop_min_epoch", 0)):
            if ma_val + 1e-12 < best_metric:
                best_metric = ma_val
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= int(getattr(cfg, "patience", 10)):
                print(f"[EARLYSTOP] epoch={epoch} best_ma={best_metric:.6f} current_ma={ma_val:.6f}")
                break

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(float(eval_ate) if eval_ate is not None else mean_loss)
        else:
            scheduler.step()
