"""
Total loss composition for FNO-EVIO (strategy-based, refactored).

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fno_evio.config.schema import TrainingConfig


def _compute_velocity_loss(new_v: torch.Tensor, y: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """
    Velocity supervision against ground-truth current velocity if available.
    """
    if y.ndim != 2 or y.shape[1] < 17:
        return torch.tensor(0.0, device=device)
    v_gt_curr = y[:, 14:17]
    return F.smooth_l1_loss(new_v, v_gt_curr, beta=0.1)


def _get_actual_model(model: nn.Module) -> nn.Module:
    m = model.module if hasattr(model, "module") else model
    m = m.orig_mod if hasattr(m, "orig_mod") else m
    return m


def _loss_base(
    *,
    lt: torch.Tensor,
    lr: torch.Tensor,
    lv: torch.Tensor,
    lp: torch.Tensor,
    lo: torch.Tensor,
    config: TrainingConfig,
    current_epoch_physics_weight: float,
    adaptive_loss_fn: Optional[Any],
    is_amp: bool,
    device: torch.device,
    batch_idx: int,
) -> torch.Tensor:
    p_term = (float(current_epoch_physics_weight) * lp) if float(current_epoch_physics_weight) > 1e-6 else torch.tensor(0.0, device=device)

    if adaptive_loss_fn is not None:
        lt_w = float(config.loss_w_t) * lt
        lr_w = float(config.loss_w_r) * lr
        lv_w = float(config.loss_w_v) * lv
        loss_list = [lt_w, lr_w, lv_w, p_term]
        loss = adaptive_loss_fn(loss_list) + float(config.loss_w_ortho) * lo
        if float(loss.detach().item()) > 1000.0:
            log_vars_vals = adaptive_loss_fn.log_vars.detach().float().cpu().numpy()
            amp_tag = "AMP " if is_amp else ""
            print(
                f"[WARN] {amp_tag}High Loss Batch {int(batch_idx)}: {loss.item():.4f} | "
                f"T={lt.detach().item():.4f}, R={lr.detach().item():.4f}, V={lv.detach().item():.4f}, "
                f"P={p_term.detach().item():.4f} | LogVars={log_vars_vals}"
            )
        return loss

    loss = float(config.loss_w_t) * lt + float(config.loss_w_r) * lr + p_term + float(config.loss_w_ortho) * lo + float(config.loss_w_v) * lv
    threshold = 1000.0 if is_amp else 200.0
    if float(loss.detach().item()) > threshold:
        amp_tag = "AMP " if is_amp else ""
        print(
            f"[WARN] {amp_tag}High Loss Batch {int(batch_idx)}: {loss.detach().item():.4f} | "
            f"T={lt.detach().item():.4f}, R={lr.detach().item():.4f}, V={lv.detach().item():.4f}, "
            f"P={p_term.detach().item():.4f} (Fixed Weights)"
        )
    return loss


def _loss_aux_motion(
    *,
    loss: torch.Tensor,
    y: torch.Tensor,
    config: TrainingConfig,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    w_aux_motion = float(getattr(config, "loss_w_aux_motion", 0.0))
    if w_aux_motion <= 0.0:
        return loss

    actual_model = _get_actual_model(model)
    dbg = getattr(actual_model, "_last_step_debug", None)
    aux_motion = dbg.get("aux_motion_tensor") if isinstance(dbg, dict) else None
    if not (isinstance(aux_motion, torch.Tensor) and aux_motion.ndim == 2 and aux_motion.shape[1] >= 7 and y.ndim == 2 and y.shape[1] >= 7):
        return loss

    aux_t = aux_motion[:, 0:3]
    aux_q = F.normalize(aux_motion[:, 3:7], p=2, dim=1)
    gt_t = y[:, 0:3]
    gt_q = F.normalize(y[:, 3:7], p=2, dim=1)

    t_imu = dbg.get("t_imu_tensor") if isinstance(dbg, dict) else None
    if not isinstance(t_imu, torch.Tensor) and isinstance(dbg, dict):
        t_hat_body_vec = dbg.get("t_hat_body_vec")
        if isinstance(t_hat_body_vec, np.ndarray) and t_hat_body_vec.shape == (gt_t.shape[0], 3):
            t_imu = torch.from_numpy(t_hat_body_vec).to(device=gt_t.device, dtype=gt_t.dtype)

    gt_t_for_aux = gt_t - t_imu.detach() if (isinstance(t_imu, torch.Tensor) and t_imu.shape == gt_t.shape) else gt_t
    lt_aux = F.smooth_l1_loss(aux_t, gt_t_for_aux, beta=0.1)
    dot = torch.sum(aux_q * gt_q, dim=1).abs()
    lr_aux = (1.0 - dot).mean()
    return loss + w_aux_motion * (lt_aux + lr_aux)


def _loss_seq_scale_reg(
    *,
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    config: TrainingConfig,
    dt_tensor: Optional[torch.Tensor],
    dt_window_fallback: float,
    batch_idx: int,
    step_idx: int,
) -> torch.Tensor:
    if not bool(getattr(config, "use_seq_scale", False)):
        return loss
    w = float(getattr(config, "seq_scale_reg", 0.0))
    if w <= 0.0:
        return loss

    dp_pred = pred[:, 0:3]
    dp_gt = y[:, 0:3]
    dp_norm = dp_gt.norm(dim=1)
    if dt_tensor is not None:
        dt_local = dt_tensor.view(-1).to(dp_norm.device).clamp(min=1e-6)
    else:
        dt_local = torch.full_like(dp_norm, float(dt_window_fallback)).clamp(min=1e-6)
    moving = dp_norm > (float(config.speed_thresh) * dt_local)
    if not torch.any(moving):
        return loss

    num = torch.sum((dp_pred[moving] * dp_gt[moving]).sum(dim=1))
    den = torch.sum(dp_pred[moving].norm(dim=1) ** 2) + 1e-6
    s_step = torch.clamp(num / den, min=1e-6, max=1e6)
    loss = loss + w * (torch.log(s_step) ** 2)
    if batch_idx == 0 and step_idx == 0:
        print(f"[TRAIN SCALE] seq_scale_reg enabled (w={w:.4f}) | s_step_ols={float(s_step.detach().item()):.6f}")
    return loss


def _loss_bias_prior(
    *,
    loss: torch.Tensor,
    ba_pred: Optional[torch.Tensor],
    bg_pred: Optional[torch.Tensor],
    config: TrainingConfig,
    batch_idx: int,
    step_idx: int,
) -> torch.Tensor:
    w_ba = float(getattr(config, "loss_w_bias_a", 0.0))
    w_bg = float(getattr(config, "loss_w_bias_g", 0.0))

    if w_ba > 0.0 and ba_pred is not None:
        prior = getattr(config, "bias_prior_accel", None)
        if prior is not None and len(prior) == 3:
            ba0 = torch.tensor([float(prior[0]), float(prior[1]), float(prior[2])], device=ba_pred.device, dtype=ba_pred.dtype).view(1, 3)
            loss = loss + w_ba * (ba_pred - ba0).pow(2).mean()
            if batch_idx == 0 and step_idx == 0 and not hasattr(_loss_bias_prior, "_printed_ba"):
                print(f"[BIAS PRIOR] accel={float(prior[0]):+.6e},{float(prior[1]):+.6e},{float(prior[2]):+.6e} (normalized)")
                _loss_bias_prior._printed_ba = True
        else:
            loss = loss + w_ba * ba_pred.pow(2).mean()

    if w_bg > 0.0 and bg_pred is not None:
        prior = getattr(config, "bias_prior_gyro", None)
        if prior is not None and len(prior) == 3:
            bg0 = torch.tensor([float(prior[0]), float(prior[1]), float(prior[2])], device=bg_pred.device, dtype=bg_pred.dtype).view(1, 3)
            loss = loss + w_bg * (bg_pred - bg0).pow(2).mean()
            if batch_idx == 0 and step_idx == 0 and not hasattr(_loss_bias_prior, "_printed_bg"):
                print(f"[BIAS PRIOR] gyro={float(prior[0]):+.6e},{float(prior[1]):+.6e},{float(prior[2]):+.6e} (normalized)")
                _loss_bias_prior._printed_bg = True
        else:
            loss = loss + w_bg * bg_pred.pow(2).mean()

    return loss


def _loss_correction_reg(
    *,
    loss: torch.Tensor,
    config: TrainingConfig,
    model: nn.Module,
    device: torch.device,
    batch_idx: int,
    step_idx: int,
) -> torch.Tensor:
    w_correction = float(getattr(config, "loss_w_correction", 0.0))
    if w_correction <= 0.0:
        return loss

    actual_model = _get_actual_model(model)
    debug_info = getattr(actual_model, "_last_step_debug", None)
    if not isinstance(debug_info, dict):
        return loss
    pos_res_vec = debug_info.get("pos_res_vec")
    if not (pos_res_vec is not None and isinstance(pos_res_vec, np.ndarray)):
        return loss

    pos_res_tensor = torch.from_numpy(pos_res_vec).to(device=device, dtype=torch.float32)
    correction_loss = pos_res_tensor.pow(2).mean()
    loss = loss + w_correction * correction_loss
    if batch_idx == 0 and step_idx == 0:
        print(f"[DEIO] correction_reg enabled (w={w_correction:.4f}) | ||r||={correction_loss.item():.6f}")
    return loss


def _loss_uncertainty_nll(
    *,
    loss: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    s: Optional[torch.Tensor],
    config: TrainingConfig,
    model: nn.Module,
) -> torch.Tensor:
    w_uncertainty = float(getattr(config, "loss_w_uncertainty", 0.1))
    w_uncertainty_calib = float(getattr(config, "loss_w_uncertainty_calib", 0.0))
    if not hasattr(_loss_uncertainty_nll, "_printed_cfg"):
        print(f"[NLL_CONFIG] loss_w_uncertainty={w_uncertainty:.4f} (>0 enables uncertainty learning)")
        _loss_uncertainty_nll._printed_cfg = True
    if w_uncertainty <= 0.0 and w_uncertainty_calib <= 0.0:
        return loss

    actual_model = _get_actual_model(model)
    debug_info = getattr(actual_model, "_last_step_debug", None)
    if not isinstance(debug_info, dict):
        return loss

    log_var_v = debug_info.get("log_var_v_tensor")
    log_var_i = debug_info.get("log_var_i_tensor")
    t_imu = debug_info.get("t_imu_tensor")
    t_visual = debug_info.get("t_visual_tensor")

    if not hasattr(_loss_uncertainty_nll, "_printed_tensors"):
        has_all = (log_var_v is not None and log_var_i is not None and t_imu is not None and t_visual is not None)
        print(
            f"[NLL_TENSORS] log_var_v={log_var_v is not None}, log_var_i={log_var_i is not None}, "
            f"t_imu={t_imu is not None}, t_visual={t_visual is not None} | all_present={has_all}"
        )
        if has_all:
            print(f"[NLL_TENSORS] log_var_v.requires_grad={getattr(log_var_v, 'requires_grad', False)}, log_var_i.requires_grad={getattr(log_var_i, 'requires_grad', False)}")
        _loss_uncertainty_nll._printed_tensors = True

    if not (isinstance(log_var_v, torch.Tensor) and isinstance(log_var_i, torch.Tensor) and isinstance(t_imu, torch.Tensor) and isinstance(t_visual, torch.Tensor)):
        return loss

    target_t = y[:, 0:3]
    residual_imu = t_imu - target_t
    residual_visual = t_visual - target_t

    log_var_v_eff = log_var_v
    if bool(getattr(actual_model, "uncertainty_use_gate", False)) and s is not None:
        s_eff = torch.clamp(s.to(dtype=log_var_v.dtype), min=1e-3, max=1e6).view(-1, 1)
        log_var_v_eff = log_var_v_eff - 2.0 * torch.log(s_eff)

    var_v_eff = torch.exp(torch.clamp(log_var_v_eff, min=-10.0, max=10.0))
    var_i = torch.exp(torch.clamp(log_var_i, min=-10.0, max=10.0))

    nll_visual = 0.5 * (log_var_v_eff + residual_visual.pow(2) / (var_v_eff + 1e-6))
    nll_imu = 0.5 * (log_var_i + residual_imu.pow(2) / (var_i + 1e-6))
    nll_loss = nll_visual.mean() + nll_imu.mean()

    if w_uncertainty > 0.0:
        loss = loss + w_uncertainty * nll_loss

    if w_uncertainty_calib > 0.0:
        rv2 = residual_visual.detach().pow(2) + 1e-4
        ri2 = residual_imu.detach().pow(2) + 1e-4
        target_log_var_v = torch.log(rv2).to(dtype=log_var_v_eff.dtype)
        target_log_var_i = torch.log(ri2).to(dtype=log_var_i.dtype)
        calib_loss = F.smooth_l1_loss(log_var_v_eff, target_log_var_v, beta=1.0) + F.smooth_l1_loss(log_var_i, target_log_var_i, beta=1.0)
        loss = loss + w_uncertainty_calib * calib_loss

    if torch.rand(1).item() < 0.001:
        nll_v_m = float(nll_visual.mean().detach().cpu().item())
        nll_i_m = float(nll_imu.mean().detach().cpu().item())
        nll_m = float(nll_loss.detach().cpu().item())
        print(f"[NLL_DEBUG] nll_v={nll_v_m:.4f} nll_i={nll_i_m:.4f} w={w_uncertainty:.4f} total_nll={nll_m:.4f}")

    return loss


def compute_total_loss(
    *,
    lt: torch.Tensor,
    lr: torch.Tensor,
    lv: torch.Tensor,
    lp: torch.Tensor,
    lo: torch.Tensor,
    pred: torch.Tensor,
    y: torch.Tensor,
    s: Optional[torch.Tensor],
    ba_pred: Optional[torch.Tensor],
    bg_pred: Optional[torch.Tensor],
    config: TrainingConfig,
    current_epoch_physics_weight: float,
    adaptive_loss_fn: Optional[Any],
    model: nn.Module,
    device: torch.device,
    batch_idx: int,
    step_idx: int,
    is_amp: bool,
    dt_tensor: Optional[torch.Tensor],
    dt_window_fallback: float,
) -> torch.Tensor:
    """
    Strategy-driven total loss composition (refactor of baseline _compute_total_loss).
    """
    strategies: "OrderedDict[str, Callable[[torch.Tensor], torch.Tensor]]" = OrderedDict()
    strategies["base"] = lambda loss0: _loss_base(
        lt=lt,
        lr=lr,
        lv=lv,
        lp=lp,
        lo=lo,
        config=config,
        current_epoch_physics_weight=current_epoch_physics_weight,
        adaptive_loss_fn=adaptive_loss_fn,
        is_amp=is_amp,
        device=device,
        batch_idx=batch_idx,
    )
    strategies["aux_motion"] = lambda loss0: _loss_aux_motion(loss=loss0, y=y, config=config, model=model, device=device)
    strategies["seq_scale_reg"] = lambda loss0: _loss_seq_scale_reg(
        loss=loss0,
        pred=pred,
        y=y,
        config=config,
        dt_tensor=dt_tensor,
        dt_window_fallback=dt_window_fallback,
        batch_idx=batch_idx,
        step_idx=step_idx,
    )
    strategies["bias_prior"] = lambda loss0: _loss_bias_prior(loss=loss0, ba_pred=ba_pred, bg_pred=bg_pred, config=config, batch_idx=batch_idx, step_idx=step_idx)
    strategies["correction_reg"] = lambda loss0: _loss_correction_reg(loss=loss0, config=config, model=model, device=device, batch_idx=batch_idx, step_idx=step_idx)
    strategies["uncertainty_nll"] = lambda loss0: _loss_uncertainty_nll(loss=loss0, pred=pred, y=y, s=s, config=config, model=model)

    loss = torch.tensor(0.0, device=device, dtype=pred.dtype)
    for name, fn in strategies.items():
        loss = fn(loss)
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite loss after strategy '{name}'")
    return loss

