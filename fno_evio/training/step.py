"""
Single-step forward + loss accumulation for FNO-EVIO.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from fno_evio.config.schema import TrainingConfig
from fno_evio.training.loss_components import LossComposer
from fno_evio.training.loss_total import _compute_velocity_loss, compute_total_loss


@dataclass
class StepResult:
    loss: torch.Tensor
    lt: torch.Tensor
    lr: torch.Tensor
    lv: torch.Tensor
    lp: torch.Tensor
    lo: torch.Tensor
    pred: torch.Tensor
    hidden: Any


def unpack_and_validate_batch(batch: Any) -> Tuple[Any, List]:
    """
    Normalize dataloader batch formats.

    The baseline uses either:
      - (batch_data, starts_list) where batch_data is a list of time steps, or
      - a single-element list/tuple.
    """
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], list):
        batch_data, starts_list = batch
    else:
        batch_data = batch[0] if isinstance(batch, (tuple, list)) and len(batch) > 0 else batch
        starts_list = []
    return batch_data, starts_list


def unpack_step_item(item: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Unpack one temporal step item.
    """
    if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (list, tuple)):
        item = item[0]
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        raise ValueError("Invalid step item format")
    ev = item[0]
    imu = item[1]
    y = item[2]
    dt_tensor = item[3] if len(item) > 3 else None
    return ev, imu, y, dt_tensor


def train_one_step(
    *,
    model: nn.Module,
    ev: torch.Tensor,
    imu: torch.Tensor,
    y: torch.Tensor,
    hidden: Any,
    device: torch.device,
    config: TrainingConfig,
    loss_composer: LossComposer,
    physics_module: Optional[nn.Module],
    physics_config: dict,
    current_epoch_physics_weight: float,
    adaptive_loss_fn: Optional[Any],
    batch_idx: int,
    step_idx: int,
    dt_window_fallback: float,
    is_amp: bool,
    dt_tensor: Optional[torch.Tensor],
) -> StepResult:
    """
    Compute forward and total loss for a single temporal step.

    This is the refactored equivalent of the baseline `_forward_and_compute_loss`, with the
    responsibility split into:
      - forward pass and output unpack
      - loss component computation
      - total loss composition (strategy loop)
    """
    vox = ev.to(device, non_blocking=True)
    imu_batch = imu.to(device, non_blocking=True)
    y_dev = y.to(device, non_blocking=True)

    if vox.ndim == 3:
        vox = vox.unsqueeze(0)
    if imu_batch.ndim == 2:
        imu_batch = imu_batch.unsqueeze(0)

    out, hidden, new_v, _, raw_6d, s, ba_pred, bg_pred = model(
        vox,
        imu_batch,
        hidden,
        dt_window=float(dt_window_fallback),
        debug=(batch_idx == 0 and step_idx == 0),
    )

    lt, lr, lp, lo = loss_composer.compute_components(
        out,
        y_dev,
        raw_6d,
        vox,
        dt_tensor,
        physics_module,
        float(config.speed_thresh),
        float(dt_window_fallback),
        physics_config,
        scale_weight=float(getattr(config, "loss_w_scale", 0.0)),
        scale_reg_weight=float(getattr(config, "loss_w_scale_reg", 0.0)),
        static_weight=float(getattr(config, "loss_w_static", 0.0)),
        scale_reg_center=None,
        min_step_threshold=float(getattr(config, "min_step_threshold", 0.0)),
        min_step_weight=float(getattr(config, "min_step_weight", 0.0)),
        path_scale_weight=float(getattr(config, "loss_w_path_scale", 0.0)),
        s=s,
    )

    lv = _compute_velocity_loss(new_v, y_dev, device=device)

    loss = compute_total_loss(
        lt=lt,
        lr=lr,
        lv=lv,
        lp=lp,
        lo=lo,
        pred=out,
        y=y_dev,
        s=s,
        ba_pred=ba_pred,
        bg_pred=bg_pred,
        config=config,
        current_epoch_physics_weight=float(current_epoch_physics_weight),
        adaptive_loss_fn=adaptive_loss_fn,
        model=model,
        device=device,
        batch_idx=int(batch_idx),
        step_idx=int(step_idx),
        is_amp=bool(is_amp),
        dt_tensor=dt_tensor,
        dt_window_fallback=float(dt_window_fallback),
    )

    return StepResult(loss=loss, lt=lt, lr=lr, lv=lv, lp=lp, lo=lo, pred=out, hidden=hidden)

