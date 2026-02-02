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
from fno_evio.training.step import IMUStateManager, StepResult, quat_to_rot, train_one_step, unpack_and_validate_batch, unpack_step_item, validate_dt_consistency
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
    """Manual gradient clipping that handles complex numbers (FNO-FAST compatible)."""
    total_norm = 0.0
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))

    for p in parameters:
        if p.grad.is_complex():
            # For complex parameters, use abs() before computing norm
            param_norm = p.grad.detach().abs().norm(2)
        else:
            param_norm = p.grad.detach().norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)


def _detach_tree(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach()
    if isinstance(v, (tuple, list)):
        return type(v)(_detach_tree(x) for x in v)
    if isinstance(v, dict):
        return {k: _detach_tree(val) for k, val in v.items()}
    return v


def _detach_hidden(hidden: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Detach hidden state for TBPTT (FNO-FAST compatible)."""
    if hidden is None:
        return None
    h, c = hidden
    return (h.detach().contiguous(), c.detach().contiguous())


def _adjust_hidden_size(hidden: Optional[Tuple[torch.Tensor, torch.Tensor]], B: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Adjust hidden state batch size (FNO-FAST compatible)."""
    if hidden is None:
        return None
    h, c = hidden
    # h, c: [L, B_curr, H]
    B_curr = h.size(1)
    if B_curr == B:
        return (h.contiguous(), c.contiguous())
    if B_curr > B:
        return (h[:, :B, :].contiguous(), c[:, :B, :].contiguous())
    # B_curr < B → pad zeros for new entries
    pad = B - B_curr
    dev = h.device
    H = h.size(2)
    zeros_h = torch.zeros(h.size(0), pad, H, device=dev, dtype=h.dtype)
    zeros_c = torch.zeros(c.size(0), pad, H, device=dev, dtype=c.dtype)
    return (
        torch.cat([h, zeros_h], dim=1).contiguous(),
        torch.cat([c, zeros_c], dim=1).contiguous(),
    )


def _get_ids(base_seq: Any, base_base: Any, starts_list: Optional[List[int]], s_idx: int, j: int, b: Optional[int] = None) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Get segment IDs for segment-aware reset (FNO-FAST compatible).

    Returns:
        (contig_id, gt_id, segment_id) tuple
    """
    start_b = starts_list[b] if (starts_list is not None and b is not None) else (base_seq.starts[s_idx] if hasattr(base_seq, "starts") else 0)
    idx = int(start_b) + int(j)
    if idx < 0:
        return None, None, None

    try:
        base_len = len(base_base)
        if base_len <= 0:
            return None, None, None
        if idx >= base_len:
            return None, None, None
    except Exception:
        pass

    contig_id = idx
    inner_idx = idx
    if hasattr(base_base, "indices"):
        try:
            if idx >= len(base_base.indices):
                return None, None, None
            inner_idx = int(base_base.indices[idx])
            contig_id = inner_idx
        except Exception:
            return None, None, None

    ds = base_base.dataset if hasattr(base_base, "dataset") else base_base
    gt_id = inner_idx
    if hasattr(ds, "sample_indices"):
        try:
            if inner_idx < 0 or inner_idx >= len(ds.sample_indices):
                return None, None, None
            gt_id = int(ds.sample_indices[inner_idx])
        except Exception:
            return None, None, None

    segment_id = None
    if hasattr(ds, "segment_ids"):
        try:
            if inner_idx >= 0 and inner_idx < len(ds.segment_ids):
                segment_id = int(ds.segment_ids[inner_idx])
        except Exception:
            segment_id = None

    return contig_id, gt_id, segment_id


def train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[Any],
    cfg: TrainingConfig,
    device: torch.device,
    dt_window_fallback: float,
    output_dir: Optional[str] = None,
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

    out_dir = None
    if output_dir is not None and str(output_dir).strip():
        out_dir = Path(str(output_dir)).expanduser().resolve()
        os.makedirs(str(out_dir), exist_ok=True)

    checkpoint_saving_enabled = True
    best_ckpt_path = (out_dir / "hybrid_vio_best.pth") if out_dir is not None else None
    last_ckpt_path = (out_dir / "hybrid_vio_last.pth") if out_dir is not None else None
    best_ckpt_saved = False

    def _try_save_checkpoint(tag: str, payload: Any, dst: Optional[Path]) -> bool:
        nonlocal checkpoint_saving_enabled
        if dst is None or not checkpoint_saving_enabled:
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
        try:
            try:
                torch.save(payload, str(tmp), _use_new_zipfile_serialization=False)
                os.replace(str(tmp), str(dst))
                print(f"[CKPT {tag}] saved to {dst}")
                return True
            except OSError as e:
                err_no = getattr(e, "errno", None)
                if err_no in (28, 122):
                    try:
                        torch.save(payload, str(dst), _use_new_zipfile_serialization=False)
                        print(f"[CKPT {tag}] saved to {dst} (non-atomic fallback)")
                        return True
                    except Exception:
                        checkpoint_saving_enabled = False
                raise
        except Exception as e:
            print(f"[CKPT {tag}] save failed: {e}")
            return False
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    best_metric = float("inf")  # For early stopping (uses composite/ma)
    best_ate = float("inf")  # For best checkpoint saving (pure ATE, like FNO-FAST)
    bad_epochs = 0
    metric_hist: List[float] = []

    imu_state_mgr = IMUStateManager(device)

    last_segment_id: Optional[int] = None
    global_hidden: Any = None  # Cross-batch hidden state (FNO-FAST behavior)

    # Get base dataset for segment_ids access
    base_seq = train_loader.dataset
    base_base = base_seq.dataset if hasattr(base_seq, "dataset") else base_seq

    for epoch in range(int(cfg.epochs)):
        total = 0.0
        count = 0
        # Reset segment tracking at epoch start (like FNO-FAST)
        last_segment_id = None
        global_hidden = None

        prev_pred_full: Optional[torch.Tensor] = None
        prev_gt_full: Optional[torch.Tensor] = None
        prev_dp_pred: Optional[torch.Tensor] = None  # For smoothness loss (FNO-FAST: prev_t_pred)
        for batch_idx, batch in enumerate(train_loader):
            batch_data, starts_list = unpack_and_validate_batch(batch)
            if not isinstance(batch_data, list):
                continue

            contig0, sid0, seg0 = _get_ids(base_seq, base_base, starts_list, batch_idx, 0, 0)

            # Check segment continuity
            contiguous = True
            if seg0 is not None and last_segment_id is not None:
                contiguous = (int(seg0) == int(last_segment_id))

            # Reset hidden state at segment boundaries (FNO-FAST behavior)
            if not contiguous:
                global_hidden = None
                print(f"[TRAIN RESET] batch_idx={batch_idx} seg_id={seg0} (segment boundary detected)")

            # Use global_hidden from previous batch (or None if reset)
            hidden = global_hidden

            # Reset IMU state at segment boundary or first batch
            if not contiguous or batch_idx == 0:
                imu_state_mgr.reset()

            # Initialize IMU rotation from first GT absolute quaternion if available
            # GT format: [0:3] delta_p, [3:7] delta_q, [7:11] q_prev, [11:14] v_prev, [14:17] v_curr
            # This matches FNO-FAST's behavior (train_fno_vio.py:4931-4941)
            if len(batch_data) > 0:
                try:
                    ev0, imu0, y0, _, _ = unpack_step_item(batch_data[0])
                    y0_dev = y0.to(device, non_blocking=True)
                    if y0_dev.ndim == 1:
                        y0_dev = y0_dev.unsqueeze(0)

                    # Adjust hidden size for batch size changes
                    B = ev0.size(0) if hasattr(ev0, 'size') else y0_dev.size(0)
                    hidden = _adjust_hidden_size(hidden, B) if hidden is not None else None

                    # Initialize IMU state at step_idx == 0 (FNO-FAST behavior)
                    imu_state_mgr.initialize(B)

                    # Use absolute quaternion at y[:, 7:11] for IMU rotation initialization
                    if y0_dev.shape[-1] >= 11:
                        q0 = y0_dev[:, 7:11]  # absolute quaternion [x, y, z, w]
                        q0 = torch.nn.functional.normalize(q0, p=2, dim=1)
                        imu_state_mgr.rotation = quat_to_rot(q0)
                    # Also initialize velocity from v_prev if available
                    if y0_dev.shape[-1] >= 14:
                        v0 = y0_dev[:, 11:14]  # v_prev
                        imu_state_mgr.velocity = v0
                except Exception:
                    pass

            step_idx = 0
            while step_idx < len(batch_data):
                optimizer.zero_grad(set_to_none=True)
                accum_loss: Optional[torch.Tensor] = None
                local_steps = 0
                seg_end = min(step_idx + tbptt_len, len(batch_data))
                for j in range(step_idx, seg_end):
                    ev, imu, y, dt_tensor, intr_tensor = unpack_step_item(batch_data[j])

                    # Move tensors to device
                    ev = ev.to(device, non_blocking=True)
                    imu = imu.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    # Validate and process dt_tensor (FNO-FAST compatible)
                    actual_dt = float(dt_window_fallback)
                    if dt_tensor is not None:
                        dt_tensor = dt_tensor.to(device, non_blocking=True)
                        actual_dt = validate_dt_consistency(dt_tensor, batch_idx)

                    # Process intrinsics tensor (FNO-FAST compatible)
                    fx_step, fy_step = float(physics_config.get('fx', 1.0)), float(physics_config.get('fy', 1.0))
                    if intr_tensor is not None:
                        intr_tensor = intr_tensor.to(device, non_blocking=True)
                        if intr_tensor.numel() >= 2:
                            intr_flat = intr_tensor.view(intr_tensor.shape[0], -1)
                            if intr_flat.shape[1] >= 2:
                                fx_step = float(intr_flat[:, 0].mean().item())
                                fy_step = float(intr_flat[:, 1].mean().item())

                    # Update physics config with per-step intrinsics
                    step_physics_config = dict(physics_config)
                    step_physics_config['fx'] = fx_step
                    step_physics_config['fy'] = fy_step

                    is_amp = bool(cfg.mixed_precision) and scaler is not None
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_amp):
                        res: StepResult = train_one_step(
                            model=model,
                            ev=ev,
                            imu=imu,
                            y=y,
                            hidden=hidden,
                            imu_state_mgr=imu_state_mgr,
                            device=device,
                            config=cfg,
                            loss_composer=loss_composer,
                            physics_module=physics_module,
                            physics_config=step_physics_config,
                            current_epoch_physics_weight=float(cfg.loss_w_physics),
                            adaptive_loss_fn=adaptive_loss_fn,
                            batch_idx=int(batch_idx),
                            step_idx=int(j),
                            dt_window_fallback=actual_dt,
                            is_amp=is_amp,
                            dt_tensor=dt_tensor,
                        )
                    hidden = res.hidden
                    loss_j = res.loss

                    # Progressive warmup (FNO-FAST compatible) - apply warmup multiplier instead of skipping
                    warmup_frames = int(getattr(cfg, "warmup_frames", 0))
                    if warmup_frames > 0 and j < warmup_frames:
                        warmup_factor = min(1.0, (j + 1) / warmup_frames)
                        loss_j = loss_j * warmup_factor
                        if j == 0 and batch_idx % 50 == 0:
                            print(f"[PROGRESSIVE_WARMUP] Step {j}/{warmup_frames}, factor: {warmup_factor:.2f}")
                        # Reset prev buffers during warmup (but still accumulate loss)
                        prev_pred_full = None
                        prev_gt_full = None
                        prev_dp_pred = None

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

                    # Smoothness loss (FNO-FAST compatible: mse_loss on displacement prediction differences)
                    if float(getattr(cfg, "loss_w_smooth", 0.0)) > 0.0:
                        try:
                            dp_pred = res.pred[:, 0:3]
                            if prev_dp_pred is not None:
                                # FNO-FAST uses mse_loss between current and previous prediction
                                smooth = torch.nn.functional.mse_loss(dp_pred, prev_dp_pred)
                                loss_j = loss_j + float(getattr(cfg, "loss_w_smooth", 0.0)) * smooth
                            prev_dp_pred = dp_pred.detach()
                        except Exception:
                            prev_dp_pred = None

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

                hidden = _detach_tree(hidden)
                imu_state_mgr.detach_states()

                step_idx += tbptt_stride

            # ========== FNO-FAST compatible: preserve hidden state for next batch ==========
            global_hidden = hidden
            last_segment_id = seg0

        mean_loss = total / max(count, 1)
        print(f"[TRAIN] epoch={epoch} mean_loss={mean_loss:.6f}")

        eval_ate = None
        eval_rpe_t = None
        eval_rpe_r = None
        if val_loader is not None and int(cfg.eval_interval) > 0 and ((epoch + 1) % int(cfg.eval_interval) == 0):
            # Temporarily disable plotting during regular eval (will plot only for best model)
            _old_eval_outdir = os.environ.get("FNO_EVIO_EVAL_OUTDIR", "")
            os.environ["FNO_EVIO_EVAL_OUTDIR"] = ""
            try:
                from fno_evio.eval.evaluate import evaluate

                model.eval()
                if isinstance(val_loader, dict):
                    ates: List[float] = []
                    rpes_t: List[float] = []
                    rpes_r: List[float] = []
                    for ld in val_loader.values():
                        ate_i, rpe_t_i, rpe_r_i, _ = evaluate(
                            model=model,
                            loader=ld,
                            device=device,
                            rpe_dt=float(cfg.rpe_dt),
                            dt=float(dt_window_fallback),
                            eval_sim3_mode=str(cfg.eval_sim3_mode),
                            speed_thresh=float(getattr(cfg, "speed_thresh", 0.0)),
                        )
                        ates.append(float(ate_i))
                        rpes_t.append(float(rpe_t_i))
                        rpes_r.append(float(rpe_r_i))
                    if ates:
                        eval_ate = float(sum(ates) / float(len(ates)))
                        eval_rpe_t = float(sum(rpes_t) / float(len(rpes_t)))
                        eval_rpe_r = float(sum(rpes_r) / float(len(rpes_r)))
                else:
                    eval_ate, eval_rpe_t, eval_rpe_r, _ = evaluate(
                        model=model,
                        loader=val_loader,
                        device=device,
                        rpe_dt=float(cfg.rpe_dt),
                        dt=float(dt_window_fallback),
                        eval_sim3_mode=str(cfg.eval_sim3_mode),
                        speed_thresh=float(getattr(cfg, "speed_thresh", 0.0)),
                    )
            except Exception:
                eval_ate = None
                eval_rpe_t = None
                eval_rpe_r = None
            finally:
                os.environ["FNO_EVIO_EVAL_OUTDIR"] = _old_eval_outdir
                model.train()

        actual_model = model.orig_mod if hasattr(model, "orig_mod") else model
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

        if eval_ate is not None and float(eval_ate) < best_ate:
            best_ate = float(eval_ate)
            if best_ckpt_path is not None:
                payload = {
                    "epoch": int(epoch + 1),
                    "metric": float(eval_ate),
                    "eval_ate": float(eval_ate),
                    "eval_rpe_t": float(eval_rpe_t) if eval_rpe_t is not None else None,
                    "eval_rpe_r": float(eval_rpe_r) if eval_rpe_r is not None else None,
                    "state_dict": actual_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                }
                best_ckpt_saved = _try_save_checkpoint("best", payload, best_ckpt_path) or best_ckpt_saved

                # Generate trajectory plots for best model
                _eval_outdir = os.environ.get("FNO_EVIO_EVAL_OUTDIR", "").strip()
                if _eval_outdir and val_loader is not None:
                    try:
                        from fno_evio.eval.evaluate import evaluate
                        model.eval()
                        _loader = list(val_loader.values())[0] if isinstance(val_loader, dict) else val_loader
                        _ = evaluate(
                            model=model,
                            loader=_loader,
                            device=device,
                            rpe_dt=float(cfg.rpe_dt),
                            dt=float(dt_window_fallback),
                            eval_sim3_mode=str(cfg.eval_sim3_mode),
                            speed_thresh=float(getattr(cfg, "speed_thresh", 0.0)),
                        )
                        print(f"[PLOT] Best model (ATE={best_ate:.6f}) trajectory saved to {_eval_outdir}")
                    except Exception as e:
                        print(f"[PLOT] Failed to generate best model plot: {e}")
                    finally:
                        model.train()

        # ========== Early stopping (composite/ma, like FNO-FAST) ==========
        # This controls when to STOP training, NOT when to save best checkpoint
        metric_mode = str(getattr(cfg, "earlystop_metric", "composite")).strip().lower()
        alpha = float(getattr(cfg, "earlystop_alpha", 1.0))
        beta = float(getattr(cfg, "earlystop_beta", 0.2))
        if metric_mode == "train_loss":
            metric_val = float(mean_loss)
        elif metric_mode in ("eval_loss", "ate"):
            metric_val = float(eval_ate) if eval_ate is not None else float(mean_loss)
        elif metric_mode == "composite":
            # FNO-FAST: ate + alpha * rpe_t + beta * rpe_r
            if eval_ate is not None:
                rpe_t_val = float(eval_rpe_t) if eval_rpe_t is not None else 0.0
                rpe_r_val = float(eval_rpe_r) if eval_rpe_r is not None else 0.0
                metric_val = float(eval_ate) + alpha * rpe_t_val + beta * rpe_r_val
            else:
                metric_val = float(mean_loss)
        else:
            metric_val = float(eval_ate) if eval_ate is not None else float(mean_loss)

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

        if last_ckpt_path is not None:
            payload = {
                "epoch": int(epoch + 1),
                "metric": float(ma_val),
                "eval_ate": float(eval_ate) if eval_ate is not None else None,
                "eval_rpe_t": float(eval_rpe_t) if eval_rpe_t is not None else None,
                "eval_rpe_r": float(eval_rpe_r) if eval_rpe_r is not None else None,
                "state_dict": actual_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            }
            _try_save_checkpoint("last", payload, last_ckpt_path)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(float(eval_ate) if eval_ate is not None else mean_loss)
        else:
            scheduler.step()

    if best_ckpt_saved and best_ckpt_path is not None:
        try:
            ckpt_obj = torch.load(str(best_ckpt_path), map_location="cpu")
            state = ckpt_obj.get("state_dict") if isinstance(ckpt_obj, dict) else ckpt_obj
            if isinstance(state, dict):
                actual_model = model.orig_mod if hasattr(model, "orig_mod") else model
                missing, unexpected = actual_model.load_state_dict(state, strict=False)
                print(f"[CKPT best] restored | missing={len(missing)} unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[CKPT best] restore failed: {e}")
