"""
FNO-EVIO evaluation pipeline (ATE/RPE, optional SIM(3) diagnostics).

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fno_evio.utils.quaternion_np import QuaternionUtils
from fno_evio.utils.trajectory import (
    align_trajectory_with_timestamps,
    align_trajectory_with_timestamps_sim3,
)


@dataclass
class EvalDiagnostics:
    """Container for verbose evaluation statistics."""

    gate_mode: bool
    eval_sim3_mode: str
    dt_list: List[float]
    s_list: List[float]
    w_imu_list: List[float]
    w_visual_list: List[float]
    sigma_imu_list: List[float]
    sigma_visual_list: List[float]
    win_step_pred: List[float]
    win_step_gt: List[float]
    base_sample_stride: int
    contiguous_sid_step: int
    has_window_ts: bool


@dataclass
class EvalTrajectory:
    """Aligned trajectory inputs and bookkeeping for metrics."""

    est_pos: np.ndarray
    est_quat: np.ndarray
    gt_pos: np.ndarray
    gt_quat: np.ndarray
    gt_t: np.ndarray
    sample_ids: np.ndarray
    contig_ids: np.ndarray
    seg_keys: np.ndarray


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    rpe_dt: float,
    dt: float,
    eval_sim3_mode: str = "diagnose",
    speed_thresh: float = 0.0,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Evaluate a VIO model on a sequence loader.

    Args:
        model: Model returning (out, hidden, v, R, rot6d, s, ba, bg) per step.
        loader: Sequence DataLoader (collate returns a list of time steps).
        device: torch device.
        rpe_dt: Target temporal separation (seconds) for RPE.
        dt: Fallback dt if per-sample dt is unavailable.
        eval_sim3_mode: SIM(3) diagnostic mode; kept for compatibility.

    Returns:
        ate: Absolute Trajectory Error (meters), after SE(3) alignment.
        rpe_t: Relative Pose Error in translation (meters).
        rpe_r: Relative Pose Error in rotation (degrees).
        state: Dictionary containing additional evaluation metrics and diagnostics.
    """
    model.eval()

    gate_mode = _infer_gate_mode(model)
    if hasattr(model, "scale_min") and hasattr(model, "scale_max"):
        smin = float(getattr(model, "scale_min"))
        smax = float(getattr(model, "scale_max"))
        mode_str = "gate" if gate_mode else "experimental_scale"
        print(f"[EVAL START] scale_head range=[{smin:.6f}, {smax:.6f}] | mode={mode_str}")

    base_seq = loader.dataset
    base_base = base_seq.base if hasattr(base_seq, "base") else loader.dataset
    base_ds = base_base.dataset if hasattr(base_base, "dataset") else base_base
    has_window_ts = bool(
        hasattr(base_ds, "window_t_curr")
        and getattr(base_ds, "window_t_curr") is not None
        and hasattr(base_ds, "interpolate_gt_data")
    )

    base_sample_stride, contiguous_sid_step = _extract_sample_stride(base_ds, base_seq, base_base)

    diag = EvalDiagnostics(
        gate_mode=gate_mode,
        eval_sim3_mode=str(eval_sim3_mode),
        dt_list=[],
        s_list=[],
        w_imu_list=[],
        w_visual_list=[],
        sigma_imu_list=[],
        sigma_visual_list=[],
        win_step_pred=[],
        win_step_gt=[],
        base_sample_stride=base_sample_stride,
        contiguous_sid_step=contiguous_sid_step,
        has_window_ts=has_window_ts,
    )

    traj = align_trajectory_eval(
        model=model,
        loader=loader,
        device=device,
        dt_fallback=float(dt),
        speed_thresh=float(speed_thresh),
        base_sample_stride=base_sample_stride,
        contiguous_sid_step=contiguous_sid_step,
        has_window_ts=has_window_ts,
        diag=diag,
    )

    ate, rpe_t, rpe_r, log_state = compute_eval_metrics(
        traj=traj,
        diag=diag,
        rpe_dt=float(rpe_dt),
    )

    log_eval_summary(ate=ate, rpe_t=rpe_t, rpe_r=rpe_r, diag=diag, state=log_state)
    plot_eval(traj=traj, diag=diag, state=log_state)
    save_eval_outputs(traj=traj, diag=diag, state=log_state)

    return ate, rpe_t, rpe_r, log_state


def align_trajectory_eval(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dt_fallback: float,
    speed_thresh: float,
    base_sample_stride: int,
    contiguous_sid_step: int,
    has_window_ts: bool,
    diag: EvalDiagnostics,
) -> EvalTrajectory:
    """
    Run the model on the loader and construct matched (est, gt) trajectories.

    This function implements the data plumbing: state reset across sequences, optional
    window-timestamp GT interpolation, and rich debug-stat collection.
    """
    est_pos: List[np.ndarray] = []
    est_quat: List[np.ndarray] = []
    sample_ids: List[int] = []
    contig_ids: List[int] = []
    seg_keys: List[int] = []

    base_seq = loader.dataset
    base_base = base_seq.base if hasattr(base_seq, "base") else loader.dataset
    base_ds = base_base.dataset if hasattr(base_base, "dataset") else base_base

    t = np.zeros(3, dtype=np.float64)
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    with torch.no_grad():
        diag_zero_prev_v = os.environ.get("EVAL_DIAG_ZERO_PREV_V", "0").strip() == "1"
        diag_zero_prev_R = os.environ.get("EVAL_DIAG_ZERO_PREV_R", "0").strip() == "1"
        if diag_zero_prev_v or diag_zero_prev_R:
            print(f"[EVAL DIAG FLAGS] zero_prev_v={diag_zero_prev_v} zero_prev_R={diag_zero_prev_R}")

        global_eval_v = None
        global_eval_R = None
        global_hidden = None
        global_t_batch = None
        global_q_batch = None
        last_sid_b0 = None
        last_segment_id = None
        printed_norm_debug = False

        prev_v_probe = None
        prev_R_probe = None

        for s_idx, batch in enumerate(loader):
            batched_seq, starts_list = _unpack_eval_batch(batch)
            hidden = None
            if not batched_seq:
                continue

            B = batched_seq[0][0].shape[0]

            contig0, sid0, seg0 = _get_ids(base_seq, base_base, starts_list, s_idx, 0, 0)
            contiguous = _is_contiguous(
                seg0=seg0,
                last_segment_id=last_segment_id,
                has_window_ts=has_window_ts,
                contig0=contig0,
                last_sid_b0=last_sid_b0,
                sid0=sid0,
                contiguous_sid_step=contiguous_sid_step,
            )
            if bool(getattr(base_ds, "_force_continuous_eval", False)):
                contiguous = True

            t_batch, q_batch = _init_batch_pose(B, contiguous, global_t_batch, global_q_batch)

            if contiguous:
                eval_v = global_eval_v
                eval_R = global_eval_R
                hidden = global_hidden
            else:
                eval_v = None
                eval_R = None
                hidden = None
                prev_v_probe = None
                prev_R_probe = None
                if global_eval_v is not None:
                    try:
                        print(
                            f"[EVAL RESET] s_idx={int(s_idx)} sid0={int(sid0)} | "
                            f"drop_prev_v_norm={float(global_eval_v.detach().norm(dim=1).mean().item()):.3e}"
                        )
                    except Exception:
                        print(f"[EVAL RESET] s_idx={int(s_idx)} sid0={int(sid0)}")

            for j, item in enumerate(batched_seq):
                if j % base_sample_stride != 0:
                    continue

                ev, imu, y, actual_dt = _unpack_eval_item(item, dt_fallback=dt_fallback)
                diag.dt_list.append(float(actual_dt))

                vox = ev.to(device, non_blocking=True)
                imu_batch = imu.to(device, non_blocking=True)
                if vox.ndim == 3:
                    vox = vox.unsqueeze(0)
                if imu_batch.ndim == 2:
                    imu_batch = imu_batch.unsqueeze(0)

                hidden = _adjust_hidden_size(hidden, vox.size(0))
                y_dev = y.to(device, non_blocking=True)

                if eval_R is None and y_dev.ndim >= 2 and y_dev.shape[1] >= 14:
                    q_prev = F.normalize(y_dev[:, 7:11], p=2, dim=1)
                    eval_R = _quat_to_rot(q_prev)
                    eval_v = y_dev[:, 11:14].contiguous()

                pv_in = None if diag_zero_prev_v else eval_v
                pr_in = None if diag_zero_prev_R else eval_R

                out, hidden, eval_v, eval_R, _, s, _, _ = model(
                    vox,
                    imu_batch,
                    hidden,
                    prev_v=pv_in,
                    prev_R=pr_in,
                    dt_window=float(actual_dt),
                    debug=(not printed_norm_debug),
                )

                if not printed_norm_debug:
                    _print_eval_norm_debug(model=model, y=y_dev, actual_dt=float(actual_dt))
                    printed_norm_debug = True

                if s is not None:
                    diag.s_list.extend([float(v) for v in s.detach().flatten().cpu().tolist()])
                _collect_uncertainty_stats(model=model, diag=diag)

                global_eval_v = eval_v
                global_eval_R = eval_R
                global_hidden = hidden

                out_np = out.detach().float().cpu().numpy()
                td = out_np[:, 0:3].astype(np.float64)
                qd = out_np[:, 3:7].astype(np.float64)
                tdelta_gt = y.detach().float().cpu().numpy()[:, 0:3].astype(np.float64)

                s_np = None
                if s is not None:
                    try:
                        s_np = s.detach().float().flatten().cpu().numpy().astype(np.float64)
                    except Exception:
                        s_np = None

                if s_idx == 0 and j == 0:
                    print(f"[EVAL DEBUG] First prediction td={td[0]} (norm={np.linalg.norm(td[0]):.6f})")
                    if s_np is not None and s_np.size > 0:
                        mode_str = "gate" if diag.gate_mode else "scale"
                        print(f"[EVAL DEBUG] scale_s median={float(np.median(s_np)):.6f} | mode={mode_str}")

                for b in range(int(B)):
                    tb, qb = QuaternionUtils.compose_se3(
                        t_batch[b],
                        q_batch[b],
                        td[b],
                        QuaternionUtils.normalize(qd[b]),
                    )
                    t_batch[b] = tb
                    q_batch[b] = qb

                    contig_id, sid, segment_id = _get_ids(base_seq, base_base, starts_list, s_idx, j, b)
                    if sid is None:
                        continue

                    if b == 0:
                        _maybe_print_window_probe(
                            sid=int(sid),
                            s_idx=int(s_idx),
                            j=int(j),
                            b=int(b),
                            base_ds=base_ds,
                            contig_id=contig_id,
                            base_sample_stride=int(base_sample_stride),
                            contiguous_sid_step=int(contiguous_sid_step),
                            has_window_ts=bool(has_window_ts),
                            imu_batch=imu_batch,
                            y_dev=y_dev,
                            td=td,
                            tdelta_gt=tdelta_gt,
                            s_np=s_np,
                            actual_dt=float(actual_dt),
                            speed_thresh=float(speed_thresh),
                            eval_v=eval_v,
                            eval_R=eval_R,
                            prev_v_probe=prev_v_probe,
                            diag=diag,
                            model=model,
                        )

                        prev_v_probe = eval_v.detach().clone() if isinstance(eval_v, torch.Tensor) else None
                        prev_R_probe = eval_R.detach().clone() if isinstance(eval_R, torch.Tensor) else None

                    est_pos.append(tb.copy())
                    est_quat.append(qb.copy())
                    sample_ids.append(int(sid))
                    contig_ids.append(int(contig_id) if contig_id is not None else -1)
                    # Use actual segment_id from dataset instead of s_idx to avoid
                    # splitting continuous trajectories across SequenceDataset batches
                    seg_key = int(segment_id) if segment_id is not None else int(s_idx) * 1_000_000 + int(b)
                    seg_keys.append(seg_key)
                    diag.win_step_pred.append(float(np.linalg.norm(td[b])))
                    diag.win_step_gt.append(float(np.linalg.norm(tdelta_gt[b])))

                    if b == 0:
                        last_sid_b0 = contig_id if has_window_ts else sid
                        last_segment_id = segment_id
                    if b == B - 1:
                        global_t_batch = t_batch.copy()
                        global_q_batch = q_batch.copy()

    est_pos_arr = np.asarray(est_pos, dtype=np.float64)
    est_quat_arr = np.asarray(est_quat, dtype=np.float64)
    sample_ids_arr = np.asarray(sample_ids, dtype=np.int64)
    contig_ids_arr = np.asarray(contig_ids, dtype=np.int64)
    seg_keys_arr = np.asarray(seg_keys, dtype=np.int64)

    gt_pos, gt_quat, gt_t = _lookup_gt(base_ds, sample_ids_arr, contig_ids_arr, has_window_ts=has_window_ts)

    valid = np.isfinite(est_pos_arr).all(axis=1) & np.isfinite(gt_pos).all(axis=1) & np.isfinite(gt_t)
    if valid.size > 0 and not np.all(valid):
        est_pos_arr = est_pos_arr[valid]
        est_quat_arr = est_quat_arr[valid]
        sample_ids_arr = sample_ids_arr[valid]
        contig_ids_arr = contig_ids_arr[valid]
        seg_keys_arr = seg_keys_arr[valid]
        gt_pos = gt_pos[valid]
        gt_quat = gt_quat[valid]
        gt_t = gt_t[valid]

    if gt_t.size > 1:
        order = np.argsort(gt_t)
        est_pos_arr = est_pos_arr[order]
        est_quat_arr = est_quat_arr[order]
        sample_ids_arr = sample_ids_arr[order]
        contig_ids_arr = contig_ids_arr[order]
        seg_keys_arr = seg_keys_arr[order]
        gt_pos = gt_pos[order]
        gt_quat = gt_quat[order]
        gt_t = gt_t[order]

    return EvalTrajectory(
        est_pos=est_pos_arr,
        est_quat=est_quat_arr,
        gt_pos=gt_pos,
        gt_quat=gt_quat,
        gt_t=gt_t,
        sample_ids=sample_ids_arr,
        contig_ids=contig_ids_arr,
        seg_keys=seg_keys_arr,
    )


def compute_eval_metrics(
    *,
    traj: EvalTrajectory,
    diag: EvalDiagnostics,
    rpe_dt: float,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Compute ATE and RPE metrics, including optional SIM(3) diagnostics.

    Returns:
        ate, rpe_t, rpe_r, state: Extra values for logging/plotting/saving.
    """
    m = int(min(len(traj.est_pos), len(traj.gt_pos)))
    if m <= 0:
        return float("nan"), float("nan"), float("nan"), {}

    state: Dict[str, Any] = {
        "m": m,
        "contiguous_sid_step": int(diag.contiguous_sid_step),
    }

    ate = float("nan")
    ate_sim3 = float("nan")
    s_sim3 = 1.0

    if m >= 3:
        sid_diff = traj.sample_ids[1:] - traj.sample_ids[:-1]
        seg_diff = traj.seg_keys[1:] - traj.seg_keys[:-1]
        jump_mask = (seg_diff != 0) | (sid_diff < 0)
        jump_indices = np.nonzero(jump_mask)[0] + 1
        segment_splits = np.split(np.arange(m), jump_indices)

        se3_sq_errors: List[float] = []
        sim3_sq_errors: List[float] = []
        sim3_scales: List[float] = []
        se3_ate_seq: List[float] = []
        sim3_ate_seq: List[float] = []

        ts_val = float(np.median(np.asarray(diag.s_list, dtype=np.float64))) if (len(diag.s_list) and (not diag.gate_mode)) else 1.0
        state["ts_val"] = ts_val

        for seg_idx in segment_splits:
            if len(seg_idx) < 3:
                continue

            est_seg = traj.est_pos[seg_idx]
            gt_seg = traj.gt_pos[seg_idx]
            t_seg = traj.gt_t[seg_idx]

            R, t_off, _, _ = align_trajectory_with_timestamps(est_seg, t_seg, gt_seg, t_seg)
            est_seg_aligned = (R @ est_seg.T).T + t_off
            sq_err = np.sum((est_seg_aligned - gt_seg) ** 2, axis=1)
            se3_sq_errors.extend(sq_err.tolist())
            se3_ate_seq.append(float(np.sqrt(np.mean(sq_err))))

            mean_gt_step = float(np.mean(np.linalg.norm(gt_seg[1:] - gt_seg[:-1], axis=1))) if gt_seg.shape[0] >= 2 else 0.0
            if (not np.isfinite(mean_gt_step)) or (mean_gt_step < 1e-4):
                sim3_sq_errors.extend(sq_err.tolist())
                sim3_scales.append(1.0)
                sim3_ate_seq.append(float(np.sqrt(np.mean(sq_err))))
                continue

            s_s_seg = 1.0
            if diag.eval_sim3_mode in ("diagnose", "use_learned"):
                est_for_sim3 = est_seg.copy()
                if diag.eval_sim3_mode == "use_learned" and (not diag.gate_mode):
                    est_for_sim3 = est_for_sim3 / (ts_val + 1e-12)
                R_s, t_s, s_s_seg, _, _ = align_trajectory_with_timestamps_sim3(est_for_sim3, t_seg, gt_seg, t_seg)
                est_seg_sim3 = (s_s_seg * R_s @ est_for_sim3.T).T + t_s
                sq_err_sim3 = np.sum((est_seg_sim3 - gt_seg) ** 2, axis=1)
                sim3_sq_errors.extend(sq_err_sim3.tolist())
                sim3_scales.append(float(s_s_seg))
                sim3_ate_seq.append(float(np.sqrt(np.mean(sq_err_sim3))))
            else:
                sim3_sq_errors.extend(sq_err.tolist())
                sim3_scales.append(1.0)
                sim3_ate_seq.append(float(np.sqrt(np.mean(sq_err))))

        ate = float(np.sqrt(np.mean(np.asarray(se3_sq_errors, dtype=np.float64)))) if se3_sq_errors else float("nan")
        ate_sim3 = float(np.sqrt(np.mean(np.asarray(sim3_sq_errors, dtype=np.float64)))) if sim3_sq_errors else float("nan")
        s_sim3 = float(np.mean(np.asarray(sim3_scales, dtype=np.float64))) if sim3_scales else 1.0

        state.update({"ate": ate, "ate_sim3": ate_sim3, "s_sim3": s_sim3})

        if se3_ate_seq:
            se3_seq_arr = np.asarray(se3_ate_seq, dtype=np.float64)
            sim3_seq_arr = np.asarray(sim3_ate_seq, dtype=np.float64) if sim3_ate_seq else np.asarray([], dtype=np.float64)
            se3_mean = float(np.nanmean(se3_seq_arr))
            se3_median = float(np.nanmedian(se3_seq_arr))
            sim3_mean = float(np.nanmean(sim3_seq_arr)) if sim3_seq_arr.size else float("nan")
            sim3_median = float(np.nanmedian(sim3_seq_arr)) if sim3_seq_arr.size else float("nan")
            print(f"[ATE PER-SEQ] N={len(se3_ate_seq)} | SE3(mean/median)={se3_mean:.4f}/{se3_median:.4f} | SIM3(mean/median)={sim3_mean:.4f}/{sim3_median:.4f}")
        else:
            print("[ATE PER-SEQ] N=0")

        _log_step_diagnostics(traj=traj, diag=diag, state=state)
    else:
        ate = float(np.sqrt(np.mean(np.sum((traj.est_pos[:m] - traj.gt_pos[:m]) ** 2, axis=1))))
        state["ate"] = ate

    rpe_t, rpe_r = _compute_rpe(traj=traj, state=state, rpe_dt=float(rpe_dt))
    return ate, rpe_t, rpe_r, state


def plot_eval(*, traj: EvalTrajectory, diag: EvalDiagnostics, state: Dict[str, Any]) -> None:
    """
    Plot trajectory visualization if output directory is set.

    Set FNO_EVIO_EVAL_OUTDIR environment variable to enable plotting.
    """
    outdir = os.environ.get("FNO_EVIO_EVAL_OUTDIR", "").strip()
    if not outdir:
        return

    try:
        from fno_evio.utils.plot_trajectory import (
            plot_trajectory,
            plot_trajectory_simple,
            plot_trajectory_3d,
        )

        os.makedirs(outdir, exist_ok=True)

        # Debug: print trajectory lengths
        print(f"[PLOT DEBUG] est_pos.shape={traj.est_pos.shape}, gt_pos.shape={traj.gt_pos.shape}")

        # Get ATE for title
        ate = state.get("ate", float("nan"))
        ate_sim3 = state.get("ate_sim3", float("nan"))
        title = f"ATE={ate:.4f}m (SIM3: {ate_sim3:.4f}m)" if np.isfinite(ate_sim3) else f"ATE={ate:.4f}m"

        # Try evo-based plot first, fall back to simple plot
        try:
            plot_trajectory(
                pred_pos=traj.est_pos,
                pred_quat=traj.est_quat,
                pred_t=traj.gt_t,
                gt_pos=traj.gt_pos,
                gt_quat=traj.gt_quat,
                gt_t=traj.gt_t,
                title=title,
                filename=os.path.join(outdir, "trajectory.png"),
                align=True,
                correct_scale=True,
            )
        except Exception as e:
            print(f"[PLOT] evo plot failed ({e}), using simple plot")
            plot_trajectory_simple(
                pred_pos=traj.est_pos,
                gt_pos=traj.gt_pos,
                title=title,
                filename=os.path.join(outdir, "trajectory.png"),
            )

        # Also save 3D plot
        try:
            plot_trajectory_3d(
                pred_pos=traj.est_pos,
                gt_pos=traj.gt_pos,
                title=title,
                filename=os.path.join(outdir, "trajectory_3d.png"),
            )
        except Exception as e:
            print(f"[PLOT] 3D plot failed: {e}")

    except ImportError as e:
        print(f"[PLOT] Skipping plot - missing dependency: {e}")


def save_eval_outputs(*, traj: EvalTrajectory, diag: EvalDiagnostics, state: Dict[str, Any]) -> None:
    """
    Save evaluation artifacts including trajectories in TUM format.

    Set FNO_EVIO_EVAL_OUTDIR environment variable to enable saving.
    Outputs:
        - est_pos.npy, gt_pos.npy, gt_t.npy: numpy arrays
        - metrics.npy: evaluation metrics
        - pred_traj.tum, gt_traj.tum: TUM format trajectories
    """
    outdir = os.environ.get("FNO_EVIO_EVAL_OUTDIR", "").strip()
    if not outdir:
        return
    os.makedirs(outdir, exist_ok=True)

    # Save numpy arrays
    np.save(os.path.join(outdir, "est_pos.npy"), traj.est_pos)
    np.save(os.path.join(outdir, "est_quat.npy"), traj.est_quat)
    np.save(os.path.join(outdir, "gt_pos.npy"), traj.gt_pos)
    np.save(os.path.join(outdir, "gt_quat.npy"), traj.gt_quat)
    np.save(os.path.join(outdir, "gt_t.npy"), traj.gt_t)

    # Save metrics
    meta = {
        "ate": float(state.get("ate", float("nan"))),
        "ate_sim3": float(state.get("ate_sim3", float("nan"))),
        "rpe_t": float(state.get("rpe_t", float("nan"))),
        "rpe_r": float(state.get("rpe_r", float("nan"))),
        "rpe_dt": float(state.get("rpe_dt", float("nan"))),
        "s_sim3": float(state.get("s_sim3", 1.0)),
        "direct_ratio": float(state.get("direct_ratio", float("nan"))),
        "path_ratio": float(state.get("path_ratio", float("nan"))),
    }
    np.save(os.path.join(outdir, "metrics.npy"), meta, allow_pickle=True)

    # Save TUM format trajectories
    try:
        from fno_evio.utils.plot_trajectory import save_trajectory_tum

        save_trajectory_tum(
            positions=traj.est_pos,
            orientations=traj.est_quat,
            timestamps=traj.gt_t,
            filename=os.path.join(outdir, "pred_traj.tum"),
        )
        save_trajectory_tum(
            positions=traj.gt_pos,
            orientations=traj.gt_quat,
            timestamps=traj.gt_t,
            filename=os.path.join(outdir, "gt_traj.tum"),
        )
    except Exception as e:
        print(f"[SAVE] Failed to save TUM trajectories: {e}")


def log_eval_summary(*, ate: float, rpe_t: float, rpe_r: float, diag: EvalDiagnostics, state: Dict[str, Any]) -> None:
    """
    Print evaluation summary and high-level diagnostics.
    """
    _ = (diag, state)
    print(f"[EVAL RESULT] ATE={float(ate):.6f} | RPE_t={float(rpe_t):.6f} | RPE_r={float(rpe_r):.6f}")


def _infer_gate_mode(model: torch.nn.Module) -> bool:
    actual_model = model.orig_mod if hasattr(model, "orig_mod") else model
    if hasattr(actual_model, "scale_min") and hasattr(actual_model, "scale_max"):
        smin = float(getattr(actual_model, "scale_min"))
        smax = float(getattr(actual_model, "scale_max"))
        return bool(smin >= -1e-6 and smax <= 1.0 + 1e-6)
    return False


def _extract_sample_stride(base_ds: Any, base_seq: Any, base_base: Any) -> Tuple[int, int]:
    base_sample_stride = 1
    found_stride_attr = False
    current = base_ds
    for _ in range(10):
        if hasattr(current, "sample_stride"):
            base_sample_stride = max(int(current.sample_stride), 1)
            found_stride_attr = True
            print(f"[EVAL] Found sample_stride={base_sample_stride} at {type(current).__name__}")
            break
        if hasattr(current, "dataset"):
            current = current.dataset
        elif hasattr(current, "base"):
            current = current.base
        else:
            break
    if not found_stride_attr:
        print("[EVAL WARNING] Could not find sample_stride attribute, defaulting to 1. This may cause time window mismatch!")
        print(f"[EVAL WARNING] Dataset hierarchy: base_seq={type(base_seq).__name__} -> base_base={type(base_base).__name__} -> base_ds={type(base_ds).__name__}")
    contiguous_sid_step = base_sample_stride
    if base_sample_stride > 1:
        print(f"[EVAL] Dataset sample_stride={base_sample_stride} | Eval subsample={base_sample_stride} | Contiguous sid step={contiguous_sid_step}")
    return base_sample_stride, contiguous_sid_step


def _unpack_eval_batch(batch: Any) -> Tuple[List[Any], Any]:
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch[0], batch[1]
    if isinstance(batch, (tuple, list)):
        return batch[0], None
    return batch, None


def _unpack_eval_item(item: Any, *, dt_fallback: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (list, tuple)):
        item = item[0]
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        raise ValueError("Invalid eval item format")
    ev, imu, y = item[0], item[1], item[2]
    if len(item) > 3:
        dt_tensor = item[3]
        dt_flat = dt_tensor.view(dt_tensor.shape[0], -1) if dt_tensor.ndim > 1 else dt_tensor.view(-1, 1)
        dt_per_sample = dt_flat.mean(dim=1)
        dt_min = float(dt_per_sample.min().item())
        dt_max = float(dt_per_sample.max().item())
        dt_mean = max(float(dt_per_sample.mean().item()), 1e-6)
        if abs(dt_max - dt_min) / dt_mean > 1e-3:
            raise ValueError(
                f"[EVAL DT CHECK] Inconsistent dt_window in eval batch: "
                f"range=({dt_min:.6e}, {dt_max:.6e}), mean={dt_mean:.6e}. "
                f"Please ensure all samples in a batch share the same dt_window."
            )
        actual_dt = float(dt_mean)
    else:
        actual_dt = float(dt_fallback)
    return ev, imu, y, actual_dt


def _get_ids(base_seq: Any, base_base: Any, starts_list: Any, s_idx: int, j: int, b: int):
    ds = base_seq.base if hasattr(base_seq, "base") else base_seq
    try:
        if starts_list is not None and isinstance(starts_list, (list, tuple)) and b < len(starts_list):
            inner_idx = int(starts_list[b]) + int(j)
        else:
            inner_idx = int(s_idx)
    except Exception:
        inner_idx = int(s_idx)

    contig_id = None
    gt_id = None

    try:
        mapped_idx = inner_idx
        cur = ds
        for _ in range(8):
            # Handle Subset (has indices + dataset)
            if hasattr(cur, "indices") and hasattr(cur, "dataset"):
                mapped_idx = int(cur.indices[mapped_idx])
                cur = cur.dataset
                continue
            # Handle SequenceDataset or other wrappers (has base but no indices)
            if hasattr(cur, "base") and not hasattr(cur, "segment_ids"):
                cur = cur.base
                continue
            break

        if hasattr(cur, "contig_ids"):
            try:
                contig_id = int(cur.contig_ids[mapped_idx])
            except Exception:
                contig_id = None
        else:
            contig_id = int(mapped_idx)

        if hasattr(cur, "sample_indices"):
            try:
                gt_id = int(cur.sample_indices[mapped_idx])
            except Exception:
                gt_id = int(mapped_idx)
        else:
            gt_id = int(mapped_idx)

        segment_id = None
        if hasattr(cur, "segment_ids"):
            try:
                if mapped_idx >= 0 and mapped_idx < len(cur.segment_ids):
                    segment_id = int(cur.segment_ids[mapped_idx])
            except Exception:
                segment_id = None

        return contig_id, gt_id, segment_id
    except Exception:
        return None, None, None


def _is_contiguous(
    *,
    seg0: Any,
    last_segment_id: Any,
    has_window_ts: bool,
    contig0: Any,
    last_sid_b0: Any,
    sid0: Any,
    contiguous_sid_step: int,
) -> bool:
    if seg0 is not None and last_segment_id is not None:
        return int(seg0) == int(last_segment_id)
    if has_window_ts:
        return contig0 is not None and last_sid_b0 is not None and contig0 == last_sid_b0 + 1
    return sid0 is not None and last_sid_b0 is not None and sid0 == last_sid_b0 + contiguous_sid_step


def _init_batch_pose(B: int, contiguous: bool, global_t_batch: Any, global_q_batch: Any) -> Tuple[np.ndarray, np.ndarray]:
    t_batch = np.zeros((B, 3), dtype=np.float64)
    q_batch = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (B, 1))
    if contiguous and global_t_batch is not None and global_q_batch is not None:
        if global_t_batch.shape[0] == B and global_q_batch.shape[0] == B:
            t_batch = global_t_batch.copy()
            q_batch = global_q_batch.copy()
    return t_batch, q_batch


def _adjust_hidden_size(hidden: Any, batch_size: int):
    if hidden is None:
        return None
    if not isinstance(hidden, (tuple, list)) or len(hidden) != 2:
        return hidden
    h, c = hidden
    if not torch.is_tensor(h) or not torch.is_tensor(c):
        return hidden
    if h.size(1) == batch_size and c.size(1) == batch_size:
        return hidden
    if h.size(1) > batch_size:
        return (h[:, :batch_size].contiguous(), c[:, :batch_size].contiguous())
    pad_h = torch.zeros(h.size(0), batch_size - h.size(1), h.size(2), device=h.device, dtype=h.dtype)
    pad_c = torch.zeros(c.size(0), batch_size - c.size(1), c.size(2), device=c.device, dtype=c.dtype)
    return (torch.cat([h, pad_h], dim=1), torch.cat([c, pad_c], dim=1))


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R


def _print_eval_norm_debug(*, model: torch.nn.Module, y: torch.Tensor, actual_dt: float) -> None:
    actual_model = model.orig_mod if hasattr(model, "orig_mod") else model
    dbg = getattr(actual_model, "_last_step_debug", None)

    t_hat_body_norm = float(dbg.get("t_hat_body_norm")) if isinstance(dbg, dict) and "t_hat_body_norm" in dbg else float("nan")
    pos_res_norm = float(dbg.get("pos_res_norm")) if isinstance(dbg, dict) and "pos_res_norm" in dbg else float("nan")
    scale_s_val = float(dbg.get("scale_s")) if isinstance(dbg, dict) and "scale_s" in dbg else float("nan")
    if not np.isfinite(scale_s_val) and isinstance(dbg, dict):
        scale_s_val = float(dbg.get("ts")) if "ts" in dbg else float("nan")

    if isinstance(dbg, dict):
        if not np.isfinite(t_hat_body_norm):
            t_hat_body_vec = dbg.get("t_hat_body_vec")
            if isinstance(t_hat_body_vec, np.ndarray) and t_hat_body_vec.ndim == 2 and t_hat_body_vec.shape[1] == 3:
                t_hat_body_norm = float(np.linalg.norm(t_hat_body_vec, axis=1).mean())
        if not np.isfinite(pos_res_norm):
            pos_res_vec = dbg.get("pos_res_vec")
            if isinstance(pos_res_vec, np.ndarray) and pos_res_vec.ndim == 2 and pos_res_vec.shape[1] == 3:
                pos_res_norm = float(np.linalg.norm(pos_res_vec, axis=1).mean())
        if not np.isfinite(scale_s_val):
            for k in ("scale_s_vec", "ts_vec"):
                v = dbg.get(k)
                if isinstance(v, np.ndarray) and v.size > 0:
                    scale_s_val = float(np.median(v.reshape(-1)))
                    break

    t_delta_gt_norm = float(y[:, 0:3].detach().norm(dim=1).mean().item()) if y.ndim >= 2 and y.shape[1] >= 3 else float("nan")
    print(
        f"[DEBUG NORMS][EVAL] ||t_hat_body||={t_hat_body_norm:.6e} ||pos_res||={pos_res_norm:.6e} "
        f"||t_delta_gt||={t_delta_gt_norm:.6e} | scale_s={scale_s_val:.6e} | dt={float(actual_dt):.6e}"
    )


def _collect_uncertainty_stats(*, model: torch.nn.Module, diag: EvalDiagnostics) -> None:
    actual_model = model.orig_mod if hasattr(model, "orig_mod") else model
    dbg_unc = getattr(actual_model, "_last_step_debug", None)
    if not isinstance(dbg_unc, dict):
        return
    w_imu_np = dbg_unc.get("weight_imu")
    w_visual_np = dbg_unc.get("weight_visual")
    if isinstance(w_imu_np, np.ndarray) and w_imu_np.size > 0:
        diag.w_imu_list.extend(w_imu_np.mean(axis=1).flatten().tolist())
    if isinstance(w_visual_np, np.ndarray) and w_visual_np.size > 0:
        diag.w_visual_list.extend(w_visual_np.mean(axis=1).flatten().tolist())
    var_i_np = dbg_unc.get("var_i")
    var_v_np = dbg_unc.get("var_v")
    if isinstance(var_i_np, np.ndarray) and var_i_np.size > 0:
        diag.sigma_imu_list.extend(np.sqrt(var_i_np.mean(axis=1)).flatten().tolist())
    if isinstance(var_v_np, np.ndarray) and var_v_np.size > 0:
        diag.sigma_visual_list.extend(np.sqrt(var_v_np.mean(axis=1)).flatten().tolist())


def _maybe_print_window_probe(**kwargs) -> None:
    s_idx = int(kwargs["s_idx"])
    j = int(kwargs["j"])
    sid = int(kwargs["sid"])
    base_sample_stride = int(kwargs["base_sample_stride"])
    td = kwargs["td"]
    tdelta_gt = kwargs["tdelta_gt"]
    s_np = kwargs["s_np"]
    actual_dt = float(kwargs["actual_dt"])
    speed_thresh = float(kwargs.get("speed_thresh", 0.0) or 0.0)
    b = int(kwargs["b"])

    td_norm = float(np.linalg.norm(td[b]))
    y_norm = float(np.linalg.norm(tdelta_gt[b]))
    ratio_y = td_norm / (y_norm + 1e-12)

    probe_y_min = float(os.getenv("FNO_EVAL_PROBE_YMIN", "1e-3"))
    moving_y_min = max(probe_y_min, max(speed_thresh, 0.0) * float(actual_dt))

    is_first = s_idx == 0 and j == 0
    if not is_first:
        if not (np.isfinite(ratio_y) and ratio_y > 10.0 and y_norm > moving_y_min):
            return

    s_b = float("nan")
    if s_np is not None and getattr(s_np, "size", 0) > b:
        s_b = float(s_np[b])

    print(
        f"[WINDOW PROBE] sid={int(sid)} stride={int(base_sample_stride)} dt={float(actual_dt):.6e} "
        f"||td||={td_norm:.6e} ||y_dpos||={y_norm:.6e} ratio(td/y)={ratio_y:.3f} s={s_b:.6f}"
    )


def _lookup_gt(base_ds: Any, sample_ids: np.ndarray, contig_ids: np.ndarray, *, has_window_ts: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if has_window_ts:
        try:
            wtc = np.asarray(base_ds.window_t_curr, dtype=np.float64)
            if contig_ids.size > 0 and wtc.size > 0:
                contig_ids_i = np.asarray(contig_ids, dtype=np.int64)
                valid = contig_ids_i >= 0
                if np.any(valid) and int(np.max(contig_ids_i[valid])) < int(wtc.size):
                    gt_t = wtc[contig_ids_i]
                    gt_pos_list = []
                    gt_quat_list = []
                    for tt in gt_t.tolist():
                        p_i, q_i = base_ds.interpolate_gt_data(float(tt))
                        gt_pos_list.append(p_i.astype(np.float64))
                        gt_quat_list.append(q_i.astype(np.float64))
                    return np.asarray(gt_pos_list, dtype=np.float64), np.asarray(gt_quat_list, dtype=np.float64), gt_t
        except Exception:
            pass
    gt_pos = base_ds.gt_pos[sample_ids]
    gt_quat = base_ds.gt_quat[sample_ids]
    gt_t = base_ds.gt_t[sample_ids]
    return gt_pos, gt_quat, gt_t


def _compute_rpe(*, traj: EvalTrajectory, state: Dict[str, Any], rpe_dt: float) -> Tuple[float, float]:
    m = int(state.get("m", 0))
    gt_t_m = traj.gt_t[:m]
    if len(gt_t_m) < 2:
        return float("nan"), float("nan")

    idx0_list: List[int] = []
    idx1_list: List[int] = []
    for i0 in range(0, m - 1):
        t0 = float(gt_t_m[i0])
        t_target = t0 + float(rpe_dt)
        j = int(np.searchsorted(gt_t_m, t_target, side="left"))
        if j >= m:
            break
        idx0_list.append(i0)
        idx1_list.append(j)

    if len(idx0_list) == 0:
        idx0 = np.arange(0, m - 1, dtype=np.int64)
        idx1 = idx0 + 1
    else:
        idx0 = np.asarray(idx0_list, dtype=np.int64)
        idx1 = np.asarray(idx1_list, dtype=np.int64)

    dp_est = traj.est_pos[idx1] - traj.est_pos[idx0]
    dp_gt = traj.gt_pos[idx1] - traj.gt_pos[idx0]
    errs = np.linalg.norm(dp_est - dp_gt, axis=1)
    rpe_t = float(np.sqrt(np.mean(errs ** 2))) if errs.size > 0 else float("nan")

    q_gt_rel = np.array([QuaternionUtils.multiply(QuaternionUtils.inverse(traj.gt_quat[i]), traj.gt_quat[j]) for i, j in zip(idx0, idx1)])
    q_est_rel = np.array([QuaternionUtils.multiply(QuaternionUtils.inverse(traj.est_quat[i]), traj.est_quat[j]) for i, j in zip(idx0, idx1)])
    q_diff = np.array([QuaternionUtils.multiply(qg, QuaternionUtils.inverse(qe)) for qg, qe in zip(q_gt_rel, q_est_rel)])
    ang = 2.0 * np.arccos(np.clip(np.abs(q_diff[:, 3]), -1.0, 1.0))
    rpe_r = float(np.degrees(np.mean(ang))) if ang.size > 0 else float("nan")
    state["rpe_dt"] = float(rpe_dt)
    return rpe_t, rpe_r


def _log_step_diagnostics(*, traj: EvalTrajectory, diag: EvalDiagnostics, state: Dict[str, Any]) -> None:
    m = int(state.get("m", 0))
    if m < 2:
        return

    seg_keys_m = traj.seg_keys[:m]
    uniq_keys = np.unique(seg_keys_m)
    step_est_vals: List[float] = []
    step_gt_vals: List[float] = []
    dp_est_all_list: List[np.ndarray] = []
    dp_gt_all_list: List[np.ndarray] = []

    for k in uniq_keys:
        idx = np.nonzero(seg_keys_m == k)[0]
        if idx.size < 2:
            continue
        sid_step_k = traj.sample_ids[idx[1:]] - traj.sample_ids[idx[:-1]]
        ok = (sid_step_k == int(diag.contiguous_sid_step))
        if not np.any(ok):
            continue
        dp_est_k = traj.est_pos[idx[1:]] - traj.est_pos[idx[:-1]]
        dp_gt_k = traj.gt_pos[idx[1:]] - traj.gt_pos[idx[:-1]]
        dp_est_k = dp_est_k[ok]
        dp_gt_k = dp_gt_k[ok]
        step_est_vals.extend(np.linalg.norm(dp_est_k, axis=1).astype(np.float64).tolist())
        step_gt_vals.extend(np.linalg.norm(dp_gt_k, axis=1).astype(np.float64).tolist())
        dp_est_all_list.append(dp_est_k)
        dp_gt_all_list.append(dp_gt_k)

    has_steps = bool(len(step_est_vals) > 1 and len(step_gt_vals) > 1 and dp_est_all_list and dp_gt_all_list)
    dt_stats = (
        float(np.min(diag.dt_list)) if diag.dt_list else float("nan"),
        float(np.mean(diag.dt_list)) if diag.dt_list else float("nan"),
        float(np.max(diag.dt_list)) if diag.dt_list else float("nan"),
    )

    if len(diag.win_step_pred) > 0 and len(diag.win_step_gt) > 0:
        win_pred = np.asarray(diag.win_step_pred, dtype=np.float64)
        win_gt = np.asarray(diag.win_step_gt, dtype=np.float64)
        win_mean_pred = float(np.mean(win_pred))
        win_mean_gt = float(np.mean(win_gt))
        win_ratio = win_mean_pred / (win_mean_gt + 1e-12)
        wp10, wp50, wp90 = [float(np.quantile(win_pred, q)) for q in (0.10, 0.50, 0.90)]
        wg10, wg50, wg90 = [float(np.quantile(win_gt, q)) for q in (0.10, 0.50, 0.90)]
        print(f"[WINDOW STEP] Pred(mean)={win_mean_pred:.6f} GT(mean)={win_mean_gt:.6f} | ratio(mean)={win_ratio:.3f}")
        print(f"[WINDOW STEP] Pred(p10/p50/p90)=[{wp10:.6f},{wp50:.6f},{wp90:.6f}] | GT(p10/p50/p90)=[{wg10:.6f},{wg50:.6f},{wg90:.6f}] | ratio(p50)={wp50/(wg50 + 1e-12):.3f}")

    if not has_steps:
        print(f"[STEP ANALYSIS] insufficient steps for ratio/OLS | dt[min/mean/max]={dt_stats}")
        return

    mean_step_est = float(np.mean(np.asarray(step_est_vals, dtype=np.float64)))
    mean_step_gt = float(np.mean(np.asarray(step_gt_vals, dtype=np.float64)))
    direct_ratio = mean_step_est / (mean_step_gt + 1e-12)

    dp_est_all = np.concatenate(dp_est_all_list, axis=0)
    dp_gt_all = np.concatenate(dp_gt_all_list, axis=0)
    dot = np.einsum("ij,ij->i", dp_est_all, dp_gt_all)
    num = float(np.sum(dot))
    den_pred = float(np.sum(np.linalg.norm(dp_est_all, axis=1) ** 2)) + 1e-12
    den_gt = float(np.sum(np.linalg.norm(dp_gt_all, axis=1) ** 2)) + 1e-12
    s_pred_to_gt = num / den_pred
    s_gt_to_pred = num / den_gt
    sum_pred = float(np.sum(np.linalg.norm(dp_est_all, axis=1)))
    sum_gt = float(np.sum(np.linalg.norm(dp_gt_all, axis=1))) + 1e-12
    path_ratio = sum_pred / sum_gt

    if (path_ratio > 1.0 and direct_ratio < 1.0) or (path_ratio < 1.0 and direct_ratio > 1.0):
        print(f"[WARN] direct_ratio={direct_ratio:.3f} and PATH_RATIO={path_ratio:.3f} disagree. Check pred/gt swap or trajectory mismatch.")

    print(f"[STEP ANALYSIS] Mean Step: Pred={mean_step_est:.6f} vs GT={mean_step_gt:.6f} | direct_ratio={direct_ratio:.3f} | OLS_pred_to_gt={s_pred_to_gt:.6f} | OLS_gt_to_pred={s_gt_to_pred:.6f} | PATH_RATIO={path_ratio:.6f}")
    print(f"[STEP RATIO] Pred/GT = {direct_ratio:.4f} | dt[min/mean/max]={dt_stats}")

    state.update(
        {
            "mean_step_est": mean_step_est,
            "mean_step_gt": mean_step_gt,
            "direct_ratio": direct_ratio,
            "s_pred_to_gt": s_pred_to_gt,
            "s_gt_to_pred": s_gt_to_pred,
            "path_ratio": path_ratio,
        }
    )
