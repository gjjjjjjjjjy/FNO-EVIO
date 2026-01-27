"""Loss component computation for FNO-EVIO.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from fno_evio.common.constants import NumericalConstants, safe_divide

class LossComposer:
    """
    Compute core supervised loss components for one timestep.

    This class is a direct refactor-friendly wrapper around the baseline loss decomposition:
      - Translation loss (direction + log-magnitude)
      - Rotation loss (geodesic distance in quaternion space)
      - Physics loss (optional module)
      - Orthonormality regularizer for 6D rotation head (optional)
    """

    def compute_components(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        raw_6d: Optional[torch.Tensor],
        voxel: torch.Tensor,
        dt_tensor: Optional[torch.Tensor],
        physics_module,
        speed_thresh: float,
        dt: float,
        physics_config: Dict[str, Any],
        scale_weight: float = 1.0,
        scale_reg_weight: float = 0.0,
        static_weight: float = 0.0,
        scale_reg_center: Optional[float] = None,
        min_step_threshold: float = 0.0,
        min_step_weight: float = 0.0,
        path_scale_weight: float = 0.0,
        s: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute (lt, lr, lp, lo) for a single prediction/target pair.

        Args:
            pred: (B, 7) predicted relative pose [t(3), q(4)].
            target: (B, >=7) ground-truth relative pose [t(3), q(4), ...].
            raw_6d: Optional (B, 6) raw 6D rotation head output.
            voxel: (B, C, H, W) event voxel input (used for physics loss).
            dt_tensor: Optional (B,) per-sample dt.
            physics_module: Optional physics consistency module.
            speed_thresh: Speed threshold (m/s) for static/moving gating.
            dt: Fallback dt in seconds.
            physics_config: Dict of physics-module hyperparameters.
            scale_weight: Translation scale loss weight inside lt.
            scale_reg_weight: Optional scale regularizer.
            static_weight: Optional static penalty.
            scale_reg_center: Optional scale center.
            min_step_threshold: Optional step-length floor.
            min_step_weight: Weight for step-length floor penalty.
            path_scale_weight: Optional path-length ratio penalty.
            s: Optional learned scale/gate scalar(s).

        Returns:
            lt: Translation loss.
            lr: Rotation loss.
            lp: Physics loss (or zero if disabled).
            lo: Orthonormality regularizer (or zero if disabled).
        """
        t_pred = pred[:, 0:3].contiguous()
        t_gt = target[:, 0:3].contiguous()

        eps = float(NumericalConstants.DIVISION_EPS)
        t_pred_norm = F.normalize(t_pred, p=2, dim=1, eps=eps)
        t_gt_norm = F.normalize(t_gt, p=2, dim=1, eps=eps)
        l_dir_vec = (1.0 - torch.sum(t_pred_norm * t_gt_norm, dim=1))

        mag_pred = torch.norm(t_pred, p=2, dim=1)
        mag_gt = torch.norm(t_gt, p=2, dim=1)
        l_mag_vec = torch.abs(torch.log(mag_pred + eps) - torch.log(mag_gt + eps))

        lt_vec = l_dir_vec + 0.5 * l_mag_vec
        lt = lt_vec.mean()

        if s is not None and float(scale_reg_weight) > 0.0:
            if scale_reg_center is None:
                l_scale_reg = (torch.log(s + eps) ** 2).mean()
            else:
                l_scale_reg = ((s - float(scale_reg_center)) ** 2).mean()
            lt = lt + float(scale_reg_weight) * 0.5 * l_scale_reg

        if True:
            disp_pred = pred[:, 0:3].contiguous()
            disp_gt = target[:, 0:3].contiguous()
            step_norm_pred = disp_pred.norm(dim=1)
            step_norm_gt = disp_gt.norm(dim=1)

            if dt_tensor is not None:
                dt_local = dt_tensor.view(-1).to(pred.device).clamp(min=1e-6)
            else:
                dt_local = torch.full_like(step_norm_gt, dt).clamp(min=1e-6)

            disp_thresh = float(speed_thresh) * dt_local
            tau = torch.clamp(0.5 * disp_thresh, min=1e-4)
            soft_moving = torch.sigmoid((step_norm_gt - disp_thresh) / (tau + eps))
            soft_static = 1.0 - soft_moving

            if float(static_weight) > 0.0:
                denom = disp_thresh.clamp(min=eps)
                ratio_static = (step_norm_pred + eps) / denom
                ratio_static = torch.clamp(ratio_static, min=1.0, max=1e6)
                l_static = (torch.log(ratio_static) ** 2)
                w_sum = soft_static.sum().clamp(min=eps)
                lt = lt + float(static_weight) * (l_static * soft_static).sum() / w_sum

            if float(scale_weight) > 0.0 or float(min_step_weight) > 0.0 or float(path_scale_weight) > 0.0:
                w_sum = soft_moving.sum().clamp(min=eps)

                mean_norm_pred = (step_norm_pred * soft_moving).sum() / w_sum
                mean_norm_gt = (step_norm_gt * soft_moving).sum() / w_sum

                ratio = (step_norm_pred + eps) / (step_norm_gt + eps)
                ratio = torch.clamp(ratio, min=1e-6, max=1e6)
                l_scale = (torch.log(ratio) ** 2)
                if float(scale_weight) > 0.0:
                    lt = lt + float(scale_weight) * (l_scale * soft_moving).sum() / w_sum

                if float(min_step_threshold) > 0.0 and float(min_step_weight) > 0.0:
                    mean_step_penalty = F.relu(float(min_step_threshold) - mean_norm_pred) ** 2
                    lt = lt + float(min_step_weight) * mean_step_penalty

                if float(path_scale_weight) > 0.0:
                    sum_pred = (step_norm_pred * soft_moving).sum()
                    sum_gt = (step_norm_gt * soft_moving).sum()
                    path_ratio = safe_divide(sum_pred, sum_gt, eps=float(NumericalConstants.DIVISION_EPS), fallback=1.0)
                    path_ratio = torch.clamp(path_ratio, min=1e-4, max=1e4)
                    l_path = (torch.log(path_ratio)) ** 2
                    lt = lt + float(path_scale_weight) * l_path

        q_pred = pred[:, 3:7].contiguous()
        q_gt = target[:, 3:7].contiguous()
        q_pred = q_pred / (q_pred.norm(dim=1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        q_gt = q_gt / (q_gt.norm(dim=1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        lr = GeometryUtils.geodesic_rot_loss(q_pred, q_gt).mean()
        lo = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        if raw_6d is not None:
            a1 = raw_6d[:, :3]
            a2 = raw_6d[:, 3:]
            l_norm = ((a1.norm(dim=1) - 1.0) ** 2).mean() + ((a2.norm(dim=1) - 1.0) ** 2).mean()
            l_dot = (torch.sum(a1 * a2, dim=1) ** 2).mean()
            lo = l_norm + l_dot

        if physics_module is not None:
            displacement = pred[:, 0:3].contiguous()
            lp = physics_module(voxel, displacement,
                                fx=float(physics_config.get('fx', 1.0)),
                                fy=float(physics_config.get('fy', 1.0)),
                                q=float(physics_config.get('physics_scale_quantile', 0.95)),
                                mask_thresh=float(physics_config.get('physics_event_mask_thresh', 0.05)),
                                dt=dt_local)
        else:
            lp = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return lt, lr, lp, lo


class GeometryUtils:    
    @staticmethod
    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def quat_conj(q: torch.Tensor) -> torch.Tensor:
        """Conjugate of quaternion [x, y, z, w] -> [-x, -y, -z, w]"""
        return torch.stack([-q[..., 0], -q[..., 1], -q[..., 2], q[..., 3]], dim=-1)

    @staticmethod
    def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
        q = q / (q.norm(dim=-1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        R = torch.stack([
            1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
            2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
            2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
        ], dim=-1).reshape(q.shape[:-1] + (3, 3))
        return R

    @staticmethod
    def safe_geodesic_loss(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        q1 = q1 / (q1.norm(dim=-1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        q2 = q2 / (q2.norm(dim=-1, keepdim=True) + NumericalConstants.QUATERNION_EPS)
        dot = torch.sum(q1 * q2, dim=-1).abs()
        dot = torch.clamp(dot, min=eps, max=1.0 - eps)
        return 2.0 * torch.acos(dot)

    @staticmethod
    def geodesic_rot_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:
        return GeometryUtils.safe_geodesic_loss(q_pred, q_gt)

    @staticmethod
    def robust_rot_loss(q_pred: torch.Tensor, q_gt: torch.Tensor) -> torch.Tensor:

        return GeometryUtils.safe_geodesic_loss(q_pred, q_gt)
