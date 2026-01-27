"""
Physics consistency modules for training.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsBrightnessLoss(nn.Module):
    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self._init_gaussian_kernel(self.sigma)

    def _init_gaussian_kernel(self, sigma: float) -> None:
        kernel_size = int(2 * 4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        x = torch.arange(kernel_size)
        xx = x.repeat(kernel_size).view(kernel_size, kernel_size)
        yy = xx.t()
        mean = (kernel_size - 1) / 2.0
        var = sigma * sigma
        gk = (1.0 / (2.0 * np.pi * var)) * torch.exp(-((xx - mean) ** 2 + (yy - mean) ** 2) / (2 * var))
        gk = gk / torch.sum(gk)
        self.register_buffer("gaussian_kernel", gk.view(1, 1, kernel_size, kernel_size))
        self.padding = kernel_size // 2

    def _get_sobel_kernels(self, device, dtype):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=dtype, device=device).view(1, 1, 3, 3)
        return sobel_x, sobel_y

    def forward(
        self,
        voxel_grid: torch.Tensor,
        motion_pred: torch.Tensor,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        q: float = 0.95,
        mask_thresh: float = 0.05,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = voxel_grid.shape[0]
        raw_img = voxel_grid[:, 0:1]
        p = torch.quantile(raw_img.view(B, -1), float(q), dim=1).view(-1, 1, 1, 1)
        L = raw_img / (p + 1e-6)
        L_smooth = F.conv2d(L, self.gaussian_kernel.to(L.dtype), padding=self.padding)
        sobel_x, sobel_y = self._get_sobel_kernels(L.device, L.dtype)
        grad_x = F.conv2d(L_smooth, sobel_x, padding=1)
        grad_y = F.conv2d(L_smooth, sobel_y, padding=1)
        f_x = float(fx) if fx is not None else 1.0
        f_y = float(fy) if fy is not None else 1.0

        if dt is None:
            dt_vec = motion_pred.new_ones((B, 1))
        else:
            dt_vec = dt.view(B, 1).to(device=motion_pred.device, dtype=motion_pred.dtype).clamp(min=1e-6)
        vel_pred = motion_pred / dt_vec

        vx = (vel_pred[:, 0:1] * f_x).view(-1, 1, 1, 1).expand_as(grad_x)
        vy = (vel_pred[:, 1:2] * f_y).view(-1, 1, 1, 1).expand_as(grad_y)

        if voxel_grid.shape[1] >= 5:
            dt_term = voxel_grid[:, 3:4] - voxel_grid[:, 4:5]
        elif voxel_grid.shape[1] >= 3:
            dt_term = voxel_grid[:, 1:2] - voxel_grid[:, 2:3]
        else:
            dt_term = torch.zeros_like(raw_img)

        pde_residual = dt_term + grad_x * vx + grad_y * vy
        event_mask = torch.sigmoid((L - float(mask_thresh)) * 10.0)
        loss = torch.sum((pde_residual * event_mask) ** 2) / (torch.sum(event_mask) + 1.0)
        return torch.clamp(loss, max=10.0)


__all__ = ["PhysicsBrightnessLoss"]

