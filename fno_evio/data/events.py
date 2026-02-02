"""
Event voxelization utilities for FNO-EVIO.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0

Notes:
  The event representation follows the 5-channel voxel used in the baseline:
    - ch0: normalized total count
    - ch1: normalized positive count
    - ch2: normalized negative count
    - ch3: mean normalized timestamp for positive events
    - ch4: mean normalized timestamp for negative events
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from fno_evio.common.constants import NumericalConstants, safe_divide


@dataclass
class EventProcessor:
    """
    Vectorized event voxelizer with internal reusable tensor buffers.

    Args:
        resolution: Target voxel resolution (H, W).
        device: Target torch device.
        std_norm: If True, per-channel standardization is applied.
        log_norm: If True, log-normalize count channels.
    """

    resolution: Tuple[int, int]
    device: torch.device
    std_norm: bool = False
    log_norm: bool = False
    voxel_cache: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        H, W = self.resolution
        self.voxel_cache = torch.zeros((5, H, W), dtype=torch.float32, device=self.device)

    def voxelize_events_vectorized(
        self,
        xw: Union[np.ndarray, torch.Tensor],
        yw: Union[np.ndarray, torch.Tensor],
        tw: Union[np.ndarray, torch.Tensor],
        pw: Union[np.ndarray, torch.Tensor],
        src_w: int,
        src_h: int,
        t_prev: float,
        t_curr: float,
    ) -> torch.Tensor:
        """
        Convert an event packet to a fixed-size voxel grid.

        Args:
            xw, yw: Event coordinates.
            tw: Event timestamps.
            pw: Event polarities (can be None).
            src_w, src_h: Sensor resolution.
            t_prev, t_curr: Window boundaries (seconds).

        Returns:
            (5, H, W) voxel tensor.
        """
        H, W = self.resolution
        if self.voxel_cache is None:
            self.voxel_cache = torch.zeros((5, H, W), dtype=torch.float32, device=self.device)
        else:
            self.voxel_cache.zero_()

        def _to_dev(v):
            if isinstance(v, torch.Tensor):
                return v.to(device=self.device, dtype=torch.float32)
            return torch.from_numpy(v.astype(np.float32)).to(self.device)

        x = _to_dev(xw)
        y = _to_dev(yw)
        t = _to_dev(tw)
        p = _to_dev(pw) if pw is not None else torch.zeros_like(t)

        if torch.any(torch.isnan(x)) or torch.any(torch.isnan(y)) or torch.any(torch.isnan(t)):
            return torch.zeros((5, H, W), dtype=torch.float32, device=self.device)

        x = x.view(-1)
        y = y.view(-1)
        t = t.view(-1)
        p = p.view(-1)
        n = min(x.numel(), y.numel(), t.numel(), p.numel())
        x = x[:n]
        y = y[:n]
        t = t[:n]
        p = p[:n]
        if n == 0:
            return self.voxel_cache.clone()

        x_max = float(torch.amax(x).detach().cpu()) if x.numel() > 0 else 0.0
        y_max = float(torch.amax(y).detach().cpu()) if y.numel() > 0 else 0.0
        if x_max <= 1.01 and y_max <= 1.01:
            xs_float = torch.clamp(x * float(W - 1), 0.0, float(W - 1))
            ys_float = torch.clamp(y * float(H - 1), 0.0, float(H - 1))
        else:
            src_w_i = max(int(src_w), 1)
            src_h_i = max(int(src_h), 1)
            x_scaled = x * float(W - 1) / float(max(src_w_i - 1, 1))
            y_scaled = y * float(H - 1) / float(max(src_h_i - 1, 1))
            xs_float = torch.clamp(x_scaled, 0.0, float(W - 1))
            ys_float = torch.clamp(y_scaled, 0.0, float(H - 1))

        xs = torch.round(xs_float).long()
        ys = torch.round(ys_float).long()
        xs = torch.clamp(xs, 0, W - 1)
        ys = torch.clamp(ys, 0, H - 1)
        idx = ys * W + xs

        dt = max(float(t_curr - t_prev), 1e-6)
        norm_t = torch.clamp((t - float(t_prev)) / float(dt), 0.0, 1.0)

        total = max(int(x.numel()), 1)
        voxel = self.voxel_cache
        ch0, ch1, ch2, ch3, ch4 = voxel[0], voxel[1], voxel[2], voxel[3], voxel[4]
        ch0_flat = ch0.view(-1)
        ch1_flat = ch1.view(-1)
        ch2_flat = ch2.view(-1)
        ch3_flat = ch3.view(-1)
        ch4_flat = ch4.view(-1)

        ones = torch.ones_like(idx, dtype=torch.float32)
        ch0_flat.index_add_(0, idx, ones)

        pos_mask = p > 0
        neg_mask = p < 0
        if pos_mask.any():
            pos_idx = idx[pos_mask]
            ch1_flat.index_add_(0, pos_idx, ones[pos_mask])
            ch3_flat.index_add_(0, pos_idx, norm_t[pos_mask])
        if neg_mask.any():
            neg_idx = idx[neg_mask]
            ch2_flat.index_add_(0, neg_idx, ones[neg_mask])
            ch4_flat.index_add_(0, neg_idx, norm_t[neg_mask])

        total_inv = 1.0 / float(total)
        pos_nz = ch1_flat > NumericalConstants.DIVISION_EPS
        neg_nz = ch2_flat > NumericalConstants.DIVISION_EPS
        ch3_flat[pos_nz] = safe_divide(ch3_flat[pos_nz], ch1_flat[pos_nz])
        ch4_flat[neg_nz] = safe_divide(ch4_flat[neg_nz], ch2_flat[neg_nz])

        ch0.mul_(total_inv)
        ch1.mul_(total_inv)
        ch2.mul_(total_inv)

        if self.log_norm:
            scale = torch.log1p(torch.tensor(float(total), device=self.device))
            counts = voxel[0:3, :, :]
            counts.mul_(float(total))
            counts = torch.log1p(counts)
            counts = counts / scale
            voxel[0:3, :, :] = counts

        if self.std_norm:
            m = voxel.view(5, -1).mean(dim=1).view(5, 1, 1)
            s = voxel.view(5, -1).std(dim=1).view(5, 1, 1)
            s = torch.clamp(s, min=1e-6)
            voxel = (voxel - m) / s

        if not torch.isfinite(voxel).all():
            voxel = torch.nan_to_num(voxel, nan=0.0, posinf=0.0, neginf=0.0)

        return voxel.clone()


@dataclass
class AdaptiveEventProcessor:
    """
    Adaptive event voxelizer that optionally applies multi-scale pooling.

    This implements the same behavior as the baseline AdaptiveEventProcessor.
    """

    resolution: Tuple[int, int]
    device: torch.device
    std_norm: bool = False
    log_norm: bool = False

    def __post_init__(self) -> None:
        self.base = EventProcessor(
            resolution=self.resolution,
            device=self.device,
            std_norm=self.std_norm,
            log_norm=self.log_norm,
        )

    def get_adaptive_params(self, event_count: int) -> Tuple[int, int]:
        H, W = self.resolution
        b_h = max(3, H // 60)
        b_w = max(3, W // 60)
        max_e = max(1, (H * W) // 12)
        den = min(10.0, float(event_count) / float(max_e))
        scale = max(den, 1e-6)
        if scale > 2.0:
            scale = 2.0 + np.log1p(scale - 2.0)
        kh_raw = int(b_h / scale)
        kw_raw = int(b_w / scale)
        kh = max(2, min(kh_raw, H // 2))
        kw = max(2, min(kw_raw, W // 2))
        if H % kh != 0 or W % kw != 0:
            kh = max(2, min(kh, H // 4))
            kw = max(2, min(kw, W // 4))
        return kh, kw

    def voxelize_events_adaptive(
        self,
        xw: Union[np.ndarray, torch.Tensor],
        yw: Union[np.ndarray, torch.Tensor],
        tw: Union[np.ndarray, torch.Tensor],
        pw: Union[np.ndarray, torch.Tensor],
        src_w: int,
        src_h: int,
        t_prev: float,
        t_curr: float,
    ) -> torch.Tensor:
        v = self.base.voxelize_events_vectorized(xw, yw, tw, pw, src_w, src_h, t_prev, t_curr)
        H, W = self.resolution
        expected_shape = (5, H, W)
        if v.shape != expected_shape:
            if v.numel() == 5 * H * W:
                v = v.view(expected_shape)
            else:
                v = self._pad_to_shape(v, expected_shape)
        n = int(min(len(xw), len(yw)))
        kh, kw = self.get_adaptive_params(n)
        if kh <= 1 and kw <= 1:
            return v
        try:
            counts = v[0:3, :, :]
            times = v[3:5, :, :]
            area = float(kh * kw)
            counts_p = F.avg_pool2d(counts, kernel_size=(kh, kw), stride=(kh, kw)) * area
            times_p = F.avg_pool2d(times, kernel_size=(kh, kw), stride=(kh, kw))
            counts_u = F.interpolate(counts_p.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            times_u = F.interpolate(times_p.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            return torch.cat([counts_u, times_u], dim=0)
        except Exception as e:
            print(f"Adaptive voxelization failed: {e}")
            return v

    def _pad_to_shape(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        if tensor.shape == target_shape:
            return tensor
        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        if tensor.dim() != len(target_shape):
            src = tensor.reshape(-1)
            dst = result.view(-1)
            n = min(int(src.numel()), int(dst.numel()))
            if n > 0:
                dst[:n] = src[:n]
            return result
        slices = []
        for src_dim, tgt_dim in zip(tensor.shape, target_shape):
            min_dim = min(int(src_dim), int(tgt_dim))
            slices.append(slice(0, min_dim))
        s = tuple(slices)
        result[s] = tensor[s]
        return result
