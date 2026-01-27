"""
Numerical and training constants for FNO-EVIO.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NumericalConstants:
    QUATERNION_EPS: float = 1e-8
    DIVISION_EPS: float = 1e-12
    GRADIENT_CLIP_NORM: float = 1.0
    MIN_EVENT_COUNT: int = 1
    MAX_TEMPORAL_WINDOW: float = 1.0
    ROTATION_EPS: float = 1e-6
    TIME_ALIGNMENT_EPS: float = 1e-9


@dataclass
class TrainingConstants:
    DEFAULT_SEQUENCE_LENGTH: int = 50
    DEFAULT_TBPTT_LENGTH: int = 20
    IMU_FREQUENCY_HZ: int = 200
    ADAPTIVE_SEQ_LENGTH_MULTIPLIER: float = 1.5
    ADAGN_SCALE_OFFSET: float = 0.8
    ADAGN_SCALE_RANGE: float = 0.4
    PHYSICS_QUANTILE_DEFAULT: float = 0.95
    RPE_STRIDE_MULTIPLIER: float = 0.5


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = NumericalConstants.DIVISION_EPS,
    fallback: float = 0.0,
) -> torch.Tensor:
    safe_denominator = torch.where(
        denominator.abs() > eps,
        denominator,
        torch.tensor(eps, device=denominator.device, dtype=denominator.dtype),
    )
    return torch.where(
        denominator.abs() > eps,
        numerator / safe_denominator,
        torch.tensor(fallback, device=numerator.device, dtype=numerator.dtype),
    )


def compute_adaptive_sequence_length(
    dt: float,
    imu_freq: int = TrainingConstants.IMU_FREQUENCY_HZ,
    multiplier: float = TrainingConstants.ADAPTIVE_SEQ_LENGTH_MULTIPLIER,
) -> int:
    return max(int(float(dt) * int(imu_freq) * float(multiplier)), 20)


__all__ = ["NumericalConstants", "TrainingConstants", "safe_divide", "compute_adaptive_sequence_length"]
