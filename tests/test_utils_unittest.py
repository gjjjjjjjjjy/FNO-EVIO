"""
Unit tests for FNO-EVIO utility modules.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from fno_evio.config.loader import load_experiment_config
from fno_evio.utils.camera import kb4_unproject
from fno_evio.utils.events_warp import warp_events_flow_torch
from fno_evio.utils.quaternion_np import QuaternionUtils
from fno_evio.utils.rotation import rotation_6d_to_matrix
from fno_evio.utils.trajectory import align_trajectory_with_timestamps


class TestQuaternionUtilsNP(unittest.TestCase):
    def test_inverse_identity(self) -> None:
        q = np.array([0.2, -0.1, 0.05, 0.97], dtype=np.float64)
        q = QuaternionUtils.normalize(q)
        q_inv = QuaternionUtils.inverse(q)
        prod = QuaternionUtils.multiply(q, q_inv)
        self.assertTrue(np.allclose(prod, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6))

    def test_rotation_matrix_orthonormal(self) -> None:
        q = QuaternionUtils.normalize(np.array([0.3, 0.4, -0.2, 0.83], dtype=np.float64))
        R = QuaternionUtils.to_rotation_matrix(q)
        I = R.T @ R
        self.assertTrue(np.allclose(I, np.eye(3), atol=1e-6))


class TestTrajectoryAlignment(unittest.TestCase):
    def test_align_se3_recovers_transform(self) -> None:
        t = np.linspace(0.0, 1.0, 20)
        gt = np.stack([t, 0.2 * t, 0.0 * t], axis=1)
        theta = 0.3
        Rz = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0.0],
                [math.sin(theta), math.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        trans = np.array([0.1, -0.2, 0.05], dtype=np.float64)
        est = (Rz.T @ (gt - trans).T).T
        R_hat, t_hat, ie, ig = align_trajectory_with_timestamps(est, t, gt, t, max_dt=1e-6)
        self.assertGreaterEqual(ie.size, 3)
        self.assertTrue(np.allclose(R_hat @ Rz.T, np.eye(3), atol=1e-3))
        self.assertTrue(np.allclose(t_hat, trans, atol=1e-2))


class TestRotation6D(unittest.TestCase):
    def test_rotation_6d_matrix_is_orthonormal(self) -> None:
        d6 = torch.randn(4, 6)
        R = rotation_6d_to_matrix(d6)
        RtR = torch.matmul(R.transpose(-1, -2), R)
        I = torch.eye(3).expand_as(RtR)
        self.assertTrue(torch.allclose(RtR, I, atol=1e-4, rtol=1e-4))


class TestWarp(unittest.TestCase):
    def test_zero_omega_no_warp(self) -> None:
        x = torch.tensor([10.0, 20.0, 30.0])
        y = torch.tensor([15.0, 25.0, 35.0])
        t = torch.tensor([0.0, 0.01, 0.02])
        omega = torch.zeros(3)
        K = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0}
        x2, y2 = warp_events_flow_torch(x, y, t, omega, K, (64, 64), 0.0)
        self.assertTrue(torch.allclose(x2, x, atol=1e-6))
        self.assertTrue(torch.allclose(y2, y, atol=1e-6))


class TestKB4(unittest.TestCase):
    def test_center_unprojects_forward(self) -> None:
        u = np.array([50.0], dtype=np.float64)
        v = np.array([60.0], dtype=np.float64)
        X, Y, Z = kb4_unproject(u, v, fx=100.0, fy=100.0, cx=50.0, cy=60.0)
        self.assertTrue(np.allclose(X, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(Y, 0.0, atol=1e-6))
        self.assertTrue(np.allclose(Z, 1.0, atol=1e-6))


class TestConfigLoader(unittest.TestCase):
    def test_load_sample_yaml(self) -> None:
        cfg = load_experiment_config("/Users/gjy/eventlearning/code/FNO-EVIO/configs/train.yaml")
        self.assertIsNotNone(cfg.dataset.root)
        self.assertGreater(cfg.training.epochs, 0)


if __name__ == "__main__":
    unittest.main()

