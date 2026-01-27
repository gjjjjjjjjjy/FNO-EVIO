"""
Hyper-parameter application tests for the refactored training pipeline.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fno_evio.config.schema import ModelConfig, TrainingConfig
from fno_evio.data.sequence import CollateSequence, SequenceDataset
from fno_evio.training.loop import _build_optimizer_and_scheduler


class _TinyStepDataset(Dataset):
    def __init__(self, n: int = 10) -> None:
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        ev = torch.zeros((5, 8, 8), dtype=torch.float32)
        imu = torch.zeros((50, 6), dtype=torch.float32)
        y = torch.zeros((17,), dtype=torch.float32)
        return (ev, imu, y)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.ones(()))

    def forward(self, *args, **kwargs):
        raise RuntimeError("not used")


class TestHyperparamApplication(unittest.TestCase):
    def test_optimizer_adamw_weight_decay(self) -> None:
        cfg = TrainingConfig(
            epochs=9,
            batch_size=2,
            lr=1e-3,
            optimizer="adamw",
            weight_decay=0.01,
            mixed_precision=False,
            compile=False,
            scheduler="step",
            gamma=0.5,
        )
        device = torch.device("cpu")
        model = _TinyModel().to(device)
        model2, opt, sched, adaptive, scaler = _build_optimizer_and_scheduler(model=model, cfg=cfg, device=device)
        self.assertIs(model2, model)
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertAlmostEqual(opt.param_groups[0]["weight_decay"], 0.01, places=8)
        self.assertIsNone(adaptive)
        self.assertIsNone(scaler)
        self.assertIsNotNone(sched)

    def test_sequence_collate_uses_window_stack_k(self) -> None:
        base = _TinyStepDataset(n=10)
        seq = SequenceDataset(base, sequence_len=4, stride=4)
        model_cfg = ModelConfig(window_stack_K=2, voxel_stack_mode="abs")
        loader = DataLoader(seq, batch_size=2, shuffle=False, collate_fn=CollateSequence(model_cfg.window_stack_K, model_cfg.voxel_stack_mode))
        batch_seq, starts = next(iter(loader))
        self.assertEqual(len(starts), 2)
        self.assertEqual(len(batch_seq), 4)
        ev0, imu0, y0 = batch_seq[0][0], batch_seq[0][1], batch_seq[0][2]
        self.assertEqual(tuple(ev0.shape), (2, 10, 8, 8))
        self.assertEqual(tuple(imu0.shape), (2, 50, 6))
        self.assertEqual(tuple(y0.shape), (2, 17))


if __name__ == "__main__":
    unittest.main()
