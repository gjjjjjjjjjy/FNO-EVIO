"""
Sequence slicing and collation utilities for FNO-EVIO.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch
from torch.utils.data import Dataset


class CollateSequence:
    def __init__(self, window_stack_k: int = 1, voxel_stack_mode: str = "abs") -> None:
        self.k = int(window_stack_k)
        self.mode = str(voxel_stack_mode).strip().lower()

    def __call__(self, batch: List[Tuple]) -> Any:
        first_seq = batch[0][0]
        seq_len = len(first_seq)
        batch_size = len(batch)
        batched_seq = []
        for i in range(seq_len):
            evs = []
            imus = []
            ys = []
            dts = []
            intrs = []
            for b in range(batch_size):
                seq, _ = batch[b]
                item = seq[i]
                if self.k > 1:
                    start = max(0, i - self.k + 1)
                    parts_abs = []
                    for j in range(i, start - 1, -1):
                        parts_abs.append(seq[j][0])
                    if len(parts_abs) < self.k:
                        need = self.k - len(parts_abs)
                        parts_abs = parts_abs + [parts_abs[-1] for _ in range(need)]
                    if self.mode == "delta":
                        parts = [parts_abs[0]]
                        for j in range(1, self.k):
                            parts.append(parts_abs[j - 1] - parts_abs[j])
                        ev_stack = torch.cat(parts, dim=0)
                    else:
                        ev_stack = torch.cat(parts_abs, dim=0)
                else:
                    ev_stack = item[0]
                evs.append(ev_stack)
                imus.append(item[1])
                ys.append(item[2])
                if len(item) > 3:
                    dts.append(item[3])
                if len(item) > 4:
                    intrs.append(item[4])
            batched_ev = torch.stack(evs, dim=0)
            batched_imu = torch.stack(imus, dim=0)
            batched_y = torch.stack(ys, dim=0)
            if len(dts) == batch_size:
                batched_dt = torch.stack(dts, dim=0)
                if len(intrs) == batch_size:
                    batched_intr = torch.stack(intrs, dim=0)
                    batched_seq.append((batched_ev, batched_imu, batched_y, batched_dt, batched_intr))
                else:
                    batched_seq.append((batched_ev, batched_imu, batched_y, batched_dt))
            else:
                batched_seq.append((batched_ev, batched_imu, batched_y))
        starts = [s for _, s in batch]
        return (batched_seq, starts)


def collate_sequence(batch: List[Tuple]) -> Any:
    return CollateSequence(window_stack_k=1, voxel_stack_mode="abs")(batch)


class SequenceDataset(Dataset):
    def __init__(self, base_ds: Dataset, sequence_len: int = 200, stride: int = 200) -> None:
        self.base = base_ds
        self.seq_len = int(sequence_len)
        self.stride = int(stride)
        N = len(base_ds)
        if self.stride < self.seq_len:
            print(
                f"[WARN] SequenceDataset uses overlapping sequences: stride={self.stride} < seq_len={self.seq_len}. "
                "Initializing state from GT at each sequence start can introduce inconsistency across overlaps."
            )
            if N >= int(self.seq_len) * 3:
                orig_stride = int(self.stride)
                self.stride = int(self.seq_len)
                print(f"[WARN] Clamping sequence_stride to seq_len for large dataset: stride={orig_stride} -> {self.stride}")
        self.starts = list(range(0, max(N - self.seq_len + 1, 1), self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        N = len(self.base)
        if N <= 0:
            raise IndexError("Empty base dataset")

        s0 = int(self.starts[idx])
        if N >= self.seq_len:
            s = s0
            if s + self.seq_len > N:
                s = max(0, N - self.seq_len)
            e = s + self.seq_len
            return ([self.base[i] for i in range(s, e)], s)

        seq = [self.base[i] for i in range(0, N)]
        last = seq[-1]
        if len(seq) < self.seq_len:
            seq.extend([last for _ in range(self.seq_len - len(seq))])
        return (seq, 0)


__all__ = ["SequenceDataset", "CollateSequence", "collate_sequence"]
