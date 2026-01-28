from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from fno_evio.data.datasets import Davis240Dataset, UZHFPVDataset


def _write_txt(path: Path, rows: np.ndarray) -> None:
    lines = [" ".join([f"{float(v):.9f}" for v in r]) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def _write_csv(path: Path, rows: np.ndarray) -> None:
    lines = [",".join([f"{float(v):.9f}" for v in r]) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestDatasetKinds(unittest.TestCase):
    def test_davis240_dataset_loads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            imu_t = np.linspace(0.0, 1.0, num=21, dtype=np.float64)
            gyro = np.zeros((imu_t.size, 3), dtype=np.float64)
            acc = np.tile(np.array([0.0, 0.0, 9.81], dtype=np.float64), (imu_t.size, 1))
            imu_rows = np.concatenate([imu_t[:, None], gyro, acc], axis=1)
            _write_csv(root / "imu_data.csv", imu_rows)

            gt_t = np.linspace(0.0, 1.0, num=21, dtype=np.float64)
            pos = np.stack([gt_t, gt_t * 0.0, gt_t * 0.0], axis=1).astype(np.float64)
            quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (gt_t.size, 1))
            gt_rows = np.concatenate([gt_t[:, None], pos, quat], axis=1)
            _write_txt(root / "gt_stamped_left.txt", gt_rows)

            rng = np.random.default_rng(0)
            n_ev = 2000
            ev_x = rng.integers(0, 32, size=n_ev, dtype=np.int64)
            ev_y = rng.integers(0, 32, size=n_ev, dtype=np.int64)
            ev_t = rng.random(n_ev, dtype=np.float64)
            ev_t.sort()
            ev_p = rng.integers(0, 2, size=n_ev, dtype=np.int64)
            ev_p = np.where(ev_p == 0, -1, 1).astype(np.int64)

            with h5py.File(str(root / "evs_left.h5"), "w") as f:
                f.create_dataset("x", data=ev_x.astype(np.float32))
                f.create_dataset("y", data=ev_y.astype(np.float32))
                f.create_dataset("t", data=ev_t.astype(np.float64))
                f.create_dataset("p", data=ev_p.astype(np.int64))
                f.attrs["height"] = 32
                f.attrs["width"] = 32

            rectify = np.zeros((32, 32, 2), dtype=np.float32)
            xs = np.arange(32, dtype=np.float32)[None, :].repeat(32, axis=0)
            ys = np.arange(32, dtype=np.float32)[:, None].repeat(32, axis=1)
            rectify[:, :, 0] = xs
            rectify[:, :, 1] = ys
            with h5py.File(str(root / "rectify_map_left.h5"), "w") as f:
                f.create_dataset("rectify_map", data=rectify)

            calib = {
                "K": {"fx": 32.0, "fy": 32.0, "cx": 16.0, "cy": 16.0, "camera_type": "pinhole"},
                "resolution": (32, 32),
                "event_rectify_enable": True,
                "event_rectify_map_h5": str(root / "rectify_map_left.h5"),
                "imu_time_unit": "s",
                "gt_time_unit": "s",
                "events_time_unit": "s",
            }

            ds = Davis240Dataset(
                root=str(root),
                dt=0.2,
                resolution=(32, 32),
                sensor_resolution=(32, 32),
                windowing_mode="imu",
                window_dt=0.2,
                calib=calib,
                side="left",
            )
            self.assertGreater(len(ds), 0)
            vox, imu_feat, y, dt_win, intr = ds[0]
            self.assertEqual(tuple(vox.shape), (5, 32, 32))
            self.assertEqual(tuple(imu_feat.shape[1:]), (6,))
            self.assertEqual(int(y.numel()), 17)
            self.assertEqual(tuple(dt_win.shape), (1,))
            self.assertEqual(tuple(intr.shape), (2,))

    def test_uzhfpv_dataset_loads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            imu_t = np.linspace(0.0, 1.0, num=21, dtype=np.float64)
            gyro = np.zeros((imu_t.size, 3), dtype=np.float64)
            acc = np.tile(np.array([0.0, 0.0, 9.81], dtype=np.float64), (imu_t.size, 1))
            imu_rows = np.concatenate([imu_t[:, None], gyro, acc], axis=1)
            _write_txt(root / "imu.txt", imu_rows)

            gt_t_us = (np.linspace(0.0, 1.0, num=21, dtype=np.float64) * 1e6).astype(np.float64)
            pos = np.stack([gt_t_us * 0.0, gt_t_us * 0.0, gt_t_us * 0.0], axis=1).astype(np.float64)
            quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (gt_t_us.size, 1))
            gt_rows = np.concatenate([gt_t_us[:, None], pos, quat], axis=1)
            _write_txt(root / "stamped_groundtruth_us.txt", gt_rows)

            rng = np.random.default_rng(1)
            n_ev = 2000
            ev_t = rng.random(n_ev, dtype=np.float64)
            ev_t.sort()
            ev_x = rng.integers(0, 32, size=n_ev, dtype=np.int64).astype(np.float64)
            ev_y = rng.integers(0, 32, size=n_ev, dtype=np.int64).astype(np.float64)
            ev_p = rng.integers(0, 2, size=n_ev, dtype=np.int64).astype(np.float64)
            ev_rows = np.stack([ev_t, ev_x, ev_y, ev_p], axis=1)
            _write_txt(root / "events.txt", ev_rows)

            calib = {
                "K": {"fx": 32.0, "fy": 32.0, "cx": 16.0, "cy": 16.0, "camera_type": "pinhole"},
                "resolution": (32, 32),
                "imu_time_unit": "s",
                "gt_time_unit": "us",
                "events_time_unit": "s",
                "events_txt_path": str(root / "events.txt"),
            }

            ds = UZHFPVDataset(
                root=str(root),
                dt=0.2,
                resolution=(32, 32),
                sensor_resolution=(32, 32),
                windowing_mode="imu",
                window_dt=0.2,
                calib=calib,
            )
            self.assertGreater(len(ds), 0)
            vox, imu_feat, y, dt_win, intr = ds[0]
            self.assertEqual(tuple(vox.shape), (5, 32, 32))
            self.assertEqual(tuple(imu_feat.shape[1:]), (6,))
            self.assertEqual(int(y.numel()), 17)
            self.assertEqual(tuple(dt_win.shape), (1,))
            self.assertEqual(tuple(intr.shape), (2,))


if __name__ == "__main__":
    unittest.main()
