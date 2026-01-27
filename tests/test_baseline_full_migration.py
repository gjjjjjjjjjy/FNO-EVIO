"""
Baseline full migration verification.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TestBaselineFullMigration(unittest.TestCase):
    def test_train_file_bytes_equal(self) -> None:
        root = Path("/Users/gjy/eventlearning/code/FNO-EVIO")
        base = root / "_baseline" / "train_fno_vio.py"
        legacy = root / "fno_evio" / "legacy" / "train_fno_vio.py"
        self.assertTrue(base.exists())
        self.assertTrue(legacy.exists())
        self.assertEqual(_sha256(base), _sha256(legacy))

    def test_utils_file_bytes_equal(self) -> None:
        root = Path("/Users/gjy/eventlearning/code/FNO-EVIO")
        base = root / "_baseline" / "utils.py"
        legacy = root / "fno_evio" / "legacy" / "utils.py"
        self.assertTrue(base.exists())
        self.assertTrue(legacy.exists())
        self.assertEqual(_sha256(base), _sha256(legacy))

    def test_no_runtime_import_baseline_dir(self) -> None:
        root = Path("/Users/gjy/eventlearning/code/FNO-EVIO")
        vio = (root / "fno_evio" / "models" / "vio.py").read_text(encoding="utf-8")
        self.assertNotIn("_baseline", vio)

    def test_utils_exports_baseline_symbols(self) -> None:
        import utils as u

        required = [
            "kb4_project",
            "kb4_unproject",
            "rescale_intrinsics_kb4",
            "rescale_intrinsics_pinhole",
            "warp_events_flow",
            "warp_events_flow_torch",
            "warp_events_flow_torch_kb4",
            "QuaternionUtils",
            "compute_rpe_loss",
            "rotation_6d_to_matrix",
            "matrix_to_rotation_6d",
            "associate_by_timestamp",
            "align_trajectory_with_timestamps",
            "align_trajectory_with_timestamps_sim3",
            "compute_ols_scale_stats",
        ]
        for name in required:
            self.assertTrue(hasattr(u, name), msg=f"utils missing symbol: {name}")


if __name__ == "__main__":
    unittest.main()

