"""
Tests for baseline-compatible calibration YAML loading.

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

import unittest

from fno_evio.config.calib import infer_dataset_root_from_calib, load_calibration


class TestCalibYaml(unittest.TestCase):
    def test_load_calib_test_yaml(self) -> None:
        path = "/Users/gjy/eventlearning/code/FNO-EVIO/yaml/calib-test.yaml"
        calib = load_calibration(path)
        self.assertIsInstance(calib, dict)
        self.assertIn("K", calib)
        K = calib["K"]
        self.assertIn("fx", K)
        self.assertIn("fy", K)
        self.assertIn("cx", K)
        self.assertIn("cy", K)
        self.assertIn("camera_type", K)
        self.assertIn("R_IC", calib)

    def test_load_mvsec_yaml(self) -> None:
        path = "/Users/gjy/eventlearning/code/FNO-EVIO/yaml/mvsec_left_calib.yaml"
        calib = load_calibration(path)
        self.assertIsInstance(calib, dict)
        self.assertIn("K", calib)
        self.assertIn("R_IC", calib)

    def test_infer_root_from_multi_root(self) -> None:
        path = "/Users/gjy/eventlearning/code/FNO-EVIO/yaml/tumive_full_calib.yaml"
        calib = load_calibration(path)
        self.assertIsInstance(calib, dict)
        mr = calib.get("multi_root")
        self.assertIsInstance(mr, list)
        inferred = infer_dataset_root_from_calib(calib, path)
        if inferred is None:
            self.assertTrue(len(mr) > 0)


if __name__ == "__main__":
    unittest.main()

