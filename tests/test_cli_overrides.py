from __future__ import annotations

import unittest


class TestCliOverrides(unittest.TestCase):
    def test_config_not_overridden_by_defaults(self) -> None:
        from train_fno_vio import build_arg_parser, build_cfg_from_args

        p = build_arg_parser()
        args = p.parse_args(["--config", "configs/base.yaml", "--root", "/tmp/fno_evio_dummy_root"])
        cfg = build_cfg_from_args(args)
        self.assertAlmostEqual(float(cfg.dataset.dt), 0.00833, places=6)


if __name__ == "__main__":
    unittest.main()
