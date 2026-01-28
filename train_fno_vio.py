from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from fno_evio.app import main as run_app
from fno_evio.config.calib import infer_dataset_root_from_calib, load_calibration
from fno_evio.config.loader import load_experiment_config, merge_cli_overrides
from fno_evio.config.schema import ExperimentConfig


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("train_fno_vio")
    p.add_argument("--config", type=str, required=True)

    p.add_argument("--root", type=str, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--resolution", type=int, nargs=2, default=None)
    p.add_argument("--sensor_resolution", type=int, nargs=2, default=None)
    p.add_argument("--sample_stride", type=int, default=None)
    p.add_argument("--dataset_kind", type=str, default=None)

    p.add_argument("--calib_yaml", type=str, default="")
    p.add_argument("--multi_calib_yaml", type=str, nargs="+", default=None)

    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    return p


def build_cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    cfg = load_experiment_config(str(args.config))
    overrides: Dict[str, Any] = {
        "root": getattr(args, "root", None),
        "dt": getattr(args, "dt", None),
        "resolution": tuple(getattr(args, "resolution")) if getattr(args, "resolution", None) is not None else None,
        "sensor_resolution": tuple(getattr(args, "sensor_resolution")) if getattr(args, "sensor_resolution", None) is not None else None,
        "sample_stride": getattr(args, "sample_stride", None),
        "dataset_kind": getattr(args, "dataset_kind", None),
        "device": getattr(args, "device", None),
        "seed": getattr(args, "seed", None),
        "output_dir": getattr(args, "output_dir", None),
    }
    cfg = merge_cli_overrides(cfg, overrides=overrides)

    multi_calib_yaml = getattr(args, "multi_calib_yaml", None)
    calib_yaml = str(getattr(args, "calib_yaml", "") or "").strip()

    if isinstance(multi_calib_yaml, list) and multi_calib_yaml:
        calibs_by_root: Dict[str, Any] = {}
        roots = []
        for y in multi_calib_yaml:
            y_p = Path(str(y)).expanduser().resolve()
            if not y_p.exists():
                raise FileNotFoundError(str(y_p))
            calib_obj = load_calibration(str(y_p))
            if calib_obj is None:
                raise RuntimeError(f"Failed to parse calib_yaml: {str(y_p)}")
            inferred = infer_dataset_root_from_calib(calib_obj, str(y_p))
            if not inferred:
                raise ValueError(f"Missing dataset root for calib_yaml: {str(y_p)}")
            root = str(Path(inferred).expanduser().resolve())
            roots.append(root)
            calibs_by_root[root] = calib_obj
        cfg.dataset.multi_root = roots
        cfg.dataset.root = roots[0] if roots else cfg.dataset.root
        cfg.dataset.calib = calibs_by_root
        return cfg

    if calib_yaml:
        y_p = Path(calib_yaml).expanduser().resolve()
        if not y_p.exists():
            raise FileNotFoundError(str(y_p))
        calib_obj = load_calibration(str(y_p))
        if calib_obj is None:
            raise RuntimeError(f"Failed to parse calib_yaml: {str(y_p)}")
        cfg.dataset.calib = calib_obj
        if not str(cfg.dataset.root or "").strip():
            inferred = infer_dataset_root_from_calib(calib_obj, str(y_p))
            if inferred:
                cfg.dataset.root = str(Path(inferred).expanduser().resolve())

    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = build_cfg_from_args(args)
    _ = asdict(cfg)
    run_app(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
