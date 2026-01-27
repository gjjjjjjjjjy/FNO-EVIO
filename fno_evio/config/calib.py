"""
Calibration YAML loading utilities (baseline-compatible).

Author: gjjjjjjjjjy
Created: 2026-01-27
Version: 0.1.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _pick_shortest_path(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    paths = [p for p in paths if isinstance(p, Path)]
    if not paths:
        return None
    paths = sorted(paths, key=lambda p: (len(p.as_posix()), p.as_posix()))
    return paths[0]


def _resolve_existing_path(v: Any, *, bases: List[Path]) -> Optional[Path]:
    if v is None:
        return None
    try:
        s = str(v)
    except Exception:
        return None
    if not s:
        return None

    cand: List[Path] = []
    p0 = Path(s).expanduser()
    if p0.is_absolute():
        if p0.exists():
            return p0.resolve()
        return None
    for b in bases:
        try:
            p = (b / p0).expanduser()
            if p.exists():
                cand.append(p.resolve())
        except Exception:
            continue
    return _pick_shortest_path(cand)


def infer_dataset_root_from_calib(calib: Dict[str, Any], calib_yaml: Optional[str]) -> Optional[str]:
    for k in ("dataset_root", "root_dir", "root", "data_root"):
        v = calib.get(k)
        if v:
            return str(v)

    yml_dir = Path(".")
    if calib_yaml:
        try:
            yml_dir = Path(str(calib_yaml)).expanduser().resolve().parent
        except Exception:
            yml_dir = Path(str(calib_yaml)).expanduser().parent

    cand: List[Path] = []
    for k in (
        "events_h5",
        "events_path",
        "event_path",
        "imu_path",
        "imu_file",
        "gt_path",
        "gt_file",
        "mocap_path",
        "mocap_file",
    ):
        v = calib.get(k)
        if not v:
            continue
        p = _resolve_existing_path(v, bases=[yml_dir])
        if p is not None:
            cand.append(p)

    if not cand:
        return None

    common = Path(os.path.commonpath([c.as_posix() for c in cand]))
    if common.is_file():
        common = common.parent
    if common.name == "mocap-6dof-vi_gt_data":
        common = common.parent
    return common.as_posix()


def load_calibration(calib_yaml: str) -> Optional[Dict[str, Any]]:
    if not calib_yaml:
        return None
    try:
        import yaml  # type: ignore

        with open(calib_yaml, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            return None
        calib = raw.get("calib") or raw.get("value0") or raw
        if not isinstance(calib, dict):
            return None

        if "K" not in calib:
            cam = calib.get("camera", {})
            if isinstance(cam, dict):
                calib["K"] = {
                    "fx": float(cam.get("fx", 1.0)),
                    "fy": float(cam.get("fy", 1.0)),
                    "cx": float(cam.get("cx", 0.0)),
                    "cy": float(cam.get("cy", 0.0)),
                    "camera_type": cam.get("camera_type", "pinhole"),
                    "k1": float(cam.get("k1", 0.0)),
                    "k2": float(cam.get("k2", 0.0)),
                    "k3": float(cam.get("k3", 0.0)),
                    "k4": float(cam.get("k4", 0.0)),
                }
                if "width" in cam and "height" in cam:
                    calib["resolution"] = (int(cam["width"]), int(cam["height"]))

        if "R_IC" not in calib:
            if "T_imu_cam" in calib:
                T_val = calib["T_imu_cam"]
                if isinstance(T_val, dict) and all(k in T_val for k in ("px", "py", "pz", "qx", "qy", "qz", "qw")):
                    q = np.asarray([T_val["qx"], T_val["qy"], T_val["qz"], T_val["qw"]], dtype=np.float64)
                    n = float(np.linalg.norm(q))
                    if n > 0.0:
                        q = q / n
                    x, y, z, w = q
                    R = np.array(
                        [
                            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
                        ],
                        dtype=np.float64,
                    )
                    calib["R_IC"] = R.tolist()
                else:
                    M = np.asarray(T_val, dtype=np.float64)
                    if M.ndim == 2 and M.shape[0] >= 3 and M.shape[1] >= 3:
                        calib["R_IC"] = M[:3, :3].tolist()
            elif "R_imu_cam" in calib:
                R = np.asarray(calib["R_imu_cam"], dtype=np.float64)
                calib["R_IC"] = R.tolist()

        return calib
    except Exception:
        return None


__all__ = ["load_calibration", "infer_dataset_root_from_calib"]

