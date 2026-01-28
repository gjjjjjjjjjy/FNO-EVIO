from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _resolve_script(repo_root: Path, rel: str) -> Path:
    p = (repo_root / rel).resolve()
    if p.is_file():
        return p
    raise FileNotFoundError(str(p))


def _run_script(repo_root: Path, rel: str, argv: List[str]) -> int:
    script = _resolve_script(repo_root, rel)
    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    root_str = str(repo_root)
    env["PYTHONPATH"] = root_str if not py_path else (root_str + os.pathsep + py_path)
    cmd = [sys.executable, "-u", str(script), *argv]
    return int(subprocess.call(cmd, env=env))


def main(argv: Optional[List[str]] = None) -> int:
    repo_root = Path(__file__).resolve().parent

    p = argparse.ArgumentParser("FNO-EVIO")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("args", nargs=argparse.REMAINDER)

    p_test_fno = sub.add_parser("test-fno")
    p_test_fno.add_argument("args", nargs=argparse.REMAINDER)

    p_test_mvsec = sub.add_parser("test-mvsec")
    p_test_mvsec.add_argument("args", nargs=argparse.REMAINDER)

    ns = p.parse_args(argv)

    if ns.cmd == "train":
        return _run_script(repo_root, "apps/train_fno_vio.py", list(ns.args))
    if ns.cmd == "test-fno":
        return _run_script(repo_root, "apps/TUM-VIE/test_fno_vio.py", list(ns.args))
    if ns.cmd == "test-mvsec":
        return _run_script(repo_root, "apps/MVSEC/test_mvsec.py", list(ns.args))

    raise RuntimeError(str(ns.cmd))


if __name__ == "__main__":
    raise SystemExit(main())
