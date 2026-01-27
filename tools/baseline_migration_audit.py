from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Sym:
    kind: str
    name: str
    signature: str


def _unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _format_args(a: ast.arguments) -> str:
    parts: List[str] = []

    def add_arg(arg: ast.arg, default: Optional[ast.AST]) -> None:
        ann = _unparse(arg.annotation)
        name = arg.arg
        s = name
        if ann:
            s += f": {ann}"
        if default is not None:
            s += f"={_unparse(default)}"
        parts.append(s)

    posonly = list(a.posonlyargs)
    args = list(a.args)
    defaults = list(a.defaults)
    first_default_idx = len(args) - len(defaults)
    for arg in posonly:
        add_arg(arg, None)
    if posonly:
        parts.append("/")
    for i, arg in enumerate(args):
        default = defaults[i - first_default_idx] if i >= first_default_idx else None
        add_arg(arg, default)
    if a.vararg is not None:
        parts.append(f"*{a.vararg.arg}")
    elif a.kwonlyargs:
        parts.append("*")
    for i, arg in enumerate(a.kwonlyargs):
        default = a.kw_defaults[i] if i < len(a.kw_defaults) else None
        add_arg(arg, default)
    if a.kwarg is not None:
        parts.append(f"**{a.kwarg.arg}")
    return ", ".join([p for p in parts if p != ""])


def extract_top_level_symbols(py_path: Path) -> List[Sym]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    out: List[Sym] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = _format_args(node.args)
            ret = _unparse(node.returns)
            sig = f"({args})"
            if ret:
                sig += f" -> {ret}"
            out.append(Sym(kind="def", name=node.name, signature=sig))
        elif isinstance(node, ast.ClassDef):
            out.append(Sym(kind="class", name=node.name, signature=""))
    return sorted(out, key=lambda s: (s.kind, s.name, s.signature))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def diff_syms(a: List[Sym], b: List[Sym]) -> Dict[str, object]:
    set_a = {(s.kind, s.name, s.signature) for s in a}
    set_b = {(s.kind, s.name, s.signature) for s in b}
    missing = sorted(list(set_a - set_b))
    extra = sorted(list(set_b - set_a))
    return {"missing_in_target": missing, "extra_in_target": extra}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline_train = root / "_baseline" / "train_fno_vio.py"
    baseline_utils = root / "_baseline" / "utils.py"
    legacy_train = root / "fno_evio" / "legacy" / "train_fno_vio.py"
    legacy_utils = root / "fno_evio" / "legacy" / "utils.py"

    res: Dict[str, object] = {
        "train": {
            "baseline": str(baseline_train),
            "legacy": str(legacy_train),
            "baseline_sha256": sha256(baseline_train) if baseline_train.exists() else None,
            "legacy_sha256": sha256(legacy_train) if legacy_train.exists() else None,
        },
        "utils": {
            "baseline": str(baseline_utils),
            "legacy": str(legacy_utils),
            "baseline_sha256": sha256(baseline_utils) if baseline_utils.exists() else None,
            "legacy_sha256": sha256(legacy_utils) if legacy_utils.exists() else None,
        },
    }

    if baseline_train.exists() and legacy_train.exists():
        b = extract_top_level_symbols(baseline_train)
        l = extract_top_level_symbols(legacy_train)
        res["train"]["symbol_diff"] = diff_syms(b, l)
        res["train"]["baseline_symbol_count"] = len(b)
        res["train"]["legacy_symbol_count"] = len(l)

    if baseline_utils.exists() and legacy_utils.exists():
        b = extract_top_level_symbols(baseline_utils)
        l = extract_top_level_symbols(legacy_utils)
        res["utils"]["symbol_diff"] = diff_syms(b, l)
        res["utils"]["baseline_symbol_count"] = len(b)
        res["utils"]["legacy_symbol_count"] = len(l)

    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

