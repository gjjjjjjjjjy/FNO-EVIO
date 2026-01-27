from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FieldInfo:
    name: str
    annotation: str
    default: str


def _unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _literal_default(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        v = ast.literal_eval(node)
        return repr(v)
    except Exception:
        return _unparse(node)


def extract_dataclass_fields(py_path: Path, class_name: str) -> Dict[str, FieldInfo]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"))
    out: Dict[str, FieldInfo] = {}
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            for stmt in n.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    name = stmt.target.id
                    ann = _unparse(stmt.annotation)
                    default = _literal_default(stmt.value)
                    out[name] = FieldInfo(name=name, annotation=ann, default=default)
            break
    return out


def diff_fields(
    base: Dict[str, FieldInfo], cur: Dict[str, FieldInfo]
) -> Tuple[List[str], List[str], List[Tuple[str, str, str, str, str]]]:
    missing_in_cur = sorted([k for k in base.keys() if k not in cur])
    extra_in_cur = sorted([k for k in cur.keys() if k not in base])
    mismatches: List[Tuple[str, str, str, str, str]] = []
    for k in sorted(set(base.keys()) & set(cur.keys())):
        b = base[k]
        c = cur[k]
        if b.annotation != c.annotation or b.default != c.default:
            mismatches.append((k, b.annotation, c.annotation, b.default, c.default))
    return missing_in_cur, extra_in_cur, mismatches


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = root / "_baseline" / "train_fno_vio.py"
    schema = root / "fno_evio" / "config" / "schema.py"
    classes = ["DatasetConfig", "ModelConfig", "TrainingConfig"]
    report: Dict[str, Any] = {"baseline": str(baseline), "current": str(schema), "classes": {}}
    for cls in classes:
        b = extract_dataclass_fields(baseline, cls)
        c = extract_dataclass_fields(schema, cls)
        missing, extra, mismatches = diff_fields(b, c)
        report["classes"][cls] = {
            "baseline_count": len(b),
            "current_count": len(c),
            "missing_in_current": missing,
            "extra_in_current": extra,
            "mismatches": [
                {"name": k, "baseline_type": bt, "current_type": ct, "baseline_default": bd, "current_default": cd}
                for (k, bt, ct, bd, cd) in mismatches
            ],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

