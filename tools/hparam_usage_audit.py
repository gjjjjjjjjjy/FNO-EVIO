from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class ClassFields:
    name: str
    fields: List[str]


def extract_fields(schema_path: Path, class_name: str) -> ClassFields:
    tree = ast.parse(schema_path.read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            fields: List[str] = []
            for stmt in n.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
            return ClassFields(name=class_name, fields=fields)
    return ClassFields(name=class_name, fields=[])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = root / "fno_evio" / "config" / "schema.py"
    code_root = root / "fno_evio"
    cfg_fields = extract_fields(schema, "TrainingConfig").fields + extract_fields(schema, "ModelConfig").fields + extract_fields(schema, "DatasetConfig").fields

    py_files = [p for p in code_root.rglob("*.py") if p.is_file()]
    text_by_file: Dict[str, str] = {str(p): p.read_text(encoding="utf-8") for p in py_files}

    used: Set[str] = set()
    for f in cfg_fields:
        token_attr = f".{f}"
        token_getattr = f"\"{f}\""
        for txt in text_by_file.values():
            if token_attr in txt:
                used.add(f)
                break
            if "getattr" in txt and token_getattr in txt:
                used.add(f)
                break

    unused = sorted([f for f in cfg_fields if f not in used])
    print(json.dumps({"total_fields": len(cfg_fields), "unused_fields": unused}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
