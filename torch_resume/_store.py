"""原子写盘的小工具：先写临时文件再 rename。

写一半崩了不会毁掉已有文件 —— 检查点缓存都靠它。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

import torch


def atomic_torch_save(obj: Any, path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, p)


def atomic_write_text(text: str, path: Union[str, Path], encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, p)


def atomic_write_json(obj: Any, path: Union[str, Path]) -> None:
    atomic_write_text(json.dumps(obj, ensure_ascii=False, indent=1), path)


def read_json(path: Union[str, Path], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default
