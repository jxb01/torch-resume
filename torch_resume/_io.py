"""输出编码兼容层。

Windows 控制台默认不是 UTF-8 —— 英文系统是 cp1252，中文系统是 cp936。
直接 print 中文会抛 UnicodeEncodeError，**整个训练脚本就崩了**，这是不能接受的。

两层保护：
  1. 库自己的输出全部走 safe_print，绝不因为编码问题抛异常
  2. 导入时若发现当前编码装不下中文，自动切到 UTF-8（只修坏掉的，不动正常的）

想关掉自动切换：设置环境变量 TORCH_RESUME_NO_UTF8=1
"""
from __future__ import annotations

import os
import sys
from typing import Any

_PROBE = "中"


def _encoding() -> str:
    return getattr(sys.stdout, "encoding", None) or "ascii"


def _can_encode(text: str, enc: str) -> bool:
    try:
        text.encode(enc)
        return True
    except Exception:
        return False


def enable_utf8() -> bool:
    """把 stdout/stderr 切到 UTF-8。成功返回 True。"""
    ok = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
            ok = True
        except Exception:
            pass
    return ok


def safe_print(*args: Any, **kwargs: Any) -> None:
    """打印，绝不因编码问题抛异常。装不下就退化成可显示的字符。"""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        enc = _encoding()
        fallback = [str(a).encode(enc, "replace").decode(enc, "replace") for a in args]
        try:
            print(*fallback, **kwargs)
        except Exception:
            pass


def auto_fix() -> bool:
    """只在编码确实装不下中文时才切 UTF-8。返回是否做了切换。"""
    if os.environ.get("TORCH_RESUME_NO_UTF8") == "1":
        return False
    if _can_encode(_PROBE, _encoding()):
        return False
    return enable_utf8()
