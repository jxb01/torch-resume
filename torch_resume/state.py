"""训练状态快照：RNG、以及逐参数的名称映射工具。

暂停、检查点、崩溃恢复三者用的是同一套状态，所以放在一处。
"""
from __future__ import annotations

import random
from typing import Any, Dict

import torch


def capture_rng() -> Dict[str, Any]:
    """抓取全部随机数状态。不记这些，续训就不可复现。"""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def name_map(model: torch.nn.Module) -> Dict[str, torch.nn.Parameter]:
    """参数名 -> 参数。用于把优化器状态按名字对上。"""
    return {n: p for n, p in model.named_parameters()}


def opt_state_by_name(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, Dict[str, Any]]:
    """把优化器状态从"按参数对象"重排成"按参数名"。

    PyTorch 的 optimizer.state 用参数张量做 key，跨模型不可比；
    换成名字之后新旧模型才能对上。
    """
    id2name = {id(p): n for n, p in model.named_parameters()}
    out: Dict[str, Dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = id2name.get(id(p))
            if name is None:
                continue
            st = optimizer.state.get(p)
            if st:
                out[name] = st
    return out


def clone_tree(obj: Any) -> Any:
    """递归克隆，张量搬到 CPU。

    注意：state_dict() 返回的是**引用**，训练继续跑时值会变。
    异步保存前必须先 clone，否则存下来的是"当时的指针、之后的值"。
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(clone_tree(v) for v in obj)
    return obj
