"""模型手术：改结构不用重写整个模型。

设计要点：
  1. 每个函数默认返回**新模型**（深拷贝），旧模型留着做迁移对比
  2. 每个函数都会在返回的模型上记录这次改动是**有意的**：
     - renames      下标位移 -> 显式重命名映射
     - copies       新增的份数从哪一份复制权重
     - partial_ok   这些参数允许无条件切片继承（不受重叠比例门槛限制）
     迁移时 diff 会读这份记录。**没有记录就不猜** —— 这是"不做静默迁移"的延续。

  3. nn.Sequential 的下标是位置命名，插入/复制会让后面的下标整体位移。
     手术函数会把位移算出来记进 renames，所以迁移仍然对得上。
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .plan import SURGERY_ATTR, surgery_info

_NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)


# ---------------------------------------------------------------- 基础工具

def clone_model(model: nn.Module) -> nn.Module:
    """深拷贝。改结构前先留一份旧的，迁移才有对比对象。"""
    return copy.deepcopy(model)


def _split(name: str) -> Tuple[Optional[str], str]:
    """'a.b.c' -> ('a.b', 'c')；'c' -> (None, 'c')。"""
    if "." not in name:
        return None, name
    parent, _, attr = name.rpartition(".")
    return parent, attr


def _get(model: nn.Module, name: str) -> nn.Module:
    return model.get_submodule(name) if name else model


def _set(model: nn.Module, name: str, module: nn.Module) -> None:
    """替换子模块。nn.Module.__setattr__ 对 Sequential 的数字下标同样有效。"""
    parent, attr = _split(name)
    setattr(_get(model, parent) if parent else model, attr, module)


def _record(model: nn.Module, **kw) -> None:
    info = dict(surgery_info(model))
    for k, v in kw.items():
        if k in ("renames", "copies", "partial_ok"):
            merged = dict(info.get(k) or {}) if k != "partial_ok" else list(info.get(k) or [])
            if k == "partial_ok":
                merged = sorted(set(merged) | set(v))
            else:
                merged.update(v)
            info[k] = merged
        else:
            info[k] = v
    setattr(model, SURGERY_ATTR, info)


def _param_names(module: nn.Module, prefix: str) -> List[str]:
    out = []
    for k in module.state_dict().keys():
        out.append(("%s.%s" % (prefix, k)) if prefix else k)
    return out


def _sequential_index(name: str) -> Tuple[Optional[str], int]:
    parent, attr = _split(name)
    try:
        return parent, int(attr)
    except ValueError:
        raise ValueError(
            "这个操作目前只支持 nn.Sequential 里的下标（像 '3' 或 'blocks.3'），"
            "得到的是 %r。命名子模块请用 replace_module。" % name)


def _seq(model: nn.Module, parent: Optional[str]) -> nn.Sequential:
    c = _get(model, parent) if parent else model
    if not isinstance(c, nn.Sequential):
        raise TypeError("父容器不是 nn.Sequential，而是 %s" % type(c).__name__)
    return c


# ---------------------------------------------------------------- 手术

def _copy_overlap(old_mod: nn.Module, new_mod: nn.Module) -> int:
    """把旧模块里同名同位置的参数搬到新模块（形状不同就按重叠区域切片）。"""
    osd, nsd = old_mod.state_dict(), new_mod.state_dict()
    moved = 0
    with torch.no_grad():
        for k, nv in nsd.items():
            ov = osd.get(k)
            if not torch.is_tensor(ov) or not torch.is_tensor(nv):
                continue
            if tuple(ov.shape) == tuple(nv.shape):
                nv.copy_(ov)
                moved += 1
            elif ov.ndim == nv.ndim and all(a <= b for a, b in zip(ov.shape, nv.shape)):
                sl = tuple(slice(0, int(a)) for a in ov.shape)
                nv[sl].copy_(ov)
                moved += 1
    return moved


def replace_module(
    model: nn.Module,
    name: str,
    new_module: nn.Module,
    *,
    copy_weights: bool = True,
    inplace: bool = False,
) -> nn.Module:
    """换掉一个子模块。名字不变，所以迁移会天然对上（形状变了就切片）。

    Args:
        copy_weights: 默认把旧模块能对上的权重搬过去。设为 False 就完全用新模块自己的初始化。
    """
    m = model if inplace else clone_model(model)
    moved = 0
    # 新模块常常是刚 nn.Linear(...) 出来的（在 CPU），自动对齐到被替换模块的设备/dtype
    try:
        old = _get(m, name)
        first = next(old.parameters(), None)
        if first is not None:
            new_module = new_module.to(device=first.device, dtype=first.dtype)
        if copy_weights:
            moved = _copy_overlap(old, new_module)
    except Exception:
        pass
    _set(m, name, new_module)
    _record(m, notes=list(surgery_info(m).get("notes") or []) +
            ["replace_module %s -> %s（搬了 %d 个张量）"
             % (name, type(new_module).__name__, moved)])
    return m


def widen(
    model: nn.Module,
    name: str,
    new_out: int,
    *,
    next_name: Optional[str] = None,
    inplace: bool = False,
) -> nn.Module:
    """把一个 nn.Linear 的输出加宽，并**同步加宽下游**（中间夹的 norm 也一起）。

    这是改结构里最有价值的操作：加宽隐藏层之后，已训练的部分原样保留。

        m = tr.widen(model, "0", 128)      # (64,20) -> (128,20)，下游 (4,64) -> (4,128)

    自动找下游只对 nn.Sequential 生效；其他结构请传 next_name。
    """
    m = model if inplace else clone_model(model)
    target = _get(m, name)
    if not isinstance(target, nn.Linear):
        raise TypeError("widen 目前只支持 nn.Linear，得到 %s" % type(target).__name__)

    old_in, old_out = target.in_features, target.out_features
    if new_out <= old_out:
        raise ValueError("new_out 必须大于当前 out_features（%d），得到 %d" % (old_out, new_out))

    dev, dt = target.weight.device, target.weight.dtype
    new_layer = nn.Linear(old_in, new_out, bias=target.bias is not None,
                          device=dev, dtype=dt)
    # 立刻把已训练的部分搬过去。这样即使不调 migrate，加宽后的模型也能直接用；
    # 迁移仍然需要，因为优化器状态不在模型里。
    with torch.no_grad():
        new_layer.weight[:old_out].copy_(target.weight)
        if new_layer.bias is not None and target.bias is not None:
            new_layer.bias[:old_out].copy_(target.bias)
    _set(m, name, new_layer)
    touched = list(_param_names(target, name))

    # 找下游：显式给的优先，否则在同一个 Sequential 里往后找
    norms: List[str] = []
    if next_name is None:
        parent, idx = _sequential_index(name)
        container = _seq(m, parent)
        for j in range(idx + 1, len(container)):
            sub = container[j]
            p = ("%s.%d" % (parent, j)) if parent else str(j)
            if isinstance(sub, nn.Linear):
                next_name = p
                break
            if isinstance(sub, _NORM_TYPES):
                norms.append(p)

    if next_name is not None:
        down = _get(m, next_name)
        if not isinstance(down, nn.Linear):
            raise TypeError("下游 %s 不是 nn.Linear，而是 %s"
                            % (next_name, type(down).__name__))
        if down.in_features != old_out:
            raise ValueError("下游 %s 的 in_features 是 %d，与期望的 %d 不符"
                             % (next_name, down.in_features, old_out))
        new_down = nn.Linear(new_out, down.out_features, bias=down.bias is not None,
                             device=down.weight.device, dtype=down.weight.dtype)
        with torch.no_grad():
            new_down.weight[:, :old_out].copy_(down.weight)
            if new_down.bias is not None and down.bias is not None:
                new_down.bias.copy_(down.bias)
        _set(m, next_name, new_down)
        touched += list(_param_names(down, next_name))

    # 中间夹的归一化层也要跟着变
    for nname in norms:
        mod = _get(m, nname)
        new_mod = _resize_norm(mod, old_out, new_out)
        if new_mod is None:
            raise ValueError("中间层 %s 的形状与 %d 不匹配，无法自动调整" % (nname, old_out))
        _copy_norm_overlap(mod, new_mod, old_out)
        _set(m, nname, new_mod)
        touched += list(_param_names(mod, nname))

    _record(m,
            partial_ok=touched,
            notes=list(surgery_info(m).get("notes") or []) +
            ["widen %s: %d -> %d（下游 %s）" % (name, old_out, new_out, next_name)])
    return m


def insert_after(
    model: nn.Module, name: str, new_module: nn.Module, *, inplace: bool = False
) -> nn.Module:
    """在 nn.Sequential 的某一项之后插入一个模块。

    后面的下标会整体位移，函数会把位移记成显式重命名映射，迁移仍然对得上。
    """
    m = model if inplace else clone_model(model)
    parent, idx = _sequential_index(name)
    container = _seq(m, parent)

    shifted: Dict[str, str] = {}
    for i in range(idx + 1, len(container)):
        old_p = ("%s.%d" % (parent, i)) if parent else str(i)
        new_p = ("%s.%d" % (parent, i + 1)) if parent else str(i + 1)
        for pname in _param_names(container[i], old_p):
            shifted[pname] = pname.replace(old_p, new_p, 1)

    container.insert(idx + 1, new_module)
    _record(m, renames=shifted,
            notes=list(surgery_info(m).get("notes") or []) +
            ["insert_after %s: +1（%d 个参数位移）" % (name, len(shifted))])
    return m


def duplicate_layer(
    model: nn.Module, name: str, n: int = 1, *, inplace: bool = False
) -> nn.Module:
    """把 nn.Sequential 里的某一层复制 n 份插在后面（扩深）。

    新增的份会**直接从原层复制权重**（深度堆叠的标准做法），
    而不是从零初始化 —— 否则等于白加了几层。
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    m = model if inplace else clone_model(model)
    parent, idx = _sequential_index(name)
    container = _seq(m, parent)
    src = container[idx]

    shifted: Dict[str, str] = {}
    for i in range(idx + 1, len(container)):
        old_p = ("%s.%d" % (parent, i)) if parent else str(i)
        new_p = ("%s.%d" % (parent, i + n)) if parent else str(i + n)
        for pname in _param_names(container[i], old_p):
            shifted[pname] = pname.replace(old_p, new_p, 1)

    copies: Dict[str, str] = {}
    for j in range(idx + 1, idx + n + 1):
        new_p = ("%s.%d" % (parent, j)) if parent else str(j)
        for k in src.state_dict().keys():
            copies[new_p + "." + k] = (("%s.%s" % (name, k)) if name else k)
        container.insert(j, copy.deepcopy(src))

    _record(m, renames=shifted, copies=copies,
            notes=list(surgery_info(m).get("notes") or []) +
            ["duplicate_layer %s x%d（%d 个参数位移，%d 份复制权重）"
             % (name, n, len(shifted), len(copies))])
    return m


def freeze_module(model: nn.Module, name: str, value: bool = True) -> int:
    """冻结/解冻一个子模块下的全部参数。返回改了几个。"""
    mod = _get(model, name)
    n = 0
    for p in mod.parameters():
        p.requires_grad = not value
        n += 1
    return n


def describe(model: nn.Module) -> str:
    """把这次会话里做过的手术列出来。"""
    info = surgery_info(model)
    if not info:
        return "（这个模型没有手术记录）"
    lines = ["模型手术记录", "=" * 60]
    for note in info.get("notes") or []:
        lines.append("  " + str(note))
    for key, label in (("renames", "重命名"), ("copies", "复制权重"), ("partial_ok", "显式继承")):
        v = info.get(key) or {}
        if v:
            lines.append("  %s: %d 项" % (label, len(v)))
    return "\n".join(lines)


# ---------------------------------------------------------------- 内部

def _copy_norm_overlap(old_mod: nn.Module, new_mod: nn.Module, old_n: int) -> None:
    """把归一化层已训练的参数/统计量搬到新形状的重叠区域。"""
    with torch.no_grad():
        if getattr(old_mod, "weight", None) is not None and new_mod.weight is not None:
            new_mod.weight[:old_n].copy_(old_mod.weight)
        if getattr(old_mod, "bias", None) is not None and new_mod.bias is not None:
            new_mod.bias[:old_n].copy_(old_mod.bias)
        for attr in ("running_mean", "running_var"):
            o = getattr(old_mod, attr, None)
            n = getattr(new_mod, attr, None)
            if o is not None and n is not None:
                n[:old_n].copy_(o)
        if hasattr(new_mod, "num_batches_tracked") and new_mod.num_batches_tracked is not None \
                and getattr(old_mod, "num_batches_tracked", None) is not None:
            new_mod.num_batches_tracked.copy_(old_mod.num_batches_tracked)


def _resize_norm(mod: nn.Module, old_n: int, new_n: int) -> Optional[nn.Module]:
    """把归一化层的维度从 old_n 改成 new_n（形状对不上就返回 None）。"""
    if isinstance(mod, nn.LayerNorm):
        if tuple(mod.normalized_shape) != (old_n,):
            return None
        return nn.LayerNorm(new_n, eps=mod.eps,
                            elementwise_affine=mod.elementwise_affine,
                            device=mod.weight.device if mod.weight is not None else None,
                            dtype=mod.weight.dtype if mod.weight is not None else None)
    if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
        if mod.num_features != old_n:
            return None
        cls = nn.BatchNorm1d if isinstance(mod, nn.BatchNorm1d) else nn.BatchNorm2d
        return cls(new_n, eps=mod.eps, momentum=mod.momentum, affine=mod.affine,
                   track_running_stats=mod.track_running_stats,
                   device=mod.weight.device if mod.weight is not None else None,
                   dtype=mod.weight.dtype if mod.weight is not None else None)
    return None
