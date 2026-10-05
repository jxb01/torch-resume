"""结构对比：算出新旧模型的参数迁移计划。

原则：**不做静默迁移**。每个决定都要能解释、能报告。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# 手术模块会在返回的模型上挂这个属性，记录"这次改动是有意的"。
# diff 读它来决定重命名映射、以及哪些参数允许无条件切片继承。
SURGERY_ATTR = "_torch_resume_surgery"


def surgery_info(model: torch.nn.Module) -> Dict[str, object]:
    """读取 surgery 模块记录的结构改动提示（没有就返回空）。"""
    return dict(getattr(model, SURGERY_ATTR, None) or {})


@dataclass
class Action:
    name: str
    kind: str          # keep | partial | reset | drop | rename
    reason: str
    old_shape: Optional[Tuple[int, ...]] = None
    new_shape: Optional[Tuple[int, ...]] = None
    source: Optional[str] = None
    slices: Optional[Tuple[slice, ...]] = None


@dataclass
class MigrationPlan:
    actions: List[Action] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # 预处理之后的旧权重（键已按手术记录重命名/复制）。
    # apply_plan 必须用这一份，否则会找不到重命名后的键。
    old_state: Optional[Dict[str, torch.Tensor]] = None

    def by_kind(self, kind: str) -> List[Action]:
        return [a for a in self.actions if a.kind == kind]

    @property
    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for a in self.actions:
            out[a.kind] = out.get(a.kind, 0) + 1
        return out

    def report(self) -> str:
        lines = ["迁移计划", "=" * 66]
        for a in self.actions:
            if a.kind == "keep":
                detail = "形状一致"
            elif a.kind == "partial":
                detail = "部分继承 %s  %s -> %s" % (a.source, a.old_shape, a.new_shape)
            elif a.kind == "rename":
                detail = "重命名自 %s" % a.source
            elif a.kind == "drop":
                detail = "已删除"
            else:
                detail = "从零初始化（%s）" % a.reason
            lines.append("  %-34s %-8s %s" % (a.name, a.kind, detail))
        lines.append("-" * 66)
        lines.append("  汇总: " + "  ".join("%s=%d" % kv for kv in sorted(self.summary.items())))
        for w in self.warnings:
            lines.append("  [警告] " + w)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "<MigrationPlan %s>" % self.summary


def _slice_for(old: Sequence[int], new: Sequence[int]) -> Optional[Tuple[slice, ...]]:
    """维度数相同、且新形状每维都不小于旧形状时，给出可继承区域。"""
    if len(old) != len(new):
        return None
    if any(n < o for o, n in zip(old, new)):
        return None
    return tuple(slice(0, int(o)) for o in old)


def _overlap_ratio(old: Sequence[int], new: Sequence[int]) -> float:
    """重叠区域占新形状的比例（各维取最小）。

    这个门槛是必须的：Linear(64,4) -> Linear(64,32) 在"每维都不小于"的意义上
    是兼容的，但重叠只有 4/32 = 12.5% —— 把旧的输出层塞进新的隐藏层，语义上是错的。
    """
    if len(old) != len(new) or not new:
        return 0.0
    if any(n < o for o, n in zip(old, new)):
        return 0.0
    return min(float(o) / float(n) for o, n in zip(old, new))


def effective_old_state(
    new_model: torch.nn.Module,
    old_state: Dict[str, torch.Tensor],
    *,
    renames: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    """按手术记录把旧 state_dict 的键调到新模型的名字上。

    diff 和 apply_plan 都必须用这一份，否则重命名/复制过的键会两边对不上。
    """
    info = surgery_info(new_model)
    if renames is None:
        renames = dict(info.get("renames") or {})
    copies = dict(info.get("copies") or {})
    warnings: List[str] = []

    out = dict(old_state)
    if renames:
        moved = 0
        for old_n, new_n in renames.items():
            if old_n in out and new_n not in out:
                out[new_n] = out.pop(old_n)
                moved += 1
        if moved:
            warnings.append("按手术记录重命名了 %d 个参数（显式映射，非猜测）" % moved)

    if copies:
        made = 0
        for new_n, src_n in copies.items():
            if src_n in out and new_n not in out:
                out[new_n] = out[src_n].clone()
                made += 1
        if made:
            warnings.append("按手术记录复制了 %d 份权重（扩深的份不是随机初始化）" % made)

    return out, warnings


def diff(
    new_model: torch.nn.Module,
    old_state: Dict[str, torch.Tensor],
    *,
    allow_rename: bool = False,
    partial_min_ratio: float = 0.5,
    renames: Optional[Dict[str, str]] = None,
) -> MigrationPlan:
    """对比新模型与旧 state_dict，产出迁移计划。

    Args:
        allow_rename: 是否允许按形状猜测重命名（默认关闭，容易猜错）
        partial_min_ratio: 做切片继承所需的最小重叠比例。
            0.5 表示重叠必须覆盖新形状的一半以上，否则宁可从零初始化。
            设为 0.0 可恢复"只要每维不小于就继承"的行为。
        renames: 显式的 旧名 -> 新名 映射。不传则读模型中手术记录的那份。
    """
    plan = MigrationPlan()
    info = surgery_info(new_model)
    forced_partial = set(info.get("partial_ok") or [])

    old_state, prep_warnings = effective_old_state(new_model, old_state, renames=renames)
    plan.old_state = old_state
    plan.warnings.extend(prep_warnings)

    new_all = dict(new_model.named_parameters())
    new_all.update(dict(new_model.named_buffers()))

    old_names = set(old_state.keys())
    consumed: set = set()

    for name, tensor in new_all.items():
        new_shape = tuple(tensor.shape)
        if name in old_state:
            old_shape = tuple(old_state[name].shape)
            if old_shape == new_shape:
                plan.actions.append(Action(name, "keep", "形状一致", old_shape, new_shape))
            else:
                sl = _slice_for(old_shape, new_shape)
                ratio = _overlap_ratio(old_shape, new_shape)
                forced = name in forced_partial
                if sl is not None and (forced or ratio >= partial_min_ratio):
                    why = ("显式指定继承（手术记录）" if forced
                           else "新形状更大，重叠 %.0f%%，可继承" % (ratio * 100))
                    plan.actions.append(Action(
                        name, "partial", why, old_shape, new_shape,
                        source=name, slices=sl))
                elif sl is not None:
                    plan.actions.append(Action(
                        name, "reset",
                        "重叠仅 %.0f%%（低于 %.0f%% 门槛）%s -> %s，宁可从零初始化"
                        % (ratio * 100, partial_min_ratio * 100, old_shape, new_shape),
                        old_shape, new_shape))
                else:
                    plan.actions.append(Action(
                        name, "reset", "形状不兼容 %s -> %s" % (old_shape, new_shape),
                        old_shape, new_shape))
            consumed.add(name)
        else:
            plan.actions.append(Action(name, "reset", "新增参数", None, new_shape))

    for name in sorted(old_names - consumed):
        plan.actions.append(Action(
            name, "drop", "旧模型独有", tuple(old_state[name].shape), None))

    n_partial = sum(1 for a in plan.actions if a.kind == "partial")
    if n_partial:
        plan.warnings.append(
            "%d 处参数形状兼容、做了切片继承 —— "
            "但形状兼容不代表语义正确（例如上游特征含义变了）。请人工确认。" % n_partial)

    if allow_rename:
        pending_new = [a for a in plan.actions if a.kind == "reset" and a.old_shape is None]
        pending_old = [a for a in plan.actions if a.kind == "drop"]
        used: set = set()
        for a in pending_new:
            for b in pending_old:
                if b.name not in used and b.old_shape == a.new_shape:
                    a.kind, a.source = "rename", b.name
                    a.reason = "按形状匹配到 %s" % b.name
                    used.add(b.name)
                    break
        if used:
            plan.actions = [a for a in plan.actions
                            if not (a.kind == "drop" and a.name in used)]
            plan.warnings.append(
                "按形状猜测了 %d 处重命名，请人工确认: %s" % (len(used), sorted(used)))

    return plan
